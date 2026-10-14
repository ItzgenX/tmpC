import math

import carla
import numpy as np

from utils import build_camera_intrinsics, project_vertices_to_2d, lerp_location
from spawner import spawn_pedestrian_only, start_walker_controller


def project_bbox_for_transform(bounding_box, transform, camera):
    """
    Projects an arbitrary (bounding_box, transform) pair to a 2D image-space
    bbox without needing the actor to actually be at that transform yet.
    """
    K, image_w, image_h = build_camera_intrinsics(camera)
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())

    verts = bounding_box.get_world_vertices(transform)
    points_2d, _ = project_vertices_to_2d(verts, world_to_camera, K, image_w, image_h)

    if len(points_2d) < 2:
        return None

    points_2d = np.array(points_2d)
    if (np.max(points_2d[:, 0]) < 0 or np.min(points_2d[:, 0]) > image_w or
        np.max(points_2d[:, 1]) < 0 or np.min(points_2d[:, 1]) > image_h):
        return None

    x_min = float(np.clip(np.min(points_2d[:, 0]), 0, image_w))
    y_min = float(np.clip(np.min(points_2d[:, 1]), 0, image_h))
    x_max = float(np.clip(np.max(points_2d[:, 0]), 0, image_w))
    y_max = float(np.clip(np.max(points_2d[:, 1]), 0, image_h))
    return [x_min, y_min, x_max, y_max]


def compute_bbox_overlap_fraction(target_bbox, occluder_bboxes, grid=64):
    """
    Fraction of target_bbox's area covered by the union of occluder_bboxes,
    approximated on a grid so multiple overlapping occluders are handled
    correctly (rasterized union, not naive sum of individual overlaps).
    """
    tx0, ty0, tx1, ty1 = target_bbox
    tw, th = tx1 - tx0, ty1 - ty0
    if tw <= 0 or th <= 0:
        return 0.0

    covered = np.zeros((grid, grid), dtype=bool)
    for (ox0, oy0, ox1, oy1) in occluder_bboxes:
        gx0 = int(np.clip((ox0 - tx0) / tw * grid, 0, grid))
        gx1 = int(np.clip((ox1 - tx0) / tw * grid, 0, grid))
        gy0 = int(np.clip((oy0 - ty0) / th * grid, 0, grid))
        gy1 = int(np.clip((oy1 - ty0) / th * grid, 0, grid))
        if gx1 > gx0 and gy1 > gy0:
            covered[gy0:gy1, gx0:gx1] = True

    return float(covered.sum()) / float(grid * grid)


def camera_forward_depth(world_to_camera, location):
    # same depth axis project_vertices_to_2d uses to drop anything behind
    # the camera - lets us tell which of two boxes is actually nearer
    world_point = np.array([location.x, location.y, location.z, 1.0])
    p_camera = np.dot(world_to_camera, world_point)
    return p_camera[0]


def find_best_occlusion_transform(pedestrian_bb, crossing_start, crossing_end,
                                   occluder_actors, camera, target_percent,
                                   samples=25, progress_range=(0.15, 0.85), grid=64):
    """
    Searches along the crossing line for the pedestrian position whose
    projected bbox overlap with the (already-placed) occluder_actors best
    matches target_percent. Returns (transform, achieved_percent, progress_t)
    or None if nothing projects into frame.

    Only counts an occluder at a candidate point if it's actually nearer to
    the camera than the pedestrian would be there - a box overlapping on
    screen means nothing if the "occluder" is actually behind the
    pedestrian, it has to be in front to really block the view.
    """
    best = None
    best_diff = None
    lo, hi = progress_range

    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())

    occluder_entries = []
    for occ in occluder_actors:
        occ_transform = occ.get_transform()
        bbox = project_bbox_for_transform(occ.bounding_box, occ_transform, camera)
        if bbox:
            depth = camera_forward_depth(world_to_camera, occ_transform.location)
            occluder_entries.append((bbox, depth))

    if not occluder_entries:
        return None

    for i in range(samples):
        t = lo + (hi - lo) * (i / max(1, samples - 1))
        loc = lerp_location(crossing_start, crossing_end, t)
        yaw = math.degrees(math.atan2(crossing_end.y - crossing_start.y, crossing_end.x - crossing_start.x))
        candidate_transform = carla.Transform(
            carla.Location(loc.x, loc.y, loc.z + 0.9),
            carla.Rotation(yaw=yaw)
        )

        ped_bbox = project_bbox_for_transform(pedestrian_bb, candidate_transform, camera)
        if not ped_bbox:
            continue

        ped_depth = camera_forward_depth(world_to_camera, candidate_transform.location)
        near_bboxes = [bbox for bbox, depth in occluder_entries if depth < ped_depth]
        if not near_bboxes:
            continue

        overlap_pct = compute_bbox_overlap_fraction(ped_bbox, near_bboxes, grid=grid) * 100.0
        diff = abs(overlap_pct - target_percent)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (candidate_transform, overlap_pct, t)

    return best


def place_pedestrian_for_occlusion(world, pedestrian_bps, walker_controller_bp,
                                    crossing_start, crossing_end, occluder_candidates,
                                    camera, target_percent, samples=25,
                                    progress_range=(0.15, 0.85), grid=64,
                                    tolerance_percent=12.0, speed_range=(0.9, 1.6)):
    """
    One call that does the whole occlusion-targeting job: spawns a
    pedestrian on the crossing line, searches for the spot along it that
    best matches target_percent against whatever vehicles/trucks/cyclists
    are already in the scene, repositions the pedestrian there, and only
    then starts its AI walker controller.

    That ordering matters: the controller never gets attached until the
    pedestrian is at its final transform. Teleporting an actor that
    already has a live walker controller navigating it is a known way to
    desync CARLA's crowd-nav state - this avoids ever doing that.

    This is the main control point if you want to tune how occlusion
    generation behaves - samples/progress_range/grid affect how hard the
    search tries, tolerance_percent only affects what gets reported back.

    Returns (pedestrian, controller, achieved_percent, within_tolerance).
    pedestrian is None if nothing could even be spawned.
    """
    pedestrian, start_loc, _, _ = spawn_pedestrian_only(
        world, pedestrian_bps, crossing_start, crossing_end, progress=0.5
    )
    if not pedestrian:
        return None, None, 0.0, False

    # the search always assumes walking crossing_start -> crossing_end
    # (that's the fixed yaw it scores candidates with), so once we've
    # repositioned onto the search's line, the walk target follows suit
    forward_yaw = math.degrees(math.atan2(crossing_end.y - crossing_start.y, crossing_end.x - crossing_start.x))

    if not occluder_candidates:
        controller = start_walker_controller(
            world, walker_controller_bp, pedestrian, start_loc, crossing_end, forward_yaw,
            speed_range=speed_range
        )
        return pedestrian, controller, 0.0, target_percent <= 0

    best = find_best_occlusion_transform(
        pedestrian.bounding_box, crossing_start, crossing_end,
        occluder_candidates, camera, target_percent,
        samples=samples, progress_range=progress_range, grid=grid
    )
    if not best:
        controller = start_walker_controller(
            world, walker_controller_bp, pedestrian, start_loc, crossing_end, forward_yaw,
            speed_range=speed_range
        )
        return pedestrian, controller, 0.0, False

    best_transform, achieved_percent, _ = best
    pedestrian.set_transform(best_transform)  # safe - no controller attached yet

    controller = start_walker_controller(
        world, walker_controller_bp, pedestrian, best_transform.location, crossing_end, forward_yaw,
        speed_range=speed_range
    )
    within_tolerance = abs(achieved_percent - target_percent) <= tolerance_percent
    return pedestrian, controller, achieved_percent, within_tolerance
