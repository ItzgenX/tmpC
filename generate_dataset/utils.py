import os
import math
import random
import numpy as np
import carla
import cv2


def carla_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2.0)
    y_center = y_min + (height / 2.0)
    return [x_center / img_w, y_center / img_h, width / img_w, height / img_h]


def get_distance(loc1, loc2):
    return math.sqrt(
        (loc1.x - loc2.x)**2 +
        (loc1.y - loc2.y)**2 +
        (loc1.z - loc2.z)**2
    )


def get_world_corners(box):
    """Manually calculates the 8 corners of the static bounding box in world space."""
    box_transform = carla.Transform(box.location, box.rotation)
    ex, ey, ez = box.extent.x, box.extent.y, box.extent.z
   
    local_corners = [
        carla.Location(ex, ey, ez), carla.Location(ex, -ey, ez),
        carla.Location(-ex, ey, ez), carla.Location(-ex, -ey, ez),
        carla.Location(ex, ey, -ez), carla.Location(ex, -ey, -ez),
        carla.Location(-ex, ey, -ez), carla.Location(-ex, -ey, -ez)
    ]
   
    world_corners = []
    for corner in local_corners:
        p = carla.Location(corner.x, corner.y, corner.z)
        box_transform.transform(p)
        world_corners.append(p)
       
    return world_corners


# ============================================================
# ✅ CORE FIX: Strict FOV + Depth Validation
# ============================================================
def project_vertices_to_2d(verts, world_to_camera, K, image_w, image_h):
    """
    Projects 3D world vertices to 2D image plane.
    Returns (points_2d, all_behind_camera)
    - Filters out vertices behind the camera
    - Tracks how many vertices are actually in front
    """
    points_2d = []
    valid_depths = []


    for v in verts:
        world_point = np.array([v.x, v.y, v.z, 1.0])
        p_camera = np.dot(world_to_camera, world_point)


        # p_camera[0] is forward depth in CARLA camera space
        if p_camera[0] <= 0:
            # Vertex is BEHIND the camera — skip
            continue


        p_camera_std = np.array([
            p_camera[1],
            -p_camera[2],
            p_camera[0]
        ])


        p_image = np.dot(K, p_camera_std)
        p_image /= p_image[2]


        points_2d.append(p_image[:2])
        valid_depths.append(p_camera[0])


    return points_2d, valid_depths




def is_bbox_valid(x_min, y_min, x_max, y_max,
                  image_w, image_h,
                  min_area=200,
                  max_area_ratio=0.85,
                  min_dim=10):
    """
    ✅ Strict bbox validation to eliminate ghost boxes.


    Checks:
    - bbox is within image bounds
    - minimum area threshold (removes tiny noise boxes)
    - maximum area ratio (removes full-frame ghost boxes)
    - minimum width and height dimension
    - aspect ratio sanity check
    """
    # Must overlap with image
    if x_max <= 0 or y_max <= 0:
        return False
    if x_min >= image_w or y_min >= image_h:
        return False


    w = x_max - x_min
    h = y_max - y_min


    # Minimum dimension check
    if w < min_dim or h < min_dim:
        return False


    area = w * h
    image_area = image_w * image_h


    # Too small — likely noise
    if area < min_area:
        return False


    # ✅ KEY FIX: Too large — ghost box covering whole frame
    if area > max_area_ratio * image_area:
        return False


    # Aspect ratio sanity: vehicles shouldn't be extremely thin
    aspect = w / (h + 1e-6)
    if aspect > 10.0 or aspect < 0.1:
        return False


    return True




def build_camera_intrinsics(camera):
    image_w = int(camera.attributes['image_size_x'])
    image_h = int(camera.attributes['image_size_y'])
    fov = float(camera.attributes['fov'])
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.array([
        [focal, 0,     image_w / 2.0],
        [0,     focal, image_h / 2.0],
        [0,     0,     1.0]
    ])
    return K, image_w, image_h




def is_visible_and_not_occluded(world, camera, actor_or_loc, max_distance):
    """
    Occlusion check using ray casting.
    Works for both dynamic actors and static objects.
    """
    if hasattr(actor_or_loc, 'get_transform'):
        actor_loc = actor_or_loc.get_transform().location
    elif hasattr(actor_or_loc, 'transform'):
        actor_loc = actor_or_loc.transform.location
    else:
        actor_loc = actor_or_loc


    cam_loc = camera.get_transform().location
   
    # ✅ Elevate the ray target to prevent hitting the road/ground
    ray_target = carla.Location(actor_loc.x, actor_loc.y, actor_loc.z + 0.8)
    dist = get_distance(cam_loc, ray_target)


    if dist > max_distance:
        return False


    ray_hits = world.cast_ray(cam_loc, ray_target)
    for hit in ray_hits:
        label_str = str(hit.label).split('.')[-1]
        if label_str in ['Buildings', 'Walls', 'Fences', 'Vegetation', 'Other']:
            hit_dist = get_distance(cam_loc, hit.location)
            if hit_dist < dist - 2.5:
                return False
    return True




# ============================================================
# ✅ UNIFIED: Dynamic + Static Vehicle BBox Extraction
# ============================================================
def get_all_vehicle_bboxes(world, camera, max_distance=50.0,
                            min_area=200, check_occlusion=True,
                            min_vertices_in_front=2):
    """
    Returns bounding boxes for ALL vehicles:
    - Dynamic actors (world.get_actors())
    - Static environment objects (world.get_environment_objects())


    Ghost box fixes applied:
    - Minimum vertices in front of camera required
    - Strict area + dimension validation
    - Occlusion check
    - Aspect ratio check
    - Max area ratio check
    """
    K, image_w, image_h = build_camera_intrinsics(camera)
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())
    cam_loc = cam_transform.location


    results = []


    # ----------------------------------------------------------
    # PART 1: Dynamic Vehicles
    # ----------------------------------------------------------
    dynamic_actors = world.get_actors().filter('vehicle.*')


    for actor in dynamic_actors:
        actor_loc = actor.get_transform().location
        dist = get_distance(cam_loc, actor_loc)


        if dist > max_distance:
            continue


        # Occlusion check
        if check_occlusion:
            if not is_visible_and_not_occluded(
                world, camera, actor_loc, max_distance
            ):
                continue


        bb = actor.bounding_box
        verts = bb.get_world_vertices(actor.get_transform())


        points_2d, valid_depths = project_vertices_to_2d(
            verts, world_to_camera, K, image_w, image_h
        )


        # ✅ Ghost Fix: Require minimum vertices in front of camera
        if len(points_2d) < min_vertices_in_front:
            continue


        points_2d = np.array(points_2d)


        # ✅ Ghost Fix: Strictly filter objects completely outside image bounds BEFORE clipping
        if (np.max(points_2d[:, 0]) < 0 or np.min(points_2d[:, 0]) > image_w or
            np.max(points_2d[:, 1]) < 0 or np.min(points_2d[:, 1]) > image_h):
            continue


        x_min = int(np.clip(np.min(points_2d[:, 0]), 0, image_w))
        y_min = int(np.clip(np.min(points_2d[:, 1]), 0, image_h))
        x_max = int(np.clip(np.max(points_2d[:, 0]), 0, image_w))
        y_max = int(np.clip(np.max(points_2d[:, 1]), 0, image_h))


        # ✅ Strict validation
        if not is_bbox_valid(x_min, y_min, x_max, y_max,
                              image_w, image_h, min_area=min_area):
            continue


        results.append({
            'bbox':     [x_min, y_min, x_max, y_max],
            'label':    'Vehicle',
            'type':     'dynamic',
            'id':       actor.id,
            'type_id':  actor.type_id,
            'distance': dist,
            'actor_obj': actor
        })


    # ----------------------------------------------------------
    # PART 2: Static / Environment Vehicles
    # ----------------------------------------------------------
    env_objects = []
    if hasattr(carla.CityObjectLabel, 'Vehicles'):
        env_objects.extend(world.get_environment_objects(carla.CityObjectLabel.Vehicles))
    else:
        for label_str in ['Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle']:
            if hasattr(carla.CityObjectLabel, label_str):
                env_objects.extend(world.get_environment_objects(getattr(carla.CityObjectLabel, label_str)))


    for obj in env_objects:
        bb = obj.bounding_box
       
        # Filter out broken map assets with zero volume
        if bb.extent.x == 0 or bb.extent.y == 0:
            continue
           
        verts = get_world_corners(bb)
        if not verts:
            continue


        # ✅ Extract absolute world center to prevent (0,0,0) transform bugs
        center_x = sum(v.x for v in verts) / len(verts)
        center_y = sum(v.y for v in verts) / len(verts)
        center_z = sum(v.z for v in verts) / len(verts)
        obj_world_loc = carla.Location(x=center_x, y=center_y, z=center_z)
       
        dist = get_distance(cam_loc, obj_world_loc)


        if dist > max_distance:
            continue


        # Occlusion check
        if check_occlusion:
            if not is_visible_and_not_occluded(
                world, camera, obj_world_loc, max_distance
            ):
                continue


        points_2d, valid_depths = project_vertices_to_2d(
            verts, world_to_camera, K, image_w, image_h
        )


        # ✅ Ghost Fix: Require minimum vertices in front of camera
        if len(points_2d) < min_vertices_in_front:
            continue


        points_2d = np.array(points_2d)


        # ✅ Ghost Fix: Strictly filter objects completely outside image bounds BEFORE clipping
        if (np.max(points_2d[:, 0]) < 0 or np.min(points_2d[:, 0]) > image_w or
            np.max(points_2d[:, 1]) < 0 or np.min(points_2d[:, 1]) > image_h):
            continue


        x_min = int(np.clip(np.min(points_2d[:, 0]), 0, image_w))
        y_min = int(np.clip(np.min(points_2d[:, 1]), 0, image_h))
        x_max = int(np.clip(np.max(points_2d[:, 0]), 0, image_w))
        y_max = int(np.clip(np.max(points_2d[:, 1]), 0, image_h))


        # ✅ Strict validation
        if not is_bbox_valid(x_min, y_min, x_max, y_max,
                              image_w, image_h, min_area=min_area):
            continue


        results.append({
            'bbox':     [x_min, y_min, x_max, y_max],
            'label':    'Vehicle',
            'type':     'static',
            'id':       obj.id,
            'type_id':  obj.name,
            'distance': dist,
            'actor_obj': obj
        })


    return results




def get_actor_label(actor):
    if hasattr(actor, 'type_id'):
        if 'walker.pedestrian' in actor.type_id:
            return 'Pedestrian'
        if 'vehicle' in actor.type_id:
            return 'Vehicle'
    elif hasattr(actor, 'type'):
        type_name = str(actor.type).split('.')[-1]
        if type_name in ['Vehicles', 'Car', 'Truck',
                         'Bus', 'Motorcycle', 'Bicycle']:
            return 'Vehicle'
    return 'Actor'




def get_2d_bbox_from_3d(actor, camera):
    """
    Given an actor and a camera, returns the 2D bounding box on the camera plane.
    """
    K, image_w, image_h = build_camera_intrinsics(camera)
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())


    bb = actor.bounding_box
    if hasattr(actor, 'get_transform'):
        verts = bb.get_world_vertices(actor.get_transform())
    else:
        verts = get_world_corners(bb)


    points_2d, valid_depths = project_vertices_to_2d(
        verts, world_to_camera, K, image_w, image_h
    )


    if len(points_2d) < 2:
        return None


    points_2d = np.array(points_2d)


    # ✅ Ghost Fix: Ensure object intersects screen before bounding it to screen
    if (np.max(points_2d[:, 0]) < 0 or np.min(points_2d[:, 0]) > image_w or
        np.max(points_2d[:, 1]) < 0 or np.min(points_2d[:, 1]) > image_h):
        return None


    x_min = int(np.clip(np.min(points_2d[:, 0]), 0, image_w))
    y_min = int(np.clip(np.min(points_2d[:, 1]), 0, image_h))
    x_max = int(np.clip(np.max(points_2d[:, 0]), 0, image_w))
    y_max = int(np.clip(np.max(points_2d[:, 1]), 0, image_h))


    if not is_bbox_valid(x_min, y_min, x_max, y_max, image_w, image_h, min_area=100, min_dim=5):
        return None


    return [x_min, y_min, x_max, y_max]




def draw_3d_bbox_on_image(img, actor, camera, color=(0, 255, 255), thickness=2):
    """
    Draws a 3D bounding box overlay on the provided OpenCV image.
    Handles resizing logic if the camera resolution differs from the output image.
    """
    K, image_w, image_h = build_camera_intrinsics(camera)
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())


    bb = actor.bounding_box
    if hasattr(actor, 'get_transform'):
        verts = bb.get_world_vertices(actor.get_transform())
    else:
        verts = get_world_corners(bb)


    if not verts:
        return


    points_2d = []
    scale_x = img.shape[1] / image_w
    scale_y = img.shape[0] / image_h


    for v in verts:
        world_point = np.array([v.x, v.y, v.z, 1.0])
        p_camera = np.dot(world_to_camera, world_point)


        if p_camera[0] <= 0:
            points_2d.append(None)
            continue


        p_camera_std = np.array([p_camera[1], -p_camera[2], p_camera[0]])
        p_image = np.dot(K, p_camera_std)
        p_image /= p_image[2]


        points_2d.append((int(p_image[0] * scale_x), int(p_image[1] * scale_y)))


    # Edges mapped using CARLA's get_world_vertices output order
    edges = [(0, 2), (2, 6), (6, 4), (4, 0), (1, 3), (3, 7), (7, 5), (5, 1), (0, 1), (2, 3), (4, 5), (6, 7)]


    for p1_idx, p2_idx in edges:
        p1, p2 = points_2d[p1_idx], points_2d[p2_idx]
        if p1 is not None and p2 is not None:
            cv2.line(img, p1, p2, color, thickness)




def draw_2d_bbox_on_image(img, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a 2D bounding box overlay on the provided OpenCV image.
    """
    if bbox is not None:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)




def save_dataset_sample(img, vehicles_info, camera, output_dir, frame_id):
    """
    Saves raw images, 2D labeled images, 3D labeled images, and YOLO 2D labels
    into separate folders to organize the dataset.
    """
    # Define target directories
    raw_dir = os.path.join(output_dir, "raw_images")
    img_2d_dir = os.path.join(output_dir, "images_2d_bbox")
    img_3d_dir = os.path.join(output_dir, "images_3d_bbox")
    labels_2d_dir = os.path.join(output_dir, "labels_2d")


    # Create directories if they do not exist
    for folder in [raw_dir, img_2d_dir, img_3d_dir, labels_2d_dir]:
        os.makedirs(folder, exist_ok=True)


    img_w = img.shape[1]
    img_h = img.shape[0]
    frame_name = f"{frame_id:06d}"


    # 1. Save the Raw Image
    raw_img_path = os.path.join(raw_dir, f"{frame_name}.png")
    cv2.imwrite(raw_img_path, img)


    # 2. Draw and Save 2D Bounding Box Image & Build YOLO Labels
    img_2d = img.copy()
    yolo_labels = []
    for v in vehicles_info:
        bbox = v['bbox']  # [x_min, y_min, x_max, y_max]
        draw_2d_bbox_on_image(img_2d, bbox, color=(0, 255, 0), thickness=2)
       
        # Convert to YOLO format
        yolo_box = carla_to_yolo(bbox, img_w, img_h)
        # Using class ID 0 for 'Vehicle'
        yolo_labels.append(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")


    cv2.imwrite(os.path.join(img_2d_dir, f"{frame_name}.png"), img_2d)


    # 3. Save YOLO 2D Labels
    label_path = os.path.join(labels_2d_dir, f"{frame_name}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_labels))


    # 4. Draw and Save 3D Bounding Box Image
    img_3d = img.copy()
    for v in vehicles_info:
        actor_obj = v['actor_obj']
        draw_3d_bbox_on_image(img_3d, actor_obj, camera, color=(0, 255, 255), thickness=2)
   
    cv2.imwrite(os.path.join(img_3d_dir, f"{frame_name}.png"), img_3d)




def get_crosswalk_approach_transforms(carla_map,
                                       min_dist=2.0,
                                       max_dist=10.0):
    crosswalks = []
    current_polygon = []
    for point in carla_map.get_crosswalks():
        current_polygon.append(point)
        if len(current_polygon) > 2 and point == current_polygon[0]:
            center = carla.Location(
                x=sum(p.x for p in current_polygon) / len(current_polygon),
                y=sum(p.y for p in current_polygon) / len(current_polygon),
                z=sum(p.z for p in current_polygon) / len(current_polygon)
            )
            crosswalks.append(center)
            current_polygon = []


    valid_transforms = []
    for cw_center in crosswalks:
        cw_waypoint = carla_map.get_waypoint(
            cw_center,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if cw_waypoint:
            dist = random.uniform(min_dist, max_dist)
            prev_wps = cw_waypoint.previous(dist)
            t = prev_wps[0].transform if prev_wps else cw_waypoint.transform
            t.location.z += 0.5
            valid_transforms.append((t, cw_center))
    return valid_transforms


# ============================================================
# ✅ SCENARIO PIPELINE: crosswalk-to-crosswalk crossing + occlusion targeting
# ============================================================
def lerp_location(a, b, t):
    return carla.Location(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t
    )


def get_crosswalk_crossing_segments(carla_map, ego_min_dist=5.0, ego_max_dist=12.0):
    """
    For each crosswalk polygon, returns the two sidewalk-side endpoints that
    bound the actual crossing path (not just its center), plus a suggested
    ego-vehicle approach transform on the road before it.
    """
    segments = []
    current_polygon = []
    for point in carla_map.get_crosswalks():
        current_polygon.append(point)
        if len(current_polygon) > 2 and point == current_polygon[0]:
            corners = current_polygon[:-1]
            if len(corners) < 3:
                current_polygon = []
                continue

            center = carla.Location(
                x=sum(p.x for p in corners) / len(corners),
                y=sum(p.y for p in corners) / len(corners),
                z=sum(p.z for p in corners) / len(corners)
            )

            wp = carla_map.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                yaw_rad = math.radians(wp.transform.rotation.yaw)
                road_dir_x, road_dir_y = math.cos(yaw_rad), math.sin(yaw_rad)
                cross_dir_x, cross_dir_y = -road_dir_y, road_dir_x

                side_a, side_b = [], []
                for p in corners:
                    rel_x, rel_y = p.x - center.x, p.y - center.y
                    proj = rel_x * cross_dir_x + rel_y * cross_dir_y
                    (side_a if proj >= 0 else side_b).append(p)

                if side_a and side_b:
                    def avg(pts):
                        return carla.Location(
                            x=sum(pp.x for pp in pts) / len(pts),
                            y=sum(pp.y for pp in pts) / len(pts),
                            z=sum(pp.z for pp in pts) / len(pts)
                        )

                    start = avg(side_a)
                    end = avg(side_b)

                    dist = random.uniform(ego_min_dist, ego_max_dist)
                    prev_wps = wp.previous(dist)
                    ego_transform = prev_wps[0].transform if prev_wps else wp.transform
                    ego_transform.location.z += 0.5

                    segments.append({
                        'start': start,
                        'end': end,
                        'center': center,
                        'ego_transform': ego_transform
                    })
            current_polygon = []
    return segments


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


def find_best_occlusion_transform(pedestrian_bb, crossing_start, crossing_end,
                                   occluder_actors, camera, target_percent,
                                   samples=25, progress_range=(0.15, 0.85)):
    """
    Searches along the crosswalk line for the pedestrian position whose
    projected bbox overlap with the (already-placed) occluder_actors best
    matches target_percent. Returns (transform, achieved_percent, progress_t)
    or None if nothing projects into frame.
    """
    best = None
    best_diff = None
    lo, hi = progress_range

    occluder_bboxes_cache = []
    for occ in occluder_actors:
        bbox = project_bbox_for_transform(occ.bounding_box, occ.get_transform(), camera)
        if bbox:
            occluder_bboxes_cache.append(bbox)

    if not occluder_bboxes_cache:
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

        overlap_pct = compute_bbox_overlap_fraction(ped_bbox, occluder_bboxes_cache) * 100.0
        diff = abs(overlap_pct - target_percent)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (candidate_transform, overlap_pct, t)

    return best


def get_actor_label_for_scenario(actor, truck_ids, cyclist_ids, escooter_ids, separate_truck_class=True):
    """
    Scenario-pipeline label resolver: distinguishes Cyclist / E-Scooter+Rider
    / Truck (togglable) / Vehicle / Pedestrian by blueprint id.
    """
    if hasattr(actor, 'type_id'):
        if 'walker.pedestrian' in actor.type_id:
            return 'Pedestrian'
        if actor.type_id in escooter_ids:
            return 'E-Scooter + Rider'
        if actor.type_id in cyclist_ids:
            return 'Cyclist'
        if actor.type_id in truck_ids:
            return 'Truck' if separate_truck_class else 'Vehicle'
        if 'vehicle' in actor.type_id:
            return 'Vehicle'
    elif hasattr(actor, 'type'):
        type_name = str(actor.type).split('.')[-1]
        if type_name in ['Vehicles', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle']:
            return 'Vehicle'
    return 'Actor'

