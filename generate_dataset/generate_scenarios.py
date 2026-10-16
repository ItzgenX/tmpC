import argparse
import carla
import os
import csv
import random
import time
import queue
import traceback
import cv2
import numpy as np
import json


from scenario_config import (
    MAP_NAME, MAP_ROTATION, DRAW_2D_BBOX, DRAW_3D_BBOX,
    resolve_output_dir, OUTPUT_BASE_DIR,
    SEPARATE_TRUCK_CLASS, CLASS_MAP, SCENARIOS, IMAGE_TYPE_SPLIT, OCCLUSION_PERCENT_RANGE,
    REQUIRE_PEDESTRIAN, MIN_OPTIONAL_CATEGORIES_SATISFIED, MAX_COUNT_REROLLS,
    OCCLUSION_SEARCH_SAMPLES, OCCLUSION_PROGRESS_RANGE, OCCLUSION_OVERLAP_GRID,
    OCCLUSION_TOLERANCE_PERCENT, OCCLUSION_STRICT, OCCLUSION_MIN_ACHIEVED_PERCENT,
    PEDESTRIAN_WALK_SPEED_RANGE, CROSSING_WARMUP_TICKS_RANGE, MAX_RENDER_DISTANCE,
    JAYWALK_CROSSWALK_EXCLUSION_RADIUS,
    TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX, TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE,
    TRUCK_SPAWN_DISTANCE_MIN, TRUCK_SPAWN_DISTANCE_MAX, TRUCK_SPAWN_LATERAL_RANGE,
    CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN, CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX, CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE,
    MIN_SEPARATION_BETWEEN_VEHICLES, MIN_SEPARATION_TRUCK_TO_CYCLIST, MAX_SPAWN_ATTEMPTS_PER_ACTOR,
    CROSSWALK_VEHICLE_EXCLUSION_RADIUS,
    ALLOWED_CAR_VEHICLES, ALLOWED_TRUCK_VEHICLES, ALLOWED_CYCLIST_SCOOTER_VEHICLES,
    ALLOWED_ESCOOTER_VEHICLES, ALLOWED_TWO_WHEELER_VEHICLES,
    RECONNECT_RETRY_DELAY_SECONDS, RECONNECT_MAX_ATTEMPTS,
    POST_DESTROY_SETTLE_TICKS, POST_DESTROY_SETTLE_SECONDS, EGO_REFRESH_EVERY_N_IMAGES,
    SPAWN_SIDEWALK_PEDESTRIANS, SIDEWALK_PEDESTRIAN_COUNT_RANGE,
    SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MIN, SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MAX,
)
from utils import (
    get_2d_bbox_from_3d, is_visible_and_not_occluded, carla_to_yolo,
    get_all_vehicle_bboxes, draw_3d_bbox_on_image, should_use_lights,
    get_crosswalk_crossing_segments,
    get_crosswalk_centers, find_jaywalk_waypoint, get_full_road_crossing_segment,
    resolve_scenario_counts,
)
from spawner import spawn_actor, spawn_pedestrian_crossing, spawn_traffic_vehicles, spawn_pedestrian_on_sidewalk
from occlusion import place_pedestrian_for_occlusion
from image_plan import load_or_build_plan, consume_plan_item, pick_next_bucket, total_remaining


# shortcuts for the three "one type only" batches
MODE_SPLITS = {
    "crosswalk_only": {"crosswalk_clear": 50, "crosswalk_occluded": 50, "jaywalk_clear": 0, "jaywalk_occluded": 0},
    "jaywalk_only": {"crosswalk_clear": 0, "crosswalk_occluded": 0, "jaywalk_clear": 50, "jaywalk_occluded": 50},
    "occlusion_only": {"crosswalk_clear": 0, "crosswalk_occluded": 50, "jaywalk_clear": 0, "jaywalk_occluded": 50},
}

CONNECTION_ERROR_HINTS = [
    "time-out", "timeout", "time out", "failed to connect", "connection",
    "broken pipe", "not responding", "unable to connect", "disconnected",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the CARLA pedestrian-crossing dataset.")
    parser.add_argument(
        "--mode", choices=["mixed", "crosswalk_only", "jaywalk_only", "occlusion_only"],
        default="mixed",
        help="mixed (default): the split from IMAGE_TYPE_SPLIT in scenario_config.py. "
             "crosswalk_only/jaywalk_only/occlusion_only: restrict to that type, this run only."
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="override number_images_to_generate for every scenario (per scenario, not total)."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="path to a previous run folder, or 'latest' for the most recent one. "
             "Resumes exactly where that run left off."
    )
    return parser.parse_args()


def resolve_split(mode):
    if mode == "mixed":
        return IMAGE_TYPE_SPLIT
    return MODE_SPLITS[mode]


def looks_like_connection_error(exc):
    msg = str(exc).lower()
    return any(hint in msg for hint in CONNECTION_ERROR_HINTS)


def resolve_latest_run_dir():
    # most recently modified folder under dataset_scenarios/
    if not os.path.isdir(OUTPUT_BASE_DIR):
        return None
    candidates = [
        os.path.join(OUTPUT_BASE_DIR, d) for d in os.listdir(OUTPUT_BASE_DIR)
        if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def scenario_dirs(output_dir, scenario_name):
    base = os.path.join(output_dir, scenario_name)
    dirs = {
        'base': base,
        'images': os.path.join(base, "images"),
        'labels': os.path.join(base, "labels"),
        'labels_3d': os.path.join(base, "labels_3d"),
        'seg_raw': os.path.join(base, "images_segmentation_raw"),
        'seg_color': os.path.join(base, "images_segmentation_color"),
    }
    if DRAW_2D_BBOX:
        dirs['bbox_2d'] = os.path.join(base, "images_with_2d_bbox")
    if DRAW_3D_BBOX:
        dirs['bbox_3d'] = os.path.join(base, "images_with_3d_bbox")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def next_frame_index(images_dir):
    # resumes numbering from what's already on disk
    if not os.path.isdir(images_dir):
        return 0
    nums = []
    for f in os.listdir(images_dir):
        if f.startswith('frame_') and f.endswith('.png'):
            try:
                nums.append(int(f[len('frame_'):-len('.png')]))
            except ValueError:
                pass
    return (max(nums) + 1) if nums else 0


def log_error(log_path, context, exc):
    with open(log_path, 'a') as f:
        f.write(f"\n--- {context} ---\n")
        f.write(traceback.format_exc())


def destroy_frame_actors(client, actors):
    # ego vehicle and cameras live outside this now, reused per chunk.
    # stop controllers, then destroy everything in one batch - no settle
    # tick in between, matching CARLA's own reference examples. an earlier
    # version of this inserted ticks here on a theory that didn't hold up.
    for actor in actors:
        if actor and actor.is_alive and hasattr(actor, 'type_id') and actor.type_id.startswith('controller.'):
            actor.stop()

    commands = [carla.command.DestroyActor(actor) for actor in actors if actor and actor.is_alive]
    if commands:
        client.apply_batch(commands)


def spawn_ego_and_cameras(world, vehicle_bp, rgb_camera_bp, seg_camera_bp, ego_transform):
    # one ego + two cameras per chunk, reused every frame. not listening
    # yet - each frame does its own brief listen()/stop() window, so a
    # chunk-end destroy() can never race an in-flight sensor callback
    ego_vehicle = spawn_actor(world, vehicle_bp, ego_transform)
    if not ego_vehicle:
        return None, None, None

    for _ in range(15):
        world.tick()
    ego_vehicle.set_simulate_physics(False)

    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.8), carla.Rotation(pitch=-10))
    rgb_cam = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
    seg_cam = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=ego_vehicle)
    world.tick()

    return ego_vehicle, rgb_cam, seg_cam


def destroy_ego_and_cameras(rgb_cam, seg_cam, ego_vehicle):
    if rgb_cam and rgb_cam.is_alive:
        if rgb_cam.is_listening:
            rgb_cam.stop()
        rgb_cam.destroy()
    if seg_cam and seg_cam.is_alive:
        if seg_cam.is_listening:
            seg_cam.stop()
        seg_cam.destroy()
    if ego_vehicle and ego_vehicle.is_alive:
        ego_vehicle.destroy()


def generate_one_frame(client, world, carla_map, blueprints, pedestrian_bps,
                        walker_controller_bp, ego_vehicle, rgb_cam, seg_cam,
                        crossing_segments, spawn_points, crosswalk_centers,
                        scenario, dirs, frame_index, manifest_writer, map_label, bucket):
    # ego_vehicle/rgb_cam/seg_cam belong to the caller, just reposition/read here
    actors = []

    try:
        crossing_kind, occlusion_kind = bucket.split("_", 1)
        is_jaywalk = crossing_kind == "jaywalk"
        wants_occlusion = occlusion_kind == "occluded"

        crossing_start = crossing_end = None

        if is_jaywalk:
            jw_wp = find_jaywalk_waypoint(carla_map, spawn_points, crosswalk_centers,
                                           JAYWALK_CROSSWALK_EXCLUSION_RADIUS)
            if jw_wp is None:
                # fail and retry, don't fall back to a crosswalk
                return False
            dist = random.uniform(8.0, 15.0)
            prev = jw_wp.previous(dist)
            ego_transform = prev[0].transform if prev else jw_wp.transform
            ego_transform.location.z += 0.5
            crossing_start, crossing_end = get_full_road_crossing_segment(jw_wp)
        else:
            segment = random.choice(crossing_segments)
            ego_transform = segment['ego_transform']
            crossing_start, crossing_end = segment['start'], segment['end']

        ego_vehicle.set_transform(ego_transform)
        for _ in range(15):
            world.tick()

        lights_on = should_use_lights(scenario['weather'])
        if lights_on:
            try:
                ego_vehicle.set_light_state(
                    carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                )
            except Exception:
                pass

        counts = resolve_scenario_counts(scenario, MIN_OPTIONAL_CATEGORIES_SATISFIED,
                                          REQUIRE_PEDESTRIAN, MAX_COUNT_REROLLS,
                                          occluded_this_image=wants_occlusion)
        target_occlusion_percent = random.randint(*OCCLUSION_PERCENT_RANGE) if wants_occlusion else 0

        # keeps traffic off the marked crosswalk stripe itself, not the full jaywalk road width
        avoid_segment = (crossing_start, crossing_end) if not is_jaywalk else None

        vehicles_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['vehicles'],
            TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX,
            TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_CAR_VEHICLES,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR,
            avoid_segment=avoid_segment, avoid_segment_dist=CROSSWALK_VEHICLE_EXCLUSION_RADIUS
        )
        actors.extend(vehicles_list)

        trucks_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['trucks'],
            TRUCK_SPAWN_DISTANCE_MIN, TRUCK_SPAWN_DISTANCE_MAX,
            TRUCK_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_TRUCK_VEHICLES,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR,
            avoid_segment=avoid_segment, avoid_segment_dist=CROSSWALK_VEHICLE_EXCLUSION_RADIUS
        )
        actors.extend(trucks_list)

        # cyclists/scooters go last, kept clear of trucks
        cyclists_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['cyclists_scooters'],
            CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN, CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX,
            CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_CYCLIST_SCOOTER_VEHICLES,
            prefer_bike_lane=True, keep_away_from=trucks_list,
            keep_away_dist=MIN_SEPARATION_TRUCK_TO_CYCLIST,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR,
            avoid_segment=avoid_segment, avoid_segment_dist=CROSSWALK_VEHICLE_EXCLUSION_RADIUS
        )
        actors.extend(cyclists_list)

        occluder_candidates = vehicles_list + trucks_list + cyclists_list

        remaining_pedestrians = counts['pedestrians']
        achieved_occlusion_pct = 0.0
        occlusion_within_tolerance = True

        if remaining_pedestrians > 0 and wants_occlusion:
            if not occluder_candidates:
                # nothing to occlude with, retry
                return False

            occluded_ped, occluded_controller, achieved_occlusion_pct, occlusion_within_tolerance = place_pedestrian_for_occlusion(
                world, pedestrian_bps, walker_controller_bp,
                crossing_start, crossing_end, occluder_candidates,
                rgb_cam, target_occlusion_percent,
                samples=OCCLUSION_SEARCH_SAMPLES, progress_range=OCCLUSION_PROGRESS_RANGE,
                grid=OCCLUSION_OVERLAP_GRID, tolerance_percent=OCCLUSION_TOLERANCE_PERCENT,
                speed_range=PEDESTRIAN_WALK_SPEED_RANGE
            )
            if occluded_ped:
                actors.append(occluded_ped)
                if occluded_controller:
                    actors.append(occluded_controller)
                remaining_pedestrians -= 1

            if achieved_occlusion_pct < OCCLUSION_MIN_ACHIEVED_PERCENT:
                # not occluded enough, retry
                return False

            if OCCLUSION_STRICT and not occlusion_within_tolerance:
                # missed its own target by too much, retry
                return False

        for _ in range(remaining_pedestrians):
            ped, controller = spawn_pedestrian_crossing(
                world, pedestrian_bps, walker_controller_bp,
                crossing_start, crossing_end,
                speed_range=PEDESTRIAN_WALK_SPEED_RANGE
            )
            if ped:
                actors.append(ped)
                if controller:
                    actors.append(controller)

        # extra background people on the footpath, not part of the crossing
        if SPAWN_SIDEWALK_PEDESTRIANS:
            for _ in range(random.randint(*SIDEWALK_PEDESTRIAN_COUNT_RANGE)):
                sidewalk_ped, _ = spawn_pedestrian_on_sidewalk(
                    world, pedestrian_bps, ego_transform,
                    SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MIN, SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MAX
                )
                if sidewalk_ped:
                    actors.append(sidewalk_ped)

        warmup_ticks = random.randint(*CROSSING_WARMUP_TICKS_RANGE)
        for _ in range(warmup_ticks):
            world.tick()

        # listen/stop scoped tightly to this capture, not left on across frames
        image_queue = queue.Queue()
        seg_queue = queue.Queue()
        rgb_cam.listen(image_queue.put)
        seg_cam.listen(seg_queue.put)

        for _ in range(10):
            world.tick()

        # drain the backlog from those 10 ticks and take one fresh tick -
        # labels are read right after this, so they need to match the exact
        # tick the saved image came from, not an older queued one. without
        # this, a walking pedestrian's box drifts from several steps of
        # movement between the captured frame and the label read; stationary
        # vehicles never showed it, which is why only the person looked wrong
        while not image_queue.empty():
            image_queue.get()
        while not seg_queue.empty():
            seg_queue.get()
        world.tick()

        rgb_cam.stop()
        seg_cam.stop()

        try:
            rgb_image = image_queue.get(timeout=10.0)
        except queue.Empty:
            rgb_image = None
        try:
            seg_image = seg_queue.get(timeout=10.0)
        except queue.Empty:
            seg_image = None

        if not rgb_image:
            return False

        image_name = f"frame_{frame_index:06d}"
        img_path = os.path.join(dirs['images'], f"{image_name}.png")
        label_path = os.path.join(dirs['labels'], f"{image_name}.txt")

        img_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (rgb_image.height, rgb_image.width, 4))
        img_array = img_array[:, :, :3]
        img_1080p = cv2.resize(img_array, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, img_1080p)

        if seg_image:
            seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
            seg_array = np.reshape(seg_array, (seg_image.height, seg_image.width, 4))
            raw_classes = seg_array[:, :, 2]
            raw_classes_1080p = cv2.resize(raw_classes, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dirs['seg_raw'], f"{image_name}.png"), raw_classes_1080p)

            seg_image.convert(carla.ColorConverter.CityScapesPalette)
            color_seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
            color_seg_array = np.reshape(color_seg_array, (seg_image.height, seg_image.width, 4))
            color_seg_array = color_seg_array[:, :, :3]
            color_seg_1080p = cv2.resize(color_seg_array, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dirs['seg_color'], f"{image_name}.png"), color_seg_1080p)

        image_w, image_h = 3840, 2160
        yolo_annotations = []
        valid_detections = []

        # cyclists/scooters get their own pass below, skip here
        vehicle_results = get_all_vehicle_bboxes(world, rgb_cam, max_distance=MAX_RENDER_DISTANCE)
        for v_res in vehicle_results:
            actor_obj = v_res['actor_obj']
            if v_res['id'] == ego_vehicle.id:
                continue
            type_id = getattr(actor_obj, 'type_id', None)
            if type_id in ALLOWED_CYCLIST_SCOOTER_VEHICLES:
                continue
            label_name = 'Truck' if (type_id in ALLOWED_TRUCK_VEHICLES and SEPARATE_TRUCK_CLASS) else 'Vehicle'
            class_id = CLASS_MAP.get(label_name, CLASS_MAP.get('Vehicle'))
            valid_detections.append((class_id, label_name, v_res['bbox'], actor_obj))

        all_world_actors = world.get_actors()
        for ped in all_world_actors.filter('walker.pedestrian.*'):
            if ped and ped.is_alive:
                if not is_visible_and_not_occluded(world, rgb_cam, ped, MAX_RENDER_DISTANCE):
                    continue
                bbox_2d = get_2d_bbox_from_3d(ped, rgb_cam)
                if bbox_2d:
                    class_id = CLASS_MAP.get('Pedestrian')
                    valid_detections.append((class_id, 'Pedestrian', bbox_2d, ped))

        for bike in all_world_actors.filter('vehicle.*'):
            if bike.type_id not in ALLOWED_CYCLIST_SCOOTER_VEHICLES:
                continue
            if not is_visible_and_not_occluded(world, rgb_cam, bike, MAX_RENDER_DISTANCE):
                continue
            bbox_2d = get_2d_bbox_from_3d(bike, rgb_cam)
            if bbox_2d:
                if bike.type_id in ALLOWED_ESCOOTER_VEHICLES:
                    label_name = 'E-Scooter + Rider'
                elif bike.type_id in ALLOWED_TWO_WHEELER_VEHICLES:
                    label_name = 'Motorcycle'
                else:
                    label_name = 'Cyclist'
                class_id = CLASS_MAP.get(label_name)
                if class_id is not None:
                    valid_detections.append((class_id, label_name, bbox_2d, bike))

        annotations_3d = []
        for class_id, label_name, bbox_2d, actor_obj in valid_detections:
            yolo_bbox = carla_to_yolo(bbox_2d, image_w, image_h)
            yolo_annotations.append(
                f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            )

            actor_transform = actor_obj.get_transform() if hasattr(actor_obj, 'get_transform') else getattr(actor_obj, 'transform', None)
            if actor_transform and hasattr(actor_obj, 'bounding_box'):
                bb = actor_obj.bounding_box
                if hasattr(actor_obj, 'get_transform'):
                    world_bb_center = actor_transform.transform(bb.location)
                    rot = actor_transform.rotation
                else:
                    world_bb_center = bb.location
                    rot = bb.rotation

                cam_transform = rgb_cam.get_transform()
                world_to_camera = np.array(cam_transform.get_inverse_matrix())
                world_point = np.array([world_bb_center.x, world_bb_center.y, world_bb_center.z, 1.0])
                p_camera = np.dot(world_to_camera, world_point)
                extent = bb.extent

                detection_3d = {
                    "class_id": class_id,
                    "label_name": label_name,
                    "2d_bbox_pixel": bbox_2d,
                    "3d_world_location": {"x": world_bb_center.x, "y": world_bb_center.y, "z": world_bb_center.z},
                    "3d_world_rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                    "extent_dimensions": {
                        "x_half_size": extent.x, "y_half_size": extent.y, "z_half_size": extent.z,
                        "height": extent.z * 2.0, "width": extent.y * 2.0, "length": extent.x * 2.0
                    },
                    "camera_relative_location": {"x": p_camera[1], "y": -p_camera[2], "z": p_camera[0]},
                }
                if label_name == 'Pedestrian':
                    detection_3d["target_occlusion_percent"] = target_occlusion_percent
                    detection_3d["achieved_occlusion_percent"] = round(achieved_occlusion_pct, 1)
                    detection_3d["occlusion_within_tolerance"] = occlusion_within_tolerance
                    detection_3d["is_jaywalking"] = is_jaywalk
                    detection_3d["map"] = map_label
                annotations_3d.append(detection_3d)

        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

        with open(os.path.join(dirs['labels_3d'], f"{image_name}.json"), 'w') as f:
            json.dump(annotations_3d, f, indent=4)

        if DRAW_2D_BBOX:
            img_2d = img_1080p.copy()
            for _, label_name, bbox_2d, _ in valid_detections:
                x_min, y_min, x_max, y_max = bbox_2d
                x_min, y_min, x_max, y_max = x_min // 2, y_min // 2, x_max // 2, y_max // 2
                cv2.rectangle(img_2d, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
                cv2.putText(img_2d, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.imwrite(os.path.join(dirs['bbox_2d'], f"{image_name}_bbox_2d.png"), img_2d)

        if DRAW_3D_BBOX:
            img_3d = img_1080p.copy()
            for _, _, _, actor_obj in valid_detections:
                draw_3d_bbox_on_image(img_3d, actor_obj, rgb_cam, color=(0, 255, 255), thickness=2)
            cv2.imwrite(os.path.join(dirs['bbox_3d'], f"{image_name}_bbox_3d.png"), img_3d)

        manifest_writer.writerow([
            image_name, scenario['name'], map_label, bucket, counts['pedestrians'], counts['vehicles'],
            counts['cyclists_scooters'], counts['trucks'], target_occlusion_percent,
            f"{achieved_occlusion_pct:.1f}", occlusion_within_tolerance, is_jaywalk
        ])

        return True

    finally:
        # always runs, keeps actors from piling up
        destroy_frame_actors(client, actors)


def cleanup_stale_ego_actors(client, world, vehicle_bp):
    # a previous run that aborted (uncaught native crash) never tears
    # down its actors - the server keeps them. destroys any leftover
    # actor matching our ego vehicle blueprint before this run spawns
    # its own, so a fresh run doesn't fail to spawn (or touch a dangling
    # actor) because of a leftover ego from a crashed previous process
    stale = [a for a in world.get_actors() if a.type_id == vehicle_bp.id]
    if stale:
        print(f"  found {len(stale)} leftover ego vehicle(s) from a previous run, cleaning up")
        client.apply_batch([carla.command.DestroyActor(a) for a in stale])
        world.tick()


def setup_actor_blueprints(blueprints):
    vehicle_bp = blueprints.filter('my_car_xyz')[0]
    pedestrian_bps = blueprints.filter('walker.pedestrian.*')
    walker_controller_bp = blueprints.find('controller.ai.walker')

    rgb_camera_bp = blueprints.find('sensor.camera.rgb')
    rgb_camera_bp.set_attribute('image_size_x', '3840')
    rgb_camera_bp.set_attribute('image_size_y', '2160')
    rgb_camera_bp.set_attribute('fov', '90')
    rgb_camera_bp.set_attribute('sensor_tick', '0.05')
    rgb_camera_bp.set_attribute('lens_circle_multiplier', '1.5')
    rgb_camera_bp.set_attribute('lens_circle_falloff', '4.0')
    rgb_camera_bp.set_attribute('chromatic_aberration_intensity', '0.5')
    rgb_camera_bp.set_attribute('chromatic_aberration_offset', '0')

    seg_camera_bp = blueprints.find('sensor.camera.semantic_segmentation')
    seg_camera_bp.set_attribute('image_size_x', '3840')
    seg_camera_bp.set_attribute('image_size_y', '2160')
    seg_camera_bp.set_attribute('fov', '90')
    seg_camera_bp.set_attribute('sensor_tick', '0.05')

    return vehicle_bp, pedestrian_bps, walker_controller_bp, rgb_camera_bp, seg_camera_bp


def scan_map_geometry(world):
    carla_map = world.get_map()
    crossing_segments = get_crosswalk_crossing_segments(carla_map)
    crosswalk_centers = get_crosswalk_centers(carla_map)
    spawn_points = carla_map.get_spawn_points()
    return carla_map, crossing_segments, crosswalk_centers, spawn_points


def apply_sync_settings(world):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return settings


def switch_map(client, map_name):
    print(f"switching map: {map_name}")
    world = client.load_world(map_name)
    time.sleep(2.0)
    apply_sync_settings(world)
    return world


def connect_to_server(map_name_hint=None):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if map_name_hint:
        current_map = world.get_map().name.split('/')[-1]
        if current_map != map_name_hint:
            world = switch_map(client, map_name_hint)

    settings = apply_sync_settings(world)
    blueprints = world.get_blueprint_library()
    actor_bps = setup_actor_blueprints(blueprints)
    return client, world, settings, blueprints, actor_bps


def reconnect_with_retries(map_name_hint, log_path):
    # all progress is already on disk, safe to keep retrying a dead connection
    attempt = 0
    while RECONNECT_MAX_ATTEMPTS is None or attempt < RECONNECT_MAX_ATTEMPTS:
        attempt += 1
        limit_str = f"/{RECONNECT_MAX_ATTEMPTS}" if RECONNECT_MAX_ATTEMPTS else ""
        print(f"  connection to CARLA looks dead, retrying ({attempt}{limit_str}) "
              f"in {RECONNECT_RETRY_DELAY_SECONDS:.0f}s - restart the server now if it crashed")
        time.sleep(RECONNECT_RETRY_DELAY_SECONDS)
        try:
            return connect_to_server(map_name_hint)
        except Exception as exc:
            log_error(log_path, f"reconnect attempt {attempt} failed", exc)

    raise RuntimeError(f"could not reconnect to CARLA after {attempt} attempts, giving up for now")


def split_evenly(total, n):
    base = total // n
    remainder = total % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


def resolve_map_chunks(number_images_to_generate):
    # splits across MAP_ROTATION maps if set, else one chunk on the current map
    if not MAP_ROTATION:
        return [(None, number_images_to_generate)]
    counts = split_evenly(number_images_to_generate, len(MAP_ROTATION))
    return list(zip(MAP_ROTATION, counts))


def write_run_summary(manifest_path, summary_path):
    # tallies the manifest into per-scenario totals, written to summary.txt
    if not os.path.exists(manifest_path):
        return

    totals = {}
    with open(manifest_path, newline='') as f:
        for row in csv.DictReader(f):
            name = row['scenario']
            bucket = totals.setdefault(name, {
                'total': 0, 'jaywalk': 0, 'crosswalk': 0,
                'occluded': 0, 'clear': 0, 'maps': {}
            })
            bucket['total'] += 1
            if row['is_jaywalking'] == 'True':
                bucket['jaywalk'] += 1
            else:
                bucket['crosswalk'] += 1
            if float(row['target_occlusion_percent']) > 0:
                bucket['occluded'] += 1
            else:
                bucket['clear'] += 1
            map_name = row.get('map') or 'current'
            bucket['maps'][map_name] = bucket['maps'].get(map_name, 0) + 1

    lines = ["Run summary", "=" * 60]
    for name, bucket in totals.items():
        lines.append(f"\n{name}: {bucket['total']} images")
        lines.append(f"  crosswalk: {bucket['crosswalk']}   jaywalk: {bucket['jaywalk']}")
        lines.append(f"  clear: {bucket['clear']}   occluded: {bucket['occluded']}")
        if len(bucket['maps']) > 1 or 'current' not in bucket['maps']:
            map_breakdown = ", ".join(f"{m}={c}" for m, c in sorted(bucket['maps'].items()))
            lines.append(f"  maps: {map_breakdown}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(summary_path, 'w') as f:
        f.write(report + "\n")


def main(cli_args=None):
    if cli_args is None:
        cli_args = parse_args()

    active_split = resolve_split(cli_args.mode)
    if cli_args.count is not None:
        for scenario in SCENARIOS:
            scenario['number_images_to_generate'] = cli_args.count

    if cli_args.resume == "latest":
        output_dir = resolve_latest_run_dir()
        if not output_dir:
            print(f"no previous run found under {OUTPUT_BASE_DIR}/ to resume")
            return
        print(f"resuming latest run: {output_dir}")
    else:
        output_dir = cli_args.resume or resolve_output_dir(active_split, cli_args.mode)
    os.makedirs(output_dir, exist_ok=True)

    client, world, settings, blueprints, actor_bps = connect_to_server(MAP_NAME if not MAP_ROTATION else None)
    vehicle_bp, pedestrian_bps, walker_controller_bp, rgb_camera_bp, seg_camera_bp = actor_bps
    cleanup_stale_ego_actors(client, world, vehicle_bp)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_is_new = not os.path.exists(manifest_path)
    manifest_file = open(manifest_path, 'a', newline='')
    manifest_writer = csv.writer(manifest_file)
    if manifest_is_new:
        manifest_writer.writerow([
            "image_name", "scenario", "map", "bucket", "pedestrians", "vehicles",
            "cyclists_scooters", "trucks", "target_occlusion_percent",
            "achieved_occlusion_percent", "occlusion_within_tolerance", "is_jaywalking"
        ])

    log_path = os.path.join(output_dir, "run_log.txt")

    try:
        for scenario in SCENARIOS:
            dirs = scenario_dirs(output_dir, scenario['name'])
            frame_index = next_frame_index(dirs['images'])
            plan = load_or_build_plan(
                dirs['base'], scenario['number_images_to_generate'], active_split,
                scenario.get('allow_jaywalk', False), scenario.get('allow_occlusion', False)
            )

            if total_remaining(plan) == 0:
                print(f"\nscenario '{scenario['name']}': plan already complete, skipping")
                continue

            map_chunks = resolve_map_chunks(total_remaining(plan))

            for map_name, chunk_target in map_chunks:
                if chunk_target <= 0 or total_remaining(plan) == 0:
                    continue

                if map_name and world.get_map().name.split('/')[-1] != map_name:
                    world = switch_map(client, map_name)
                    settings = apply_sync_settings(world)
                    blueprints = world.get_blueprint_library()
                    vehicle_bp, pedestrian_bps, walker_controller_bp, rgb_camera_bp, seg_camera_bp = setup_actor_blueprints(blueprints)

                carla_map, crossing_segments, crosswalk_centers, spawn_points = scan_map_geometry(world)
                map_label = map_name or world.get_map().name.split('/')[-1]

                if not crossing_segments:
                    print(f"no crosswalks found on {map_label}, skipping this chunk "
                          f"({chunk_target} images not generated)")
                    continue

                world.set_weather(scenario['weather'])
                if should_use_lights(scenario['weather']):
                    try:
                        light_manager = world.get_lightmanager()
                        if light_manager:
                            light_manager.turn_on(light_manager.get_all_lights())
                    except Exception:
                        pass

                # one ego + cameras per chunk, reused every frame (fixes the core dump)
                ego_vehicle, rgb_cam, seg_cam = spawn_ego_and_cameras(
                    world, vehicle_bp, rgb_camera_bp, seg_camera_bp, crossing_segments[0]['ego_transform']
                )
                if not ego_vehicle:
                    print(f"  could not spawn ego vehicle/cameras on {map_label}, skipping this chunk "
                          f"({chunk_target} images not generated)")
                    continue

                try:
                    chunk_generated = 0
                    attempts = 0
                    # ceiling so a broken chunk can't spin forever
                    max_total_attempts = chunk_target * 3 + 20

                    print(f"\nscenario '{scenario['name']}' on {map_label}: target {chunk_target} images "
                          f"({total_remaining(plan)} total remaining in plan), resuming at frame {frame_index}")

                    while chunk_generated < chunk_target and total_remaining(plan) > 0 and attempts < max_total_attempts:
                        attempts += 1
                        bucket = pick_next_bucket(plan)
                        try:
                            saved = generate_one_frame(
                                client, world, carla_map, blueprints, pedestrian_bps,
                                walker_controller_bp, ego_vehicle, rgb_cam, seg_cam,
                                crossing_segments, spawn_points, crosswalk_centers,
                                scenario, dirs, frame_index, manifest_writer, map_label, bucket
                            )
                            manifest_file.flush()
                            if saved:
                                consume_plan_item(dirs['base'], plan, bucket)
                                chunk_generated += 1
                                frame_index += 1
                                if chunk_generated % 25 == 0:
                                    print(f"  {chunk_generated}/{chunk_target}")

                            # breathing room before the next spawn batch
                            for _ in range(POST_DESTROY_SETTLE_TICKS):
                                world.tick()
                            time.sleep(POST_DESTROY_SETTLE_SECONDS)

                            # caps how long one ego/camera set stays alive,
                            # rather than reusing it for an entire chunk.
                            # spawn the replacement BEFORE destroying the
                            # current one, so a failed respawn just skips
                            # this round instead of leaving no valid ego
                            if saved and chunk_generated % EGO_REFRESH_EVERY_N_IMAGES == 0:
                                new_ego, new_rgb_cam, new_seg_cam = spawn_ego_and_cameras(
                                    world, vehicle_bp, rgb_camera_bp, seg_camera_bp,
                                    random.choice(crossing_segments)['ego_transform']
                                )
                                if new_ego:
                                    old_ego, old_rgb_cam, old_seg_cam = ego_vehicle, rgb_cam, seg_cam
                                    ego_vehicle, rgb_cam, seg_cam = new_ego, new_rgb_cam, new_seg_cam
                                    destroy_ego_and_cameras(old_rgb_cam, old_seg_cam, old_ego)

                        except Exception as exc:
                            log_error(log_path, f"scenario={scenario['name']} map={map_label} "
                                                 f"bucket={bucket} frame_index={frame_index} attempt={attempts}", exc)

                            if looks_like_connection_error(exc):
                                try:
                                    # old ego/cams are on a dead connection, ignore cleanup errors
                                    try:
                                        destroy_ego_and_cameras(rgb_cam, seg_cam, ego_vehicle)
                                    except Exception:
                                        pass
                                    client, world, settings, blueprints, actor_bps = reconnect_with_retries(map_label, log_path)
                                    vehicle_bp, pedestrian_bps, walker_controller_bp, rgb_camera_bp, seg_camera_bp = actor_bps
                                    cleanup_stale_ego_actors(client, world, vehicle_bp)
                                    world.set_weather(scenario['weather'])
                                    carla_map, crossing_segments, crosswalk_centers, spawn_points = scan_map_geometry(world)
                                    ego_vehicle, rgb_cam, seg_cam = spawn_ego_and_cameras(
                                        world, vehicle_bp, rgb_camera_bp, seg_camera_bp, crossing_segments[0]['ego_transform']
                                    )
                                    if not ego_vehicle:
                                        raise RuntimeError("could not respawn ego vehicle/cameras after reconnecting")
                                    print("  reconnected, resuming from the same plan/frame position")
                                except Exception as reconnect_exc:
                                    log_error(log_path, "gave up trying to reconnect", reconnect_exc)
                                    print(f"  could not reconnect to CARLA - stopping here. "
                                          f"Nothing already saved is lost: re-run with "
                                          f"--resume {output_dir} once the server is back up.")
                                    raise
                            else:
                                print(f"  frame failed, logged to {log_path}, continuing")
                                time.sleep(0.5)

                    if chunk_generated < chunk_target:
                        print(f"  only got {chunk_generated}/{chunk_target} for '{scenario['name']}' on {map_label} "
                              f"after {attempts} attempts, moving on")
                finally:
                    try:
                        destroy_ego_and_cameras(rgb_cam, seg_cam, ego_vehicle)
                    except Exception:
                        pass
    finally:
        manifest_file.close()
        # don't let a dead connection's cleanup mask the real error above
        try:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

    write_run_summary(manifest_path, os.path.join(output_dir, "summary.txt"))


if __name__ == "__main__":
    main()
