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
    MAP_NAME, MAP_ROTATION, OUTPUT_DIR, DRAW_2D_BBOX, DRAW_3D_BBOX,
    SEPARATE_TRUCK_CLASS, CLASS_MAP, SCENARIOS,
    REQUIRE_PEDESTRIAN, MIN_OPTIONAL_CATEGORIES_SATISFIED, MAX_COUNT_REROLLS,
    OCCLUSION_SEARCH_SAMPLES, OCCLUSION_PROGRESS_RANGE, OCCLUSION_OVERLAP_GRID,
    OCCLUSION_TOLERANCE_PERCENT, OCCLUSION_PROBABILITY, OCCLUSION_STRICT,
    PEDESTRIAN_WALK_SPEED_RANGE, CROSSING_WARMUP_TICKS_RANGE, MAX_RENDER_DISTANCE,
    JAYWALK_CROSSWALK_EXCLUSION_RADIUS, JAYWALK_PROBABILITY, FORCE_CROSSING_MODE,
    TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX, TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE,
    TRUCK_SPAWN_DISTANCE_MIN, TRUCK_SPAWN_DISTANCE_MAX, TRUCK_SPAWN_LATERAL_RANGE,
    CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN, CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX, CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE,
    MIN_SEPARATION_BETWEEN_VEHICLES, MIN_SEPARATION_TRUCK_TO_CYCLIST, MAX_SPAWN_ATTEMPTS_PER_ACTOR,
    ALLOWED_CAR_VEHICLES, ALLOWED_TRUCK_VEHICLES, ALLOWED_CYCLIST_SCOOTER_VEHICLES,
    ALLOWED_ESCOOTER_VEHICLES, ALLOWED_TWO_WHEELER_VEHICLES,
)
from utils import (
    get_2d_bbox_from_3d, is_visible_and_not_occluded, carla_to_yolo,
    get_all_vehicle_bboxes, draw_3d_bbox_on_image, should_use_lights,
    get_crosswalk_crossing_segments,
    get_crosswalk_centers, find_jaywalk_waypoint, get_full_road_crossing_segment,
    resolve_scenario_counts,
)
from spawner import spawn_actor, spawn_pedestrian_crossing, spawn_traffic_vehicles
from occlusion import place_pedestrian_for_occlusion


def scenario_dirs(scenario_name):
    base = os.path.join(OUTPUT_DIR, scenario_name)
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
    # resumes numbering from whatever's already on disk instead of starting
    # over at 0 and overwriting a previous run's images
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


def destroy_frame_actors(client, actors, ego_vehicle, rgb_cam, seg_cam):
    # the cameras get destroyed directly (they're sensors, not batch-friendly
    # while listening), everything else - including walker AI controllers,
    # which are attached to their pedestrian and were previously getting
    # skipped here, which is what caused actors to pile up run after run -
    # goes through the batch destroy below.
    if rgb_cam and rgb_cam.is_alive:
        if rgb_cam.is_listening:
            rgb_cam.stop()
        rgb_cam.destroy()
    if seg_cam and seg_cam.is_alive:
        if seg_cam.is_listening:
            seg_cam.stop()
        seg_cam.destroy()

    for actor in actors:
        if actor and actor.is_alive and hasattr(actor, 'type_id') and actor.type_id.startswith('controller.'):
            actor.stop()

    commands = []
    for actor in actors:
        if not actor or not actor.is_alive:
            continue
        if rgb_cam and actor.id == rgb_cam.id:
            continue
        if seg_cam and actor.id == seg_cam.id:
            continue
        commands.append(carla.command.DestroyActor(actor))
    if ego_vehicle and ego_vehicle.is_alive:
        commands.append(carla.command.DestroyActor(ego_vehicle))
    if commands:
        client.apply_batch(commands)


def generate_one_frame(client, world, carla_map, blueprints, vehicle_bp, pedestrian_bps,
                        walker_controller_bp, rgb_camera_bp, seg_camera_bp,
                        crossing_segments, spawn_points, crosswalk_centers,
                        scenario, dirs, frame_index, manifest_writer, map_label):
    ego_vehicle = None
    rgb_cam = None
    seg_cam = None
    actors = []

    try:
        if FORCE_CROSSING_MODE == "crosswalk_only":
            is_jaywalk = False
        elif FORCE_CROSSING_MODE == "jaywalk_only":
            is_jaywalk = True
        else:
            is_jaywalk = scenario.get('allow_jaywalk', False) and random.random() < JAYWALK_PROBABILITY

        crossing_start = crossing_end = None

        if is_jaywalk:
            jw_wp = find_jaywalk_waypoint(carla_map, spawn_points, crosswalk_centers,
                                           JAYWALK_CROSSWALK_EXCLUSION_RADIUS)
            if jw_wp is not None:
                dist = random.uniform(8.0, 15.0)
                prev = jw_wp.previous(dist)
                ego_transform = prev[0].transform if prev else jw_wp.transform
                ego_transform.location.z += 0.5
                crossing_start, crossing_end = get_full_road_crossing_segment(jw_wp)
            elif FORCE_CROSSING_MODE == "jaywalk_only":
                # forced jaywalk-only mode: don't silently swap to a legal
                # crossing just because this attempt couldn't find a spot -
                # fail the attempt instead, the outer loop will retry
                return False
            else:
                # couldn't find a spot far enough from every crosswalk this
                # time, fall back to a normal legal crossing instead
                is_jaywalk = False

        if not is_jaywalk:
            segment = random.choice(crossing_segments)
            ego_transform = segment['ego_transform']
            crossing_start, crossing_end = segment['start'], segment['end']

        ego_vehicle = spawn_actor(world, vehicle_bp, ego_transform)
        if not ego_vehicle:
            return False

        lights_on = should_use_lights(scenario['weather'])
        if lights_on:
            try:
                ego_vehicle.set_light_state(
                    carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                )
            except Exception:
                pass

        for _ in range(15):
            world.tick()
        ego_vehicle.set_simulate_physics(False)

        counts = resolve_scenario_counts(scenario, MIN_OPTIONAL_CATEGORIES_SATISFIED,
                                          REQUIRE_PEDESTRIAN, MAX_COUNT_REROLLS,
                                          occlusion_probability=OCCLUSION_PROBABILITY)
        has_other_actors = counts['vehicles'] + counts['cyclists_scooters'] + counts['trucks'] > 0
        target_occlusion_percent = counts['occlusion_percent'] if has_other_actors else 0

        vehicles_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['vehicles'],
            TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX,
            TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_CAR_VEHICLES,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR
        )
        actors.extend(vehicles_list)

        trucks_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['trucks'],
            TRUCK_SPAWN_DISTANCE_MIN, TRUCK_SPAWN_DISTANCE_MAX,
            TRUCK_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_TRUCK_VEHICLES,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR
        )
        actors.extend(trucks_list)

        # cyclists/scooters go last so they can be kept clear of trucks -
        # a truck shouldn't spawn right on top of a bike lane in real traffic
        cyclists_list = spawn_traffic_vehicles(
            world, blueprints, ego_transform, counts['cyclists_scooters'],
            CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN, CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX,
            CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE, lights_on, ALLOWED_CYCLIST_SCOOTER_VEHICLES,
            prefer_bike_lane=True, keep_away_from=trucks_list,
            keep_away_dist=MIN_SEPARATION_TRUCK_TO_CYCLIST,
            min_spacing=MIN_SEPARATION_BETWEEN_VEHICLES, max_attempts=MAX_SPAWN_ATTEMPTS_PER_ACTOR
        )
        actors.extend(cyclists_list)

        occluder_candidates = vehicles_list + trucks_list + cyclists_list

        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.8), carla.Rotation(pitch=-10))
        rgb_cam = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
        actors.append(rgb_cam)
        seg_cam = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=ego_vehicle)
        actors.append(seg_cam)
        world.tick()

        remaining_pedestrians = counts['pedestrians']
        achieved_occlusion_pct = 0.0
        occlusion_within_tolerance = True

        if remaining_pedestrians > 0 and target_occlusion_percent > 0 and occluder_candidates:
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

            if OCCLUSION_STRICT and not occlusion_within_tolerance:
                # targeted occlusion but couldn't actually place it close
                # enough to the target - don't save a frame that claims to
                # be occluded but isn't, retry with a fresh roll instead
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

        warmup_ticks = random.randint(*CROSSING_WARMUP_TICKS_RANGE)
        for _ in range(warmup_ticks):
            world.tick()

        image_queue = queue.Queue()
        seg_queue = queue.Queue()
        rgb_cam.listen(image_queue.put)
        seg_cam.listen(seg_queue.put)

        for _ in range(10):
            world.tick()

        while not image_queue.empty():
            image_queue.get()
        while not seg_queue.empty():
            seg_queue.get()

        world.tick()

        try:
            rgb_image = image_queue.get(timeout=10.0)
        except queue.Empty:
            rgb_image = None
        try:
            seg_image = seg_queue.get(timeout=10.0)
        except queue.Empty:
            seg_image = None

        if rgb_cam.is_listening:
            rgb_cam.stop()
        if seg_cam.is_listening:
            seg_cam.stop()

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

        # vehicles + static parked cars, minus cyclists/scooters which get
        # their own pass below so they aren't double-counted as "Vehicle" too
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
            image_name, scenario['name'], map_label, counts['pedestrians'], counts['vehicles'],
            counts['cyclists_scooters'], counts['trucks'], target_occlusion_percent,
            f"{achieved_occlusion_pct:.1f}", occlusion_within_tolerance, is_jaywalk
        ])

        return True

    finally:
        # runs whether the frame was saved, skipped, or blew up halfway
        # through - this is what actually prevents actors from piling up
        destroy_frame_actors(client, actors, ego_vehicle, rgb_cam, seg_cam)


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


def split_evenly(total, n):
    base = total // n
    remainder = total % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


def resolve_map_chunks(number_images_to_generate):
    """
    Splits a scenario's image count across MAP_ROTATION if one is set, so
    the same weather scenario gets built from several different maps
    instead of repeating the same streets over and over. Returns a list
    of (map_name_or_None, image_count) pairs; map_name is None when no
    rotation is configured, meaning "don't touch the current map".
    """
    if not MAP_ROTATION:
        return [(None, number_images_to_generate)]
    counts = split_evenly(number_images_to_generate, len(MAP_ROTATION))
    return list(zip(MAP_ROTATION, counts))


def write_run_summary(manifest_path, summary_path):
    """
    Reads the manifest back and tallies what actually got generated, so
    "did I get the mix I asked for" is a quick look instead of a guess -
    total per scenario, crosswalk vs jaywalk, occluded vs clear, and the
    spread across maps if MAP_ROTATION was used.
    """
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


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if not MAP_ROTATION and MAP_NAME:
        current_map = world.get_map().name.split('/')[-1]
        if current_map != MAP_NAME:
            world = switch_map(client, MAP_NAME)

    settings = apply_sync_settings(world)
    blueprints = world.get_blueprint_library()
    vehicle_bp, pedestrian_bps, walker_controller_bp, rgb_camera_bp, seg_camera_bp = setup_actor_blueprints(blueprints)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_is_new = not os.path.exists(manifest_path)
    manifest_file = open(manifest_path, 'a', newline='')
    manifest_writer = csv.writer(manifest_file)
    if manifest_is_new:
        manifest_writer.writerow([
            "image_name", "scenario", "map", "pedestrians", "vehicles",
            "cyclists_scooters", "trucks", "target_occlusion_percent",
            "achieved_occlusion_percent", "occlusion_within_tolerance", "is_jaywalking"
        ])

    log_path = os.path.join(OUTPUT_DIR, "run_log.txt")

    try:
        for scenario in SCENARIOS:
            dirs = scenario_dirs(scenario['name'])
            frame_index = next_frame_index(dirs['images'])
            map_chunks = resolve_map_chunks(scenario['number_images_to_generate'])

            for map_name, chunk_target in map_chunks:
                if chunk_target <= 0:
                    continue

                if map_name:
                    current_map_name = world.get_map().name.split('/')[-1]
                    if current_map_name != map_name:
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

                generated = 0
                attempts = 0
                # a hard ceiling so one broken chunk (bad map, no valid
                # spawns, etc) can't spin forever without ever finishing
                max_total_attempts = chunk_target * 3 + 20

                print(f"\nscenario '{scenario['name']}' on {map_label}: "
                      f"target {chunk_target} images, resuming at frame {frame_index}")

                while generated < chunk_target and attempts < max_total_attempts:
                    attempts += 1
                    try:
                        saved = generate_one_frame(
                            client, world, carla_map, blueprints, vehicle_bp, pedestrian_bps,
                            walker_controller_bp, rgb_camera_bp, seg_camera_bp,
                            crossing_segments, spawn_points, crosswalk_centers,
                            scenario, dirs, frame_index, manifest_writer, map_label
                        )
                        manifest_file.flush()
                        if saved:
                            generated += 1
                            frame_index += 1
                            if generated % 25 == 0:
                                print(f"  {generated}/{chunk_target}")
                    except Exception as exc:
                        log_error(log_path, f"scenario={scenario['name']} map={map_label} "
                                             f"frame_index={frame_index} attempt={attempts}", exc)
                        print(f"  frame failed, logged to {log_path}, continuing")
                        time.sleep(0.5)

                if generated < chunk_target:
                    print(f"  only got {generated}/{chunk_target} for '{scenario['name']}' on {map_label} "
                          f"after {attempts} attempts, moving on")
    finally:
        manifest_file.close()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    write_run_summary(manifest_path, os.path.join(OUTPUT_DIR, "summary.txt"))


if __name__ == "__main__":
    main()
