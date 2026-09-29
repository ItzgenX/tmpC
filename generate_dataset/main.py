import carla
import os
import random
import time
import queue
import cv2
import numpy as np
import json


# --- Import from our custom modules ---
from config import *
from utils import get_2d_bbox_from_3d, is_visible_and_not_occluded, get_actor_label, carla_to_yolo, get_all_vehicle_bboxes, draw_3d_bbox_on_image
from spawner import spawn_actor, spawn_pedestrian_on_sidewalk, spawn_traffic_vehicles


def main():


    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels_3d"), exist_ok=True)
    if DRAW_2D_BBOX:
        os.makedirs(os.path.join(OUTPUT_DIR, "images_with_2d_bbox"), exist_ok=True)
    if DRAW_3D_BBOX:
        os.makedirs(os.path.join(OUTPUT_DIR, "images_with_3d_bbox"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images_segmentation_raw"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images_segmentation_color"), exist_ok=True)


    # --- SETUP CARLA ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)


    world = client.get_world()
    blueprints = world.get_blueprint_library()


    settings = world.get_settings()
    print(f"Setting weather to: {CURRENT_WEATHER}")
    world.set_weather(WEATHER_CONDITIONS[CURRENT_WEATHER])
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)


    if CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
        print("Turning on street lights...")
        try:
            light_manager = world.get_lightmanager()
            if light_manager:
                lights = light_manager.get_all_lights()
                light_manager.turn_on(lights)
        except Exception as e:
            pass


    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('my_car_xyz')[0]
    pedestrian_bps = blueprints.filter('walker.pedestrian.*')


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


    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        print("Error: No spawn points found on the map.")
        return


    ego_vehicle = None
    actors = []
    try:
        for i in range(NUM_IMAGES):
            print(f"Generating image {i+1}/{NUM_IMAGES}")
           
            ego_transform = random.choice(spawn_points)
            ego_vehicle = spawn_actor(world, vehicle_bp, ego_transform)
            if not ego_vehicle:
                continue
               
            if CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
                try:
                    ego_vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                except Exception: pass


            for _ in range(15): world.tick()
            ego_vehicle.set_simulate_physics(False)


            actors = []


            for _ in range(NUM_PEDESTRIANS):
                ped, controller = spawn_pedestrian_on_sidewalk(world, pedestrian_bps, ego_transform, PEDESTRIAN_SPAWN_DISTANCE_MIN, PEDESTRIAN_SPAWN_DISTANCE_MAX)
                if ped:
                    actors.append(ped)
                    if controller:
                        actors.append(controller)


            traffic_vehicles = spawn_traffic_vehicles(
                world, blueprints, ego_transform, NUM_TRAFFIC_VEHICLES,
                TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX, TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE,
                CURRENT_WEATHER, ALLOWED_TRAFFIC_VEHICLES
            )
            actors.extend(traffic_vehicles)


            camera_transform = carla.Transform(carla.Location(x=0.8, z=1.8), carla.Rotation(pitch=-10))
            rgb_cam = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
            actors.append(rgb_cam)


            seg_cam = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=ego_vehicle)
            actors.append(seg_cam)
         
            for _ in range(50): world.tick()


            image_queue = queue.Queue()
            seg_queue = queue.Queue()
            rgb_cam.listen(image_queue.put)
            seg_cam.listen(seg_queue.put)
           
            for _ in range(10): world.tick()
           
            while not image_queue.empty(): image_queue.get()
            while not seg_queue.empty(): seg_queue.get()
               
            world.tick()
           
            try: rgb_image = image_queue.get(timeout=10.0)
            except queue.Empty: rgb_image = None


            try: seg_image = seg_queue.get(timeout=10.0)
            except queue.Empty: seg_image = None
               
            if rgb_cam.is_listening: rgb_cam.stop()
            if seg_cam.is_listening: seg_cam.stop()


            if rgb_image:
                frame_number = START_FRAME_NUMBER + i
                image_name = f"frame_{frame_number:06d}"
                img_path = os.path.join(OUTPUT_DIR, "images", f"{image_name}.png")
                label_path = os.path.join(OUTPUT_DIR, "labels", f"{image_name}.txt")
               
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
                    raw_seg_path = os.path.join(OUTPUT_DIR, "images_segmentation_raw", f"{image_name}.png")
                    cv2.imwrite(raw_seg_path, raw_classes_1080p)
                   
                    seg_image.convert(carla.ColorConverter.CityScapesPalette)
                    color_seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
                    color_seg_array = np.reshape(color_seg_array, (seg_image.height, seg_image.width, 4))
                    color_seg_array = color_seg_array[:, :, :3]
                    color_seg_1080p = cv2.resize(color_seg_array, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                    color_seg_path = os.path.join(OUTPUT_DIR, "images_segmentation_color", f"{image_name}.png")
                    cv2.imwrite(color_seg_path, color_seg_1080p)
               
                image_w = 3840
                image_h = 2160
                yolo_annotations = []
               
                all_world_actors = world.get_actors()
               
                valid_detections = []


                # 1. Get ALL vehicles (dynamic + static parked cars) using the robust unified function
                vehicle_results = get_all_vehicle_bboxes(world, rgb_cam, max_distance=MAX_RENDER_DISTANCE)
                for v_res in vehicle_results:
                    # Exclude the ego vehicle
                    if v_res['id'] == ego_vehicle.id:
                        continue
                    class_id = CLASS_MAP.get('Vehicle')
                    if class_id is not None:
                        valid_detections.append((class_id, 'Vehicle', v_res['bbox'], v_res['actor_obj']))


                # 2. Process Pedestrians separately
                for ped in all_world_actors.filter('walker.pedestrian.*'):
                    if ped and ped.is_alive:
                        if not is_visible_and_not_occluded(world, rgb_cam, ped, MAX_RENDER_DISTANCE):
                            continue
                           
                        bbox_2d = get_2d_bbox_from_3d(ped, rgb_cam)
                        if bbox_2d:
                            label_name = get_actor_label(ped)
                            class_id = CLASS_MAP.get(label_name)
                            if class_id is not None:
                                valid_detections.append((class_id, label_name, bbox_2d, ped))


                annotations_3d = []
                for class_id, label_name, bbox_2d, actor_obj in valid_detections:
                    yolo_bbox = carla_to_yolo(bbox_2d, image_w, image_h)
                    line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    yolo_annotations.append(line)


                    # Extract 3D Bounding Box details
                    actor_transform = actor_obj.get_transform() if hasattr(actor_obj, 'get_transform') else getattr(actor_obj, 'transform', None)
                    if actor_transform and hasattr(actor_obj, 'bounding_box'):
                        bb = actor_obj.bounding_box
                       
                        # Static environment objects (carla.EnvironmentObject) have bounding_box in world space.
                        # Dynamic actors have bounding_box relative to the actor's transform.
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
                       
                        # Camera relative positions (X = Right, Y = Down, Z = Depth Forward)
                        cam_rel_x = p_camera[1]
                        cam_rel_y = -p_camera[2]
                        cam_rel_z = p_camera[0]
                       
                        extent = bb.extent
                       
                        detection_3d = {
                            "class_id": class_id,
                            "label_name": label_name,
                            "2d_bbox_pixel": bbox_2d,
                            "3d_world_location": {
                                "x": world_bb_center.x,
                                "y": world_bb_center.y,
                                "z": world_bb_center.z
                            },
                            "3d_world_rotation": {
                                "pitch": rot.pitch,
                                "yaw": rot.yaw,
                                "roll": rot.roll
                            },
                            "extent_dimensions": {
                                "x_half_size": extent.x,
                                "y_half_size": extent.y,
                                "z_half_size": extent.z,
                                "height": extent.z * 2.0,
                                "width": extent.y * 2.0,
                                "length": extent.x * 2.0
                            },
                            "camera_relative_location": {
                                "x": cam_rel_x,
                                "y": cam_rel_y,
                                "z": cam_rel_z
                            }
                        }
                        annotations_3d.append(detection_3d)


                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_annotations))


                label_3d_path = os.path.join(OUTPUT_DIR, "labels_3d", f"{image_name}.json")
                with open(label_3d_path, 'w') as f:
                    json.dump(annotations_3d, f, indent=4)


                if DRAW_2D_BBOX:
                    img_2d = img_1080p.copy()
                    for _, label_name, bbox_2d, _ in valid_detections:
                        x_min, y_min, x_max, y_max = bbox_2d
                        x_min, y_min = x_min // 2, y_min // 2
                        x_max, y_max = x_max // 2, y_max // 2
                       
                        cv2.rectangle(img_2d, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
                        cv2.putText(img_2d, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                   
                    bbox_img_path = os.path.join(OUTPUT_DIR, "images_with_2d_bbox", f"{image_name}_bbox_2d.png")
                    cv2.imwrite(bbox_img_path, img_2d)


                if DRAW_3D_BBOX:
                    img_3d = img_1080p.copy()
                    for _, _, _, actor_obj in valid_detections:
                        draw_3d_bbox_on_image(img_3d, actor_obj, rgb_cam, color=(0, 255, 255), thickness=2)
                   
                    bbox_img_path = os.path.join(OUTPUT_DIR, "images_with_3d_bbox", f"{image_name}_bbox_3d.png")
                    cv2.imwrite(bbox_img_path, img_3d)


            if rgb_cam and rgb_cam.is_alive:
                if rgb_cam.is_listening: rgb_cam.stop()
                rgb_cam.destroy()
               
            if seg_cam and seg_cam.is_alive:
                if seg_cam.is_listening: seg_cam.stop()
                seg_cam.destroy()


            for actor in actors:
                if actor and actor.is_alive and hasattr(actor, 'type_id') and actor.type_id.startswith('controller.'):
                    actor.stop()
                           
            commands = []
            for actor in actors:
                if actor and actor.is_alive and actor.parent is None and actor.id != rgb_cam.id:
                    commands.append(carla.command.DestroyActor(actor))
            if ego_vehicle and ego_vehicle.is_alive:
                commands.append(carla.command.DestroyActor(ego_vehicle))
               
            client.apply_batch(commands)
            actors = []
            ego_vehicle = None
           
            while not image_queue.empty(): image_queue.get()
            while not seg_queue.empty(): seg_queue.get()


            for _ in range(10): world.tick()
            time.sleep(0.2)
    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == "__main__":
    main()



