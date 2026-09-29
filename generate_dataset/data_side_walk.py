import carla
import random
import time
import os
import math
import json
import numpy as np
from PIL import Image
import cv2
import datetime
import queue


# --- NEW YOLO CONFIGURATION ---
NUM_IMAGES = 20 # Increased to generate 40+ frames
START_FRAME_NUMBER = 340 # Set your starting number here
OUTPUT_DIR = "dataset_w_segmetaion"
# Define your classes here
CLASS_MAP = {
    'Pedestrian': 0,
    'E-Scooter + Rider': 1,
    'E-Scooter': 2,
    'Cyclist': 3,
    'Vehicle': 4
}


# Spawning parameters
NUM_PEDESTRIANS_MIN = 3
NUM_PEDESTRIANS_MAX = 5
PEDESTRIAN_SPAWN_DISTANCE_MIN = 10
PEDESTRIAN_SPAWN_DISTANCE_MAX = 25
NUM_ESCOOTERS_MIN = 0
NUM_ESCOOTERS_MAX = 0
ESCOOTER_SPAWN_DISTANCE_MIN = 10
ESCOOTER_SPAWN_DISTANCE_MAX = 15
MAX_RENDER_DISTANCE = 50.0 # Exclude actors beyond this distance (in meters)
NUM_TRAFFIC_VEHICLES_MIN = 0
NUM_TRAFFIC_VEHICLES_MAX = 2
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN = 15
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX = 35
TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE = 8.0


# Define allowed traffic vehicles (Add or remove specific car, truck, and 2-wheeler IDs here)
ALLOWED_TRAFFIC_VEHICLES = [
    'vehicle.audi.etron',
    'vehicle.audi.tt',
    'vehicle.bmw.grandtourer',
    'vehicle.ford.mustang',
    'vehicle.lincoln.mkz_2020',
    'vehicle.mercedes.coupe',
    'vehicle.nissan.patrol_2021',
    'vehicle.tesla.model3',
    'vehicle.toyota.prius',
    'vehicle.volkswagen.t2_2021',
    'vehicle.carlamotors.firetruck',
    'vehicle.diamondback.century',
    'vehicle.kawasaki.ninja',
    'vehicle.yamaha.yzf'  # Specific 2-wheeler example
]


# Select the exact weather you want for this dataset run
CURRENT_WEATHER = "Rain" # Options: "Clear", "Partly cloudy", "Overcast", "Rain", "Snowfall"


# Weather parameters
WEATHER_CONDITIONS = {
    "Clear": carla.WeatherParameters.ClearNoon,
    "Partly_cloudy": carla.WeatherParameters.CloudyNoon,
    "Overcast": carla.WeatherParameters.WetCloudyNoon,
    "Rain": carla.WeatherParameters.HardRainNoon
}






# ------------------------
# Functions
# ------------------------


def carla_to_yolo(bbox, img_w, img_h):
    """
    Converts [x_min, y_min, x_max, y_max] to YOLO [x_center, y_center, width, height]
    all normalized 0.0 to 1.0
    """
    x_min, y_min, x_max, y_max = bbox
   
    # Calculate pixel width and height
    width = x_max - x_min
    height = y_max - y_min
   
    # Calculate pixel centers
    x_center = x_min + (width / 2.0)
    y_center = y_min + (height / 2.0)
   
    # Normalize by image dimensions
    return [x_center / img_w, y_center / img_h, width / img_w, height / img_h]


def spawn_actor(world, bp, transform, max_attempts=10, attach_to=None):
    for _ in range(max_attempts):
        if attach_to:
            actor = world.try_spawn_actor(bp, carla.Transform(), attach_to=attach_to)
        else:
            actor = world.try_spawn_actor(bp, transform)
        if actor:
            return actor
    return None


def random_location_in_camera_fov(ego_transform, min_dist=10, max_dist=20, lateral_range=5.0):
    yaw_rad = math.radians(ego_transform.rotation.yaw)
    forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad))
    dist = random.uniform(min_dist, max_dist)
    lateral = random.uniform(-lateral_range, lateral_range)
    loc = ego_transform.location + forward * dist + carla.Vector3D(
        x=-forward.y * lateral,
        y=forward.x * lateral,
        z=0
    )
    return loc


def spawn_pedestrian_on_sidewalk(world, pedestrian_bps, ego_transform, min_dist, max_dist):
    max_attempts = 30
    carla_map = world.get_map()


    for attempt in range(max_attempts):
        test_loc = random_location_in_camera_fov(ego_transform, min_dist=min_dist, max_dist=max_dist, lateral_range=15.0)
       
        # Try to find a driving waypoint first to search for a connected sidewalk
        wp_driving = carla_map.get_waypoint(test_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        sidewalk_loc = None


        if wp_driving:
            # Check right lanes for sidewalk
            curr = wp_driving
            while curr is not None:
                if curr.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_loc = curr.transform.location
                    break
                curr = curr.get_right_lane()
               
            # If not found, check left lanes for sidewalk
            if sidewalk_loc is None:
                curr = wp_driving
                while curr is not None:
                    if curr.lane_type == carla.LaneType.Sidewalk:
                        sidewalk_loc = curr.transform.location
                        break
                    curr = curr.get_left_lane()


        if sidewalk_loc:
            # Add some lateral noise
            sidewalk_loc.x += random.uniform(-1.0, 1.0)
            sidewalk_loc.y += random.uniform(-1.0, 1.0)
           
            spawn_transform = carla.Transform(
                carla.Location(sidewalk_loc.x, sidewalk_loc.y, sidewalk_loc.z + 0.5),
                carla.Rotation(yaw=random.uniform(-180, 180))
            )


            ped_bp = random.choice(pedestrian_bps)
            pedestrian = spawn_actor(world, ped_bp, spawn_transform)
            if pedestrian:
                world.tick()
                ped_yaw = math.radians(spawn_transform.rotation.yaw)
                ped_forward = carla.Vector3D(x=math.cos(ped_yaw), y=math.sin(ped_yaw))
                control = carla.WalkerControl()
                control.direction = ped_forward
                control.speed = random.uniform(0.8, 1.5)
                pedestrian.apply_control(control)
                return pedestrian, None


    # Fallback to random nav location
    for attempt in range(max_attempts):
        loc = world.get_random_location_from_navigation()
        if loc:
            dist = loc.distance(ego_transform.location)
            if dist < max_dist + 20.0:
                spawn_transform = carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + 0.5),
                    carla.Rotation(yaw=random.uniform(-180, 180))
                )
                ped_bp = random.choice(pedestrian_bps)
                pedestrian = spawn_actor(world, ped_bp, spawn_transform)
                if pedestrian:
                    world.tick()
                    ped_yaw = math.radians(spawn_transform.rotation.yaw)
                    ped_forward = carla.Vector3D(x=math.cos(ped_yaw), y=math.sin(ped_yaw))
                    control = carla.WalkerControl()
                    control.direction = ped_forward
                    control.speed = random.uniform(0.8, 1.5)
                    pedestrian.apply_control(control)
                    return pedestrian, None


    return None, None


def spawn_escooter(world, escooter_bp, ego_transform, min_dist, max_dist):
    """
    Tries to spawn an e-scooter in the ego vehicle's FOV.
    If that fails, it falls back to spawning at a random spawn point on the map.
    """
    if escooter_bp is None:
        print("E-scooter blueprint not found.")
        return None


    # --- 1. Try to spawn in Field of View (FOV) ---
    max_attempts = 20
    carla_map = world.get_map()
    for i in range(max_attempts):
        candidate_loc = random_location_in_camera_fov(ego_transform, min_dist=min_dist, max_dist=max_dist, lateral_range=4.0)
        wp = carla_map.get_waypoint(candidate_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp:
            continue


        spawn_transform = wp.transform
        spawn_transform.location.z += 0.18 # Increased Z offset for better ground clearance
        spawn_transform.rotation.yaw += 270.0


        escooter = spawn_actor(world, escooter_bp, spawn_transform)
        if escooter:
            #print("Successfully spawned e-scooter in FOV.")
            escooter.set_simulate_physics(True)
            if CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
                try:
                    escooter.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                except Exception:
                    pass
           
            # Calculate forward velocity based on new randomized yaw
            yaw_rad = math.radians(spawn_transform.rotation.yaw)
            escooter.set_target_velocity(carla.Vector3D(x=2.0 * math.cos(yaw_rad), y=2.0 * math.sin(yaw_rad)))
           
            for _ in range(10):
                world.tick()
            return escooter


    # --- 2. Fallback: Spawn at a random map spawn point ---
    print("Could not spawn e-scooter in FOV, falling back to a random spawn point.")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Could not find any spawn points in the map.")
        return None


    spawn_transform = random.choice(spawn_points)
    spawn_transform.location.z += 0.5 # Raise it slightly to avoid collision
    print(f"Spawning at random map spawn point: {spawn_transform.location}")
    escooter = spawn_actor(world, escooter_bp, spawn_transform)
    if escooter and CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
        try:
            escooter.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        except Exception:
            pass
    return escooter


def spawn_traffic_vehicles(world, blueprints, ego_transform, num_vehicles, min_dist, max_dist, lateral_range):
    """Spawns background traffic vehicles near the ego vehicle on valid roads."""
    carla_map = world.get_map()
    spawned_vehicles = []
    max_attempts = 20
   
    # Filter blueprints to only include the ones in ALLOWED_TRAFFIC_VEHICLES
    traffic_bps = [bp for bp in blueprints.filter('vehicle.*') if bp.id in ALLOWED_TRAFFIC_VEHICLES]
    if not traffic_bps:
        print("Warning: No valid traffic vehicle blueprints found from ALLOWED_TRAFFIC_VEHICLES list.")
        return []


    for _ in range(num_vehicles):
        for _ in range(max_attempts):
            loc = random_location_in_camera_fov(ego_transform, min_dist=min_dist, max_dist=max_dist, lateral_range=lateral_range)
            wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                t = wp.transform
                t.location.z += 0.5
                # Prevent vehicles from spawning exactly inside each other
                t.location.x += random.uniform(-1.0, 1.0)
                t.location.y += random.uniform(-1.0, 1.0)
                v_bp = random.choice(traffic_bps)
                v_actor = spawn_actor(world, v_bp, t)
                if v_actor:
                    if CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
                        try:
                            v_actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                        except Exception:
                            pass
                    spawned_vehicles.append(v_actor)
                    break # Successfully spawned, move to the next vehicle


    # Let the vehicles settle to the ground before freezing them
    for _ in range(15):
        world.tick()
    for v in spawned_vehicles:
        v.set_simulate_physics(False) # Keep static
       
    return spawned_vehicles


# ------------------------
# Helper: Check if actor is in camera FOV
# ------------------------


def is_actor_in_fov(actor, camera, image_w, image_h):
    """
    Checks if an actor is visible in the camera's field of view.
    Returns True if the actor is visible, False otherwise.
    """
    # Get the actor's 3D bounding box vertices in world coordinates
    bb = actor.bounding_box
    transform = actor.get_transform() if hasattr(actor, 'get_transform') else actor.transform
    verts = bb.get_world_vertices(transform)


    # Get camera transformation matrices
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())


    # Project each vertex onto the 2D image plane
    points_in_camera_view = []
    for v in verts:
        world_point = np.array([v.x, v.y, v.z, 1.0])
        cam_point = np.dot(world_to_camera, world_point)
 
        # UE4's camera space is X-forward, Y-right, Z-up.
        # We need to check if the point is in front of the camera (positive X).
        if cam_point[0] > 0:
            # Now, we transform to the standard camera coordinate system (Z-forward, X-right, Y-down)
            # for the projection. This is a simple remapping.
            p_camera_std = np.array([cam_point[1], -cam_point[2], cam_point[0]])
 
            # Project to 2D using the camera's intrinsic parameters.
            # For a simple FOV check, we can normalize by depth (Z).
            x_2d = p_camera_std[0] / p_camera_std[2]
            y_2d = p_camera_std[1] / p_camera_std[2]
 
            # This gives coordinates in the range [-tan(FOV/2), tan(FOV/2)].
            # A simpler check is to see if it's within the normalized screen space [-1, 1]
            # which is what the original code was trying to do, but with the wrong depth axis.
            # The projection logic in CARLA's client-side drawing functions is more complex,
            # but this check is a good approximation for "is it in the frustum".
            points_in_camera_view.append(True)


    # If any vertex of the bounding box is inside the camera's frustum, the actor is visible.
    return any(points_in_camera_view)


# ------------------------
# Helper: Distance and Raycast Occlusion Check
# ------------------------
def get_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)


def is_visible_and_not_occluded(world, camera, actor, max_distance):
    cam_loc = camera.get_transform().location
    actor_transform = actor.get_transform() if hasattr(actor, 'get_transform') else actor.transform
    actor_loc = actor_transform.location
   
    dist = get_distance(cam_loc, actor_loc)
    if dist > max_distance:
        return False
       
    # Cast a ray from the camera to the actor to check for static occlusions
    ray_hits = world.cast_ray(cam_loc, actor_loc)
    for hit in ray_hits:
        label_str = str(hit.label).split('.')[-1]
        # Check if the ray hits building, wall, fence, or vegetation before the actor
        if label_str in ['Buildings', 'Walls', 'Fences', 'Vegetation', 'Other']:
            hit_dist = get_distance(cam_loc, hit.location)
            # If the obstacle is closer than the actor (with 2 meter tolerance)
            if hit_dist < dist - 2.0:
                return False
    return True


# ------------------------
# Helper: Project 3D box to 2D
# ------------------------
def get_2d_bbox_from_3d(actor, camera):
    """
    Projects the 3D bounding box of an actor to a 2D bounding box in the image.
    Returns [x_min, y_min, x_max, y_max] or None if the box is not in the frame.
    """
    # Get camera attributes
    image_w = int(camera.attributes['image_size_x'])
    image_h = int(camera.attributes['image_size_y'])
    fov = float(camera.attributes['fov'])
   
    # Calculate camera intrinsics
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.array([[focal, 0, image_w / 2.0],
                  [0, focal, image_h / 2.0],
                  [0, 0, 1.0]])


    # Get actor's 3D bounding box vertices
    bb = actor.bounding_box
    transform = actor.get_transform() if hasattr(actor, 'get_transform') else actor.transform
    verts = bb.get_world_vertices(transform)
   
    # Get camera transform
    cam_transform = camera.get_transform()
    world_to_camera = np.array(cam_transform.get_inverse_matrix())


    # Project vertices to 2D
    points_2d = []
    for v in verts:
        world_point = np.array([v.x, v.y, v.z, 1.0])
        p_camera = np.dot(world_to_camera, world_point)
       
        # Check if point is in front of camera (UE4's X-forward)
        if p_camera[0] > 0:
            # Transform to standard camera coordinates (Z-forward, X-right, Y-down)
            p_camera_std = np.array([p_camera[1], -p_camera[2], p_camera[0]])
           
            # Project to image plane
            p_image = np.dot(K, p_camera_std)
            p_image /= p_image[2] # Normalize
            points_2d.append(p_image[:2])


    if not points_2d:
        return None


    points_2d = np.array(points_2d)
    x_min = int(np.clip(np.min(points_2d[:, 0]), 0, image_w))
    y_min = int(np.clip(np.min(points_2d[:, 1]), 0, image_h))
    x_max = int(np.clip(np.max(points_2d[:, 0]), 0, image_w))
    y_max = int(np.clip(np.max(points_2d[:, 1]), 0, image_h))


    # Return None if the box is completely outside the image
    if x_min >= image_w or y_min >= image_h or x_max <= 0 or y_max <= 0:
        return None


    return [x_min, y_min, x_max, y_max]


# ------------------------
# Helper: Get actor label for drawing
# ------------------------
def get_actor_label(actor):
    """Returns a display-friendly label for an actor."""
    if hasattr(actor, 'type_id'):
        if 'walker.pedestrian' in actor.type_id:
            return 'Pedestrian'
        if 'vehicle.diamondback.century' in actor.type_id:
            return 'Cyclist' # Matches CLASS_MAP
        if 'vehicle.bh.rider' in actor.type_id:       # rider  vehicle.bh.escooter
            return 'E-Scooter + Rider' # Matches CLASS_MAP       #  E-Scooter + Rider  E-Scooter
        if 'vehicle' in actor.type_id:
            return 'Vehicle' # Matches CLASS_MAP
    elif hasattr(actor, 'type'):
        type_name = str(actor.type).split('.')[-1]
        if type_name in ['Vehicles', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle']:
            return 'Vehicle'
    return 'Actor' # Fallback


# ------------------------
# Helper: Find Spawn Points Approaching Crosswalks
# ------------------------
def get_crosswalk_approach_transforms(carla_map, min_dist=2.0, max_dist=10.0):
    """Finds crosswalks and returns spawn transforms approaching them."""
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
        cw_waypoint = carla_map.get_waypoint(cw_center, project_to_road=True, lane_type=carla.LaneType.Driving)
        if cw_waypoint:
            dist = random.uniform(min_dist, max_dist)
            prev_wps = cw_waypoint.previous(dist)
            t = prev_wps[0].transform if prev_wps else cw_waypoint.transform
            t.location.z += 0.5 # Elevate slightly to avoid collision with ground
            valid_transforms.append((t, cw_center))
    return valid_transforms


# ------------------------
# Main loop
# ------------------------
def main():
    # Create YOLO folder structure
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images_with_bbox"), exist_ok=True)
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
            print(f"Warning: Could not turn on street lights: {e}")


    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('my_car_xyz')[0]
    pedestrian_bps = blueprints.filter('walker.pedestrian.*')
    walker_controller_bp = blueprints.find('controller.ai.walker')
   
    escooter_bp = blueprints.filter('vehicle.bh.rider')[0] if blueprints.filter('vehicle.bh.rider') else None
    if not escooter_bp:
        print("Warning: E-scooter blueprint 'vehicle.bh.escooter' not found. E-scooters will not be spawned.")


    rgb_camera_bp = blueprints.find('sensor.camera.rgb')
   
    # Set to 4K for Supersampling Anti-Aliasing (SSAA)
    rgb_camera_bp.set_attribute('image_size_x', '3840')
    rgb_camera_bp.set_attribute('image_size_y', '2160')
    rgb_camera_bp.set_attribute('fov', '90')
    rgb_camera_bp.set_attribute('sensor_tick', '0.05')
    rgb_camera_bp.set_attribute('lens_circle_multiplier', '1.5')
    rgb_camera_bp.set_attribute('lens_circle_falloff', '4.0')
    rgb_camera_bp.set_attribute('chromatic_aberration_intensity', '0.5')
    rgb_camera_bp.set_attribute('chromatic_aberration_offset', '0')


    # Set up Semantic Segmentation Camera
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


    # Cache environment vehicles once outside the loop.
    # Calling this every frame causes memory leaks and UE4 Signal 11 Engine Crashes.
    print("Caching static environment vehicles...")
    env_vehicles = []
    if hasattr(carla.CityObjectLabel, 'Vehicles'):
        env_vehicles.extend(world.get_environment_objects(carla.CityObjectLabel.Vehicles))
    else:
        for label_str in ['Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle']:
            if hasattr(carla.CityObjectLabel, label_str):
                env_vehicles.extend(world.get_environment_objects(getattr(carla.CityObjectLabel, label_str)))


    ego_vehicle = None
    actors = []
    try:
        for i in range(NUM_IMAGES):
            print(f"Generating image {i+1}/{NUM_IMAGES}")
           
            ego_transform = random.choice(spawn_points)
            ego_vehicle = spawn_actor(world, vehicle_bp, ego_transform)
            if not ego_vehicle:
                print("Failed to spawn ego vehicle, skipping this frame.")
                continue
               
            if CURRENT_WEATHER in ["Partly_cloudy", "Overcast", "Rain"]:
                try:
                    ego_vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                except Exception:
                    pass


            # Let the ego vehicle drop to the ground before freezing it
            for _ in range(15):
                world.tick()
            ego_vehicle.set_simulate_physics(False) # Keep ego vehicle perfectly static


            actors = []


            # Determine random counts for this frame
            num_pedestrians = random.randint(NUM_PEDESTRIANS_MIN, NUM_PEDESTRIANS_MAX)
            num_traffic_vehicles = random.randint(NUM_TRAFFIC_VEHICLES_MIN, NUM_TRAFFIC_VEHICLES_MAX)
            num_escooters = random.randint(NUM_ESCOOTERS_MIN, NUM_ESCOOTERS_MAX)


            # Spawn pedestrians
            for _ in range(num_pedestrians):
                ped, controller = spawn_pedestrian_on_sidewalk(world, pedestrian_bps, ego_transform, PEDESTRIAN_SPAWN_DISTANCE_MIN, PEDESTRIAN_SPAWN_DISTANCE_MAX)
                if ped:
                    actors.append(ped)
                    if controller:
                        actors.append(controller)
                else:
                    print("Failed to spawn pedestrian on sidewalk.")


            # Spawn traffic vehicles
            traffic_vehicles = spawn_traffic_vehicles(
                world, blueprints, ego_transform, num_traffic_vehicles,
                TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN, TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX, TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE
            )
            actors.extend(traffic_vehicles)


            # Spawn e-scooters
            for _ in range(num_escooters):
                escooter = spawn_escooter(world, escooter_bp, ego_transform, ESCOOTER_SPAWN_DISTANCE_MIN, ESCOOTER_SPAWN_DISTANCE_MAX)
                if escooter:
                    actors.append(escooter)
                    time.sleep(0.1)  # Give a little time for the escooter to spawn
                else:
                    print("Failed to spawn e-scooter in FOV.")


            # Spawn RGB camera
            camera_transform = carla.Transform(carla.Location(x=0.8, z=1.8), carla.Rotation(pitch=-10)) # This transform is for the RGB camera now
            rgb_cam = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
            actors.append(rgb_cam) # Add RGB camera to actor list for cleanup


            # Spawn Semantic Segmentation camera
            seg_cam = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=ego_vehicle)
            actors.append(seg_cam) # Add Segmentation camera to actor list for cleanup
         
            # Give the simulation time to fully settle (physics and auto-exposure)
            for _ in range(50):
                world.tick()


            # Capture frames using a queue to prevent memory leaks
            image_queue = queue.Queue()
            seg_queue = queue.Queue()
            rgb_cam.listen(image_queue.put)
            seg_cam.listen(seg_queue.put)
           
            # Tick a few more times while listening to initialize the sensor pipeline
            for _ in range(10):
                world.tick()
           
            # Clear out all the transitional/unsettled frames
            while not image_queue.empty():
                image_queue.get()
            while not seg_queue.empty():
                seg_queue.get()
               
            # Take one final tick to capture the perfect, fully settled frame
            world.tick()
           
            try:
                rgb_image = image_queue.get(timeout=10.0)
            except queue.Empty:
                print("Failed to capture image from camera.")
                rgb_image = None


            try:
                seg_image = seg_queue.get(timeout=10.0)
            except queue.Empty:
                print("Failed to capture image from segmentation camera.")
                seg_image = None
               
            if rgb_cam.is_listening:
                rgb_cam.stop() # Immediately stop listening to free CARLA resources
            if seg_cam.is_listening:
                seg_cam.stop()


            if rgb_image:
                frame_number = START_FRAME_NUMBER + i
                image_name = f"frame_{frame_number:06d}"
                img_path = os.path.join(OUTPUT_DIR, "images", f"{image_name}.png")
                label_path = os.path.join(OUTPUT_DIR, "labels", f"{image_name}.txt")
               
                # Save Image
                # Apply SSAA: Convert 4K raw image to numpy array, then downscale to 1080p
                img_array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
                img_array = np.reshape(img_array, (rgb_image.height, rgb_image.width, 4))
                img_array = img_array[:, :, :3] # Remove alpha channel
               
                # Resize to 1080p using high-quality Area interpolation
                img_1080p = cv2.resize(img_array, (1920, 1080), interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_path, img_1080p)


                if seg_image:
                    # 1. Save Raw Semantic IDs (useful for training models)
                    # CARLA Semantic Segmentation stores class IDs in the Red channel (index 2 in BGRA)
                    seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
                    seg_array = np.reshape(seg_array, (seg_image.height, seg_image.width, 4))
                    raw_classes = seg_array[:, :, 2]
                   
                    # Resize using INTER_NEAREST to avoid interpolating class IDs (which would create invalid classes)
                    raw_classes_1080p = cv2.resize(raw_classes, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                    raw_seg_path = os.path.join(OUTPUT_DIR, "images_segmentation_raw", f"{image_name}.png")
                    cv2.imwrite(raw_seg_path, raw_classes_1080p)
                   
                    # 2. Save Colorized Version (useful for human visualization)
                    seg_image.convert(carla.ColorConverter.CityScapesPalette)
                    color_seg_array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
                    color_seg_array = np.reshape(color_seg_array, (seg_image.height, seg_image.width, 4))
                    color_seg_array = color_seg_array[:, :, :3] # Remove alpha channel
                    color_seg_1080p = cv2.resize(color_seg_array, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                    color_seg_path = os.path.join(OUTPUT_DIR, "images_segmentation_color", f"{image_name}.png")
                    cv2.imwrite(color_seg_path, color_seg_1080p)
               
                # Save YOLO Labels
                # The camera outputs 4K bounding boxes, so we normalize using 4K dimensions
                image_w = 3840
                image_h = 2160
                yolo_annotations = []
               
                # Get all vehicles and pedestrians in the world, excluding ego_vehicle
                all_world_actors = world.get_actors()
                visible_actors = [a for a in all_world_actors.filter('vehicle.*') if a.id != ego_vehicle.id]
                visible_actors.extend(list(all_world_actors.filter('walker.pedestrian.*')))


                # Include static parked vehicles natively placed in the CARLA environment
                visible_actors.extend(env_vehicles)
               
                valid_detections = []


                for actor in visible_actors:
                    if actor and (not hasattr(actor, 'is_alive') or actor.is_alive):
                        # Verify the actor is close enough and not occluded by geometry
                        if not is_visible_and_not_occluded(world, rgb_cam, actor, MAX_RENDER_DISTANCE):
                            continue
                           
                        bbox_2d = get_2d_bbox_from_3d(actor, rgb_cam)
                        if bbox_2d:
                            label_name = get_actor_label(actor)
                            class_id = CLASS_MAP.get(label_name)
                            if class_id is not None:
                                valid_detections.append((class_id, label_name, bbox_2d))


                # 1. Format and write YOLO Labels
                for class_id, label_name, bbox_2d in valid_detections:
                    yolo_bbox = carla_to_yolo(bbox_2d, image_w, image_h)
                    line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    yolo_annotations.append(line)


                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_annotations))


                # 2. Draw Bounding Boxes on Image
                img_for_drawing = img_1080p.copy()
                for _, label_name, bbox_2d in valid_detections:
                    x_min, y_min, x_max, y_max = bbox_2d
                   
                    # Scale 4K coordinates down to 1080p for drawing
                    x_min, y_min = x_min // 2, y_min // 2
                    x_max, y_max = x_max // 2, y_max // 2
                   
                    cv2.rectangle(img_for_drawing, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
                    cv2.putText(img_for_drawing, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                   
                bbox_img_path = os.path.join(OUTPUT_DIR, "images_with_bbox", f"{image_name}_bbox.png")
                cv2.imwrite(bbox_img_path, img_for_drawing)


            # Cleanup at the end of each iteration
            # Explicitly stop and destroy the camera first to prevent stream warnings and SIGSEGV
            if rgb_cam and rgb_cam.is_alive:
                if rgb_cam.is_listening:
                    rgb_cam.stop()
                rgb_cam.destroy()
               
            if seg_cam and seg_cam.is_alive:
                if seg_cam.is_listening:
                    seg_cam.stop()
                seg_cam.destroy()


            # Stop controllers explicitly
            for actor in actors:
                if actor and actor.is_alive and hasattr(actor, 'type_id') and actor.type_id.startswith('controller.'):
                    actor.stop()
                           
            # Use batch destruction to prevent CARLA desync and memory corruption
            commands = []
            for actor in actors:
                if actor and actor.is_alive and actor.parent is None and actor.id != rgb_cam.id:
                    commands.append(carla.command.DestroyActor(actor))
            if ego_vehicle and ego_vehicle.is_alive:
                commands.append(carla.command.DestroyActor(ego_vehicle))
               
            client.apply_batch(commands)
            actors = []
            ego_vehicle = None
           
            # Clear queue to free unreferenced CARLA image objects from RAM
            while not image_queue.empty():
                image_queue.get()
            while not seg_queue.empty():
                seg_queue.get()


            # Give CARLA some time to process the destruction of actors
            for _ in range(10):
                world.tick()
            time.sleep(0.2)
    finally:
        # Reset synchronous mode
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == "__main__":
    main()



