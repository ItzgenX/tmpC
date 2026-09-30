import carla
import random
import math

from utils import lerp_location


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
        wp_driving = carla_map.get_waypoint(test_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        sidewalk_loc = None


        if wp_driving:
            curr = wp_driving
            while curr is not None:
                if curr.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_loc = curr.transform.location
                    break
                curr = curr.get_right_lane()
               
            if sidewalk_loc is None:
                curr = wp_driving
                while curr is not None:
                    if curr.lane_type == carla.LaneType.Sidewalk:
                        sidewalk_loc = curr.transform.location
                        break
                    curr = curr.get_left_lane()


        if sidewalk_loc:
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


def spawn_traffic_vehicles(world, blueprints, ego_transform, num_vehicles, min_dist, max_dist, lateral_range, current_weather, allowed_vehicles):
    carla_map = world.get_map()
    spawned_vehicles = []
    max_attempts = 20
   
    traffic_bps = [bp for bp in blueprints.filter('vehicle.*') if bp.id in allowed_vehicles]
    if not traffic_bps:
        print("Warning: No valid traffic vehicle blueprints found.")
        return []


    for _ in range(num_vehicles):
        for _ in range(max_attempts):
            loc = random_location_in_camera_fov(ego_transform, min_dist=min_dist, max_dist=max_dist, lateral_range=lateral_range)
            wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                t = wp.transform
                t.location.z += 0.5
                t.location.x += random.uniform(-1.0, 1.0)
                t.location.y += random.uniform(-1.0, 1.0)
                v_bp = random.choice(traffic_bps)
                v_actor = spawn_actor(world, v_bp, t)
                if v_actor:
                    if current_weather in ["Partly_cloudy", "Overcast", "Rain"]:
                        try:
                            v_actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                        except Exception:
                            pass
                    spawned_vehicles.append(v_actor)
                    break


    for _ in range(15):
        world.tick()
    for v in spawned_vehicles:
        v.set_simulate_physics(False)

    return spawned_vehicles


def spawn_pedestrian_crossing(world, pedestrian_bps, walker_controller_bp,
                               crossing_start, crossing_end,
                               progress=None, speed_range=(0.9, 1.6)):
    """
    Spawns a pedestrian on the crosswalk line between crossing_start and
    crossing_end, walking toward the far side via CARLA's AI walker
    controller (proper crowd-nav pathing + walk animation), so it reads as
    genuinely crossing rather than standing still on the zebra crossing.

    progress: 0..1 position along the line to start from. If None, a random
    start near one side is chosen (so the walk has room to play out).
    Returns (pedestrian, controller_or_None).
    """
    if progress is None:
        progress = random.uniform(0.05, 0.35)

    if random.random() < 0.5:
        start_loc = lerp_location(crossing_start, crossing_end, progress)
        target_loc = crossing_end
        yaw = math.degrees(math.atan2(crossing_end.y - crossing_start.y, crossing_end.x - crossing_start.x))
    else:
        start_loc = lerp_location(crossing_end, crossing_start, progress)
        target_loc = crossing_start
        yaw = math.degrees(math.atan2(crossing_start.y - crossing_end.y, crossing_start.x - crossing_end.x))

    spawn_transform = carla.Transform(
        carla.Location(start_loc.x, start_loc.y, start_loc.z + 0.5),
        carla.Rotation(yaw=yaw)
    )

    ped_bp = random.choice(pedestrian_bps)
    if ped_bp.has_attribute('is_invincible'):
        ped_bp.set_attribute('is_invincible', 'false')

    pedestrian = spawn_actor(world, ped_bp, spawn_transform)
    if not pedestrian:
        return None, None

    world.tick()

    controller = None
    speed = random.uniform(*speed_range)
    try:
        controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=pedestrian)
        world.tick()
        controller.start()
        beyond = lerp_location(start_loc, target_loc, 1.4)
        controller.go_to_location(carla.Location(beyond.x, beyond.y, target_loc.z))
        controller.set_max_speed(speed)
    except RuntimeError:
        if controller is not None and controller.is_alive:
            controller.destroy()
        controller = None
        yaw_rad = math.radians(yaw)
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad))
        control.speed = speed
        pedestrian.apply_control(control)

    return pedestrian, controller



