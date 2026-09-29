import carla
import random
import math


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



