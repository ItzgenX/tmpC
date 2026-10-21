"""Spawns what the planner decided. No decisions are made here."""
import carla

import advanced as C


class SpawnError(Exception):
    pass


# ---------- blueprints ----------

def _attr(bp, name):
    return bp.get_attribute(name).as_str().lower() if bp.has_attribute(name) else ""


def walker_type(bp):
    if bp.id in C.PED_TYPE_OVERRIDES:
        return C.PED_TYPE_OVERRIDES[bp.id]
    if _attr(bp, "age") == "child":
        return "child"
    gender = _attr(bp, "gender")
    if gender == "female":
        return "woman"
    if gender == "male":
        return "man"
    return None


def walker_pools(bp_lib):
    pools = {"child": [], "woman": [], "man": []}
    for bp in bp_lib.filter("walker.pedestrian.*"):
        if bp.id in C.PED_BLUEPRINT_EXCLUDE:
            continue
        t = walker_type(bp)
        if t in pools:
            pools[t].append(bp.id)
    return {k: sorted(v) for k, v in pools.items()}


def vehicle_pool(bp_lib, ids):
    """{bp_id: [recommended colors]} for the ids that exist on this server."""
    pool = {}
    for bp_id in ids:
        found = bp_lib.filter(bp_id)
        if not found:
            print(f"  warning: blueprint {bp_id} not on server, skipped")
            continue
        bp = found[0]
        pool[bp.id] = list(bp.get_attribute("color").recommended_values) if bp.has_attribute("color") else []
    return pool


def ego_blueprint(bp_lib):
    found = bp_lib.filter(C.EGO_BLUEPRINT)
    if found:
        return found[0]
    print(f"  warning: ego blueprint '{C.EGO_BLUEPRINT}' not found, using {C.EGO_FALLBACK_BLUEPRINT}")
    return bp_lib.find(C.EGO_FALLBACK_BLUEPRINT)


def _set_attrs(bp, attrs):
    for k, v in attrs.items():
        if bp.has_attribute(k):
            bp.set_attribute(k, str(v))


def camera_blueprints(bp_lib):
    rgb = bp_lib.find("sensor.camera.rgb")
    _set_attrs(rgb, {
        "image_size_x": C.IMAGE_W * C.RGB_SUPERSAMPLE, "image_size_y": C.IMAGE_H * C.RGB_SUPERSAMPLE,
        "fov": C.CAMERA_FOV, "sensor_tick": 0.0, **C.RGB_ATTRIBUTES,
    })
    seg = bp_lib.find("sensor.camera.semantic_segmentation")
    # rendered straight at output size, pixel-aligned with the downscaled RGB
    _set_attrs(seg, {"image_size_x": C.IMAGE_W, "image_size_y": C.IMAGE_H,
                     "fov": C.CAMERA_FOV, "sensor_tick": 0.0})
    depth = bp_lib.find("sensor.camera.depth")
    _set_attrs(depth, {"image_size_x": C.IMAGE_W, "image_size_y": C.IMAGE_H,
                       "fov": C.CAMERA_FOV, "sensor_tick": 0.0})
    return rgb, seg, depth


# ---------- world housekeeping ----------

def cleanup_world(client, world):
    """Removes every vehicle, walker, sensor and controller, e.g. left by a crashed run.

    A crash leaves its cameras on the server still rendering every tick. Without
    this they pile up with every auto-restart until the server itself falls over,
    and leftover cars/people would show up unplanned (and mislabeled) in images.
    """
    actors = world.get_actors()
    doomed = [a for pat in ("sensor.*", "controller.*", "walker.*", "vehicle.*") for a in actors.filter(pat)]
    for s in actors.filter("sensor.*"):
        try:
            if s.is_listening:
                s.stop()
        except RuntimeError:
            pass
    if doomed:
        print(f"  removing {len(doomed)} leftover actors from a previous run")
        client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in doomed], True)
        world.tick()


def destroy_actors(client, world, actors):
    ids = [a.id for a in actors if a is not None]
    if ids:
        client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
    world.tick()


# ---------- ego + cameras ----------

def to_transform(d, z_offset=0.0):
    return carla.Transform(
        carla.Location(x=d["x"], y=d["y"], z=d["z"] + z_offset),
        carla.Rotation(pitch=d.get("pitch", 0.0), yaw=d["yaw"], roll=d.get("roll", 0.0)),
    )


def spawn_ego_and_cameras(world, bp_lib, ego_pose):
    """Spawned once per map. Returns (ego, rgb_cam, seg_cam, depth_cam or None, z_offset)."""
    ego = None
    for lift in (0.5, 1.0, 2.0):
        ego = world.try_spawn_actor(ego_blueprint(bp_lib), to_transform(ego_pose, lift))
        if ego:
            break
    if ego is None:
        raise SpawnError("could not spawn ego vehicle")
    for _ in range(20):         # let it drop onto the road, then freeze it
        world.tick()
    z_offset = ego.get_location().z - ego_pose["z"]
    ego.set_simulate_physics(False)

    rgb_bp, seg_bp, depth_bp = camera_blueprints(bp_lib)
    cam_tf = carla.Transform(carla.Location(*C.CAMERA_LOCATION), carla.Rotation(pitch=C.CAMERA_PITCH))
    rgb = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
    seg = world.spawn_actor(seg_bp, cam_tf, attach_to=ego)
    depth = world.spawn_actor(depth_bp, cam_tf, attach_to=ego) if C.SAVE_DEPTH else None
    world.tick()
    return ego, rgb, seg, depth, z_offset


def destroy_ego_and_cameras(client, world, ego, cams):
    for c in cams:
        try:
            if c is not None and c.is_listening:
                c.stop()
        except RuntimeError:
            pass
    try:
        world.tick()            # let in-flight sensor callbacks drain before destroy
    except RuntimeError:
        pass
    destroy_actors(client, world, list(cams) + [ego])


# ---------- scene actors ----------

def spawn_vehicles(world, bp_lib, plan_vehicles, actors):
    """Appends into `actors` as it goes, so the caller can always clean up."""
    spawned = []
    for v in plan_vehicles:
        bp = bp_lib.find(v["blueprint"])
        if v["color"] and bp.has_attribute("color"):
            bp.set_attribute("color", v["color"])
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "traffic")
        actor = world.try_spawn_actor(bp, to_transform(v, 0.3))
        if actor is None:
            raise SpawnError(f"vehicle {v['blueprint']} collided at spawn")
        actors.append(actor)
        spawned.append(actor)
        actor.apply_control(carla.VehicleControl(hand_brake=True))
    return spawned


def freeze(actors):
    for a in actors:
        a.set_simulate_physics(False)


def spawn_pedestrians(world, bp_lib, plan_peds, actors):
    spawned = []
    for p in plan_peds:
        bp = bp_lib.find(p["blueprint"])
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("use_wheelchair"):
            bp.set_attribute("use_wheelchair", "false")
        x, y, z = p["spawn"]
        tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=p["yaw"]))
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            raise SpawnError(f"pedestrian {p['blueprint']} collided at spawn")
        actors.append(actor)
        spawned.append(actor)
    return spawned


def start_walking(walkers, plan_peds):
    # plain WalkerControl, no AI controller: nothing tied to the crowd-nav manager
    for w, p in zip(walkers, plan_peds):
        dx, dy = p["direction"]
        w.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=dx, y=dy, z=0.0), speed=p["speed"]))
