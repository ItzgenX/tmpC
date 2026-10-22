"""Talking to CARLA: spawning / removing actors (spawner), and everything that
lives for one map (session): world, crossing sites, ego + cameras, weather, capture.
"""
import queue
import random
import time

import carla

import advanced as C
from engine.labels import hood_mask, seg_tags
from engine.planner import PlanError, ScenePlanner
from engine.roads import find_all_sites
from engine.stop import protected


# ============================================================================
# spawner
# ============================================================================
# Spawns what the planner decided. No decisions are made here.

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

def cleanup_world(client, world, why="left over from a previous run"):
    """Removes every vehicle, walker, sensor and controller from the world.

    Used at start-up (a crash leaves its cameras on the server still rendering
    every tick; they would pile up with every restart, and leftover cars/people
    would show up unplanned in images) and on Ctrl+C / normal exit.
    """
    sync = world.get_settings().synchronous_mode
    actors = world.get_actors()
    doomed = [a for pat in ("sensor.*", "controller.*", "walker.*", "vehicle.*") for a in actors.filter(pat)]
    for s in actors.filter("sensor.*"):
        try:
            if s.is_listening:
                s.stop()
        except RuntimeError:
            pass
    if not doomed:
        return
    if sync:
        world.tick()            # let in-flight sensor images arrive before their camera is destroyed
    client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in doomed], sync)
    if sync:
        world.tick()
    try:
        print(f"  removed {len(doomed)} actors {why}")
    except OSError:
        pass                    # terminal already closed


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
    ego, bp = None, ego_blueprint(bp_lib)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")   # large maps (Town12/13) stream tiles around the "hero"
    for lift in (0.5, 1.0, 2.0):
        ego = world.try_spawn_actor(bp, to_transform(ego_pose, lift))
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
    want_depth = C.SAVE.get("depth") or C.SAVE.get("depth_color")
    depth = world.spawn_actor(depth_bp, cam_tf, attach_to=ego) if want_depth else None
    world.tick()
    return ego, rgb, seg, depth, z_offset


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


def hide(actors):
    """Moves (frozen) vehicles 200 m underground; returns their transforms for restore()."""
    saved = [a.get_transform() for a in actors]
    for a, tf in zip(actors, saved):
        loc = tf.location
        a.set_transform(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z - 200.0), tf.rotation))
    return saved


def restore(actors, saved):
    for a, tf in zip(actors, saved):
        a.set_transform(tf)


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


# ============================================================================
# session
# ============================================================================
# Everything that lives for one map: the world, crossing sites, the planner,
# the ego vehicle with its three cameras (spawned once, reused for every image),
# the weather, and frame-exact capture.

class MapSession:
    def __init__(self, client, carla_module, map_name):
        self.client = client
        self.carla = carla_module
        self.map_name = map_name
        self.world = None
        self.ego = self.rgb = self.seg = self.depth = None
        self.weather_key = None

    # ---------- world ----------

    def _set_sync(self, on):
        s = self.world.get_settings()
        s.synchronous_mode = on
        s.fixed_delta_seconds = C.FIXED_DELTA_SECONDS if on else None
        self.world.apply_settings(s)

    def _load(self, reload=False):
        self.world = self.client.get_world()
        self._set_sync(False)
        if reload or self.world.get_map().name.split("/")[-1] != self.map_name:
            print(f"  loading {self.map_name}")
            self.client.set_timeout(C.MAP_LOAD_TIMEOUT)
            try:
                self.world = self.client.reload_world(False) if reload else self.client.load_world(self.map_name)
            finally:
                self.client.set_timeout(C.CARLA_TIMEOUT)
        self._set_sync(True)
        self.world.tick()
        self.weather_key = None

    def open(self, reload=False):
        """Load the map, scan it, spawn ego + cameras. Raises if the map is unusable."""
        self._load(reload)
        cleanup_world(self.client, self.world)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        sites = find_all_sites(self.map, C)
        # parked cars baked into the map: read once per map (reading them every
        # frame leaks server memory and was a known crash cause)
        self.parked = []
        for label_name, cls in C.PARKED_VEHICLE_CLASS.items():
            label = getattr(self.carla.CityObjectLabel, label_name, None)
            if label is not None:
                self.parked += [(o, cls, label_name) for o in self.world.get_environment_objects(label)]
        env = [(o.bounding_box.location.x, o.bounding_box.location.y, o.bounding_box.extent.x,
                o.bounding_box.extent.y, o.bounding_box.rotation.yaw) for o, _, _ in self.parked]

        pools = walker_pools(self.bp_lib)
        print("  pedestrian blueprints: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
        blueprints = {"car": vehicle_pool(self.bp_lib, C.CAR_BLUEPRINTS),
                      "truck": vehicle_pool(self.bp_lib, C.TRUCK_BLUEPRINTS),
                      "cyclist": vehicle_pool(self.bp_lib, C.CYCLIST_BLUEPRINTS)}
        self.planner = ScenePlanner(self.map, sites, env, pools, blueprints)
        if not sites:
            raise RuntimeError(f"no crossing sites found on {self.map_name}")
        self._spawn_ego()

    def _spawn_ego(self):
        rng = random.Random(f"{C.SEED}:{self.map_name}:ego")
        poses = []
        for _ in range(300):
            try:
                poses.append(self.planner.plan_ego(rng, rng.choice(self.planner.sites))[0])
            except PlanError:
                continue
            if len(poses) == 3:
                break
        if not poses:
            raise RuntimeError(f"no usable ego position on {self.map_name}")
        for pose in poses:
            try:
                self.ego, self.rgb, self.seg, self.depth, z_off = \
                    spawn_ego_and_cameras(self.world, self.bp_lib, pose)
                break
            except SpawnError:
                continue
        else:
            raise RuntimeError(f"ego vehicle could not be spawned on {self.map_name}")
        self.planner.ego_z_offset = z_off

        # bonnet mask: vehicle pixels at the bottom that stay put across poses
        segs = []
        for pose in poses:
            self.ego.set_transform(to_transform(pose, z_off))
            _, (img,) = self.capture([self.seg], C.CAPTURE_SETTLE_TICKS)
            segs.append(seg_tags(img))
        self.hood = hood_mask(segs)

    def close(self, why="at shutdown"):
        """Empty the world (ego, cameras, anything left from a scene) and hand the
        server back in async mode. Runs to the end even if Ctrl+C is pressed again;
        each step is tried even if an earlier one failed."""
        if self.world is None:
            return
        with protected():
            try:
                cleanup_world(self.client, self.world, why)
            except RuntimeError:
                pass        # connection already gone
            try:
                self._set_sync(False)
            except RuntimeError:
                pass
            self.ego = self.rgb = self.seg = self.depth = None

    def reload(self):
        print("  periodic world reload")
        self.close("before the periodic world reload")
        self.open(reload=True)

    # ---------- weather ----------

    def set_weather(self, name, key, rng):
        """Applies a new look of weather `name` only when `key` changes (start of a
        weather block, or every WEATHER_RESAMPLE_EVERY jobs). Returns the parameters."""
        if key == self.weather_key:
            return self.weather_params
        params = {k: round(rng.uniform(*r), 2) for k, r in C.WEATHER_LOOK[name].items()}
        params.update({"sun_altitude_angle": round(rng.uniform(*C.SUN_ALTITUDE_RANGE), 2),
                       "sun_azimuth_angle": round(rng.uniform(*C.SUN_AZIMUTH_RANGE), 2)})
        for k in ("precipitation", "precipitation_deposits", "wetness"):
            params.setdefault(k, 0.0)
        weather = self.carla.WeatherParameters()
        for k, v in params.items():
            if hasattr(weather, k):
                setattr(weather, k, v)
        self.world.set_weather(weather)
        self.weather_key, self.weather_params = key, {"name": name, **params}
        return self.weather_params

    # ---------- capture ----------

    def cameras(self):
        return [self.rgb, self.seg] + ([self.depth] if self.depth is not None else [])

    def capture(self, cams, settle_ticks):
        """Listen, tick, and return the images from exactly the last tick."""
        qs = [queue.Queue() for _ in cams]
        for cam, q in zip(cams, qs):
            cam.listen(q.put)
        try:
            frame = None
            for _ in range(max(1, settle_ticks)):
                frame = self.world.tick()
            imgs = [self._frame_from(q, frame) for q in qs]
        finally:
            for cam in cams:
                cam.stop()
        return frame, imgs

    @staticmethod
    def _frame_from(q, frame, timeout=20.0):
        end = time.time() + timeout
        while True:
            img = q.get(timeout=max(0.1, end - time.time()))
            if img.frame == frame:
                return img
            if img.frame > frame:
                raise RuntimeError("camera skipped the capture frame")
