"""Shared set-up for the tests: the fake CARLA (tests/fake/carla.py), a fixed test
config (so the tests do not depend on what is in your config.py), and helpers to plan
or render one scene with a chosen occluder layout."""
import contextlib
import copy
import io
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "fake"), os.path.join(HERE, "..", "generate_dataset")]

import carla  # noqa: E402  (the fake one, found first on sys.path)
import advanced as C  # noqa: E402

_DEFAULTS = {k: copy.deepcopy(getattr(C, k)) for k in dir(C) if k.isupper()}

TEST_CONFIG = dict(
    MAPS=["FakeCity"],
    WEATHERS={
        "clear": dict(enabled=True, images=6, pedestrians=1, vehicles=0, cyclists=0, trucks=0, occlusion=0),
        "rain": dict(enabled=True, images=30, pedestrians=1, vehicles=2, cyclists=0, trucks=0, occlusion=(50, 95)),
    },
    SCENES={s: dict(enabled=True, share=1) for s in C.SCENE_SITE_KINDS},
    SEQUENCE=dict(enabled=True, frames=5, walk_speed=(0.6, 1.0)),
    SIDEWALK_PEDESTRIANS=0, SCENE_FOLLOWS_WEATHER=True, MAX_USES_PER_SPOT=None,
    PED_RUNNER_PROBABILITY=0.0, SKIP_FLAGGED_IMAGES=False, RGB_SUPERSAMPLE=1,
    EGO_BLUEPRINT="vehicle.lincoln.mkz_2020",
    CAR_BLUEPRINTS=["vehicle.audi.tt", "vehicle.tesla.model3"],
    TRUCK_BLUEPRINTS=["vehicle.carlamotors.european_hgv"],
    CYCLIST_BLUEPRINTS=["vehicle.bh.crossbike"],
)

from engine.selftest import LAYOUTS, PARAMS as SCENE_PARAMS  # noqa: E402,F401  (every occluder layout)


def configure(**overrides):
    """Reset advanced/config settings to their defaults, then apply the test config."""
    for k, v in _DEFAULTS.items():
        setattr(C, k, copy.deepcopy(v))
    for k, v in {**TEST_CONFIG, **overrides}.items():
        setattr(C, k, copy.deepcopy(v))


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def make_planner():
    """A ScenePlanner on FakeCity, as MapSession.open() builds it (without spawning anything)."""
    from engine.carla_io import vehicle_pool, walker_pools
    from engine.planner import ScenePlanner
    from engine.roads import find_all_sites
    world = carla.Client().get_world()
    lib = world.get_blueprint_library()
    with quiet():
        sites = find_all_sites(world.get_map(), C)
        blueprints = {"car": vehicle_pool(lib, C.CAR_BLUEPRINTS), "truck": vehicle_pool(lib, C.TRUCK_BLUEPRINTS),
                      "cyclist": vehicle_pool(lib, C.CYCLIST_BLUEPRINTS)}
    return ScenePlanner(world.get_map(), sites, [], walker_pools(lib), blueprints)


def plan_layout(planner, scene, layout, seed, tries=300):
    """A plan of `scene` with exactly this occluder layout, or None."""
    from engine.selftest import plan_layout as plan
    return plan(planner, scene, layout, "rain", f"test{seed}", tries)


def open_session():
    """A real MapSession on the fake world: sites found, ego + 3 cameras spawned, rain set."""
    from engine.carla_io import MapSession
    session = MapSession(carla.Client(), carla, "FakeCity", "/Game/Carla/Maps/FakeCity")
    with quiet():
        session.open()
    session.set_weather("rain", "test", random.Random(0))
    return session


# ---------- plain 3D sight lines, independent of the planner's screen-box estimate ----------

def camera_xyz(ego, z_offset=0.0):
    """Where the dashcam is."""
    yaw = math.radians(ego["yaw"])
    cx, cy, cz = C.CAMERA_LOCATION
    return (ego["x"] + math.cos(yaw) * cx - math.sin(yaw) * cy,
            ego["y"] + math.sin(yaw) * cx + math.cos(yaw) * cy, ego["z"] + z_offset + cz)


def segment_hits_box(p, q, box):
    """Does the 3D line p -> q pass through box = (x, y, z0, yaw, length, width, height)
    (standing on z0, turned by yaw)?"""
    x, y, z0, yaw, length, width, height = box
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

    def local(pt):
        dx, dy = pt[0] - x, pt[1] - y
        return dx * c + dy * s, -dx * s + dy * c, pt[2] - z0

    a, b = local(p), local(q)
    t0, t1 = 0.0, 1.0
    for i, (lo, hi) in enumerate(((-length / 2, length / 2), (-width / 2, width / 2), (0.0, height))):
        d = b[i] - a[i]
        if abs(d) < 1e-12:
            if not lo <= a[i] <= hi:
                return False
            continue
        ta, tb = sorted(((lo - a[i]) / d, (hi - a[i]) / d))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def vehicle_solids(v):
    """A vehicle as solid boxes: a car is its low body plus the cabin in the middle (as real
    cars: you can see over the bonnet), a truck / cyclist is one block."""
    length, width, height = {"car": C.CAR_SIZE, "truck": C.TRUCK_SIZE, "cyclist": C.CYCLIST_SIZE}[v["category"]]
    if v["category"] != "car":
        return [(v["x"], v["y"], v["z"], v["yaw"], length, width, height)]
    return [(v["x"], v["y"], v["z"], v["yaw"], length, width, C.CAR_BODY_HEIGHT),
            (v["x"], v["y"], v["z"], v["yaw"], length * C.CAR_CABIN_LENGTH_PART, width * 0.9, height)]


def hidden_share(cam, ped, x, y, solids):
    """Share of sight lines from the camera to points on the pedestrian's body (3 heights x
    3 across) that pass through a vehicle."""
    h = C.PED_HEIGHT[ped["type"]]
    dx, dy = x - cam[0], y - cam[1]
    n = math.hypot(dx, dy)
    side = (-dy / n, dx / n)
    lines = [(x + side[0] * o, y + side[1] * o, ped["ground_z"] + h * f)
             for o in (-0.15, 0.0, 0.15) for f in (0.1, 0.5, 0.9)]
    return sum(any(segment_hits_box(cam, p, b) for b in solids) for p in lines) / len(lines)
