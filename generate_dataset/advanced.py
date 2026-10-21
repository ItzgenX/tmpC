"""Fine tuning. You normally never need to edit this file - config.py has the
everyday settings. The code reads every setting through this module, which
includes everything from config.py.
"""
from config import *  # noqa: F401,F403

# --- server / timing ---
CARLA_ARGS = ["-RenderOffScreen", "-quality-level=Epic"]
CARLA_STARTUP_SECONDS = 60
CARLA_TIMEOUT = 30.0                # seconds for a normal server call
MAP_LOAD_TIMEOUT = 180.0
FIXED_DELTA_SECONDS = 0.05          # 20 ticks per simulated second
SERVER_WAIT_SECONDS = 1800          # how long to wait for an unreachable server

# --- weather look (daytime only) ---
# sun 70-90 deg high: always >60 deg above the top of the image, never in frame or blinding
SUN_ALTITUDE_RANGE = (70.0, 90.0)
SUN_AZIMUTH_RANGE = (0.0, 360.0)
WEATHER_LOOK = {                    # (min, max) sampled per image; rain/wetness always 0
    "clear":         {"cloudiness": (0, 10),   "fog_density": (0, 2),  "wind_intensity": (0, 15)},
    "partly_cloudy": {"cloudiness": (30, 60),  "fog_density": (0, 3),  "wind_intensity": (5, 30)},
    "overcast":      {"cloudiness": (85, 100), "fog_density": (2, 8),  "wind_intensity": (10, 40)},
}

# --- pedestrians ---
PED_SPEED_RANGE = {"child": (0.7, 1.4), "woman": (0.9, 1.5), "man": (1.0, 1.7)}   # m/s
PED_RUNNER_PROBABILITY = 0.08
PED_RUN_SPEED_RANGE = (2.3, 3.2)
PED_PROGRESS_RANGE = (0.12, 0.88)   # where on the crossing at capture (0 = start curb, 1 = far curb)
PED_PREROLL_SECONDS = (1.0, 2.0)    # walk this long before capture, so they are mid-stride
PED_MIN_GAP = 0.9
PED_HEIGHT = {"child": 1.25, "woman": 1.65, "man": 1.78}   # planning estimate only
SIDEWALK_PED_DISTANCE = (4.0, 25.0)
SIDEWALK_PED_STANDING_PROBABILITY = 0.3
PED_TYPE_OVERRIDES = {}             # blueprint id -> "child"/"woman"/"man", if list_blueprints.py shows a wrong type
PED_BLUEPRINT_EXCLUDE = []

# --- occlusion ---
OCCLUSION_LEVELS = {"none": (0.0, 0.10), "partial": (0.10, 0.40), "heavy": (0.40, 0.80)}
OCCLUSION_TOLERANCE = 0.07          # measured % may miss the level range by this much
OCCLUDER_TRUCK_PROBABILITY = 0.3    # when the image has a truck, chance the truck is the occluder
OCCLUDED_EGO_DISTANCE_RANGE = (16.0, 30.0)
OCCLUDER_MIN_AHEAD = 5.0
PROTECT_TARGET_MAX_COVER = 0.05     # other traffic may cover at most this much of the target

# --- crossing sites ---
JAYWALK_MIN_DIST_FROM_CROSSWALK = 30.0
JAYWALK_SITE_SPACING = 20.0
JAYWALK_ROAD_WIDTH_RANGE = (5.0, 25.0)

# --- ego / camera ---
EGO_FALLBACK_BLUEPRINT = "vehicle.lincoln.mkz_2020"
EGO_DISTANCE_RANGE = (10.0, 30.0)   # ego to crossing centre, metres
DISTANCE_BIN = 2.0                  # no two images share (site, lane, distance bin, weather, occlusion level)
EGO_YAW_JITTER_DEG = 2.0
EGO_LATERAL_JITTER = 0.25
MAX_VIEW_BEARING_DEG = 30.0
EGO_SIZE = (5.0, 2.2)
IMAGE_W, IMAGE_H = 1920, 1080
RGB_SUPERSAMPLE = 2                 # RGB rendered at 2x then downscaled; 1 = faster/lighter
CAMERA_FOV = 90.0
CAMERA_LOCATION = (0.8, 0.0, 1.8)   # x forward, y right, z up, relative to ego
CAMERA_PITCH = -10.0
RGB_ATTRIBUTES = {
    "motion_blur_intensity": 0.0, "lens_flare_intensity": 0.0,
    "exposure_speed_up": 15.0, "exposure_speed_down": 15.0,
    "lens_circle_multiplier": 1.5, "lens_circle_falloff": 4.0,
    "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0,
}
CAPTURE_SETTLE_TICKS = 6
JPEG_QUALITY = 95

# --- traffic placement ---
EGO_LANE_WIDTH = 2.4                # same-direction traffic drifts at most (2.4 - width) / 2 off lane centre
ONCOMING_MAX_DRIFT = 0.3
CAR_SIZE = (5.0, 2.1, 1.6)          # length, width, height for planning
TRUCK_SIZE = (10.0, 2.7, 3.6)
TRAFFIC_FAR_MAX = 35.0
NEAR_SIDE_PROBABILITY = 0.4
EGO_CLEARANCE = 8.0
VEHICLE_GAP = 1.5
CROSSING_CLEARANCE = 1.0

# --- labels / quality gates ---
MIN_VISIBLE_PIXELS = {"pedestrian": 80, "vehicle": 250, "truck": 250}
MIN_BOX_SIZE_PX = 8
MAX_LABEL_DISTANCE = 80.0
LABEL_PARKED_VEHICLES = True
PARKED_VEHICLE_CLASS = {"Car": "vehicle", "Truck": "truck", "Bus": "truck"}
MIN_MAIN_PED_PIXELS = 400
MIN_ROAD_FRACTION = 0.12
SEG_TAGS = {                        # CARLA >= 0.9.14
    "pedestrian": [12],
    "vehicle": [14, 15, 16],
    "road_like": [1, 2, 10, 24, 25],
}

# --- run control ---
MAX_ATTEMPTS_PER_SCENE = 8
RELOAD_WORLD_EVERY = 400
MAX_SCENES_PER_PROCESS = 1500
MAX_CRASHES_WITHOUT_PROGRESS = 6


# ---------- helpers ----------

class _Scene:
    def __init__(self, values):
        self.__dict__.update(values)


def settings_for(weather=None):
    """SCENE with this weather's changes applied, as attributes (s.cars, s.truck_chance, ...)."""
    values = dict(SCENE)
    if weather:
        values.update({k: v for k, v in WEATHERS[weather].items() if k != "share"})
    return _Scene(values)


def weather_shares():
    return {w: spec.get("share", 1) for w, spec in WEATHERS.items()}


def class_names():
    names = ["pedestrian", "vehicle", "truck"]
    if SPLIT_CROSSING_CLASS:
        names = ["crossing_pedestrian"] + names
    return names


def validate():
    """Raises ValueError on a config mistake, before any image is generated."""
    def check_range(name, w, rng, lo, hi):
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2 and lo <= rng[0] <= rng[1] <= hi):
            raise ValueError(f"{w}: '{name}' must be (min, max) within {lo}..{hi}, got {rng}")

    for w, spec in WEATHERS.items():
        if w not in WEATHER_LOOK:
            raise ValueError(f"weather '{w}' has no look in advanced.WEATHER_LOOK")
        bad = [k for k in spec if k != "share" and k not in SCENE]
        if bad:
            raise ValueError(f"WEATHERS['{w}'] has unknown keys {bad}; allowed: share, {list(SCENE)}")
        s = settings_for(w)
        check_range("crossing_pedestrians", w, s.crossing_pedestrians, 1, 3)
        check_range("sidewalk_pedestrians", w, s.sidewalk_pedestrians, 0, 5)
        check_range("cars", w, s.cars, 0, 6)
        if set(s.pedestrian_types) - set(PED_HEIGHT) or not any(v > 0 for v in s.pedestrian_types.values()):
            raise ValueError(f"{w}: 'pedestrian_types' keys must be child/woman/man")
        if set(s.occlusion) - set(OCCLUSION_LEVELS) or not any(v > 0 for v in s.occlusion.values()):
            raise ValueError(f"{w}: 'occlusion' keys must be {list(OCCLUSION_LEVELS)}")
        occluded = any(v > 0 for k, v in s.occlusion.items() if k != "none")
        if occluded and s.cars[1] == 0 and s.truck_chance <= 0:
            raise ValueError(f"{w}: occluded images need a car or truck to hide the pedestrian")
