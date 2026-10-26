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
WEATHER_LOOK = {                    # (min, max) sampled; anything not listed is 0
    "clear":         {"cloudiness": (0, 10), "fog_density": (0, 2), "wind_intensity": (0, 15)},
    "partly_cloudy": {"cloudiness": (30, 60), "fog_density": (0, 3), "wind_intensity": (5, 30)},
    "overcast":      {"cloudiness": (85, 100), "fog_density": (2, 8), "wind_intensity": (10, 40)},
    "rain":          {"cloudiness": (80, 100), "precipitation": (40, 80), "precipitation_deposits": (30, 70),
                      "wetness": (50, 90), "fog_density": (3, 10), "wind_intensity": (20, 50)},
    # CARLA 0.9.16 has no snow assets: this is only a cold, hazy, sleet-like approximation
    # "snowfall":      {"cloudiness": (95, 100), "precipitation": (15, 35), "precipitation_deposits": (40, 70),
    #                   "wetness": (20, 40), "fog_density": (20, 35), "fog_distance": (10, 30),
    #                   "wind_intensity": (30, 60)},
}
WEATHER_RESAMPLE_EVERY = 5          # new sun/cloud values (same weather type) every N jobs; 0 = never

# --- pedestrians ---
PED_SPEED_RANGE = {"child": (0.7, 1.4), "woman": (0.9, 1.5), "man": (1.0, 1.7)}   # m/s
PED_RUNNER_PROBABILITY = 0.08
PED_RUN_SPEED_RANGE = (2.3, 3.2)
PED_PROGRESS_RANGE = (0.12, 0.88)   # where on the crossing at capture (0 = start curb, 1 = far curb)
PED_PREROLL_SECONDS = (1.0, 2.0)    # walk this long before the first frame, so they are mid-stride
PED_MIN_GAP = 0.9
PED_HEIGHT = {"child": 1.25, "woman": 1.65, "man": 1.78}   # planning estimate only
SIDEWALK_PED_DISTANCE = (4.0, 25.0)
SIDEWALK_PED_STANDING_PROBABILITY = 0.3
PED_TYPE_OVERRIDES = {}             # blueprint id -> "child"/"woman"/"man", if "python main.py --check" shows a wrong type
PED_BLUEPRINT_EXCLUDE = []

# --- scene types ---
OCCLUDED_SCENES = ("occluded_by_traffic", "occluded_by_parked")
SCENE_SITE_KINDS = {                # where each scene type may happen
    "crosswalk": ["crosswalk"],
    "jaywalk_road": ["road"],
    "jaywalk_junction": ["junction"],
    "occluded_by_traffic": ["crosswalk", "road", "junction"],
    "occluded_by_parked": ["road", "junction"],     # nobody parks on a marked crosswalk
}

# --- occlusion ---
OCCLUSION_TOLERANCE = 0.15          # frame is flagged if measured occlusion misses the planned value by more
OCCLUSION_NUMBER_SPREAD = 10.0      # a single occlusion number v in config means the range v-10 .. v+10 %
LOWEST_REACHABLE_OCCLUSION = 0.5    # behind a car the most hidden moment is rarely below this (warning only)
OCCLUDER_TRUCK_CHANCE = 0.3         # when trucks are available, chance the occluder is a truck
OCCLUDED_EGO_DISTANCE_RANGE = (12.0, 35.0)
ADJACENT_CARS_AHEAD = (1.0, 25.0)   # occluding cars next to the ego: metres ahead of the ego
PARKED_NOSE_GAP = (0.5, 6.0)        # gap between parked car's front and the pedestrian's path
SEQUENCE_MAX_SECONDS = 15.0         # longest walk to cover all frames of a sequence
FULLY_VISIBLE = 0.03                # "completely visible" = at most this much hidden
PROTECT_TARGET_MAX_COVER = 0.05     # non-occluder traffic may cover at most this much of the target

# --- crossing sites ---
JAYWALK_MIN_DIST_FROM_CROSSWALK = 30.0
JAYWALK_SITE_SPACING = 10.0         # metres between two mid-block crossing spots
JAYWALK_ROAD_WIDTH_RANGE = (5.0, 25.0)
JUNCTION_ARM_OFFSET = 4.0           # crossing line this far out from a T-junction's edge
JUNCTION_MIN_DIST_FROM_CROSSWALK = 10.0   # "crosswalk is missing": skip arms that have one

# --- ego / camera ---
EGO_FALLBACK_BLUEPRINT = "vehicle.lincoln.mkz_2020"
EGO_DISTANCE_RANGE = (8.0, 40.0)    # ego to crossing centre, metres
MIN_VIEW_DISTANCE_GAP = 6.0         # same spot + lane + weather: views must differ by this many metres
EGO_YAW_JITTER_DEG = 5.0
EGO_LATERAL_JITTER = 0.25
MAX_VIEW_BEARING_DEG = 30.0
EGO_SIZE = (5.0, 2.2)
IMAGE_W, IMAGE_H = 1920, 1080
RGB_SUPERSAMPLE = 2                 # RGB rendered at 2x then downscaled; 1 = faster/lighter
CAMERA_FOV = 90.0
CAMERA_LOCATION = (0.8, 0.0, 1.8)   # x forward, y right, z up, relative to ego
CAMERA_PITCH = -10.0
CAMERA_LIFT_RANGE = (0.0, 0.4)      # per image: camera this much higher (car .. van), by lifting the ego
CAMERA_PITCH_JITTER_DEG = 3.0       # per image: camera tilt -10 +/- this
RGB_ATTRIBUTES = {
    "motion_blur_intensity": 0.0, "lens_flare_intensity": 0.0,
    "exposure_speed_up": 15.0, "exposure_speed_down": 15.0,
    "lens_circle_multiplier": 1.5, "lens_circle_falloff": 4.0,
    "chromatic_aberration_intensity": 0.5, "chromatic_aberration_offset": 0.0,
}
CAPTURE_SETTLE_TICKS = 6
JPEG_QUALITY = 95
BOX_COLOR = (0, 255, 0)             # drawn box colour, OpenCV B, G, R (green). no text is drawn
BOX_THICKNESS = 2

# config.SAVE flag -> output folder (labels_3d/ is always written)
OUTPUT_FOLDERS = {
    "image": "images", "depth": "depth", "depth_color": "depth_color",
    "seg_color": "seg_color", "seg_raw": "seg_raw",
    "bbox_2d_image": "images_2d_bbox", "bbox_3d_image": "images_3d_bbox", "labels_yolo": "labels",
}

# --- traffic placement ---
EGO_LANE_WIDTH = 2.4                # same-direction traffic drifts at most (2.4 - width) / 2 off lane centre
ONCOMING_MAX_DRIFT = 0.3
CAR_SIZE = (5.0, 2.1, 1.6)          # length, width, height for planning
CAR_BODY_HEIGHT = 0.95              # occlusion estimate: car body height (bonnet / boot) ...
CAR_CABIN_LENGTH_PART = 0.55        # ... plus a full-height cabin over this middle part of the length
TRUCK_SIZE = (10.0, 2.7, 3.6)
CYCLIST_SIZE = (2.0, 0.8, 1.8)
TRAFFIC_FAR_MAX = 45.0
NEAR_SIDE_PROBABILITY = 0.4
EGO_CLEARANCE = 8.0
VEHICLE_GAP = 1.5
CROSSING_CLEARANCE = 1.0

# --- labels / quality checks ---
MIN_VISIBLE_PIXELS = {"pedestrian": 80, "vehicle": 250, "truck": 250, "cyclist": 120}
MIN_BOX_SIZE_PX = 8
MAX_LABEL_DISTANCE = 80.0
LABEL_PARKED_VEHICLES = True        # the map's own parked cars
PARKED_VEHICLE_CLASS = {"Car": "vehicle", "Truck": "truck", "Bus": "truck"}
MIN_TARGET_PIXELS = 300             # target pedestrian (unoccluded silhouette) must be at least this big
MIN_ROAD_FRACTION = 0.12            # below this the camera is probably blocked
SEG_TAGS = {                        # CARLA >= 0.9.14
    "pedestrian": [12],
    "vehicle": [14, 15, 16],        # car, truck, bus (used for the ego bonnet mask)
    "road_like": [1, 2, 10, 24, 25],
}
# a box's class comes from CARLA's own semantic tag on that object's pixels
TAG_CLASS = {12: "pedestrian", 14: "vehicle", 15: "truck", 16: "truck",
             18: "cyclist", 19: "cyclist", 13: "cyclist"}
TAG_NAMES = {12: "Pedestrian", 13: "Rider", 14: "Car", 15: "Truck", 16: "Bus",
             18: "Motorcycle", 19: "Bicycle"}
RIDER_TAG = 13                      # a rider counts with the bicycle / motorcycle it sits on
DEPTH_MATCH_MARGIN = 0.75           # a pixel belongs to an object if its depth is within the object's box +- this (m)

# --- run control ---
MAX_RENDERS_PER_JOB = 6             # renders that crash / fail to spawn before a job is skipped
MAX_PLANS_PER_JOB = 400
RELOAD_WORLD_EVERY = 300            # jobs
MAX_JOBS_PER_PROCESS = 1000
MAX_CRASHES_WITHOUT_PROGRESS = 6
EXIT_DONE, EXIT_FATAL, EXIT_NO_SERVER, EXIT_RECYCLE = 0, 2, 3, 10   # worker -> supervisor
EXIT_STOPPED = 130                  # stopped by Ctrl+C: never restarted
STOP_WAIT_SECONDS = 60              # on Ctrl+C, time the generator gets to empty the CARLA world

COUNT_KEYS = ("pedestrians", "vehicles", "cyclists", "trucks")


# ---------- helpers ----------

def roll(value, rng):
    """Config value -> number: an int is exact, a (min, max) pair is random."""
    if isinstance(value, (tuple, list)):
        return rng.randint(int(value[0]), int(value[1]))
    return int(value or 0)


def occlusion_range(value):
    """Occlusion % from config -> (min, max) as fractions 0..1: the hidden share of the
    target at its most hidden moment. A single number v means v +- OCCLUSION_NUMBER_SPREAD."""
    if isinstance(value, (tuple, list)):
        return value[0] / 100.0, value[1] / 100.0
    v = float(value or 0)
    return max(0.0, v - OCCLUSION_NUMBER_SPREAD) / 100.0, min(100.0, v + OCCLUSION_NUMBER_SPREAD) / 100.0


def max_of(value):
    return max(value) if isinstance(value, (tuple, list)) else (value or 0)


def enabled_weathers():
    return [w for w, spec in WEATHERS.items() if spec.get("enabled", True) and spec.get("images", 0) > 0]


def enabled_scenes():
    return [s for s, spec in SCENES.items() if spec.get("enabled", True) and spec.get("share", 0) > 0]


def frames_per_job(scene):
    if scene in OCCLUDED_SCENES and SEQUENCE.get("enabled", True):
        return max(1, int(SEQUENCE.get("frames", 1)))
    return 1


def can_occlude(row):
    return max_of(row.get("vehicles")) + max_of(row.get("trucks")) > 0 and max_of(row.get("occlusion")) > 0


def class_names():
    names = ["pedestrian", "vehicle", "truck", "cyclist"]
    if SPLIT_CROSSING_CLASS:
        names = ["crossing_pedestrian"] + names
    return names


def validate():
    """Raises ValueError on a config mistake, before any image is generated."""
    def is_count(v, lo, hi):
        if isinstance(v, (tuple, list)):
            return len(v) == 2 and lo <= v[0] <= v[1] <= hi
        return v is None or (isinstance(v, int) and lo <= v <= hi)

    if not enabled_weathers():
        raise ValueError("no weather is enabled with images > 0")
    for w in enabled_weathers():
        row = WEATHERS[w]
        if w not in WEATHER_LOOK:
            raise ValueError(f"weather '{w}' has no look in advanced.WEATHER_LOOK")
        unknown = set(row) - {"enabled", "images", "occlusion", *COUNT_KEYS}
        if unknown:
            raise ValueError(f"WEATHERS['{w}'] has unknown keys {sorted(unknown)}")
        if not is_count(row.get("pedestrians"), 1, 8):
            raise ValueError(f"{w}: 'pedestrians' must be 1..8 or (min, max)")
        for k in ("vehicles", "cyclists", "trucks"):
            if not is_count(row.get(k), 0, 8):
                raise ValueError(f"{w}: '{k}' must be 0..8 or (min, max)")
        if not is_count(row.get("occlusion"), 0, 100):
            raise ValueError(f"{w}: 'occlusion' must be 0..100 or (min, max)")
    unknown = set(SCENES) - set(SCENE_SITE_KINDS)
    if unknown:
        raise ValueError(f"unknown scene types {sorted(unknown)}; allowed {list(SCENE_SITE_KINDS)}")
    if not enabled_scenes():
        raise ValueError("no scene type is enabled")
    if set(PEDESTRIAN_MIX) - set(PED_HEIGHT) or not any(v > 0 for v in PEDESTRIAN_MIX.values()):
        raise ValueError("PEDESTRIAN_MIX keys must be child/woman/man")
    unknown = set(SAVE) - set(OUTPUT_FOLDERS)
    if unknown:
        raise ValueError(f"unknown SAVE flags {sorted(unknown)}")
    if MAX_USES_PER_SPOT is not None and (not isinstance(MAX_USES_PER_SPOT, int) or MAX_USES_PER_SPOT < 1):
        raise ValueError("MAX_USES_PER_SPOT must be a whole number >= 1, or None")
    if not MAPS or not all(isinstance(m, str) and m for m in MAPS):
        raise ValueError("MAPS must list at least one map name")
    if TRAIN_VAL_TEST and abs(sum(TRAIN_VAL_TEST) - 1.0) > 1e-6:
        raise ValueError("TRAIN_VAL_TEST must add up to 1")
