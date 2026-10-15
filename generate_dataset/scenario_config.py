import os
import datetime

import carla


# None uses whatever map the server already has loaded. Set a name to load one.
# Ignored if MAP_ROTATION is non-empty.
MAP_NAME = None

# List of maps to cycle through instead of running on one map.
# Each scenario's image count splits evenly across them.
MAP_ROTATION = []

# everything gets written under this folder
OUTPUT_BASE_DIR = "dataset_scenarios"
DRAW_2D_BBOX = True
DRAW_3D_BBOX = True

# trucks get their own YOLO class when True, else labeled "Vehicle"
SEPARATE_TRUCK_CLASS = True


def build_class_map(separate_truck_class=SEPARATE_TRUCK_CLASS):
    class_map = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'E-Scooter + Rider': 2,
        'Vehicle': 3,
        'Motorcycle': 4,
    }
    if separate_truck_class:
        class_map['Truck'] = 5
    return class_map


CLASS_MAP = build_class_map()


# --- exact image-type split ---
# each scenario's image count splits across these four buckets by exact
# count, not probability. percentages don't need to sum to 100.
IMAGE_TYPE_SPLIT = {
    "crosswalk_clear": 25,
    "crosswalk_occluded": 25,
    "jaywalk_clear": 25,
    "jaywalk_occluded": 25,
}

# target occlusion % for "*_occluded" images, rolled uniformly in this range
OCCLUSION_PERCENT_RANGE = (1, 100)


# each row is one weather scenario. pedestrians/vehicles/cyclists_scooters/
# trucks are (min, max) ranges rolled per image.
# allow_jaywalk/allow_occlusion: whether this scenario is eligible for those
# buckets. a disallowed bucket folds into the nearest allowed one.
SCENARIOS = [
    dict(
        name="clear",
        weather=carla.WeatherParameters.ClearNoon,
        pedestrians=(1, 2), vehicles=(0, 1), cyclists_scooters=(0, 1), trucks=(0, 1),
        allow_jaywalk=False, allow_occlusion=False,
        number_images_to_generate=700,
    ),
    dict(
        name="partly_cloudy",
        weather=carla.WeatherParameters.CloudyNoon,
        pedestrians=(1, 3), vehicles=(0, 2), cyclists_scooters=(0, 2), trucks=(0, 1),
        allow_jaywalk=True, allow_occlusion=False,
        number_images_to_generate=800,
    ),
    dict(
        name="overcast",
        weather=carla.WeatherParameters.WetCloudyNoon,
        pedestrians=(2, 5), vehicles=(1, 3), cyclists_scooters=(1, 3), trucks=(0, 2),
        allow_jaywalk=True, allow_occlusion=True,
        number_images_to_generate=1700,
    ),
    dict(
        name="rain",
        weather=carla.WeatherParameters.HardRainNoon,
        pedestrians=(2, 5), vehicles=(1, 3), cyclists_scooters=(0, 2), trucks=(0, 2),
        allow_jaywalk=True, allow_occlusion=True,
        number_images_to_generate=1100,
    ),
    # snowfall disabled, no real snow assets in CARLA. uncomment to restore.
    # dict(
    #     name="snowfall",
    #     weather=carla.WeatherParameters(
    #         cloudiness=100.0, precipitation=60.0, precipitation_deposits=80.0,
    #         wind_intensity=30.0, sun_azimuth_angle=0.0, sun_altitude_angle=25.0,
    #         fog_density=25.0, fog_distance=60.0, fog_falloff=1.5, wetness=30.0,
    #     ),
    #     pedestrians=(3, 5), vehicles=(2, 4), cyclists_scooters=(0, 0), trucks=(0, 0),
    #     allow_jaywalk=False, allow_occlusion=True,
    #     number_images_to_generate=700,
    # ),
]


# --- acceptance rule (actor counts only, not crossing/occlusion type) ---
# every image needs a pedestrian, plus this many of vehicles/cyclists/trucks/occlusion
REQUIRE_PEDESTRIAN = True
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2
MAX_COUNT_REROLLS = 8

# --- background pedestrians on the sidewalk, not crossing ---
SPAWN_SIDEWALK_PEDESTRIANS = False
SIDEWALK_PEDESTRIAN_COUNT_RANGE = (0, 2)   # rolled fresh per image
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MIN = 8.0
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MAX = 20.0


# --- occlusion search tuning (see occlusion.py) ---
OCCLUSION_SEARCH_SAMPLES = 25          # points along the crossing line to test
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)  # middle stretch only, not the edges
OCCLUSION_TOLERANCE_PERCENT = 12.0     # how close counts as "close enough"
OCCLUSION_OVERLAP_GRID = 64            # resolution of the overlap scoring grid

# hard floor: an "*_occluded" image below this achieved % gets retried, never saved
OCCLUSION_MIN_ACHIEVED_PERCENT = 15.0

# extra check: also retry if achieved misses its own rolled target by more
# than OCCLUSION_TOLERANCE_PERCENT. off by default, costs extra retries.
OCCLUSION_STRICT = False


# --- crossing / walking ---
PEDESTRIAN_WALK_SPEED_RANGE = (0.9, 1.6)
CROSSING_WARMUP_TICKS_RANGE = (35, 80)
MAX_RENDER_DISTANCE = 50.0

# min distance from a jaywalk spot to any crosswalk, so they never share a frame
JAYWALK_CROSSWALK_EXCLUSION_RADIUS = MAX_RENDER_DISTANCE + 15.0


# --- traffic realism ---
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN = 15
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX = 35
TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE = 8.0

TRUCK_SPAWN_DISTANCE_MIN = 15
TRUCK_SPAWN_DISTANCE_MAX = 35
TRUCK_SPAWN_LATERAL_RANGE = 8.0

CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN = 8
CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX = 25
CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE = 6.0

MIN_SEPARATION_BETWEEN_VEHICLES = 3.5
MIN_SEPARATION_TRUCK_TO_CYCLIST = 6.0   # trucks kept further from bikes
MAX_SPAWN_ATTEMPTS_PER_ACTOR = 20


# --- crash / disconnect resilience ---
# keeps retrying a dropped connection instead of giving up. all progress is
# already persisted to disk, so this is safe to retry indefinitely.
RECONNECT_RETRY_DELAY_SECONDS = 10.0
RECONNECT_MAX_ATTEMPTS = 30    # None = retry forever

# settle time between frames, avoids spawn-collision instability
POST_DESTROY_SETTLE_TICKS = 20
POST_DESTROY_SETTLE_SECONDS = 0.3


# --- blueprint categories (CARLA 0.9.16 + vehicle.bh.rider) ---
ALLOWED_CAR_VEHICLES = [
    'vehicle.audi.etron', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer',
    'vehicle.ford.mustang', 'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe',
    'vehicle.nissan.patrol_2021', 'vehicle.tesla.model3', 'vehicle.toyota.prius',
    'vehicle.volkswagen.t2_2021',
]

ALLOWED_TRUCK_VEHICLES = [
    'vehicle.carlamotors.firetruck',
    'vehicle.carlamotors.carlacola',
    'vehicle.carlamotors.european_hgv',
    'vehicle.tesla.cybertruck',
]

ALLOWED_CYCLIST_VEHICLES = [
    'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets',
    'vehicle.bh.crossbike',
]

ALLOWED_ESCOOTER_VEHICLES = [
    'vehicle.bh.rider',
]

# motorcycles/mopeds, labeled separately from pedal cyclists and e-scooters
ALLOWED_TWO_WHEELER_VEHICLES = [
    'vehicle.yamaha.yzf',
    'vehicle.kawasaki.ninja',
    'vehicle.harley-davidson.low_rider',
    'vehicle.vespa.zx125',
]

ALLOWED_CYCLIST_SCOOTER_VEHICLES = (
    ALLOWED_CYCLIST_VEHICLES + ALLOWED_ESCOOTER_VEHICLES + ALLOWED_TWO_WHEELER_VEHICLES
)


# --- output folder naming ---
def resolve_output_dir(split_pct=None, mode_name=None):
    """Builds this run's output folder name from the active split + timestamp."""
    split_pct = split_pct or IMAGE_TYPE_SPLIT
    active = sorted((k, v) for k, v in split_pct.items() if v > 0)

    if mode_name and mode_name != "mixed":
        split_tag = mode_name
    elif len(active) == 1:
        split_tag = active[0][0]
    elif not active:
        split_tag = "empty_split"
    elif active == sorted(IMAGE_TYPE_SPLIT.items()):
        split_tag = "default_split"
    else:
        split_tag = "custom_split"

    if MAP_ROTATION:
        map_tag = "map_rotation"
    elif MAP_NAME:
        map_tag = MAP_NAME.lower()
    else:
        map_tag = "current_map"

    run_tag = split_tag if map_tag == "current_map" else f"{split_tag}__{map_tag}"
    return os.path.join(OUTPUT_BASE_DIR, f"{run_tag}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}")
