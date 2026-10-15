import os
import datetime

import carla


# leave as None to use whatever map the server already has loaded,
# or set a map name and the script will call load_world() for you.
# ignored if MAP_ROTATION below is non-empty.
MAP_NAME = None

# set this to a list of map names to cycle through instead of running the
# whole dataset on one map - e.g. ["Town01", "Town02", "Town03", "Town10HD_Opt"].
# each scenario's number_images_to_generate gets split evenly across these
# maps, so "overcast" at 1700 images with 4 maps in rotation means ~425
# images per map. this is what actually fixes images looking repetitive -
# one map only has so many crosswalks and buildings to work with, cycling
# maps is the real source of variety. leave as [] to stay on one map.
MAP_ROTATION = []

# everything gets written under this folder
OUTPUT_BASE_DIR = "dataset_scenarios"
DRAW_2D_BBOX = True
DRAW_3D_BBOX = True

# truck-type blueprints get their own YOLO class when True, otherwise
# they still spawn per-scenario but get labeled "Vehicle" like a car
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
# every scenario's number_images_to_generate gets split across these four
# buckets by EXACT count, not probability - ask for 100 images with the
# defaults below and you get exactly 25 of each, guaranteed, not "roughly
# a quarter on average". percentages don't need to sum to 100 (they're
# normalized), but keep them in the ratio you actually want.
IMAGE_TYPE_SPLIT = {
    "crosswalk_clear": 25,
    "crosswalk_occluded": 25,
    "jaywalk_clear": 25,
    "jaywalk_occluded": 25,
}

# when an image's bucket is "*_occluded", the target percentage is rolled
# uniformly across this whole range - this is the one knob for "a person
# can be occluded anywhere from barely to almost completely". the lower
# bound is nudged up to 1 at roll time so an "occluded" bucket image can
# never silently land on 0% and look identical to a "clear" one.
OCCLUSION_PERCENT_RANGE = (1, 100)


# Each row is one weather scenario. pedestrians/vehicles/cyclists_scooters/
# trucks are (min, max) ranges rolled per image - see
# MIN_OPTIONAL_CATEGORIES_SATISFIED below for how that ties into which
# rolled combinations actually get saved.
# allow_jaywalk / allow_occlusion: whether this scenario's images are even
# eligible for those buckets at all. If a scenario disallows one, that
# bucket's quota (from IMAGE_TYPE_SPLIT above) folds into the nearest
# allowed bucket instead of being dropped - see image_plan.py.
# weather takes any carla.WeatherParameters object directly, so adding a new
# scenario is just one more dict here, nothing else to touch.
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
    # snowfall temporarily disabled - CARLA has no real snow assets and this
    # was just a fog/wetness approximation anyway. uncomment to bring it back.
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

# CARLA has no snow ground/particle assets in the base town content, so
# "snowfall" above is a cold/overcast/wet-fog approximation rather than
# actual snow. Swap in a snow-enabled map + weather if you have one.


# --- variety / acceptance rule (actor counts only - crossing type and
# occlusion-or-not are now decided exactly by the plan above, not rolled) ---
# every saved image must have at least one pedestrian, and at least this
# many of {vehicles, cyclists_scooters, trucks, this image's occlusion
# bucket} present, or it gets re-rolled. keeps images varied instead of
# every single one maxed out, while still avoiding a bunch of empty
# "one person on a sidewalk" shots.
REQUIRE_PEDESTRIAN = True
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2
MAX_COUNT_REROLLS = 8

# --- ambient sidewalk pedestrians (separate from the crossing pedestrian(s)
# above - these just walk along the footpath, never into the road) ---
# off by default; set True to add background pedestrians on the sidewalk
# for busier-looking street scenes. Doesn't affect the crossing/occlusion
# pedestrian(s) at all - this is purely extra background people.
SPAWN_SIDEWALK_PEDESTRIANS = False
SIDEWALK_PEDESTRIAN_COUNT_RANGE = (0, 2)   # rolled fresh per image
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MIN = 8.0
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MAX = 20.0


# --- occlusion search tuning (see occlusion.py for the actual logic) ---
# how many points along the crossing line to test for a placement
OCCLUSION_SEARCH_SAMPLES = 25
# only search this middle stretch of the crossing, not right at the edges
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)
# how far achieved can be from target before we flag it as "not close enough"
# in the manifest - raise this if your map/occluders can't hit tight targets
OCCLUSION_TOLERANCE_PERCENT = 12.0
# resolution of the overlap grid used to score each candidate placement -
# higher is more precise but slower, this is plenty for a bbox-level estimate
OCCLUSION_OVERLAP_GRID = 64

# hard floor on an "*_occluded" image: if the placement search can't
# actually get at least this much of the pedestrian covered, the attempt
# is thrown away and retried - never saved. this is the real guarantee
# that "occluded" means occluded: it's checked unconditionally, it doesn't
# depend on OCCLUSION_STRICT below, and it can't be satisfied by a low
# rolled target plus a close-enough miss (e.g. target 5%, achieved 3%
# would pass OCCLUSION_STRICT's tolerance check but isn't visibly occluded -
# this catches that case too).
OCCLUSION_MIN_ACHIEVED_PERCENT = 15.0

# when True, an "*_occluded" image that cleared the floor above but still
# landed outside OCCLUSION_TOLERANCE_PERCENT of its specific rolled target
# gets discarded and retried too - use this if you need occlusion amounts
# to track their rolled target closely, not just "occluded at all".
# off by default since it costs extra retries on maps with few occluders.
OCCLUSION_STRICT = False


# --- crossing / walking ---
PEDESTRIAN_WALK_SPEED_RANGE = (0.9, 1.6)
CROSSING_WARMUP_TICKS_RANGE = (35, 80)
MAX_RENDER_DISTANCE = 50.0

# a crosswalk must be at least this far from a jaywalk spot to count as
# "not visible" - a bit more than render distance so it can't peek into
# frame at the edges
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

# how far apart same-frame vehicles need to land before we accept a spawn
MIN_SEPARATION_BETWEEN_VEHICLES = 3.5
# trucks are big and fast, keep them further from cyclists/scooters than
# from other cars
MIN_SEPARATION_TRUCK_TO_CYCLIST = 6.0
MAX_SPAWN_ATTEMPTS_PER_ACTOR = 20


# --- crash / disconnect resilience ---
# if the client loses its connection to the server (server crashed, core
# dumped, got killed) mid-run, the script doesn't just give up - it waits
# and keeps trying to reconnect, since all per-image progress is already
# persisted (frame numbering + the remaining plan are both read back from
# disk). set RECONNECT_MAX_ATTEMPTS to None for "retry forever" (useful if
# you're not standing at the keyboard to restart the server by hand).
RECONNECT_RETRY_DELAY_SECONDS = 10.0
RECONNECT_MAX_ATTEMPTS = 30

# extra settle time after a batch of actors is destroyed, before the next
# frame starts spawning - a tight spawn/destroy loop with no breathing
# room is a known way to upset the UE4 side of CARLA over a long run
POST_DESTROY_SETTLE_TICKS = 20
POST_DESTROY_SETTLE_SECONDS = 0.3


# --- blueprint categories ---
# these ids match the CARLA 0.9.16 catalogue plus 'vehicle.bh.rider', which
# was already used for e-scooters elsewhere in this project. edit freely if
# your content pack differs - unknown ids are just skipped at spawn time.
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

# motorized two-wheelers (motorcycles/mopeds) - separate from pedal cyclists
# and e-scooters since they're visually and behaviorally different, but
# still spawned from the same "cyclists_scooters" scenario count
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
# every fresh run gets its own folder under OUTPUT_BASE_DIR, named after
# the image-type split that was actually active plus a timestamp, so two
# different-config runs never land in the same folder by accident. to
# continue a specific previous run after a crash/restart, pass its exact
# folder via --resume instead of letting a new one get created - see
# generate_scenarios.py's CLI help.
#
# this is a function, not a precomputed constant, because generate_scenarios.py
# can override IMAGE_TYPE_SPLIT via CLI flags after this module is imported -
# call resolve_output_dir() *after* applying those overrides so the folder
# name stays honest about what actually ran.
def resolve_output_dir(split_pct=None, mode_name=None):
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

    run_tag = f"{split_tag}__{map_tag}"
    return os.path.join(OUTPUT_BASE_DIR, f"{run_tag}__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
