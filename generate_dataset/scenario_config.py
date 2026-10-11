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


# Each row is one scenario. Counts (including occlusion_percent) are
# (min, max) ranges rolled per image, not fixed numbers - see
# MIN_OPTIONAL_CATEGORIES_SATISFIED below for how that ties into which
# rolled combinations actually get saved. Use occlusion_percent=(0, 0) for
# "never occluded" or a wide range like (0, 75) to cover several occlusion
# tiers from one scenario row instead of duplicating rows per percentage.
# occlusion_percent is a target, not a guarantee - the placement search gets
# close to it but real geometry doesn't always allow an exact match.
# allow_jaywalk: whether this scenario is even eligible for a jaywalking
# frame - the actual odds are one shared knob, JAYWALK_PROBABILITY below,
# so you're not re-tuning a probability on every row.
# weather takes any carla.WeatherParameters object directly, so adding a new
# scenario is just one more dict here, nothing else to touch.
SCENARIOS = [
    dict(
        name="clear",
        weather=carla.WeatherParameters.ClearNoon,
        pedestrians=(1, 2), vehicles=(0, 1), cyclists_scooters=(0, 1), trucks=(0, 1),
        occlusion_percent=(0, 0), allow_jaywalk=False,
        number_images_to_generate=700,
    ),
    dict(
        name="partly_cloudy",
        weather=carla.WeatherParameters.CloudyNoon,
        pedestrians=(1, 3), vehicles=(0, 2), cyclists_scooters=(0, 2), trucks=(0, 1),
        occlusion_percent=(0, 0), allow_jaywalk=True,
        number_images_to_generate=800,
    ),
    dict(
        name="overcast",
        weather=carla.WeatherParameters.WetCloudyNoon,
        pedestrians=(2, 5), vehicles=(1, 3), cyclists_scooters=(1, 3), trucks=(0, 2),
        # single knob for the whole 0-75% occlusion range instead of
        # separate 25/50/75 rows - each image rolls its own value in here
        occlusion_percent=(0, 75), allow_jaywalk=True,
        number_images_to_generate=1700,
    ),
    dict(
        name="rain",
        weather=carla.WeatherParameters.HardRainNoon,
        pedestrians=(2, 5), vehicles=(1, 3), cyclists_scooters=(0, 2), trucks=(0, 2),
        occlusion_percent=(0, 75), allow_jaywalk=True,
        number_images_to_generate=1100,
    ),
    dict(
        name="snowfall",
        weather=carla.WeatherParameters(
            cloudiness=100.0, precipitation=60.0, precipitation_deposits=80.0,
            wind_intensity=30.0, sun_azimuth_angle=0.0, sun_altitude_angle=25.0,
            fog_density=25.0, fog_distance=60.0, fog_falloff=1.5, wetness=30.0,
        ),
        pedestrians=(3, 5), vehicles=(2, 4), cyclists_scooters=(0, 0), trucks=(0, 0),
        occlusion_percent=(0, 100), allow_jaywalk=False,
        number_images_to_generate=700,
    ),
]

# CARLA has no snow ground/particle assets in the base town content, so
# "snowfall" above is a cold/overcast/wet-fog approximation rather than
# actual snow. Swap in a snow-enabled map + weather if you have one.


# --- variety / acceptance rule ---
# every saved image must have at least one pedestrian, and at least this
# many of {vehicles, cyclists_scooters, trucks, occlusion} present, or it
# gets re-rolled. keeps images varied instead of every single one maxed out,
# while still avoiding a bunch of empty "one person on a sidewalk" shots.
REQUIRE_PEDESTRIAN = True
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2
MAX_COUNT_REROLLS = 8


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

# fraction of eligible images that actually get an occlusion target rolled
# at all - without this, a scenario with occlusion_percent=(0, 75) would
# roll *some* nonzero value on almost every single image (only 1/76 chance
# of landing exactly on 0), which doesn't give you a real occluded/clear
# mix. this is the knob for "how many of my images should be occlusion
# images" - 0.6 means roughly 60% of eligible images get occluded, the
# rest are left clear (occlusion_percent forced to 0 for that image).
# set to 1.0 for a dedicated "occlusion only" run (combine with
# OCCLUSION_STRICT below if you want that run to only keep frames that
# actually hit the target, not just ones that tried to).
OCCLUSION_PROBABILITY = 0.6

# when True, a frame that targeted occlusion but landed outside
# OCCLUSION_TOLERANCE_PERCENT of that target gets discarded and retried
# instead of saved anyway - the search can't always find a usable occluder
# in frame, and without this a "100% occlusion" image could quietly save
# with 0% actually achieved. off by default since it costs extra retries;
# turn on for a run where every occluded image needs to actually be occluded.
OCCLUSION_STRICT = False


# --- crossing / walking ---
PEDESTRIAN_WALK_SPEED_RANGE = (0.9, 1.6)
CROSSING_WARMUP_TICKS_RANGE = (35, 80)
MAX_RENDER_DISTANCE = 50.0

# odds of a jaywalk frame in any scenario with allow_jaywalk=True - one
# shared knob instead of a separate probability on every scenario row
JAYWALK_PROBABILITY = 0.12

# override for a dedicated run instead of the normal random mix:
#   None             -> normal behavior, JAYWALK_PROBABILITY decides per image
#   "crosswalk_only" -> every image is a legal crosswalk crossing
#   "jaywalk_only"   -> every image is a mid-road jaywalk (ignores each
#                        scenario's allow_jaywalk flag - if you're forcing
#                        this mode you clearly want it everywhere)
# run the script once with this set, then flip it back (or comment it out)
# for your normal mixed run.
FORCE_CROSSING_MODE = None

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
# every run gets its own folder under OUTPUT_BASE_DIR, named after the
# choices that were actually active (crossing mode, occlusion mode, map
# setup) plus a timestamp - so a jaywalk_only batch, an occlusion-forced
# batch, and a normal mixed run never land in the same folder and get
# mixed together or have their manifests/frame numbering collide.
def _describe_run():
    if FORCE_CROSSING_MODE == "crosswalk_only":
        crossing_tag = "crosswalk_only"
    elif FORCE_CROSSING_MODE == "jaywalk_only":
        crossing_tag = "jaywalk_only"
    else:
        crossing_tag = "mixed_crossing"

    if OCCLUSION_PROBABILITY >= 0.95:
        occlusion_tag = "occlusion_forced"
    elif OCCLUSION_PROBABILITY <= 0.0:
        occlusion_tag = "no_occlusion"
    else:
        occlusion_tag = f"occlusion_{round(OCCLUSION_PROBABILITY * 100)}pct"

    if MAP_ROTATION:
        map_tag = "map_rotation"
    elif MAP_NAME:
        map_tag = MAP_NAME.lower()
    else:
        map_tag = "current_map"

    return f"{crossing_tag}__{occlusion_tag}__{map_tag}"


OUTPUT_DIR = os.path.join(
    OUTPUT_BASE_DIR,
    f"{_describe_run()}__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
