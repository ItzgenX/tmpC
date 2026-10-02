import carla


# leave as None to use whatever map the server already has loaded,
# or set a map name and the script will call load_world() for you
MAP_NAME = None

OUTPUT_DIR = "dataset_scenarios"
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
    }
    if separate_truck_class:
        class_map['Truck'] = 4
    return class_map


CLASS_MAP = build_class_map()


# Each row is one scenario. Counts are (min, max) ranges rolled per image,
# not fixed numbers - see MIN_OPTIONAL_CATEGORIES_SATISFIED below for how
# that ties into which rolled combinations actually get saved.
# occlusion_percent is a target, not a guarantee - the placement search gets
# close to it but real geometry doesn't always allow an exact match.
# jaywalk_probability: chance a frame in this scenario is generated with a
# pedestrian illegally crossing mid-road instead of on a crosswalk. Kept
# separate from occlusion/other counts on purpose, see notes further down.
# weather takes any carla.WeatherParameters object directly, so adding a new
# scenario is just one more dict here, nothing else to touch.
SCENARIOS = [
    dict(
        name="clear_baseline",
        weather=carla.WeatherParameters.ClearNoon,
        pedestrians=(1, 2), vehicles=(0, 1), cyclists_scooters=(0, 1), trucks=(0, 1),
        occlusion_percent=0, jaywalk_probability=0.0,
        number_images_to_generate=800,
    ),
    dict(
        name="partly_cloudy_light",
        weather=carla.WeatherParameters.CloudyNoon,
        pedestrians=(1, 3), vehicles=(0, 2), cyclists_scooters=(0, 2), trucks=(0, 1),
        occlusion_percent=0, jaywalk_probability=0.1,
        number_images_to_generate=900,
    ),
    dict(
        name="overcast_25pct",
        weather=carla.WeatherParameters.WetCloudyNoon,
        pedestrians=(2, 4), vehicles=(1, 3), cyclists_scooters=(1, 3), trucks=(0, 2),
        occlusion_percent=25, jaywalk_probability=0.1,
        number_images_to_generate=900,
    ),
    dict(
        name="overcast_50pct",
        weather=carla.WeatherParameters.WetCloudyNoon,
        pedestrians=(2, 5), vehicles=(1, 3), cyclists_scooters=(1, 3), trucks=(0, 2),
        occlusion_percent=50, jaywalk_probability=0.15,
        number_images_to_generate=900,
    ),
    dict(
        name="overcast_75pct",
        weather=carla.WeatherParameters.WetCloudyNoon,
        pedestrians=(2, 4), vehicles=(1, 3), cyclists_scooters=(1, 3), trucks=(0, 2),
        occlusion_percent=75, jaywalk_probability=0.15,
        number_images_to_generate=800,
    ),
    dict(
        name="snowfall_100pct",
        weather=carla.WeatherParameters(
            cloudiness=100.0, precipitation=60.0, precipitation_deposits=80.0,
            wind_intensity=30.0, sun_azimuth_angle=0.0, sun_altitude_angle=25.0,
            fog_density=25.0, fog_distance=60.0, fog_falloff=1.5, wetness=30.0,
        ),
        pedestrians=(3, 5), vehicles=(2, 4), cyclists_scooters=(0, 0), trucks=(0, 0),
        occlusion_percent=100, jaywalk_probability=0.0,
        number_images_to_generate=700,
    ),
]

# CARLA has no snow ground/particle assets in the base town content, so
# "snowfall_100pct" above is a cold/overcast/wet-fog approximation rather
# than actual snow. Swap in a snow-enabled map + weather if you have one.


# --- variety / acceptance rule ---
# every saved image must have at least one pedestrian, and at least this
# many of {vehicles, cyclists_scooters, trucks, occlusion} present, or it
# gets re-rolled. keeps images varied instead of every single one maxed out,
# while still avoiding a bunch of empty "one person on a sidewalk" shots.
REQUIRE_PEDESTRIAN = True
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2
MAX_COUNT_REROLLS = 8


# --- occlusion search tuning ---
OCCLUSION_SEARCH_SAMPLES = 25
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)
OCCLUSION_TOLERANCE_PERCENT = 12.0


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

ALLOWED_CYCLIST_SCOOTER_VEHICLES = ALLOWED_CYCLIST_VEHICLES + ALLOWED_ESCOOTER_VEHICLES
