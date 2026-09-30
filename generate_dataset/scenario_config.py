import carla


# --- OUTPUT / VOLUME CONFIGURATION ---
OUTPUT_DIR = "dataset_scenarios"
START_FRAME_NUMBER = 0
TOTAL_IMAGES = 5000


# --- Bounding Box Drawing Configuration ---
DRAW_2D_BBOX = True
DRAW_3D_BBOX = True


# --- CLASS SWITCH: set to True/False and flip freely ---
# When True, truck-type blueprints get their own YOLO class ("Truck").
# When False, trucks are still spawned per-scenario but labeled as "Vehicle".
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


# --- SCENARIO TABLE (exact spec you gave) ---
# occlusion_percent: target fraction of one pedestrian's bounding box that
# should be covered by another actor (vehicle/truck/cyclist) in-frame.
SCENARIOS = [
    dict(name="clear_baseline",       weather="Clear",         pedestrians=1, vehicles=0, cyclists_scooters=0, trucks=0, occlusion_percent=0),
    dict(name="partly_cloudy_light",  weather="Partly_cloudy", pedestrians=2, vehicles=1, cyclists_scooters=1, trucks=1, occlusion_percent=0),
    dict(name="overcast_25pct",       weather="Overcast",      pedestrians=3, vehicles=2, cyclists_scooters=2, trucks=2, occlusion_percent=25),
    dict(name="overcast_50pct",       weather="Overcast",      pedestrians=4, vehicles=2, cyclists_scooters=2, trucks=1, occlusion_percent=50),
    dict(name="overcast_75pct",       weather="Overcast",      pedestrians=3, vehicles=2, cyclists_scooters=2, trucks=2, occlusion_percent=75),
    dict(name="snowfall_100pct",      weather="Snowfall",      pedestrians=5, vehicles=4, cyclists_scooters=0, trucks=0, occlusion_percent=100),
]


def build_images_per_scenario(total_images=TOTAL_IMAGES, scenarios=SCENARIOS):
    """Spreads total_images as evenly as possible across all scenario rows."""
    n = len(scenarios)
    base = total_images // n
    remainder = total_images % n
    counts = [base + (1 if i < remainder else 0) for i in range(n)]
    return counts


IMAGES_PER_SCENARIO = build_images_per_scenario()


# --- Occlusion search tuning ---
OCCLUSION_SEARCH_SAMPLES = 25
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)


# --- Spawning / crossing parameters ---
PEDESTRIAN_WALK_SPEED_RANGE = (0.9, 1.6)
CROSSING_WARMUP_TICKS_RANGE = (35, 80)   # lets pedestrians actually walk before capture
MAX_RENDER_DISTANCE = 50.0


TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN = 15
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX = 35
TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE = 8.0

TRUCK_SPAWN_DISTANCE_MIN = 15
TRUCK_SPAWN_DISTANCE_MAX = 35
TRUCK_SPAWN_LATERAL_RANGE = 8.0

CYCLIST_SCOOTER_SPAWN_DISTANCE_MIN = 8
CYCLIST_SCOOTER_SPAWN_DISTANCE_MAX = 25
CYCLIST_SCOOTER_SPAWN_LATERAL_RANGE = 6.0


# --- Blueprint category lists ---
# NOTE: these ids follow the CARLA 0.9.13+ standard content plus the
# 'vehicle.bh.rider' e-scooter asset already used elsewhere in this project.
# If your install/content pack differs, just edit these lists; unknown ids
# are silently skipped by the spawn filter.
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


# --- Weather parameters (Snowfall is an approximation: CARLA's base town
# content has no snow ground/particle assets, so this simulates a cold,
# overcast, wet-and-foggy look via WeatherParameters rather than true snow) ---
WEATHER_CONDITIONS = {
    "Clear": carla.WeatherParameters.ClearNoon,
    "Partly_cloudy": carla.WeatherParameters.CloudyNoon,
    "Overcast": carla.WeatherParameters.WetCloudyNoon,
    "Rain": carla.WeatherParameters.HardRainNoon,
    "Snowfall": carla.WeatherParameters(
        cloudiness=100.0,
        precipitation=60.0,
        precipitation_deposits=80.0,
        wind_intensity=30.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=25.0,
        fog_density=25.0,
        fog_distance=60.0,
        fog_falloff=1.5,
        wetness=30.0,
    ),
}

WEATHERS_WITH_LIGHTS_ON = {"Partly_cloudy", "Overcast", "Rain", "Snowfall"}
