import carla


# --- NEW YOLO CONFIGURATION ---
NUM_IMAGES = 20
START_FRAME_NUMBER = 45
OUTPUT_DIR = "dataset_w_segmetaion"


CLASS_MAP = {
    'Pedestrian': 0,
    'Vehicle': 4
}


# --- Bounding Box Drawing Configuration ---
DRAW_2D_BBOX = True
DRAW_3D_BBOX = True


# Spawning parameters
NUM_PEDESTRIANS = 1
PEDESTRIAN_SPAWN_DISTANCE_MIN = 10
PEDESTRIAN_SPAWN_DISTANCE_MAX = 25
MAX_RENDER_DISTANCE = 50.0


NUM_TRAFFIC_VEHICLES = 1
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MIN = 15
TRAFFIC_VEHICLE_SPAWN_DISTANCE_MAX = 35
TRAFFIC_VEHICLE_SPAWN_LATERAL_RANGE = 8.0


# Define allowed traffic vehicles
ALLOWED_TRAFFIC_VEHICLES = [
    'vehicle.audi.etron', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer',
    'vehicle.ford.mustang', 'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe',
    'vehicle.nissan.patrol_2021', 'vehicle.tesla.model3', 'vehicle.toyota.prius',
    'vehicle.volkswagen.t2_2021', 'vehicle.carlamotors.firetruck',
    'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf'
]


# Weather parameters
CURRENT_WEATHER = "Partly_cloudy"
WEATHER_CONDITIONS = {
    "Clear": carla.WeatherParameters.ClearNoon,
    "Partly_cloudy": carla.WeatherParameters.CloudyNoon,
    "Overcast": carla.WeatherParameters.WetCloudyNoon,
    "Rain": carla.WeatherParameters.HardRainNoon
}
