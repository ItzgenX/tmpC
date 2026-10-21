"""The settings you normally change. Fine tuning lives in advanced.py."""

# --- run ---
TOTAL_IMAGES = 6000
OUTPUT_DIR = "dataset"              # run again with the same folder to resume
MAPS = ["Town10HD_Opt"]             # urban downtown. other urban maps: "Town03_Opt", "Town05_Opt"
SEED = 2026                         # same seed + same settings = same dataset

# --- CARLA server ---
CARLA_HOST, CARLA_PORT = "localhost", 2000
CARLA_EXECUTABLE = ""               # optional path to CarlaUE4.exe: lets main.py (re)start the server

# --- what is in each image (defaults for every weather) ---
SCENE = {
    "crossing_pedestrians": (1, 3),           # (min, max) people crossing the road
    "sidewalk_pedestrians": (0, 2),           # (min, max) people on the sidewalk, not crossing
    "pedestrian_types": {"child": 1, "woman": 1.5, "man": 1.5},   # relative mix
    "cars": (2, 3),                           # (min, max)
    "truck_chance": 0.3,                      # chance of one truck in the image
    "same_direction_chance": 0.5,             # each vehicle: same way as the ego, else oncoming
    "crosswalk_chance": 0.75,                 # marked crosswalk, else jaywalking
    "occlusion": {"none": 30, "partial": 35, "heavy": 35},   # target pedestrian hidden <10% / 10-40% / 40-80%
}

# --- weathers: share of images, plus any SCENE values to change for that weather ---
# e.g. "clear": {"share": 1, "crossing_pedestrians": (2, 2), "cars": (3, 3), "truck_chance": 1.0}
WEATHERS = {
    "clear":         {"share": 1},
    "partly_cloudy": {"share": 1},
    "overcast":      {"share": 1},
}

# --- vehicles ---
EGO_BLUEPRINT = "my_car_xyz"
CAR_BLUEPRINTS = [
    "vehicle.audi.a2", "vehicle.audi.etron", "vehicle.audi.tt", "vehicle.bmw.grandtourer",
    "vehicle.chevrolet.impala", "vehicle.citroen.c3", "vehicle.dodge.charger_2020",
    "vehicle.ford.mustang", "vehicle.lincoln.mkz_2020", "vehicle.mercedes.coupe_2020",
    "vehicle.mini.cooper_s_2021", "vehicle.nissan.patrol_2021", "vehicle.seat.leon",
    "vehicle.tesla.model3", "vehicle.toyota.prius", "vehicle.volkswagen.t2_2021",
]
TRUCK_BLUEPRINTS = [
    "vehicle.carlamotors.european_hgv", "vehicle.carlamotors.carlacola", "vehicle.carlamotors.firetruck",
]

# --- outputs ---
PED_YOLO_BOX = "full"               # "full" = whole body even where hidden, "visible" = visible part only
LABEL_VEHICLES = True               # False = YOLO files contain pedestrians only
SPLIT_CROSSING_CLASS = False        # True = crossing people get their own YOLO class
SAVE_DEPTH = True
SAVE_DEBUG_OVERLAY = True           # images with boxes drawn, for checking by eye
IMAGE_EXT = ".png"                  # ".jpg" to save disk space
