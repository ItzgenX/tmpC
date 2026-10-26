"""The settings you normally change. Fine tuning lives in advanced.py."""

# --- run ---
OUTPUT_DIR = "dataset"              # run again with the same folder to resume
# any map the server has, by short name ("Town03_Opt", "MyCity") or full path
# ("/Game/Custom/Maps/MyCity"); "python main.py --check" lists them. images are
# split evenly across the maps; custom maps work the same way.
MAPS = ["Town10HD_Opt", "Town03_Opt", "Town05_Opt"]
SEED = 2026                         # same seed + same settings = same dataset

# --- CARLA server ---
CARLA_HOST, CARLA_PORT = "localhost", 2000
CARLA_EXECUTABLE = ""               # optional, e.g. "/opt/carla/CarlaUE4.sh": lets main.py (re)start the server

# --- weathers: generated one after another, in this order ---
# images                : images for this weather
# pedestrians           : people crossing the road
# vehicles / trucks     : cars / trucks (occluding cars count as vehicles)
# cyclists              : cyclists and scooters
# occlusion             : (min, max) % of the target pedestrian hidden at its most hidden moment,
#                         while it walks behind a car (0 = no occluded scenes for this weather).
#                         Behind a car this moment is naturally 50-100 % hidden (a parked car:
#                         70-100 %); lower levels are in the frames before / after it.
# counts: a number is exact; (min, max) picks a random value per image
WEATHERS = {
    "clear":         dict(enabled=True, images=1000, pedestrians=1, vehicles=0, cyclists=0, trucks=0, occlusion=0),
    "partly_cloudy": dict(enabled=True, images=1000, pedestrians=2, vehicles=1, cyclists=1, trucks=1, occlusion=(50, 75)),
    "overcast":      dict(enabled=True, images=1000, pedestrians=3, vehicles=2, cyclists=2, trucks=2, occlusion=(60, 85)),
    "rain":          dict(enabled=True, images=1000, pedestrians=4, vehicles=3, cyclists=3, trucks=0, occlusion=(70, 95)),
    # CARLA 0.9.16 has no real snow - this weather is only a hazy approximation (see advanced.WEATHER_LOOK)
    # "snowfall":      dict(enabled=True, images=1000, pedestrians=5, vehicles=4, cyclists=0, trucks=0, occlusion=(85, 100)),
}

# --- scene types made inside each weather (share = relative part of that weather's images) ---
SCENES = {
    "crosswalk":           dict(enabled=True, share=1),   # crossing at a marked crosswalk
    "jaywalk_road":        dict(enabled=True, share=1),   # crossing mid-block, no crosswalk
    "jaywalk_junction":    dict(enabled=True, share=1),   # crossing where 3 roads meet, no crosswalk
    "occluded_by_traffic": dict(enabled=True, share=1),   # pedestrian walks behind 1-2 cars next to the ego
    "occluded_by_parked":  dict(enabled=True, share=1),   # pedestrian walks behind a car parked at the curb
}

# --- occluded pedestrians: seen from the ego, they walk behind the car and out again ---
SEQUENCE = dict(
    enabled=True,
    frames=5,                       # images per walk: visible -> hidden -> visible again
    walk_speed=(0.6, 1.0),          # m/s, slow so the reveal is gradual
)

# --- variety ---
MAX_USES_PER_SPOT = 8               # one crossing spot is used at most this often per weather; None = no limit

# --- people and traffic ---
PEDESTRIAN_MIX = {"child": 1, "woman": 1.5, "man": 1.5}   # relative mix of crossing people
SIDEWALK_PEDESTRIANS = 0            # extra people on the sidewalk, not crossing (number or (min, max))
SAME_DIRECTION_CHANCE = 0.5         # each vehicle / cyclist: same way as the ego, else oncoming
# True: each weather uses its own row above. False: counts and occlusion are taken from a
# random row instead, so the model cannot learn "rain = 4 people" (better for training)
SCENE_FOLLOWS_WEATHER = True

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
CYCLIST_BLUEPRINTS = [              # bicycles and scooters (rider included)
    "vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets", "vehicle.vespa.zx125",
]

# --- outputs (True = save) ---
SAVE = dict(
    image=True,                     # images/          RGB
    depth=True,                     # depth/           16-bit PNG, 1 unit = 1 cm
    seg_color=True,                 # seg_color/       CityScapes colours
    bbox_2d_image=True,             # images_2d_bbox/  RGB with 2D boxes drawn
    bbox_3d_image=True,             # images_3d_bbox/  RGB with 3D boxes drawn
    labels_yolo=True,               # labels/          YOLO 2D labels
    depth_color=False,              # depth_color/     depth preview
    seg_raw=False,                  # seg_raw/         class id per pixel
)
# labels_3d/<image>.json (2D + 3D boxes, occlusion, camera) is always written
PED_YOLO_BOX = "full"               # "full" = whole body even where hidden, "visible" = visible part only
LABEL_VEHICLES = True               # False = YOLO files contain pedestrians only
LABEL_FULLY_HIDDEN_PEDESTRIANS = False   # YOLO box for a pedestrian that cannot be seen at all
SPLIT_CROSSING_CLASS = False        # True = crossing people get their own YOLO class
IMAGE_EXT = ".png"                  # RGB image format: ".png" (lossless) or ".jpg" (smaller)
SAVE_IN_FOLDERS = True              # True = images/clear/crosswalk/clear_crosswalk_00012_00.png
                                    # False = all in one folder: images/clear_crosswalk_00012_00.png

# --- problem images ---
# False: images that fail a check are saved anyway and listed in flagged.csv (review them,
#        then python main.py --delete-flagged). True: they are skipped, never saved.
SKIP_FLAGGED_IMAGES = False
TRAIN_VAL_TEST = (0.8, 0.1, 0.1)    # split lists, grouped by location so no place is in two splits; None = off
