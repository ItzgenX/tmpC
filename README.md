# CARLA Pedestrian-Crossing Dataset Generator

Generates labeled dashcam images of pedestrians crossing the road in CARLA 0.9.16,
built for training **pedestrian / occluded-pedestrian / crossing** models.

Every image comes with:
- YOLO 2D boxes (pedestrian full-body or visible-part, plus vehicles and trucks)
- a JSON label per image with 2D + 3D boxes, true occlusion %, and crossing attributes
- semantic segmentation: raw CARLA class ids and a CityScapes-coloured copy
- an optional debug image with all boxes drawn on it

## Quick start

```bash
# 1. start CARLA 0.9.16 and wait until it has loaded
# 2. in an env with: carla (matching the server), numpy, opencv-python
cd generate_dataset
python list_blueprints.py     # once: checks ego car, pedestrian types, car ids, maps
python main.py --plan-only 50 # optional: plans 50 scenes per map without rendering
python main.py                # generates the dataset (run it again any time to resume)
```

`python main.py` is the only command you need. It runs the generator in a child
process and restarts it automatically if it crashes. This includes native CARLA
crashes and core dumps that no Python code can catch. Every saved image is final,
so a crash only loses the image being rendered at that moment. If you set
`CARLA_EXECUTABLE` in `config.py`, it also starts and restarts the CARLA server itself.

## What each image contains

| | |
|---|---|
| Map | Town10HD (CARLA's downtown city map) only, no map rotation |
| Weather | clear, partly cloudy or overcast (exact split, each weather's `share`). Daytime, sun 70–90° high, so it is never in frame and never blinds the camera. No rain or wet roads. |
| Crossing | a real marked crosswalk (75%) or a jaywalk spot on a straight road with sidewalks on both sides (fallback) |
| Pedestrians crossing | 1–3, a mix of children, women and men, each with its own speed, walking direction and position on the crossing. Some hurry across. |
| Pedestrians on the sidewalk | 0–2, walking along or standing, **not** crossing. These are negatives so a model learns crossing vs not crossing. |
| Traffic | 2–3 cars, plus a truck in about 30% of images. Each vehicle independently drives the same way as the ego or oncoming. Same-direction cars stay within a 2.4 m ego-lane band. |
| Occlusion | one *target* crossing pedestrian per image with an exact occlusion quota: `none` (<10%), `partial` (10–40%), `heavy` (40–80%). Default split 30/35/35. |

## How an image is made

1. **Plan.** The whole scene is decided as plain data before anything is spawned:
   ego pose, every pedestrian (type, blueprint, speed, direction, where on the crossing),
   every vehicle (lane, same/oncoming, distance, colour). For occluded images, a car or
   truck is placed between the camera and the target pedestrian. Its position is chosen
   by projecting both into the camera, so it covers the requested amount.
2. **Check the plan.** It is rejected and re-planned if vehicles overlap each other,
   the ego, a parked car or the crossing; if the crossing is not in view; if a
   pedestrian would walk into something; or if the viewpoint (crossing + lane +
   distance + weather + occlusion level) was already used. **No two images share a
   viewpoint**, and crossings are used evenly.
3. **Render.** The ego teleports, the planned actors spawn, pedestrians walk for 1–2 s
   so they are mid-stride, then RGB (rendered at 2x and downscaled for anti-aliasing)
   and segmentation are captured from **exactly the same tick**.
4. **Measure.** Our traffic is moved out of sight and a second segmentation frame is
   taken. Comparing the two gives each pedestrian's unoccluded silhouette, so you get a
   true full-body box, a visible box and a real occlusion %.
5. **Quality gates.** The image is re-planned, never saved, if the target pedestrian is
   not clearly visible, its measured occlusion misses the requested level, or the
   camera is blocked.
6. **Write** all files. The JSON is written last, so it marks the image as finished.

## Output

```
dataset/
  images/          Town10HD_Opt_00042.png         1920x1080 RGB
  labels/          Town10HD_Opt_00042.txt         YOLO, pedestrian box = PED_YOLO_BOX ("full" by default)
  labels_visible/  Town10HD_Opt_00042.txt         YOLO, pedestrian box = visible part only
  labels_3d/       Town10HD_Opt_00042.json        everything below + the full scene plan
  seg_raw/         Town10HD_Opt_00042.png         CARLA semantic class id per pixel
  seg_color/       Town10HD_Opt_00042.png         CityScapes palette
  debug/           Town10HD_Opt_00042.jpg         boxes drawn (SAVE_DEBUG_OVERLAY)
  data.yaml, classes.txt, manifest.csv, summary.txt, run_log.txt, skipped.json
```

YOLO classes: `0 pedestrian, 1 vehicle, 2 truck`. With `SPLIT_CROSSING_CLASS = True`
the classes are `0 crossing_pedestrian, 1 pedestrian, 2 vehicle, 3 truck`. With
`LABEL_VEHICLES = False` the YOLO files contain pedestrians only.

Each JSON label has `bbox_visible`, `bbox_full` (pedestrians), `bbox_projected`,
`occlusion`, `visible_pixels`, `truncated`, 3D `location_world` / `rotation_world` /
`dimensions`, `location_camera` (x right, y down, z forward), the 8 projected 3D
corners, plus `camera.K` and `camera.world_to_camera`. Pedestrians also carry `type`
(child/woman/man), `role` (crossing/sidewalk), `crossing`, `on_road`, `crossing_type`,
`crossing_progress`, `walking_direction`, `speed` and `target`.

`occlusion` counts occlusion by traffic. Something static, like a pole or a parked car,
in front of a pedestrian reduces `visible_pixels` but is not part of the occlusion %.

## Configuration

Two files:
- [generate_dataset/config.py](generate_dataset/config.py) holds the everyday settings
  (about 50 lines): dataset size, map, server, what is in each image, per-weather
  changes, vehicle blueprints and which outputs to save.
- [generate_dataset/advanced.py](generate_dataset/advanced.py) holds fine tuning: camera,
  weather look, pedestrian speeds, quality gates, placement distances, timing. You
  normally never edit it.

What is in each image is one table, `SCENE`:

```python
SCENE = {
    "crossing_pedestrians": (1, 3),           # (min, max) people crossing the road
    "sidewalk_pedestrians": (0, 2),           # (min, max) people on the sidewalk, not crossing
    "pedestrian_types": {"child": 1, "woman": 1.5, "man": 1.5},   # relative mix
    "cars": (2, 3),
    "truck_chance": 0.3,
    "same_direction_chance": 0.5,
    "crosswalk_chance": 0.75,
    "occlusion": {"none": 30, "partial": 35, "heavy": 35},
}
```

`python main.py --total 300 --out test_run` is a good first test run.

### Different settings per weather

Each entry in `WEATHERS` has its `share` of the images, plus any `SCENE` values to
change for that weather. Anything not listed keeps the `SCENE` value. For example,
"clear: exactly 2 people crossing, nobody on the sidewalk, 3 cars and 1 truck;
overcast: 3 people, no truck":

```python
WEATHERS = {
    "clear":         {"share": 1, "crossing_pedestrians": (2, 2), "sidewalk_pedestrians": (0, 0),
                      "cars": (3, 3), "truck_chance": 1.0},
    "partly_cloudy": {"share": 1},
    "overcast":      {"share": 1, "crossing_pedestrians": (3, 3), "truck_chance": 0.0},
}
```

- Counts are exact. If a requested car or truck cannot be placed, the scene is
  re-planned rather than saved with fewer.
- In an occluded image, the occluder is one of that image's cars, or its truck.
  "3 cars + 1 truck" stays 3 cars + 1 truck.
- A typo or an impossible value (e.g. 5 crossing pedestrians) stops the run before
  any image is generated.

### One map and variety

With a single map, every image comes from Town10HD's own crossings. No two images
share a viewpoint (crossing + lane + 2 m distance step + weather + occlusion level),
so run `python main.py --plan-only 200` first. If it reports many
`viewpoint already used` rejections, set `DISTANCE_BIN = 1.0` in `advanced.py` to double the unique
viewpoints, or add `Town03_Opt` / `Town05_Opt` (both urban).

## Reliability

- Ego and cameras are spawned once per map, not once per frame, and the world is fully
  reloaded every `RELOAD_WORLD_EVERY` images.
- Parked-vehicle data is read once per map. Reading it every frame leaks server memory
  and crashes CARLA (Signal 11).
- At startup, every vehicle, walker and sensor left over from a crashed run is removed.
  Otherwise orphaned 4K cameras keep rendering and pile up on the server.
- The worker restarts itself every `MAX_SCENES_PER_PROCESS` images to cap slow leaks.
- An image slot that fails `MAX_ATTEMPTS_PER_SCENE` renders is recorded in `skipped.json`
  with the reasons. Delete that file to retry those slots.

## Size and time

At 1920x1080 PNG, expect about 4–5 MB per image including segmentation and debug files
(about 30 GB for 6000). Use `IMAGE_EXT = ".jpg"` or `SAVE_DEBUG_OVERLAY = False` to cut
that. Expect roughly 3–6 s per image on a good GPU. Run CARLA with `-RenderOffScreen`.

## Files

| File | Role |
|---|---|
| `config.py` | everyday settings |
| `advanced.py` | fine tuning + config helpers |
| `main.py` | supervisor + generator loop, capture, quality gates, writing |
| `planner.py` | plans and validates a whole scene before spawning |
| `crosswalk.py` | finds crossing sites: marked crosswalks, jaywalk fallback |
| `sidewalk.py` | road cross-section: lanes on both sides, sidewalks, waypoint stepping |
| `spawner.py` | spawns what the plan says, ego/cameras, cleanup |
| `utils.py` | projection, segmentation-based boxes and occlusion, file writing, drawing |
| `list_blueprints.py` | checks a live server against the config |
