# CARLA Pedestrian-Crossing Dataset Generator

Generates labeled dashcam images of pedestrians crossing urban roads in CARLA 0.9.16,
including pedestrians hidden behind traffic, for training pedestrian / occluded-pedestrian /
crossing models.

For every image you get (each output can be switched off):

| Folder | Content |
|---|---|
| `images/` | RGB, 1920x1080 |
| `depth/` | depth, 16-bit PNG, 1 unit = 1 cm |
| `seg_color/` | semantic segmentation, CityScapes colours |
| `images_2d_bbox/` | RGB with the 2D boxes drawn: one colour, no text |
| `images_3d_bbox/` | RGB with the 3D boxes drawn: one colour, no text |
| `labels/` | YOLO 2D labels |
| `labels_3d/` | JSON: every 2D + 3D box, occlusion %, camera, the scene plan (always written) |

## Quick start

1. Install the Python packages, with `carla` matching your server version:
   ```bash
   pip install numpy opencv-python
   pip install carla==0.9.16
   ```
   If pip has no `carla` build for your Python version, install the wheel shipped with
   CARLA instead: `pip install <CARLA folder>/PythonAPI/carla/dist/carla-0.9.16-<your python>.whl`.
2. Start CARLA 0.9.16 and wait until the map has loaded.
3. Run everything from the `generate_dataset` folder:
   ```bash
   cd generate_dataset
   python main.py --check           # once: checks ego car, pedestrian types, car/truck/cyclist ids, maps
   python main.py --self-test       # once: renders every occlusion layout and checks it came out right
   python main.py --weather-preview # optional: one picture sheet of every weather setting at 0..100
   python main.py --plan-only 5     # optional: plans 5 jobs per weather + scene, renders nothing
   python main.py                   # generates the dataset into generate_dataset/dataset/
   python main.py --delete-flagged  # lists problem images; add --yes to delete them
   python main.py --summary         # refreshes manifest / summary / train-val-test lists
   ```

`python main.py` restarts its own generator process after any crash (including CARLA
core dumps) and continues where it stopped. Running it again later also resumes. Set
`CARLA_EXECUTABLE` in `config.py` (e.g. `"/opt/carla/CarlaUE4.sh"`) to let it start / restart
the CARLA server too.

## All commands

Run every command from the `generate_dataset` folder. Start CARLA first, except for
`--delete-flagged` and `--summary`, which only read files on disk.

### `python main.py`

**Makes the dataset.** This is the main command.

1. Connects to CARLA, loads the map from `MAPS`, and removes any leftover cars, people
   or cameras.
2. Finds the places where people can cross: marked crosswalks, straight roads,
   T-junctions, and the roads cars turn into at junctions.
3. Builds the job list: all clear-weather images first (crosswalk, jaywalk, occluded),
   then the next weather, and so on.
4. For each job:
   - **plans** the whole scene first (where the ego, people and cars go, and how fast
     people walk);
   - **spawns** it in CARLA, lets people walk, and **captures** image, depth and
     segmentation;
   - **measures** the 2D/3D boxes and how much of the pedestrian is hidden;
   - **saves** all outputs.
5. Writes `jobs_done.txt` after every job, so nothing is lost.

If CARLA crashes, it restarts by itself and continues. If you stop it (Ctrl+C) and run
it again later, it **continues** where it stopped.

### `python main.py --out /data/run1`

Same as above, but the dataset goes into `/data/run1` instead of the default `dataset/`
folder. Pointing it at an existing folder **resumes** that dataset.

Use a **new** folder after you change settings in `config.py`, such as image counts,
weathers or scene types.

### `python main.py --check`

**Checks your CARLA setup before a long run.** It generates nothing. It prints:

- whether your ego car (`my_car_xyz`) exists;
- every pedestrian model and whether it counts as child, woman or man;
- which car, truck and cyclist models from `config.py` exist on the server
  (`ok` / `MISSING`);
- every vehicle model available, in case you want to add some;
- whether the maps in `MAPS` are installed.

Run it once, before your first real run.

### `python main.py --plan-only 5`

**A quick test that renders nothing.** It loads each map and *plans* 5 jobs for every
weather + scene type, without spawning anything. It prints:

- how many crossing places the map has (crosswalks, mid-block, T-junctions);
- how many plans succeeded for each weather and scene type;
- the most common reasons plans were rejected.

Use it to see whether a map gives enough variety, or why some scene type keeps failing.

### `python main.py --delete-flagged`

**Shows the images that failed a quality check.** These images were saved anyway and
listed in `flagged.csv`. It prints how many there are, grouped by reason, plus the first
20 with their reason, and **deletes nothing**. The reasons are:

- the target pedestrian is not in view, or too small;
- the measured occlusion is far from what was planned;
- the camera looks blocked.

Look at those images in `images_2d_bbox/` before deciding.

- `--reason occlusion`: only show flagged images whose reason contains that text.
- `--yes`: actually **delete** them, every output of each image (image, depth,
  segmentation, box images, labels). Their rows move from `flagged.csv` to `deleted.csv`.
- `--reason "not in view" --yes`: delete only that kind and keep the rest.
- `--out /data/run1`: work on a different dataset folder.

### `python main.py --summary`

**Rebuilds the overview files** from the images on disk:

- `manifest.csv`: one row per image (weather, scene type, place, counts, occlusion,
  flagged or not);
- `summary.txt`: totals per weather, scene type and map, plus flag reasons;
- `train.txt`, `val.txt`, `test.txt`: lists for training. Images of the same place always
  go to the same list, so near-identical images can't leak from training into validation.
  Flagged images are left out.

This runs automatically at the end of a full run. Run it yourself after deleting images,
to bring the lists up to date.

### `python main.py --cleanup`

**Empties the CARLA world.** It removes every car, person and camera, and switches CARLA
back to normal mode, so it doesn't stay frozen.

You normally don't need it: Ctrl+C already does this, and every run cleans up first. It's
for a program killed hard (`kill -9`, a crash, closing the machine) when CARLA is left
frozen or full of leftover actors.

### `python main.py --self-test`

**Checks in CARLA that occluded scenes come out as planned.** Run it once on a new
machine, map or CARLA version, before a long run. For every occluder layout (see
"Occluded scenes" below) it plans and renders 2 scenes on the first map in `MAPS` and
checks each one:

- every car and the pedestrian really are where the plan put them, at every frame;
- measured from the segmentation, the pedestrian is clearly visible in the first and
  last frame and mostly hidden in between (visible -> hidden -> visible);
- the pedestrian is labeled `Pedestrian`, the occluding cars `Car` / `Truck` / `Bus`.

It prints `ok` / `FAIL` per scene, with the measured occlusion of each frame, and saves
the frames with their 2D boxes to `dataset/self_test/` so you can look at them. Nothing is
added to the dataset. A layout the map has no place for is reported as `skip`.

- `--self-test 5`: 5 scenes per layout instead of 2.

### `python main.py --weather-preview`

**Shows what each weather setting looks like**, so you can choose the `(min, max)` ranges
in `WEATHER_LOOK` (advanced.py). It sets up one street view on the first map in `MAPS`
(a crossing with 3 people and a car, so you can judge how well people stay visible) and
photographs it with one setting at a time at 0, 25, 50, 75 and 100, everything else at 0
and the sun high:

| Row | Setting |
|---|---|
| 1-6 | `cloudiness`, `fog_density`, `wind_intensity`, `precipitation`, `precipitation_deposits` (puddles), `wetness` (wet road look) |
| 7 | rain: `cloudiness`, `precipitation`, `precipitation_deposits` and `wetness` all at that level |
| 8 | camera brightness `exposure_compensation` -1.5 .. +0.5 on a cloudy sky (-1.5 is CARLA's default on Linux; +1 = 2x brighter) |

Result: `dataset/weather_preview/sheet.jpg` (all rows in one picture) and every picture
full size next to it (`cloudiness_050.jpg`, `exposure_-1.0.jpg`, ...). `wind_intensity` only
moves trees and flags, so it barely shows in a still picture. To use a brightness, add
`"exposure_compensation": <value>` to `RGB_ATTRIBUTES` in advanced.py.

## Stopping

Stop any time with **Ctrl+C**. **Ctrl+Z**, `kill <pid>` and closing the terminal do the
same. Normally Ctrl+Z only *pauses* a program, which would leave CARLA frozen with our
cars and people in it, so here it stops cleanly instead. Before the program exits it:

1. removes the cars and people of the scene being made,
2. removes the ego vehicle and its three cameras,
3. switches the CARLA server back to normal (asynchronous) mode, so it does not freeze,
4. runs a final `--cleanup` pass to make sure nothing of ours is left.

This takes a few seconds (at most 60). Pressing Ctrl+C again during it is ignored, so
the cleanup always finishes. A CARLA server started by `main.py` (`CARLA_EXECUTABLE`)
keeps running. Run `python main.py` again to continue from the last finished job; a
scene stopped halfway is simply made again.

After a hard kill (`kill -9`, power loss) nothing can clean up. Run
`python main.py --cleanup` once, or just start `python main.py` again: it always
empties the world first.

## What gets generated

Everything is set in [generate_dataset/config.py](generate_dataset/config.py).

**Maps**: `MAPS` lists the maps to use; images are split evenly across them. Any map
the server has works, including your own custom maps: give its short name
(`"Town03_Opt"`, `"MyCity"`) or full path (`"/Game/Custom/Maps/MyCity"`).
`python main.py --check` shows which maps the server has. A map without crosswalks or
sidewalks simply gets the scene types it can support, and the run says which it skipped.

**Variety**: no two images look the same.

- **Spots used evenly**: a crossing spot is reused only after all spots have been used
  equally often, and at most `MAX_USES_PER_SPOT` times per weather (config.py). If a
  weather asks for more images than its spots allow, the run warns at the start and the
  extra jobs are listed in `skipped.csv`. Fix by adding maps, raising
  `MAX_USES_PER_SPOT`, or lowering `images`.
- **Different views**: at the same spot, the same lane needs at least 6 m more or less
  distance to count as a new picture (`MIN_VIEW_DISTANCE_GAP`).
- **Camera and light vary**: ego distance 8-40 m, heading +/-5 deg, camera height
  +0-0.4 m, camera tilt +/-3 deg, and new sun / cloud values every 5 jobs within the same
  weather.
- **More spots**: mid-block crossing spots every 10 m along suitable roads.

**Weathers** run one after another, in the order listed. The weather is set once per
block, not per image.

```python
WEATHERS = {
    "clear":         dict(enabled=True, images=1000, pedestrians=1, vehicles=0, cyclists=0, trucks=0, occlusion=0),
    "partly_cloudy": dict(enabled=True, images=1000, pedestrians=2, vehicles=1, cyclists=1, trucks=1, occlusion=(50, 75)),
    "overcast":      dict(enabled=True, images=1000, pedestrians=3, vehicles=2, cyclists=2, trucks=2, occlusion=(60, 85)),
    "rain":          dict(enabled=True, images=1000, pedestrians=4, vehicles=3, cyclists=3, trucks=0, occlusion=(70, 95)),
    "snowfall":      dict(enabled=True, images=1000, pedestrians=5, vehicles=4, cyclists=0, trucks=0, occlusion=(85, 100)),
}
```

- Counts: a number is exact; `(min, max)` picks a random value per image. Counts are
  always met exactly; a scene that cannot fit them is re-planned.
- `occlusion` is a `(min, max)` range in %: how much of the *target* pedestrian is hidden
  at the most hidden moment of its walk behind a car. `0` = this weather has no occluded
  scenes. Behind a car that moment is naturally 50-100 % (behind a parked car 70-100 %),
  because a car covers a person from the feet up to about the chest; the run warns if a
  range is mostly below 50 %. Lower levels are in the frames before and after it.
- All weathers are daytime with the sun 70–90° high, so it is never in frame.

**Scene types** inside each weather (`SCENES`, each with an on/off switch and a share):

| Scene | What happens |
|---|---|
| `crosswalk` | crossing at a marked crosswalk |
| `jaywalk_road` | crossing mid-block, no crosswalk |
| `jaywalk_junction` | crossing an arm of a T-junction (3 roads meet) with no crosswalk |
| `occluded_by_traffic` | 1–2 cars ahead of the ego, right in front of it or in the lane next to it; the pedestrian walks behind them |
| `occluded_by_parked` | a car parked at the curb; the pedestrian walks behind it |
| `occluded_turning` | the ego turns left or right at a junction behind 1–2 cars turning the same way; the pedestrian crosses the road they turn into, behind the car(s) |

**Occluded scenes are sequences** (`SEQUENCE`). Seen from the ego, the pedestrian is first
clearly visible, walks behind the car, and comes out clearly visible again. The planner
predicts how much of the person is hidden at every 10 cm of the walk and only accepts a
walk whose most hidden moment lies in the weather's `occlusion` range. With `frames=5`
the frames are spread over the walk, e.g. 0 % -> 40 % -> 75 % -> 35 % -> 0 % hidden
(visible, going behind, most hidden, coming out, visible). "Visible" frames show the
whole person inside the picture. The person walks slowly (0.6-1.0 m/s). With
`SEQUENCE.enabled = False`, only the most hidden frame is taken.

The occluder layouts, picked at random for each scene:

| Layout | Scene |
|---|---|
| ego -> car -> pedestrian (same lane) | `occluded_by_traffic` |
| ego -> car -> car -> pedestrian (same lane) | `occluded_by_traffic` |
| 1 or 2 cars in the lane next to the ego -> pedestrian | `occluded_by_traffic` |
| parked car -> pedestrian | `occluded_by_parked` |
| turning left: ego -> car (-> car) -> pedestrian | `occluded_turning` |
| turning right: ego -> car (-> car) -> pedestrian | `occluded_turning` |

Same lane or next lane is 50/50 (`OCCLUDER_SAME_LANE_CHANCE` in `advanced.py`). In a turn the
pedestrian crosses either at the junction exit (its crosswalk, or a line there) or a
little further down that road (`TURN_FAR_CROSSING`, 12 m), where the turning cars are
straight again. Two cars in a left turn fit only on bigger junctions, so that layout is
the rarest; a layout that does not fit at a spot is simply re-planned.

**Traffic** is cars, trucks and cyclists/scooters. Each is randomly the same direction as
the ego or oncoming. Same-direction traffic stays within a 2.4 m ego-lane band.

## How one job is made

1. **Plan** the whole scene as data: crossing site, ego pose, occluders, the target
   pedestrian's walk and frame times, the other pedestrians, and the traffic. Placement
   uses a camera model to predict how much of the pedestrian is hidden at every step.
2. **Check the plan**: no overlaps, crossing in view, normal traffic never hides the
   target, and no near-identical view (see "Variety" above).
3. **Render**: spawn, let the pedestrians walk, and capture RGB, segmentation and depth
   from exactly the same tick. Our traffic is then hidden for one tick to see each
   pedestrian unoccluded.
4. **Label** from segmentation: the visible box, the full-body box and the **measured**
   occlusion %.
5. **Check** each frame. A frame that fails is **saved anyway** and listed in
   `flagged.csv` with the reason.

## Problem images: flagged, not thrown away

A frame is flagged when:
- the target pedestrian is not in view, or too small;
- the measured occlusion misses the planned value by more than 15 %;
- the camera looks blocked.

1. `python main.py --delete-flagged` lists them with the reasons.
2. Look at them in `dataset/images_2d_bbox/`.
3. `python main.py --delete-flagged --yes` deletes them, or add `--reason <text>` to delete only
   one kind. Every output of each image is removed, and the rows move to `deleted.csv`.
4. `python main.py --summary` refreshes the manifest, summary and split lists.

Set `SKIP_FLAGGED_IMAGES = True` in config.py to **skip** failing frames instead: they are
never saved (the reason is written to `run_log.txt`), and a job whose frames all fail is
re-planned. A job that never produces an image is listed in `skipped.csv`.

## Output files

```
dataset/
  images/ depth/ seg_color/ images_2d_bbox/ images_3d_bbox/ labels/ labels_3d/
  data.yaml  train.txt  val.txt  test.txt     YOLO dataset, split by location
  manifest.csv  summary.txt                   one row per image / totals
  jobs_done.txt  flagged.csv  skipped.csv  deleted.csv  run_log.txt
```

Every output folder is flat (no sub-folders), and one image has the same name in all of
them. The name is `<weather>_<scene>_<number>`, the number counting up from `00001`:

```
images/clear_crosswalk_00001.png
images/rain_occluded_00001.png      \
images/rain_occluded_00002.png       |
images/rain_occluded_00003.png       |  one walk behind a car: 5 numbers in a row
images/rain_occluded_00004.png       |
images/rain_occluded_00005.png      /
images/rain_occluded_00006.png         the next walk
depth/rain_occluded_00001.png          same name in every output folder
labels/rain_occluded_00001.txt
```

| name part | weather / scene |
|---|---|
| `clear`, `cloudy`, `overcast`, `rain`, `snow` | clear, partly_cloudy, overcast, rain, snowfall |
| `crosswalk`, `jaywalk`, `junction` | crosswalk, jaywalk_road, jaywalk_junction |
| `occluded` | occluded_by_traffic, occluded_by_parked, occluded_turning |

The short names are in `NAME_CODES` in `advanced.py`; a weather you add yourself keeps
its own name. Scenes that share a short name share one counter, so names never clash.
With more than one map in `MAPS`, a map code comes first: `t03_rain_occluded_00001.png`
(Town03_Opt -> `t03`, Town10HD_Opt -> `t10hd`, a custom map keeps its name). Weather,
exact scene type, map and the walk an image belongs to (`job`) are in its `labels_3d`
JSON and in `manifest.csv`. YOLO training uses the folders as they are (`data.yaml`
lists every image). With `SKIP_FLAGGED_IMAGES = True` a skipped frame leaves a gap in the
numbers.

The box images (`images_2d_bbox/`, `images_3d_bbox/`) draw exactly the objects in the
YOLO file, in one colour with no text. Set `BOX_COLOR` (OpenCV B, G, R; default green) and
`BOX_THICKNESS` in `advanced.py`.

**How boxes are labeled.** Every box is measured on the segmentation image, and its class
comes from CARLA's own semantic tag on that object's pixels: Pedestrian -> `pedestrian`,
Car -> `vehicle`, Truck / Bus -> `truck`, Bicycle / Motorcycle (with its Rider) ->
`cyclist`. So a model is labeled by what CARLA says it is, not by which config list it
came from; the JSON also keeps CARLA's own name (`carla_class`, e.g. `Bus`). A pixel
only counts for an object if its depth also matches that object's distance, so an object
hidden behind a wall can never take the box of another, visible object. The full-body
box of a pedestrian comes from a frame with our traffic moved away, so the car in front
never cuts it.

YOLO classes: `0 pedestrian, 1 vehicle, 2 truck, 3 cyclist`. `SPLIT_CROSSING_CLASS = True`
adds `crossing_pedestrian` in front. Pedestrian boxes are full-body by default
(`PED_YOLO_BOX`). A pedestrian hidden completely is in the JSON but not in YOLO, unless
`LABEL_FULLY_HIDDEN_PEDESTRIANS = True`.

`train.txt` / `val.txt` / `test.txt` are grouped by location (`TRAIN_VAL_TEST`). All
frames of a sequence, and every image from the same crossing, land in the same split.
Without this, near-identical images would leak from train into val.

## Files

You normally only edit `config.py` and run `main.py`. The code that does the work is in
`engine/`, one small file per job. [CODE_GUIDE.md](CODE_GUIDE.md) explains every file
and function.

```
generate_dataset/
  config.py        everyday settings: images, weathers, scene types, counts, outputs
  advanced.py      fine tuning: weather look, speeds, tolerances, camera, box colour
  main.py          the only script you run: the list of commands
  engine/
    stop.py          clean stop on Ctrl+C / Ctrl+Z / kill
    supervisor.py    auto-restart after crashes, optional CARLA server start
    tools.py         --check, --delete-flagged, --cleanup
    roads.py         lanes, sidewalks, crossing sites (crosswalk, mid-block, T-junction,
                     turn), planning camera: how much of a pedestrian a vehicle hides
    people.py        pedestrian planning: who walks where, how fast
    traffic.py       car / truck / cyclist placement, occluder layouts
    occlusion.py     occluder + walk behind it: visible -> hidden -> visible
    planner.py       puts a whole scene plan together and checks it
    carla_io.py      spawning / removing actors; per-map session, weather, capture
    render.py        runs one plan in CARLA and captures its frames
    labels.py        labels from segmentation, quality checks, saving outputs
    run.py           job list, job loop, progress files, summary
    selftest.py      --self-test: every occluder layout, rendered and checked in CARLA
    weather_preview.py  --weather-preview: picture sheet of every weather setting
tests/             unit tests, run without CARLA (see "Tests")
```

## Tests

The tests run **without CARLA**, on a small fake CARLA in `tests/fake/carla.py`: a city
block with a 4-way junction (turn lanes, crosswalks, sidewalks) that draws segmentation
and depth from the actors' shapes. They use their own fixed settings, so your
`config.py` does not change the result. From the `Dataset_Generation` folder:

```bash
python -m unittest discover -s tests            # everything (about 2 minutes)
python -m unittest discover -s tests -v         # with the name of every check
python -m unittest discover -s tests -p test_layouts.py     # one file
```

| File | Checks |
|---|---|
| `test_layouts.py` | every occluder layout (same lane, next lane, parked, turning left / right, 1 or 2 cars): cars in the right lane and order (ego -> car(s) -> pedestrian), walk visible -> hidden -> visible, and 3D sight lines from the camera really blocked by a car at the most hidden frame |
| `test_render.py` | every layout through the real render code: cars and pedestrian exactly where planned at every frame, the rendered segmentation goes visible -> hidden -> visible, CARLA classes on the boxes |
| `test_labels.py` | CARLA classes (Truck, Bicycle + rider), a hidden car never takes a visible car's box, the full-body box is not cut by the car in front |
| `test_names.py` | flat, unique file names; numbers in a row per walk; map code with several maps |
| `test_run.py` | a small full run: every image has all its files, every scene type is made, resume adds nothing, flagged images kept / deleted / skipped |
| `test_variety.py` | spots used evenly and at most `MAX_USES_PER_SPOT`, views 6 m apart, camera height varies, maps by short name or path |
| `test_stop.py` | Ctrl+C / Ctrl+Z / kill stop cleanly, and the world is emptied even with Ctrl+C pressed again during cleanup |
| `test_selftest.py`, `test_weather_preview.py` | `--self-test` and `--weather-preview` themselves |

The tests check the code; `python main.py --self-test` checks the same things in real CARLA.

## Known limitations

- **Snow:** CARLA 0.9.16 has no snow. `snowfall` is only a hazy, sleet-like
  approximation. Disable it or replace it if you have real snow assets.
- **100 % occlusion** means the pedestrian cannot be seen at all in that frame. It is
  labeled in the JSON (`visible: false`) but left out of YOLO by default.
- **Occlusion** counts only occlusion by the traffic we place. A pole or parked map car
  in front of a pedestrian lowers `visible_pixels` but not the occlusion %.
- **Static traffic:** vehicles are frozen in place (no motion blur, no driving).
