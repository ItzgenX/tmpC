# CARLA Pedestrian-Crossing Dataset Generator

Generates a labeled image dataset from a running CARLA simulator, aimed at
training pedestrian-crossing detection models. Each generated frame comes
with YOLO-format 2D boxes, a richer 3D JSON label, semantic segmentation
(raw + colorized), and optional debug images with boxes drawn on them.

The dataset is built around three things that matter for this kind of
model: pedestrians actually walking across a **real crosswalk**, a
controlled fraction of pedestrians **jaywalking** mid-road instead, and a
controlled fraction of pedestrians being **partially occluded** by traffic.

Everything lives in `generate_dataset/`:

| File | What it's for |
|---|---|
| `scenario_config.py` | Every setting you'd want to change. Start here. |
| `generate_scenarios.py` | The script you actually run. |
| `spawner.py` | Spawns ego vehicle, traffic, cyclists/scooters/motorcycles, and walking pedestrians. |
| `occlusion.py` | The occlusion placement search (where exactly to stand a pedestrian so they're covered by X%). |
| `utils.py` | Camera projection math, bounding-box helpers, crosswalk/lane geometry, the per-image count roller. |

---

## 1. Requirements

- A running CARLA server (this was built against **CARLA 0.9.16**) that you
  can reach at `localhost:2000`.
- A Python environment with:
  - the `carla` client package matching your server's version (`pip install carla`,
    or whatever wheel your CARLA install ships)
  - `opencv-python` (`cv2`)
  - `numpy`
- Your CARLA content needs two things that **aren't stock CARLA**:
  - a vehicle blueprint literally named `my_car_xyz` — this is your ego
    vehicle. If it's missing, the script fails immediately with an
    `IndexError` on the very first line that looks it up.
  - a `vehicle.bh.rider` blueprint, used as the e-scooter+rider actor.
    If your content pack doesn't have it, it's silently skipped at spawn
    time (not fatal), you'll just never get e-scooter detections.
- Every other vehicle/truck/cyclist/motorcycle blueprint ID used is
  standard CARLA 0.9.16 content (see §5 to swap any of them out).

---

## 2. Quick start

```bash
# 1. start your CARLA server first, wait for it to fully load
# 2. activate the env that has `carla` + opencv + numpy installed
cd generate_dataset
python generate_scenarios.py
```

That's it for a default run. It will:

1. Connect to the server, switch synchronous mode on.
2. Scan the currently-loaded map for crosswalks.
3. Loop through the 5 weather scenarios in `SCENARIOS` (clear, partly
   cloudy, overcast, rain, snowfall), generating the number of images each
   one asks for.
4. Write everything into a timestamped folder under `dataset_scenarios/`
   (see §4).
5. Print — and save to `summary.txt` — a breakdown of what it actually
   generated: how many were crosswalk vs. jaywalk, how many were occluded
   vs. clear, per scenario.

There is **no CLI flag support** right now — every choice (which map, how
many images, forced crosswalk-only/jaywalk-only, occlusion rate, etc.) is
made by editing `scenario_config.py` before you run it. See §6 for the
exact edits for common cases.

---

## 3. How a single image gets made

For every frame, `generate_scenarios.py` does roughly this:

1. **Decide crosswalk or jaywalk.** Normally this is a coin-flip weighted
   by `JAYWALK_PROBABILITY`, and only for scenarios with `allow_jaywalk=True`.
   If jaywalking, it first finds a spot on the map that's nowhere near a
   crosswalk (`JAYWALK_CROSSWALK_EXCLUSION_RADIUS`) — a crosswalk and a
   jaywalker are never allowed to appear in the same frame.
2. **Spawn the ego vehicle** at an approach point before that crossing, let
   it settle, freeze its physics.
3. **Roll actor counts** for this specific image — pedestrians, cars,
   trucks, cyclists/scooters/motorcycles, and an occlusion target — from
   each scenario's `(min, max)` ranges (see §5.2 for the acceptance rule
   that keeps this from producing boring/empty images).
4. **Spawn traffic**: cars, then trucks, then cyclists/scooters/motorcycles
   last (kept a minimum distance from trucks — a bike doesn't ride
   hugging a truck in real traffic).
5. **Spawn pedestrians.** If this image has a nonzero occlusion target, one
   pedestrian goes through the occlusion placement search in `occlusion.py`
   first; the rest walk normally across the crossing via CARLA's AI walker
   controller (so it's an actual walk cycle, not a frozen pose).
6. **Let the simulation tick** for a random warm-up period so everyone is
   mid-stride, then capture an RGB + semantic segmentation frame.
7. **Build labels**: project every visible actor's 3D bounding box into
   the 2D image, filter out anything occluded by static geometry or
   outside render distance, write YOLO `.txt` + a richer 3D `.json`, and
   optionally draw debug images with boxes on them.
8. **Destroy every actor spawned this frame** (including walker AI
   controllers — this used to leak and pile up actors run after run; fixed).
9. **Log one row to `manifest.csv`.**

---

## 4. Output layout

Each full run of `generate_scenarios.py` creates **one new folder**, named
after whatever was actually active when you ran it:

```
dataset_scenarios/
  <crossing_mode>__<occlusion_mode>__<map_setup>__<timestamp>/
    manifest.csv
    run_log.txt
    summary.txt
    clear/
      images/              frame_000000.png ...
      labels/               frame_000000.txt            (YOLO format)
      labels_3d/             frame_000000.json           (richer per-detection data)
      images_segmentation_raw/
      images_segmentation_color/
      images_with_2d_bbox/   (only if DRAW_2D_BBOX = True)
      images_with_3d_bbox/   (only if DRAW_3D_BBOX = True)
    partly_cloudy/
      ... same structure ...
    overcast/
    rain/
    snowfall/
```

Example folder names:
- `mixed_crossing__occlusion_60pct__current_map__20261012_153000` — a
  normal default run.
- `jaywalk_only__occlusion_60pct__current_map__20261012_160512` — a
  dedicated jaywalk batch (see §6).
- `mixed_crossing__occlusion_forced__map_rotation__20261012_161030` — a
  normal crossing mix, every image occluded, cycling across multiple maps.

The timestamp means **two runs never collide or get merged**, even with
identical settings. Frame numbering inside a given run's scenario folder
resumes from whatever's already there, so if a run gets interrupted and
you re-run it (pointing at the same folder — not something the script does
automatically, see the note at the end of §5.1), you won't lose or
overwrite anything.

### Label formats

**`labels/frame_XXXXXX.txt`** — standard YOLO: one line per detection,
`class_id x_center y_center width height`, all normalized 0–1.

**`labels_3d/frame_XXXXXX.json`** — a list of per-detection dicts with:
`class_id`, `label_name`, `2d_bbox_pixel`, `3d_world_location`,
`3d_world_rotation`, `extent_dimensions`, `camera_relative_location`, and
— for pedestrians only — `target_occlusion_percent`,
`achieved_occlusion_percent`, `occlusion_within_tolerance`,
`is_jaywalking`, `map`.

**`manifest.csv`** — one row per saved image across the whole run:
`image_name, scenario, map, pedestrians, vehicles, cyclists_scooters,
trucks, target_occlusion_percent, achieved_occlusion_percent,
occlusion_within_tolerance, is_jaywalking`.

### Class IDs (`CLASS_MAP` in `scenario_config.py`)

| ID | Class |
|---|---|
| 0 | Pedestrian |
| 1 | Cyclist (pedal bike) |
| 2 | E-Scooter + Rider |
| 3 | Vehicle |
| 4 | Motorcycle (motorized two-wheeler: motorcycle/moped) |
| 5 | Truck *(only if `SEPARATE_TRUCK_CLASS = True`; otherwise trucks are labeled Vehicle)* |

---

## 5. Configuration reference (`scenario_config.py`)

### 5.1 Map

```python
MAP_NAME = None        # e.g. "Town10HD_Opt" - loads it if different from what's running
MAP_ROTATION = []      # e.g. ["Town01", "Town02", "Town03"] - cycles through these instead
```

`MAP_ROTATION` is ignored unless empty; if you set it, each scenario's
`number_images_to_generate` is split evenly across the listed maps. This
is the real fix for images looking repetitive — one map only has so many
distinct crosswalks and buildings, so cycling maps (not just re-randomizing
within one) is what actually introduces new backgrounds.

### 5.2 The scenario table

```python
SCENARIOS = [
    dict(
        name="overcast",
        weather=carla.WeatherParameters.WetCloudyNoon,   # any carla.WeatherParameters object
        pedestrians=(2, 5), vehicles=(1, 3),
        cyclists_scooters=(1, 3), trucks=(0, 2),
        occlusion_percent=(0, 75),                        # rolled per image, not fixed
        allow_jaywalk=True,
        number_images_to_generate=1700,
    ),
    ...
]
```

Every count (including `occlusion_percent`) is a `(min, max)` range rolled
fresh **per image**, not a fixed number. This is governed by:

```python
REQUIRE_PEDESTRIAN = True              # every saved image has >=1 pedestrian
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2  # need >=2 of {vehicles, cyclists_scooters, trucks, occlusion} nonzero
MAX_COUNT_REROLLS = 8                  # reroll this many times before just bumping the best attempt
```

This is what gives you *varied but not boring* images: a lone pedestrian
on an empty street never gets saved, but you also don't get every single
image maxed out with every category present.

Adding a new scenario is just one more dict in the list — nothing else to
touch. `weather` takes any `carla.WeatherParameters` object directly,
including a custom one you construct inline (see the `snowfall` entry,
which approximates snow via fog/wetness/cold light since CARLA's base town
content has no actual snow assets).

> **Output folders don't auto-resume across different configs.** Since
> every run gets a fresh timestamped folder (§4), if you want to literally
> top up an existing run's folder instead of starting a new one, you'd
> need to point `OUTPUT_BASE_DIR`/rename things manually — this isn't
> wired up as an automatic "resume this specific run" flag.

### 5.3 Jaywalking

```python
JAYWALK_PROBABILITY = 0.12                           # odds per eligible image
FORCE_CROSSING_MODE = None                           # None | "crosswalk_only" | "jaywalk_only"
JAYWALK_CROSSWALK_EXCLUSION_RADIUS = MAX_RENDER_DISTANCE + 15.0
```

`allow_jaywalk` on a scenario just makes it *eligible*; the actual odds are
this one shared `JAYWALK_PROBABILITY` knob, not re-tuned per row.
`FORCE_CROSSING_MODE` overrides everything for a dedicated batch — see §6.

### 5.4 Occlusion

```python
OCCLUSION_SEARCH_SAMPLES = 25        # how many points along the crossing to test
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)
OCCLUSION_TOLERANCE_PERCENT = 12.0   # how close to target counts as "close enough"
OCCLUSION_OVERLAP_GRID = 64          # resolution of the overlap-scoring grid
OCCLUSION_PROBABILITY = 0.6          # fraction of eligible images that get occluded at all
OCCLUSION_STRICT = False             # discard+retry instead of saving a miss
```

How it actually works: `occlusion.py` only ever uses vehicles/trucks/
cyclists that are **already spawned** in the scene as occluders — it never
spawns an extra prop just to block the view. It searches 25 points along
the crosswalk line and picks whichever one makes the pedestrian's 2D
bounding box overlap the nearest-to-camera traffic actors by the closest
percentage to your target (and it checks camera depth, not just 2D overlap
— an "occluder" that's actually behind the pedestrian doesn't count).

`OCCLUSION_PROBABILITY` exists because rolling `occlusion_percent` from a
range like `(0, 75)` would give you *some* nonzero value on 75/76 images —
not a real occluded/clear split. The probability gate decides per image
whether occlusion is attempted at all; if it fires, the rolled value is
floored to at least 1 so the gate can't be silently undone by drawing 0.

`OCCLUSION_STRICT` closes the last gap: even when occlusion is attempted,
the geometry search might not find a usable occluder and only achieve 0%.
With `OCCLUSION_STRICT = True`, that frame is discarded and retried instead
of being saved with a false "occluded" label.

### 5.5 Traffic realism

```python
MIN_SEPARATION_BETWEEN_VEHICLES = 3.5     # no two same-frame vehicles closer than this
MIN_SEPARATION_TRUCK_TO_CYCLIST = 6.0     # trucks kept further from cyclists specifically
```

Vehicles are jittered along their lane's own right-vector (not raw x/y
noise), so they stay parallel to the road instead of clipping sideways on
a curve. Cyclists/scooters/motorcycles try a dedicated `LaneType.Biking`
lane first if the map has one.

### 5.6 Blueprint categories

```python
ALLOWED_CAR_VEHICLES = [...]
ALLOWED_TRUCK_VEHICLES = [...]
ALLOWED_CYCLIST_VEHICLES = [...]        # pedal bikes -> "Cyclist"
ALLOWED_ESCOOTER_VEHICLES = [...]       # -> "E-Scooter + Rider"
ALLOWED_TWO_WHEELER_VEHICLES = [...]    # motorcycles/mopeds -> "Motorcycle"
```

All four of these feed from the same `cyclists_scooters` scenario count
but get detected and labeled separately. Edit any list freely — an ID
your content pack doesn't have is just skipped at spawn time, not fatal.

### 5.7 Class map

```python
SEPARATE_TRUCK_CLASS = True   # False -> trucks get labeled "Vehicle" instead of their own class
```

---

## 6. Common recipes

All of these are edits to `scenario_config.py`, then `python generate_scenarios.py`.

**Only legal crosswalk crossings, no jaywalking at all:**
```python
FORCE_CROSSING_MODE = "crosswalk_only"
```

**Only jaywalking (ignores each scenario's `allow_jaywalk` — if you're
forcing this mode you want it everywhere):**
```python
FORCE_CROSSING_MODE = "jaywalk_only"
```

**Every image occluded, and only keep ones that actually hit the target:**
```python
OCCLUSION_PROBABILITY = 1.0
OCCLUSION_STRICT = True
```

**No occlusion at all:**
```python
OCCLUSION_PROBABILITY = 0.0
```

**A small test batch (e.g. 50 images) instead of the full production run:**
Lower `number_images_to_generate` on the scenarios you care about (or all
five). There's no `--count` flag yet — see the note at the end of this
section.

**Spread one scenario across several maps for variety:**
```python
MAP_ROTATION = ["Town01", "Town02", "Town03", "Town10HD_Opt"]
```

Remember to flip `FORCE_CROSSING_MODE` back to `None` and
`OCCLUSION_PROBABILITY`/`OCCLUSION_STRICT` back to their normal values
afterward if you want your next run to go back to the default mixed
behavior.

> **Known gap:** doing "50 crosswalk-only + 50 jaywalk-only + 50
> occlusion-only" today means editing the config and re-running the script
> three separate times. A `--mode`/`--count` CLI flag to do this in one
> command was planned but not finished — ask if you want it added.

---

## 7. Troubleshooting

- **`IndexError` on `blueprints.filter('my_car_xyz')[0]`** — your CARLA
  content doesn't have the `my_car_xyz` ego vehicle blueprint loaded.
- **"no crosswalks found on \<map\>, skipping this chunk"** — that map has
  no crosswalk geometry CARLA's API can see. Try a different map.
- **A frame fails but the run keeps going** — expected. Every frame is
  wrapped in its own try/except; failures get logged to `run_log.txt`
  inside that run's output folder with the scenario/map/frame context, and
  generation continues. Check that file first if something looks off.
- **Actors seem to be piling up across runs** — this was a real bug (walker
  AI controllers were never destroyed because cleanup skipped anything
  with a parent) and is fixed; if you still see it, check `run_log.txt` for
  exceptions during cleanup specifically.
- **Running the CARLA server on Linux and getting an opaque core dump** —
  the Python-side error that *caused* it (if any) will be in `run_log.txt`
  before the crash; check there first rather than the core dump itself.

---

## 8. Known limitations

- **Snow isn't real snow.** CARLA's base town content has no snow
  ground/particle assets, so the `snowfall` scenario approximates it with
  fog, wetness, and cold lighting via `WeatherParameters`. Swap in a
  snow-enabled map/weather preset if you have access to one.
- **Occlusion is 2D-bbox-based, not pixel-perfect.** It's a solid proxy
  (and depth-checked, so it won't place an occluder that's actually
  behind the pedestrian), but a car's rectangular bounding box can stick
  out slightly past its visible silhouette, so "achieved occlusion %" is
  an estimate, not an exact pixel count.
- **No CLI arguments.** Every run is driven entirely by editing
  `scenario_config.py`. See the note in §6.
- **This hasn't been run against a live CARLA server from the environment
  this was developed in** — it's been checked for internal consistency
  (every cross-module import resolves, every function signature matches
  its call sites) but the first real end-to-end test is on you.
