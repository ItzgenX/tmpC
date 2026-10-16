# CARLA Pedestrian-Crossing Dataset Generator

Generates a labeled image dataset from a running CARLA simulator, aimed at
training pedestrian-crossing detection models. Each generated frame comes
with YOLO-format 2D boxes, a richer 3D JSON label, semantic segmentation
(raw + colorized), and optional debug images with boxes drawn on them.

Every image falls into exactly one of four types, split by an **exact
quota** (not a probability) you set in config: legal **crosswalk**
crossing, mid-road **jaywalking**, each either **clear** or with the
pedestrian **occluded** by traffic. Ask for 100 images with the default
25/25/25/25 split and you get exactly 25 of each, guaranteed.

Everything lives in `generate_dataset/`:

| File | What it's for |
|---|---|
| `scenario_config.py` | Every setting you'd want to change. Start here. |
| `generate_scenarios.py` | The script you actually run. |
| `run_until_done.ps1` | Runs `generate_scenarios.py` in a loop, auto-resuming after any crash (including native ones) until the full plan is done. See §2.1. |
| `image_plan.py` | Turns the exact-quota split into a shuffled per-scenario image plan, persisted to disk so progress survives a crash. |
| `spawner.py` | Spawns ego vehicle, traffic, cyclists/scooters/motorcycles, and walking pedestrians. |
| `occlusion.py` | The occlusion placement search (where exactly to stand a pedestrian so they're covered by X%). |
| `utils.py` | Camera projection math, bounding-box helpers, crosswalk/lane geometry, the per-image actor-count roller. |

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
  standard CARLA 0.9.16 content (see §5.5 to swap any of them out).

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
2. For each weather scenario in `SCENARIOS` (clear, partly cloudy,
   overcast, rain — snowfall is currently commented out, see §5.2),
   build (or resume) that scenario's exact image-type plan.
3. Scan the map for crosswalks, generate images by working through that
   plan, and write everything into a fresh timestamped folder under
   `dataset_scenarios/` (see §4).
4. Print — and save to `summary.txt` — a breakdown of what actually got
   generated: how many were crosswalk vs. jaywalk, how many were occluded
   vs. clear, per scenario.

Three CLI flags cover the common cases without editing anything:

```bash
python generate_scenarios.py --mode jaywalk_only --count 50
python generate_scenarios.py --mode crosswalk_only --count 50
python generate_scenarios.py --mode occlusion_only --count 50
python generate_scenarios.py --mode mixed   # or just omit --mode entirely

# continue a previous run (e.g. after a CARLA crash) instead of starting fresh:
python generate_scenarios.py --resume dataset_scenarios/default_split__current_map__20261014_120000
```

`--mode` restricts the image-type split for this run only (doesn't touch
`scenario_config.py`); `--count` overrides `number_images_to_generate` on
every scenario in `SCENARIOS` (so `--count 50` means 50 images for *each*
scenario, not 50 total — see §8); `--resume` points at an existing run
folder to continue exactly where it left off, or `--resume latest` to
auto-pick the most recently modified one. Weather still cycles
through the full `SCENARIOS` table regardless of `--mode`, so an
`occlusion_only` or `jaywalk_only` batch still covers every configured
weather condition, not just one.

### 2.1 Running it reliably through crashes: `run_until_done.ps1`

CARLA client-side native crashes (core dumps / aborts) kill the Python
process outright — nothing inside `generate_scenarios.py` can catch or
retry around that, by definition. But every image is persisted to disk
the instant it's saved, so a crash only ever loses the one frame in
flight, not the dataset. `run_until_done.ps1` turns that into a
complete, hands-off pipeline: it runs `generate_scenarios.py`, and if
the process exits for *any* reason other than success, it waits a few
seconds and relaunches it with `--resume latest`, repeating until the
full plan actually finishes.

```powershell
cd generate_dataset
.\run_until_done.ps1                              # default mixed run
.\run_until_done.ps1 -Mode occlusion_only -Count 100
```

`-Mode`/`-Count` only matter for the very first launch — every relaunch
after that uses `--resume latest`, since the plan (and the split that
was baked into it) already lives on disk. If it crashes 5 times in a
row in under 8 seconds each, it stops and tells you to restart the
CARLA server by hand instead of burning through retries — that pattern
means the server itself is the problem, not a one-off client crash.

---

## 3. How a single image gets made

For every frame, `generate_scenarios.py` does roughly this:

1. **Pull the next item off this scenario's image-type plan** — one of
   `crosswalk_clear`, `crosswalk_occluded`, `jaywalk_clear`,
   `jaywalk_occluded`. This is an exact queue, not a dice roll (§5.2) — it
   only advances once an image is *actually saved*; a failed attempt
   retries the same bucket.
2. If the bucket calls for jaywalking, finds a spot on the map nowhere
   near a crosswalk (`JAYWALK_CROSSWALK_EXCLUSION_RADIUS`) — a crosswalk
   and a jaywalker are never allowed to appear in the same frame. If no
   such spot can be found this attempt, it fails and retries rather than
   silently substituting a crosswalk image.
3. **Reposition the ego vehicle** to an approach point before that
   crossing. The ego vehicle and both cameras are spawned only *once* per
   map chunk (not per frame — see §8 point 5) and reused across every
   frame in it; each frame just teleports the already-spawned ego vehicle
   to this frame's approach point and lets it settle.
4. **Roll actor counts** for this image — pedestrians, cars, trucks,
   cyclists/scooters/motorcycles — from each scenario's `(min, max)`
   ranges (see §5.3 for the acceptance rule that keeps this from
   producing boring/empty images).
5. **Spawn traffic**: cars, then trucks, then cyclists/scooters/motorcycles
   last (kept a minimum distance from trucks — a bike doesn't ride
   hugging a truck in real traffic).
6. **Spawn pedestrians.** If the bucket is `*_occluded`, one pedestrian
   goes through the occlusion placement search in `occlusion.py` first
   (target % rolled from `OCCLUSION_PERCENT_RANGE`, see §5.4); the rest
   walk normally across the crossing via CARLA's AI walker controller
   (so it's an actual walk cycle, not a frozen pose).
7. **Let the simulation tick** for a random warm-up period so everyone is
   mid-stride, then capture an RGB + semantic segmentation frame.
8. **Build labels**: project every visible actor's 3D bounding box into
   the 2D image, filter out anything occluded by static geometry or
   outside render distance, write YOLO `.txt` + a richer 3D `.json`, and
   optionally draw debug images with boxes on them.
9. **Destroy every actor spawned this frame** (including walker AI
   controllers), then give the simulation a short settle period
   (`POST_DESTROY_SETTLE_TICKS`/`_SECONDS`) before the next frame starts
   spawning — see §8 for why this matters.
10. **Pop the consumed item off the plan and persist what's left to
    `plan.json`**, log one row to `manifest.csv`.

---

## 4. Output layout

Each fresh run of `generate_scenarios.py` (without `--resume`) creates
**one new folder**, named after whatever image-type split was actually
active:

```
dataset_scenarios/
  <split_tag>__<map_setup>__<timestamp>/
    manifest.csv
    run_log.txt
    summary.txt
    clear/
      plan.json              remaining (unconsumed) image-type queue
      images/                frame_000000.png ...
      labels/                frame_000000.txt   (YOLO format)
      labels_3d/             frame_000000.json  (richer per-detection data)
      images_segmentation_raw/
      images_segmentation_color/
      images_with_2d_bbox/   (only if DRAW_2D_BBOX = True)
      images_with_3d_bbox/   (only if DRAW_3D_BBOX = True)
    partly_cloudy/
      ... same structure ...
    overcast/
    rain/
```

Example folder names: `default_split__current_map__20261014_120000` (a
normal run), `occlusion_only__map_rotation__20261014_130500` (a dedicated
occlusion batch cycling across multiple maps).

The timestamp means two fresh runs never collide. To continue a specific
run instead of starting a new one — e.g. after a CARLA crash — pass its
exact folder to `--resume` (§2). Both frame numbering (from what's
already in `images/`) and each scenario's exact remaining plan (from
`plan.json`) resume from disk, so a run interrupted partway through
doesn't lose or duplicate any progress — see §8 for the full crash story.

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
`image_name, scenario, map, bucket, pedestrians, vehicles,
cyclists_scooters, trucks, target_occlusion_percent,
achieved_occlusion_percent, occlusion_within_tolerance, is_jaywalking`.
`bucket` is the exact plan item this image fulfilled (e.g.
`jaywalk_occluded`).

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
remaining image count is split evenly across the listed maps. This is the
real fix for images looking repetitive — one map only has so many
distinct crosswalks and buildings, so cycling maps (not just
re-randomizing within one) is what actually introduces new backgrounds.

### 5.2 Exact image-type split

```python
IMAGE_TYPE_SPLIT = {
    "crosswalk_clear": 25,
    "crosswalk_occluded": 25,
    "jaywalk_clear": 25,
    "jaywalk_occluded": 25,
}
OCCLUSION_PERCENT_RANGE = (1, 100)   # how occluded an "*_occluded" image's pedestrian is
```

Every scenario's `number_images_to_generate` is split across these four
buckets by **exact count** (via `image_plan.build_image_plan`, largest-
remainder rounding so nothing is lost), not probability — 100 images
with the defaults above gives you exactly 25 of each, every time.
Percentages don't need to sum to 100 (they're normalized), just keep them
in the ratio you want.

Each scenario can opt a bucket type out entirely:

```python
SCENARIOS = [
    dict(
        name="overcast",
        weather=carla.WeatherParameters.WetCloudyNoon,   # any carla.WeatherParameters object
        pedestrians=(2, 5), vehicles=(1, 3),
        cyclists_scooters=(1, 3), trucks=(0, 2),
        allow_jaywalk=True, allow_occlusion=True,
        number_images_to_generate=1700,
    ),
    ...
]
```

If `allow_jaywalk=False` or `allow_occlusion=False`, that bucket's quota
folds into its nearest allowed counterpart instead of being dropped (e.g.
`jaywalk_occluded`'s quota becomes `crosswalk_occluded` if jaywalking is
disallowed) — so every one of `number_images_to_generate` is still
accounted for, exactly.

Adding a new scenario is just one more dict in the list — nothing else to
touch. `weather` takes any `carla.WeatherParameters` object directly,
including a custom one you construct inline. The `snowfall` entry (fog/
wetness/cold-light approximation, since CARLA's base town content has no
real snow assets) is currently commented out in the list — uncomment it
to bring it back.

### 5.3 Actor counts

```python
REQUIRE_PEDESTRIAN = True              # every saved image has >=1 pedestrian
MIN_OPTIONAL_CATEGORIES_SATISFIED = 2  # need >=2 of {vehicles, cyclists_scooters, trucks, this image's occlusion bucket} nonzero
MAX_COUNT_REROLLS = 8                  # reroll this many times before just bumping the best attempt
```

`pedestrians`/`vehicles`/`cyclists_scooters`/`trucks` in each scenario
dict are `(min, max)` ranges rolled fresh per image (`resolve_scenario_counts`
in `utils.py`). This is what gives you *varied but not boring* images: a
lone pedestrian on an empty street never gets saved, but you also don't
get every single image maxed out with every category present. Note this
only governs actor counts now — which crossing type and whether this
image is occluded come from the exact plan (§5.2), not from a roll here.

**Ambient sidewalk pedestrians** — separate flag, off by default:

```python
SPAWN_SIDEWALK_PEDESTRIANS = False          # background people on the footpath, not crossing
SIDEWALK_PEDESTRIAN_COUNT_RANGE = (0, 2)    # rolled fresh per image
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MIN = 8.0
SIDEWALK_PEDESTRIAN_SPAWN_DISTANCE_MAX = 20.0
```

These are purely extra background people walking along the sidewalk near
the ego vehicle — they never enter the road and have nothing to do with
the crossing/occlusion pedestrian(s) above. Set `SPAWN_SIDEWALK_PEDESTRIANS
= True` to turn them on; they get labeled `Pedestrian` the same as any
other walker, no extra config needed for that. Set True and tune the
count range in `scenario_config.py` for busier-looking street scenes.

### 5.4 Occlusion

```python
OCCLUSION_SEARCH_SAMPLES = 25        # how many points along the crossing to test
OCCLUSION_PROGRESS_RANGE = (0.15, 0.85)
OCCLUSION_TOLERANCE_PERCENT = 12.0   # how close to target counts as "close enough" (OCCLUSION_STRICT only)
OCCLUSION_OVERLAP_GRID = 64          # resolution of the overlap-scoring grid
OCCLUSION_MIN_ACHIEVED_PERCENT = 15.0  # hard floor - see below, this is the real guarantee
OCCLUSION_STRICT = False             # extra: discard+retry a miss vs its own target too
```

How it actually works: `occlusion.py` only ever uses vehicles/trucks/
cyclists that are **already spawned** in the scene as occluders — it
never spawns an extra prop just to block the view. For an `*_occluded`
image, it rolls a target from `OCCLUSION_PERCENT_RANGE` (§5.2 — this is
the "0 to 100%" knob), then searches 25 points along the crossing line
for whichever one makes the pedestrian's 2D bounding box overlap the
nearest-to-camera traffic actors by the closest percentage to that
target. It checks camera depth, not just 2D overlap — an "occluder"
that's actually behind the pedestrian doesn't count.

**The guarantee that `*_occluded` actually means occluded** is
`OCCLUSION_MIN_ACHIEVED_PERCENT` — this is checked unconditionally, every
time, independent of everything else below it. If the achieved coverage
comes in under this floor (nothing to occlude with, a bad camera angle,
or just a weak match), the attempt is thrown away and retried with a
fresh roll — never saved. This also covers a case a pure "close to
target" check wouldn't: a low rolled target (say 5%) matched almost
exactly (achieved 4%) would pass a tolerance check but isn't visibly
occluded to the eye — the floor rejects it regardless of how close it
was to its own target.

`OCCLUSION_STRICT` is a separate, optional, stricter check on top: even
after clearing the floor, if you want the achieved amount to also track
its own specific rolled target closely (not just "occluded at all"), set
`OCCLUSION_STRICT = True` and misses outside `OCCLUSION_TOLERANCE_PERCENT`
get discarded too.

### 5.5 Traffic realism & blueprint categories

```python
MIN_SEPARATION_BETWEEN_VEHICLES = 3.5     # no two same-frame vehicles closer than this
MIN_SEPARATION_TRUCK_TO_CYCLIST = 6.0     # trucks kept further from cyclists specifically
CROSSWALK_VEHICLE_EXCLUSION_RADIUS = 2.0  # keeps traffic off the crosswalk stripe itself
```

Vehicles are jittered along their lane's own right-vector (not raw x/y
noise), so they stay parallel to the road instead of clipping sideways on
a curve. Cyclists/scooters/motorcycles try a dedicated `LaneType.Biking`
lane first if the map has one.

On crosswalk buckets (not jaywalk), vehicles/trucks/cyclists are also kept
at least `CROSSWALK_VEHICLE_EXCLUSION_RADIUS` away from the crosswalk's own
crossing line, so a cyclist can no longer end up riding across the same
marked crosswalk at the same moment the pedestrian is - that was happening
because `find_lane_waypoint`'s bike-lane search had no awareness of where
the crosswalk itself was. This doesn't hurt occlusion: an occluder just
needs to overlap the pedestrian's 2D bbox from the camera's angle, which a
vehicle a couple meters off the stripe still does fine.

```python
ALLOWED_CAR_VEHICLES = [...]
ALLOWED_TRUCK_VEHICLES = [...]
ALLOWED_CYCLIST_VEHICLES = [...]        # pedal bikes -> "Cyclist"
ALLOWED_ESCOOTER_VEHICLES = [...]       # -> "E-Scooter + Rider"
ALLOWED_TWO_WHEELER_VEHICLES = [...]    # motorcycles/mopeds -> "Motorcycle"
SEPARATE_TRUCK_CLASS = True             # False -> trucks labeled "Vehicle" instead of their own class
```

The four vehicle-ish lists all feed from the same `cyclists_scooters`
scenario count but get detected and labeled separately. Edit any list
freely — an ID your content pack doesn't have is just skipped at spawn
time, not fatal.

Cyclists/scooters/motorcycles try a dedicated `LaneType.Biking` waypoint
near their rolled spawn point first. CARLA's waypoint lookup snaps to the
*nearest* lane of that type anywhere on the map with no distance cap, so
on a map with no real bike lane nearby it could previously snap one onto
a bike lane meters away that runs alongside (or is tagged over) a
sidewalk — visibly placing a cyclist on the footpath. `find_lane_waypoint`
in `utils.py` now rejects a bike-lane match further than
`max_bike_lane_dist` (4.0m) from the rolled point and falls back to the
ordinary driving lane instead.

### 5.6 Crash / disconnect resilience

```python
RECONNECT_RETRY_DELAY_SECONDS = 10.0
RECONNECT_MAX_ATTEMPTS = 30          # None = retry forever
POST_DESTROY_SETTLE_TICKS = 20
POST_DESTROY_SETTLE_SECONDS = 0.3
```

See §8 for what these actually do and why they exist.

---

## 6. Common recipes

**Only legal crosswalk crossings, no jaywalking at all:**
```bash
python generate_scenarios.py --mode crosswalk_only
```

**Only jaywalking:**
```bash
python generate_scenarios.py --mode jaywalk_only
```

**Only occlusion (both crossing types, always occluded):**
```bash
python generate_scenarios.py --mode occlusion_only
```

**50 images of each type, as three separate batches:**
```bash
python generate_scenarios.py --mode crosswalk_only --count 50
python generate_scenarios.py --mode jaywalk_only --count 50
python generate_scenarios.py --mode occlusion_only --count 50
```
Each writes to its own timestamped folder, so these three never collide.
`--count 50` applies to *every* scenario in `SCENARIOS`, so each command
above actually produces 50 × (number of active scenarios) images, not 50
total — see §9.

**Resume a run that got interrupted (CARLA crash, you stopped it, etc.):**
```bash
python generate_scenarios.py --resume dataset_scenarios/<that run's folder name>
```

**No occlusion at all, ever:**
```python
IMAGE_TYPE_SPLIT = {"crosswalk_clear": 50, "crosswalk_occluded": 0, "jaywalk_clear": 50, "jaywalk_occluded": 0}
```

**Spread one scenario across several maps for variety:**
```python
MAP_ROTATION = ["Town01", "Town02", "Town03", "Town10HD_Opt"]
```

Anything set via `--mode`/`--count` only applies to that one run — it
doesn't modify `scenario_config.py`.

---

## 7. Troubleshooting

- **`IndexError` on `blueprints.filter('my_car_xyz')[0]`** — your CARLA
  content doesn't have the `my_car_xyz` ego vehicle blueprint loaded.
- **"no crosswalks found on \<map\>, skipping this chunk"** — that map has
  no crosswalk geometry CARLA's API can see. Try a different map.
- **A frame fails but the run keeps going** — expected. Every frame is
  wrapped in its own try/except; failures get logged to `run_log.txt`
  inside that run's output folder with the scenario/map/bucket/frame
  context, and generation continues. Check that file first if something
  looks off.
- **The run says "connection to CARLA looks dead, retrying..."** —
  expected if the server crashed or got restarted; see §8. If it keeps
  failing past `RECONNECT_MAX_ATTEMPTS`, the server probably needs a
  manual restart — once it's back up, re-run with `--resume <folder>`.
- **Running the CARLA server on Linux and getting an opaque core dump** —
  the Python-side error that *caused* it (if any) will be in `run_log.txt`
  before the crash; check there first rather than the core dump itself.

---

## 8. Crash resilience — what actually happens on a core dump

A CARLA server crash kills the client connection outright; no amount of
retrying the same call fixes that, and a previous version of this script
would just silently waste all images generated so far. Three separate
things now address this:

1. **Every image's progress is persisted immediately, not batched.**
   Frame numbering comes from what's actually in `images/` on disk, and
   each scenario's remaining image-type queue is written to `plan.json`
   the instant an image is saved (`image_plan.consume_plan_item`) — never
   held only in memory. A crash loses at most the one frame in flight.
2. **A dead connection triggers an active reconnect loop, not a dead
   retry.** If an exception looks like a connection failure (timeout,
   broken pipe, etc.), the script waits `RECONNECT_RETRY_DELAY_SECONDS`
   and tries opening a brand new `carla.Client` connection, repeating up
   to `RECONNECT_MAX_ATTEMPTS` (or forever if set to `None`) — giving you
   time to restart the server by hand if it actually crashed. Once
   reconnected, it re-applies sync settings, re-fetches blueprints,
   re-scans the map, and continues from the exact same plan/frame
   position. If reconnecting is given up on, the script tells you the
   exact `--resume <folder>` command to pick back up once the server's
   running again.
3. **A breathing gap between frames.** Previously, the moment one frame's
   actors were destroyed, the next frame started spawning immediately —
   zero ticks, zero pause. A tight spawn/destroy loop with no settle time
   is a known way to upset CARLA's UE4 side over a long run (spawn
   collisions, instability). `POST_DESTROY_SETTLE_TICKS`/`_SECONDS` now
   add a short pause after every single frame, success or failure.
4. **A specific native-crash fix.** If your symptom is the *script*
   segfaulting while the CARLA server itself stays up and fine (the
   giveaway: no Python traceback at all, just "core dumped," and
   restarting the script immediately works again) — that's not something
   point 2 above catches, since a segfault isn't a catchable Python
   exception. The most likely cause found in this code: the occlusion
   placement search (`occlusion.py`) used to attach and start a
   pedestrian's AI walker controller *first*, then call
   `pedestrian.set_transform(...)` afterward to move it into position.
   Teleporting an actor that already has a live crowd-nav controller
   running is a known way to corrupt CARLA's native-side navigation
   state. This is now fixed: `spawner.py`'s pedestrian spawning is split
   into `spawn_pedestrian_only` (just the actor, no controller) and
   `start_walker_controller` (attaches + starts it), and the occlusion
   path always repositions the pedestrian via `spawn_pedestrian_only` +
   `set_transform` *before* the controller is ever attached. (Superseded
   by point 7 below: the AI controller this ordering was protecting
   against has since been removed entirely, which closes off this whole
   class of crash rather than just one ordering within it.)

5. **Ego vehicle and both cameras are no longer respawned every frame.**
   The RGB and semantic segmentation cameras run at 4K (3840x2160).
   Spawning and destroying a pair of 4K camera sensors (plus the ego
   vehicle they're attached to) on every single frame leaks native-side
   sensor/render resources in CARLA, and is what was causing the client
   to core-dump after roughly 20-25 images regardless of anything else
   in this list — the server stayed up throughout, which is the
   signature of this specific leak. Now the ego vehicle and both
   cameras are spawned exactly once per map chunk and reused for every
   frame in it: each frame just teleports the ego vehicle to its next
   approach point, instead of respawning it. They're only destroyed
   when the chunk ends, the map switches, or a reconnect happens.
6. **Each frame's camera capture is its own short `listen()`/`stop()`
   window, not left running for the whole chunk.** An earlier version of
   this fix had both cameras call `listen()` once per chunk and stay
   subscribed across every frame in it. That's unsafe: synchronous mode
   only guarantees a sensor callback has finished by the time
   `world.tick()` returns for *that* tick — it says nothing about a
   `destroy()` issued much later (chunk end, map switch, reconnect)
   never racing a callback still in flight on CARLA's native streaming
   thread. That race throws a C++ exception on a thread Python's
   `try/except` can't reach, which is an unconditional process abort —
   `terminate called after throwing an instance of 'std::runtime_error'
   ... trying to operate on destroyed actor ... Aborted (core dumped)`.
   Each frame now calls `listen()` right before its capture ticks and
   `stop()` right after, the same tightly-scoped pattern the original
   per-frame-respawn code used — just without respawning the actors
   themselves.
7. **Pedestrians no longer use an AI walker controller at all.** Every
   version of the original reference script drives a pedestrian with a
   manual `carla.WalkerControl()` (direction + speed) applied directly,
   never a `controller.ai.walker` actor - that blueprint gets looked up
   but is never actually spawned anywhere in the reference code. An
   earlier version of this project added the AI controller for a more
   natural walk cycle, but that ties the pedestrian into CARLA's
   crowd-nav manager, and destroying an actor the manager still
   considers "active" is an uncatchable native crash (`terminate called
   after throwing an instance of 'std::runtime_error' ... trying to
   operate on destroyed actor ... Aborted (core dumped)`). Two attempts
   to fix this by inserting settle ticks between stopping and destroying
   a controller didn't hold up - the crash kept coming back - because the
   real fix was removing the AI controller from the picture entirely,
   not timing its teardown more carefully. `start_walker_controller` in
   `spawner.py` now only ever applies manual `WalkerControl` and returns
   `None`; no controller actor is ever created, so there's nothing for
   the crowd-nav manager to still consider active when destroy time
   comes.
8. **Ego vehicle + cameras are refreshed periodically within a chunk,
   not kept for its entire length.** `EGO_REFRESH_EVERY_N_IMAGES` (200)
   respawns the ego vehicle and both cameras every N images instead of
   reusing the exact same sensor pair for a whole chunk, which could be
   thousands of frames. The replacement is spawned *before* the current
   one is destroyed, so a failed respawn attempt just skips that round
   instead of leaving the chunk with no valid ego.

### Pedestrian bounding boxes drifting from the actual image

Separate from the crash fixes above: each frame's capture ticks 10 times
after `listen()` starts, which queues up 10 images (one per tick) -
`image_queue.get()` was pulling the *first* (oldest) of those 10, while
labels were then computed from actors' *current* transforms (after all 10
ticks, plus whatever ticks ran after). A walking pedestrian moves during
that gap, so their box ended up drawn where they'd walked to, not where
they were in the saved image. Vehicles don't move (physics frozen, no
controller), so this never showed up on them - only ever on people. Fixed
by draining that backlog and taking one fresh tick right before grabbing
the image, so the frame and the actor transforms used for labels come from
the exact same tick.

None of this can make Python relaunch a CarlaUE4 process that's actually
dead — something (you, or a supervisor/systemd unit) still has to restart
the server binary itself. What this guarantees is that once it's back up,
`--resume` picks up exactly where things stopped, so **"60 out of 100"
never means those 60 were wasted** — they're just already-done work that
doesn't get redone.

---

## 9. Known limitations

- **Snow is commented out**, and when re-enabled it still isn't real
  snow — CARLA's base town content has no snow ground/particle assets,
  so it's a cold/fog/wetness approximation via `WeatherParameters`. Swap
  in a snow-enabled map/weather preset if you have access to one.
- **Occlusion is 2D-bbox-based, not pixel-perfect.** It's a solid proxy
  (and depth-checked, so it won't place an occluder that's actually
  behind the pedestrian), but a car's rectangular bounding box can stick
  out slightly past its visible silhouette, so "achieved occlusion %" is
  an estimate, not an exact pixel count.
- **`--count` scales with the scenario table, not a flat total.**
  `--count 50` means 50 images *per active scenario*, not 50 images
  total. If you want an exact total regardless of weather, temporarily
  trim `SCENARIOS` down to one entry, or ask and a `--count-is-total`
  style flag can be added.
- **This hasn't been run against a live CARLA server from the environment
  this was developed in** — it's been checked for internal consistency
  (every cross-module import resolves, every function signature matches
  its call sites, the exact-quota math was tested in isolation) but the
  first real end-to-end test, including whatever CARLA-side instability
  originally prompted the crash-resilience work in §8, is on you.
