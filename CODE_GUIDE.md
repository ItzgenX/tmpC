# Code Guide

This guide explains how the generator works inside, file by file and function by
function, so you can change it yourself. For *using* the generator (commands,
settings, outputs) see [README.md](README.md).

Contents:

1. [The big picture](#1-the-big-picture)
2. [Words used in the code](#2-words-used-in-the-code)
3. [Coordinates and units](#3-coordinates-and-units)
4. [Settings: config.py and advanced.py](#4-settings-configpy-and-advancedpy)
5. [Data passed between files](#5-data-passed-between-files)
6. [Every file and function](#6-every-file-and-function)
7. [How to change common things](#7-how-to-change-common-things)
8. [Rules to keep when you change code](#8-rules-to-keep-when-you-change-code)
9. [Finding problems](#9-finding-problems)

---

## 1. The big picture

```
python main.py
   |
   |  main.py ............ reads the command, starts the supervisor
   v
engine/supervisor.py .... starts the generator as a CHILD process, restarts it if it
   |                       crashes, passes Ctrl+C on, runs a final cleanup
   v
python main.py --worker   (the child)
   |
   v
engine/run.py  Generator.run()
   |  for each map in MAPS:
   |     engine/carla_io.py  MapSession.open()   load map, find crossing sites
   |                                             (roads.py), spawn ego + 3 cameras
   |     run.py  build_jobs()                    the ordered job list:
   |                                             weather blocks -> scene types
   |     for each job not done yet:
   |        run.py  run_job()
   |           |-- engine/planner.py  ScenePlanner.plan()   the whole scene as DATA
   |           |      uses camera.py, people.py, traffic.py, occlusion.py, roads.py
   |           |-- engine/render.py   render_job()          spawn the plan in CARLA,
   |           |                                            walk, capture frames
   |           |      uses carla_io.py (spawn / remove), labels.py (measure, check)
   |           |-- engine/labels.py   write_frame()         save files of each frame
   |           |-- run.py  Progress                         jobs_done.txt, flagged.csv
   |  write_summary()   manifest.csv, summary.txt, train/val/test lists
```

**The key idea:** every scene is first decided completely as plain Python data (a
*plan*): every position, person, vehicle, speed and the frame times. It is checked
(no overlaps, crossing in view, ...) before anything is spawned in CARLA. Only a good
plan is rendered. The plan is saved inside each image's JSON, so every image can be
explained and reproduced.

**Why two processes?** A CARLA crash can kill Python itself (core dump), and no Python
code can catch that. So the real work runs in a child process. The parent (supervisor)
only watches it and starts it again, which continues from `jobs_done.txt`.

---

## 2. Words used in the code

| Word | Meaning |
|---|---|
| **job** | one planned scene. Makes 1 image, or several (a *sequence*) for occluded scenes |
| **frame** | one saved image of a job, with all its outputs |
| **scene type** | `crosswalk`, `jaywalk_road`, `jaywalk_junction`, `occluded_by_traffic`, `occluded_by_parked` |
| **site** | a place where people cross: a straight line from one curb to the other (`CrossingSite`) |
| **site kind** | `crosswalk` (marked), `road` (mid-block), `junction` (T-junction arm) |
| **ego** | our own car; the cameras are attached to it |
| **target pedestrian** | the crossing person whose occlusion is controlled and checked |
| **occluder** | a car/truck placed to hide the target |
| **plan** | the dict describing a whole job (section 5) |
| **tick** | one simulation step = `FIXED_DELTA_SECONDS` = 0.05 s |
| **sync mode** | CARLA only moves forward when we call `world.tick()`; set during a run, switched off at the end |
| **spot use** | how many jobs a crossing site has had in one weather; capped by `MAX_USES_PER_SPOT` |
| **new view** | at the same site and weather, the same lane needs `MIN_VIEW_DISTANCE_GAP` m more / less distance |
| **flagged** | an image that failed a quality check: saved anyway and listed in `flagged.csv` |
| **skipped** | a job that never produced an image (listed in `skipped.csv`) |
| **ctx** | `SceneContext`: everything the planning helpers need about the scene being built |
| **long / lat** | distance along the road / across it, in the road frame (section 3) |

---

## 3. Coordinates and units

- **Distances** are metres, **speeds** m/s, **angles** degrees, **time** seconds or ticks.
- **CARLA world** is left-handed: x forward, y right, z up. A heading (`yaw`) points to
  `(cos yaw, sin yaw)`; its right side is `(-sin yaw, cos yaw)` (`roads.forward_right`).
- **Road frame** (`camera.Frame`), used by the planner: origin at the crossing centre,
  `long` = along the road in the ego's driving direction (positive = past the crossing,
  the ego is at about `-distance`), `lat` = to the right of that direction.
- **Lanes** have an id: one sign drives one way, the other sign the other way.
  `road_cross_section` sorts lanes into `same` (as the site's waypoint) and `opposite`.
- **Camera (planning, `PlanCamera`) and UE camera coordinates**: x forward, y right, z up.
- **Camera coordinates in the JSON** (`location_camera`): KITTI style, x right, y down, z forward.
- **Pixels**: origin top-left, x to the right, y down, image 1920 x 1080.

---

## 4. Settings: config.py and advanced.py

`config.py` is imported by `advanced.py` (`from config import *`), and **all code reads
settings through `advanced`**, written `C.SETTING`. So a setting can be moved between
the two files without touching the code.

### config.py (everyday)

| Setting | Used in | Meaning |
|---|---|---|
| `OUTPUT_DIR` | main.py | dataset folder (also the resume folder) |
| `MAPS` | run.resolve_maps | maps to use, by short name or full path (custom maps too); images are split evenly across them |
| `MAX_USES_PER_SPOT` | planner._pick_site, run.spot_shortages | how often one crossing spot may be used per weather; None = no limit |
| `SEED` | run.py, carla_io.py | random seed; same seed + settings = same plans |
| `CARLA_HOST`, `CARLA_PORT` | everywhere a client is made | server address |
| `CARLA_EXECUTABLE` | supervisor.py | optional path to start / restart the server |
| `WEATHERS` | run.py, planner.py | per weather: `enabled`, `images`, `pedestrians`, `vehicles`, `cyclists`, `trucks`, `occlusion`. A number is exact, `(min, max)` is random |
| `SCENES` | run.py | scene types with `enabled` and `share` |
| `SEQUENCE` | advanced.frames_per_job, occlusion.py | `enabled`, `frames`, `walk_speed` of the occluded walk |
| `PEDESTRIAN_MIX` | people.pick_type | relative mix of child / woman / man |
| `SIDEWALK_PEDESTRIANS` | planner.py | extra non-crossing people (number or range) |
| `SAME_DIRECTION_CHANCE` | traffic.place_traffic | chance a vehicle drives the ego's way |
| `SCENE_FOLLOWS_WEATHER` | run.Generator._params | False = counts/occlusion come from a random weather row |
| `EGO_BLUEPRINT`, `CAR_BLUEPRINTS`, `TRUCK_BLUEPRINTS`, `CYCLIST_BLUEPRINTS` | carla_io.py | vehicle models |
| `SAVE` | labels.write_frame | which outputs to write (the depth camera always runs: labels need it) |
| `PED_YOLO_BOX`, `LABEL_VEHICLES`, `LABEL_FULLY_HIDDEN_PEDESTRIANS`, `SPLIT_CROSSING_CLASS` | labels.py | what goes into the YOLO files |
| `IMAGE_EXT` | labels.write_frame | `.png` or `.jpg` for RGB |
| `SAVE_IN_FOLDERS` | run.build_jobs | True = `<weather>/<scene>/` sub-folders, False = all files in one folder per output type |
| `SKIP_FLAGGED_IMAGES` | run.py | True = images that fail a check are not saved; False = saved and listed in flagged.csv |
| `TRAIN_VAL_TEST` | run.write_summary | split ratios, or None |

### advanced.py (fine tuning), by group

| Group | Settings | Used in |
|---|---|---|
| server / timing | `CARLA_ARGS`, `CARLA_STARTUP_SECONDS`, `CARLA_TIMEOUT`, `MAP_LOAD_TIMEOUT`, `FIXED_DELTA_SECONDS`, `SERVER_WAIT_SECONDS` | supervisor, carla_io, run |
| weather look | `SUN_ALTITUDE_RANGE`, `SUN_AZIMUTH_RANGE`, `WEATHER_LOOK`, `WEATHER_RESAMPLE_EVERY` | carla_io.MapSession.set_weather, run._weather |
| pedestrians | `PED_SPEED_RANGE`, `PED_RUNNER_PROBABILITY`, `PED_RUN_SPEED_RANGE`, `PED_PROGRESS_RANGE`, `PED_PREROLL_SECONDS`, `PED_MIN_GAP`, `PED_HEIGHT`, `SIDEWALK_PED_DISTANCE`, `SIDEWALK_PED_STANDING_PROBABILITY`, `PED_TYPE_OVERRIDES`, `PED_BLUEPRINT_EXCLUDE` | people.py, carla_io.py |
| scene types | `OCCLUDED_SCENES`, `SCENE_SITE_KINDS` | run.py, planner.py |
| occlusion | `OCCLUSION_TOLERANCE`, `OCCLUSION_NUMBER_SPREAD`, `LOWEST_REACHABLE_OCCLUSION`, `CAR_BODY_HEIGHT`, `CAR_CABIN_LENGTH_PART`, `OCCLUDER_TRUCK_CHANCE`, `OCCLUDED_EGO_DISTANCE_RANGE`, `ADJACENT_CARS_AHEAD`, `PARKED_NOSE_GAP`, `FULLY_VISIBLE`, `PROTECT_TARGET_MAX_COVER` | occlusion.py, traffic.py, labels.frame_flags |
| crossing sites | `JAYWALK_MIN_DIST_FROM_CROSSWALK`, `JAYWALK_SITE_SPACING`, `JAYWALK_ROAD_WIDTH_RANGE`, `JUNCTION_ARM_OFFSET`, `JUNCTION_MIN_DIST_FROM_CROSSWALK` | roads.py |
| ego / camera | `EGO_FALLBACK_BLUEPRINT`, `EGO_DISTANCE_RANGE`, `MIN_VIEW_DISTANCE_GAP`, `CAMERA_LIFT_RANGE`, `CAMERA_PITCH_JITTER_DEG`, `EGO_YAW_JITTER_DEG`, `EGO_LATERAL_JITTER`, `MAX_VIEW_BEARING_DEG`, `EGO_SIZE`, `IMAGE_W`, `IMAGE_H`, `RGB_SUPERSAMPLE`, `CAMERA_FOV`, `CAMERA_LOCATION`, `CAMERA_PITCH`, `RGB_ATTRIBUTES`, `CAPTURE_SETTLE_TICKS`, `JPEG_QUALITY` | planner, camera, carla_io, labels |
| drawing / folders | `BOX_COLOR`, `BOX_THICKNESS`, `OUTPUT_FOLDERS` | labels.py, tools.py |
| traffic | `EGO_LANE_WIDTH`, `ONCOMING_MAX_DRIFT`, `CAR_SIZE`, `TRUCK_SIZE`, `CYCLIST_SIZE`, `TRAFFIC_FAR_MAX`, `NEAR_SIDE_PROBABILITY`, `EGO_CLEARANCE`, `VEHICLE_GAP`, `CROSSING_CLEARANCE` | traffic.py |
| labels / checks | `TAG_CLASS`, `TAG_NAMES`, `RIDER_TAG`, `DEPTH_MATCH_MARGIN`, `MIN_VISIBLE_PIXELS`, `MIN_BOX_SIZE_PX`, `MAX_LABEL_DISTANCE`, `LABEL_PARKED_VEHICLES`, `PARKED_VEHICLE_CLASS`, `MIN_TARGET_PIXELS`, `MIN_ROAD_FRACTION`, `SEG_TAGS` | labels.py, render.py, carla_io.py |
| run control | `MAX_RENDERS_PER_JOB`, `MAX_PLANS_PER_JOB`, `RELOAD_WORLD_EVERY`, `MAX_JOBS_PER_PROCESS`, `MAX_CRASHES_WITHOUT_PROGRESS`, `EXIT_*`, `STOP_WAIT_SECONDS` | run.py, supervisor.py |

`SEQUENCE_MAX_SECONDS` is defined but **not used anywhere** (changing it has no effect).

`COUNT_KEYS` = `("pedestrians", "vehicles", "cyclists", "trucks")`, the count keys of a weather row.

Exit codes the generator (child) returns to the supervisor:

| Code | Name | Supervisor does |
|---|---|---|
| 0 | `EXIT_DONE` | stops: dataset complete |
| 2 | `EXIT_FATAL` | stops: configuration error (nothing to retry) |
| 3 | `EXIT_NO_SERVER` | waits for / restarts the server, then starts the generator again |
| 10 | `EXIT_RECYCLE` | starts a fresh generator at once (after `MAX_JOBS_PER_PROCESS` jobs) |
| 130 | `EXIT_STOPPED` | stops: Ctrl+C / Ctrl+Z / kill, after the final cleanup |
| anything else | (a crash) | starts it again; gives up after `MAX_CRASHES_WITHOUT_PROGRESS` crashes in a row with no new job |

### Helper functions in advanced.py

| Function | What it does |
|---|---|
| `roll(value, rng)` | config count -> number: an int stays, a `(min, max)` pair becomes a random int |
| `occlusion_range(value)` | config occlusion -> (min, max) fractions; a single number v means v +- `OCCLUSION_NUMBER_SPREAD` |
| `max_of(value)` | largest value a count can take (used by checks) |
| `enabled_weathers()` | weather names that are enabled and have images > 0, in config order |
| `enabled_scenes()` | scene types that are enabled with share > 0 |
| `frames_per_job(scene)` | `SEQUENCE.frames` for occluded scenes (if enabled), else 1 |
| `can_occlude(row)` | True if a weather row has occlusion > 0 and at least one car or truck |
| `class_names()` | YOLO class list (adds `crossing_pedestrian` if `SPLIT_CROSSING_CLASS`) |
| `validate()` | checks config for mistakes; raises `ValueError` before anything runs |

---

## 5. Data passed between files

All of these are plain dicts / lists, so they can be saved to JSON.

**job** (made by `run.build_jobs`)

```
{"name": "rain/occluded_by_parked/rain_occluded_by_parked_00012", "map": "Town10HD_Opt",
 "weather": "rain", "scene": "occluded_by_parked", "frames": 5, "block_index": 37}
```
`block_index` = place of the job inside its weather block (for weather re-sampling).
The name is also where the job's files go: `images/rain/occluded_by_parked/rain_occluded_by_parked_00012_00.png`.
The file name alone is unique; with `SAVE_IN_FOLDERS = False` the folders are dropped
(`"rain_occluded_by_parked_00012"`). With more than one map the map comes first in both.

**plan** (made by `planner.ScenePlanner.plan`)

| Key | Content |
|---|---|
| `job`, `scene`, `weather` | from the job |
| `signature` | site, lane, distance, weather, scene of this view (for reference) |
| `counts` | the rolled pedestrians / vehicles / cyclists / trucks |
| `occlusion_range`, `occlusion_peak` | the weather's (min, max) range, and the planned most-hidden share of the target's walk |
| `site` | `CrossingSite.to_json()`: id, kind, start, end, length, half_width |
| `ego` | x, y, z, yaw, pitch, roll, lane, distance, approach |
| `target_pedestrian` | index into `pedestrians` |
| `frames` | list of `{"tick": t, "planned_occlusion": p}`; ticks counted from when people start walking |
| `pedestrians` | list of pedestrian dicts (below) |
| `vehicles` | list of vehicle dicts (below) |
| `seed_key` | added by run.py: the exact random seed of this plan |

**pedestrian** (made by `people.make_ped`): `role` (`crossing`/`sidewalk`), `type`
(child/woman/man), `blueprint`, `speed`, `runner`, `spawn` [x, y, z], `ground_z`, `yaw`,
`direction` [dx, dy], `capture_xy` (position at the first frame), and for crossing
people `capture_progress`, `lateral_offset`.

**vehicle** (made by `traffic.vehicle_dict`): `blueprint`, `category` (car/truck/cyclist),
`role` (traffic/occluder), `color`, `x`, `y`, `z`, `yaw`, `pitch`, `direction`
(same/oncoming), `long`, `lane`, and `parked: True` for a parked occluder.

**frame** (made by `render._finish_frame`, written by `labels.write_frame`): `rgb`,
`seg` (class ids), `seg_color`, `depth_m` (metres), `labels`, `flags`, and `info` (goes
into the JSON: map, job, scene, weather, planned / measured occlusion, flags, image
size, classes, camera K and world_to_camera, the plan).

**label** (made by `labels.compute_labels`): `kind` (pedestrian/vehicle/truck/cyclist, from
CARLA's tags), `carla_class` (CARLA's own name: Pedestrian, Car, Truck, Bus, Motorcycle, Bicycle),
`source` (spawned/parked), `meta`, `visible`, `bbox_visible`, `bbox_full`
(pedestrians), `bbox_projected`, `visible_pixels`, `full_pixels`, `occlusion`,
`truncated`, `depth`, `corners_2d`, `location_world`, `rotation_world`, `dimensions`,
`location_camera`. Boxes are `[x0, y0, x1, y1]` in pixels.

---

## 6. Every file and function

Functions starting with `_` are internal helpers of their file.

### main.py, the commands

| Function | What it does |
|---|---|
| `main()` | reads the command line and calls the right part: `--check`, `--delete-flagged`, `--summary`, `--cleanup`, the worker, or (no option) the supervisor |
| `run_worker(args)` | runs `run.Generator`; turns Ctrl+C, a lost connection, or a server time-out into the exit codes the supervisor understands |

### engine/stop.py, clean stop

| Function | What it does |
|---|---|
| `EXTRA_STOP_SIGNALS` | Ctrl+Z (SIGTSTP), kill (SIGTERM), closed terminal (SIGHUP) |
| `install_stop_handlers()` | makes those signals behave like Ctrl+C |
| `_stop(signum, frame)` | the handler: raises `KeyboardInterrupt` |
| `protected()` | `with protected():` ignores every stop key until the block ends, so cleanup can't be cut short |
| `release_server(client)` | removes all vehicles, people and sensors, then switches CARLA back to async mode |

### engine/supervisor.py, auto-restart

| Function | What it does |
|---|---|
| `MAIN_SCRIPT` | path of main.py, which is started as the child |
| `server_reachable()` | True if the CARLA port answers |
| `ServerKeeper.restart()` | (re)starts CARLA from `CARLA_EXECUTABLE`, in its own session so Ctrl+C can't kill it |
| `ServerKeeper.wait()` | waits until the server answers (starting it if possible); False after `SERVER_WAIT_SECONDS` |
| `count_done(out)` | number of lines in `jobs_done.txt` (to see if a crashed run made progress) |
| `run_child(args)` | starts `main.py <args>` in its own session and waits; on a stop key it sends exactly one SIGINT, waits up to `STOP_WAIT_SECONDS`, kills only if needed |
| `final_cleanup()` | runs `main.py --cleanup` once at the end of a stop |
| `supervise(out)` | entry point; returns the "stopped" exit code on a stop key while no child is running |
| `_supervise(out)` | the loop: start the child, read its exit code (done / config error / recycle / no server / crash), restart or stop. Gives up after `MAX_CRASHES_WITHOUT_PROGRESS` crashes with no new job |

### engine/tools.py, helper commands

| Function | What it does |
|---|---|
| `check_server()` | `--check`: ego model, child/woman/man sorting of every pedestrian model, which configured vehicle ids exist, all vehicles, installed maps |
| `FLAG_FIELDS` | the columns of `flagged.csv` / `deleted.csv` (also used by run.Progress) |
| `_files_of(out, stem)` | every output file of one image, in every output folder |
| `delete_flagged(out, reason, yes)` | `--delete-flagged`: lists flagged images; with `yes` deletes all their files and moves the rows to `deleted.csv` |
| `cleanup_only()` | `--cleanup`: connects and calls `stop.release_server` |

### engine/roads.py, map geometry

Runs on the map's OpenDRIVE data on our side, no server calls.

| Function | What it does |
|---|---|
| `MAX_LANE_STEPS` | safety limit when walking across lanes (12) |
| `forward_right(yaw_deg)` | unit forward and right vectors of a heading |
| `heading_diff(a, b)` | smallest angle between two headings (0..180) |
| `dot(a, b)`, `sub(a, b)` | 2D vector helpers |
| `xy(loc)` | (x, y) of a location. **Not used anywhere** |
| `step_along(wp, dist, forward)` | waypoint `dist` m ahead / behind on the same lane; at a junction takes the straightest branch |
| `same_side(a, b)` | True if two lanes drive the same way (same lane-id sign) |
| `_walk_outward(start)` | from a lane outward with `get_right_lane()`: the driving lanes and the first sidewalk |
| `road_cross_section(wp)` | all driving lanes on both sides of the road at `wp` and the sidewalk on each side. Handles the CARLA quirk that "left" flips direction after the centre line |
| `lane_key(wp)` | "road:section:lane" id of a lane |
| `ground_z(carla_map, x, y, z_hint)` | road / sidewalk height at a point |
| `CrossingSite` | one crossing place: start, end, centre, length, axis (start->end), along (road direction), half_width, road_wp. `to_json()` for saving |
| `find_all_sites(carla_map, cfg)` | runs the three finders below, prints the counts, returns all sites |
| `parse_crosswalk_polygons(carla_map)` | splits CARLA's flat crosswalk point list into polygons |
| `_principal_axis(pts)` | long axis of a polygon and how elongated it is |
| `_find_road_wp(carla_map, center, major, anisotropy)` | driving lane under a crosswalk whose road runs across it; prefers non-junction lanes |
| `find_crosswalk_sites(carla_map)` | marked crosswalk sites (and how many polygons could not be used) |
| `_is_straight(wp, dist, max_turn)` | road is straight and junction-free `dist` m both ways |
| `_curb_to_curb(wp, width_range)` | sidewalk centre on each side of the road, if the road width fits |
| `_far_from(x, y, centers, dist)` | point is at least `dist` from all given centres |
| `find_road_sites(...)` | mid-block jaywalk sites: straight roads with sidewalks both sides, far from crosswalks, spaced apart |
| `find_junction_sites(...)` | one site across each arm of every 3-way junction that has no crosswalk |

### engine/camera.py, planning-time camera

| Function | What it does |
|---|---|
| `Frame(origin, yaw)`, `.local(x, y)` | road frame; turns world (x, y) into (long, lat) |
| `rects_overlap(a, b, gap_long, gap_lat)` | do two (long, lat, length, width) rectangles overlap, with gaps |
| `PlanCamera(ego, z_offset)` | pinhole model of the dashcam at a planned ego pose |
| `PlanCamera.box(corners)` | 2D box + mean depth of 3D corners, None if behind the camera |
| `box_corners(x, y, z, yaw, l, w, h)` | 8 corners of a box standing on the ground |
| `in_view(box)` | box is at least partly inside the image |
| `coverage(target, occluders, samples)` | share of the target's box hidden by nearer occluder boxes (estimated on a grid) |

### engine/people.py, pedestrian planning

| Function | What it does |
|---|---|
| `weighted_choice(rng, weights)` | random key by relative weights |
| `pick_type(ctx)` | child / woman / man by `PEDESTRIAN_MIX`, only types with models on the server |
| `track(x, y, d, speed, seconds, step)` | positions every `step` s of a straight walk |
| `walk_is_clear(ctx, path, tracks)` | walk never comes within `PED_MIN_GAP` of another walk and never hits a vehicle |
| `ped_box(ctx, ped, x, y)` | planning-camera box of a pedestrian standing at (x, y) |
| `position_at(ped, seconds)` | where a planned walker is after `seconds` |
| `make_ped(ctx, role, ...)` | builds a pedestrian dict (picks the model, ground heights, yaw) |
| `crossing_line(site, direction, lateral)` | start point and direction of a walk across the site, shifted along the road |
| `plan_crossing(ctx, n, first_frame_s, total_s, tracks)` | `n` crossing people, each at a random spot on the crossing at the first frame; None if they can't all fit |
| `plan_sidewalk(ctx, n, ...)` | people walking along or standing on the sidewalk (not crossing) |

### engine/traffic.py, vehicle placement

| Function | What it does |
|---|---|
| `SIZE` | planning length / width / height per category |
| `vehicle_at(ctx, lane, s, drift)` | pose `s` m along the road frame on `lane`, shifted `drift` m to the lane's right |
| `vehicle_dict(ctx, category, wp, x, y, direction, role)` | builds a vehicle dict (picks model and colour) |
| `boxes_of(ctx, v)`, `rect_of(ctx, v)` | planning-camera silhouette boxes (a car = low body + middle cabin; truck / cyclist = one block) / road-frame rectangle of a vehicle |
| `max_drift(direction, lane, width)` | how far a vehicle may sit off its lane centre (2.4 m ego-lane rule for same-direction traffic) |
| `_direction_of`, `_lanes_by_direction` | lanes for "same" / "oncoming" traffic |
| `place_traffic(ctx, category, protect_boxes)` | one normal car / truck / cyclist: before or past the crossing, never on it, never overlapping, never hiding the target |
| `adjacent_occluders(ctx, categories, ped_long, rects)` | 1-2 cars in the lane next to the ego, before the pedestrian's path |
| `parked_occluder(ctx, category, ped_long, rects)` | a car pulled over at the curb, its front just before the pedestrian's path |

### engine/occlusion.py, occluded walk

| Function | What it does |
|---|---|
| `STEP` | 0.1 m between predicted positions |
| `EXTEND`, `CLEAR_VIEW` | the walk may go 3 m onto each sidewalk; frames start / end 0.8 m into the visible part |
| `_occluder_categories(ctx, scene, cars, trucks)` | whether the occluder(s) are cars or a truck (from the job's own counts) |
| `_walk(ctx, direction, lateral)` | positions every 10 cm along the walk, from one sidewalk to the other |
| `_arcs(curve, lo, hi)` | stretches of the walk "visible -> hidden -> visible" whose most hidden point is in [lo, hi] |
| `_frame_indices(start, peak, end, n)` | `n` frames spread over that stretch, one exactly at the most hidden point |
| `plan_occluded(ctx, scene, occ_range, cars, trucks, n_frames)` | tries up to 40 layouts (occluders + a walk across the road, random direction); keeps a walk that goes visible -> hidden -> visible with its most hidden moment closest to a random goal inside `occ_range`; returns target pedestrian, occluders, frame ticks, occupied space, planned peak |

### engine/planner.py, whole scene plan

| Function | What it does |
|---|---|
| `PlanError` | raised when a plan fails a check; the job loop just tries the next seed |
| `CATEGORY_KEY` | car -> vehicles, truck -> trucks, cyclist -> cyclists |
| `SceneContext` | bag of everything the helpers need (rng, site, ego, lanes, frame, camera, occupied rectangles, models) |
| `ScenePlanner(...)` | made once per map with its sites, parked map cars, pedestrian and vehicle models |
| `.register(plan)` | counts a saved plan: spot use per weather, and its view (lane, distance) |
| `._view_is_new(site, weather, ego)` | False if the same lane at this spot (same weather) was used within `MIN_VIEW_DISTANCE_GAP` m |
| `.sites_of(kinds)`, `._pick_site(rng, kinds, weather)` | sites of given kinds; picks a random one among the least used in this weather, never over `MAX_USES_PER_SPOT` |
| `.cross_section(site)` | cached `road_cross_section` of a site |
| `._env_rects(frame)` | the map's parked cars as road-frame rectangles |
| `.plan_ego(rng, site, distance_range)` | ego pose: lane, distance, heading jitter, camera lift + tilt (the ego is raised / tilted; the camera is fixed to it); checks the crossing is ahead and in view |
| `.plan(rng, job, params)` | the full plan in order: counts -> site -> ego -> new-view check -> target (+ occluders) -> other people -> traffic -> sidewalk people |

### engine/carla_io.py, talking to CARLA

Spawner part:

| Function | What it does |
|---|---|
| `SpawnError` | an actor could not be spawned (collision at the spot) |
| `_attr(bp, name)`, `_set_attrs(bp, attrs)` | read one blueprint attribute (lower case, "" if missing); set several, skipping ones the blueprint does not have |
| `walker_type(bp)` | child / woman / man of a pedestrian model (from its age / gender, or `PED_TYPE_OVERRIDES`) |
| `walker_pools(bp_lib)` | all pedestrian models grouped by type |
| `vehicle_pool(bp_lib, ids)` | configured vehicle ids that exist, with their colours |
| `ego_blueprint(bp_lib)` | the ego model (or the fallback) |
| `camera_blueprints(bp_lib)` | RGB (2x size), segmentation and depth camera settings |
| `cleanup_world(client, world, why)` | removes every vehicle, walker, sensor and controller (start-up and stop) |
| `destroy_actors(client, world, actors)` | removes a list of actors and ticks |
| `to_transform(d, z_offset)` | plan dict -> `carla.Transform` |
| `spawn_ego_and_cameras(world, bp_lib, ego_pose)` | ego (named "hero") + RGB, segmentation, depth cameras; returns the ego's height offset |
| `spawn_vehicles(world, bp_lib, plan_vehicles, actors)` | spawns planned vehicles, handbrake on; adds each to `actors` at once |
| `freeze`, `hide`, `restore` | stop physics; move vehicles 200 m under ground; put them back |
| `spawn_pedestrians(world, bp_lib, plan_peds, actors)` | spawns planned people (no wheelchair) |
| `start_walking(walkers, plan_peds)` | gives each person its direction and speed (no AI controller) |

Session part, `MapSession` (one per map):

| Method | What it does |
|---|---|
| `open(reload)` | loads the map, cleans it, finds sites, reads parked map cars, builds the planner, spawns the ego |
| `_load(reload)` | switches map (or reloads it) and turns sync mode on |
| `_set_sync(on)` | sync mode on / off |
| `_spawn_ego()` | finds 3 ego poses, spawns ego + cameras, measures the car-bonnet mask from those poses |
| `close(why)` | removes everything and switches back to async mode, even if Ctrl+C is pressed again |
| `reload()` | `close` + full world reload (clears server-side leftovers) |
| `set_weather(name, key, rng)` | applies a new look of a weather only when `key` changes |
| `cameras()` | the cameras in capture order (RGB, segmentation, depth) |
| `capture(cams, settle_ticks)` | listen, tick, return the images of exactly the last tick |
| `_frame_from(q, frame, timeout)` | waits for the image of a given tick in a camera's queue |

### engine/render.py, running one plan

| Function | What it does |
|---|---|
| `KIND` | plan category -> label kind |
| `render_job(session, plan)` | moves the ego, spawns and freezes traffic, spawns people and starts them walking, captures every planned frame (plus the "traffic hidden" segmentation), always removes what it spawned |
| `_finish_frame(...)` | from raw images to a frame dict: labels, flags, RGB downscale, colour segmentation, depth, JSON info |
| `scene_objects(session, plan, walkers, vehicles, cam_tf)` | everything that may get a label, with its 3D box and meta (crossing progress, walking direction, ...) |

### engine/labels.py, labels, checks, saving

| Function | What it does |
|---|---|
| `intrinsics(width, height, fov)` | camera matrix K |
| `project(points, world_to_cam, K)` | 3D world points -> pixels and depth |
| `cam_coords(cam_xyz)` | UE camera axes -> KITTI camera axes |
| `seg_tags(img)` | class id per pixel from a segmentation image (a copy) |
| `decode_depth(img)` | depth image -> metres |
| `_group(kind)` | pedestrian pixels or vehicle-like pixels (car, truck, bus, motorcycle, bicycle, rider) |
| `_search_box(...)` | an object's projected box, slightly grown |
| `_assign_pixels(objs, masks, depth, w, h)` | nearest object first takes the pixels in its box whose tag fits **and** whose depth matches its own distance |
| `_carla_class(seg, pix)` | most common CARLA tag on an object's pixels -> (our class, CARLA name); a rider counts with its bicycle / motorcycle |
| `_big_enough(box, pixels, kind)` | enough pixels and big enough box to label |
| `compute_labels(objects, seg, seg_full, hood, world_to_cam, K, depth, depth_full)` | all labels of a frame; `seg_full` (traffic hidden) gives full-body boxes and real occlusion |
| `road_fraction(seg, hood)` | share of road-like pixels (camera-blocked check) |
| `hood_mask(seg_frames)` | the ego's bonnet pixels at the bottom of the image |
| `target_label(labels)` | the target pedestrian's label |
| `frame_flags(labels, seg, hood, planned_occlusion)` | reasons a frame looks wrong (empty = fine) |
| `FOLDERS`, `make_folders(out)` | output folders of the outputs switched on |
| `class_id(label)`, `yolo_box(label)`, `yolo_lines(labels)` | YOLO class, box and lines |
| `EDGES`, `draw_2d(img, labels)`, `draw_3d(img, labels)` | box images, one colour, no text |
| `write_image`, `write_text_atomic`, `_json_default` | file helpers (text written to `.tmp` then renamed) |
| `write_frame(out, name, frame)` | writes every switched-on output of a frame; JSON last |

### engine/run.py, the run

Progress part:

| Function | What it does |
|---|---|
| `Progress(out)` | reads `jobs_done.txt` |
| `.mark_done(job_name)` | appends a finished job (flushed to disk at once) |
| `.flag(...)`, `.skip(...)` | rows in `flagged.csv` / `skipped.csv` (both through `._append`, which writes the header for a new file) |
| `_split_of(map_name, site_id)` | train / val / test of a location, always the same |
| `write_summary(out)` | `manifest.csv`, `summary.txt`, `train.txt` / `val.txt` / `test.txt` from the JSON files |

Generator part:

| Function | What it does |
|---|---|
| `ConnectionLost`, `is_connection_error(exc)` | recognise a dead / hung server |
| `split_evenly(total, n)`, `exact_counts(n, shares)` | exact integer splits |
| `scenes_for(weather, available_kinds)` | scene types this weather can have on this map |
| `resolve_maps(wanted, available)` | `MAPS` entries -> (short name, path to load); short name or full path, any case |
| `spot_shortages(planner, jobs)` | weather/scene combinations asking for more jobs than spots x `MAX_USES_PER_SPOT` (warned at start) |
| `build_jobs(map_name, map_index, n_maps, available_kinds)` | the ordered job list: weather blocks, scene types inside; job names are folder paths |
| `Generator(out)` | output folders, progress |
| `.run(plan_only)` | the whole run (section 1) |
| `._register_done(session, jobs)` | feeds finished jobs to the planner (spot use, views) |
| `._weather(session, job)` | sets the weather for the job's block |
| `._params(job, rng)` | the weather row used for this job's counts |
| `.run_job(session, job)` | plan -> render -> save, retrying until a good one or the limits |
| `._save(job, frames)` | writes frames, records flags, prints progress |
| `.plan_only(session, jobs, per_scene)` | `--plan-only`: plans without rendering and prints rejection reasons |
| `.log(text)`, `.write_dataset_yaml()` | `run_log.txt`, `data.yaml` |

---

## 7. How to change common things

| You want to... | Change |
|---|---|
| change counts or occlusion of a weather | `WEATHERS` in config.py |
| add a weather (e.g. `fog`) | add it to `WEATHERS` (config.py) and its look to `WEATHER_LOOK` (advanced.py) |
| change frames per occluded walk | `SEQUENCE["frames"]`; spread by `occlusion._frame_indices` |
| change how hidden the pedestrian gets | `occlusion` (min, max) per weather in config.py |
| make the reveal faster / slower | `SEQUENCE["walk_speed"]` |
| change where occluding cars stand | `ADJACENT_CARS_AHEAD`, `PARKED_NOSE_GAP`; logic in `traffic.adjacent_occluders` / `parked_occluder` |
| add a new scene type | 1) `SCENES` in config.py, 2) `SCENE_SITE_KINDS` (and `OCCLUDED_SCENES` if occluded) in advanced.py, 3) its behaviour in `planner.ScenePlanner.plan` |
| add a new kind of crossing place | a finder in roads.py returning `CrossingSite(..., kind="mykind", ...)`, call it in `find_all_sites`, use the kind in `SCENE_SITE_KINDS` |
| change camera height / angle / lens | `CAMERA_LOCATION`, `CAMERA_PITCH`, `CAMERA_FOV` (labels follow automatically) |
| change image size | `IMAGE_W`, `IMAGE_H` |
| add vehicle models | the `*_BLUEPRINTS` lists in config.py; check with `python main.py --check` |
| fix a wrongly typed pedestrian model | `PED_TYPE_OVERRIDES` in advanced.py |
| change pedestrian speeds | `PED_SPEED_RANGE`, `PED_RUNNER_PROBABILITY` |
| add / reorder YOLO classes | `class_names()` in advanced.py, `TAG_CLASS` (CARLA tag -> class), `labels.class_id`, `MIN_VISIBLE_PIXELS` |
| change when an image is flagged | `labels.frame_flags` and its thresholds (`OCCLUSION_TOLERANCE`, `MIN_TARGET_PIXELS`, `MIN_ROAD_FRACTION`) |
| change what is saved | `SAVE` in config.py; a new output: add a flag + folder in `OUTPUT_FOLDERS` and write it in `labels.write_frame` |
| change box drawing | `BOX_COLOR`, `BOX_THICKNESS`; code in `labels.draw_2d` / `draw_3d` |
| more variety | add maps to `MAPS`; lower `MAX_USES_PER_SPOT`; raise `MIN_VIEW_DISTANCE_GAP`; widen `EGO_DISTANCE_RANGE`, `CAMERA_LIFT_RANGE`, `CAMERA_PITCH_JITTER_DEG` |
| use a custom map | add its name or path to `MAPS`; check with `python main.py --check` |

---

## 8. Rules to keep when you change code

1. **Plans are data only.** Nothing in planner.py, people.py, traffic.py, occlusion.py,
   camera.py or roads.py may spawn anything in CARLA. Spawning happens only in
   render.py / carla_io.py.
2. **Everything spawned goes into the `actors` list at once** (see `spawn_vehicles`), so
   `render_job`'s `finally` can always remove it, even after an error or Ctrl+C.
3. **Cleanup code runs inside `with protected():`** so a second Ctrl+C can't cut it short.
4. **The JSON is written last** in `write_frame`, and a job is marked done only after
   all its frames are written. Resume relies on this.
5. **Read actor positions right after `capture()`**, before any further tick, so labels
   match the saved image.
6. **Job names come from the config.** Changing `WEATHERS`, `SCENES` or `SEQUENCE` changes
   the job list. After such a change, start a new output folder (`--out`) instead of
   resuming an old one.
7. **Random choices use the `rng` passed in**, never the global `random`, so a seed
   reproduces the same plan.
8. **Settings are read as `C.NAME`** (`import advanced as C`), so a value can move between
   config.py and advanced.py freely.

---

## 9. Finding problems

| Symptom | Look at |
|---|---|
| a map makes few or no plans | `python main.py --plan-only 5`: site counts and rejection reasons |
| errors during a run | `run_log.txt` in the output folder (full tracebacks) |
| images look wrong | `flagged.csv` reasons, then `images_2d_bbox/` / `images_3d_bbox/` |
| a job produced nothing | `skipped.csv` (reasons, most common first) |
| why one image looks the way it does | `labels_3d/<image>.json` -> `plan` (every position) and `seed_key` |
| CARLA left frozen / full of actors | `python main.py --cleanup` |
| quick test | `python main.py --out /tmp/test` with small `images` values in config.py |
