"""python main.py --weather-preview: one picture sheet that shows what each weather setting does.

Everything is on the same street view with people standing on the crossing, as two kinds
of output, kept separate on purpose so each comparison is easy to read on its own:

  sheet.jpg        - single settings only: each one alone, 0/25/50/75/100, everything
                     else 0 (sun high), ending with the 4 settings combined ("rain: all 4
                     + cloudiness") and exposure_compensation.
  insights_*.jpg   - one small sheet per meaningful combination, not crammed together:
                       insights_weather_look.jpg      - your conditions from advanced.py,
                         all their parameters combined, at low/mid/high of each range.
                         This is what clear / partly_cloudy / overcast / rain actually
                         look like.
                       insights_combos.jpg            - your own hand-picked mixes
                         (edit the COMBOS list below) to test before touching advanced.py.
                       insights_matrix_<name>.jpg     - two parameters together, one per
                         axis (edit the MATRICES list below), so you can see how a pair
                         interacts instead of guessing from two separate single sweeps.

The exact numbers behind every insights_*.jpg picture are printed to the console and
written to <out>/weather_preview/values.txt.
Saved to <out>/weather_preview/: the sheets above, plus every picture full size.
Settings: https://carla.readthedocs.io/en/0.9.16/python_api/#carla.WeatherParameters
"""
import os
import random

import cv2
import numpy as np

import advanced as C
from engine.carla_io import (MapSession, camera_blueprints, destroy_actors, freeze, spawn_pedestrians,
                             spawn_vehicles, to_transform)
from engine.planner import PlanError
from engine.run import resolve_maps

LEVELS = (0, 25, 50, 75, 100)
ROWS = [                                    # (row name, settings set to the level)
    ("cloudiness", ["cloudiness"]),
    ("fog_density", ["fog_density"]),
    ("wind_intensity", ["wind_intensity"]),
    ("precipitation", ["precipitation"]),
    ("precipitation_deposits", ["precipitation_deposits"]),
    ("wetness", ["wetness"]),
    ("rain: all 4\n+ cloudiness", ["cloudiness", "precipitation", "precipitation_deposits", "wetness"]),
]
WEATHER_KEYS = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "fog_density", "wetness")
LOOK_POINTS = (0.0, 0.5, 1.0)                # fraction of each WEATHER_LOOK range: low / mid / high
LOOK_LABELS = ("low", "mid", "high")
# --- your own test mixes: edit freely, each one renders as one picture ---
# values left out default to 0. try a guess here before writing it into advanced.WEATHER_LOOK.
COMBOS = [
    ("light haze",     {"cloudiness": 20, "fog_density": 5, "wind_intensity": 10}),
    ("heavy overcast", {"cloudiness": 95, "fog_density": 6, "wind_intensity": 25}),
    ("drizzle",        {"cloudiness": 70, "precipitation": 30, "precipitation_deposits": 20,
                         "wetness": 30, "wind_intensity": 15}),
]
# --- x/y grids: see two parameters TOGETHER, one on each axis, everything else 0 ---
# (name, x param, x values, y param, y values) - edit freely, pick any two WEATHER_KEYS.
# e.g. this answers "what does wetness + precipitation together actually look like", which
# a single-parameter sweep can't show since wetness alone barely changes anything visually.
MATRICES = [
    ("precipitation x wetness", "precipitation", (0, 50, 100), "wetness", (0, 50, 100)),
    ("cloudiness x fog_density", "cloudiness", (0, 50, 100), "fog_density", (0, 10, 30)),
]
SUN = {"sun_altitude_angle": 70.0, "sun_azimuth_angle": 45.0}
# camera brightness: 0 = as measured, -1 = 2x darker, +1 = 2x brighter (CARLA's Linux default is -1.5)
# https://carla.readthedocs.io/en/0.9.16/ref_sensors/#rgb-camera
EXPOSURES = (-1.5, -1.0, -0.5, 0.0, 0.5)
EXPOSURE_SKY = {"cloudiness": 90.0}
SETTLE_TICKS = 10                           # let wet roads / auto exposure settle
CELL_W, CELL_H, LABEL_W, HEADER_H = 384, 216, 300, 40


def _scene(session):
    """A crossing with 3 people and a car, planned like a normal clear-weather job."""
    kinds = {s.kind for s in session.planner.sites}
    scene = "crosswalk" if "crosswalk" in kinds else "jaywalk_road"
    job = {"name": "weather_preview", "weather": "preview", "scene": scene, "frames": 1}
    counts = dict(pedestrians=3, vehicles=1, cyclists=0, trucks=0, occlusion=0)
    for attempt in range(C.MAX_PLANS_PER_JOB):
        try:
            return session.planner.plan(random.Random(f"{C.SEED}:preview:{attempt}"), job, counts)
        except PlanError:
            continue
    raise RuntimeError("could not plan a street view on this map")


def _weather(session, values):
    w = session.carla.WeatherParameters()
    for k, v in {**{k: 0.0 for k in WEATHER_KEYS}, **SUN, **values}.items():
        setattr(w, k, float(v))
    session.world.set_weather(w)


def _values_label(values):
    return " ".join(f"{k}={v:g}" for k, v in sorted(values.items()))


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def _matrix_rows():
    """MATRICES -> one (matrix name, rows) per matrix, each a standalone table: x-values
    as a header row on top (no photo), the y-value as the row name on the left, one
    column per x-value, both params set together.
    """
    out = []
    for name, xk, xs, yk, ys in MATRICES:
        header = [(f"{xk}={x:g}", None) for x in xs]          # None = text-only cell, no photo
        rows = [(f"(rows: {yk})", header)]
        for y in ys:
            cells = [(f"{xk}={x:g}", {xk: x, yk: y}) for x in xs]
            rows.append((f"{yk}={y:g}", cells))
        out.append((name, rows))
    return out


def _look_rows():
    """advanced.WEATHER_LOOK, every parameter in a condition combined, at low/mid/high of its range."""
    rows = []
    for cond, ranges in C.WEATHER_LOOK.items():
        cells = []
        for t, label in zip(LOOK_POINTS, LOOK_LABELS):
            cells.append((label, {k: lo + (hi - lo) * t for k, (lo, hi) in ranges.items()}))
        rows.append((cond, cells))
    return rows


def _photo(session, cam):
    _, (img,) = session.capture([cam], SETTLE_TICKS)
    rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
    return cv2.resize(rgb, (C.IMAGE_W, C.IMAGE_H), interpolation=cv2.INTER_AREA)


def _text(img, text, x, y, scale=0.6):
    """White text on a solid black box - clearer than an outline against a busy photo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _sheet(rows):
    """rows: [(name, [(column title, image)])] -> one labelled grid image.

    Rows can have different numbers of cells (e.g. 5 LEVELS vs 3 low/mid/high). Every
    cell keeps its native CELL_W x CELL_H (matches the images' real aspect ratio, so
    nothing gets stretched/warped) - a shorter row is just centred in the row instead,
    so the leftover space is split evenly either side rather than dumped as one block.
    """
    cols = max(len(cells) for _, cells in rows)
    width = LABEL_W + cols * CELL_W
    sheet = np.full((HEADER_H + len(rows) * CELL_H, width, 3), 40, np.uint8)
    _text(sheet, "setting  \\  value", 10, 28)
    for r, (name, cells) in enumerate(rows):
        y = HEADER_H + r * CELL_H
        for i, line in enumerate(name.split("\n")):
            _text(sheet, line, 10, y + CELL_H // 2 + i * 24, 0.55)
        row_start = LABEL_W + (width - LABEL_W - len(cells) * CELL_W) // 2
        for c, (title, img) in enumerate(cells):
            x = row_start + c * CELL_W
            if img is None:                                    # header cell: label only, no photo
                sheet[y:y + CELL_H, x:x + CELL_W] = 55
                _text(sheet, title, x + 8, y + CELL_H // 2, 0.7)
            else:
                sheet[y:y + CELL_H, x:x + CELL_W] = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
                _text(sheet, title, x + 8, y + 26)
    return sheet


def run_weather_preview(out):
    import carla
    client = carla.Client(C.CARLA_HOST, C.CARLA_PORT)
    client.set_timeout(C.CARLA_TIMEOUT)
    print(f"connected to CARLA {client.get_server_version()}")
    maps, _ = resolve_maps(C.MAPS, client.get_available_maps())
    if not maps:
        print("none of the maps in MAPS is on this server (see python main.py --check)")
        return C.EXIT_FATAL
    folder = os.path.join(out, "weather_preview")
    os.makedirs(folder, exist_ok=True)

    session, actors, rows, legend = MapSession(client, carla, *maps[0]), [], [], []
    try:
        session.open()
        plan = _scene(session)
        session.ego.set_transform(to_transform(plan["ego"], session.planner.ego_z_offset))
        vehicles = spawn_vehicles(session.world, session.bp_lib, plan["vehicles"], actors)
        for _ in range(10):
            session.world.tick()
        freeze(vehicles)
        # people stand where they would be at the first frame
        standing = [dict(p, spawn=[*p["capture_xy"], p["spawn"][2]]) for p in plan["pedestrians"]]
        spawn_pedestrians(session.world, session.bp_lib, standing, actors)

        for name, keys in ROWS:
            cells = []
            for v in LEVELS:
                _weather(session, {k: v for k in keys})
                img = _photo(session, session.rgb)
                cv2.imwrite(os.path.join(folder, f"{name.split(':')[0]}_{v:03d}.jpg"), img)
                cells.append((f"{v}", img))
            rows.append((name, cells))
            print(f"  {name}: done")

        insight_sheets = []                 # [(output filename, rows)] - one small sheet each

        look_rows = []
        for cond, cells in _look_rows():
            rendered = []
            for label, values in cells:
                _weather(session, values)
                img = _photo(session, session.rgb)
                fname = f"look_{cond}_{label}.jpg"
                cv2.imwrite(os.path.join(folder, fname), img)
                rendered.append((label, img))
                line = f"{fname}: {_values_label(values)}"
                legend.append(line)
                print(f"  {line}")
            look_rows.append((f"WEATHER_LOOK\n{cond}", rendered))
        if look_rows:
            insight_sheets.append(("insights_weather_look.jpg", look_rows))

        if COMBOS:
            cells = []
            for name, values in COMBOS:
                _weather(session, values)
                img = _photo(session, session.rgb)
                fname = f"combo_{name.replace(' ', '_')}.jpg"
                cv2.imwrite(os.path.join(folder, fname), img)
                cells.append((name, img))
                line = f"{fname}: {_values_label(values)}"
                legend.append(line)
                print(f"  {line}")
            insight_sheets.append(("insights_combos.jpg", [("your COMBOS\n(edit in this file)", cells)]))

        # all matrices stacked into ONE sheet (insights_matrices.jpg) - no per-cell files,
        # only that one combined image, so there is nothing separate to get confused by.
        matrix_rows = []
        for mat_name, mat_rows in _matrix_rows():
            for row_name, cells in mat_rows:
                rendered_cells = []
                for title, values in cells:
                    if values is None:                          # header row: labels only
                        rendered_cells.append((title, None))
                        continue
                    _weather(session, values)
                    img = _photo(session, session.rgb)
                    rendered_cells.append((title, img))
                    legend.append(f"insights_matrices.jpg [{mat_name} / {row_name} / {title}]: "
                                  f"{_values_label(values)}")
                # first row of each matrix carries the matrix's own name too
                label = f"{mat_name}\n{row_name}" if row_name.startswith("(rows:") else row_name
                matrix_rows.append((label, rendered_cells))
            print(f"  matrix '{mat_name}': done")
        if matrix_rows:
            insight_sheets.append(("insights_matrices.jpg", matrix_rows))

        _weather(session, EXPOSURE_SKY)
        cells = []
        for e in EXPOSURES:                 # brightness can only be set when a camera is made
            bp = camera_blueprints(session.bp_lib)[0]
            if bp.has_attribute("exposure_compensation"):
                bp.set_attribute("exposure_compensation", str(e))
            cam_tf = carla.Transform(carla.Location(*C.CAMERA_LOCATION), carla.Rotation(pitch=C.CAMERA_PITCH))
            cam = session.world.spawn_actor(bp, cam_tf, attach_to=session.ego)
            try:
                img = _photo(session, cam)
            finally:
                destroy_actors(session.client, session.world, [cam])
            cv2.imwrite(os.path.join(folder, f"exposure_{e:+.1f}.jpg"), img)
            cells.append((f"{e:+.1f}", img))
        rows.append(("exposure_compensation\n(at cloudiness 90)", cells))
    finally:
        if session.world is not None:
            destroy_actors(session.client, session.world, actors)
        session.close()

    if legend:
        with open(os.path.join(folder, "values.txt"), "w") as f:
            f.write("\n".join(legend) + "\n")

    path = os.path.join(folder, "sheet.jpg")
    cv2.imwrite(path, _sheet(rows))
    print(f"\nweather preview: {path} (single pictures full size next to it)")
    for fname, sheet_rows in insight_sheets:
        ipath = os.path.join(folder, fname)
        cv2.imwrite(ipath, _sheet(sheet_rows))
        print(f"insight sheet: {ipath}")
    if legend:
        print(f"exact numbers behind the insight sheets: {os.path.join(folder, 'values.txt')}")
    return C.EXIT_DONE
