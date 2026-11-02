"""python main.py --weather-preview: one picture sheet that shows what each weather setting does.

Every row changes one setting from 0 to 100 (everything else 0, sun high), on the same
street view with people standing on the crossing, so you can see how strong a value looks
and pick the (min, max) ranges of WEATHER_LOOK in advanced.py. The last rows show the
rain settings together, and the camera brightness (exposure_compensation) on a cloudy day.
Saved to <out>/weather_preview/: sheet.jpg, plus every picture full size.
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


def _photo(session, cam):
    _, (img,) = session.capture([cam], SETTLE_TICKS)
    rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
    return cv2.resize(rgb, (C.IMAGE_W, C.IMAGE_H), interpolation=cv2.INTER_AREA)


def _text(img, text, x, y, scale=0.6):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _sheet(rows):
    """rows: [(name, [(column title, image)])] -> one labelled grid image."""
    cols = max(len(cells) for _, cells in rows)
    sheet = np.full((HEADER_H + len(rows) * CELL_H, LABEL_W + cols * CELL_W, 3), 40, np.uint8)
    _text(sheet, "setting  \\  value", 10, 28)
    for r, (name, cells) in enumerate(rows):
        y = HEADER_H + r * CELL_H
        for i, line in enumerate(name.split("\n")):
            _text(sheet, line, 10, y + CELL_H // 2 + i * 24, 0.55)
        for c, (title, img) in enumerate(cells):
            x = LABEL_W + c * CELL_W
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

    session, actors, rows = MapSession(client, carla, *maps[0]), [], []
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

    path = os.path.join(folder, "sheet.jpg")
    cv2.imwrite(path, _sheet(rows))
    print(f"\nweather preview: {path} (single pictures full size next to it)")
    return C.EXIT_DONE
