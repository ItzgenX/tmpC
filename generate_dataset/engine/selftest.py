"""python main.py --self-test: checks in real CARLA that occluded scenes come out as planned.

For every occluder layout (car(s) right in front of the ego or in the next lane, a parked
car, turning left / right behind 1 or 2 cars) a few scenes are planned, rendered and measured:
  - the cars and the pedestrian really are where the plan put them, at every frame
  - measured from the segmentation, the pedestrian is visible in the first and last frame
    and mostly hidden in between (visible -> hidden -> visible)
  - the pedestrian is labeled Pedestrian, the occluding cars Car / Truck / Bus
The frames (with their 2D boxes) go to <out>/self_test/ so you can look at them.
Nothing is added to the dataset.
"""
import math
import os
import random

import advanced as C
from engine.carla_io import MapSession, SpawnError
from engine.labels import draw_2d, target_label, write_image
from engine.people import position_at
from engine.planner import PlanError
from engine.render import render_job
from engine.run import resolve_maps


# every occluder layout the generator makes: (name, scene, layout); also used by the tests
LAYOUTS = [
    ("ego -> car -> pedestrian (same lane)", "occluded_by_traffic", {"lane": "ego", "cars": 1}),
    ("ego -> car -> car -> pedestrian (same lane)", "occluded_by_traffic", {"lane": "ego", "cars": 2}),
    ("ego, car in the next lane -> pedestrian", "occluded_by_traffic", {"lane": "next", "cars": 1}),
    ("ego, 2 cars in the next lane -> pedestrian", "occluded_by_traffic", {"lane": "next", "cars": 2}),
    ("parked car -> pedestrian", "occluded_by_parked", {"cars": 1}),
    ("turn left: ego -> car -> pedestrian", "occluded_turning", {"side": "left", "lane": "ego", "cars": 1}),
    ("turn left: ego -> car -> car -> pedestrian", "occluded_turning", {"side": "left", "lane": "ego", "cars": 2}),
    ("turn right: ego -> car -> pedestrian", "occluded_turning", {"side": "right", "lane": "ego", "cars": 1}),
    ("turn right: ego -> car -> car -> pedestrian", "occluded_turning", {"side": "right", "lane": "ego", "cars": 2}),
]
PARAMS = dict(pedestrians=1, vehicles=2, cyclists=0, trucks=0, occlusion=(50, 95))

WALKER_TOLERANCE = 0.6      # metres; a real walker needs a few ticks to reach its speed
VEHICLE_TOLERANCE = 0.5     # metres; cars settle on the road before they are frozen
VISIBLE_MAX = 0.25          # measured occlusion of the first / last frame
HIDDEN_MIN = 0.35           # measured occlusion of the most hidden frame


def plan_layout(planner, scene, layout, weather, key, tries=C.MAX_PLANS_PER_JOB):
    job = {"name": f"selftest_{key}", "weather": weather, "scene": scene, "frames": 5, "layout": layout}
    for attempt in range(tries):
        try:
            return planner.plan(random.Random(f"{C.SEED}:selftest:{key}:{attempt}"), job, PARAMS)
        except PlanError:
            continue
    return None


def check(plan, frames):
    """Measured occlusion per frame and the list of problems (empty = as planned)."""
    problems, occ = [], []
    i = plan["target_pedestrian"]
    target = plan["pedestrians"][i]
    for k, (planned, frame) in enumerate(zip(plan["frames"], frames)):
        want = position_at(target, planned["tick"] * C.FIXED_DELTA_SECONDS)
        for p in frame["positions"]:
            if p["kind"] == "pedestrian":
                if p["plan_index"] == i:
                    d = math.hypot(p["center"][0] - want[0], p["center"][1] - want[1])
                    if d > WALKER_TOLERANCE:
                        problems.append(f"frame {k}: pedestrian {d:.2f} m from its planned spot")
                continue
            v = plan["vehicles"][p["plan_index"]]
            d = math.hypot(p["center"][0] - v["x"], p["center"][1] - v["y"])
            if d > VEHICLE_TOLERANCE:
                problems.append(f"frame {k}: {v['role']} car {d:.2f} m from its planned spot")
        lb = target_label(frame["labels"])
        occ.append(None if lb is None else lb["occlusion"])
        if lb is not None and lb["carla_class"] != "Pedestrian":
            problems.append(f"frame {k}: pedestrian labeled {lb['carla_class']}")
        for other in frame["labels"]:
            if other["meta"].get("role") == "occluder" and other["carla_class"] not in ("Car", "Truck", "Bus"):
                problems.append(f"frame {k}: occluding car labeled {other['carla_class']}")
    if None in occ:
        problems.append("pedestrian has no label in some frames")
    else:
        if occ[0] > VISIBLE_MAX or occ[-1] > VISIBLE_MAX:
            problems.append("not clearly visible in the first / last frame")
        if max(occ) < HIDDEN_MIN:
            problems.append("never really hidden")
        elif occ.index(max(occ)) in (0, len(occ) - 1):
            problems.append("most hidden at the first / last frame")
    return occ, problems


def run_self_test(out, per_layout):
    import carla
    client = carla.Client(C.CARLA_HOST, C.CARLA_PORT)
    client.set_timeout(C.CARLA_TIMEOUT)
    print(f"connected to CARLA {client.get_server_version()}")
    maps, _ = resolve_maps(C.MAPS, client.get_available_maps())
    if not maps:
        print("none of the maps in MAPS is on this server (see python main.py --check)")
        return C.EXIT_FATAL
    folder = os.path.join(out, "self_test")
    os.makedirs(folder, exist_ok=True)
    weather = "clear" if "clear" in C.WEATHER_LOOK else next(iter(C.WEATHER_LOOK))

    session = MapSession(client, carla, *maps[0])
    results = []
    try:
        session.open()
        session.set_weather(weather, "self-test", random.Random(C.SEED))
        for n, (name, scene, layout) in enumerate(LAYOUTS, 1):
            for k in range(1, per_layout + 1):
                plan = plan_layout(session.planner, scene, layout, weather, f"{n}_{k}")
                if plan is None:
                    results.append(("skip", name, "no place for this layout on this map"))
                    print(f"  skip  {name}: no place for it on {maps[0][0]}")
                    continue
                try:
                    frames = render_job(session, plan)
                except SpawnError as exc:
                    results.append(("FAIL", name, f"spawn failed: {exc}"))
                    print(f"  FAIL  {name}: spawn failed: {exc}")
                    continue
                occ, problems = check(plan, frames)
                for f, frame in enumerate(frames):
                    write_image(os.path.join(folder, f"{n:02d}_{k}_frame{f}.jpg"), draw_2d(frame["rgb"], frame["labels"]))
                measured = " ".join("-" if o is None else f"{o:.0%}" for o in occ)
                status = "ok" if not problems else "FAIL"
                results.append((status, name, f"hidden {measured}" + ("" if not problems else " | " + "; ".join(problems))))
                print(f"  {status:4s}  {name}: hidden {measured}" + ("" if not problems else "\n        " + "; ".join(problems)))
    finally:
        session.close()

    failed = [r for r in results if r[0] == "FAIL"]
    skipped = [r for r in results if r[0] == "skip"]
    print(f"\n{len(results) - len(failed) - len(skipped)} ok, {len(failed)} failed, {len(skipped)} skipped "
          f"(no place on this map). frames with boxes: {folder}")
    return C.EXIT_DONE if not failed else C.EXIT_FATAL
