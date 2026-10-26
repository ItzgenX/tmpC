"""The generation run: the job list and job loop (generator), and the bookkeeping
files in the output folder (progress).
"""
import collections
import csv
import glob
import hashlib
import json
import os
import random
import traceback

import advanced as C
from advanced import EXIT_DONE, EXIT_FATAL, EXIT_RECYCLE
from engine.carla_io import MapSession, SpawnError
from engine.labels import make_folders, write_frame
from engine.planner import PlanError
from engine.render import render_job
from engine.tools import FLAG_FIELDS


# ============================================================================
# progress
# ============================================================================
# Bookkeeping files in the output folder.
#
# jobs_done.txt  one line per finished job (resume skips these)
# flagged.csv    images that failed a check: kept, listed here for review (python main.py --delete-flagged)
# skipped.csv    jobs that never produced an image (spawn failures / crashes), with the reasons
# manifest.csv   one row per image, rebuilt from labels_3d/*.json
# summary.txt    counts per weather / scene / flag
# train.txt, val.txt, test.txt   image lists, split by location so no place is in two splits

class Progress:
    def __init__(self, out):
        self.out = out
        self.done_path = os.path.join(out, "jobs_done.txt")
        self.done = set()
        if os.path.exists(self.done_path):
            with open(self.done_path, encoding="utf-8") as f:
                self.done = {line.strip() for line in f if line.strip()}

    def mark_done(self, job_name):
        with open(self.done_path, "a", encoding="utf-8") as f:
            f.write(job_name + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add(job_name)

    def _append(self, filename, fields, row):
        path = os.path.join(self.out, filename)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def flag(self, image, weather, scene, reasons, planned, measured):
        self._append("flagged.csv", FLAG_FIELDS, {
            "image": image, "weather": weather, "scene": scene, "reasons": "; ".join(reasons),
            "planned_occlusion": planned, "measured_occlusion": "" if measured is None else measured})

    def skip(self, job_name, reasons):
        self._append("skipped.csv", ["job", "reasons"],
                     {"job": job_name, "reasons": "; ".join(f"{r} x{n}" for r, n in reasons.most_common(4))})


def _split_of(map_name, site_id):
    """Same location -> same split, always."""
    h = int(hashlib.md5(f"{map_name}|{site_id}".encode()).hexdigest(), 16) % 10000 / 10000.0
    train, val, _ = C.TRAIN_VAL_TEST
    return "train" if h < train else ("val" if h < train + val else "test")


def write_summary(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "labels_3d", "**", "*.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        plan, labels = d["plan"], d["labels"]
        peds = [lb for lb in labels if lb["kind"] == "pedestrian"]
        rows.append({
            "image": d["image"], "map": d["map"], "weather": plan["weather"], "scene": plan["scene"],
            "site": plan["site"]["id"], "ego_distance": plan["ego"]["distance"],
            "planned_occlusion": d["planned_occlusion"],
            "measured_occlusion": "" if d["measured_occlusion"] is None else d["measured_occlusion"],
            "flagged": bool(d["flags"]),
            "pedestrians": sum(p["role"] == "crossing" for p in plan["pedestrians"]),
            "vehicles": sum(v["category"] == "car" for v in plan["vehicles"]),
            "trucks": sum(v["category"] == "truck" for v in plan["vehicles"]),
            "cyclists": sum(v["category"] == "cyclist" for v in plan["vehicles"]),
            "labeled_pedestrians": len(peds), "labeled_other": len(labels) - len(peds),
        })
    if not rows:
        print("no images yet")
        return
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if C.TRAIN_VAL_TEST:
        splits = collections.defaultdict(list)
        for r in rows:
            if not r["flagged"]:
                splits[_split_of(r["map"], r["site"])].append(os.path.join("images", r["image"]))
        for name in ("train", "val", "test"):
            with open(os.path.join(out, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(splits[name]) + ("\n" if splits[name] else ""))

    lines = [f"images: {len(rows)}  (flagged: {sum(r['flagged'] for r in rows)})"]
    for key in ("weather", "scene", "map"):
        cnt = collections.Counter(str(r[key]) for r in rows)
        lines.append(f"{key:10s} " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    lines.append(f"{'locations':10s} {len({(r['map'], r['site']) for r in rows})}")
    flagged = os.path.join(out, "flagged.csv")
    if os.path.exists(flagged):
        with open(flagged, newline="", encoding="utf-8") as f:
            reasons = collections.Counter("occlusion off target" if r.startswith("occlusion") else r
                                          for row in csv.DictReader(f) for r in row["reasons"].split("; "))
        lines.append("flag reasons: " + "  ".join(f"{k}={v}" for k, v in reasons.most_common(6)))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


# ============================================================================
# generator
# ============================================================================
# The generation loop (runs in the worker process).
#
# For every map, weather by weather (in config order), scene type by scene type:
# plan a job, render it, save its frames. Weather is set once per weather block
# (new sun/cloud values every WEATHER_RESAMPLE_EVERY jobs), never per image.
# Finished jobs are listed in jobs_done.txt, so a restart continues where it stopped.

class ConnectionLost(Exception):
    pass


def is_connection_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ("time-out", "timeout", "timed out", "connection", "rpc", "closed"))


def split_evenly(total, n):
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def exact_counts(n, shares):
    """n split by shares with exact counts (largest remainder)."""
    total = sum(shares.values())
    raw = {k: n * v / total for k, v in shares.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def scenes_for(weather, available_kinds):
    """Scene types this weather can have on this map."""
    rows = [C.WEATHERS[weather]] if C.SCENE_FOLLOWS_WEATHER else [C.WEATHERS[w] for w in C.enabled_weathers()]
    out = []
    for scene in C.enabled_scenes():
        if scene in C.OCCLUDED_SCENES and not any(C.can_occlude(r) for r in rows):
            continue        # e.g. clear weather with occlusion 0
        if not set(C.SCENE_SITE_KINDS[scene]) & available_kinds:
            continue        # e.g. no T-junction without a crosswalk on this map
        out.append(scene)
    return out


def resolve_maps(wanted, available):
    """MAPS entries -> ([(short name, path to load)], [entries not on the server]).
    An entry can be a short name ("Town03_Opt", "MyCity") or a full path
    ("/Game/Custom/Maps/MyCity"); short names match case-insensitively."""
    by_name = {}
    for path in available:
        by_name.setdefault(path.split("/")[-1].lower(), path)
    found, missing = [], []
    for entry in wanted:
        path = entry if entry in available else by_name.get(entry.split("/")[-1].lower())
        if path is None:
            missing.append(entry)
        elif path not in [p for _, p in found]:
            found.append((path.split("/")[-1], path))
    return found, missing


def spot_shortages(planner, jobs):
    """(weather/scene, jobs, spots) where a scene asks for more jobs than its spots allow."""
    if not C.MAX_USES_PER_SPOT:
        return []
    need = collections.Counter((j["weather"], j["scene"]) for j in jobs)
    out = []
    for (weather, scene), n in need.items():
        spots = len(planner.sites_of(C.SCENE_SITE_KINDS[scene]))
        if n > spots * C.MAX_USES_PER_SPOT:
            out.append((f"{weather}/{scene}", n, spots))
    return out


def build_jobs(map_name, map_index, n_maps, available_kinds):
    """The full, ordered job list for one map: weather blocks, scene types inside each.

    A job's name is also where its files go, e.g. "rain/occluded_by_parked/rain_occluded_by_parked_00012"
    -> images/rain/occluded_by_parked/rain_occluded_by_parked_00012_00.png. The file name
    alone is unique in the whole dataset; SAVE_IN_FOLDERS = False drops the folders.
    The map is added (folder and name) only when more than one map is used.
    """
    folder = [map_name] if n_maps > 1 else []
    jobs = []
    for weather in C.enabled_weathers():
        n = split_evenly(C.WEATHERS[weather]["images"], n_maps)[map_index]
        scenes = scenes_for(weather, available_kinds)
        if not scenes or n == 0:
            continue
        per_scene = exact_counts(n, {s: C.SCENES[s]["share"] for s in scenes})
        for scene in scenes:
            left, j, size = per_scene[scene], 0, C.frames_per_job(scene)
            while left > 0:
                frames = min(size, left)
                parts = folder + [weather, scene]
                name = "_".join(parts + [f"{j:05d}"])
                if C.SAVE_IN_FOLDERS:
                    name = "/".join(parts + [name])
                jobs.append({"name": name, "map": map_name,
                             "weather": weather, "scene": scene, "frames": frames})
                left -= frames
                j += 1
    return jobs


class Generator:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        make_folders(out)
        self.progress = Progress(out)
        self.log_path = os.path.join(out, "run_log.txt")

    def log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def write_dataset_yaml(self):
        names = C.class_names()
        with open(os.path.join(self.out, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(self.out)}\n")
            if C.TRAIN_VAL_TEST:
                f.write("train: train.txt\nval: val.txt\ntest: test.txt\n")
            else:
                f.write("train: images\nval: images\n")
            f.write(f"nc: {len(names)}\nnames: {json.dumps(names)}\n")

    # ---------- top level ----------

    def run(self, plan_only=0):
        try:
            C.validate()
        except ValueError as exc:
            print(f"config error: {exc}")
            return EXIT_FATAL
        import carla
        self.carla = carla
        self.client = carla.Client(C.CARLA_HOST, C.CARLA_PORT)
        self.client.set_timeout(C.CARLA_TIMEOUT)
        try:
            print(f"connected to CARLA {self.client.get_server_version()}")
        except RuntimeError as exc:
            raise ConnectionLost(exc)
        self.write_dataset_yaml()

        for w in C.enabled_weathers():
            lo, hi = C.occlusion_range(C.WEATHERS[w].get("occlusion"))
            if 0 < hi < C.LOWEST_REACHABLE_OCCLUSION:
                print(f"  warning: {w} occlusion range {C.WEATHERS[w]['occlusion']} is below what a car "
                      f"normally hides ({C.LOWEST_REACHABLE_OCCLUSION:.0%}+ at the most hidden moment); "
                      f"most occluded jobs will be skipped. use e.g. (50, 75)")
        maps, missing = resolve_maps(C.MAPS, self.client.get_available_maps())
        for m in missing:
            print(f"  warning: map {m} is not on this server (see python main.py --check), skipped")
        if not maps:
            print("none of the maps in MAPS is on this server")
            return EXIT_FATAL

        processed, usable_maps = 0, 0
        for map_index, (map_name, map_path) in enumerate(maps):
            session = MapSession(self.client, carla, map_name, map_path)
            try:
                try:
                    session.open()
                except Exception as exc:     # noqa: BLE001 - an unusable map must not stop the run
                    if is_connection_error(exc):
                        raise ConnectionLost(exc)
                    self.log(f"{map_name} unusable: {exc}\n{traceback.format_exc()}")
                    print(f"  {map_name} unusable, skipped: {exc}")
                    continue
                kinds = {s.kind for s in session.planner.sites}
                missing = [sc for sc in C.enabled_scenes() if not set(C.SCENE_SITE_KINDS[sc]) & kinds]
                if missing:
                    print(f"  {map_name}: no place for {', '.join(missing)} on this map - "
                          f"those images go to the other scene types")
                jobs = build_jobs(map_name, map_index, len(maps), kinds)
                if jobs:
                    usable_maps += 1
                for what, n, spots in spot_shortages(session.planner, jobs):
                    print(f"  not enough spots on {map_name}: {what} needs {n} jobs, but {spots} spots x "
                          f"{C.MAX_USES_PER_SPOT} uses = {spots * C.MAX_USES_PER_SPOT}. the rest is skipped "
                          f"(skipped.csv); raise MAX_USES_PER_SPOT, add maps, or lower images")
                block = collections.Counter()
                for job in jobs:                # place of each job inside its weather block
                    job["block_index"] = block[job["weather"]]
                    block[job["weather"]] += 1
                if plan_only:
                    self.plan_only(session, jobs, plan_only)
                    continue
                self._register_done(session, jobs)
                todo = [j for j in jobs if j["name"] not in self.progress.done]
                print(f"\n{map_name}: {len(jobs) - len(todo)}/{len(jobs)} jobs done, {len(todo)} to go")
                for job in todo:
                    self.run_job(session, job)
                    processed += 1
                    if processed >= C.MAX_JOBS_PER_PROCESS:
                        return EXIT_RECYCLE
                    if processed % C.RELOAD_WORLD_EVERY == 0:
                        session.reload()
            finally:
                session.close()

        if not usable_maps:
            print("no map could be used (no crossing sites, or the ego could not be placed) - "
                  "nothing generated. check MAPS in config.py and run_log.txt")
            return EXIT_FATAL
        if not plan_only:
            write_summary(self.out)
        return EXIT_DONE

    def _register_done(self, session, jobs):
        """Saved jobs count toward site balance and the no-repeated-viewpoint rule."""
        for job in jobs:
            if job["name"] not in self.progress.done:
                continue
            path = os.path.join(self.out, "labels_3d", job["name"] + "_00.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    session.planner.register(json.load(f)["plan"])

    # ---------- one job ----------

    def _weather(self, session, job):
        # keyed by the job's place in the whole block, so a resumed run gets the same look
        every = C.WEATHER_RESAMPLE_EVERY or 10 ** 9
        key = (job["weather"], job["block_index"] // every)
        session.set_weather(job["weather"], key, random.Random(f"{C.SEED}:{job['map']}:{key}"))

    def _params(self, job, rng):
        if C.SCENE_FOLLOWS_WEATHER:
            return C.WEATHERS[job["weather"]]
        rows = [C.WEATHERS[w] for w in C.enabled_weathers()]
        if job["scene"] in C.OCCLUDED_SCENES:
            rows = [r for r in rows if C.can_occlude(r)]
        return rng.choice(rows)

    def run_job(self, session, job):
        self._weather(session, job)
        reasons, renders = collections.Counter(), 0
        for attempt in range(C.MAX_PLANS_PER_JOB):
            rng = random.Random(f"{C.SEED}:{job['name']}:{attempt}")
            try:
                plan = session.planner.plan(rng, job, self._params(job, rng))
            except PlanError as exc:
                reasons[f"plan: {exc}"] += 1
                continue
            plan["seed_key"] = f"{C.SEED}:{job['name']}:{attempt}"
            renders += 1
            try:
                frames = render_job(session, plan)
            except Exception as exc:     # noqa: BLE001 - one bad scene must not stop the run
                if is_connection_error(exc):
                    raise ConnectionLost(exc)
                why = str(exc) if isinstance(exc, SpawnError) else type(exc).__name__
                reasons[f"render: {why}"] += 1
                self.log(f"{job['name']} attempt {attempt}: {exc}\n{traceback.format_exc()}")
                print(f"  {job['name']}: render failed ({why}), retrying")
                if renders >= C.MAX_RENDERS_PER_JOB:
                    break
                continue

            if C.SKIP_FLAGGED_IMAGES and all(f["flags"] for f in frames):
                reasons["all frames failed checks"] += 1
                if renders >= C.MAX_RENDERS_PER_JOB:
                    break
                continue
            self._save(job, frames)
            session.planner.register(plan)
            self.progress.mark_done(job["name"])
            return True

        self.progress.skip(job["name"], reasons)
        self.log(f"{job['name']} skipped: {dict(reasons)}")
        print(f"  {job['name']} skipped: {reasons.most_common(2)}")
        return False

    def _save(self, job, frames):
        saved = 0
        for k, frame in enumerate(frames):
            if frame["flags"] and C.SKIP_FLAGGED_IMAGES:
                self.log(f"{job['name']}_{k:02d} not saved: {'; '.join(frame['flags'])}")
                continue
            name = f"{job['name']}_{k:02d}"
            write_frame(self.out, name, frame)
            saved += 1
            if frame["flags"]:
                info = frame["info"]
                self.progress.flag(name + C.IMAGE_EXT, job["weather"], job["scene"], frame["flags"],
                                   info["planned_occlusion"], info["measured_occlusion"])
        flagged = sum(1 for f in frames if f["flags"])
        note = (f", {flagged} {'skipped (failed a check)' if C.SKIP_FLAGGED_IMAGES else 'flagged'}"
                if flagged else "")
        print(f"  {job['name']}: {saved} image(s){note}")

    # ---------- dry run ----------

    def plan_only(self, session, jobs, per_scene):
        """Plans up to `per_scene` jobs per (weather, scene) without rendering."""
        seen, reasons, ok = collections.Counter(), collections.Counter(), collections.Counter()
        for job in jobs:
            key = (job["weather"], job["scene"])
            if seen[key] >= per_scene:
                continue
            seen[key] += 1
            for attempt in range(C.MAX_PLANS_PER_JOB):
                rng = random.Random(f"{C.SEED}:{job['name']}:{attempt}")
                try:
                    plan = session.planner.plan(rng, job, self._params(job, rng))
                except PlanError as exc:
                    reasons[str(exc)] += 1
                    continue
                session.planner.register(plan)
                ok[key] += 1
                break
        for key in sorted(seen):
            print(f"  {key[0]:14s} {key[1]:20s} planned {ok[key]}/{seen[key]}")
        for r, c in reasons.most_common(8):
            print(f"    rejected {c:5d}x  {r}")
