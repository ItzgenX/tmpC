"""Pedestrian-crossing dataset generator. Run:  python main.py

The process you start is a small supervisor. It runs the real generator in a
child process and restarts it (resuming exactly where it stopped) if it dies,
including native CARLA crashes / core dumps that no Python code can catch.
If CARLA_EXECUTABLE is set in config.py it also (re)starts the CARLA server.

    python main.py                  generate / resume the dataset
    python main.py --plan-only 50   plan 50 scenes per map without rendering (sanity check)
    python main.py --summary        rebuild manifest.csv + summary.txt from what is on disk
"""
import argparse
import collections
import csv
import glob
import json
import os
import queue
import random
import socket
import subprocess
import sys
import time
import traceback

import advanced as C

EXIT_DONE, EXIT_FATAL, EXIT_NO_SERVER, EXIT_RECYCLE = 0, 2, 3, 10
# labels/ uses PED_YOLO_BOX; the other pedestrian box variant goes to ALT_LABELS
ALT_LABELS = "labels_visible" if C.PED_YOLO_BOX == "full" else "labels_full"
SUBDIRS = ["images", "labels", ALT_LABELS, "labels_3d", "seg_raw", "seg_color", "debug"]
SUBDIRS += (["depth_raw", "depth_color"] if C.SAVE_DEPTH else [])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=C.OUTPUT_DIR, help="output folder (resumes if it exists)")
    ap.add_argument("--total", type=int, default=C.TOTAL_IMAGES, help="total images across all maps")
    ap.add_argument("--plan-only", type=int, default=0, metavar="N", help="plan N scenes per map, no rendering")
    ap.add_argument("--summary", action="store_true", help="only rebuild manifest/summary")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


# ====================================================================== supervisor

def server_reachable():
    try:
        with socket.create_connection((C.CARLA_HOST, C.CARLA_PORT), timeout=3):
            return True
    except OSError:
        return False


class ServerKeeper:
    """Starts / restarts CARLA if CARLA_EXECUTABLE is configured, else just waits for it."""

    def __init__(self):
        self.proc = None

    def restart(self):
        if not C.CARLA_EXECUTABLE:
            return False
        if self.proc is not None and self.proc.poll() is None:
            print("  stopping CARLA server")
            self.proc.kill()
            self.proc.wait()
        print(f"  starting CARLA: {C.CARLA_EXECUTABLE}")
        self.proc = subprocess.Popen([C.CARLA_EXECUTABLE] + list(C.CARLA_ARGS))
        time.sleep(C.CARLA_STARTUP_SECONDS)
        return True

    def wait(self):
        if server_reachable():
            return True
        if self.restart() and server_reachable():
            return True
        print(f"  CARLA not reachable at {C.CARLA_HOST}:{C.CARLA_PORT} - start the server, waiting...")
        end = time.time() + C.SERVER_WAIT_SECONDS
        while time.time() < end:
            time.sleep(10)
            if server_reachable():
                time.sleep(15)  # let it finish loading
                return True
        return False


def count_done(out):
    return len(glob.glob(os.path.join(out, "labels_3d", "*.json")))


def supervise(args):
    keeper = ServerKeeper()
    passthrough = ["--out", args.out, "--total", str(args.total)]
    crashes, no_server = 0, 0
    while True:
        if not keeper.wait():
            print("gave up waiting for the CARLA server")
            return 1
        before = count_done(args.out)
        code = subprocess.call([sys.executable, os.path.abspath(__file__), "--worker"] + passthrough)
        after = count_done(args.out)
        if code == EXIT_DONE:
            write_manifest(args.out)
            print(f"\ndone: {after} images in {args.out}")
            return 0
        if code == EXIT_FATAL:
            print("stopped on a configuration error (see above)")
            return 2
        if code == EXIT_RECYCLE:
            crashes = no_server = 0
            print("\nworker recycled, continuing")
            continue
        if code == EXIT_NO_SERVER:
            no_server += 1
            print(f"\nserver connection lost ({no_server})")
            if no_server >= 2 and server_reachable():
                keeper.restart()    # port open but not answering: hung server
            continue
        crashes = 0 if after > before else crashes + 1
        print(f"\nworker died (exit code {code}), {after - before} new images this run, "
              f"crash streak {crashes}/{C.MAX_CRASHES_WITHOUT_PROGRESS}")
        if crashes >= C.MAX_CRASHES_WITHOUT_PROGRESS:
            print("no progress across repeated crashes - check run_log.txt in the output folder")
            return 1
        if not server_reachable():
            keeper.restart()
        time.sleep(5)


# ====================================================================== manifest

def write_manifest(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "labels_3d", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s, labels = d["scene"], d["labels"]
        peds = [lb for lb in labels if lb["kind"] == "pedestrian"]
        target = next((lb for lb in peds if lb["meta"].get("target")), None)
        rows.append({
            "image": d["image"], "map": d["map"], "weather": s["weather"]["name"],
            "crossing": s["site"]["kind"], "site": s["site"]["id"], "ego_distance": s["ego"]["distance"],
            "occlusion_level": s["occlusion_level"],
            "target_occlusion": target["occlusion"] if target else "",
            "crossing_peds": sum(1 for p in s["pedestrians"] if p["role"] == "crossing"),
            "sidewalk_peds": sum(1 for p in s["pedestrians"] if p["role"] == "sidewalk"),
            "ped_types": "+".join(p["type"] for p in s["pedestrians"] if p["role"] == "crossing"),
            "cars": sum(1 for v in s["vehicles"] if v["category"] == "car"),
            "truck": any(v["category"] == "truck" for v in s["vehicles"]),
            "labeled_peds": len(peds), "labeled_vehicles": len(labels) - len(peds),
        })
    if not rows:
        return
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"images: {len(rows)}"]
    for key in ("map", "weather", "crossing", "occlusion_level", "truck", "crossing_peds", "sidewalk_peds"):
        cnt = collections.Counter(str(r[key]) for r in rows)
        lines.append(f"{key:16s} " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    types = collections.Counter(t for r in rows for t in r["ped_types"].split("+") if t)
    lines.append(f"{'ped types':16s} " + "  ".join(f"{k}={v}" for k, v in sorted(types.items())))
    lines.append(f"{'sites used':16s} {len({(r['map'], r['site']) for r in rows})}")
    skipped = os.path.join(out, "skipped.json")
    if os.path.exists(skipped):
        with open(skipped, encoding="utf-8") as f:
            lines.append(f"{'skipped':16s} {len(json.load(f))} image slots (see skipped.json)")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


# ====================================================================== worker

class ConnectionLost(Exception):
    pass


def is_connection_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ("time-out", "timeout", "timed out", "connection", "rpc", "closed"))


def exact_sequence(n, weights, rng):
    """n items split by weights with exact counts (largest remainder), shuffled."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    seq = [k for k in weights for _ in range(counts[k])]
    rng.shuffle(seq)
    return seq


def build_schedule(n, map_name):
    """Weather and occlusion level for each of a map's n image slots, exact counts.
    Each weather's slots are split by that weather's own occlusion mix."""
    rng = random.Random(f"{C.SEED}:{map_name}:schedule")
    weathers = exact_sequence(n, C.weather_shares(), rng)
    levels = [None] * n
    for w in C.WEATHERS:
        slots = [i for i, x in enumerate(weathers) if x == w]
        for i, lv in zip(slots, exact_sequence(len(slots), C.settings_for(w).occlusion, rng)):
            levels[i] = lv
    return weathers, levels


def split_evenly(total, n):
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def scene_name(map_name, i):
    return f"{map_name}_{i:05d}"


def run_worker(args):
    try:
        return Worker(args).run()
    except ConnectionLost as exc:
        print(f"  lost connection to CARLA: {exc}")
        return EXIT_NO_SERVER
    except RuntimeError as exc:
        if is_connection_error(exc):
            print(f"  lost connection to CARLA: {exc}")
            return EXIT_NO_SERVER
        raise


class Worker:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        for d in SUBDIRS:
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.log_path = os.path.join(self.out, "run_log.txt")
        self.skipped_path = os.path.join(self.out, "skipped.json")
        self.skipped = {}
        if os.path.exists(self.skipped_path):
            with open(self.skipped_path, encoding="utf-8") as f:
                self.skipped = json.load(f)

    def log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")

    def write_dataset_yaml(self):
        names = C.class_names()
        with open(os.path.join(self.out, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(self.out)}\ntrain: images\nval: images\n")
            f.write(f"nc: {len(names)}\nnames: {json.dumps(names)}\n")
        with open(os.path.join(self.out, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")

    def run(self):
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

        maps = self.resolve_maps()
        if not maps:
            print("none of the maps in config.MAPS exist on this server")
            return EXIT_FATAL
        done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.out, "labels_3d", "*.json"))}
        processed = 0

        for map_name, n in zip(maps, split_evenly(self.args.total, len(maps))):
            if self.args.plan_only:
                MapSession(self, map_name).plan_only(min(n, self.args.plan_only))
                continue
            todo = [i for i in range(n)
                    if scene_name(map_name, i) not in done and scene_name(map_name, i) not in self.skipped]
            if not todo:
                continue
            weathers, levels = build_schedule(n, map_name)
            print(f"\n{map_name}: {n - len(todo)}/{n} done, {len(todo)} to go")

            session = MapSession(self, map_name)
            try:
                try:
                    session.open(done)
                except Exception as exc:     # noqa: BLE001 - an unusable map must not stop the run
                    if is_connection_error(exc):
                        raise ConnectionLost(exc)
                    self.log(f"{map_name} unusable: {exc}\n{traceback.format_exc()}")
                    print(f"  {map_name} unusable, skipped: {exc}")
                    continue
                for k, i in enumerate(todo):
                    session.generate(i, scene_name(map_name, i), weathers[i], levels[i])
                    processed += 1
                    if (k + 1) % 25 == 0:
                        print(f"  {map_name}: {k + 1}/{len(todo)}")
                    if processed >= C.MAX_SCENES_PER_PROCESS:
                        return EXIT_RECYCLE
                    if processed % C.RELOAD_WORLD_EVERY == 0:
                        session.reload()
            finally:
                session.close()

        if not self.args.plan_only:
            write_manifest(self.out)
        return EXIT_DONE

    def resolve_maps(self):
        current = self.client.get_world().get_map().name.split("/")[-1]
        if not C.MAPS:
            return [current]
        available = {m.split("/")[-1] for m in self.client.get_available_maps()}
        for m in C.MAPS:
            if m not in available:
                print(f"  warning: map {m} not on this server, skipped")
        return [m for m in C.MAPS if m in available]


class MapSession:
    """Everything that lives for one map: world, sites, ego + cameras, planner."""

    def __init__(self, worker, map_name):
        self.w = worker
        self.carla = worker.carla
        self.client = worker.client
        self.map_name = map_name
        self.world = None
        self.ego = self.rgb = self.seg = self.depth = None

    # ---------- setup ----------

    def _set_sync(self, on):
        s = self.world.get_settings()
        s.synchronous_mode = on
        s.fixed_delta_seconds = C.FIXED_DELTA_SECONDS if on else None
        self.world.apply_settings(s)

    def _load_world(self, reload=False):
        self.world = self.client.get_world()
        self._set_sync(False)
        if reload or self.world.get_map().name.split("/")[-1] != self.map_name:
            print(f"  loading {self.map_name}")
            self.client.set_timeout(C.MAP_LOAD_TIMEOUT)
            try:
                self.world = self.client.reload_world(False) if reload else self.client.load_world(self.map_name)
            finally:
                self.client.set_timeout(C.CARLA_TIMEOUT)
        self._set_sync(True)
        self.world.tick()

    def _prepare(self):
        from crosswalk import find_crosswalk_sites, find_jaywalk_sites
        from planner import ScenePlanner
        import spawner
        carla = self.carla

        spawner.cleanup_world(self.client, self.world)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        cw, bad = find_crosswalk_sites(self.map)
        jw = find_jaywalk_sites(self.map, cw, C.JAYWALK_MIN_DIST_FROM_CROSSWALK,
                                C.JAYWALK_SITE_SPACING, C.JAYWALK_ROAD_WIDTH_RANGE)
        print(f"  {len(cw)} crosswalk sites ({bad} unusable polygons), {len(jw)} jaywalk sites")

        # parked cars baked into the map: cached once per map (calling this every
        # frame leaks memory on the server and was a known crash cause)
        self.parked = []
        for label_name, cls in C.PARKED_VEHICLE_CLASS.items():
            label = getattr(carla.CityObjectLabel, label_name, None)
            if label is not None:
                self.parked += [(o, cls, label_name) for o in self.world.get_environment_objects(label)]
        env = [(o.bounding_box.location.x, o.bounding_box.location.y, o.bounding_box.extent.x,
                o.bounding_box.extent.y, o.bounding_box.rotation.yaw) for o, _, _ in self.parked]

        pools = spawner.walker_pools(self.bp_lib)
        print("  pedestrian blueprints: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
        self.planner = ScenePlanner(self.map, cw + jw, env, pools,
                                    spawner.vehicle_pool(self.bp_lib, C.CAR_BLUEPRINTS),
                                    spawner.vehicle_pool(self.bp_lib, C.TRUCK_BLUEPRINTS))

    def open(self, done):
        import spawner
        from planner import PlanError
        import utils
        self._load_world()
        self._prepare()
        for name in sorted(done):
            if name.startswith(self.map_name + "_"):
                with open(os.path.join(self.w.out, "labels_3d", name + ".json"), encoding="utf-8") as f:
                    self.planner.register(json.load(f)["scene"])

        # ego + cameras once per map (respawning 4K sensors every frame leaks)
        rng = random.Random(f"{C.SEED}:{self.map_name}:ego")
        poses = []
        for _ in range(200):
            try:
                poses.append(self.planner.plan_ego(rng, rng.choice(self.planner.sites))[0])
            except PlanError:
                continue
            if len(poses) == 3:
                break
        if not poses:
            raise RuntimeError(f"no usable ego position on {self.map_name}")
        for pose in poses:
            try:
                self.ego, self.rgb, self.seg, self.depth, z_off = (
                    spawner.spawn_ego_and_cameras(self.world, self.bp_lib, pose))
                break
            except spawner.SpawnError:
                continue
        else:
            raise RuntimeError(f"ego vehicle could not be spawned on {self.map_name}")
        self.planner.ego_z_offset = z_off

        # bonnet mask: vehicle pixels at the bottom that stay put across poses
        segs = []
        for pose in poses:
            self.ego.set_transform(spawner.to_transform(pose, z_off))
            _, (img,) = self.capture([self.seg], C.CAPTURE_SETTLE_TICKS)
            segs.append(self.seg_tags(img))
        self.hood = utils.hood_mask(segs)

    def close(self):
        import spawner
        if self.world is None:
            return
        try:
            if self.ego is not None:
                spawner.destroy_ego_and_cameras(self.client, self.world, self.ego, [self.rgb, self.seg, self.depth])
            self._set_sync(False)
        except RuntimeError as exc:
            if is_connection_error(exc):
                return
            raise
        finally:
            self.ego = self.rgb = self.seg = self.depth = None

    def reload(self):
        print("  periodic world reload")
        self.close()
        self._load_world(reload=True)
        done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.w.out, "labels_3d", "*.json"))}
        self.open(done)

    # ---------- capture ----------

    def capture(self, cams, settle_ticks):
        """Listen, tick, and return the images from exactly the last tick."""
        qs = [queue.Queue() for _ in cams]
        for cam, q in zip(cams, qs):
            cam.listen(q.put)
        try:
            frame = None
            for _ in range(max(1, settle_ticks)):
                frame = self.world.tick()
            imgs = [self._frame_from(q, frame) for q in qs]
        finally:
            for cam in cams:
                cam.stop()
        return frame, imgs

    @staticmethod
    def _frame_from(q, frame, timeout=20.0):
        end = time.time() + timeout
        while True:
            img = q.get(timeout=max(0.1, end - time.time()))
            if img.frame == frame:
                return img
            if img.frame > frame:
                raise RuntimeError("camera skipped the capture frame")

    @staticmethod
    def seg_tags(img):
        import numpy as np
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        return arr[:, :, 2].copy()      # copy: convert() later rewrites the buffer

    # ---------- one image ----------

    def generate(self, i, name, weather, level):
        from planner import PlanError
        import utils
        reasons = collections.Counter()
        renders = 0
        for attempt in range(300):
            rng = random.Random(f"{C.SEED}:{self.map_name}:{i}:{attempt}")
            try:
                plan = self.planner.plan(rng, weather, level)
            except PlanError as exc:
                reasons[f"plan: {exc}"] += 1
                continue
            plan["seed_key"] = f"{C.SEED}:{self.map_name}:{i}:{attempt}"
            renders += 1
            try:
                ok, why = self.render(plan, name)
            except Exception as exc:     # noqa: BLE001 - one bad scene must not stop the run
                if is_connection_error(exc):
                    raise ConnectionLost(exc)
                self.w.log(f"{name} attempt {attempt}: {exc}\n{traceback.format_exc()}")
                ok, why = False, f"error: {type(exc).__name__}"
            if ok:
                self.planner.register(plan)
                return True
            reasons[f"render: {why}"] += 1
            if renders >= C.MAX_ATTEMPTS_PER_SCENE:
                break
        self.w.skipped[name] = dict(reasons)
        utils.write_json_atomic(self.w.skipped_path, self.w.skipped)
        self.w.log(f"{name} skipped: {dict(reasons)}")
        print(f"  {name} skipped after {renders} renders: {reasons.most_common(2)}")
        return False

    def render(self, plan, name):
        import numpy as np
        import cv2
        import spawner
        import utils
        carla = self.carla
        world = self.world
        actors = []
        try:
            weather = carla.WeatherParameters()
            for k, v in plan["weather"]["params"].items():
                if hasattr(weather, k):
                    setattr(weather, k, v)
            world.set_weather(weather)
            self.ego.set_transform(spawner.to_transform(plan["ego"], self.planner.ego_z_offset))

            vehicles = spawner.spawn_vehicles(world, self.bp_lib, plan["vehicles"], actors)
            for _ in range(10):         # settle on the road, then freeze in place
                world.tick()
            spawner.freeze(vehicles)
            walkers = spawner.spawn_pedestrians(world, self.bp_lib, plan["pedestrians"], actors)
            world.tick()
            spawner.start_walking(walkers, plan["pedestrians"])
            for _ in range(max(0, plan["preroll_ticks"] - C.CAPTURE_SETTLE_TICKS)):
                world.tick()

            cams = [self.rgb, self.seg] + ([self.depth] if C.SAVE_DEPTH else [])
            _, (rgb_img, seg_img, *depth_img) = self.capture(cams, C.CAPTURE_SETTLE_TICKS)
            # read every transform now: they belong to the captured tick
            cam_tf = self.rgb.get_transform()
            world_to_cam = np.array(cam_tf.get_inverse_matrix())
            objects = self._objects(plan, walkers, vehicles, cam_tf)

            # second segmentation pass with our traffic moved away -> each
            # pedestrian's unoccluded silhouette (full-body box, true occlusion %)
            for v in vehicles:
                loc = v.get_location()
                v.set_transform(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z - 200.0)))
            _, (seg_full_img,) = self.capture([self.seg], 1)

            seg = self.seg_tags(seg_img)
            seg_full = self.seg_tags(seg_full_img)
            labels = utils.compute_labels(objects, seg, seg_full, self.hood, world_to_cam, utils.intrinsics())

            # ---------- quality gates ----------
            if utils.road_fraction(seg, self.hood) < C.MIN_ROAD_FRACTION:
                return False, "camera blocked"
            target = next((lb for lb in labels if lb["meta"].get("target")), None)
            if target is None:
                return False, "target pedestrian not visible"
            if (target["full_pixels"] or 0) < C.MIN_MAIN_PED_PIXELS:
                return False, "target pedestrian too small"
            lo, hi = C.OCCLUSION_LEVELS[plan["occlusion_level"]]
            if not (lo - C.OCCLUSION_TOLERANCE <= target["occlusion"] <= hi + C.OCCLUSION_TOLERANCE):
                return False, f"occlusion {target['occlusion']:.2f} outside '{plan['occlusion_level']}'"

            # ---------- write ----------
            out = self.w.out
            s = C.RGB_SUPERSAMPLE
            rgb = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape(rgb_img.height, rgb_img.width, 4)[:, :, :3]
            if s > 1:
                rgb = cv2.resize(rgb, (C.IMAGE_W, C.IMAGE_H), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(rgb)
            seg_img.convert(carla.ColorConverter.CityScapesPalette)
            seg_color = np.frombuffer(seg_img.raw_data, dtype=np.uint8).reshape(C.IMAGE_H, C.IMAGE_W, 4)[:, :, :3]

            utils.write_image(os.path.join(out, "images", name + C.IMAGE_EXT), rgb)
            utils.write_image(os.path.join(out, "seg_raw", name + ".png"), seg)
            utils.write_image(os.path.join(out, "seg_color", name + ".png"), np.ascontiguousarray(seg_color))
            if depth_img:
                depth_m = utils.decode_depth(depth_img[0])
                utils.write_image(os.path.join(out, "depth_raw", name + ".png"), utils.depth_to_png16(depth_m))
                utils.write_image(os.path.join(out, "depth_color", name + ".png"), utils.depth_preview(depth_m))
            if C.SAVE_DEBUG_OVERLAY:
                utils.write_image(os.path.join(out, "debug", name + ".jpg"), utils.draw_overlay(rgb, labels))
            other = "visible" if C.PED_YOLO_BOX == "full" else "full"
            utils.write_text_atomic(os.path.join(out, "labels", name + ".txt"),
                                    "\n".join(utils.yolo_lines(labels, C.PED_YOLO_BOX)))
            utils.write_text_atomic(os.path.join(out, ALT_LABELS, name + ".txt"),
                                    "\n".join(utils.yolo_lines(labels, other)))
            loc, rot = cam_tf.location, cam_tf.rotation
            utils.write_json_atomic(os.path.join(out, "labels_3d", name + ".json"), {   # written last = done marker
                "image": name + C.IMAGE_EXT, "map": self.map_name,
                "image_size": [C.IMAGE_W, C.IMAGE_H], "classes": C.class_names(),
                "camera": {"K": utils.intrinsics().tolist(), "fov": C.CAMERA_FOV,
                           "location": [loc.x, loc.y, loc.z],
                           "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                           "world_to_camera": world_to_cam.tolist()},
                "yolo_pedestrian_box": C.PED_YOLO_BOX,
                "labels": labels,
                "scene": plan,
            })
            return True, "ok"
        finally:
            spawner.destroy_actors(self.client, world, actors)

    def _objects(self, plan, walkers, vehicles, cam_tf):
        """Everything that may get a label, with its 3D box at the captured tick."""
        import numpy as np
        carla = self.carla
        site = plan["site"]
        sx, sy = site["start"][:2]
        ex, ey = site["end"][:2]
        L = max(site["length"], 1e-3)
        ax, ay = (ex - sx) / L, (ey - sy) / L
        cam_right = cam_tf.get_right_vector()
        objs = []

        def box_obj(actor, kind, meta):
            tf = actor.get_transform()
            bb = actor.bounding_box
            verts = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(tf)])
            return {"kind": kind, "source": "spawned", "meta": meta, "verts": verts,
                    "center": verts.mean(axis=0).tolist(),
                    "rotation": (tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
                    "extent": (bb.extent.x, bb.extent.y, bb.extent.z)}

        for idx, (wk, p) in enumerate(zip(walkers, plan["pedestrians"])):
            loc = wk.get_location()
            progress = ((loc.x - sx) * ax + (loc.y - sy) * ay) / L
            crossing = p["role"] == "crossing"
            dx, dy = p["direction"]
            moving = p["speed"] > 0
            objs.append(box_obj(wk, "pedestrian", {
                "actor_id": wk.id, "plan_index": idx, "target": idx == plan["target_pedestrian"],
                "role": p["role"], "crossing": crossing, "type": p["type"], "blueprint": p["blueprint"],
                "speed": p["speed"], "runner": p["runner"],
                "crossing_type": site["kind"] if crossing else None,
                "crossing_progress": round(progress, 3) if crossing else None,
                "on_road": bool(crossing and 0.0 < progress < 1.0),
                "walking_direction": (("left_to_right" if dx * cam_right.x + dy * cam_right.y > 0
                                       else "right_to_left") if crossing else
                                      ("along_sidewalk" if moving else "standing")),
            }))
        for v_actor, v in zip(vehicles, plan["vehicles"]):
            objs.append(box_obj(v_actor, v["category"], {
                "actor_id": v_actor.id, "blueprint": v["blueprint"], "direction": v["direction"],
                "side": v["side"], "occluder": v.get("occluder", False)}))

        if C.LABEL_PARKED_VEHICLES:
            cl = cam_tf.location
            for obj, cls, label_name in self.parked:
                bb = obj.bounding_box
                if bb.location.distance(cl) > C.MAX_LABEL_DISTANCE or bb.extent.x <= 0 or bb.extent.y <= 0:
                    continue
                verts = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(carla.Transform())])
                objs.append({"kind": cls, "source": "parked", "meta": {"env_id": obj.id, "env_type": label_name},
                             "verts": verts, "center": [bb.location.x, bb.location.y, bb.location.z],
                             "rotation": (bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll),
                             "extent": (bb.extent.x, bb.extent.y, bb.extent.z)})
        return objs

    # ---------- dry run ----------

    def plan_only(self, n):
        """Loads the map and plans n scenes without spawning anything: shows whether
        the map has enough variety and why plans get rejected."""
        from planner import PlanError
        self._load_world()
        self._prepare()
        weathers, levels = build_schedule(n, self.map_name)
        reasons, ok = collections.Counter(), 0
        for i in range(n):
            for attempt in range(300):
                try:
                    plan = self.planner.plan(random.Random(f"{C.SEED}:{self.map_name}:{i}:{attempt}"),
                                             weathers[i], levels[i])
                except PlanError as exc:
                    reasons[str(exc)] += 1
                    continue
                self.planner.register(plan)
                ok += 1
                break
        used = sum(1 for v in self.planner.usage.values() if v)
        print(f"  planned {ok}/{n}, sites used {used}/{len(self.planner.sites)}")
        for r, c in reasons.most_common(8):
            print(f"    rejected {c:5d}x  {r}")
        self._set_sync(False)


def main():
    args = parse_args()
    if args.summary:
        write_manifest(args.out)
        return 0
    if args.worker or args.plan_only:
        return run_worker(args)
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
