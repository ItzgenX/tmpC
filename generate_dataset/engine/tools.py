"""Helper commands: --check (server against config.py), --delete-flagged, --cleanup.
"""
import collections
import csv
import os

import advanced as C
from engine.stop import release_server


# ============================================================================
# server check  (--check)
# ============================================================================

def check_server():
    """Ego blueprint, how pedestrian models sort into child/woman/man, which configured
    car/truck/cyclist ids exist, every vehicle you could add, installed maps."""
    import carla
    from engine.carla_io import walker_type
    client = carla.Client(C.CARLA_HOST, C.CARLA_PORT)
    client.set_timeout(C.CARLA_TIMEOUT)
    world = client.get_world()
    lib = world.get_blueprint_library()
    print(f"server {client.get_server_version()}, map {world.get_map().name}\n")

    ego = lib.filter(C.EGO_BLUEPRINT)
    print(f"ego '{C.EGO_BLUEPRINT}': {ego[0].id if ego else 'NOT FOUND -> fallback ' + C.EGO_FALLBACK_BLUEPRINT}\n")

    print("pedestrians (id, age, gender -> type used):")
    counts = {}
    for bp in sorted(lib.filter("walker.pedestrian.*"), key=lambda b: b.id):
        age = bp.get_attribute("age").as_str() if bp.has_attribute("age") else "-"
        gender = bp.get_attribute("gender").as_str() if bp.has_attribute("gender") else "-"
        t = "EXCLUDED" if bp.id in C.PED_BLUEPRINT_EXCLUDE else (walker_type(bp) or "UNUSED (add to PED_TYPE_OVERRIDES)")
        counts[t] = counts.get(t, 0) + 1
        print(f"  {bp.id:28s} {age:8s} {gender:8s} -> {t}")
    print("  totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")

    def check(title, ids):
        print(title)
        for i in ids:
            print(f"  {i:40s} {'ok' if lib.filter(i) else 'MISSING'}")
        print()

    check("configured cars:", C.CAR_BLUEPRINTS)
    check("configured trucks:", C.TRUCK_BLUEPRINTS)
    check("configured cyclists / scooters:", C.CYCLIST_BLUEPRINTS)

    print("all vehicles on this server (id, wheels, base_type):")
    for bp in sorted(lib.filter("vehicle.*"), key=lambda b: b.id):
        wheels = bp.get_attribute("number_of_wheels").as_int() if bp.has_attribute("number_of_wheels") else 4
        base = bp.get_attribute("base_type").as_str() if bp.has_attribute("base_type") else "-"
        print(f"  {bp.id:40s} {wheels}  {base}")
    print()

    available = {m.split("/")[-1] for m in client.get_available_maps()}
    print("maps:")
    for m in C.MAPS or [world.get_map().name.split("/")[-1]]:
        print(f"  {m:20s} {'ok' if m in available else 'MISSING'}")
    print("  installed: " + ", ".join(sorted(available)))
    return 0


# ============================================================================
# flagged images  (--delete-flagged)
# ============================================================================
# Every output of a deleted image goes (image, depth, segmentation, box images,
# labels). Deleted rows move from flagged.csv to deleted.csv.

FLAG_FIELDS = ["image", "weather", "scene", "reasons", "planned_occlusion", "measured_occlusion"]


def _files_of(out, stem):
    found = []
    for folder in list(C.OUTPUT_FOLDERS.values()) + ["labels_3d"]:
        for ext in (".png", ".jpg", ".txt", ".json"):
            p = os.path.join(out, folder, stem + ext)
            if os.path.exists(p):
                found.append(p)
    return found


def delete_flagged(out, reason, yes):
    path = os.path.join(out, "flagged.csv")
    if not os.path.exists(path):
        print(f"no flagged.csv in {out} - nothing flagged")
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chosen = [r for r in rows if reason.lower() in r["reasons"].lower()]
    keep = [r for r in rows if r not in chosen]

    by_reason = collections.Counter(r["reasons"].split(",")[0] for r in chosen)
    print(f"{len(chosen)} of {len(rows)} flagged images match")
    for why, n in by_reason.most_common():
        print(f"  {n:5d}  {why}")
    if not yes:
        for r in chosen[:20]:
            print(f"    {r['image']}: {r['reasons']}")
        if len(chosen) > 20:
            print(f"    ... and {len(chosen) - 20} more")
        print("\nnothing deleted. look at images_2d_bbox/ first, then add --yes")
        return 0

    removed = 0
    for r in chosen:
        for p in _files_of(out, os.path.splitext(r["image"])[0]):
            os.remove(p)
            removed += 1
    deleted_path = os.path.join(out, "deleted.csv")
    new = not os.path.exists(deleted_path)
    with open(deleted_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(chosen)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAG_FIELDS)
        w.writeheader()
        w.writerows(keep)
    print(f"deleted {len(chosen)} images ({removed} files). run: python main.py --summary")
    return 0


# ============================================================================
# cleanup  (--cleanup)
# ============================================================================

def cleanup_only():
    import carla
    client = carla.Client(C.CARLA_HOST, C.CARLA_PORT)
    client.set_timeout(C.CARLA_TIMEOUT)
    try:
        release_server(client)
    except RuntimeError as exc:
        print(f"  could not reach CARLA to clean up: {exc}")
        return 1
    print("  CARLA world emptied and back in async mode")
    return 0
