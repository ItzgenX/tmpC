"""Checks a live CARLA server against config.py before a long run.

    python list_blueprints.py

Prints the ego blueprint, how pedestrian blueprints sort into child/woman/man,
which configured car/truck ids exist, every 4-wheeled vehicle you could add,
and which configured maps are installed.
"""
import carla

import advanced as C
from spawner import walker_type


def main():
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

    print("all 4-wheeled vehicles on this server (id, base_type):")
    for bp in sorted(lib.filter("vehicle.*"), key=lambda b: b.id):
        wheels = bp.get_attribute("number_of_wheels").as_int() if bp.has_attribute("number_of_wheels") else 4
        if wheels == 4:
            base = bp.get_attribute("base_type").as_str() if bp.has_attribute("base_type") else "-"
            print(f"  {bp.id:40s} {base}")
    print()

    available = {m.split("/")[-1] for m in client.get_available_maps()}
    check_maps = C.MAPS or [world.get_map().name.split("/")[-1]]
    print("maps:")
    for m in check_maps:
        print(f"  {m:20s} {'ok' if m in available else 'MISSING'}")
    print("  installed: " + ", ".join(sorted(available)))


if __name__ == "__main__":
    main()
