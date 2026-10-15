import json
import os
import random


# the four buckets every scenario's images get split across. keys must
# match exactly - "crosswalk"/"jaywalk" x "clear"/"occluded".
BUCKETS = ["crosswalk_clear", "crosswalk_occluded", "jaywalk_clear", "jaywalk_occluded"]


def allocate_counts(n, split_pct):
    # percentages -> exact counts summing to n, largest-remainder rounding
    total_pct = sum(split_pct.values())
    if total_pct <= 0:
        return {k: 0 for k in split_pct}

    raw = {k: n * (p / total_pct) for k, p in split_pct.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(floors.values())

    by_fraction = sorted(split_pct.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in by_fraction[:remainder]:
        floors[k] += 1
    return floors


def build_image_plan(n, split_pct, allow_jaywalk, allow_occlusion):
    # shuffled list of n bucket labels. a disallowed bucket's quota folds
    # into its nearest allowed counterpart instead of getting dropped
    counts = allocate_counts(n, split_pct)
    merged = {b: 0 for b in BUCKETS}

    for bucket, count in counts.items():
        crossing, occlusion = bucket.split("_", 1)
        if crossing == "jaywalk" and not allow_jaywalk:
            crossing = "crosswalk"
        if occlusion == "occluded" and not allow_occlusion:
            occlusion = "clear"
        merged[f"{crossing}_{occlusion}"] += count

    plan = []
    for bucket, count in merged.items():
        plan.extend([bucket] * count)
    random.shuffle(plan)
    return plan


def plan_path(scenario_dir):
    return os.path.join(scenario_dir, "plan.json")


def load_or_build_plan(scenario_dir, n, split_pct, allow_jaywalk, allow_occlusion):
    # resumes from disk if a plan file exists - including an empty one,
    # which means this scenario already finished, not "build a new one"
    path = plan_path(scenario_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    plan = build_image_plan(n, split_pct, allow_jaywalk, allow_occlusion)
    save_plan(scenario_dir, plan)
    return plan


def save_plan(scenario_dir, remaining_plan):
    with open(plan_path(scenario_dir), 'w') as f:
        json.dump(remaining_plan, f)


def consume_plan_item(scenario_dir, remaining_plan):
    # pops + persists, so a crash only loses the image in flight
    remaining_plan.pop(0)
    save_plan(scenario_dir, remaining_plan)
