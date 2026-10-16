import json
import os
import random


# the four buckets every scenario's images split across
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


def build_remaining_counts(n, split_pct, allow_jaywalk, allow_occlusion):
    # remaining count per bucket, disallowed buckets fold into allowed ones
    counts = allocate_counts(n, split_pct)
    merged = {b: 0 for b in BUCKETS}

    for bucket, count in counts.items():
        crossing, occlusion = bucket.split("_", 1)
        if crossing == "jaywalk" and not allow_jaywalk:
            crossing = "crosswalk"
        if occlusion == "occluded" and not allow_occlusion:
            occlusion = "clear"
        merged[f"{crossing}_{occlusion}"] += count

    return merged


def plan_path(scenario_dir):
    return os.path.join(scenario_dir, "plan.json")


def load_or_build_plan(scenario_dir, n, split_pct, allow_jaywalk, allow_occlusion):
    # resumes from disk if present; all-zero counts means already finished
    path = plan_path(scenario_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    remaining = build_remaining_counts(n, split_pct, allow_jaywalk, allow_occlusion)
    save_plan(scenario_dir, remaining)
    return remaining


def save_plan(scenario_dir, remaining):
    with open(plan_path(scenario_dir), 'w') as f:
        json.dump(remaining, f, indent=2)


def total_remaining(remaining):
    return sum(remaining.values())


def pick_next_bucket(remaining):
    # weighted by remaining count - same distribution as popping from the
    # front of a shuffled list of repeated bucket names, without storing
    # one entry per image
    buckets = [b for b, c in remaining.items() if c > 0]
    weights = [remaining[b] for b in buckets]
    return random.choices(buckets, weights=weights)[0]


def consume_plan_item(scenario_dir, remaining, bucket):
    # decrements + persists, so a crash only loses the image in flight
    remaining[bucket] -= 1
    save_plan(scenario_dir, remaining)
