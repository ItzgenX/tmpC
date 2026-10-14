import json
import os
import random


# the four buckets every scenario's images get split across. keys must
# match exactly - "crosswalk"/"jaywalk" x "clear"/"occluded".
BUCKETS = ["crosswalk_clear", "crosswalk_occluded", "jaywalk_clear", "jaywalk_occluded"]


def allocate_counts(n, split_pct):
    """
    Turns percentages into exact integer counts that sum to exactly n,
    using largest-remainder so rounding doesn't lose or add an image.
    split_pct values don't need to sum to exactly 100 - they're
    normalized here - but they should be in the same ratio you intend.
    """
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
    """
    Builds a shuffled list of exactly n bucket labels for one scenario.

    If a scenario doesn't allow jaywalking or doesn't allow occlusion,
    that bucket's quota is folded into its nearest allowed counterpart
    (jaywalk_occluded -> crosswalk_occluded if no jaywalk, -> jaywalk_clear
    if no occlusion, -> crosswalk_clear if neither) rather than just being
    dropped, so every one of the n images is still accounted for.
    """
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
    """
    Resumes a scenario's remaining plan from disk if a plan file exists
    (this is what makes a crash/restart not throw away progress - the
    plan file is only ever the *remaining* work, consumed and rewritten
    one image at a time by consume_plan_item), otherwise builds a fresh
    one. An existing file that's an empty list is a real, meaningful
    state - it means this scenario already finished - and must be
    returned as-is, not treated as "nothing there yet" and rebuilt.
    """
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
    """
    Pops the next bucket off the front and persists what's left, so a
    crash right after this call still only re-does the image currently in
    flight, not the whole scenario.
    """
    remaining_plan.pop(0)
    save_plan(scenario_dir, remaining_plan)
