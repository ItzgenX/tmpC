"""Places cars, trucks and cyclists for a scene (as plan data, nothing is spawned).

Normal traffic   : any lane, same way as the ego or oncoming, before or past the crossing.
Cars ahead       : 1-2 cars between ego and crossing, right in front of the ego (its own
                   lane) or in the lane next to it (occluders).
Turning cars     : 1-2 cars ahead of the turning ego on its turn path (occluders).
Parked car       : a car at the curb right before the pedestrian's path (occluder).
"""
import advanced as C
from engine.roads import TURN_STEP, box_corners, coverage, dot, footprint, forward_right, lane_key, rects_overlap, step_along


SIZE = {"car": C.CAR_SIZE, "truck": C.TRUCK_SIZE, "cyclist": C.CYCLIST_SIZE}


def vehicle_at(ctx, lane, s, drift):
    """Pose `s` metres along the road frame (+ = past the crossing) on `lane`, shifted `drift` to the lane's right."""
    loc = lane.transform.location
    lane_f, _ = forward_right(lane.transform.rotation.yaw)
    delta = s - ctx.frame.local(loc.x, loc.y)[0]
    aligned = dot(lane_f, ctx.frame.f) > 0
    wp = step_along(lane, abs(delta), forward=(delta > 0) == aligned)
    if wp is None:
        return None
    t = wp.transform
    _, r = forward_right(t.rotation.yaw)
    return wp, t.location.x + r[0] * drift, t.location.y + r[1] * drift


def vehicle_dict(ctx, category, wp, x, y, direction, role):
    pool = ctx.blueprints[category]
    bp_id = ctx.rng.choice(sorted(pool))
    t = wp.transform
    lo, _ = ctx.frame.local(x, y)
    return {
        "blueprint": bp_id, "category": category, "role": role,
        "color": ctx.rng.choice(pool[bp_id]) if pool[bp_id] else None,
        "x": x, "y": y, "z": t.location.z, "yaw": t.rotation.yaw, "pitch": t.rotation.pitch,
        "direction": direction, "long": round(lo, 2), "lane": lane_key(wp),
    }


def boxes_of(ctx, v):
    """On-screen boxes of a vehicle's silhouette, for occlusion estimates. A car is a low
    body over its full length plus a cabin over the middle part (not one tall block,
    which would hide far too much); a truck or cyclist is one block."""
    length, width, height = SIZE[v["category"]]
    if v["category"] != "car":
        return [ctx.cam.box(box_corners(v["x"], v["y"], v["z"], v["yaw"], length, width, height))]
    body_h, cabin_part = C.CAR_BODY_HEIGHT, C.CAR_CABIN_LENGTH_PART
    return [ctx.cam.box(box_corners(v["x"], v["y"], v["z"], v["yaw"], length, width, body_h)),
            ctx.cam.box(box_corners(v["x"], v["y"], v["z"], v["yaw"], length * cabin_part, width * 0.9, height))]


def rect_of(ctx, v):
    length, width, _ = SIZE[v["category"]]
    return footprint(ctx.frame, v["x"], v["y"], v["yaw"], length, width)


def max_drift(direction, lane, width):
    # same-direction traffic stays inside a lane treated as EGO_LANE_WIDTH wide
    if direction == "same":
        return max(0.0, C.EGO_LANE_WIDTH / 2 - width / 2)
    return min(C.ONCOMING_MAX_DRIFT, max(0.0, (lane.lane_width - width) / 2))


def _direction_of(ctx, group):
    return "same" if group == ctx.ego_group else "oncoming"


def _lanes_by_direction(ctx, direction):
    other = "opposite" if ctx.ego_group == "same" else "same"
    group = ctx.ego_group if direction == "same" else other
    if not ctx.xs[group]:
        group = other if group == ctx.ego_group else ctx.ego_group
    return group, ctx.xs[group]


# ---------- normal traffic ----------

def place_traffic(ctx, category, protect_boxes):
    """One car / truck / cyclist anywhere sensible that does not hide the target pedestrian."""
    length, width, _ = SIZE[category]
    corridor = ctx.site.half_width + C.CROSSING_CLEARANCE + length / 2
    near_lo = -(ctx.ego["distance"] - C.EGO_CLEARANCE - length / 2)

    for _ in range(60):
        want = "same" if ctx.rng.random() < C.SAME_DIRECTION_CHANCE else "oncoming"
        group, lanes = _lanes_by_direction(ctx, want)
        if not lanes:
            return None
        lane = ctx.rng.choice(lanes)
        direction = _direction_of(ctx, group)
        in_ego_lane = lane_key(lane) == ctx.ego["lane"]
        # turning ego: the part of the road before the crossing is the junction, keep it free
        near_ok = (category == "cyclist" or not in_ego_lane) and near_lo < -corridor and not ctx.turn
        if near_ok and ctx.rng.random() < C.NEAR_SIDE_PROBABILITY:
            s = ctx.rng.uniform(near_lo, -corridor)
        elif corridor < C.TRAFFIC_FAR_MAX:
            s = ctx.rng.uniform(corridor, C.TRAFFIC_FAR_MAX)
        else:
            continue

        if category == "cyclist":       # ride near the lane's outer edge
            drift = max(0.0, lane.lane_width / 2 - width / 2 - 0.3) + ctx.rng.uniform(-0.2, 0.2)
        else:
            m = max_drift(direction, lane, width)
            drift = ctx.rng.uniform(-m, m)
        placed = vehicle_at(ctx, lane, s, drift)
        if placed is None:
            continue
        wp, x, y = placed
        v = vehicle_dict(ctx, category, wp, x, y, direction, "traffic")
        lo = v["long"]
        if abs(lo) < corridor or lo < near_lo or lo > C.TRAFFIC_FAR_MAX + 5:
            continue    # curvature moved it onto the crossing / next to the ego
        rect = rect_of(ctx, v)
        if any(rects_overlap(rect, r, C.VEHICLE_GAP, 0.3) for r in ctx.rects):
            continue
        boxes = boxes_of(ctx, v)
        if any(coverage(p, boxes) > C.PROTECT_TARGET_MAX_COVER for p in protect_boxes):
            continue
        ctx.rects.append(rect)
        return v
    return None


# ---------- occluders ----------

def _occluder_lane(ctx, have_own, have_next):
    """'ego' (cars right in front of the ego) or 'next' (the lane beside it), or None.
    A layout given for the job (tests, --self-test) is used as it is."""
    want = ctx.layout.get("lane")
    if want:
        return want if {"ego": have_own, "next": have_next}[want] else None
    want = "ego" if ctx.rng.random() < C.OCCLUDER_SAME_LANE_CHANCE else "next"
    if not {"ego": have_own, "next": have_next}[want]:
        want = "next" if want == "ego" else "ego"
    return want if {"ego": have_own, "next": have_next}[want] else None


def _check_occluder(ctx, v, ped_long, rects, placed_rects):
    """Rectangle of an occluder, or None if it reaches the pedestrian's path or hits something."""
    rect = rect_of(ctx, v)
    if rect[0] + rect[2] / 2 > ped_long - 0.8:
        return None                 # no room before the pedestrian's path
    if any(rects_overlap(rect, r, C.VEHICLE_GAP, 0.3) for r in rects + placed_rects):
        return None
    return rect


def lane_occluders(ctx, categories, ped_long, rects):
    """1-2 cars between the ego and the pedestrian's path: right in front of the ego
    (ego -> car -> car -> pedestrian) or in the lane next to it."""
    ego_lo, ego_la = ctx.frame.local(ctx.ego["x"], ctx.ego["y"])
    own, beside = [], []
    for group in ("same", "opposite"):
        for lane in ctx.xs[group]:
            lo, la = ctx.frame.local(lane.transform.location.x, lane.transform.location.y)
            if abs(la - ego_la) < 1.5 and group == ctx.ego_group:
                own.append((lane, group))
            elif 2.3 < abs(la - ego_la) < 4.8:
                beside.append((lane, group))
    which = _occluder_lane(ctx, bool(own), bool(beside))
    if which is None:
        return None
    lane, group = ctx.rng.choice(own if which == "ego" else beside)
    direction = _direction_of(ctx, group)

    out, placed_rects = [], []
    if which == "ego":              # first car's rear this far ahead of the ego's centre
        s = ego_lo + C.EGO_SIZE[0] / 2 + ctx.rng.uniform(*C.SAME_LANE_GAP)
    else:
        s = ego_lo + ctx.rng.uniform(*C.ADJACENT_CARS_AHEAD)
    for category in categories:
        length, width, _ = SIZE[category]
        s += length / 2
        if s + length / 2 > ped_long - 0.8:
            return None
        m = max_drift(direction, lane, width)
        placed = vehicle_at(ctx, lane, s, ctx.rng.uniform(-m, m))
        if placed is None:
            return None
        wp, x, y = placed
        v = vehicle_dict(ctx, category, wp, x, y, direction, "occluder")
        v["occluder_lane"] = which
        rect = _check_occluder(ctx, v, ped_long, rects, placed_rects)
        if rect is None:
            return None
        out.append(v)
        placed_rects.append(rect)
        s += length / 2 + ctx.rng.uniform(1.5, 4.0)
    return out, placed_rects


def turning_occluders(ctx, categories, ped_long, rects):
    """1-2 cars ahead of the turning ego, turning the same way: on its own turn path,
    or on a turn lane right next to it (wide junctions turn with 2 lanes)."""
    turn = ctx.turn
    which = _occluder_lane(ctx, True, bool(turn["parallel"]))
    if which is None:
        return None
    path = turn["path"] if which == "ego" else ctx.rng.choice(turn["parallel"])
    ex, ey = ctx.ego["x"], ctx.ego["y"]
    i0 = min(range(len(path)), key=lambda i: (path[i][0] - ex) ** 2 + (path[i][1] - ey) ** 2)
    s = i0 * TURN_STEP + (C.EGO_SIZE[0] / 2 + ctx.rng.uniform(*C.SAME_LANE_GAP) if which == "ego"
                          else ctx.rng.uniform(*C.ADJACENT_CARS_AHEAD))

    out, placed_rects = [], []
    for category in categories:
        length = SIZE[category][0]
        s += length / 2
        i = int(round(s / TURN_STEP))
        if i >= len(path):
            return None
        x, y, _, _, wp = path[i]
        v = vehicle_dict(ctx, category, wp, x, y, "same", "occluder")
        v["occluder_lane"] = which
        rect = _check_occluder(ctx, v, ped_long, rects, placed_rects)
        if rect is None:
            return None
        out.append(v)
        placed_rects.append(rect)
        s += length / 2 + ctx.rng.uniform(*C.TURN_CAR_GAP)      # cars queue close together through a turn
    return out, placed_rects


def parked_occluder(ctx, category, ped_long, rects):
    """A car at the curb whose front ends just before the pedestrian's path."""
    choices = []
    for group in ("same", "opposite"):
        if ctx.xs[group]:
            outer = ctx.xs[group][-1]
            if lane_key(outer) != ctx.ego["lane"]:     # never parked inside the ego's own lane
                choices.append((outer, group))
    if not choices:
        return None
    lane, group = ctx.rng.choice(choices)
    length, width, _ = SIZE[category]
    target_long = ped_long - ctx.rng.uniform(*C.PARKED_NOSE_GAP) - length / 2
    drift = max(0.0, lane.lane_width / 2 - width / 2 - 0.15)     # pulled over to the curb
    placed = vehicle_at(ctx, lane, target_long, drift)
    if placed is None:
        return None
    wp, x, y = placed
    v = vehicle_dict(ctx, category, wp, x, y, _direction_of(ctx, group), "occluder")
    v["parked"] = True
    v["occluder_lane"] = "parked"
    rect = rect_of(ctx, v)
    if abs(rect[0] - target_long) > 1.0 or rect[0] + length / 2 > ped_long - 0.3:
        return None
    if any(rects_overlap(rect, r, 0.5, 0.2) for r in rects):
        return None
    return [v], [rect]
