"""Map geometry: the lanes across a road, its sidewalks, the places where
pedestrians cross (marked crosswalks, mid-block, T-junctions, roads a car turns
into), and the planning camera that predicts what the dashcam will see.

All of this runs client-side on the OpenDRIVE map, no server round trips.
"""
import math

import carla

import advanced as C


# ============================================================================
# roads
# ============================================================================
# Road cross-section geometry: lanes across the road, sidewalks, stepping along a lane.
#
# All of this runs client-side on the OpenDRIVE map, no server round trips.

MAX_LANE_STEPS = 12


def forward_right(yaw_deg):
    # CARLA is left-handed: right of heading (cos, sin) is (-sin, cos)
    yaw = math.radians(yaw_deg)
    fx, fy = math.cos(yaw), math.sin(yaw)
    return (fx, fy), (-fy, fx)


def heading_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


# waypoints and lanes: https://carla.readthedocs.io/en/0.9.16/core_map/#waypoints
def step_along(wp, dist, forward=True):
    """Waypoint `dist` metres ahead/behind on the same lane, straightest branch at junctions."""
    if dist < 0.05:
        return wp
    cands = wp.next(dist) if forward else wp.previous(dist)
    if not cands:
        return None
    yaw = wp.transform.rotation.yaw
    return min(cands, key=lambda c: heading_diff(c.transform.rotation.yaw, yaw))


def same_side(a, b):
    return (a.lane_id > 0) == (b.lane_id > 0)


def _walk_outward(start):
    # get_right_lane() always points away from the centre line on both sides
    lanes, sidewalk = [], None
    cur = start
    for _ in range(MAX_LANE_STEPS):
        if cur is None:
            break
        if cur.lane_type == carla.LaneType.Driving:
            lanes.append(cur)
        elif cur.lane_type == carla.LaneType.Sidewalk:
            sidewalk = cur
            break
        cur = cur.get_right_lane()
    return lanes, sidewalk


def road_cross_section(wp):
    """Driving lanes on both sides of the road at wp, plus the sidewalk on each edge.

    get_left_lane() flips meaning once it crosses the centre line (the left of an
    oncoming lane points back at you), so naive left-walking ping-pongs. Instead:
    walk inward to the innermost lane on wp's side, hop the centre once, then
    walk outward on each side with get_right_lane().
    """
    inner = wp
    for _ in range(MAX_LANE_STEPS):
        left = inner.get_left_lane()
        if left is None or not same_side(left, wp):
            break
        inner = left
    across = inner.get_left_lane()
    if across is not None and same_side(across, wp):
        across = None

    same_lanes, same_sidewalk = _walk_outward(inner)
    opp_lanes, opp_sidewalk = _walk_outward(across) if across is not None else ([], None)
    return {
        "same": same_lanes,             # travel in wp's direction
        "opposite": opp_lanes,          # travel against wp's direction
        "same_sidewalk": same_sidewalk,
        "opposite_sidewalk": opp_sidewalk,
    }


def lane_key(wp):
    return f"{wp.road_id}:{wp.section_id}:{wp.lane_id}"


def ground_z(carla_map, x, y, z_hint):
    """Surface height at (x, y): road, sidewalk, whatever lane is there."""
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z_hint), project_to_road=True,
                                lane_type=carla.LaneType.Any)
    return wp.transform.location.z if wp is not None else z_hint


# ============================================================================
# sites
# ============================================================================
# Where pedestrians cross. Three kinds of site:
#
# crosswalk : a real marked crosswalk
# road      : mid-block on a straight road with sidewalks on both sides (jaywalking)
# junction  : an arm of a T-junction (3 roads meet) with no crosswalk (jaywalking)
# turn      : a road a car can turn into (left or right) at any junction; the crossing
#             is the crosswalk just past the junction, or a line across the road there
#
# A site is a straight crossing line across the road (start curb -> end curb)
# plus the driving waypoint whose lanes the traffic and ego are planned on.

class CrossingSite:
    def __init__(self, site_id, kind, start, end, half_width, road_wp):
        self.id = site_id
        self.kind = kind
        self.start = start                    # (x, y, z) on one curb
        self.end = end                        # (x, y, z) on the other curb
        self.half_width = half_width          # half the walking corridor width, along the road
        self.road_wp = road_wp                # driving waypoint at this cross-section
        self.center = tuple((a + b) / 2.0 for a, b in zip(start, end))
        dx, dy = end[0] - start[0], end[1] - start[1]
        self.length = math.hypot(dx, dy)
        self.axis = (dx / self.length, dy / self.length)    # unit, start -> end
        self.along, _ = forward_right(road_wp.transform.rotation.yaw)
        self.turns = []                       # turn sites: the junction lanes that turn into this road

    def to_json(self):
        return {
            "id": self.id, "kind": self.kind,
            "start": [round(v, 3) for v in self.start], "end": [round(v, 3) for v in self.end],
            "length": round(self.length, 2), "half_width": round(self.half_width, 2),
        }


def find_all_sites(carla_map, cfg):
    cw, bad = find_crosswalk_sites(carla_map)
    road = find_road_sites(carla_map, cw, cfg.JAYWALK_MIN_DIST_FROM_CROSSWALK,
                           cfg.JAYWALK_SITE_SPACING, cfg.JAYWALK_ROAD_WIDTH_RANGE)
    junction = find_junction_sites(carla_map, cw, cfg.JUNCTION_MIN_DIST_FROM_CROSSWALK,
                                   cfg.JUNCTION_ARM_OFFSET, cfg.JAYWALK_ROAD_WIDTH_RANGE)
    turn = find_turn_sites(carla_map, cw, cfg.JUNCTION_ARM_OFFSET, cfg.JAYWALK_ROAD_WIDTH_RANGE,
                           cfg.TURN_ANGLE_RANGE, cfg.TURN_FAR_CROSSING)
    print(f"  sites: {len(cw)} crosswalk ({bad} unusable polygons), {len(road)} mid-block, "
          f"{len(junction)} T-junction, {len(turn)} turn")
    return cw + road + junction + turn


# ---------- marked crosswalks ----------

# https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Map.get_crosswalks
def parse_crosswalk_polygons(carla_map):
    # get_crosswalks() is a flat point list; each polygon repeats its first point to close
    polys, cur = [], []
    for p in carla_map.get_crosswalks():
        cur.append(p)
        if len(cur) > 2 and p.distance(cur[0]) < 1e-3:
            if len(cur) - 1 >= 3:
                polys.append(cur[:-1])
            cur = []
    return polys


def _principal_axis(pts):
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    sxx = sum((p.x - cx) ** 2 for p in pts)
    syy = sum((p.y - cy) ** 2 for p in pts)
    sxy = sum((p.x - cx) * (p.y - cy) for p in pts)
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
    return (math.cos(ang), math.sin(ang)), math.sqrt(l1 / l2)


def _find_road_wp(carla_map, center, major, anisotropy):
    """Driving lane whose road runs across the polygon's long axis. Prefers a lane
    outside any junction (crosswalks often sit right at a junction's edge)."""
    minor = (-major[1], major[0])
    for allow_junction in (False, True):
        for k in (0.0, 1.5, -1.5, 3.0, -3.0, 5.0, -5.0, 8.0, -8.0, 11.0, -11.0):
            for ax in (minor, major):
                probe = carla.Location(x=center.x + ax[0] * k, y=center.y + ax[1] * k, z=center.z)
                wp = carla_map.get_waypoint(probe, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None or (wp.is_junction and not allow_junction):
                    continue
                if wp.transform.location.distance(probe) > 4.0:
                    continue
                f, _ = forward_right(wp.transform.rotation.yaw)
                if anisotropy > 1.3 and abs(dot(f, major)) > 0.35:
                    continue
                return wp
    return None


def find_crosswalk_sites(carla_map):
    sites, skipped = [], 0
    for poly in parse_crosswalk_polygons(carla_map):
        c = carla.Location(x=sum(p.x for p in poly) / len(poly),
                           y=sum(p.y for p in poly) / len(poly),
                           z=sum(p.z for p in poly) / len(poly))
        major, anis = _principal_axis(poly)
        wp = _find_road_wp(carla_map, c, major, anis)
        if wp is None:
            skipped += 1
            continue
        f, r = forward_right(wp.transform.rotation.yaw)
        across = [dot(sub((p.x, p.y), (c.x, c.y)), r) for p in poly]
        along = [dot(sub((p.x, p.y), (c.x, c.y)), f) for p in poly]
        length, width = max(across) - min(across), max(along) - min(along)
        if not (4.0 <= length <= 40.0 and 1.0 <= width <= 12.0):
            skipped += 1
            continue
        mid_f = (max(along) + min(along)) / 2.0
        bx, by = c.x + f[0] * mid_f, c.y + f[1] * mid_f
        start = (bx + r[0] * min(across), by + r[1] * min(across), c.z)
        end = (bx + r[0] * max(across), by + r[1] * max(across), c.z)
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if any(math.hypot(s.center[0] - mid[0], s.center[1] - mid[1]) < 2.0 for s in sites):
            continue    # duplicate polygon
        sites.append(CrossingSite(f"cw{len(sites):03d}", "crosswalk", start, end, width / 2.0, wp))
    return sites, skipped


# ---------- mid-block (jaywalking) ----------

def _is_straight(wp, dist=15.0, max_turn=15.0):
    for fwd in (True, False):
        other = step_along(wp, dist, fwd)
        if other is None or other.is_junction:
            return False
        if heading_diff(other.transform.rotation.yaw, wp.transform.rotation.yaw) > max_turn:
            return False
    return True


def _curb_to_curb(wp, width_range):
    """Sidewalk centre on each side of the road at wp, or None."""
    xs = road_cross_section(wp)
    if xs["same_sidewalk"] is None or xs["opposite_sidewalk"] is None:
        return None
    a, b = xs["same_sidewalk"].transform.location, xs["opposite_sidewalk"].transform.location
    width = math.hypot(a.x - b.x, a.y - b.y)
    if not (width_range[0] <= width <= width_range[1]):
        return None
    return (a.x, a.y, a.z), (b.x, b.y, b.z)


def _far_from(x, y, centers, dist):
    return all(math.hypot(x - c[0], y - c[1]) >= dist for c in centers)


def find_road_sites(carla_map, crosswalk_sites, min_dist_from_crosswalk, spacing, width_range, max_sites=300):
    """Straight road stretches with a sidewalk on both sides, far from any crosswalk."""
    cw = [s.center for s in crosswalk_sites]
    sites = []
    for wp in carla_map.generate_waypoints(4.0):
        if len(sites) >= max_sites:
            break
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        loc = wp.transform.location
        if not _far_from(loc.x, loc.y, cw, min_dist_from_crosswalk):
            continue
        if not _far_from(loc.x, loc.y, [s.center for s in sites], spacing):
            continue
        ends = _curb_to_curb(wp, width_range)
        if ends is None or not _is_straight(wp):
            continue
        sites.append(CrossingSite(f"rd{len(sites):03d}", "road", ends[0], ends[1], 1.5, wp))
    return sites


# ---------- T-junctions (jaywalking) ----------

def find_junction_sites(carla_map, crosswalk_sites, min_dist_from_crosswalk, arm_offset, width_range):
    """One crossing line across each arm of every 3-way junction, where there is no crosswalk."""
    cw = [s.center for s in crosswalk_sites]
    seen, sites = set(), []
    for wp in carla_map.generate_waypoints(3.0):
        if not wp.is_junction:
            continue
        junction = wp.get_junction()
        if junction is None or junction.id in seen:
            continue
        seen.add(junction.id)

        # arms = the roads touching the junction, found from each lane path through it
        arms = {}
        for a, b in junction.get_waypoints(carla.LaneType.Driving):
            entry = step_along(a, arm_offset, forward=False)   # on the road leading in
            exit_ = step_along(b, arm_offset, forward=True)    # on the road leading out
            for w in (entry, exit_):
                if w is not None and not w.is_junction:
                    arms.setdefault(w.road_id, w)
        if len(arms) != 3:
            continue

        for road_id, arm in sorted(arms.items()):
            loc = arm.transform.location
            if not _far_from(loc.x, loc.y, cw, min_dist_from_crosswalk):
                continue    # this arm already has a crosswalk
            ends = _curb_to_curb(arm, width_range)
            if ends is None:
                continue
            sites.append(CrossingSite(f"tj{junction.id}_{road_id}", "junction", ends[0], ends[1], 1.5, arm))
    return sites


# ---------- roads a car turns into (any junction) ----------

TURN_STEP = 0.5             # metres between the points of a turn path


def _crosswalk_past(wp, crosswalk_sites, search=15.0):
    """The marked crosswalk across wp's road just past the junction, or None."""
    loc = wp.transform.location
    f, r = forward_right(wp.transform.rotation.yaw)
    best = None
    for s in crosswalk_sites:
        d = sub(s.center[:2], (loc.x, loc.y))
        ahead = dot(d, f)
        if -6.0 <= ahead <= search and abs(dot(d, r)) < 12.0 and abs(dot(s.along, f)) > 0.8:
            if best is None or abs(ahead) < abs(best[0]):
                best = (ahead, s)
    return best[1] if best else None


# junction lanes: https://carla.readthedocs.io/en/0.9.16/core_map/#junctions
def find_turn_sites(carla_map, crosswalk_sites, arm_offset, width_range, angle_range, far_offset):
    """Two sites per road leaving a junction that cars can turn into: the crossing at the
    junction exit (its crosswalk, or a line arm_offset m into the road) and a line far_offset
    m into the road (people crossing a little past the junction, where the turning cars are
    straight again). Their `turns` are the junction lanes turning into that road:
    {"start", "end" (junction waypoints), "side", "entry_road"}."""
    exits, seen = {}, set()
    for wp in carla_map.generate_waypoints(3.0):
        if not wp.is_junction:
            continue
        junction = wp.get_junction()
        if junction is None or junction.id in seen:
            continue
        seen.add(junction.id)
        for a, b in junction.get_waypoints(carla.LaneType.Driving):
            turn = (b.transform.rotation.yaw - a.transform.rotation.yaw + 180.0) % 360.0 - 180.0
            if not angle_range[0] <= abs(turn) <= angle_range[1]:
                continue
            entry = step_along(a, 2.0, forward=False)
            out = step_along(b, arm_offset, forward=True)
            if entry is None or out is None or out.is_junction:
                continue
            e = exits.setdefault((junction.id, out.road_id), {"wp": out, "turns": []})
            # CARLA is left-handed: a growing yaw turns right
            e["turns"].append({"start": a, "end": b, "side": "right" if turn > 0 else "left",
                               "entry_road": entry.road_id})

    sites = []
    for (jid, road_id), e in sorted(exits.items()):
        lines = []
        cw = _crosswalk_past(e["wp"], crosswalk_sites)
        if cw is not None:
            lines.append(("near", cw.start, cw.end, cw.half_width, e["wp"]))
        else:
            ends = _curb_to_curb(e["wp"], width_range)
            if ends is not None:
                lines.append(("near", ends[0], ends[1], 1.5, e["wp"]))
        far = step_along(e["wp"], far_offset - arm_offset, forward=True)
        if far is not None and not far.is_junction:
            ends = _curb_to_curb(far, width_range)
            if ends is not None:
                lines.append(("far", ends[0], ends[1], 1.5, far))
        for name, start, end, half, wp in lines:
            site = CrossingSite(f"tn{jid}_{road_id}_{name}", "turn", start, end, half, wp)
            site.turns = e["turns"]
            sites.append(site)
    return sites


def turn_path(start, end, back, ahead, step=TURN_STEP):
    """Points (x, y, z, yaw, waypoint) every `step` m along one turn: from `back` m before
    the junction, along the junction lane start -> end, to `ahead` m past the junction."""
    before, wp = [], start
    for _ in range(int(back / step)):
        wp = step_along(wp, step, forward=False)
        if wp is None:
            break
        before.append(wp)
    pts, wp, past = list(reversed(before)) + [start], start, None
    end_loc = end.transform.location
    for _ in range(4000):
        nxt = wp.next(step)
        if not nxt:
            break
        if past is None:            # still inside the junction: stay on the lane that leads to `end`
            wp = min(nxt, key=lambda c: c.transform.location.distance(end_loc))
            if not wp.is_junction or wp.transform.location.distance(end_loc) < step:
                past = 0.0
        else:
            yaw = wp.transform.rotation.yaw
            wp = min(nxt, key=lambda c: heading_diff(c.transform.rotation.yaw, yaw))
            past += step
            if past > ahead:
                break
        pts.append(wp)
    return [(w.transform.location.x, w.transform.location.y, w.transform.location.z,
             w.transform.rotation.yaw, w) for w in pts]


# ============================================================================
# planning camera
# ============================================================================
# An ego-aligned 2D frame, overlap checks, and a pinhole model of the dashcam used to
# predict how much of a pedestrian traffic hides. Pure math: nothing here spawns
# anything or asks the server.

class Frame:
    """2D frame centred on the crossing: long = along the road ahead of the ego, lat = to its right."""

    def __init__(self, origin, yaw):
        self.o = origin
        self.f, self.r = forward_right(yaw)

    def local(self, x, y):
        d = sub((x, y), self.o)
        return dot(d, self.f), dot(d, self.r)


def footprint(frame, x, y, yaw, length, width):
    """(long, lat, length, width) rectangle in `frame` that holds a box turned by `yaw`."""
    lo, la = frame.local(x, y)
    f, _ = forward_right(yaw)
    c, s = abs(dot(f, frame.f)), abs(dot(f, frame.r))
    return lo, la, length * c + width * s, length * s + width * c


def rects_overlap(a, b, gap_long, gap_lat):
    # a, b = (long, lat, length, width), axis-aligned in the frame
    return (abs(a[0] - b[0]) < (a[2] + b[2]) / 2 + gap_long and
            abs(a[1] - b[1]) < (a[3] + b[3]) / 2 + gap_lat)


class PlanCamera:
    """The dashcam at a planned ego pose (road roll ignored)."""

    def __init__(self, ego, z_offset):
        yaw = math.radians(ego["yaw"])
        pitch = math.radians(ego["pitch"] + C.CAMERA_PITCH)
        cx, cy, cz = C.CAMERA_LOCATION
        fx, fy = math.cos(yaw), math.sin(yaw)
        self.pos = (ego["x"] + fx * cx - fy * cy, ego["y"] + fy * cx + fx * cy, ego["z"] + z_offset + cz)
        cp, sp = math.cos(pitch), math.sin(pitch)
        self.F = (cp * fx, cp * fy, sp)
        self.R = (-fy, fx, 0.0)
        self.U = (-sp * fx, -sp * fy, cp)
        self.focal = C.IMAGE_W / (2.0 * math.tan(math.radians(C.CAMERA_FOV) / 2.0))

    def box(self, corners):
        """2D box [x0, y0, x1, y1], mean depth and on-screen outline (convex polygon) of
        3D corners, or None if behind the camera."""
        us, vs, ds = [], [], []
        for p in corners:
            d = (p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2])
            depth = sum(a * b for a, b in zip(d, self.F))
            if depth < 0.5:
                return None
            us.append(C.IMAGE_W / 2 + self.focal * sum(a * b for a, b in zip(d, self.R)) / depth)
            vs.append(C.IMAGE_H / 2 - self.focal * sum(a * b for a, b in zip(d, self.U)) / depth)
            ds.append(depth)
        return [min(us), min(vs), max(us), max(vs)], sum(ds) / len(ds), _hull(list(zip(us, vs)))


def _hull(points):
    """Convex hull (counter-clockwise) of 2D points."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def _inside(poly, x, y):
    """Point inside a convex counter-clockwise polygon."""
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0:
            return False
    return n >= 3


def box_corners(x, y, z, yaw_deg, length, width, height):
    f, r = forward_right(yaw_deg)
    out = []
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            px = x + f[0] * a * length + r[0] * b * width
            py = y + f[1] * a * length + r[1] * b * width
            out += [(px, py, z), (px, py, z + height)]
    return out


def in_view(box):
    if box is None:
        return False
    b = box[0]
    return b[2] > 0 and b[0] < C.IMAGE_W and b[3] > 0 and b[1] < C.IMAGE_H


def coverage(target, occluders, samples=12):
    """Fraction of the target's on-screen box hidden by nearer occluders (union of their
    on-screen outlines), estimated on a samples x samples grid. Only the part of the
    target inside the image counts."""
    if target is None:
        return 0.0
    t = target[0]
    x0, y0 = max(t[0], 0.0), max(t[1], 0.0)
    x1, y1 = min(t[2], float(C.IMAGE_W)), min(t[3], float(C.IMAGE_H))
    # only nearer occluders whose box actually touches the target's box (fast path: none)
    near = [(o[0], o[2]) for o in occluders if o is not None and o[1] < target[1]
            and o[0][0] < x1 and o[0][2] > x0 and o[0][1] < y1 and o[0][3] > y0]
    if x1 <= x0 or y1 <= y0 or not near:
        return 0.0
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if any(all(_inside(hull, *c) for c in corners) for _, hull in near):
        return 1.0                                  # one occluder covers it completely
    hidden = 0
    for i in range(samples):
        px = x0 + (i + 0.5) * (x1 - x0) / samples
        for j in range(samples):
            py = y0 + (j + 0.5) * (y1 - y0) / samples
            if any(b[0] <= px <= b[2] and b[1] <= py <= b[3] and _inside(hull, px, py) for b, hull in near):
                hidden += 1
    return hidden / (samples * samples)
