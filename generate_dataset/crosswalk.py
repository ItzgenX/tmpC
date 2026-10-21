"""Finds where pedestrians cross: real marked crosswalks first, jaywalk spots as fallback.

A site is a straight crossing line across the road (start curb -> end curb) plus
the driving waypoint whose lanes the traffic and ego are planned on.
"""
import math

import carla

from sidewalk import forward_right, road_cross_section, step_along, heading_diff, dot, sub


class CrossingSite:
    def __init__(self, site_id, kind, start, end, half_width, road_wp):
        self.id = site_id
        self.kind = kind                      # "crosswalk" or "jaywalk"
        self.start = start                    # (x, y, z) on one curb
        self.end = end                        # (x, y, z) on the other curb
        self.half_width = half_width          # half the corridor width along the road
        self.road_wp = road_wp                # driving waypoint at this cross-section
        self.center = tuple((a + b) / 2.0 for a, b in zip(start, end))
        dx, dy = end[0] - start[0], end[1] - start[1]
        self.length = math.hypot(dx, dy)
        self.axis = (dx / self.length, dy / self.length)    # unit, start -> end
        f, _ = forward_right(road_wp.transform.rotation.yaw)
        self.along = f                                       # road direction

    def to_json(self):
        return {
            "id": self.id, "kind": self.kind,
            "start": [round(v, 3) for v in self.start], "end": [round(v, 3) for v in self.end],
            "length": round(self.length, 2), "half_width": round(self.half_width, 2),
        }


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
    # nearest non-junction driving lane whose road runs across the polygon's long axis
    minor = (-major[1], major[0])
    for k in (0.0, 1.5, -1.5, 3.0, -3.0, 5.0, -5.0):
        for ax in (minor, major):
            probe = carla.Location(x=center.x + ax[0] * k, y=center.y + ax[1] * k, z=center.z)
            wp = carla_map.get_waypoint(probe, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
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
        if any(math.hypot(s.center[0] - (start[0] + end[0]) / 2, s.center[1] - (start[1] + end[1]) / 2) < 2.0
               for s in sites):
            continue    # duplicate polygon
        sites.append(CrossingSite(f"cw{len(sites):03d}", "crosswalk", start, end, width / 2.0, wp))
    return sites, skipped


def _is_straight(wp, dist=15.0, max_turn=15.0):
    for fwd in (True, False):
        other = step_along(wp, dist, fwd)
        if other is None or other.is_junction:
            return False
        if heading_diff(other.transform.rotation.yaw, wp.transform.rotation.yaw) > max_turn:
            return False
    return True


def find_jaywalk_sites(carla_map, crosswalk_sites, min_dist_from_crosswalk, spacing,
                       road_width_range, max_sites=300):
    """Straight road stretches with a sidewalk on both sides, far from any crosswalk."""
    cw_centers = [s.center for s in crosswalk_sites]
    sites = []
    for wp in carla_map.generate_waypoints(4.0):
        if len(sites) >= max_sites:
            break
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        loc = wp.transform.location
        if any(math.hypot(loc.x - c[0], loc.y - c[1]) < min_dist_from_crosswalk for c in cw_centers):
            continue
        if any(math.hypot(loc.x - s.center[0], loc.y - s.center[1]) < spacing for s in sites):
            continue
        xs = road_cross_section(wp)
        if xs["same_sidewalk"] is None or xs["opposite_sidewalk"] is None:
            continue
        a, b = xs["same_sidewalk"].transform.location, xs["opposite_sidewalk"].transform.location
        width = math.hypot(a.x - b.x, a.y - b.y)
        if not (road_width_range[0] <= width <= road_width_range[1]):
            continue
        if not _is_straight(wp):
            continue
        sites.append(CrossingSite(f"jw{len(sites):03d}", "jaywalk",
                                  (a.x, a.y, a.z), (b.x, b.y, b.z), 1.5, wp))
    return sites
