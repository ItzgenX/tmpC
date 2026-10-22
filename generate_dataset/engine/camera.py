"""Planning-time geometry: an ego-aligned 2D frame, overlap checks, and a pinhole
model of the dashcam used to predict how much of a pedestrian traffic hides.
Pure Python, nothing here talks to the simulator.
"""
import math

import advanced as C
from engine.roads import dot, forward_right, sub


class Frame:
    """2D frame centred on the crossing: long = along the road ahead of the ego, lat = to its right."""

    def __init__(self, origin, yaw):
        self.o = origin
        self.f, self.r = forward_right(yaw)

    def local(self, x, y):
        d = sub((x, y), self.o)
        return dot(d, self.f), dot(d, self.r)


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
        """2D box [x0, y0, x1, y1] and mean depth of 3D corners, None if behind the camera."""
        us, vs, ds = [], [], []
        for p in corners:
            d = (p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2])
            depth = sum(a * b for a, b in zip(d, self.F))
            if depth < 0.5:
                return None
            us.append(C.IMAGE_W / 2 + self.focal * sum(a * b for a, b in zip(d, self.R)) / depth)
            vs.append(C.IMAGE_H / 2 - self.focal * sum(a * b for a, b in zip(d, self.U)) / depth)
            ds.append(depth)
        return [min(us), min(vs), max(us), max(vs)], sum(ds) / len(ds)


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
    """Fraction of the target's on-screen box hidden by nearer occluder boxes (union),
    estimated on a samples x samples grid. Only the part of the target inside the image counts."""
    if target is None:
        return 0.0
    near = [o[0] for o in occluders if o is not None and o[1] < target[1]]
    t = target[0]
    x0, y0 = max(t[0], 0.0), max(t[1], 0.0)
    x1, y1 = min(t[2], float(C.IMAGE_W)), min(t[3], float(C.IMAGE_H))
    if x1 <= x0 or y1 <= y0 or not near:
        return 0.0
    hidden = 0
    for i in range(samples):
        px = x0 + (i + 0.5) * (x1 - x0) / samples
        for j in range(samples):
            py = y0 + (j + 0.5) * (y1 - y0) / samples
            if any(o[0] <= px <= o[2] and o[1] <= py <= o[3] for o in near):
                hidden += 1
    return hidden / (samples * samples)
