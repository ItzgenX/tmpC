"""A small stand-in for the `carla` module, so the tests run without a CARLA server.

Map "FakeCity": a 4-way junction at (0, 0) with four straight roads (arms) of 150 m.
Every road has 2 lanes each way (3.5 m) and a sidewalk on both edges. Inside the
junction, lanes connect the roads: straight on (both lanes), right turns (outer lane)
and left turns (inner lane), drawn as smooth curves. Marked crosswalks: just past the
junction on the east and south roads, and one mid-block on the east road.

The simulator moves walkers by their WalkerControl every tick and "renders" each camera
image: the ground is road, every actor is its bounding box (a car: low body + cabin)
filled with its semantic tag, nearest in front, with matching depth. That is enough for the real label code to measure
boxes and occlusion. The ego (the cameras' parent) and actors moved underground are not drawn.
"""
import fnmatch
import itertools
import math

import cv2
import numpy as np


# ============================================================================
# basic types
# ============================================================================

class LaneType:
    Driving, Sidewalk, Shoulder, Parking, Any = 1, 2, 4, 8, -1


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Location(Vector3D):
    def distance(self, o):
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


def _rot_matrix(rot):
    """UE / CARLA rotation matrix (columns: forward, right, up in world coordinates)."""
    cy, sy = math.cos(math.radians(rot.yaw)), math.sin(math.radians(rot.yaw))
    cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
    cr, sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr]]


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_matrix(self):
        R, loc = _rot_matrix(self.rotation), self.location
        return [R[0] + [loc.x], R[1] + [loc.y], R[2] + [loc.z], [0.0, 0.0, 0.0, 1.0]]

    def get_inverse_matrix(self):
        return np.linalg.inv(np.array(self.get_matrix())).tolist()

    def get_right_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(-math.sin(y), math.cos(y), 0.0)


class BoundingBox:
    def __init__(self, location, extent, rotation=None):
        self.location, self.extent, self.rotation = location, extent, rotation or Rotation()

    def get_world_vertices(self, tf):
        R = _rot_matrix(tf.rotation)
        out = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    p = (self.location.x + sx * self.extent.x, self.location.y + sy * self.extent.y,
                         self.location.z + sz * self.extent.z)
                    w = [sum(R[i][k] * p[k] for k in range(3)) for i in range(3)]
                    out.append(Location(w[0] + tf.location.x, w[1] + tf.location.y, w[2] + tf.location.z))
        return out


# ============================================================================
# map
# ============================================================================

J = 14.0                                    # the junction is the square |x|, |y| <= J
ARM_LEN = 150.0
LANE_OFFSET = {1: 1.75, 2: 5.25, 3: 8.0}    # |lane id| -> distance from the centre line
SIDEWALK_Z = 0.15


def _yaw_of(d):
    return math.degrees(math.atan2(d[1], d[0]))


class Arm:
    """A straight road leaving the junction in direction u. Lanes with negative ids
    drive away from the junction (on the right of u), positive ids drive towards it."""

    def __init__(self, road_id, u):
        self.road_id, self.u = road_id, u
        self.r = (-u[1], u[0])              # right of u (CARLA is left-handed)
        self.o = (u[0] * J, u[1] * J)

    def lateral(self, lane_id):
        return LANE_OFFSET[abs(lane_id)] * (1 if lane_id < 0 else -1)

    def point(self, lane_id, s):
        lat = self.lateral(lane_id)
        return (self.o[0] + self.u[0] * s + self.r[0] * lat, self.o[1] + self.u[1] * s + self.r[1] * lat)


class Connector:
    """A lane inside the junction, from an arm's incoming lane to another arm's outgoing lane."""

    def __init__(self, index, arm_in, lane_in, arm_out, lane_out):
        self.index, self.arm_in, self.lane_in, self.arm_out, self.lane_out = index, arm_in, lane_in, arm_out, lane_out
        p0, p1 = arm_in.point(lane_in, 0.0), arm_out.point(lane_out, 0.0)
        h0, h1 = (-arm_in.u[0], -arm_in.u[1]), arm_out.u
        den = h0[0] * h1[1] - h0[1] * h1[0]
        if abs(den) < 1e-9:                 # straight on: a line
            q = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        else:                               # a turn: control point where both lane lines meet
            t = ((p1[0] - p0[0]) * h1[1] - (p1[1] - p0[1]) * h1[0]) / den
            q = (p0[0] + h0[0] * t, p0[1] + h0[1] * t)
        ts = np.linspace(0.0, 1.0, 400)
        self.pts = np.stack([(1 - ts) ** 2 * p0[0] + 2 * (1 - ts) * ts * q[0] + ts ** 2 * p1[0],
                             (1 - ts) ** 2 * p0[1] + 2 * (1 - ts) * ts * q[1] + ts ** 2 * p1[1]], axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(self.pts, axis=0).T))])
        self.length = float(self.cum[-1])

    def at(self, t):
        t = min(max(t, 0.0), self.length)
        i = int(min(np.searchsorted(self.cum, t), len(self.cum) - 1))
        i0, i1 = max(i - 1, 0), max(i, 1)
        a = (t - self.cum[i0]) / max(self.cum[i1] - self.cum[i0], 1e-9)
        x = self.pts[i0, 0] + (self.pts[i1, 0] - self.pts[i0, 0]) * a
        y = self.pts[i0, 1] + (self.pts[i1, 1] - self.pts[i0, 1]) * a
        d = self.pts[i1] - self.pts[i0]
        return x, y, _yaw_of(d)


class Waypoint:
    """A point on an arm's lane (s = metres from the junction edge)."""
    is_junction = False
    section_id = 0

    def __init__(self, arm, lane_id, s):
        self.arm, self.lane_id, self.s = arm, lane_id, s
        self.road_id = arm.road_id
        sidewalk = abs(lane_id) == 3
        self.lane_type = LaneType.Sidewalk if sidewalk else LaneType.Driving
        self.lane_width = 2.0 if sidewalk else 3.5
        x, y = arm.point(lane_id, s)
        heading = arm.u if lane_id < 0 else (-arm.u[0], -arm.u[1])
        self.transform = Transform(Location(x, y, SIDEWALK_Z if sidewalk else 0.0), Rotation(yaw=_yaw_of(heading)))

    def _at(self, s):
        return Waypoint(self.arm, self.lane_id, s)

    def next(self, d):
        if self.lane_id < 0:                # driving away from the junction
            return [self._at(self.s + d)] if self.s + d <= ARM_LEN else []
        if self.s - d >= 0:
            return [self._at(self.s - d)]
        if self.lane_type != LaneType.Driving:
            return []
        rest = d - self.s                   # into the junction: every lane starting here
        return [JunctionWaypoint(c, rest) for c in MAP.connectors if c.arm_in is self.arm and c.lane_in == self.lane_id]

    def previous(self, d):
        if self.lane_id > 0:
            return [self._at(self.s + d)] if self.s + d <= ARM_LEN else []
        if self.s - d >= 0:
            return [self._at(self.s - d)]
        if self.lane_type != LaneType.Driving:
            return []
        rest = d - self.s
        return [JunctionWaypoint(c, c.length - rest) for c in MAP.connectors
                if c.arm_out is self.arm and c.lane_out == self.lane_id]

    def get_left_lane(self):
        nid = {-1: 1, 1: -1}.get(self.lane_id, self.lane_id + 1 if self.lane_id < 0 else self.lane_id - 1)
        return self._with(nid)

    def get_right_lane(self):
        nid = self.lane_id - 1 if self.lane_id < 0 else self.lane_id + 1
        return self._with(nid) if abs(nid) in LANE_OFFSET else None

    def _with(self, lane_id):
        return Waypoint(self.arm, lane_id, self.s)

    def get_junction(self):
        return None


class JunctionWaypoint:
    """A point `t` metres along a lane inside the junction."""
    is_junction = True
    section_id = 0
    lane_id = -1
    lane_type = LaneType.Driving
    lane_width = 3.5

    def __init__(self, conn, t):
        self.conn, self.t = conn, min(max(t, 0.0), conn.length)
        self.road_id = 100 + conn.index
        x, y, yaw = conn.at(self.t)
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=yaw))

    def next(self, d):
        if self.t + d <= self.conn.length:
            return [JunctionWaypoint(self.conn, self.t + d)]
        s = self.t + d - self.conn.length
        return [Waypoint(self.conn.arm_out, self.conn.lane_out, s)] if s <= ARM_LEN else []

    def previous(self, d):
        if self.t - d >= 0:
            return [JunctionWaypoint(self.conn, self.t - d)]
        return [Waypoint(self.conn.arm_in, self.conn.lane_in, d - self.t)]

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None

    def get_junction(self):
        return MAP.junction


class Junction:
    id = 1

    def get_waypoints(self, lane_type):
        return [(JunctionWaypoint(c, 0.0), JunctionWaypoint(c, c.length)) for c in MAP.connectors]


class Map:
    name = "Carla/Maps/FakeCity"

    def __init__(self):
        self.arms = [Arm(1, (1.0, 0.0)), Arm(2, (0.0, 1.0)), Arm(3, (-1.0, 0.0)), Arm(4, (0.0, -1.0))]
        self.junction = Junction()
        self.connectors = []
        for a in self.arms:
            for b in self.arms:
                if a is b:
                    continue
                cross = (-a.u[0]) * b.u[1] - (-a.u[1]) * b.u[0]    # > 0: right turn (left-handed)
                if abs(cross) < 1e-9:
                    pairs = [(1, -1), (2, -2)]
                elif cross > 0:
                    pairs = [(2, -2)]                              # right turn from the outer lane
                else:
                    pairs = [(1, -1)]                              # left turn from the inner lane
                for lin, lout in pairs:
                    self.connectors.append(Connector(len(self.connectors), a, lin, b, lout))

    def _lane_ids(self, lane_type):
        if lane_type == LaneType.Driving:
            return [-1, -2, 1, 2]
        if lane_type == LaneType.Sidewalk:
            return [-3, 3]
        return [-1, -2, -3, 1, 2, 3]

    def get_waypoint(self, loc, project_to_road=True, lane_type=LaneType.Driving):
        if abs(loc.x) <= J and abs(loc.y) <= J and lane_type != LaneType.Sidewalk:
            best = None
            for c in self.connectors:
                d = np.hypot(c.pts[:, 0] - loc.x, c.pts[:, 1] - loc.y)
                i = int(d.argmin())
                if best is None or d[i] < best[0]:
                    best = (d[i], c, float(c.cum[i]))
            return JunctionWaypoint(best[1], best[2])
        arm, s, lat = None, 0.0, 0.0
        for a in self.arms:
            sa = (loc.x - a.o[0]) * a.u[0] + (loc.y - a.o[1]) * a.u[1]
            la = (loc.x - a.o[0]) * a.r[0] + (loc.y - a.o[1]) * a.r[1]
            score = abs(la) + max(0.0, -sa) * 10
            if arm is None or score < arm[0]:
                arm = (score, a, sa, la)
        _, a, s, lat = arm
        lid = min(self._lane_ids(lane_type), key=lambda i: abs(a.lateral(i) - lat))
        wp = Waypoint(a, lid, min(max(s, 0.0), ARM_LEN))
        if not project_to_road and abs(a.lateral(lid) - lat) > wp.lane_width / 2:
            return None
        return wp

    def get_crosswalks(self):
        pts = []
        for arm, s0 in ((self.arms[0], 2.0), (self.arms[1], 2.0), (self.arms[0], 100.0)):
            corners = []
            for s, lat in ((s0, -7.2), (s0 + 4.0, -7.2), (s0 + 4.0, 7.2), (s0, 7.2)):
                x = arm.o[0] + arm.u[0] * s + arm.r[0] * lat
                y = arm.o[1] + arm.u[1] * s + arm.r[1] * lat
                corners.append(Location(x, y, 0.0))
            pts += corners + [Location(corners[0].x, corners[0].y, 0.0)]
        return pts

    def generate_waypoints(self, d):
        out = []
        for a in self.arms:
            for lid in (-1, -2, 1, 2):
                s = 0.0
                while s <= ARM_LEN:
                    out.append(Waypoint(a, lid, s))
                    s += d
        for c in self.connectors:
            t = 0.0
            while t <= c.length:
                out.append(JunctionWaypoint(c, t))
                t += d
        return out


MAP = Map()


# ============================================================================
# blueprints and actors
# ============================================================================

class _Attr:
    def __init__(self, v, rec=()):
        self.v, self.recommended_values = v, list(rec)

    def as_str(self):
        return str(self.v)

    def as_int(self):
        return int(self.v)


class Blueprint:
    def __init__(self, bp_id, attrs=None):
        self.id, self.attrs = bp_id, dict(attrs or {})

    def has_attribute(self, k):
        return k in self.attrs

    def get_attribute(self, k):
        return self.attrs[k]

    def set_attribute(self, k, v):
        if k in self.attrs:
            self.attrs[k].v = v
        else:
            self.attrs[k] = _Attr(v)


class BlueprintLibrary(list):
    def filter(self, pattern):
        return BlueprintLibrary(b for b in self if fnmatch.fnmatch(b.id, pattern) or fnmatch.fnmatch(b.id, "*" + pattern))

    def find(self, bp_id):
        return next(b for b in self if b.id == bp_id)


CARS = ["vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2020"]
TRUCKS = ["vehicle.carlamotors.european_hgv"]
CYCLES = ["vehicle.bh.crossbike", "vehicle.vespa.zx125"]


def _library():
    lib = BlueprintLibrary()
    people = [("adult", "male"), ("adult", "female"), ("child", "male"), ("adult", "female")]
    for i, (age, gender) in enumerate(people):
        lib.append(Blueprint(f"walker.pedestrian.{i + 1:04d}", {"age": _Attr(age), "gender": _Attr(gender)}))
    for v in CARS + TRUCKS + CYCLES:
        lib.append(Blueprint(v, {"color": _Attr("0,0,0", ["10,10,10", "200,0,0"]), "role_name": _Attr("")}))
    for s in ("sensor.camera.rgb", "sensor.camera.semantic_segmentation", "sensor.camera.depth"):
        lib.append(Blueprint(s, {"image_size_x": _Attr(800), "image_size_y": _Attr(600), "fov": _Attr(90)}))
    return lib


def _tag_and_box(bp):
    """Semantic tag and bounding box (centre above the actor's origin, half sizes)."""
    bp_id = bp.id
    if bp_id.startswith("walker"):
        half_h = 0.62 if bp.attrs["age"].v == "child" else 0.88                 # 1.25 m / 1.75 m tall
        return 12, BoundingBox(Location(0, 0, 0), Vector3D(0.25, 0.25, half_h))  # origin = body centre
    if bp_id in TRUCKS:
        return 15, BoundingBox(Location(0, 0, 1.8), Vector3D(4.5, 1.3, 1.8))
    if bp_id in CYCLES:
        return (19 if "bh." in bp_id else 18), BoundingBox(Location(0, 0, 0.8), Vector3D(0.9, 0.35, 0.8))
    return 14, BoundingBox(Location(0, 0, 0.75), Vector3D(2.3, 1.0, 0.75))     # origin = under the car


_ids = itertools.count(100)


class Actor:
    def __init__(self, world, bp, tf, parent=None):
        self.world, self.id, self.type_id, self.bp = world, next(_ids), bp.id, bp
        self.tf, self.parent, self.is_alive = tf, parent, True
        self.tag, self.bounding_box = _tag_and_box(bp)
        self.listener, self.is_listening, self.control = None, False, None

    def get_transform(self):
        if self.parent is None:
            return self.tf
        p = self.parent.get_transform()
        y = math.radians(p.rotation.yaw)
        o = self.tf.location
        return Transform(Location(p.location.x + o.x * math.cos(y) - o.y * math.sin(y),
                                  p.location.y + o.x * math.sin(y) + o.y * math.cos(y), p.location.z + o.z),
                         Rotation(pitch=p.rotation.pitch + self.tf.rotation.pitch, yaw=p.rotation.yaw))

    def get_location(self):
        return self.get_transform().location

    def set_transform(self, tf):
        self.tf = tf

    def set_simulate_physics(self, on):
        pass

    def apply_control(self, c):
        self.control = c

    def listen(self, cb):
        self.listener, self.is_listening = cb, True

    def stop(self):
        self.listener, self.is_listening = None, False


class ActorList(list):
    def filter(self, pattern):
        return ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


class VehicleControl:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class WalkerControl:
    def __init__(self, direction=None, speed=0.0, jump=False):
        self.direction, self.speed = direction, speed


class WeatherParameters:
    def __init__(self):
        for k in ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "sun_azimuth_angle",
                  "sun_altitude_angle", "fog_density", "fog_distance", "wetness"):
            setattr(self, k, 0.0)


class CityObjectLabel:
    Car, Truck, Bus = 14, 15, 16


class ColorConverter:
    CityScapesPalette = 1


# ============================================================================
# rendering
# ============================================================================

ROAD_TAG, SKY_TAG = 1, 11
# a car is drawn like a real one, not as its bounding box: a 1 m high body over the full
# length (bonnet and boot) and the cabin over the middle half
CAR_SHAPE = [BoundingBox(Location(0, 0, 0.5), Vector3D(2.3, 1.0, 0.5)),
             BoundingBox(Location(0, 0, 0.75), Vector3D(1.15, 0.95, 0.75))]
PALETTE = {0: (0, 0, 0), 1: (128, 64, 128), 11: (180, 130, 70), 12: (60, 20, 220), 13: (0, 0, 255),
           14: (142, 0, 0), 15: (70, 0, 0), 16: (100, 60, 0), 18: (230, 0, 0), 19: (32, 11, 119)}   # B, G, R


def render(cam_tf, w, h, fov, shapes):
    """Semantic tags (h, w) and planar depth in metres for a camera at cam_tf.
    shapes: [(tag, 8 world corners)] for every actor to draw."""
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    R = np.array(_rot_matrix(cam_tf.rotation))
    cz = cam_tf.location.z
    tags = np.full((h, w), SKY_TAG, np.uint8)
    depth = np.full((h, w), 1000.0, np.float32)

    # ground plane z = 0: the ray of pixel (u, v) is (1, (u - w/2) / f, -(v - h/2) / f) in the camera
    u = (np.arange(w) + 0.5 - w / 2.0) / f
    v = -(np.arange(h) + 0.5 - h / 2.0) / f
    ray_z = R[2, 0] + R[2, 1] * u[None, :] + R[2, 2] * v[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ray_z < -1e-6, -cz / ray_z, np.inf)
    ground = t < 1000.0
    tags[ground], depth[ground] = ROAD_TAG, t[ground]

    world_to_cam = np.array(cam_tf.get_inverse_matrix())
    drawn = []
    for tag, corners in shapes:
        pts = np.hstack([np.array(corners), np.ones((8, 1))]) @ world_to_cam.T
        if (pts[:, 0] < 0.2).any():
            continue                        # partly behind the camera: not drawn
        uv = np.stack([f * pts[:, 1] / pts[:, 0] + w / 2.0, -f * pts[:, 2] / pts[:, 0] + h / 2.0], axis=1)
        drawn.append((float(pts[:, 0].mean()), tag, np.clip(uv, -1e5, 1e5)))
    for d, tag, uv in sorted(drawn, key=lambda x: -x[0]):          # far first, near on top
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        m = mask.astype(bool)
        tags[m], depth[m] = tag, d
    return tags, depth


class Image:
    """Rendered only when its pixels are read (most captured ticks are never looked at)."""

    def __init__(self, frame, sensor, cam_tf, shapes):
        self.frame, self.sensor = frame, sensor.type_id
        self.width = int(sensor.bp.attrs["image_size_x"].v)
        self.height = int(sensor.bp.attrs["image_size_y"].v)
        self.fov = float(sensor.bp.attrs["fov"].v)
        self._cam_tf, self._shapes, self._raw, self._palette = cam_tf, shapes, None, False

    def convert(self, converter):
        self._palette = True
        self._raw = None

    @property
    def raw_data(self):
        if self._raw is None:
            tags, depth = render(self._cam_tf, self.width, self.height, self.fov, self._shapes)
            out = np.zeros((self.height, self.width, 4), np.uint8)
            out[:, :, 3] = 255
            if self.sensor.endswith("depth"):
                n = np.round(np.clip(depth, 0, 1000) / 1000.0 * (256 ** 3 - 1)).astype(np.int64)
                out[:, :, 2], out[:, :, 1], out[:, :, 0] = n % 256, (n // 256) % 256, n // 65536
            elif self.sensor.endswith("segmentation") and not self._palette:
                out[:, :, 2] = tags
            else:
                lut = np.zeros((256, 3), np.uint8)
                for k, c in PALETTE.items():
                    lut[k] = c
                out[:, :, :3] = lut[tags]
            self._raw = out.tobytes()
        return self._raw


# ============================================================================
# world and client
# ============================================================================

class _Settings:
    synchronous_mode, fixed_delta_seconds = False, None


class World:
    def __init__(self):
        self.actors, self.frame, self.map, self.lib = {}, 0, MAP, _library()
        self.settings = _Settings()
        self.weather_sets = 0

    def get_map(self):
        return self.map

    def get_blueprint_library(self):
        return self.lib

    def get_settings(self):
        return self.settings

    def apply_settings(self, s):
        self.settings = s

    def set_weather(self, w):
        self.weather = w
        self.weather_sets += 1

    def get_environment_objects(self, label):
        return []

    def get_actors(self):
        return ActorList(self.actors.values())

    def try_spawn_actor(self, bp, tf, attach_to=None):
        a = Actor(self, bp, tf, attach_to)
        if attach_to is None and not bp.id.startswith("sensor"):
            # "physics": vehicles stand on the ground, walkers on their feet
            loc = tf.location
            ground = self.map.get_waypoint(loc, lane_type=LaneType.Any).transform.location.z
            z = ground + (a.bounding_box.extent.z if bp.id.startswith("walker") else 0.0)
            a.tf = Transform(Location(loc.x, loc.y, z), tf.rotation)
        self.actors[a.id] = a
        return a

    spawn_actor = try_spawn_actor

    def _shapes(self, camera):
        out = []
        for a in self.actors.values():
            if a is camera.parent or a.type_id.startswith("sensor") or a.tf.location.z < -50:
                continue
            tf = a.get_transform()
            for box in CAR_SHAPE if a.tag == 14 else [a.bounding_box]:
                out.append((a.tag, [(p.x, p.y, p.z) for p in box.get_world_vertices(tf)]))
        return out

    def tick(self, seconds=10.0):
        self.frame += 1
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in list(self.actors.values()):
            c = a.control
            if isinstance(c, WalkerControl) and c.direction is not None:
                loc = a.tf.location
                a.tf = Transform(Location(loc.x + c.direction.x * c.speed * dt,
                                          loc.y + c.direction.y * c.speed * dt, loc.z), a.tf.rotation)
        for a in list(self.actors.values()):
            if a.listener:
                a.listener(Image(self.frame, a, a.get_transform(), self._shapes(a)))
        return self.frame


class command:
    class DestroyActor:
        def __init__(self, a):
            self.id = a if isinstance(a, int) else a.id


class Client:
    world = None

    def __init__(self, host="localhost", port=2000):
        if Client.world is None:
            Client.world = World()
        self.world = Client.world

    def set_timeout(self, t):
        pass

    def get_server_version(self):
        return "fake-0.9.16"

    def get_world(self):
        return self.world

    def get_available_maps(self):
        return ["/Game/" + Map.name]

    def load_world(self, name):
        return self.world

    def reload_world(self, reset=True):
        return self.world

    def apply_batch_sync(self, cmds, do_tick=False):
        for c in cmds:
            a = self.world.actors.pop(c.id, None)
            if a:
                a.is_alive = False
        if do_tick:
            self.world.tick()
