"""Label measurement on hand-made segmentation / depth images: every box gets CARLA's own
class, an object hidden behind something never takes another object's pixels, and a
pedestrian's full-body box is not cut by a car that was moved away for that pass."""
import unittest

import numpy as np

import helpers as H
from helpers import C
import engine.labels as L

W2C = np.eye(4)             # camera at the origin looking along +x (UE: x forward, y right, z up)


def box(x, y, half_w, half_l, z0, z1):
    return np.array([[a, b, c] for a in (x - half_l, x + half_l) for b in (y - half_w, y + half_w) for c in (z0, z1)])


def region(verts):
    uv, _, _ = L.project(verts, W2C, L.intrinsics())
    return int(uv[:, 0].min()) + 2, int(uv[:, 1].min()) + 2, int(uv[:, 0].max()) - 2, int(uv[:, 1].max()) - 2


def obj(kind, verts, name, **meta):
    return {"kind": kind, "source": "spawned", "meta": {"name": name, **meta}, "verts": verts,
            "center": verts.mean(axis=0).tolist(), "rotation": (0, 0, 0), "extent": (1, 1, 1)}


class LabelTest(unittest.TestCase):
    def setUp(self):
        H.configure()
        self.seg = np.full((C.IMAGE_H, C.IMAGE_W), 1, np.uint8)            # road
        self.depth = np.full((C.IMAGE_H, C.IMAGE_W), 200.0, np.float32)
        self.hood = np.zeros((C.IMAGE_H, C.IMAGE_W), bool)

    def paint(self, verts, tag, dist, seg=None, depth=None):
        x0, y0, x1, y1 = region(verts)
        (self.seg if seg is None else seg)[y0:y1, x0:x1] = tag
        (self.depth if depth is None else depth)[y0:y1, x0:x1] = dist
        return [x0, y0, x1, y1]

    def labels(self, objects, seg_full=None, depth_full=None):
        out = L.compute_labels(objects, self.seg, self.seg if seg_full is None else seg_full, self.hood, W2C,
                               L.intrinsics(), depth=self.depth,
                               depth_full=self.depth if depth_full is None else depth_full)
        return {lb["meta"]["name"]: lb for lb in out}

    def test_classes_come_from_carla_tags(self):
        truck = box(15, -6, 1.2, 3.0, -1.8, 0.5)            # a model from the car list that CARLA tags Truck
        self.paint(truck, 15, 15.0)
        bike = box(12, 5, 0.4, 1.0, -1.8, 0.0)              # bicycle (lower half) with its rider (upper half)
        bx0, by0, bx1, by1 = region(bike)
        mid = (by0 + by1) // 2
        self.seg[mid:by1, bx0:bx1], self.seg[by0:mid, bx0:bx1], self.depth[by0:by1, bx0:bx1] = 19, 13, 12.0
        got = self.labels([obj("vehicle", truck, "truck"), obj("cyclist", bike, "bike")])
        self.assertEqual((got["truck"]["kind"], got["truck"]["carla_class"]), ("truck", "Truck"))
        self.assertEqual((got["bike"]["kind"], got["bike"]["carla_class"]), ("cyclist", "Bicycle"))
        self.assertEqual(got["bike"]["bbox_visible"], [bx0, by0, bx1, by1], "rider not inside the cyclist's box")

    def test_hidden_object_does_not_take_a_visible_ones_box(self):
        far = box(20, 0, 1.0, 2.3, -1.8, -0.3)              # visible car, 20 m
        near = box(10, 0.3, 1.0, 2.3, -1.8, -0.3)           # car at 10 m, but behind a wall
        far_box = self.paint(far, 14, 20.0)
        nx0, ny0, nx1, ny1 = region(near)
        wall = np.zeros_like(self.seg, bool)
        wall[ny0:ny1, nx0:nx1] = True
        wall[far_box[1]:far_box[3], far_box[0]:far_box[2]] = False      # the far car shows through a gap
        self.seg[wall], self.depth[wall] = 3, 6.0
        got = self.labels([obj("vehicle", far, "far"), obj("vehicle", near, "near")])
        self.assertNotIn("near", got, "the car behind the wall got a box")
        self.assertEqual(got["far"]["bbox_visible"], far_box)

    def test_full_body_box_ignores_the_moved_away_car(self):
        """A car right in front of a pedestrian, seen lengthwise (its depth range reaches the
        pedestrian): in the pass with traffic moved away, the pedestrian's whole body counts."""
        ped = box(12.6, 0, 0.25, 0.25, -1.8, 0.0)           # right behind the car, camera 1.8 m up
        car = box(9.5, 0, 1.0, 2.6, -1.8, -0.3)             # 6.9 .. 12.1 m: depth range reaches 12.85 m
        ped_box = self.paint(ped, 12, 12.6)
        car_box = self.paint(car, 14, 9.5)                  # hides the pedestrian's lower part
        seg_full, depth_full = self.seg.copy(), self.depth.copy()
        seg_full[seg_full == 14], depth_full[self.seg == 14] = 1, 50.0
        self.paint(ped, 12, 12.6, seg_full, depth_full)
        got = self.labels([obj("pedestrian", ped, "ped", target=True), obj("vehicle", car, "car")],
                          seg_full, depth_full)
        p = got["ped"]
        self.assertEqual(p["bbox_full"], ped_box, "full-body box cut by the car")
        hidden_rows = max(0, ped_box[3] - car_box[1])
        expect = hidden_rows / (ped_box[3] - ped_box[1])
        self.assertAlmostEqual(p["occlusion"], expect, delta=0.03)


if __name__ == "__main__":
    unittest.main()
