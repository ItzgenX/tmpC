"""Camera math, segmentation-based labels, quality gates, file writing, drawing."""
import json
import math
import os

import cv2
import numpy as np

import advanced as C


# ---------- camera ----------

def intrinsics(width=C.IMAGE_W, height=C.IMAGE_H, fov=C.CAMERA_FOV):
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def project(points, world_to_cam, K):
    """World points (N,3) -> pixels (N,2) and depth (N,). UE camera: x fwd, y right, z up."""
    homog = np.hstack([points, np.ones((len(points), 1))])
    cam = homog @ world_to_cam.T
    depth = cam[:, 0]
    safe = np.where(depth > 1e-3, depth, 1e-3)
    u = K[0, 0] * cam[:, 1] / safe + K[0, 2]
    v = -K[1, 1] * cam[:, 2] / safe + K[1, 2]
    return np.stack([u, v], axis=1), depth, cam


def cam_coords(cam_xyz):
    # UE (fwd, right, up) -> conventional camera (right, down, forward), KITTI style
    return [float(cam_xyz[1]), float(-cam_xyz[2]), float(cam_xyz[0])]


# ---------- depth ----------

def decode_depth(img):
    """CARLA depth image -> float32 metres (H, W). Sky reads as ~1000 m."""
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4).astype(np.float32)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0 ** 3 - 1) * 1000.0


def depth_to_png16(depth_m):
    """uint16 PNG, 1 unit = 1 cm (max 655 m). Divide by 100 to get metres."""
    return np.clip(np.round(depth_m * 100.0), 0, 65535).astype(np.uint16)


def depth_preview(depth_m, max_m=80.0):
    """Colour preview, near = warm. Not for training."""
    d = np.clip(depth_m, 0.0, max_m) / max_m
    return cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ---------- segmentation labels ----------

def _tight(mask, box):
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return None, None
    return [int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max() + 1), int(y0 + ys.max() + 1)], (ys + y0, xs + x0)


def _search_box(box, w, h, grow=0.08):
    x0, y0, x1, y1 = box
    gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
    return [int(max(0, x0 - gx)), int(max(0, y0 - gy)), int(min(w, x1 + gx + 1)), int(min(h, y1 + gy + 1))]


def _assign_pixels(objs, tag_masks, w, h):
    """Near-first: each object takes its class pixels inside its projected box that a
    nearer object has not already taken. Returns {index: (tight_box, pixel_count)}."""
    claimed = np.zeros((h, w), dtype=bool)
    out = {}
    for i, o in sorted(enumerate(objs), key=lambda io: io[1]["depth"]):
        mask = tag_masks[o["group"]] & ~claimed
        sb = _search_box(o["proj_box"], w, h)
        box, pix = _tight(mask, sb)
        if box is None:
            out[i] = (None, 0)
            continue
        claimed[pix] = True
        out[i] = (box, len(pix[0]))
    return out


def compute_labels(objects, seg, seg_full, hood, world_to_cam, K):
    """Projects every candidate object and measures it against segmentation.

    seg      = tags as rendered (traffic present)
    seg_full = tags with our traffic moved away: each pedestrian's unoccluded
               silhouette, giving the true full-body box and occlusion %
    """
    h, w = seg.shape
    tags = C.SEG_TAGS
    ped = np.isin(seg, tags["pedestrian"])
    veh = np.isin(seg, tags["vehicle"]) & ~hood
    masks = {"pedestrian": ped, "vehicle": veh}
    masks_full = {"pedestrian": np.isin(seg_full, tags["pedestrian"]) if seg_full is not None else ped,
                  "vehicle": veh}

    cand = []
    for o in objects:
        uv, depth, cam = project(o["verts"], world_to_cam, K)
        c_uv, c_depth, c_cam = project(np.array([o["center"]]), world_to_cam, K)
        if c_depth[0] <= 0.5 or c_depth[0] > C.MAX_LABEL_DISTANCE or (depth > 0.1).sum() < 2:
            continue
        front = uv[depth > 0.1]
        raw = [front[:, 0].min(), front[:, 1].min(), front[:, 0].max(), front[:, 1].max()]
        if raw[2] < 0 or raw[3] < 0 or raw[0] > w or raw[1] > h:
            continue
        clipped = [int(max(0, raw[0])), int(max(0, raw[1])), int(min(w, raw[2])), int(min(h, raw[3]))]
        if clipped[2] - clipped[0] < 2 or clipped[3] - clipped[1] < 2:
            continue
        o = dict(o)
        o.update({
            "group": "pedestrian" if o["kind"] == "pedestrian" else "vehicle",
            "proj_box": clipped, "depth": float(c_depth[0]),
            "truncated": bool(raw[0] < 0 or raw[1] < 0 or raw[2] > w or raw[3] > h or (depth <= 0.1).any()),
            "corners_2d": [[round(float(x), 1), round(float(y), 1)] if d > 0.1 else None
                           for (x, y), d in zip(uv, depth)],
            "cam_xyz": c_cam[0],
        })
        cand.append(o)

    visible = _assign_pixels(cand, masks, w, h)
    full = _assign_pixels(cand, masks_full, w, h)

    labels = []
    for i, o in enumerate(cand):
        vis_box, vis_px = visible[i]
        min_px = C.MIN_VISIBLE_PIXELS[o["kind"]]
        if vis_box is None or vis_px < min_px:
            continue
        if min(vis_box[2] - vis_box[0], vis_box[3] - vis_box[1]) < C.MIN_BOX_SIZE_PX:
            continue
        full_box, full_px = full[i]
        if o["kind"] == "pedestrian" and full_box is not None:
            full_px = max(full_px, vis_px)
            full_box = [min(full_box[0], vis_box[0]), min(full_box[1], vis_box[1]),
                        max(full_box[2], vis_box[2]), max(full_box[3], vis_box[3])]
            occlusion = round(1.0 - vis_px / full_px, 3)
        else:
            full_box, full_px, occlusion = None, None, None
        labels.append({
            **{k: o[k] for k in ("kind", "source", "meta", "truncated", "corners_2d", "depth")},
            "bbox_visible": vis_box,
            "bbox_full": full_box,
            "bbox_projected": o["proj_box"],
            "visible_pixels": int(vis_px),
            "full_pixels": int(full_px) if full_px is not None else None,
            "occlusion": occlusion,
            "location_world": [round(v, 3) for v in o["center"]],
            "rotation_world": {"pitch": o["rotation"][0], "yaw": o["rotation"][1], "roll": o["rotation"][2]},
            "dimensions": {"length": 2 * o["extent"][0], "width": 2 * o["extent"][1], "height": 2 * o["extent"][2]},
            "location_camera": [round(v, 3) for v in cam_coords(o["cam_xyz"])],
        })
    return labels


def road_fraction(seg, hood):
    keep = ~hood
    return float((np.isin(seg, C.SEG_TAGS["road_like"]) & keep).sum()) / max(1, int(keep.sum()))


def hood_mask(seg_frames):
    """Ego bonnet = vehicle pixels in the lower image present in every given frame."""
    mask = None
    for seg in seg_frames:
        m = np.isin(seg, C.SEG_TAGS["vehicle"])
        m[: int(seg.shape[0] * 0.6), :] = False
        mask = m if mask is None else (mask & m)
    return mask


# ---------- outputs ----------

def class_id(label):
    names = C.class_names()
    if label["kind"] == "pedestrian":
        if C.SPLIT_CROSSING_CLASS and label["meta"].get("role") == "crossing":
            return names.index("crossing_pedestrian")
        return names.index("pedestrian")
    return names.index(label["kind"])


def yolo_lines(labels, ped_box):
    lines = []
    for lb in labels:
        if lb["kind"] != "pedestrian" and not C.LABEL_VEHICLES:
            continue
        box = lb["bbox_full"] if (lb["kind"] == "pedestrian" and ped_box == "full" and lb["bbox_full"]) \
            else lb["bbox_visible"]
        x0, y0, x1, y1 = box
        lines.append(f"{class_id(lb)} {(x0 + x1) / 2 / C.IMAGE_W:.6f} {(y0 + y1) / 2 / C.IMAGE_H:.6f} "
                     f"{(x1 - x0) / C.IMAGE_W:.6f} {(y1 - y0) / C.IMAGE_H:.6f}")
    return lines


def write_text_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json_atomic(path, data):
    write_text_atomic(path, json.dumps(data, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_image(path, img):
    params = [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY] if path.endswith(".jpg") else []
    if not cv2.imwrite(path, img, params):
        raise IOError(f"could not write {path}")


# ---------- debug drawing ----------

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
COLORS = {"pedestrian": (60, 220, 60), "vehicle": (255, 160, 40), "truck": (40, 140, 255)}


def draw_overlay(img, labels):
    out = img.copy()
    for lb in labels:
        col = COLORS[lb["kind"]]
        pts = lb["corners_2d"]
        for i, j in EDGES:
            if pts[i] is not None and pts[j] is not None:
                cv2.line(out, tuple(map(int, pts[i])), tuple(map(int, pts[j])), col, 1)
        if lb["bbox_full"]:
            x0, y0, x1, y1 = lb["bbox_full"]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
        x0, y0, x1, y1 = lb["bbox_visible"]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        text = lb["kind"]
        if lb["kind"] == "pedestrian":
            text = f"{lb['meta'].get('type')} {lb['meta'].get('role')} occ {lb['occlusion']:.0%}"
        cv2.putText(out, text, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out
