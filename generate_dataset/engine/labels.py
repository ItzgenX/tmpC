"""From a captured frame to files on disk.

Sections:
  labels   - 2D/3D boxes and real occlusion %, measured from segmentation
  checks   - quality checks (a failing frame is saved and flagged, not thrown away)
  writer   - saving only the outputs switched on in config.SAVE
"""
import json
import math
import os

import cv2
import numpy as np

import advanced as C


# ============================================================================
# labels
# ============================================================================
# Turns a captured frame into labels.
#
# Each object's 3D box is projected into the image, then measured against the
# segmentation image: the pixels of its class inside that box give the visible
# 2D box. A second segmentation frame taken with our traffic moved away gives
# each pedestrian's unoccluded silhouette, so we also get its full-body box and
# the real occlusion %.

# ---------- camera ----------

def intrinsics(width=C.IMAGE_W, height=C.IMAGE_H, fov=C.CAMERA_FOV):
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def project(points, world_to_cam, K):
    """World points (N,3) -> pixels (N,2), depth (N,), camera coords. UE camera: x fwd, y right, z up."""
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


# ---------- raw images ----------

def seg_tags(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return arr[:, :, 2].copy()      # copy: convert() later rewrites the buffer


def decode_depth(img):
    """CARLA depth image -> float32 metres (H, W). Sky reads as ~1000 m."""
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4).astype(np.float32)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0 ** 3 - 1) * 1000.0


# ---------- labels ----------

def _group(kind):
    return {"pedestrian": "pedestrian", "cyclist": "cyclist"}.get(kind, "vehicle")


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


def _assign_pixels(objs, masks, w, h):
    """Near-first: each object takes its class pixels inside its projected box that a
    nearer object has not already taken. Returns {index: (tight_box, pixel_count)}."""
    claimed = np.zeros((h, w), dtype=bool)
    out = {}
    for i, o in sorted(enumerate(objs), key=lambda io: io[1]["depth"]):
        box, pix = _tight(masks[o["group"]] & ~claimed, _search_box(o["proj_box"], w, h))
        if box is None:
            out[i] = (None, 0)
            continue
        claimed[pix] = True
        out[i] = (box, len(pix[0]))
    return out


def _big_enough(box, pixels, kind):
    return (box is not None and pixels >= C.MIN_VISIBLE_PIXELS[kind]
            and min(box[2] - box[0], box[3] - box[1]) >= C.MIN_BOX_SIZE_PX)


def compute_labels(objects, seg, seg_full, hood, world_to_cam, K):
    """objects: dicts with kind, source, meta, verts (8,3), center, rotation, extent.
    Pedestrians hidden completely are kept (visible = False) when their unoccluded
    silhouette is big enough; vehicles and cyclists need to be visible."""
    h, w = seg.shape
    tags = C.SEG_TAGS
    masks = {"pedestrian": np.isin(seg, tags["pedestrian"]),
             "vehicle": np.isin(seg, tags["vehicle"]) & ~hood,
             "cyclist": np.isin(seg, tags["cyclist"])}
    masks_full = dict(masks, pedestrian=np.isin(seg_full, tags["pedestrian"]))

    cand = []
    for o in objects:
        uv, depth, _ = project(o["verts"], world_to_cam, K)
        _, c_depth, c_cam = project(np.array([o["center"]]), world_to_cam, K)
        if c_depth[0] <= 0.5 or c_depth[0] > C.MAX_LABEL_DISTANCE or (depth > 0.1).sum() < 2:
            continue
        front = uv[depth > 0.1]
        raw = [front[:, 0].min(), front[:, 1].min(), front[:, 0].max(), front[:, 1].max()]
        if raw[2] < 0 or raw[3] < 0 or raw[0] > w or raw[1] > h:
            continue
        clipped = [int(max(0, raw[0])), int(max(0, raw[1])), int(min(w, raw[2])), int(min(h, raw[3]))]
        if clipped[2] - clipped[0] < 2 or clipped[3] - clipped[1] < 2:
            continue
        cand.append(dict(o, group=_group(o["kind"]), proj_box=clipped, depth=float(c_depth[0]),
                         truncated=bool(raw[0] < 0 or raw[1] < 0 or raw[2] > w or raw[3] > h
                                        or (depth <= 0.1).any()),
                         corners_2d=[[round(float(x), 1), round(float(y), 1)] if d > 0.1 else None
                                     for (x, y), d in zip(uv, depth)],
                         cam_xyz=c_cam[0]))

    visible = _assign_pixels(cand, masks, w, h)
    full = _assign_pixels(cand, masks_full, w, h)

    labels = []
    for i, o in enumerate(cand):
        vis_box, vis_px = visible[i]
        is_visible = _big_enough(vis_box, vis_px, o["kind"])
        full_box, full_px, occlusion = None, None, None
        if o["kind"] == "pedestrian":
            full_box, full_px = full[i]
            if not _big_enough(full_box, full_px, "pedestrian"):
                continue                # not in view at all, even without traffic
            if vis_box is not None:
                full_px = max(full_px, vis_px)
                full_box = [min(full_box[0], vis_box[0]), min(full_box[1], vis_box[1]),
                            max(full_box[2], vis_box[2]), max(full_box[3], vis_box[3])]
            occlusion = round(1.0 - vis_px / full_px, 3)
        elif not is_visible:
            continue
        labels.append({
            "kind": o["kind"], "source": o["source"], "meta": o["meta"],
            "visible": is_visible,
            "bbox_visible": vis_box if is_visible else None,
            "bbox_full": full_box,
            "bbox_projected": o["proj_box"],
            "visible_pixels": int(vis_px),
            "full_pixels": int(full_px) if full_px is not None else None,
            "occlusion": occlusion,
            "truncated": o["truncated"], "depth": o["depth"],
            "corners_2d": o["corners_2d"],
            "location_world": [round(v, 3) for v in o["center"]],
            "rotation_world": {"pitch": o["rotation"][0], "yaw": o["rotation"][1], "roll": o["rotation"][2]},
            "dimensions": {"length": 2 * o["extent"][0], "width": 2 * o["extent"][1], "height": 2 * o["extent"][2]},
            "location_camera": [round(v, 3) for v in cam_coords(o["cam_xyz"])],
        })
    return labels


# ---------- whole-image measures ----------

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


# ============================================================================
# checks
# ============================================================================
# Quality checks on a rendered frame. A failed check does not throw the image
# away: the image is saved and listed in flagged.csv with the reason, so you can
# look at it and delete it later with "python main.py --delete-flagged".

def target_label(labels):
    return next((lb for lb in labels if lb["meta"].get("target")), None)


def frame_flags(labels, seg, hood, planned_occlusion):
    """Reasons this frame looks wrong; empty list = fine."""
    flags = []
    if road_fraction(seg, hood) < C.MIN_ROAD_FRACTION:
        flags.append("camera blocked")
    target = target_label(labels)
    if target is None:
        flags.append("target pedestrian not in view")
        return flags
    if (target["full_pixels"] or 0) < C.MIN_TARGET_PIXELS:
        flags.append("target pedestrian too small")
    if abs(target["occlusion"] - planned_occlusion) > C.OCCLUSION_TOLERANCE:
        flags.append(f"occlusion {target['occlusion']:.0%}, planned {planned_occlusion:.0%}")
    return flags


# ============================================================================
# writer
# ============================================================================
# Writes one frame to disk: only the outputs switched on in config.SAVE.
#
# images/<name>.png          RGB
# depth/<name>.png           16-bit, 1 unit = 1 cm
# seg_color/<name>.png       CityScapes colours
# images_2d_bbox/<name>.jpg  RGB with 2D boxes
# images_3d_bbox/<name>.jpg  RGB with 3D boxes
# labels/<name>.txt          YOLO
# labels_3d/<name>.json      always: every box (2D + 3D), occlusion, camera, scene plan

FOLDERS = C.OUTPUT_FOLDERS


def make_folders(out):
    for flag, folder in FOLDERS.items():
        if C.SAVE.get(flag):
            os.makedirs(os.path.join(out, folder), exist_ok=True)
    os.makedirs(os.path.join(out, "labels_3d"), exist_ok=True)


# ---------- YOLO ----------

def class_id(label):
    names = C.class_names()
    if label["kind"] == "pedestrian":
        if C.SPLIT_CROSSING_CLASS and label["meta"].get("role") == "crossing":
            return names.index("crossing_pedestrian")
        return names.index("pedestrian")
    return names.index(label["kind"])


def yolo_box(label):
    """Box used in the YOLO file, or None if this label is left out of it."""
    if label["kind"] != "pedestrian":
        return label["bbox_visible"] if C.LABEL_VEHICLES else None
    if not label["visible"]:
        return label["bbox_full"] if C.LABEL_FULLY_HIDDEN_PEDESTRIANS else None
    return label["bbox_full"] if C.PED_YOLO_BOX == "full" else label["bbox_visible"]


def yolo_lines(labels):
    lines = []
    for lb in labels:
        box = yolo_box(lb)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        lines.append(f"{class_id(lb)} {(x0 + x1) / 2 / C.IMAGE_W:.6f} {(y0 + y1) / 2 / C.IMAGE_H:.6f} "
                     f"{(x1 - x0) / C.IMAGE_W:.6f} {(y1 - y0) / C.IMAGE_H:.6f}")
    return lines


# ---------- drawing ----------

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def draw_2d(img, labels):
    """The same boxes as the YOLO file, one colour, no text."""
    out = img.copy()
    for lb in labels:
        box = yolo_box(lb)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1, y1), C.BOX_COLOR, C.BOX_THICKNESS)
    return out


def draw_3d(img, labels):
    """3D box edges, one colour, no text."""
    out = img.copy()
    for lb in labels:
        if yolo_box(lb) is None:
            continue
        pts = lb["corners_2d"]
        for i, j in EDGES:
            if pts[i] is not None and pts[j] is not None:
                cv2.line(out, tuple(map(int, pts[i])), tuple(map(int, pts[j])), C.BOX_COLOR, C.BOX_THICKNESS)
    return out


# ---------- files ----------

def write_image(path, img):
    params = [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY] if path.endswith(".jpg") else []
    if not cv2.imwrite(path, img, params):
        raise IOError(f"could not write {path}")


def write_text_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_frame(out, name, frame):
    """frame: dict with rgb, depth_m, seg, seg_color, labels, info (json extras)."""
    def path(flag, ext):
        return os.path.join(out, FOLDERS[flag], name + ext)

    rgb, labels = frame["rgb"], frame["labels"]
    if C.SAVE.get("image"):
        write_image(path("image", C.IMAGE_EXT), rgb)
    if C.SAVE.get("depth") and frame["depth_m"] is not None:
        write_image(path("depth", ".png"), np.clip(np.round(frame["depth_m"] * 100), 0, 65535).astype(np.uint16))
    if C.SAVE.get("depth_color") and frame["depth_m"] is not None:
        d = np.clip(frame["depth_m"], 0.0, 80.0) / 80.0
        write_image(path("depth_color", ".png"), cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    if C.SAVE.get("seg_color"):
        write_image(path("seg_color", ".png"), frame["seg_color"])
    if C.SAVE.get("seg_raw"):
        write_image(path("seg_raw", ".png"), frame["seg"])
    if C.SAVE.get("bbox_2d_image"):
        write_image(path("bbox_2d_image", ".jpg"), draw_2d(rgb, labels))
    if C.SAVE.get("bbox_3d_image"):
        write_image(path("bbox_3d_image", ".jpg"), draw_3d(rgb, labels))
    if C.SAVE.get("labels_yolo"):
        write_text_atomic(path("labels_yolo", ".txt"), "\n".join(yolo_lines(labels)))
    write_text_atomic(os.path.join(out, "labels_3d", name + ".json"),
                      json.dumps({"image": name + C.IMAGE_EXT, **frame["info"], "labels": labels},
                                 indent=1, default=_json_default))
