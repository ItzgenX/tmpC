"""Runs one planned job in the simulator and returns its frames (not yet written).

1. teleport the ego, spawn traffic, let it settle, freeze it
2. spawn pedestrians and start them walking
3. at every planned tick: capture RGB + segmentation + depth from that exact
   tick, then hide our traffic for one tick to see each pedestrian unoccluded
4. measure labels from segmentation, run the quality checks
Everything spawned is destroyed at the end, even if something fails.
"""
import cv2
import numpy as np

import advanced as C
from engine.carla_io import destroy_actors, freeze, hide, restore, spawn_pedestrians, spawn_vehicles, start_walking, to_transform
from engine.labels import compute_labels, decode_depth, frame_flags, intrinsics, seg_tags, target_label
from engine.stop import protected


KIND = {"car": "vehicle", "truck": "truck", "cyclist": "cyclist"}


def render_job(session, plan):
    world = session.world
    actors = []
    try:
        session.ego.set_transform(to_transform(plan["ego"], session.planner.ego_z_offset))
        vehicles = spawn_vehicles(world, session.bp_lib, plan["vehicles"], actors)
        for _ in range(10):         # settle on the road, then freeze in place
            world.tick()
        freeze(vehicles)
        walkers = spawn_pedestrians(world, session.bp_lib, plan["pedestrians"], actors)
        world.tick()
        start_walking(walkers, plan["pedestrians"])

        frames, tick = [], 0
        for planned in plan["frames"]:
            wait = max(0, planned["tick"] - tick - C.CAPTURE_SETTLE_TICKS)
            for _ in range(wait):
                world.tick()
            _, imgs = session.capture(session.cameras(), C.CAPTURE_SETTLE_TICKS)
            tick += wait + C.CAPTURE_SETTLE_TICKS
            # every transform read now belongs to the captured tick
            cam_tf = session.rgb.get_transform()
            world_to_cam = np.array(cam_tf.get_inverse_matrix())
            objects = scene_objects(session, plan, walkers, vehicles, cam_tf)

            saved = hide(vehicles)
            _, full_imgs = session.capture([session.seg, session.depth], 1)
            restore(vehicles, saved)
            tick += 1

            frames.append(_finish_frame(session, plan, planned, imgs, full_imgs, objects, cam_tf, world_to_cam))
        return frames
    finally:
        with protected():           # Ctrl+C during a scene: its actors still get removed
            destroy_actors(session.client, world, actors)


def _finish_frame(session, plan, planned, imgs, full_imgs, objects, cam_tf, world_to_cam):
    rgb_img, seg_img, depth_img = imgs
    seg = seg_tags(seg_img)
    depth_m = decode_depth(depth_img)
    labels = compute_labels(objects, seg, seg_tags(full_imgs[0]), session.hood, world_to_cam, intrinsics(),
                            depth=depth_m, depth_full=decode_depth(full_imgs[1]))
    flags = frame_flags(labels, seg, session.hood, planned["planned_occlusion"])
    target = target_label(labels)

    rgb = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape(rgb_img.height, rgb_img.width, 4)[:, :, :3]
    if C.RGB_SUPERSAMPLE > 1:
        rgb = cv2.resize(rgb, (C.IMAGE_W, C.IMAGE_H), interpolation=cv2.INTER_AREA)
    seg_img.convert(session.carla.ColorConverter.CityScapesPalette)
    seg_color = np.frombuffer(seg_img.raw_data, dtype=np.uint8).reshape(C.IMAGE_H, C.IMAGE_W, 4)[:, :, :3]

    loc, rot = cam_tf.location, cam_tf.rotation
    return {
        "rgb": np.ascontiguousarray(rgb),
        "seg": seg,
        "seg_color": np.ascontiguousarray(seg_color),
        "depth_m": depth_m,
        "labels": labels,
        "flags": flags,
        "info": {
            "map": session.map_name, "job": plan["job"], "scene": plan["scene"],
            "weather": session.weather_params,
            "planned_occlusion": planned["planned_occlusion"],
            "measured_occlusion": target["occlusion"] if target else None,
            "flags": flags,
            "image_size": [C.IMAGE_W, C.IMAGE_H], "classes": C.class_names(),
            "camera": {"K": intrinsics().tolist(), "fov": C.CAMERA_FOV,
                       "location": [loc.x, loc.y, loc.z],
                       "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                       "world_to_camera": world_to_cam.tolist()},
            "plan": plan,
        },
    }


def scene_objects(session, plan, walkers, vehicles, cam_tf):
    """Everything that may get a label, with its 3D box at the captured tick."""
    carla = session.carla
    site = plan["site"]
    sx, sy = site["start"][:2]
    ex, ey = site["end"][:2]
    L = max(site["length"], 1e-3)
    ax, ay = (ex - sx) / L, (ey - sy) / L
    cam_right = cam_tf.get_right_vector()
    objs = []

    def box_obj(actor, kind, meta):
        tf = actor.get_transform()
        bb = actor.bounding_box
        verts = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(tf)])
        return {"kind": kind, "source": "spawned", "meta": meta, "verts": verts,
                "center": verts.mean(axis=0).tolist(),
                "rotation": (tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
                "extent": (bb.extent.x, bb.extent.y, bb.extent.z)}

    for idx, (wk, p) in enumerate(zip(walkers, plan["pedestrians"])):
        loc = wk.get_location()
        crossing = p["role"] == "crossing"
        progress = ((loc.x - sx) * ax + (loc.y - sy) * ay) / L
        dx, dy = p["direction"]
        objs.append(box_obj(wk, "pedestrian", {
            "actor_id": wk.id, "plan_index": idx, "target": idx == plan["target_pedestrian"],
            "role": p["role"], "crossing": crossing, "type": p["type"], "blueprint": p["blueprint"],
            "speed": p["speed"], "runner": p["runner"],
            "crossing_type": site["kind"] if crossing else None,
            "crossing_progress": round(progress, 3) if crossing else None,
            "on_road": bool(crossing and 0.0 < progress < 1.0),
            "walking_direction": (("left_to_right" if dx * cam_right.x + dy * cam_right.y > 0 else "right_to_left")
                                  if crossing else ("along_sidewalk" if p["speed"] > 0 else "standing")),
        }))
    for actor, v in zip(vehicles, plan["vehicles"]):
        objs.append(box_obj(actor, KIND[v["category"]], {
            "actor_id": actor.id, "blueprint": v["blueprint"], "direction": v["direction"],
            "role": v["role"], "parked": v.get("parked", False)}))

    if C.LABEL_PARKED_VEHICLES:
        cl = cam_tf.location
        for obj, cls, label_name in session.parked:
            bb = obj.bounding_box
            if bb.location.distance(cl) > C.MAX_LABEL_DISTANCE or bb.extent.x <= 0 or bb.extent.y <= 0:
                continue
            verts = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(carla.Transform())])
            objs.append({"kind": cls, "source": "parked", "meta": {"env_id": obj.id, "env_type": label_name},
                         "verts": verts, "center": [bb.location.x, bb.location.y, bb.location.z],
                         "rotation": (bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll),
                         "extent": (bb.extent.x, bb.extent.y, bb.extent.z)})
    return objs
