"""Auto-place candidate cameras in world.blend's "Views" collection for review.

Cameras stand at eye height on sidewalks, plazas, park paths and the skate
area, mostly looking along the street or path; --targeted first places
cameras facing entrances, bridges and skate platforms. A spot is kept only when the ground ahead is open
and at least one task target (bench, door, crosswalk, terrace, ...) is in
view. Open world.blend afterwards to move, delete, or add cameras; any camera
in "Views" is rendered by render_views.py.

  blender --background blender/world/world.blend --python blender/place_views.py -- --count 40

Existing cameras are kept; new ones continue the numbering. Use --replace to start over.
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldlib import layout as lay  # noqa: E402
from worldlib.classes import CLASS_ID  # noqa: E402
from worldlib.meshes import get_collection  # noqa: E402
from worldlib.navmap import VisTree  # noqa: E402
from worldlib.planner import HumanPlanner  # noqa: E402
from worldlib.scene_setup import look_rotation, make_view_camera  # noqa: E402
from worldlib.tasks import ViewTasks  # noqa: E402

STAND_CLASSES = ("sidewalk", "plaza", "park_path", "skate_area", "cobblestone", "boardwalk", "gravel_path",
                 "running_track", "terrace")
CLEAR_AHEAD_M = 7.0
HEADING_JITTER_DEG = 14.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Auto-place candidate view cameras.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_spacing", type=float, default=8.0)
    parser.add_argument("--replace", action="store_true", help="Delete existing view cameras first.")
    parser.add_argument("--new_only", action="store_true", help="Only place cameras in Phase 2 blocks.")
    parser.add_argument("--targeted", action="store_true",
                        help="First place cameras facing entrances, bridges and skate platforms.")
    return parser.parse_args(argv)


def world_dir():
    return os.path.dirname(bpy.data.filepath)


def camera_heading(cam):
    forward = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    return math.atan2(forward.y, forward.x)


def clear_ahead(planner, x, y, heading, dist=CLEAR_AHEAD_M):
    """Open walkable ground for ``dist`` metres straight ahead, with no height jumps."""
    step = planner.cell
    prev = planner.height_at(x, y)
    for k in range(1, int(dist / step) + 1):
        px = x + math.cos(heading) * k * step
        py = y + math.sin(heading) * k * step
        row, col = planner.to_cell(px, py)
        if not planner.inside(row, col) or not math.isfinite(planner.cost[row, col]):
            return False
        h = planner.height[row, col]
        if abs(h - prev) > 0.2:
            return False
        prev = h
    return True


def park_heading(block, x, y, rng):
    """Along the nearest park path segment, either way."""
    best = None
    for x0, y0, x1, y1 in block["park"]["segments"]:
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / ((dx * dx + dy * dy) or 1e-9)))
        d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        if best is None or d < best[0]:
            best = (d, math.atan2(dy, dx))
    return best[1] + (math.pi if rng.random() < 0.5 else 0.0)


def street_heading(layout, x, y, rng):
    """Old town: along the nearest street centerline, either way."""
    best = None
    for seg in layout["streets"]:
        (x0, y0), (x1, y1) = seg["a"], seg["b"]
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / ((dx * dx + dy * dy) or 1e-9)))
        d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        if best is None or d < best[0]:
            best = (d, math.atan2(dy, dx))
    return best[1] + (math.pi if rng.random() < 0.5 else 0.0)


def vary_camera(cam, heading, rng):
    """World 3: hand-held phone variety: lens 24-35 mm, eye height 1.4-1.75 m, pitch and a slight roll."""
    from mathutils import Quaternion
    ground = cam["ground_z"]
    cam.data.lens = rng.uniform(24.0, 35.0)
    cam.location.z = ground + rng.uniform(1.4, 1.75)
    rot = look_rotation(cam.location, heading, look_height=rng.uniform(0.2, 1.6), look_dist=rng.uniform(12.0, 18.0),
                        ground_z=ground).to_quaternion()
    roll = Quaternion((0.0, 0.0, 1.0), math.radians(rng.uniform(-2.5, 2.5)))
    cam.rotation_euler = (rot @ roll).to_euler()
    cam["lens_mm"] = cam.data.lens


def campus_heading(layout, face, x, y, cls, rng):
    """World 3: along the promenade, the pier and the paths, across plazas, along the walkways."""
    qy = layout["quay_y"]
    if y < qy - 0.5:   # pier or a lower quay ledge
        pier = layout["pier"]
        along = math.pi / 2 if abs(x - pier["x"]) < pier["width"] else 0.0
        return along + (math.pi if rng.random() < 0.5 else 0.0)
    if y < qy + 14.0:  # the promenade
        roll = rng.random()
        if roll < 0.72:
            return rng.choice((0.0, math.pi))
        if roll < 0.88:
            return -math.pi / 2 + math.radians(rng.uniform(-45, 45))   # out over the water
        return math.pi / 2 + math.radians(rng.uniform(-45, 45))
    from worldlib.oldtown_layout import point_in_poly
    if point_in_poly((x, y), face["inset"]):
        paths = face.get("paths") or []
        best = None
        for (x0, y0), (x1, y1) in paths:
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / ((dx * dx + dy * dy) or 1e-9)))
            d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
            if best is None or d < best[0]:
                best = (d, math.atan2(dy, dx))
        if best is not None and best[0] < 2.5 and rng.random() < 0.8:
            return best[1] + (math.pi if rng.random() < 0.5 else 0.0)
        cx = sum(p[0] for p in face["inset"]) / 4
        cy = sum(p[1] for p in face["inset"]) / 4
        return math.atan2(cy - y, cx - x) + math.radians(rng.uniform(-75, 75))
    return street_heading(layout, x, y, rng)


def stair_spots(rng):
    """Old town: (x, y, heading) spots 6-12 m from stairs and ramps, facing them, from above and below."""
    spots = []
    for obj in bpy.context.scene.objects:
        if obj.get("category") not in ("stairs", "ramp"):
            continue
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        gx, gy = sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8
        for _ in range(10):
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(6.0, 12.0)
            x, y = gx + math.cos(a) * d, gy + math.sin(a) * d
            spots.append((x, y, math.atan2(gy - y, gx - x)))
    return spots


def feature_spots(layout, rng):
    """(x, y, heading) camera spots 7-12 m in front of entrances, bridges and skate platforms, facing them."""
    spots = []
    for ent in layout.get("entrances", []):
        fx, fy = ent["facing"]
        gx, gy = ent["xy"]
        for _ in range(2):
            d, lateral = rng.uniform(7.0, 12.0), rng.uniform(-3.0, 3.0)
            x, y = gx + fx * d - fy * lateral, gy + fy * d + fx * lateral
            spots.append((x, y, math.atan2(gy - y, gx - x)))
    for block in layout["blocks"].values():
        park = block.get("park")
        if park:
            br = park["bridge"]
            for sign in (-1.0, 1.0):
                d = br["half_len"] + rng.uniform(5.0, 9.0)
                x, y = (br["x"] + sign * d, br["y"]) if br["axis"] == "x" else (br["x"], br["y"] + sign * d)
                spots.append((x, y, math.atan2(br["y"] - y, br["x"] - x)))
    for plat in layout.get("platforms", []):
        gx, gy = plat["goal"]
        for _ in range(3):
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(7.0, 11.0)
            x, y = gx + math.cos(a) * d, gy + math.sin(a) * d
            spots.append((x, y, math.atan2(gy - y, gx - x)))
    if layout.get("world") in ("oldtown", "campus"):
        spots += stair_spots(rng)
    rng.shuffle(spots)
    return spots


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data = np.load(os.path.join(world_dir(), "navmap.npz"))
    planner = HumanPlanner(data)
    with open(os.path.join(world_dir(), "layout.json"), encoding="utf-8") as handle:
        layout = json.load(handle)

    views = get_collection("Views")
    if args.replace:
        for obj in list(views.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    existing = [obj for obj in views.objects if obj.type == "CAMERA"]
    taken = [(obj.location.x, obj.location.y, camera_heading(obj)) for obj in existing]
    state = {"next": 1 + max([int(o.name.split("_")[-1]) for o in existing if o.name.split("_")[-1].isdigit()] or [-1]),
             "placed": 0, "tried": 0}
    scene = bpy.context.scene
    vis = VisTree()
    oldtown = layout.get("world") == "oldtown"
    campus = layout.get("world") == "campus"
    # World 2 and 3 names keep sample ids and image names unique.
    prefix = "ot_view" if oldtown else ("cw_view" if campus else "view")

    def try_camera(x, y, heading, spacing, allowed=None):
        allowed = allowed or stand_ids
        block, _side = lay.block_side_at(layout, x, y)
        if block is None or (args.new_only and not block.get("new")):
            return False
        row, col = planner.to_cell(x, y)
        if not planner.inside(row, col) or int(planner.cls[row, col]) not in allowed or not math.isfinite(planner.cost[row, col]):
            return False
        for tx, ty, th in taken:
            same_dir = math.cos(th - heading) > 0.0
            if math.hypot(tx - x, ty - y) < spacing * (1.0 if same_dir else 0.5):
                return False
        if not clear_ahead(planner, x, y, heading):
            return False
        if layout.get("world") in ("oldtown", "campus"):
            nx0, nx1, ny0, ny1 = layout["nav_extent"]
            ax, ay = x + math.cos(heading) * 12.0, y + math.sin(heading) * 12.0
            if not (nx0 <= ax <= nx1 and ny0 <= ay <= ny1):
                return False  # facing out of the town
        state["tried"] += 1
        name = f"{prefix}_{state['next']:04d}"
        cam = make_view_camera(name, x, y, planner.height_at(x, y), heading)
        if campus:
            vary_camera(cam, heading, random.Random(name + "/camera"))
        cam["block"] = block["id"]
        cam["variant"] = block["variant"]
        bpy.context.view_layer.update()
        tasks = ViewTasks(scene, cam, planner, layout, rng=random.Random(name), vis=vis).generate(max_tasks=1)
        if not tasks:
            bpy.data.objects.remove(cam, do_unlink=True)
            return False
        cam["preview_task"] = tasks[0]["task"]
        taken.append((x, y, heading))
        state["placed"] += 1
        state["next"] += 1
        if state["placed"] % 50 == 0:
            bpy.ops.wm.save_mainfile()  # keep progress if the run is interrupted
        print(f"view_placed {name} block={block['id']} variant={block['variant']} task={tasks[0]['task']!r}")
        return True

    stand_ids = {CLASS_ID[name] for name in STAND_CLASSES}
    if args.targeted:
        for x, y, heading in feature_spots(layout, rng):
            if state["placed"] >= args.count:
                break
            try_camera(x, y, heading + math.radians(rng.uniform(-8, 8)), args.min_spacing * 0.6,
                       allowed=stand_ids | {CLASS_ID["grass"], CLASS_ID["driveway"]})
        print(f"targeted_placed {state['placed']}")

    rows, cols = np.nonzero(np.isin(planner.cls, list(stand_ids)) & (planner.cost <= 1.0))
    order = list(range(len(rows)))
    rng.shuffle(order)
    print(f"candidate_cells {len(order)}")
    for index in order:
        if state["placed"] >= args.count:
            break
        x, y = planner.to_world(int(rows[index]), int(cols[index]))
        block, side = lay.block_side_at(layout, x, y)
        if block is None or (args.new_only and not block.get("new")):
            continue
        if campus:
            heading = campus_heading(layout, block, x, y, int(planner.cls[rows[index], cols[index]]), rng)
        elif oldtown:
            cls = int(planner.cls[rows[index], cols[index]])
            if cls == CLASS_ID["plaza"]:
                # In a square: look across it, roughly toward its middle.
                cx = sum(p[0] for p in block["inset"]) / 4
                cy = sum(p[1] for p in block["inset"]) / 4
                heading = math.atan2(cy - y, cx - x) + math.radians(rng.uniform(-70, 70))
            else:
                heading = street_heading(layout, x, y, rng)
        elif block.get("park") and int(planner.cls[rows[index], cols[index]]) == CLASS_ID["park_path"]:
            heading = park_heading(block, x, y, rng)
        elif int(planner.cls[rows[index], cols[index]]) == CLASS_ID["skate_area"]:
            heading = rng.choice((0.0, 0.5, 1.0, 1.5)) * math.pi
        else:
            _o, tangent, inward = lay.side_frame(block, side)
            base = math.atan2(tangent[1], tangent[0])
            roll = rng.random()
            if roll < 0.45:
                heading = base
            elif roll < 0.9:
                heading = base + math.pi
            else:
                heading = math.atan2(-inward[1], -inward[0])  # face across the street
        heading += math.radians(rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG))
        try_camera(x, y, heading, args.min_spacing)

    bpy.ops.wm.save_mainfile()
    print(f"place_views_done placed={state['placed']} tried={state['tried']} total_views={len(views.objects)}")


if __name__ == "__main__":
    main()
