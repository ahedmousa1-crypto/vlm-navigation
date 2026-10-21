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
from worldlib.scene_setup import make_view_camera  # noqa: E402
from worldlib.tasks import ViewTasks  # noqa: E402

STAND_CLASSES = ("sidewalk", "plaza", "park_path", "skate_area")
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
        state["tried"] += 1
        name = f"view_{state['next']:04d}"
        cam = make_view_camera(name, x, y, planner.height_at(x, y), heading)
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
        if block.get("park") and int(planner.cls[rows[index], cols[index]]) == CLASS_ID["park_path"]:
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
