"""Auto-place candidate cameras in world.blend's "Views" collection for review.

Cameras stand at eye height on sidewalks, plazas, and the terrace, mostly
looking along the street. A spot is kept only when the ground ahead is open
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
from worldlib.planner import HumanPlanner  # noqa: E402
from worldlib.scene_setup import make_view_camera  # noqa: E402
from worldlib.tasks import ViewTasks  # noqa: E402

STAND_CLASSES = ("sidewalk", "plaza")
CLEAR_AHEAD_M = 7.0
HEADING_JITTER_DEG = 14.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Auto-place candidate view cameras.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_spacing", type=float, default=8.0)
    parser.add_argument("--replace", action="store_true", help="Delete existing view cameras first.")
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
    next_index = 1 + max([int(obj.name.split("_")[-1]) for obj in existing if obj.name.split("_")[-1].isdigit()] or [-1])

    stand_ids = [CLASS_ID[name] for name in STAND_CLASSES]
    rows, cols = np.nonzero(np.isin(planner.cls, stand_ids) & (planner.cost <= 1.0))
    order = list(range(len(rows)))
    rng.shuffle(order)
    print(f"candidate_cells {len(order)}")

    scene = bpy.context.scene
    placed, tried = 0, 0
    for index in order:
        if placed >= args.count:
            break
        x, y = planner.to_world(int(rows[index]), int(cols[index]))
        block, side = lay.block_side_at(layout, x, y)
        if block is None:
            continue
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

        too_close = False
        for tx, ty, th in taken:
            gap = math.hypot(tx - x, ty - y)
            same_dir = math.cos(th - heading) > 0.0
            if gap < args.min_spacing * (1.0 if same_dir else 0.5):
                too_close = True
                break
        if too_close or not clear_ahead(planner, x, y, heading):
            continue

        tried += 1
        name = f"view_{next_index:04d}"
        ground_z = planner.height_at(x, y)
        cam = make_view_camera(name, x, y, ground_z, heading)
        cam["block"] = block["id"]
        cam["variant"] = block["variant"]
        bpy.context.view_layer.update()
        tasks = ViewTasks(scene, cam, planner, layout, rng=random.Random(name)).generate(max_tasks=1)
        if not tasks:
            bpy.data.objects.remove(cam, do_unlink=True)
            continue
        cam["preview_task"] = tasks[0]["task"]
        taken.append((x, y, heading))
        placed += 1
        next_index += 1
        print(f"view_placed {name} block={block['id']} variant={block['variant']} task={tasks[0]['task']!r}")

    bpy.ops.wm.save_mainfile()
    print(f"place_views_done placed={placed} tried={tried} total_views={len(views.objects)}")


if __name__ == "__main__":
    main()
