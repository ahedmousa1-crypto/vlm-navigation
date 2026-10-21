"""Recompute pixel traces for existing samples from their stored world paths (no re-render).

Each record's ``synthetic.world_paths`` is projected through its own camera and
simplified with the current rules (corners kept, then resampled to 6-15 points),
and ``ground_truth["Human"]`` is replaced. Records are updated in place.

  blender --background blender/world/world.blend --python blender/retrace.py -- --dataset dataset
"""

import argparse
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from worldlib.scene_setup import RES_X, RES_Y  # noqa: E402
from worldlib.tasks import project, simplify_trace  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Recompute traces from stored world paths.")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(HERE), "dataset"))
    args = parser.parse_args(argv)
    path = os.path.join(os.path.abspath(args.dataset), "samples.jsonl")
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = RES_X, RES_Y
    changed, missing = 0, 0
    before, after = [], []
    for record in records:
        cam = scene.objects.get(record["view"])
        world_paths = record.get("synthetic", {}).get("world_paths")
        if cam is None or not world_paths:
            missing += 1
            continue
        new_paths = []
        for points in world_paths:
            pix = [project(scene, cam, p) for p in points]
            dense = [(min(max(px, 0.0), RES_X - 0.5), min(max(py, 0.0), RES_Y - 0.5)) for px, py, _d in pix]
            new_paths.append(simplify_trace(dense))
        before.extend(len(p) for p in record["ground_truth"]["Human"])
        after.extend(len(p) for p in new_paths)
        record["ground_truth"]["Human"] = new_paths
        record.setdefault("synthetic", {})["retraced"] = True
        changed += 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    print(f"retrace_done records={changed} missing={missing} points_before={mean(before):.1f} "
          f"points_after={mean(after):.1f} min={min(after or [0])} max={max(after or [0])}")


if __name__ == "__main__":
    main()
