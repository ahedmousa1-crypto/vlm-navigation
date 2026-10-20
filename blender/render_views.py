"""Render every camera in world.blend's "Views" collection into NaviTrace-style samples.

For each view: plan Human tasks, render the RGB image (EEVEE, legacy look) and
a flat-color class mask (Workbench), then write one record per task.

  blender --background blender/world/world.blend --python blender/render_views.py -- --out dataset

Outputs under --out:
  images/<view>.png            RGB render, 1365x1024
  work/masks_rgb/<view>.png    class colors; tools/build_dataset.py turns these into id masks
  samples.jsonl                one record per task (records for re-rendered views are replaced)
  render_log.csv               view, status, tasks, seconds
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import uuid

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from trace_format import empty_ground_truth, project_trace_to_pixels, set_embodiment_paths  # noqa: E402
from worldlib.assets_look import AssetLook  # noqa: E402
from worldlib.classes import CLASSES, class_for  # noqa: E402
from worldlib.lighting import PRESETS, Lighting, preset_for  # noqa: E402
from worldlib.planner import HumanPlanner  # noqa: E402
from worldlib.scene_setup import RES_X, RES_Y, configure_eevee, configure_mask_render  # noqa: E402
from worldlib.tasks import ViewTasks  # noqa: E402

SAMPLE_NS = uuid.UUID("6f1c3a52-5b0e-4f7e-9d2a-7c1b0e4a9f10")
METADATA = {
    "city": "Synthetic",
    "country": "Synthetic",
    "lighting_conditions": "Daylight",
    "natural_structured": "Structured",
    "task_type": "Goal",
    "urban_rural": "Urban",
    "weather_conditions": "Clear",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render view cameras into dataset samples.")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "dataset"))
    parser.add_argument("--views", nargs="*", default=None, help="Camera names to render (default: all).")
    parser.add_argument("--limit", type=int, default=0, help="Render at most this many views (0 = all).")
    parser.add_argument("--max_tasks", type=int, default=2, help="Tasks (samples) per view.")
    parser.add_argument("--samples", type=int, default=48, help="EEVEE render samples.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip views that already have an image.")
    parser.add_argument("--look", choices=("procedural", "assets"), default="procedural",
                        help="'assets' swaps in Poly Haven HDRI skies and ground textures (tools/fetch_assets.py).")
    parser.add_argument("--lighting", choices=("mix",) + tuple(PRESETS), default="mix",
                        help="'mix' picks one preset per view (55%% day, 15%% overcast, 15%% dusk, 10%% night, "
                             "5%% moonlight); a preset name uses it for every view. Procedural look only.")
    return parser.parse_args(argv)


def srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def assign_mask_colors():
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        cls = class_for(obj.get("surface_type"), obj.get("category"))
        rgb = CLASSES[cls][1]
        obj.color = (*[srgb_to_linear(v) for v in rgb], 1.0)


def load_records(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_records(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def make_record(cam, layout, view_tasks, task, index, look):
    for path in task["paths"]:
        project_trace_to_pixels(path, RES_X, RES_Y)  # rejects normalized coordinates
    ground_truth = empty_ground_truth()
    set_embodiment_paths(ground_truth, "Human", task["paths"])
    loc = cam.matrix_world.translation
    return {
        "sample_id": str(uuid.uuid5(SAMPLE_NS, f"{layout['seed']}/{cam.name}/{index}")),
        "view": cam.name,
        "image": f"images/{cam.name}.png",
        "segmentation_mask": f"masks/{cam.name}.png",
        "task": task["task"],
        "embodiments": ["Human"],
        "ground_truth": ground_truth,
        "category": task["category"],
        "context": view_tasks.context([task]),
        "metadata": {**METADATA, "weather_conditions": look["weather"], "lighting_conditions": look["lighting"]},
        "synthetic": {
            "look": look["name"],
            "hdri": look["hdri"],
            "lighting_preset": look["preset"],
            "world_seed": layout["seed"],
            "block": view_tasks.block["id"] if view_tasks.block else None,
            "variant": view_tasks.block["variant"] if view_tasks.block else None,
            "target_kind": task["target_kind"],
            "target_object": task["target_object"],
            "path_surfaces": task["path_surfaces"],
            "world_paths": task["world_paths"],
            "camera": {
                "location": [round(v, 3) for v in loc],
                "heading_deg": round(math.degrees(view_tasks.heading), 2),
                "lens_mm": cam.data.lens,
                "sensor_mm": cam.data.sensor_width,
            },
        },
    }


def render_to(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    world = os.path.dirname(bpy.data.filepath)
    for sub in ("images", os.path.join("work", "masks_rgb")):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    planner = HumanPlanner(np.load(os.path.join(world, "navmap.npz")))
    with open(os.path.join(world, "layout.json"), encoding="utf-8") as handle:
        layout = json.load(handle)

    scene = bpy.context.scene
    views = bpy.data.collections.get("Views")
    cams = sorted((obj for obj in (views.objects if views else []) if obj.type == "CAMERA"), key=lambda o: o.name)
    if args.views:
        cams = [cam for cam in cams if cam.name in set(args.views)]
    if args.skip_existing:
        cams = [cam for cam in cams if not os.path.isfile(os.path.join(out_dir, "images", cam.name + ".png"))]
    if args.limit:
        cams = cams[: args.limit]
    print(f"render_views count={len(cams)} out={out_dir}")

    assign_mask_colors()
    asset_look = None
    lighting = None
    if args.look == "assets":
        asset_look = AssetLook(os.path.join(HERE, "assets"))
        asset_look.apply_textures()
    else:
        lighting = Lighting(scene)
    samples_path = os.path.join(out_dir, "samples.jsonl")
    records = load_records(samples_path)
    log_path = os.path.join(out_dir, "render_log.csv")
    new_log = not os.path.isfile(log_path)
    total_start = time.perf_counter()
    rendered = 0
    with open(log_path, "a", newline="", encoding="utf-8") as log_handle:
        log = csv.writer(log_handle)
        if new_log:
            log.writerow(["view", "status", "tasks", "seconds"])
        for cam in cams:
            start = time.perf_counter()
            scene.camera = cam
            look = {"name": args.look, "hdri": None, "weather": "Clear", "lighting": "Daylight", "preset": "day"}
            light_info = None
            if lighting is not None:
                light_info = lighting.apply(preset_for(cam.name, args.lighting))
                look.update(weather=light_info["weather"], lighting=light_info["lighting"], preset=light_info["preset"])
            elif asset_look is not None:
                look["hdri"], look["weather"] = asset_look.apply_sky(cam.name)
            bpy.context.view_layer.update()
            view_tasks = ViewTasks(scene, cam, planner, layout, rng=random.Random(cam.name), lighting=light_info)
            tasks = view_tasks.generate(max_tasks=args.max_tasks)
            records = [r for r in records if r.get("view") != cam.name]
            if not tasks:
                log.writerow([cam.name, "no_task", 0, f"{time.perf_counter() - start:.1f}"])
                print(f"view_skipped {cam.name} no valid task")
                continue

            configure_eevee(scene, samples=args.samples)
            if light_info is not None:
                scene.view_settings.exposure = PRESETS[light_info["preset"]]["exposure"]
            render_to(scene, os.path.join(out_dir, "images", cam.name + ".png"))
            configure_mask_render(scene)
            render_to(scene, os.path.join(out_dir, "work", "masks_rgb", cam.name + ".png"))

            for index, task in enumerate(tasks):
                records.append(make_record(cam, layout, view_tasks, task, index, look))
            write_records(samples_path, records)
            seconds = time.perf_counter() - start
            log.writerow([cam.name, "ok", len(tasks), f"{seconds:.1f}"])
            log_handle.flush()
            rendered += 1
            print(f"view_done {cam.name} tasks={[t['task'] for t in tasks]} seconds={seconds:.1f}")

    elapsed = time.perf_counter() - total_start
    print(f"render_views_done rendered={rendered}/{len(cams)} records={len(records)} "
          f"total={elapsed:.1f}s avg={elapsed / max(1, rendered):.1f}s")


if __name__ == "__main__":
    main()
