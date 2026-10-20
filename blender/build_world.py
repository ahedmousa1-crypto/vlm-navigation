"""Build the persistent street world once and save it.

Writes into --out (default blender/world/):
  world.blend     the world; cameras in "Views" are carried over from the previous
                  world.blend unless --keep_views false
  navmap.npz      top-down class/height grid used for Human paths
  layout.json     the world plan (blocks, variants, crosswalks)
  overview.png    top-down render (skip with --overview false)
  build_qa.json   props that do not sit on the ground

  blender --background --python blender/build_world.py -- --seed 0

Blender only forwards arguments that come after ``--``.
"""

import argparse
import json
import os
import shutil
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldlib import layout as lay  # noqa: E402
from worldlib import navmap  # noqa: E402
from worldlib.animals import AnimalTemplates  # noqa: E402
from worldlib.build import WorldBuilder  # noqa: E402
from worldlib.materials import build_materials  # noqa: E402
from worldlib.meshes import get_collection, world_aabb  # noqa: E402
from worldlib.scene_setup import configure_eevee, make_overview_camera, setup_world  # noqa: E402

GROUND_TOL_M = 0.03


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build the street world.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "world"))
    parser.add_argument("--cell", type=float, default=0.25, help="Navmap cell size in metres.")
    parser.add_argument("--overview", type=lambda v: str(v).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--keep_views", type=lambda v: str(v).lower() in ("true", "1", "yes"), default=True,
                        help="Carry over cameras from an existing world.blend (default true).")
    return parser.parse_args(argv)


def ground_contact_failures(tree):
    failures = []
    checked = 0
    for obj in bpy.context.scene.objects:
        if obj.get("surface_type") != "obstacle":
            continue
        checked += 1
        box = world_aabb(obj)
        cx, cy = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
        ground = tree.ground_below(cx, cy, from_z=box[4] + 0.5)
        if ground is None or abs(box[4] - ground) > GROUND_TOL_M:
            failures.append({"object": obj.name, "lowest_z": round(box[4], 4), "ground_z": None if ground is None else round(ground, 4)})
    print(f"qa_ground_contact checked={checked} failures={len(failures)}")
    return checked, failures


def restore_views(old_blend):
    """Copy the cameras from a previous world.blend's "Views" collection, so a rebuild keeps hand edits."""
    if not os.path.isfile(old_blend):
        return 0
    # Read from a copy: Blender refuses to overwrite a file it has loaded data from.
    source = old_blend + ".views_tmp"
    shutil.copyfile(old_blend, source)
    with bpy.data.libraries.load(source, link=False) as (src, dst):
        dst.collections = [name for name in src.collections if name == "Views"]
    for library in list(bpy.data.libraries):
        if os.path.normcase(bpy.path.abspath(library.filepath)) == os.path.normcase(source):
            bpy.data.libraries.remove(library)
    os.remove(source)
    old = dst.collections[0] if dst.collections else None
    views = get_collection("Views")
    kept = 0
    if old is not None:
        for obj in list(old.objects):
            if obj.type == "CAMERA":
                views.objects.link(obj)
                kept += 1
        bpy.data.collections.remove(old)
    print(f"views_restored {kept} from {old_blend}")
    return kept


def render_overview(out_dir, extent):
    scene = bpy.context.scene
    cam = make_overview_camera(extent)
    scene.camera = cam
    res = (scene.render.resolution_x, scene.render.resolution_y)
    scene.render.resolution_x = scene.render.resolution_y = 2048
    final = os.path.join(out_dir, "overview.png")
    tmp = os.path.join(out_dir, "overview_tmp.png")
    scene.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.camera = None
    try:
        os.replace(tmp, final)
    except PermissionError:
        # overview.png is open in an image viewer; keep the new render beside it.
        final = os.path.join(out_dir, "overview_new.png")
        os.replace(tmp, final)
    print("overview_saved", final)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    start = time.perf_counter()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    layout = lay.make_layout(args.seed)
    print(lay.describe(layout))

    animals = AnimalTemplates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
    if not animals:
        print("animals_missing run tools/fetch_assets.py to add cats and dogs")
    builder = WorldBuilder(layout, build_materials(), animals)
    builder.build()
    get_collection("Views")
    blend_path = os.path.join(out_dir, "world.blend")
    if args.keep_views:
        restore_views(blend_path)
    setup_world()
    configure_eevee(bpy.context.scene)
    bpy.context.view_layer.update()
    print(f"objects={len(bpy.context.scene.objects)} build_time={time.perf_counter() - start:.1f}s")

    tree = navmap.SceneTree()
    checked, failures = ground_contact_failures(tree)
    with open(os.path.join(out_dir, "build_qa.json"), "w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "obstacles_checked": checked, "ground_contact_failures": failures}, handle, indent=2)

    grid = navmap.rasterize(tree, layout["nav_extent"], cell=args.cell)
    navmap.save(grid, os.path.join(out_dir, "navmap.npz"))
    with open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=1)

    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    print("world_saved", blend_path)

    # Rendered after saving so the overview camera stays out of world.blend.
    if args.overview:
        render_overview(out_dir, layout["extent"])
    print(f"build_world_done total={time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
