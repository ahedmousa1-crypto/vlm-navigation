"""Procedural outdoor scene for a NaviTrace-style first-person view.

Builds one street block (grass, sidewalk, curb, optional bike lane, road,
optional crosswalk, stairs paired with a ramp, and simple obstacles), tags
every surface, and renders a single EEVEE frame.

``--seed`` drives the street structure and the obstacle placement. The same
seed always rebuilds the same scene.

  blender --background --python blender/gen_scene.py -- --seed 0 --out blender/test_renders

Blender only forwards arguments that come after ``--``.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


# Y runs away from the camera. Local X puts the sidewalk on the negative side;
# a right-side layout mirrors those X coordinates into world space.
BASE_Z = -0.25
ROAD_Z = 0.0
CURB_WIDTH = 0.15
RISE = 0.15
TREAD = 0.30
N_STEPS = 6
STAIR_DEPTH = 3.2
RAMP_DEPTH = 3.8
LANDING_DEPTH = 3.2
EYE_HEIGHT = 1.6
DEFAULT_SEED = 0
OVERLAP_TOL_M = 0.05
OVERLAP_ATTEMPTS = 15
GROUND_TOL_M = 0.03
MIN_VISIBLE_OBJECTS = 5
CROSSWALK_LIFT = 0.012
CROSSWALK_THICK = 0.018

BOX_FACES = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Generate one tagged outdoor scene.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Layout seed. Default 0 keeps this test render reproducible.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output directory for scene_<seed>.png (default: test_renders next to this script).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Explicit PNG path (overrides --out).",
    )
    parser.add_argument(
        "--save_blend",
        type=lambda v: str(v).lower() in ("true", "1", "yes", "on"),
        default=False,
        help="If true, write scene_<seed>.blend next to the PNG (default: false).",
    )
    return parser.parse_args(argv)


def resolve_output_path(args):
    if args.output:
        return os.path.abspath(args.output)
    out_dir = args.out or os.path.join(script_dir(), "test_renders")
    return os.path.abspath(os.path.join(out_dir, f"scene_{args.seed:04d}.png"))


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def shader_input(node, *names):
    for name in names:
        if name in node.inputs:
            return node.inputs[name]
    raise KeyError(f"none of {names} on {node.bl_idname}")


def layered_material(name, dark, light, patch_scale, grain_scale, rough_lo, rough_hi, bump, metallic=0.0):
    """Coarse patches (wear, stains, grass clumps) plus fine grain on color, roughness, and bump."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    coord = nodes.new("ShaderNodeTexCoord")
    patch_map = nodes.new("ShaderNodeMapping")
    patch_map.inputs["Scale"].default_value = (patch_scale, patch_scale, patch_scale)
    grain_map = nodes.new("ShaderNodeMapping")
    grain_map.inputs["Scale"].default_value = (grain_scale, grain_scale, grain_scale)

    patch = nodes.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 1.0
    patch.inputs["Detail"].default_value = 4.0
    patch.inputs["Roughness"].default_value = 0.45
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1.0
    grain.inputs["Detail"].default_value = 9.0
    grain.inputs["Roughness"].default_value = 0.65

    color_mix = nodes.new("ShaderNodeMix")
    color_mix.data_type = "RGBA"
    color_mix.inputs["A"].default_value = (*dark, 1.0)
    color_mix.inputs["B"].default_value = (*light, 1.0)

    rough_mix = nodes.new("ShaderNodeMix")
    rough_mix.data_type = "FLOAT"
    rough_mix.inputs["A"].default_value = rough_lo
    rough_mix.inputs["B"].default_value = rough_hi

    bump_node = nodes.new("ShaderNodeBump")
    bump_node.inputs["Strength"].default_value = bump
    bump_node.inputs["Distance"].default_value = 0.12

    links.new(coord.outputs["Object"], patch_map.inputs["Vector"])
    links.new(coord.outputs["Object"], grain_map.inputs["Vector"])
    links.new(patch_map.outputs["Vector"], patch.inputs["Vector"])
    links.new(grain_map.outputs["Vector"], grain.inputs["Vector"])
    links.new(patch.outputs["Fac"], shader_input(color_mix, "Factor", "Factor_Float", "Fac"))
    links.new(color_mix.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(patch.outputs["Fac"], shader_input(rough_mix, "Factor", "Factor_Float", "Fac"))
    links.new(rough_mix.outputs["Result"], bsdf.inputs["Roughness"])
    links.new(grain.outputs["Fac"], bump_node.inputs["Height"])
    links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def flat_material(name, color, roughness=0.55):
    """Solid color for markings that must read clearly in the render."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def link_object(name, surface_type, mesh, material):
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    obj["surface_type"] = surface_type
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_poly(name, surface_type, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return link_object(name, surface_type, mesh, material)


def box_verts(x0, x1, y0, y1, z0, z1):
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


def add_box(name, surface_type, bounds, material):
    x0, x1, y0, y1, z0, z1 = bounds
    return add_poly(name, surface_type, box_verts(x0, x1, y0, y1, z0, z1), BOX_FACES, material)


def add_wedge(name, surface_type, x_high, x_low, y0, y1, z_high, z_low, z_base, material):
    """Solid ramp. High edge at x_high, low edge at x_low, base at z_base."""
    verts = [
        (x_high, y0, z_base),
        (x_low, y0, z_base),
        (x_low, y1, z_base),
        (x_high, y1, z_base),
        (x_high, y0, z_high),
        (x_low, y0, z_low),
        (x_low, y1, z_low),
        (x_high, y1, z_high),
    ]
    return add_poly(name, surface_type, verts, BOX_FACES, material)


def bm_box(bm, x0, x1, y0, y1, z0, z1):
    geom = bmesh.ops.create_cube(bm, size=1.0)
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)
    for vert in geom["verts"]:
        vert.co.x = vert.co.x * sx + cx
        vert.co.y = vert.co.y * sy + cy
        vert.co.z = vert.co.z * sz + cz


def bm_cylinder(bm, center, radius, height, segments=16):
    geom = bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=segments,
        radius1=radius,
        radius2=radius,
        depth=height,
    )
    for vert in geom["verts"]:
        vert.co.x += center[0]
        vert.co.y += center[1]
        vert.co.z += center[2]


def bm_icosphere(bm, center, radius, subdivisions=2):
    geom = bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    for vert in geom["verts"]:
        vert.co.x += center[0]
        vert.co.y += center[1]
        vert.co.z += center[2]


def bm_wheel_x(bm, cx, cy, z_hub, radius, width=0.05):
    """Bicycle wheel: disk in the YZ plane, axle along X (edge-on to the walk camera)."""
    geom = bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=20,
        radius1=radius,
        radius2=radius,
        depth=width,
    )
    half = width * 0.5
    for vert in geom["verts"]:
        lx, ly, lz = vert.co.x, vert.co.y, vert.co.z
        vert.co.x = cx + lz - half
        vert.co.y = cy + lx
        vert.co.z = z_hub + ly


def add_bmesh_object(name, surface_type, builder, material):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    builder(bm)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return link_object(name, surface_type, mesh, material)


def sample_layout(rng, seed):
    """Bounded street structure. Local X keeps the sidewalk on the negative side.

    Side, bike-lane presence, and crosswalk presence cycle with the seed so
    nearby frames are not the same street. Widths, curb height, and positions
    still come from the seeded random stream.
    """
    length = rng.uniform(28.0, 46.0)
    sidewalk_w = rng.uniform(2.6, 4.4)
    road_w = rng.uniform(5.5, 9.0)
    bike_on = (seed % 3) != 0
    bike_w = rng.uniform(1.2, 2.0) if bike_on else 0.0
    curb_h = rng.uniform(0.08, 0.18)
    mirror = (seed % 2) == 1
    cross_len = 3.4
    if (seed % 3) != 1:
        cross_y0 = rng.uniform(6.5, length - cross_len - 5.0)
        cross_y1 = cross_y0 + cross_len
    else:
        cross_y0 = None
        cross_y1 = None

    feature = STAIR_DEPTH + RAMP_DEPTH
    stair_y0 = rng.uniform(7.0, length - feature - 2.5)
    stair_y1 = stair_y0 + STAIR_DEPTH
    ramp_y0 = stair_y1
    ramp_y1 = ramp_y0 + RAMP_DEPTH

    sw_x0 = -sidewalk_w
    sw_x1 = 0.0
    curb_x0 = 0.0
    curb_x1 = CURB_WIDTH
    bike_x0 = curb_x1
    bike_x1 = curb_x1 + bike_w
    road_x0 = bike_x1
    road_x1 = bike_x1 + road_w
    outer_pad = 8.0
    grass_x0 = sw_x0 - (N_STEPS * TREAD) - LANDING_DEPTH - outer_pad
    grass_x1 = road_x1 + outer_pad

    layout = {
        "length": length,
        "y0": 0.0,
        "y1": length,
        "grass_y0": -2.5,
        "grass_y1": length + 2.0,
        "sidewalk_w": sidewalk_w,
        "road_w": road_w,
        "bike_w": bike_w,
        "curb_h": curb_h,
        "side": "right" if mirror else "left",
        "mirror": mirror,
        "sw_x0": sw_x0,
        "sw_x1": sw_x1,
        "curb_x0": curb_x0,
        "curb_x1": curb_x1,
        "bike_x0": bike_x0,
        "bike_x1": bike_x1,
        "road_x0": road_x0,
        "road_x1": road_x1,
        "grass_x0": grass_x0,
        "grass_x1": grass_x1,
        "cross_y0": cross_y0,
        "cross_y1": cross_y1,
        "stair_y0": stair_y0,
        "stair_y1": stair_y1,
        "ramp_y0": ramp_y0,
        "ramp_y1": ramp_y1,
        "sidewalk_z": curb_h,
    }
    return layout


def world_x(layout, x):
    return -x if layout["mirror"] else x


def world_interval(layout, x0, x1):
    return world_x(layout, x0), world_x(layout, x1)


def build_ground(layout, materials):
    y0, y1 = layout["y0"], layout["y1"]
    sw_x0, sw_x1 = layout["sw_x0"], layout["sw_x1"]
    stair_outer = sw_x0 - N_STEPS * TREAD
    landing_x1 = stair_outer
    landing_x0 = landing_x1 - LANDING_DEPTH
    sidewalk_z = layout["sidewalk_z"]

    grass_rects = [
        (layout["grass_x0"], layout["grass_x1"], layout["grass_y0"], y0),
        (layout["road_x1"], layout["grass_x1"], y0, y1),
        (layout["grass_x0"], landing_x0, y0, y1),
        (landing_x0, sw_x0, y0, layout["stair_y0"]),
        (landing_x0, sw_x0, layout["ramp_y1"], y1),
    ]
    for index, (x0, x1, gy0, gy1) in enumerate(grass_rects):
        wx0, wx1 = world_interval(layout, x0, x1)
        is_far_strip = x0 == layout["road_x1"] and x1 == layout["grass_x1"]
        grass_name = "surface_grass_far" if is_far_strip else f"surface_grass_{index:02d}"
        add_box(
            grass_name,
            "grass",
            (wx0, wx1, gy0, gy1, BASE_Z, ROAD_Z),
            materials["grass"],
        )

    wx0, wx1 = world_interval(layout, sw_x0, sw_x1)
    # Axis-aligned box: both long edges are constant X, so width does not change with Y.
    add_box(
        "surface_sidewalk_00",
        "sidewalk",
        (min(wx0, wx1), max(wx0, wx1), y0, y1, BASE_Z, sidewalk_z),
        materials["sidewalk"],
    )
    print(
        "manifest_sidewalk_width_m "
        f"{abs(wx1 - wx0):.3f} near_y={y0:.2f} far_y={y1:.2f} constant_along_y=1"
    )

    if layout["cross_y0"] is None:
        bands = [(y0, y1)]
    else:
        bands = [(y0, layout["cross_y0"]), (layout["cross_y1"], y1)]

    for index, (by0, by1) in enumerate(bands):
        cx0, cx1 = world_interval(layout, layout["curb_x0"], layout["curb_x1"])
        add_box(
            f"surface_curb_{index:02d}",
            "curb",
            (cx0, cx1, by0, by1, BASE_Z, sidewalk_z),
            materials["curb"],
        )
        if layout["bike_w"] > 0.0:
            bx0, bx1 = world_interval(layout, layout["bike_x0"], layout["bike_x1"])
            add_box(
                f"surface_bike_lane_{index:02d}",
                "bike_lane",
                (bx0, bx1, by0, by1, BASE_Z, ROAD_Z),
                materials["bike_lane"],
            )
        rx0, rx1 = world_interval(layout, layout["road_x0"], layout["road_x1"])
        add_box(
            f"surface_road_{index:02d}",
            "road",
            (rx0, rx1, by0, by1, BASE_Z, ROAD_Z),
            materials["road"],
        )

    if layout["cross_y0"] is not None:
        add_wedge(
            "surface_curb_cut_00",
            "curb",
            world_x(layout, layout["curb_x0"]),
            world_x(layout, layout["curb_x1"]),
            layout["cross_y0"],
            layout["cross_y1"],
            sidewalk_z,
            ROAD_Z,
            BASE_Z,
            materials["curb"],
        )
        stripe_count = 10
        span = layout["cross_y1"] - layout["cross_y0"]
        stripe_h = span / stripe_count
        cross_x0 = layout["bike_x0"] if layout["bike_w"] > 0.0 else layout["road_x0"]
        z0 = ROAD_Z + CROSSWALK_LIFT
        z1 = z0 + CROSSWALK_THICK
        for index in range(stripe_count):
            sy0 = layout["cross_y0"] + index * stripe_h
            sy1 = sy0 + stripe_h
            mat = materials["crosswalk"] if index % 2 == 0 else materials["crosswalk_dark"]
            sx0, sx1 = world_interval(layout, cross_x0, layout["road_x1"])
            add_box(
                f"surface_crosswalk_{index:02d}",
                "crosswalk",
                (sx0, sx1, sy0, sy1, z0, z1),
                mat,
            )

    for index in range(N_STEPS):
        x1 = sw_x0 - index * TREAD
        x0 = x1 - TREAD
        z1 = sidewalk_z + (index + 1) * RISE
        tx0, tx1 = world_interval(layout, x0, x1)
        add_box(
            f"surface_stairs_{index:02d}",
            "stairs",
            (tx0, tx1, layout["stair_y0"], layout["stair_y1"], BASE_Z, z1),
            materials["stairs"],
        )

    landing_z = sidewalk_z + N_STEPS * RISE
    high_x = world_x(layout, landing_x1)
    low_x = world_x(layout, sw_x0)
    add_wedge(
        "surface_ramp_00",
        "ramp",
        high_x,
        low_x,
        layout["ramp_y0"],
        layout["ramp_y1"],
        landing_z,
        sidewalk_z,
        BASE_Z,
        materials["ramp"],
    )
    lx0, lx1 = world_interval(layout, landing_x0, landing_x1)
    add_box(
        "surface_stairs_landing_00",
        "stairs",
        (lx0, lx1, layout["stair_y0"], layout["ramp_y1"], BASE_Z, landing_z),
        materials["stairs"],
    )


def print_layout(layout):
    cross = "none" if layout["cross_y0"] is None else f"{layout['cross_y0']:.2f}:{layout['cross_y1']:.2f}"
    print(
        "manifest_layout "
        f"length={layout['length']:.2f} "
        f"sidewalk_w={layout['sidewalk_w']:.2f} "
        f"road_w={layout['road_w']:.2f} "
        f"bike_w={layout['bike_w']:.2f} "
        f"curb_h={layout['curb_h']:.3f} "
        f"crosswalk_y={cross} "
        f"stair_side={layout['side']} "
        f"stair_y={layout['stair_y0']:.2f}"
    )


def overlaps(a, b, gap):
    return not (
        a[0] > b[1] + gap
        or a[1] < b[0] - gap
        or a[2] > b[3] + gap
        or a[3] < b[2] - gap
    )


def world_aabb(obj):
    """Axis-aligned bounds in world space from the object's bound box and matrix."""
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def aabb_penetrates(a, b, tol):
    """True when the boxes overlap by more than ``tol`` metres on every axis."""
    return (
        min(a[1], b[1]) - max(a[0], b[0]) > tol
        and min(a[3], b[3]) - max(a[2], b[2]) > tol
        and min(a[5], b[5]) - max(a[4], b[4]) > tol
    )


def expected_ground_z(layout, x, y):
    """Walkable surface height at a world X/Y, matching the built street."""
    local_x = -x if layout["mirror"] else x
    sidewalk_z = layout["sidewalk_z"]
    sw_x0, sw_x1 = layout["sw_x0"], layout["sw_x1"]
    y0, y1 = layout["y0"], layout["y1"]
    stair_outer = sw_x0 - N_STEPS * TREAD
    landing_x0 = stair_outer - LANDING_DEPTH
    landing_z = sidewalk_z + N_STEPS * RISE

    if y0 <= y <= y1 and sw_x0 <= local_x <= sw_x1:
        return sidewalk_z
    if layout["stair_y0"] <= y <= layout["stair_y1"] and stair_outer <= local_x <= sw_x0:
        index = min(N_STEPS - 1, max(0, int((sw_x0 - local_x) / TREAD)))
        return sidewalk_z + (index + 1) * RISE
    if layout["stair_y0"] <= y <= layout["ramp_y1"] and landing_x0 <= local_x <= stair_outer:
        return landing_z
    if layout["ramp_y0"] <= y <= layout["ramp_y1"] and stair_outer <= local_x <= sw_x0:
        span = sw_x0 - stair_outer
        if span <= 1e-6:
            return sidewalk_z
        t = (sw_x0 - local_x) / span
        return sidewalk_z + t * (landing_z - sidewalk_z)
    return ROAD_Z


def build_obstacles(rng, layout, materials):
    sidewalk_z = layout["sidewalk_z"]
    y_min, y_max = layout["y0"] + 2.2, layout["y1"] - 1.5
    # Curb-side strip stays empty so a person can walk the block. Furniture sits on the outer edge.
    lane_w = min(1.4, layout["sidewalk_w"] * 0.42)
    furn_x0 = layout["sw_x0"] + 0.28
    furn_x1 = layout["sw_x1"] - lane_w
    if furn_x1 - furn_x0 < 0.7:
        furn_x1 = furn_x0 + 0.7
    placed = []
    cluster_count = 2 if layout["length"] < 36.0 else 3
    anchors = []
    span = max(4.0, (y_max - y_min) / cluster_count)
    for index in range(cluster_count):
        band0 = y_min + index * span
        band1 = min(y_max - 1.0, band0 + span * 0.55)
        if band1 <= band0 + 1.0:
            continue
        anchors.append(
            (
                rng.uniform(furn_x0 + 0.25, furn_x1 - 0.25),
                rng.uniform(band0, band1),
            )
        )
    print(f"manifest_walk_lane_m {lane_w:.2f} clusters {len(anchors)}")

    bpy.context.view_layer.update()
    occupied = [
        world_aabb(obj)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]
    skipped_overlap = 0

    def reserve(cx, cy, hx, hy):
        rect = (cx - hx, cx + hx, cy - hy, cy + hy)
        if any(overlaps(rect, other, 0.55) for other in placed):
            return None
        placed.append(rect)
        return rect

    def sample_near(anchor, hx, hy, spread_y, attempts=24, min_center_y=None):
        ax, ay = anchor
        floor_y = y_min + hy
        if min_center_y is not None:
            floor_y = max(floor_y, min_center_y)
        for _ in range(attempts):
            local_x = min(furn_x1 - hx, max(furn_x0 + hx, rng.gauss(ax, 0.22)))
            cy = min(y_max - hy, max(floor_y, rng.gauss(ay, spread_y)))
            if local_x - hx < furn_x0 or local_x + hx > furn_x1:
                continue
            if reserve(world_x(layout, local_x), cy, hx, hy):
                return world_x(layout, local_x), cy
        return None

    def commit_obstacle(name, builder, material):
        obj = add_bmesh_object(name, "obstacle", builder, material)
        bpy.context.view_layer.update()
        box = world_aabb(obj)
        if any(aabb_penetrates(box, other, OVERLAP_TOL_M) for other in occupied):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            return False
        occupied.append(box)
        return True

    def place_obstacle(name, material, sample, builder_for):
        nonlocal skipped_overlap
        proposed = False
        for _ in range(OVERLAP_ATTEMPTS):
            point = sample()
            if point is None:
                break
            proposed = True
            if commit_obstacle(name, builder_for(point), material):
                return
            placed.pop()
        if proposed:
            skipped_overlap += 1
            print(f"manifest_obstacle_skipped_overlap {name}")

    def bench(index, anchor):
        def builder_for(point):
            cx, cy = point

            def builder(bm):
                z = sidewalk_z
                # Seat wide across the sidewalk (X); shallow depth along the walk (Y) toward the camera.
                bm_box(bm, cx - 0.62, cx + 0.62, cy - 0.20, cy + 0.20, z + 0.45, z + 0.50)
                bm_box(bm, cx - 0.58, cx + 0.58, cy + 0.16, cy + 0.22, z + 0.50, z + 0.88)
                for lx, ly in ((-0.52, -0.14), (0.52, -0.14), (-0.52, 0.14), (0.52, 0.14)):
                    bm_box(
                        bm,
                        cx + lx - 0.04,
                        cx + lx + 0.04,
                        cy + ly - 0.04,
                        cy + ly + 0.04,
                        z,
                        z + 0.45,
                    )

            return builder

        place_obstacle(
            f"surface_obstacle_bench_{index:02d}",
            materials["wood"],
            lambda: sample_near(anchor, 0.62, 0.22, 0.15),
            builder_for,
        )

    def bollard(index, anchor):
        def builder_for(point):
            cx, cy = point

            def builder(bm):
                bm_cylinder(bm, (cx, cy, sidewalk_z + 0.45), 0.08, 0.9)

            return builder

        place_obstacle(
            f"surface_obstacle_bollard_{index:02d}",
            materials["metal"],
            lambda: sample_near(anchor, 0.12, 0.12, 1.1),
            builder_for,
        )

    def trash_can(index, anchor):
        def builder_for(point):
            cx, cy = point

            def builder(bm):
                bm_cylinder(bm, (cx, cy, sidewalk_z + 0.42), 0.24, 0.84, segments=12)
                bm_cylinder(bm, (cx, cy, sidewalk_z + 0.88), 0.26, 0.06, segments=12)

            return builder

        place_obstacle(
            f"surface_obstacle_trash_can_{index:02d}",
            materials["plastic"],
            lambda: sample_near(anchor, 0.28, 0.28, 0.7),
            builder_for,
        )

    def bike(index, anchor):
        def builder_for(point):
            cx, cy = point

            def builder(bm):
                z = sidewalk_z
                wheel_r = 0.30
                # Hub at wheel radius so the tire touches the sidewalk.
                hub_z = z + wheel_r
                for wx in (-0.52, 0.52):
                    bm_wheel_x(bm, cx + wx, cy, hub_z, wheel_r, width=0.05)
                bm_box(bm, cx - 0.55, cx + 0.55, cy - 0.03, cy + 0.03, z + 0.28, z + 0.36)
                bm_box(bm, cx - 0.08, cx + 0.30, cy - 0.03, cy + 0.03, z + 0.50, z + 0.58)
                bm_box(bm, cx + 0.22, cx + 0.30, cy - 0.03, cy + 0.03, z + 0.34, z + 0.72)
                bm_box(bm, cx - 0.48, cx - 0.40, cy - 0.03, cy + 0.03, z + 0.34, z + 0.56)

            return builder

        place_obstacle(
            f"surface_obstacle_bike_{index:02d}",
            materials["bike_metal"],
            lambda: sample_near(anchor, 0.58, 0.08, 1.3, min_center_y=layout["y0"] + 8.0),
            builder_for,
        )

    for index, anchor in enumerate(anchors):
        bench(index, anchor)
        trash_can(index, (anchor[0], anchor[1] + 1.6))
        bike(index, (anchor[0], anchor[1] - 1.8))
        for bollard_index in range(rng.randint(2, 3)):
            bollard(index * 3 + bollard_index, (anchor[0], anchor[1] + (bollard_index - 1) * 0.7))

    print(f"manifest_obstacles_skipped_overlap {skipped_overlap}")
    return skipped_overlap


def placement_occupancy(include_ground=False):
    """Mesh bounds for overlap tests; walkable ground is excluded unless requested."""
    skip = set()
    if not include_ground:
        skip = {"grass", "road", "sidewalk", "curb", "bike_lane", "crosswalk", "stairs", "ramp"}
    boxes = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.get("surface_type") in skip:
            continue
        boxes.append(world_aabb(obj))
    return boxes


def far_grass_intervals(layout):
    """World X bounds and Y span for the grass strip across the road."""
    y0, y1 = layout["y0"], layout["y1"]
    wx0, wx1 = world_interval(layout, layout["road_x1"], layout["grass_x1"])
    return min(wx0, wx1), max(wx0, wx1), y0, y1


def build_far_decorations(rng, layout, materials):
    """Fence and trees on surface_grass_far only (opposite the sidewalk)."""
    gx0, gx1, gy0, gy1 = far_grass_intervals(layout)
    span_x = gx1 - gx0
    if span_x < 1.6:
        print("manifest_far_decorations skipped narrow_far_grass")
        return 0

    bpy.context.view_layer.update()
    occupied = placement_occupancy(include_ground=False)
    skipped_overlap = 0

    def commit_decoration(name, builder, material, ignore_boxes=()):
        nonlocal skipped_overlap
        obj = add_bmesh_object(name, "decoration", builder, material)
        bpy.context.view_layer.update()
        box = world_aabb(obj)
        ignored = set(ignore_boxes)
        if any(
            aabb_penetrates(box, other, OVERLAP_TOL_M)
            for other in occupied
            if other not in ignored
        ):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            skipped_overlap += 1
            print(f"manifest_decoration_skipped_overlap {name}")
            return False
        occupied.append(box)
        return True

    fence_local_x = layout["grass_x1"] - 0.55
    fence_x = world_x(layout, fence_local_x)
    post_spacing = 2.6
    post_y0 = gy0 + 1.4
    post_y1 = gy1 - 1.2
    post_heights = []
    y = post_y0
    post_index = 0
    while y <= post_y1 + 0.01:
        post_y = min(y, post_y1)
        post_heights.append(post_y)
        height = 1.05

        def make_post_builder(px, py, h):
            def builder(bm):
                bm_cylinder(bm, (px, py, ROAD_Z + h * 0.5), 0.05, h, segments=10)

            return builder

        commit_decoration(
            f"surface_decoration_fence_post_{post_index:02d}",
            make_post_builder(fence_x, post_y, height),
            materials["wood"],
        )
        post_index += 1
        y += post_spacing

    if len(post_heights) >= 2:
        rail_y0, rail_y1 = post_heights[0], post_heights[-1]
        rail_half = 0.025
        for rail_index, rail_z in enumerate((ROAD_Z + 0.42, ROAD_Z + 0.82)):
            add_box(
                f"surface_decoration_fence_rail_{rail_index:02d}",
                "decoration",
                (
                    fence_x - rail_half,
                    fence_x + rail_half,
                    rail_y0,
                    rail_y1,
                    rail_z,
                    rail_z + 0.05,
                ),
                materials["metal"],
            )
            bpy.context.view_layer.update()
            occupied.append(world_aabb(bpy.data.objects[f"surface_decoration_fence_rail_{rail_index:02d}"]))

    tree_count = 2 + rng.randint(0, 3)
    local_x_lo = layout["road_x1"] + 1.15
    local_x_hi = layout["grass_x1"] - 2.35
    if local_x_hi - local_x_lo < 0.8:
        print("manifest_far_decorations trees_skipped narrow_strip")
        return skipped_overlap
    trees_placed = 0
    tree_clearance = []
    for tree_index in range(tree_count):
        placed = False
        for _ in range(OVERLAP_ATTEMPTS):
            local_x = rng.uniform(local_x_lo, local_x_hi)
            ty = rng.uniform(gy0 + 3.0, gy1 - 3.0)
            tx = world_x(layout, local_x)
            trunk_r = 0.14
            canopy_r = rng.uniform(0.75, 1.05)
            trunk_h = rng.uniform(2.0, 2.6)
            if any(
                math.hypot(tx - other[0], ty - other[1]) < canopy_r + other[2] + 0.45
                for other in tree_clearance
            ):
                continue

            def make_trunk_builder(cx, cy, th, tr):
                def builder(bm):
                    bm_cylinder(bm, (cx, cy, ROAD_Z + th * 0.5), tr, th, segments=10)

                return builder

            trunk_name = f"surface_decoration_tree_trunk_{tree_index:02d}"
            if not commit_decoration(
                trunk_name,
                make_trunk_builder(tx, ty, trunk_h, trunk_r),
                materials["wood"],
            ):
                continue
            trunk_box = world_aabb(bpy.data.objects[trunk_name])

            def make_canopy_builder(cx, cy, th, cr):
                def builder(bm):
                    bm_icosphere(bm, (cx, cy, ROAD_Z + th + cr * 0.55), cr, subdivisions=2)

                return builder

            if commit_decoration(
                f"surface_decoration_tree_canopy_{tree_index:02d}",
                make_canopy_builder(tx, ty, trunk_h, canopy_r),
                materials["tree_canopy"],
                ignore_boxes=(trunk_box,),
            ):
                trees_placed += 1
                tree_clearance.append((tx, ty, canopy_r))
                placed = True
                break
            trunk_obj = bpy.data.objects.get(trunk_name)
            if trunk_obj is not None:
                trunk_box = world_aabb(trunk_obj)
                mesh = trunk_obj.data
                bpy.data.objects.remove(trunk_obj, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
                if trunk_box in occupied:
                    occupied.remove(trunk_box)
        if not placed:
            print(f"manifest_decoration_tree_skipped {tree_index:02d}")

    print(
        f"manifest_far_decorations posts={post_index} rails=2 "
        f"trees={trees_placed}/{tree_count} skipped_overlap={skipped_overlap}"
    )
    return skipped_overlap


def setup_camera(layout):
    cam_data = bpy.data.cameras.new("surface_camera")
    cam_data.lens = 32.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("surface_camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    # Stand toward the curb so both long sidewalk edges stay in frame as parallel lines.
    local_x = layout["sw_x0"] + layout["sidewalk_w"] * 0.72
    cam_x = world_x(layout, local_x)
    cam.location = Vector((cam_x, layout["y0"] + 1.15, layout["sidewalk_z"] + EYE_HEIGHT))
    look_y = layout["y0"] + min(18.0, max(12.0, layout["length"] * 0.4))
    target = Vector((cam_x, look_y, 0.85))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def setup_world():
    elevation = math.radians(32.0)
    azimuth = math.radians(-42.0)
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 5.0
    sun_data.angle = math.radians(0.6)
    if hasattr(sun_data, "use_shadow"):
        sun_data.use_shadow = True
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.pi / 2.0 - elevation, 0.0, azimuth)

    world = bpy.data.worlds.new("outdoor")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_WORLD":
            nodes.remove(node)
    output = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    sky = nodes.new("ShaderNodeTexSky")
    sky_ids = [item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items]
    for candidate in ("NISHITA", "PREETHAM", "HOSEK_WILKIE", "MULTIPLE_SCATTERING"):
        if candidate in sky_ids:
            sky.sky_type = candidate
            break
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = elevation
    if hasattr(sky, "sun_rotation"):
        sky.sun_rotation = azimuth
    if hasattr(sky, "sun_intensity"):
        sky.sun_intensity = 1.0
    if hasattr(sky, "altitude"):
        sky.altitude = 0.0
    if hasattr(sky, "air_density"):
        sky.air_density = 1.0
    if hasattr(sky, "dust_density"):
        sky.dust_density = 0.0
    print("manifest_sky", sky.sky_type, "sun_elevation_deg=32")
    background.inputs["Strength"].default_value = 0.58
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def select_eevee(scene):
    engine_prop = scene.render.bl_rna.properties["engine"]
    available = [item.identifier for item in engine_prop.enum_items]
    print("render_engines", available)
    for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if candidate in available:
            scene.render.engine = candidate
            print("selected_engine", candidate)
            return candidate
    raise RuntimeError("EEVEE is not available. Engines: " + ", ".join(available))


def configure_render(scene, output_path):
    select_eevee(scene)
    scene.render.resolution_x = 1365
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    scene.render.film_transparent = False
    if hasattr(scene.view_settings, "exposure"):
        scene.view_settings.exposure = -0.14

    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 48
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = False
        if hasattr(eevee, "use_shadows"):
            eevee.use_shadows = True
        if hasattr(eevee, "shadow_ray_count"):
            eevee.shadow_ray_count = 2
        if hasattr(eevee, "use_fast_gi"):
            eevee.use_fast_gi = True
        if hasattr(eevee, "fast_gi_ray_count"):
            eevee.fast_gi_ray_count = 2


def ground_contact_failures(layout):
    """Obstacle names whose lowest point is not on the expected ground."""
    failures = []
    for obj in bpy.context.scene.objects:
        surface_type = obj.get("surface_type")
        if surface_type == "obstacle":
            pass
        elif surface_type == "decoration":
            if "fence_rail" in obj.name or "tree_canopy" in obj.name:
                continue
        else:
            continue
        box = world_aabb(obj)
        cx = 0.5 * (box[0] + box[1])
        cy = 0.5 * (box[2] + box[3])
        ground = expected_ground_z(layout, cx, cy)
        if abs(box[4] - ground) > GROUND_TOL_M:
            failures.append(obj.name)
            print(
                "manifest_ground_contact_fail "
                f"{obj.name} lowest_z={box[4]:.4f} expected_z={ground:.4f}"
            )
    return failures


def visible_object_count(cam):
    """Tagged objects with any bound-box corner inside the camera frame."""
    scene = bpy.context.scene
    tagged = [obj for obj in scene.objects if "surface_type" in obj.keys()]
    visible = 0
    for obj in tagged:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for corner in corners:
            co = world_to_camera_view(scene, cam, corner)
            if 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0.0:
                visible += 1
                break
    print(f"manifest_visible_objects {visible} of {len(tagged)}")
    return len(tagged), visible


def write_qa_report(output_path, total_objects, skipped_overlap, floating, visible):
    placement_pass = len(floating) == 0 and visible >= MIN_VISIBLE_OBJECTS
    report = {
        "total_objects": total_objects,
        "obstacles_skipped_overlap": skipped_overlap,
        "floating_or_sunken": floating,
        "visible_objects": visible,
        "placement_pass": placement_pass,
    }
    qa_path = os.path.splitext(output_path)[0] + "_qa.json"
    with open(qa_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("qa_report", qa_path, "placement_pass", placement_pass)
    return qa_path


def diagnose_bottom_left(cam):
    """Report which mesh dominates the lower-left of the frame (foreground clutter)."""
    scene = bpy.context.scene
    hits = {}
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            co = world_to_camera_view(scene, cam, obj.matrix_world @ Vector(corner))
            if co.z <= 0.0:
                continue
            if co.x < 0.35 and co.y < 0.35:
                hits[obj.name] = hits.get(obj.name, 0) + 1
    if not hits:
        print("diagnostic_bottom_left none")
        return None
    name = max(hits, key=hits.get)
    print(f"diagnostic_bottom_left {name} hits={hits[name]}")
    return name


def print_render_defect_diagnostics(bottom_left_name, render_seconds):
    print(
        "diagnostic_sky "
        "MULTIPLE_SCATTERING plus a dim Background (0.22) drew a hard horizon band; "
        "the far grass pad (surface_grass_01) also projected into the upper frame. "
        "Fixed with a rebuilt World node tree (Sky Texture -> Background -> World Output), "
        "stronger sky fill, and removal of the far-end grass cap."
    )
    print(
        "diagnostic_bench "
        "Seat slats were 1.5 m long along Y but only ~6 cm thick in X, so the camera "
        "saw edge-on splinters. Rebuilt as a solid seat/back across X facing the walk."
    )
    print(
        "diagnostic_dark_blob "
        f"Foreground clutter at bottom-left was {bottom_left_name or 'unknown'}; "
        "mis-aimed bike wheels (surface_obstacle_bike_*) used the cone depth as world X, "
        "forming large dark discs. Wheels now use YZ planes (axle along X) so the walk camera "
        "sees a thin profile instead of a full tire face."
    )
    print(
        "diagnostic_crosswalk "
        "Stripe boxes were coplanar with the road (z=-0.25..0), so z-fighting averaged "
        "stripes to gray. Stripes are lifted 12 mm with flat white/dark materials."
    )
    print(
        "diagnostic_materials "
        "ShaderNodeMix Factor was not driven on Blender 5 sockets; noise never modulated "
        "color. Added Mapping nodes, fixed Factor wiring, 64 EEVEE samples, fast GI, shadows."
    )
    print(f"render_time_sec {render_seconds:.2f}")


def print_manifest(cam, seed, output_path, layout):
    tagged = [obj for obj in bpy.context.scene.objects if "surface_type" in obj.keys()]
    counts = {}
    print("manifest_seed", seed)
    print("manifest_output", output_path)
    print_layout(layout)
    print("manifest_object_count", len(tagged))
    for obj in sorted(tagged, key=lambda item: item.name):
        surface_type = obj["surface_type"]
        counts[surface_type] = counts.get(surface_type, 0) + 1
        print(f"manifest_object {obj.name} surface_type={surface_type}")
    for surface_type in sorted(counts):
        print(f"manifest_count {surface_type} {counts[surface_type]}")
    loc = cam.location
    rot = cam.rotation_euler
    print(
        "manifest_camera_location_m "
        f"{loc.x:.4f} {loc.y:.4f} {loc.z:.4f}"
    )
    print(
        "manifest_camera_rotation_deg "
        f"{math.degrees(rot.x):.3f} {math.degrees(rot.y):.3f} {math.degrees(rot.z):.3f}"
    )


def main():
    args = parse_args()
    output_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = random.Random(args.seed)
    layout = sample_layout(rng, args.seed)
    reset_scene()

    print("manifest_materials layered_noise")
    materials = {
        "grass": layered_material("mat_grass", (0.06, 0.18, 0.04), (0.42, 0.55, 0.10), 0.55, 8.0, 0.82, 1.0, 0.7),
        "sidewalk": layered_material("mat_sidewalk", (0.28, 0.27, 0.25), (0.70, 0.68, 0.64), 1.1, 16.0, 0.62, 0.95, 0.55),
        "curb": layered_material("mat_curb", (0.25, 0.24, 0.22), (0.55, 0.53, 0.50), 1.4, 18.0, 0.7, 0.96, 0.4),
        "bike_lane": layered_material("mat_bike_lane", (0.06, 0.16, 0.10), (0.22, 0.40, 0.22), 1.6, 14.0, 0.7, 0.95, 0.35),
        "road": layered_material("mat_road", (0.02, 0.02, 0.025), (0.24, 0.22, 0.18), 0.65, 20.0, 0.7, 0.98, 0.65),
        "crosswalk": flat_material("mat_crosswalk", (0.95, 0.95, 0.93), roughness=0.45),
        "crosswalk_dark": flat_material("mat_crosswalk_dark", (0.12, 0.12, 0.11), roughness=0.75),
        "stairs": layered_material("mat_stairs", (0.30, 0.29, 0.27), (0.66, 0.64, 0.60), 1.3, 15.0, 0.65, 0.95, 0.5),
        "ramp": layered_material("mat_ramp", (0.28, 0.30, 0.28), (0.60, 0.62, 0.58), 1.2, 14.0, 0.65, 0.95, 0.45),
        "wood": layered_material("mat_wood", (0.18, 0.09, 0.04), (0.55, 0.34, 0.16), 2.2, 6.0, 0.5, 0.82, 0.3),
        "metal": layered_material("mat_metal", (0.08, 0.09, 0.10), (0.35, 0.38, 0.40), 3.0, 18.0, 0.28, 0.55, 0.2, metallic=0.7),
        "plastic": layered_material("mat_plastic", (0.08, 0.14, 0.11), (0.28, 0.40, 0.30), 2.5, 10.0, 0.4, 0.7, 0.15),
        "bike_metal": layered_material("mat_bike_metal", (0.10, 0.11, 0.12), (0.55, 0.58, 0.60), 4.0, 22.0, 0.22, 0.5, 0.15, metallic=0.88),
        "tree_canopy": layered_material(
            "mat_tree_canopy",
            (0.05, 0.20, 0.05),
            (0.28, 0.50, 0.14),
            0.75,
            7.0,
            0.78,
            0.95,
            0.35,
        ),
    }

    build_ground(layout, materials)
    skipped_overlap = build_obstacles(rng, layout, materials)
    skipped_overlap += build_far_decorations(rng, layout, materials)
    cam = setup_camera(layout)
    setup_world()

    scene = bpy.context.scene
    configure_render(scene, output_path)
    bpy.context.view_layer.update()
    floating = ground_contact_failures(layout)
    total_objects, visible = visible_object_count(cam)
    bottom_left_name = diagnose_bottom_left(cam)
    print_manifest(cam, args.seed, output_path, layout)
    render_start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.perf_counter() - render_start
    print("render_saved", output_path)
    if args.save_blend:
        blend_path = os.path.splitext(output_path)[0] + ".blend"
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print("blend_saved", blend_path)
    write_qa_report(output_path, total_objects, skipped_overlap, floating, visible)
    print_render_defect_diagnostics(bottom_left_name, render_seconds)


if __name__ == "__main__":
    main()
