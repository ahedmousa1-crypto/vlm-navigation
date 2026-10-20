"""Prop builders in a side-local frame: +X along the street, +Y away from the road, Z=0 on the ground.

Each ``make_*`` returns a Prop: bmesh builder, material keys (index order),
surface_type, category, and half extents (hx, hy) used for spacing.
Legacy props (bench, bollard, trash can, bike, fence, tree) keep the
dimensions from legacy/blender/gen_scene.py.
"""

import math
from collections import namedtuple

from .meshes import bm_box, bm_cylinder, bm_icosphere, bm_wheel

Prop = namedtuple("Prop", "builder materials surface_type category hx hy")


def make_bench():
    def builder(bm):
        # Seat along the street, backrest on the far side so it faces the road.
        bm_box(bm, -0.62, 0.62, -0.20, 0.20, 0.45, 0.50)
        bm_box(bm, -0.58, 0.58, 0.16, 0.22, 0.50, 0.88)
        for lx, ly in ((-0.52, -0.14), (0.52, -0.14), (-0.52, 0.14), (0.52, 0.14)):
            bm_box(bm, lx - 0.04, lx + 0.04, ly - 0.04, ly + 0.04, 0.0, 0.45)

    return Prop(builder, ["wood"], "obstacle", "bench", 0.62, 0.22)


def make_bollard():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.45), 0.08, 0.9)

    return Prop(builder, ["metal"], "obstacle", "bollard", 0.12, 0.12)


def make_trash_can():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.42), 0.24, 0.84, segments=12)
        bm_cylinder(bm, (0.0, 0.0, 0.88), 0.26, 0.06, segments=12)

    return Prop(builder, ["plastic"], "obstacle", "trash_can", 0.28, 0.28)


def make_bike():
    def builder(bm):
        # Parked across the sidewalk: wheels at +/-Y, axle along X.
        wheel_r = 0.30
        hub_z = wheel_r
        for wy in (-0.52, 0.52):
            bm_wheel(bm, 0.0, wy, hub_z, wheel_r, 0.05, axle="x")
        bm_box(bm, -0.03, 0.03, -0.55, 0.55, 0.28, 0.36)
        bm_box(bm, -0.03, 0.03, -0.30, 0.08, 0.50, 0.58)
        bm_box(bm, -0.03, 0.03, -0.30, -0.22, 0.34, 0.72)
        bm_box(bm, -0.03, 0.03, 0.40, 0.48, 0.34, 0.56)

    return Prop(builder, ["bike_metal"], "obstacle", "bike", 0.08, 0.58)


def make_tree_trunk(height, radius=0.14):
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, height * 0.5), radius, height, segments=10)

    return Prop(builder, ["wood"], "obstacle", "tree", radius, radius)


def make_tree_canopy(trunk_h, canopy_r):
    def builder(bm):
        bm_icosphere(bm, (0.0, 0.0, trunk_h + canopy_r * 0.55), canopy_r, subdivisions=2)

    return Prop(builder, ["tree_canopy"], "overhead", "tree", canopy_r, canopy_r)


def make_conifer(height, radius):
    """Short trunk with three stacked cones; low branches, so the whole thing blocks walking."""

    def builder(bm):
        trunk_h = height * 0.18
        bm_cylinder(bm, (0.0, 0.0, trunk_h * 0.5), 0.12, trunk_h, segments=10)
        tiers = ((0.15, 1.0, 0.45), (0.42, 0.78, 0.38), (0.66, 0.55, 0.34))
        for base, scale, depth in tiers:
            h = height * depth
            bm_cylinder(bm, (0.0, 0.0, height * base + h * 0.5), radius * scale, h, segments=12,
                        radius_top=0.02, material_index=1)

    return Prop(builder, ["wood", "shrub"], "obstacle", "tree", radius, radius)


def make_tall_canopy(trunk_h, canopy_r):
    """Oval canopy for a tall, narrow tree (trunk from make_tree_trunk)."""

    def builder(bm):
        bm_icosphere(bm, (0.0, 0.0, trunk_h + canopy_r * 1.2), canopy_r, subdivisions=2, scale_z=1.8)

    return Prop(builder, ["tree_canopy"], "overhead", "tree", canopy_r, canopy_r)


def make_bush(radius, rng):
    """Two to four overlapping blobs, flattened, sitting on the ground."""

    def builder(bm):
        count = rng.randint(2, 4)
        for index in range(count):
            angle = index * 6.283 / count + rng.uniform(-0.4, 0.4)
            offset = radius * 0.35
            r = radius * rng.uniform(0.55, 0.8)
            # Center at r * 0.85 so the flattened blob's bottom touches the ground exactly.
            bm_icosphere(bm, (offset * math.cos(angle), offset * math.sin(angle), r * 0.85), r, subdivisions=2,
                         scale_z=0.85)

    return Prop(builder, ["shrub"], "obstacle", "bush", radius, radius)


def make_fence(length, height=1.05, spacing=2.6):
    """Posts every ``spacing`` metres along X, centered on the origin, with two rails (legacy fence)."""
    half = length / 2

    def builder(bm):
        count = max(2, int(length / spacing) + 1)
        step = length / (count - 1)
        for index in range(count):
            bm_cylinder(bm, (-half + index * step, 0.0, height * 0.5), 0.05, height, segments=10)
        for rail_z in (0.42, 0.82):
            bm_box(bm, -half, half, -0.025, 0.025, rail_z, rail_z + 0.05, material_index=1)

    return Prop(builder, ["wood", "metal"], "obstacle", "fence", length / 2, 0.05)


def make_lamp_pole(height=5.0):
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.15), 0.14, 0.3, segments=12)
        bm_cylinder(bm, (0.0, 0.0, height * 0.5), 0.07, height, segments=12)

    return Prop(builder, ["metal"], "obstacle", "street_lamp", 0.14, 0.14)


def make_lamp_head(height=5.0):
    def builder(bm):
        # Arm reaches over the road (-Y); kept separate so it doesn't block walking.
        bm_box(bm, -0.04, 0.04, -1.1, 0.0, height - 0.08, height)
        bm_box(bm, -0.16, 0.16, -1.35, -0.85, height - 0.2, height - 0.05)
        bm_box(bm, -0.13, 0.13, -1.32, -0.88, height - 0.24, height - 0.2, material_index=1)

    return Prop(builder, ["metal", "lamp_glass"], "overhead", "street_lamp", 0.16, 0.7)


def make_planter(size=1.1):
    half = size / 2

    def builder(bm):
        bm_box(bm, -half, half, -half, half, 0.0, 0.5)
        bm_icosphere(bm, (0.0, 0.0, 0.75), half * 0.85, subdivisions=2, scale_z=0.7, material_index=1)

    return Prop(builder, ["curb", "shrub"], "obstacle", "planter", half, half)


def make_car(paint):
    def builder(bm):
        bm_box(bm, -2.2, 2.2, -0.88, 0.88, 0.32, 0.95)
        bm_box(bm, -1.1, 1.0, -0.80, 0.80, 0.95, 1.45)
        # Windows: thin glass panels just outside the cabin faces.
        bm_box(bm, -1.0, 0.9, -0.82, -0.80, 1.0, 1.4, material_index=1)
        bm_box(bm, -1.0, 0.9, 0.80, 0.82, 1.0, 1.4, material_index=1)
        bm_box(bm, 1.0, 1.02, -0.72, 0.72, 1.0, 1.4, material_index=1)
        bm_box(bm, -1.12, -1.10, -0.72, 0.72, 1.0, 1.4, material_index=1)
        for wx in (-1.4, 1.4):
            for wy in (-0.78, 0.78):
                bm_wheel(bm, wx, wy, 0.32, 0.32, 0.22, axle="y", material_index=2)

    return Prop(builder, [paint, "car_glass", "tire"], "obstacle", "car", 2.2, 0.9)


def make_cone():
    def builder(bm):
        bm_box(bm, -0.19, 0.19, -0.19, 0.19, 0.0, 0.04, material_index=1)
        bm_cylinder(bm, (0.0, 0.0, 0.39), 0.16, 0.7, segments=16, radius_top=0.03)

    return Prop(builder, ["cone_orange", "tire"], "obstacle", "traffic_cone", 0.19, 0.19)


def make_barrier(length=1.6):
    """Two A-frame legs and a red/white striped plank, spanning X from -length/2 to length/2."""
    half = length / 2

    def builder(bm):
        for lx in (-half + 0.08, half - 0.08):
            bm_box(bm, lx - 0.04, lx + 0.04, -0.25, 0.25, 0.0, 0.05, material_index=0)
            bm_box(bm, lx - 0.03, lx + 0.03, -0.03, 0.03, 0.0, 1.0, material_index=0)
        stripes = 6
        width = length / stripes
        for index in range(stripes):
            x0 = -half + index * width
            bm_box(bm, x0, x0 + width, -0.03, 0.03, 0.75, 0.97, material_index=1 + index % 2)

    return Prop(builder, ["metal", "barrier_red", "barrier_white"], "obstacle", "barrier", half, 0.25)


def make_hoarding(length, height=2.4):
    def builder(bm):
        bm_box(bm, -length / 2, length / 2, -0.04, 0.04, 0.0, height)

    return Prop(builder, ["hoarding"], "obstacle", "construction_fence", length / 2, 0.04)


def make_crate(sx, sy, sz):
    def builder(bm):
        bm_box(bm, -sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz)

    return Prop(builder, ["wood"], "obstacle", "construction_material", sx / 2, sy / 2)
