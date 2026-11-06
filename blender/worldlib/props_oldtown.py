"""World 2 (old town) props. Same conventions as props.py: local frame, +X along the street,
Z=0 on the ground; each returns a Prop. Canopies and umbrellas are separate "overhead"
props so the ground under them stays walkable in the navmap.
"""

import math

from .meshes import bm_box, bm_cylinder, bm_icosphere, bm_obox, bm_wheel
from .props import Prop

CHAIR_R = 0.62  # chair distance from the table center


def make_cafe_set(chairs=4):
    """Round table with chairs around it (one object, so tasks can name "the café table")."""

    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.73), 0.38, 0.04, segments=18)
        bm_cylinder(bm, (0.0, 0.0, 0.37), 0.04, 0.72, segments=8, material_index=1)
        bm_cylinder(bm, (0.0, 0.0, 0.015), 0.22, 0.03, segments=12, material_index=1)
        for index in range(chairs):
            a = 2 * math.pi * index / chairs + math.pi / 4
            cx, cy = math.cos(a) * CHAIR_R, math.sin(a) * CHAIR_R
            bm_box(bm, cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2, 0.43, 0.47, material_index=1)
            for lx, ly in ((-0.16, -0.16), (0.16, -0.16), (-0.16, 0.16), (0.16, 0.16)):
                bm_box(bm, cx + lx - 0.02, cx + lx + 0.02, cy + ly - 0.02, cy + ly + 0.02, 0.0, 0.43, material_index=1)
            # Thin backrest on the side away from the table.
            bx, by = math.cos(a) * (CHAIR_R + 0.18), math.sin(a) * (CHAIR_R + 0.18)
            bm_obox(bm, (bx, by), (-math.sin(a), math.cos(a)), -0.19, 0.19, -0.02, 0.02, 0.47, 0.88, 1)

    reach = CHAIR_R + 0.25
    return Prop(builder, ["wood", "metal"], "obstacle", "cafe_table", reach, reach)


def chair_spots(chairs=4):
    """(local x, local y, yaw facing the table) for each chair of make_cafe_set."""
    out = []
    for index in range(chairs):
        a = 2 * math.pi * index / chairs + math.pi / 4
        out.append((math.cos(a) * CHAIR_R, math.sin(a) * CHAIR_R, a + math.pi))
    return out


def make_umbrella(fabric):
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 1.4), 0.025, 1.2, segments=6, material_index=1)
        bm_cylinder(bm, (0.0, 0.0, 2.15), 1.25, 0.35, segments=8, radius_top=0.05)

    return Prop(builder, [fabric, "metal"], "overhead", "cafe_table", 1.25, 1.25)


def make_market_stall(length=2.6, depth=1.4):
    """Counter with produce crates; the canopy is make_stall_canopy."""
    hl, hd = length / 2, depth / 2

    def builder(bm):
        bm_box(bm, -hl, hl, -hd, hd, 0.0, 0.9)
        for k in range(3):
            x = -hl + (k + 0.5) * length / 3
            bm_box(bm, x - 0.35, x + 0.35, -hd + 0.05, -hd + 0.6, 0.9, 1.05, material_index=1)
            bm_icosphere(bm, (x, -hd + 0.33, 1.08), 0.22, subdivisions=1, scale_z=0.45, material_index=2 + k % 2)
        for px, py in ((-hl, -hd), (hl, -hd), (-hl, hd), (hl, hd)):
            bm_box(bm, px - 0.03, px + 0.03, py - 0.03, py + 0.03, 0.9, 2.3, material_index=1)

    return Prop(builder, ["wood", "wood", "paint_orange", "paint_green"], "obstacle", "market_stall", hl, hd)


def make_stall_canopy(fabric, length=2.6, depth=1.4):
    hl, hd = length / 2 + 0.15, depth / 2 + 0.25

    def builder(bm):
        bm_box(bm, -hl, hl, -hd, hd, 2.3, 2.38)
        bm_box(bm, -hl, hl, -hd - 0.02, -hd, 2.05, 2.38)

    return Prop(builder, [fabric], "overhead", "market_stall", hl, hd)


def make_flower_pot(size=0.55):
    half = size / 2

    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, size * 0.45), half, size * 0.9, segments=12, radius_top=half * 1.15)
        bm_icosphere(bm, (0.0, 0.0, size * 1.05), half * 1.05, subdivisions=2, scale_z=0.8, material_index=1)
        for a in range(5):
            ang = a * 2 * math.pi / 5
            bm_icosphere(bm, (math.cos(ang) * half * 0.6, math.sin(ang) * half * 0.6, size * 1.3), 0.06,
                         subdivisions=1, material_index=2)

    return Prop(builder, ["terracotta", "shrub", "paint_red"], "obstacle", "flower_pot", half * 1.15, half * 1.15)


def make_scooter(paint):
    """Parked motor scooter along X."""

    def builder(bm):
        for wx in (-0.62, 0.62):
            bm_wheel(bm, wx, 0.0, 0.2, 0.2, 0.09, axle="y", material_index=2)
        bm_box(bm, -0.75, 0.25, -0.17, 0.17, 0.25, 0.62)           # body
        bm_box(bm, -0.72, -0.05, -0.15, 0.15, 0.62, 0.74, material_index=1)  # seat
        bm_box(bm, 0.25, 0.45, -0.16, 0.16, 0.2, 0.36)             # footboard nose
        bm_box(bm, 0.45, 0.55, -0.12, 0.12, 0.25, 1.0)             # front shield
        bm_box(bm, 0.48, 0.56, -0.32, 0.32, 0.98, 1.03, material_index=1)  # handlebar

    return Prop(builder, [paint, "paint_black", "tire"], "obstacle", "scooter", 0.85, 0.33)


def make_lantern_post(height=4.2):
    """Old-style post; the arm and lantern are make_lantern_head (overhead), like the street lamps."""

    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.25), 0.16, 0.5, segments=10)
        bm_cylinder(bm, (0.0, 0.0, height * 0.5), 0.06, height, segments=10)
        bm_icosphere(bm, (0.0, 0.0, height + 0.04), 0.09, subdivisions=1)

    return Prop(builder, ["paint_black"], "obstacle", "street_lamp", 0.16, 0.16)


def make_lantern_head(height=4.2):
    """Arm toward local -Y with a lantern 1.1 m out, where lighting.py puts the lamp light."""

    def builder(bm):
        bm_box(bm, -0.03, 0.03, -1.1, 0.0, height - 0.12, height - 0.06)
        # Lantern ends just above the lamp light (0.3 m under the top), so the light is not boxed in.
        bm_box(bm, -0.17, 0.17, -1.27, -0.93, height - 0.27, height - 0.12)
        bm_box(bm, -0.14, 0.14, -1.24, -0.96, height - 0.28, height - 0.13, material_index=1)
        bm_cylinder(bm, (0.0, -1.1, height - 0.06), 0.22, 0.12, segments=4, radius_top=0.02)

    return Prop(builder, ["paint_black", "lamp_glass"], "overhead", "street_lamp", 0.2, 0.7)


def make_scooter_paints():
    return ("paint_red", "paint_blue", "paint_white", "paint_yellow", "paint_green", "paint_black")
