"""Phase 2 props (neighborhoods, parks, skate park, scattered street objects).

Same conventions as props.py: built in a local frame with +X along the street,
+Y away from the road, Z=0 on the ground; each returns a Prop.
"""

from .meshes import bm_box, bm_cylinder, bm_icosphere, bm_wheel
from .props import Prop


def make_mailbox():
    def builder(bm):
        bm_box(bm, -0.04, 0.04, -0.04, 0.04, 0.0, 1.0)
        bm_box(bm, -0.25, 0.25, -0.12, 0.12, 1.0, 1.25, material_index=1)

    return Prop(builder, ["metal", "paint_blue"], "obstacle", "mailbox", 0.25, 0.12)


def make_wheelie_bin():
    def builder(bm):
        bm_box(bm, -0.30, 0.30, -0.36, 0.36, 0.05, 1.0)
        bm_box(bm, -0.32, 0.32, -0.40, 0.40, 1.0, 1.06)
        for wx in (-0.25, 0.25):
            bm_wheel(bm, wx, 0.34, 0.05, 0.05, 0.05, axle="x", material_index=1)

    return Prop(builder, ["paint_green", "tire"], "obstacle", "wheelie_bin", 0.32, 0.40)


def make_hedge(length, height=1.1, depth=0.7):
    def builder(bm):
        bm_box(bm, -length / 2, length / 2, -depth / 2, depth / 2, 0.0, height)

    return Prop(builder, ["shrub"], "obstacle", "hedge", length / 2, depth / 2)


def make_fire_hydrant():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.05), 0.16, 0.1, segments=12)
        bm_cylinder(bm, (0.0, 0.0, 0.40), 0.11, 0.6, segments=12)
        bm_cylinder(bm, (0.0, 0.0, 0.74), 0.13, 0.08, segments=12, radius_top=0.05)
        bm_box(bm, -0.20, 0.20, -0.05, 0.05, 0.45, 0.55)

    return Prop(builder, ["paint_red"], "obstacle", "fire_hydrant", 0.2, 0.16)


def make_signpost():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 1.3), 0.04, 2.6, segments=8)
        bm_box(bm, -0.45, 0.45, -0.02, 0.02, 2.1, 2.5, material_index=1)

    return Prop(builder, ["metal", "paint_green"], "obstacle", "signpost", 0.45, 0.05)


def make_bike_rack(count=4):
    def builder(bm):
        for k in range(count):
            x = (k - (count - 1) / 2) * 0.7
            bm_box(bm, x - 0.03, x + 0.03, -0.35, -0.29, 0.0, 0.8)
            bm_box(bm, x - 0.03, x + 0.03, 0.29, 0.35, 0.0, 0.8)
            bm_box(bm, x - 0.03, x + 0.03, -0.35, 0.35, 0.74, 0.8)

    return Prop(builder, ["metal"], "obstacle", "bike_rack", (count - 1) * 0.35 + 0.05, 0.35)


def make_bus_stop():
    """Shelter along the street (+X), open toward the road (-Y), with a bench and a sign."""

    def builder(bm):
        for x in (-1.6, 1.6):
            bm_box(bm, x - 0.05, x + 0.05, 0.45, 0.55, 0.0, 2.5)
        bm_box(bm, -1.65, 1.65, 0.50, 0.55, 0.3, 2.4, material_index=1)
        bm_box(bm, -1.8, 1.8, -0.6, 0.6, 2.5, 2.6)
        bm_box(bm, -1.2, 1.2, 0.15, 0.45, 0.42, 0.48, material_index=2)
        for x in (-1.0, 1.0):
            bm_box(bm, x - 0.04, x + 0.04, 0.25, 0.35, 0.0, 0.42, material_index=2)
        bm_cylinder(bm, (2.1, -0.4, 1.3), 0.04, 2.6, segments=8)
        bm_box(bm, 1.95, 2.25, -0.42, -0.38, 2.2, 2.6, material_index=3)

    return Prop(builder, ["metal", "glass", "wood", "paint_yellow"], "obstacle", "bus_stop", 2.25, 0.6)


def make_kiosk(paint):
    def builder(bm):
        bm_box(bm, -1.2, 1.2, -1.0, 1.0, 0.0, 2.4)
        bm_box(bm, -1.0, 1.0, -1.02, -1.0, 1.0, 2.0, material_index=1)
        bm_box(bm, -1.5, 1.5, -1.4, 1.3, 2.4, 2.6, material_index=2)

    return Prop(builder, [paint, "glass", "roof"], "obstacle", "kiosk", 1.5, 1.4)


def make_food_truck(paint):
    def builder(bm):
        bm_box(bm, -2.8, 2.8, -1.1, 1.1, 0.45, 2.9)
        bm_box(bm, -1.6, 1.2, -1.12, -1.10, 1.4, 2.3, material_index=1)
        bm_box(bm, -1.7, 1.3, -1.7, -1.1, 2.3, 2.4, material_index=3)
        for wx in (-1.9, 1.9):
            for wy in (-1.0, 1.0):
                bm_wheel(bm, wx, wy, 0.42, 0.42, 0.28, axle="y", material_index=2)

    return Prop(builder, [paint, "glass", "tire", "paint_white"], "obstacle", "food_truck", 2.8, 1.7)


def make_vending_machine():
    def builder(bm):
        bm_box(bm, -0.45, 0.45, -0.40, 0.40, 0.0, 1.9)
        bm_box(bm, -0.35, 0.20, -0.42, -0.40, 0.8, 1.7, material_index=1)

    return Prop(builder, ["paint_red", "glass"], "obstacle", "vending_machine", 0.45, 0.42)


def make_recycling_bins():
    def builder(bm):
        for k in range(3):
            x = (k - 1) * 0.7
            bm_box(bm, x - 0.3, x + 0.3, -0.3, 0.3, 0.0, 1.0, material_index=k)

    return Prop(builder, ["paint_blue", "paint_green", "paint_yellow"], "obstacle", "recycling_bin", 1.0, 0.3)


def make_sculpture(rng):
    def builder(bm):
        bm_box(bm, -0.6, 0.6, -0.6, 0.6, 0.0, 0.5, material_index=1)
        z = 0.5
        for _ in range(rng.randint(2, 4)):
            r = rng.uniform(0.25, 0.5)
            bm_icosphere(bm, (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), z + r), r, subdivisions=1)
            z += 1.6 * r

    return Prop(builder, ["bronze", "stone"], "obstacle", "sculpture", 0.6, 0.6)


def make_statue():
    def builder(bm):
        bm_box(bm, -0.7, 0.7, -0.7, 0.7, 0.0, 1.2)
        bm_box(bm, -0.25, 0.25, -0.18, 0.18, 1.2, 2.3, material_index=1)
        bm_icosphere(bm, (0.0, 0.0, 2.5), 0.2, subdivisions=2, material_index=1)
        bm_box(bm, -0.45, 0.45, -0.08, 0.08, 1.9, 2.05, material_index=1)

    return Prop(builder, ["stone", "bronze"], "obstacle", "statue", 0.7, 0.7)


def make_fountain():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.25), 2.1, 0.5, segments=28)
        bm_cylinder(bm, (0.0, 0.0, 0.47), 1.9, 0.06, segments=28, material_index=1)
        bm_cylinder(bm, (0.0, 0.0, 1.0), 0.22, 1.1, segments=12)
        bm_cylinder(bm, (0.0, 0.0, 1.55), 0.8, 0.12, segments=20, radius_top=0.9)
        bm_cylinder(bm, (0.0, 0.0, 1.62), 0.7, 0.03, segments=20, material_index=1)

    return Prop(builder, ["stone", "water"], "obstacle", "fountain", 2.1, 2.1)


def make_swings():
    def builder(bm):
        for x in (-1.8, 1.8):
            for y in (-0.8, 0.8):
                bm_box(bm, x - 0.05, x + 0.05, y - 0.05, y + 0.05, 0.0, 2.3)
        bm_box(bm, -1.85, 1.85, -0.06, 0.06, 2.25, 2.37)
        for x in (-0.8, 0.8):
            bm_box(bm, x - 0.01, x + 0.01, -0.01, 0.01, 0.55, 2.25, material_index=1)
            bm_box(bm, x - 0.25, x + 0.25, -0.12, 0.12, 0.5, 0.55, material_index=2)

    return Prop(builder, ["paint_blue", "metal", "paint_red"], "obstacle", "swings", 1.85, 0.85)


def make_slide():
    def builder(bm):
        bm_box(bm, -1.4, -0.6, -0.4, 0.4, 0.0, 1.5, material_index=1)
        bm_box(bm, -1.5, -0.5, -0.5, 0.5, 1.5, 1.6)
        chute = bm_box(bm, -0.5, 1.8, -0.3, 0.3, 0.0, 0.08)
        for vert in chute:  # raise the tower end so the chute slopes down to +X
            if vert.co.x < 0.0:
                vert.co.z += 1.45

    return Prop(builder, ["paint_yellow", "paint_red"], "obstacle", "slide", 1.8, 0.5)


def make_picnic_table():
    def builder(bm):
        bm_box(bm, -0.9, 0.9, -0.4, 0.4, 0.72, 0.78)
        for y in (-0.7, 0.7):
            bm_box(bm, -0.9, 0.9, y - 0.15, y + 0.15, 0.42, 0.47)
        for x in (-0.7, 0.7):
            bm_box(bm, x - 0.05, x + 0.05, -0.75, 0.75, 0.0, 0.42)
            bm_box(bm, x - 0.05, x + 0.05, -0.05, 0.05, 0.42, 0.72)

    return Prop(builder, ["wood"], "obstacle", "picnic_table", 0.9, 0.85)


def make_basketball_hoop():
    def builder(bm):
        bm_box(bm, -0.08, 0.08, -0.08, 0.08, 0.0, 3.0)
        bm_box(bm, -0.08, 0.08, -0.08, 0.9, 2.9, 3.0)
        bm_box(bm, -0.9, 0.9, 0.9, 0.95, 2.75, 3.8, material_index=1)
        bm_cylinder(bm, (0.0, 0.7, 3.05), 0.23, 0.03, segments=16, material_index=2)

    return Prop(builder, ["metal", "paint_white", "paint_orange"], "obstacle", "basketball_hoop", 0.9, 0.95)


def make_quarter_pipe(width=3.0, depth=2.4, height=1.2):
    """Curved transition approximated by a stepped profile, deck at the back (+Y)."""

    def builder(bm):
        steps = 6
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            y0, y1 = -depth / 2 + t0 * depth * 0.8, -depth / 2 + t1 * depth * 0.8
            bm_box(bm, -width / 2, width / 2, y0, y1, 0.0, max(0.05, height * t1 * t1))
        bm_box(bm, -width / 2, width / 2, -depth / 2 + depth * 0.8, depth / 2, 0.0, height)

    return Prop(builder, ["concrete"], "obstacle", "skate_ramp", width / 2, depth / 2)


def make_funbox(length=3.0, width=2.0, height=0.6):
    def builder(bm):
        bm_box(bm, -length / 4, length / 4, -width / 2, width / 2, 0.0, height)
        for sign in (-1.0, 1.0):
            a, b = sorted((sign * length / 4, sign * length / 2))
            wedge = bm_box(bm, a, b, -width / 2, width / 2, 0.0, height)
            for vert in wedge:  # slope down toward the outer end
                if abs(abs(vert.co.x) - length / 2) < 1e-6 and vert.co.z > 0.01:
                    vert.co.z = 0.02

    return Prop(builder, ["concrete"], "obstacle", "skate_ramp", length / 2, width / 2)


def make_rail(length=3.0, height=0.4):
    def builder(bm):
        for x in (-length / 2 + 0.2, length / 2 - 0.2):
            bm_box(bm, x - 0.03, x + 0.03, -0.03, 0.03, 0.0, height)
        bm_box(bm, -length / 2, length / 2, -0.03, 0.03, height - 0.05, height + 0.01)

    return Prop(builder, ["metal"], "obstacle", "rail", length / 2, 0.05)


def make_ledge(length=3.0, depth=0.5, height=0.45):
    def builder(bm):
        bm_box(bm, -length / 2, length / 2, -depth / 2, depth / 2, 0.0, height)

    return Prop(builder, ["concrete"], "obstacle", "ledge", length / 2, depth / 2)
