"""Procedural low-poly people (our own geometry, so no asset licence is involved).

A person is one mesh: shoes, legs, torso, arms, neck, head and hair, built
facing +X with the feet at Z=0. Poses are standing, walking (stride and arm
swing) and sitting (on a chair seat at 0.46 m). Clothes, skin and hair come
from the materials in materials.py; the top's color names the person in
task wording ("the person in the red jacket").
"""

import math

from .meshes import bm_box, bm_cylinder, bm_icosphere
from .props import Prop

TOP_COLORS = ("red", "blue", "green", "yellow", "white", "black", "gray", "orange", "purple", "pink")
BOTTOMS = ("cloth_jeans", "cloth_black", "cloth_beige", "cloth_gray")
SKINS = ("skin_1", "skin_2", "skin_3", "skin_4")
HAIRS = ("hair_black", "hair_brown", "hair_blond", "hair_gray")
GARMENTS = ("shirt", "jacket", "coat", "sweater")


def random_look(rng, child=False):
    """Colors, garment and height for one person."""
    return {
        "top": rng.choice(TOP_COLORS),
        "garment": rng.choice(GARMENTS),
        "bottom": rng.choice(BOTTOMS),
        "skin": rng.choice(SKINS),
        "hair": rng.choice(HAIRS[:3]) if child else rng.choice(HAIRS),
        "height": rng.uniform(1.15, 1.35) if child else rng.uniform(1.58, 1.90),
        "long_hair": rng.random() < 0.4,
        "dress": not child and rng.random() < 0.15,
    }


def _limb(bm, x0, z0, x1, z1, y, radius, material_index, radius_end=None):
    """Rounded limb from (x0, z0) to (x1, z1) at lateral offset y, optionally tapering to ``radius_end``."""
    length = math.hypot(x1 - x0, z1 - z0)
    verts = bm_cylinder(bm, (0.0, y, length / 2), radius, length, segments=8,
                        radius_top=radius if radius_end is None else radius_end, material_index=material_index)
    angle = math.atan2(x1 - x0, z1 - z0)  # lean from vertical toward +X
    c, s = math.cos(angle), math.sin(angle)
    for v in verts:
        lx, lz = v.co.x, v.co.z
        v.co.x = x0 + lx * c + lz * s
        v.co.z = z0 - lx * s + lz * c
    bm_icosphere(bm, (x0, y, z0), radius, subdivisions=1, material_index=material_index)  # rounded joint


def _torso(bm, x, z0, z1, half_w, half_d, material_index, top_scale=1.0):
    """Elliptic torso from z0 to z1, ``top_scale`` wider at the shoulders."""
    verts = bm_cylinder(bm, (0.0, 0.0, (z0 + z1) / 2), half_w, z1 - z0, segments=12, radius_top=half_w * top_scale,
                        material_index=material_index)
    for v in verts:
        v.co.x = x + v.co.x * half_d / half_w


def make_person(look, pose="standing", stride=0.0):
    """``pose``: "standing", "walking" (``stride`` in -1..1 picks the leg phase) or "sitting"."""
    h = look["height"]
    k = h / 1.75
    hip_z = 0.92 * k
    shoulder_z = 1.43 * k
    head_r = 0.11 * k
    leg_r, arm_r = 0.068 * k, 0.055 * k
    hip_w, shoulder_w = 0.10 * k, 0.20 * k
    # material slots: 0 skin, 1 top, 2 bottom, 3 hair, 4 shoes
    materials = [look["skin"], f"cloth_{look['top']}", look["bottom"], look["hair"], "cloth_black"]

    def builder(bm):
        knee_z = 0.50 * k
        if pose == "sitting":
            hz = 0.54
            for side in (-1, 1):
                y = side * hip_w
                _limb(bm, 0.0, hz, 0.42 * k, hz, y, leg_r, 2, leg_r * 0.85)                    # thigh, forward
                _limb(bm, 0.42 * k, hz, 0.45 * k, 0.08, y, leg_r * 0.85, 2, leg_r * 0.7)       # shin, down
                bm_box(bm, 0.40 * k, 0.62 * k, y - 0.05, y + 0.05, 0.0, 0.08, material_index=4)
            base, top = hz, hz + (shoulder_z - hip_z)
            lean = -0.03
        else:
            swing = 0.30 * stride if pose == "walking" else 0.0
            for side in (-1, 1):
                y = side * hip_w
                foot_x = swing * side
                knee_x = foot_x * 0.55 + (0.04 if pose == "walking" and foot_x * side < 0 else 0.0)
                _limb(bm, 0.0, hip_z, knee_x, knee_z, y, leg_r, 2, leg_r * 0.85)
                _limb(bm, knee_x, knee_z, foot_x, 0.08, y, leg_r * 0.85, 2, leg_r * 0.65)
                bm_box(bm, foot_x - 0.07, foot_x + 0.17, y - 0.05, y + 0.05, 0.0, 0.08, material_index=4)
            base, top = hip_z, shoulder_z
            lean = 0.03 if pose == "walking" else 0.0
        # Hips in the trousers' color, torso in the top's color; a dress adds a skirt.
        _torso(bm, lean * 0.5, base - 0.06, base + 0.10, 0.17 * k, 0.11 * k, 2)
        _torso(bm, lean, base + 0.08, top, 0.16 * k, 0.11 * k, 1, top_scale=1.25)
        if look.get("dress") and pose != "sitting":
            bm_cylinder(bm, (lean, 0.0, base - 0.16), 0.24 * k, 0.38, segments=12, radius_top=0.16 * k,
                        material_index=1)
        # Shoulders, then arms in two parts (sleeve, then forearm) swinging opposite to the legs.
        for side in (-1, 1):
            y = side * shoulder_w
            bm_icosphere(bm, (lean, y, top - 0.05), arm_r * 1.5, subdivisions=1, material_index=1)
            elbow_z = top - 0.32 * k
            if pose == "sitting":
                ex, hx, hz2 = 0.10 * k, 0.32 * k, top - 0.42 * k
            elif pose == "walking":
                ex = lean - 0.14 * stride * side * k
                hx, hz2 = ex - 0.10 * stride * side * k, top - 0.62 * k
            else:
                ex, hx, hz2 = 0.0, 0.03 * k, top - 0.62 * k
            _limb(bm, lean, top - 0.05, ex, elbow_z, y, arm_r * 1.1, 1, arm_r)
            sleeve = 1 if look["garment"] in ("jacket", "coat", "sweater") else 0
            _limb(bm, ex, elbow_z, hx, hz2, y, arm_r, sleeve, arm_r * 0.8)
            bm_icosphere(bm, (hx, y, hz2 - 0.03), arm_r * 1.05, subdivisions=1, material_index=0)
        # Neck, head (with a nose so the facing reads) and hair.
        bm_cylinder(bm, (lean, 0.0, top + 0.03), 0.045 * k, 0.09, segments=8, material_index=0)
        head_z = top + 0.07 + head_r
        bm_icosphere(bm, (lean + 0.01, 0.0, head_z), head_r, subdivisions=2, scale_z=1.15, material_index=0)
        bm_box(bm, lean + head_r * 0.9, lean + head_r * 1.15, -0.012, 0.012, head_z - 0.03, head_z + 0.01,
               material_index=0)
        bm_icosphere(bm, (lean - 0.015, 0.0, head_z + head_r * 0.3), head_r * 1.06, subdivisions=2, scale_z=0.8,
                     material_index=3)
        if look.get("long_hair"):
            bm_cylinder(bm, (lean - 0.06 * k, 0.0, head_z - 0.06 * k), head_r * 0.95, 0.22 * k, segments=8,
                        radius_top=head_r * 1.05, material_index=3)

    reach = 0.30 * k if pose == "sitting" else 0.22
    return Prop(builder, materials, "obstacle", "person", reach + 0.1, 0.30 * k)
