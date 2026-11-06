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


def _limb(bm, x0, z0, x1, z1, y, radius, material_index):
    """Box-section limb from (x0, z0) to (x1, z1) at lateral offset y."""
    length = math.hypot(x1 - x0, z1 - z0)
    verts = bm_box(bm, -radius, radius, y - radius, y + radius, 0.0, length, material_index=material_index)
    angle = math.atan2(x1 - x0, z1 - z0)  # lean from vertical toward +X
    c, s = math.cos(angle), math.sin(angle)
    for v in verts:
        lx, lz = v.co.x, v.co.z
        v.co.x = x0 + lx * c + lz * s
        v.co.z = z0 - lx * s + lz * c


def make_person(look, pose="standing", stride=0.0):
    """``pose``: "standing", "walking" (``stride`` in -1..1 picks the leg phase) or "sitting"."""
    h = look["height"]
    k = h / 1.75
    hip_z = 0.92 * k
    shoulder_z = 1.43 * k
    head_r = 0.11 * k
    leg_r, arm_r = 0.065 * k, 0.045 * k
    hip_w, shoulder_w = 0.10 * k, 0.20 * k
    # material slots: 0 skin, 1 top, 2 bottom, 3 hair, 4 shoes
    materials = [look["skin"], f"cloth_{look['top']}", look["bottom"], look["hair"], "cloth_black"]

    def builder(bm):
        if pose == "sitting":
            seat_z = 0.46
            hz = seat_z + 0.08
            for side in (-1, 1):
                y = side * hip_w
                _limb(bm, 0.0, hz, 0.42 * k, hz, y, leg_r, 2)              # thigh, forward
                _limb(bm, 0.42 * k, hz, 0.45 * k, 0.06, y, leg_r * 0.9, 2)  # shin, down
                bm_box(bm, 0.40 * k, 0.60 * k, y - 0.05, y + 0.05, 0.0, 0.07, material_index=4)
            base, top = hz, hz + (shoulder_z - hip_z)
            lean = 0.0
        else:
            swing = 0.28 * stride if pose == "walking" else 0.0
            for side in (-1, 1):
                y = side * hip_w
                foot_x = swing * side
                _limb(bm, 0.0, hip_z, foot_x, 0.07, y, leg_r, 2)
                bm_box(bm, foot_x - 0.08, foot_x + 0.16, y - 0.05, y + 0.05, 0.0, 0.07, material_index=4)
            base, top = hip_z, shoulder_z
            lean = 0.02 if pose == "walking" else 0.0
        # Torso: hips to shoulders, a little wider at the chest; a dress adds a skirt.
        bm_box(bm, -0.10 * k + lean, 0.10 * k + lean, -0.17 * k, 0.17 * k, base - 0.05, top, material_index=1)
        if look.get("dress") and pose != "sitting":
            bm_cylinder(bm, (lean, 0.0, base - 0.18), 0.22 * k, 0.36, segments=10, radius_top=0.15 * k,
                        material_index=1)
        # Arms hang from the shoulders, swinging opposite to the legs when walking.
        for side in (-1, 1):
            y = side * (shoulder_w + arm_r)
            if pose == "sitting":
                _limb(bm, 0.0, top - 0.04, 0.30 * k, top - 0.38 * k, y, arm_r, 1)
            else:
                hand_x = (-0.25 * stride * side if pose == "walking" else 0.02) * k
                _limb(bm, lean, top - 0.04, hand_x, top - 0.62 * k, y, arm_r, 1)
                bm_icosphere(bm, (hand_x, y, top - 0.66 * k), arm_r * 1.1, subdivisions=1, material_index=0)
        # Neck, head and hair.
        bm_cylinder(bm, (lean, 0.0, top + 0.04), 0.045 * k, 0.08, segments=8, material_index=0)
        head_z = top + 0.08 + head_r
        bm_icosphere(bm, (lean + 0.01, 0.0, head_z), head_r, subdivisions=2, scale_z=1.15, material_index=0)
        bm_icosphere(bm, (lean - 0.02, 0.0, head_z + head_r * 0.35), head_r * 1.05, subdivisions=2, scale_z=0.75,
                     material_index=3)
        if look.get("long_hair"):
            bm_box(bm, lean - 0.13 * k, lean - 0.02, -head_r, head_r, head_z - 0.20 * k, head_z + 0.04,
                   material_index=3)

    reach = 0.30 * k if pose == "sitting" else 0.22
    return Prop(builder, materials, "obstacle", "person", reach + 0.1, 0.30 * k)
