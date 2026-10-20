"""Procedural materials. The street set matches the legacy gen_scene.py look."""

import bpy


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


def flat_material(name, color, roughness=0.55, metallic=0.0):
    """Solid color for markings and painted parts that must read clearly in the render."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_materials():
    """Every material the world uses, keyed by short name."""
    return {
        # Legacy street set (same parameters as legacy/blender/gen_scene.py).
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
        "tree_canopy": layered_material("mat_tree_canopy", (0.05, 0.20, 0.05), (0.28, 0.50, 0.14), 0.75, 7.0, 0.78, 0.95, 0.35),
        # World additions, same layered style.
        "plaza": layered_material("mat_plaza", (0.36, 0.33, 0.29), (0.74, 0.70, 0.62), 0.9, 12.0, 0.6, 0.92, 0.5),
        "terrace": layered_material("mat_terrace", (0.33, 0.31, 0.28), (0.68, 0.65, 0.60), 1.0, 14.0, 0.62, 0.94, 0.5),
        "soil": layered_material("mat_soil", (0.10, 0.07, 0.04), (0.30, 0.22, 0.14), 1.5, 12.0, 0.85, 1.0, 0.6),
        "shrub": layered_material("mat_shrub", (0.04, 0.15, 0.04), (0.20, 0.40, 0.12), 1.2, 9.0, 0.8, 0.95, 0.5),
        "facade_light": layered_material("mat_facade_light", (0.55, 0.50, 0.42), (0.80, 0.76, 0.68), 0.6, 10.0, 0.75, 0.95, 0.25),
        "facade_brick": layered_material("mat_facade_brick", (0.30, 0.12, 0.08), (0.55, 0.28, 0.20), 0.8, 14.0, 0.8, 0.97, 0.35),
        "facade_gray": layered_material("mat_facade_gray", (0.30, 0.31, 0.32), (0.55, 0.56, 0.57), 0.7, 12.0, 0.7, 0.93, 0.25),
        "facade_beige": layered_material("mat_facade_beige", (0.55, 0.45, 0.32), (0.78, 0.68, 0.52), 0.7, 11.0, 0.75, 0.95, 0.25),
        "glass": flat_material("mat_glass", (0.06, 0.09, 0.12), roughness=0.08, metallic=0.4),
        # Separate from window glass so car windows do not glow when windows light up at night.
        "car_glass": flat_material("mat_car_glass", (0.05, 0.07, 0.09), roughness=0.06, metallic=0.4),
        "door": flat_material("mat_door", (0.22, 0.13, 0.07), roughness=0.5),
        "roof": layered_material("mat_roof", (0.10, 0.10, 0.11), (0.25, 0.24, 0.24), 1.0, 15.0, 0.75, 0.95, 0.3),
        "car_red": flat_material("mat_car_red", (0.45, 0.04, 0.03), roughness=0.25, metallic=0.5),
        "car_blue": flat_material("mat_car_blue", (0.05, 0.12, 0.35), roughness=0.25, metallic=0.5),
        "car_white": flat_material("mat_car_white", (0.80, 0.80, 0.78), roughness=0.25, metallic=0.3),
        "car_gray": flat_material("mat_car_gray", (0.25, 0.26, 0.27), roughness=0.3, metallic=0.6),
        "car_black": flat_material("mat_car_black", (0.03, 0.03, 0.035), roughness=0.3, metallic=0.5),
        "tire": flat_material("mat_tire", (0.02, 0.02, 0.02), roughness=0.9),
        "cone_orange": flat_material("mat_cone_orange", (0.90, 0.25, 0.02), roughness=0.5),
        "barrier_red": flat_material("mat_barrier_red", (0.75, 0.05, 0.03), roughness=0.5),
        "barrier_white": flat_material("mat_barrier_white", (0.92, 0.92, 0.90), roughness=0.5),
        "hoarding": layered_material("mat_hoarding", (0.45, 0.36, 0.22), (0.70, 0.60, 0.42), 1.8, 8.0, 0.7, 0.9, 0.2),
        "lamp_glass": flat_material("mat_lamp_glass", (0.90, 0.88, 0.75), roughness=0.3),
    }


CAR_PAINTS = ("car_red", "car_blue", "car_white", "car_gray", "car_black")
FACADES = ("facade_light", "facade_brick", "facade_gray", "facade_beige")
