"""Cat and dog meshes from the Quaternius Animal Pack Vol.2 (CC0), as reusable templates (bpy).

Each OBJ is imported once, joined into one mesh, scaled to a real-world body
length, and moved so the origin sits on the ground under its middle with the
body along +X. Color variants are separate mesh copies with our own flat
materials (the pack's MTL colors are all the same gray).
"""

import json
import os

import bpy
from mathutils import Matrix

from .materials import flat_material

# Body length in metres (nose to rump along the longest horizontal axis).
LENGTH_M = {"cat": 0.55, "dog": 0.95}
# Variant name -> colors for the pack's material slots.
VARIANTS = {
    "cat": {
        "black": {"Grey": (0.035, 0.035, 0.04), "White": (0.85, 0.85, 0.83), "Pink": (0.85, 0.45, 0.55)},
        "gray": {"Grey": (0.33, 0.33, 0.35), "White": (0.85, 0.85, 0.83), "Pink": (0.85, 0.45, 0.55)},
        "orange": {"Grey": (0.62, 0.30, 0.07), "White": (0.90, 0.85, 0.75), "Pink": (0.85, 0.45, 0.55)},
    },
    "dog": {
        "brown": {"Dog": (0.30, 0.17, 0.08)},
        "gray": {"Dog": (0.35, 0.35, 0.36)},
        "golden": {"Dog": (0.75, 0.55, 0.25)},
        "black": {"Dog": (0.05, 0.05, 0.05)},
    },
}


def _import_obj(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path)
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def _normalized_mesh(objects, length_m, name):
    """One mesh from ``objects``: world transform applied, scaled, body along +X, origin on the ground."""
    import bmesh

    bm = bmesh.new()
    materials = []
    for obj in objects:
        mesh = obj.data.copy()
        mesh.transform(obj.matrix_world)
        offset = len(materials)
        materials.extend(mesh.materials)
        start = len(bm.faces)
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        for face in bm.faces[start:]:
            face.material_index += offset
        bpy.data.meshes.remove(mesh)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    center = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, min(zs))
    scale = length_m / max(span_x, span_y)
    matrix = Matrix.Scale(scale, 4) @ Matrix.Translation((-center[0], -center[1], -center[2]))
    if span_y > span_x:
        matrix = Matrix.Rotation(-1.5707963, 4, "Z") @ matrix
    bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    for material in materials:
        out.materials.append(material)
    return out


class AnimalTemplates:
    def __init__(self, asset_dir):
        self.templates = {}  # (kind, variant) -> (mesh, half_length, half_width)
        manifest_path = os.path.join(asset_dir, "assets.json")
        if not os.path.isfile(manifest_path):
            return
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for pack in manifest.get("animals", {}).values():
            for kind, rel_path in pack["files"].items():
                if kind not in LENGTH_M:
                    continue
                objects = _import_obj(os.path.join(asset_dir, rel_path))
                if not objects:
                    continue
                base = _normalized_mesh(objects, LENGTH_M[kind], f"template_{kind}")
                for obj in objects:
                    mesh = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
                xs = [v.co.x for v in base.vertices]
                ys = [v.co.y for v in base.vertices]
                half_l, half_w = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
                for variant, colors in VARIANTS[kind].items():
                    mesh = base.copy()
                    mesh.name = f"template_{kind}_{variant}"
                    for index, slot in enumerate(mesh.materials):
                        key = slot.name.split(".")[0] if slot else ""
                        color = colors.get(key, (0.5, 0.5, 0.5))
                        mesh.materials[index] = flat_material(f"mat_{kind}_{variant}_{key or index}", color, roughness=0.8)
                    self.templates[(kind, variant)] = (mesh, half_l, half_w)
                bpy.data.meshes.remove(base)
        print(f"animal_templates {sorted(self.templates)}")

    def variants(self, kind):
        return [variant for k, variant in self.templates if k == kind]

    def __bool__(self):
        return bool(self.templates)
