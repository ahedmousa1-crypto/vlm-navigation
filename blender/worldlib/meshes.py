"""Mesh helpers: boxes, wedges, bmesh primitives, tagged objects."""

import math

import bmesh
import bpy
from mathutils import Vector

BOX_FACES = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]


def get_collection(name, parent=None):
    """Return a collection by name, creating and linking it on first use."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def link_object(name, surface_type, mesh, materials, collection, category=None):
    if not isinstance(materials, (list, tuple)):
        materials = [materials]
    for material in materials:
        mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    obj["surface_type"] = surface_type
    if category:
        obj["category"] = category
    collection.objects.link(obj)
    return obj


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


def add_poly(name, surface_type, verts, faces, material, collection, category=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return link_object(name, surface_type, mesh, material, collection, category)


def add_box(name, surface_type, bounds, material, collection, category=None):
    x0, x1, y0, y1, z0, z1 = bounds
    return add_poly(
        name, surface_type, box_verts(x0, x1, y0, y1, z0, z1), BOX_FACES, material, collection, category
    )


def add_wedge(name, surface_type, axis, high, low, a0, a1, z_high, z_low, z_base, material, collection, category=None):
    """Solid ramp sloping along ``axis`` ("x" or "y"): top runs from z_high at ``high`` to z_low at ``low``.

    ``a0``/``a1`` bound the other horizontal axis; the base sits at z_base.
    """
    profile = [(high, z_base), (low, z_base), (low, z_base), (high, z_base),
               (high, z_high), (low, z_low), (low, z_low), (high, z_high)]
    across = [a0, a0, a1, a1, a0, a0, a1, a1]
    verts = []
    for (along, z), other in zip(profile, across):
        verts.append((along, other, z) if axis == "x" else (other, along, z))
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_verts = [bm.verts.new(co) for co in verts]
    for face in BOX_FACES:
        bm.faces.new([bm_verts[index] for index in face])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return link_object(name, surface_type, mesh, material, collection, category)


def faces_of(verts):
    return {face for vert in verts for face in vert.link_faces}


def set_material(verts, index):
    for face in faces_of(verts):
        face.material_index = index


def bm_box(bm, x0, x1, y0, y1, z0, z1, material_index=0):
    geom = bmesh.ops.create_cube(bm, size=1.0)
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)
    for vert in geom["verts"]:
        vert.co.x = vert.co.x * sx + cx
        vert.co.y = vert.co.y * sy + cy
        vert.co.z = vert.co.z * sz + cz
    if material_index:
        set_material(geom["verts"], material_index)
    return geom["verts"]


def bm_cylinder(bm, center, radius, height, segments=16, radius_top=None, material_index=0):
    geom = bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=segments,
        radius1=radius,
        radius2=radius if radius_top is None else radius_top,
        depth=height,
    )
    for vert in geom["verts"]:
        vert.co.x += center[0]
        vert.co.y += center[1]
        vert.co.z += center[2]
    if material_index:
        set_material(geom["verts"], material_index)
    return geom["verts"]


def bm_icosphere(bm, center, radius, subdivisions=2, scale_z=1.0, material_index=0):
    geom = bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    for vert in geom["verts"]:
        vert.co.x += center[0]
        vert.co.y += center[1]
        vert.co.z = vert.co.z * scale_z + center[2]
    if material_index:
        set_material(geom["verts"], material_index)
    return geom["verts"]


def bm_wheel(bm, cx, cy, z_hub, radius, width, axle="x", segments=20, material_index=0):
    """Wheel disk standing on its rim; ``axle`` is the local axis it spins around."""
    geom = bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=segments,
        radius1=radius,
        radius2=radius,
        depth=width,
    )
    for vert in geom["verts"]:
        lx, ly, lz = vert.co.x, vert.co.y, vert.co.z
        if axle == "x":
            vert.co.x = cx + lz
            vert.co.y = cy + lx
        else:
            vert.co.x = cx + lx
            vert.co.y = cy + lz
        vert.co.z = z_hub + ly
    if material_index:
        set_material(geom["verts"], material_index)
    return geom["verts"]


def add_bmesh_object(name, surface_type, builder, materials, collection, location=None, yaw=0.0, category=None):
    """Build a mesh in local coordinates, then place it at ``location`` rotated by ``yaw`` (radians)."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    builder(bm)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = link_object(name, surface_type, mesh, materials, collection, category)
    if location is not None:
        obj.location = Vector(location)
    obj.rotation_euler = (0.0, 0.0, yaw)
    return obj


def remove_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


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


def yaw_of(tangent):
    return math.atan2(tangent[1], tangent[0])
