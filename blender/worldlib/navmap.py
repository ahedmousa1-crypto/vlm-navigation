"""Rasterize the world into a top-down class/height grid by casting rays straight down (bpy).

All non-overhead meshes are merged into one BVH tree with a class id per
polygon; one tree query per cell is far faster than ``scene.ray_cast``.
"""

import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .classes import GROUND_TYPES, class_for

DOWN = Vector((0.0, 0.0, -1.0))
TOP_Z = 80.0


class SceneTree:
    """BVH over every mesh except overhead parts, with per-polygon class id and ground flag."""

    def __init__(self, skip_types=("overhead",)):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, polys, poly_cls, poly_ground = [], [], [], []
        ground_polys = []
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or obj.get("surface_type") in skip_types:
                continue
            surface_type = obj.get("surface_type")
            cls = class_for(surface_type, obj.get("category"))
            is_ground = surface_type in GROUND_TYPES
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            matrix = obj.matrix_world
            base = len(verts)
            verts.extend(matrix @ vert.co for vert in mesh.vertices)
            for poly in mesh.polygons:
                indices = [base + index for index in poly.vertices]
                polys.append(indices)
                poly_cls.append(cls)
                poly_ground.append(is_ground)
                if is_ground:
                    ground_polys.append(indices)
            evaluated.to_mesh_clear()
        self.tree = BVHTree.FromPolygons(verts, polys)
        # Separate ground-only tree: prop bottoms sit exactly on the ground, so
        # stepping through props in the full tree can skip the coincident surface.
        self.ground_tree = BVHTree.FromPolygons(verts, ground_polys)
        self.poly_cls = poly_cls
        self.poly_ground = poly_ground

    def first_hit(self, x, y, from_z=TOP_Z):
        loc, _normal, index, _dist = self.tree.ray_cast(Vector((x, y, from_z)), DOWN)
        if loc is None:
            return 0, None
        return self.poly_cls[index], loc.z

    def ground_below(self, x, y, from_z=TOP_Z):
        """Height of the first ground surface below (x, y, from_z), ignoring props."""
        loc, _normal, _index, _dist = self.ground_tree.ray_cast(Vector((x, y, from_z)), DOWN)
        return None if loc is None else loc.z


class VisTree:
    """BVH over every mesh (overhead parts included) for line-of-sight checks from cameras.

    One tree query is far faster than ``scene.ray_cast`` in a large scene; the
    owner name of each polygon lets a hit on the target itself count as visible.
    """

    def __init__(self):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, polys, owners = [], [], []
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            matrix = obj.matrix_world
            base = len(verts)
            verts.extend(matrix @ vert.co for vert in mesh.vertices)
            for poly in mesh.polygons:
                polys.append([base + index for index in poly.vertices])
                owners.append(obj.name)
            evaluated.to_mesh_clear()
        self.tree = BVHTree.FromPolygons(verts, polys)
        self.owners = owners

    def first_hit(self, origin, direction, distance):
        """(hit distance, owner name) of the first surface along the ray, or (None, None)."""
        loc, _normal, index, dist = self.tree.ray_cast(origin, direction, distance)
        if loc is None:
            return None, None
        return dist, self.owners[index]


def rasterize(tree, extent, cell=0.25):
    x0, x1, y0, y1 = extent
    nx = int(np.ceil((x1 - x0) / cell))
    ny = int(np.ceil((y1 - y0) / cell))
    cls = np.zeros((ny, nx), dtype=np.uint8)
    height = np.zeros((ny, nx), dtype=np.float32)
    start = time.perf_counter()
    for row in range(ny):
        y = y0 + (row + 0.5) * cell
        for col in range(nx):
            value, z = tree.first_hit(x0 + (col + 0.5) * cell, y)
            if z is not None:
                cls[row, col] = value
                height[row, col] = z
    print(f"navmap {nx}x{ny} cells={nx * ny} cell={cell} took={time.perf_counter() - start:.1f}s")
    return {"cls": cls, "height": height, "origin": np.array([x0, y0]), "cell": np.float32(cell)}


SOCIAL_COST = 6.0  # extra walking cost inside a group, a queue or café seating (finite: still passable)


def add_social(navmap, zones):
    """Add a "social" cost layer from polygons (groups, queues, café seating), so paths go around them."""
    cls = navmap["cls"]
    social = np.zeros(cls.shape, dtype=np.float32)
    x0, y0 = navmap["origin"]
    cell = float(navmap["cell"])
    for zone in zones:
        poly = zone["poly"]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        c0, c1 = max(0, int((min(xs) - x0) / cell)), min(cls.shape[1] - 1, int((max(xs) - x0) / cell))
        r0, r1 = max(0, int((min(ys) - y0) / cell)), min(cls.shape[0] - 1, int((max(ys) - y0) / cell))
        for row in range(r0, r1 + 1):
            y = y0 + (row + 0.5) * cell
            for col in range(c0, c1 + 1):
                x = x0 + (col + 0.5) * cell
                inside = False
                for k in range(len(poly)):
                    (ax, ay), (bx, by) = poly[k], poly[(k + 1) % len(poly)]
                    if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                        inside = not inside
                if inside:
                    social[row, col] = SOCIAL_COST
    navmap["social"] = social
    print(f"social_cells {int((social > 0).sum())} from {len(zones)} zones")
    return navmap


def save(navmap, path):
    np.savez_compressed(path, **navmap)
