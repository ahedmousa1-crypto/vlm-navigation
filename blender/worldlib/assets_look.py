"""Optional photo-real look: Poly Haven HDRI sky and PBR ground textures (bpy).

Applied at render time on top of the procedural world, so geometry, navmap,
and traces are identical to the default look. Needs tools/fetch_assets.py.
"""

import math
import os

import bpy
import numpy as np

# World material name -> surface key in assets.json.
MATERIAL_SURFACE = {
    "mat_road": "road",
    "mat_sidewalk": "sidewalk",
    "mat_curb": "curb",
    "mat_plaza": "plaza",
    "mat_terrace": "terrace",
    # Grass stays procedural: the leafy_grass photo texture renders dry and yellow at street scale.
}
WEATHER = {"Clear": "Clear", "Partly cloudy": "Cloudy", "Cloudy": "Cloudy"}


class AssetLook:
    def __init__(self, asset_dir):
        import json

        self.asset_dir = asset_dir
        with open(os.path.join(asset_dir, "assets.json"), encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.hdris = sorted(self.manifest["hdris"])
        self.textured = False
        self.sun_cache = {}

    def path(self, relative):
        return os.path.join(self.asset_dir, relative)

    # ---- textures ------------------------------------------------------

    def apply_textures(self):
        if self.textured:
            return
        by_surface = {rec["surface"]: rec for rec in self.manifest["textures"].values()}
        for mat_name, surface in MATERIAL_SURFACE.items():
            mat = bpy.data.materials.get(mat_name)
            record = by_surface.get(surface)
            if mat is None or record is None:
                continue
            self._pbr(mat, record)
        self.textured = True

    def _pbr(self, mat, record):
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in list(nodes):
            if node.type not in ("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"):
                nodes.remove(node)
        bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
        coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        sx, sy = record["size_m"]
        mapping.inputs["Scale"].default_value = (1.0 / sx, 1.0 / sy, 1.0 / sx)
        # Ground objects are built in world coordinates with their origin at 0, so Object == world.
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        def image_node(key, non_color):
            node = nodes.new("ShaderNodeTexImage")
            node.image = bpy.data.images.load(self.path(record["maps"][key]), check_existing=True)
            node.projection = "BOX"
            node.projection_blend = 0.2
            if non_color:
                node.image.colorspace_settings.name = "Non-Color"
            links.new(mapping.outputs["Vector"], node.inputs["Vector"])
            return node

        links.new(image_node("diff", False).outputs["Color"], bsdf.inputs["Base Color"])
        links.new(image_node("rough", True).outputs["Color"], bsdf.inputs["Roughness"])
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(image_node("nor_gl", True).outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Metallic"].default_value = 0.0

    # ---- sky -----------------------------------------------------------

    def hdri_for(self, view_name):
        return self.hdris[sum(map(ord, view_name)) % len(self.hdris)]

    def apply_sky(self, view_name):
        """Swap the world to this view's HDRI and aim the sun lamp at its brightest spot. Returns weather."""
        asset_id = self.hdri_for(view_name)
        record = self.manifest["hdris"][asset_id]
        image = bpy.data.images.load(self.path(record["file"]), check_existing=True)

        world = bpy.context.scene.world
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        for node in list(nodes):
            if node.type != "OUTPUT_WORLD":
                nodes.remove(node)
        output = next(node for node in nodes if node.type == "OUTPUT_WORLD")
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = image
        background = nodes.new("ShaderNodeBackground")
        background.inputs["Strength"].default_value = 1.0
        links.new(env.outputs["Color"], background.inputs["Color"])
        links.new(background.outputs["Background"], output.inputs["Surface"])

        elevation, azimuth, contrast = self.sun_from(image, asset_id)
        sun = bpy.data.objects.get("sun")
        if sun is not None:
            sun.rotation_euler = (math.pi / 2.0 - elevation, 0.0, azimuth)
            # A sharp sun only where the sky has one; overcast skies get soft light.
            sun.data.energy = 4.0 if contrast > 50.0 else 0.6
            sun.data.angle = math.radians(0.6 if contrast > 50.0 else 20.0)
        return asset_id, WEATHER.get(record["weather"], "Clear")

    def sun_from(self, image, asset_id):
        if asset_id in self.sun_cache:
            return self.sun_cache[asset_id]
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgba = pixels.reshape(height, width, 4)
        lum = rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722
        row, col = np.unravel_index(np.argmax(lum), lum.shape)
        contrast = float(lum[row, col] / max(1e-6, np.median(lum)))
        # Equirectangular: rows go from -90 (bottom, row 0 in Blender) to +90 degrees.
        elevation = ((row + 0.5) / height - 0.5) * math.pi
        u = (col + 0.5) / width
        # Blender's environment lookup: u = atan2(dir.y, -dir.x) / (2 pi) + 0.5.
        phi = (u - 0.5) * 2.0 * math.pi
        direction = (-math.cos(phi), math.sin(phi))
        # With rotation (pi/2 - elevation, 0, az) the lamp sits toward (sin az, -cos az); solve for az.
        azimuth = math.atan2(direction[0], -direction[1])
        result = (max(math.radians(8.0), elevation), azimuth, contrast)
        self.sun_cache[asset_id] = result
        return result
