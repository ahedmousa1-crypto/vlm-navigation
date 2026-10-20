"""Lighting presets applied at render time: day, overcast, dusk, night, moonlight (bpy).

Day is the legacy look (sky + sun from scene_setup). The others change the
sun, the sky, the exposure, and switch on street-lamp lights and lit windows.
Geometry, navmap, and traces are the same under every preset.
"""

import hashlib
import math

import bpy
from mathutils import Vector

# name: share of views when mixing.
MIX = {"day": 0.55, "overcast": 0.15, "dusk": 0.15, "night": 0.10, "moonlight": 0.05}

PRESETS = {
    # sun_elev/azim in degrees; sky_mix blends the physical sky toward a flat color.
    "day": dict(sun_elev=32, sun_azim=-42, sun_energy=5.0, sun_angle=0.6, sun_color=(1.0, 1.0, 1.0),
                sky_strength=0.58, sky_mix=0.0, sky_color=(0.5, 0.5, 0.5), exposure=-0.14,
                lamps=False, windows=0.0, window_glow=0.0, lighting="Daylight", weather="Clear"),
    "overcast": dict(sun_elev=40, sun_azim=-42, sun_energy=0.7, sun_angle=25.0, sun_color=(0.9, 0.93, 1.0),
                     sky_strength=0.9, sky_mix=0.85, sky_color=(0.55, 0.58, 0.62), exposure=-0.3,
                     lamps=False, windows=0.0, window_glow=0.0, lighting="Daylight", weather="Cloudy"),
    "dusk": dict(sun_elev=3, sun_azim=-70, sun_energy=3.0, sun_angle=1.0, sun_color=(1.0, 0.5, 0.25),
                 sky_strength=0.3, sky_mix=0.55, sky_color=(0.55, 0.3, 0.22), exposure=-0.2,
                 lamps=True, windows=0.25, window_glow=0.6, lighting="Low Light", weather="Clear"),
    "night": dict(sun_elev=35, sun_azim=60, sun_energy=0.03, sun_angle=1.0, sun_color=(0.6, 0.7, 1.0),
                  sky_strength=1.0, sky_mix=1.0, sky_color=(0.004, 0.006, 0.014), exposure=1.6,
                  lamps=True, windows=0.4, window_glow=0.5, lighting="Night", weather="Clear"),
    "moonlight": dict(sun_elev=40, sun_azim=60, sun_energy=0.12, sun_angle=1.0, sun_color=(0.55, 0.65, 1.0),
                      sky_strength=1.0, sky_mix=1.0, sky_color=(0.008, 0.011, 0.024), exposure=2.0,
                      lamps=False, windows=0.0, window_glow=0.0, lighting="Night", weather="Clear"),
}

LAMP_HEAD_LOCAL = Vector((0.0, -1.1, -0.3))  # light position relative to the lamp-head top, in the pole's frame


def preset_for(view_name, mode):
    """Fixed preset name, or a deterministic weighted pick per view when ``mode`` is "mix"."""
    if mode != "mix":
        return mode
    h = int(hashlib.sha1(view_name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    total = 0.0
    for name, share in MIX.items():
        total += share
        if h <= total:
            return name
    return "day"


class Lighting:
    def __init__(self, scene):
        self.scene = scene
        self.sun = bpy.data.objects.get("sun")
        self.lamp_lights = []
        self._prepare_world()
        self._prepare_windows()

    # ---- setup ---------------------------------------------------------

    def _prepare_world(self):
        """Insert a Mix node between the sky and the background so presets can fade to a flat color."""
        tree = self.scene.world.node_tree
        nodes, links = tree.nodes, tree.links
        self.sky = next((n for n in nodes if n.type == "TEX_SKY"), None)
        self.background = next((n for n in nodes if n.type == "BACKGROUND"), None)
        self.mix = nodes.get("preset_mix")
        if self.mix is None and self.sky is not None and self.background is not None:
            self.mix = nodes.new("ShaderNodeMix")
            self.mix.name = "preset_mix"
            self.mix.data_type = "RGBA"
            links.new(self.sky.outputs["Color"], self.mix.inputs["A"])
            links.new(self.mix.outputs["Result"], self.background.inputs["Color"])

    def _prepare_windows(self):
        """Window glass glows in random cells (one cell per window bay) at night."""
        mat = bpy.data.materials.get("mat_glass")
        self.window_value = None
        self.window_strength = None
        if mat is None:
            return
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        if nodes.get("lit_fraction") is not None:
            self.window_value = nodes["lit_fraction"]
            self.window_strength = nodes["lit_strength"]
            return
        bsdf = nodes.get("Principled BSDF")
        geo = nodes.new("ShaderNodeNewGeometry")
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "DIVIDE"
        scale.inputs[1].default_value = (3.0, 3.0, 3.2)  # window pitch and floor height
        floor = nodes.new("ShaderNodeVectorMath")
        floor.operation = "FLOOR"
        noise = nodes.new("ShaderNodeTexWhiteNoise")
        noise.noise_dimensions = "3D"
        lit = nodes.new("ShaderNodeValue")
        lit.name = "lit_fraction"
        lit.outputs[0].default_value = 0.0
        threshold = nodes.new("ShaderNodeMath")
        threshold.operation = "LESS_THAN"
        strength = nodes.new("ShaderNodeValue")
        strength.name = "lit_strength"
        strength.outputs[0].default_value = 0.0
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        links.new(geo.outputs["Position"], scale.inputs[0])
        links.new(scale.outputs["Vector"], floor.inputs[0])
        links.new(floor.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Value"], threshold.inputs[0])
        links.new(lit.outputs[0], threshold.inputs[1])
        links.new(threshold.outputs[0], multiply.inputs[0])
        links.new(strength.outputs[0], multiply.inputs[1])
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.78, 0.45, 1.0)
        links.new(multiply.outputs[0], bsdf.inputs["Emission Strength"])
        self.window_value, self.window_strength = lit, strength

    def _ensure_lamp_lights(self):
        if self.lamp_lights:
            return
        for obj in self.scene.objects:
            if not obj.name.startswith("surface_overhead_lamp"):
                continue
            top = max((obj.matrix_world @ Vector(c)).z for c in obj.bound_box)
            yaw = obj.rotation_euler.z
            local = LAMP_HEAD_LOCAL
            offset = Vector((local.x * math.cos(yaw) - local.y * math.sin(yaw),
                             local.x * math.sin(yaw) + local.y * math.cos(yaw), 0.0))
            data = bpy.data.lights.new(f"light_{obj.name}", "SPOT")
            data.energy = 900.0
            data.color = (1.0, 0.82, 0.55)
            data.spot_size = math.radians(120.0)
            data.spot_blend = 0.6
            data.shadow_soft_size = 0.15
            light = bpy.data.objects.new(f"light_{obj.name}", data)
            light.location = obj.location + offset
            light.location.z = top + local.z
            light.rotation_euler = (0.0, 0.0, 0.0)  # spot points down -Z
            self.scene.collection.objects.link(light)
            self.lamp_lights.append(light)

    # ---- apply ---------------------------------------------------------

    def apply(self, name):
        p = PRESETS[name]
        if self.sun is not None:
            elev, azim = math.radians(p["sun_elev"]), math.radians(p["sun_azim"])
            self.sun.rotation_euler = (math.pi / 2.0 - elev, 0.0, azim)
            self.sun.data.energy = p["sun_energy"]
            self.sun.data.angle = math.radians(p["sun_angle"])
            self.sun.data.color = p["sun_color"]
        if self.sky is not None:
            if hasattr(self.sky, "sun_elevation"):
                self.sky.sun_elevation = math.radians(max(p["sun_elev"], 1.0))
            if hasattr(self.sky, "sun_rotation"):
                self.sky.sun_rotation = math.radians(p["sun_azim"])
        if self.mix is not None:
            self.mix.inputs["Factor"].default_value = p["sky_mix"]
            self.mix.inputs["B"].default_value = (*p["sky_color"], 1.0)
        if self.background is not None:
            self.background.inputs["Strength"].default_value = p["sky_strength"]
        self.scene.view_settings.exposure = p["exposure"]

        if p["lamps"]:
            self._ensure_lamp_lights()
        for light in self.lamp_lights:
            light.hide_render = not p["lamps"]
        glass = bpy.data.materials.get("mat_lamp_glass")
        if glass is not None:
            bsdf = glass.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Emission Color"].default_value = (1.0, 0.85, 0.6, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 25.0 if p["lamps"] else 0.0
        if self.window_value is not None:
            self.window_value.outputs[0].default_value = p["windows"]
            self.window_strength.outputs[0].default_value = p["window_glow"]
        return {"preset": name, "lighting": p["lighting"], "weather": p["weather"], "lamps_on": p["lamps"]}
