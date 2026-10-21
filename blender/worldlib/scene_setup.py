"""Sky, sun, render settings, and cameras. Values match legacy/blender/gen_scene.py."""

import math

import bpy
from mathutils import Vector

from .layout import EYE_HEIGHT
from .meshes import get_collection

RES_X, RES_Y = 1365, 1024
LENS_MM = 32.0
SENSOR_MM = 36.0
LOOK_HEIGHT = 0.85  # legacy aim point: 0.85 m above the ground ...
LOOK_DIST = 15.0    # ... about 15 m ahead


def setup_world():
    elevation = math.radians(32.0)
    azimuth = math.radians(-42.0)
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 5.0
    sun_data.angle = math.radians(0.6)
    if hasattr(sun_data, "use_shadow"):
        sun_data.use_shadow = True
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.pi / 2.0 - elevation, 0.0, azimuth)

    world = bpy.data.worlds.new("outdoor")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_WORLD":
            nodes.remove(node)
    output = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    sky = nodes.new("ShaderNodeTexSky")
    sky_ids = [item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items]
    for candidate in ("NISHITA", "PREETHAM", "HOSEK_WILKIE", "MULTIPLE_SCATTERING"):
        if candidate in sky_ids:
            sky.sky_type = candidate
            break
    for attr, value in (
        ("sun_elevation", elevation),
        ("sun_rotation", azimuth),
        ("sun_intensity", 1.0),
        ("altitude", 0.0),
        ("air_density", 1.0),
        ("dust_density", 0.0),
    ):
        if hasattr(sky, attr):
            setattr(sky, attr, value)
    background.inputs["Strength"].default_value = 0.58
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def select_eevee(scene):
    available = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if candidate in available:
            scene.render.engine = candidate
            return candidate
    raise RuntimeError("EEVEE is not available. Engines: " + ", ".join(available))


def configure_eevee(scene, samples=48):
    select_eevee(scene)
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    # Factory default (what the legacy renders used); set explicitly because the mask pass switches it.
    if "AgX" in _view_transforms(scene):
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    if hasattr(scene.view_settings, "exposure"):
        scene.view_settings.exposure = -0.14
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        return
    for attr, value in (
        ("taa_render_samples", samples),
        ("use_raytracing", False),
        ("use_shadows", True),
        ("shadow_ray_count", 2),
        ("use_fast_gi", True),
        ("fast_gi_ray_count", 2),
        ("shadow_pool_size", "1024"),  # the bigger town overflows the default shadow memory
    ):
        if hasattr(eevee, attr):
            setattr(eevee, attr, value)


def configure_mask_render(scene):
    """Flat per-object colors, no anti-aliasing, no color transform: pixels equal class colors."""
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "OBJECT"
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_object_outline = False
    shading.show_specular_highlight = False
    if hasattr(shading, "show_xray"):
        shading.show_xray = False
    scene.display.render_aa = "OFF"
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.world.color = (0.0, 0.0, 0.0)


def _view_transforms(scene):
    return [item.identifier for item in scene.view_settings.bl_rna.properties["view_transform"].enum_items]


def look_rotation(location, heading, look_height=LOOK_HEIGHT, look_dist=LOOK_DIST, ground_z=None):
    """Euler rotation looking along ``heading`` (radians, 0 = +X) at a point ``look_dist`` ahead."""
    ground = location.z - EYE_HEIGHT if ground_z is None else ground_z
    target = Vector((
        location.x + math.cos(heading) * look_dist,
        location.y + math.sin(heading) * look_dist,
        ground + look_height,
    ))
    return (target - location).to_track_quat("-Z", "Y").to_euler()


def make_view_camera(name, x, y, ground_z, heading, collection_name="Views"):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = LENS_MM
    cam_data.sensor_width = SENSOR_MM
    cam_data.clip_start = 0.1
    cam_data.clip_end = 400.0
    cam = bpy.data.objects.new(name, cam_data)
    get_collection(collection_name).objects.link(cam)
    cam.location = Vector((x, y, ground_z + EYE_HEIGHT))
    cam.rotation_euler = look_rotation(cam.location, heading, ground_z=ground_z)
    cam["ground_z"] = ground_z
    cam["heading_deg"] = math.degrees(heading)
    return cam


def make_overview_camera(extent):
    ex0, ex1, ey0, ey1 = extent
    cam_data = bpy.data.cameras.new("overview")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(ex1 - ex0, ey1 - ey0) * 0.8
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("overview", cam_data)
    get_collection("Overview").objects.link(cam)
    cam.location = Vector(((ex0 + ex1) / 2, (ey0 + ey1) / 2, 300.0))
    cam.rotation_euler = (0.0, 0.0, 0.0)
    return cam
