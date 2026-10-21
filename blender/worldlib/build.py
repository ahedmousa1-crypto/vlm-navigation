"""Turn a layout (see layout.py) into tagged Blender geometry."""

import math
import random

import bpy

from . import layout as lay
from . import props
from .materials import CAR_PAINTS, FACADES
from .newblocks import NewBlocksMixin
from .meshes import (
    add_bmesh_object,
    add_box,
    add_wedge,
    bm_box,
    get_collection,
)

FLOOR_H = 3.2
BAR_W = 0.5  # zebra bar and gap width


def ring_boxes(bm, outer, inner, z_top):
    """Four boxes filling ``outer`` minus ``inner`` (both x0, x1, y0, y1) from BASE_Z to z_top."""
    ox0, ox1, oy0, oy1 = outer
    ix0, ix1, iy0, iy1 = inner
    bm_box(bm, ox0, ox1, oy0, iy0, lay.BASE_Z, z_top)
    bm_box(bm, ox0, ox1, iy1, oy1, lay.BASE_Z, z_top)
    bm_box(bm, ox0, ix0, iy0, iy1, lay.BASE_Z, z_top)
    bm_box(bm, ix1, ox1, iy0, iy1, lay.BASE_Z, z_top)


def rects_overlap(a, b, gap):
    return not (a[0] > b[1] + gap or a[1] < b[0] - gap or a[2] > b[3] + gap or a[3] < b[2] - gap)


class WorldBuilder(NewBlocksMixin):
    def __init__(self, layout, materials, animals=None):
        self.layout = layout
        self.mat = materials
        self.animals = animals
        self.rng = random.Random(layout["seed"] * 7919 + 1)
        root = get_collection("World")
        self.coll = {
            name: get_collection(name, root)
            for name in ("Ground", "Markings", "Blocks", "Props", "Buildings", "Vegetation", "Animals", "Traffic")
        }
        self.reserved = []
        self.counter = {}
        self.benches = []  # (x, y, block id) of sidewalk benches, for greenery placed beside them

    # ---- helpers -------------------------------------------------------

    def next_name(self, prefix):
        index = self.counter.get(prefix, 0)
        self.counter[prefix] = index + 1
        return f"{prefix}_{index:03d}"

    def materials_for(self, prop):
        return [self.mat[key] for key in prop.materials]

    def place(self, prop, block, side, s, d, ground_z, prefix, turn=0.0, gap=0.4, reserve=True, collection="Props"):
        """Place a prop at side-local (s, d). ``turn`` rotates it from the side's tangent (radians)."""
        _origin, tangent, _inward = lay.side_frame(block, side)
        x, y = lay.side_point(block, side, s, d)
        yaw = math.atan2(tangent[1], tangent[0]) + turn
        return self.place_world(prop, x, y, ground_z, yaw, prefix, gap, reserve, collection)

    def place_world(self, prop, x, y, ground_z, yaw, prefix, gap=0.4, reserve=True, collection="Props"):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex = prop.hx * c + prop.hy * s
        ey = prop.hx * s + prop.hy * c
        rect = (x - ex, x + ex, y - ey, y + ey)
        if reserve:
            if any(rects_overlap(rect, other, gap) for other in self.reserved):
                return None
            self.reserved.append(rect)
        return add_bmesh_object(
            self.next_name(prefix),
            prop.surface_type,
            prop.builder,
            self.materials_for(prop),
            self.coll[collection],
            location=(x, y, ground_z),
            yaw=yaw,
            category=prop.category,
        )

    def place_mesh(self, mesh, half_l, half_w, x, y, ground_z, yaw, prefix, category, gap=0.3, collection="Animals"):
        """Place a shared template mesh (e.g. an animal) with the same spacing rules as props."""
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = half_l * c + half_w * s, half_l * s + half_w * c
        rect = (x - ex, x + ex, y - ey, y + ey)
        if any(rects_overlap(rect, other, gap) for other in self.reserved):
            return None
        self.reserved.append(rect)
        obj = bpy.data.objects.new(self.next_name(prefix), mesh)
        obj["surface_type"] = "obstacle"
        obj["category"] = category
        obj.location = (x, y, ground_z)
        obj.rotation_euler = (0.0, 0.0, yaw)
        self.coll[collection].objects.link(obj)
        return obj

    def keep_clear(self, rect):
        """Reserve an area so later props stay out of it (stair access, closures, crossings)."""
        self.reserved.append(rect)

    # ---- ground --------------------------------------------------------

    def build_ground(self):
        rx0, rx1, ry0, ry1 = self.layout["road_extent"]
        ex0, ex1, ey0, ey1 = self.layout["extent"]
        ground = self.coll["Ground"]
        add_box("surface_road_slab", "road", (rx0, rx1, ry0, ry1, lay.BASE_Z, lay.ROAD_Z), self.mat["road"], ground)

        # Curb and sidewalk around the outside of the outer streets.
        c, w, z = lay.CURB_WIDTH, lay.OUTER_SIDEWALK_W, lay.OUTER_CURB_H
        road = (rx0, rx1, ry0, ry1)
        curb = (rx0 - c, rx1 + c, ry0 - c, ry1 + c)
        walk = (rx0 - c - w, rx1 + c + w, ry0 - c - w, ry1 + c + w)
        add_bmesh_object("surface_curb_outer", "curb", lambda bm: ring_boxes(bm, curb, road, z), [self.mat["curb"]], ground)
        add_bmesh_object("surface_sidewalk_outer", "sidewalk", lambda bm: ring_boxes(bm, walk, curb, z),
                         [self.mat["sidewalk"]], ground)

        wx0, wx1, wy0, wy1 = walk
        outer = [
            (ex0, ex1, ey0, wy0),
            (ex0, ex1, wy1, ey1),
            (ex0, wx0, wy0, wy1),
            (wx1, ex1, wy0, wy1),
        ]
        for index, (x0, x1, y0, y1) in enumerate(outer):
            add_box(f"surface_grass_outer_{index}", "grass", (x0, x1, y0, y1, lay.BASE_Z, lay.GRASS_Z), self.mat["grass"], ground)

    def build_center_lines(self):
        xs, ys, wx, wy = self.layout["xs"], self.layout["ys"], self.layout["wx"], self.layout["wy"]
        dash, period, half_w = 3.0, 9.0, 0.06
        z0, z1 = lay.ROAD_Z + 0.002, lay.ROAD_Z + 0.006

        def builder(bm):
            for i, x in enumerate(xs):
                for j in range(len(ys) - 1):
                    start = ys[j] + wy[j] / 2 + 6.0
                    stop = ys[j + 1] - wy[j + 1] / 2 - 6.0
                    y = start
                    while y + dash <= stop:
                        bm_box(bm, x - half_w, x + half_w, y, y + dash, z0, z1)
                        y += period
            for j, y in enumerate(ys):
                for i in range(len(xs) - 1):
                    start = xs[i] + wx[i] / 2 + 6.0
                    stop = xs[i + 1] - wx[i + 1] / 2 - 6.0
                    x = start
                    while x + dash <= stop:
                        bm_box(bm, x, x + dash, y - half_w, y + half_w, z0, z1)
                        x += period

        add_bmesh_object("surface_lane_marking_center", "lane_marking", builder, [self.mat["crosswalk"]],
                         self.coll["Markings"], category="lane_marking")

    def build_crosswalks(self, core):
        """Crosswalks between two core blocks (``core=True``) or touching a new block (``core=False``)."""
        markings = self.coll["Markings"]
        ramp = lay.CURB_RAMP_LEN
        z0 = lay.ROAD_Z + lay.CROSSWALK_LIFT
        z1 = z0 + lay.CROSSWALK_THICK
        landing = lay.CURB_WIDTH + 0.8
        blocks = self.layout["blocks"]
        for cw in self.layout["crosswalks"]:
            touches_new = any(blocks[b].get("new") for b in cw["ends"])
            if touches_new == core:
                continue
            x0, x1, y0, y1 = cw["x0"], cw["x1"], cw["y0"], cw["y1"]
            h_a, h_b = cw["curb_h"]
            # Zebra bars run parallel to the traffic, so a walker crosses them like rungs.
            if cw["walk_axis"] == "x":
                count = max(5, round((x1 - x0 - 2 * ramp) / BAR_W))
                step = (x1 - x0 - 2 * ramp) / count

                def builder(bm, x0=x0, y0=y0, y1=y1, step=step, count=count):
                    for index in range(count):
                        sx = x0 + ramp + index * step
                        bm_box(bm, sx, sx + step, y0, y1, z0, z1, material_index=index % 2)

                ym = (y0 + y1) / 2
                ends = ((x0 - landing, ym), (x1 + landing, ym))
                add_wedge(f"surface_curb_ramp_{cw['id']}_a", "curb_ramp", "x", x0, x0 + ramp, y0, y1, h_a, lay.ROAD_Z, lay.BASE_Z, self.mat["curb"], markings, "curb_ramp")
                add_wedge(f"surface_curb_ramp_{cw['id']}_b", "curb_ramp", "x", x1, x1 - ramp, y0, y1, h_b, lay.ROAD_Z, lay.BASE_Z, self.mat["curb"], markings, "curb_ramp")
            else:
                count = max(5, round((y1 - y0 - 2 * ramp) / BAR_W))
                step = (y1 - y0 - 2 * ramp) / count

                def builder(bm, x0=x0, x1=x1, y0=y0, step=step, count=count):
                    for index in range(count):
                        sy = y0 + ramp + index * step
                        bm_box(bm, x0, x1, sy, sy + step, z0, z1, material_index=index % 2)

                xm = (x0 + x1) / 2
                ends = ((xm, y0 - landing), (xm, y1 + landing))
                add_wedge(f"surface_curb_ramp_{cw['id']}_a", "curb_ramp", "y", y0, y0 + ramp, x0, x1, h_a, lay.ROAD_Z, lay.BASE_Z, self.mat["curb"], markings, "curb_ramp")
                add_wedge(f"surface_curb_ramp_{cw['id']}_b", "curb_ramp", "y", y1, y1 - ramp, x0, x1, h_b, lay.ROAD_Z, lay.BASE_Z, self.mat["curb"], markings, "curb_ramp")
            obj = add_bmesh_object(f"surface_crosswalk_{cw['id']}", "crosswalk", builder,
                                   [self.mat["crosswalk"], self.mat["crosswalk_dark"]], markings, category="crosswalk")
            obj["end_a"] = ends[0]
            obj["end_b"] = ends[1]
            # Landing areas on both sidewalks stay free of props.
            for ex, ey in ends:
                self.keep_clear((ex - 1.6, ex + 1.6, ey - 1.6, ey + 1.6))

    # ---- blocks --------------------------------------------------------

    def build_blocks(self, blocks):
        for block in blocks:
            self.build_block_ground(block)
        # Access areas first so furniture never blocks stairs, ramps, or closures.
        for block in blocks:
            if block["variant"] == "terrace":
                self.build_terrace(block)
            if block["variant"] == "construction":
                self.build_closures(block)
        for block in blocks:
            self.build_block_props(block)

    def build_block_ground(self, block):
        c = lay.CURB_WIDTH
        z = block["sidewalk_z"]
        outer = (block["x0"], block["x1"], block["y0"], block["y1"])
        curb_inner = (block["x0"] + c, block["x1"] - c, block["y0"] + c, block["y1"] - c)
        interior = (block["ix0"], block["ix1"], block["iy0"], block["iy1"])
        walk_type = "plaza" if block["variant"] == "plaza" else "sidewalk"
        blocks = self.coll["Blocks"]
        add_bmesh_object(f"surface_curb_{block['id']}", "curb", lambda bm: ring_boxes(bm, outer, curb_inner, z),
                         [self.mat["curb"]], blocks)
        add_bmesh_object(f"surface_{walk_type}_{block['id']}", walk_type,
                         lambda bm: ring_boxes(bm, curb_inner, interior, z), [self.mat[walk_type]], blocks)

        for side, info in block["sides"].items():
            if info["bike_w"] > 0:
                x0, x1, y0, y1 = lay.side_rect(block, side, 0.0, info["length"], -info["bike_w"], 0.0)
                z0 = lay.ROAD_Z + lay.BIKE_LANE_LIFT
                add_box(f"surface_bike_lane_{block['id']}_{side}", "bike_lane",
                        (x0, x1, y0, y1, z0, z0 + lay.BIKE_LANE_THICK), self.mat["bike_lane"], self.coll["Markings"])

        variant = block["variant"]
        if variant in ("street", "construction", "neighborhood", "park", "skate"):
            add_box(f"surface_grass_{block['id']}", "grass", (*interior, lay.BASE_Z, lay.GRASS_Z), self.mat["grass"], blocks)
        elif variant in ("plaza", "narrow"):
            add_box(f"surface_{walk_type}_{block['id']}_inner", walk_type, (*interior, lay.BASE_Z, z), self.mat[walk_type], blocks)
            self.build_buildings(block)
        if variant == "neighborhood":
            self.build_neighborhood(block)
        elif variant == "park":
            self.build_park(block)
        elif variant == "skate":
            self.build_skate(block)
        # Terrace interior is built in build_terrace.

    def build_terrace(self, block):
        t = block["terrace"]
        side = t["side"]
        z = block["sidewalk_z"]
        tz = t["z"]
        ring = lay.CURB_WIDTH + block["sidewalk_w"]
        depth = lay.N_STEPS * lay.TREAD
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        plateau = {
            "south": (ix0, ix1, iy0 + depth, iy1),
            "north": (ix0, ix1, iy0, iy1 - depth),
            "west": (ix0 + depth, ix1, iy0, iy1),
            "east": (ix0, ix1 - depth, iy0, iy1),
        }[side]
        blocks = self.coll["Blocks"]
        add_box(f"surface_terrace_{block['id']}", "terrace", (*plateau, lay.BASE_Z, tz), self.mat["terrace"], blocks)
        t["plateau"] = plateau

        length = block["sides"][side]["length"]
        s_in0, s_in1 = ring, length - ring
        s0 = t["s0"]
        s_stairs = (s0, s0 + lay.STAIR_W)
        s_ramp = (s0 + lay.STAIR_W, s0 + lay.STAIR_W + lay.RAMP_W)

        def stairs_builder(bm):
            for k in range(lay.N_STEPS):
                rect = lay.side_rect(block, side, s_stairs[0], s_stairs[1], ring + k * lay.TREAD, ring + (k + 1) * lay.TREAD)
                bm_box(bm, *rect, lay.BASE_Z, z + (k + 1) * lay.RISE)

        add_bmesh_object(f"surface_stairs_{block['id']}", "stairs", stairs_builder, [self.mat["stairs"]], blocks, category="stairs")

        rx0, rx1, ry0, ry1 = lay.side_rect(block, side, s_ramp[0], s_ramp[1], ring, ring + depth)
        hi = lay.side_point(block, side, s_ramp[0], ring + depth)
        lo = lay.side_point(block, side, s_ramp[0], ring)
        if side in ("south", "north"):
            add_wedge(f"surface_ramp_{block['id']}", "ramp", "y", hi[1], lo[1], rx0, rx1, tz, z, lay.BASE_Z, self.mat["ramp"], blocks, "ramp")
        else:
            add_wedge(f"surface_ramp_{block['id']}", "ramp", "x", hi[0], lo[0], ry0, ry1, tz, z, lay.BASE_Z, self.mat["ramp"], blocks, "ramp")

        for index, (a, b) in enumerate(((s_in0, s_stairs[0]), (s_ramp[1], s_in1))):
            if b - a > 0.05:
                rect = lay.side_rect(block, side, a, b, ring, ring + depth)
                add_box(f"surface_grass_{block['id']}_strip{index}", "grass", (*rect, lay.BASE_Z, lay.GRASS_Z), self.mat["grass"], blocks)
                # Railing along the terrace edge above the grass strip.
                if b - a > 2.0:
                    fence = props.make_fence(b - a - 0.6)
                    self.place(fence, block, side, (a + b) / 2, ring + depth + 0.15, tz, "surface_obstacle_railing",
                               reserve=False)

        # Keep the sidewalk in front of the stairs and ramp free.
        self.keep_clear(lay.side_rect(block, side, s_stairs[0] - 0.8, s_ramp[1] + 0.8, lay.CURB_WIDTH + 0.6, ring + depth + 1.5))
        t["stairs_s"] = s_stairs
        t["ramp_s"] = s_ramp

        # A few things on the terrace itself.
        px0, px1, py0, py1 = plateau
        for index in range(self.rng.randint(2, 3)):
            for _ in range(20):
                x = self.rng.uniform(px0 + 2.5, px1 - 2.5)
                y = self.rng.uniform(py0 + 2.5, py1 - 2.5)
                if self.add_tree(x, y, tz, prefix="surface_obstacle_tree_terrace"):
                    break
        for index in range(2):
            for _ in range(20):
                x = self.rng.uniform(px0 + 2.0, px1 - 2.0)
                y = self.rng.uniform(py0 + 2.0, py1 - 2.0)
                if self.place_world(props.make_bench(), x, y, tz, self.rng.choice((0.0, math.pi / 2)) , "surface_obstacle_bench"):
                    break

    def build_closures(self, block):
        """Construction: part of the sidewalk fenced off with barriers and cones."""
        z = block["sidewalk_z"]
        c = lay.CURB_WIDTH
        ring = c + block["sidewalk_w"]
        for closure in block["closures"]:
            side, s0, length = closure["side"], closure["s0"], closure["len"]
            gap = self.rng.uniform(0.9, 1.2)
            d_line = c + gap + 0.25
            closure["d_line"] = d_line
            count = max(1, round(length / 1.6))
            unit = length / count
            for index in range(count):
                self.place(props.make_barrier(unit - 0.05), block, side, s0 + (index + 0.5) * unit, d_line, z,
                           "surface_obstacle_barrier", reserve=False)
            end_len = ring - d_line - 0.1
            if end_len > 0.6:
                for s_end in (s0, s0 + length):
                    self.place(props.make_barrier(end_len), block, side, s_end, d_line + end_len / 2, z,
                               "surface_obstacle_barrier", turn=math.pi / 2, reserve=False)
            for s_cone in (s0 - 0.5, s0 + length + 0.5):
                self.place(props.make_cone(), block, side, s_cone, d_line, z, "surface_obstacle_cone", reserve=False)
            mid = s0 + length / 2
            self.place(props.make_crate(1.2, 0.8, 0.6), block, side, mid, (d_line + ring) / 2 + 0.1, z,
                       "surface_obstacle_crate", reserve=False)
            # The free walking gap and the work area stay clear of other furniture.
            self.keep_clear(lay.side_rect(block, side, s0 - 1.5, s0 + length + 1.5, c, ring))

    def build_block_props(self, block):
        variant = block["variant"]
        z = block["sidewalk_z"]
        for side, info in block["sides"].items():
            length = info["length"]
            if variant != "narrow":
                self.add_lamps(block, side, length, z, spacing=self.rng.uniform(22.0, 26.0))
            if variant in ("street", "terrace", "construction"):
                self.add_furniture_clusters(block, side, length, z)
            elif variant == "plaza":
                self.add_plaza_row(block, side, length, z)
            elif variant == "narrow":
                self.add_lamps(block, side, length, z, spacing=30.0)
                self.add_tree_pits(block, side, length, z)
                self.add_parked_cars(block, side, length)
            elif variant == "neighborhood" and info["parked_cars"]:
                self.add_parked_cars(block, side, length)
        if variant == "street":
            self.add_interior_trees_and_fence(block)
            self.add_greenery(block)
        if variant == "construction":
            self.add_construction_site(block)
        if block.get("extras"):
            self.add_scattered(block)
        self.add_animals(block)

    def add_lamps(self, block, side, length, z, spacing):
        s = 7.0
        while s < length - 7.0:
            if self.place(props.make_lamp_pole(), block, side, s, lay.CURB_WIDTH + 0.4, z, "surface_obstacle_lamp"):
                self.place(props.make_lamp_head(), block, side, s, lay.CURB_WIDTH + 0.4, z, "surface_overhead_lamp",
                           reserve=False)
            s += spacing

    def add_furniture_clusters(self, block, side, length, z):
        """Legacy cluster: bench, trash can, bike, and 2-3 bollards on the inner edge of the sidewalk."""
        c = lay.CURB_WIDTH
        width = block["sidewalk_w"]
        lane_w = min(1.4, width * 0.42)
        d0, d1 = c + lane_w, c + width - 0.28
        if d1 - d0 < 0.7:
            return
        clusters = 1 if length < 45.0 else 2
        band = (length - 16.0) / clusters
        for index in range(clusters):
            s_a = 8.0 + index * band + self.rng.uniform(0.15, 0.55) * band
            d_mid = (d0 + d1) / 2
            if self.place(props.make_bench(), block, side, s_a, d1 - 0.22, z, "surface_obstacle_bench"):
                self.benches.append({"block": block["id"], "side": side, "s": s_a})
            self.place(props.make_trash_can(), block, side, s_a + 1.6, d_mid, z, "surface_obstacle_trash_can")
            bike_d = min(d1 - 0.58, max(d0 + 0.58, d_mid))
            self.place(props.make_bike(), block, side, s_a - 1.9, bike_d, z, "surface_obstacle_bike", gap=0.25)
            for k in range(self.rng.randint(2, 3)):
                self.place(props.make_bollard(), block, side, s_a + 3.0 + k * 0.9, d_mid, z, "surface_obstacle_bollard",
                           gap=0.3)

    def add_plaza_row(self, block, side, length, z):
        d = lay.CURB_WIDTH + block["sidewalk_w"] * 0.62
        s = 8.0
        index = 0
        while s < length - 8.0:
            if index % 2 == 0:
                self.place(props.make_planter(self.rng.uniform(1.0, 1.4)), block, side, s, d, z, "surface_obstacle_planter")
            else:
                self.place(props.make_bench(), block, side, s, d, z, "surface_obstacle_bench")
                if index % 4 == 1:
                    self.place(props.make_trash_can(), block, side, s + 1.5, d, z, "surface_obstacle_trash_can")
            s += self.rng.uniform(4.0, 5.0)
            index += 1

    def add_tree_pits(self, block, side, length, z):
        d = lay.CURB_WIDTH + 0.5
        s = 7.0 + self.rng.uniform(0.0, 2.0)
        blocks = self.coll["Blocks"]
        while s < length - 7.0:
            x, y = lay.side_point(block, side, s, d)
            rect = (x - 0.45, x + 0.45, y - 0.45, y + 0.45)
            if not any(rects_overlap(rect, other, 0.3) for other in self.reserved):
                add_box(self.next_name("surface_tree_pit"), "tree_pit", (*rect, lay.BASE_Z, z + 0.01), self.mat["soil"], blocks)
                self.add_tree(x, y, z + 0.01, prefix="surface_obstacle_tree_pit", trunk_r=0.1, canopy=(0.7, 0.95), reserve_gap=0.0)
            s += self.rng.uniform(9.0, 11.0)

    def add_parked_cars(self, block, side, length):
        s = 9.0 + self.rng.uniform(0.0, 2.0)
        while s < length - 9.0:
            if self.rng.random() > 0.25:
                turn = 0.0 if self.rng.random() < 0.5 else math.pi
                paint = self.rng.choice(CAR_PAINTS)
                car = self.place(props.make_car(paint), block, side, s, -1.15, lay.ROAD_Z, "surface_obstacle_car",
                                 turn=turn, gap=0.3)
                if car is not None:
                    car["color_name"] = paint.split("_", 1)[1]
            s += self.rng.uniform(5.8, 6.8)

    def add_tree(self, x, y, ground_z, prefix, trunk_r=0.14, canopy=(0.75, 1.05), reserve_gap=0.45, style="round",
                 trunk_only_check=False):
        """Place a tree. ``style``: "round" (legacy), "tall" (oval canopy), or "conifer" (solid, low branches)."""
        if style == "conifer":
            height, radius = self.rng.uniform(3.0, 5.0), self.rng.uniform(0.8, 1.2)
            obj = self.place_world(props.make_conifer(height, radius), x, y, ground_z, 0.0, prefix,
                                   gap=reserve_gap, collection="Vegetation")
            if obj is not None:
                obj["tree_style"] = "conifer"
            return obj is not None
        if style == "tall":
            trunk_h = self.rng.uniform(2.6, 3.4)
            canopy_r = self.rng.uniform(0.55, 0.8)
        else:
            trunk_h = self.rng.uniform(2.0, 2.6)
            canopy_r = self.rng.uniform(*canopy)
        reach = trunk_r if trunk_only_check else canopy_r
        rect = (x - reach, x + reach, y - reach, y + reach)
        if reserve_gap and any(rects_overlap(rect, other, reserve_gap) for other in self.reserved):
            return False
        trunk = self.place_world(props.make_tree_trunk(trunk_h, trunk_r), x, y, ground_z, 0.0, prefix, gap=0.0,
                                 reserve=False, collection="Vegetation")
        trunk["tree_style"] = style
        self.reserved.append((x - trunk_r, x + trunk_r, y - trunk_r, y + trunk_r))
        canopy_prop = props.make_tall_canopy(trunk_h, canopy_r) if style == "tall" else props.make_tree_canopy(trunk_h, canopy_r)
        self.place_world(canopy_prop, x, y, ground_z, 0.0, "surface_overhead_canopy", reserve=False,
                         collection="Vegetation")
        return trunk is not None

    def add_bush(self, x, y, ground_z, radius=None, gap=0.3):
        radius = radius or self.rng.uniform(0.4, 0.75)
        return self.place_world(props.make_bush(radius, self.rng), x, y, ground_z, self.rng.uniform(0, math.pi),
                                "surface_obstacle_bush", gap=gap, collection="Vegetation")

    def add_greenery(self, block):
        """Bushes and mixed trees on the grass, some right behind benches ("the tree beside the bench")."""
        ring = lay.CURB_WIDTH + block["sidewalk_w"]
        for bench in (b for b in self.benches if b["block"] == block["id"]):
            if self.rng.random() < 0.7:
                side, s = bench["side"], bench["s"] + self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.6, 1.2)
                if self.rng.random() < 0.5:
                    x, y = lay.side_point(block, side, s, ring + 1.5)
                    self.add_tree(x, y, lay.GRASS_Z, "surface_obstacle_tree", style=self.rng.choice(("round", "tall")),
                                  reserve_gap=0.2, trunk_only_check=True)
                else:
                    radius = self.rng.uniform(0.4, 0.6)
                    x, y = lay.side_point(block, side, s, ring + 0.8 + radius)
                    self.add_bush(x, y, lay.GRASS_Z, radius, gap=0.1)
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        for _ in range(self.rng.randint(2, 4)):
            for _attempt in range(20):
                x, y = self.rng.uniform(ix0 + 2.5, ix1 - 2.5), self.rng.uniform(iy0 + 2.5, iy1 - 2.5)
                if self.add_tree(x, y, lay.GRASS_Z, "surface_obstacle_tree", style=self.rng.choice(("conifer", "tall"))):
                    break
        # Bushes along the grass edge, where sidewalk cameras can see them.
        for side in lay.SIDES:
            length = block["sides"][side]["length"]
            for _ in range(self.rng.randint(1, 3)):
                s = self.rng.uniform(ring + 3.0, length - ring - 3.0)
                radius = self.rng.uniform(0.45, 0.75)
                x, y = lay.side_point(block, side, s, ring + 1.0 + radius + self.rng.uniform(0.0, 2.0))
                self.add_bush(x, y, lay.GRASS_Z, radius)

    def add_animals(self, block):
        """One or two cats/dogs per block: on the grass near the edge, on the sidewalk, or on the plaza."""
        if not self.animals:
            return
        variant = block["variant"]
        if variant == "construction":
            return
        ring = lay.CURB_WIDTH + block["sidewalk_w"]
        # New blocks get more animals so cat/dog tasks and Dynamic Obstacle cases are not rare.
        count = self.rng.randint(2, 4) if block.get("new") else self.rng.randint(1, 2)
        for _ in range(count):
            kind = self.rng.choice(("cat", "dog"))
            color = self.rng.choice(self.animals.variants(kind))
            mesh, half_l, half_w = self.animals.templates[(kind, color)]
            for _attempt in range(25):
                side = self.rng.choice(lay.SIDES)
                length = block["sides"][side]["length"]
                s = self.rng.uniform(ring + 2.0, length - ring - 2.0)
                on_grass = variant in ("street", "park", "neighborhood") and self.rng.random() < 0.6
                if on_grass:
                    d, z = ring + self.rng.uniform(0.8, 3.0), lay.GRASS_Z
                else:
                    d, z = lay.CURB_WIDTH + self.rng.uniform(0.4, block["sidewalk_w"] - 0.4), block["sidewalk_z"]
                x, y = lay.side_point(block, side, s, d)
                obj = self.place_mesh(mesh, half_l, half_w, x, y, z, self.rng.uniform(0, 2 * math.pi),
                                      f"surface_obstacle_{kind}", kind)
                if obj is not None:
                    obj["color_name"] = color
                    break

    def add_traffic(self):
        """Cars stopped in the driving lanes of one inner street (parked cars stay at the curb)."""
        xs, ys, wx, wy = self.layout["xs"], self.layout["ys"], self.layout["wx"], self.layout["wy"]
        axis, index = self.layout["traffic_street"]
        if axis == "h":
            center, width, cross = ys[index], wy[index], [(x, w) for x, w in zip(xs, wx)]
        else:
            center, width, cross = xs[index], wx[index], [(y, w) for y, w in zip(ys, wy)]
        lane = width / 4
        placed = 0
        for (a, wa), (b, wb) in zip(cross, cross[1:]):
            # Keep clear of the intersections and the crosswalks next to them.
            start, stop = a + wa / 2 + 6.0, b - wb / 2 - 6.0
            for sign in (1.0, -1.0):
                pos = start + self.rng.uniform(0.0, 4.0)
                while pos < stop - 2.5:
                    if self.rng.random() < 0.6:
                        yaw = 0.0 if sign > 0 else math.pi
                        if axis == "v":
                            yaw += math.pi / 2
                        x, y = (pos, center - sign * lane) if axis == "h" else (center + sign * lane, pos)
                        paint = self.rng.choice(CAR_PAINTS)
                        car = self.place_world(props.make_car(paint), x, y, lay.ROAD_Z, yaw,
                                               "surface_obstacle_car_traffic", gap=0.6, collection="Traffic")
                        if car is not None:
                            car["in_traffic"] = True
                            car["color_name"] = paint.split("_", 1)[1]
                            placed += 1
                    pos += self.rng.uniform(7.0, 14.0)
        print(f"traffic_cars {placed} on {axis}{index}")

    def add_interior_trees_and_fence(self, block):
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        ring = lay.CURB_WIDTH + block["sidewalk_w"]
        for side in self.rng.sample(lay.SIDES, self.rng.randint(1, 2)):
            length = block["sides"][side]["length"]
            span = length - 2 * ring - 2.0
            if span > 4.0:
                self.place(props.make_fence(span), block, side, length / 2, ring + 0.6, lay.GRASS_Z,
                           "surface_obstacle_fence", gap=0.2)
        for _ in range(self.rng.randint(3, 6)):
            for _attempt in range(20):
                x = self.rng.uniform(ix0 + 2.5, ix1 - 2.5)
                y = self.rng.uniform(iy0 + 2.5, iy1 - 2.5)
                if self.add_tree(x, y, lay.GRASS_Z, prefix="surface_obstacle_tree"):
                    break

    def add_construction_site(self, block):
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        ring = lay.CURB_WIDTH + block["sidewalk_w"]
        for side in lay.SIDES:
            length = block["sides"][side]["length"]
            span = length - 2 * ring - 0.8
            self.place(props.make_hoarding(span), block, side, length / 2, ring + 0.4, lay.GRASS_Z,
                       "surface_obstacle_hoarding", reserve=False)
        for _ in range(self.rng.randint(3, 5)):
            x = self.rng.uniform(ix0 + 3.0, ix1 - 3.0)
            y = self.rng.uniform(iy0 + 3.0, iy1 - 3.0)
            prop = props.make_crate(self.rng.uniform(1.0, 3.0), self.rng.uniform(1.0, 2.5), self.rng.uniform(0.6, 1.8))
            self.place_world(prop, x, y, lay.GRASS_Z, self.rng.uniform(0, math.pi), "surface_obstacle_crate", gap=0.5)

    # ---- buildings -----------------------------------------------------

    def build_buildings(self, block):
        plaza = block["variant"] == "plaza"
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        if plaza:
            nx, ny = self.rng.choice((1, 2)), self.rng.choice((1, 2))
            heights = (14.0, 26.0)
            facades = ("facade_light", "facade_gray", "facade_beige")
        else:
            nx, ny = 3, 3
            heights = (7.0, 11.0)
            facades = ("facade_brick", "facade_beige", "facade_light")
        alley = 3.0
        cw = (ix1 - ix0 - alley * (nx - 1)) / nx
        ch = (iy1 - iy0 - alley * (ny - 1)) / ny
        for cx in range(nx):
            for cy in range(ny):
                if nx == 3 and ny == 3 and cx == 1 and cy == 1:
                    continue  # courtyard
                x0 = ix0 + cx * (cw + alley)
                y0 = iy0 + cy * (ch + alley)
                rect = (x0, x0 + cw, y0, y0 + ch)
                faces = []
                if cy == 0:
                    faces.append("south")
                if cy == ny - 1:
                    faces.append("north")
                if cx == 0:
                    faces.append("west")
                if cx == nx - 1:
                    faces.append("east")
                height = self.rng.uniform(*heights)
                self.build_building(block, rect, block["sidewalk_z"], height, self.rng.choice(facades), plaza, faces,
                                    entrance=bool(block.get("extras")))

    def build_building(self, block, rect, z, height, facade, storefront, street_faces, entrance=False):
        """``entrance``: doors sit on a porch reached by steps and (usually) a ramp (new blocks only)."""
        x0, x1, y0, y1 = rect
        floors = max(2, int(height / FLOOR_H))
        height = floors * FLOOR_H
        name = self.next_name(f"surface_building_{block['id']}")
        protrude = 0.04

        def face_runs():
            # (face, along-axis start, end, fixed coordinate, outward sign, axis)
            return [
                ("south", x0, x1, y0, -1.0, "x"),
                ("north", x0, x1, y1, 1.0, "x"),
                ("west", y0, y1, x0, -1.0, "y"),
                ("east", y0, y1, x1, 1.0, "y"),
            ]

        def panel(bm, axis, fixed, sign, a0, a1, zb, zt, out, index):
            if axis == "x":
                f0, f1 = sorted((fixed, fixed + sign * out))
                bm_box(bm, a0, a1, f0, f1, zb, zt, material_index=index)
            else:
                f0, f1 = sorted((fixed, fixed + sign * out))
                bm_box(bm, f0, f1, a0, a1, zb, zt, material_index=index)

        def builder(bm):
            bm_box(bm, x0, x1, y0, y1, z, z + height)
            bm_box(bm, x0 - 0.15, x1 + 0.15, y0 - 0.15, y1 + 0.15, z + height, z + height + 0.35, material_index=2)
            for face, a0, a1, fixed, sign, axis in face_runs():
                length = a1 - a0
                columns = max(1, int(length / 3.0))
                pitch = length / columns
                first_floor = 1 if (storefront and face in street_faces) else 0
                if first_floor:
                    panel(bm, axis, fixed, sign, a0 + 0.5, a1 - 0.5, z + 0.3, z + 2.8, 0.03, 1)
                for floor in range(first_floor, floors):
                    zb = z + floor * FLOOR_H + 0.9
                    for col in range(columns):
                        mid = a0 + (col + 0.5) * pitch
                        if floor == 0 and face in street_faces and col == columns // 2:
                            continue  # door goes here
                        panel(bm, axis, fixed, sign, mid - 0.65, mid + 0.65, zb, zb + 1.5, protrude, 1)

        building = add_bmesh_object(name, "building", builder, [self.mat[facade], self.mat["glass"], self.mat["roof"]],
                                    self.coll["Buildings"], category="building")
        building["height"] = height

        for face, a0, a1, fixed, sign, axis in face_runs():
            if face not in street_faces:
                continue
            length = a1 - a0
            columns = max(1, int(length / 3.0))
            mid = a0 + (columns // 2 + 0.5) * (length / columns)
            dz = lay.PORCH_RISE if entrance else 0.0

            def door_builder(bm, axis=axis, fixed=fixed, sign=sign, mid=mid, dz=dz):
                panel(bm, axis, fixed, sign, mid - 0.6, mid + 0.6, z + dz, z + dz + 2.3, 0.07, 0)

            door = add_bmesh_object(self.next_name(f"surface_door_{block['id']}"), "building", door_builder,
                                    [self.mat["door"]], self.coll["Buildings"], category="door")
            door["building"] = building.name
            if axis == "x":
                door["front"] = (mid, fixed + sign * 1.0)
                door_xy, facing = (mid, fixed), (0.0, sign)
            else:
                door["front"] = (fixed + sign * 1.0, mid)
                door_xy, facing = (fixed, mid), (sign, 0.0)
            if entrance:
                self.add_entrance(door, door_xy, facing, z, ramp=self.rng.random() < 0.75)

    # ---- surroundings --------------------------------------------------

    def add_background_buildings(self):
        """A row of buildings behind the outer sidewalk so views never look out at an empty edge."""
        rx0, rx1, ry0, ry1 = self.layout["road_extent"]
        front = lay.CURB_WIDTH + lay.OUTER_SIDEWALK_W + lay.OUTER_SETBACK
        fx0, fx1, fy0, fy1 = rx0 - front, rx1 + front, ry0 - front, ry1 + front
        outer = {"id": "outer"}
        facades = ("facade_light", "facade_brick", "facade_gray", "facade_beige")

        def row(start, stop, place):
            pos = start
            while pos < stop - 6.0:
                length = min(self.rng.uniform(12.0, 24.0), stop - pos)
                depth = self.rng.uniform(10.0, 14.0)
                rect, face = place(pos, pos + length, depth)
                self.build_building(outer, rect, lay.GRASS_Z, self.rng.uniform(8.0, 22.0),
                                    self.rng.choice(facades), False, [face])
                self.reserved.append(rect)
                pos += length + self.rng.uniform(2.5, 5.0)

        row(fx0, fx1, lambda a, b, d: ((a, b, fy0 - d, fy0), "north"))
        row(fx0, fx1, lambda a, b, d: ((a, b, fy1, fy1 + d), "south"))
        row(fy0, fy1, lambda a, b, d: ((fx0 - d, fx0, a, b), "east"))
        row(fy0, fy1, lambda a, b, d: ((fx1, fx1 + d, a, b), "west"))

    def add_margin_trees(self):
        rx0, rx1, ry0, ry1 = self.layout["road_extent"]
        ex0, ex1, ey0, ey1 = self.layout["extent"]
        keep_out = lay.CURB_WIDTH + lay.OUTER_SIDEWALK_W + 1.5
        placed = 0
        attempts = 0
        while placed < 90 and attempts < 2000:
            attempts += 1
            x = self.rng.uniform(ex0 + 2.0, ex1 - 2.0)
            y = self.rng.uniform(ey0 + 2.0, ey1 - 2.0)
            if rx0 - keep_out < x < rx1 + keep_out and ry0 - keep_out < y < ry1 + keep_out:
                continue
            style = self.rng.choice(("round", "round", "tall", "conifer"))
            if self.add_tree(x, y, lay.GRASS_Z, prefix="surface_obstacle_tree_margin", canopy=(1.0, 1.8), reserve_gap=0.6,
                             style=style):
                placed += 1

    def build(self):
        self.build_ground()
        self.build_center_lines()
        blocks = list(self.layout["blocks"].values())
        core = [b for b in blocks if not b.get("new")]
        new = [b for b in blocks if b.get("new")]
        # Core blocks first, with the original random stream, so they come out exactly as in Phase 1.
        self.build_crosswalks(core=True)
        self.build_blocks(core)
        core_rng, self.rng = self.rng, random.Random(self.layout["seed"] * 7919 + 2)
        self.build_crosswalks(core=False)
        self.build_blocks(new)
        self.rng = core_rng
        self.add_traffic()
        self.add_background_buildings()
        self.add_margin_trees()
