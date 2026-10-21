"""Builders for the Phase 2 block types: neighborhood, park, skate park, scattered objects, entrances.

Mixed into WorldBuilder, so they share its materials, collections, random
stream, and prop-spacing rules. Every walkable surface carries a ground class
(porch, park_path, bridge, court, skate_area, driveway, ...) so the navmap and
planner handle it like the Phase 1 surfaces.
"""

import math

import bmesh

from . import layout as lay
from . import props
from . import props_extra as px
from .materials import CAR_PAINTS, HOUSE_PAINTS
from .meshes import add_bmesh_object, add_box, add_wedge, bm_box

TOP_PATH = lay.GRASS_Z + 0.03   # park paths, playground, plaza paving
TOP_PAD = lay.GRASS_Z + 0.04    # skate pad and court
TOP_WALK = lay.GRASS_Z + 0.02   # driveways and front walks
WALL_H = 2.9


def frame_rect(origin, t, f, u0, u1, v0, v1):
    """World rect (x0, x1, y0, y1) for a box spanning u along ``t`` and v along ``f`` from ``origin``."""
    xs, ys = [], []
    for u in (u0, u1):
        for v in (v0, v1):
            xs.append(origin[0] + u * t[0] + v * f[0])
            ys.append(origin[1] + u * t[1] + v * f[1])
    return (min(xs), max(xs), min(ys), max(ys))


def yaw_facing(dx, dy):
    """Yaw that turns a prop's local -Y (its front) toward direction (dx, dy)."""
    return math.atan2(dx, -dy)


class NewBlocksMixin:
    # ---- entrances: porch + steps + optional ramp -------------------------

    def add_entrance(self, door, door_xy, facing, ground_z, ramp, ramp_side=None, width=2.2, depth=1.4):
        """Raised porch in front of ``door`` with steps outward and, if ``ramp``, a 1:8 ramp along the wall.

        Sets door["front"] to a point on the porch, so a Human path to the door must climb the
        steps or the ramp; door["stairs"]/door["ramp"] name those objects for two-path tasks.
        """
        fx, fy = facing
        t = (-fy, fx)
        rise = lay.PORCH_RISE
        top = ground_z + rise
        coll = self.coll["Buildings"]
        name = door.name.replace("surface_door", "surface_entrance")
        porch = frame_rect(door_xy, t, facing, -width / 2, width / 2, 0.0, depth)
        add_box(f"{name}_porch", "porch", (*porch, lay.BASE_Z, top), self.mat["terrace"], coll)

        def steps_builder(bm):
            for k in range(2):  # two steps up to the porch (third rise is the porch itself)
                rect = frame_rect(door_xy, t, facing, -0.8, 0.8, depth + k * lay.TREAD, depth + (k + 1) * lay.TREAD)
                bm_box(bm, *rect, lay.BASE_Z, ground_z + (2 - k) * lay.RISE)

        stairs = add_bmesh_object(f"{name}_stairs", "stairs", steps_builder, [self.mat["stairs"]], coll, category="stairs")
        door["stairs"] = stairs.name
        door["ramp"] = ""
        rects = [porch, frame_rect(door_xy, t, facing, -1.0, 1.0, depth, depth + 2 * lay.TREAD + 1.2)]
        if ramp:
            side = ramp_side if ramp_side is not None else self.rng.choice((-1.0, 1.0))
            length = rise / lay.RAMP_SLOPE
            u_hi, u_lo = side * width / 2, side * (width / 2 + length)
            rect = frame_rect(door_xy, t, facing, min(u_hi, u_lo), max(u_hi, u_lo), 0.1, 1.3)
            axis = "x" if abs(t[0]) > 0.5 else "y"
            hi = door_xy[0] + u_hi * t[0] if axis == "x" else door_xy[1] + u_hi * t[1]
            lo = door_xy[0] + u_lo * t[0] if axis == "x" else door_xy[1] + u_lo * t[1]
            a0, a1 = (rect[2], rect[3]) if axis == "x" else (rect[0], rect[1])
            ramp_obj = add_wedge(f"{name}_ramp", "ramp", axis, hi, lo, a0, a1, top, ground_z, lay.BASE_Z,
                                 self.mat["ramp"], coll, "ramp")
            door["ramp"] = ramp_obj.name
            rects.append(rect)
        goal = (door_xy[0] + fx * depth * 0.55, door_xy[1] + fy * depth * 0.55)
        door["front"] = goal
        door["porch_z"] = top
        for rect in rects:
            self.reserved.append(rect)
        self.layout.setdefault("entrances", []).append({
            "door": door.name, "goal": goal, "z": top, "stairs": stairs.name, "ramp": door["ramp"],
            "facing": facing, "xy": door_xy,
        })

    # ---- neighborhood ------------------------------------------------------

    def build_neighborhood(self, block):
        g = lay.GRASS_Z
        blocks = self.coll["Blocks"]
        for index, lot in enumerate(block["lots"]):
            south = lot["side"] == "south"
            sign = 1.0 if south else -1.0
            edge = block["iy0"] if south else block["iy1"]
            facing = (0.0, -sign)
            width = lot["x1"] - lot["x0"]
            house_w = min(lot["house_w"], width - 4.6)
            if lot["driveway_left"]:
                drive = (lot["x0"] + 0.5, lot["x0"] + 3.3)
                hx0 = lot["x0"] + 3.8
            else:
                drive = (lot["x1"] - 3.3, lot["x1"] - 0.5)
                hx0 = lot["x1"] - 3.8 - house_w
            hx1 = hx0 + house_w
            front_y = edge + sign * lot["setback"]
            back_y = front_y + sign * lot["house_d"]
            hy0, hy1 = sorted((front_y, back_y))
            floor_z = g + lay.PORCH_RISE
            top_z = floor_z + lot["floors"] * WALL_H
            color = lot["color"]
            # Door toward the side away from the driveway; the ramp runs away from the driveway too.
            toward_drive = -1.0 if lot["driveway_left"] else 1.0
            door_x = (hx0 + hx1) / 2 - toward_drive * house_w * 0.18
            house = self.build_house(block, (hx0, hx1, hy0, hy1), g, floor_z, top_z, color, front_y, sign, door_x)
            door = self.add_house_door(block, house, door_x, front_y, sign, floor_z, color)
            self.add_entrance(door, (door_x, front_y), facing, g, ramp=lot["ramp"], ramp_side=-toward_drive)
            self.reserved.append((hx0, hx1, hy0, hy1))

            walk_end = front_y - sign * (1.4 + 2 * lay.TREAD)
            wy0, wy1 = sorted((edge, walk_end))
            add_box(f"surface_sidewalk_{block['id']}_walk{index}", "sidewalk",
                    (door_x - 0.6, door_x + 0.6, wy0, wy1, lay.BASE_Z, TOP_WALK), self.mat["sidewalk"], blocks)
            dy0, dy1 = sorted((edge, front_y + sign * lot["house_d"] * 0.6))
            add_box(f"surface_driveway_{block['id']}_{index}", "driveway",
                    (drive[0], drive[1], dy0, dy1, lay.BASE_Z, TOP_WALK), self.mat["concrete"], blocks)
            self.reserved.append((door_x - 0.6, door_x + 0.6, wy0, wy1))
            dcx = (drive[0] + drive[1]) / 2
            if lot["car"]:
                paint = self.rng.choice(CAR_PAINTS)
                car = self.place_world(props.make_car(paint), dcx, front_y + sign * 0.5, TOP_WALK,
                                       math.pi / 2 if self.rng.random() < 0.5 else -math.pi / 2,
                                       "surface_obstacle_car_driveway", gap=0.1)
                if car is not None:
                    car["color_name"] = paint.split("_", 1)[1]
            self.reserved.append((drive[0], drive[1], dy0, dy1))
            if lot["mailbox"]:
                self.place_world(px.make_mailbox(), door_x - toward_drive * 1.1, edge + sign * 0.6, g, 0.0,
                                 "surface_obstacle_mailbox", gap=0.1)
            if lot["bin"]:
                self.place_world(px.make_wheelie_bin(), drive[0] - 0.5 if lot["driveway_left"] else drive[1] + 0.5,
                                 edge + sign * 0.7, g, 0.0, "surface_obstacle_wheelie_bin", gap=0.1)
            if lot["hedge"] and abs(lot["x0"] - block["ix0"]) > 0.01:  # hedge between this lot and the previous one
                length = lot["setback"] - 1.6
                if length > 1.0:
                    self.place_world(px.make_hedge(length), lot["x0"], edge + sign * (1.2 + length / 2), g,
                                     math.pi / 2, "surface_obstacle_hedge", gap=0.05, collection="Vegetation")
            bx, by = (hx0 + hx1) / 2, back_y + sign * 4.0
            if block["iy0"] + 2 < by < block["iy1"] - 2:
                self.add_tree(bx, by, g, "surface_obstacle_tree", style=self.rng.choice(("round", "tall", "conifer")))
        # Side lawns (east/west) get bushes and trees.
        for side in ("east", "west"):
            length = block["sides"][side]["length"]
            ring = lay.CURB_WIDTH + block["sidewalk_w"]
            for _ in range(self.rng.randint(2, 4)):
                s = self.rng.uniform(ring + 6.0, length - ring - 6.0)
                x, y = lay.side_point(block, side, s, ring + self.rng.uniform(1.2, 3.0))
                if self.rng.random() < 0.5:
                    self.add_bush(x, y, g)
                else:
                    self.add_tree(x, y, g, "surface_obstacle_tree", style=self.rng.choice(("round", "tall")))

    def build_house(self, block, rect, ground_z, floor_z, top_z, color, front_y, sign, door_x):
        x0, x1, y0, y1 = rect
        roof_h = self.rng.uniform(1.6, 2.4)
        ym = (y0 + y1) / 2

        def builder(bm):
            bm_box(bm, x0, x1, y0, y1, ground_z, top_z)
            o = 0.35
            v = [bm.verts.new(co) for co in (
                (x0 - o, y0 - o, top_z), (x1 + o, y0 - o, top_z), (x1 + o, y1 + o, top_z), (x0 - o, y1 + o, top_z),
                (x0 - o, ym, top_z + roof_h), (x1 + o, ym, top_z + roof_h))]
            roof_faces = [bm.faces.new((v[0], v[1], v[5], v[4])), bm.faces.new((v[2], v[3], v[4], v[5])),
                          bm.faces.new((v[0], v[4], v[3])), bm.faces.new((v[1], v[2], v[5])),
                          bm.faces.new((v[0], v[3], v[2], v[1]))]
            for face in roof_faces:
                face.material_index = 2
            bmesh.ops.recalc_face_normals(bm, faces=roof_faces)
            floors = max(1, round((top_z - floor_z) / WALL_H))
            fy = front_y
            for floor in range(floors):
                zb = floor_z + floor * WALL_H + 0.8
                for wx in (x0 + (x1 - x0) * 0.22, x1 - (x1 - x0) * 0.22):
                    if floor == 0 and abs(wx - door_x) < 1.4:
                        continue
                    a, b = sorted((fy, fy - sign * 0.04))
                    bm_box(bm, wx - 0.6, wx + 0.6, a, b, zb, zb + 1.3, material_index=1)
                for wx in (x0, x1):
                    a, b = sorted((wx, wx + (0.04 if wx == x1 else -0.04)))
                    bm_box(bm, a, b, ym - 0.6, ym + 0.6, zb, zb + 1.3, material_index=1)

        house = add_bmesh_object(self.next_name(f"surface_house_{block['id']}"), "building", builder,
                                 [self.mat[HOUSE_PAINTS[color]], self.mat["glass"], self.mat["roof_tile"]],
                                 self.coll["Buildings"], category="house")
        house["house_color"] = color
        return house

    def add_house_door(self, block, house, door_x, front_y, sign, floor_z, color):
        def builder(bm):
            a, b = sorted((front_y, front_y - sign * 0.07))
            bm_box(bm, door_x - 0.5, door_x + 0.5, a, b, floor_z, floor_z + 2.2)

        door = add_bmesh_object(self.next_name(f"surface_door_{block['id']}"), "building", builder,
                                [self.mat["door"]], self.coll["Buildings"], category="door")
        door["building"] = house.name
        door["house_color"] = color
        return door

    # ---- park --------------------------------------------------------------

    def build_park(self, block):
        pk = block["park"]
        w = pk["path_w"]
        g = lay.GRASS_Z
        br = pk["bridge"]
        blocks = self.coll["Blocks"]

        def seg_rects(seg):
            x0, y0, x1, y1 = seg
            if abs(x1 - x0) > abs(y1 - y0):  # along X
                a, b = sorted((x0, x1))
                pieces = [(a - w / 2, b + w / 2)]
                if br["axis"] == "x" and abs(y0 - br["y"]) < 0.01 and a < br["x"] < b:
                    pieces = [(a - w / 2, br["x"] - br["half_len"] + 0.1), (br["x"] + br["half_len"] - 0.1, b + w / 2)]
                return [(p0, p1, y0 - w / 2, y0 + w / 2) for p0, p1 in pieces if p1 > p0]
            a, b = sorted((y0, y1))
            pieces = [(a - w / 2, b + w / 2)]
            if br["axis"] == "y" and abs(x0 - br["x"]) < 0.01 and a < br["y"] < b:
                pieces = [(a - w / 2, br["y"] - br["half_len"] + 0.1), (br["y"] + br["half_len"] - 0.1, b + w / 2)]
            return [(x0 - w / 2, x0 + w / 2, p0, p1) for p0, p1 in pieces if p1 > p0]

        rects = [r for seg in pk["segments"] for r in seg_rects(seg)] + [pk["plaza"]]

        def path_builder(bm):
            for rect in rects:
                bm_box(bm, *rect, lay.BASE_Z, TOP_PATH)

        add_bmesh_object(f"surface_park_path_{block['id']}", "park_path", path_builder, [self.mat["park_path"]], blocks)
        for rect in rects:
            self.reserved.append((rect[0] - 0.2, rect[1] + 0.2, rect[2] - 0.2, rect[3] + 0.2))

        self.build_pond_and_bridge(block, pk, w)

        cx = (pk["plaza"][0] + pk["plaza"][1]) / 2
        cy = (pk["plaza"][2] + pk["plaza"][3]) / 2
        half = (pk["plaza"][1] - pk["plaza"][0]) / 2
        self.place_world(px.make_fountain(), cx, cy, TOP_PATH, 0.0, "surface_obstacle_fountain", reserve=False)
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            # Benches at the plaza edge, beside the spoke, facing the fountain.
            bx, by = cx + dx * (half - 0.7) + dy * 2.6, cy + dy * (half - 0.7) - dx * 2.6
            self.place_world(props.make_bench(), bx, by, TOP_PATH, yaw_facing(-dx, -dy), "surface_obstacle_bench",
                             gap=0.1, reserve=False)
        for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            lx, ly = cx + sx_ * (half - 0.5), cy + sy_ * (half - 0.5)
            if self.place_world(props.make_lamp_pole(), lx, ly, TOP_PATH, 0.0, "surface_obstacle_lamp", reserve=False):
                self.place_world(props.make_lamp_head(), lx, ly, TOP_PATH, yaw_facing(-sx_, -sy_),
                                 "surface_overhead_lamp", reserve=False)
        sx, sy = pk["statue"]
        self.place_world(px.make_statue(), sx, sy, g, self.rng.uniform(0, math.pi), "surface_obstacle_statue", gap=0.3)

        p = pk["playground"]
        add_box(f"surface_playground_{block['id']}", "playground", (*p, lay.BASE_Z, TOP_PATH), self.mat["rubber"], blocks)
        self.reserved.append(p)
        pcx, pcy = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2
        self.place_world(px.make_swings(), pcx, pcy + 1.8, TOP_PATH, 0.0, "surface_obstacle_swings", reserve=False)
        self.place_world(px.make_slide(), pcx - 1.0, pcy - 2.2, TOP_PATH, 0.0, "surface_obstacle_slide", reserve=False)

        q = pk["picnic"]
        for k in range(3):
            tx = q[0] + (k + 0.5) * (q[1] - q[0]) / 3
            ty = (q[2] + q[3]) / 2 + self.rng.uniform(-2.0, 2.0)
            self.place_world(px.make_picnic_table(), tx, ty, g, self.rng.choice((0.0, math.pi / 2)),
                             "surface_obstacle_picnic_table", gap=0.4)

        self.add_path_furniture(block, pk, w)
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        for _ in range(self.rng.randint(18, 26)):
            for _attempt in range(20):
                x, y = self.rng.uniform(ix0 + 1.5, ix1 - 1.5), self.rng.uniform(iy0 + 1.5, iy1 - 1.5)
                style = self.rng.choice(("round", "round", "tall", "conifer"))
                if self.add_tree(x, y, g, "surface_obstacle_tree", style=style):
                    break
        for _ in range(self.rng.randint(12, 18)):
            for _attempt in range(20):
                x, y = self.rng.uniform(ix0 + 1.0, ix1 - 1.0), self.rng.uniform(iy0 + 1.0, iy1 - 1.0)
                if self.add_bush(x, y, g):
                    break

    def build_pond_and_bridge(self, block, pk, w):
        pond, br = pk["pond"], pk["bridge"]
        blocks = self.coll["Blocks"]

        def water_builder(bm):
            geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=36, radius1=1.0, radius2=1.0, depth=1.0)
            for vert in geom["verts"]:
                vert.co.x = pond["x"] + vert.co.x * pond["rx"]
                vert.co.y = pond["y"] + vert.co.y * pond["ry"]
                vert.co.z = lay.BASE_Z if vert.co.z < 0 else lay.GRASS_Z + 0.01

        add_bmesh_object(f"surface_water_{block['id']}", "water", water_builder, [self.mat["water"]], blocks)
        self.reserved.append((pond["x"] - pond["rx"] - 0.4, pond["x"] + pond["rx"] + 0.4,
                              pond["y"] - pond["ry"] - 0.4, pond["y"] + pond["ry"] + 0.4))

        def rim_builder(bm):
            count = 30
            for k in range(count):
                a = 2 * math.pi * k / count
                x = pond["x"] + math.cos(a) * (pond["rx"] + 0.2)
                y = pond["y"] + math.sin(a) * (pond["ry"] + 0.2)
                across = abs(y - br["y"]) if br["axis"] == "x" else abs(x - br["x"])
                if across < br["half_w"] + 0.3:
                    continue  # leave the bridge corridor open
                bm_box(bm, x - 0.25, x + 0.25, y - 0.25, y + 0.25, lay.GRASS_Z, lay.GRASS_Z + 0.18)

        rim = add_bmesh_object(f"surface_obstacle_pond_edge_{block['id']}", "obstacle", rim_builder, [self.mat["stone"]],
                               self.coll["Props"], category="ledge")
        rim["qa_skip"] = True  # a ring: the ground under its center is the pond, not the stones

        deck_top = TOP_PATH + 0.1
        if br["axis"] == "x":
            rect = (br["x"] - br["half_len"], br["x"] + br["half_len"], br["y"] - br["half_w"], br["y"] + br["half_w"])
            ends = ((rect[0] - 1.5, br["y"]), (rect[1] + 1.5, br["y"]))
            rail_yaw = 0.0
            rails = [(br["x"], br["y"] - br["half_w"] + 0.06), (br["x"], br["y"] + br["half_w"] - 0.06)]
        else:
            rect = (br["x"] - br["half_w"], br["x"] + br["half_w"], br["y"] - br["half_len"], br["y"] + br["half_len"])
            ends = ((br["x"], rect[2] - 1.5), (br["x"], rect[3] + 1.5))
            rail_yaw = math.pi / 2
            rails = [(br["x"] - br["half_w"] + 0.06, br["y"]), (br["x"] + br["half_w"] - 0.06, br["y"])]
        bridge = add_box(f"surface_bridge_{block['id']}", "bridge", (*rect, -0.2, deck_top), self.mat["wood"], blocks,
                         category="bridge")
        bridge["end_a"], bridge["end_b"] = ends
        for rx, ry in rails:
            self.place_world(props.make_fence(2 * br["half_len"] - 0.2, height=0.95, spacing=1.2), rx, ry, deck_top,
                             rail_yaw, "surface_obstacle_bridge_rail", reserve=False)

    def add_path_furniture(self, block, pk, w):
        """Benches (some with a trash can) and lamps beside the spokes and the loop."""
        for index, (x0, y0, x1, y1) in enumerate(pk["segments"]):
            length = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            nx, ny = -uy, ux
            loop = index >= 4
            step = 18.0 if loop else 11.0
            s = 6.0 + self.rng.uniform(0.0, 3.0)
            k = 0
            while s < length - 5.0:
                side = 1.0 if k % 2 == 0 else -1.0
                bx, by = x0 + ux * s + nx * side * (w / 2 + 0.55), y0 + uy * s + ny * side * (w / 2 + 0.55)
                if loop:
                    if self.place_world(props.make_lamp_pole(), bx, by, lay.GRASS_Z, 0.0, "surface_obstacle_lamp", gap=0.2):
                        self.place_world(props.make_lamp_head(), bx, by, lay.GRASS_Z,
                                         yaw_facing(-nx * side, -ny * side), "surface_overhead_lamp", reserve=False)
                else:
                    if self.place_world(props.make_bench(), bx, by, lay.GRASS_Z, yaw_facing(-nx * side, -ny * side),
                                        "surface_obstacle_bench", gap=0.2) and k % 2 == 0:
                        self.place_world(props.make_trash_can(), bx + ux * 1.4, by + uy * 1.4, lay.GRASS_Z, 0.0,
                                         "surface_obstacle_trash_can", gap=0.1)
                s += step
                k += 1

    # ---- skate park and court ----------------------------------------------

    def build_skate(self, block):
        sk = block["skate"]
        blocks = self.coll["Blocks"]
        add_box(f"surface_skate_area_{block['id']}", "skate_area", (*sk["pad"], lay.BASE_Z, TOP_PAD), self.mat["concrete"], blocks)
        c = sk["court"]
        add_box(f"surface_court_{block['id']}", "court", (*c, lay.BASE_Z, TOP_PAD), self.mat["court"], blocks)
        self.reserved.append(sk["pad"])
        self.reserved.append(c)
        cx0, cx1, cy0, cy1 = c[0] + 1.5, c[1] - 1.5, c[2] + 1.5, c[3] - 1.5
        long_x = (cx1 - cx0) > (cy1 - cy0)

        def lines_builder(bm):
            z0, z1 = TOP_PAD, TOP_PAD + 0.004
            lw = 0.06
            bm_box(bm, cx0, cx1, cy0, cy0 + lw, z0, z1)
            bm_box(bm, cx0, cx1, cy1 - lw, cy1, z0, z1)
            bm_box(bm, cx0, cx0 + lw, cy0, cy1, z0, z1)
            bm_box(bm, cx1 - lw, cx1, cy0, cy1, z0, z1)
            if long_x:
                mx = (cx0 + cx1) / 2
                bm_box(bm, mx - lw / 2, mx + lw / 2, cy0, cy1, z0, z1)
            else:
                my = (cy0 + cy1) / 2
                bm_box(bm, cx0, cx1, my - lw / 2, my + lw / 2, z0, z1)

        add_bmesh_object(f"surface_court_lines_{block['id']}", "court", lines_builder, [self.mat["court_line"]], blocks)
        if long_x:
            hoops = [(cx0 + 0.4, (cy0 + cy1) / 2, 1.0, 0.0), (cx1 - 0.4, (cy0 + cy1) / 2, -1.0, 0.0)]
        else:
            hoops = [((cx0 + cx1) / 2, cy0 + 0.4, 0.0, 1.0), ((cx0 + cx1) / 2, cy1 - 0.4, 0.0, -1.0)]
        for hx, hy, dx, dy in hoops:
            # The hoop's backboard and rim reach toward local +Y, i.e. into the court.
            self.place_world(px.make_basketball_hoop(), hx, hy, TOP_PAD, math.atan2(-dx, dy), "surface_obstacle_hoop",
                             reserve=False)
        for kind_feature in sk["features"]:
            self.add_skate_feature(block, kind_feature)
        # Benches and a bike rack beside the court; trees around the edge of the block.
        self.place_world(props.make_bench(), (c[0] + c[1]) / 2, c[2] + 0.6 if not long_x else (c[2] + c[3]) / 2,
                         TOP_PAD, 0.0, "surface_obstacle_bench", gap=0.1, reserve=False)
        ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
        for _ in range(8):
            for _attempt in range(20):
                x, y = self.rng.uniform(ix0 + 1.0, ix1 - 1.0), self.rng.uniform(iy0 + 1.0, iy1 - 1.0)
                if self.add_tree(x, y, lay.GRASS_Z, "surface_obstacle_tree", style=self.rng.choice(("round", "tall"))):
                    break

    def add_skate_feature(self, block, feat):
        x, y, yaw = feat["x"], feat["y"], feat["yaw"]
        kind = feat["kind"]
        if kind == "quarter_pipe":
            self.place_world(px.make_quarter_pipe(), x, y, TOP_PAD, yaw, "surface_obstacle_quarter_pipe", reserve=False)
        elif kind == "funbox":
            self.place_world(px.make_funbox(), x, y, TOP_PAD, yaw, "surface_obstacle_funbox", reserve=False)
        elif kind == "rail":
            self.place_world(px.make_rail(), x, y, TOP_PAD, yaw, "surface_obstacle_rail", reserve=False)
        elif kind == "ledge":
            self.place_world(px.make_ledge(), x, y, TOP_PAD, yaw, "surface_obstacle_ledge", reserve=False)
        elif kind == "stair_set":
            self.build_stair_set(block, x, y, yaw)

    def build_stair_set(self, block, x, y, yaw):
        """Raised platform with steps on one side and a 1:8 bank on the other (two valid ways up)."""
        f = (round(math.cos(yaw)), round(math.sin(yaw)))
        t = (-f[1], f[0])
        size = 3.0
        steps = 4
        top = TOP_PAD + steps * lay.RISE
        coll = self.coll["Blocks"]
        name = f"surface_platform_{block['id']}_{self.counter.get('platform', 0)}"
        self.counter["platform"] = self.counter.get("platform", 0) + 1
        center = (x, y)
        plat = frame_rect(center, t, f, -size / 2, size / 2, -size / 2, size / 2)
        platform = add_box(name, "skate_area", (*plat, lay.BASE_Z, top), self.mat["concrete"], coll)

        def steps_builder(bm):
            for k in range(steps - 1):
                rect = frame_rect(center, t, f, -size / 2, size / 2, size / 2 + k * lay.TREAD, size / 2 + (k + 1) * lay.TREAD)
                bm_box(bm, *rect, lay.BASE_Z, TOP_PAD + (steps - 1 - k) * lay.RISE)

        stairs = add_bmesh_object(f"{name}_stairs", "stairs", steps_builder, [self.mat["stairs"]], coll, category="stairs")
        length = steps * lay.RISE / lay.RAMP_SLOPE
        bank = frame_rect(center, t, f, -size / 2, size / 2, -size / 2 - length, -size / 2)
        axis = "x" if abs(f[0]) > 0.5 else "y"
        hi = (x if axis == "x" else y) - size / 2 * (f[0] if axis == "x" else f[1])
        lo = (x if axis == "x" else y) - (size / 2 + length) * (f[0] if axis == "x" else f[1])
        a0, a1 = (bank[2], bank[3]) if axis == "x" else (bank[0], bank[1])
        ramp = add_wedge(f"{name}_ramp", "ramp", axis, hi, lo, a0, a1, top, TOP_PAD, lay.BASE_Z, self.mat["ramp"], coll, "ramp")
        # Handrail along one edge of the steps.
        rx = x + t[0] * (size / 2 - 0.1) + f[0] * (size / 2 + 0.45)
        ry = y + t[1] * (size / 2 - 0.1) + f[1] * (size / 2 + 0.45)
        self.place_world(px.make_rail(1.2, 0.9), rx, ry, TOP_PAD + 0.3, math.atan2(f[1], f[0]),
                         "surface_obstacle_handrail", reserve=False)
        self.layout.setdefault("platforms", []).append({
            "name": platform.name, "goal": center, "z": top, "stairs": stairs.name, "ramp": ramp.name,
        })

    # ---- scattered objects ---------------------------------------------------

    def add_scattered(self, block):
        """Street objects in open spots: hydrants, signs, bike racks, bus stops, kiosks, food trucks, ..."""
        variant = block["variant"]
        z = block["sidewalk_z"]
        c = lay.CURB_WIDTH
        width = block["sidewalk_w"]
        sides = list(lay.SIDES)
        self.rng.shuffle(sides)
        for side in sides:
            length = block["sides"][side]["length"]
            # Hydrant near the curb and a street sign near a corner on every side.
            self.place(px.make_fire_hydrant(), block, side, self.rng.uniform(10.0, length - 10.0), c + 0.45, z,
                       "surface_obstacle_hydrant", gap=0.3)
            self.place(px.make_signpost(), block, side, self.rng.choice((5.5, length - 5.5)), c + 0.35, z,
                       "surface_obstacle_signpost", gap=0.3)
        if variant in ("street", "plaza", "park"):
            side = sides[0]
            length = block["sides"][side]["length"]
            if width >= 2.6:
                self.place(px.make_bus_stop(), block, side, length * self.rng.uniform(0.35, 0.65), c + 0.75, z,
                           "surface_obstacle_bus_stop", gap=0.4)
        inner_d = c + width - 0.6
        candidates = [
            (px.make_bike_rack(), "surface_obstacle_bike_rack"),
            (px.make_recycling_bins(), "surface_obstacle_recycling"),
            (px.make_vending_machine(), "surface_obstacle_vending"),
        ]
        if variant in ("plaza", "park"):
            candidates.append((px.make_kiosk(self.rng.choice(("paint_green", "paint_blue"))), "surface_obstacle_kiosk"))
            candidates.append((px.make_sculpture(self.rng), "surface_obstacle_sculpture"))
        for prop, prefix in candidates:
            for _attempt in range(10):
                side = self.rng.choice(lay.SIDES)
                length = block["sides"][side]["length"]
                d = min(inner_d, c + width * self.rng.uniform(0.45, 0.8)) if variant == "plaza" else inner_d
                if prop.hy * 2 > width - 1.2:
                    break  # no room on this sidewalk
                if self.place(prop, block, side, self.rng.uniform(9.0, length - 9.0), d, z, prefix, gap=0.4):
                    break
        if variant in ("plaza", "street"):
            # A food truck parked at the curb.
            side = self.rng.choice(lay.SIDES)
            length = block["sides"][side]["length"]
            paint = self.rng.choice(("paint_yellow", "paint_white", "paint_blue"))
            truck = self.place(px.make_food_truck(paint), block, side, length * self.rng.uniform(0.3, 0.7), -1.4,
                               lay.ROAD_Z, "surface_obstacle_food_truck", gap=0.5)
            if truck is not None:
                truck["color_name"] = paint.split("_", 1)[1]
