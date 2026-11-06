"""World 2: build the old town (see oldtown_layout.py) as tagged Blender geometry.

Reuses WorldBuilder's placement helpers (spacing reservations, trees, animals)
and adds the old-town pieces: two ground levels with a retaining wall,
stairs and ramps between them, the main street, perimeter blocks of houses
with shutters, balconies and shopfronts, furnished squares, and people.
Groups, queues and café seating are recorded as social zones, which the
navmap turns into an extra walking cost (paths go around, not through).
"""

import math
import random

import bmesh
import bpy

from . import oldtown_layout as ot
from . import people as ppl
from . import props
from . import props_extra as px
from . import props_oldtown as po
from .build import WorldBuilder
from .materials import CAR_PAINTS, FABRICS, SHUTTERS, STUCCOS
from .meshes import add_bmesh_object, bm_obox, bm_prism, get_collection
from .oldtown_layout import add, intersect, left, lerp, mul, norm, sub

FLOOR_H, GROUND_FLOOR_H = 3.0, 3.4
WINDOW_PITCH = 2.6


def angle_of(u):
    return math.atan2(u[1], u[0])


def circle(center, radius, count=12):
    return [[center[0] + math.cos(2 * math.pi * k / count) * radius,
             center[1] + math.sin(2 * math.pi * k / count) * radius] for k in range(count)]


class OldTownBuilder(WorldBuilder):
    def __init__(self, layout, materials, animals=None):
        super().__init__(layout, materials, animals)
        self.rng = random.Random(layout["seed"] * 7919 + 3)
        self.coll["People"] = get_collection("People", get_collection("World"))
        self.nodes = {tuple(int(v) for v in key.split(",")): tuple(p) for key, p in layout["nodes"].items()}
        self.H = layout["upper_z"]
        self.wall = [tuple(p) for p in layout["wall"]]
        self.wall_gaps = []      # (x0, x1) stretches of the wall top left open for stairs and ramps
        self.faces = layout["blocks"]
        self.people = 0

    # ---- small helpers ---------------------------------------------------

    def wall_y(self, x):
        for (x0, y0), (x1, y1) in zip(self.wall, self.wall[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / ((x1 - x0) or 1e-9)
        return self.wall[0][1] if x < self.wall[0][0] else self.wall[-1][1]

    def level_z(self, x, y):
        return self.H if y > self.wall_y(x) else 0.0

    def frame(self, face, k):
        inset = face["inset"]
        a = tuple(inset[k])
        u = norm(sub(tuple(inset[(k + 1) % 4]), a))
        return a, u, left(u)

    def fpt(self, face, k, s, d):
        a, u, m = self.frame(face, k)
        return add(add(a, mul(u, s)), mul(m, d))

    def edge_len(self, face, k):
        inset = face["inset"]
        return math.hypot(*sub(tuple(inset[(k + 1) % 4]), tuple(inset[k])))

    def neighbour(self, face, k):
        c, r = face["col"], face["row"]
        nc, nr = [(c, r - 1), (c + 1, r), (c, r + 1), (c - 1, r)][k]
        return self.faces.get(f"f{nc}{nr}")

    def add_zone(self, kind, poly):
        self.layout["social_zones"].append({"kind": kind, "poly": [[round(x, 3), round(y, 3)] for x, y in poly]})

    def add_person(self, x, y, z, yaw, pose="standing", look=None, reserve=True, gap=0.15):
        look = look or ppl.random_look(self.rng, child=self.rng.random() < 0.08)
        prop = ppl.make_person(look, pose, stride=self.rng.uniform(-1.0, 1.0))
        obj = self.place_world(prop, x, y, z, yaw, "surface_obstacle_person", gap=gap, reserve=reserve,
                               collection="People")
        if obj is not None:
            obj["color_name"] = look["top"]
            obj["garment"] = "dress" if look.get("dress") else look["garment"]
            obj["pose"] = pose
            self.people += 1
        return obj

    def add_lantern(self, x, y, z, yaw):
        if self.place_world(po.make_lantern_post(), x, y, z, yaw, "surface_obstacle_lamp"):
            self.place_world(po.make_lantern_head(), x, y, z, yaw, "surface_overhead_lamp", reserve=False)

    def railing(self, name, p0, z0, p1, z1, height=1.0):
        """Railing from (p0, z0) to (p1, z1) whose top follows the slope (stairs and ramp edges)."""
        u = norm(sub(p1, p0))
        length = math.hypot(*sub(p1, p0))

        def builder(bm):
            count = max(2, int(length / 1.4) + 1)
            for k in range(count):
                t = k / (count - 1)
                p = lerp(p0, p1, t)
                z = z0 + (z1 - z0) * t
                bm_obox(bm, p, u, -0.035, 0.035, -0.035, 0.035, z, z + height, 1)
            n = left(u)
            poly = [add(p0, mul(n, -0.03)), add(p1, mul(n, -0.03)), add(p1, mul(n, 0.03)), add(p0, mul(n, 0.03))]
            bm_prism(bm, poly, min(z0, z1) + height - 0.06, [z0 + height, z1 + height, z1 + height, z0 + height], 1)

        obj = add_bmesh_object(self.next_name(name), "obstacle", builder, [self.mat["metal"], self.mat["paint_black"]],
                               self.coll["Props"], category="fence")
        obj["qa_skip"] = True  # sloped: its lowest point is not over the ground at its middle
        return obj

    # ---- ground and levels --------------------------------------------------

    def build_ground(self):
        ex0, ex1, ey0, ey1 = self.layout["extent"]
        ground = self.coll["Ground"]
        lower = [(ex0, ey0), (ex1, ey0), (ex1, ey1), (ex0, ey1)]
        add_bmesh_object("surface_cobblestone_lower", "cobblestone", lambda bm: bm_prism(bm, lower, ot.BASE_Z, 0.0),
                         [self.mat["cobblestone"]], ground)

        def upper(bm):
            for (x0, y0), (x1, y1) in zip(self.wall, self.wall[1:]):
                bm_prism(bm, [(x0, y0), (x1, y1), (x1, ey1), (x0, ey1)], ot.BASE_Z, self.H, 0, side_index=1)

        add_bmesh_object("surface_cobblestone_upper", "cobblestone", upper,
                         [self.mat["cobblestone"], self.mat["wall_stone"]], ground)

        for face in self.faces.values():
            if face["square"]:
                z = face["ground_z"]
                quad = [tuple(p) for p in face["inset"]]
                add_bmesh_object(f"surface_plaza_{face['id']}", "plaza", lambda bm, q=quad, z=z: bm_prism(bm, q, z - 0.1, z + 0.02),
                                 [self.mat["pavers"]], self.coll["Blocks"])

    def build_main_street(self):
        road = [tuple(p) for p in self.layout["road"]]
        half = ot.ROAD_W / 2
        lft = ot.offset_polyline(road, half)
        rgt = ot.offset_polyline(road, -half)

        def road_builder(bm):
            for k in range(len(road) - 1):
                bm_prism(bm, [rgt[k], rgt[k + 1], lft[k + 1], lft[k]], ot.BASE_Z, 0.01)

        add_bmesh_object("surface_road_main", "road", road_builder, [self.mat["road"]], self.coll["Ground"])

        # Raised sidewalks between the road and the house fronts of the faces on both sides.
        j = self.layout["main_line"]
        for face in self.faces.values():
            for k, row in ((2, j - 1), (0, j)):
                if face["row"] != row:
                    continue
                a, u, m = self.frame(face, k)
                b = self.fpt(face, k, self.edge_len(face, k), 0.0)
                walk_w = (face["edge_widths"][k] - ot.ROAD_W) / 2
                pa, pb = add(a, mul(m, -walk_w)), add(b, mul(m, -walk_w))
                add_bmesh_object(f"surface_sidewalk_{face['id']}", "sidewalk",
                                 lambda bm, q=[pa, pb, b, a]: bm_prism(bm, q, ot.BASE_Z, ot.SIDEWALK_Z, 0, side_index=1),
                                 [self.mat["sidewalk"], self.mat["curb"]], self.coll["Blocks"])

        # Two zebra crossings, bars parallel to the traffic.
        for seg in (1, 3):
            p0, p1 = road[seg + 1], road[seg + 2]   # road[0] is the extension to the world edge
            u = norm(sub(p1, p0))
            mid = lerp(p0, p1, 0.5)
            n = left(u)

            def zebra(bm, mid=mid, u=u):
                bars = int(ot.ROAD_W / 0.5)
                for b in range(bars):
                    d0 = -half + b * ot.ROAD_W / bars
                    bm_obox(bm, mid, u, -1.6, 1.6, d0, d0 + ot.ROAD_W / bars, 0.012, 0.03, b % 2)

            obj = add_bmesh_object(f"surface_crosswalk_main{seg}", "crosswalk", zebra,
                                   [self.mat["crosswalk"], self.mat["crosswalk_dark"]], self.coll["Markings"],
                                   category="crosswalk")
            ends = (add(mid, mul(n, half + 1.3)), add(mid, mul(n, -half - 1.3)))
            obj["end_a"], obj["end_b"] = ends
            for ex, ey in ends:
                self.keep_clear((ex - 1.8, ex + 1.8, ey - 1.8, ey + 1.8))
            self.keep_clear((mid[0] - 2.2, mid[0] + 2.2, mid[1] - half, mid[1] + half))

        # Parked cars on the south side, a few stopped in the north lane, lanterns on both sidewalks.
        for k in range(1, len(road) - 2):
            p0, p1 = road[k], road[k + 1]
            u = norm(sub(p1, p0))
            length = math.hypot(*sub(p1, p0))
            n = left(u)
            s = 8.0 + self.rng.uniform(0, 2)
            while s < length - 8.0:
                if self.rng.random() < 0.7:
                    p = add(add(p0, mul(u, s)), mul(n, -half + 1.05))
                    paint = self.rng.choice(CAR_PAINTS)
                    car = self.place_world(props.make_car(paint), p[0], p[1], 0.01, angle_of(u), "surface_obstacle_car",
                                           gap=0.3)
                    if car is not None:
                        car["color_name"] = paint.split("_", 1)[1]
                s += self.rng.uniform(5.8, 7.0)
            if k in (2, 4):
                p = add(add(p0, mul(u, length * self.rng.uniform(0.3, 0.7))), mul(n, half * 0.5))
                paint = self.rng.choice(CAR_PAINTS)
                car = self.place_world(props.make_car(paint), p[0], p[1], 0.01, angle_of(u) + math.pi,
                                       "surface_obstacle_car_traffic", gap=0.6, collection="Traffic")
                if car is not None:
                    car["in_traffic"] = True
                    car["color_name"] = paint.split("_", 1)[1]
            for side, d in ((1, half + 0.45), (-1, -half - 0.45)):
                s = 7.0
                while s < length - 7.0:
                    p = add(add(p0, mul(u, s)), mul(n, d))
                    yaw = angle_of(u) if side > 0 else angle_of(u) + math.pi
                    self.add_lantern(p[0], p[1], ot.SIDEWALK_Z, yaw)
                    s += 17.0

    def wall_crossing(self, a, u):
        """Point and wall segment where the line a + t u crosses the wall."""
        for p, q in zip(self.wall, self.wall[1:]):
            w = norm(sub(q, p))
            hit = intersect(a, u, p, w)
            if hit is not None and min(p[0], q[0]) - 0.01 <= hit[0] <= max(p[0], q[0]) + 0.01:
                return hit, w
        return None, None

    def build_wall_links(self):
        """Stairs or a ramp at the top of each linked cross street, up to the promenade."""
        j = self.layout["wall_line"]
        for link in self.layout["wall_links"]:
            i, width = link["i"], link["width"]
            a, b = self.nodes[(i, j - 1)], self.nodes[(i, j)]
            v = norm(sub(b, a))
            hit, w = self.wall_crossing(a, v)
            if hit is None:
                continue
            down = left(w)  # wall runs west to east; its left is the upper town
            down = (-down[0], -down[1])
            side_a = (add(a, mul(left(v), width / 2)), v)
            side_b = (add(a, mul(left(v), -width / 2)), v)
            room = math.hypot(*sub(hit, a)) - ot.H_WIDTH[self.layout["main_line"]] / 2 - 2.0

            def strip(d0, d1):
                p0, p1 = add(hit, mul(down, d0)), add(hit, mul(down, d1))
                return [intersect(p0, w, *side_b), intersect(p0, w, *side_a), intersect(p1, w, *side_a),
                        intersect(p1, w, *side_b)]

            kind = link["kind"]
            run = self.H / ot.RAMP_SLOPE
            if kind == "ramp" and room < run:
                run = max(room, self.H * 6.0)
            name = f"surface_{'ramp' if kind == 'ramp' else 'stairs'}_link{i}"
            if kind == "ramp":
                # strip() is counter-clockwise with its two wall-side corners first: H there, 0 at the foot.
                poly = strip(0.0, run)
                obj = add_bmesh_object(name, "ramp", lambda bm: bm_prism(bm, poly, ot.BASE_Z, [self.H, self.H, 0.0, 0.0], 0, 1),
                                       [self.mat["ramp"], self.mat["wall_stone"]], self.coll["Blocks"], category="ramp")
            else:
                steps = round(self.H / ot.RISE)
                rise = self.H / steps

                def stairs(bm):
                    for k in range(steps):
                        bm_prism(bm, strip(k * ot.TREAD, (k + 1) * ot.TREAD), ot.BASE_Z, self.H - k * rise, 0, 1)

                run = steps * ot.TREAD
                obj = add_bmesh_object(name, "stairs", stairs, [self.mat["stairs"], self.mat["wall_stone"]],
                                       self.coll["Blocks"], category="stairs")
            obj["link"] = i
            foot = add(hit, mul(down, run + 1.0))
            self.keep_clear((min(hit[0], foot[0]) - width / 2, max(hit[0], foot[0]) + width / 2,
                             min(hit[1], foot[1]) - 0.5, max(hit[1], foot[1]) + 0.5))
            self.wall_gaps.append((hit[0] - width / 2 - 0.3, hit[0] + width / 2 + 0.3))

    def build_parapet(self):
        """Low stone wall along the top of the retaining wall where no house stands against it,
        open where stairs and ramps arrive."""
        nav_x0, nav_x1 = self.layout["nav_extent"][:2]
        gaps = list(self.wall_gaps)
        for face in self.faces.values():
            if face["row"] == self.layout["wall_line"] - 1 and not face["square"]:
                xs = (face["inset"][2][0], face["inset"][3][0])
                gaps.append((min(xs) - 0.05, max(xs) + 0.05))

        def builder(bm):
            for p, q in zip(self.wall, self.wall[1:]):
                u = norm(sub(q, p))
                length = math.hypot(*sub(q, p))
                cuts = [(0.0, length)]
                for g0, g1 in gaps + [(-1e9, nav_x0 - 4.0), (nav_x1 + 4.0, 1e9)]:
                    out = []
                    for s0, s1 in cuts:
                        t0 = (g0 - p[0]) / (u[0] or 1e-9)
                        t1 = (g1 - p[0]) / (u[0] or 1e-9)
                        if t1 <= s0 or t0 >= s1:
                            out.append((s0, s1))
                            continue
                        if t0 > s0:
                            out.append((s0, t0))
                        if t1 < s1:
                            out.append((t1, s1))
                    cuts = out
                for s0, s1 in cuts:
                    if s1 - s0 > 0.3:
                        bm_obox(bm, p, u, s0, s1, 0.0, 0.4, self.H - 0.05, self.H + 0.85)

        obj = add_bmesh_object("surface_obstacle_parapet", "obstacle", builder, [self.mat["wall_stone"]],
                               self.coll["Blocks"], category="fence")
        obj["qa_skip"] = True

    # ---- houses ---------------------------------------------------------------

    def build_faces(self):
        j_main, j_wall = self.layout["main_line"], self.layout["wall_line"]
        for face in self.faces.values():
            if face["square"]:
                continue
            edges = [0, 1, 2, 3]
            inset = [tuple(p) for p in face["inset"]]
            short = min(self.edge_len(face, k) for k in range(4))
            depth = min(10.0, max(6.0, short * 0.36))
            lots = ot.face_lots(inset, edges, lambda k: self.rng.uniform(depth * 0.8, depth), (6.0, 10.5), self.rng)
            for lot in lots:
                k = lot["edge"]
                nb = self.neighbour(face, k)
                on_main = (k == 0 and face["row"] == j_main) or (k == 2 and face["row"] == j_main - 1)
                on_promenade = (k == 0 and face["row"] == j_wall) or (k == 2 and face["row"] == j_wall - 1)
                shop = on_main or on_promenade or (nb is not None and nb["square"] is not None)
                if k == 2 and face["row"] == j_wall - 1:
                    # Built into the slope: stands on the lower town, front door on the promenade above.
                    self.build_house(face, lot, shop and self.rng.random() < 0.8, ground_z=self.H, base_z=0.0)
                else:
                    self.build_house(face, lot, shop and self.rng.random() < 0.8)
            # A tree or two in the courtyard (seen over the parapet from the promenade).
            cx = sum(p[0] for p in inset) / 4
            cy = sum(p[1] for p in inset) / 4
            for _ in range(self.rng.randint(1, 2)):
                x, y = cx + self.rng.uniform(-3, 3), cy + self.rng.uniform(-3, 3)
                self.add_tree(x, y, face["ground_z"], "surface_obstacle_tree_yard", style=self.rng.choice(("round", "tall")),
                              reserve_gap=0.3)
        self.build_background()

    def build_house(self, face, lot, shop, ground_z=None, floors=None, base_z=None):
        """``base_z``: foundation height when it differs from the entrance level (houses in the slope)."""
        z = face["ground_z"] if ground_z is None else ground_z
        base = z if base_z is None else base_z
        quad = [tuple(p) for p in lot["quad"]]
        a, b = quad[0], quad[1]
        u = norm(sub(b, a))
        m = left(u)
        length = math.hypot(*sub(b, a))
        floors = floors or self.rng.randint(2, 4)
        height = GROUND_FLOOR_H + (floors - 1) * FLOOR_H
        stucco = self.rng.choice(STUCCOS)
        shutter = self.rng.choice(SHUTTERS)
        cols = max(1, int(length / WINDOW_PITCH))
        pitch = length / cols
        door_col = cols // 2
        flat = self.rng.random() < 0.3

        def builder(bm):
            bm_prism(bm, quad, base - 0.3, z + height, 0)
            top = z + height
            for f in range(1, floors):
                zf = z + GROUND_FLOOR_H + (f - 1) * FLOOR_H
                for c in range(cols):
                    mid = (c + 0.5) * pitch
                    bm_obox(bm, a, u, mid - 0.5, mid + 0.5, -0.03, 0.02, zf + 0.75, zf + 2.25, 1)
                    if pitch > 2.1:
                        for s0, s1 in ((mid - 0.95, mid - 0.52), (mid + 0.52, mid + 0.95)):
                            bm_obox(bm, a, u, s0, s1, -0.07, 0.0, zf + 0.75, zf + 2.25, 2)
                    bm_obox(bm, a, u, mid - 0.62, mid + 0.62, -0.12, 0.0, zf + 0.66, zf + 0.75, 3)
            for c in range(cols):
                if c == door_col:
                    continue
                mid = (c + 0.5) * pitch
                if shop:
                    bm_obox(bm, a, u, mid - pitch / 2 + 0.25, mid + pitch / 2 - 0.25, -0.03, 0.02, z + 0.35, z + 2.7, 1)
                else:
                    bm_obox(bm, a, u, mid - 0.45, mid + 0.45, -0.03, 0.02, z + 1.0, z + 2.3, 1)
            # Plinth and cornice in stone.
            bm_obox(bm, a, u, 0.0, length, -0.06, 0.0, z, z + 0.35, 3)
            bm_obox(bm, a, u, -0.05, length + 0.05, -0.3, 0.0, top - 0.25, top, 3)
            if flat:
                bm_obox(bm, a, u, 0.0, length, 0.0, 0.3, top, top + 0.7, 0)
                return
            # Gable roof with the ridge along the street; the front eave overhangs the street.
            ridge_h = self.rng.uniform(1.4, 2.2)
            c_, d_ = quad[2], quad[3]
            ra, rb = lerp(a, d_, 0.5), lerp(b, c_, 0.5)
            ea, eb = add(a, mul(m, -0.4)), add(b, mul(m, -0.4))
            verts = {key: bm.verts.new(co) for key, co in (
                ("ea", (*ea, top - 0.15)), ("eb", (*eb, top - 0.15)), ("a", (*a, top)), ("b", (*b, top)),
                ("c", (*c_, top)), ("d", (*d_, top)), ("ra", (*ra, top + ridge_h)), ("rb", (*rb, top + ridge_h)))}
            new = [bm.faces.new([verts[k] for k in keys]) for keys in (
                ("ea", "eb", "rb", "ra"), ("d", "ra", "rb", "c"), ("a", "ra", "d"), ("b", "c", "rb"))]
            for face_ in new:
                face_.material_index = 4
            bmesh.ops.recalc_face_normals(bm, faces=new)

        building = add_bmesh_object(self.next_name(f"surface_building_{face['id']}"), "building", builder,
                                    [self.mat[stucco], self.mat["glass"], self.mat[shutter], self.mat["stone"],
                                     self.mat["roof_terracotta"]], self.coll["Buildings"], category="building")
        building["height"] = height

        mid = (door_col + 0.5) * pitch
        door = add_bmesh_object(self.next_name(f"surface_door_{face['id']}"), "building",
                                lambda bm: bm_obox(bm, a, u, mid - 0.55, mid + 0.55, -0.07, 0.0, z, z + 2.35),
                                [self.mat["door"]], self.coll["Buildings"], category="door")
        door["building"] = building.name
        front = add(add(a, mul(u, mid)), mul(m, -1.0))
        door["front"] = front
        out = (-m[0], -m[1])
        if shop:
            fabric = self.rng.choice(list(FABRICS.values()))
            add_bmesh_object(self.next_name("surface_overhead_awning"), "overhead",
                             lambda bm: (bm_obox(bm, a, u, 0.3, length - 0.3, -1.3, 0.0, z + 2.85, z + 2.92),
                                         bm_obox(bm, a, u, 0.3, length - 0.3, -1.33, -1.28, z + 2.6, z + 2.92)),
                             [self.mat[fabric]], self.coll["Buildings"], category="building")
        elif floors >= 3 and self.rng.random() < 0.35:
            zf = z + GROUND_FLOOR_H + FLOOR_H
            bmid = (cols // 2 + 0.5) * pitch
            add_bmesh_object(self.next_name("surface_overhead_balcony"), "overhead",
                             lambda bm: (bm_obox(bm, a, u, bmid - 0.9, bmid + 0.9, -0.8, 0.0, zf - 0.12, zf),
                                         bm_obox(bm, a, u, bmid - 0.9, bmid + 0.9, -0.8, -0.76, zf, zf + 0.95, 1),
                                         bm_obox(bm, a, u, bmid - 0.9, bmid - 0.86, -0.8, 0.0, zf, zf + 0.95, 1),
                                         bm_obox(bm, a, u, bmid + 0.86, bmid + 0.9, -0.8, 0.0, zf, zf + 0.95, 1)),
                             [self.mat["stone"], self.mat["paint_black"]], self.coll["Buildings"], category="building")
        # A flower pot beside some doors.
        if self.rng.random() < 0.35:
            side = self.rng.choice((-1, 1))
            p = add(add(a, mul(u, mid + side * 0.95)), mul(out, 0.35))
            self.place_world(po.make_flower_pot(self.rng.uniform(0.45, 0.6)), p[0], p[1], self.street_z(p, z), 0.0,
                             "surface_obstacle_flower_pot", gap=0.1)
        return building

    def street_z(self, p, face_z):
        """Ground height on the street just outside a house front (raised on the main street's sidewalks)."""
        if face_z > 0.0:
            return face_z
        road = self.layout["road"]
        for p0, p1 in zip(road, road[1:]):
            u = norm(sub(tuple(p1), tuple(p0)))
            rel = sub(p, tuple(p0))
            along = rel[0] * u[0] + rel[1] * u[1]
            across = abs(rel[0] * -u[1] + rel[1] * u[0])
            seg = math.hypot(*sub(tuple(p1), tuple(p0)))
            if -1.0 <= along <= seg + 1.0 and across <= ot.H_WIDTH[self.layout["main_line"]] / 2 + 0.05:
                return ot.SIDEWALK_Z
        return 0.0

    def build_background(self):
        """Houses along the outside of the outer streets, so views never end in empty ground."""
        nx, ny = ot.NX, ot.NY
        runs = []
        for i in range(nx - 1):
            runs.append((self.nodes[(i + 1, 0)], self.nodes[(i, 0)], ("h", i, 0)))           # south, facing north
            runs.append((self.nodes[(i, ny - 1)], self.nodes[(i + 1, ny - 1)], ("h", i, ny - 1)))  # north
        for j in range(ny - 1):
            runs.append((self.nodes[(0, j)], self.nodes[(0, j + 1)], ("v", 0, j)))           # west
            runs.append((self.nodes[(nx - 1, j + 1)], self.nodes[(nx - 1, j)], ("v", nx - 1, j)))  # east
        widths = {tuple(k): w for k, w in ((("h", s["i"], s["j"]), s["width"]) for s in self.layout["streets"]
                                           if s["axis"] == "h")}
        widths.update({("v", s["i"], s["j"]): s["width"] for s in self.layout["streets"] if s["axis"] == "v"})
        road_exits = [self.nodes[(0, self.layout["main_line"])], self.nodes[(nx - 1, self.layout["main_line"])]]
        outer = {"id": "outer", "ground_z": 0.0}
        for p, q, key in runs:
            u = norm(sub(q, p))
            m = left(u)                       # away from the town
            w = widths[key]
            a, b = add(p, mul(m, w / 2)), add(q, mul(m, w / 2))
            length = math.hypot(*sub(b, a))
            s = 0.0
            while s < length - 4.0:
                lot_w = min(self.rng.uniform(7.0, 12.0), length - s)
                quad = [lerp(a, b, s / length), lerp(a, b, (s + lot_w) / length)]
                depth = ot.BACKGROUND_DEPTH
                quad += [add(quad[1], mul(m, depth)), add(quad[0], mul(m, depth))]
                s += lot_w + self.rng.uniform(0.0, 1.5)
                mid = lerp(quad[0], quad[1], 0.5)
                if any(math.hypot(*sub(mid, e)) < ot.ROAD_W + 4.0 for e in road_exits):
                    continue
                zs = {self.level_z(*pt) for pt in quad}
                if len(zs) > 1:
                    continue  # straddles the retaining wall
                lot = {"quad": quad, "edge": 0}   # street corners first; left of the street edge is the lot
                outer["ground_z"] = zs.pop()
                self.build_house(outer, lot, False, floors=self.rng.randint(3, 5))

    # ---- squares ------------------------------------------------------------

    def build_squares(self):
        for face in self.faces.values():
            kind = face["square"]
            if not kind:
                continue
            z = face["ground_z"] + 0.02
            if kind == "stair_square":
                self.build_grand_stairs(face)
                self.furnish_square(face, z, centre="statue", cafe_edge=3, stall_edge=None)
            elif kind == "market_square":
                self.furnish_square(face, z, centre="fountain", cafe_edge=3, stall_edge=1)
            else:
                self.furnish_square(face, z, centre="fountain", cafe_edge=1, stall_edge=None)

    def build_grand_stairs(self, face):
        """Wide stairs from the square up to the promenade, with a ramp along the wall beside them."""
        k = 2                                   # north edge = the retaining wall
        a, u, m = self.frame(face, k)           # u runs east to west along the wall; m points into the square
        length = self.edge_len(face, k)
        steps = round(self.H / ot.RISE)
        rise = self.H / steps
        s0, s1 = length - 10.0, length - 3.0    # stairs near the west end
        ramp_run = min(self.H / ot.RAMP_SLOPE, s0 - 3.5)
        r_hi, r_lo = s0 - 1.5, s0 - 1.5 - ramp_run   # ramp high end next to the stairs
        z0 = face["ground_z"] + 0.02

        def stairs(bm):
            for j in range(steps):
                bm_obox(bm, a, u, s0, s1, j * ot.TREAD, (j + 1) * ot.TREAD, ot.BASE_Z, self.H - j * rise, 0)

        stairs_obj = add_bmesh_object("surface_stairs_square", "stairs", stairs, [self.mat["stairs"]], self.coll["Blocks"],
                                      category="stairs")
        ramp_w = 2.4
        p = lambda s, d: add(add(a, mul(u, s)), mul(m, d))  # noqa: E731
        poly = [p(r_lo, 0.0), p(r_hi, 0.0), p(r_hi, ramp_w), p(r_lo, ramp_w)]   # counter-clockwise (m = left of u)
        tops = [z0, self.H, self.H, z0]
        ramp_obj = add_bmesh_object("surface_ramp_square", "ramp", lambda bm: bm_prism(bm, poly, ot.BASE_Z, tops, 0, 1),
                                    [self.mat["ramp"], self.mat["wall_stone"]], self.coll["Blocks"], category="ramp")
        # Railings on the open sides: along the ramp and down both sides of the stairs.
        self.railing("surface_obstacle_railing", p(r_lo + 1.2, ramp_w), z0 + 1.2 * ot.RAMP_SLOPE, p(r_hi, ramp_w), self.H)
        depth = steps * ot.TREAD
        for s in (s0, s1):
            self.railing("surface_obstacle_railing", p(s, 0.15), self.H, p(s, depth - 0.2), z0 + rise)
        corners = [p(r_lo - 1.0, 0.0), p(s1 + 1.0, 0.0), p(s1 + 1.0, depth + 1.8), p(r_lo - 1.0, depth + 1.8)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        self.keep_clear((min(xs), max(xs), min(ys), max(ys)))
        hi_s = (s0 + r_hi) / 2
        top = add(add(a, mul(u, hi_s)), mul(m, -2.5))   # on the promenade above both
        self.layout["platforms"].append({"kind": "promenade", "stairs": stairs_obj.name, "ramp": ramp_obj.name,
                                         "goal": [round(top[0], 3), round(top[1], 3)], "z": self.H})
        for s_a, s_b in ((r_hi - 0.3, s1 + 0.3),):
            pa, pb = p(s_a, 0.0), p(s_b, 0.0)
            self.wall_gaps.append((min(pa[0], pb[0]), max(pa[0], pb[0])))

    def furnish_square(self, face, z, centre, cafe_edge, stall_edge):
        inset = [tuple(p) for p in face["inset"]]
        cx = sum(p[0] for p in inset) / 4
        cy = sum(p[1] for p in inset) / 4
        if centre == "fountain":
            self.place_world(px.make_fountain(), cx, cy, z, 0.0, "surface_obstacle_fountain", gap=1.0)
        else:
            self.place_world(px.make_statue(), cx, cy, z, self.rng.uniform(0, math.pi), "surface_obstacle_statue", gap=1.0)
        # Lanterns near the corners, trees and benches between them.
        for k in range(4):
            p = lerp(inset[k], (cx, cy), 0.12)
            self.add_lantern(p[0], p[1], z, self.rng.uniform(0, 2 * math.pi))
        for k in range(4):
            if k in (cafe_edge, stall_edge):
                continue
            length = self.edge_len(face, k)
            a, u, _m = self.frame(face, k)
            for s in (length * 0.3, length * 0.7):
                p = self.fpt(face, k, s, 3.0)
                if self.rng.random() < 0.6:
                    self.add_tree(p[0], p[1], z, "surface_obstacle_tree_square", style="round", reserve_gap=0.4)
                q = self.fpt(face, k, s + 2.0, 3.0)
                # Benches face local -Y: turn them to face into the square.
                self.place_world(props.make_bench(), q[0], q[1], z, angle_of(u) + math.pi, "surface_obstacle_bench")
        if cafe_edge is not None:
            self.add_cafe(face, cafe_edge, z)
        if stall_edge is not None:
            self.add_market(face, stall_edge, z)
        # Groups of people talking, and single people crossing the square.
        for _ in range(self.rng.randint(2, 3)):
            for _attempt in range(15):
                gx, gy = cx + self.rng.uniform(-9, 9), cy + self.rng.uniform(-8, 8)
                if self.add_group(gx, gy, z):
                    break
        for _ in range(self.rng.randint(3, 6)):
            x, y = cx + self.rng.uniform(-12, 12), cy + self.rng.uniform(-10, 10)
            self.add_person(x, y, z, self.rng.uniform(0, 2 * math.pi), pose=self.rng.choice(("walking", "standing")))
        self.add_animals_at([(cx + self.rng.uniform(-10, 10), cy + self.rng.uniform(-10, 10), z) for _ in range(2)])

    def add_group(self, x, y, z):
        count = self.rng.randint(2, 4)
        radius = self.rng.uniform(0.55, 0.8)
        rect = (x - radius - 0.4, x + radius + 0.4, y - radius - 0.4, y + radius + 0.4)
        if any(not (rect[1] < o[0] - 0.5 or rect[0] > o[1] + 0.5 or rect[3] < o[2] - 0.5 or rect[2] > o[3] + 0.5)
               for o in self.reserved):
            return False
        start = self.rng.uniform(0, 2 * math.pi)
        for k in range(count):
            a = start + 2 * math.pi * k / count
            self.add_person(x + math.cos(a) * radius, y + math.sin(a) * radius, z, a + math.pi, reserve=False)
        self.keep_clear(rect)
        self.add_zone("group", circle((x, y), radius + 0.9))
        return True

    def add_cafe(self, face, k, z):
        """Tables with chairs and umbrellas along one edge; some chairs taken."""
        length = self.edge_len(face, k)
        a, u, m = self.frame(face, k)
        s0, s1 = 4.0, min(length - 4.0, 4.0 + self.rng.uniform(11.0, 16.0))
        rows = (2.2, 4.6)
        s = s0
        while s <= s1:
            for d in rows:
                p = self.fpt(face, k, s, d)
                chairs = self.rng.choice((2, 4, 4))
                yaw = angle_of(u)
                table = self.place_world(po.make_cafe_set(chairs), p[0], p[1], z, yaw, "surface_obstacle_cafe_table",
                                         gap=0.2)
                if table is None:
                    continue
                if self.rng.random() < 0.7:
                    fabric = self.rng.choice(list(FABRICS.values()))
                    self.place_world(po.make_umbrella(fabric), p[0], p[1], z, yaw, "surface_overhead_umbrella",
                                     reserve=False)
                c, sn = math.cos(yaw), math.sin(yaw)
                for lx, ly, face_yaw in po.chair_spots(chairs):
                    if self.rng.random() < 0.35:
                        wx, wy = p[0] + lx * c - ly * sn, p[1] + lx * sn + ly * c
                        self.add_person(wx, wy, z, yaw + face_yaw, pose="sitting", reserve=False)
            s += 2.6
        zone = [self.fpt(face, k, s0 - 1.2, 1.0), self.fpt(face, k, s1 + 1.2, 1.0),
                self.fpt(face, k, s1 + 1.2, rows[-1] + 1.4), self.fpt(face, k, s0 - 1.2, rows[-1] + 1.4)]
        self.add_zone("cafe", zone)

    def add_market(self, face, k, z):
        """A row of stalls facing into the square, with a queue at one of them."""
        length = self.edge_len(face, k)
        a, u, m = self.frame(face, k)
        count = max(2, min(5, int((length - 8.0) / 3.6)))
        d = 2.2
        queue_at = self.rng.randrange(count)
        for idx in range(count):
            s = 4.0 + idx * 3.6 + 1.3
            p = self.fpt(face, k, s, d)
            yaw = angle_of(u) + math.pi   # crates (local -Y) face into the square
            stall = self.place_world(po.make_market_stall(), p[0], p[1], z, yaw, "surface_obstacle_market_stall", gap=0.3)
            if stall is None:
                continue
            fabric_name, fabric = self.rng.choice(list(FABRICS.items()))
            stall["color_name"] = fabric_name
            self.place_world(po.make_stall_canopy(fabric), p[0], p[1], z, yaw, "surface_overhead_canopy_stall",
                             reserve=False)
            self.add_person(*self.fpt(face, k, s, d - 1.1), z, angle_of(m), reserve=False)  # vendor behind
            if idx == queue_at:
                n = self.rng.randint(3, 5)
                for q in range(n):
                    qp = self.fpt(face, k, s, d + 1.35 + q * 0.75)
                    self.add_person(qp[0], qp[1], z, angle_of((-m[0], -m[1])), reserve=False)
                end = d + 1.35 + (n - 1) * 0.75
                self.add_zone("queue", [self.fpt(face, k, s - 0.9, d + 0.9), self.fpt(face, k, s + 0.9, d + 0.9),
                                        self.fpt(face, k, s + 0.9, end + 0.9), self.fpt(face, k, s - 0.9, end + 0.9)])
                xs = [self.fpt(face, k, s + ds, dd)[0] for ds in (-0.9, 0.9) for dd in (d, end + 0.9)]
                ys = [self.fpt(face, k, s + ds, dd)[1] for ds in (-0.9, 0.9) for dd in (d, end + 0.9)]
                self.keep_clear((min(xs), max(xs), min(ys), max(ys)))

    # ---- streets ---------------------------------------------------------------

    def street_spot(self, seg, margin=1.0, along=None):
        """(x, y, z, yaw) at a random spot on a street, away from intersections and off the road."""
        a, b = tuple(seg["a"]), tuple(seg["b"])
        u = norm(sub(b, a))
        length = math.hypot(*sub(b, a))
        if length < 14.0:
            return None
        t = along if along is not None else self.rng.uniform(7.0, length - 7.0)
        j_wall, j_main = self.layout["wall_line"], self.layout["main_line"]
        half = seg["width"] / 2 - margin
        if seg["axis"] == "h" and seg["j"] == j_main:
            off = self.rng.choice((-1, 1)) * self.rng.uniform(ot.ROAD_W / 2 + 0.6, ot.ROAD_W / 2 + 2.0)
            z = ot.SIDEWALK_Z
        else:
            off = self.rng.uniform(-half, half) if half > 0 else 0.0
            upper = seg["j"] >= j_wall
            z = self.H if upper else 0.0
            if seg["axis"] == "v" and seg["j"] == j_wall - 1 and t > length - 18.0:
                return None  # near the wall: stairs and ramps
        p = add(add(a, mul(u, t)), mul(left(u), off))
        return p[0], p[1], z, angle_of(u)

    def furnish_streets(self):
        streets = self.layout["streets"]
        # People walking along the streets, mostly in the walking direction of the street.
        for _ in range(70):
            spot = self.street_spot(self.rng.choice(streets))
            if spot:
                x, y, z, yaw = spot
                self.add_person(x, y, z, yaw + self.rng.choice((0.0, math.pi)) + self.rng.uniform(-0.3, 0.3),
                                pose="walking" if self.rng.random() < 0.75 else "standing")
        # A few groups on the wider streets and the promenade.
        for _ in range(8):
            seg = self.rng.choice([s for s in streets if s["width"] >= 6.0])
            spot = self.street_spot(seg, margin=2.0)
            if spot:
                self.add_group(spot[0], spot[1], spot[2])
        # Parked scooters and bikes against the house fronts, benches and lanterns on the promenade.
        for _ in range(26):
            seg = self.rng.choice(streets)
            spot = self.street_spot(seg)
            if spot is None:
                continue
            x, y, z, yaw = spot
            a, b = tuple(seg["a"]), tuple(seg["b"])
            u = norm(sub(b, a))
            rel = sub((x, y), a)
            t = rel[0] * u[0] + rel[1] * u[1]
            side = self.rng.choice((-1, 1))
            edge_off = side * (seg["width"] / 2 - 0.55)
            if seg["axis"] == "h" and seg["j"] == self.layout["main_line"]:
                edge_off = side * (seg["width"] / 2 - 0.55)
            p = add(add(a, mul(u, t)), mul(left(u), edge_off))
            if self.rng.random() < 0.65:
                paint = self.rng.choice(po.make_scooter_paints())
                obj = self.place_world(po.make_scooter(paint), p[0], p[1], z, yaw, "surface_obstacle_scooter", gap=0.3)
                if obj is not None:
                    obj["color_name"] = paint.split("_", 1)[1]
            else:
                self.place_world(props.make_bike(), p[0], p[1], z, yaw + math.pi / 2, "surface_obstacle_bike", gap=0.3)
        promenade = [s for s in streets if s["axis"] == "h" and s["j"] == self.layout["wall_line"]]
        for seg in promenade:
            a, b = tuple(seg["a"]), tuple(seg["b"])
            u = norm(sub(b, a))
            length = math.hypot(*sub(b, a))
            s = 5.0
            while s < length - 5.0:
                # On the promenade, 1.3 m in from its south edge, facing along the promenade's middle
                # (benches face local -Y).
                p = add(add(a, mul(u, s)), mul(left(u), -seg["width"] / 2 + 1.3))
                choice = self.rng.random()
                if choice < 0.45:
                    self.place_world(props.make_bench(), p[0], p[1], self.H, angle_of(u) + math.pi,
                                     "surface_obstacle_bench")
                elif choice < 0.7:
                    self.place_world(po.make_flower_pot(0.6), p[0], p[1], self.H, 0.0, "surface_obstacle_flower_pot")
                else:
                    self.add_lantern(p[0], p[1], self.H, angle_of(u) + math.pi / 2)
                s += self.rng.uniform(5.0, 8.0)
        # Trash cans and cats or dogs here and there.
        for _ in range(12):
            spot = self.street_spot(self.rng.choice(streets), margin=0.7)
            if spot:
                self.place_world(props.make_trash_can(), spot[0], spot[1], spot[2], 0.0, "surface_obstacle_trash_can")
        spots = [self.street_spot(self.rng.choice(streets)) for _ in range(14)]
        self.add_animals_at([s[:3] for s in spots if s])

    def add_animals_at(self, spots):
        if not self.animals:
            return
        for x, y, z in spots:
            kind = self.rng.choice(("cat", "dog"))
            color = self.rng.choice(self.animals.variants(kind))
            mesh, half_l, half_w = self.animals.templates[(kind, color)]
            obj = self.place_mesh(mesh, half_l, half_w, x, y, z, self.rng.uniform(0, 2 * math.pi),
                                  f"surface_obstacle_{kind}", kind)
            if obj is not None:
                obj["color_name"] = color

    # ---- all -------------------------------------------------------------------

    def build(self):
        self.build_ground()
        self.build_main_street()
        self.build_wall_links()
        self.build_squares()      # grand stairs add their wall gap before the parapet is built
        self.build_parapet()
        self.build_faces()
        self.furnish_streets()
        print(f"oldtown people={self.people} zones={len(self.layout['social_zones'])} "
              f"platforms={len(self.layout['platforms'])}")
