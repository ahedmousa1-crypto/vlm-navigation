"""World 3: build the campus on a hillside above a waterfront (see campus_layout.py) as tagged geometry.

Reuses the old town's helpers (people, groups, cafes, markets, railings,
lanterns, animals) and adds the campus pieces: the lake, quay, promenade with
a bike lane and a boardwalk, lower quay ledges reached by steps and a ramp, a
pier with moored boats, a canal with footbridges, the retaining wall with its
stairs and ramps, an amphitheater and terrace gardens on the slope, lawn quads
crossed by paths, faculty buildings with stepped and ramped entrances,
libraries on plinths, a sports field, parking, a construction site, a park
belt with gravel paths, and a road with bike lanes along the top.
"""

import math

from . import campus_layout as cl
from . import props
from . import props_extra as px
from . import props_oldtown as po
from .campus_layout import add, left, mul, norm, point_in_poly, sub
from .materials import CAR_PAINTS, FABRICS
from .meshes import add_bmesh_object, bm_box, bm_cylinder, bm_obox, bm_prism
from .oldtown import OldTownBuilder, angle_of, circle
from .props import Prop

FLOOR_H = 3.6
FACADES = ("facade_brick", "facade_gray", "facade_light", "facade_beige", "concrete")
BOAT_PAINTS = ("paint_white", "paint_blue", "paint_red", "paint_green", "paint_yellow")
PATH_W = 3.0


def strip_poly(p0, p1, w):
    """Counter-clockwise quad of width ``w`` around the segment p0 -> p1."""
    u = norm(sub(p1, p0))
    n = left(u)
    h = w / 2
    return [add(p0, mul(n, -h)), add(p1, mul(n, -h)), add(p1, mul(n, h)), add(p0, mul(n, h))]


def seg_dist(p, a, b):
    ab = sub(b, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / ((ab[0] ** 2 + ab[1] ** 2) or 1e-9)))
    return math.hypot(p[0] - (a[0] + ab[0] * t), p[1] - (a[1] + ab[1] * t))


def make_boat(paint, length, cabin):
    """Small boat, bow toward local +X, hull bottom 0.35 below z=0 (the waterline)."""
    half_l, half_w = length / 2, length * 0.18

    def builder(bm):
        hull = [(-half_l, -half_w), (half_l * 0.55, -half_w), (half_l, 0.0), (half_l * 0.55, half_w), (-half_l, half_w)]
        inner = [(x * 0.9, y * 0.85) for x, y in hull]
        bm_prism(bm, hull, -0.35, 0.45, 0)
        bm_prism(bm, inner, 0.3, 0.47, 2)          # deck
        if cabin:
            bm_box(bm, -half_l * 0.55, half_l * 0.05, -half_w * 0.65, half_w * 0.65, 0.47, 1.35, material_index=3)
            bm_box(bm, -half_l * 0.56, half_l * 0.06, -half_w * 0.66, half_w * 0.66, 0.95, 1.2, material_index=1)

    return Prop(builder, [paint, "glass", "wood", "paint_white"], "obstacle", "boat", half_l, half_w)


def make_mooring_post():
    def builder(bm):
        bm_cylinder(bm, (0.0, 0.0, 0.275), 0.14, 0.55, segments=10)
        bm_cylinder(bm, (0.0, 0.0, 0.59), 0.18, 0.08, segments=10)

    return Prop(builder, ["paint_black"], "obstacle", "bollard", 0.18, 0.18)


def make_goal_frame():
    def builder(bm):
        for y in (-3.66, 3.66):
            bm_box(bm, -0.06, 0.06, y - 0.06, y + 0.06, 0.0, 2.44)
        bm_box(bm, -0.06, 0.06, -3.72, 3.72, 2.38, 2.5)
        bm_box(bm, 1.2, 1.3, -3.66, 3.66, 0.0, 0.05)

    return Prop(builder, ["paint_white"], "obstacle", "barrier", 0.7, 3.72)


def make_bleachers(length):
    def builder(bm):
        for k in range(4):
            bm_box(bm, -length / 2, length / 2, -1.6 + k * 0.8, 1.6, k * 0.4, (k + 1) * 0.4)

    return Prop(builder, ["concrete"], "obstacle", "barrier", length / 2, 1.6)


class CampusBuilder(OldTownBuilder):
    def __init__(self, layout, materials, animals=None):
        super().__init__(layout, materials, animals)
        self.quay_y = layout["quay_y"]
        canal = layout["canal"]
        self.canal_x = canal["a"][0]
        self.canal_x0 = self.canal_x - canal["width"] / 2
        self.canal_x1 = self.canal_x + canal["width"] / 2
        self.canal_head = canal["b"][1]
        self.quay_gaps = []      # (x0, x1) stretches of the quay edge without railing
        self.lamps = 0

    # ---- small helpers ---------------------------------------------------

    def slab(self, name, surface, poly, z0, z1, mat, coll="Blocks", category=None, side=None):
        mats = [self.mat[mat]] + ([self.mat[side]] if side else [])
        return add_bmesh_object(self.next_name(name), surface,
                                lambda bm: bm_prism(bm, poly, z0, z1, 0, 1 if side else None), mats, self.coll[coll],
                                category=category)

    def fquad(self, face, k, s0, s1, d0, d1):
        return [self.fpt(face, k, s0, d0), self.fpt(face, k, s1, d0), self.fpt(face, k, s1, d1), self.fpt(face, k, s0, d1)]

    def centre(self, face):
        inset = face["inset"]
        return (sum(p[0] for p in inset) / 4, sum(p[1] for p in inset) / 4)

    def in_face(self, face, p, margin=0.0):
        if not point_in_poly(p, [tuple(q) for q in face["inset"]]):
            return False
        if margin <= 0:
            return True
        inset = [tuple(q) for q in face["inset"]]
        return all(seg_dist(p, inset[k], inset[(k + 1) % 4]) >= margin for k in range(4))

    def lamp(self, x, y, z, yaw):
        if self.place_world(props.make_lamp_pole(), x, y, z, yaw, "surface_obstacle_lamp", gap=0.3):
            self.place_world(props.make_lamp_head(), x, y, z, yaw, "surface_overhead_lamp", reserve=False)
            self.lamps += 1

    def bench(self, p, z, yaw, sit=0.3):
        obj = self.place_world(props.make_bench(), p[0], p[1], z, yaw, "surface_obstacle_bench", gap=0.3)
        if obj is not None and self.rng.random() < sit:
            c, s = math.cos(yaw), math.sin(yaw)
            lx = self.rng.choice((-0.3, 0.3))
            self.add_person(p[0] + lx * c + 0.05 * s, p[1] + lx * s - 0.05 * c, z, yaw - math.pi / 2, pose="sitting",
                            reserve=False)
        return obj

    def walkers(self, face, z, count, region=None):
        cx, cy = self.centre(face)
        for _ in range(count):
            for _attempt in range(8):
                p = (cx + self.rng.uniform(-16, 16), cy + self.rng.uniform(-14, 14))
                if region is None or point_in_poly(p, region):
                    if self.add_person(p[0], p[1], z, self.rng.uniform(0, 2 * math.pi),
                                       pose="walking" if self.rng.random() < 0.7 else "standing"):
                        break

    def groups(self, face, z, count, spread=10.0):
        cx, cy = self.centre(face)
        for _ in range(count):
            for _attempt in range(15):
                p = (cx + self.rng.uniform(-spread, spread), cy + self.rng.uniform(-spread, spread))
                if self.in_face(face, p, 2.0) and self.add_group(p[0], p[1], z):
                    break

    # ---- ground, water, levels -------------------------------------------

    def build_ground(self):
        ex0, ex1, ey0, ey1 = self.layout["extent"]
        qy = self.quay_y
        wall = self.wall
        # The lake, with the canal cut into the lower level.
        self.slab("surface_water_lake", "water", [(ex0, ey0), (ex1, ey0), (ex1, qy), (ex0, qy)],
                  cl.WATER_Z - 1.0, cl.WATER_Z, "water", coll="Ground")
        self.slab("surface_water_canal", "water", [(self.canal_x0, qy - 0.2), (self.canal_x1, qy - 0.2),
                                                   (self.canal_x1, self.canal_head), (self.canal_x0, self.canal_head)],
                  cl.WATER_Z - 1.0, cl.WATER_Z, "water", coll="Ground")

        # Lower level: vertical slices between the quay and the wall, around the canal.
        xs = sorted({p[0] for p in wall} | {self.canal_x0, self.canal_x1})
        xs = [x for x in xs if ex0 <= x <= ex1]

        def lower(bm):
            for xa, xb in zip(xs, xs[1:]):
                mid = (xa + xb) / 2
                south = self.canal_head if self.canal_x0 <= mid <= self.canal_x1 else qy
                poly = [(xa, south), (xb, south), (xb, self.wall_y(xb)), (xa, self.wall_y(xa))]
                bm_prism(bm, poly, cl.WATER_Z - 1.2, 0.0, 0, 1)

        add_bmesh_object("surface_sidewalk_lower", "sidewalk", lower, [self.mat["sidewalk"], self.mat["wall_stone"]],
                         self.coll["Ground"])

        def upper(bm):
            for (x0, y0), (x1, y1) in zip(wall, wall[1:]):
                bm_prism(bm, [(x0, y0), (x1, y1), (x1, ey1), (x0, ey1)], cl.BASE_Z, self.H, 0, 1)

        add_bmesh_object("surface_sidewalk_upper", "sidewalk", upper, [self.mat["sidewalk"], self.mat["wall_stone"]],
                         self.coll["Ground"])

        # Far bank across the lake, closing the views over the water.
        fy = qy - cl.FAR_BANK
        self.slab("surface_grass_farbank", "grass", [(ex0, ey0), (ex1, ey0), (ex1, fy), (ex0, fy)],
                  cl.WATER_Z - 1.0, 0.6, "grass", coll="Ground", side="wall_stone")
        x = ex0 + 6.0
        while x < ex1 - 10.0:
            w = self.rng.uniform(12.0, 22.0)
            d = self.rng.uniform(10.0, 16.0)
            poly = [(x, fy - 4.0 - d), (x + w, fy - 4.0 - d), (x + w, fy - 4.0), (x, fy - 4.0)]
            self.campus_building_poly(poly, 0.6, self.rng.randint(3, 7), self.rng.choice(FACADES), "farbank")
            if self.rng.random() < 0.5:
                self.add_tree(x + w + 2.0, fy - 2.5, 0.6, "surface_obstacle_tree_farbank", style="round", reserve_gap=0.2)
            x += w + self.rng.uniform(1.0, 6.0)

    # ---- the promenade along the quay -------------------------------------

    def build_promenade(self):
        ex0, ex1 = self.layout["extent"][:2]
        qy = self.quay_y
        y_bike0 = qy + (cl.PROM_W - cl.PROM_WALK - cl.PROM_BIKE)
        y_bike1 = y_bike0 + cl.PROM_BIKE
        cx0, cx1 = self.canal_x0, self.canal_x1
        for x0, x1 in ((ex0, cx0), (cx1, ex1)):
            self.slab("surface_boardwalk_prom", "boardwalk", [(x0, qy), (x1, qy), (x1, y_bike0), (x0, y_bike0)],
                      -0.1, 0.03, "deck", coll="Ground")
            obj = self.slab("surface_bike_lane_prom", "bike_lane", [(x0, y_bike0), (x1, y_bike0), (x1, y_bike1), (x0, y_bike1)],
                            -0.1, 0.02, "bike_lane", coll="Markings")
            obj["category"] = "bike_lane"
        # Bridge over the canal mouth, carrying the whole promenade.
        bridge = self.slab("surface_bridge_canal_mouth", "bridge", [(cx0 - 0.6, qy), (cx1 + 0.6, qy),
                                                                     (cx1 + 0.6, qy + cl.PROM_W), (cx0 - 0.6, qy + cl.PROM_W)],
                           -0.25, 0.05, "deck", category="bridge")
        bridge["end_a"], bridge["end_b"] = [cx0 - 2.5, qy + 3.0], [cx1 + 2.5, qy + 3.0]
        for y in (qy + 0.1, qy + cl.PROM_W - 0.1):
            self.railing("surface_obstacle_railing", (cx0 - 0.6, y), 0.05, (cx1 + 0.6, y), 0.05)
        self.quay_gaps.append((cx0 - 0.8, cx1 + 0.8))

        self.build_ledges()
        self.build_pier()
        # Buildings closing both ends of the promenade.
        for x0, x1 in ((ex0, ex0 + 14.0), (ex1 - 14.0, ex1)):
            self.campus_building_poly([(x0, qy), (x1, qy), (x1, qy + cl.PROM_W + 30.0), (x0, qy + cl.PROM_W + 30.0)], 0.0,
                                      self.rng.randint(3, 5), self.rng.choice(FACADES), "prom_end")
            self.keep_clear((x0, x1, qy, qy + cl.PROM_W + 30.0))
        ex0, ex1 = ex0 + 15.0, ex1 - 15.0   # the promenade runs between the two end buildings

        # Railing along the quay edge, open at the ledges and the pier.
        x = ex0 + 2.0
        cuts = sorted(self.quay_gaps)
        segments, start = [], x
        for g0, g1 in cuts:
            if g0 > start:
                segments.append((start, g0))
            start = max(start, g1)
        segments.append((start, ex1 - 2.0))
        for s0, s1 in segments:
            if s1 - s0 > 1.0:
                self.railing("surface_obstacle_railing_quay", (s0, qy + 0.12), 0.03, (s1, qy + 0.12), 0.03)
        self.keep_clear((ex0, ex1, qy - 0.2, qy + 0.5))

        # Lamps and trees along the north edge of the paved walk, benches facing the water on the boardwalk.
        y_lamp = qy + cl.PROM_W - 0.6
        x = ex0 + 8.0
        k = 0
        while x < ex1 - 8.0:
            if not (cx0 - 3.0 < x < cx1 + 3.0):
                if k % 2 == 0:
                    self.lamp(x, y_lamp, 0.0, -math.pi / 2)
                else:
                    self.add_tree(x, y_lamp - 0.3, 0.0, "surface_obstacle_tree_prom", style="round", reserve_gap=0.4)
            x += 9.0
            k += 1
        x = ex0 + 12.0
        while x < ex1 - 12.0:
            if not (cx0 - 4.0 < x < cx1 + 4.0) and not any(g0 - 2.0 < x < g1 + 2.0 for g0, g1 in self.quay_gaps):
                if self.rng.random() < 0.55:
                    self.bench((x, qy + 1.6), 0.03, 0.0, sit=0.4)   # faces local -Y = the water
                elif self.rng.random() < 0.3:
                    self.place_world(props.make_trash_can(), x, qy + 1.2, 0.03, 0.0, "surface_obstacle_trash_can")
            x += self.rng.uniform(7.0, 11.0)
        # People: walkers both ways, a few standing at the railing looking out, groups on the boardwalk.
        for _ in range(70):
            x = self.rng.uniform(ex0 + 10.0, ex1 - 10.0)
            if cx0 - 1.0 < x < cx1 + 1.0:
                continue
            lane = self.rng.random()
            y = self.rng.uniform(qy + 2.5, y_bike0 - 0.8) if lane < 0.6 else self.rng.uniform(y_bike1 + 0.6, qy + cl.PROM_W - 1.2)
            self.add_person(x, y, 0.03 if y < y_bike0 else 0.0, self.rng.choice((0.0, math.pi)) + self.rng.uniform(-0.25, 0.25),
                            pose="walking" if self.rng.random() < 0.8 else "standing")
        for _ in range(10):
            x = self.rng.uniform(ex0 + 10.0, ex1 - 10.0)
            self.add_group(x, self.rng.uniform(qy + 3.0, y_bike0 - 1.5), 0.03)
        for _ in range(12):
            x = self.rng.uniform(ex0 + 10.0, ex1 - 10.0)
            if not any(g0 - 1.0 < x < g1 + 1.0 for g0, g1 in self.quay_gaps):
                self.add_person(x, qy + 1.05, 0.03, -math.pi / 2, pose="standing")
        spots = [(self.rng.uniform(ex0 + 10, ex1 - 10), self.rng.uniform(qy + 2.0, y_bike0 - 1.0), 0.03) for _ in range(8)]
        self.add_animals_at(spots)
        # Boats out on the lake (landmarks, never reachable).
        for _ in range(9):
            bx = self.rng.uniform(ex0 + 20, ex1 - 20)
            by = self.rng.uniform(qy - cl.FAR_BANK + 12, qy - 30)
            boat = self.place_world(make_boat(self.rng.choice(BOAT_PAINTS), self.rng.uniform(5, 9), self.rng.random() < 0.6),
                                    bx, by, cl.WATER_Z, self.rng.uniform(0, 2 * math.pi), "surface_obstacle_boat_lake",
                                    gap=2.0, collection="Traffic")
            if boat is not None:
                boat["qa_skip"] = True
                boat["in_traffic"] = True   # out on the water: a landmark, never a goal

    def build_ledges(self):
        """Lower quay ledges just above the water: stairs down at the west end, a ramp up at the east end."""
        qy = self.quay_y
        depth = 3.2
        steps = round(-cl.LEDGE_Z / cl.RISE)
        rise = -cl.LEDGE_Z / steps
        run_s = steps * cl.TREAD
        run_r = -cl.LEDGE_Z * 12.0
        for index, ledge in enumerate(self.layout["ledges"]):
            x0, x1 = ledge["x0"], ledge["x1"]
            yo, yi = qy - depth, qy
            # Landings at promenade level at both ends, then stairs, the low ledge, and the ramp.
            for la, lb in ((x0 - 3.0, x0), (x1, x1 + 3.0)):
                self.slab("surface_boardwalk_landing", "boardwalk", [(la, yo), (lb, yo), (lb, yi), (la, yi)],
                          cl.WATER_Z - 1.2, 0.03, "deck", side="wall_stone")

            def stairs(bm, x0=x0):
                for k in range(steps):
                    bm_prism(bm, [(x0 + k * cl.TREAD, yo), (x0 + (k + 1) * cl.TREAD, yo), (x0 + (k + 1) * cl.TREAD, yi),
                                  (x0 + k * cl.TREAD, yi)], cl.WATER_Z - 1.2, -(k + 1) * rise + 0.03, 0, 1)

            stairs_obj = add_bmesh_object(f"surface_stairs_ledge{index}", "stairs", stairs,
                                          [self.mat["stairs"], self.mat["wall_stone"]], self.coll["Blocks"], category="stairs")
            lx0, lx1 = x0 + run_s, x1 - run_r
            self.slab(f"surface_plaza_ledge{index}", "plaza", [(lx0, yo), (lx1, yo), (lx1, yi), (lx0, yi)],
                      cl.WATER_Z - 1.2, cl.LEDGE_Z, "stone", side="wall_stone")
            poly = [(lx1, yo), (x1, yo), (x1, yi), (lx1, yi)]
            ramp_obj = add_bmesh_object(f"surface_ramp_ledge{index}", "ramp",
                                        lambda bm, poly=poly: bm_prism(bm, poly, cl.WATER_Z - 1.2,
                                                                       [cl.LEDGE_Z, 0.03, 0.03, cl.LEDGE_Z], 0, 1),
                                        [self.mat["ramp"], self.mat["wall_stone"]], self.coll["Blocks"], category="ramp")
            # The step down along the quay between the promenade and the ledge gets a railing,
            # except where the stairs and the ramp meet the landings.
            self.railing("surface_obstacle_railing_ledge", (x0 + 0.4, qy + 0.1), 0.03, (x1 - 0.4, qy + 0.1), 0.03)
            self.quay_gaps.append((x0 - 3.1, x1 + 3.1))
            # Mooring posts and moored boats along the outer edge.
            x = lx0 + 1.5
            while x < lx1 - 1.0:
                self.place_world(make_mooring_post(), x, yo + 0.35, cl.LEDGE_Z, 0.0, "surface_obstacle_mooring", gap=0.1)
                x += 6.0
            x = lx0 + 4.0
            while x < x1 - 4.0:
                if self.rng.random() < 0.7:
                    length = self.rng.uniform(4.5, 7.0)
                    boat = self.place_world(make_boat(self.rng.choice(BOAT_PAINTS), length, self.rng.random() < 0.5),
                                            x, yo - length * 0.18 - 0.5, cl.WATER_Z, self.rng.choice((0.0, math.pi)),
                                            "surface_obstacle_boat", gap=0.3, collection="Traffic")
                    if boat is not None:
                        boat["qa_skip"] = True
                        boat["goal"] = [x, yo + 1.0]
                        boat["goal_z"] = cl.LEDGE_Z
                        boat["color_name"] = boat.data.materials[0].name.replace("mat_paint_", "")
                x += self.rng.uniform(7.0, 10.0)
            # A person or two on the ledge.
            for _ in range(self.rng.randint(1, 3)):
                self.add_person(self.rng.uniform(lx0 + 2, lx1 - 2), self.rng.uniform(yo + 1.0, yi - 0.8), cl.LEDGE_Z,
                                self.rng.uniform(0, 2 * math.pi))
            goal = ((lx0 + lx1) / 2, qy - depth / 2)
            self.layout["platforms"].append({"kind": "quay", "stairs": stairs_obj.name, "ramp": ramp_obj.name,
                                             "goal": [round(goal[0], 3), round(goal[1], 3)], "z": cl.LEDGE_Z, "down": True})
            self.keep_clear((x0 - 3.0, x1 + 3.0, qy - 0.1, qy + 1.4))

    def build_pier(self):
        pier = self.layout["pier"]
        qy = self.quay_y
        x, length, w = pier["x"], pier["length"], pier["width"]
        y_end = qy - length
        poly = [(x - w / 2, y_end), (x + w / 2, y_end), (x + w / 2, qy + 0.2), (x - w / 2, qy + 0.2)]

        def deck(bm):
            bm_prism(bm, poly, -0.25, 0.03, 0)
            yy = y_end + 0.4
            while yy < qy:
                for px_ in (x - w / 2 + 0.2, x + w / 2 - 0.2):
                    bm_cylinder(bm, (px_, yy, (cl.WATER_Z - 1.0 - 0.25) / 2), 0.15, 0.75 - cl.WATER_Z, segments=8,
                                material_index=1)
                yy += 3.0

        obj = add_bmesh_object("surface_boardwalk_pier", "boardwalk", deck, [self.mat["deck"], self.mat["wood"]],
                               self.coll["Blocks"], category="pier")
        obj["goal"] = [x, y_end + 2.0]
        for sx in (x - w / 2 + 0.06, x + w / 2 - 0.06):
            self.railing("surface_obstacle_railing_pier", (sx, qy - 0.6), 0.03, (sx, y_end + 0.1), 0.03)
        self.railing("surface_obstacle_railing_pier", (x - w / 2 + 0.06, y_end + 0.08), 0.03, (x + w / 2 - 0.06, y_end + 0.08), 0.03)
        self.quay_gaps.append((x - w / 2 - 0.2, x + w / 2 + 0.2))
        yy = qy - 5.0
        while yy > y_end + 3.0:
            for side in (-1, 1):
                if self.rng.random() < 0.65:
                    blen = self.rng.uniform(4.5, 8.0)
                    bx = x + side * (w / 2 + blen * 0.18 + 0.4)
                    boat = self.place_world(make_boat(self.rng.choice(BOAT_PAINTS), blen, self.rng.random() < 0.6),
                                            bx, yy, cl.WATER_Z, -math.pi / 2, "surface_obstacle_boat", gap=0.2,
                                            collection="Traffic")
                    if boat is not None:
                        boat["qa_skip"] = True
                        boat["goal"] = [x + side * (w / 2 - 0.9), yy]
                        boat["color_name"] = boat.data.materials[0].name.replace("mat_paint_", "")
            yy -= self.rng.uniform(6.5, 9.0)
        for _ in range(4):
            self.add_person(x + self.rng.uniform(-1.0, 1.0), self.rng.uniform(y_end + 2, qy - 2), 0.03,
                            self.rng.choice((math.pi / 2, -math.pi / 2)), pose=self.rng.choice(("walking", "standing")))
        self.keep_clear((x - w / 2, x + w / 2, qy, qy + 2.0))

    # ---- canal and footbridges --------------------------------------------

    def build_canal(self):
        qy = self.quay_y
        y0, y1 = qy + cl.PROM_W, self.canal_head
        bridges = []
        for t in self.layout["footbridges"]:
            yb = y0 + (y1 - y0) * t
            bw = 3.2
            obj = self.slab("surface_bridge_foot", "bridge", [(self.canal_x0 - 0.8, yb - bw / 2), (self.canal_x1 + 0.8, yb - bw / 2),
                                                              (self.canal_x1 + 0.8, yb + bw / 2), (self.canal_x0 - 0.8, yb + bw / 2)],
                            -0.3, 0.06, "deck", category="bridge")
            obj["end_a"], obj["end_b"] = [self.canal_x0 - 2.4, yb], [self.canal_x1 + 2.4, yb]
            for y in (yb - bw / 2 + 0.06, yb + bw / 2 - 0.06):
                self.railing("surface_obstacle_railing_bridge", (self.canal_x0 - 0.7, y), 0.06, (self.canal_x1 + 0.7, y), 0.06)
            bridges.append((yb - bw / 2 - 0.1, yb + bw / 2 + 0.1))
            self.keep_clear((self.canal_x0 - 3.0, self.canal_x1 + 3.0, yb - 2.0, yb + 2.0))
        # Railings along both canal edges, open at the bridges.
        for x in (self.canal_x0 - 0.12, self.canal_x1 + 0.12):
            start = y0 + 0.2
            for b0, b1 in sorted(bridges) + [(y1 - 0.1, y1 + 1.0)]:
                if b0 - start > 1.0:
                    self.railing("surface_obstacle_railing_canal", (x, start), 0.0, (x, b0), 0.0)
                start = b1
        self.railing("surface_obstacle_railing_canal", (self.canal_x0, y1 + 0.12), 0.0, (self.canal_x1, y1 + 0.12), 0.0)
        self.keep_clear((self.canal_x0 - 0.6, self.canal_x1 + 0.6, y0, y1 + 0.5))

    # ---- retaining wall, links, parapet --------------------------------------

    def build_wall_links(self):
        j = self.layout["wall_line"]
        for link in self.layout["wall_links"]:
            i, width = link["i"], link["width"]
            a, b = self.nodes[(i, j - 1)], self.nodes[(i, j)]
            v = norm(sub(b, a))
            hit, w = self.wall_crossing(a, v)
            if hit is None:
                continue
            down = left(w)
            down = (-down[0], -down[1])
            side_a = (add(a, mul(left(v), width / 2)), v)
            side_b = (add(a, mul(left(v), -width / 2)), v)
            from .oldtown_layout import intersect

            def strip(d0, d1):
                p0, p1 = add(hit, mul(down, d0)), add(hit, mul(down, d1))
                return [intersect(p0, w, *side_b), intersect(p0, w, *side_a), intersect(p1, w, *side_a),
                        intersect(p1, w, *side_b)]

            kind = link["kind"]
            name = f"surface_{'ramp' if kind == 'ramp' else 'stairs'}_link{i}"
            if kind == "ramp":
                run = self.H / cl.RAMP_SLOPE
                poly = strip(0.0, run)
                obj = add_bmesh_object(name, "ramp", lambda bm, poly=poly: bm_prism(bm, poly, cl.BASE_Z, [self.H, self.H, 0.0, 0.0], 0, 1),
                                       [self.mat["ramp"], self.mat["wall_stone"]], self.coll["Blocks"], category="ramp")
            else:
                steps = round(self.H / cl.RISE)
                rise = self.H / steps

                def stairs(bm, steps=steps, rise=rise, strip=strip):
                    for k in range(steps):
                        bm_prism(bm, strip(k * cl.TREAD, (k + 1) * cl.TREAD), cl.BASE_Z, self.H - k * rise, 0, 1)

                run = steps * cl.TREAD
                obj = add_bmesh_object(name, "stairs", stairs, [self.mat["stairs"], self.mat["wall_stone"]],
                                       self.coll["Blocks"], category="stairs")
            obj["link"] = i
            foot = add(hit, mul(down, run + 1.0))
            self.keep_clear((min(hit[0], foot[0]) - width / 2, max(hit[0], foot[0]) + width / 2,
                             min(hit[1], foot[1]) - 0.5, max(hit[1], foot[1]) + 0.5))
            self.wall_gaps.append((hit[0] - width / 2 - 0.3, hit[0] + width / 2 + 0.3))
            # A railing along the open side of the stairs or ramp.
            for side in (side_a, side_b):
                p_top = intersect(hit, w, *side)
                p_foot = intersect(add(hit, mul(down, run)), w, *side)
                if p_top and p_foot:
                    pt = add(p_top, mul(down, 0.3))
                    self.railing("surface_obstacle_railing_link", pt, self.H, p_foot, 0.0 + (0.15 if kind == "stairs" else 0.0))

    def build_parapet(self):
        """Low wall along the top of the retaining wall, open at links and slope features."""
        nav_x0, nav_x1 = self.layout["nav_extent"][:2]
        gaps = list(self.wall_gaps)
        for face in self.faces.values():
            if face["row"] == self.layout["wall_line"] - 1 and face["kind"] == "faculty":
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
                        bm_obox(bm, p, u, s0, s1, 0.0, 0.35, self.H - 0.05, self.H + 0.95)

        obj = add_bmesh_object("surface_obstacle_parapet", "obstacle", builder, [self.mat["wall_stone"]],
                               self.coll["Blocks"], category="fence")
        obj["qa_skip"] = True

    # ---- the north road ---------------------------------------------------

    def build_road(self):
        road = [tuple(p) for p in self.layout["road"]]
        z = self.H
        half = cl.ROAD_W / 2
        from .oldtown_layout import offset_polyline
        layers = [("road", -half, half, 0.012, "road"), ("bike_lane", half, half + cl.BIKE_W, 0.016, "bike_lane"),
                  ("bike_lane", -half - cl.BIKE_W, -half, 0.016, "bike_lane")]
        for cls_name, d0, d1, top, mat in layers:
            lo, hi = offset_polyline(road, d0), offset_polyline(road, d1)

            def builder(bm, lo=lo, hi=hi, top=top):
                for k in range(len(road) - 1):
                    bm_prism(bm, [lo[k], lo[k + 1], hi[k + 1], hi[k]], z - 0.2, z + top)

            obj = add_bmesh_object(self.next_name(f"surface_{cls_name}_north"), cls_name, builder, [self.mat[mat]],
                                   self.coll["Ground"])
            obj["category"] = cls_name
        street_w = cl.H_WIDTH[cl.NY - 1]
        walk = (street_w - cl.ROAD_W - 2 * cl.BIKE_W) / 2
        for d0, d1 in ((half + cl.BIKE_W, half + cl.BIKE_W + walk), (-half - cl.BIKE_W - walk, -half - cl.BIKE_W)):
            lo, hi = offset_polyline(road, d0), offset_polyline(road, d1)

            def builder(bm, lo=lo, hi=hi):
                for k in range(len(road) - 1):
                    bm_prism(bm, [lo[k], lo[k + 1], hi[k + 1], hi[k]], z - 0.2, z + cl.SIDEWALK_Z, 0, 1)

            add_bmesh_object(self.next_name("surface_sidewalk_north"), "sidewalk", builder,
                             [self.mat["sidewalk"], self.mat["curb"]], self.coll["Blocks"])
        # Crosswalks over road and bike lanes, parked cars, a stopped car, bus stops and lamps.
        span = half + cl.BIKE_W
        for seg in (2, 5, 8):
            p0, p1 = road[seg], road[seg + 1]
            u = norm(sub(p1, p0))
            mid = add(p0, mul(sub(p1, p0), 0.5))
            n = left(u)

            def zebra(bm, mid=mid, u=u):
                bars = int(2 * span / 0.5)
                for b in range(bars):
                    d0 = -span + b * 2 * span / bars
                    bm_obox(bm, mid, u, -1.6, 1.6, d0, d0 + 2 * span / bars, z + 0.017, z + 0.03, b % 2)

            obj = add_bmesh_object(f"surface_crosswalk_north{seg}", "crosswalk", zebra,
                                   [self.mat["crosswalk"], self.mat["crosswalk_dark"]], self.coll["Markings"],
                                   category="crosswalk")
            ends = (add(mid, mul(n, span + 1.3)), add(mid, mul(n, -span - 1.3)))
            obj["end_a"], obj["end_b"] = ends
            for ex, ey in ends:
                self.keep_clear((ex - 1.8, ex + 1.8, ey - 1.8, ey + 1.8))
            self.keep_clear((mid[0] - 2.2, mid[0] + 2.2, mid[1] - span, mid[1] + span))
        for k in range(1, len(road) - 2):
            p0, p1 = road[k], road[k + 1]
            u = norm(sub(p1, p0))
            length = math.hypot(*sub(p1, p0))
            n = left(u)
            s = 8.0
            while s < length - 8.0:
                if self.rng.random() < 0.55:
                    p = add(add(p0, mul(u, s)), mul(n, -half + 1.05))
                    paint = self.rng.choice(CAR_PAINTS)
                    car = self.place_world(props.make_car(paint), p[0], p[1], z + 0.012, angle_of(u), "surface_obstacle_car",
                                           gap=0.3)
                    if car is not None:
                        car["color_name"] = paint.split("_", 1)[1]
                s += self.rng.uniform(5.8, 7.5)
            if k % 3 == 1:
                p = add(add(p0, mul(u, length * self.rng.uniform(0.3, 0.7))), mul(n, half * 0.5))
                paint = self.rng.choice(CAR_PAINTS)
                car = self.place_world(props.make_car(paint), p[0], p[1], z + 0.012, angle_of(u) + math.pi,
                                       "surface_obstacle_car_traffic", gap=0.6, collection="Traffic")
                if car is not None:
                    car["in_traffic"] = True
                    car["color_name"] = paint.split("_", 1)[1]
            for side in (1, -1):
                d = side * (span + 0.45)
                s = 6.0
                while s < length - 6.0:
                    p = add(add(p0, mul(u, s)), mul(n, d))
                    self.lamp(p[0], p[1], z + cl.SIDEWALK_Z, angle_of(u) + (math.pi / 2 if side < 0 else -math.pi / 2))
                    s += 18.0
            if k in (3, 7):
                p = add(add(p0, mul(u, length * 0.5)), mul(n, -(span + walk * 0.6)))
                self.place_world(px.make_bus_stop(), p[0], p[1], z + cl.SIDEWALK_Z, angle_of(u), "surface_obstacle_bus_stop",
                                 gap=0.4)
                for q in range(self.rng.randint(1, 3)):
                    pp = add(add(p0, mul(u, length * 0.5 + 2.2 + q * 0.8)), mul(n, -(span + walk * 0.5)))
                    self.add_person(pp[0], pp[1], z + cl.SIDEWALK_Z, angle_of(n), reserve=False)

    # ---- campus buildings --------------------------------------------------

    def campus_building_poly(self, poly, z, floors, facade, tag, base_z=None):
        """Flat-roofed block over a convex quad with ribbon windows on every side."""
        base = z if base_z is None else base_z
        height = floors * FLOOR_H

        style = self.rng.random()
        pitch = self.rng.uniform(1.7, 2.6)
        win_w = pitch * self.rng.uniform(0.5, 0.75)
        glazed_ground = self.rng.random() < 0.45

        def builder(bm):
            bm_prism(bm, poly, base - 0.3, z + height, 0)
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                u = norm(sub(b, a))
                length = math.hypot(*sub(b, a))
                if length < 2.0:
                    continue
                cols = max(1, int((length - 0.6) / pitch))
                step = (length - 0.6) / cols
                for f in range(floors):
                    zf = z + f * FLOOR_H
                    if f == 0 and glazed_ground:
                        bm_obox(bm, a, u, 0.5, length - 0.5, -0.04, 0.02, zf + 0.3, zf + 3.0, 1)   # shop-front glazing
                        continue
                    if style < 0.3:
                        bm_obox(bm, a, u, 0.4, length - 0.4, -0.04, 0.02, zf + 0.9, zf + 2.6, 1)   # ribbon window
                        continue
                    for c in range(cols):
                        mid = 0.3 + (c + 0.5) * step
                        bm_obox(bm, a, u, mid - win_w / 2, mid + win_w / 2, -0.04, 0.02, zf + 0.85, zf + 2.55, 1)
                        bm_obox(bm, a, u, mid - win_w / 2 - 0.05, mid + win_w / 2 + 0.05, -0.1, 0.0, zf + 0.78, zf + 0.85, 2)
                bm_obox(bm, a, u, -0.05, length + 0.05, -0.1, 0.3, z + height, z + height + 0.5, 0)   # parapet

        obj = add_bmesh_object(self.next_name(f"surface_building_{tag}"), "building", builder,
                               [self.mat[facade], self.mat["glass"], self.mat["stone"]], self.coll["Buildings"],
                               category="building")
        obj["height"] = height
        return obj

    def campus_entrance(self, face, k, s_mid, d_front, z, steps=4, ramp=True):
        """Door in a facade along face edge k at depth ``d_front`` (facade line), with a porch, steps and a ramp.

        The building stands deeper than ``d_front`` (into the face); the porch and steps reach toward the street.
        """
        a, u, m = self.frame(face, k)
        rise = steps * cl.RISE
        top = z + rise
        porch_d = 1.8
        pw = 1.8

        def P(s, d):
            return self.fpt(face, k, s, d)

        porch = [P(s_mid - pw, d_front - porch_d), P(s_mid + pw, d_front - porch_d), P(s_mid + pw, d_front),
                 P(s_mid - pw, d_front)]
        add_bmesh_object(self.next_name("surface_porch_entrance"), "porch", lambda bm: bm_prism(bm, porch, cl.BASE_Z + z, top),
                         [self.mat["terrace"]], self.coll["Buildings"])

        def stairs(bm):
            for j in range(steps - 1):
                d1 = d_front - porch_d - j * cl.TREAD
                d0 = d1 - cl.TREAD
                bm_prism(bm, [P(s_mid - 1.2, d0), P(s_mid + 1.2, d0), P(s_mid + 1.2, d1), P(s_mid - 1.2, d1)],
                         cl.BASE_Z + z, top - (j + 1) * cl.RISE)

        stairs_obj = add_bmesh_object(self.next_name("surface_stairs_entrance"), "stairs", stairs, [self.mat["stairs"]],
                                      self.coll["Buildings"], category="stairs")
        ramp_obj = None
        if ramp:
            run = rise / cl.RAMP_SLOPE
            side = self.rng.choice((-1, 1))
            s_hi = s_mid + side * pw
            s_lo = s_hi + side * run
            sa, sb = min(s_hi, s_lo), max(s_hi, s_lo)
            poly = [P(sa, d_front - 1.4), P(sb, d_front - 1.4), P(sb, d_front - 0.05), P(sa, d_front - 0.05)]
            tops = [top if abs(sa - s_hi) < 1e-6 else z, top if abs(sb - s_hi) < 1e-6 else z] * 1
            tops = [tops[0], tops[1], tops[1], tops[0]]
            ramp_obj = add_bmesh_object(self.next_name("surface_ramp_entrance"), "ramp",
                                        lambda bm: bm_prism(bm, poly, cl.BASE_Z + z, tops), [self.mat["ramp"]],
                                        self.coll["Buildings"], category="ramp")
            self.railing("surface_obstacle_railing_ramp", P(s_lo - side * 0.6, d_front - 1.45), z + 0.075,
                         P(s_hi, d_front - 1.45), top)
            rect_pts = [P(sa - 0.3, d_front - 1.6), P(sb + 0.3, d_front)]
            self.keep_clear((min(p[0] for p in rect_pts), max(p[0] for p in rect_pts),
                             min(p[1] for p in rect_pts), max(p[1] for p in rect_pts)))
        door = add_bmesh_object(self.next_name(f"surface_door_{face['id']}"), "building",
                                lambda bm: bm_obox(bm, a, u, s_mid - 0.8, s_mid + 0.8, d_front - 0.06, d_front + 0.02,
                                                   top, top + 2.5), [self.mat["door"]], self.coll["Buildings"], category="door")
        front = P(s_mid, d_front - porch_d * 0.5)
        door["front"] = list(front)
        door["stairs"] = stairs_obj.name
        door["ramp"] = ramp_obj.name if ramp_obj else ""
        door["porch_z"] = top
        foot = [P(s_mid - 1.6, d_front - porch_d - steps * cl.TREAD - 1.2), P(s_mid + 1.6, d_front)]
        self.keep_clear((min(p[0] for p in foot), max(p[0] for p in foot), min(p[1] for p in foot), max(p[1] for p in foot)))
        facing = norm((-m[0], -m[1]))
        self.layout["entrances"].append({"door": door.name, "goal": list(front), "z": top, "stairs": stairs_obj.name,
                                         "ramp": door["ramp"], "facing": list(facing), "xy": list(P(s_mid, d_front))})
        return door, top

    def build_faculty(self, face, z):
        """One or two teaching buildings set back from the streets, entrances with steps and ramps,
        a paved court with trees, bike racks and benches."""
        self.slab(f"surface_plaza_{face['id']}", "plaza", [tuple(p) for p in face["inset"]], z - 0.1, z + 0.02, "plaza")
        zt = z + 0.02
        edges = self.rng.sample([0, 1, 2, 3], 2) if self.rng.random() < 0.6 else [self.rng.randrange(4)]
        for k in edges:
            length = self.edge_len(face, k)
            if length < 18.0:
                continue
            d_front = self.rng.uniform(4.5, 6.5)
            depth = self.rng.uniform(11.0, 14.0)
            s0, s1 = 3.5, length - 3.5
            poly = self.fquad(face, k, s0, s1, d_front, d_front + depth)
            if any(not self.in_face(face, p, 0.5) for p in poly):
                continue
            rect = (min(p[0] for p in poly), max(p[0] for p in poly), min(p[1] for p in poly), max(p[1] for p in poly))
            if any(not (rect[1] < o[0] or rect[0] > o[1] or rect[3] < o[2] or rect[2] > o[3]) for o in self.reserved[-400:]
                   if (o[1] - o[0]) > 8.0 and (o[3] - o[2]) > 8.0):
                continue
            floors = self.rng.randint(3, 5)
            self.campus_building_poly(poly, zt, floors, self.rng.choice(FACADES), face["id"])
            self.keep_clear(rect)
            for s_mid in ([(s0 + s1) / 2] if length < 34 else [s0 + (s1 - s0) * 0.3, s0 + (s1 - s0) * 0.72]):
                self.campus_entrance(face, k, s_mid, d_front, zt, steps=4, ramp=self.rng.random() < 0.85)
            # Bike racks and benches in the forecourt.
            for s in (s0 + 2.0, s1 - 2.0):
                p = self.fpt(face, k, s, d_front - 2.6)
                if self.rng.random() < 0.6:
                    _a, u, _m = self.frame(face, k)
                    self.place_world(px.make_bike_rack(), p[0], p[1], zt, angle_of(u), "surface_obstacle_bike_rack", gap=0.3)
                    for b in range(self.rng.randint(1, 3)):
                        q = self.fpt(face, k, s - 1.0 + b * 0.7, d_front - 2.6)
                        self.place_world(props.make_bike(), q[0], q[1], zt, angle_of(u) + math.pi / 2,
                                         "surface_obstacle_bike", gap=0.05)
        cx, cy = self.centre(face)
        for _ in range(4):
            p = (cx + self.rng.uniform(-8, 8), cy + self.rng.uniform(-8, 8))
            if self.in_face(face, p, 3.0):
                self.add_tree(p[0], p[1], zt, "surface_obstacle_tree_court", style=self.rng.choice(("round", "tall")))
        for _ in range(3):
            p = (cx + self.rng.uniform(-9, 9), cy + self.rng.uniform(-9, 9))
            if self.in_face(face, p, 2.5):
                self.bench(p, zt, self.rng.choice((0.0, math.pi / 2, math.pi, -math.pi / 2)), sit=0.5)
        self.walkers(face, zt, self.rng.randint(4, 8), region=[tuple(p) for p in face["inset"]])
        self.groups(face, zt, self.rng.randint(1, 2))

    # ---- lawns, quads and parks --------------------------------------------

    def lawn_with_paths(self, face, z, paths, path_surface="sidewalk", path_mat="pavers", path_w=PATH_W):
        """Lawn over the face with path strips on top; returns the path segments."""
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_grass_{face['id']}", "grass", inset, z - 0.1, z + 0.06, "grass", coll="Blocks")
        for p0, p1 in paths:
            self.slab(f"surface_{path_surface}_path_{face['id']}", path_surface, strip_poly(p0, p1, path_w), z - 0.05,
                      z + 0.08, path_mat, coll="Blocks")
        return paths

    def near_path(self, p, paths, d):
        return any(seg_dist(p, a, b) < d for a, b in paths)

    def build_quad(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        c = self.centre(face)
        mids = [add(inset[k], mul(sub(inset[(k + 1) % 4], inset[k]), 0.5)) for k in range(4)]
        style = self.rng.random()
        if style < 0.4:
            paths = [(inset[0], inset[2]), (inset[1], inset[3])]
        elif style < 0.75:
            paths = [(mids[0], mids[2]), (mids[1], mids[3])]
        else:
            paths = [(inset[0], inset[2]), (mids[1], mids[3]), (mids[0], c)]
        self.lawn_with_paths(face, z, paths)
        face["paths"] = [[list(a), list(b)] for a, b in paths]
        zp = z + 0.08
        # Round plaza with a statue or sculpture where the paths cross.
        self.slab(f"surface_plaza_quadcentre_{face['id']}", "plaza", circle(c, 3.6, 16), z - 0.05, z + 0.085, "pavers")
        if self.rng.random() < 0.5:
            self.place_world(px.make_statue(), c[0], c[1], zp, self.rng.uniform(0, math.pi), "surface_obstacle_statue", gap=0.6)
        else:
            self.place_world(px.make_sculpture(self.rng), c[0], c[1], zp, 0.0, "surface_obstacle_sculpture", gap=0.6)
        # Benches beside the paths, lamps along them.
        for a, b in paths:
            u = norm(sub(b, a))
            n = left(u)
            length = math.hypot(*sub(b, a))
            s = 6.0
            while s < length - 6.0:
                if abs(s - length / 2) > 5.0:
                    side = self.rng.choice((-1, 1))
                    p = add(add(a, mul(u, s)), mul(n, side * (PATH_W / 2 + 0.55)))
                    roll = self.rng.random()
                    if roll < 0.35:
                        self.bench(p, z + 0.06, angle_of(u) + (0.0 if side > 0 else math.pi), sit=0.45)
                    elif roll < 0.55:
                        self.lamp(p[0], p[1], z + 0.06, angle_of(u) + (-math.pi / 2 if side > 0 else math.pi / 2))
                s += self.rng.uniform(6.0, 9.0)
        # Trees on the lawn, away from the paths.
        for _ in range(14):
            p = (c[0] + self.rng.uniform(-18, 18), c[1] + self.rng.uniform(-16, 16))
            if self.in_face(face, p, 2.5) and not self.near_path(p, paths, 3.2) and math.hypot(*sub(p, c)) > 6.0:
                self.add_tree(p[0], p[1], z + 0.06, "surface_obstacle_tree_quad", style=self.rng.choice(("round", "round", "tall")))
        # Students: groups on the lawn, walkers on the paths, a picnic table, a dog.
        for _ in range(self.rng.randint(2, 4)):
            for _attempt in range(12):
                p = (c[0] + self.rng.uniform(-14, 14), c[1] + self.rng.uniform(-12, 12))
                if self.in_face(face, p, 3.0) and not self.near_path(p, paths, 2.8) and self.add_group(p[0], p[1], z + 0.06):
                    break
        for a, b in paths:
            for _ in range(self.rng.randint(1, 3)):
                t = self.rng.uniform(0.15, 0.85)
                p = add(a, mul(sub(b, a), t))
                self.add_person(p[0], p[1], zp, angle_of(sub(b, a)) + self.rng.choice((0.0, math.pi)),
                                pose="walking" if self.rng.random() < 0.8 else "standing")
        if self.rng.random() < 0.5:
            for _attempt in range(10):
                p = (c[0] + self.rng.uniform(-14, 14), c[1] + self.rng.uniform(-12, 12))
                if self.in_face(face, p, 3.0) and not self.near_path(p, paths, 3.5):
                    if self.place_world(px.make_picnic_table(), p[0], p[1], z + 0.06, self.rng.uniform(0, math.pi),
                                        "surface_obstacle_picnic_table", gap=0.6):
                        break
        self.add_animals_at([(c[0] + self.rng.uniform(-12, 12), c[1] + self.rng.uniform(-12, 12), z + 0.06)])

    def park_paths(self, face, z, surface="gravel_path", mat="gravel"):
        """Winding gravel paths from edge midpoints, each with one bend."""
        inset = [tuple(p) for p in face["inset"]]
        c = self.centre(face)
        mids = [add(inset[k], mul(sub(inset[(k + 1) % 4], inset[k]), self.rng.uniform(0.35, 0.65))) for k in range(4)]
        hub = (c[0] + self.rng.uniform(-6, 6), c[1] + self.rng.uniform(-5, 5))
        paths = []
        for k in self.rng.sample(range(4), self.rng.choice((3, 4))):
            bend = add(add(mids[k], mul(sub(hub, mids[k]), 0.5)), (self.rng.uniform(-4, 4), self.rng.uniform(-4, 4)))
            paths += [(mids[k], bend), (bend, hub)]
        self.lawn_with_paths(face, z, paths, path_surface=surface, path_mat=mat, path_w=2.6)
        face["paths"] = [[list(a), list(b)] for a, b in paths]
        self.slab(f"surface_{surface}_hub_{face['id']}", surface, circle(hub, 2.2, 12), z - 0.05, z + 0.085, mat)
        return paths, hub

    def build_park(self, face, z, kind):
        paths, hub = self.park_paths(face, z)
        c = self.centre(face)
        zg = z + 0.06
        trees = {"woods": 34, "meadow": 7, "playground": 10}[kind]
        for _ in range(trees * 2):
            if trees <= 0:
                break
            p = (c[0] + self.rng.uniform(-19, 19), c[1] + self.rng.uniform(-17, 17))
            if self.in_face(face, p, 1.5) and not self.near_path(p, paths, 2.6) and math.hypot(*sub(p, hub)) > 4.0:
                style = self.rng.choice(("conifer", "round", "tall")) if kind == "woods" else "round"
                if self.add_tree(p[0], p[1], zg, "surface_obstacle_tree_park", style=style, reserve_gap=0.6):
                    trees -= 1
        for _ in range(10 if kind == "woods" else 4):
            p = (c[0] + self.rng.uniform(-18, 18), c[1] + self.rng.uniform(-16, 16))
            if self.in_face(face, p, 1.0) and not self.near_path(p, paths, 2.0):
                self.add_bush(p[0], p[1], zg)
        # Few lamps (dark at night), benches along the paths.
        for idx, (a, b) in enumerate(paths):
            u = norm(sub(b, a))
            n = left(u)
            mid = add(a, mul(sub(b, a), 0.5))
            side = self.rng.choice((-1, 1))
            p = add(mid, mul(n, side * 1.9))
            if idx % 3 == 0 and kind != "woods":
                self.lamp(p[0], p[1], zg, angle_of(n) + (math.pi if side > 0 else 0.0))
            elif self.rng.random() < 0.5:
                self.bench(p, zg, angle_of(u) + (0.0 if side > 0 else math.pi), sit=0.35)
        if kind == "playground":
            pad = [add(hub, (dx, dy)) for dx, dy in ((4.0, 3.0), (12.0, 3.0), (12.0, 10.0), (4.0, 10.0))]
            if all(self.in_face(face, p, 1.0) for p in pad):
                self.slab(f"surface_playground_{face['id']}", "playground", pad, z - 0.05, z + 0.09, "rubber")
                pc = add(hub, (8.0, 6.5))
                self.place_world(px.make_swings(), pc[0] - 2.0, pc[1], z + 0.09, 0.0, "surface_obstacle_swings", gap=0.5)
                self.place_world(px.make_slide(), pc[0] + 2.4, pc[1], z + 0.09, math.pi / 2, "surface_obstacle_slide", gap=0.5)
                for _ in range(3):
                    self.add_person(pc[0] + self.rng.uniform(-3, 3), pc[1] + self.rng.uniform(-2.5, 2.5), z + 0.09,
                                    self.rng.uniform(0, 2 * math.pi), look=None, reserve=True)
        if kind == "meadow":
            for _ in range(3):
                for _attempt in range(10):
                    p = (c[0] + self.rng.uniform(-14, 14), c[1] + self.rng.uniform(-12, 12))
                    if self.in_face(face, p, 3.0) and not self.near_path(p, paths, 3.0):
                        if self.place_world(px.make_picnic_table(), p[0], p[1], zg, self.rng.uniform(0, math.pi),
                                            "surface_obstacle_picnic_table", gap=0.8):
                            break
            for _ in range(2):
                for _attempt in range(10):
                    p = (c[0] + self.rng.uniform(-14, 14), c[1] + self.rng.uniform(-12, 12))
                    if self.in_face(face, p, 3.0) and not self.near_path(p, paths, 2.5) and self.add_group(p[0], p[1], zg):
                        break
        for a, b in paths[: self.rng.randint(1, 4)]:
            p = add(a, mul(sub(b, a), self.rng.uniform(0.2, 0.8)))
            self.add_person(p[0], p[1], z + 0.08, angle_of(sub(b, a)) + self.rng.choice((0.0, math.pi)), pose="walking")
        self.add_animals_at([(c[0] + self.rng.uniform(-12, 12), c[1] + self.rng.uniform(-12, 12), zg) for _ in range(2)])

    # ---- library, sports, parking, construction ---------------------------------

    def build_library(self, face, z):
        """Large building on a plinth: wide stairs across the front, a ramp along the side (two paths up)."""
        self.slab(f"surface_plaza_{face['id']}", "plaza", [tuple(p) for p in face["inset"]], z - 0.1, z + 0.02, "pavers")
        zt = z + 0.02
        k = self.rng.randrange(4)
        length = self.edge_len(face, k)
        depth_face = min(self.edge_len(face, (k + 1) % 4), self.edge_len(face, (k + 3) % 4))
        steps = 6
        rise = steps * cl.RISE
        top = zt + rise
        d_plinth0 = depth_face * 0.42
        d_build0 = d_plinth0 + 5.0
        d_build1 = depth_face - 2.5
        s0, s1 = 3.0, length - 3.0
        plinth = self.fquad(face, k, s0, s1, d_plinth0, d_build1)
        self.slab(f"surface_terrace_plinth_{face['id']}", "terrace", plinth, z - 0.2, top, "stone", side="wall_stone")
        building = self.fquad(face, k, s0 + 2.0, s1 - 2.0, d_build0, d_build1)
        self.campus_building_poly(building, top, self.rng.randint(3, 4), self.rng.choice(("facade_light", "concrete", "facade_brick")),
                                  f"library_{face['id']}")
        st_s0, st_s1 = length * 0.5 - 4.0, length * 0.5 + 4.0

        def stairs(bm):
            for j in range(steps):
                d1 = d_plinth0 - j * cl.TREAD
                bm_prism(bm, self.fquad(face, k, st_s0, st_s1, d1 - cl.TREAD, d1), z - 0.2, top - (j + 1) * rise / steps + 0.0)

        stairs_obj = add_bmesh_object(self.next_name("surface_stairs_library"), "stairs", stairs, [self.mat["stairs"]],
                                      self.coll["Blocks"], category="stairs")
        # Ramp along the plinth front, rising toward the stairs from the west end.
        run = rise / cl.RAMP_SLOPE
        r_hi = st_s0 - 1.0
        r_lo = r_hi - run
        ramp_obj = None
        if r_lo > s0 + 0.5:
            poly = self.fquad(face, k, r_lo, r_hi, d_plinth0 - 1.6, d_plinth0)
            ramp_obj = add_bmesh_object(self.next_name("surface_ramp_library"), "ramp",
                                        lambda bm: bm_prism(bm, poly, z - 0.2, [zt, top, top, zt]), [self.mat["ramp"]],
                                        self.coll["Blocks"], category="ramp")
            self.railing("surface_obstacle_railing_ramp", self.fpt(face, k, r_lo + 0.8, d_plinth0 - 1.65), zt + 0.1,
                         self.fpt(face, k, r_hi, d_plinth0 - 1.65), top)
        # Door and goal on the plinth.
        door = add_bmesh_object(self.next_name(f"surface_door_{face['id']}"), "building",
                                lambda bm: bm_obox(bm, *self.frame(face, k)[:2], length * 0.5 - 1.2, length * 0.5 + 1.2,
                                                   d_build0 - 0.06, d_build0 + 0.02, top, top + 3.0),
                                [self.mat["door"]], self.coll["Buildings"], category="door")
        goal = self.fpt(face, k, length * 0.5, d_plinth0 + 2.2)
        door["front"] = list(goal)
        door["stairs"] = stairs_obj.name
        door["ramp"] = ramp_obj.name if ramp_obj else ""
        if ramp_obj is not None:
            self.layout["platforms"].append({"kind": "library_terrace", "stairs": stairs_obj.name, "ramp": ramp_obj.name,
                                             "goal": [round(goal[0], 3), round(goal[1], 3)], "z": top})
        rect_pts = plinth + [self.fpt(face, k, r_lo - 1.0, d_plinth0 - steps * cl.TREAD - 1.5)]
        self.keep_clear((min(p[0] for p in rect_pts), max(p[0] for p in rect_pts),
                         min(p[1] for p in rect_pts), max(p[1] for p in rect_pts)))
        self.layout["entrances"].append({"door": door.name, "goal": list(goal), "z": top, "stairs": stairs_obj.name,
                                         "ramp": door["ramp"], "facing": list(norm((-self.frame(face, k)[2][0], -self.frame(face, k)[2][1]))),
                                         "xy": list(self.fpt(face, k, length * 0.5, d_plinth0))})
        # People on the plinth and the steps.
        for _ in range(self.rng.randint(2, 5)):
            p = self.fpt(face, k, self.rng.uniform(s0 + 2, s1 - 2), self.rng.uniform(d_plinth0 + 0.8, d_build0 - 0.8))
            self.add_person(p[0], p[1], top, self.rng.uniform(0, 2 * math.pi))
        # Front plaza: fountain, food truck with a queue, benches, bikes, groups.
        fp = self.fpt(face, k, length * 0.5, d_plinth0 * 0.45)
        self.place_world(px.make_fountain(), fp[0], fp[1], zt, 0.0, "surface_obstacle_fountain", gap=1.0)
        _a, u, m = self.frame(face, k)
        tp = self.fpt(face, k, length * 0.8, d_plinth0 * 0.4)
        paint = self.rng.choice(("paint_red", "paint_blue", "paint_yellow", "paint_green"))
        # Serving hatch (local -Y) toward the fountain side.
        truck = self.place_world(px.make_food_truck(paint), tp[0], tp[1], zt, angle_of(u) + math.pi,
                                 "surface_obstacle_food_truck", gap=0.6)
        if truck is not None:
            truck["color_name"] = paint.split("_", 1)[1]
            out = (m[0], m[1])  # local -Y of yaw (u + pi) is +m: the hatch faces into the face
            n = self.rng.randint(3, 5)
            for q in range(n):
                qp = add(tp, mul(out, 2.0 + q * 0.75))
                self.add_person(qp[0], qp[1], zt, angle_of((-out[0], -out[1])), reserve=False)
            end = add(tp, mul(out, 2.0 + (n - 1) * 0.75 + 0.9))
            truck["queue_end"] = list(end)
            q0, q1 = add(tp, mul(out, 1.4)), end
            self.add_zone("queue", strip_poly(q0, q1, 1.6))
            self.keep_clear((min(q0[0], q1[0]) - 0.9, max(q0[0], q1[0]) + 0.9, min(q0[1], q1[1]) - 0.9, max(q0[1], q1[1]) + 0.9))
        for s in (length * 0.2, length * 0.35):
            p = self.fpt(face, k, s, d_plinth0 * 0.5)
            self.bench(p, zt, angle_of(u) + math.pi / 2, sit=0.5)
        self.groups(face, zt, 2, spread=7.0)
        self.walkers(face, zt, 5, region=[tuple(p) for p in face["inset"]])

    def build_sports(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_running_track_{face['id']}", "running_track", inset, z - 0.1, z + 0.04, "track")
        k = 0
        length = self.edge_len(face, k)
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3))
        field = self.fquad(face, k, 7.0, length - 7.0, 7.0, depth - 7.0)
        self.slab(f"surface_sports_field_{face['id']}", "sports_field", field, z - 0.05, z + 0.06, "field")
        _a, u, m = self.frame(face, k)
        for s in (8.2, length - 8.2):
            p = self.fpt(face, k, s, depth / 2)
            self.place_world(make_goal_frame(), p[0], p[1], z + 0.06, angle_of(u) + (0.0 if s < length / 2 else math.pi),
                             "surface_obstacle_goal", gap=0.3)
        # Bleachers along the far side and a fence around the outside with two gates.
        bp = self.fpt(face, k, length / 2, depth - 2.4)
        self.place_world(make_bleachers(min(24.0, length - 14.0)), bp[0], bp[1], z + 0.04, angle_of(u), "surface_obstacle_bleachers",
                         gap=0.3)
        for kk in range(4):
            a, uu, mm = self.frame(face, kk)
            el = self.edge_len(face, kk)
            gate = el * 0.5
            for s0, s1 in ((0.6, gate - 2.0), (gate + 2.0, el - 0.6)):
                if s1 - s0 > 1.0:
                    p0, p1 = self.fpt(face, kk, s0, 0.3), self.fpt(face, kk, s1, 0.3)
                    fence = props.make_fence(math.hypot(*sub(p1, p0)), height=1.2)
                    mid = add(p0, mul(sub(p1, p0), 0.5))
                    self.place_world(fence, mid[0], mid[1], z + 0.04, angle_of(uu), "surface_obstacle_fence", gap=0.0,
                                     reserve=False)
        for _ in range(8):
            t = self.rng.random()
            p = self.fpt(face, k, 3.0 + t * (length - 6.0), self.rng.choice((3.2, depth - 4.0)))
            self.add_person(p[0], p[1], z + 0.04, angle_of(u) + self.rng.choice((0.0, math.pi)), pose="walking")
        for _ in range(2):
            fc = self.fpt(face, k, length * self.rng.uniform(0.3, 0.7), depth * self.rng.uniform(0.35, 0.65))
            self.add_group(fc[0], fc[1], z + 0.06)

    def build_parking(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        lot = self.fquad(face, 0, 3.0, self.edge_len(face, 0) - 3.0, 3.0,
                         min(self.edge_len(face, 1), self.edge_len(face, 3)) - 3.0)
        self.slab(f"surface_plaza_{face['id']}", "plaza", inset, z - 0.1, z + 0.02, "plaza")
        self.slab(f"surface_parking_lot_{face['id']}", "parking_lot", lot, z - 0.05, z + 0.03, "road")
        _a, u, m = self.frame(face, 0)
        length = self.edge_len(face, 0) - 6.0
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3)) - 6.0
        rows = [d for d in (5.5, 11.5, depth - 5.5) if d < depth - 3.0]
        for d in rows:
            s = 5.5
            while s < length + 1.0:
                p = self.fpt(face, 0, s, d)
                if self.rng.random() < 0.75:
                    paint = self.rng.choice(CAR_PAINTS)
                    car = self.place_world(props.make_car(paint), p[0], p[1], z + 0.03, angle_of(m) + self.rng.choice((0, math.pi)),
                                           "surface_obstacle_car", gap=0.15)
                    if car is not None:
                        car["color_name"] = paint.split("_", 1)[1]
                s += 2.7
        pm = self.fpt(face, 0, 4.0, 2.0)
        self.place_world(px.make_vending_machine(), pm[0], pm[1], z + 0.02, angle_of(u), "surface_obstacle_vending_machine")
        for _ in range(4):
            p = self.fpt(face, 0, self.rng.uniform(4, length), self.rng.uniform(4, depth))
            self.add_person(p[0], p[1], z + 0.03, self.rng.uniform(0, 2 * math.pi), pose="walking")

    def build_construction(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_plaza_{face['id']}", "plaza", inset, z - 0.1, z + 0.02, "plaza")
        zt = z + 0.02
        site = self.fquad(face, 0, 6.0, self.edge_len(face, 0) - 6.0, 6.0, min(self.edge_len(face, 1), self.edge_len(face, 3)) - 6.0)
        self.slab(f"surface_soil_site_{face['id']}", "grass", site, z - 0.05, z + 0.03, "soil")
        for kk in range(4):
            p0, p1 = site[kk], site[(kk + 1) % 4]
            mid = add(p0, mul(sub(p1, p0), 0.5))
            self.place_world(props.make_hoarding(math.hypot(*sub(p1, p0)) - 0.4), mid[0], mid[1], z + 0.03,
                             angle_of(sub(p1, p0)), "surface_obstacle_hoarding", gap=0.0)
        self.keep_clear((min(p[0] for p in site), max(p[0] for p in site), min(p[1] for p in site), max(p[1] for p in site)))
        # Barriers and cones narrowing the walk around the site, a pile of materials, a detour sign.
        for kk in range(4):
            a, uu, mm = self.frame(face, kk)
            el = self.edge_len(face, kk)
            for s in (el * 0.3, el * 0.6):
                p = self.fpt(face, kk, s, 2.6)
                if self.rng.random() < 0.6:
                    self.place_world(props.make_barrier(), p[0], p[1], zt, angle_of(uu), "surface_obstacle_barrier", gap=0.3)
                else:
                    for c in range(3):
                        q = self.fpt(face, kk, s + c * 0.9, 2.4)
                        self.place_world(props.make_cone(), q[0], q[1], zt, 0.0, "surface_obstacle_cone", gap=0.1)
        p = self.fpt(face, 1, 3.0, 3.0)
        self.place_world(props.make_crate(1.8, 1.2, 0.9), p[0], p[1], zt, 0.3, "surface_obstacle_construction_material")
        self.walkers(face, zt, 3, region=inset)

    # ---- waterfront faces ----------------------------------------------------

    def build_boathouse(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_plaza_{face['id']}", "plaza", inset, z - 0.1, z + 0.02, "pavers")
        zt = z + 0.02
        k = 2  # north edge: the building backs onto the street behind
        length = self.edge_len(face, k)
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3))
        poly = self.fquad(face, k, 3.0, length - 3.0, 1.5, min(14.0, depth * 0.5))
        self.campus_building_poly(poly, zt, 2, self.rng.choice(("facade_brick", "facade_beige", "wood")), face["id"])
        self.keep_clear((min(p[0] for p in poly), max(p[0] for p in poly), min(p[1] for p in poly), max(p[1] for p in poly)))
        # Door facing the water, and a cafe terrace toward the promenade.
        a, u, m = self.frame(face, k)
        d_door = min(14.0, depth * 0.5)
        mid = length * 0.5
        door = add_bmesh_object(self.next_name(f"surface_door_{face['id']}"), "building",
                                lambda bm: bm_obox(bm, a, u, mid - 0.8, mid + 0.8, d_door - 0.02, d_door + 0.06, zt, zt + 2.5),
                                [self.mat["door"]], self.coll["Buildings"], category="door")
        door["front"] = list(self.fpt(face, k, mid, d_door + 1.2))
        self.add_cafe(face, 0, zt)
        self.groups(face, zt, 1, spread=6.0)
        self.walkers(face, zt, 4, region=inset)

    def build_harbor_square(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_plaza_{face['id']}", "plaza", inset, z - 0.1, z + 0.02, "pavers")
        zt = z + 0.02
        c = self.centre(face)
        self.place_world(px.make_fountain(), c[0], c[1], zt, 0.0, "surface_obstacle_fountain", gap=1.0)
        self.add_market(face, 2, zt)
        paint = self.rng.choice(("paint_blue", "paint_green", "paint_red"))
        kp = self.fpt(face, 1, self.edge_len(face, 1) * 0.5, 3.0)
        self.place_world(px.make_kiosk(paint), kp[0], kp[1], zt, angle_of(self.frame(face, 1)[1]), "surface_obstacle_kiosk", gap=0.6)
        for kk in (1, 3):
            el = self.edge_len(face, kk)
            for s in (el * 0.25, el * 0.75):
                p = self.fpt(face, kk, s, 2.5)
                self.add_tree(p[0], p[1], zt, "surface_obstacle_tree_square", style="round", reserve_gap=0.4)
                q = self.fpt(face, kk, s + 2.2, 2.5)
                self.bench(q, zt, angle_of(self.frame(face, kk)[1]) + math.pi, sit=0.4)
        self.groups(face, zt, 3, spread=8.0)
        self.walkers(face, zt, 7, region=inset)
        self.add_animals_at([(c[0] + self.rng.uniform(-8, 8), c[1] + self.rng.uniform(-8, 8), zt)])

    # ---- the slope -----------------------------------------------------------

    def build_terrace_garden(self, face, z):
        """Two levels inside the face (z and H/2) and the upper campus behind: stairs and ramps between them."""
        inset = [tuple(p) for p in face["inset"]]
        mid_z = self.H / 2
        k = 0  # south edge
        length = self.edge_len(face, k)
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3))
        d_terr = depth * 0.45
        lower = self.fquad(face, k, 0.0, length, 0.0, d_terr)
        terrace = self.fquad(face, k, 0.0, length, d_terr, depth)
        self.slab(f"surface_grass_{face['id']}", "grass", lower, z - 0.1, z + 0.06, "grass")
        self.slab(f"surface_terrace_{face['id']}", "terrace", terrace, z - 0.2, mid_z, "terrace", side="wall_stone")
        steps = round(mid_z / cl.RISE)
        rise = mid_z / steps
        s_st = length * self.rng.uniform(0.3, 0.45)

        def stairs(bm):
            for j in range(steps):
                d1 = d_terr - j * cl.TREAD
                bm_prism(bm, self.fquad(face, k, s_st - 1.6, s_st + 1.6, d1 - cl.TREAD, d1), z - 0.2, mid_z - (j + 1) * rise + 0.0)

        stairs_obj = add_bmesh_object(self.next_name("surface_stairs_garden"), "stairs", stairs, [self.mat["stairs"]],
                                      self.coll["Blocks"], category="stairs")
        run = mid_z / cl.RAMP_SLOPE
        r_hi = s_st + 3.0
        r_lo = min(length - 1.0, r_hi + run)
        poly = self.fquad(face, k, r_hi, r_lo, d_terr - 1.6, d_terr)
        ramp_obj = add_bmesh_object(self.next_name("surface_ramp_garden"), "ramp",
                                    lambda bm: bm_prism(bm, poly, z - 0.2, [mid_z, z + 0.06, z + 0.06, mid_z]), [self.mat["ramp"]],
                                    self.coll["Blocks"], category="ramp")
        self.railing("surface_obstacle_railing_ramp", self.fpt(face, k, r_hi, d_terr - 1.65), mid_z,
                     self.fpt(face, k, r_lo - 0.8, d_terr - 1.65), z + 0.15)
        # A path across the lower lawn to the stairs and the ramp foot.
        self.slab(f"surface_sidewalk_gardenpath_{face['id']}", "sidewalk",
                  strip_poly(self.fpt(face, k, s_st, 0.0), self.fpt(face, k, s_st, d_terr - steps * cl.TREAD), 2.6),
                  z - 0.05, z + 0.08, "pavers")
        goal = self.fpt(face, k, (s_st + r_hi) / 2, d_terr + 2.5)
        self.layout["platforms"].append({"kind": "terrace", "stairs": stairs_obj.name, "ramp": ramp_obj.name,
                                         "goal": [round(goal[0], 3), round(goal[1], 3)], "z": mid_z})
        self.keep_clear(self.rect_of(self.fquad(face, k, s_st - 2.0, r_lo + 0.5, d_terr - steps * cl.TREAD - 1.5, d_terr + 1.0)))
        # From the terrace up to the campus: stairs against the wall at the far end.
        k2 = 2
        l2 = self.edge_len(face, k2)
        s2 = l2 * self.rng.uniform(0.3, 0.7)
        steps2 = round((self.H - mid_z) / cl.RISE)
        rise2 = (self.H - mid_z) / steps2

        def stairs2(bm):
            for j in range(steps2):
                d0 = j * cl.TREAD
                bm_prism(bm, self.fquad(face, k2, s2 - 1.5, s2 + 1.5, d0, d0 + cl.TREAD), z - 0.2, self.H - j * rise2)

        add_bmesh_object(self.next_name("surface_stairs_garden_up"), "stairs", stairs2, [self.mat["stairs"]],
                         self.coll["Blocks"], category="stairs")
        gp = [self.fpt(face, k2, s2 - 1.8, 0.0), self.fpt(face, k2, s2 + 1.8, 0.0)]
        self.wall_gaps.append((min(p[0] for p in gp), max(p[0] for p in gp)))
        self.keep_clear(self.rect_of(self.fquad(face, k2, s2 - 2.0, s2 + 2.0, 0.0, steps2 * cl.TREAD + 1.5)))
        # Planting, benches and people on both levels.
        c = self.centre(face)
        for _ in range(8):
            p = self.fpt(face, k, self.rng.uniform(2, length - 2), self.rng.uniform(2, d_terr - 3))
            if not self.near_path(p, [(self.fpt(face, k, s_st, 0.0), self.fpt(face, k, s_st, d_terr))], 2.5):
                self.add_bush(p[0], p[1], z + 0.06) if self.rng.random() < 0.6 else self.add_tree(p[0], p[1], z + 0.06,
                                                                                                    "surface_obstacle_tree_garden")
        for _ in range(4):
            p = self.fpt(face, k, self.rng.uniform(3, length - 3), self.rng.uniform(d_terr + 2, depth - 3))
            self.bench(p, mid_z, angle_of(self.frame(face, k)[1]), sit=0.5)
        for _ in range(4):
            p = self.fpt(face, k, self.rng.uniform(3, length - 3), self.rng.uniform(d_terr + 1.5, depth - 2))
            self.add_person(p[0], p[1], mid_z, self.rng.uniform(0, 2 * math.pi), pose=self.rng.choice(("walking", "standing")))
        if self.rng.random() < 0.6:
            self.add_group(*self.fpt(face, k, self.rng.uniform(5, length - 5), self.rng.uniform(d_terr + 3, depth - 3)), mid_z)
        del c

    def rect_of(self, pts):
        return (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))

    def build_amphitheater(self, face, z):
        """Seating tiers from a stage up to the campus level, aisle stairs and a ramp beside them."""
        k = 0
        length = self.edge_len(face, k)
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3))
        tiers = 6
        tier_h = self.H / tiers
        steps = round(self.H / cl.RISE)
        rise = self.H / steps
        run = steps * cl.TREAD
        tier_d = run / tiers
        d_top = depth            # the north edge meets the wall top
        d_bottom = d_top - run
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_plaza_{face['id']}", "plaza", self.fquad(face, k, 0.0, length, 0.0, d_bottom), z - 0.1, z + 0.02, "pavers")
        ramp_w = 2.4
        s_ramp1 = length - 1.0
        s_ramp0 = s_ramp1 - ramp_w
        s_seat0, s_seat1 = 1.0, s_ramp0 - 1.2
        aisles = [s_seat0 + 1.0, (s_seat0 + s_seat1) / 2, s_seat1 - 1.0]

        bounds = [s_seat0] + [v for a in aisles for v in (a - 0.7, a + 0.7)] + [s_seat1]
        blocks = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2) if bounds[i + 1] - bounds[i] > 0.3]

        def seats(bm):
            for sa, sb in blocks:
                for t in range(tiers):
                    d0 = d_bottom + t * tier_d
                    bm_prism(bm, self.fquad(face, k, sa, sb, d0, d0 + tier_d), z - 0.2, z + (t + 1) * tier_h)

        add_bmesh_object(self.next_name("surface_obstacle_amph_seats"), "obstacle", seats, [self.mat["stone"]],
                         self.coll["Blocks"], category="barrier")["qa_skip"] = True

        def aisle_stairs(bm):
            for s in aisles:
                for j in range(steps):
                    d0 = d_bottom + j * cl.TREAD
                    bm_prism(bm, self.fquad(face, k, s - 0.7, s + 0.7, d0, d0 + cl.TREAD), z - 0.2, z + (j + 1) * rise + 0.002)

        stairs_obj = add_bmesh_object(self.next_name("surface_stairs_amph"), "stairs", aisle_stairs, [self.mat["stairs"]],
                                      self.coll["Blocks"], category="stairs")
        ramp_run = self.H / cl.RAMP_SLOPE
        r_d0 = max(1.0, d_top - ramp_run)
        poly = self.fquad(face, k, s_ramp0, s_ramp1, r_d0, d_top)
        ramp_obj = add_bmesh_object(self.next_name("surface_ramp_amph"), "ramp",
                                    lambda bm: bm_prism(bm, poly, z - 0.2, [z + 0.02, z + 0.02, self.H, self.H]),
                                    [self.mat["ramp"]], self.coll["Blocks"], category="ramp")
        for s_edge in (s_ramp0 - 0.05, s_ramp1 + 0.05):
            self.railing("surface_obstacle_railing_ramp", self.fpt(face, k, s_edge, r_d0 + 0.8), z + 0.12,
                         self.fpt(face, k, s_edge, d_top - 0.2), self.H)
        # Stage, people sitting on the tiers, a lecturer.
        stage = self.fquad(face, k, s_seat0 + 4.0, s_seat1 - 4.0, d_bottom - 6.0, d_bottom - 1.5)
        self.slab(f"surface_obstacle_stage_{face['id']}", "obstacle", stage, z + 0.02, z + 0.45, "wood", category="barrier")
        self.keep_clear(self.rect_of(self.fquad(face, k, 0.0, length, d_bottom - 0.2, d_top + 1.0)))
        _a, u, m = self.frame(face, k)
        facing = angle_of((-m[0], -m[1]))
        for t in range(tiers - 1):
            d_feet = d_bottom + t * tier_d + tier_d * 0.35
            for _ in range(self.rng.randint(2, 6)):
                s = self.rng.uniform(s_seat0 + 0.5, s_seat1 - 0.5)
                if min(abs(s - a) for a in aisles) < 1.2:
                    continue
                p = self.fpt(face, k, s, d_feet + tier_d * 0.3)
                person = self.add_person(p[0], p[1], z + t * tier_h + tier_h - 0.5, facing, pose="sitting", reserve=False)
                if person is not None:
                    person["qa_skip"] = True   # sits on the seating tiers, not on walkable ground
        lp = self.fpt(face, k, length * 0.4, d_bottom - 3.5)
        lecturer = self.add_person(lp[0], lp[1], z + 0.45, angle_of(m), reserve=False)
        if lecturer is not None:
            lecturer["qa_skip"] = True
        goal = self.fpt(face, k, (s_seat0 + s_seat1) / 2, d_top + 2.5)
        self.layout["platforms"].append({"kind": "amphitheater", "stairs": stairs_obj.name, "ramp": ramp_obj.name,
                                         "goal": [round(goal[0], 3), round(goal[1], 3)], "z": self.H})
        gp = [self.fpt(face, k, s_seat0 - 0.2, d_top), self.fpt(face, k, s_ramp1 + 0.2, d_top)]
        self.wall_gaps.append((min(p[0] for p in gp), max(p[0] for p in gp)))
        # Trees and benches on the front plaza.
        for s in (length * 0.15, length * 0.85):
            p = self.fpt(face, k, s, 3.0)
            self.add_tree(p[0], p[1], z + 0.02, "surface_obstacle_tree_square", style="round")
        self.walkers(face, z + 0.02, 4, region=self.fquad(face, k, 0.0, length, 0.0, d_bottom - 6.5))
        del inset

    def build_cafe_plaza(self, face, z):
        inset = [tuple(p) for p in face["inset"]]
        self.slab(f"surface_plaza_{face['id']}", "plaza", inset, z - 0.1, z + 0.02, "pavers")
        zt = z + 0.02
        self.add_cafe(face, 0, zt)
        self.add_cafe(face, 2, zt)
        c = self.centre(face)
        self.place_world(px.make_sculpture(self.rng), c[0], c[1], zt, 0.0, "surface_obstacle_sculpture", gap=0.8)
        for kk in (1, 3):
            el = self.edge_len(face, kk)
            p = self.fpt(face, kk, el * 0.5, 2.5)
            self.add_tree(p[0], p[1], zt, "surface_obstacle_tree_square", style="tall")
        self.groups(face, zt, 2, spread=6.0)
        self.walkers(face, zt, 6, region=inset)

    def build_lower_faculty(self, face, z):
        """Building built into the slope: stands on the waterfront level, its roof above the campus level."""
        self.slab(f"surface_plaza_{face['id']}", "plaza", [tuple(p) for p in face["inset"]], z - 0.1, z + 0.02, "plaza")
        zt = z + 0.02
        k0 = 0
        depth = min(self.edge_len(face, 1), self.edge_len(face, 3))
        l0 = self.edge_len(face, k0)
        d_front = depth * 0.45
        poly = self.fquad(face, k0, 0.0, l0, d_front, depth - 0.2)
        self.campus_building_poly(poly, zt, self.rng.randint(2, 3), self.rng.choice(FACADES), face["id"])
        self.keep_clear(self.rect_of(poly))
        # Entrances on the south facade (toward the water), with steps and a ramp.
        for s in ([l0 * 0.5] if l0 < 34 else [l0 * 0.3, l0 * 0.7]):
            self.campus_entrance_from_south(face, s, d_front, zt)
        c = self.centre(face)
        for _ in range(3):
            p = self.fpt(face, k0, self.rng.uniform(3, l0 - 3), self.rng.uniform(1.5, d_front - 4.5))
            self.add_tree(p[0], p[1], zt, "surface_obstacle_tree_court", style="round")
        self.walkers(face, zt, 4, region=self.fquad(face, k0, 0.0, l0, 0.0, d_front - 3.0))
        del c

    def campus_entrance_from_south(self, face, s, d_front, z):
        """Entrance on a facade that faces the south edge (k=0) at depth d_front from it."""
        return self.campus_entrance(face, 0, s, d_front, z, steps=4, ramp=self.rng.random() < 0.85)

    # ---- outer ring --------------------------------------------------------------

    def build_background(self):
        """Buildings beyond the outer cross streets and north of the road, and a taller ring at the world edge."""
        ex0, ex1, ey0, ey1 = self.layout["extent"]
        nav = self.layout["nav_extent"]
        road_half = cl.H_WIDTH[cl.NY - 1] / 2
        # North: a row of buildings behind the road's north sidewalk.
        top = [self.nodes[(i, cl.NY - 1)] for i in range(cl.NX)]
        x = ex0 + 2.0
        while x < ex1 - 8.0:
            w = self.rng.uniform(14.0, 24.0)
            yb = max(p[1] for p in top) + road_half + 2.5
            poly = [(x, yb), (x + w, yb), (x + w, yb + 14.0), (x, yb + 14.0)]
            self.campus_building_poly(poly, self.H, self.rng.randint(3, 6), self.rng.choice(FACADES), "bg")
            x += w + self.rng.uniform(2.0, 6.0)
        # West and east: buildings beyond the outer cross streets, on both levels.
        for i, sign in ((0, -1), (cl.NX - 1, 1)):
            for j in range(cl.NY - 1):
                a, b = self.nodes[(i, j)], self.nodes[(i, j + 1)]
                xo = (a[0] + b[0]) / 2 + sign * (3.0 + 2.5)
                y0, y1 = a[1] + 4.0, b[1] - 4.0
                z = self.H if j >= cl.WALL_LINE else 0.0
                if j == cl.WALL_LINE - 1:
                    continue  # the wall crosses this stretch
                x0, x1 = (xo - 14.0, xo) if sign < 0 else (xo, xo + 14.0)
                if j == 0:
                    y0 = max(y0, self.quay_y + cl.PROM_W + 1.0)
                if y1 - y0 > 6.0:
                    self.campus_building_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z, self.rng.randint(3, 5),
                                              self.rng.choice(FACADES), "bg")
        del nav, ey0, ey1

    # ---- people along the walkways --------------------------------------------

    def furnish_streets(self):
        streets = [s for s in self.layout["streets"] if not (s["axis"] == "h" and s["j"] in (0, cl.NY - 1))]

        def spot(seg, margin=1.0):
            a, b = tuple(seg["a"]), tuple(seg["b"])
            u = norm(sub(b, a))
            length = math.hypot(*sub(b, a))
            if length < 14.0:
                return None
            t = self.rng.uniform(6.0, length - 6.0)
            half = seg["width"] / 2 - margin
            off = self.rng.uniform(-half, half) if half > 0 else 0.0
            p = add(add(a, mul(u, t)), mul(left(u), off))
            if seg["axis"] == "v" and seg["j"] == cl.WALL_LINE - 1:
                return None   # stairs and ramps
            if seg["axis"] == "v" and seg["j"] == 0 and seg["i"] == cl.CANAL_I:
                return None   # the canal
            z = self.level_z(*p)
            return p[0], p[1], z, angle_of(u)

        for _ in range(170):
            sp = spot(self.rng.choice(streets))
            if sp:
                x, y, z, yaw = sp
                self.add_person(x, y, z, yaw + self.rng.choice((0.0, math.pi)) + self.rng.uniform(-0.3, 0.3),
                                pose="walking" if self.rng.random() < 0.78 else "standing")
        for _ in range(18):
            seg = self.rng.choice([s for s in streets if s["width"] >= 6.0])
            sp = spot(seg, margin=2.0)
            if sp:
                self.add_group(sp[0], sp[1], sp[2])
        for _ in range(40):
            seg = self.rng.choice(streets)
            sp = spot(seg, margin=0.6)
            if sp is None:
                continue
            x, y, z, yaw = sp
            roll = self.rng.random()
            if roll < 0.3:
                self.place_world(props.make_bike(), x, y, z, yaw, "surface_obstacle_bike", gap=0.3)
            elif roll < 0.5:
                paint = self.rng.choice(po.make_scooter_paints())
                obj = self.place_world(po.make_scooter(paint), x, y, z, yaw, "surface_obstacle_scooter", gap=0.3)
                if obj is not None:
                    obj["color_name"] = paint.split("_", 1)[1]
            elif roll < 0.75:
                self.place_world(props.make_trash_can(), x, y, z, 0.0, "surface_obstacle_trash_can")
            else:
                self.lamp(x, y, z, yaw)
        spots = [spot(self.rng.choice(streets)) for _ in range(24)]
        self.add_animals_at([s[:3] for s in spots if s])

    # ---- all -----------------------------------------------------------------------

    def build_faces(self):
        for face in self.faces.values():
            z = face["ground_z"]
            kind = face["kind"]
            if kind == "quad" or kind == "water_park":
                self.build_quad(face, z)
            elif kind == "faculty":
                if face["row"] == cl.WALL_LINE - 1:
                    self.build_lower_faculty(face, z)
                else:
                    self.build_faculty(face, z)
            elif kind == "library":
                self.build_library(face, z)
            elif kind == "sports":
                self.build_sports(face, z)
            elif kind == "parking":
                self.build_parking(face, z)
            elif kind == "construction":
                self.build_construction(face, z)
            elif kind in ("woods", "meadow", "playground"):
                self.build_park(face, z, kind)
            elif kind == "boathouse":
                self.build_boathouse(face, z)
            elif kind == "harbor_square":
                self.build_harbor_square(face, z)
            elif kind == "terrace_garden":
                self.build_terrace_garden(face, z)
            elif kind == "amphitheater":
                self.build_amphitheater(face, z)
            elif kind == "cafe_plaza":
                self.build_cafe_plaza(face, z)

    def build(self):
        self.build_ground()
        self.build_promenade()
        self.build_canal()
        self.build_wall_links()
        self.build_road()
        self.build_faces()          # slope features add their wall gaps before the parapet
        self.build_parapet()
        self.build_background()
        self.furnish_streets()
        print(f"campus people={self.people} lamps={self.lamps} zones={len(self.layout['social_zones'])} "
              f"platforms={len(self.layout['platforms'])} entrances={len(self.layout['entrances'])}")
