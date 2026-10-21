"""Turn one camera view into NaviTrace-style Human tasks (bpy).

For every visible target (bench, door, crosswalk, terrace, fountain, ...) plan
a Human path from the bottom center of the image, project it into the image,
and keep it when the path stays on screen and is mostly unoccluded. Each view
yields up to one "Goal" task ("Go to the fountain") and one "Directions" task
("Go straight for about 10 meters, then turn left and cross the bridge"),
whose wording is read off the planned path's turns and surfaces.

Targets are chosen against dataset-wide counts (``balance``), so common
targets such as street lamps stay rare and stairs, ramps, animals, and
crossings are favoured.
"""

import math
import random

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from . import layout as lay
from .classes import CLASS_ID, GROUND_TYPES
from .meshes import world_aabb
from .scene_setup import RES_X, RES_Y

MIN_DIST, MAX_DIST = 4.0, 32.0
MAX_OCCLUDED = 0.15
RDP_EPS_PX = 6.0
# NaviTrace traces have about 6-15 points; ours are resampled along the true projected path.
TRACE_MIN_PTS, TRACE_MAX_PTS, TRACE_PX_PER_PT = 6, 15, 70.0

GO = ["Go to the {noun}{where}", "Walk to the {noun}{where}"]
OBJECT_TARGETS = {
    "bench": GO + ["Go sit on the {noun}{where}"],
    "trash_can": GO + ["Walk over to the {noun}{where}"],
    "bike": GO,
    "planter": GO,
    "door": GO + ["Head to the {noun}{where}"],
    "street_lamp": ["Go to the {noun}{where}"],
    "tree": GO,
    "bush": GO + ["Walk over to the {noun}{where}"],
    "cat": ["Walk toward the {noun}{where}", "Go to the {noun}{where}", "Walk over to the {noun}{where}"],
    "dog": ["Walk toward the {noun}{where}", "Go to the {noun}{where}", "Walk up to the {noun}{where}"],
    "car": GO,
    "fountain": GO + ["Walk over to the {noun}{where}"],
    "statue": GO,
    "swings": GO + ["Go over to the {noun}{where}"],
    "slide": GO,
    "picnic_table": GO + ["Go sit at the {noun}{where}"],
    "kiosk": GO,
    "food_truck": GO + ["Walk up to the {noun}{where}"],
    "bus_stop": GO + ["Go wait at the {noun}{where}"],
    "bike_rack": GO,
    "fire_hydrant": GO,
    "vending_machine": GO,
    "recycling_bin": GO,
    "sculpture": GO,
    "mailbox": GO,
    "basketball_hoop": GO,
    "skate_ramp": GO,
    "wheelie_bin": GO,
}
BASE_NOUN = {
    "bench": "bench", "trash_can": "trash can", "bike": "parked bike", "planter": "planter",
    "door": "building entrance", "street_lamp": "street lamp", "tree": "tree", "bush": "bush",
    "cat": "cat", "dog": "dog", "car": "car", "fountain": "fountain", "statue": "statue",
    "swings": "swings", "slide": "slide", "picnic_table": "picnic table", "kiosk": "kiosk",
    "food_truck": "food truck", "bus_stop": "bus stop", "bike_rack": "bike rack",
    "fire_hydrant": "fire hydrant", "vending_machine": "vending machine", "recycling_bin": "recycling bins",
    "sculpture": "sculpture", "mailbox": "mailbox", "basketball_hoop": "basketball hoop",
    "skate_ramp": "skate ramp", "wheelie_bin": "garbage bin",
}
# Relative weight per target kind when balancing the dataset (higher = picked more often).
KIND_WEIGHT = {
    "street_lamp": 0.15, "tree": 0.6, "bush": 0.6, "car": 0.6, "trash_can": 0.7, "planter": 0.7,
    "terrace": 3.0, "platform": 3.0, "door": 1.6, "crosswalk": 1.6, "bridge": 2.0,
    "cat": 1.8, "dog": 1.8, "fountain": 1.5, "swings": 1.3, "slide": 1.3,
}
RELATION_M = 2.5          # landmark must be this close to the target to be "beside" it
RELATION_WORDS = (" beside the {}", " next to the {}", " near the {}")
RELATION_RATE = 0.35      # share of already-unique targets that still get a landmark phrase, for variety
DYNAMIC_KINDS = ("cat", "dog")
CROSSWALK_TASKS = ["Cross the street", "Cross the street at the crosswalk", "Get to the other side of the street"]
BRIDGE_TASKS = ["Cross the bridge", "Walk over the bridge", "Cross the pond on the bridge"]
TERRACE_TASKS = ["Go up to the terrace{where}", "Get onto the raised terrace{where}"]
PLATFORM_TASKS = ["Get up onto the platform{where}", "Go up to the top of the skate platform{where}"]
UPPER_KINDS = ("terrace", "platform")
TERRAIN_PHRASES = {
    "crosswalk": "cross the street at the crosswalk",
    "bridge": "cross the bridge",
    "stairs_up": "take the stairs up",
    "stairs_down": "take the stairs down",
    "ramp_up": "take the ramp up",
    "ramp_down": "take the ramp down",
}

VARIANT_CONTEXT = {
    "street": "Sidewalk along a street with a grass area behind it",
    "plaza": "Wide paved plaza in front of large buildings",
    "narrow": "Narrow sidewalk between parked cars and row houses",
    "construction": "Sidewalk next to a construction site",
    "terrace": "Sidewalk next to a raised terrace with stairs and a ramp",
    "neighborhood": "Residential street with houses, front lawns and driveways",
    "park": "Park with paths, trees, a fountain, a pond and a playground",
    "skate": "Skate park and basketball court",
}


def project(scene, cam, point):
    co = world_to_camera_view(scene, cam, Vector(point))
    return co.x * RES_X, (1.0 - co.y) * RES_Y, co.z


def in_frame(px, py, depth, margin=0.0):
    return depth > 0.05 and -margin <= px <= RES_X + margin and -margin <= py <= RES_Y + margin


def visible_from(scene, depsgraph, cam_loc, point, target_names=None, tol=0.25, vis=None):
    """True when nothing blocks the line from the camera to ``point`` (or the first hit is a target).

    ``vis`` (navmap.VisTree) answers the same question much faster than ``scene.ray_cast``.
    """
    direction = Vector(point) - cam_loc
    dist = direction.length
    if vis is not None:
        hit_dist, owner = vis.first_hit(cam_loc, direction.normalized(), dist)
        if hit_dist is None:
            return True
        if target_names and owner in target_names:
            return True
        return hit_dist >= dist - tol
    hit, loc, _n, _i, obj, _m = scene.ray_cast(depsgraph, cam_loc, direction.normalized(), distance=dist)
    if not hit:
        return True
    source = obj.original if hasattr(obj, "original") else obj
    if target_names and source.name in target_names:
        return True
    return (loc - cam_loc).length >= dist - tol


def rdp_indices(points, eps, lo=0, hi=None):
    """Indices of the Ramer-Douglas-Peucker simplification of points[lo..hi]."""
    hi = len(points) - 1 if hi is None else hi
    if hi - lo < 2:
        return [lo, hi]
    (x0, y0), (x1, y1) = points[lo], points[hi]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1e-9
    best_i, best_d = lo, -1.0
    for i in range(lo + 1, hi):
        px, py = points[i]
        d = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_i, best_d = i, d
    if best_d <= eps:
        return [lo, hi]
    return rdp_indices(points, eps, lo, best_i)[:-1] + rdp_indices(points, eps, best_i, hi)


def simplify_trace(dense):
    """Pixel trace from a dense projected path: keep the corners, then add points evenly along
    each stretch until the trace has a NaviTrace-like 6-15 points."""
    if len(dense) < 2:
        return [list(p) for p in dense]
    eps = RDP_EPS_PX
    corners = rdp_indices(dense, eps)
    while len(corners) > TRACE_MAX_PTS:
        eps *= 1.5
        corners = rdp_indices(dense, eps)
    seg = [math.hypot(dense[i + 1][0] - dense[i][0], dense[i + 1][1] - dense[i][1]) for i in range(len(dense) - 1)]
    cum = [0.0]
    for length in seg:
        cum.append(cum[-1] + length)
    total = cum[-1]
    target = max(len(corners), min(TRACE_MAX_PTS, max(TRACE_MIN_PTS, round(total / TRACE_PX_PER_PT) + 1)))
    extra = target - len(corners)

    def point_at(arc):
        i = max(0, min(len(seg) - 1, next((k for k in range(len(seg)) if cum[k + 1] >= arc), len(seg) - 1)))
        t = 0.0 if seg[i] == 0 else (arc - cum[i]) / seg[i]
        return (dense[i][0] + (dense[i + 1][0] - dense[i][0]) * t, dense[i][1] + (dense[i + 1][1] - dense[i][1]) * t)

    out = [dense[corners[0]]]
    spans = [(cum[a], cum[b]) for a, b in zip(corners, corners[1:])]
    shares = [(b - a) / total * extra if total > 0 else 0.0 for a, b in spans]
    counts = [int(s) for s in shares]
    for k in sorted(range(len(shares)), key=lambda k: shares[k] - counts[k], reverse=True)[: extra - sum(counts)]:
        counts[k] += 1
    for (a, b), n, end in zip(spans, counts, corners[1:]):
        for j in range(1, n + 1):
            out.append(point_at(a + (b - a) * j / (n + 1)))
        out.append(dense[end])
    return [[round(x, 2), round(y, 2)] for x, y in out]


def direction_word(px):
    if px < RES_X * 0.36:
        return " on the left"
    if px > RES_X * 0.64:
        return " on the right"
    return " ahead"


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def landmark_phrase(cand, group, landmarks, rng):
    """" beside the bench" when a landmark next to ``cand`` singles it out within ``group``, else None.

    A landmark of noun N works when no other object in the group has an N
    within RELATION_M, so the phrase points at exactly one object.
    """
    def near(obj, noun):
        return [l for l in landmarks if l["noun"] == noun and l["object"] != obj["object"]
                and math.hypot(l["xy"][0] - obj["xy"][0], l["xy"][1] - obj["xy"][1]) <= RELATION_M]

    options = []
    for lm in landmarks:
        if lm["kind"] == cand["kind"] or lm["object"] == cand["object"]:
            continue
        d = math.hypot(lm["xy"][0] - cand["xy"][0], lm["xy"][1] - cand["xy"][1])
        if d > RELATION_M:
            continue
        if len(near(cand, lm["noun"])) != 1:
            continue  # two of that landmark nearby would be ambiguous too
        if any(near(other, lm["noun"]) for other in group if other["object"] != cand["object"]):
            continue
        options.append((d, lm["noun"]))
    if not options:
        return None
    return rng.choice(RELATION_WORDS).format(min(options)[1])


def disambiguate(cands, landmarks, rng):
    """Keep only targets the task wording can single out, and say how.

    Targets are grouped by their noun ("black cat", "pine tree"). A target
    alone in its group can use the direction word, or sometimes a landmark
    ("the tree beside the bench"). In a group of several, a target stays only
    when a landmark or its direction word singles it out; failing both, only
    the nearest stays, worded "nearest to you".
    """
    by_noun = {}
    for cand in cands:
        by_noun.setdefault(cand["noun"], []).append(cand)
    out = []
    for group in by_noun.values():
        if len(group) == 1:
            cand = dict(group[0])
            phrase = landmark_phrase(cand, group, landmarks, rng) if rng.random() < RELATION_RATE else None
            if phrase:
                cand["where"] = phrase
            out.append(cand)
            continue
        words = [direction_word(c["px"]) for c in group]
        kept = False
        for cand, word in zip(group, words):
            phrase = landmark_phrase(cand, group, landmarks, rng)
            if phrase:
                out.append({**cand, "where": phrase})
                kept = True
            elif words.count(word) == 1:
                out.append(dict(cand))
                kept = True
        if not kept:
            out.append({**min(group, key=lambda c: c["dist"]), "where": " nearest to you"})
    return out


def noun_of(obj, lamps_on=False):
    category = obj.get("category")
    color = obj.get("color_name")
    if category in ("cat", "dog", "car", "food_truck") and color:
        return f"{color} {BASE_NOUN[category]}"
    if category == "tree" and obj.get("tree_style") == "conifer":
        return "pine tree"
    if category == "street_lamp" and lamps_on:
        return "lit street lamp"
    if category == "door" and obj.get("house_color"):
        return f"front door of the {obj['house_color']} house"
    return BASE_NOUN.get(category, category.replace("_", " "))


class ViewTasks:
    def __init__(self, scene, cam, planner, layout, rng=None, lighting=None, balance=None, vis=None):
        """``lighting``: preset info from lighting.Lighting.apply (None = day).
        ``balance``: dict of target-kind counts so far, shared across views (updated by the caller).
        ``vis``: a navmap.VisTree for fast line-of-sight checks (built once per run)."""
        self.vis = vis
        self.lighting = lighting or {"preset": "day", "lamps_on": False}
        self.lamps_on = self.lighting["lamps_on"]
        self.balance = balance if balance is not None else {}
        self.scene = scene
        self.cam = cam
        self.planner = planner
        self.layout = layout
        self.rng = rng or random.Random(cam.name)
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        # Read pose from the camera itself so cameras moved or added by hand in Blender work too.
        self.cam_loc = cam.matrix_world.translation.copy()
        forward = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
        self.heading = math.atan2(forward.y, forward.x)
        ground_z = planner.height_at(self.cam_loc.x, self.cam_loc.y)
        self.feet = (self.cam_loc.x, self.cam_loc.y, ground_z)
        self.block, _side = lay.block_side_at(layout, self.cam_loc.x, self.cam_loc.y)
        self.start_xy = self.bottom_center_ground()
        self.start = None
        if self.start_xy is not None:
            self.start = planner.nearest_walkable(self.start_xy[0], self.start_xy[1], radius=0.5, max_cost=1.6)

    def bottom_center_ground(self):
        """Ground point straight ahead that shows at the bottom center of the image.

        NaviTrace traces start at the bottom center, so paths are planned from
        here rather than from the camera's feet (which are below the frame).
        """
        dx, dy = math.cos(self.heading), math.sin(self.heading)
        for k in range(10, 200):
            s = k * 0.05
            x, y = self.feet[0] + dx * s, self.feet[1] + dy * s
            _px, py, depth = project(self.scene, self.cam, (x, y, self.planner.height_at(x, y)))
            if depth > 0 and py <= RES_Y - 1:
                return (x, y)
        return None

    # ---- trace from a world path --------------------------------------

    def dense_pixels(self, points):
        """Projected pixels of world points, or None if any point is off screen."""
        pix = [project(self.scene, self.cam, p) for p in points]
        # The start cell can sit a few pixels below the frame; anything more means it is not in view.
        if any(not (d > 0.05 and -2 <= px <= RES_X + 2 and -2 <= py <= RES_Y + 25) for px, py, d in pix):
            return None
        return [(min(max(px, 0.0), RES_X - 0.5), min(max(py, 0.0), RES_Y - 0.5)) for px, py, _d in pix]

    def trace(self, points, target_names=()):
        """Pixel trace for world points (starting at the bottom center), or (None, reason)."""
        dense = self.dense_pixels(points)
        if dense is None:
            return None, "leaves the frame"
        occluded = 0
        for (x, y, z) in points[1:]:
            if not visible_from(self.scene, self.depsgraph, self.cam_loc, (x, y, z + 0.05), target_names, vis=self.vis):
                occluded += 1
        if occluded > MAX_OCCLUDED * len(points):
            return None, f"occluded {occluded}/{len(points)}"
        if math.hypot(dense[-1][0] - dense[0][0], dense[-1][1] - dense[0][1]) < 40:
            return None, "too short on screen"
        return simplify_trace(dense), "ok"

    def plan(self, goal_xy, banned=()):
        """(points, cost, corners): resampled world path, its cost, and its smoothed corner points."""
        goal = self.planner.nearest_walkable(goal_xy[0], goal_xy[1], radius=1.5, max_cost=1.6, prefer=self.start_xy)
        if self.start is None or goal is None:
            return None, None, None
        cells = self.planner.astar(self.start, goal, banned_classes=banned)
        if cells is None:
            return None, None, None
        cells = self.planner.smooth(cells)
        corners = [self.planner.to_world(*c) for c in cells]
        return self.planner.to_points(cells), self.planner.path_cost(cells), corners

    # ---- candidates ----------------------------------------------------

    def object_candidates(self):
        """(targets, landmarks): visible objects that tasks can name, and those that can describe them."""
        targets, landmarks = [], []
        for obj in self.scene.objects:
            category = obj.get("category")
            if category not in OBJECT_TARGETS or obj.get("surface_type") == "overhead":
                continue
            box = world_aabb(obj)
            center = ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2, (box[4] + box[5]) / 2)
            dist = math.hypot(center[0] - self.feet[0], center[1] - self.feet[1])
            if dist > MAX_DIST + 5.0:
                continue
            px, py, depth = project(self.scene, self.cam, center)
            if not in_frame(px, py, depth, margin=-RES_X * 0.04):
                continue
            if not visible_from(self.scene, self.depsgraph, self.cam_loc, center, {obj.name}, vis=self.vis):
                continue
            cand = {"kind": category, "noun": noun_of(obj, self.lamps_on), "object": obj.name, "xy": center[:2],
                    "px": px, "dist": dist}
            landmarks.append(cand)
            if obj.get("in_traffic") or not MIN_DIST <= dist <= MAX_DIST:
                continue  # cars stopped in the lanes are landmarks, never goals (the goal would be on the road)
            if category == "door":
                goal = tuple(obj["front"])
                if obj.get("stairs") and obj.get("ramp"):
                    cand["alts"] = (obj["stairs"], obj["ramp"])  # porch: steps or ramp both lead up
            elif category == "bench":
                # In front of the seat (benches face local -Y), not behind the backrest.
                front = obj.matrix_world @ Vector((0.0, -0.8, 0.0))
                goal = (front.x, front.y)
            else:
                goal = center[:2]
            targets.append({**cand, "goal": goal})
        return targets, landmarks

    def crossing_candidates(self, category, max_near=18.0):
        """Crosswalks and bridges: the goal is the far end, and the path must use the crossing."""
        out = []
        for obj in self.scene.objects:
            if obj.get("category") != category or "end_a" not in obj.keys():
                continue
            a, b = tuple(obj["end_a"]), tuple(obj["end_b"])
            da = math.hypot(a[0] - self.feet[0], a[1] - self.feet[1])
            db = math.hypot(b[0] - self.feet[0], b[1] - self.feet[1])
            far = b if da < db else a
            if min(da, db) > max_near or max(da, db) > MAX_DIST:
                continue
            box = world_aabb(obj)
            center = ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2, box[5])
            px, py, depth = project(self.scene, self.cam, center)
            if not in_frame(px, py, depth, margin=-RES_X * 0.04):
                continue
            out.append({"kind": category, "object": obj.name, "goal": far, "px": px, "dist": max(da, db),
                        "noun": BASE_NOUN.get(category, category)})
        return out

    def upper_level_candidates(self):
        """Raised places reached by both stairs and a ramp: the terrace and the skate platforms."""
        out = []
        for block in self.layout["blocks"].values():
            terrace = block.get("terrace")
            if not terrace or "stairs_s" not in terrace:
                continue
            ring = lay.CURB_WIDTH + block["sidewalk_w"]
            depth = lay.N_STEPS * lay.TREAD
            s_mid = (terrace["stairs_s"][0] + terrace["ramp_s"][1]) / 2
            gx, gy = lay.side_point(block, terrace["side"], s_mid, ring + depth + 2.5)
            out.append(("terrace", f"surface_stairs_{block['id']}", f"surface_ramp_{block['id']}", (gx, gy), terrace["z"]))
        for platform in self.layout.get("platforms", []):
            out.append(("platform", platform["stairs"], platform["ramp"], tuple(platform["goal"]), platform["z"]))
        cands = []
        for kind, stairs, ramp, (gx, gy), z in out:
            dist = math.hypot(gx - self.feet[0], gy - self.feet[1])
            if not MIN_DIST <= dist <= MAX_DIST:
                continue
            if self.scene.objects.get(stairs) is None or self.scene.objects.get(ramp) is None:
                continue
            px, py, d = project(self.scene, self.cam, (gx, gy, z))
            if not in_frame(px, py, d, margin=-RES_X * 0.04):
                continue
            cands.append({"kind": kind, "object": stairs, "alts": (stairs, ramp), "goal": (gx, gy), "px": px,
                          "dist": dist, "noun": kind})
        return cands

    # ---- path checks ----------------------------------------------------

    def single_path(self, cand):
        """(cost, trace, points, corners) for a candidate, or None when it is not a sensible Human task."""
        points, cost, corners = self.plan(cand["goal"])
        if points is None:
            return None
        names = self.planner.classes_along(points)
        road_steps = sum(1 for name in names if name in ("road", "bike_lane", "lane_marking"))
        if road_steps * 0.25 > 0.6:
            return None  # would need to walk on the road
        if cand["kind"] in ("crosswalk", "bridge") and cand["kind"] not in names:
            return None
        trace, _reason = self.trace(points, {cand["object"]})
        if not trace:
            return None
        return (cost, trace, points, corners)

    def two_paths(self, cand):
        """Stairs-only and ramp-only paths to a raised goal; both kept when neither is a big detour."""
        stairs, ramp = cand["alts"]
        paths = []
        for banned in ((CLASS_ID["ramp"],), (CLASS_ID["stairs"],)):
            points, cost, corners = self.plan(cand["goal"], banned=banned)
            if points is None:
                continue
            trace, _reason = self.trace(points, {stairs, ramp, cand["object"]})
            if trace:
                paths.append((cost, trace, points, corners))
        if not paths:
            return None
        best = min(p[0] for p in paths)
        return [p for p in paths if p[0] <= best * 1.6]

    # ---- wording --------------------------------------------------------

    def goal_task(self, cand, paths):
        kind = cand["kind"]
        if kind == "crosswalk":
            template = self.rng.choice(CROSSWALK_TASKS)
        elif kind == "bridge":
            template = self.rng.choice(BRIDGE_TASKS)
        elif kind == "terrace":
            template = self.rng.choice(TERRACE_TASKS)
        elif kind == "platform":
            template = self.rng.choice(PLATFORM_TASKS)
        else:
            template = self.rng.choice(OBJECT_TARGETS[kind])
        text = template.format(noun=cand.get("noun", ""), where=cand.get("where", direction_word(cand["px"])))
        return self.finish(cand, text, paths, "Goal")

    def directions_task(self, cand, path):
        """Turn-by-turn wording of one path, or None when the path has no turn or terrain event to describe."""
        _cost, _trace, points, corners = path
        clauses = self.describe(points, corners)
        if clauses is None:
            return None
        kind = cand["kind"]
        if kind in OBJECT_TARGETS:
            clauses.append(f"continue to the {cand['noun']}")
        elif kind in UPPER_KINDS:
            clauses.append("continue up onto the " + ("terrace" if kind == "terrace" else "platform"))
        text = clauses[0][0].upper() + clauses[0][1:]
        for clause in clauses[1:]:
            text += ", then " + clause
        return self.finish(cand, text, [path], "Directions")

    def describe(self, points, corners):
        """Clauses like ["go straight for about 10 meters", "turn left", "cross the bridge"]."""
        if len(corners) < 2:
            return None
        # Merge nearly straight corners and drop very short legs.
        legs = []
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 0.3:
                continue
            angle = math.atan2(y1 - y0, x1 - x0)
            if legs and (abs(wrap(angle - legs[-1][0])) < math.radians(25) or length < 1.5):
                a0, l0, s0, e0 = legs[-1]
                legs[-1] = (a0 if l0 >= length else angle, l0 + length, s0, (x1, y1))
            else:
                legs.append((angle, length, (x0, y0), (x1, y1)))
        if not legs or len(legs) > 4:
            return None
        events = self.terrain_events(points, legs)
        if len(legs) < 2 and not events:
            return None  # a straight walk with nothing on the way is just a Goal task
        clauses = []
        rel = wrap(legs[0][0] - self.heading)
        first = "go straight" if abs(rel) < math.radians(25) else ("head left" if rel > 0 else "head right")
        if legs[0][1] >= 6.0:
            first += f" for about {max(5, int(round(legs[0][1] / 5.0)) * 5)} meters"
        clauses.append(first)
        clauses.extend(events.get(0, []))
        for k in range(1, len(legs)):
            turn = wrap(legs[k][0] - legs[k - 1][0])
            side = "left" if turn > 0 else "right"
            clause = ("bear " if abs(turn) < math.radians(60) else "turn ") + side
            if legs[k][1] >= 8.0 and k < len(legs) - 1:
                clause += f" and walk about {int(round(legs[k][1] / 5.0)) * 5} meters"
            clauses.append(clause)
            clauses.extend(events.get(k, []))
        return clauses

    def terrain_events(self, points, legs):
        """Crossings, bridges, stairs and ramps, keyed by the leg they happen on (first occurrence each)."""
        events, seen = {}, set()
        prev_name, prev_z = None, None
        for x, y, z in points:
            name = self.planner.classes_along([(x, y, z)])[0]
            key = None
            if name in ("crosswalk", "bridge") and name != prev_name:
                key = name
            elif name in ("stairs", "ramp") and name != prev_name and prev_z is not None:
                ahead = [p[2] for p in points if math.hypot(p[0] - x, p[1] - y) < 2.5]
                key = f"{name}_{'up' if max(ahead) > z + 0.1 else 'down'}"
            if key and key not in seen:
                seen.add(key)
                leg = min(range(len(legs)), key=lambda k: self._dist_to_leg((x, y), legs[k]))
                events.setdefault(leg, []).append(TERRAIN_PHRASES[key])
            prev_name, prev_z = name, z
        return events

    @staticmethod
    def _dist_to_leg(p, leg):
        _a, _l, (x0, y0), (x1, y1) = leg
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / ((dx * dx + dy * dy) or 1e-9)))
        return math.hypot(p[0] - (x0 + t * dx), p[1] - (y0 + t * dy))

    def finish(self, cand, text, paths, task_type):
        names = set()
        for path in paths:
            names.update(self.planner.classes_along(path[2]))
        categories = []
        if "crosswalk" in names:
            categories.append("Social Norms")
        if names & {"stairs", "ramp", "curb_ramp", "bridge"}:
            categories.append("Geometric Terrain")
        if len(paths) > 1 or "ramp" in names:
            categories.append("Accessibility")
        if self.passes_obstacles(paths[0][2], cand.get("object")):
            categories.append("Stationary Obstacle")
        if self.passes_dynamic(paths[0][2], cand.get("object")):
            categories.append("Dynamic Obstacle")
        if self.lighting["preset"] in ("night", "moonlight"):
            categories.append("Visibility")
        if not categories:
            categories.append("Semantic Terrain")
        return {
            "task": text,
            "task_type": task_type,
            "target_kind": cand["kind"],
            "target_object": cand["object"],
            "paths": [p[1] for p in paths],
            "world_paths": [[[round(v, 3) for v in q] for q in p[2]] for p in paths],
            "category": categories,
            "path_surfaces": sorted(names),
        }

    def passes_obstacles(self, points, target_name, radius=0.9):
        """True when the path passes close to an obstacle other than the target."""
        planner = self.planner
        reach = int(radius / planner.cell)
        obstacle_ids = {index for index in range(256)} - {CLASS_ID[name] for name in GROUND_TYPES} - {0}
        for x, y, _z in points[4:-4]:
            row, col = planner.to_cell(x, y)
            window = planner.cls[max(0, row - reach): row + reach + 1, max(0, col - reach): col + reach + 1]
            if any(int(v) in obstacle_ids for v in set(window.flatten().tolist())):
                return True
        return False

    def passes_dynamic(self, points, target_name, radius=1.5):
        """True when the path passes near something that could move: an animal or a car stopped in a lane."""
        movers = []
        for obj in self.scene.objects:
            if obj.name != target_name and (obj.get("category") in DYNAMIC_KINDS or obj.get("in_traffic")):
                movers.append(obj.matrix_world.translation)
        for x, y, _z in points:
            for m in movers:
                if math.hypot(m.x - x, m.y - y) <= radius:
                    return True
        return False

    # ---- choosing tasks ---------------------------------------------------

    def score(self, kind):
        """Lower = more needed: dataset count so far divided by the kind's weight, with a little jitter."""
        return (self.balance.get(kind, 0) + 1) / KIND_WEIGHT.get(kind, 1.0) * (1.0 + 0.3 * self.rng.random())

    def generate(self, max_tasks=2):
        """Up to one Goal and one Directions task, for different targets."""
        if self.start is None:
            return []
        targets, self.landmarks = self.object_candidates()
        cands = (disambiguate(targets, self.landmarks, self.rng)
                 + self.crossing_candidates("crosswalk") + self.crossing_candidates("bridge", max_near=14.0)
                 + self.upper_level_candidates())
        cands.sort(key=lambda c: self.score(c["kind"]))
        tasks, used = [], set()
        want = ["Goal", "Directions"][:max_tasks] if max_tasks > 1 else ["Goal"]
        for task_type in want:
            for cand in cands:
                if cand["object"] in used or len(tasks) >= max_tasks:
                    continue
                if task_type == "Goal":
                    paths = self.two_paths(cand) if "alts" in cand else None
                    if not paths:
                        path = self.single_path(cand)
                        paths = [path] if path else None
                    task = self.goal_task(cand, paths) if paths else None
                else:
                    path = self.single_path(cand)
                    task = self.directions_task(cand, path) if path else None
                if task:
                    tasks.append(task)
                    used.add(cand["object"])
                    break
        return tasks

    def context(self, tasks):
        block = self.block
        parts = [VARIANT_CONTEXT.get(block["variant"], "City street") if block else "City street"]
        if block and any(info["bike_w"] > 0 for info in block["sides"].values()):
            parts.append("Bike lane next to the curb")
        preset = self.lighting["preset"]
        if preset != "day":
            parts.append({"overcast": "Overcast sky", "dusk": "Dusk, street lamps on",
                          "night": "Night, street lamps and some windows lit",
                          "moonlight": "Night, only moonlight"}[preset])
        landmarks = getattr(self, "landmarks", [])
        if any(self.scene.objects[l["object"]].get("in_traffic") for l in landmarks):
            parts.append("Cars stopped in the driving lanes")
        animals = sorted({l["noun"] for l in landmarks if l["kind"] in DYNAMIC_KINDS})
        if animals:
            parts.append(", ".join(a.capitalize() for a in animals) + " nearby")
        seen = sorted({t["target_kind"].replace("_", " ") for t in tasks})
        if seen:
            parts.append("Visible targets: " + ", ".join(seen))
        return "; ".join(parts)
