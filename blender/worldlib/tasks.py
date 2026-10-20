"""Turn one camera view into NaviTrace-style Human tasks (bpy).

For every visible target (bench, door, crosswalk, terrace, ...) plan a Human
path from the camera's feet, project it into the image, and keep it when the
path is on screen and mostly unoccluded. Traces start where the path enters
the bottom of the frame, like NaviTrace ground truth.
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
MAX_TRACE_POINTS = 20

OBJECT_TARGETS = {
    "bench": ["Go to the {noun}{where}", "Walk to the {noun}{where}", "Go sit on the {noun}{where}"],
    "trash_can": ["Go to the {noun}{where}", "Walk over to the {noun}{where}"],
    "bike": ["Go to the {noun}{where}", "Walk to the {noun}{where}"],
    "planter": ["Go to the {noun}{where}", "Walk to the {noun}{where}"],
    "door": ["Go to the {noun}{where}", "Walk to the {noun}{where}"],
    "street_lamp": ["Go to the {noun}{where}"],
    "tree": ["Go to the {noun}{where}", "Walk to the {noun}{where}"],
    "bush": ["Go to the {noun}{where}", "Walk over to the {noun}{where}"],
    "cat": ["Walk toward the {noun}{where}", "Go to the {noun}{where}", "Walk over to the {noun}{where}"],
    "dog": ["Walk toward the {noun}{where}", "Go to the {noun}{where}", "Walk up to the {noun}{where}"],
    "car": ["Go to the {noun}{where}", "Walk to the {noun}{where}"],
}
BASE_NOUN = {
    "bench": "bench", "trash_can": "trash can", "bike": "parked bike", "planter": "planter",
    "door": "building entrance", "street_lamp": "street lamp", "tree": "tree", "bush": "bush",
    "cat": "cat", "dog": "dog", "car": "car",
}
RELATION_M = 2.5          # landmark must be this close to the target to be "beside" it
RELATION_WORDS = (" beside the {}", " next to the {}", " near the {}")
RELATION_RATE = 0.35      # share of already-unique targets that still get a landmark phrase, for variety
DYNAMIC_KINDS = ("cat", "dog")
CROSSWALK_TASKS = ["Cross the street", "Cross the street at the crosswalk", "Get to the other side of the street"]
TERRACE_TASKS = ["Go up to the terrace{where}", "Get onto the raised terrace{where}"]

VARIANT_CONTEXT = {
    "street": "Sidewalk along a street with a grass area behind it",
    "plaza": "Wide paved plaza in front of large buildings",
    "narrow": "Narrow sidewalk between parked cars and row houses",
    "construction": "Sidewalk next to a construction site",
    "terrace": "Sidewalk next to a raised terrace with stairs and a ramp",
}


def project(scene, cam, point):
    co = world_to_camera_view(scene, cam, Vector(point))
    return co.x * RES_X, (1.0 - co.y) * RES_Y, co.z


def in_frame(px, py, depth, margin=0.0):
    return depth > 0.05 and -margin <= px <= RES_X + margin and -margin <= py <= RES_Y + margin


def visible_from(scene, depsgraph, cam_loc, point, target_names=None, tol=0.25):
    """True when nothing blocks the line from the camera to ``point`` (or the first hit is a target)."""
    direction = Vector(point) - cam_loc
    dist = direction.length
    hit, loc, _n, _i, obj, _m = scene.ray_cast(depsgraph, cam_loc, direction.normalized(), distance=dist)
    if not hit:
        return True
    source = obj.original if hasattr(obj, "original") else obj
    if target_names and source.name in target_names:
        return True
    return (loc - cam_loc).length >= dist - tol


def rdp(points, eps):
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1e-9
    best_i, best_d = 0, -1.0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_i, best_d = i, d
    if best_d <= eps:
        return [points[0], points[-1]]
    left = rdp(points[: best_i + 1], eps)
    right = rdp(points[best_i:], eps)
    return left[:-1] + right


def direction_word(px):
    if px < RES_X * 0.36:
        return " on the left"
    if px > RES_X * 0.64:
        return " on the right"
    return " ahead"


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


def noun_of(obj):
    category = obj.get("category")
    color = obj.get("color_name")
    if category in ("cat", "dog", "car") and color:
        return f"{color} {category}"
    if category == "tree" and obj.get("tree_style") == "conifer":
        return "pine tree"
    return BASE_NOUN.get(category, category.replace("_", " "))


class ViewTasks:
    def __init__(self, scene, cam, planner, layout, rng=None):
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

    def trace(self, points, target_names=()):
        """Pixel trace for world points (starting at the bottom center), or (None, reason)."""
        pix = [project(self.scene, self.cam, p) for p in points]
        # The start cell can sit a few pixels below the frame; anything more means it is not in view.
        if any(not (d > 0.05 and -2 <= px <= RES_X + 2 and -2 <= py <= RES_Y + 25) for px, py, d in pix):
            return None, "leaves the frame"
        occluded = 0
        for (x, y, z) in points[1:]:
            if not visible_from(self.scene, self.depsgraph, self.cam_loc, (x, y, z + 0.05), target_names):
                occluded += 1
        if occluded > MAX_OCCLUDED * len(points):
            return None, f"occluded {occluded}/{len(points)}"
        trace = [(min(max(px, 0.0), RES_X - 0.5), min(max(py, 0.0), RES_Y - 0.5)) for px, py, _d in pix]
        simple = rdp(trace, RDP_EPS_PX)
        eps = RDP_EPS_PX
        while len(simple) > MAX_TRACE_POINTS:
            eps *= 1.5
            simple = rdp(trace, eps)
        if len(simple) < 2 or math.hypot(simple[-1][0] - simple[0][0], simple[-1][1] - simple[0][1]) < 40:
            return None, "too short on screen"
        return [[round(px, 2), round(py, 2)] for px, py in simple], "ok"

    def plan(self, goal_xy, banned=()):
        goal = self.planner.nearest_walkable(goal_xy[0], goal_xy[1], radius=1.5, max_cost=1.6, prefer=self.start_xy)
        if self.start is None or goal is None:
            return None, None
        cells = self.planner.astar(self.start, goal, banned_classes=banned)
        if cells is None:
            return None, None
        cells = self.planner.smooth(cells)
        points = self.planner.to_points(cells)
        return points, self.planner.path_cost(cells)

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
            if not visible_from(self.scene, self.depsgraph, self.cam_loc, center, {obj.name}):
                continue
            cand = {"kind": category, "noun": noun_of(obj), "object": obj.name, "xy": center[:2], "px": px,
                    "dist": dist}
            landmarks.append(cand)
            if obj.get("in_traffic") or not MIN_DIST <= dist <= MAX_DIST:
                continue  # cars stopped in the lanes are landmarks, never goals (the goal would be on the road)
            if category == "door":
                goal = tuple(obj["front"])
            elif category == "bench":
                # In front of the seat (benches face local -Y), not behind the backrest.
                front = obj.matrix_world @ Vector((0.0, -0.8, 0.0))
                goal = (front.x, front.y)
            else:
                goal = center[:2]
            targets.append({**cand, "goal": goal})
        return targets, landmarks

    def crosswalk_candidates(self):
        out = []
        for obj in self.scene.objects:
            if obj.get("category") != "crosswalk":
                continue
            a, b = tuple(obj["end_a"]), tuple(obj["end_b"])
            da = math.hypot(a[0] - self.feet[0], a[1] - self.feet[1])
            db = math.hypot(b[0] - self.feet[0], b[1] - self.feet[1])
            near, far = (a, b) if da < db else (b, a)
            if min(da, db) > 18.0 or max(da, db) > MAX_DIST:
                continue
            box = world_aabb(obj)
            center = ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0.02)
            px, py, depth = project(self.scene, self.cam, center)
            if not in_frame(px, py, depth, margin=-RES_X * 0.04):
                continue
            out.append({"kind": "crosswalk", "object": obj.name, "goal": far, "px": px, "dist": max(da, db)})
        return out

    def terrace_candidates(self):
        out = []
        for block in self.layout["blocks"].values():
            terrace = block.get("terrace")
            if not terrace or "stairs_s" not in terrace:
                continue
            ring = lay.CURB_WIDTH + block["sidewalk_w"]
            depth = lay.N_STEPS * lay.TREAD
            s_mid = (terrace["stairs_s"][0] + terrace["ramp_s"][1]) / 2
            gx, gy = lay.side_point(block, terrace["side"], s_mid, ring + depth + 2.5)
            dist = math.hypot(gx - self.feet[0], gy - self.feet[1])
            if not MIN_DIST <= dist <= MAX_DIST:
                continue
            stairs = self.scene.objects.get(f"surface_stairs_{block['id']}")
            ramp = self.scene.objects.get(f"surface_ramp_{block['id']}")
            if stairs is None or ramp is None:
                continue
            px, py, d = project(self.scene, self.cam, (gx, gy, terrace["z"]))
            if not in_frame(px, py, d, margin=-RES_X * 0.04):
                continue
            out.append({"kind": "terrace", "object": stairs.name, "ramp": ramp.name, "goal": (gx, gy),
                        "px": px, "dist": dist, "block": block["id"]})
        return out

    # ---- tasks ---------------------------------------------------------

    def build_task(self, cand):
        if cand["kind"] == "terrace":
            paths = []
            targets = {cand["object"], cand["ramp"]}
            for banned in ((CLASS_ID["ramp"],), (CLASS_ID["stairs"],)):
                points, cost = self.plan(cand["goal"], banned=banned)
                if points is None:
                    continue
                trace, _reason = self.trace(points, targets)
                if trace:
                    paths.append((cost, trace, points))
            if not paths:
                return None
            best = min(cost for cost, _t, _p in paths)
            # Stairs and ramp are both valid for a Human when neither is a big detour.
            paths = [p for p in paths if p[0] <= best * 1.6]
            template = self.rng.choice(TERRACE_TASKS)
            return self.finish(cand, template, paths)

        points, cost = self.plan(cand["goal"])
        if points is None:
            return None
        names = self.planner.classes_along(points)
        road_steps = sum(1 for name in names if name in ("road", "bike_lane", "lane_marking"))
        if road_steps * 0.25 > 0.6:
            return None  # would need to walk on the road; not a sensible Human task
        if cand["kind"] == "crosswalk" and "crosswalk" not in names:
            return None
        trace, _reason = self.trace(points, {cand["object"]})
        if not trace:
            return None
        if cand["kind"] == "crosswalk":
            template = self.rng.choice(CROSSWALK_TASKS)
        else:
            template = self.rng.choice(OBJECT_TARGETS[cand["kind"]])
        return self.finish(cand, template, [(cost, trace, points)])

    def finish(self, cand, template, paths):
        task = template.format(noun=cand.get("noun", ""), where=cand.get("where", direction_word(cand["px"])))
        names = set()
        for _cost, _trace, points in paths:
            names.update(self.planner.classes_along(points))
        categories = []
        if "crosswalk" in names:
            categories.append("Social Norms")
        if names & {"stairs", "ramp", "curb_ramp"}:
            categories.append("Geometric Terrain")
        if cand["kind"] == "terrace":
            categories.append("Accessibility")
        if self.passes_obstacles(paths[0][2], cand.get("object")):
            categories.append("Stationary Obstacle")
        if self.passes_dynamic(paths[0][2], cand.get("object")):
            categories.append("Dynamic Obstacle")
        if not categories:
            categories.append("Semantic Terrain")
        return {
            "task": task,
            "target_kind": cand["kind"],
            "target_object": cand["object"],
            "paths": [trace for _cost, trace, _points in paths],
            "world_paths": [[[round(v, 3) for v in p] for p in points] for _cost, _trace, points in paths],
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

    def generate(self, max_tasks=2):
        if self.start is None:
            return []
        targets, self.landmarks = self.object_candidates()
        cands = (disambiguate(targets, self.landmarks, self.rng)
                 + self.crosswalk_candidates() + self.terrace_candidates())
        self.rng.shuffle(cands)
        # Terrain and crossings first, then animals and entrances; lamps last.
        priority = {"terrace": 0, "crosswalk": 1, "cat": 2, "dog": 2, "door": 3, "street_lamp": 5}
        cands.sort(key=lambda c: priority.get(c["kind"], 4))
        tasks, kinds = [], set()
        for cand in cands:
            if cand["kind"] in kinds:
                continue
            task = self.build_task(cand)
            if task:
                tasks.append(task)
                kinds.add(cand["kind"])
            if len(tasks) >= max_tasks:
                break
        return tasks

    def context(self, tasks):
        block = self.block
        parts = [VARIANT_CONTEXT.get(block["variant"], "City street") if block else "City street"]
        if block and any(info["bike_w"] > 0 for info in block["sides"].values()):
            parts.append("Bike lane next to the curb")
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
