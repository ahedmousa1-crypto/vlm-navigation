"""World 3 plan: a university campus on a hillside above a waterfront. Plain Python (no bpy).

Like the old town (oldtown_layout.py) the street network is a jittered grid of
nodes; the spaces between the grid lines ("faces") are the blocks. Rows of
faces form zones, from the water up the hill:

  quay   grid line 0 is the waterfront promenade (paved walk, a bike lane and a
         wooden boardwalk to the quay edge); south of it is the lake, with a
         pier, moored boats and lower quay ledges reached by steps and ramps.
         A canal cuts into the first row, crossed by footbridges.
  row 0  waterfront: boathouses with cafe terraces, harbor squares, lawns.
  row 1  the slope: amphitheater, terrace gardens with intermediate levels,
         lower faculty buildings. A retaining wall along grid line WALL_LINE
         lifts the campus UPPER_Z above the waterfront; stairs and ramps at the
         cross streets, and the amphitheater, connect the two levels.
  rows 2-5  the campus: lawn quads crossed by paths, faculty buildings with
         stepped and ramped entrances, libraries on plinths, a sports field,
         parking and a construction site.
  row 6  the park belt: gravel paths, woods, meadows and a playground.
  north  grid line NY-1 is a road with bike lanes, bus stops and crosswalks.

Faces are the "blocks" of this world (``world == "campus"``): whole faces are
held out for the test split. Each face also has a ``region`` polygon, which
for faces on the edge reaches out to the world edge, so cameras on the
promenade, the pier and the north road still belong to a face.
"""

import math
import random

from .oldtown_layout import add, inset_quad, intersect, left, mul, norm, offset_polyline, point_in_poly, sub  # noqa: F401

NX, NY = 10, 8                # grid nodes; faces are (NX - 1) x (NY - 1)
PITCH_X, PITCH_Y = 40.0, 38.0
JITTER = 4.0
EDGE_JITTER = 1.5
WALL_LINE = 2                 # horizontal grid line along the top of the slope
UPPER_Z = 3.0                 # campus level above the waterfront
RISE, TREAD = 0.15, 0.32
RAMP_SLOPE = 1.0 / 8.0
ROAD_W = 6.4
BIKE_W = 1.6
SIDEWALK_Z = 0.14
BASE_Z = -0.3
WATER_Z = -1.4                # lake and canal surface
LEDGE_Z = -0.9                # lower quay ledges, just above the water
PROM_W = 14.0                 # promenade: 4.6 m paved walk, 2.4 m bike lane, 7 m boardwalk
PROM_WALK, PROM_BIKE = 4.6, 2.4
CANAL_I = 4                   # the canal runs along vertical street CANAL_I in row 0
CANAL_W = 6.0
MARGIN = 30.0
FAR_BANK = 70.0               # distance from the quay to the buildings across the lake

# Horizontal street widths per grid line.
H_WIDTH = {0: PROM_W, 1: 7.0, 2: 9.0, 3: 6.5, 4: 6.5, 5: 6.5, 6: 6.0, 7: ROAD_W + 2 * (BIKE_W + 2.4)}
# Cross streets that meet the retaining wall: what connects them to the campus above.
WALL_LINKS = {1: "stairs", 3: "ramp", 5: "stairs", 6: "ramp", 8: "stairs"}

# Face kinds by (col, row).
ROW_KINDS = {
    0: ["water_park", "boathouse", "harbor_square", "water_park", "boathouse", "harbor_square", "water_park",
        "boathouse", "water_park"],
    1: ["faculty", "terrace_garden", "faculty", "amphitheater", "cafe_plaza", "terrace_garden", "faculty",
        "terrace_garden", "faculty"],
    2: ["quad", "faculty", "library", "quad", "faculty", "quad", "construction", "faculty", "quad"],
    3: ["faculty", "quad", "quad", "faculty", "sports", "faculty", "quad", "library", "faculty"],
    4: ["quad", "faculty", "parking", "quad", "quad", "faculty", "quad", "faculty", "quad"],
    5: ["faculty", "quad", "faculty", "library", "quad", "parking", "faculty", "quad", "construction"],
    6: ["woods", "playground", "meadow", "woods", "meadow", "playground", "woods", "meadow", "woods"],
}
VARIANT = {
    "water_park": "waterfront", "boathouse": "waterfront", "harbor_square": "waterfront",
    "faculty": "campus_buildings", "terrace_garden": "campus_slope", "amphitheater": "campus_slope",
    "cafe_plaza": "campus_slope", "quad": "campus_quad", "library": "campus_library",
    "construction": "campus_construction", "sports": "campus_sports", "parking": "campus_parking",
    "woods": "campus_park", "playground": "campus_park", "meadow": "campus_park",
}
# Held-out faces: one or two of every zone.
TEST_FACES = ("f30", "f40", "f71", "f12", "f52", "f73", "f44", "f25", "f16", "f66")


def make_layout(seed=0):
    rng = random.Random(seed * 15485863 + 11)
    nodes = {}
    for i in range(NX):
        for j in range(NY):
            boundary = i in (0, NX - 1) or j in (0, NY - 1)
            amp = EDGE_JITTER if boundary else JITTER
            dx, dy = rng.uniform(-amp, amp), rng.uniform(-amp, amp)
            if j == 0:
                dy = 0.0              # the quay is straight
            if j == WALL_LINE:
                dy *= 0.4             # and the retaining wall nearly so
            nodes[(i, j)] = (i * PITCH_X + dx, j * PITCH_Y + dy)
    # The canal runs straight up its cross street.
    nodes[(CANAL_I, 1)] = (nodes[(CANAL_I, 0)][0], nodes[(CANAL_I, 1)][1])

    widths = {}
    for i in range(NX - 1):
        for j in range(NY):
            widths[("h", i, j)] = H_WIDTH[j]
    for i in range(NX):
        for j in range(NY - 1):
            if j == WALL_LINE - 1 and WALL_LINKS.get(i) == "ramp":
                w = 5.0
            elif j == 0 and i == CANAL_I:
                w = CANAL_W + 2 * 3.2   # canal with a walk on each side
            elif i in (0, NX - 1):
                w = 6.0
            else:
                w = rng.choice((4.5, 5.5, 6.5, 7.5))
            widths[("v", i, j)] = w

    ex0, ex1 = -MARGIN, (NX - 1) * PITCH_X + MARGIN
    quay_y = -PROM_W / 2
    ey0, ey1 = quay_y - FAR_BANK - 14.0, (NY - 1) * PITCH_Y + MARGIN
    layout = {
        "world": "campus",
        "seed": seed,
        "nodes": {f"{i},{j}": list(p) for (i, j), p in nodes.items()},
        "upper_z": UPPER_Z,
        "wall_line": WALL_LINE,
        "quay_y": quay_y,
        "test_blocks": list(TEST_FACES),
        "blocks": {},
        "streets": [],
        "platforms": [],
        "entrances": [],
        "social_zones": [],
    }

    nav = [-10.0, (NX - 1) * PITCH_X + 10.0, quay_y - 30.0, (NY - 1) * PITCH_Y + 12.0]
    for c in range(NX - 1):
        for r in range(NY - 1):
            corners = [nodes[(c, r)], nodes[(c + 1, r)], nodes[(c + 1, r + 1)], nodes[(c, r + 1)]]
            edge_keys = [("h", c, r), ("v", c + 1, r), ("h", c, r + 1), ("v", c, r)]
            inset = inset_quad(corners, [widths[k] / 2 for k in edge_keys])
            fid = f"f{c}{r}"
            upper = r >= WALL_LINE
            kind = ROW_KINDS[r][c]
            # Region: the quad, pushed out to the navmap edge on the world's border.
            region = [list(p) for p in corners]
            if r == 0:
                region[0][1] = region[1][1] = nav[2]
            if r == NY - 2:
                region[2][1] = region[3][1] = nav[3]
            if c == 0:
                region[0][0] = region[3][0] = nav[0]
            if c == NX - 2:
                region[1][0] = region[2][0] = nav[1]
            layout["blocks"][fid] = {
                "id": fid,
                "col": c,
                "row": r,
                "kind": kind,
                "variant": VARIANT[kind],
                "square": None,
                "upper": upper,
                "ground_z": UPPER_Z if upper else 0.0,
                "quad": [list(p) for p in corners],
                "region": region,
                "inset": [list(p) for p in inset],
                "edge_keys": [list(k) for k in edge_keys],
                "edge_widths": [widths[k] for k in edge_keys],
            }

    for (axis, i, j), w in widths.items():
        a = nodes[(i, j)]
        b = nodes[(i + 1, j)] if axis == "h" else nodes[(i, j + 1)]
        layout["streets"].append({"axis": axis, "i": i, "j": j, "a": list(a), "b": list(b), "width": w})

    # Retaining wall: the south edge of the street on WALL_LINE, extended to the world edge.
    line = [nodes[(i, WALL_LINE)] for i in range(NX)]
    wall = offset_polyline(line, -H_WIDTH[WALL_LINE] / 2)
    first_u = norm(sub(wall[1], wall[0]))
    last_u = norm(sub(wall[-1], wall[-2]))
    start = intersect(wall[0], first_u, (ex0, 0.0), (0.0, 1.0))
    end = intersect(wall[-1], last_u, (ex1, 0.0), (0.0, 1.0))
    layout["wall"] = [list(p) for p in [start] + wall + [end]]
    layout["wall_links"] = [{"i": i, "kind": kind, "width": widths[("v", i, WALL_LINE - 1)]}
                            for i, kind in WALL_LINKS.items()]

    # North road along the top grid line, extended to the world edge.
    top = [nodes[(i, NY - 1)] for i in range(NX)]
    u0, u1 = norm(sub(top[1], top[0])), norm(sub(top[-1], top[-2]))
    top = [add(top[0], mul(u0, -(top[0][0] - ex0) / max(0.2, u0[0])))] + top
    top = top + [add(top[-1], mul(u1, (ex1 - top[-1][0]) / max(0.2, u1[0])))]
    layout["road"] = [list(p) for p in top]

    # Canal: from the quay up the cross street CANAL_I to the south edge of grid line 1's street.
    ca, cb = nodes[(CANAL_I, 0)], nodes[(CANAL_I, 1)]
    head = add(cb, mul(norm(sub(ca, cb)), H_WIDTH[1] / 2 + 0.5))
    layout["canal"] = {"a": [ca[0], quay_y - 0.5], "b": list(head), "width": CANAL_W}

    # Promenade features along the quay: lower ledges (steps + ramp), a pier, footbridges over the canal.
    span = (NX - 1) * PITCH_X
    layout["ledges"] = [{"x0": x0, "x1": x0 + 34.0} for x0 in (18.0, span * 0.62, span - 58.0)]
    layout["pier"] = {"x": span * 0.38, "length": 26.0, "width": 4.0}
    layout["footbridges"] = [0.35, 0.72]   # fractions along the canal

    layout["extent"] = [ex0, ex1, ey0, ey1]
    layout["nav_extent"] = nav
    return layout


def face_at(layout, x, y):
    for face in layout["blocks"].values():
        if point_in_poly((x, y), face.get("region", face["quad"])):
            return face
    return None


def describe(layout):
    lines = [f"campus seed={layout['seed']} extent={tuple(round(v, 1) for v in layout['extent'])}"]
    kinds = {}
    for face in layout["blocks"].values():
        kinds[face["kind"]] = kinds.get(face["kind"], 0) + 1
    lines.append(f"  faces={len(layout['blocks'])} kinds={kinds}")
    lines.append(f"  test={layout['test_blocks']}")
    return "\n".join(lines)
