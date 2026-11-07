"""World 2 plan: an old town on two levels with irregular streets. Plain Python (no bpy).

The street network is a jittered grid of nodes; streets run along the grid
lines and the spaces between them ("faces") hold perimeter blocks of
narrow houses around a courtyard, or open squares. A stone retaining wall
along grid line ``WALL_LINE`` splits the lower town from the upper town
(UPPER_Z higher); stairs and ramps at the ends of the cross streets, and a
grand stair with a ramp in one square, connect the two levels. Grid line
``MAIN_LINE`` in the lower town is the only street with a road and cars.

Faces are the "blocks" of this world: whole faces are held out for the test
split, and ``block_side_at`` (layout.py) finds the face around a point.
"""

import math
import random

NX, NY = 8, 7                 # grid nodes; faces are (NX - 1) x (NY - 1)
PITCH_X, PITCH_Y = 36.0, 32.0
JITTER = 6.5                  # interior node jitter (m)
EDGE_JITTER = 2.5             # boundary nodes move less, so the outline stays tidy
MAIN_LINE = 1                 # horizontal grid line with the road
WALL_LINE = 3                 # horizontal grid line whose street is the upper promenade
UPPER_Z = 2.1                 # height of the upper town
RISE, TREAD = 0.15, 0.32
RAMP_SLOPE = 1.0 / 8.0
ROAD_W = 6.2
SIDEWALK_Z = 0.14
BASE_Z = -0.3
BACKGROUND_DEPTH = 11.0
MARGIN = 26.0

# Horizontal street widths per grid line (outer lines are narrower lanes).
H_WIDTH = {0: 5.5, 1: ROAD_W + 2 * 2.6, 2: 6.0, 3: 7.5, 4: 6.0, 5: 5.5, 6: 5.0}
# Cross streets that meet the retaining wall: what connects them to the upper town.
WALL_LINKS = {1: "stairs", 2: "ramp", 4: "stairs", 6: "ramp", 7: "stairs"}
SQUARES = {(3, 2): "stair_square", (5, 0): "market_square", (1, 4): "church_square", (5, 4): "upper_square",
           (1, 1): "small_square"}
TEST_FACES = ("f00", "f20", "f42", "f61", "f33", "f14", "f64", "f55")


# ---- 2D geometry ------------------------------------------------------------

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, k):
    return (a[0] * k, a[1] * k)


def norm(a):
    length = math.hypot(*a) or 1e-9
    return (a[0] / length, a[1] / length)


def left(u):
    return (-u[1], u[0])


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersect(p, u, q, v):
    """Intersection of the lines p + t u and q + s v (None when parallel)."""
    det = cross(u, v)
    if abs(det) < 1e-9:
        return None
    t = cross(sub(q, p), v) / det
    return add(p, mul(u, t))


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def point_in_poly(pt, poly):
    inside = False
    x, y = pt
    for k in range(len(poly)):
        (x0, y0), (x1, y1) = poly[k], poly[(k + 1) % len(poly)]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def offset_polyline(points, dist):
    """Polyline shifted ``dist`` to its left, with mitered joints."""
    out = []
    for k, p in enumerate(points):
        if k == 0:
            u = norm(sub(points[1], p))
            out.append(add(p, mul(left(u), dist)))
        elif k == len(points) - 1:
            u = norm(sub(p, points[k - 1]))
            out.append(add(p, mul(left(u), dist)))
        else:
            u0 = norm(sub(p, points[k - 1]))
            u1 = norm(sub(points[k + 1], p))
            a = add(points[k - 1], mul(left(u0), dist))
            b = add(p, mul(left(u1), dist))
            out.append(intersect(a, u0, b, u1) or add(p, mul(left(u1), dist)))
    return out


# ---- plan ---------------------------------------------------------------------

def make_layout(seed=0):
    rng = random.Random(seed * 104729 + 7)
    nodes = {}
    for i in range(NX):
        for j in range(NY):
            boundary = i in (0, NX - 1) or j in (0, NY - 1)
            amp = EDGE_JITTER if boundary else JITTER
            nodes[(i, j)] = (i * PITCH_X + rng.uniform(-amp, amp), j * PITCH_Y + rng.uniform(-amp, amp))

    # Street width per edge: horizontal edges follow their grid line; cross streets vary.
    widths = {}
    for i in range(NX - 1):
        for j in range(NY):
            widths[("h", i, j)] = H_WIDTH[j]
    for i in range(NX):
        for j in range(NY - 1):
            if j == WALL_LINE - 1 and WALL_LINKS.get(i) == "ramp":
                w = 4.6
            elif i in (0, NX - 1):
                w = 5.0
            else:
                w = rng.choice((3.2, 3.8, 4.4, 5.2, 6.0))
            widths[("v", i, j)] = w

    layout = {
        "world": "oldtown",
        "seed": seed,
        "nodes": {f"{i},{j}": list(p) for (i, j), p in nodes.items()},
        "upper_z": UPPER_Z,
        "main_line": MAIN_LINE,
        "wall_line": WALL_LINE,
        "test_blocks": list(TEST_FACES),
        "blocks": {},
        "streets": [],
        "platforms": [],
        "entrances": [],
        "social_zones": [],
    }

    # Faces: corners counter-clockwise from the south-west node; edge k runs corner k -> k + 1.
    for c in range(NX - 1):
        for r in range(NY - 1):
            corners = [nodes[(c, r)], nodes[(c + 1, r)], nodes[(c + 1, r + 1)], nodes[(c, r + 1)]]
            edge_keys = [("h", c, r), ("v", c + 1, r), ("h", c, r + 1), ("v", c, r)]
            inset = inset_quad(corners, [widths[k] / 2 for k in edge_keys])
            fid = f"f{c}{r}"
            upper = r >= WALL_LINE
            kind = SQUARES.get((c, r))
            layout["blocks"][fid] = {
                "id": fid,
                "col": c,
                "row": r,
                "variant": "old_square" if kind else ("old_upper" if upper else "old_lower"),
                "square": kind,
                "upper": upper,
                "ground_z": UPPER_Z if upper else 0.0,
                "quad": [list(p) for p in corners],
                "inset": [list(p) for p in inset],
                "edge_keys": [list(k) for k in edge_keys],
                "edge_widths": [widths[k] for k in edge_keys],
                "sides": {},
            }

    # Street centerline segments (for camera headings) with their level.
    for (axis, i, j), w in widths.items():
        a = nodes[(i, j)]
        b = nodes[(i + 1, j)] if axis == "h" else nodes[(i, j + 1)]
        layout["streets"].append({"axis": axis, "i": i, "j": j, "a": list(a), "b": list(b), "width": w})

    # The retaining wall: south edge of the promenade street, extended to the world edge.
    line = [nodes[(i, WALL_LINE)] for i in range(NX)]
    wall = offset_polyline(line, -H_WIDTH[WALL_LINE] / 2)
    ex0, ex1 = -MARGIN, (NX - 1) * PITCH_X + MARGIN
    ey0, ey1 = -MARGIN, (NY - 1) * PITCH_Y + MARGIN
    first_u = norm(sub(wall[1], wall[0]))
    last_u = norm(sub(wall[-1], wall[-2]))
    start = intersect(wall[0], first_u, (ex0, 0.0), (0.0, 1.0))
    end = intersect(wall[-1], last_u, (ex1, 0.0), (0.0, 1.0))
    layout["wall"] = [list(p) for p in [start] + wall + [end]]

    # The main street's road, extended to the world edge.
    main = [nodes[(i, MAIN_LINE)] for i in range(NX)]
    u0 = norm(sub(main[1], main[0]))
    u1 = norm(sub(main[-1], main[-2]))
    main = [add(main[0], mul(u0, -(main[0][0] - ex0) / max(0.2, u0[0])))] + main
    main = main + [add(main[-1], mul(u1, (ex1 - main[-1][0]) / max(0.2, u1[0])))]
    layout["road"] = [list(p) for p in main]

    # Wall links: cross street i between node rows WALL_LINE-1 and WALL_LINE.
    layout["wall_links"] = []
    for i, kind in WALL_LINKS.items():
        layout["wall_links"].append({"i": i, "kind": kind, "width": widths[("v", i, WALL_LINE - 1)]})

    layout["extent"] = [ex0, ex1, ey0, ey1]
    nm = 8.0
    layout["nav_extent"] = [-nm, (NX - 1) * PITCH_X + nm, -nm, (NY - 1) * PITCH_Y + nm]
    return layout


def inset_quad(corners, half_widths):
    """Quad shrunk by a different distance along each edge (edge k runs corner k -> k + 1)."""
    lines = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        u = norm(sub(b, a))
        lines.append((add(a, mul(left(u), half_widths[k])), u))
    out = []
    for k in range(4):
        p, u = lines[(k - 1) % 4]
        q, v = lines[k]
        out.append(intersect(p, u, q, v))
    return out


def face_lots(inset, edges, depth_of, width_range, rng):
    """Narrow house lots around the inside of a face.

    ``edges``: which inset edges (0-3) get houses. Edges 0 and 2 own the
    corners, so lots never overlap. Returns dicts with a counter-clockwise
    quad (street corner a, street corner b, inner b, inner a) and the edge.
    """
    order = [k for k in (0, 2, 1, 3) if k in edges]
    inner = {}
    lots = []
    for k in order:
        a, b = inset[k], inset[(k + 1) % 4]
        u = norm(sub(b, a))
        m = left(u)  # into the face
        depth = depth_of(k)
        prev_k, next_k = (k - 1) % 4, (k + 1) % 4

        def limit(other):
            if other in inner:
                return inner[other]
            p, q = inset[other], inset[(other + 1) % 4]
            return (p, norm(sub(q, p)))

        lp, lu = limit(prev_k)
        np_, nu = limit(next_k)
        s = intersect(a, u, lp, lu)
        e = intersect(a, u, np_, nu)
        ia = add(a, mul(m, depth))
        si = intersect(ia, u, lp, lu)
        ei = intersect(ia, u, np_, nu)
        if None in (s, e, si, ei):
            continue
        length = math.hypot(*sub(e, s))
        if length < 4.0:
            inner[k] = (ia, u)
            continue
        cuts = [0.0]
        while length * (1.0 - cuts[-1]) > width_range[1]:
            cuts.append(cuts[-1] + rng.uniform(*width_range) / length)
        cuts.append(1.0)
        if len(cuts) > 2 and (cuts[-1] - cuts[-2]) * length < width_range[0] * 0.7:
            cuts.pop(-2)
        for t0, t1 in zip(cuts, cuts[1:]):
            quad = [lerp(s, e, t0), lerp(s, e, t1), lerp(si, ei, t1), lerp(si, ei, t0)]
            lots.append({"quad": quad, "edge": k, "u": u})
        inner[k] = (ia, u)
    return lots


def describe(layout):
    lines = [f"old town seed={layout['seed']} extent={tuple(round(v, 1) for v in layout['extent'])}"]
    for face in layout["blocks"].values():
        lines.append(f"  {face['id']} {face['variant']:<11} {face['square'] or ''}")
    lines.append(f"  wall_links={[(w['i'], w['kind']) for w in layout['wall_links']]}")
    return "\n".join(lines)
