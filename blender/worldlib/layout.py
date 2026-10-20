"""World plan: a 3x3 grid of blocks separated by streets. Plain Python, no bpy.

Lines (street centerlines) run at X[i] (vertical, i = 0..3) and Y[j]
(horizontal, j = 0..3). Block (c, r) sits between X[c]/X[c+1] and Y[r]/Y[r+1].
A block's footprint stops at the road edge; inside it a curb ring, a sidewalk
ring, then the interior (grass, buildings, terrace, construction site).

Each block side has a local frame: ``s`` runs along the side (0..length) and
``d`` runs inward from the road edge, so ``d = 0`` is the curb face and
``d = CURB_WIDTH + sidewalk_w`` is the interior edge. Sides are walked
counter-clockwise (south, east, north, west), so inward is always the tangent
rotated +90 degrees.
"""

import random

BASE_Z = -0.25
ROAD_Z = 0.0
GRASS_Z = 0.02  # interior grass sits just above the asphalt slab underneath it
CURB_WIDTH = 0.15
RISE = 0.15
TREAD = 0.30
N_STEPS = 6
STAIR_W = 3.2
RAMP_W = 3.8
EYE_HEIGHT = 1.6
CROSSWALK_LIFT = 0.012
CROSSWALK_THICK = 0.018
BIKE_LANE_LIFT = 0.004
BIKE_LANE_THICK = 0.006
CURB_RAMP_LEN = 0.9
OUTER_MARGIN = 40.0
OUTER_SIDEWALK_W = 3.0   # sidewalk ring outside the outer streets
OUTER_CURB_H = 0.13
OUTER_SETBACK = 1.0      # background buildings stand this far behind the outer sidewalk
NAV_MARGIN = 8.0         # navmap reaches this far past the outer road edge

GRID = 3
VARIANTS = ("street", "plaza", "narrow", "construction", "terrace")
# Seed 0 layout; other seeds shuffle it. Every variant appears at least once.
DEFAULT_VARIANTS = [
    "street", "plaza", "narrow",
    "construction", "terrace", "street",
    "narrow", "street", "plaza",
]
SIDEWALK_RANGE = {
    "street": (2.6, 4.4),
    "terrace": (2.8, 4.0),
    "construction": (3.0, 4.2),
    "plaza": (7.0, 9.0),
    "narrow": (1.6, 2.0),
}
SIDES = ("south", "east", "north", "west")


def make_layout(seed=0):
    rng = random.Random(seed)
    xs, ys = [0.0], [0.0]
    for _ in range(GRID):
        xs.append(xs[-1] + rng.uniform(56.0, 68.0))
        ys.append(ys[-1] + rng.uniform(56.0, 68.0))
    wx = [rng.uniform(7.0, 9.5) for _ in range(GRID + 1)]
    wy = [rng.uniform(7.0, 9.5) for _ in range(GRID + 1)]

    variants = list(DEFAULT_VARIANTS)
    if seed != 0:
        rng.shuffle(variants)

    blocks = {}
    for row in range(GRID):
        for col in range(GRID):
            variant = variants[row * GRID + col]
            lo, hi = SIDEWALK_RANGE[variant]
            block = {
                "id": f"b{col}{row}",
                "col": col,
                "row": row,
                "variant": variant,
                "x0": xs[col] + wx[col] / 2,
                "x1": xs[col + 1] - wx[col + 1] / 2,
                "y0": ys[row] + wy[row] / 2,
                "y1": ys[row + 1] - wy[row + 1] / 2,
                "curb_h": rng.uniform(0.10, 0.16),
                "sidewalk_w": rng.uniform(lo, hi),
                "sides": {},
            }
            block["sidewalk_z"] = block["curb_h"]
            ring = CURB_WIDTH + block["sidewalk_w"]
            block["ix0"] = block["x0"] + ring
            block["ix1"] = block["x1"] - ring
            block["iy0"] = block["y0"] + ring
            block["iy1"] = block["y1"] - ring
            for side in SIDES:
                road_w = road_width_at(side, col, row, wx, wy)
                bike = variant in ("street", "terrace", "construction") and rng.random() < 0.5
                block["sides"][side] = {
                    "length": side_length(block, side),
                    "bike_w": min(1.6, 0.18 * road_w) if bike else 0.0,
                    "parked_cars": variant == "narrow",
                    "road_w": road_w,
                }
            if variant == "terrace":
                block["terrace"] = {
                    "side": rng.choice(SIDES),
                    "s0": None,  # filled below once the side length is known
                    "z": block["sidewalk_z"] + N_STEPS * RISE,
                }
                length = block["sides"][block["terrace"]["side"]]["length"]
                block["terrace"]["s0"] = rng.uniform(0.3 * length, 0.7 * length - STAIR_W - RAMP_W)
            if variant == "construction":
                closed = rng.sample(SIDES, 2)
                block["closures"] = []
                for side in closed:
                    length = block["sides"][side]["length"]
                    block["closures"].append({"side": side, "s0": rng.uniform(12.0, length - 18.0), "len": rng.uniform(5.0, 7.0)})
            blocks[block["id"]] = block

    layout = {
        "seed": seed,
        "xs": xs,
        "ys": ys,
        "wx": wx,
        "wy": wy,
        "blocks": blocks,
        "extent": (
            xs[0] - wx[0] / 2 - OUTER_MARGIN,
            xs[-1] + wx[-1] / 2 + OUTER_MARGIN,
            ys[0] - wy[0] / 2 - OUTER_MARGIN,
            ys[-1] + wy[-1] / 2 + OUTER_MARGIN,
        ),
        "road_extent": (xs[0] - wx[0] / 2, xs[-1] + wx[-1] / 2, ys[0] - wy[0] / 2, ys[-1] + wy[-1] / 2),
    }
    rx0, rx1, ry0, ry1 = layout["road_extent"]
    layout["nav_extent"] = (rx0 - NAV_MARGIN, rx1 + NAV_MARGIN, ry0 - NAV_MARGIN, ry1 + NAV_MARGIN)
    layout["crosswalks"] = make_crosswalks(layout)
    # One inner street gets cars stopped in its driving lanes.
    layout["traffic_street"] = rng.choice([("h", j) for j in range(1, GRID)] + [("v", i) for i in range(1, GRID)])
    return layout


def road_width_at(side, col, row, wx, wy):
    return {
        "south": wy[row],
        "north": wy[row + 1],
        "west": wx[col],
        "east": wx[col + 1],
    }[side]


def side_length(block, side):
    if side in ("south", "north"):
        return block["x1"] - block["x0"]
    return block["y1"] - block["y0"]


def side_frame(block, side):
    """Origin (s=0, d=0), tangent, and inward unit vectors for one block side."""
    x0, x1, y0, y1 = block["x0"], block["x1"], block["y0"], block["y1"]
    if side == "south":
        return (x0, y0), (1.0, 0.0), (0.0, 1.0)
    if side == "east":
        return (x1, y0), (0.0, 1.0), (-1.0, 0.0)
    if side == "north":
        return (x1, y1), (-1.0, 0.0), (0.0, -1.0)
    return (x0, y1), (0.0, -1.0), (1.0, 0.0)


def side_point(block, side, s, d):
    origin, tangent, inward = side_frame(block, side)
    return (
        origin[0] + s * tangent[0] + d * inward[0],
        origin[1] + s * tangent[1] + d * inward[1],
    )


def side_rect(block, side, s0, s1, d0, d1):
    """Axis-aligned world rect (x0, x1, y0, y1) covering a side-local s/d box."""
    a = side_point(block, side, s0, d0)
    b = side_point(block, side, s1, d1)
    return (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]))


def block_side_at(layout, x, y):
    """(block, side) whose road edge is nearest to (x, y), or (None, None) outside every block."""
    for block in layout["blocks"].values():
        if block["x0"] <= x <= block["x1"] and block["y0"] <= y <= block["y1"]:
            dists = {
                "south": y - block["y0"],
                "north": block["y1"] - y,
                "west": x - block["x0"],
                "east": block["x1"] - x,
            }
            return block, min(dists, key=dists.get)
    return None, None


def make_crosswalks(layout):
    """Zebra crossings over inner streets, next to every block corner that has a block across."""
    xs, ys, wx, wy = layout["xs"], layout["ys"], layout["wx"], layout["wy"]
    blocks = layout["blocks"]
    crossings = []

    def ring(block):
        return CURB_WIDTH + block["sidewalk_w"]

    # Crossing vertical line i (walk along X), legs above and below each horizontal line j.
    for i in range(1, GRID):
        for j in range(GRID + 1):
            for row, sign in ((j, 1.0), (j - 1, -1.0)):
                if not 0 <= row < GRID:
                    continue
                west, east = blocks[f"b{i - 1}{row}"], blocks[f"b{i}{row}"]
                width = max(1.4, min(3.4, min(ring(west), ring(east)) - 0.4))
                edge = ys[j] + sign * wy[j] / 2
                near = edge + sign * 0.4
                far = near + sign * width
                crossings.append({
                    "id": f"cw_v{i}_{j}_{'n' if sign > 0 else 's'}",
                    "walk_axis": "x",
                    "x0": xs[i] - wx[i] / 2,
                    "x1": xs[i] + wx[i] / 2,
                    "y0": min(near, far),
                    "y1": max(near, far),
                    "ends": (west["id"], east["id"]),
                    "curb_h": (west["curb_h"], east["curb_h"]),
                })
    # Crossing horizontal line j (walk along Y), legs left and right of each vertical line i.
    for j in range(1, GRID):
        for i in range(GRID + 1):
            for col, sign in ((i, 1.0), (i - 1, -1.0)):
                if not 0 <= col < GRID:
                    continue
                south, north = blocks[f"b{col}{j - 1}"], blocks[f"b{col}{j}"]
                width = max(1.4, min(3.4, min(ring(south), ring(north)) - 0.4))
                edge = xs[i] + sign * wx[i] / 2
                near = edge + sign * 0.4
                far = near + sign * width
                crossings.append({
                    "id": f"cw_h{j}_{i}_{'e' if sign > 0 else 'w'}",
                    "walk_axis": "y",
                    "x0": min(near, far),
                    "x1": max(near, far),
                    "y0": ys[j] - wy[j] / 2,
                    "y1": ys[j] + wy[j] / 2,
                    "ends": (south["id"], north["id"]),
                    "curb_h": (south["curb_h"], north["curb_h"]),
                })
    return crossings


def describe(layout):
    lines = [f"world seed={layout['seed']} extent={tuple(round(v, 1) for v in layout['extent'])}"]
    for block in layout["blocks"].values():
        bikes = [side for side, info in block["sides"].items() if info["bike_w"] > 0]
        lines.append(
            f"  {block['id']} {block['variant']:<12} sidewalk={block['sidewalk_w']:.2f} "
            f"curb={block['curb_h']:.3f} bike_lanes={','.join(bikes) or '-'}"
        )
    lines.append(f"  crosswalks={len(layout['crosswalks'])}")
    return "\n".join(lines)
