"""World plan: a grid of blocks separated by streets (3x3 core, extended to 5x4). Plain Python, no bpy.

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

GRID = 3  # the original core town; extension blocks are added around it
VARIANTS = ("street", "plaza", "narrow", "construction", "terrace", "neighborhood", "park", "skate")
# Seed 0 layout; other seeds shuffle it. Every variant appears at least once.
DEFAULT_VARIANTS = [
    "street", "plaza", "narrow",
    "construction", "terrace", "street",
    "narrow", "street", "plaza",
]
# Phase 2 extension: (col, row) -> variant, growing the 3x3 core to 5 columns x 4 rows.
EXTENSION = {
    (0, 3): "neighborhood", (1, 3): "park", (2, 3): "neighborhood",
    (3, 0): "plaza", (3, 1): "park", (3, 2): "skate", (3, 3): "neighborhood",
    (4, 0): "street", (4, 1): "neighborhood", (4, 2): "park", (4, 3): "street",
}
EXT_COLS, EXT_ROWS = 5, 4
# Whole blocks held out for testing: two core blocks plus one new park and one new neighborhood.
TEST_BLOCKS = ("b12", "b22", "b42", "b03")
SIDEWALK_RANGE = {
    "street": (2.6, 4.4),
    "terrace": (2.8, 4.0),
    "construction": (3.0, 4.2),
    "plaza": (7.0, 9.0),
    "narrow": (1.6, 2.0),
    "neighborhood": (1.8, 2.4),
    "park": (2.6, 3.4),
    "skate": (3.0, 4.0),
}
SIDES = ("south", "east", "north", "west")


def make_layout(seed=0, extended=True):
    """Core 3x3 town (identical to Phase 1 for the same seed) plus, if ``extended``, the Phase 2 blocks.

    The core consumes the main random stream exactly as before, so its blocks
    and every prop placed in them stay where they were; the extension uses its
    own stream.
    """
    layout = make_core_layout(seed)
    if extended:
        extend_layout(layout, seed)
    return layout


def make_core_layout(seed):
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
        "cols": GRID,
        "rows": GRID,
        "blocks": blocks,
        "test_blocks": [b for b in TEST_BLOCKS if b in blocks],
    }
    set_extents(layout)
    layout["crosswalks"] = make_crosswalks(layout)
    # One inner street gets cars stopped in its driving lanes.
    layout["traffic_street"] = rng.choice([("h", j) for j in range(1, GRID)] + [("v", i) for i in range(1, GRID)])
    return layout


def set_extents(layout):
    xs, ys, wx, wy = layout["xs"], layout["ys"], layout["wx"], layout["wy"]
    layout["extent"] = (
        xs[0] - wx[0] / 2 - OUTER_MARGIN,
        xs[-1] + wx[-1] / 2 + OUTER_MARGIN,
        ys[0] - wy[0] / 2 - OUTER_MARGIN,
        ys[-1] + wy[-1] / 2 + OUTER_MARGIN,
    )
    layout["road_extent"] = (xs[0] - wx[0] / 2, xs[-1] + wx[-1] / 2, ys[0] - wy[0] / 2, ys[-1] + wy[-1] / 2)
    rx0, rx1, ry0, ry1 = layout["road_extent"]
    layout["nav_extent"] = (rx0 - NAV_MARGIN, rx1 + NAV_MARGIN, ry0 - NAV_MARGIN, ry1 + NAV_MARGIN)


def make_block(rng, col, row, variant, xs, ys, wx, wy):
    """A block record with the same fields the core uses (sides, rings, interior)."""
    lo, hi = SIDEWALK_RANGE[variant]
    block = {
        "id": f"b{col}{row}",
        "col": col,
        "row": row,
        "variant": variant,
        "new": True,
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
    block["ix0"], block["ix1"] = block["x0"] + ring, block["x1"] - ring
    block["iy0"], block["iy1"] = block["y0"] + ring, block["y1"] - ring
    for side in SIDES:
        road_w = road_width_at(side, col, row, wx, wy)
        bike = variant in ("street", "park") and rng.random() < 0.5
        block["sides"][side] = {
            "length": side_length(block, side),
            "bike_w": min(1.6, 0.18 * road_w) if bike else 0.0,
            "parked_cars": variant == "neighborhood" and rng.random() < 0.5,
            "road_w": road_w,
        }
    return block


def extend_layout(layout, seed):
    """Add the Phase 2 columns/row and their blocks without touching the core."""
    rng = random.Random(seed * 1000003 + 17)
    xs, ys, wx, wy = layout["xs"], layout["ys"], layout["wx"], layout["wy"]
    while len(xs) < EXT_COLS + 1:
        xs.append(xs[-1] + rng.uniform(56.0, 68.0))
        wx.append(rng.uniform(7.0, 9.5))
    while len(ys) < EXT_ROWS + 1:
        ys.append(ys[-1] + rng.uniform(56.0, 68.0))
        wy.append(rng.uniform(7.0, 9.5))
    cells = sorted(EXTENSION)
    variants = [EXTENSION[cell] for cell in cells]
    if seed != 0:
        rng.shuffle(variants)
    for (col, row), variant in zip(cells, variants):
        block = make_block(rng, col, row, variant, xs, ys, wx, wy)
        if variant == "neighborhood":
            block["lots"] = plan_lots(rng, block)
        elif variant == "park":
            block["park"] = plan_park(rng, block)
        elif variant == "skate":
            block["skate"] = plan_skate(rng, block)
        block["extras"] = True  # scattered street objects and accessible entrances
        layout["blocks"][block["id"]] = block
    layout["cols"], layout["rows"] = EXT_COLS, EXT_ROWS
    layout["test_blocks"] = [b for b in TEST_BLOCKS if b in layout["blocks"]]
    set_extents(layout)
    layout["crosswalks"] = make_crosswalks(layout)


# ---- plans for the new block types (all world coordinates) --------------

HOUSE_COLORS = ("white", "yellow", "blue", "red", "gray", "green")
PORCH_RISE = 3 * RISE      # three steps up to the front door
RAMP_SLOPE = 1.0 / 8.0     # accessible ramp gradient


def plan_lots(rng, block):
    """Houses facing the south and north streets, 3-4 per side, each with lawn, driveway and front walk."""
    lots = []
    for side in ("south", "north"):
        length = block["ix1"] - block["ix0"]
        count = rng.choice((3, 4))
        width = length / count
        for k in range(count):
            a = block["ix0"] + k * width
            house_w = rng.uniform(0.55, 0.68) * width
            lots.append({
                "side": side,
                "x0": a,
                "x1": a + width,
                "house_w": house_w,
                "house_d": rng.uniform(8.0, 10.0),
                "setback": rng.uniform(5.0, 7.0),
                "floors": rng.choice((1, 2, 2)),
                "color": rng.choice(HOUSE_COLORS),
                "ramp": rng.random() < 0.5,
                "driveway_left": rng.random() < 0.5,
                "car": rng.random() < 0.5,
                "mailbox": rng.random() < 0.85,
                "bin": rng.random() < 0.6,
                "hedge": rng.random() < 0.6,
            })
    return lots


def plan_park(rng, block):
    """Central plaza with a fountain, four spokes, an inner loop, a pond crossed by a bridge, a playground."""
    ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
    path_w = rng.uniform(2.4, 3.0)
    half = 4.5  # central plaza half-size (fountain radius 2.1)
    inset = rng.uniform(5.0, 6.0)
    lx0, lx1, ly0, ly1 = ix0 + inset, ix1 - inset, iy0 + inset, iy1 - inset
    segments = [
        (cx, iy0 - 0.2, cx, cy - half), (cx, cy + half, cx, iy1 + 0.2),  # spokes reach into the sidewalk ring
        (ix0 - 0.2, cy, cx - half, cy), (cx + half, cy, ix1 + 0.2, cy),
        (lx0, ly0, lx1, ly0), (lx1, ly0, lx1, ly1), (lx1, ly1, lx0, ly1), (lx0, ly1, lx0, ly0),
    ]
    # The pond sits on one loop side; that stretch of the loop becomes a bridge.
    pond_side = rng.randrange(4, 8)
    sx0, sy0, sx1, sy1 = segments[pond_side]
    t = rng.uniform(0.25, 0.35) if rng.random() < 0.5 else rng.uniform(0.65, 0.75)
    px, py = sx0 + (sx1 - sx0) * t, sy0 + (sy1 - sy0) * t
    along_x = abs(sx1 - sx0) > abs(sy1 - sy0)
    pond = {"x": px, "y": py, "rx": rng.uniform(5.0, 6.5) if along_x else 3.6,
            "ry": 3.6 if along_x else rng.uniform(5.0, 6.5)}
    span = (pond["rx"] if along_x else pond["ry"]) + 0.8
    bridge = {"axis": "x" if along_x else "y", "x": px, "y": py, "half_len": span, "half_w": path_w / 2 + 0.2}
    # Playground and picnic area in two quadrants between the plaza and the loop, away from the pond.
    pond_quad = (1 if px > cx else -1, 1 if py > cy else -1)
    quads = [q for q in ((-1, -1), (1, -1), (1, 1), (-1, 1)) if q != pond_quad]
    rng.shuffle(quads)
    hx, hy = (lx1 - lx0) / 2, (ly1 - ly0) / 2
    room = min(hx, hy) - half - path_w - 1.0

    def quad_rect(q, size):
        size = min(size, room)
        qx = cx + q[0] * (half + (hx - half) / 2)
        qy = cy + q[1] * (half + (hy - half) / 2)
        return (qx - size / 2, qx + size / 2, qy - size / 2, qy + size / 2)
    return {
        "path_w": path_w,
        "plaza": (cx - half, cx + half, cy - half, cy + half),
        "segments": segments,
        "loop": (lx0, lx1, ly0, ly1),
        "pond": pond,
        "bridge": bridge,
        "playground": quad_rect(quads[0], 9.0),
        "picnic": quad_rect(quads[1], 9.0),
        "statue": (cx + quads[2][0] * (half + 3.0), cy + quads[2][1] * (half + 3.0)),
    }


def plan_skate(rng, block):
    """Concrete skate area on one half (ramps, boxes, rails, a stair set) and a basketball court on the other."""
    ix0, ix1, iy0, iy1 = block["ix0"], block["ix1"], block["iy0"], block["iy1"]
    split_x = rng.random() < 0.5
    if split_x:
        mid = (ix0 + ix1) / 2
        pad, court = (ix0, mid - 1.0, iy0, iy1), (mid + 1.0, ix1, iy0, iy1)
    else:
        mid = (iy0 + iy1) / 2
        pad, court = (ix0, ix1, iy0, mid - 1.0), (ix0, ix1, mid + 1.0, iy1)
    px0, px1, py0, py1 = pad
    features = []
    kinds = ["quarter_pipe", "quarter_pipe", "funbox", "rail", "rail", "ledge", "ledge", "stair_set"]
    rng.shuffle(kinds)
    cols, rows = 4, 2
    for k, kind in enumerate(kinds):
        fx = px0 + (k % cols + 0.5) * (px1 - px0) / cols
        fy = py0 + (k // cols + 0.5) * (py1 - py0) / rows
        yaw = rng.choice((0.0, 1.5708, 3.1416, -1.5708))
        if kind == "stair_set":  # its 9 m footprint runs along the pad's long axis, where cells are longest
            yaw = rng.choice((1.5708, -1.5708)) if (py1 - py0) > (px1 - px0) else rng.choice((0.0, 3.1416))
        features.append({"kind": kind, "x": fx + rng.uniform(-1.0, 1.0), "y": fy + rng.uniform(-1.0, 1.0), "yaw": yaw})
    return {"pad": pad, "court": court, "features": features}


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
    cols, rows = layout.get("cols", GRID), layout.get("rows", GRID)
    crossings = []

    def ring(block):
        return CURB_WIDTH + block["sidewalk_w"]

    # Crossing vertical line i (walk along X), legs above and below each horizontal line j.
    for i in range(1, cols):
        for j in range(rows + 1):
            for row, sign in ((j, 1.0), (j - 1, -1.0)):
                if not 0 <= row < rows:
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
    for j in range(1, rows):
        for i in range(cols + 1):
            for col, sign in ((i, 1.0), (i - 1, -1.0)):
                if not 0 <= col < cols:
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
