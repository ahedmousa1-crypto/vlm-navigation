"""Human path planning on the navmap. Plain Python + numpy (runs inside or outside Blender).

A* over 8-connected cells. Each step costs distance x mean surface cost
(classes.HUMAN_COST); cells next to obstacles are blocked so paths keep
about a shoulder's width of clearance; height jumps above MAX_STEP_M are not
allowed, so walls and terrace edges stop a path while curbs and stair steps do not.
"""

import heapq
import math

import numpy as np

from .classes import CLASSES, HUMAN_COST, MAX_STEP_M

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
CLEARANCE_CELLS = 1      # blocked ring around obstacles (x cell size)
NEAR_OBSTACLE_COST = 0.6  # extra cost one more ring out, so paths don't hug furniture


def dilate(mask, cells):
    out = mask.copy()
    for _ in range(cells):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


class HumanPlanner:
    def __init__(self, navmap):
        self.cls = np.asarray(navmap["cls"])
        self.height = np.asarray(navmap["height"], dtype=np.float64)
        self.origin = np.asarray(navmap["origin"], dtype=np.float64)
        self.cell = float(navmap["cell"])
        self.ny, self.nx = self.cls.shape

        lut = np.full(256, np.inf)
        for index, (name, _rgb) in CLASSES.items():
            if name in HUMAN_COST:
                lut[index] = HUMAN_COST[name]
        base = lut[self.cls]
        obstacle = ~np.isfinite(base)
        blocked = dilate(obstacle, CLEARANCE_CELLS)
        near = dilate(blocked, 1) & ~blocked
        self.cost = np.where(blocked, np.inf, base + np.where(near, NEAR_OBSTACLE_COST, 0.0))
        self.surface_cost = base

    # ---- coordinates ---------------------------------------------------

    def to_cell(self, x, y):
        col = int((x - self.origin[0]) / self.cell)
        row = int((y - self.origin[1]) / self.cell)
        return row, col

    def to_world(self, row, col):
        return (self.origin[0] + (col + 0.5) * self.cell, self.origin[1] + (row + 0.5) * self.cell)

    def inside(self, row, col):
        return 0 <= row < self.ny and 0 <= col < self.nx

    def class_at(self, x, y):
        row, col = self.to_cell(x, y)
        return int(self.cls[row, col]) if self.inside(row, col) else 0

    def height_at(self, x, y):
        row, col = self.to_cell(x, y)
        return float(self.height[row, col]) if self.inside(row, col) else 0.0

    def walkable(self, row, col, max_cost=np.inf):
        return self.inside(row, col) and self.cost[row, col] <= max_cost

    def nearest_walkable(self, x, y, radius=2.0, max_cost=1.6, prefer=None):
        """Walkable cell (cost <= max_cost) within ``radius`` of (x, y); closest to ``prefer`` if given, else to (x, y)."""
        row0, col0 = self.to_cell(x, y)
        reach = int(math.ceil(radius / self.cell))
        best, best_d = None, math.inf
        target = prefer if prefer is not None else (x, y)
        for row in range(row0 - reach, row0 + reach + 1):
            for col in range(col0 - reach, col0 + reach + 1):
                if not self.walkable(row, col, max_cost):
                    continue
                wx, wy = self.to_world(row, col)
                if math.hypot(wx - x, wy - y) > radius:
                    continue
                d = math.hypot(wx - target[0], wy - target[1])
                if d < best_d:
                    best, best_d = (row, col), d
        return best

    # ---- search --------------------------------------------------------

    def astar(self, start, goal, banned_classes=(), margin_m=30.0):
        """Cell path from ``start`` to ``goal`` (row, col tuples), or None."""
        if not (self.walkable(*start) and self.walkable(*goal)):
            return None
        margin = int(margin_m / self.cell)
        rmin = max(0, min(start[0], goal[0]) - margin)
        rmax = min(self.ny - 1, max(start[0], goal[0]) + margin)
        cmin = max(0, min(start[1], goal[1]) - margin)
        cmax = min(self.nx - 1, max(start[1], goal[1]) + margin)
        cost = self.cost
        if banned_classes:
            cost = cost.copy()
            cost[np.isin(self.cls, list(banned_classes))] = np.inf
        height = self.height

        open_heap = [(0.0, 0.0, start)]
        came = {start: None}
        g_best = {start: 0.0}
        gr, gc = goal
        while open_heap:
            _f, g, node = heapq.heappop(open_heap)
            if node == goal:
                path = []
                while node is not None:
                    path.append(node)
                    node = came[node]
                return path[::-1]
            if g > g_best.get(node, math.inf):
                continue
            row, col = node
            c_here = cost[row, col]
            h_here = height[row, col]
            for dr, dc, dist in NEIGHBOURS:
                nr, nc = row + dr, col + dc
                if nr < rmin or nr > rmax or nc < cmin or nc > cmax:
                    continue
                c_next = cost[nr, nc]
                if not math.isfinite(c_next):
                    continue
                if abs(height[nr, nc] - h_here) > MAX_STEP_M:
                    continue
                if dr and dc and not (math.isfinite(cost[row, nc]) and math.isfinite(cost[nr, col])):
                    continue  # no corner cutting past obstacles
                ng = g + dist * self.cell * 0.5 * (c_here + c_next)
                if ng < g_best.get((nr, nc), math.inf):
                    g_best[(nr, nc)] = ng
                    came[(nr, nc)] = node
                    h = math.hypot(nr - gr, nc - gc) * self.cell
                    heapq.heappush(open_heap, (ng + h, ng, (nr, nc)))
        return None

    def path_cost(self, cells):
        total = 0.0
        for a, b in zip(cells, cells[1:]):
            dist = math.hypot(a[0] - b[0], a[1] - b[1]) * self.cell
            total += dist * 0.5 * (self.cost[a] + self.cost[b])
        return total

    # ---- smoothing -----------------------------------------------------

    def segment_ok(self, a, b, max_surface_cost):
        """Straight walk a->b stays on cells no worse than ``max_surface_cost`` without big height jumps."""
        (ar, ac), (br, bc) = a, b
        steps = max(1, int(math.hypot(br - ar, bc - ac) * 2.5))
        prev_h = self.height[a]
        for k in range(1, steps + 1):
            t = k / steps
            row = int(round(ar + (br - ar) * t))
            col = int(round(ac + (bc - ac) * t))
            if not math.isfinite(self.cost[row, col]) or self.surface_cost[row, col] > max_surface_cost + 1e-6:
                return False
            h = self.height[row, col]
            if abs(h - prev_h) > MAX_STEP_M:
                return False
            prev_h = h
        return True

    def smooth(self, cells):
        """Greedy line-of-sight shortcutting that never cuts across worse surfaces than the A* path used."""
        if len(cells) < 3:
            return list(cells)
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1:
                worst = max(self.surface_cost[c] for c in cells[i:j + 1])
                if self.segment_ok(cells[i], cells[j], worst):
                    break
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def to_points(self, cells, spacing=0.25):
        """World (x, y, z) points along the cell polyline, resampled every ``spacing`` metres."""
        pts = [self.to_world(*c) for c in cells]
        out = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(length / spacing))
            for k in range(n):
                t = k / n
                x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                out.append((x, y, self.height_at(x, y)))
        x, y = pts[-1]
        out.append((x, y, self.height_at(x, y)))
        return out

    def classes_along(self, points):
        return [CLASSES[self.class_at(x, y)][0] for x, y, _z in points]
