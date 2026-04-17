"""
2D SLAM for Aliengo quadruped competition.

Components:
- OdometryTracker: dead-reckoning from body velocity integration
- OccupancyGrid: 2D log-odds grid built from depth camera
- FrontierExplorer: frontier-based exploration (free cells adjacent to unknown)
- PathPlanner: BFS on inflated occupancy grid
- SlamController: orchestrates mapping, planning, and navigation
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------

class OdometryTracker:
    """Integrates body-frame velocities to estimate global pose (x, y, theta)."""

    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0

    def update(self, vx: float, vy: float, wz: float, dt: float) -> None:
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        self.x += (vx * c - vy * s) * dt
        self.y += (vx * s + vy * c) * dt
        self.theta += wz * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta


# ---------------------------------------------------------------------------
# Occupancy Grid
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """2D occupancy grid with log-odds updates from a forward-facing depth camera."""

    def __init__(
        self,
        size_m: float = 40.0,
        resolution: float = 0.05,
        depth_hfov_deg: float = 86.0,
        depth_max_m: float = 4.0,
        depth_min_m: float = 0.4,
    ) -> None:
        self.resolution = resolution
        self.size_m = size_m
        self.w = int(size_m / resolution)
        self.h = int(size_m / resolution)
        self.origin = size_m / 2.0  # robot starts at grid centre

        # log-odds map: 0 = unknown, >0 = occupied, <0 = free
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)

        self.L_MAX = 5.0
        self.L_MIN = -5.0
        self.L_OCC = 0.85
        self.L_FREE = -0.4
        self.OCC_THRESH = 1.5
        self.FREE_THRESH = -1.0

        self.depth_hfov = math.radians(depth_hfov_deg)
        self.depth_max = depth_max_m
        self.depth_min = depth_min_m

        # cached inflated grid for path planning
        self._inflated: Optional[np.ndarray] = None
        self._dirty = True

    # -- coordinate helpers --------------------------------------------------

    def w2g(self, wx: float, wy: float) -> Tuple[int, int]:
        """World → grid cell."""
        return (
            int((wx + self.origin) / self.resolution),
            int((wy + self.origin) / self.resolution),
        )

    def g2w(self, gx: int, gy: int) -> Tuple[float, float]:
        """Grid cell → world (cell centre)."""
        return (
            gx * self.resolution - self.origin,
            gy * self.resolution - self.origin,
        )

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.w and 0 <= gy < self.h

    # -- map update ----------------------------------------------------------

    def update_from_depth(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        depth_img: np.ndarray,
    ) -> None:
        """Project a depth image into the 2-D grid (vectorised)."""
        H, W = depth_img.shape
        fx = W / (2.0 * math.tan(self.depth_hfov / 2.0))
        cx = W / 2.0

        # horizontal strip from the middle of the depth image
        mid = H // 2
        band = 10
        strip = np.nanmedian(depth_img[max(0, mid - band):min(H, mid + band), :], axis=0)

        col_step = 3
        u_arr = np.arange(0, W, col_step)
        d_arr = strip[u_arr].astype(np.float32)

        bearings = np.arctan2(cx - u_arr, fx)
        ga_arr = rtheta + bearings
        cos_b = np.cos(bearings)
        cos_ga = np.cos(ga_arr)
        sin_ga = np.sin(ga_arr)

        rgx, rgy = self.w2g(rx, ry)
        finite = np.isfinite(d_arr)

        # --- occupied rays ---
        hit = finite & (d_arr > self.depth_min) & (d_arr < self.depth_max)
        if hit.any():
            safe_cos = np.where(np.abs(cos_b[hit]) > 1e-6, cos_b[hit], 1.0)
            range_d = d_arr[hit] / safe_cos
            px = rx + range_d * cos_ga[hit]
            py = ry + range_d * sin_ga[hit]
            ogx_arr = ((px + self.origin) / self.resolution).astype(int)
            ogy_arr = ((py + self.origin) / self.resolution).astype(int)
            in_b = (ogx_arr >= 0) & (ogx_arr < self.w) & (ogy_arr >= 0) & (ogy_arr < self.h)
            np.add.at(self.grid, (ogy_arr[in_b], ogx_arr[in_b]), self.L_OCC)
            np.clip(self.grid, self.L_MIN, self.L_MAX, out=self.grid)
            for ogx_i, ogy_i in zip(ogx_arr[in_b], ogy_arr[in_b]):
                self._ray_free(rgx, rgy, ogx_i, ogy_i)

        # --- far rays (all free) ---
        far = finite & (d_arr >= self.depth_max)
        if far.any():
            safe_cos_f = np.where(np.abs(cos_b[far]) > 1e-6, cos_b[far], 1.0)
            max_range = self.depth_max / safe_cos_f
            ex = rx + max_range * cos_ga[far]
            ey = ry + max_range * sin_ga[far]
            egx_arr = ((ex + self.origin) / self.resolution).astype(int)
            egy_arr = ((ey + self.origin) / self.resolution).astype(int)
            for egx_i, egy_i in zip(egx_arr, egy_arr):
                self._ray_free(rgx, rgy, egx_i, egy_i)

        self._dirty = True

    def _ray_free(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Mark cells along a ray (exclusive of endpoint) as free (DDA)."""
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        skip = max(2, steps // 50)
        for i in range(0, steps, skip):
            t = i / steps
            gx = int(x0 + t * dx)
            gy = int(y0 + t * dy)
            if 0 <= gx < self.w and 0 <= gy < self.h:
                self.grid[gy, gx] = max(
                    self.grid[gy, gx] + self.L_FREE, self.L_MIN
                )

    # -- inflated obstacle mask for planning ---------------------------------

    def inflated(self, r: int = 4) -> np.ndarray:
        """Boolean mask with obstacles inflated by *r* cells (cached)."""
        if not self._dirty and self._inflated is not None:
            return self._inflated

        occ = self.grid > self.OCC_THRESH
        try:
            import cv2 as _cv2
            y_idx, x_idx = np.ogrid[-r:r + 1, -r:r + 1]
            kernel = ((x_idx ** 2 + y_idx ** 2) <= r * r).astype(np.uint8)
            self._inflated = _cv2.dilate(
                occ.astype(np.uint8), kernel
            ).astype(bool)
        except ImportError:
            # fallback: manual numpy dilation
            result = occ.copy()
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy > r * r:
                        continue
                    sy0, sy1 = max(0, -dy), min(self.h, self.h - dy)
                    sx0, sx1 = max(0, -dx), min(self.w, self.w - dx)
                    dy0, dy1 = max(0, dy), min(self.h, self.h + dy)
                    dx0, dx1 = max(0, dx), min(self.w, self.w + dx)
                    result[dy0:dy1, dx0:dx1] |= occ[sy0:sy1, sx0:sx1]
            self._inflated = result

        self._dirty = False
        return self._inflated

    # -- debug visualisation -------------------------------------------------

    def to_image(self) -> np.ndarray:
        """Export the grid as an RGB image (white=free, black=occupied, grey=unknown)."""
        img = np.full((self.h, self.w, 3), 128, dtype=np.uint8)
        img[self.grid < self.FREE_THRESH] = [255, 255, 255]
        img[self.grid > self.OCC_THRESH] = [0, 0, 0]
        return img


# ---------------------------------------------------------------------------
# Frontier Explorer
# ---------------------------------------------------------------------------

class FrontierExplorer:
    """Finds frontier clusters (free cells adjacent to unknown space)."""

    def __init__(self, grid: OccupancyGrid, min_size: int = 5) -> None:
        self.grid = grid
        self.min_size = min_size

    def find_targets(
        self,
        rx: float,
        ry: float,
        n: int = 5,
        strategy: str = "least_explored",
    ) -> List[Tuple[float, float]]:
        """Return up to *n* frontier targets.

        ``strategy='least_explored'`` prefers frontiers bordering larger unknown areas.
        ``strategy='nearest'`` prefers closer frontiers (legacy behavior).
        """
        g = self.grid
        free = g.grid < g.FREE_THRESH
        unknown = (g.grid >= g.FREE_THRESH) & (g.grid <= g.OCC_THRESH)

        # frontier = free cell with at least one unknown 4-neighbour
        padded = np.pad(unknown, 1, constant_values=False)
        has_unknown_nb = (
            padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
        frontier_mask = free & has_unknown_nb

        fy, fx = np.where(frontier_mask)
        if len(fx) == 0:
            return []

        clusters = self._cluster(fx, fy)

        _UNKNOWN_RADIUS = int(1.8 / g.resolution)
        rgx, rgy = g.w2g(rx, ry)

        scored = []  # type: List[Tuple[float, float, float]]
        for cl in clusters:
            if len(cl) < self.min_size:
                continue
            mean_gx = sum(c[0] for c in cl) / len(cl)
            mean_gy = sum(c[1] for c in cl) / len(cl)
            wx, wy = g.g2w(int(mean_gx), int(mean_gy))
            dist = math.hypot(mean_gx - rgx, mean_gy - rgy) * g.resolution

            cx, cy = int(mean_gx), int(mean_gy)
            unknown_frac = self._local_unknown_ratio(cx, cy, radius_cells=_UNKNOWN_RADIUS)

            if strategy == "nearest":
                score = dist - 0.3 * math.sqrt(len(cl)) * g.resolution
            else:
                size_bonus = min(math.sqrt(len(cl)) * g.resolution, 1.5)
                score = unknown_frac + 0.15 * size_bonus - 0.03 * dist

            scored.append((wx, wy, score))

        if strategy == "nearest":
            scored.sort(key=lambda t: t[2])
        else:
            scored.sort(key=lambda t: t[2], reverse=True)

        return [(t[0], t[1]) for t in scored[:n]]

    def find_least_explored_target(
        self,
        rx: float,
        ry: float,
        sample_step_cells: int = 20,
        patch_radius_cells: int = 20,
    ) -> Optional[Tuple[float, float]]:
        """Pick a free target near the least explored region of the grid."""
        g = self.grid
        blocked = g.inflated(r=4)
        rgx, rgy = g.w2g(rx, ry)

        best_score = -1.0
        best_cell = None  # type: Optional[Tuple[int, int]]

        x0 = patch_radius_cells
        y0 = patch_radius_cells
        x1 = max(g.w - patch_radius_cells, x0 + 1)
        y1 = max(g.h - patch_radius_cells, y0 + 1)
        step = max(sample_step_cells, 1)

        for gy in range(y0, y1, step):
            for gx in range(x0, x1, step):
                if blocked[gy, gx]:
                    continue

                unknown_ratio = self._local_unknown_ratio(
                    gx,
                    gy,
                    radius_cells=patch_radius_cells,
                )
                if unknown_ratio < 0.25:
                    continue

                dist = math.hypot(gx - rgx, gy - rgy) * g.resolution
                if dist < 1.0:
                    continue

                score = unknown_ratio + 0.02 * min(dist, 8.0)
                if score > best_score:
                    best_score = score
                    best_cell = (gx, gy)

        if best_cell is None:
            return None

        return g.g2w(*best_cell)

    def _local_unknown_ratio(self, gx: int, gy: int, radius_cells: int) -> float:
        """Fraction of unknown cells around a grid coordinate."""
        g = self.grid
        x0 = max(0, gx - radius_cells)
        x1 = min(g.w, gx + radius_cells + 1)
        y0 = max(0, gy - radius_cells)
        y1 = min(g.h, gy + radius_cells + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0

        patch = g.grid[y0:y1, x0:x1]
        unknown = (patch >= g.FREE_THRESH) & (patch <= g.OCC_THRESH)
        return float(np.mean(unknown)) if unknown.size > 0 else 0.0

    @staticmethod
    def _cluster(
        fx: np.ndarray, fy: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """Connected-component clustering via flood fill."""
        fset = set(zip(fx.tolist(), fy.tolist()))
        visited = set()  # type: set
        clusters = []  # type: List[List[Tuple[int, int]]]

        for pt in fset:
            if pt in visited:
                continue
            cl = []  # type: List[Tuple[int, int]]
            q = deque([pt])  # type: deque
            visited.add(pt)
            while q:
                cx, cy = q.popleft()
                cl.append((cx, cy))
                if len(cl) > 500:
                    break  # cap cluster size for performance
                for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (cx + ddx, cy + ddy)
                    if nb in fset and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            clusters.append(cl)

        return clusters


# ---------------------------------------------------------------------------
# Path Planner (A* with turn penalty + line-of-sight smoothing)
# ---------------------------------------------------------------------------

class PathPlanner:
    """A* path planner with turn penalty and line-of-sight path smoothing."""

    MAX_ITER = 200_000
    LAMBDA_TURN = 0.3   # turn penalty weight (radians → grid-cost units)

    _NEIGHBORS = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    )
    _SQRT2 = math.sqrt(2.0)

    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid

    def plan(
        self,
        sx: float,
        sy: float,
        gx: float,
        gy: float,
        inflate_r: int = 4,
        dense_step_m: float = 0.25,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan a path and return smoothed waypoints in world coords, or None."""
        g = self.grid
        start = g.w2g(sx, sy)
        goal = g.w2g(gx, gy)

        if not g.in_bounds(*start) or not g.in_bounds(*goal):
            return None

        blocked = g.inflated(inflate_r)

        if blocked[goal[1], goal[0]]:
            new_goal = self._nearest_free(goal, blocked)
            if new_goal is None:
                return None
            goal = new_goal

        if blocked[start[1], start[0]]:
            new_start = self._nearest_free(start, blocked)
            if new_start is None:
                return None
            start = new_start

        # --- A* search with 8-connectivity and turn penalty ---
        def h(a: Tuple[int, int]) -> float:
            return math.hypot(a[0] - goal[0], a[1] - goal[1])

        # heap entries: (f_cost, g_cost, node)
        open_heap = [(h(start), 0.0, start)]  # type: list
        g_cost = {start: 0.0}    # type: dict
        came_from = {start: None}  # type: dict
        in_dir = {}  # type: dict  # node → (dx, dy) incoming direction

        found = False
        iters = 0

        while open_heap and iters < self.MAX_ITER:
            f, gc, cur = heapq.heappop(open_heap)
            iters += 1

            if cur == goal:
                found = True
                break

            if gc > g_cost.get(cur, float("inf")):
                continue  # stale entry

            cur_dir = in_dir.get(cur)  # None for the start node

            for ddx, ddy in self._NEIGHBORS:
                nb = (cur[0] + ddx, cur[1] + ddy)
                if not g.in_bounds(*nb) or blocked[nb[1], nb[0]]:
                    continue

                step_dist = self._SQRT2 if (ddx != 0 and ddy != 0) else 1.0

                # turn penalty: angle change between incoming and outgoing
                turn_cost = 0.0
                if cur_dir is not None:
                    a_in = math.atan2(cur_dir[1], cur_dir[0])
                    a_out = math.atan2(ddy, ddx)
                    delta = abs(math.atan2(
                        math.sin(a_out - a_in), math.cos(a_out - a_in)
                    ))
                    turn_cost = self.LAMBDA_TURN * delta

                new_g = gc + step_dist + turn_cost

                if new_g < g_cost.get(nb, float("inf")):
                    g_cost[nb] = new_g
                    came_from[nb] = cur
                    in_dir[nb] = (ddx, ddy)
                    heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))

        if not found:
            return None

        # --- reconstruct grid path ---
        path_grid = []  # type: List[Tuple[int, int]]
        c = goal  # type: Optional[Tuple[int, int]]
        while c is not None:
            path_grid.append(c)
            c = came_from[c]
        path_grid.reverse()

        # --- line-of-sight smoothing + densification ---
        coarse = self._los_simplify(path_grid, blocked)
        return self._densify_world_path(coarse, step_m=dense_step_m)

    # -- line-of-sight path smoothing ----------------------------------------

    def _los_simplify(
        self,
        path: List[Tuple[int, int]],
        blocked: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Remove redundant waypoints: keep only those where LOS is broken."""
        if len(path) <= 2:
            return [self.grid.g2w(*p) for p in path]

        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            # find the furthest point still visible from path[i]
            j = len(path) - 1
            while j > i + 1:
                if self._los_clear(path[i], path[j], blocked):
                    break
                j -= 1
            result.append(path[j])
            i = j

        return [self.grid.g2w(*p) for p in result]

    @staticmethod
    def _densify_world_path(
        path: List[Tuple[float, float]],
        step_m: float,
    ) -> List[Tuple[float, float]]:
        """Insert intermediate points so the default path has many waypoints."""
        if len(path) <= 1 or step_m <= 0.0:
            return path

        dense = [path[0]]
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-6:
                continue
            steps = max(int(math.ceil(seg / step_m)), 1)
            for k in range(1, steps + 1):
                t = k / steps
                dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return dense

    @staticmethod
    def _los_clear(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        blocked: np.ndarray,
    ) -> bool:
        """Return True if a straight line from p1 to p2 crosses no obstacle."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        steps = max(abs(dx), abs(dy), 1)
        bh, bw = blocked.shape
        for k in range(steps + 1):
            t = k / steps
            gx = int(round(p1[0] + t * dx))
            gy = int(round(p1[1] + t * dy))
            if gx < 0 or gx >= bw or gy < 0 or gy >= bh or blocked[gy, gx]:
                return False
        return True

    # -- helpers -------------------------------------------------------------

    def _nearest_free(
        self,
        pt: Tuple[int, int],
        blocked: np.ndarray,
        max_r: int = 30,
    ) -> Optional[Tuple[int, int]]:
        """BFS outward from *pt* to find the nearest non-blocked cell."""
        q = deque([pt])  # type: deque
        vis = {pt}  # type: set
        while q:
            cx, cy = q.popleft()
            if not blocked[cy, cx]:
                return (cx, cy)
            if math.hypot(cx - pt[0], cy - pt[1]) > max_r:
                continue
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cx + ddx, cy + ddy)
                if nb not in vis and self.grid.in_bounds(*nb):
                    vis.add(nb)
                    q.append(nb)
        return None


# ---------------------------------------------------------------------------
# SLAM Controller  (top-level façade)
# ---------------------------------------------------------------------------

class SlamController:
    """Orchestrates mapping, frontier exploration, path planning, and navigation.

    Usage from the control loop::

        slam = SlamController(control_dt=0.02)

        # each tick:
        vx, vy, vw = slam.update(step_index, state, camera_data)
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        self.odom = OdometryTracker()
        self.grid = OccupancyGrid(size_m=40.0, resolution=0.05)
        self.explorer = FrontierExplorer(self.grid, min_size=5)
        self.planner = PathPlanner(self.grid)
        self.dt = control_dt

        # navigation state
        self.path: Optional[List[Tuple[float, float]]] = None
        self.path_idx: int = 0
        self.target: Optional[Tuple[float, float]] = None
        self._explicit_target: Optional[Tuple[float, float]] = None
        self._exclusion_zones: List[Tuple[float, float]] = []  # visited object positions

        # step counters for throttled updates
        self._last_map: int = -999
        self._last_frontier: int = -999
        self._last_plan: int = -999
        self.cached_frontiers: List[Tuple[float, float]] = []

        # update intervals (in simulation steps)
        self.MAP_INTERVAL: int = 10      # ~5 Hz
        self.FRONTIER_INTERVAL: int = 50  # ~1 Hz
        self.PLAN_INTERVAL: int = 50      # ~1 Hz
        self.FRONTIER_CANDIDATES: int = 8
        self.FRONTIER_STRATEGY: str = "least_explored"
        self.PATH_DENSE_STEP_M: float = 0.20

        # navigation tuning
        self.WP_REACH: float = 0.3       # waypoint reached radius (m)
        self.MAX_VX: float = 0.55
        self.MIN_VX: float = 0.10
        self.MAX_WZ: float = 0.8
        self.TURN_THRESH: float = 0.4    # turn-in-place if angle error > this
        self.OBS_STOP: float = 0.55      # stop if obstacle closer than this (m)
        self.OBS_SLOW: float = 1.0       # slow down zone (m)
        self.INFLATE_R: int = 10         # path inflation radius (cells = 0.5m)
        self.LAT_AVOID_THRESH: float = 0.6  # lateral avoidance threshold (m)

        # Pure Pursuit
        self.LOOKAHEAD: float = 0.8      # look-ahead distance on path (m)

        # Straight-line acceleration
        self.BOOST_VX: float = 0.90                    # boosted speed on straights
        self.STRAIGHT_THRESH: float = math.radians(8.0)  # boost when heading error < this

        # Low-pass filter for angular velocity (reduces jitter)
        self._prev_wz: float = 0.0
        self.WZ_ALPHA: float = 0.3       # 0 = freeze, 1 = no filter

        # Dead zone: skip yaw corrections smaller than this
        self.YAW_DEADZONE: float = math.radians(4.0)  # ~4°

        # Anti-stuck: when no frontier is found, pick a random direction
        self._no_target_since: int = 0     # step when target was last None
        self._random_heading: Optional[float] = None  # fallback heading
        self._STUCK_STEPS: int = 100       # ~2s without target → random walk

        # Wall-stuck detection: consecutive steps where front is blocked
        self._blocked_steps: int = 0
        self._BLOCKED_REPLAN: int = 10     # force replan after this many blocked ticks (~0.2s)
        self._BLOCKED_HARD_TURN: int = 30  # aggressive 90° turn after ~0.6s stuck

        # Sticky backing: after obstacle clears, keep backing for this many steps
        # Prevents + - + - oscillation when robot wobbles near a wall at a shallow angle
        self._step_index: int = 0
        self._backing_until_step: int = 0
        self._backing_turn_dir: float = 1.0
        self._BACKING_STICKY_STEPS: int = 65  # ~1.2 s at 50 Hz

    # -- public API ----------------------------------------------------------

    def reset_pose(self) -> None:
        """Call after robot fall / reset. Keeps the map, clears pose & plans."""
        self.odom.reset()
        self.path = None
        self.path_idx = 0
        self.target = None
        self._explicit_target = None
        self._last_frontier = -999
        self._last_plan = -999
        self._prev_wz = 0.0
        self._blocked_steps = 0
        self._backing_until_step = 0

    def set_navigation_target(self, wx: float, wy: float) -> None:
        """Override exploration with a specific world-coordinate goal."""
        if (self._explicit_target is None
                or math.hypot(self._explicit_target[0] - wx,
                              self._explicit_target[1] - wy) > 0.3):
            self._last_plan = -999  # force replan only on significant target change
        self._explicit_target = (wx, wy)

    def clear_navigation_target(self) -> None:
        """Resume frontier-based exploration."""
        self._explicit_target = None

    def force_replan(self) -> None:
        """Force immediate frontier search and path replan on next update()."""
        self._last_frontier = -999
        self._last_plan = -999
        self.path = None
        self.target = None

    def set_exclusion_zones(self, positions: List[Tuple[float, float]]) -> None:
        """Mark visited object positions; frontiers within 1.5 m are deprioritized."""
        self._exclusion_zones = list(positions)

    def update(
        self,
        step_index: int,
        state: object,
        camera_data: Optional[dict],
    ) -> Tuple[float, float, float]:
        """Run one SLAM tick. Returns ``(vx, vy, vw)`` velocity command."""
        dt = getattr(state, "dt", self.dt)

        # 1. odometry
        vx_meas = float(getattr(state, "vx", 0.0))
        vy_meas = float(getattr(state, "vy", 0.0))
        wz_meas = float(getattr(state, "wz", 0.0))
        self.odom.update(vx_meas, vy_meas, wz_meas, dt)
        rx, ry, rt = self.odom.pose

        # 2. map update (throttled)
        depth = camera_data.get("depth") if camera_data else None
        if depth is not None and (step_index - self._last_map) >= self.MAP_INTERVAL:
            self.grid.update_from_depth(rx, ry, rt, depth)
            self._last_map = step_index

        # 3. choose target
        if self._explicit_target is not None:
            self.target = self._explicit_target
            self._no_target_since = step_index
            self._random_heading = None
        elif (step_index - self._last_frontier) >= self.FRONTIER_INTERVAL:
            targets = self.explorer.find_targets(
                rx,
                ry,
                n=self.FRONTIER_CANDIDATES,
                strategy=self.FRONTIER_STRATEGY,
            )
            # Filter out frontiers within 1.5 m of visited objects
            if self._exclusion_zones and targets:
                filtered = [
                    t for t in targets
                    if not any(math.hypot(t[0] - zx, t[1] - zy) < 1.5
                               for zx, zy in self._exclusion_zones)
                ]
                targets = filtered if filtered else targets
            self.cached_frontiers = targets
            if targets:
                self.target = targets[0]
                self._no_target_since = step_index
                self._random_heading = None
            else:
                # No frontiers found: go to the least explored reachable area.
                fallback_target = self.explorer.find_least_explored_target(rx, ry)
                if fallback_target is not None:
                    self.target = fallback_target
                    self._no_target_since = step_index
                    self._random_heading = None
                    print(
                        "[SLAM] No frontiers — heading to least explored area "
                        f"({fallback_target[0]:.2f}, {fallback_target[1]:.2f})"
                    )
                else:
                    # Last-resort anti-stuck: random walk.
                    if (self._random_heading is None
                            or step_index - self._no_target_since > self._STUCK_STEPS):
                        import random as _rnd
                        self._random_heading = _rnd.uniform(-math.pi, math.pi)
                        self._no_target_since = step_index
                        print(f"[SLAM] No frontiers/unknown area — random heading "
                              f"{math.degrees(self._random_heading):.0f}°")
                    _rw_dist = 3.0
                    self.target = (
                        rx + _rw_dist * math.cos(self._random_heading),
                        ry + _rw_dist * math.sin(self._random_heading),
                    )
            self._last_frontier = step_index
            self._last_plan = -999  # force replan on new target

        # 4. plan path (throttled)
        if self.target is not None and (step_index - self._last_plan) >= self.PLAN_INTERVAL:
            p = self.planner.plan(
                rx,
                ry,
                self.target[0],
                self.target[1],
                inflate_r=self.INFLATE_R,
                dense_step_m=self.PATH_DENSE_STEP_M,
            )
            if p and len(p) > 1:
                self.path = p
                self.path_idx = 1  # skip the start cell
            else:
                self.path = None
            self._last_plan = step_index

        # 5. generate velocity command
        self._step_index = step_index
        return self._navigate(rx, ry, rt, depth)

    # -- debug ---------------------------------------------------------------

    def save_map_image(self, path: str) -> None:
        """Save occupancy grid to a PNG for debugging (requires cv2)."""
        try:
            import cv2

            img = self.grid.to_image()
            # robot position — red dot
            rgx, rgy = self.grid.w2g(self.odom.x, self.odom.y)
            if self.grid.in_bounds(rgx, rgy):
                cv2.circle(img, (rgx, rgy), 4, (0, 0, 255), -1)
            # target — green dot
            if self.target is not None:
                tgx, tgy = self.grid.w2g(*self.target)
                if self.grid.in_bounds(tgx, tgy):
                    cv2.circle(img, (tgx, tgy), 4, (0, 255, 0), -1)
            # path — blue line
            if self.path:
                for i in range(len(self.path) - 1):
                    p1 = self.grid.w2g(*self.path[i])
                    p2 = self.grid.w2g(*self.path[i + 1])
                    cv2.line(img, p1, p2, (255, 0, 0), 1)
            cv2.imwrite(path, img)
        except Exception:
            pass

    # -- internals -----------------------------------------------------------

    def _navigate(
        self,
        rx: float,
        ry: float,
        rt: float,
        depth: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """Pure Pursuit controller + reactive obstacle avoidance.

        Compared to the previous proportional controller this adds:
        - lookahead-based waypoint selection (smoother arcs)
        - curvature-based speed scaling (slow down on turns)
        - yaw dead zone (skip tiny corrections)
        - low-pass filter on angular velocity (remove jitter)
        """
        front_clear, turn_dir, speed_scale, lat_vy = self._obstacle_check(depth)

        if not front_clear:
            self._blocked_steps += 1
            # Arm sticky backing: even after front clears, keep retreating
            self._backing_until_step = self._step_index + self._BACKING_STICKY_STEPS
            self._backing_turn_dir = turn_dir

            # Invalidate current path — it leads into the wall
            if self._blocked_steps >= self._BLOCKED_REPLAN:
                if self.path is not None:
                    self.path = None
                    self._last_plan = -999  # force replan next cycle
                # Also clear random walk heading so we pick a new direction
                self._random_heading = None

            # Progressive turn: the longer we're stuck, the harder we turn
            if self._blocked_steps >= self._BLOCKED_HARD_TURN:
                # Aggressive: fast turn + moderate backup
                wz = turn_dir * self.MAX_WZ
                self._prev_wz = wz
                return -0.20, lat_vy, wz
            else:
                wz = turn_dir * 0.6
                self._prev_wz = wz
                return -0.15, lat_vy, wz
        else:
            self._blocked_steps = 0
            # Sticky: continue backing even though front just cleared.
            # This prevents + - + - oscillation when the robot wobbles near
            # a wall at a shallow angle (center depth flickers around OBS_STOP).
            if self._step_index < self._backing_until_step:
                wz = self._backing_turn_dir * 0.5
                self._prev_wz = wz
                return -0.10, 0.0, wz

        wp = self._pure_pursuit_target(rx, ry)
        if wp is None:
            # no target — drive forward while turning to discover new space
            # (pure spin keeps the robot stuck in the same spot)
            self._prev_wz = 0.6
            return 0.25, 0.0, 0.6

        dx = wp[0] - rx
        dy = wp[1] - ry
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        err = math.atan2(
            math.sin(target_angle - rt), math.cos(target_angle - rt)
        )

        # Dead zone: ignore tiny yaw errors
        if abs(err) < self.YAW_DEADZONE:
            err = 0.0

        abs_err = abs(err)

        # Near an obstacle: scale MIN_VX down so the robot can decelerate to 0
        # before the backing phase kicks in (smooth stop, not abrupt jump).
        # speed_scale < 0.3 ≈ closer than ~0.69 m; at OBS_STOP it reaches 0.
        min_vx_eff = self.MIN_VX * min(speed_scale / 0.3, 1.0)

        if abs_err > self.TURN_THRESH:
            # large heading error — turn in place (crawl forward slightly)
            vx = min_vx_eff
            wz_raw = max(min(err * 2.0, self.MAX_WZ), -self.MAX_WZ)
        else:
            # curvature-based speed: reduce vx proportionally to yaw error
            curv_factor = 1.0 - 0.6 * (abs_err / self.TURN_THRESH) if self.TURN_THRESH > 0 else 1.0
            # Boost speed on straight segments (small heading error)
            top_speed = self.BOOST_VX if abs_err < self.STRAIGHT_THRESH else self.MAX_VX
            speed = top_speed * speed_scale * curv_factor
            if dist < 0.5:
                speed *= max(dist / 0.5, 0.25)
            vx = max(speed, min_vx_eff)
            wz_raw = max(min(err * 1.5, self.MAX_WZ), -self.MAX_WZ)

        # Low-pass filter on angular velocity
        wz = self.WZ_ALPHA * wz_raw + (1.0 - self.WZ_ALPHA) * self._prev_wz
        self._prev_wz = wz

        return vx, lat_vy, wz

    def _pure_pursuit_target(
        self, rx: float, ry: float
    ) -> Optional[Tuple[float, float]]:
        """Pure Pursuit: aim at a point *LOOKAHEAD* metres ahead on the path."""
        if not self.path or self.path_idx >= len(self.path):
            return self.target

        # advance past reached waypoints
        while self.path_idx < len(self.path) - 1:
            if math.hypot(self.path[self.path_idx][0] - rx,
                           self.path[self.path_idx][1] - ry) >= self.WP_REACH:
                break
            self.path_idx += 1

        # walk the path to find the point at LOOKAHEAD distance
        for i in range(self.path_idx, len(self.path)):
            wp = self.path[i]
            d = math.hypot(wp[0] - rx, wp[1] - ry)
            if d >= self.LOOKAHEAD:
                # interpolate between previous and this waypoint for a smooth arc
                if i > self.path_idx:
                    prev = self.path[i - 1]
                    d_prev = math.hypot(prev[0] - rx, prev[1] - ry)
                    if d_prev < self.LOOKAHEAD and d > d_prev:
                        t = (self.LOOKAHEAD - d_prev) / (d - d_prev)
                        t = max(0.0, min(1.0, t))
                        return (prev[0] + t * (wp[0] - prev[0]),
                                prev[1] + t * (wp[1] - prev[1]))
                return wp

        # entire remaining path is within lookahead — aim at last waypoint
        return self.path[-1]

    def _obstacle_check(
        self, depth: Optional[np.ndarray]
    ) -> Tuple[bool, float, float, float]:
        """Check the depth image for close obstacles.

        Returns ``(front_is_clear, turn_direction, speed_scale, lateral_vy)``.
        ``turn_direction`` is +1 (turn left) or -1 (turn right).
        ``speed_scale`` is 0.0–1.0, used to slow down near obstacles.
        ``lateral_vy`` is a lateral dodge velocity (positive = left).
        """
        if depth is None:
            return True, 0.0, 1.0, 0.0

        H, W = depth.shape
        row_lo, row_hi = H // 3, 2 * H // 3

        def safe_min(arr: np.ndarray) -> float:
            v = arr[np.isfinite(arr) & (arr > 0.1)]
            if len(v) == 0:
                return 99.0
            return float(np.percentile(v, 10))

        left_dist = safe_min(depth[row_lo:row_hi, : W // 3])
        center_dist = safe_min(depth[row_lo:row_hi, W // 3: 2 * W // 3])
        right_dist = safe_min(depth[row_lo:row_hi, 2 * W // 3:])

        # Lateral avoidance: push away from nearby side obstacles
        lat_vy = 0.0
        lat_thresh = self.LAT_AVOID_THRESH
        if left_dist < lat_thresh and right_dist >= lat_thresh:
            lat_vy = -0.15 * (1.0 - left_dist / lat_thresh)   # dodge right
        elif right_dist < lat_thresh and left_dist >= lat_thresh:
            lat_vy = 0.15 * (1.0 - right_dist / lat_thresh)   # dodge left

        if center_dist <= self.OBS_STOP:
            # blocked — decide which way to turn
            turn = 1.0 if left_dist > right_dist else -1.0
            return False, turn, 0.0, lat_vy

        if center_dist < self.OBS_SLOW:
            # slow-down zone — scale speed linearly to 0 at OBS_STOP
            scale = (center_dist - self.OBS_STOP) / (self.OBS_SLOW - self.OBS_STOP)
            return True, 0.0, max(scale, 0.0), lat_vy

        return True, 0.0, 1.0, lat_vy
