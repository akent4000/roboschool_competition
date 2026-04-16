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
        size_m: float = 20.0,
        resolution: float = 0.05,
        depth_hfov_deg: float = 86.0,
        depth_max_m: float = 3.5,
        depth_min_m: float = 0.15,
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
        """Project a depth image into the 2-D grid."""
        H, W = depth_img.shape
        fx = W / (2.0 * math.tan(self.depth_hfov / 2.0))
        cx = W / 2.0

        # horizontal strip from the middle of the depth image
        mid = H // 2
        band = 10
        row_lo = max(0, mid - band)
        row_hi = min(H, mid + band)
        strip = np.nanmedian(depth_img[row_lo:row_hi, :], axis=0)

        rgx, rgy = self.w2g(rx, ry)
        col_step = 3

        for u in range(0, W, col_step):
            d = float(strip[u])
            bearing = math.atan2(cx - u, fx)  # positive = left
            ga = rtheta + bearing
            cos_ga = math.cos(ga)
            sin_ga = math.sin(ga)

            if self.depth_min < d < self.depth_max and math.isfinite(d):
                # z-depth → euclidean range along the ray
                cos_bearing = math.cos(bearing)
                range_d = d / cos_bearing if abs(cos_bearing) > 1e-6 else d
                # occupied endpoint
                px = rx + range_d * cos_ga
                py = ry + range_d * sin_ga
                ogx, ogy = self.w2g(px, py)
                # free-space ray
                self._ray_free(rgx, rgy, ogx, ogy)
                # mark occupied
                if self.in_bounds(ogx, ogy):
                    self.grid[ogy, ogx] = min(
                        self.grid[ogy, ogx] + self.L_OCC, self.L_MAX
                    )
            elif math.isfinite(d) and d >= self.depth_max:
                # ray to max range — all free, no obstacle
                cos_bearing = math.cos(bearing)
                max_range = self.depth_max / cos_bearing if abs(cos_bearing) > 1e-6 else self.depth_max
                ex = rx + max_range * cos_ga
                ey = ry + max_range * sin_ga
                egx, egy = self.w2g(ex, ey)
                self._ray_free(rgx, rgy, egx, egy)

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
        result = occ.copy()

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                # source slice
                sy0 = max(0, -dy)
                sy1 = min(self.h, self.h - dy)
                sx0 = max(0, -dx)
                sx1 = min(self.w, self.w - dx)
                # destination slice
                dy0 = max(0, dy)
                dy1 = min(self.h, self.h + dy)
                dx0 = max(0, dx)
                dx1 = min(self.w, self.w + dx)
                result[dy0:dy1, dx0:dx1] |= occ[sy0:sy1, sx0:sx1]

        self._inflated = result
        self._dirty = False
        return result

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
        self, rx: float, ry: float, n: int = 5
    ) -> List[Tuple[float, float]]:
        """Return up to *n* frontier cluster centroids, scored by distance & size."""
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

        rgx, rgy = g.w2g(rx, ry)
        scored = []  # type: List[Tuple[float, float, float]]
        for cl in clusters:
            if len(cl) < self.min_size:
                continue
            mean_gx = sum(c[0] for c in cl) / len(cl)
            mean_gy = sum(c[1] for c in cl) / len(cl)
            wx, wy = g.g2w(int(mean_gx), int(mean_gy))
            dist = math.hypot(mean_gx - rgx, mean_gy - rgy) * g.resolution
            # prefer closer, larger frontiers
            score = dist - 0.3 * math.sqrt(len(cl)) * g.resolution
            scored.append((wx, wy, score))

        scored.sort(key=lambda t: t[2])
        return [(t[0], t[1]) for t in scored[:n]]

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

        # --- line-of-sight smoothing ---
        return self._los_simplify(path_grid, blocked)

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
        self.grid = OccupancyGrid(size_m=20.0, resolution=0.05)
        self.explorer = FrontierExplorer(self.grid, min_size=5)
        self.planner = PathPlanner(self.grid)
        self.dt = control_dt

        # navigation state
        self.path: Optional[List[Tuple[float, float]]] = None
        self.path_idx: int = 0
        self.target: Optional[Tuple[float, float]] = None
        self._explicit_target: Optional[Tuple[float, float]] = None

        # step counters for throttled updates
        self._last_map: int = -999
        self._last_frontier: int = -999
        self._last_plan: int = -999

        # update intervals (in simulation steps)
        self.MAP_INTERVAL: int = 5       # ~10 Hz
        self.FRONTIER_INTERVAL: int = 50  # ~1 Hz
        self.PLAN_INTERVAL: int = 50      # ~1 Hz

        # navigation tuning
        self.WP_REACH: float = 0.3       # waypoint reached radius (m)
        self.MAX_VX: float = 0.4
        self.MIN_VX: float = 0.08
        self.MAX_WZ: float = 0.8
        self.TURN_THRESH: float = 0.4    # turn-in-place if angle error > this
        self.OBS_STOP: float = 0.45      # stop if obstacle closer than this (m)
        self.OBS_SLOW: float = 0.80      # slow down zone (m)

        # Pure Pursuit
        self.LOOKAHEAD: float = 0.8      # look-ahead distance on path (m)

        # Low-pass filter for angular velocity (reduces jitter)
        self._prev_wz: float = 0.0
        self.WZ_ALPHA: float = 0.3       # 0 = freeze, 1 = no filter

        # Dead zone: skip yaw corrections smaller than this
        self.YAW_DEADZONE: float = math.radians(4.0)  # ~4°

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

    def set_navigation_target(self, wx: float, wy: float) -> None:
        """Override exploration with a specific world-coordinate goal."""
        self._explicit_target = (wx, wy)
        self._last_plan = -999  # force replan

    def clear_navigation_target(self) -> None:
        """Resume frontier-based exploration."""
        self._explicit_target = None

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
        elif (step_index - self._last_frontier) >= self.FRONTIER_INTERVAL:
            targets = self.explorer.find_targets(rx, ry, n=3)
            self.target = targets[0] if targets else None
            self._last_frontier = step_index
            self._last_plan = -999  # force replan on new target

        # 4. plan path (throttled)
        if self.target is not None and (step_index - self._last_plan) >= self.PLAN_INTERVAL:
            p = self.planner.plan(rx, ry, self.target[0], self.target[1])
            if p and len(p) > 1:
                self.path = p
                self.path_idx = 1  # skip the start cell
            else:
                self.path = None
            self._last_plan = step_index

        # 5. generate velocity command
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
        front_clear, turn_dir, speed_scale = self._obstacle_check(depth)

        if not front_clear:
            # obstacle too close — back up while turning away
            wz = turn_dir * 0.6
            self._prev_wz = wz
            return -0.15, 0.0, wz

        wp = self._pure_pursuit_target(rx, ry)
        if wp is None:
            # no target — spin slowly to discover space
            self._prev_wz = 0.5
            return 0.0, 0.0, 0.5

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

        if abs_err > self.TURN_THRESH:
            # large heading error — turn in place (crawl forward slightly)
            vx = self.MIN_VX
            wz_raw = max(min(err * 2.0, self.MAX_WZ), -self.MAX_WZ)
        else:
            # curvature-based speed: reduce vx proportionally to yaw error
            curv_factor = 1.0 - 0.6 * (abs_err / self.TURN_THRESH) if self.TURN_THRESH > 0 else 1.0
            speed = self.MAX_VX * speed_scale * curv_factor
            if dist < 0.5:
                speed *= max(dist / 0.5, 0.25)
            vx = max(speed, self.MIN_VX)
            wz_raw = max(min(err * 1.5, self.MAX_WZ), -self.MAX_WZ)

        # Low-pass filter on angular velocity
        wz = self.WZ_ALPHA * wz_raw + (1.0 - self.WZ_ALPHA) * self._prev_wz
        self._prev_wz = wz

        return vx, 0.0, wz

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
    ) -> Tuple[bool, float, float]:
        """Check the depth image for close obstacles.

        Returns ``(front_is_clear, turn_direction, speed_scale)``.
        ``turn_direction`` is +1 (turn left) or -1 (turn right).
        ``speed_scale`` is 0.0–1.0, used to slow down near obstacles.
        """
        if depth is None:
            return True, 0.0, 1.0

        H, W = depth.shape
        row_lo, row_hi = H // 3, 2 * H // 3

        def safe_min(arr: np.ndarray) -> float:
            v = arr[np.isfinite(arr) & (arr > 0.1)]
            if len(v) == 0:
                return 99.0
            return float(np.percentile(v, 10))

        center_dist = safe_min(depth[row_lo:row_hi, W // 3: 2 * W // 3])

        if center_dist <= self.OBS_STOP:
            # blocked — decide which way to turn
            left_dist = safe_min(depth[row_lo:row_hi, : W // 3])
            right_dist = safe_min(depth[row_lo:row_hi, 2 * W // 3:])
            turn = 1.0 if left_dist > right_dist else -1.0
            return False, turn, 0.0

        if center_dist < self.OBS_SLOW:
            # slow-down zone — scale speed linearly
            scale = (center_dist - self.OBS_STOP) / (self.OBS_SLOW - self.OBS_STOP)
            return True, 0.0, max(scale, 0.15)

        return True, 0.0, 1.0
