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
# Path Planner (BFS)
# ---------------------------------------------------------------------------

class PathPlanner:
    """BFS shortest-path planner on the inflated occupancy grid."""

    MAX_ITER = 200_000

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
        """Plan a path and return simplified waypoints in world coords, or None."""
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

        came_from = {start: None}  # type: dict
        q = deque([start])  # type: deque
        found = False
        iters = 0

        while q and iters < self.MAX_ITER:
            cur = q.popleft()
            iters += 1
            if cur == goal:
                found = True
                break
            for ddx, ddy in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ):
                nb = (cur[0] + ddx, cur[1] + ddy)
                if nb in came_from:
                    continue
                if not g.in_bounds(*nb):
                    continue
                if blocked[nb[1], nb[0]]:
                    continue
                came_from[nb] = cur
                q.append(nb)

        if not found:
            return None

        # reconstruct
        path_grid = []  # type: List[Tuple[int, int]]
        c = goal  # type: Optional[Tuple[int, int]]
        while c is not None:
            path_grid.append(c)
            c = came_from[c]
        path_grid.reverse()

        # simplify to ~20 waypoints
        wp = []  # type: List[Tuple[float, float]]
        step = max(len(path_grid) // 20, 1)
        for i in range(0, len(path_grid), step):
            wp.append(g.g2w(*path_grid[i]))
        last = g.g2w(*goal)
        if not wp or wp[-1] != last:
            wp.append(last)
        return wp

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
        """Proportional controller + reactive obstacle avoidance."""
        front_clear, turn_dir, speed_scale = self._obstacle_check(depth)

        if not front_clear:
            # obstacle too close — stop and turn away
            return 0.0, 0.0, turn_dir * 0.6

        wp = self._current_waypoint(rx, ry)
        if wp is None:
            # no target — spin slowly to discover space
            return 0.0, 0.0, 0.5

        dx = wp[0] - rx
        dy = wp[1] - ry
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        err = math.atan2(
            math.sin(target_angle - rt), math.cos(target_angle - rt)
        )

        if abs(err) > self.TURN_THRESH:
            # large heading error — turn in place (crawl forward slightly)
            vx = self.MIN_VX
            wz = max(min(err * 2.0, self.MAX_WZ), -self.MAX_WZ)
        else:
            speed = self.MAX_VX * speed_scale
            if dist < 0.5:
                speed *= max(dist / 0.5, 0.25)
            vx = max(speed, self.MIN_VX)
            wz = max(min(err * 1.5, self.MAX_WZ), -self.MAX_WZ)

        return vx, 0.0, wz

    def _current_waypoint(
        self, rx: float, ry: float
    ) -> Optional[Tuple[float, float]]:
        """Advance along the planned path and return the current waypoint."""
        if not self.path or self.path_idx >= len(self.path):
            return self.target

        wp = self.path[self.path_idx]
        while math.hypot(wp[0] - rx, wp[1] - ry) < self.WP_REACH:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                return self.target
            wp = self.path[self.path_idx]
        return wp

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
