"""
Real-time dashboard visualizer for the Aliengo navigation pipeline.

Shows:
- Camera feed with YOLO detection overlays
- Occupancy map with robot, path, target, and frontiers
- Status bar with velocities, queue progress, and timing
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Class name mapping (must match data.yaml / YOLO training order)
CLASS_NAMES: Dict[int, str] = {
    0: "backpack",
    1: "bottle",
    2: "chair",
    3: "cup",
    4: "laptop",
}

# Distinct colors (BGR) for each class
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 165, 255),   # orange
    1: (0, 255, 0),     # green
    2: (255, 0, 0),     # blue
    3: (0, 255, 255),   # yellow
    4: (255, 0, 255),   # magenta
}

# Map overlay colors (BGR)
_CLR_ROBOT = (0, 0, 255)       # red
_CLR_TRAIL = (0, 0, 180)       # dark red
_CLR_PATH = (255, 160, 0)      # light blue
_CLR_WAYPOINT = (255, 100, 0)  # blue
_CLR_TARGET = (0, 220, 0)      # green
_CLR_FRONTIER = (0, 200, 255)  # yellow
_CLR_EXPLICIT = (255, 0, 255)  # magenta — explicit nav target (detected object)
_CLR_LOOKAHEAD = (255, 255, 0) # cyan


class DashboardVisualizer:
    """All-in-one OpenCV dashboard for the Aliengo navigation system."""

    MAP_DISPLAY_SIZE = 720  # pixels — map panel height & width

    def __init__(
        self,
        enabled: bool = True,
        depth_max_m: float = 4.0,
        trail_length: int = 500,
    ) -> None:
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._trail_length = trail_length
        self._cv2: Any = None
        self._active = False
        self._window_name = "Aliengo Dashboard"
        self._trail: List[Tuple[float, float]] = []
        self._fps_t0 = time.time()
        self._fps_frames = 0
        self._fps_value = 0.0

        if not self.enabled:
            return

        if "QT_QPA_FONTDIR" not in os.environ:
            for font_dir in (
                "/usr/share/fonts/truetype/dejavu",
                "/usr/share/fonts/dejavu",
                "/usr/share/fonts/truetype/liberation2",
            ):
                if os.path.isdir(font_dir):
                    os.environ["QT_QPA_FONTDIR"] = font_dir
                    break

        try:
            import cv2
        except ImportError as exc:
            print(f"[Visualizer] cv2 unavailable ({exc}) — dashboard disabled")
            self.enabled = False
            return

        self._cv2 = cv2
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        self._active = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        detections: Optional[list] = None,
        target_cls: Optional[int] = None,
        slam=None,
        vx_cmd: float = 0.0,
        vy_cmd: float = 0.0,
        wz_cmd: float = 0.0,
        queue: Optional[list] = None,
        queue_idx: int = 0,
        sim_t: float = 0.0,
        confirm_count: int = 0,
        confirm_needed: int = 3,
        known_objects: Optional[dict] = None,
        visited_positions: Optional[list] = None,
    ) -> None:
        """Render one dashboard frame. Call once per control tick (or throttled)."""
        if not self._active:
            return

        cv2 = self._cv2

        # -- camera panel (left) --
        cam_panel = self._render_camera(rgb, depth, detections, target_cls)

        # -- map panel (right) --
        map_panel = self._render_map(slam, known_objects, visited_positions)

        # Scale map to configured display size
        map_panel = cv2.resize(
            map_panel,
            (self.MAP_DISPLAY_SIZE, self.MAP_DISPLAY_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        # Pad camera panel to match map height if needed
        cam_h = cam_panel.shape[0]
        map_h = map_panel.shape[0]
        if cam_h < map_h:
            pad = np.full((map_h - cam_h, cam_panel.shape[1], 3), 30, dtype=np.uint8)
            cam_panel = np.concatenate((cam_panel, pad), axis=0)

        # -- status bar (bottom) --
        status_w = cam_panel.shape[1] + map_panel.shape[1]
        status = self._render_status(
            status_w, vx_cmd, vy_cmd, wz_cmd,
            queue, queue_idx, sim_t, target_cls,
            slam, confirm_count, confirm_needed,
        )

        # -- compose --
        top = np.concatenate((cam_panel, map_panel), axis=1)
        dashboard = np.concatenate((top, status), axis=0)

        cv2.imshow(self._window_name, dashboard)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Camera panel
    # ------------------------------------------------------------------

    def _render_camera(
        self,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        detections: Optional[list],
        target_cls: Optional[int],
    ) -> np.ndarray:
        cv2 = self._cv2

        if rgb is None:
            panel = np.full((360, 640, 3), 40, dtype=np.uint8)
            cv2.putText(panel, "No camera", (220, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            return panel

        img = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # -- draw detections --
        if detections:
            for det in detections:
                cls_id, uc, vc, conf = det[:4]
                # Unpack box size if provided, else use defaults
                bw = det[4] if len(det) > 4 else 80
                bh = det[5] if len(det) > 5 else 60

                is_target = (target_cls is not None and cls_id == target_cls)
                color = (0, 255, 0) if is_target else CLASS_COLORS.get(cls_id, (200, 200, 200))
                thickness = 3 if is_target else 2

                x1 = int(uc - bw / 2)
                y1 = int(vc - bh / 2)
                x2 = int(uc + bw / 2)
                y2 = int(vc + bh / 2)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

                name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                label = f"{name} {conf:.2f}"
                if is_target:
                    label = ">>> " + label + " <<<"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # -- mini depth bar at bottom --
        if depth is not None:
            depth_m = np.asarray(depth, dtype=np.float32)
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
            depth_m = np.clip(depth_m, 0.0, self.depth_max_m)

            # Compress depth to a thin strip the width of the RGB image
            H, W = depth_m.shape
            mid = H // 2
            band = depth_m[mid - 20:mid + 20, :]
            strip = np.nanmedian(band, axis=0)
            strip_u8 = (strip * (255.0 / self.depth_max_m)).astype(np.uint8)

            bar_h = 20
            bar = cv2.applyColorMap(
                np.tile(strip_u8, (bar_h, 1)),
                cv2.COLORMAP_TURBO,
            )
            bar = cv2.resize(bar, (img_bgr.shape[1], bar_h), interpolation=cv2.INTER_LINEAR)

            # distance labels on the bar
            cv2.putText(bar, "Depth", (5, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, f"{self.depth_max_m:.0f}m", (img_bgr.shape[1] - 35, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            img_bgr = np.concatenate((img_bgr, bar), axis=0)

        return img_bgr

    # ------------------------------------------------------------------
    # Map panel
    # ------------------------------------------------------------------

    def _render_map(self, slam, known_objects=None, visited_positions=None) -> np.ndarray:
        cv2 = self._cv2

        if slam is None:
            panel = np.full((400, 400, 3), 128, dtype=np.uint8)
            cv2.putText(panel, "No SLAM", (130, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
            return panel

        # Base occupancy image — flip horizontally to correct mirror
        img = slam.grid.to_image()  # (H, W, 3) uint8
        img = cv2.flip(img, 1)
        H, W = img.shape[:2]

        rx, ry, rt = slam.odom.pose

        # Helper: world → display pixel (accounts for horizontal flip)
        def w2d(wx, wy):
            gx, gy = slam.grid.w2g(wx, wy)
            return (W - 1 - gx, gy)

        def in_disp(dx, dy):
            return 0 <= dx < W and 0 <= dy < H

        # -- robot trail --
        self._trail.append((rx, ry))
        if len(self._trail) > self._trail_length:
            self._trail = self._trail[-self._trail_length:]

        fade_step = max(1, len(self._trail))
        for i, (tx, ty) in enumerate(self._trail):
            dx, dy = w2d(tx, ty)
            if in_disp(dx, dy):
                alpha = int(60 + 140 * (i / fade_step))
                cv2.circle(img, (dx, dy), 1, (0, 0, alpha), -1)

        # -- frontiers (use cached result from SLAM update, not recomputed) --
        frontier_targets = getattr(slam, "cached_frontiers", [])
        for fx, fy in frontier_targets:
            dx, dy = w2d(fx, fy)
            if in_disp(dx, dy):
                cv2.circle(img, (dx, dy), 4, _CLR_FRONTIER, 1)

        # -- planned path --
        if slam.path and len(slam.path) > 1:
            pts = [w2d(*p) for p in slam.path]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], _CLR_PATH, 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                color = _CLR_WAYPOINT
                r = 3
                if i == min(slam.path_idx, len(pts) - 1):
                    color = _CLR_LOOKAHEAD
                    r = 4
                cv2.circle(img, pt, r, color, -1)

        # -- exclusion zones around visited objects (1.5 m circles) --
        if visited_positions:
            for vx, vy in visited_positions:
                dx, dy = w2d(vx, vy)
                r_px = int(1.5 / slam.grid.resolution)
                if in_disp(dx, dy):
                    cv2.circle(img, (dx, dy), r_px, (0, 180, 0), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (dx, dy), (0, 180, 0),
                                   cv2.MARKER_TILTED_CROSS, 8, 2)

        # -- known (remembered) objects --
        if known_objects:
            for cls_id, (kx, ky) in known_objects.items():
                dx, dy = w2d(kx, ky)
                if in_disp(dx, dy):
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    cv2.drawMarker(img, (dx, dy), color,
                                   cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
                    name = CLASS_NAMES.get(cls_id, f"#{cls_id}")
                    cv2.putText(img, name, (dx + 7, dy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

        # -- navigation target --
        if slam.target is not None:
            dx, dy = w2d(*slam.target)
            if in_disp(dx, dy):
                is_explicit = slam._explicit_target is not None
                color = _CLR_EXPLICIT if is_explicit else _CLR_TARGET
                cv2.drawMarker(img, (dx, dy), color,
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                label = "OBJ" if is_explicit else "FRN"
                cv2.putText(img, label, (dx + 8, dy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # -- robot position + heading arrow --
        rdx, rdy = w2d(rx, ry)
        if in_disp(rdx, rdy):
            cv2.circle(img, (rdx, rdy), 5, _CLR_ROBOT, -1)
            arrow_len = 12
            # Negate cos(rt) because x-axis is flipped
            ax = int(rdx - arrow_len * math.cos(rt))
            ay = int(rdy + arrow_len * math.sin(rt))
            cv2.arrowedLine(img, (rdx, rdy), (ax, ay), _CLR_ROBOT, 2, tipLength=0.4)

        # -- legend --
        legend_y = 15
        items = [
            ("Robot", _CLR_ROBOT),
            ("Path", _CLR_PATH),
            ("Frontier", _CLR_FRONTIER),
            ("Target", _CLR_TARGET),
            ("Object", _CLR_EXPLICIT),
        ]
        if known_objects:
            items.append(("Known", (200, 200, 0)))
        if visited_positions:
            items.append(("Visited", (0, 180, 0)))
        for label, color in items:
            cv2.circle(img, (10, legend_y), 4, color, -1)
            cv2.putText(img, label, (18, legend_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            legend_y += 16

        return img

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _render_status(
        self,
        width: int,
        vx: float,
        vy: float,
        wz: float,
        queue: Optional[list],
        queue_idx: int,
        sim_t: float,
        target_cls: Optional[int],
        slam,
        confirm_count: int,
        confirm_needed: int,
    ) -> np.ndarray:
        cv2 = self._cv2

        # FPS calculation
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps_value = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0 = now

        bar_h = 36
        bar = np.full((bar_h, width, 3), 30, dtype=np.uint8)

        # Velocity info
        vel_txt = f"Vel: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}"
        cv2.putText(bar, vel_txt, (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 180), 1, cv2.LINE_AA)

        # Queue progress
        if queue:
            total = len(queue)
            visited = min(queue_idx, total)
            progress_txt = f"Queue: {visited}/{total}"
            if target_cls is not None:
                name = CLASS_NAMES.get(target_cls, f"#{target_cls}")
                progress_txt += f"  Target: {name}"
            if confirm_count > 0:
                progress_txt += f"  Confirm: {confirm_count:.1f}/{confirm_needed:.1f}s"
            cv2.putText(bar, progress_txt, (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 255), 1, cv2.LINE_AA)

            # progress bar
            bar_x0 = width - 160
            bar_x1 = width - 10
            bar_w = bar_x1 - bar_x0
            cv2.rectangle(bar, (bar_x0, 22), (bar_x1, 32), (80, 80, 80), 1)
            fill_w = int(bar_w * visited / max(total, 1))
            if fill_w > 0:
                cv2.rectangle(bar, (bar_x0, 22), (bar_x0 + fill_w, 32), (0, 200, 0), -1)

        # Time + FPS (right side, top line)
        time_txt = f"t={sim_t:.1f}s  FPS={self._fps_value:.0f}"
        (tw, _), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(bar, time_txt, (width - tw - 10, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Odometry position
        if slam is not None:
            rx, ry, rt = slam.odom.pose
            odom_txt = f"Pos: ({rx:.2f}, {ry:.2f})  Yaw: {math.degrees(rt):.0f} deg"
            x_start = width // 3
            cv2.putText(bar, odom_txt, (x_start, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1, cv2.LINE_AA)

        return bar
