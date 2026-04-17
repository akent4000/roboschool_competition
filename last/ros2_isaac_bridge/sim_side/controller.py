#!/usr/bin/env python3
"""
ROS 2 competition controller — autonomous navigation with SLAM + YOLO.

Ported from src/aliengo_competition/controllers/main_controller.py.
Runs as a standalone ROS 2 node subscribing to sensor topics from bridge_node
and publishing velocity commands + detected objects.
"""
import sys
import math
import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Int32, String

# ---------------------------------------------------------------------------
# Add project root to path so we can import SLAM and visualizer modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aliengo_competition.controllers.slam import SlamController

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Map overlay colors (BGR) — match visualizer.py
_CLR_ROBOT = (0, 0, 255)
_CLR_PATH = (255, 160, 0)
_CLR_WAYPOINT = (255, 100, 0)
_CLR_TARGET = (0, 220, 0)
_CLR_FRONTIER = (0, 200, 255)
_CLR_EXPLICIT = (255, 0, 255)
_CLR_LOOKAHEAD = (255, 255, 0)
_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 165, 255),  1: (0, 255, 0),  2: (255, 0, 0),
    3: (0, 255, 255),  4: (255, 0, 255),
}


# ---------------------------------------------------------------------------
# Minimal state adapter for SlamController.update()
# ---------------------------------------------------------------------------
class _SimpleState:
    """SlamController.update() reads state.vx, state.vy, state.wz, state.dt."""

    def __init__(self, vx: float, vy: float, wz: float, dt: float):
        self.vx = vx
        self.vy = vy
        self.wz = wz
        self.dt = dt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NAMES: Dict[int, str] = {
    0: "backpack",
    1: "bottle",
    2: "chair",
    3: "cup",
    4: "laptop",
}


# ===========================================================================
# Main controller node
# ===========================================================================
class NavigationController(Node):

    def __init__(self):
        super().__init__("controller")

        self._sensor_cb_group = ReentrantCallbackGroup()
        self._control_cb_group = MutuallyExclusiveCallbackGroup()

        # ======================== ROS I/O ========================
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.detected_object_pub = self.create_publisher(
            Int32, "/competition/detected_object", 10
        )
        self.ctrl_state_pub = self.create_publisher(
            String, "/competition/controller_state", 10
        )
        self.map_image_pub = self.create_publisher(
            Image, "/slam/map_image", 10
        )

        self.vel_sub = self.create_subscription(
            TwistStamped,
            "/aliengo/base_velocity",
            self._vel_cb,
            qos_profile_sensor_data,
            callback_group=self._sensor_cb_group,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            "/aliengo/joint_states",
            self._joint_cb,
            qos_profile_sensor_data,
            callback_group=self._sensor_cb_group,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/aliengo/imu",
            self._imu_cb,
            qos_profile_sensor_data,
            callback_group=self._sensor_cb_group,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            "/aliengo/camera/color/image_raw",
            self._rgb_cb,
            qos_profile_sensor_data,
            callback_group=self._sensor_cb_group,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/aliengo/camera/depth/image_raw",
            self._depth_cb,
            qos_profile_sensor_data,
            callback_group=self._sensor_cb_group,
        )
        self.seq_sub = self.create_subscription(
            String,
            "/competition/object_sequence",
            self._seq_cb,
            10,
            callback_group=self._sensor_cb_group,
        )

        # ======================== Sensor cache ========================
        self.latest_vx: float = 0.0
        self.latest_vy: float = 0.0
        self.latest_wz: float = 0.0
        self.vel_stamp: Optional[float] = None

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        self.latest_joint_state: Dict = {
            "names": [],
            "position": [],
            "velocity": [],
            "stamp_sec": None,
        }
        self.latest_imu: Dict = {"wx": 0.0, "wy": 0.0, "wz": 0.0, "stamp_sec": None}

        # ======================== Object queue ========================
        self.object_queue: List[int] = []
        self._seq_received: bool = False

        # ======================== Timing ========================
        self.control_dt: float = 0.02
        self.warmup_s: float = 3.5
        self.step_count: int = 0
        self.sim_t: float = 0.0
        self._last_loop_wall_t: Optional[float] = None
        self._vel_timeout_s: float = 0.25
        self._last_stale_warn_t: float = -999.0

        # ======================== Camera intrinsics ========================
        # RGB 640×360 @ 70° HFOV
        self._RGB_W, self._RGB_H = 640, 360
        self._RGB_FX = self._RGB_W / (2.0 * math.tan(math.radians(35.0)))
        self._RGB_CX = self._RGB_W / 2.0
        self._RGB_CY = self._RGB_H / 2.0

        # Depth 848×480 @ 86° HFOV
        self._DEPTH_W, self._DEPTH_H = 848, 480
        self._DEPTH_FX = self._DEPTH_W / (2.0 * math.tan(math.radians(43.0)))
        self._DEPTH_FY = self._DEPTH_FX
        self._DEPTH_CX = self._DEPTH_W / 2.0
        self._DEPTH_CY = self._DEPTH_H / 2.0

        # ======================== Detection tuning ========================
        self._DETECT_EVERY: int = max(int(os.environ.get("CTRL_DETECT_EVERY", "3")), 1)
        self._DETECT_CONF: float = 0.6
        self._CONFIRM_DIST_M: float = 0.8
        self._CONFIRM_WAIT_S: float = 2.1
        self._DEPTH_PATCH: int = 15
        self._YOLO_IMGSZ: int = max(int(os.environ.get("CTRL_YOLO_IMGSZ", "416")), 160)

        self._CLASS_CONF: Dict[int, float] = {
            2: 0.82,  # chair — higher threshold to avoid false positives
        }

        self._MEMORY_SIGHT_THRESH: int = 3
        self._STABLE_DETECT_STEPS: int = 10
        self._BACKUP_DURATION_S: float = 1.5
        self._BACKUP_VX: float = -0.25
        self._PASSBY_OFFSET: float = 0.4

        self._OBJ_HEIGHT_M: Dict[int, float] = {
            0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5,
        }

        self._VIS_EVERY: int = max(int(os.environ.get("CTRL_VIS_EVERY", "6")), 1)
        self._MAP_PUB_INTERVAL_S: float = max(
            float(os.environ.get("CTRL_MAP_PUB_INTERVAL_S", "0.33")),
            0.0,
        )
        self._MAP_IMAGE_SIZE: int = max(int(os.environ.get("CTRL_MAP_IMAGE_SIZE", "560")), 256)
        self._SLAM_DEPTH_STRIDE: int = max(int(os.environ.get("CTRL_SLAM_DEPTH_STRIDE", "2")), 1)
        self._last_map_pub_t: float = -1e9

        # ======================== SLAM ========================
        self.slam = SlamController(control_dt=self.control_dt)

        # ======================== YOLO ========================
        self._yolo_model = None
        _YOLO_MODEL_PATH = str(
            _PROJECT_ROOT / "runs" / "yolo_detector" / "train" / "weights" / "best1.pt"
        )
        try:
            if os.path.isfile(_YOLO_MODEL_PATH):
                from ultralytics.models.yolo.model import YOLO

                self._yolo_model = YOLO(_YOLO_MODEL_PATH)
                _model_names = getattr(self._yolo_model, "names", {})
                self.get_logger().info(f"[Detector] YOLO loaded: {_YOLO_MODEL_PATH}")
                self.get_logger().info(f"[Detector] Model classes: {_model_names}")
            else:
                self.get_logger().warn(
                    f"[Detector] No model at {_YOLO_MODEL_PATH} — detection disabled"
                )
        except Exception as e:
            self.get_logger().warn(f"[Detector] Could not load YOLO: {e}")

        # ======================== Dashboard (optional) ========================
        self.dashboard = None
        _enable_dashboard = os.environ.get("ENABLE_DASHBOARD", "0") == "1"
        if _enable_dashboard:
            try:
                from aliengo_competition.controllers.visualizer import DashboardVisualizer

                self.dashboard = DashboardVisualizer(enabled=True, depth_max_m=4.0)
                self.get_logger().info("[Visualizer] Dashboard enabled")
            except Exception as e:
                self.get_logger().warn(f"[Visualizer] Could not start dashboard: {e}")

        # ======================== Navigation state ========================
        self.last_detect_step: int = -999
        self.detections: list = []
        self.queue_idx: int = 0
        self.announced_queue_idx: int = -1
        self.target_world: Optional[Tuple[float, float]] = None
        self.confirming_since_t: Optional[float] = None
        self.nav_active: bool = False
        self.target_visible: bool = False
        self.backup_until_t: float = 0.0
        self.known_objects: Dict[int, Tuple[float, float]] = {}
        self.visited_positions: List[Tuple[float, float]] = []
        self.sight_counts: Dict[int, Tuple[int, float, float]] = {}
        self.yolo_stable_counts: Dict[int, int] = {}
        self.yolo_logged: set = set()
        self.passby_active: bool = False
        self.confirming_proximity: bool = False
        self.queue_completed: bool = False
        self.motion_mode: str = "idle"
        self._trail: List[Tuple[float, float]] = []
        self._trail_length: int = 500
        self._rgb_frame_id: int = 0

        self._yolo_lock = threading.Lock()
        self._yolo_busy: bool = False
        self._yolo_last_request_step: int = -999
        self._yolo_last_request_frame_id: int = -1
        self._yolo_result_version: int = 0
        self._yolo_last_consumed_version: int = 0
        self._yolo_last_log_wall_t: float = -1e9

        # ======================== Timer (50 Hz matching sim dt) ========================
        self.create_timer(
            self.control_dt,
            self._main_loop,
            callback_group=self._control_cb_group,
        )

        self.get_logger().info(
            "Navigation controller started. Waiting for sensor data + object_sequence..."
        )

    # ===================================================================
    # ROS callbacks
    # ===================================================================
    def _vel_cb(self, msg: TwistStamped) -> None:
        self.latest_vx = float(msg.twist.linear.x)
        self.latest_vy = float(msg.twist.linear.y)
        self.latest_wz = float(msg.twist.angular.z)
        self.vel_stamp = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )

    def _joint_cb(self, msg: JointState) -> None:
        self.latest_joint_state = {
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "stamp_sec": (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            ),
        }

    def _imu_cb(self, msg: Imu) -> None:
        self.latest_imu = {
            "wx": float(msg.angular_velocity.x),
            "wy": float(msg.angular_velocity.y),
            "wz": float(msg.angular_velocity.z),
            "stamp_sec": (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            ),
        }

    def _rgb_cb(self, msg: Image) -> None:
        try:
            self.latest_rgb = (
                np.frombuffer(msg.data, dtype=np.uint8)
                .reshape((msg.height, msg.width, 3))
                .copy()
            )
            self._rgb_frame_id += 1
        except ValueError:
            self.get_logger().warning("Failed to reshape RGB image.")

    def _depth_cb(self, msg: Image) -> None:
        try:
            self.latest_depth = (
                np.frombuffer(msg.data, dtype=np.float32)
                .reshape((msg.height, msg.width))
                .copy()
            )
        except ValueError:
            self.get_logger().warning("Failed to reshape depth image.")

    def _seq_cb(self, msg: String) -> None:
        if self._seq_received:
            return
        try:
            raw = json.loads(msg.data)
            self.object_queue = [
                item[0] if isinstance(item, (tuple, list)) else int(item)
                for item in raw
            ]
            self._seq_received = True
            self.get_logger().info(f"Received object_queue: {self.object_queue}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse object_sequence: {e}")

    # ===================================================================
    # Publishing helpers
    # ===================================================================
    def send_command(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.send_command(0.0, 0.0, 0.0)

    def publish_detected_object(self, object_id: int) -> None:
        msg = Int32()
        msg.data = int(object_id)
        self.detected_object_pub.publish(msg)
        self.get_logger().info(
            f"Published detected object: {object_id} ({_NAMES.get(object_id, '?')})"
        )

    # ===================================================================
    # YOLO inference (asynchronous + throttled)
    # ===================================================================
    def _run_yolo(self, rgb: Optional[np.ndarray], step_idx: int) -> Tuple[list, bool]:
        if self._yolo_model is None or rgb is None:
            with self._yolo_lock:
                self.detections = []
            return [], False

        frame_id = self._rgb_frame_id
        frame_copy = None

        with self._yolo_lock:
            should_schedule = (
                (not self._yolo_busy)
                and (step_idx - self._yolo_last_request_step >= self._DETECT_EVERY)
                and (frame_id != self._yolo_last_request_frame_id)
            )
            if should_schedule:
                self._yolo_busy = True
                self._yolo_last_request_step = step_idx
                self._yolo_last_request_frame_id = frame_id
                frame_copy = rgb.copy()

        if frame_copy is not None:
            worker = threading.Thread(
                target=self._yolo_worker,
                args=(frame_copy,),
                daemon=True,
            )
            worker.start()

        with self._yolo_lock:
            dets = list(self.detections)
            fresh = self._yolo_result_version != self._yolo_last_consumed_version
            if fresh:
                self._yolo_last_consumed_version = self._yolo_result_version

        return dets, fresh

    def _yolo_worker(self, rgb_frame: np.ndarray) -> None:
        dets: list = []
        t0 = time.monotonic()
        try:
            model = self._yolo_model
            if model is None:
                return

            results = model(
                rgb_frame[..., ::-1],
                conf=self._DETECT_CONF,
                verbose=False,
                imgsz=self._YOLO_IMGSZ,
            )
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    min_conf = self._CLASS_CONF.get(cls_id, self._DETECT_CONF)
                    if conf < min_conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append((
                        cls_id,
                        (x1 + x2) / 2.0,
                        (y1 + y2) / 2.0,
                        conf,
                        x2 - x1,
                        y2 - y1,
                    ))
        except Exception as e:
            self.get_logger().warning(f"[YOLO] inference error: {e}")
        finally:
            latency_ms = (time.monotonic() - t0) * 1000.0
            with self._yolo_lock:
                self.detections = dets
                self._yolo_result_version += 1
                self._yolo_busy = False

            if dets:
                now = time.monotonic()
                if now - self._yolo_last_log_wall_t >= 1.0:
                    summary = ", ".join(
                        f"{_NAMES.get(d[0], '?')}({d[3]:.2f})" for d in dets
                    )
                    self.get_logger().info(
                        f"[YOLO] {summary} | latency={latency_ms:.0f}ms imgsz={self._YOLO_IMGSZ}"
                    )
                    self._yolo_last_log_wall_t = now

    # ===================================================================
    # Depth sampling / geometry
    # ===================================================================
    def _sample_depth_at(
        self, depth_img: Optional[np.ndarray], u_rgb: float, v_rgb: float
    ) -> Optional[float]:
        if depth_img is None:
            return None
        u_d = int(self._DEPTH_CX + self._DEPTH_FX * (u_rgb - self._RGB_CX) / self._RGB_FX)
        v_d = int(self._DEPTH_CY + self._DEPTH_FY * (v_rgb - self._RGB_CY) / self._RGB_FX)
        H, W = depth_img.shape
        p = self._DEPTH_PATCH
        patch = depth_img[max(0, v_d - p) : min(H, v_d + p),
                          max(0, u_d - p) : min(W, u_d + p)]
        valid = patch[(patch > 0.15) & (patch < 4.0) & np.isfinite(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _pixel_to_world(
        self, u_rgb: float, depth_z: float, ox: float, oy: float, ot: float
    ) -> Tuple[float, float]:
        x_r = depth_z
        y_r = -depth_z * (u_rgb - self._RGB_CX) / self._RGB_FX
        c, s = math.cos(ot), math.sin(ot)
        return (ox + x_r * c - y_r * s, oy + x_r * s + y_r * c)

    def _estimate_dist_from_bbox(self, cls_id: int, box_h: float) -> float:
        real_h = self._OBJ_HEIGHT_M.get(cls_id, 0.35)
        if box_h < 5:
            return 10.0
        dist = (real_h * self._RGB_FX) / box_h
        return max(4.5, min(dist, 12.0))

    def _get_next_target_pos(self) -> Optional[Tuple[float, float]]:
        next_idx = self.queue_idx + 1
        if next_idx >= len(self.object_queue):
            return None
        return self.known_objects.get(self.object_queue[next_idx])

    def _compute_approach_point(
        self,
        target_pos: Tuple[float, float],
        next_pos: Tuple[float, float],
        robot_pos: Tuple[float, float],
    ) -> Tuple[float, float]:
        tx, ty = target_pos
        nx, ny = next_pos
        rx, ry = robot_pos
        dx, dy = nx - tx, ny - ty
        d = math.hypot(dx, dy)
        if d < 0.1:
            return target_pos
        ux, uy = dx / d, dy / d
        perp1 = (-uy, ux)
        perp2 = (uy, -ux)
        offset = self._PASSBY_OFFSET
        ap1 = (tx + perp1[0] * offset, ty + perp1[1] * offset)
        ap2 = (tx + perp2[0] * offset, ty + perp2[1] * offset)
        t1 = math.hypot(ap1[0] - rx, ap1[1] - ry) + math.hypot(ap1[0] - nx, ap1[1] - ny)
        t2 = math.hypot(ap2[0] - rx, ap2[1] - ry) + math.hypot(ap2[0] - nx, ap2[1] - ny)
        return ap1 if t1 < t2 else ap2

    # ===================================================================
    # Object detection + visit confirmation
    # ===================================================================
    def _get_found_object_id(
        self,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        step_idx: int,
        sim_t: float,
    ) -> Optional[int]:
        dets, yolo_ran = self._run_yolo(rgb, step_idx)

        if not self.object_queue or self.queue_idx >= len(self.object_queue):
            return None

        target_cls = self.object_queue[self.queue_idx]

        # --- Update consecutive YOLO detection counters ---
        if yolo_ran:
            seen_classes = set(d[0] for d in dets)
            for sc in seen_classes:
                self.yolo_stable_counts[sc] = self.yolo_stable_counts.get(sc, 0) + 1
            for sc in list(self.yolo_stable_counts):
                if sc not in seen_classes:
                    self.yolo_stable_counts[sc] = 0
            # Log objects that crossed the stable threshold
            visited_set = set(self.object_queue[: self.queue_idx])
            for sc in seen_classes:
                if (
                    sc in self.object_queue
                    and sc not in visited_set
                    and sc not in self.yolo_logged
                    and self.yolo_stable_counts[sc] >= self._STABLE_DETECT_STEPS
                ):
                    self.publish_detected_object(sc)
                    self.yolo_logged.add(sc)
                    self.get_logger().info(
                        f"[LOG] Object {sc} ({_NAMES.get(sc, '?')}) detected "
                        f"(stable {self.yolo_stable_counts[sc]} steps) at t={sim_t:.2f}s"
                    )

        # --- Save world positions of ALL detected objects ---
        rx, ry, rt = self.slam.odom.pose
        visited_set = set(self.object_queue[: self.queue_idx])

        for det in dets if yolo_ran else []:
            cls_id_det = det[0]
            if cls_id_det in visited_set:
                continue
            uc_det, vc_det = det[1], det[2]
            d_det = self._sample_depth_at(depth, uc_det, vc_det)

            if d_det is not None:
                # Depth-confirmed position
                wx_det, wy_det = self._pixel_to_world(uc_det, d_det, rx, ry, rt)
                if cls_id_det not in self.known_objects:
                    self.get_logger().info(
                        f"[Memory] Saved {_NAMES.get(cls_id_det, '?')}"
                        f" at ({wx_det:.2f}, {wy_det:.2f}) [depth]"
                    )
                self.known_objects[cls_id_det] = (wx_det, wy_det)
                # Evict stale wrong-class entries at the same physical position
                for stale_cls in list(self.known_objects.keys()):
                    if stale_cls == cls_id_det:
                        continue
                    sx, sy = self.known_objects[stale_cls]
                    if math.hypot(sx - wx_det, sy - wy_det) < 1.5:
                        del self.known_objects[stale_cls]
                        self.sight_counts.pop(stale_cls, None)
                        self.get_logger().info(
                            f"[Memory] Evicted stale {_NAMES.get(stale_cls, '?')}"
                            f" -> reclassified as {_NAMES.get(cls_id_det, '?')}"
                        )
                        break
            else:
                # Beyond depth range — estimate from bbox size
                bearing = math.atan2(self._RGB_CX - uc_det, self._RGB_FX)
                ga = rt + bearing
                est_dist = self._estimate_dist_from_bbox(cls_id_det, det[5])
                wx_det = rx + est_dist * math.cos(ga)
                wy_det = ry + est_dist * math.sin(ga)
                too_close = any(
                    math.hypot(wx_det - vpx, wy_det - vpy) < 1.5
                    for vpx, vpy in self.visited_positions
                )
                if too_close:
                    continue
                # Accumulate sightings
                prev = self.sight_counts.get(cls_id_det)
                if prev is not None:
                    cnt, px, py = prev
                    if math.hypot(wx_det - px, wy_det - py) < 2.0:
                        cnt += 1
                        avg_x = (px * (cnt - 1) + wx_det) / cnt
                        avg_y = (py * (cnt - 1) + wy_det) / cnt
                        self.sight_counts[cls_id_det] = (cnt, avg_x, avg_y)
                    else:
                        self.sight_counts[cls_id_det] = (1, wx_det, wy_det)
                else:
                    self.sight_counts[cls_id_det] = (1, wx_det, wy_det)
                cnt, sx, sy = self.sight_counts[cls_id_det]
                if cnt >= self._MEMORY_SIGHT_THRESH:
                    if cls_id_det not in self.known_objects:
                        self.get_logger().info(
                            f"[Memory] Saved {_NAMES.get(cls_id_det, '?')}"
                            f" at ({sx:.2f}, {sy:.2f}) [estimated, {cnt} sightings]"
                        )
                    self.known_objects[cls_id_det] = (sx, sy)

        # --- Pick highest-confidence detection of the target class ---
        best, best_conf = None, 0.0
        for det in dets:
            if det[0] == target_cls and det[3] > best_conf:
                best, best_conf = det, det[3]

        if best is None:
            if self.target_visible and self.confirming_since_t is None:
                pass  # target lost
            self.target_visible = False
            if self.confirming_since_t is not None and yolo_ran and not self.confirming_proximity:
                self.get_logger().info(
                    "[Detector] Confirmation cancelled — target absent in fresh YOLO scan"
                )
                self.confirming_since_t = None
                self.target_world = None
                self.nav_active = False
            elif self.confirming_since_t is None:
                self.target_world = None
                self.nav_active = False
            return None

        _, uc, vc, _, _, _ = best

        if not yolo_ran:
            # Cached pixel coords — don't update world position from stale data
            self.target_visible = True
            return None

        d_m = self._sample_depth_at(depth, uc, vc)
        if d_m is None:
            # Object visible but beyond depth range — estimate from bbox
            bearing = math.atan2(self._RGB_CX - uc, self._RGB_FX)
            est_d = self._estimate_dist_from_bbox(target_cls, best[5])
            ga = rt + bearing
            wx = rx + est_d * math.cos(ga)
            wy = ry + est_d * math.sin(ga)
            self.target_world = (wx, wy)
            self.nav_active = True
            self.target_visible = True
            self.confirming_since_t = None
            return None

        wx, wy = self._pixel_to_world(uc, d_m, rx, ry, rt)
        self.target_world = (wx, wy)
        self.nav_active = True
        if not self.target_visible:
            self.get_logger().info(
                f"[Detector] Target acquired: {_NAMES.get(target_cls, '?')} at {d_m:.2f}m"
            )
        self.target_visible = True

        dist = math.hypot(wx - rx, wy - ry)
        if dist < self._CONFIRM_DIST_M:
            if self.confirming_since_t is None:
                self.confirming_since_t = sim_t
                self.get_logger().info(
                    f"[Detector] Target in range ({dist:.2f}m), "
                    f"stopping for {self._CONFIRM_WAIT_S}s confirmation..."
                )
            elapsed = sim_t - self.confirming_since_t
            if elapsed >= self._CONFIRM_WAIT_S:
                confirmed_id = target_cls
                self.queue_idx += 1
                self.confirming_since_t = None
                self.confirming_proximity = False
                self.visited_positions.append((wx, wy))
                self.known_objects.pop(confirmed_id, None)
                self.target_world = None
                self.nav_active = False
                self.backup_until_t = sim_t + (
                    0.3 if self.passby_active else self._BACKUP_DURATION_S
                )
                self.target_visible = False
                self.get_logger().info(
                    f"[Detector] CONFIRMED object {confirmed_id} "
                    f"({_NAMES.get(confirmed_id, '?')}) "
                    f"({self.queue_idx}/{len(self.object_queue)})"
                )
                return confirmed_id
        else:
            if self.confirming_since_t is not None:
                self.get_logger().info(
                    f"[Detector] Confirmation cancelled — left radius ({dist:.2f}m)"
                )
            self.confirming_since_t = None

        return None

    # ===================================================================
    # Main control loop (called by timer at 50 Hz)
    # ===================================================================
    def _main_loop(self) -> None:
        # Wait for at least one velocity measurement before acting
        if self.vel_stamp is None:
            return

        now_wall = time.monotonic()
        if self._last_loop_wall_t is None:
            loop_dt = self.control_dt
        else:
            loop_dt = now_wall - self._last_loop_wall_t
            if loop_dt <= 0.0:
                loop_dt = self.control_dt
            elif loop_dt > 0.2:
                loop_dt = 0.2
        self._last_loop_wall_t = now_wall

        step_idx = self.step_count
        self.step_count += 1
        self.sim_t += loop_dt
        sim_t = self.sim_t

        now_ros_s = float(self.get_clock().now().nanoseconds) * 1e-9
        vel_age = now_ros_s - self.vel_stamp
        if vel_age < 0.0:
            vel_age = 0.0
        vel_fresh = vel_age <= self._vel_timeout_s
        if not vel_fresh and (sim_t - self._last_stale_warn_t) >= 1.0:
            self.get_logger().warn(
                f"[ROS] Stale /aliengo/base_velocity: age={vel_age:.3f}s. Holding robot."
            )
            self._last_stale_warn_t = sim_t

        rgb = self.latest_rgb
        depth = self.latest_depth
        camera_data: Dict = {"image": rgb, "depth": depth}

        depth_for_slam = depth
        if depth is not None and self._SLAM_DEPTH_STRIDE > 1:
            depth_for_slam = depth[:: self._SLAM_DEPTH_STRIDE, :: self._SLAM_DEPTH_STRIDE]
        slam_camera_data: Dict = {"image": rgb, "depth": depth_for_slam}

        # --- Queue progress announcement ---
        if self.queue_idx < len(self.object_queue) and self.announced_queue_idx != self.queue_idx:
            target = self.object_queue[self.queue_idx]
            self.get_logger().info(
                f"[Nav] Target #{self.queue_idx}: {_NAMES.get(target, '?')} (id={target}), "
                f"remaining={len(self.object_queue) - self.queue_idx}"
            )
            self.announced_queue_idx = self.queue_idx
        elif self.queue_idx >= len(self.object_queue) and not self.queue_completed and self.object_queue:
            self.get_logger().info("[Nav] All objects visited!")
            self.queue_completed = True

        # --- Object detection ---
        detected_object_id = self._get_found_object_id(rgb, depth, step_idx, sim_t)

        # --- Proximity-based confirmation for known objects ---
        if detected_object_id is None and self.queue_idx < len(self.object_queue):
            prox_cls = self.object_queue[self.queue_idx]
            prox_pos = self.known_objects.get(prox_cls)
            if prox_pos is not None:
                prx, pry, _ = self.slam.odom.pose
                prox_dist = math.hypot(prox_pos[0] - prx, prox_pos[1] - pry)
                if prox_dist < self._CONFIRM_DIST_M:
                    if self.confirming_since_t is None:
                        self.confirming_since_t = sim_t
                        self.confirming_proximity = True
                        self.get_logger().info(
                            f"[Detector] PROXIMITY in range ({prox_dist:.2f}m), "
                            f"stopping for {self._CONFIRM_WAIT_S}s confirmation..."
                        )
                    elif self.confirming_proximity:
                        elapsed = sim_t - self.confirming_since_t
                        if elapsed >= self._CONFIRM_WAIT_S:
                            detected_object_id = prox_cls
                            self.queue_idx += 1
                            self.confirming_since_t = None
                            self.confirming_proximity = False
                            self.visited_positions.append(prox_pos)
                            self.known_objects.pop(prox_cls, None)
                            self.target_world = None
                            self.nav_active = False
                            self.backup_until_t = sim_t + 0.3
                            self.target_visible = False
                            self.get_logger().info(
                                f"[Detector] PROXIMITY CONFIRMED {_NAMES.get(prox_cls, '?')} "
                                f"at dist={prox_dist:.2f}m "
                                f"({self.queue_idx}/{len(self.object_queue)})"
                            )
                elif self.confirming_proximity:
                    self.confirming_since_t = None
                    self.confirming_proximity = False
                    self.get_logger().info(
                        "[Detector] Proximity confirmation cancelled — left radius"
                    )

        # --- Publish detection + force SLAM replan ---
        if detected_object_id is not None:
            self.publish_detected_object(detected_object_id)
            self.slam.force_replan()

        # --- SLAM navigation ---
        self.slam.set_exclusion_zones(self.visited_positions)

        # Choose navigation target
        nav_pos: Optional[Tuple[float, float]] = None
        if self.nav_active and self.target_world is not None:
            nav_pos = self.target_world
        elif (
            self.queue_idx < len(self.object_queue)
            and self.object_queue[self.queue_idx] in self.known_objects
        ):
            nav_pos = self.known_objects[self.object_queue[self.queue_idx]]

        if nav_pos is not None:
            next_pos = self._get_next_target_pos()
            if next_pos is not None:
                prx, pry, _ = self.slam.odom.pose
                approach = self._compute_approach_point(nav_pos, next_pos, (prx, pry))
                self.slam.set_navigation_target(*approach)
                self.passby_active = True
            else:
                self.slam.set_navigation_target(*nav_pos)
                self.passby_active = False
        else:
            self.slam.clear_navigation_target()
            self.passby_active = False

        # Feed SLAM with sensor data and get velocity command
        vx_meas = self.latest_vx if vel_fresh else 0.0
        vy_meas = self.latest_vy if vel_fresh else 0.0
        wz_meas = self.latest_wz if vel_fresh else 0.0
        state = _SimpleState(vx_meas, vy_meas, wz_meas, loop_dt)
        vx_cmd, vy_cmd, vw_cmd = self.slam.update(step_idx, state, slam_camera_data)

        # --- Motion mode selection ---
        if not vel_fresh:
            vx, vy, vw = 0.0, 0.0, 0.0
            self.motion_mode = "stale_data_stop"
        elif self.queue_idx >= len(self.object_queue) and len(self.object_queue) > 0:
            vx, vy, vw = 0.0, 0.0, 0.0
            self.motion_mode = "all_done_stop"
        elif sim_t < self.warmup_s:
            vx, vy, vw = 0.0, 0.0, 0.0
            self.motion_mode = "warmup_stop"
        elif sim_t < self.backup_until_t:
            vx, vy, vw = self._BACKUP_VX, 0.0, 0.0
            self.motion_mode = "backup"
        elif self.confirming_since_t is not None:
            vx, vy, vw = 0.0, 0.0, 0.0
            self.motion_mode = "confirm_wait"
        else:
            vx, vy, vw = vx_cmd, vy_cmd, vw_cmd
            self.motion_mode = "navigate"

        self.send_command(vx, vy, vw)

        # --- Publish controller state + map for web dashboard ---
        if step_idx % self._VIS_EVERY == 0:
            self._publish_dashboard_data(sim_t, vx, vy, vw)

        # --- Dashboard visualization (optional, throttled) ---
        if self.dashboard is not None and step_idx % self._VIS_EVERY == 0:
            cur_target = None
            if self.object_queue and self.queue_idx < len(self.object_queue):
                cur_target = self.object_queue[self.queue_idx]
            try:
                self.dashboard.update(
                    rgb=rgb,
                    depth=depth,
                    detections=self.detections,
                    target_cls=cur_target,
                    slam=self.slam,
                    vx_cmd=vx,
                    vy_cmd=vy,
                    wz_cmd=vw,
                    queue=self.object_queue,
                    queue_idx=self.queue_idx,
                    sim_t=sim_t,
                    confirm_count=(
                        int(sim_t - self.confirming_since_t)
                        if self.confirming_since_t is not None
                        else 0
                    ),
                    confirm_needed=int(self._CONFIRM_WAIT_S),
                    known_objects=self.known_objects,
                    visited_positions=self.visited_positions,
                )
            except Exception as e:
                self.get_logger().warning(f"[Visualizer] update failed: {e}")
                try:
                    self.dashboard.close()
                except Exception:
                    pass
                self.dashboard = None


    # ===================================================================
    # Dashboard publishing (controller state + SLAM map)
    # ===================================================================
    def _publish_dashboard_data(self, sim_t: float, vx: float, vy: float, vw: float) -> None:
        """Publish controller state JSON and rendered SLAM map for the web dashboard."""
        self._publish_controller_state(sim_t, vx, vy, vw)
        if (sim_t - self._last_map_pub_t) >= self._MAP_PUB_INTERVAL_S:
            self._publish_map_image()
            self._last_map_pub_t = sim_t

    def _publish_controller_state(self, sim_t: float, vx: float, vy: float, vw: float) -> None:
        rx, ry, rt = self.slam.odom.pose

        target_cls = None
        target_cls_name = ""
        if self.object_queue and self.queue_idx < len(self.object_queue):
            target_cls = self.object_queue[self.queue_idx]
            target_cls_name = _NAMES.get(target_cls, f"#{target_cls}")

        confirm_elapsed = 0.0
        if self.confirming_since_t is not None:
            confirm_elapsed = sim_t - self.confirming_since_t

        # Build known objects list
        known_list = [
            {"id": cid, "name": _NAMES.get(cid, f"#{cid}"), "x": round(pos[0], 3), "y": round(pos[1], 3)}
            for cid, pos in self.known_objects.items()
        ]

        # Build visited positions list
        visited_list = [
            {"x": round(p[0], 3), "y": round(p[1], 3)}
            for p in self.visited_positions
        ]

        # Build detections list
        det_list = [
            {"cls": d[0], "u": round(d[1], 1), "v": round(d[2], 1),
             "conf": round(d[3], 3), "w": round(d[4], 1), "h": round(d[5], 1)}
            for d in self.detections
        ]

        # Nav target
        nav_target = None
        nav_target_type = ""
        if self.slam.target is not None:
            is_explicit = self.slam._explicit_target is not None
            nav_target = {"x": round(self.slam.target[0], 3), "y": round(self.slam.target[1], 3)}
            nav_target_type = "object" if is_explicit else "frontier"

        state_dict = {
            "sim_time": round(sim_t, 2),
            "motion_mode": self.motion_mode,
            "odom": {"x": round(rx, 3), "y": round(ry, 3), "yaw_deg": round(math.degrees(rt), 1)},
            "target_cls": target_cls,
            "target_cls_name": target_cls_name,
            "confirm_elapsed": round(confirm_elapsed, 2),
            "confirm_needed": round(self._CONFIRM_WAIT_S, 2),
            "queue_idx": self.queue_idx,
            "known_objects": known_list,
            "visited_positions": visited_list,
            "nav_target": nav_target,
            "nav_target_type": nav_target_type,
            "detections": det_list,
        }

        msg = String()
        msg.data = json.dumps(state_dict, ensure_ascii=False)
        self.ctrl_state_pub.publish(msg)

    def _publish_map_image(self) -> None:
        if not _HAS_CV2:
            return
        slam = self.slam
        if slam is None or not hasattr(slam, "grid"):
            return

        try:
            img = slam.grid.to_image()  # (H, W, 3) uint8
            img = cv2.flip(img, 1)
            H, W = img.shape[:2]

            rx, ry, rt = slam.odom.pose

            def w2d(wx, wy):
                gx, gy = slam.grid.w2g(wx, wy)
                return (W - 1 - gx, gy)

            def in_disp(dx, dy):
                return 0 <= dx < W and 0 <= dy < H

            # Robot trail
            self._trail.append((rx, ry))
            if len(self._trail) > self._trail_length:
                self._trail = self._trail[-self._trail_length:]

            fade_step = max(1, len(self._trail))
            for i, (tx, ty) in enumerate(self._trail):
                dx, dy = w2d(tx, ty)
                if in_disp(dx, dy):
                    alpha = int(60 + 140 * (i / fade_step))
                    cv2.circle(img, (dx, dy), 1, (0, 0, alpha), -1)

            # Frontiers
            frontier_targets = getattr(slam, "cached_frontiers", [])
            for fx, fy in frontier_targets:
                dx, dy = w2d(fx, fy)
                if in_disp(dx, dy):
                    cv2.circle(img, (dx, dy), 4, _CLR_FRONTIER, 1)

            # Planned path
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

            # Exclusion zones around visited objects
            for vpos in self.visited_positions:
                dx, dy = w2d(*vpos)
                r_px = int(1.5 / slam.grid.resolution)
                if in_disp(dx, dy):
                    cv2.circle(img, (dx, dy), r_px, (0, 180, 0), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (dx, dy), (0, 180, 0),
                                   cv2.MARKER_TILTED_CROSS, 8, 2)

            # Known objects
            for cls_id, (kx, ky) in self.known_objects.items():
                dx, dy = w2d(kx, ky)
                if in_disp(dx, dy):
                    color = _CLASS_COLORS.get(cls_id, (200, 200, 200))
                    cv2.drawMarker(img, (dx, dy), color,
                                   cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
                    name = _NAMES.get(cls_id, f"#{cls_id}")
                    cv2.putText(img, name, (dx + 7, dy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

            # Navigation target
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

            # Robot position + heading arrow
            rdx, rdy = w2d(rx, ry)
            if in_disp(rdx, rdy):
                cv2.circle(img, (rdx, rdy), 5, _CLR_ROBOT, -1)
                arrow_len = 12
                ax = int(rdx - arrow_len * math.cos(rt))
                ay = int(rdy + arrow_len * math.sin(rt))
                cv2.arrowedLine(img, (rdx, rdy), (ax, ay), _CLR_ROBOT, 2, tipLength=0.4)

            # Legend
            legend_y = 15
            items = [
                ("Robot", _CLR_ROBOT), ("Path", _CLR_PATH),
                ("Frontier", _CLR_FRONTIER), ("Target", _CLR_TARGET),
                ("Object", _CLR_EXPLICIT),
            ]
            if self.known_objects:
                items.append(("Known", (200, 200, 0)))
            if self.visited_positions:
                items.append(("Visited", (0, 180, 0)))
            for label, color in items:
                cv2.circle(img, (10, legend_y), 4, color, -1)
                cv2.putText(img, label, (18, legend_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
                legend_y += 16

            # Resize to display size
            disp_size = self._MAP_IMAGE_SIZE
            img = cv2.resize(img, (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)

            # Convert BGR to RGB for ROS Image message
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height = rgb_img.shape[0]
            msg.width = rgb_img.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = rgb_img.shape[1] * 3
            msg.data = rgb_img.tobytes()
            self.map_image_pub.publish(msg)

        except Exception as e:
            self.get_logger().debug(f"Map render failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.stop_robot()
        if node.dashboard is not None:
            node.dashboard.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
