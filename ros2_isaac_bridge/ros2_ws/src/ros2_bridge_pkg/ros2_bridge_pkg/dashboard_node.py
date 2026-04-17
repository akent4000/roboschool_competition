#!/usr/bin/env python3
"""
Web dashboard for the Aliengo competition ROS 2 stack.

Runs an HTTP server (default port 8080) that provides:
  /                  — single-page dashboard (HTML)
  /stream/rgb        — MJPEG stream of the RGB camera (with detection overlays)
  /stream/depth      — MJPEG stream of the depth camera (colorized)
  /stream/map        — MJPEG stream of the occupancy map
  /api/state         — JSON snapshot (velocity, IMU, queue, detections, rates, controller state)

No external dependencies beyond stdlib + cv2 + numpy (already in the Docker image).
"""
import json
import math
import struct
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, JointState, Imu
from std_msgs.msg import Int32, String

# ---------------------------------------------------------------------------
# Object class names (same mapping as controller.py)
# ---------------------------------------------------------------------------
_NAMES: Dict[int, str] = {
    0: "backpack",
    1: "bottle",
    2: "chair",
    3: "cup",
    4: "laptop",
}

# Detection overlay colors (BGR for OpenCV) — match visualizer.py
_CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 165, 255),   # orange — backpack
    1: (0, 255, 0),     # green — bottle
    2: (255, 0, 0),     # blue — chair
    3: (0, 255, 255),   # yellow — cup
    4: (255, 0, 255),   # magenta — laptop
}

# ---------------------------------------------------------------------------
# Thread-safe shared state between ROS callbacks and HTTP threads
# ---------------------------------------------------------------------------

_RATE_WINDOW_S = 3.0  # seconds kept for Hz calculation


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.monotonic()

        # Latest JPEG-encoded frames
        self.rgb_jpeg: Optional[bytes] = None
        self.depth_jpeg: Optional[bytes] = None
        self.map_jpeg: Optional[bytes] = None

        # Telemetry
        self.velocity = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.imu = {"wx": 0.0, "wy": 0.0, "wz": 0.0}
        self.joint_names: List[str] = []
        self.joint_positions: List[float] = []
        self.joint_velocities: List[float] = []

        # Mission
        self.object_queue: List[int] = []
        self.detected_objects: List[dict] = []  # [{id, name, t}]

        # Controller state (from /competition/controller_state JSON topic)
        self.controller: dict = {
            "sim_time": 0.0,
            "state": "idle",
            "odom": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
            "target_cls": None,
            "target_cls_name": "",
            "confirm_elapsed": 0.0,
            "confirm_needed": 3.0,
            "queue_idx": 0,
            "known_objects": [],
            "visited_positions": [],
            "nav_target": None,
            "nav_target_type": "",
            "detections": [],
            "motion_mode": "idle",
        }

        # Message timestamps for rate calculation
        self._stamps: Dict[str, deque] = {
            k: deque(maxlen=200)
            for k in ("rgb", "depth", "vel", "cmd", "imu", "joint", "seq", "map", "ctrl")
        }

    # --- writers (called from ROS callbacks) ---

    def set_rgb(self, jpeg: bytes):
        with self._lock:
            self.rgb_jpeg = jpeg
            self._stamps["rgb"].append(time.monotonic())

    def set_depth(self, jpeg: bytes):
        with self._lock:
            self.depth_jpeg = jpeg
            self._stamps["depth"].append(time.monotonic())

    def set_map(self, jpeg: bytes):
        with self._lock:
            self.map_jpeg = jpeg
            self._stamps["map"].append(time.monotonic())

    def set_velocity(self, vx: float, vy: float, wz: float):
        with self._lock:
            self.velocity = {"vx": vx, "vy": vy, "wz": wz}
            self._stamps["vel"].append(time.monotonic())

    def set_cmd_vel(self, vx: float, vy: float, wz: float):
        with self._lock:
            self.cmd_vel = {"vx": vx, "vy": vy, "wz": wz}
            self._stamps["cmd"].append(time.monotonic())

    def set_imu(self, wx: float, wy: float, wz: float):
        with self._lock:
            self.imu = {"wx": wx, "wy": wy, "wz": wz}
            self._stamps["imu"].append(time.monotonic())

    def set_joints(self, names: list, pos: list, vel: list):
        with self._lock:
            self.joint_names = names
            self.joint_positions = pos
            self.joint_velocities = vel
            self._stamps["joint"].append(time.monotonic())

    def set_object_queue(self, queue: list):
        with self._lock:
            self.object_queue = queue
            self._stamps["seq"].append(time.monotonic())

    def set_controller_state(self, data: dict):
        with self._lock:
            self.controller.update(data)
            self._stamps["ctrl"].append(time.monotonic())

    def add_detection(self, obj_id: int):
        with self._lock:
            self.detected_objects.append({
                "id": obj_id,
                "name": _NAMES.get(obj_id, f"unknown-{obj_id}"),
                "t": round(time.monotonic() - self.start_time, 2),
            })

    # --- readers (called from HTTP threads) ---

    def get_rgb(self) -> Optional[bytes]:
        with self._lock:
            return self.rgb_jpeg

    def get_depth(self) -> Optional[bytes]:
        with self._lock:
            return self.depth_jpeg

    def get_map(self) -> Optional[bytes]:
        with self._lock:
            return self.map_jpeg

    def get_detections_for_overlay(self) -> Tuple[list, Optional[int]]:
        """Return (detections_list, target_cls) for drawing overlays on RGB."""
        with self._lock:
            return (
                list(self.controller.get("detections", [])),
                self.controller.get("target_cls"),
            )

    def _hz(self, key: str) -> float:
        stamps = self._stamps.get(key, deque())
        if len(stamps) < 2:
            return 0.0
        now = time.monotonic()
        cutoff = now - _RATE_WINDOW_S
        recent = [s for s in stamps if s > cutoff]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.monotonic() - self.start_time, 1),
                "velocity": self.velocity.copy(),
                "cmd_vel": self.cmd_vel.copy(),
                "imu": self.imu.copy(),
                "joints": {
                    "names": list(self.joint_names),
                    "positions": [round(p, 4) for p in self.joint_positions],
                    "velocities": [round(v, 4) for v in self.joint_velocities],
                },
                "object_queue": list(self.object_queue),
                "detected_objects": list(self.detected_objects),
                "controller": dict(self.controller),
                "rates_hz": {
                    "rgb": round(self._hz("rgb"), 1),
                    "depth": round(self._hz("depth"), 1),
                    "velocity": round(self._hz("vel"), 1),
                    "cmd_vel": round(self._hz("cmd"), 1),
                    "imu": round(self._hz("imu"), 1),
                    "joint_state": round(self._hz("joint"), 1),
                    "map": round(self._hz("map"), 1),
                    "controller": round(self._hz("ctrl"), 1),
                },
            }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

# Shared state — set once before server starts
_state: Optional[DashboardState] = None

_MJPEG_BOUNDARY = b"--frameboundary"


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard page, MJPEG streams, and JSON API."""

    # Suppress per-request log lines
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream/rgb":
            self._serve_mjpeg(lambda: _state.get_rgb())
        elif self.path == "/stream/depth":
            self._serve_mjpeg(lambda: _state.get_depth())
        elif self.path == "/stream/map":
            self._serve_mjpeg(lambda: _state.get_map())
        elif self.path == "/api/state":
            self._serve_json(_state.snapshot())
        else:
            self.send_error(404)

    def _serve_html(self):
        body = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self, getter):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frameboundary",
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        try:
            while True:
                frame = getter()
                if frame is None:
                    # 1x1 black pixel placeholder
                    frame = _BLACK_JPEG
                self.wfile.write(_MJPEG_BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)  # ~20 fps cap to limit bandwidth
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class _ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# Tiny 1x1 black JPEG used as placeholder before first real frame arrives
_BLACK_JPEG = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard")

        global _state
        _state = DashboardState()

        # --- parameters ---
        self.declare_parameter("port", 8080)
        port = self.get_parameter("port").value

        # --- subscriptions ---
        self.create_subscription(
            TwistStamped, "/aliengo/base_velocity", self._vel_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist, "/cmd_vel", self._cmd_cb, 10,
        )
        self.create_subscription(
            Image, "/aliengo/camera/color/image_raw", self._rgb_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, "/aliengo/camera/depth/image_raw", self._depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState, "/aliengo/joint_states", self._joint_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu, "/aliengo/imu", self._imu_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/competition/object_sequence", self._seq_cb, 10,
        )
        self.create_subscription(
            Int32, "/competition/detected_object", self._det_cb, 10,
        )
        # --- new: controller state (JSON) and SLAM map image ---
        self.create_subscription(
            String, "/competition/controller_state", self._ctrl_state_cb, 10,
        )
        self.create_subscription(
            Image, "/slam/map_image", self._map_cb,
            qos_profile_sensor_data,
        )

        # --- start HTTP server in background thread ---
        self._server = _ThreadedServer(("0.0.0.0", port), _Handler)
        self._http_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._http_thread.start()

        self.get_logger().info(f"Dashboard HTTP server started on http://0.0.0.0:{port}")

    # ---- ROS callbacks ----

    def _vel_cb(self, msg: TwistStamped):
        _state.set_velocity(
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z,
        )

    def _cmd_cb(self, msg: Twist):
        _state.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def _rgb_cb(self, msg: Image):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            # ROS Image is rgb8, OpenCV wants bgr for encode
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw detection overlays (from controller_state topic)
            detections, target_cls = _state.get_detections_for_overlay()
            for det in detections:
                cls_id = det.get("cls", 0)
                uc = det.get("u", 0)
                vc = det.get("v", 0)
                conf = det.get("conf", 0)
                bw = det.get("w", 80)
                bh = det.get("h", 60)

                is_target = (target_cls is not None and cls_id == target_cls)
                color = (0, 255, 0) if is_target else _CLASS_COLORS_BGR.get(cls_id, (200, 200, 200))
                thickness = 3 if is_target else 2

                x1 = int(uc - bw / 2)
                y1 = int(vc - bh / 2)
                x2 = int(uc + bw / 2)
                y2 = int(vc + bh / 2)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)

                name = _NAMES.get(cls_id, f"cls{cls_id}")
                label = f"{name} {conf:.2f}"
                if is_target:
                    label = ">> " + label + " <<"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                _state.set_rgb(buf.tobytes())
        except Exception:
            pass

    def _depth_cb(self, msg: Image):
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)
            )
            # Normalize to 0-255 and apply colormap
            d_clip = np.clip(depth, 0.0, 4.0)
            d_norm = (d_clip / 4.0 * 255).astype(np.uint8)
            colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            # Mark invalid pixels black
            colored[depth <= 0] = 0
            ok, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                _state.set_depth(buf.tobytes())
        except Exception:
            pass

    def _map_cb(self, msg: Image):
        """Receive pre-rendered SLAM map image (with all overlays)."""
        try:
            if msg.encoding in ("bgr8",):
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3)
                )
            elif msg.encoding in ("mono8",):
                mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width)
                )
                bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            else:
                # Assume rgb8 or 8UC3
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3)
                )
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                _state.set_map(buf.tobytes())
        except Exception:
            pass

    def _joint_cb(self, msg: JointState):
        _state.set_joints(
            list(msg.name), list(msg.position), list(msg.velocity),
        )

    def _imu_cb(self, msg: Imu):
        _state.set_imu(
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        )

    def _seq_cb(self, msg: String):
        try:
            raw = json.loads(msg.data)
            queue = [
                item[0] if isinstance(item, (tuple, list)) else int(item)
                for item in raw
            ]
            _state.set_object_queue(queue)
        except Exception:
            pass

    def _det_cb(self, msg: Int32):
        _state.add_detection(int(msg.data))

    def _ctrl_state_cb(self, msg: String):
        """Parse JSON controller state blob."""
        try:
            data = json.loads(msg.data)
            _state.set_controller_state(data)
        except Exception:
            pass

    def destroy_node(self):
        self._server.shutdown()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Embedded HTML dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aliengo Dashboard</title>
<style>
:root {
  --bg: #0b0d14;
  --card: #141720;
  --card-alt: #181c28;
  --border: #252a3a;
  --text: #e0e2ec;
  --muted: #7a7f98;
  --accent: #4e9af5;
  --green: #4caf84;
  --orange: #e8a838;
  --red: #e05050;
  --cyan: #40d8e0;
  --magenta: #d060e0;
  --yellow: #e0d040;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace;
  font-size: 13px;
  line-height: 1.5;
  overflow-x: hidden;
}

/* ---- Header ---- */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--card);
}
.header-left { display: flex; align-items: center; gap: 12px; }
.header h1 { font-size: 15px; font-weight: 700; letter-spacing: 0.02em; }
.header .conn-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--red);
  flex-shrink: 0;
}
.header .conn-dot.ok { background: var(--green); }
.header-right { display: flex; align-items: center; gap: 20px; color: var(--muted); font-size: 12px; }

/* ---- Status strip ---- */
.status-strip {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 20px;
  background: var(--card-alt);
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  flex-wrap: wrap;
}
.status-strip .sep { color: var(--border); }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.badge.idle      { background: #333; color: var(--muted); }
.badge.warmup    { background: #553d00; color: var(--yellow); }
.badge.explore   { background: #0d3b60; color: var(--accent); }
.badge.approach  { background: #5a3800; color: var(--orange); }
.badge.confirm   { background: #1a4a2e; color: var(--green); animation: pulse-bg 1.2s infinite; }
.badge.backup    { background: #4a1a1a; color: var(--red); }
.badge.done      { background: #1a4a2e; color: var(--green); }
@keyframes pulse-bg {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.confirm-bar {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.confirm-bar .track {
  width: 80px;
  height: 6px;
  background: #333;
  border-radius: 3px;
  overflow: hidden;
}
.confirm-bar .fill {
  height: 100%;
  background: var(--green);
  border-radius: 3px;
  transition: width 0.2s;
}

/* ---- Main grid ---- */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto;
  gap: 12px;
  padding: 12px;
  max-width: 1700px;
  margin: 0 auto;
}

/* Cameras stacked on left, map on right spanning both rows */
.cameras-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.card-title {
  padding: 8px 14px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.card-title .tag {
  font-size: 10px;
  font-weight: 500;
  color: var(--accent);
}

/* ---- Camera streams ---- */
.stream-container { position: relative; background: #000; }
.stream-container img {
  display: block;
  width: 100%;
  height: auto;
  image-rendering: auto;
}
.stream-label {
  position: absolute;
  top: 6px; left: 8px;
  background: rgba(0,0,0,0.7);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  color: var(--muted);
}
/* Map placeholder when no data */
.map-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  color: var(--muted);
  font-size: 14px;
  font-style: italic;
  background: #111;
}

/* ---- Bottom grid ---- */
.bottom-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  padding: 0 12px 12px;
  max-width: 1700px;
  margin: 0 auto;
}

/* ---- Telemetry ---- */
.telem-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  padding: 0 12px;
  max-width: 1700px;
  margin: 0 auto 12px;
}
.telem-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: var(--border);
}
.telem-cell {
  background: var(--card);
  padding: 10px 14px;
  text-align: center;
}
.telem-cell .label { color: var(--muted); font-size: 10px; margin-bottom: 2px; text-transform: uppercase; }
.telem-cell .value { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.telem-cell .unit { color: var(--muted); font-size: 10px; margin-left: 2px; }

/* ---- Rates ---- */
.rates {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
}
.rate-cell {
  background: var(--card);
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.rate-cell .name { color: var(--muted); font-size: 11px; }
.rate-cell .hz { font-weight: 600; font-variant-numeric: tabular-nums; font-size: 12px; }
.rate-cell .hz.dead { color: var(--red); }
.rate-cell .hz.slow { color: var(--orange); }
.rate-cell .hz.ok { color: var(--green); }

/* ---- Mission panel ---- */
.mission { padding: 14px; }
.section-label {
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.queue-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.queue-item {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid var(--border);
  color: var(--muted);
  background: var(--bg);
  transition: all 0.3s;
}
.queue-item.visited {
  background: var(--green);
  color: #fff;
  border-color: var(--green);
}
.queue-item.current {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
  animation: pulse 1.5s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.target-info {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: 6px;
  border: 1px solid var(--border);
}
.target-info .target-name {
  font-weight: 700;
  font-size: 14px;
  color: var(--accent);
}
.target-info .target-dist {
  color: var(--muted);
  font-size: 12px;
}

.confirm-progress {
  margin-bottom: 14px;
}
.confirm-progress .bar-bg {
  width: 100%;
  height: 10px;
  background: #222;
  border-radius: 5px;
  overflow: hidden;
  margin-top: 4px;
}
.confirm-progress .bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--green), var(--cyan));
  border-radius: 5px;
  transition: width 0.15s;
}
.confirm-progress .bar-label {
  font-size: 11px;
  color: var(--muted);
  margin-top: 3px;
  text-align: right;
}

.det-log {
  max-height: 160px;
  overflow-y: auto;
  font-size: 12px;
}
.det-log .entry {
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.det-log .entry .t { color: var(--muted); font-size: 11px; }
.det-log .entry .name { color: var(--green); font-weight: 600; }
.det-log .empty { color: var(--muted); font-style: italic; padding: 8px 0; }

/* ---- Known objects ---- */
.known-objects { padding: 14px; }
.known-obj-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}
.known-obj-item:last-child { border-bottom: none; }
.known-obj-item .obj-name {
  font-weight: 600;
}
.known-obj-item .obj-pos { color: var(--muted); font-variant-numeric: tabular-nums; }
.known-obj-item .obj-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 8px;
  flex-shrink: 0;
  display: inline-block;
}

/* ---- Joints ---- */
.joints-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.joints-table th {
  text-align: left;
  padding: 6px 10px;
  color: var(--muted);
  font-weight: 500;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
}
.joints-table td {
  padding: 4px 10px;
  border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
}
.joints-table tr:last-child td { border-bottom: none; }
.joints-scroll { max-height: 260px; overflow-y: auto; }

/* full-width */
.span2 { grid-column: span 2; }

/* responsive */
@media (max-width: 1000px) {
  .main-grid { grid-template-columns: 1fr; }
  .bottom-grid { grid-template-columns: 1fr; }
  .telem-row { grid-template-columns: 1fr; }
  .span2 { grid-column: span 1; }
  .rates { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<!-- ======== HEADER ======== -->
<div class="header">
  <div class="header-left">
    <span id="conn-dot" class="conn-dot"></span>
    <h1>Aliengo Dashboard</h1>
  </div>
  <div class="header-right">
    <span id="uptime">--</span>
  </div>
</div>

<!-- ======== STATUS STRIP ======== -->
<div class="status-strip">
  <span id="state-badge" class="badge idle">IDLE</span>
  <span class="sep">|</span>
  <span>Sim: <b id="sim-time">0.0</b>s</span>
  <span class="sep">|</span>
  <span>Pos: (<b id="odom-x">0.00</b>, <b id="odom-y">0.00</b>)</span>
  <span>Yaw: <b id="odom-yaw">0</b>&deg;</span>
  <span class="sep">|</span>
  <span class="confirm-bar" id="confirm-bar" style="display:none">
    Confirm:
    <span class="track"><span class="fill" id="confirm-fill" style="width:0%"></span></span>
    <span id="confirm-text">0/3s</span>
  </span>
  <span id="queue-progress" style="color:var(--muted)">Queue: --</span>
</div>

<!-- ======== CAMERAS + MAP ======== -->
<div class="main-grid">
  <!-- Left: RGB + Depth stacked -->
  <div class="cameras-col">
    <div class="card">
      <div class="card-title">
        <span>RGB Camera</span>
        <span class="tag"><span id="det-count">0</span> detections</span>
      </div>
      <div class="stream-container">
        <img id="rgb" src="/stream/rgb" alt="RGB">
        <span class="stream-label">640x360 &middot; <span id="rgb-hz">0</span> Hz</span>
      </div>
    </div>
    <div class="card">
      <div class="card-title">
        <span>Depth Camera</span>
        <span class="tag">max 4.0m</span>
      </div>
      <div class="stream-container">
        <img id="depth" src="/stream/depth" alt="Depth">
        <span class="stream-label">848x480 &middot; <span id="depth-hz">0</span> Hz</span>
      </div>
    </div>
  </div>

  <!-- Right: Occupancy Map -->
  <div class="card" id="map-card">
    <div class="card-title">
      <span>Occupancy Map</span>
      <span class="tag"><span id="map-hz">0</span> Hz</span>
    </div>
    <div id="map-container">
      <div class="stream-container" id="map-stream" style="display:none">
        <img id="map-img" src="/stream/map" alt="Map">
        <span class="stream-label">SLAM &middot; <span id="map-hz2">0</span> Hz</span>
      </div>
      <div class="map-placeholder" id="map-placeholder">
        Waiting for SLAM map data...<br>
        <small>Subscribe to /slam/map_image</small>
      </div>
    </div>
  </div>
</div>

<!-- ======== TELEMETRY ======== -->
<div class="telem-row">
  <div class="card">
    <div class="card-title">Commanded Velocity</div>
    <div class="telem-grid">
      <div class="telem-cell"><div class="label">vx</div><div class="value" id="cmd-vx">0.00</div><div class="unit">m/s</div></div>
      <div class="telem-cell"><div class="label">vy</div><div class="value" id="cmd-vy">0.00</div><div class="unit">m/s</div></div>
      <div class="telem-cell"><div class="label">wz</div><div class="value" id="cmd-wz">0.00</div><div class="unit">rad/s</div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Measured Velocity</div>
    <div class="telem-grid">
      <div class="telem-cell"><div class="label">vx</div><div class="value" id="vel-vx">0.00</div><div class="unit">m/s</div></div>
      <div class="telem-cell"><div class="label">vy</div><div class="value" id="vel-vy">0.00</div><div class="unit">m/s</div></div>
      <div class="telem-cell"><div class="label">wz</div><div class="value" id="vel-wz">0.00</div><div class="unit">rad/s</div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">IMU Angular Velocity</div>
    <div class="telem-grid">
      <div class="telem-cell"><div class="label">wx</div><div class="value" id="imu-wx">0.00</div><div class="unit">rad/s</div></div>
      <div class="telem-cell"><div class="label">wy</div><div class="value" id="imu-wy">0.00</div><div class="unit">rad/s</div></div>
      <div class="telem-cell"><div class="label">wz</div><div class="value" id="imu-wz">0.00</div><div class="unit">rad/s</div></div>
    </div>
  </div>
</div>

<!-- ======== MISSION + KNOWN OBJECTS + JOINTS + RATES ======== -->
<div class="bottom-grid">
  <!-- Mission -->
  <div class="card">
    <div class="card-title">Mission</div>
    <div class="mission">
      <div class="section-label">Object Queue</div>
      <div class="queue-row" id="queue"></div>

      <div id="target-section" style="display:none">
        <div class="section-label">Current Target</div>
        <div class="target-info">
          <span class="target-name" id="target-name">--</span>
          <span class="target-dist" id="target-dist"></span>
        </div>
      </div>

      <div id="confirm-section" style="display:none">
        <div class="section-label">Confirmation Progress</div>
        <div class="confirm-progress">
          <div class="bar-bg"><div class="bar-fill" id="confirm-bar-fill" style="width:0%"></div></div>
          <div class="bar-label" id="confirm-bar-label">0.0 / 3.0s</div>
        </div>
      </div>

      <div class="section-label">Detection Log</div>
      <div class="det-log" id="det-log"><div class="empty">No detections yet</div></div>
    </div>
  </div>

  <!-- Known Objects + Visited -->
  <div class="card">
    <div class="card-title">
      <span>Known Objects</span>
      <span class="tag" id="known-count">0 found</span>
    </div>
    <div class="known-objects" id="known-objects">
      <div class="empty" style="color:var(--muted);font-style:italic;padding:8px 0">No objects discovered yet</div>
    </div>
  </div>

  <!-- Joints -->
  <div class="card">
    <div class="card-title">Joint States</div>
    <div class="joints-scroll">
      <table class="joints-table">
        <thead><tr><th>Joint</th><th>Position (rad)</th><th>Velocity (rad/s)</th></tr></thead>
        <tbody id="joints"></tbody>
      </table>
    </div>
  </div>

  <!-- Topic Rates -->
  <div class="card">
    <div class="card-title">Topic Rates</div>
    <div class="rates" id="rates"></div>
  </div>
</div>

<script>
const NAMES = {0:"backpack",1:"bottle",2:"chair",3:"cup",4:"laptop"};
const CLS_COLORS = {0:"#ffa500",1:"#00ff00",2:"#4488ff",3:"#ffff00",4:"#ff00ff"};

function fmt(v) { return Number(v).toFixed(2); }

function hzClass(v) {
  if (v < 0.5) return "dead";
  if (v < 5) return "slow";
  return "ok";
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ":" + String(sec).padStart(2, "0");
}

function motionBadge(mode) {
  const map = {
    "idle":           { cls: "idle",     text: "IDLE" },
    "warmup_stop":    { cls: "warmup",   text: "WARMUP" },
    "navigate":       { cls: "explore",  text: "NAVIGATE" },
    "confirm_wait":   { cls: "confirm",  text: "CONFIRMING" },
    "backup":         { cls: "backup",   text: "BACKUP" },
    "all_done_stop":  { cls: "done",     text: "COMPLETE" },
  };
  return map[mode] || { cls: "idle", text: mode ? mode.toUpperCase() : "IDLE" };
}

let prevDetCount = 0;
let mapShown = false;

async function poll() {
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    const c = d.controller || {};

    // Connection indicator
    document.getElementById("conn-dot").className = "conn-dot ok";

    // Uptime
    document.getElementById("uptime").textContent = "uptime " + fmtTime(d.uptime_s);

    // --- Status strip ---
    // State badge
    const badge = motionBadge(c.motion_mode || "idle");
    const badgeEl = document.getElementById("state-badge");
    badgeEl.className = "badge " + badge.cls;
    badgeEl.textContent = badge.text;

    // Sim time
    document.getElementById("sim-time").textContent = (c.sim_time || 0).toFixed(1);

    // Odometry
    const odom = c.odom || {};
    document.getElementById("odom-x").textContent = (odom.x || 0).toFixed(2);
    document.getElementById("odom-y").textContent = (odom.y || 0).toFixed(2);
    document.getElementById("odom-yaw").textContent = Math.round(odom.yaw_deg || 0);

    // Confirm bar in status strip
    const confirmElapsed = c.confirm_elapsed || 0;
    const confirmNeeded = c.confirm_needed || 3;
    const confirmBar = document.getElementById("confirm-bar");
    if (confirmElapsed > 0) {
      confirmBar.style.display = "";
      const pct = Math.min(100, (confirmElapsed / confirmNeeded) * 100);
      document.getElementById("confirm-fill").style.width = pct + "%";
      document.getElementById("confirm-text").textContent =
        confirmElapsed.toFixed(1) + "/" + confirmNeeded.toFixed(1) + "s";
    } else {
      confirmBar.style.display = "none";
    }

    // Queue progress
    const qIdx = c.queue_idx || 0;
    const qTotal = d.object_queue.length;
    document.getElementById("queue-progress").textContent =
      qTotal > 0 ? "Queue: " + Math.min(qIdx, qTotal) + "/" + qTotal : "Queue: --";

    // Detection count on RGB card
    const dets = c.detections || [];
    document.getElementById("det-count").textContent = dets.length;

    // --- Velocity ---
    document.getElementById("vel-vx").textContent = fmt(d.velocity.vx);
    document.getElementById("vel-vy").textContent = fmt(d.velocity.vy);
    document.getElementById("vel-wz").textContent = fmt(d.velocity.wz);

    document.getElementById("cmd-vx").textContent = fmt(d.cmd_vel.vx);
    document.getElementById("cmd-vy").textContent = fmt(d.cmd_vel.vy);
    document.getElementById("cmd-wz").textContent = fmt(d.cmd_vel.wz);

    // --- IMU ---
    document.getElementById("imu-wx").textContent = fmt(d.imu.wx);
    document.getElementById("imu-wy").textContent = fmt(d.imu.wy);
    document.getElementById("imu-wz").textContent = fmt(d.imu.wz);

    // --- Topic rates ---
    const ratesEl = document.getElementById("rates");
    const rateLabels = {
      rgb: "RGB", depth: "Depth", velocity: "Velocity", cmd_vel: "Cmd Vel",
      imu: "IMU", joint_state: "Joints", map: "Map", controller: "Controller"
    };
    ratesEl.innerHTML = Object.entries(d.rates_hz).map(([k,v]) =>
      `<div class="rate-cell">
        <span class="name">${rateLabels[k] || k}</span>
        <span class="hz ${hzClass(v)}">${v.toFixed(1)} Hz</span>
      </div>`
    ).join("");

    // Stream Hz overlays
    document.getElementById("rgb-hz").textContent = d.rates_hz.rgb.toFixed(0);
    document.getElementById("depth-hz").textContent = d.rates_hz.depth.toFixed(0);
    document.getElementById("map-hz").textContent = (d.rates_hz.map || 0).toFixed(0);
    document.getElementById("map-hz2").textContent = (d.rates_hz.map || 0).toFixed(0);

    // Show/hide map stream vs placeholder
    if (d.rates_hz.map > 0 && !mapShown) {
      document.getElementById("map-stream").style.display = "";
      document.getElementById("map-placeholder").style.display = "none";
      mapShown = true;
    } else if (d.rates_hz.map <= 0 && mapShown) {
      document.getElementById("map-stream").style.display = "none";
      document.getElementById("map-placeholder").style.display = "";
      mapShown = false;
    }

    // --- Mission queue ---
    const qEl = document.getElementById("queue");
    const detIds = new Set(d.detected_objects.map(o => o.id));
    if (d.object_queue.length > 0) {
      let curIdx = c.queue_idx || d.detected_objects.length;
      qEl.innerHTML = d.object_queue.map((id, i) => {
        let cls = "queue-item";
        if (i < curIdx) cls += " visited";
        else if (i === curIdx) cls += " current";
        return `<div class="${cls}">${NAMES[id] || id}</div>`;
      }).join("");
    } else {
      qEl.innerHTML = '<div style="color:var(--muted);font-style:italic">Waiting for sequence...</div>';
    }

    // --- Current target ---
    const targetSec = document.getElementById("target-section");
    if (c.target_cls != null && qTotal > 0) {
      targetSec.style.display = "";
      const tName = c.target_cls_name || NAMES[c.target_cls] || "?";
      document.getElementById("target-name").textContent = tName;
      const navTarget = c.nav_target;
      if (navTarget && odom.x != null) {
        const dx = (navTarget.x || navTarget[0] || 0) - (odom.x || 0);
        const dy = (navTarget.y || navTarget[1] || 0) - (odom.y || 0);
        const dist = Math.sqrt(dx*dx + dy*dy);
        document.getElementById("target-dist").textContent = dist.toFixed(2) + "m away";
      } else {
        document.getElementById("target-dist").textContent = "";
      }
    } else {
      targetSec.style.display = "none";
    }

    // --- Confirm progress (big bar) ---
    const confirmSec = document.getElementById("confirm-section");
    if (confirmElapsed > 0) {
      confirmSec.style.display = "";
      const pct = Math.min(100, (confirmElapsed / confirmNeeded) * 100);
      document.getElementById("confirm-bar-fill").style.width = pct + "%";
      document.getElementById("confirm-bar-label").textContent =
        confirmElapsed.toFixed(1) + " / " + confirmNeeded.toFixed(1) + "s";
    } else {
      confirmSec.style.display = "none";
    }

    // --- Detection log ---
    if (d.detected_objects.length !== prevDetCount) {
      prevDetCount = d.detected_objects.length;
      const logEl = document.getElementById("det-log");
      if (d.detected_objects.length === 0) {
        logEl.innerHTML = '<div class="empty">No detections yet</div>';
      } else {
        logEl.innerHTML = d.detected_objects.map(o =>
          `<div class="entry"><span class="name">${o.name}</span><span class="t">${fmtTime(o.t)}</span></div>`
        ).join("");
        logEl.scrollTop = logEl.scrollHeight;
      }
    }

    // --- Known objects ---
    const knownObjs = c.known_objects || [];
    const visitedPos = c.visited_positions || [];
    const knownEl = document.getElementById("known-objects");
    document.getElementById("known-count").textContent = knownObjs.length + " found";
    if (knownObjs.length > 0) {
      knownEl.innerHTML = knownObjs.map(obj => {
        const clsId = obj.id != null ? obj.id : obj.cls;
        const name = obj.name || NAMES[clsId] || "unknown";
        const color = CLS_COLORS[clsId] || "#888";
        const x = (obj.x || 0).toFixed(2);
        const y = (obj.y || 0).toFixed(2);
        return `<div class="known-obj-item">
          <span><span class="obj-dot" style="background:${color}"></span><span class="obj-name" style="color:${color}">${name}</span></span>
          <span class="obj-pos">(${x}, ${y})</span>
        </div>`;
      }).join("");
      // Add visited positions
      if (visitedPos.length > 0) {
        knownEl.innerHTML += '<div style="margin-top:12px"><div class="section-label" style="margin-bottom:6px">Visited Positions</div></div>';
        knownEl.innerHTML += visitedPos.map((vp, i) => {
          const x = (vp.x || vp[0] || 0).toFixed(2);
          const y = (vp.y || vp[1] || 0).toFixed(2);
          return `<div class="known-obj-item">
            <span><span class="obj-dot" style="background:var(--green)"></span><span class="obj-name" style="color:var(--green)">Visit #${i+1}</span></span>
            <span class="obj-pos">(${x}, ${y})</span>
          </div>`;
        }).join("");
      }
    } else if (visitedPos.length > 0) {
      knownEl.innerHTML = '<div class="section-label" style="margin-bottom:6px">Visited Positions</div>';
      knownEl.innerHTML += visitedPos.map((vp, i) => {
        const x = (vp.x || vp[0] || 0).toFixed(2);
        const y = (vp.y || vp[1] || 0).toFixed(2);
        return `<div class="known-obj-item">
          <span><span class="obj-dot" style="background:var(--green)"></span><span class="obj-name" style="color:var(--green)">Visit #${i+1}</span></span>
          <span class="obj-pos">(${x}, ${y})</span>
        </div>`;
      }).join("");
    } else {
      knownEl.innerHTML = '<div class="empty" style="color:var(--muted);font-style:italic;padding:8px 0">No objects discovered yet</div>';
    }

    // --- Joints ---
    const jEl = document.getElementById("joints");
    if (d.joints.names.length > 0) {
      jEl.innerHTML = d.joints.names.map((name, i) =>
        `<tr><td>${name}</td><td>${d.joints.positions[i]?.toFixed(3) ?? "--"}</td><td>${d.joints.velocities[i]?.toFixed(3) ?? "--"}</td></tr>`
      ).join("");
    }
  } catch (e) {
    document.getElementById("conn-dot").className = "conn-dot";
  }
}

setInterval(poll, 250);
poll();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
