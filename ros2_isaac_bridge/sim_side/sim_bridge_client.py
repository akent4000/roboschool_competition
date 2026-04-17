import socket
import json
import time
import threading
import cv2
import struct
import numpy as np


CMD_IP = "127.0.0.1"
CMD_PORT = 5005

STATE_IP = "127.0.0.1"
STATE_PORT = 5006

RGB_IP = "127.0.0.1"
RGB_PORT = 5007

DEPTH_IP = "127.0.0.1"
DEPTH_PORT = 5008

JOINT_STATE_IP = "127.0.0.1"
JOINT_STATE_PORT = 5009

IMU_IP = "127.0.0.1"
IMU_PORT = 5010

DETECTED_IP = "127.0.0.1"
DETECTED_PORT = 5011

OBJECT_SEQ_IP = "127.0.0.1"
OBJECT_SEQ_PORT = 5012


class SimBridgeClient:
    def __init__(self):
        self.latest_cmd = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind((CMD_IP, CMD_PORT))
        self.cmd_sock.setblocking(False)

        self.detected_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.detected_sock.bind((DETECTED_IP, DETECTED_PORT))
        self.detected_sock.setblocking(False)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
        self.depth_sock.connect((DEPTH_IP, DEPTH_PORT))  # TCP
        self.joint_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.object_seq_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Background sender thread for heavy payloads (RGB + depth)
        self._send_lock = threading.Lock()
        self._pending_rgb = None      # latest RGB frame (numpy array)
        self._pending_depth = None    # latest depth frame (numpy array)
        self._sender_event = threading.Event()
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

    # ---- Background sender ------------------------------------------------

    def _sender_loop(self):
        """Drain pending RGB/depth in a background thread."""
        while True:
            self._sender_event.wait()
            self._sender_event.clear()

            with self._send_lock:
                rgb = self._pending_rgb
                depth = self._pending_depth
                self._pending_rgb = None
                self._pending_depth = None

            if rgb is not None:
                self._do_send_rgb(rgb)
            if depth is not None:
                self._do_send_depth(depth)

    # ---- Receive -----------------------------------------------------------

    def receive_cmd(self):
        try:
            data, _ = self.cmd_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            self.latest_cmd["vx"] = float(msg.get("vx", 0.0))
            self.latest_cmd["vy"] = float(msg.get("vy", 0.0))
            self.latest_cmd["wz"] = float(msg.get("wz", 0.0))
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"receive_cmd error: {e}")

        return self.latest_cmd.copy()

    def receive_detected_object(self):
        try:
            data, _ = self.detected_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            return int(msg.get("object_id"))
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"receive_detected_object error: {e}")
        return None

    # ---- Lightweight sends (stay on main thread) --------------------------

    def send_state(self, vx, vy, wz):
        msg = {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.state_sock.sendto(data, (STATE_IP, STATE_PORT))

    def send_joint_states(self, names, position, velocity):
        msg = {
            "names": list(names),
            "position": [float(x) for x in position],
            "velocity": [float(x) for x in velocity],
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.joint_state_sock.sendto(data, (JOINT_STATE_IP, JOINT_STATE_PORT))

    def send_imu(self, ang_vel, lin_acc):
        msg = {
            "wx": float(ang_vel[0]),
            "wy": float(ang_vel[1]),
            "wz": float(ang_vel[2]),
            "ax": float(lin_acc[0]),
            "ay": float(lin_acc[1]),
            "az": float(lin_acc[2]),
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.imu_sock.sendto(data, (IMU_IP, IMU_PORT))

    def send_object_sequence(self, sequence):
        data = json.dumps(sequence).encode("utf-8")
        self.object_seq_sock.sendto(data, (OBJECT_SEQ_IP, OBJECT_SEQ_PORT))

    # ---- Heavy sends (queued for background thread) -----------------------

    def send_rgb(self, rgb):
        """Queue RGB frame for background sending (non-blocking)."""
        with self._send_lock:
            self._pending_rgb = rgb.copy()
        self._sender_event.set()

    def send_depth(self, depth):
        """Queue depth frame for background sending (non-blocking)."""
        with self._send_lock:
            self._pending_depth = np.asarray(depth, dtype=np.float32).copy()
        self._sender_event.set()

    # ---- Actual network I/O (called from background thread) ---------------

    def _do_send_rgb(self, rgb):
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", bgr)
            if not success:
                print("send_rgb error: JPEG encoding failed")
                return
            data = encoded.tobytes()
            self.rgb_sock.sendto(data, (RGB_IP, RGB_PORT))
        except Exception as e:
            print(f"send_rgb error: {e}")

    def _do_send_depth(self, depth):
        """Send depth as uint16 millimetres (halves bandwidth vs float32)."""
        try:
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            h, w = depth_mm.shape[:2]
            payload = struct.pack("II", h, w) + depth_mm.tobytes()
            packet = struct.pack("I", len(payload)) + payload
            self.depth_sock.sendall(packet)
        except Exception as e:
            print(f"send_depth error: {e}")
