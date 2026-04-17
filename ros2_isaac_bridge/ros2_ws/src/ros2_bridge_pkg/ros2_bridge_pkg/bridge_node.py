import socket
import json
import struct

import cv2
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, JointState, Imu
from std_msgs.msg import Int32, String


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


class BridgeNode(Node):
    def __init__(self):
        super().__init__("bridge_node")

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind((STATE_IP, STATE_PORT))
        self.state_sock.setblocking(False)

        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock.bind((RGB_IP, RGB_PORT))
        self.rgb_sock.setblocking(False)

        # TCP for depth
        self.depth_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.depth_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.depth_server_sock.bind((DEPTH_IP, DEPTH_PORT))
        self.depth_server_sock.listen(1)
        self.depth_server_sock.setblocking(False)

        self.depth_conn = None
        self._depth_rx_buffer = bytearray()

        self.joint_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.joint_state_sock.bind((JOINT_STATE_IP, JOINT_STATE_PORT))
        self.joint_state_sock.setblocking(False)

        self.imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu_sock.bind((IMU_IP, IMU_PORT))
        self.imu_sock.setblocking(False)

        # detected_object: controller → bridge → isaac_controller (UDP send)
        self.detected_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # object_sequence: isaac_controller → bridge → controller (UDP receive)
        self.object_seq_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.object_seq_sock.bind((OBJECT_SEQ_IP, OBJECT_SEQ_PORT))
        self.object_seq_sock.setblocking(False)

        # --- Subscriptions ---
        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            1,
        )

        self.detected_sub = self.create_subscription(
            Int32,
            "/competition/detected_object",
            self.detected_callback,
            10,
        )

        # --- Publishers ---
        self.vel_pub = self.create_publisher(
            TwistStamped,
            "/aliengo/base_velocity",
            1,
        )

        self.rgb_pub = self.create_publisher(
            Image,
            "/aliengo/camera/color/image_raw",
            1,
        )

        self.depth_pub = self.create_publisher(
            Image,
            "/aliengo/camera/depth/image_raw",
            1,
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            "/aliengo/joint_states",
            1,
        )

        self.imu_pub = self.create_publisher(
            Imu,
            "/aliengo/imu",
            1,
        )

        self.object_seq_pub = self.create_publisher(
            String,
            "/competition/object_sequence",
            10,
        )

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info("ROS bridge node started.")

    def cmd_callback(self, msg: Twist):
        payload = {
            "vx": msg.linear.x,
            "vy": msg.linear.y,
            "wz": msg.angular.z,
        }

        data = json.dumps(payload).encode("utf-8")
        self.cmd_sock.sendto(data, (CMD_IP, CMD_PORT))

    def detected_callback(self, msg: Int32):
        payload = {
            "object_id": int(msg.data),
        }

        data = json.dumps(payload).encode("utf-8")
        self.detected_sock.sendto(data, (DETECTED_IP, DETECTED_PORT))

    def recv_exact(self, sock, size):
        chunks = []
        received = 0
        while received < size:
            chunk = sock.recv(size - received)
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_latest_udp(self, sock, max_size):
        latest = None
        while True:
            try:
                data, _ = sock.recvfrom(max_size)
                latest = data
            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().error(f"udp receive error: {e}")
                break
        return latest

    def _read_latest_depth_payload(self):
        if self.depth_conn is None:
            return None

        try:
            while True:
                chunk = self.depth_conn.recv(65536)
                if not chunk:
                    raise ConnectionError("depth socket closed")
                self._depth_rx_buffer.extend(chunk)
                if len(chunk) < 65536:
                    break
        except BlockingIOError:
            pass

        latest_payload = None
        while len(self._depth_rx_buffer) >= 4:
            payload_size = struct.unpack("I", self._depth_rx_buffer[:4])[0]
            if payload_size <= 0 or payload_size > 20 * 1024 * 1024:
                raise ValueError(f"invalid depth payload_size={payload_size}")

            full_size = 4 + payload_size
            if len(self._depth_rx_buffer) < full_size:
                break

            latest_payload = bytes(self._depth_rx_buffer[4:full_size])
            del self._depth_rx_buffer[:full_size]

        return latest_payload

    def timer_callback(self):
        # --- base velocity ---
        try:
            data = self._recv_latest_udp(self.state_sock, 4096)
            if data is not None:
                msg = json.loads(data.decode("utf-8"))

                out = TwistStamped()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = "base"

                out.twist.linear.x = float(msg.get("vx", 0.0))
                out.twist.linear.y = float(msg.get("vy", 0.0))
                out.twist.angular.z = float(msg.get("wz", 0.0))

                self.vel_pub.publish(out)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"state receive error: {e}")

        # --- RGB ---
        try:
            data = self._recv_latest_udp(self.rgb_sock, 65535)
            if data is None:
                pass
            else:
                np_arr = np.frombuffer(data, dtype=np.uint8)
                image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image_bgr is not None:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                    msg = Image()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "front_camera"
                    msg.height = image_rgb.shape[0]
                    msg.width = image_rgb.shape[1]
                    msg.encoding = "rgb8"
                    msg.is_bigendian = 0
                    msg.step = image_rgb.shape[1] * 3
                    msg.data = image_rgb.tobytes()

                    self.rgb_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"rgb receive error: {e}")

        # --- Depth (TCP, uint16 millimetres → float32 metres) ---
        try:
            if self.depth_conn is None:
                try:
                    self.depth_conn, _ = self.depth_server_sock.accept()
                    self.depth_conn.setblocking(False)
                    self._depth_rx_buffer.clear()
                    self.get_logger().info("Depth TCP client connected.")
                except BlockingIOError:
                    pass
            else:
                payload = self._read_latest_depth_payload()
                if payload is not None:
                    h, w = struct.unpack("II", payload[:8])
                    depth_bytes = payload[8:]

                    # Sender transmits uint16 (2 bytes/pixel)
                    expected_size = h * w * 2
                    if len(depth_bytes) != expected_size:
                        raise ValueError(
                            f"depth payload size mismatch: got {len(depth_bytes)}, expected {expected_size}"
                        )

                    depth_mm = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((h, w))
                    depth_image = depth_mm.astype(np.float32) / 1000.0

                    msg = Image()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "front_camera_depth"
                    msg.height = h
                    msg.width = w
                    msg.encoding = "32FC1"
                    msg.is_bigendian = 0
                    msg.step = w * 4
                    msg.data = depth_image.tobytes()

                    self.depth_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"depth receive error: {e}")
            if self.depth_conn is not None:
                try:
                    self.depth_conn.close()
                except Exception:
                    pass
                self.depth_conn = None
                self._depth_rx_buffer.clear()

        # --- Joint states ---
        try:
            data = self._recv_latest_udp(self.joint_state_sock, 65535)
            if data is None:
                pass
            else:
                msg_in = json.loads(data.decode("utf-8"))

                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.header.frame_id = "base"

                js.name = msg_in.get("names", [])
                js.position = msg_in.get("position", [])
                js.velocity = msg_in.get("velocity", [])

                self.joint_state_pub.publish(js)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"joint_state receive error: {e}")

        # --- IMU ---
        try:
            data = self._recv_latest_udp(self.imu_sock, 4096)
            if data is None:
                pass
            else:
                msg_in = json.loads(data.decode("utf-8"))

                imu = Imu()
                imu.header.stamp = self.get_clock().now().to_msg()
                imu.header.frame_id = "imu_link"

                imu.angular_velocity.x = float(msg_in.get("wx", 0.0))
                imu.angular_velocity.y = float(msg_in.get("wy", 0.0))
                imu.angular_velocity.z = float(msg_in.get("wz", 0.0))

                imu.linear_acceleration.x = float(msg_in.get("ax", 0.0))
                imu.linear_acceleration.y = float(msg_in.get("ay", 0.0))
                imu.linear_acceleration.z = float(msg_in.get("az", 0.0))

                self.imu_pub.publish(imu)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"imu receive error: {e}")

        # --- Object sequence (from isaac_controller) ---
        try:
            data = self._recv_latest_udp(self.object_seq_sock, 4096)
            if data is not None:
                out = String()
                out.data = data.decode("utf-8")
                self.object_seq_pub.publish(out)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"object_seq receive error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
