from __future__ import annotations

import numpy as np

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Отрисовка камеры отключена: не удалось импортировать cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(view, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


def run(
    robot: AliengoRobotInterface,
    steps: int = 15000,
    render_camera: bool = False,
    camera_depth_max_m: float = 4.0,
    seed: int = 0,
) -> None:
    robot.reset()
    env = getattr(robot, "env", None)
    if env is None:
        raise ValueError("Интерфейс робота должен предоставлять 'env' для обязательного логирования.")

    logger = CompetitionRunLogger(env=env, seed=int(seed))
    camera_renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    object_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    print(f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue}")

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    import math as _math
    import os as _os
    from aliengo_competition.controllers.slam import SlamController

    warmup_s = 0.4
    slam = SlamController(control_dt=control_dt)

    # Debug: save occupancy grid image every N seconds (0 = disabled)
    map_save_interval_s = 10.0
    last_map_save_t = 0.0

    # --- YOLO detector ---
    _yolo_model = None
    _YOLO_MODEL_PATH = "runs/yolo_detector/train/weights/best.pt"
    try:
        if _os.path.isfile(_YOLO_MODEL_PATH):
            from ultralytics import YOLO as _YOLO
            _yolo_model = _YOLO(_YOLO_MODEL_PATH)
            print(f"[Detector] YOLO loaded: {_YOLO_MODEL_PATH}")
        else:
            print(f"[Detector] No model at {_YOLO_MODEL_PATH} — detection disabled")
    except Exception as _e:
        print(f"[Detector] Could not load YOLO: {_e}")

    # Camera intrinsics  (RGB 640×360 @ 70° HFOV,  Depth 848×480 @ 86° HFOV)
    _RGB_W, _RGB_H = 640, 360
    _RGB_FX = _RGB_W / (2.0 * _math.tan(_math.radians(35.0)))   # ≈ 457
    _RGB_CX = _RGB_W / 2.0
    _RGB_CY = _RGB_H / 2.0

    _DEPTH_W, _DEPTH_H = 848, 480
    _DEPTH_FX = _DEPTH_W / (2.0 * _math.tan(_math.radians(43.0)))  # ≈ 455
    _DEPTH_FY = _DEPTH_FX
    _DEPTH_CX = _DEPTH_W / 2.0
    _DEPTH_CY = _DEPTH_H / 2.0

    # Detection tuning
    _DETECT_EVERY = 5        # run YOLO every N steps (~10 Hz)
    _DETECT_CONF = 0.45      # confidence threshold
    _CONFIRM_DIST_M = 0.55   # visit confirmed within this radius (m)
    _CONFIRM_FRAMES = 3      # consecutive close detections needed
    _DEPTH_PATCH = 15        # half-size of depth sampling window (px)

    # Mutable state that persists across loop iterations
    class _DS:
        last_detect_step = -999
        detections = []       # [(class_id, u_center, v_center, conf), ...]
        queue_idx = 0         # current position in object_queue
        target_world = None   # (wx, wy) estimated world pos of target
        confirm_count = 0     # consecutive frames target seen within range
        nav_active = False    # True while approaching a detected object
    _ds = _DS()

    def _run_yolo(rgb, step_idx):
        """Run YOLO inference (throttled). Caches results in _ds.detections."""
        if step_idx - _ds.last_detect_step < _DETECT_EVERY:
            return _ds.detections
        _ds.last_detect_step = step_idx
        if _yolo_model is None or rgb is None:
            _ds.detections = []
            return _ds.detections
        results = _yolo_model(rgb[..., ::-1], conf=_DETECT_CONF, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append((cls_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
        _ds.detections = dets
        return dets

    def _sample_depth_at(depth_img, u_rgb, v_rgb):
        """Sample depth (m) at an RGB-image pixel by reprojecting into the depth image."""
        if depth_img is None:
            return None
        # Map RGB pixel → depth pixel via shared ray direction
        u_d = int(_DEPTH_CX + _DEPTH_FX * (u_rgb - _RGB_CX) / _RGB_FX)
        v_d = int(_DEPTH_CY + _DEPTH_FY * (v_rgb - _RGB_CY) / _RGB_FX)
        H, W = depth_img.shape
        p = _DEPTH_PATCH
        patch = depth_img[max(0, v_d - p):min(H, v_d + p),
                          max(0, u_d - p):min(W, u_d + p)]
        valid = patch[(patch > 0.15) & (patch < 4.0) & np.isfinite(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _pixel_to_world(u_rgb, depth_z, ox, oy, ot):
        """RGB pixel + z-depth + odometry → world (x, y)."""
        x_r = depth_z                                      # forward in robot frame
        y_r = -depth_z * (u_rgb - _RGB_CX) / _RGB_FX      # left-positive
        c, s = _math.cos(ot), _math.sin(ot)
        return (ox + x_r * c - y_r * s,
                oy + x_r * s + y_r * c)
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

    try:
        initial_observation = robot.get_observation()
        initial_camera_payload = robot.get_camera()
        print(
            "[Контроллер] Предпросмотр API:"
            f" observation_type={type(initial_observation).__name__},"
            f" camera_payload={'да' if initial_camera_payload is not None else 'нет'}"
        )
        if initial_camera_payload is None:
            print(
                "[Контроллер] Предупреждение: данные фронтальной камеры недоступны. "
                "Проверьте, что симулятор не запущен в headless-режиме и что включён front_camera_enabled."
            )

        for step_index in range(total_steps):
            state = robot.get_state()

            # Камеру можно брать и из state, и напрямую через robot.get_camera().
            camera_payload = robot.get_camera()
            camera_state = state.camera
            if (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, dict):
                camera_state = CameraState(
                    rgb=camera_payload.get("image"),
                    depth=camera_payload.get("depth"),
                )
            elif (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, CameraState):
                camera_state = camera_payload
            camera_renderer.show(camera_state)

            # ================= USER CONTROL LOGIC START =================
            sim_t = state.sim_time_s

            joint_names = state.joints.name
            relative_dof_pos = state.q
            dof_vel = state.q_dot
            measured_vx = state.vx
            measured_vy = state.vy
            measured_wz = state.wz
            base_ang_vel = state.imu.angular_velocity_xyz
            base_lin_acc = np.zeros(3, dtype=np.float32)
            camera_data = camera_payload if isinstance(camera_payload, dict) else {
                "image": camera_state.rgb,
                "depth": camera_state.depth,
            }

            # --- Обязательный шаблон логирования объектов ---
            def log_found_object(object_id: int) -> None:
                """ОБЯЗАТЕЛЬНО: вызывайте при обнаружении целевого объекта."""
                logger.log_detected_object_at_time(int(object_id), float(sim_t))

            def get_found_object_id(
                current_state,
                current_camera_data,
                current_object_queue,
            ):
                """Detect current target via YOLO, estimate 3D pose, confirm visit."""
                if not current_object_queue or _ds.queue_idx >= len(current_object_queue):
                    return None

                target_cls = current_object_queue[_ds.queue_idx]
                rgb = current_camera_data.get("image") if current_camera_data else None
                depth = current_camera_data.get("depth") if current_camera_data else None

                dets = _run_yolo(rgb, step_index)

                # Pick highest-confidence detection of the target class
                best, best_conf = None, 0.0
                for det in dets:
                    if det[0] == target_cls and det[3] > best_conf:
                        best, best_conf = det, det[3]

                if best is None:
                    _ds.confirm_count = max(0, _ds.confirm_count - 1)
                    if _ds.confirm_count == 0:
                        _ds.target_world = None
                        _ds.nav_active = False
                    return None

                _, uc, vc, _ = best
                d_m = _sample_depth_at(depth, uc, vc)
                if d_m is None:
                    return None

                rx, ry, rt = slam.odom.pose
                wx, wy = _pixel_to_world(uc, d_m, rx, ry, rt)
                _ds.target_world = (wx, wy)
                _ds.nav_active = True

                dist = _math.hypot(wx - rx, wy - ry)
                if dist < _CONFIRM_DIST_M:
                    _ds.confirm_count += 1
                    if _ds.confirm_count >= _CONFIRM_FRAMES:
                        confirmed_id = target_cls
                        _ds.queue_idx += 1
                        _ds.confirm_count = 0
                        _ds.target_world = None
                        _ds.nav_active = False
                        print(
                            f"[Detector] CONFIRMED object {confirmed_id} "
                            f"({_ds.queue_idx}/{len(current_object_queue)})"
                        )
                        return confirmed_id
                else:
                    _ds.confirm_count = max(0, _ds.confirm_count - 1)

                return None

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)

            # --- SLAM-навигация ---
            local_t = max(sim_t - segment_start_t, 0.0)

            # Reset SLAM odometry after a fall (local_t near zero = just reset)
            if local_t < control_dt * 2:
                slam.reset_pose()
                _ds.confirm_count = 0
                _ds.target_world = None
                _ds.nav_active = False

            # Direct SLAM toward detected object or let it explore frontiers
            if _ds.nav_active and _ds.target_world is not None:
                slam.set_navigation_target(*_ds.target_world)
            else:
                slam.clear_navigation_target()

            # Always feed sensor data to SLAM (builds map even during warmup)
            vx_cmd, vy_cmd, vw_cmd = slam.update(step_index, state, camera_data)

            if local_t < warmup_s:
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                vx = vx_cmd
                vy = vy_cmd
                vw = vw_cmd

            # Debug: save occupancy grid snapshot periodically
            if map_save_interval_s > 0 and sim_t - last_map_save_t >= map_save_interval_s:
                slam.save_map_image(f"slam_map_{int(sim_t):04d}.png")
                last_map_save_t = sim_t
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        robot.stop()
