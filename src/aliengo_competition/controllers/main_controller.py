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
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    _raw_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    # SEQUENCE_OF_OBJECTS returns [(id, "name"), ...] — extract just the int IDs
    object_queue = [item[0] if isinstance(item, (tuple, list)) else int(item)
                    for item in _raw_queue]
    print(f"[Контроллер] отрисовка_камеры={'включена' if render_camera else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue} (raw={_raw_queue})")

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    import math as _math
    import os as _os
    from aliengo_competition.controllers.slam import SlamController
    from aliengo_competition.controllers.visualizer import DashboardVisualizer

    warmup_s = 0.4
    slam = SlamController(control_dt=control_dt)
    dashboard = DashboardVisualizer(
        enabled=render_camera,
        depth_max_m=camera_depth_max_m,
    )
    _VIS_EVERY = 3  # render dashboard every N steps (~17 Hz)

    # Debug: save occupancy grid image every N seconds (0 = disabled)
    map_save_interval_s = 10.0
    last_map_save_t = 0.0

    # --- YOLO detector ---
    _yolo_model = None
    _YOLO_MODEL_PATH = "runs/yolo_detector/train/weights/best1.pt"
    try:
        if _os.path.isfile(_YOLO_MODEL_PATH):
            from ultralytics import YOLO as _YOLO
            _yolo_model = _YOLO(_YOLO_MODEL_PATH)
            _model_names = getattr(_yolo_model, "names", {})
            print(f"[Detector] YOLO loaded: {_YOLO_MODEL_PATH}")
            print(f"[Detector] Model class names: {_model_names}")
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
    _DETECT_CONF = 0.45      # base confidence threshold (YOLO pre-filter)
    _CONFIRM_DIST_M = 0.70   # visit confirmed within this radius (m)
    _CONFIRM_WAIT_S = 2.0    # seconds to stop near object for confirmation
    _DEPTH_PATCH = 15        # half-size of depth sampling window (px)

    # Per-class confidence thresholds (override _DETECT_CONF for specific classes)
    _CLASS_CONF = {
        2: 0.82,  # chair — high false-positive rate, require 82%
    }
    # How many times an object must be seen before saving to known_objects
    _MEMORY_SIGHT_THRESH = 3

    # Post-confirmation: back up for this many seconds after visiting an object
    _BACKUP_DURATION_S = 1.5
    _BACKUP_VX = -0.25

    # Mutable state that persists across loop iterations
    class _DS:
        last_detect_step = -999
        detections = []       # [(class_id, u_center, v_center, conf), ...]
        queue_idx = 0         # current position in object_queue
        target_world = None   # (wx, wy) estimated world pos of target
        confirming_since_t = None  # sim_t when stop-and-confirm started
        nav_active = False    # True while approaching a detected object
        backup_until_t = 0.0  # sim_t until which the robot backs away
        known_objects = {}    # {class_id: (wx, wy)} — remembered positions
        visited_positions = []  # [(wx, wy)] — positions of visited objects
        sight_counts = {}     # {class_id: (count, wx, wy)} — sighting accumulator
    _ds = _DS()

    def _run_yolo(rgb, step_idx):
        """Run YOLO inference (throttled). Caches results in _ds.detections.

        Each detection: (cls_id, u_center, v_center, conf, box_w, box_h)
        """
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
                # Per-class confidence gate
                min_conf = _CLASS_CONF.get(cls_id, _DETECT_CONF)
                if conf < min_conf:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append((
                    cls_id,
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    conf,
                    x2 - x1,  # box width
                    y2 - y1,  # box height
                ))
        _ds.detections = dets
        if dets:
            _names = {0: "backpack", 1: "bottle", 2: "chair", 3: "cup", 4: "laptop"}
            _summary = ", ".join(
                f"{_names.get(d[0], '?')}({d[3]:.2f})" for d in dets
            )
            print(f"[YOLO] step={step_idx}: {_summary}")
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

                # Save world positions of ALL detected objects (not just target)
                rx, ry, rt = slam.odom.pose
                visited_set = set(current_object_queue[:_ds.queue_idx])
                for det in dets:
                    cls_id_det = det[0]
                    if cls_id_det in visited_set:
                        continue
                    uc_det, vc_det = det[1], det[2]
                    d_det = _sample_depth_at(depth, uc_det, vc_det)
                    if d_det is not None:
                        wx_det, wy_det = _pixel_to_world(
                            uc_det, d_det, rx, ry, rt
                        )
                    else:
                        # Beyond depth range — estimate position from bearing
                        _bearing_det = _math.atan2(_RGB_CX - uc_det, _RGB_FX)
                        _ga_det = rt + _bearing_det
                        wx_det = rx + 5.0 * _math.cos(_ga_det)
                        wy_det = ry + 5.0 * _math.sin(_ga_det)
                        too_close = any(
                            _math.hypot(wx_det - vpx, wy_det - vpy) < 1.5
                            for vpx, vpy in _ds.visited_positions
                        )
                        if too_close:
                            continue
                        # Accumulate sightings — only promote to known after
                        # _MEMORY_SIGHT_THRESH consistent detections
                        prev = _ds.sight_counts.get(cls_id_det)
                        if prev is not None:
                            cnt, px, py = prev
                            if _math.hypot(wx_det - px, wy_det - py) < 1.0:
                                cnt += 1
                                # running average of position
                                avg_x = (px * (cnt - 1) + wx_det) / cnt
                                avg_y = (py * (cnt - 1) + wy_det) / cnt
                                _ds.sight_counts[cls_id_det] = (cnt, avg_x, avg_y)
                            else:
                                # position jumped — reset counter
                                _ds.sight_counts[cls_id_det] = (1, wx_det, wy_det)
                        else:
                            _ds.sight_counts[cls_id_det] = (1, wx_det, wy_det)

                        cnt, sx, sy = _ds.sight_counts[cls_id_det]
                        if cnt >= _MEMORY_SIGHT_THRESH:
                            _names = {0: "backpack", 1: "bottle", 2: "chair",
                                      3: "cup", 4: "laptop"}
                            if cls_id_det not in _ds.known_objects:
                                print(
                                    f"[Memory] Saved {_names.get(cls_id_det, '?')}"
                                    f" at ({sx:.2f}, {sy:.2f})"
                                    f" after {cnt} sightings"
                                )
                            _ds.known_objects[cls_id_det] = (sx, sy)

                # Pick highest-confidence detection of the target class
                best, best_conf = None, 0.0
                for det in dets:
                    if det[0] == target_cls and det[3] > best_conf:
                        best, best_conf = det, det[3]

                if best is None:
                    if _ds.confirming_since_t is None:
                        # Not confirming and no detection → lose target
                        _ds.target_world = None
                        _ds.nav_active = False
                    # If confirming, robot is stopped near object — keep timer running
                    return None

                _, uc, vc, _, _, _ = best
                d_m = _sample_depth_at(depth, uc, vc)
                if d_m is None:
                    # Object visible but beyond depth range — navigate by bearing
                    bearing = _math.atan2(_RGB_CX - uc, _RGB_FX)
                    _FAR_ASSUMED_DIST = 5.0
                    ga = rt + bearing
                    wx = rx + _FAR_ASSUMED_DIST * _math.cos(ga)
                    wy = ry + _FAR_ASSUMED_DIST * _math.sin(ga)
                    _ds.target_world = (wx, wy)
                    _ds.nav_active = True
                    _ds.confirming_since_t = None
                    return None

                wx, wy = _pixel_to_world(uc, d_m, rx, ry, rt)
                _ds.target_world = (wx, wy)
                _ds.nav_active = True

                dist = _math.hypot(wx - rx, wy - ry)
                if dist < _CONFIRM_DIST_M:
                    # Inside confirmation radius — start or continue stop timer
                    if _ds.confirming_since_t is None:
                        _ds.confirming_since_t = sim_t
                        print(
                            f"[Detector] Target in range ({dist:.2f}m), "
                            f"stopping for {_CONFIRM_WAIT_S}s confirmation..."
                        )
                    elapsed = sim_t - _ds.confirming_since_t
                    if elapsed >= _CONFIRM_WAIT_S:
                        confirmed_id = target_cls
                        _ds.queue_idx += 1
                        _ds.confirming_since_t = None
                        _ds.visited_positions.append((wx, wy))
                        if confirmed_id in _ds.known_objects:
                            del _ds.known_objects[confirmed_id]
                        _ds.target_world = None
                        _ds.nav_active = False
                        _ds.backup_until_t = sim_t + _BACKUP_DURATION_S
                        print(
                            f"[Detector] CONFIRMED object {confirmed_id} "
                            f"({_ds.queue_idx}/{len(current_object_queue)})"
                        )
                        return confirmed_id
                else:
                    # Outside confirmation radius — reset timer
                    _ds.confirming_since_t = None

                return None

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)
                _obj_name = {0: "backpack", 1: "bottle", 2: "chair",
                             3: "cup", 4: "laptop"}.get(detected_object_id, "?")
                print(f"[LOG] Object {detected_object_id} ({_obj_name}) "
                      f"visited at t={sim_t:.2f}s")

            # --- SLAM-навигация ---
            local_t = max(sim_t - segment_start_t, 0.0)

            # Reset SLAM odometry after a fall (local_t near zero = just reset)
            if local_t < control_dt * 2:
                slam.reset_pose()
                _ds.confirming_since_t = None
                _ds.target_world = None
                _ds.nav_active = False

            # Pass visited positions to SLAM for exclusion zones
            slam.set_exclusion_zones(_ds.visited_positions)

            # Direct SLAM toward detected object, known object, or explore frontiers
            if _ds.nav_active and _ds.target_world is not None:
                slam.set_navigation_target(*_ds.target_world)
            elif (_ds.queue_idx < len(object_queue)
                  and object_queue[_ds.queue_idx] in _ds.known_objects):
                known_pos = _ds.known_objects[object_queue[_ds.queue_idx]]
                slam.set_navigation_target(*known_pos)
            else:
                slam.clear_navigation_target()

            # Always feed sensor data to SLAM (builds map even during warmup)
            vx_cmd, vy_cmd, vw_cmd = slam.update(step_index, state, camera_data)

            if _ds.queue_idx >= len(object_queue) and len(object_queue) > 0:
                # All objects visited — stop near the last one
                vx = 0.0
                vy = 0.0
                vw = 0.0
            elif local_t < warmup_s:
                vx = 0.0
                vy = 0.0
                vw = 0.0
            elif sim_t < _ds.backup_until_t:
                # Back away after confirming an object visit
                vx = _BACKUP_VX
                vy = 0.0
                vw = 0.0
            elif _ds.confirming_since_t is not None:
                # Stopped inside confirmation radius, waiting 2s
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                vx = vx_cmd
                vy = vy_cmd
                vw = vw_cmd

            # --- Dashboard visualization (throttled) ---
            if step_index % _VIS_EVERY == 0:
                _cur_target_cls = None
                if object_queue and _ds.queue_idx < len(object_queue):
                    _cur_target_cls = object_queue[_ds.queue_idx]
                dashboard.update(
                    rgb=camera_data.get("image") if camera_data else None,
                    depth=camera_data.get("depth") if camera_data else None,
                    detections=_ds.detections,
                    target_cls=_cur_target_cls,
                    slam=slam,
                    vx_cmd=vx,
                    vy_cmd=vy,
                    wz_cmd=vw,
                    queue=object_queue,
                    queue_idx=_ds.queue_idx,
                    sim_t=sim_t,
                    confirm_count=(sim_t - _ds.confirming_since_t) if _ds.confirming_since_t is not None else 0.0,
                    confirm_needed=_CONFIRM_WAIT_S,
                    known_objects=_ds.known_objects,
                    visited_positions=_ds.visited_positions,
                )

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
        dashboard.close()
        robot.stop()
