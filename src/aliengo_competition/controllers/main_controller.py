from __future__ import annotations

import math

from aliengo_competition.robot_interface.base import AliengoRobotInterface


def _extract_base_observation(obs):
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)
    if hasattr(obs, "ndim") and obs.ndim > 1:
        obs = obs[0]
    return obs


def run(robot: AliengoRobotInterface, steps: int = 1000) -> None:
    robot.reset()

    # User-editable blocks in this file:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    # Tune these values to change the demo behavior.
    warmup_steps = 180
    circle_speed = 0.5
    circle_period_steps = 720.0
    ang_vel_scale = 0.25
    yaw_rate_kp = 1.2
    yaw_rate_kd = 0.15
    # ================== USER PARAMETERS END ==================

    warmup_counter = 0
    motion_step = 0
    previous_omega_z_error = 0.0

    for step_index in range(steps):
        obs = robot.get_observation()
        camera = robot.get_camera()
        _ = obs, camera
        base_obs = _extract_base_observation(obs)
        omega_z = float(base_obs[5].item()) / ang_vel_scale if len(base_obs) > 5 else 0.0

        # ================= USER CONTROL LOGIC START =================
        # This block is the intended place for participant logic.
        # You can:
        # - read obs / camera
        # - compute desired vx, vy, vw
        # - compute desired body pitch
        #
        # Example below:
        # 1. Hold still for a short warmup after spawn.
        # 2. Then move along a large circle in the XY plane.
        # 3. Keep the robot facing roughly the same direction by applying
        #    a PD-like correction on yaw rate (omega_z).
        # 4. The circular path is created by vx / vy only, while vw is used
        #    only to suppress unwanted body rotation.
        if warmup_counter < warmup_steps:
            vx = 0.0
            vy = 0.0
            vw = 0.0
            pitch = 0.0
            warmup_counter += 1
        else:
            phase = 2.0 * math.pi * motion_step / circle_period_steps
            vx = circle_speed * math.cos(phase)
            vy = circle_speed * math.sin(phase)

            omega_z_error = -omega_z
            omega_z_error_rate = omega_z_error - previous_omega_z_error
            previous_omega_z_error = omega_z_error

            vw = yaw_rate_kp * omega_z_error + yaw_rate_kd * omega_z_error_rate
            vw = max(min(vw, 0.8), -0.8)
            pitch = 0.0
            motion_step += 1
        # ================== USER CONTROL LOGIC END ==================

        robot.set_speed(vx, vy, vw)
        robot.set_body_pitch(pitch)

        if robot.is_fallen():
            robot.stop()
            robot.reset()
            warmup_counter = 0
            motion_step = 0
            previous_omega_z_error = 0.0
            continue

        robot.step()

    robot.stop()
