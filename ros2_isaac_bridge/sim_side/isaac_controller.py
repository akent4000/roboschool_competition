import isaacgym
assert isaacgym

import glob
import pickle as pkl
import time
from pathlib import Path

import torch

from sim_bridge_client import SimBridgeClient

from aliengo_gym.envs import *
from aliengo_gym.envs.base.legged_robot_config import Cfg
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


def load_policy(logdir):
    body = torch.jit.load(logdir + "/checkpoints/body_latest.jit")
    adaptation_module = torch.jit.load(logdir + "/checkpoints/adaptation_module_latest.jit")

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(label, headless=False, seed=0):
    dirs = glob.glob(str(RUNS_DIR / label / "*"))
    if not dirs:
        raise FileNotFoundError(
            f"No runs found for label '{label}' under {RUNS_DIR}. "
            "Check that the trained run exists and that the label matches the run directory."
        )
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.terrain_length = 10.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"
    Cfg.env.episode_length_s = 300

    Cfg.env.front_camera_enabled = True
    Cfg.env.front_camera_attach_body_name = "trunk"
    Cfg.env.front_camera_color_width_px = 640
    Cfg.env.front_camera_color_height_px = 360
    Cfg.env.front_camera_depth_width_px = 848
    Cfg.env.front_camera_depth_height_px = 480
    Cfg.env.front_camera_color_fov_h_deg = 70.0
    Cfg.env.front_camera_depth_fov_h_deg = 86.0
    Cfg.env.front_camera_offset_xyz = [0.315, 0.0, 0.052]
    Cfg.env.front_camera_pitch_deg = -4.0

    # Use CPU PhysX to reduce VRAM usage (policy still runs on GPU)
    Cfg.sim.use_gpu_pipeline = False

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(seed=seed, sim_device="cpu", headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)

    return env, policy


def main():
    label = "gait-conditioned-agility/aliengo-v0/train"
    seed = 5

    bridge = SimBridgeClient()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--objects", type=int, nargs="+", default=None,
                    help="Override object visit order, e.g. --objects 3 1 4 0 2")
    args = ap.parse_args()
    if args.seed is not None:
        seed = args.seed
    env, policy = load_env(label, headless=args.headless, seed=seed)
    if args.objects is not None:
        SEQUENCE_OF_OBJECTS = args.objects
    else:
        SEQUENCE_OF_OBJECTS = env.SEQUENCE_OF_OBJECTS

    obs = env.reset()

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor([0.5, 0.0, 0.0])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ] # env.dof_names

    print("Isaac controller started.")

    from datetime import datetime
    import os
    from aliengo_gym import MINI_GYM_ROOT_DIR

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(MINI_GYM_ROOT_DIR, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"log_seed_{seed}.txt")
    log_file = open(log_path, "w+")

    print(f"[LOG] Saving log to: {log_path}")

    object_positions = env.detectable_object_positions

    log_file.write(f"seed={seed}\n\n")
    log_file.write(f"SEQUENCE_OF_OBJECTS = {SEQUENCE_OF_OBJECTS}\n")

    for obj in object_positions:
        log_file.write(
            f"object {obj['id']}: "
            f"cell=({obj['cell_x']}, {obj['cell_y']}), "
            f"world=({obj['x']:.2f}, {obj['y']:.2f})\n"
        )

    log_file.write("\n")
    log_file.write("detected_objects = {}\n")
    log_file.write("\nt,x,y,yaw\n")
    log_file.flush()

    detected_objects = {}
    i = 0

    def log_detected_object(object_id):
        nonlocal detected_objects, log_file, t, x, y, yaw

        if object_id in detected_objects:
            return

        detected_objects[object_id] = {
            "t": round(t, 3),
            "x": round(x, 4),
            "y": round(y, 4),
            "yaw": round(yaw, 4),
        }

        log_file.seek(0)
        lines = log_file.readlines()

        new_block = "detected_objects = {\n"
        for k, v in detected_objects.items():
            new_block += f"{k}: {v},\n"
        new_block += "}\n"

        start, end = None, None

        for i, line in enumerate(lines):
            if line.startswith("detected_objects"):
                start = i
                if line.strip().endswith("}"):
                    end = i
                else:
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() == "}":
                            end = j
                            break
                break

        lines[start:end+1] = [new_block]

        log_file.seek(0)
        log_file.writelines(lines)
        log_file.truncate()
        log_file.flush()

    LOOP_DT = 0.02  # target loop period (50 Hz)
    CAMERA_SEND_EVERY = 3  # send images every N steps (~17 Hz)
    LOG_FLUSH_EVERY = 25
    PRINT_EVERY = 25

    while True:
        loop_start = time.perf_counter()

        cmd = bridge.receive_cmd()

        with torch.no_grad():
            actions = policy(obs)

        env.commands[:, 0] = cmd["vx"]
        env.commands[:, 1] = cmd["vy"]
        env.commands[:, 2] = cmd["wz"]
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        obs, rew, done, info = env.step(actions)

        t = i * env.dt
        i += 1

        x = env.root_states[0, 0].item()
        y = env.root_states[0, 1].item()
        quat = env.root_states[0, 3:7]

        yaw = torch.atan2(
            2.0 * (quat[3]*quat[2] + quat[0]*quat[1]),
            1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        ).item()

        log_file.write(f"{t:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n")
        if i % LOG_FLUSH_EVERY == 0:
            log_file.flush()

        detected_object_id = bridge.receive_detected_object()
        if detected_object_id is not None:
            log_detected_object(detected_object_id)

        # Send object sequence periodically so the ROS2 controller can receive it
        if i % 50 == 0:
            bridge.send_object_sequence(SEQUENCE_OF_OBJECTS)

        # Send images at reduced rate (~17 Hz) to save bandwidth
        if i % CAMERA_SEND_EVERY == 0:
            camera_data = env.get_front_camera_data(0)
            if camera_data is None:
                print("camera_data is None")
            else:
                bridge.send_rgb(camera_data["image"])
                bridge.send_depth(camera_data["depth"])

        measured_vx = env.base_lin_vel[0, 0].item()
        measured_vy = env.base_lin_vel[0, 1].item()
        measured_wz = env.base_ang_vel[0, 2].item()

        base_ang_vel = env.base_ang_vel[0, :].detach().cpu().numpy()
        base_lin_acc = [0.0, 0.0, 0.0]

        relative_dof_pos = (env.dof_pos[0, :] - env.default_dof_pos[0, :]).detach().cpu().numpy()
        dof_vel = env.dof_vel[0, :].detach().cpu().numpy()

        bridge.send_state(
            vx=measured_vx,
            vy=measured_vy,
            wz=measured_wz,
        )

        bridge.send_joint_states(
            names=JOINT_NAMES,
            position=relative_dof_pos,
            velocity=dof_vel,
        )

        bridge.send_imu(
            ang_vel=base_ang_vel,
            lin_acc=base_lin_acc,
        )

        if i % PRINT_EVERY == 0:
            print(
                f"cmd={{'vx': {cmd['vx']:.3f}, 'vy': {cmd['vy']:.3f}, 'wz': {cmd['wz']:.3f}}} "
                f"state={{'vx': {measured_vx:.3f}, 'vy': {measured_vy:.3f}, 'wz': {measured_wz:.3f}}}"
            )

        # Sleep only the remaining time to maintain target loop rate
        elapsed = time.perf_counter() - loop_start
        remaining = LOOP_DT - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
