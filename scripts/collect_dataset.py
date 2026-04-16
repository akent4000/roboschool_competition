#!/usr/bin/env python3
"""
Collect labeled RGB frames from the Isaac Gym simulator for each of the
5 target objects (backpack, bottle, chair, cup, laptop).

The script teleports the robot to a grid of viewpoints around every object,
lets the physics settle, then saves the onboard-camera RGB frame.

Usage (inside the sim environment / conda env with isaacgym):
    python scripts/collect_dataset.py --seeds 0 1 2
    python scripts/collect_dataset.py --seeds 0 --output_dir datasets/sim_objects --settle 40
    python scripts/collect_dataset.py --seeds 0 1 2 --yolo --output_dir datasets/yolo_objects

Output layout (default — per-class folders):
    datasets/sim_objects/
    ├── backpack/          # one folder per class
    │   ├── s0_d070_a000_y+00.png
    │   └── ...
    ├── bottle/
    ├── chair/
    ├── cup/
    ├── laptop/
    └── negative/          # frames with no target object in view

Output layout (--yolo — YOLO detection format):
    datasets/yolo_objects/
    ├── images/train/      # all RGB frames
    │   ├── s0_obj0_d070_a000_y+00.png
    │   └── ...
    ├── labels/train/      # YOLO bbox labels (class xc yc w h)
    │   ├── s0_obj0_d070_a000_y+00.txt
    │   └── ...
    └── data.yaml          # ready for ultralytics YOLO training
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import isaacgym          # must be imported before torch
assert isaacgym
import cv2
import numpy as np
import torch
from isaacgym import gymtorch

from scripts.play_FOR_COLLECT_DATASET import load_env
from aliengo_competition.robot_interface.sim import SimAliengoRobot

# ── Object meta ──────────────────────────────────────────────────────
OBJECT_NAMES = {0: "backpack", 1: "bottle", 2: "chair", 3: "cup", 4: "laptop"}

# ── Viewpoint grid ───────────────────────────────────────────────────
# Distances from object centre (metres)
DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Azimuth angles around the object (degrees, 0 = +x axis)
AZIMUTHS_DEG = list(range(0, 360, 30))           # 12 angles

# Yaw offsets so the object appears at frame centre / left / right edge
YAW_OFFSETS_DEG = [0, -25, 25]


# ── Helpers ──────────────────────────────────────────────────────────
def _unwrap(env):
    """Walk .env chain down to the base LeggedRobot."""
    while hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def yaw_to_quat(yaw: float):
    """Yaw (rad) → quaternion [x, y, z, w] (Isaac Gym convention)."""
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def teleport(base_env, x: float, y: float, z: float, yaw: float):
    """Place the robot at (x, y, z) facing *yaw* and zero all velocities."""
    dev = base_env.device
    base_env.root_states[0, 0:3] = torch.tensor([x, y, z], dtype=torch.float32, device=dev)
    base_env.root_states[0, 3:7] = torch.tensor(yaw_to_quat(yaw), dtype=torch.float32, device=dev)
    base_env.root_states[0, 7:13] = 0.0                          # zero lin+ang vel
    base_env.gym.set_actor_root_state_tensor(
        base_env.sim, gymtorch.unwrap_tensor(base_env.root_states)
    )
    # Reset joints to default standing pose
    base_env.dof_pos[:] = base_env.default_dof_pos
    base_env.dof_vel[:] = 0.0
    base_env.gym.set_dof_state_tensor(
        base_env.sim, gymtorch.unwrap_tensor(base_env.dof_state)
    )


def settle(robot: SimAliengoRobot, steps: int):
    """Run *steps* physics ticks with zero velocity command."""
    robot.set_speed(0.0, 0.0, 0.0)
    for _ in range(steps):
        robot.step()
        if robot.is_fallen():
            return False
    return True


def capture_camera(robot: SimAliengoRobot) -> tuple:
    """Return (rgb, depth, segmentation) or (None, None, None)."""
    cam = robot.get_camera()
    if cam is None:
        return None, None, None
    if isinstance(cam, dict):
        return cam.get("image"), cam.get("depth"), cam.get("segmentation")
    return getattr(cam, "rgb", None), getattr(cam, "depth", None), None


def capture_rgb(robot: SimAliengoRobot) -> np.ndarray | None:
    """Return the onboard camera RGB frame (H×W×3 uint8) or None."""
    img, _, _ = capture_camera(robot)
    return img


# ── YOLO bounding-box helpers ───────────────────────────────────────

def compute_yolo_annotations(
    obj_positions: list[dict],
    img_w: int,
    img_h: int,
    segmentation: "np.ndarray | None" = None,
    min_px: int = 8,
    max_frac: float = 0.90,
) -> list[tuple[int, float, float, float, float]]:
    """Compute YOLO annotations from the per-pixel segmentation buffer.

    Each object actor has ``segmentation_id = object_id + 1`` (set in
    ``legged_robot.py``).  The rasteriser handles occlusion natively, so
    objects behind walls produce zero pixels and are automatically excluded.
    Close-up objects that fill the entire frame are also filtered correctly.

    Returns a list of ``(class_id, xc, yc, w, h)`` normalised to ``[0, 1]``.
    """
    if segmentation is None:
        return []

    annots: list[tuple[int, float, float, float, float]] = []

    for obj in obj_positions:
        seg_id = obj["id"] + 1  # segmentation_id assigned in legged_robot.py
        mask = segmentation == seg_id

        pixel_count = int(mask.sum())
        if pixel_count < min_px * min_px:
            continue

        # Bounding box from actual rendered pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        v_min, v_max = int(np.argmax(rows)), int(img_h - 1 - np.argmax(rows[::-1]))
        u_min, u_max = int(np.argmax(cols)), int(img_w - 1 - np.argmax(cols[::-1]))

        bw = u_max - u_min + 1
        bh = v_max - v_min + 1
        if bw < min_px or bh < min_px:
            continue
        if bw > img_w * max_frac and bh > img_h * max_frac:
            continue

        xc = np.clip((u_min + u_max + 1) / 2.0 / img_w, 0, 1)
        yc = np.clip((v_min + v_max + 1) / 2.0 / img_h, 0, 1)
        wn = np.clip(bw / img_w, 0, 1)
        hn = np.clip(bh / img_h, 0, 1)

        annots.append((obj["id"], float(xc), float(yc), float(wn), float(hn)))

    return annots


def save_yolo_label(
    path: Path, annots: list[tuple[int, float, float, float, float]]
) -> None:
    """Write one YOLO-format label file (empty file is valid for negatives)."""
    with open(path, "w") as f:
        for cls_id, xc, yc, wn, hn in annots:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


# ── Main collection loop ────────────────────────────────────────────
def collect_for_seed(
    seed: int,
    output_dir: Path,
    settle_steps: int = 40,
    verbose: bool = True,
    yolo: bool = False,
) -> dict:
    """Load env with *seed*, capture frames for every object, return counts."""
    print(f"\n{'=' * 60}")
    print(f"  seed={seed}")
    print(f"{'=' * 60}")

    env, policy = load_env(
        "gait-conditioned-agility/aliengo-v0/train",
        headless=True,
        seed=seed,
    )
    base_env = _unwrap(env)
    robot = SimAliengoRobot(env=env, policy=policy)
    robot.reset()

    # Read the standing height the environment chose after reset
    standing_z = float(base_env.root_states[0, 2].item())
    obj_positions = base_env.detectable_object_positions
    if verbose:
        for o in obj_positions:
            print(f"  object {o['id']} ({OBJECT_NAMES[o['id']]}): "
                  f"world=({o['x']:.2f}, {o['y']:.2f})")

    counts: dict[str, int] = {}

    # ── YOLO directory setup ─────────────────────────────────────────
    if yolo:
        img_dir = output_dir / "images" / "train"
        lbl_dir = output_dir / "labels" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        yolo_img_w = base_env.front_camera_color_props.width
        yolo_img_h = base_env.front_camera_color_props.height

    # ── Per-object viewpoints ────────────────────────────────────────
    for obj in obj_positions:
        obj_id   = obj["id"]
        ox, oy   = obj["x"], obj["y"]
        cls_name = OBJECT_NAMES[obj_id]
        if not yolo:
            cls_dir = output_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for dist in DISTANCES:
            for az_deg in AZIMUTHS_DEG:
                az = math.radians(az_deg)
                rx = ox + dist * math.cos(az)
                ry = oy + dist * math.sin(az)
                facing = math.atan2(oy - ry, ox - rx)  # toward object

                for yoff_deg in YAW_OFFSETS_DEG:
                    yaw = facing + math.radians(yoff_deg)

                    teleport(base_env, rx, ry, standing_z, yaw)
                    ok = settle(robot, settle_steps)
                    if not ok:
                        # Robot fell — reset and skip this viewpoint
                        robot.reset()
                        teleport(base_env, rx, ry, standing_z, yaw)
                        ok = settle(robot, settle_steps)
                        if not ok:
                            robot.reset()
                            continue

                    img, _, seg = capture_camera(robot)
                    if img is None:
                        continue

                    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

                    if yolo:
                        fname = (f"s{seed}_obj{obj_id}_d{int(dist * 100):03d}"
                                 f"_a{az_deg:03d}_y{yoff_deg:+03d}.png")
                        cv2.imwrite(str(img_dir / fname), bgr)
                        annots = compute_yolo_annotations(
                            obj_positions, yolo_img_w, yolo_img_h,
                            segmentation=seg,
                        )
                        save_yolo_label(
                            lbl_dir / fname.replace(".png", ".txt"), annots,
                        )
                    else:
                        fname = (f"s{seed}_d{int(dist * 100):03d}"
                                 f"_a{az_deg:03d}_y{yoff_deg:+03d}.png")
                        cv2.imwrite(str(cls_dir / fname), bgr)

                    saved += 1

        counts[cls_name] = saved
        if verbose:
            print(f"  {cls_name}: {saved} frames")

    # ── Negatives: face away from every object ───────────────────────
    if not yolo:
        neg_dir = output_dir / "negative"
        neg_dir.mkdir(parents=True, exist_ok=True)
    neg_saved = 0

    for obj in obj_positions:
        ox, oy = obj["x"], obj["y"]
        for az_deg in range(0, 360, 45):  # 8 directions
            az = math.radians(az_deg)
            rx = ox + 1.5 * math.cos(az)
            ry = oy + 1.5 * math.sin(az)
            # Face *away* from the object
            away_yaw = math.atan2(ry - oy, rx - ox)

            teleport(base_env, rx, ry, standing_z, away_yaw)
            ok = settle(robot, settle_steps)
            if not ok:
                robot.reset()
                continue

            img, _, seg = capture_camera(robot)
            if img is None:
                continue

            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            fname = f"s{seed}_neg_away_obj{obj['id']}_a{az_deg:03d}.png"

            if yolo:
                cv2.imwrite(str(img_dir / fname), bgr)
                annots = compute_yolo_annotations(
                    obj_positions, yolo_img_w, yolo_img_h,
                    segmentation=seg,
                )
                save_yolo_label(
                    lbl_dir / fname.replace(".png", ".txt"), annots,
                )
            else:
                cv2.imwrite(str(neg_dir / fname), bgr)

            neg_saved += 1

    # Random positions far from any object
    rng = np.random.default_rng(seed + 9999)
    terrain_x_max = base_env.cfg.terrain.terrain_length - 0.5
    terrain_y_max = base_env.cfg.terrain.terrain_width - 0.5
    attempts = 0
    while neg_saved < 60 and attempts < 200:
        attempts += 1
        rx = float(rng.uniform(0.5, terrain_x_max))
        ry = float(rng.uniform(0.5, terrain_y_max))
        min_d = min(math.hypot(rx - o["x"], ry - o["y"]) for o in obj_positions)
        if min_d < 4.0:
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        teleport(base_env, rx, ry, standing_z, yaw)
        ok = settle(robot, settle_steps)
        if not ok:
            robot.reset()
            continue
        img, _, seg = capture_camera(robot)
        if img is None:
            continue

        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        fname = f"s{seed}_neg_rand_{neg_saved:03d}.png"

        if yolo:
            cv2.imwrite(str(img_dir / fname), bgr)
            annots = compute_yolo_annotations(
                obj_positions, yolo_img_w, yolo_img_h,
                segmentation=seg,
            )
            save_yolo_label(
                lbl_dir / fname.replace(".png", ".txt"), annots,
            )
        else:
            cv2.imwrite(str(neg_dir / fname), bgr)

        neg_saved += 1

    counts["negative"] = neg_saved
    if verbose:
        print(f"  negative: {neg_saved} frames")

    total = sum(counts.values())
    print(f"  TOTAL for seed {seed}: {total}")

    # Cleanup
    del robot, env
    torch.cuda.empty_cache()
    return counts


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Collect labeled RGB frames from the simulator camera."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds (each seed → different room layout).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/sim_objects",
        help="Root output directory.",
    )
    parser.add_argument(
        "--settle", type=int, default=40,
        help="Physics steps to settle after each teleport (default 40 ≈ 0.8 s).",
    )
    parser.add_argument(
        "--yolo", action="store_true",
        help="Output in YOLO detection format (images/ + labels/ + data.yaml).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grand_total = 0

    if len(args.seeds) > 1:
        # Isaac Gym / PhysX can only be initialised once per process.
        # Spawn a child process for each seed to work around this.
        for seed in args.seeds:
            cmd = [
                sys.executable, __file__,
                "--seeds", str(seed),
                "--output_dir", str(out),
                "--settle", str(args.settle),
            ]
            if args.yolo:
                cmd.append("--yolo")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[ERROR] seed={seed} exited with code {result.returncode}")
                sys.exit(result.returncode)
    else:
        counts = collect_for_seed(
            args.seeds[0], out, settle_steps=args.settle, yolo=args.yolo,
        )
        grand_total += sum(counts.values())

    # ── Generate data.yaml for YOLO training ────────────────────────
    if args.yolo:
        data_yaml = out / "data.yaml"
        names_block = "\n".join(
            f"  {i}: {name}" for i, name in sorted(OBJECT_NAMES.items())
        )
        data_yaml.write_text(
            f"path: {out.resolve()}\n"
            f"train: images/train\n"
            f"val: images/train\n"
            f"\n"
            f"nc: {len(OBJECT_NAMES)}\n"
            f"names:\n"
            f"{names_block}\n"
        )
        print(f"\nYOLO data.yaml → {data_yaml.resolve()}")

    # ── Final summary (count from disk so it works for subprocess runs) ─
    print(f"\n{'=' * 60}")
    if args.yolo:
        img_count = len(list((out / "images" / "train").glob("*.png")))
        lbl_count = len(list((out / "labels" / "train").glob("*.txt")))
        print(f"Done — {img_count} frames total  →  {out.resolve()}")
        print(f"  images/train: {img_count} images")
        print(f"  labels/train: {lbl_count} labels")
    else:
        total_on_disk = 0
        print("Class breakdown:")
        for d in sorted(out.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("*.png")))
                total_on_disk += n
                print(f"  {d.name:12s} {n:>5d} images")
        print(f"Done — {total_on_disk} frames total  →  {out.resolve()}")


if __name__ == "__main__":
    main()
