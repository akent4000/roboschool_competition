from __future__ import annotations

from types import SimpleNamespace

from aliengo_competition.common.helpers import get_args
from aliengo_competition.robot_interface.sim import SimAliengoRobot
from scripts.play import DEFAULT_RUN_LABEL, load_env

DEFAULT_CAMERA_DEPTH_MAX_M = 10.0

def _clone_args(args, *, task: str, mode: str, headless: bool, checkpoint=-1):
    clone = SimpleNamespace(**vars(args))
    clone.task = task
    clone.mode = mode
    clone.headless = headless
    clone.num_envs = 1
    clone.checkpoint = checkpoint
    clone.resume = True
    return clone

def make_robot_interface(
    *,
    args=None,
    task: str = "aliengo_flat",
    mode: str = "sim",
    headless: bool = True,
    checkpoint=-1,
):
    if args is None:
        args = get_args()
    _clone_args(args, task=task, mode=mode, headless=headless, checkpoint=checkpoint)
    if checkpoint not in (-1, None):
        print("Explicit checkpoints are ignored for controller demo; using the latest exported JIT policy from the selected run.")

    print(f"Loading controller low-level policy from label: {DEFAULT_RUN_LABEL}")
    env, policy = load_env(DEFAULT_RUN_LABEL, headless=headless)
    return SimAliengoRobot(env=env, policy=policy)
