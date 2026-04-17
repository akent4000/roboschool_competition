"""
Entry-point wrapper for the competition navigation controller.

Finds the project root (ros2_isaac_bridge/sim_side/controller.py) either from
the COMPETITION_ROOT environment variable or by searching upward from this
file until the ros2_isaac_bridge/ directory is found.
"""
import os
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the roboschool_competition project root."""
    env_root = os.environ.get("COMPETITION_ROOT")
    if env_root:
        return Path(env_root)

    # Walk upward from this file until we find ros2_isaac_bridge/
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_isaac_bridge").is_dir():
            return parent

    raise RuntimeError(
        "Cannot find project root automatically. "
        "Set the COMPETITION_ROOT environment variable to the "
        "roboschool_competition directory, e.g.:\n"
        "  export COMPETITION_ROOT=/workspace/aliengo_competition"
    )


def main(args=None):
    project_root = _find_project_root()
    sim_side = project_root / "ros2_isaac_bridge" / "sim_side"
    sys.path.insert(0, str(sim_side))

    from controller import main as _main  # type: ignore[import]

    _main(args=args)
