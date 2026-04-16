# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview — Робошкола 2026 Final

**Theme:** "Quadrupeds: from locomotion to navigation"

The task is to develop software modules for **autonomous navigation** of the Aliengo quadruped robot using an onboard RGB-D camera. The robot must explore an unknown environment, detect target objects (image markers), and visit them in a prescribed sequence.

### What organizers provide
- Trained RL locomotion policy (velocity-tracking, runs inside Isaac Gym)
- Configured simulation environment (Isaac Gym + Docker + optional ROS 2 bridge)
- RGB-D sensor streams: `IMAGE_COLOR` (640×360) and `IMAGE_DEPTH` (848×480)
- Fixed debug scene with known object positions (for development/testing)
- Examples of target objects (markers/images) before the final

### What participants must implement
1. **Object detection** — identify marker images from RGB-D stream, output 3D position + object ID
2. **Active exploration** — move autonomously through unknown room when target is not visible
3. **Global path planning** — route to target using an incrementally built map
4. **Local control** — generate `(vx, vy, wz)` velocity commands for the RL walking policy
5. **Sequence management** — visit objects in the order specified by organizers; confirm each before moving on

### Scoring rules
- **Detection:** points for each correctly identified object
- **Visit:** object is "reached" when robot stops within radius R ≈ 0.5 m **and** confirms it via camera, in correct order
- **Time:** if all objects visited in order → winner = lowest total time
- **Partial:** ranked by number of correctly-ordered visits; tie-break = time to last reached object

### Testing stages
- **Stage 1 (all teams):** simulator evaluation with random room layout and object placement
- **Stage 2 (top-5):** real robot test

> **Key constraint:** map is not provided — robot must operate in a completely unknown environment. During validation, organizers randomize both room topology and object positions.

### Target objects
Five object classes in `resources/assets/objects/`: `backpack`, `bottle`, `chair`, `cup`, `laptop`. Each has a textured mesh (`.obj`, `.urdf`, texture image). These are placed as markers in the scene.

---

## Running the Simulation

### Mode 1: Python controller in Docker (simplest)
```bash
docker/ctl.sh up           # build + start simulation container (needs X11 + DISPLAY)
docker/ctl.sh exec         # shell into container
python scripts/controller.py --steps 15000 --seed 0
# Useful flags: --no_render_camera, --steps N, --seed N
```

### Mode 2: Full ROS 2 mode in Docker
```bash
docker/ctl.sh up           # start simulation container
docker/ctl.sh ros2-build   # build ROS 2 Jazzy layer (once)
docker/ctl.sh ros2-up      # start ROS 2 container

# Terminal 1 (sim container):
docker/ctl.sh exec
python ros2_isaac_bridge/sim_side/isaac_controller.py

# Terminal 2 (ROS2 container):
docker/ctl.sh ros2-exec
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
```

### Mode 3: Local Isaac Gym + ROS 2 bridge in container (lower VRAM)
```bash
# One-time setup:
conda create -y -n roboschool python=3.8
conda activate roboschool
python -m pip install -e .
cd docker/isaac-gym/isaacgym/python && python -m pip install -e .
echo 'export PYTHONPATH="$HOME/workspace/roboschool_competition/docker/isaac-gym/isaacgym/python:$HOME/workspace/roboschool_competition/src:$HOME/workspace/roboschool_competition:${PYTHONPATH}"' >> ~/.bashrc

# Each new shell after conda activate:
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"

# Verify:
python -c "import isaacgym, torch, aliengo_gym; print('ok')"

# Start bridge first (it owns TCP port 5008 for depth):
docker/ctl.sh ros2-build && docker/ctl.sh ros2-up
docker/ctl.sh ros2-exec
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh

# Then run sim locally:
python scripts/controller.py --steps 15000 --seed 0
# or for ROS mode:
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

### Docker control commands
```bash
docker/ctl.sh build       # build all docker layers
docker/ctl.sh up          # start sim container with X11
docker/ctl.sh down        # stop sim container
docker/ctl.sh exec        # shell into sim container
docker/ctl.sh ros2-build  # build ROS 2 Jazzy image
docker/ctl.sh ros2-up     # start ROS 2 container
docker/ctl.sh ros2-down   # stop ROS 2 container
docker/ctl.sh ros2-exec   # shell into ROS 2 container
```

### Troubleshooting
- **`Connection refused`** — ROS 2 bridge not running; start it in Terminal 1 before sim
- **`Device count 0`** — forgot `export LD_LIBRARY_PATH` in terminal running sim
- **NVIDIA socket missing** — `sudo systemctl restart nvidia-persistenced`
- **Stuttering/jitter** — disable V-sync: `__GL_SYNC_TO_VBLANK=0 python scripts/controller.py ...`

### Timing
Control dt = 0.02s (50 Hz). Default `--steps 15000` ≈ 5 minutes of simulated time.

---

## Vision Pipeline

`scripts/vision/` contains a HOG + SVM image classifier for target objects.

```bash
# Train (from project root):
python scripts/vision/train_object_classifier.py \
  --objects_root resources/assets/objects \
  --output_dir runs/object_classifier \
  --samples_per_class 450 --image_size 128

# Inference:
python scripts/vision/classify_image.py \
  --image path/to/image.png \
  --model runs/object_classifier/object_hog_svm.yml \
  --metadata runs/object_classifier/metadata.json
```

Reads texture images from `map_Kd` in each object's `.mtl` file, generates synthetic augmented data, trains SVM. This pipeline is optional — any detection method (YOLO, CNN, etc.) is allowed.

---

## Architecture

### Participant code — where to write logic

**Primary file: `src/aliengo_competition/controllers/main_controller.py`**

The `run()` function contains two marked sections:
- `USER PARAMETERS START / END` — tune behavior parameters
- `USER CONTROL LOGIC START / END` — implement navigation and object detection

Required template (must not be removed):
```python
def get_found_object_id(current_state, current_camera_data, current_object_queue):
    return None  # replace with real detection logic

detected_object_id = get_found_object_id(state, camera_data, object_queue)
if detected_object_id is not None:
    log_found_object(detected_object_id)
```

The `object_queue` (from `env.SEQUENCE_OF_OBJECTS`) contains the ordered list of objects the robot must visit. Only report `detected_object_id` when the robot is within ~0.5 m of the object **and** the camera confirms it.

**Variables available in the user control block:**
- `sim_t` — current simulation time (seconds)
- `measured_vx`, `measured_vy`, `measured_wz` — body velocity
- `relative_dof_pos`, `dof_vel` — joint state (12 DOF)
- `base_ang_vel` — IMU angular velocity `(wx, wy, wz)`
- `camera_data` — dict with `"image"` (640×360 RGB uint8) and `"depth"` (848×480 float32 meters)
- `object_queue` — list of object IDs to visit in order

### Robot interface layer

`AliengoRobotInterface` (abstract base in `src/aliengo_competition/robot_interface/base.py`) defines the participant-facing API:
- `set_speed(vx, vy, vw)` — velocity command (fed to the RL locomotion policy)
- `step()` — advance one simulation tick
- `get_state() -> RobotState` — full sensor readout
- `get_camera()` — returns `{"image": np.ndarray, "depth": np.ndarray}` or `None`
- `is_fallen() -> bool` / `reset()` / `stop()`

`SimAliengoRobot` (`robot_interface/sim.py`) implements this over Isaac Gym. Each `step()` runs the RL policy internally and applies the participant's velocity command as the high-level goal. **Participants never touch the low-level controller.**

The locomotion policy uses a fixed trot gait (frequency=3.0 Hz, phase=0.5, duration=0.5). Only `vx`, `vy`, `vw` are participant-controllable; all other gait parameters are locked.

### RobotState fields (`robot_interface/types.py`)

| Property | Type | Description |
|----------|------|-------------|
| `state.step_index` | `int` | Current step count |
| `state.sim_time_s` | `float` | Simulation time (seconds) |
| `state.dt` | `float` | Control timestep (0.02s) |
| `state.q` / `state.joints.positions` | `np.ndarray(12,)` | Joint positions (relative to default pose) |
| `state.q_dot` / `state.joints.velocities` | `np.ndarray(12,)` | Joint velocities |
| `state.joints.names` | `tuple[str]` | 12 joint names |
| `state.vx`, `state.vy` | `float` | Body linear velocity (x forward, y left) |
| `state.wz` | `float` | Body yaw rate |
| `state.base_linear_velocity_xyz` | `np.ndarray(3,)` | Full body linear velocity |
| `state.imu.angular_velocity_xyz` | `np.ndarray(3,)` | IMU angular velocity |
| `state.imu.wx`, `.wy`, `.wz` | `float` | Individual IMU axes |
| `state.camera.rgb` / `.image` | `np.ndarray` or `None` | RGB image (640×360) |
| `state.camera.depth` | `np.ndarray` or `None` | Depth image (848×480, meters) |

### Execution flow

```
scripts/controller.py
  → factory.make_robot_interface(headless, seed)
    → scripts/play.py::load_env() — loads VelocityTrackingEasyEnv + HistoryWrapper + TorchScript policy
    → SimAliengoRobot(env, policy)
  → main_controller.run(robot, steps, ...)
    → control loop: get_state() → [USER LOGIC] → set_speed(vx, vy, vw) → step()
```

### Environment / policy stack

`scripts/play.py::load_env()` (also mirrored in `factory.py` and `ros2_isaac_bridge/sim_side/isaac_controller.py`):
1. Loads a trained run from `runs/gait-conditioned-agility/aliengo-v0/train/`
2. Creates `VelocityTrackingEasyEnv` (Isaac Gym env defined in `aliengo_gym/`)
3. Wraps it with `HistoryWrapper` (stacks last 15 observations for the adaptation module)
4. Loads TorchScript checkpoints: `body_latest.jit` + `adaptation_module_latest.jit`

Domain randomization is disabled at evaluation time. The env is configured for single-robot, single-terrain-cell runs with the front camera enabled.

### Camera specs
- **RGB:** 640×360, 70° horizontal FOV, attached to "trunk", forward-facing
- **Depth:** 848×480, 86° horizontal FOV, max range ~4.0 m (`DEFAULT_CAMERA_DEPTH_MAX_M`)

### ROS 2 bridge socket protocol

`ros2_isaac_bridge/sim_side/isaac_controller.py` ↔ `ros2_ws/src/ros2_bridge_pkg/bridge_node.py` via `SimBridgeClient`:

| Port | Protocol | Direction | Data |
|------|----------|-----------|------|
| 5005 | UDP | ROS → sim | `vx`, `vy`, `wz` commands |
| 5006 | UDP | sim → ROS | body velocity |
| 5007 | UDP | sim → ROS | RGB image (JPEG) |
| 5008 | TCP | sim → ROS | depth image (binary float32) |
| 5009 | UDP | sim → ROS | joint states |
| 5010 | UDP | sim → ROS | IMU |

Published ROS topics: `/aliengo/base_velocity`, `/aliengo/camera/color/image_raw`, `/aliengo/camera/depth/image_raw`, `/aliengo/joint_states`, `/aliengo/imu`
Subscribed: `/cmd_vel`

ROS 2 is **optional** — all sensor data is also accessible directly via `get_state()` / `get_camera()` in Python mode.

### Logging

`CompetitionRunLogger` (`src/aliengo_competition/common/run_logger.py`) writes to `logs/<timestamp>/log_seed_<N>.txt` (under `MINI_GYM_ROOT_DIR`, which is `aliengo_gym/`). The file format is:
```
seed=N
SEQUENCE_OF_OBJECTS = [...]
object <id>: cell=(x,y), world=(x,y)
detected_objects = {}

t,x,y,yaw
<CSV rows>
```
`log_detected_object_at_time(object_id, sim_t)` rewrites the `detected_objects` block in-place; calling it more than once for the same `object_id` is a no-op.

### Key packages

- **`aliengo_gym/`** — Isaac Gym environment: `VelocityTrackingEasyEnv` (in `envs/aliengo/velocity_tracking/`), `LeggedRobot` base (in `envs/base/`), `HistoryWrapper`, terrain generation
- **`aliengo_gym_learn/`** — Training infrastructure (PPO, PPO-CSE). Not needed for competition runs.
- **Dependencies** (Python 3.8): `isaacgym`, `torch` (1.10), `numpy>=1.19.5,<1.24`, `opencv-python`, `params-proto==2.10.5`, `gym>=0.14.0`
