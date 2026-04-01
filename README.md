# AlienGo Competition Repo

Minimal RL competition repository for **Unitree AlienGo** on **flat terrain only**.

What is included:
- one task: `aliengo_flat`
- one robot: AlienGo
- one flat-ground RL pipeline
- one high-level competition controller template
- camera API stub for now

## Repository Layout

- `docker/base/` - copied shared CUDA/Ubuntu base layer
- `docker/isaac-gym/` - copied shared Isaac Gym layer
- `src/aliengo_competition/common/` - minimal Isaac Gym / legged-gym style runtime
- `src/aliengo_competition/envs/aliengo_flat.py` - the only task env
- `src/aliengo_competition/robot_interface/` - sim adapter
- `src/aliengo_competition/controllers/main_controller.py` - participant entry point
- `configs/aliengo_flat.py` - task and PPO config
- `scripts/train.py` - train the policy
- `scripts/plot_training.py` - plot a training curve from `outputs.txt` or TensorBoard scalars
- `scripts/play.py` - run the trained policy in sim mode
- `scripts/controller.py` - run the competition controller
- `models/` - training logs and checkpoints
- `resources/robots/aliengo/` - AlienGo URDF and meshes only

## Build Docker

The shared Docker layers live inside this repo now.
Use the local control script from the repository root:

```bash
./docker/ctl.sh build
```

Or, to build and start in one step:

```bash
./docker/ctl.sh up
```

## Quick Commands

```bash
./docker/ctl.sh build
./docker/ctl.sh up
./docker/ctl.sh train
./docker/ctl.sh play
./docker/ctl.sh controller
./docker/ctl.sh viz-up
./docker/ctl.sh enter
./docker/ctl.sh viz-enter
./docker/ctl.sh down
./docker/ctl.sh logs   # latest training outputs.log, fallback to container logs
python scripts/train.py --task aliengo_flat --headless
python scripts/play.py --task aliengo_flat --headless --vx 0.5 --vy 0.0 --vw 0.0 --pitch 0.0
python scripts/controller.py --task aliengo_flat --mode sim --headless
python scripts/plot_training.py
```

## Run Container

```bash
./docker/ctl.sh enter
```

## Visualization

To use the Isaac Gym viewer from inside the container on a Linux desktop, allow local X11 access first:

```bash
xhost +local:root
```

Then start the visualization-enabled container:

```bash
./docker/ctl.sh viz-up
./docker/ctl.sh viz-enter
```

When you are done, you can revoke the access with:

```bash
xhost -local:root
```

## Train

Inside the container:

```bash
python scripts/train.py --task aliengo_flat --headless
```

Training writes a run-local `outputs.txt` file into the latest folder under `models/aliengo_flat/<run_name>/`.
The same folder also contains checkpoints and the TensorBoard event file.

To build a plot from the latest run:

```bash
python scripts/plot_training.py
```

Or point it at a specific run:

```bash
python scripts/plot_training.py --run_dir models/aliengo_flat/<run_name> --output models/aliengo_flat/<run_name>/training_plot.png
```

## Play

Play runs the trained policy and lets you set only:
- `vx`
- `vy`
- `vw`
- body pitch

The rest of the gait command vector is kept on competition defaults internally.
During training, the env uses a short observation history and a small curriculum over the variable commands, while the remaining gait parameters stay fixed at defaults.

Example:

```bash
python scripts/play.py --task aliengo_flat --headless --vx 0.5 --vy 0.0 --vw 0.0 --pitch 0.0
```

## Controller

Participants should edit only:

- `src/aliengo_competition/controllers/main_controller.py`

That file contains the explicit:

```python
# PUT YOUR CODE HERE
```

Run the controller in sim mode:

```bash
python scripts/controller.py --task aliengo_flat --mode sim --headless
```

## Available Robot API

High-level controller code should use only:

- `set_speed(vx, vy, vw)`
- `set_body_pitch(pitch)`
- `stop()`
- `reset()`
- `step()`
- `get_camera()`
- `get_observation()`
- `is_fallen()`

`get_camera()` is currently a stub and returns `None`.

## What Not To Edit

- `src/aliengo_competition/common/`
- `src/aliengo_competition/envs/`
- `src/aliengo_competition/robot_interface/`
- `configs/aliengo_flat.py`
- `scripts/train.py`
- `scripts/play.py`
- `scripts/controller.py`
- `docker/base/`
- `docker/isaac-gym/`

## Notes

- Terrain is flat only.
- There is only one task.
- Rough terrain is intentionally excluded.
