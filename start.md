# Запуск ROS 2 стека с веб-дашбордом

## Предварительно

Нужны: Docker, NVIDIA GPU + драйверы, X11 (`xhost +local:root`).

## Шаг 1 — Собрать ROS 2 образ (один раз)

```bash
docker/ctl.sh ros2-build
```

## Шаг 2 — Запустить ROS 2 контейнер

```bash
docker/ctl.sh ros2-up
docker/ctl.sh ros2-exec
```

## Шаг 3 — Собрать и запустить ноды (внутри контейнера)

```bash
cd /workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws
colcon build
set +u
source install/setup.bash
ros2 launch ros2_bridge_pkg competition.launch.py
```

Это запустит три ноды:
- **bridge_node** — мост Isaac Gym -> ROS 2 топики
- **controller_node** — SLAM + YOLO + навигация
- **dashboard_node** — веб-дашборд на http://localhost:8080

## Шаг 4 — Запустить симулятор (отдельный терминал)

**Вариант A** — в sim-контейнере:
```bash
docker/ctl.sh up
docker/ctl.sh exec
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

**Вариант B** — локально (conda):
```bash
conda activate roboschool
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

## Дашборд

Открыть в браузере: **http://localhost:8080**

## После изменения кода

Пересобрать пакет (внутри ROS 2 контейнера):
```bash
cd /workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws
colcon build
```

## Полезные команды

| Команда | Описание |
|---------|----------|
| `docker/ctl.sh ros2-build` | Пересобрать Docker-образ |
| `docker/ctl.sh ros2-up` | Запустить контейнер |
| `docker/ctl.sh ros2-down` | Остановить контейнер |
| `docker/ctl.sh ros2-exec` | Зайти в контейнер |
| `ros2 topic list` | Список активных топиков |
| `ros2 topic hz /aliengo/camera/color/image_raw` | Частота публикации топика |
