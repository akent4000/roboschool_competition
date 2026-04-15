# Aliengo Competition

Репозиторий для соревнования по управлению Aliengo в Isaac Gym.

Здесь есть два способа работы:

1. Python-контроллер - вариант для локального запуска и проверки логики.
2. ROS 2 контейнеры - предпочтительный вариант для работы через ROS-интерфейс и отладки по топикам.

## Структура проекта

- `src/aliengo_competition/controllers/main_controller.py` - основной файл логики участника.
- `scripts/controller.py` - точка входа для Python-режима.
- `ros2_isaac_bridge/` - интеграция Isaac Gym и ROS 2 Jazzy.
- `docker/` - Dockerfile и `docker compose` для обоих режимов.
- `docker/ctl.sh` - скрипт-оболочка для сборки и запуска контейнеров.

## Как читать решение

Логика участника находится в `main_controller.py` и делится на два блока:

- `USER PARAMETERS START / END` - параметры поведения.
- `USER CONTROL LOGIC START / END` - основная логика шага.

Внутри логики доступны:

- положения и скорости суставов: `state.q`, `state.q_dot`
- имена суставов: `state.joints.name`
- линейная и угловая скорость корпуса: `state.vx`, `state.vy`, `state.wz`
- полный вектор линейной скорости: `state.base_linear_velocity_xyz`
- полный вектор угловой скорости: `state.base_angular_velocity_xyz`
- IMU: `state.imu.angular_velocity_xyz`
- камера: `state.camera.rgb`, `state.camera.depth`
- очередь объектов: `object_queue`

Для логирования найденного объекта предусмотрен обязательный шаблон:

- `get_found_object_id(...)` - ваша логика поиска объекта
- `log_found_object(...)` - запись найденного объекта в судейский лог

Шаблон не следует удалять, его необходимо заполнить собственной логикой.

## ROS-архитектура

ROS-режим разбит на два контейнера:

- `aliengo-competition` - Isaac Gym и симуляция
- `ros2-jazzy` - ROS 2 Jazzy, bridge node, `rqt`, `rviz2` и утилиты

Схема обмена:

```text
ROS 2 node /cmd_vel
        |
        v
bridge_node.py  <---->  SimBridgeClient
        ^                      ^
        |                      |
        |                  isaac_controller.py
        |
   ROS topics
```

Используемые порты моста:

- `5005` - команды `vx`, `vy`, `wz` из ROS в симуляцию
- `5006` - текущая скорость корпуса
- `5007` - RGB изображение
- `5008` - изображение глубины
- `5009` - суставы
- `5010` - IMU

ROS bridge публикует топики:

- `/aliengo/base_velocity`
- `/aliengo/camera/color/image_raw`
- `/aliengo/camera/depth/image_raw`
- `/aliengo/joint_states`
- `/aliengo/imu`

И принимает:

- `/cmd_vel`

## Подготовка окружения

Нужны:

- Docker
- Docker Compose
- NVIDIA драйверы и `nvidia-container-toolkit`
- `xhost` для X11, если нужен просмотр окон

Если графический интерфейс запускается через Docker, хост должен разрешить доступ к X-серверу. Скрипт `docker/ctl.sh` выполняет это автоматически, однако переменная `DISPLAY` должна быть задана.

## Запуск Python-контроллера

Этот режим используется для проверки логики без ROS.

### 1. Собрать и запустить контейнер симуляции

```bash
docker/ctl.sh up
```

### 2. Открыть терминал в контейнере

В новом терминале:

```bash
docker/ctl.sh exec
```

Это откроет shell внутри уже запущенного `aliengo-competition` контейнера.

### 3. Запустить Python-контроллер

```bash
python scripts/controller.py --steps 15000 --seed 0
```

Дополнительно:

- `--no_render_camera` - отключить окно камеры
- `--steps` - число шагов управления
- `--seed` - seed для воспроизводимости

## Запуск ROS 2 режима

Этот режим рекомендуется для работы через ROS и проверки взаимодействия через топики.

### 1. Собрать базовый контейнер симуляции

```bash
docker/ctl.sh up
```

### 2. Собрать ROS 2 слой

```bash
docker/ctl.sh ros2-build
```

### 3. Поднять ROS 2 контейнер

```bash
docker/ctl.sh ros2-up
```

### 4. Открыть первый терминал для симуляции

В отдельном терминале:

```bash
docker/ctl.sh exec
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

Этот процесс:

- читает команды от ROS-моста
- шагает симуляцию
- отправляет RGB, depth, суставы, скорость и IMU в мост

### 5. Открыть второй терминал для ROS-моста

В ещё одном терминале:

```bash
docker/ctl.sh ros2-exec
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
```

Этот процесс:

- принимает данные из симуляции
- публикует их в ROS 2 топики
- пересылает `cmd_vel` обратно в симуляцию

### 6. Открыть дополнительные терминалы

Любой новый терминал открывается так же:

```bash
docker/ctl.sh exec
```

или для ROS:

```bash
docker/ctl.sh ros2-exec
```

Полезно для:

- `ros2 topic list`
- `ros2 topic echo /aliengo/base_velocity`
- `rqt_graph`
- отладочных скриптов

## Как контейнеры общаются между собой

Связь построена через host network и UDP/TCP сокеты.

1. ROS 2 нода публикует команду `cmd_vel`.
2. `SimBridgeClient` в симуляции получает её по UDP на порту `5005`.
3. Симуляция отправляет обратно:
   - состояние корпуса
   - RGB кадр
   - depth
   - суставы
   - IMU
4. `BridgeNode` в ROS 2 контейнере принимает эти данные и публикует их как ROS-топики.

Это означает, что контейнеры не взаимодействуют друг с другом как обычные ROS-ноды. Обмен реализован через мост и сетевые сокеты.

## Как устроен Python-вариант

Python-вариант живёт в:

- `scripts/controller.py`
- `src/aliengo_competition/controllers/main_controller.py`

Что делает `scripts/controller.py`:

- поднимает аргументы командной строки
- создаёт интерфейс робота
- вызывает `run(...)`

Что делает `main_controller.py`:

- получает `RobotState`
- читает измерения
- формирует команду `vx`, `vy`, `vw`
- логирует шаги и найденные объекты

## Как устроен ROS-вариант

ROS-вариант состоит из трёх частей:

1. Симуляция: `ros2_isaac_bridge/sim_side/isaac_controller.py`
2. Bridge client: `ros2_isaac_bridge/sim_side/sim_bridge_client.py`
3. ROS node: `ros2_isaac_bridge/ros2_ws/src/ros2_bridge_pkg/ros2_bridge_pkg/bridge_node.py`

### Что делает `isaac_controller.py`

- получает команду от ROS
- применяет её к симуляции
- снимает телеметрию
- отправляет её в bridge

### Что делает `bridge_node.py`

- принимает сокеты от симуляции
- публикует данные в ROS-топики
- слушает `/cmd_vel`
- пересылает скорость обратно в симуляцию

### Что делает `sim_bridge_client.py`

- реализует сетевой клиент для передачи:
  - `vx`, `vy`, `wz`
  - `state`
  - `rgb`
  - `depth`
  - `joint_states`
  - `imu`

## Fork и работа в своей копии

Если вы работаете из своей копии репозитория:

### Вариант через GitHub fork

1. Откройте репозиторий на GitHub.
2. Нажмите `Fork`.
3. Клонируйте свой fork:

```bash
git clone git@github.com:<your-user>/roboschool_competition.git
cd roboschool_competition
```

4. Добавьте upstream:

```bash
git remote add upstream git@github.com:<original-owner>/roboschool_competition.git
git fetch upstream
```

### Рекомендуемый рабочий процесс

- создавайте отдельную ветку под задачу
- не коммитьте напрямую в `main`
- периодически подтягивайте изменения из upstream

Пример:

```bash
git checkout -b my-competition-solution
git push -u origin my-competition-solution
```

## Полезные команды

```bash
docker/ctl.sh build
docker/ctl.sh up
docker/ctl.sh exec
docker/ctl.sh down

docker/ctl.sh ros2-build
docker/ctl.sh ros2-up
docker/ctl.sh ros2-exec
docker/ctl.sh ros2-down
```

## Где редактировать решение

- Python-режим: `src/aliengo_competition/controllers/main_controller.py`
- ROS-режим: `ros2_isaac_bridge/sim_side/isaac_controller.py`