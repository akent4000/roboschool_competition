Подытожим все наши исправления. Теперь, когда мы разобрались с путями, версиями Python и «упрямым» сокетом NVIDIA, вот твоя финальная шпаргалка по запуску.

Для полноценной работы тебе нужно **два окна терминала** (и третье — по желанию — для контроля видеокарты).

---

### Терминал №1: Мост ROS 2 (в Docker)
Этот шаг обязателен, так как симуляция не начнётся, пока не увидит активный «мост» для передачи данных.

1.  **Зайди в папку и подними контейнер:**
    ```bash
    cd ~/Desktop/roboschool_competition
    docker/ctl.sh ros2-up
    ```
2.  **Зайди внутрь запущенного контейнера:**
    ```bash
    docker/ctl.sh ros2-exec
    ```
3.  **Запусти сам узел связи (Bridge):**
    ```bash
    bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
    ```
    *Оставь это окно открытым. Оно будет транслировать данные между ROS и Isaac.*

---

### Терминал №2: Симуляция Isaac Gym (на хосте)
Здесь происходит вся магия графики и физики.

1.  **Активируй окружение:**
    ```bash
    conda activate isaac
    ```
2.  **Прокинь пути к видеокарте (критично для Device count 1):**
    ```bash
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
    ```
3.  **Перейди в проект и запускай:**
    ```bash
    cd ~/Desktop/roboschool_competition
    python scripts/teleop.py
    ```

---

### Терминал №3: Мониторинг (по желанию)
Если хочешь убедиться, что железо нагружено правильно:
```bash
watch -n 1 nvidia-smi
```

---

### Что делать, если что-то пошло не так (Short Troubleshooting):

* **Ошибка `Connection refused`:** Ты забыл запустить скрипт в **Терминале №1** или контейнер упал.
* **Ошибка `Device count 0`:** Ты забыл сделать `export LD_LIBRARY_PATH` во втором терминале.
* **Ошибка `no such file or directory` (про сокет):** Служба `nvidia-persistenced` упала. Перезапусти её:
    ```bash
    sudo systemctl restart nvidia-persistenced
    ```

### Совет по производительности:
Если робот в окне движется рывками, попробуй в **Терминале №2** добавить флаг для отключения синхронизации с монитором:
`__GL_SYNC_TO_VBLANK=0 python ros2_isaac_bridge/sim_side/isaac_controller.py`



Теперь у тебя есть чёткий план. Робот готов к первому шагу?