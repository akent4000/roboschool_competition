"""
competition.launch.py — запускает bridge_node + controller_node одной командой.

Использование:
    ros2 launch ros2_bridge_pkg competition.launch.py
    ros2 launch ros2_bridge_pkg competition.launch.py enable_dashboard:=true
    ros2 launch ros2_bridge_pkg competition.launch.py log_level:=debug
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ------------------------------------------------------------------ args
    enable_dashboard_arg = DeclareLaunchArgument(
        "enable_dashboard",
        default_value="false",
        description="Show the OpenCV navigation dashboard (needs DISPLAY)",
    )

    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        choices=["debug", "info", "warn", "error", "fatal"],
        description="ROS 2 log level for both nodes",
    )

    dashboard_port_arg = DeclareLaunchArgument(
        "dashboard_port",
        default_value="8080",
        description="HTTP port for the web dashboard",
    )

    # Set ENABLE_DASHBOARD env var based on launch argument
    # NavigationController checks: os.environ.get("ENABLE_DASHBOARD") == "1"
    set_dashboard_env = SetEnvironmentVariable(
        name="ENABLE_DASHBOARD",
        value=LaunchConfiguration("enable_dashboard"),
    )

    # ------------------------------------------------------------------ nodes
    bridge_node = Node(
        package="ros2_bridge_pkg",
        executable="bridge_node",
        name="bridge_node",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    controller_node = Node(
        package="ros2_bridge_pkg",
        executable="controller_node",
        name="controller",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    dashboard_node = Node(
        package="ros2_bridge_pkg",
        executable="dashboard_node",
        name="dashboard",
        output="screen",
        parameters=[{"port": LaunchConfiguration("dashboard_port")}],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([
        enable_dashboard_arg,
        log_level_arg,
        dashboard_port_arg,
        set_dashboard_env,
        bridge_node,
        controller_node,
        dashboard_node,
    ])
