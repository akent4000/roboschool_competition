"""
competition_no_dashboard.launch.py — запускает bridge_node + controller_node без дашборда.

Использование:
    ros2 launch ros2_bridge_pkg competition_no_dashboard.launch.py
    ros2 launch ros2_bridge_pkg competition_no_dashboard.launch.py log_level:=debug
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ------------------------------------------------------------------ args
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        choices=["debug", "info", "warn", "error", "fatal"],
        description="ROS 2 log level for both nodes",
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

    return LaunchDescription([
        log_level_arg,
        bridge_node,
        controller_node,
    ])
