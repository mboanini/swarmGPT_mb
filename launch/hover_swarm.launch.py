"""ROS 2 Humble launch file for the SwarmGPT hover swarm demo.

Equivalent of the ROS 1 hover_swarm.launch file.
Launches: crazyflie_server, joy, rviz2.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_CRAZYFLIES_YAML = str(Path(__file__).resolve().parents[1] / "config" / "crazyflies.yaml")


def generate_launch_description():
    crazyflies_yaml = _CRAZYFLIES_YAML

    joy_dev_arg = DeclareLaunchArgument(
        "joy_dev",
        default_value="/dev/input/js0",
        description="Joystick device path",
    )

    crazyflie_server = Node(
        package="crazyflie",
        executable="crazyflie_server",
        name="crazyflie_server",
        output="screen",
        parameters=[crazyflies_yaml],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy",
        output="screen",
        parameters=[{"dev": LaunchConfiguration("joy_dev")}],
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
    )

    return LaunchDescription([
        joy_dev_arg,
        crazyflie_server,
        joy_node,
        rviz2,
    ])
