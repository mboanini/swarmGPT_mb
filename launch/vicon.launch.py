"""ROS 2 Humble launch file for the Vicon motion capture bridge.

Uses motion_capture_tracking (IMRCLab) as ROS 2 replacement for vicon_bridge.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "motion_capture.yaml")


def generate_launch_description():
    hostname_arg = DeclareLaunchArgument(
        "hostname",
        default_value="192.168.0.1",
        description="IP address of the Vicon PC",
    )

    mocap_node = Node(
        package="motion_capture_tracking",
        executable="motion_capture_tracking_node",
        name="motion_capture_tracking",
        output="screen",
        parameters=[
            _CONFIG,
            {"hostname": LaunchConfiguration("hostname")},
        ],
    )

    return LaunchDescription([
        hostname_arg,
        mocap_node,
    ])
