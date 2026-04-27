"""ROS 2 Humble launch file for the Vicon motion capture bridge.

Equivalent of the ROS 1 vicon.launch file.
Uses vicon_receiver (ROS 2 replacement for vicon_bridge).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # IP:port of the Vicon Windows PC (e.g. 192.168.94.81:801)
    datastream_arg = DeclareLaunchArgument(
        "datastream_hostport",
        default_value="127.0.0.1:801",
        description="IP and port of the Vicon datastream server (host:port)",
    )

    vicon_client = Node(
        package="vicon_receiver",
        executable="vicon_client",
        name="vicon",
        output="screen",
        parameters=[{
            "datastream_hostport": LaunchConfiguration("datastream_hostport"),
            "namespace": "vicon",
            "buffer_size": 200,
        }],
    )

    return LaunchDescription([
        datastream_arg,
        vicon_client,
    ])
