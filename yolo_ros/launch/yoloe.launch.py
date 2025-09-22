#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'mpc_local_planner'

    yoloe_config_path = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'yoloe_config.yaml'
    )

    yoloe_node = Node(
        package='yolo_ros',
        executable='yoloe_lifecycle_node',
        name='yoloe_node',
        output='screen',
        namespace='yolo_ros',
        parameters=[
            {'use_sim_time': True},
            yoloe_config_path
        ]
    )


    ld = LaunchDescription()
    ld.add_action(yoloe_node)
    return ld