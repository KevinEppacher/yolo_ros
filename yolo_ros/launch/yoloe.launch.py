#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_name = 'yolo_ros'

    # config files
    yoloe_config_path = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'yoloe_config.yaml'
    )
    semantic_pcl_config_path = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'semantic_pointcloud.yaml'
    )

    # YOLOE lifecycle node
    yoloe_node = Node(
        package='yolo_ros',
        executable='yolo_ros',          # matches entry point in setup.py
        name='yolo_wrapper',
        namespace='yoloe',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': True},
            yoloe_config_path
        ]
    )

    # Semantic pointcloud node
    semantic_pcl_node = Node(
        package='yolo_ros',
        executable='yolo_semantic_pointcloud',  # matches entry point in setup.py
        name='semantic_pointcloud',
        namespace='yoloe',
        output='screen',
        # arguments=['--ros-args', '--log-level', 'debug'],
        emulate_tty=True,
        parameters=[
            {'use_sim_time': True},
            semantic_pcl_config_path
        ]
    )

    ld = LaunchDescription()
    ld.add_action(yoloe_node)
    ld.add_action(semantic_pcl_node)
    return ld
