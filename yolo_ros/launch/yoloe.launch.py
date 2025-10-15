from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Flag to enable use_sim_time'
    )

    # Get the launch configuration for use_sim_time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # config files
    yoloe_config_path = os.path.join(
        get_package_share_directory('yolo_ros'),
        'config',
        'yoloe_config.yaml'
    )
    semantic_pcl_config_path = os.path.join(
        get_package_share_directory('yolo_ros'),
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

    lcm = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_detection',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': [
                '/yoloe/yolo_wrapper',
                # '/value_map/value_map'
            ]
        }]
    )

    ld = LaunchDescription()
    ld.add_action(sim_time_arg)
    ld.add_action(yoloe_node)
    ld.add_action(semantic_pcl_node)
    ld.add_action(lcm)
    return ld
