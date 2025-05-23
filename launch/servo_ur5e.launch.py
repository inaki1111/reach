#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_moveit = get_package_share_directory('moveit_ur_config')

    xacro_file = os.path.join(pkg_moveit, 'config', 'ur5.urdf.xacro')
    robot_desc_cmd = Command(['xacro ', xacro_file])
    robot_description = {'robot_description': ParameterValue(robot_desc_cmd, value_type=str)}

    srdf_file = os.path.join(pkg_moveit, 'config', 'ur5.srdf')
    with open(srdf_file, 'r') as f:
        srdf = f.read()
    robot_description_semantic = {'robot_description_semantic': srdf}

    servo_yaml = os.path.join(pkg_moveit, 'config', 'servo.yaml')
    ros2_controllers = os.path.join(pkg_moveit, 'config', 'ros2_controllers.yaml')
    kinematics_yaml = os.path.join(pkg_moveit, 'config', 'kinematics.yaml')
    rviz_config_file = os.path.join(pkg_moveit, 'launch', 'moveit.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='screen',
            parameters=[robot_description, ros2_controllers]
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen'
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller'],
            output='screen'
        ),
        Node(
            package='moveit_servo',
            executable='servo_node_main',
            name='servo_server',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                servo_yaml,
                kinematics_yaml
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[robot_description, robot_description_semantic]
        ),
    ])
