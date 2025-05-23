from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    return LaunchDescription([
        # Policy inference node
        launch_ros.actions.Node(
            package='reach',
            executable='policy_inference_node',
            name='policy_inference_node',
            output='screen'
        ),
        # Goal position publisher node
        launch_ros.actions.Node(
            package='reach',
            executable='goal_position_node',
            name='goal_position_node',
            output='screen'
        ),
        # Joint states preprocessor node
        launch_ros.actions.Node(
            package='reach',
            executable='joint_states_node',
            name='joint_states_node',
            output='screen'
        ),
        # MoveIt integration node
        """launch_ros.actions.Node(
            package='reach',
            executable='moveit_node',
            name='moveit_node',
            output='screen'
        ),"""

    ])
