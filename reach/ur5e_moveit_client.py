#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient, CancelResponse
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint, WorkspaceParameters
from moveit_msgs.srv import GetPlanningScene
from tf2_ros import TransformListener, Buffer
#not installed in the docker container
#from tf_transformations import euler_from_quaternion


# Use https://github.com/AndrejOrsula/pymoveit2/blob/master/pymoveit2/moveit2.py
# as a reference to implement the UR5eMoveItClient class

class UR5eMoveItClient(Node):
    """ UR5eMoveItClient is a ROS 2 node that interfaces with MoveIt 2 to control a UR5e robotic arm.
        It abstracts the MoveIt 2 action interface and provides methods to move the robot to specified
        joint positions and poses. It also provides methods to retrieve the current joint positions 
        and the position of a specified link in the robot.
    """	
    def __init__(self):
        super().__init__('ur5e_moveit_client')

        # action client for MoveIt 2
        self.action_client = ActionClient(self, MoveGroup, '/move_action')
        self.current_goal_handle = None
        self.joint_names = self.get_joint_names()

        # TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def get_joint_names(self):
        """
        Retrieves the joint names from the MoveIt interface.

        Returns:
            List[str]: The list of joint names.
        """
        client = self.create_client(GetPlanningScene, 'get_planning_scene')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = GetPlanningScene.Request()
        request.components.components = request.components.ROBOT_STATE

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response:
            return response.scene.robot_state.joint_state.name
        else:
            self.get_logger().error('Failed to call service get_planning_scene')
            return []

    def __prepare_goal_msg(self, pose : list) -> MoveGroup.Goal:
        """
        Prepares a MoveGroup.Goal message with the specified target position.

        Args:
            pose (list): A list of 7 floats representing the target position and orientation quaternion.

                    Returns:
            MoveGroup.Goal: The prepared goal message with the specified target position and constraints.
        """
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"

        # Define the workspace parameters
        # does not seems to have any effect
        # Has to be tuned 
        workspace_parameters = WorkspaceParameters()
        workspace_parameters.header.frame_id = "base_link"
        workspace_parameters.min_corner.x = -1.0
        workspace_parameters.min_corner.y = -1.0
        workspace_parameters.min_corner.z = .1
        workspace_parameters.max_corner.x = 1.0
        workspace_parameters.max_corner.y = 1.0
        workspace_parameters.max_corner.z = 1.0
        goal_msg.request.workspace_parameters = workspace_parameters

        # define the target pose
        pose_target = PoseStamped()
        pose_target.header.frame_id = "base_link"
        x, y, z, qx, qy, qz, qw = pose
        pose_target.pose.position.x = x
        pose_target.pose.position.y = y
        pose_target.pose.position.z = z
        pose_target.pose.orientation.x = qx
        pose_target.pose.orientation.y = qy
        pose_target.pose.orientation.z = qz
        pose_target.pose.orientation.w = qw

        # Create a position constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "base_link"
        position_constraint.link_name = "tool0"  # Name of the end-effector link
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Define the volume around the target position
        constraint_region = SolidPrimitive()
        constraint_region.type = SolidPrimitive.BOX
        constraint_region.dimensions = [0.01, 0.01, 0.01]  # Small box around the target position
        position_constraint.constraint_region.primitives.append(constraint_region)
        position_constraint.constraint_region.primitive_poses.append(pose_target.pose)

        orientation_constraint = OrientationConstraint()

        # Define reference frame and target link
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.link_name = "tool0"

        orientation_constraint.orientation.x = qx
        orientation_constraint.orientation.y = qy
        orientation_constraint.orientation.z = qz
        orientation_constraint.orientation.w = qw

        # Define tolerances
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01

        # Define parameterization (how to interpret the tolerance)
        #constraint.parameterization = parameterization

        # Set weight of the constraint
        #constraint.weight = weight

        # Add the position constraint to the goal message
        goal_constraints = Constraints()
        goal_constraints.orientation_constraints.append(orientation_constraint)        
        goal_constraints.position_constraints.append(position_constraint)
        goal_msg.request.goal_constraints.append(goal_constraints)

        goal_msg.request.allowed_planning_time = 10.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.num_planning_attempts = 5

        return goal_msg

    def move_to_joints(self,joint_positions: list) -> GoalStatus:
        """
        Moves the UR5e to the specified joint positions.

        Args:
            joint_positions (list): A list of 6 floats representing the joint positions.

        Returns:
            GoalStatus: The status of the goal.
        """

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.allowed_planning_time = 10.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        joint_constraints = []

        for name, position in zip(self.joint_names, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            joint_constraints.append(joint_constraint)

        goal_constraints = Constraints()
        goal_constraints.joint_constraints.extend(joint_constraints)
        goal_msg.request.goal_constraints.append(goal_constraints)

        # Send the goal message to the action server
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return GoalStatus.STATUS_ABORTED

        self.get_logger().info('Goal accepted')

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            return GoalStatus.STATUS_SUCCEEDED
        else:
            self.get_logger().info('Goal failed with status: {0}'.format(result.status))
            return GoalStatus.STATUS_ABORTED

    def move_to_pose(self, pose: list) -> GoalStatus: 
        """
        Moves the robot to the specified pose.

        This function sends a goal message to the action server to move the robot
        to the desired pose. It waits for the action server to accept the goal and
        then waits for the result of the action.

        Args:
            pose (list): The target pose for the robot. A pose is a list of 7 floats,
            where the first three floats represent the x, y, and z coordinates of the
            position, and the last four floats represent the orientation in quaternion

        Returns:
            GoalStatus: The status of the goal after execution. It returns
            `GoalStatus.STATUS_SUCCEEDED` if the goal was achieved successfully,
            otherwise it returns `GoalStatus.STATUS_ABORTED`.
        """
        goal_msg = self.__prepare_goal_msg(pose)

        # Send the goal message to the action server
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return GoalStatus.STATUS_ABORTED

        self.get_logger().info('Goal accepted')

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            return GoalStatus.STATUS_SUCCEEDED
        else:
            self.get_logger().info('Goal failed with status: {0}'.format(result.status))
            return GoalStatus.STATUS_ABORTED

    def move_to_pose_async(self, pose: list) -> GoalStatus: 
        """
        Asynchronously moves the robot to the specified pose.
        This method prepares a goal message with the given pose and sends it to the action server.
        If there is a current goal being executed, it attempts to cancel it before sending the new goal.
        A small delay is added to ensure the system processes the cancellation request.
        Args:
            pose (list): The target pose for the robot.
        Returns:
            GoalStatus: The status of the goal after being sent to the action server. It returns
                        STATUS_SUCCEEDED if the goal is accepted and STATUS_ABORTED if the goal is rejected.
        """

        goal_msg = self.__prepare_goal_msg(pose)

        # Cancel the current goal if it exists
        if self.current_goal_handle and self.current_goal_handle.status in [GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING]:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            cancel_response = cancel_future.result()
            
            #don't handle the response since it returns REJECT but still cancels the goal
            # if cancel_response == CancelResponse.ACCEPT:
            #     self.get_logger().info('Current goal canceled')
            # else:
            #     self.get_logger().info('Failed to cancel current goal')

        # Add a small delay to ensure the system processes the cancellation
        time.sleep(0.1)

        # Send the goal message to the action server
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.current_goal_handle = send_goal_future.result()

        if not self.current_goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return GoalStatus.STATUS_ABORTED

        self.get_logger().info('Goal accepted')
        return GoalStatus.STATUS_SUCCEEDED


    #get the current joint positions
    def get_joint_positions(self) -> list:
        """
        Retrieves the current joint positions of the robot.
        This method creates a client to call the 'get_planning_scene' service and 
        requests the robot's current state. It waits for the service to be available 
        and then sends the request. If the service call is successful, it returns 
        the joint positions of the robot. If the service call fails, it logs an 
        error message and returns an empty list.
        Returns:
            list: A list of joint positions if the service call is successful, 
                  otherwise an empty list.
        """

        client = self.create_client(GetPlanningScene, 'get_planning_scene')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = GetPlanningScene.Request()
        request.components.components = request.components.ROBOT_STATE

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response:
            return response.scene.robot_state.joint_state.position
        else:
            self.get_logger().error('Failed to call service get_planning_scene')
            return []

    def get_link_position(self, link_name: str):
        """
        Retrieves the position of the specified link.

        Args:
            link_name (str): The name of the link.

        Returns:
            tuple: A tuple containing the position (x, y, z) and orientation (quaternion) of the link.
        """
        try:
            # Wait for the transform to become available
            self.tf_buffer.can_transform("world", link_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            # Lookup the latest transform
            transform = self.tf_buffer.lookup_transform("world", link_name, rclpy.time.Time(seconds=0))
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            position = (translation.x, translation.y, translation.z)
            orientation = (rotation.x, rotation.y, rotation.z, rotation.w)
            return position, orientation
        except Exception as e:
            self.get_logger().error(f'Failed to get transform for {link_name}: {e}')
            return None, None