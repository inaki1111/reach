#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from builtin_interfaces.msg import Time as RosTime
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    WorkspaceParameters
)
from moveit_msgs.srv import GetPlanningScene
from tf2_ros import TransformListener, Buffer

class UR5eMoveItClient(Node):
    def __init__(self):
        super().__init__('ur5e_moveit_client')

        # MoveIt action client
        self.action_client = ActionClient(self, MoveGroup, '/move_action')
        self.current_goal_handle = None
        # TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Retrieve joint names
        self.joint_names = self.get_joint_names()

    def get_joint_names(self):
        client = self.create_client(GetPlanningScene, 'get_planning_scene')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_planning_scene service...')
        req = GetPlanningScene.Request()
        req.components.components = req.components.ROBOT_STATE
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res:
            return res.scene.robot_state.joint_state.name
        else:
            self.get_logger().error('Failed to call get_planning_scene')
            return []

    def __prepare_goal_msg(self, pose: list) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request.group_name = "arm"

        # Workspace (optional)
        wp = WorkspaceParameters()
        wp.header.frame_id = "base_link"
        wp.min_corner.x, wp.min_corner.y, wp.min_corner.z = -1.0, -1.0, 0.1
        wp.max_corner.x, wp.max_corner.y, wp.max_corner.z =  1.0,  1.0, 1.0
        goal.request.workspace_parameters = wp

        # Pose target
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        x, y, z, qx, qy, qz, qw = pose
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name = "wrist_3_link"
        pc.target_point_offset.x = pc.target_point_offset.y = pc.target_point_offset.z = 0.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(ps.pose)

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = "wrist_3_link"
        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01

        cons = Constraints()
        cons.position_constraints.append(pc)
        cons.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(cons)

        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.num_planning_attempts = 5
        return goal

    def move_to_pose(self, pose: list) -> GoalStatus:
        goal = self.__prepare_goal_msg(pose)
        self.action_client.wait_for_server()
        send_fut = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        handle = send_fut.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected')
            return GoalStatus.STATUS_ABORTED
        get_res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_res)
        res = get_res.result()
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded')
            return GoalStatus.STATUS_SUCCEEDED
        else:
            self.get_logger().warn(f'Goal failed: {res.status}')
            return GoalStatus.STATUS_ABORTED

    def get_link_position(self, link_name: str):
        """
        Retrieves the latest transform base_link → link_name.
        link_name must be one of the frames in /tf, e.g. 'wrist_3_link'.
        """
        try:
            zero = RosTime()  # 0: request latest
            # wait up to 1s
            if not self.tf_buffer.can_transform(
                    'base_link', link_name, zero,
                    timeout=Duration(seconds=1.0)):
                raise RuntimeError(f"Timeout waiting for base_link→{link_name}")

            tf = self.tf_buffer.lookup_transform('base_link', link_name, zero)
            t = tf.transform.translation
            r = tf.transform.rotation
            return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)

        except Exception as e:
            self.get_logger().error(f'Failed to get transform for {link_name}: {e}')
            return None, None
