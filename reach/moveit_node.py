#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from .ur5e_moveit_client import UR5eMoveItClient

class ArmActionExecutor(Node):
    def __init__(self):
        super().__init__('arm_action_executor')

        # MoveIt client
        self.client = UR5eMoveItClient()

        # Subscriber to arm_action topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/arm_action',
            self.arm_action_cb,
            10
        )


    def arm_action_cb(self, msg: Float32MultiArray):
        if len(msg.data) != 7:
            self.get_logger().warn(f"Received arm_action of invalid size: {len(msg.data)}")
            return

        dx, dy, dz = msg.data[0:3]
        qx, qy, qz, qw = msg.data[3:7]

        # Get current pose
        position, orientation = self.client.get_link_position("tool0")
        if position is None or orientation is None:
            self.get_logger().warn("Could not get current end-effector pose.")
            return

        # Apply delta to position
        new_position = [
            position[0] + dx,
            position[1] + dy,
            position[2] + dz
        ]

        # Use new orientation from the model
        new_pose = new_position + [qx, qy, qz, qw]

        # Send the pose to MoveIt
        self.client.move_to_pose(new_pose)

def main(args=None):
    rclpy.init(args=args)
    node = ArmActionExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
