#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class JointStatePreprocessor(Node):
    def __init__(self):
        super().__init__('joint_state_preprocessor')

        # Ordered list of expected joints for model input (12 total)
        self.expected_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "robotiq_85_left_inner_knuckle_joint", "robotiq_85_left_finger_tip_joint", "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint", "robotiq_85_right_finger_tip_joint", "robotiq_85_right_knuckle_joint"
        ]

        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        self.joint_pos_pub = self.create_publisher(Float32MultiArray, '/processed_joint_pos', 10)
        self.joint_vel_pub = self.create_publisher(Float32MultiArray, '/processed_joint_vel', 10)

    def joint_state_cb(self, msg: JointState):
        # Maps from joint names to values
        joint_pos_map = dict(zip(msg.name, msg.position))
        joint_vel_map = dict(zip(msg.name, msg.velocity))

        # Initialize zero arrays
        joint_pos_vec = np.zeros(12, dtype=np.float32)
        joint_vel_vec = np.zeros(12, dtype=np.float32)

        # Get value of left_knuckle_joint if available
        left_knuckle_pos = joint_pos_map.get("robotiq_85_left_knuckle_joint", 0.0)
        left_knuckle_vel = joint_vel_map.get("robotiq_85_left_knuckle_joint", 0.0)

        # Fill vector based on expected names
        for i, joint_name in enumerate(self.expected_joint_names):
            if joint_name == "robotiq_85_right_knuckle_joint":
                # Mirror value from left
                joint_pos_vec[i] = left_knuckle_pos
                joint_vel_vec[i] = left_knuckle_vel
            elif joint_name in joint_pos_map:
                joint_pos_vec[i] = joint_pos_map[joint_name]
                joint_vel_vec[i] = joint_vel_map.get(joint_name, 0.0)  # May be missing

        # Publish processed joint states
        pos_msg = Float32MultiArray(data=joint_pos_vec.tolist())
        vel_msg = Float32MultiArray(data=joint_vel_vec.tolist())

        self.joint_pos_pub.publish(pos_msg)
        self.joint_vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePreprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
