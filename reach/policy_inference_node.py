#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Point, Pose
import numpy as np
import onnxruntime as ort
from importlib.resources import files


class PolicyActionSplitter(Node):
    def __init__(self):
        super().__init__('policy_action_splitter')

        # Observation structure
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.object_position = np.zeros(3)
        self.target_object_position = np.zeros(7)
        self.last_action = np.zeros(8)  # updated internally each step

        # Load the ONNX model
        model_path = str(files("reach").joinpath("policy.onnx"))
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Subscribers for processed inputs
        self.create_subscription(Float32MultiArray, '/processed_joint_pos', self.joint_pos_cb, 10)
        self.create_subscription(Float32MultiArray, '/processed_joint_vel', self.joint_vel_cb, 10)
        self.create_subscription(Point, '/object_position', self.obj_pos_cb, 10)
        self.create_subscription(Pose, '/target_object_position', self.target_pos_cb, 10)

        # Publishers for actions
        self.arm_action_pub = self.create_publisher(Float32MultiArray, '/arm_action', 10)
        self.gripper_action_pub = self.create_publisher(Float32, '/gripper_action', 10)

        # Timer for inference loop
        self.timer = self.create_timer(0.1, self.compute_action)

    # --- Callbacks ---
    def joint_pos_cb(self, msg: Float32MultiArray):
        self.joint_pos = np.array(msg.data[:12])

    def joint_vel_cb(self, msg: Float32MultiArray):
        self.joint_vel = np.array(msg.data[:12])

    def obj_pos_cb(self, msg: Point):
        self.object_position = np.array([msg.x, msg.y, msg.z])

    def target_pos_cb(self, msg: Pose):
        self.target_object_position = np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        ])

    # --- Main loop ---
    def compute_action(self):
        obs = np.concatenate([
            self.joint_pos,               # (12,)
            self.joint_vel,              # (12,)
            self.object_position,        # (3,)
            self.target_object_position, # (7,)
            self.last_action             # (8,)
        ])

        if obs.shape != (42,):
            self.get_logger().warn(f"Incomplete observation vector: {obs.shape}")
            return

        output = self.session.run(None, {self.input_name: obs.astype(np.float32).reshape(1, -1)})[0]
        action = output[0]  # Expected shape: (8,)

        # Publish current action
        arm_action = Float32MultiArray()
        arm_action.data = action[:7].tolist()
        self.arm_action_pub.publish(arm_action)

        gripper_action = Float32()
        gripper_action.data = float(action[7])
        self.gripper_action_pub.publish(gripper_action)

        # Update last_action for next step
        self.last_action = np.concatenate([action[:7], [float(action[7])]])



def main(args=None):
    rclpy.init(args=args)
    node = PolicyActionSplitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
