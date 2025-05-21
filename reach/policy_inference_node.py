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
        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.object_position = np.zeros(3, dtype=np.float32)
        self.target_object_position = np.zeros(7, dtype=np.float32)
        self.last_action = np.zeros(8, dtype=np.float32)  # updated internally each step

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
        self.joint_pos = np.array(msg.data[:12], dtype=np.float32)

    def joint_vel_cb(self, msg: Float32MultiArray):
        self.joint_vel = np.array(msg.data[:12], dtype=np.float32)

    def obj_pos_cb(self, msg: Point):
        self.object_position = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def target_pos_cb(self, msg: Pose):
        self.target_object_position = np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        ], dtype=np.float32)

    # --- Main loop ---
    def compute_action(self):
        # Build observation vector
        obs = np.concatenate([
            self.joint_pos,
            self.joint_vel,
            self.object_position,
            self.target_object_position,
            self.last_action
        ])

        if obs.shape != (42,):
            self.get_logger().warn(f"Incomplete observation vector: {obs.shape}")
            return

        # Run inference
        output = self.session.run(
            None,
            {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        )[0]
        action = output[0]  # shape (8,)

                # 1) Position deltas: remove scaling to test raw output
        # scale factor (currently 5cm) can be adjusted here or set to 1.0 to use raw model deltas
        scale = 1.0
        dx, dy, dz = action[0:3] * scale
        # optional clamp if needed:
        # dx, dy, dz = np.clip([dx, dy, dz], -scale, scale)

        # 2) Orientation: normalize quaternion: normalize quaternion
        raw_q = action[3:7]
        norm = np.linalg.norm(raw_q)
        if norm < 1e-6:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx, qy, qz, qw = (raw_q / norm).tolist()

        # 3) Gripper: clamp (if used later)
        gripper_val = float(np.clip(action[7], 0.0, 1.0))

        # Publish arm action
        arm_action = Float32MultiArray()
        arm_action.data = [float(dx), float(dy), float(dz), qx, qy, qz, qw]
        self.arm_action_pub.publish(arm_action)

        # Publish gripper action unchanged
        gripper_action = Float32()
        gripper_action.data = gripper_val
        self.gripper_action_pub.publish(gripper_action)

        # Update last_action for next step
        arr = [dx, dy, dz, qx, qy, qz, qw, gripper_val]
        self.last_action = np.array(arr, dtype=np.float32)

        self.get_logger().info(
            f"Published RL action: pos_delta=[{dx:.3f}, {dy:.3f}, {dz:.3f}], quat=[{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}], gripper={gripper_val:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyActionSplitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
