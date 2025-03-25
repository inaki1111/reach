#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import onnxruntime as ort
from importlib.resources import files

class UR5ReachNode(Node):
    def __init__(self):
        super().__init__('ur5_reach_inference')

        self.model_path = str(files("scripts").joinpath("policy.onnx"))
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.last_action = np.zeros(6)
        self.joint_state_received = False

        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.cmd_pub = self.create_publisher(Float64MultiArray, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.goal_poses = [
            [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
            [0.6, -0.2, 0.25, 0.0, 0.0, 0.0, 1.0],
            [0.45, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0],
            [0.55, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0],
        ]

        self.current_goal_index = 0
        self.timer = self.create_timer(3.0, self.control_loop)

    def joint_state_callback(self, msg):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        try:
            self.joint_pos = np.array([msg.position[name_to_index[name]] for name in self.joint_names])
            self.joint_vel = np.array([msg.velocity[name_to_index[name]] for name in self.joint_names])
            self.joint_state_received = True
        except KeyError:
            pass

    def control_loop(self):
        if not self.joint_state_received:
            self.get_logger().info("Esperando joint_states...")
            return

        if self.current_goal_index >= len(self.goal_poses):
            self.get_logger().info("Todos los objetivos completados.")
            rclpy.shutdown()
            return

        goal = self.goal_poses[self.current_goal_index]

        obs_joint_pos = np.zeros(8)
        obs_joint_pos[:6] = self.joint_pos

        obs_joint_vel = np.zeros(8)
        obs_joint_vel[:6] = self.joint_vel

        obs_goal = np.array(goal)  # 7 elementos
        obs_last_action = np.zeros(6)
        obs_last_action[:6] = self.last_action

        obs_extra = np.zeros(8)  # Extra padding

        obs = np.concatenate([obs_joint_pos, obs_joint_vel, obs_goal, obs_last_action, obs_extra])
        observation = obs.astype(np.float32).reshape(1, -1)

        # Ejecutar inferencia
        action = self.session.run(None, {self.input_name: observation})[0]
        action = action.squeeze()

        # Aplicar acción como nuevo setpoint de posición
        cmd = Float64MultiArray()
        cmd.data = (self.joint_pos + action).tolist()
        self.cmd_pub.publish(cmd)
        self.last_action = action

        self.get_logger().info(f"Moviendo al objetivo {self.current_goal_index + 1}")
        self.current_goal_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = UR5ReachNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
