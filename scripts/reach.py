#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import onnxruntime as ort
from importlib.resources import files

class UR5ReachNode(Node):
    def __init__(self):
        super().__init__('ur5_reach_inference')

        # Cargar modelo ONNX
        self.model_path = str(files("scripts").joinpath("policy.onnx"))
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Joints del brazo
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.last_action = np.zeros(6)
        self.joint_state_received = False

        # Suscripción a /joint_states
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Publicador en scaled_joint_trajectory_controller
        self.cmd_pub = self.create_publisher(
            JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Lista de objetivos
        self.goal_poses = [
            [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
            [0.6, -0.2, 0.25, 0.0, 0.0, 0.0, 1.0],
            [0.45, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0],
            [0.55, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0],
        ]
        self.current_goal_index = 0

        # Timer para pasos de control
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

        # Construcción de la observación (padding a 37)
        goal = self.goal_poses[self.current_goal_index]

        obs_joint_pos = np.zeros(8)
        obs_joint_pos[:6] = self.joint_pos

        obs_joint_vel = np.zeros(8)
        obs_joint_vel[:6] = self.joint_vel

        obs_goal = np.array(goal)
        obs_last_action = np.zeros(6)
        obs_last_action[:6] = self.last_action

        obs_extra = np.zeros(8)

        obs = np.concatenate([obs_joint_pos, obs_joint_vel, obs_goal, obs_last_action, obs_extra])
        observation = obs.astype(np.float32).reshape(1, -1)

        # Inferencia con el modelo
        action = self.session.run(None, {self.input_name: observation})[0].squeeze()

        # Crear mensaje JointTrajectory
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = (self.joint_pos + action).tolist()
        point.time_from_start = Duration(sec=1)

        trajectory_msg.points.append(point)
        self.cmd_pub.publish(trajectory_msg)

        # Actualizar última acción
        self.last_action = action

        # Logs
        self.get_logger().info(f"Moviendo al objetivo {self.current_goal_index + 1}")
        self.get_logger().info(f"Posición actual: {self.joint_pos}")
        self.get_logger().info(f"Acción aplicada: {action}")
        self.current_goal_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = UR5ReachNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
