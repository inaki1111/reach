#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from importlib.resources import files

class UR5ReachNode(Node):
    def __init__(self):
        super().__init__('ur5_reach_inference_tf_corrected')

        # Modelo ONNX
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

        # tf2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher y Subscriber
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.cmd_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Objetivos en tool0
        self.goal_poses_tool0 = [
            [-0.138, -0.5045, 0.08998, 0.029, 3.1, -0.54],
            [-0.141, -0.65638, -0.14177, 0.028, 3.099, -0.90],
            [-0.13351, -0.09202, 0.66481, 0.008, 2.33, -2.093],
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

    def get_pose_in_base_link(self, pose_tool0):
        x, y, z, rx, ry, rz = pose_tool0
        quat = R.from_euler("xyz", [rx, ry, rz]).as_quat()

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "tool0"
        pose_stamped.header.stamp = rclpy.time.Time().to_msg()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        try:
            tfed = self.tf_buffer.transform(pose_stamped, "base_link", timeout=rclpy.duration.Duration(seconds=1.0).to_msg())
            return [
                tfed.pose.position.x,
                tfed.pose.position.y,
                tfed.pose.position.z,
                tfed.pose.orientation.x,
                tfed.pose.orientation.y,
                tfed.pose.orientation.z,
                tfed.pose.orientation.w,
            ]
        except Exception as e:
            self.get_logger().error(f"No se pudo transformar la pose: {e}")
            return None

    def control_loop(self):
        if not self.joint_state_received:
            self.get_logger().info("Esperando joint_states...")
            return

        if self.current_goal_index >= len(self.goal_poses_tool0):
            self.get_logger().info("Todos los objetivos completados.")
            rclpy.shutdown()
            return

        goal_tool0 = self.goal_poses_tool0[self.current_goal_index]
        goal_converted = self.get_pose_in_base_link(goal_tool0)
        if goal_converted is None:
            self.get_logger().warn("Saltando este objetivo por error de TF")
            return

        # Construcción de la observación (padding a 37)
        obs_joint_pos = np.zeros(8)
        obs_joint_pos[:6] = self.joint_pos

        obs_joint_vel = np.zeros(8)
        obs_joint_vel[:6] = self.joint_vel

        obs_goal = np.array(goal_converted)
        obs_last_action = np.zeros(6)
        obs_last_action[:6] = self.last_action
        obs_extra = np.zeros(8)

        obs = np.concatenate([obs_joint_pos, obs_joint_vel, obs_goal, obs_last_action, obs_extra])
        observation = obs.astype(np.float32).reshape(1, -1)

        # Inferencia y suavizado
        action = self.session.run(None, {self.input_name: observation})[0].squeeze()
        scaled_action = action * 0.5

        # Crear comando
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = (self.joint_pos + scaled_action).tolist()
        duration_sec = min(max(np.linalg.norm(scaled_action) * 2.0, 1.0), 5.0)
        point.time_from_start = Duration(sec=int(duration_sec))
        trajectory_msg.points.append(point)
        self.cmd_pub.publish(trajectory_msg)

        self.last_action = scaled_action

        # Logs
        self.get_logger().info(f"Objetivo {self.current_goal_index + 1} enviado.")
        self.get_logger().info(f"Acción: {scaled_action}")
        self.get_logger().info(f"Pose transformada: {goal_converted}")
        self.current_goal_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = UR5ReachNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
