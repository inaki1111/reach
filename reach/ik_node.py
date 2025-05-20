#!/usr/bin/env python3
import os
import tempfile

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

import pinocchio as pin
import numpy as np
import torch

from .differential_ik_ros import DifferentialIKControllerROS

class IKStaticNode(Node):
    def __init__(self):
        super().__init__('ik_static_node')

        # --- Lectura del URDF desde parameter server ---
        self.declare_parameter('robot_description', '')
        urdf_xml = self.get_parameter('robot_description') \
                       .get_parameter_value().string_value

        # --- Escribe URDF a un fichero temporal para Pinocchio ---
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
        tf.write(urdf_xml)
        tf.flush()
        tf.close()
        urdf_path = tf.name

        # --- Construye el modelo Pinocchio desde fichero URDF ---
        self.model = pin.buildModelFromUrdf(urdf_path)  # :contentReference[oaicite:0]{index=0}
        os.remove(urdf_path)

        # --- Crea el Data asociado y obtiene el frame de efector final ---
        self.data = self.model.createData()
        self.ee_frame = self.model.getFrameId('tool0')

        # --- Estado articular del UR5 (6 DoF) ---
        self.n = self.model.nq
        self.joint_pos = np.zeros(self.n, dtype=np.float32)

        # --- Pose objetivo estática (base_link) ---
        self.target_pos  = np.array([0.5, 0.0, 0.3], dtype=np.float32)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w,x,y,z)

        # --- Controlador diferencial IK standalone ---
        self.ctrl = DifferentialIKControllerROS(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
            device="cpu"
        )

        # --- Subscripción y publicación de JointTrajectory ---
        self.create_subscription(
            JointState, '/joint_states',
            self.cb_joint_state, 10
        )
        self.pub_traj = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        # Bucle de control a 30 Hz
        self.dt = 1.0 / 30.0
        self.create_timer(self.dt, self.control_loop)

    def cb_joint_state(self, msg: JointState):
        idx = {name: i for i, name in enumerate(msg.name)}
        for i, jn in enumerate(self.model.names[1:1+self.n]):
            if jn in idx:
                self.joint_pos[i] = msg.position[idx[jn]]

    def control_loop(self):
        q = self.joint_pos.copy()

        # 1) Cinemática directa
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.ee_frame]
        ee_pos = np.array(placement.translation, dtype=np.float32)

        # Convierte quaternion (x,y,z,w) → (w,x,y,z)
        coeffs = pin.Quaternion(placement.rotation).coeffs()
        ee_quat = np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]], dtype=np.float32)

        # 2) Jacobiano espacial 6×n
        J_full = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = np.array(J_full, dtype=np.float32)

        # 3) Tensores para el controlador IK
        ee_pos_t  = torch.from_numpy(ee_pos).unsqueeze(0)       # (1,3)
        ee_quat_t = torch.from_numpy(ee_quat).unsqueeze(0)      # (1,4)
        J_t       = torch.from_numpy(J).view(1,6,self.n)        # (1,6,n)
        jp_t      = torch.from_numpy(self.joint_pos).unsqueeze(0)  # (1,n)

        # 4) Comando objetivo
        cmd = torch.tensor([[*self.target_pos, *self.target_quat]],
                           dtype=torch.float32)               # (1,7)

        # 5) Invoca controlador IK
        self.ctrl.reset()
        self.ctrl.set_command(cmd, ee_pos=ee_pos_t, ee_quat=ee_quat_t)
        q_target = self.ctrl.compute(ee_pos_t, ee_quat_t, J_t, jp_t)
        q_des = q_target.squeeze(0).numpy()  # (n,)

        # 6) Publica trayectoria
        traj = JointTrajectory(joint_names=self.model.names[1:1+self.n])
        pt = JointTrajectoryPoint(
            positions=q_des.tolist(),
            time_from_start=MsgDuration(sec=int(self.dt))
        )
        traj.points.append(pt)
        self.pub_traj.publish(traj)

def main():
    rclpy.init()
    node = IKStaticNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
