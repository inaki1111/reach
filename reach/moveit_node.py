#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from action_msgs.msg import GoalStatus
import numpy as np
from reach.ur5e_moveit_client import UR5eMoveItClient

class ArmActionExecutor(Node):
    def __init__(self):
        super().__init__('arm_action_executor')
        self.client = UR5eMoveItClient()

        # Subscribirnos a la salida de tu RL
        self.create_subscription(
            Float32MultiArray,
            '/arm_action',
            self.arm_action_cb,
            10
        )
        self.get_logger().info("ArmActionExecutor ready")

    def arm_action_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) != 7:
            self.get_logger().warn(f"Invalid arm_action length: {len(data)}")
            return

        # 1) Extraer solo la parte de traslación y escalar ±5 cm
        dx, dy, dz = data[:3]
        dx, dy, dz = np.clip([dx, dy, dz], -0.05, 0.05)

        # 2) Ignorar qx,qy,qz,qw de la política; pillar orientación actual
        pos, ori = self.client.get_link_position('wrist_3_link')
        if pos is None or ori is None:
            self.get_logger().warn("Cannot retrieve current pose/orientation")
            return
        # ori es una tupla (qx, qy, qz, qw)

        # 3) Construir la nueva pose con delta en posición + orientación actual
        new_pos = [pos[0] + float(dx), pos[1] + float(dy), pos[2] + float(dz)]
        new_pose = new_pos + list(ori)

        # 4) Lanzar la planificación de forma sincrónica y verificar resultado
        self.get_logger().info(f"Executing small step to {np.round(new_pose,3).tolist()}")
        status = self.client.move_to_pose(new_pose)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ Small step executed successfully")
        else:
            self.get_logger().error(f"❌ Small step failed with status {status}")


def main(args=None):
    rclpy.init(args=args)
    node = ArmActionExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
