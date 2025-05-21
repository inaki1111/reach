#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from reach.ur5e_moveit_client import UR5eMoveItClient
from action_msgs.msg import GoalStatus

class StaticPoseSender(Node):
    def __init__(self):
        super().__init__('static_pose_sender')
        # Cliente MoveIt
        self.client = UR5eMoveItClient()
        # Creamos un timer normal
        self.timer = self.create_timer(1.0, self.send_pose)

    def send_pose(self):
        # Cancelamos el timer para que solo se ejecute una vez
        self.timer.cancel()

        # Define aquí la pose que quieras probar: [x, y, z, qx, qy, qz, qw]
        test_pose = [0.4, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        self.get_logger().info(f'👉 Sending static pose: {test_pose}')

        # Enviamos la petición
        status = self.client.move_to_pose(test_pose)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✅ MoveIt succeeded on static pose')
        else:
            self.get_logger().error(f'❌ MoveIt failed with status {status}')

def main(args=None):
    rclpy.init(args=args)
    node = StaticPoseSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
