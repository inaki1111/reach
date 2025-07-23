import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class JointStateToTrajectory(Node):
    def init(self):
        super().init('joint_states_to_trajectory')

    # Subscribe to the incoming 12-joint command
    self.subscription = self.create_subscription(
        JointState,
        'joint_states_command',
        self.listener_callback,
        10)
    # Publish to the trajectory controller (6 joints)
    self.publisher = self.create_publisher(
        JointTrajectory,
        '/joint_trajectory_controller/joint_trajectory',
        10)

def listener_callback(self, msg: JointState):
    # Ensure there are at least 6 joints in the message
    if len(msg.position) < 6:
        self.get_logger().warn('Received fewer than 6 joint positions; skipping')
        return

    # Build the trajectory message
    traj = JointTrajectory()
    traj.header.stamp = self.get_clock().now().to_msg()
    # Take only the first 6 joint names and positions
    traj.joint_names = msg.name[:6]

    point = JointTrajectoryPoint()
    point.positions = list(msg.position[:6])
    # Immediate execution (time_from_start = 0)
    point.time_from_start = Duration(sec=0, nanosec=0)

    traj.points = [point]

    # Publish to the controller
    self.publisher.publish(traj)
    self.get_logger().info(f'Published trajectory for joints: {traj.joint_names}')
def main(args=None):
    rclpy.init(args=args)
    node = JointStateToTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
