import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class JointStateToTrajectory(Node):
    def __init__(self):
        super().__init__('joint_states_to_trajectory')
        # QoS for receiving commands: reliable & volatile (default for most publishers)
        cmd_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        # QoS for publishing to controller: reliable & transient local (latching)
        traj_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscriber for 12-joint commands
        self.subscription = self.create_subscription(
            JointState,
            'joint_states_command',
            self.listener_callback,
            qos_profile=cmd_qos)
        # Publisher to controller (6 joints)
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            qos_profile=traj_qos)

        # Store last received joint state
        self.last_positions = None
        self.joint_names = []
        # Timer to publish at fixed rate
        self.timer = self.create_timer(0.1, self.publish_trajectory)  # 10 Hz

    def listener_callback(self, msg: JointState):
        # Only take first 6 joints
        if len(msg.position) < 6:
            self.get_logger().warn('Received fewer than 6 joint positions; skipping')
            return
        self.joint_names = msg.name[:6]
        # Copy positions
        self.last_positions = list(msg.position[:6])
        self.get_logger().debug(f'Received positions: {self.last_positions}')

    def publish_trajectory(self):
        if not self.last_positions or not self.joint_names:
            return  # nothing to publish

        traj = JointTrajectory()
        # Stamp now
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.last_positions
        # Small buffer into future
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)
        traj.points = [point]

        self.publisher.publish(traj)
        self.get_logger().info(
            f'Published to controller: {traj.joint_names} @ ' 
            f'{traj.header.stamp.sec}.{traj.header.stamp.nanosec}')


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
