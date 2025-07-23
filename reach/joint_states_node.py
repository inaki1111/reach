import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class JointStateToTrajectory(Node):
    def __init__(self):
        super().__init__('joint_states_to_trajectory')

        # QoS for subscribing to joint_states_command
        cmd_qos = QoSProfile(depth=10,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(
            JointState,
            'joint_states_command',
            self.listener_callback,
            qos_profile=cmd_qos)

        # Action client to send goals sequentially
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')
        # Queue for pending goals
        self._goal_queue = []
        self._goal_active = False

    def listener_callback(self, msg: JointState):
        # Only take first 6 joints
        if len(msg.position) < 6:
            self.get_logger().warn('Received fewer than 6 joint positions; skipping')
            return

        # Build trajectory message
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = msg.name[:6]
        point = JointTrajectoryPoint()
        point.positions = list(msg.position[:6])
        # small buffer into future
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)
        traj.points = [point]

        # Create goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        # Enqueue and attempt to send
        self._goal_queue.append(goal_msg)
        self.get_logger().info(f'Enqueued goal for joints: {traj.joint_names}')
        self._try_send_next()

    def _try_send_next(self):
        if not self._goal_active and self._goal_queue:
            goal_msg = self._goal_queue.pop(0)
            self._goal_active = True
            self.get_logger().info('Sending next goal...')
            send_goal_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback)
            send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by controller')
            self._goal_active = False
            self._try_send_next()
            return

        self.get_logger().info('Goal accepted, waiting for result')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _feedback_callback(self, feedback_msg):
        # Optional: process feedback
        pass

    def _get_result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info('Goal succeeded')
        else:
            self.get_logger().warn(f'Goal failed with error code: {result.error_code}')
        self._goal_active = False
        self._try_send_next()


def main(args=None):
    rclpy.init(args=args)
    node = JointStateToTrajectory()
    # Wait for action server
    node.get_logger().info('Waiting for action server...')
    node._action_client.wait_for_server()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
