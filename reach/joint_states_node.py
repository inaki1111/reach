import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

class JointStateToTrajectory(Node):
    def __init__(self):
        super().__init__('joint_states_to_trajectory')
        self._switch_cli = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller'
        )
        self._switch_cli.wait_for_service()
        self._activate_trajectory_controller()
        self.subscription = self.create_subscription(
            JointState,
            'joint_states_command',
            self.listener_callback,
            10
        )
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        self._action_client.wait_for_server()
        self._goal_queue = []
        self._goal_active = False

    def _activate_trajectory_controller(self):
        req = SwitchController.Request()
        req.activate_controllers = ['joint_trajectory_controller']
        req.deactivate_controllers = [
            'forward_position_controller',
            'forward_velocity_controller',
            'scaled_joint_trajectory_controller',
            'passthrough_trajectory_controller',
            'force_mode_controller',
            'freedrive_mode_controller'
        ]
        req.strictness = req.STRICT
        future = self._switch_cli.call_async(req)
        future.add_done_callback(self._on_switch_response)

    def _on_switch_response(self, future):
        try:
            _ = future.result()
        except Exception:
            pass

    def listener_callback(self, msg: JointState):
        if len(msg.position) < 6:
            return
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = msg.name[:6]
        point = JointTrajectoryPoint()
        point.positions = list(msg.position[:6])
        point.time_from_start = Duration(sec=3,nanosec=0)
        traj.points = [point]
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        self._goal_queue.append(goal_msg)
        self._try_send_next()

    def _try_send_next(self):
        if not self._goal_active and self._goal_queue:
            goal_msg = self._goal_queue.pop(0)
            self._goal_active = True
            send_goal_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )
            send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._goal_active = False
            self._try_send_next()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _feedback_callback(self, feedback_msg):
        pass

    def _get_result_callback(self, future):
        result = future.result().result
        self._goal_active = False
        self._try_send_next()


def main(args=None):
    rclpy.init(args=args)
    node = JointStateToTrajectory()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

