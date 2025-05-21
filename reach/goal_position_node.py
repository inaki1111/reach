#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
 
class GoalObjectPublisher(Node):
    def __init__(self):
        super().__init__('goal_object_publisher')

        self.goal_pub = self.create_publisher(Pose, '/target_object_position', 10)
        self.object_pub = self.create_publisher(Point, '/object_position', 10)

        self.goal_axes_pub = self.create_publisher(PoseStamped, '/target_object_axes', 10)
        self.object_axes_pub = self.create_publisher(PoseStamped, '/object_axes', 10)

        self.timer = self.create_timer(0.5, self.publish_data)

        self.goal_pose = Pose()
        self.goal_pose.position.x = 0.4
        self.goal_pose.position.y = 0.2
        self.goal_pose.position.z = 0.25
        self.goal_pose.orientation.x = 0.0
        self.goal_pose.orientation.y = 0.0
        self.goal_pose.orientation.z = 0.0
        self.goal_pose.orientation.w = 1.0  

        self.object_pose = Pose()
        self.object_pose.position.x = 0.3
        self.object_pose.position.y = 0.1
        self.object_pose.position.z = 0.05
        self.object_pose.orientation.x = 0.0
        self.object_pose.orientation.y = 0.0
        self.object_pose.orientation.z = 0.0
        self.object_pose.orientation.w = 1.0  

    def publish_data(self):
        now = self.get_clock().now().to_msg()

        self.goal_pub.publish(self.goal_pose)
        self.object_pub.publish(self.object_pose.position)

        goal_axes = PoseStamped()
        goal_axes.header.frame_id = "world"
        goal_axes.header.stamp = now
        goal_axes.pose = self.goal_pose
        self.goal_axes_pub.publish(goal_axes)

        object_axes = PoseStamped()
        object_axes.header.frame_id = "world"
        object_axes.header.stamp = now
        object_axes.pose = self.object_pose
        self.object_axes_pub.publish(object_axes)


def main(args=None):
    rclpy.init(args=args)
    node = GoalObjectPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
