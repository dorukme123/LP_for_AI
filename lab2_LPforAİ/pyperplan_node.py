#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pyperplan
from rclpy.executors import SingleThreadedExecutor

class SimpleNode(Node):
    def __init__(self):
        super().__init__('pyperplan_node')
        self.get_logger().info('pyperplan is ready on machine doruk@dorukvn...')
        self.create_timer(2.0, self.cleanShutdown)
    
    def cleanShutdown(self):
        self.get_logger().info('shutting down')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('shutting down: Ki')
    finally:
        node.get_logger().info('cleaning up...')
        executor.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()