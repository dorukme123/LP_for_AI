import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import subprocess
import time

class RobotPlanner(Node):

    def __init__(self):
        super().__init__('robot_planner')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(1.0, self.planMove)
        self.plan = []
        self.current_step = 0
        self.getPlan()

    def getPlan(self):
    	domain_file = '/home/doruk/ros2_ws/src/robot_planning_pkg/domain.pddl'
    	problem_file = '/home/doruk/ros2_ws/src/robot_planning_pkg/problem.pddl'
    	result = subprocess.run(
    		['pyperplan', '--heuristic', 'hff', '--search', 'astar', domain_file, problem_file],
    		stdout=subprocess.PIPE, text=True
    	)
    	plan_output = result.stdout.splitlines()
    	self.get_logger().info(f'full pyperplan output: {plan_output}')
    	self.plan = [line for line in plan_output if line.startswith("(move")]
    	self.get_logger().info(f'Plan: {self.plan}')

    def planMove(self):
        if self.current_step < len(self.plan):
            action = self.plan[self.current_step]
            self.get_logger().info(f'Executing: {action}')
            parts = action.replace('(', '').replace(')', '').split()
            _, robot, start_location, end_location = parts
            self.move()
            self.current_step += 1
        else:
            self.get_logger().info('Plan completed')

    def move(self):
        msg = Twist()
        msg.linear.x = 0.5  # forward
        msg.angular.z = 0.0
        self.get_logger().info(f'Velocity command: {msg}')
        self.publisher_.publish(msg)  
        time.sleep(6)
        msg.linear.x = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info(f'Robot stopped')

def main(args=None):
    rclpy.init(args=args)
    robot_planner = RobotPlanner()
    rclpy.spin(robot_planner)
    robot_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
