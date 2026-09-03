#!/usr/bin/env python3
"""
ball_detector_node.py
---------------------
ROS2 Node for detecting red/green objects and controlling the arm.
Subscribes to: /camera/image_raw
Publishes to: /arm_controller/commands (or as configured in arm_controller.py)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

from ball_detector import BallDetector
from arm_controller import ArmController, ArmBackend

class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector_node')
        
        # Parameters
        self.declare_parameter('target_color', 'both')
        self.declare_parameter('arm_enabled', True)
        
        self.target_color = self.get_parameter('target_color').get_parameter_value().string_value
        self.arm_enabled = self.get_parameter('arm_enabled').get_parameter_value().bool_value
        
        # CV Logic
        self.detector = BallDetector(target_color=self.target_color)
        self.bridge = CvBridge()
        
        # Arm Controller (configured for ROS backend)
        self.arm_ctrl = ArmController(
            backend=ArmBackend.ROS,
            ros_topic='/arm_controller/commands' 
        )
        self.arm_ctrl._ros_node = self
        self.arm_ctrl._ros_pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Control Subscribers (for user inputs via ROS)
        self.color_sub = self.create_subscription(
            String,
            '/detector/set_color',
            self.set_color_callback,
            10
        )
        self.enable_sub = self.create_subscription(
            Bool,
            '/detector/enable_arm',
            self.enable_arm_callback,
            10
        )
        
        self.get_logger().info(f"Ball Detector Node started. Target: {self.target_color}, Arm: {self.arm_enabled}")

    def set_color_callback(self, msg):
        color = msg.data.lower()
        if color in ["red", "green", "both"]:
            self.target_color = color
            self.detector.target_color = color
            self.get_logger().info(f"Target color set to: {color}")
        else:
            self.get_logger().warn(f"Invalid color: {color}")

    def enable_arm_callback(self, msg):
        self.arm_enabled = msg.data
        self.get_logger().info(f"Arm enabled: {self.arm_enabled}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            
            # Detection
            detections = self.detector.detect(frame)
            
            if detections:
                best_det = detections[0]
                self.get_logger().info(f"Detected {best_det.color} object at ({best_det.cx}, {best_det.cy})")
                
                # Arm control
                if self.arm_enabled:
                    self.arm_ctrl.move_to_detection(best_det, w, h)
            
            # Annotate and show (optional, useful for debugging in WSL if X11 is setup)
            annotated = self.detector.annotate(frame, detections)
            cv2.imshow("Detection Debug", annotated)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
