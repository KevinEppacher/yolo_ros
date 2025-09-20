#!/usr/bin/env python3
# simple_image_publisher.py
# Publishes /app/image.png to /rgb at a fixed rate.

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge


class SimpleImagePublisher(Node):
    def __init__(self):
        super().__init__("simple_image_publisher")

        # Fixed path
        self.image_path = "/app/image.png"

        # Params you may still want to tweak
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("fps", 5.0)
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        fps = float(self.get_parameter("fps").value)

        # Load once
        self.bridge = CvBridge()
        self.bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.bgr is None:
            raise RuntimeError(f"cv2.imread failed for: {self.image_path}")

        # Publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        self.pub = self.create_publisher(Image, "/rgb", qos)

        # Timer
        period = 1.0 / max(fps, 0.1)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(f"Publishing '{self.image_path}' to /rgb at {fps:.2f} FPS, frame_id='{self.frame_id}'")

    def _on_timer(self):
        msg = self.bridge.cv2_to_imgmsg(self.bgr, encoding="bgr8")
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id
        msg.header = hdr
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = None
    try:
        node = SimpleImagePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
