#!/usr/bin/env python3
# simple_image_publisher.py
# Publishes a single image file to /rgb at a fixed rate.
# Code comments in English.

import os
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

        # Params
        self.declare_parameter("image_path", "")
        self.declare_parameter("path_file", "/app/image.png")        # optional: text file that contains the image path
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("fps", 5.0)

        image_path = self.get_parameter("image_path").get_parameter_value().string_value
        path_file = self.get_parameter("path_file").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        fps = float(self.get_parameter("fps").value)

        # Resolve image path
        if not image_path and path_file:
            try:
                with open(path_file, "r") as f:
                    image_path = f.read().strip()
            except Exception as e:
                self.get_logger().error(f"Failed to read path_file: {e}")

        if not image_path:
            raise RuntimeError("No image_path provided. Set parameter 'image_path' or 'path_file'.")

        # Load image once
        self.bridge = CvBridge()
        self.bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.bgr is None:
            raise RuntimeError(f"cv2.imread failed for: {image_path}")

        # Publisher
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=1)
        self.pub = self.create_publisher(Image, "/rgb", qos)

        # Timer
        period = 1.0 / max(fps, 0.1)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(f"Publishing '{image_path}' to /rgb at {fps:.2f} FPS, frame_id='{self.frame_id}'")

    def _on_timer(self):
        msg = self.bridge.cv2_to_imgmsg(self.bgr, encoding="bgr8")
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id
        msg.header = hdr
        self.pub.publish(msg)


def main():
    rclpy.init()
    try:
        node = SimpleImagePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
