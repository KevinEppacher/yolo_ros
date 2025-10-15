#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
import cv2


class RGBImageSubscriber(Node):
    def __init__(self):
        super().__init__('rgb_image_subscriber')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            qos_profile
        )

        self.bridge = CvBridge()
        self.get_logger().info('RGB Image Subscriber started with Best Effort QoS.')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RGBImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
