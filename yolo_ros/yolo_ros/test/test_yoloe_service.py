#!/usr/bin/env python3
# yoloe_text_client.py
# Test client: subscribe to /rgb, call YOLOE text service, print bboxes, show image.
# Code comments in English as requested.

import argparse
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolo_ros_interfaces.srv import SemanticDetection  # adjust if your package name differs


class YOLOETextClient(Node):
    def __init__(self, rgb_topic: str, service_name: str, prompt: str, timeout: float):
        super().__init__("yoloe_text_client")
        self.bridge = CvBridge()
        self.prompt = prompt
        self.timeout = timeout
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.latest_img_msg = None
        self.sub = self.create_subscription(Image, rgb_topic, self.on_image, qos)
        self.cli = self.create_client(SemanticDetection, service_name)

    # --- ROS image callback ---
    def on_image(self, msg: Image):
        # Keep the latest frame
        self.latest_img_msg = msg

    # --- wait until we have an image ---
    def wait_for_image(self) -> bool:
        t0 = time.time()
        while rclpy.ok() and self.latest_img_msg is None and (time.time() - t0) < self.timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_img_msg is not None

    # --- call service and return response ---
    def call_service(self):
        # Wait for service
        if not self.cli.wait_for_service(timeout_sec=self.timeout):
            self.get_logger().error("Service not available.")
            return None
        req = SemanticDetection.Request()
        req.image = self.latest_img_msg
        req.text_prompt = self.prompt
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout)
        if not future.done():
            self.get_logger().error("Service call timed out.")
            return None
        return future.result()

    # --- draw detections on image ---
    @staticmethod
    def draw_and_print(bgr: np.ndarray, det_arr) -> np.ndarray:
        out = bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if det_arr is None or len(det_arr.detections) == 0:
            print("No detections.")
            return out

        for i, det in enumerate(det_arr.detections):
            # bbox: center (cx,cy), size (w,h)
            bb = det.bbox
            cx = getattr(bb.center, "x", 0.0)
            cy = getattr(bb.center, "y", 0.0)
            w = float(bb.size_x)
            h = float(bb.size_y)
            x1 = int(max(0, cx - w / 2))
            y1 = int(max(0, cy - h / 2))
            x2 = int(min(out.shape[1] - 1, cx + w / 2))
            y2 = int(min(out.shape[0] - 1, cy + h / 2))

            # pick top hypothesis (first result)
            label = ""
            score = 0.0
            if det.results:
                label = det.results[0].hypothesis.class_id
                score = det.results[0].hypothesis.score

            # print to terminal
            print(
                f"{i}: label={label}, score={score:.3f}, bbox=[{x1},{y1},{x2},{y2}] "
                f"center=({cx:.1f},{cy:.1f}) size=({w:.1f},{h:.1f})"
            )

            # draw on image
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{label} {score:.2f}",
                (x1, max(0, y1 - 5)),
                font,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out

    # --- overlay mask (mono8) onto BGR image ---
    @staticmethod
    def overlay_mask(bgr: np.ndarray, mask_mono8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        if mask_mono8 is None or mask_mono8.size == 0:
            return bgr
        mask_bool = mask_mono8 > 0
        color = np.zeros_like(bgr)
        color[:, :, 2] = 255  # red
        out = bgr.copy()
        out[mask_bool] = ((1 - alpha) * out[mask_bool] + alpha * color[mask_bool]).astype(np.uint8)
        return out


def main():
    parser = argparse.ArgumentParser(description="YOLOE text service test client.")
    parser.add_argument("--rgb", default="/rgb", help="RGB image topic")
    parser.add_argument("--service", default="/yoloe_text_detect", help="SemanticDetection service name")
    parser.add_argument("--prompt", default="chair", help="Text prompt")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout seconds")
    args = parser.parse_args()

    rclpy.init(args=None)
    node = YOLOETextClient(args.rgb, args.service, args.prompt, args.timeout)

    try:
        print(f"Waiting for image on {args.rgb} ...")
        if not node.wait_for_image():
            print("No image received within timeout.")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        print(f"Calling service {args.service} with prompt='{args.prompt}' ...")
        resp = node.call_service()
        if resp is None:
            print("Service call failed.")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(2)

        # Convert ROS images to OpenCV
        bgr = node.bridge.imgmsg_to_cv2(node.latest_img_msg, desired_encoding="bgr8")
        mask = node.bridge.imgmsg_to_cv2(resp.segmented_image, desired_encoding="mono8")

        # Draw detections and overlay mask
        vis = node.draw_and_print(bgr, resp.detections)
        vis_mask = node.overlay_mask(vis, mask)

        # Show images
        try:
            cv2.imshow("YOLOE overlay", vis_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            # Headless fallback
            cv2.imwrite("yoloe_overlay.png", vis_mask)
            print(f"No display. Saved 'yoloe_overlay.png'. Reason: {e}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
