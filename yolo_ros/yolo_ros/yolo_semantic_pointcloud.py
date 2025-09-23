#!/usr/bin/env python3
# score_depth_to_pointcloud.py

from __future__ import annotations
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

FIELDS_XYZI = [
    PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
]

class ScoreDepthToCloud(Node):
    def __init__(self):
        super().__init__("score_depth_to_cloud")
        # params
        self.declare_parameter("score_topic", "/yoloe/score_mask_raw")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("cloud_topic", "/yoloe/score_cloud")
        self.declare_parameter("score_threshold", 0.01)
        self.declare_parameter("depth_min", 0.05)
        self.declare_parameter("depth_max", 10.0)
        self.declare_parameter("stride", 1)
        self.declare_parameter("frame_id", "camera")

        p = self.get_parameter
        self.score_topic = p("score_topic").get_parameter_value().string_value
        self.depth_topic = p("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = p("camera_info_topic").get_parameter_value().string_value
        self.cloud_topic = p("cloud_topic").get_parameter_value().string_value
        self.score_threshold = float(p("score_threshold").value)
        self.depth_min = float(p("depth_min").value)
        self.depth_max = float(p("depth_max").value)
        self.stride = max(1, int(p("stride").value))
        self.frame_id = p("frame_id").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        qos_be  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        self.sub_score = Subscriber(self, Image, self.score_topic, qos_profile=qos_rel)
        self.sub_depth = Subscriber(self, Image, self.depth_topic, qos_profile=qos_be)
        self.sub_info  = Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_rel)

        self.sync = ApproximateTimeSynchronizer([self.sub_score, self.sub_depth, self.sub_info],
                                                queue_size=10, slop=0.08)
        self.sync.registerCallback(self.cb)

        self.pub_cloud = self.create_publisher(PointCloud2, self.cloud_topic, qos_rel)

        self.get_logger().info(
            f"score={self.score_topic}, depth={self.depth_topic}, info={self.camera_info_topic} -> {self.cloud_topic}"
        )

    def cb(self, score_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        # intrinsics each time (cheap and safe)
        self.fx, self.fy, self.cx, self.cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]

        # convert
        try:
            score = self.bridge.imgmsg_to_cv2(score_msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"score cv_bridge: {e}")
            return
        try:
            if depth_msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1").astype(np.float32) * 0.001
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1").astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"depth cv_bridge: {e}")
            return

        if score.shape[:2] != depth.shape[:2]:
            score = cv2.resize(score, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid = (
            np.isfinite(depth)
            & (depth > self.depth_min) & (depth < self.depth_max)
            & (score >= self.score_threshold)
        )

        if not np.any(valid):
            cloud = point_cloud2.create_cloud(self._header(depth_msg.header.stamp), FIELDS_XYZI, [])
            self.pub_cloud.publish(cloud)
            return

        ys, xs = np.where(valid)
        if self.stride > 1:
            ys = ys[::self.stride]
            xs = xs[::self.stride]

        z = depth[ys, xs]
        s = score[ys, xs]

        x = (xs.astype(np.float32) - self.cx) / self.fx * z
        y = (ys.astype(np.float32) - self.cy) / self.fy * z

        data = np.column_stack((x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), s.astype(np.float32)))
        pts_iter = (tuple(row) for row in data)

        cloud = point_cloud2.create_cloud(self._header(depth_msg.header.stamp), FIELDS_XYZI, pts_iter)
        self.pub_cloud.publish(cloud)

    def _header(self, stamp: Time) -> Header:
        h = Header()
        h.stamp = stamp
        h.frame_id = self.frame_id
        return h

def main():
    rclpy.init()
    node = ScoreDepthToCloud()
    try:
            rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
