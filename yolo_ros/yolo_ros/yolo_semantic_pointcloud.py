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

FIELDS_XYZI_ID = [
    PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='instance',  offset=16, datatype=PointField.UINT16,  count=1),
]

class SemanticPointCloud(Node):
    def __init__(self):
        super().__init__("semantic_pointcloud")
        # params
        self.declare_parameter("score_topic", "/yoloe/score_mask_raw")
        self.declare_parameter("id_topic", "/yoloe/instance_id_mask")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("cloud_topic", "/yoloe/semantic_pointcloud_xyzi")
        self.declare_parameter("score_threshold", 0.01)
        self.declare_parameter("depth_min", 0.05)
        self.declare_parameter("depth_max", 10.0)
        self.declare_parameter("stride", 1)
        self.declare_parameter("frame_id", "camera")

        p = self.get_parameter
        self.score_topic = p("score_topic").get_parameter_value().string_value
        self.id_topic = p("id_topic").get_parameter_value().string_value
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

        # sync 4 topics: score, id, depth, info
        self.sub_score = Subscriber(self, Image, self.score_topic, qos_profile=qos_rel)
        self.sub_id    = Subscriber(self, Image, self.id_topic,    qos_profile=qos_rel)
        self.sub_depth = Subscriber(self, Image, self.depth_topic, qos_profile=qos_be)
        self.sub_info  = Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_rel)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_score, self.sub_id, self.sub_depth, self.sub_info],
            queue_size=10, slop=0.08
        )
        self.sync.registerCallback(self.cb)

        self.pub_cloud = self.create_publisher(PointCloud2, self.cloud_topic, qos_rel)

        self.get_logger().info(
            f"score={self.score_topic}, id={self.id_topic}, depth={self.depth_topic}, info={self.camera_info_topic} -> {self.cloud_topic}"
        )

    def cb(self, score_msg: Image, id_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        # intrinsics
        self.fx, self.fy, self.cx, self.cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]

        self.get_logger().debug(f"cb: score {score_msg.height}x{score_msg.width} {score_msg.encoding}, "
                                f"id {id_msg.height}x{id_msg.width} {id_msg.encoding}, "
                                f"depth {depth_msg.height}x{depth_msg.width} {depth_msg.encoding}, "
                                f"info {info_msg.width}x{info_msg.height}")

        # convert score + id + depth
        try:
            score = self.bridge.imgmsg_to_cv2(score_msg, desired_encoding="32FC1")  # [0,1]
        except Exception as e:
            self.get_logger().error(f"score cv_bridge: {e}"); return
        try:
            inst = self.bridge.imgmsg_to_cv2(id_msg, desired_encoding="16UC1")      # 0..N
        except Exception as e:
            self.get_logger().error(f"id cv_bridge: {e}"); return
        try:
            if depth_msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1").astype(np.float32) * 0.001
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1").astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"depth cv_bridge: {e}"); return

        # resize masks to depth size if needed
        H, W = depth.shape[:2]
        if score.shape[:2] != (H, W):
            score = cv2.resize(score, (W, H), interpolation=cv2.INTER_NEAREST)
        if inst.shape[:2] != (H, W):
            inst  = cv2.resize(inst,  (W, H), interpolation=cv2.INTER_NEAREST)

        # strict mask: valid depth, score > thr, instance > 0
        valid = (
            np.isfinite(depth) & (depth > self.depth_min) & (depth < self.depth_max) &
            np.isfinite(score) & (score > self.score_threshold) &
            (inst > 0)
        )
        if not np.any(valid):
            cloud = point_cloud2.create_cloud(self._header(depth_msg.header.stamp), FIELDS_XYZI_ID, [])
            self.pub_cloud.publish(cloud)
            return

        ys, xs = np.where(valid)
        if self.stride > 1:
            ys = ys[::self.stride]; xs = xs[::self.stride]

        z   = depth[ys, xs].astype(np.float32)
        s   = score[ys, xs].astype(np.float32)
        iid = inst[ys, xs].astype(np.uint16)

        x = (xs.astype(np.float32) - self.cx) / self.fx * z
        y = (ys.astype(np.float32) - self.cy) / self.fy * z

        # tuples: (x, y, z, intensity, instance)
        pts_iter = ((x[i], y[i], z[i], s[i], int(iid[i])) for i in range(len(z)))
        cloud = point_cloud2.create_cloud(self._header(depth_msg.header.stamp), FIELDS_XYZI_ID, pts_iter)
        self.pub_cloud.publish(cloud)

    def _header(self, stamp: Time) -> Header:
        h = Header()
        h.stamp = stamp
        h.frame_id = self.frame_id
        return h

def main():
    rclpy.init()
    node = SemanticPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
