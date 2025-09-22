#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from ultralytics import YOLOE
from ament_index_python.packages import get_package_share_directory

from vlm_base.vlm_base import VLMBaseLifecycleNode


# ---------- helpers ----------
def _cls_name(names, idx: int) -> str:
    if isinstance(names, dict):
        return names.get(idx, str(idx))
    if isinstance(names, (list, tuple)):
        return names[idx] if 0 <= idx < len(names) else str(idx)
    return str(idx)


def _make_center(cx: float, cy: float):
    c = type(BoundingBox2D().center)()
    if hasattr(c, "x"):
        c.x, c.y = float(cx), float(cy)
        if hasattr(c, "theta"):
            c.theta = 0.0
    elif hasattr(c, "position"):
        c.position.x, c.position.y = float(cx), float(cy)
        if hasattr(c, "theta"):
            c.theta = 0.0
    return c


# ---------- inference wrapper ----------
class YOLOEInference:
    def __init__(self, node, weights: str, imgsz: int, conf: float, iou: float, prompts: list[str]):
        self.node = node
        self.model = YOLOE(weights).to(node.device).eval()
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.prompts = list(prompts)

        if self.prompts:
            pe = self.model.get_text_pe(self.prompts)
            pe3 = pe if pe.dim() == 3 else pe.unsqueeze(0)  # (1,K,D)
            self.model.set_classes(self.prompts, pe3)

    @torch.no_grad()
    def infer(self, bgr: np.ndarray):
        return self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]


# ---------- lifecycle publisher node ----------
class YoloWrapper(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__("yoloe_lifecycle_node")
        # params
        self.declare_parameter("weights", "yoloe-11l-seg.pt")    # filename or absolute path
        self.declare_parameter("model_pkg", "yolo_ros")          # package that holds the model
        self.declare_parameter("model_dir", "models")            # subdir inside package share
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("text_prompts", ["chair", "door", "bed"])
        self.declare_parameter("publish_overlay", True)

        self.inference = None
        self.pub_det = None
        self.pub_mask = None
        self.pub_overlay = None

        self.qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

    def _resolve_weights_path(self, weights_param: str) -> str:
        # Absolute file given and exists
        if os.path.isabs(weights_param) and os.path.isfile(weights_param):
            return weights_param

        # Try package share/<model_dir>/<weights_param>
        pkg = self.get_parameter("model_pkg").get_parameter_value().string_value
        sub = self.get_parameter("model_dir").get_parameter_value().string_value
        try:
            share = get_package_share_directory(pkg)
            cand = os.path.join(share, sub, weights_param)
            if os.path.isfile(cand):
                return cand
            # Also try directly under share in case you placed it there
            cand2 = os.path.join(share, weights_param)
            if os.path.isfile(cand2):
                return cand2
            self.get_logger().warn(f"Model not found at '{cand}' or '{cand2}'. Using '{weights_param}' as-is.")
        except Exception as e:
            self.get_logger().warn(f"get_package_share_directory('{pkg}') failed: {e}. Using '{weights_param}' as-is.")
        return weights_param

    # Base will call this during on_configure()
    def load_model(self):
        weights_in = self.get_parameter("weights").get_parameter_value().string_value
        weights = self._resolve_weights_path(weights_in)

        imgsz = int(self.get_parameter("imgsz").value)
        conf = float(self.get_parameter("conf").value)
        iou = float(self.get_parameter("iou").value)
        prompts = list(self.get_parameter("text_prompts").get_parameter_value().string_array_value)

        self.get_logger().info(f"Loading YOLOE weights from: {weights}")
        self.inference = YOLOEInference(self, weights, imgsz, conf, iou, prompts)
        return self.inference.model

    # Base will call this during on_activate()
    def create_services(self):
        self.pub_det = self.create_publisher(Detection2DArray, "yoloe/detections", self.qos_reliable)
        self.pub_mask = self.create_publisher(Image, "yoloe/mask_union", self.qos_reliable)
        if bool(self.get_parameter("publish_overlay").value):
            self.pub_overlay = self.create_publisher(Image, "yoloe/overlay", self.qos_reliable)

    # Override base image callback
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            res = self.inference.infer(frame)
        except Exception as e:
            self.get_logger().error(f"YOLOE inference failed: {e}")
            return

        det_arr = self._detections_from_result(res, msg.header)
        self.pub_det.publish(det_arr)

        mask = self._mask_from_result(res, frame.shape[:2])
        ros_mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        ros_mask.header = msg.header
        self.pub_mask.publish(ros_mask)

        if self.pub_overlay is not None:
            overlay = res.plot()
            ros_overlay = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            ros_overlay.header = msg.header
            self.pub_overlay.publish(ros_overlay)

    # utils
    def _detections_from_result(self, res, header) -> Detection2DArray:
        det_arr = Detection2DArray()
        det_arr.header = header
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return det_arr

        names = res.names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().view(-1).cpu().numpy()
        cls_ids = (
            boxes.cls.detach().view(-1).cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.full((len(boxes),), -1, int)
        )

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            w, h = (x2 - x1), (y2 - y1)

            det = Detection2D()
            det.header = header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = _cls_name(names, int(cls_ids[i]))
            hyp.hypothesis.score = float(confs[i])
            det.results.append(hyp)

            bb = BoundingBox2D()
            bb.center = _make_center(cx, cy)
            bb.size_x = float(w)
            bb.size_y = float(h)
            det.bbox = bb

            det_arr.detections.append(det)
        return det_arr

    def _mask_from_result(self, res, hw):
        H, W = hw
        if getattr(res, "masks", None) is None or res.masks is None or len(res.masks) == 0:
            return np.zeros((H, W), dtype=np.uint8)
        md = res.masks.data.detach().cpu().numpy()  # (N,h,w)
        small = (md > 0.5).any(axis=0).astype(np.uint8) * 255
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    rclpy.init()
    node = YoloWrapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
