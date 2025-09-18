#!/usr/bin/env python3
# Code comments in English.

from vlm_base.vlm_base import VLMBaseLifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn, State
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from ultralytics import YOLOE
from yolo_ros_interfaces.srv import SemanticDetection
import rclpy
import torch
import numpy as np
import cv2

# ---------- Inference wrapper ----------
class YOLOEInference:
    def __init__(self, node, weights: str, imgsz: int, conf: float, iou: float):
        self.node = node
        self.model = YOLOE(weights).to(node.device).eval()
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)

    def _to_bgr(self, ros_img: Image):
        return self.node.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")

    def _make_center(self, cx: float, cy: float):
        # Create exact submessage expected by BoundingBox2D.center
        center = type(BoundingBox2D().center)()
        if hasattr(center, "x"):
            center.x = float(cx); center.y = float(cy)
            if hasattr(center, "theta"): center.theta = 0.0
        elif hasattr(center, "position"):
            center.position.x = float(cx); center.position.y = float(cy)
            if hasattr(center, "theta"): center.theta = 0.0
        return center

    def _detections_from_result(self, res, header) -> Detection2DArray:
        det_arr = Detection2DArray(); det_arr.header = header
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return det_arr
        names = res.names

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().view(-1).cpu().numpy()
        cls_ids = boxes.cls.detach().view(-1).cpu().numpy().astype(int) if boxes.cls is not None \
                  else np.full((len(boxes),), -1, int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            w, h = (x2 - x1), (y2 - y1)

            det = Detection2D(); det.header = header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
            hyp.hypothesis.score = float(confs[i])
            det.results.append(hyp)

            bb = BoundingBox2D()
            bb.center = self._make_center(cx, cy)
            bb.size_x = float(w); bb.size_y = float(h)
            det.bbox = bb

            det_arr.detections.append(det)
        return det_arr

    def _mask_from_result(self, res, H: int, W: int) -> np.ndarray:
        # Returns mono8 mask (H,W). If no masks, returns zeros.
        if getattr(res, "masks", None) is None or res.masks is None or len(res.masks) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # res.masks.data: (N,h,w) float in {0,1} or prob. Combine by OR.
        md = res.masks.data.detach().cpu().numpy()  # (N,h,w)
        mask_small = (md > 0.5).any(axis=0).astype(np.uint8) * 255  # (h,w)
        if mask_small.shape[0] != H or mask_small.shape[1] != W:
            mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small
        return mask

    @torch.no_grad()
    def run_text_detect(self, ros_image: Image, prompt: str):
        # 1) Set classes from prompt
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        pe = self.model.get_text_pe(prompts)
        pe3 = pe if pe.dim() == 3 else pe.unsqueeze(0)  # (1,K,D)
        self.model.set_classes(prompts, pe3)

        # 2) Inference
        bgr = self._to_bgr(ros_image)
        res = self.model.predict(
            source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]

        # 3) Build detections
        det_arr = self._detections_from_result(res, ros_image.header)

        # 4) Build mono8 mask
        H, W = bgr.shape[:2]
        mask = self._mask_from_result(res, H, W)

        return det_arr, mask


# ---------- Service handler ----------
class YOLOEServiceHandler:
    def __init__(self, node: VLMBaseLifecycleNode, inference: YOLOEInference):
        self.node = node
        self.inference = inference

    def text_detect(self, request, response):
        try:
            det_arr, mask = self.inference.run_text_detect(request.image, request.text_prompt)
            # segmented_image: mono8
            ros_mask = self.node.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            ros_mask.header = request.image.header
            response.segmented_image = ros_mask
            response.detections = det_arr
            return response
        except Exception as e:
            self.node.get_logger().error(f"text_detect failed: {e}")
            # return empty on failure
            H = getattr(request.image, "height", 0)
            W = getattr(request.image, "width", 0)
            response.segmented_image = self.node.bridge.cv2_to_imgmsg(
                np.zeros((H, W), dtype=np.uint8), encoding="mono8"
            )
            response.segmented_image.header = request.image.header
            response.detections = Detection2DArray()
            response.detections.header = request.image.header
            return response


# ---------- Lifecycle node ----------
class YOLOELifecycleNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__("yoloe_lifecycle_node")
        self._declare_parameters()

    def _declare_parameters(self):
        self.declare_parameter("weights", "yoloe-11l-seg.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.5)

    def load_model(self):
        weights = self.get_parameter("weights").get_parameter_value().string_value
        imgsz = int(self.get_parameter("imgsz").value)
        conf = float(self.get_parameter("conf").value)
        iou = float(self.get_parameter("iou").value)
        self.inference = YOLOEInference(self, weights, imgsz, conf, iou)
        return self.inference.model  # signals success to base

    def create_services(self):
        self.handler = YOLOEServiceHandler(self, self.inference)
        self.create_service(SemanticDetection, "yoloe_text_detect", self.handler.text_detect)


# ---------- main ----------
def main():
    rclpy.init()
    node = YOLOELifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
