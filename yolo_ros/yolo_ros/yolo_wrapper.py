#!/usr/bin/env python3
# yolo_wrapper_lifecycle.py
# Lifecycle node: subscribe /rgb, run YOLOE, publish overlay + Detection2DArray.
# Adds subscriber /user_prompt (SemanticPrompt) using only text_query.

from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

from multimodal_query_msgs.msg import SemanticPrompt  # adjust if your package differs

import numpy as np
import cv2
from ultralytics import YOLOE


# ---------- path + download ----------
def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


class PathManager:
    """Relative only: <this_file>/../models/<backend>/<weights>."""
    def __init__(self):
        self.models_root = Path(__file__).resolve().parents[1] / "models"

    def ensure_weights(self, backend: str, weights_name: str, url: str | None) -> str:
        dst = (self.models_root / backend / Path(weights_name).name).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            return str(dst)
        src = url if url else (weights_name if _is_url(weights_name) else None)
        if src:
            urllib.request.urlretrieve(src, dst)
            return str(dst)
        return weights_name


# ---------- node ----------
class YoloWrapper(LifecycleNode):
    def __init__(self):
        super().__init__("yolo_wrapper")
        self.bridge = CvBridge()
        self.pm = PathManager()

        self.model = None
        self.sub_rgb = None
        self.sub_prompt = None
        self.pub_overlay = None
        self.pub_det = None

        self._declare_params()
        self._init_param_cache()

        self.qos_cam = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.qos_rel = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

    # ---- params ----
    def _declare_params(self):
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("backend", "yoloe")
        self.declare_parameter("weights", "yoloe-11l-seg.pt")
        self.declare_parameter("download_url", "")
        self.declare_parameter("prompts", ["oven", "chair", "tv", "bed", "door"])
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.50)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("publish_detections", True)

        # prompt subscriber
        self.declare_parameter("prompt_topic", "/user_prompt")
        self.declare_parameter("prompt_delimiter", ",")

    def _init_param_cache(self):
        self.rgb_topic = "/rgb"
        self.backend = "yoloe"
        self.weights_name = "yoloe-11l-seg.pt"
        self.download_url = None
        self.prompts = []
        self.imgsz = 640
        self.conf = 0.25
        self.iou = 0.50
        self.publish_overlay = True
        self.publish_detections = True
        self.prompt_topic = "/user_prompt"
        self.prompt_delim = ","

    def _read_params(self):
        p = self.get_parameter
        self.rgb_topic = p("rgb_topic").get_parameter_value().string_value
        self.backend = p("backend").get_parameter_value().string_value
        self.weights_name = p("weights").get_parameter_value().string_value
        dl = p("download_url").get_parameter_value().string_value
        self.download_url = dl if dl else None
        self.prompts = list(p("prompts").get_parameter_value().string_array_value)
        self.imgsz = int(p("imgsz").value)
        self.conf = float(p("conf").value)
        self.iou = float(p("iou").value)
        self.publish_overlay = bool(p("publish_overlay").value)
        self.publish_detections = bool(p("publish_detections").value)
        self.prompt_topic = p("prompt_topic").get_parameter_value().string_value
        self.prompt_delim = p("prompt_delimiter").get_parameter_value().string_value or ","

    # ---- lifecycle ----
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        try:
            self._read_params()
            weights_path = self.pm.ensure_weights(self.backend, self.weights_name, self.download_url)
            self.get_logger().info(f"Resolved weights: {weights_path}")

            self.model = YOLOE(weights_path)
            if self.prompts:
                self._apply_prompts(self.prompts)

            if self.publish_overlay:
                self.pub_overlay = self.create_publisher(Image, "yoloe/overlay", self.qos_rel)
            if self.publish_detections:
                self.pub_det = self.create_publisher(Detection2DArray, "yoloe/detections", self.qos_rel)

            self.get_logger().info("Configured.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        try:
            self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.on_image, self.qos_cam)
            self.sub_prompt = self.create_subscription(SemanticPrompt, self.prompt_topic, self.on_prompt, self.qos_rel)
            self.get_logger().info(f"Activated. rgb={self.rgb_topic}, prompts={self.prompt_topic}")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"activate failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        try:
            for s in (self.sub_rgb, self.sub_prompt):
                if s: self.destroy_subscription(s)
            self.sub_rgb = self.sub_prompt = None
            for p in (self.pub_overlay, self.pub_det):
                if p: self.destroy_publisher(p)
            self.pub_overlay = self.pub_det = None
            self.get_logger().info("Deactivated.")
            return TransitionCallbackReturn.SUCCESS
        except Exception:
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        try:
            self.model = None
            self.get_logger().info("Cleaned up.")
            return TransitionCallbackReturn.SUCCESS
        except Exception:
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down.")
        return TransitionCallbackReturn.SUCCESS

    def _parse_prompt_text(self, text: str) -> list[str]:
        # keep whole phrases separated by commas
        if not text:
            return []
        # normalize common comma variants
        text = text.replace("，", ",")
        parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
        return [p for p in parts if p]

    def on_prompt(self, msg: SemanticPrompt):
        text = (msg.text_query or "").strip()
        prompts = self._parse_prompt_text(text)
        if not prompts:
            self.get_logger().info("Prompt empty or invalid. Ignored.")
            return
        self._apply_prompts(prompts)
        self.prompts = prompts
        self.get_logger().info(f"Prompts updated: {prompts}")

    def _apply_prompts(self, prompts: list[str]):
        try:
            pe = self.model.get_text_pe(prompts)
            pe3 = pe if pe.dim() == 3 else pe.unsqueeze(0)
            self.model.set_classes(prompts, pe3)
        except Exception as e:
            self.get_logger().warn(f"set_classes failed: {e}")

    # ---- image callback ----
    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            res = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
        except Exception as e:
            self.get_logger().error(f"YOLOE inference failed: {e}")
            return

        if self.pub_overlay:
            overlay = res.plot()
            out_img = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            out_img.header = msg.header
            self.pub_overlay.publish(out_img)

        if self.pub_det:
            det_arr = Detection2DArray(); det_arr.header = msg.header
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                names = res.names
                for b in boxes:
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    w, h = (x2 - x1), (y2 - y1)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else -1
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                    det = Detection2D(); det.header = msg.header
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(cls_name)
                    hyp.hypothesis.score = conf
                    det.results.append(hyp)

                    det.bbox = BoundingBox2D()
                    c = det.bbox.center
                    if hasattr(c, "x"):
                        c.x, c.y = float(cx), float(cy)
                        if hasattr(c, "theta"):
                            c.theta = 0.0
                    elif hasattr(c, "position"):
                        c.position.x, c.position.y = float(cx), float(cy)
                        det.bbox.center = c
                    det.bbox.size_x = float(w); det.bbox.size_y = float(h)

                    det_arr.detections.append(det)

            self.pub_det.publish(det_arr)


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
