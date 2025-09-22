#!/usr/bin/env python3
# detector_lifecycle_node.py
# Extra logging everywhere.

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from multimodal_query_msgs.msg import SemanticPrompt

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


# ---------------- Path resolution ----------------


class PathManager:
    """Relative only: <this_file>/../models/<backend>/<weights>."""
    def __init__(self):
        self.models_root = Path(__file__).resolve().parents[1] / "models"

    def local_candidate(self, backend: str, weights_name: str) -> Path:
        return (self.models_root / backend / weights_name).resolve()

    def ensure_weights(self, backend: str, weights_name: str, url: str | None) -> str:
        dst = self.local_candidate(backend, Path(weights_name).name)
        dst.parent.mkdir(parents=True, exist_ok=True)

        # already present
        if dst.is_file():
            return str(dst)

        # decide source URL
        src_url = url if url else (weights_name if _is_url(weights_name) else None)
        if src_url:
            print(f"[PathManager] downloading '{src_url}' → '{dst}'")
            urllib.request.urlretrieve(src_url, dst)  # raises on failure
            return str(dst)

        # no URL given; fall back to input string (Ultralytics may resolve)
        print(f"[PathManager] not found locally, no URL given. Using raw path/name: '{weights_name}'")
        return weights_name



# ---------------- Utilities ----------------

def _cls_name(names, idx: int) -> str:
    if isinstance(names, dict):
        return names.get(idx, str(idx))
    if isinstance(names, (list, tuple)):
        return names[idx] if 0 <= idx < len(names) else str(idx)
    return str(idx)

def _center_msg(cx: float, cy: float):
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

def _draw_overlay(bgr: np.ndarray, dets: Detection2DArray) -> np.ndarray:
    out = bgr.copy()
    for det in dets.detections:
        bb = det.bbox
        cx = float(getattr(bb.center, "x", 0.0))
        cy = float(getattr(bb.center, "y", 0.0))
        w = float(bb.size_x); h = float(bb.size_y)
        x1 = int(max(0, cx - w/2)); y1 = int(max(0, cy - h/2))
        x2 = int(min(out.shape[1]-1, cx + w/2))
        y2 = int(min(out.shape[0]-1, cy + h/2))
        label, score, tid = "", 0.0, None
        if det.results:
            label = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score
            for r in det.results:
                if r.hypothesis.class_id.startswith("track:"):
                    tid = r.hypothesis.class_id.split(":", 1)[1]
                    break
        txt = f"{label} {score:.2f}" + (f" id:{tid}" if tid is not None else "")
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(out, txt, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return out


# ---------------- Backend result and adapters ----------------

class DetectResult:
    def __init__(self,
                 xyxy: np.ndarray,
                 conf: np.ndarray,
                 cls_ids: np.ndarray,
                 names,
                 masks: Optional[np.ndarray] = None,
                 track_ids: Optional[np.ndarray] = None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls_ids = cls_ids
        self.names = names
        self.masks = masks
        self.track_ids = track_ids

class ModelAdapter:
    def load(self, weights_path: str) -> None: ...
    def set_prompts(self, prompts: List[str]) -> None: ...
    def infer(self, bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> DetectResult: ...

class YOLOEAdapter(ModelAdapter):
    def __init__(self, node, use_tracking: bool, tracker_cfg: Optional[str]):
        from ultralytics import YOLOE as _YOLOE
        self._YOLOE = _YOLOE
        self.node = node
        self.model = None
        self.use_tracking = bool(use_tracking)
        self.tracker_cfg = tracker_cfg if tracker_cfg else "bytetrack.yaml"

    def load(self, weights_path: str) -> None:
        t0 = time.perf_counter()
        self.model = self._YOLOE(weights_path).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        dt = (time.perf_counter() - t0) * 1000
        self.node.get_logger().info(f"[YOLOE] loaded on {dev} in {dt:.1f} ms")

    def set_prompts(self, prompts: List[str]) -> None:
        try:
            if prompts:
                pe = self.model.get_text_pe(prompts)
                pe3 = pe if pe.dim() == 3 else pe.unsqueeze(0)
                self.model.set_classes(prompts, pe3)
                self.node.get_logger().info(f"[YOLOE] prompts set: {prompts}")
            else:
                self.node.get_logger().info("[YOLOE] prompts cleared")
        except Exception as e:
            self.node.get_logger().warn(f"[YOLOE] set_prompts skipped: {e}")

    @torch.no_grad()
    def infer(self, bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> DetectResult:
        t0 = time.perf_counter()
        if self.use_tracking:
            res = self.model.track(source=bgr, imgsz=imgsz, conf=conf, iou=iou,
                                   persist=True, tracker=self.tracker_cfg, verbose=False)[0]
        else:
            res = self.model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        boxes = res.boxes
        n = int(len(boxes) if boxes is not None else 0)
        # self.node.get_logger().info(f"[YOLOE] infer {imgsz=} {conf:.2f} {iou:.2f} -> {n} dets in {dt:.1f} ms")

        if boxes is None or len(boxes) == 0:
            return DetectResult(
                xyxy=np.zeros((0,4), np.float32),
                conf=np.zeros((0,), np.float32),
                cls_ids=np.zeros((0,), np.int32),
                names=res.names,
                masks=None,
                track_ids=None
            )
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().view(-1).cpu().numpy().astype(np.float32)
        cls_ids = (boxes.cls.detach().view(-1).cpu().numpy().astype(np.int32)
                   if boxes.cls is not None else np.full((len(boxes),), -1, np.int32))
        masks = None
        if getattr(res, "masks", None) is not None and res.masks is not None and len(res.masks) > 0:
            masks = res.masks.data.detach().cpu().numpy().astype(np.float32)
            self.node.get_logger().info(f"[YOLOE] masks: {masks.shape}")
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.detach().view(-1).cpu().numpy().astype(np.int32)
            self.node.get_logger().info(f"[YOLOE] tracks: {track_ids.tolist()}")
        return DetectResult(xyxy, confs, cls_ids, res.names, masks, track_ids)

class YOLOv7Adapter(ModelAdapter):
    def __init__(self, node, use_tracking: bool, tracker_cfg: Optional[str]):
        self.node = node
        self.use_tracking = bool(use_tracking)
        self.tracker_cfg = tracker_cfg
        self.model = None
    def load(self, weights_path: str) -> None:
        self.node.get_logger().error("YOLOv7Adapter not implemented.")
        raise NotImplementedError
    def set_prompts(self, prompts: List[str]) -> None:
        pass
    def infer(self, bgr: np.ndarray, imgsz: int, conf: float, iou: float) -> DetectResult:
        raise NotImplementedError


# ---------------- Lifecycle base node ----------------

class DetectorLifecycleBase(LifecycleNode):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.pm = None
        self.adapter: Optional[ModelAdapter] = None

        # Params
        self.declare_parameter("backend", "yoloe")
        self.declare_parameter("weights", "yoloe-11l-seg.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("publish_overlay", True)

        # Prompts
        self.declare_parameter("prompt_topic", "/user_prompt")
        self.declare_parameter("prompt_delimiter", ",")
        self.declare_parameter("promptless_infer", True)
        self.declare_parameter("prompts", [])

        # Tracking
        self.declare_parameter("use_tracking", False)
        self.declare_parameter("tracker_cfg", "")

        # Mask config
        self.declare_parameter("score_thresh", 0.25)
        self.declare_parameter("mask_prob_thresh", 0.5)

        # Debug cadence
        self.declare_parameter("debug_every_n", 30)
        self.declare_parameter("download_url", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt")  # optional direct URL to checkpoint

        # IO
        self.sub_rgb = None
        self.sub_prompt = None
        self.pub_det = None
        self.pub_mask = None
        self.pub_overlay = None

        # State
        self.current_prompts: List[str] = list(self.get_parameter("prompts").get_parameter_value().string_array_value)
        self.frame_idx = 0

        # QoS
        self.qos_best_effort = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                          reliability=ReliabilityPolicy.BEST_EFFORT)
        self.qos_reliable = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST,
                                       reliability=ReliabilityPolicy.RELIABLE)

    # ---- lifecycle ----
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        try:
            self.pm = PathManager()
            backend = self.get_parameter("backend").get_parameter_value().string_value
            weights_name = self.get_parameter("weights").get_parameter_value().string_value
            download_url = self.get_parameter("download_url").get_parameter_value().string_value
            weights_path = self.pm.ensure_weights(backend, weights_name, download_url or None)
            self.get_logger().info(f"models_root={self.pm.models_root}")
            self.get_logger().info(f"Resolved weights: {weights_path}")
            
            imgsz = int(self.get_parameter("imgsz").value)
            conf = float(self.get_parameter("conf").value)
            iou = float(self.get_parameter("iou").value)
            prompts = list(self.get_parameter("prompts").get_parameter_value().string_array_value)
            use_tracking = bool(self.get_parameter("use_tracking").value)
            tracker_cfg = self.get_parameter("tracker_cfg").get_parameter_value().string_value

            self.get_logger().info(f"Config params: backend={backend}, imgsz={imgsz}, conf={conf}, iou={iou}, "
                                   f"use_tracking={use_tracking}, tracker={tracker_cfg or 'bytetrack.yaml'}")
            self.get_logger().info(f"Resolved weights: {weights_path}")

            if backend.lower() == "yoloe":
                self.adapter = YOLOEAdapter(self, use_tracking, tracker_cfg)
            elif backend.lower() == "yolov7":
                self.adapter = YOLOv7Adapter(self, use_tracking, tracker_cfg)
            else:
                self.get_logger().error(f"Unsupported backend '{backend}'")
                return TransitionCallbackReturn.FAILURE

            self.adapter.load(weights_path)
            if prompts:
                self._apply_prompts(prompts)

            self.pub_det = self.create_publisher(Detection2DArray, "yolo/detections", self.qos_reliable)
            self.pub_mask = self.create_publisher(Image, "yolo/score_mask", self.qos_reliable)
            if bool(self.get_parameter("publish_overlay").value):
                self.pub_overlay = self.create_publisher(Image, "yolo/overlay", self.qos_reliable)

            self.get_logger().info("Configured.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        try:
            rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
            prompt_topic = self.get_parameter("prompt_topic").get_parameter_value().string_value

            self.sub_rgb = self.create_subscription(Image, rgb_topic, self._on_image, self.qos_best_effort)
            self.sub_prompt = self.create_subscription(SemanticPrompt, prompt_topic, self._on_prompt, self.qos_reliable)

            self.get_logger().info(f"Activated. Subscribing rgb={rgb_topic}, prompts={prompt_topic}")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"activate failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        try:
            for s in (self.sub_rgb, self.sub_prompt):
                if s: self.destroy_subscription(s)
            self.sub_rgb = self.sub_prompt = None
            self.get_logger().info("Deactivated.")
            return TransitionCallbackReturn.SUCCESS
        except Exception:
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        try:
            for p in (self.pub_det, self.pub_mask, self.pub_overlay):
                if p: self.destroy_publisher(p)
            self.pub_det = self.pub_mask = self.pub_overlay = None
            self.adapter = None
            self.get_logger().info("Cleaned up.")
            return TransitionCallbackReturn.SUCCESS
        except Exception:
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down.")
        return TransitionCallbackReturn.SUCCESS

    # ---- prompt subscriber ----
    def _on_prompt(self, msg: SemanticPrompt):
        delim = self.get_parameter("prompt_delimiter").get_parameter_value().string_value or ","
        txt = (msg.text_query or "").strip()
        prompts = [s.strip() for s in (txt.split(delim) if delim in txt else txt.split()) if s.strip()]
        self.current_prompts = prompts
        self._apply_prompts(prompts)
        self.get_logger().info(f"Prompts updated: {prompts}")

    def _apply_prompts(self, prompts: List[str]):
        try:
            if self.adapter:
                self.adapter.set_prompts(prompts)
        except Exception as e:
            self.get_logger().warn(f"set_prompts ignored: {e}")

    # ---- image callback ----
    def _on_image(self, msg: Image):
        self.frame_idx += 1
        if self.adapter is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        h, w = bgr.shape[:2]
        if (self.frame_idx % max(1, int(self.get_parameter('debug_every_n').value))) == 1:
            self.get_logger().info(f"Frame {self.frame_idx}: {w}x{h}, prompts={self.current_prompts}")

        promptless_ok = bool(self.get_parameter("promptless_infer").value)
        if not self.current_prompts and not promptless_ok:
            if (self.frame_idx % max(1, int(self.get_parameter('debug_every_n').value))) == 1:
                self.get_logger().info("Skipping inference: no prompts and promptless_infer=false")
            return

        imgsz = int(self.get_parameter("imgsz").value)
        conf = float(self.get_parameter("conf").value)
        iou = float(self.get_parameter("iou").value)
        score_thresh = float(self.get_parameter("score_thresh").value)
        prob_thresh = float(self.get_parameter("mask_prob_thresh").value)

        try:
            dr = self.adapter.infer(bgr, imgsz=imgsz, conf=conf, iou=iou)
        except Exception as e:
            self.get_logger().error(f"infer failed: {e}")
            return

        det_msg = self._to_detection_array(dr, msg.header)
        try:
            self.pub_det.publish(det_msg)
            # self.get_logger().info(f"Published detections: {len(det_msg.detections)}")
        except Exception as e:
            pass
            # self.get_logger().error(f"publish detections failed: {e}")

        mask_img = self._make_score_mask(dr, bgr.shape[:2], score_thresh, prob_thresh)
        try:
            ros_mask = self.bridge.cv2_to_imgmsg(mask_img, "mono8"); ros_mask.header = msg.header
            self.pub_mask.publish(ros_mask)
            # self.get_logger().info(f"Published score_mask: {mask_img.shape}, nonzero={int((mask_img>0).sum())}")
        except Exception as e:
            pass
            # self.get_logger().error(f"publish mask failed: {e}")

        if self.pub_overlay:
            try:
                overlay = _draw_overlay(bgr, det_msg)
                ros_overlay = self.bridge.cv2_to_imgmsg(overlay, "bgr8"); ros_overlay.header = msg.header
                self.pub_overlay.publish(ros_overlay)
                # self.get_logger().info("Published overlay.")
            except Exception as e:
                self.get_logger().error(f"publish overlay failed: {e}")

    # ---- conversions ----
    @staticmethod
    def _to_detection_array(dr: DetectResult, header: Header) -> Detection2DArray:
        det_arr = Detection2DArray(); det_arr.header = header
        N = dr.xyxy.shape[0]
        for i in range(N):
            x1, y1, x2, y2 = dr.xyxy[i]
            cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
            w, h = (x2-x1), (y2-y1)

            det = Detection2D(); det.header = header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = _cls_name(dr.names, int(dr.cls_ids[i]))
            hyp.hypothesis.score = float(dr.conf[i])
            det.results.append(hyp)

            if dr.track_ids is not None:
                tid = int(dr.track_ids[i])
                hyp_tid = ObjectHypothesisWithPose()
                hyp_tid.hypothesis.class_id = f"track:{tid}"
                hyp_tid.hypothesis.score = 1.0
                det.results.append(hyp_tid)

            bb = BoundingBox2D()
            bb.center = _center_msg(cx, cy)
            bb.size_x = float(w); bb.size_y = float(h)
            det.bbox = bb

            det_arr.detections.append(det)
        return det_arr

    @staticmethod
    def _make_score_mask(dr: DetectResult, hw: Tuple[int,int],
                         score_thresh: float, prob_thresh: float) -> np.ndarray:
        H, W = hw
        out = np.zeros((H, W), np.float32)
        N = dr.xyxy.shape[0]
        if N == 0:
            return np.zeros((H, W), np.uint8)

        if dr.masks is not None and dr.masks.size > 0:
            for i in range(N):
                s = float(dr.conf[i])
                if s < score_thresh:
                    continue
                m_small = dr.masks[i]
                m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_LINEAR)
                pix = (m >= prob_thresh)
                out[pix] = np.maximum(out[pix], s)
        else:
            for i in range(N):
                s = float(dr.conf[i])
                if s < score_thresh:
                    continue
                x1, y1, x2, y2 = dr.xyxy[i].astype(int)
                x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
                y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
                out[y1:y2+1, x1:x2+1] = np.maximum(out[y1:y2+1, x1:x2+1], s)

        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


# ---------------- Concrete node ----------------

class YoloWrapper(DetectorLifecycleBase):
    def __init__(self):
        super().__init__("detector_lifecycle_node")


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
