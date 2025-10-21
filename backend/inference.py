import time
import numpy as np
import torch
from ultralytics import YOLO
import logging
from typing import List, Dict, Any

logger = logging.getLogger("inference")

class YOLOWrapper:
    """
    Lightweight wrapper for YOLOv8 model (Ultralytics).
    Loads a model and runs inference on frames (numpy BGR).
    """
    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = None, conf_thresh: float = 0.25):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        logger.info(f"Loading YOLO model {model_path} on {self.device}")
        self.model = YOLO(model_path)
        # model returns CPU tensors by default; ensure uses GPU if available
        self.model.to(self.device)
        self.names = {int(k): v for k,v in self.model.names.items()}

    def infer(self, frame: np.ndarray, conf: float = None, classes: List[int] = None) -> Dict[str, Any]:
        """
        Run inference and return structured detections:
        [{xmin, ymin, xmax, ymax, conf, cls_id, cls_name}, ...]
        """
        conf = conf if conf is not None else self.conf_thresh
        t0 = time.time()
        results = self.model(frame, conf=conf, verbose=False)  # Ultralytics returns a Results object list
        t1 = time.time()
        res = results[0]
        detections = []
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clsids = res.boxes.cls.cpu().numpy().astype(int)
            for box, c, cid in zip(boxes, confs, clsids):
                if c < conf:
                    continue
                if classes and cid not in classes:
                    continue
                detections.append({
                    "xmin": int(box[0]), "ymin": int(box[1]),
                    "xmax": int(box[2]), "ymax": int(box[3]),
                    "conf": float(c), "cls": int(cid), "name": self.names.get(int(cid), str(int(cid)))
                })
        return {"detections": detections, "inference_time": (t1 - t0)}
