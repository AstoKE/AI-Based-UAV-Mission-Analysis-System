from ultralytics import YOLO
import numpy as np
import time

from vision.postprocess import dedup_detections, avg_confidence

class YoloV8Detector:
    def __init__(self, model_name: str, conf: float, iou: float, imgsz:int=1920):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.imgsz= imgsz

    def detect(self, image_bgr: np.ndarray, dedup_iou: float = 0.65):
        t0 = time.perf_counter()

        # returns a list of detections
        results = self.model.predict(
            source = image_bgr,
            conf=self.conf,
            iou = self.iou,
            imgsz=self.imgsz,
            classes=[0,1,2,3,5,7],
            agnostic_nms =True,
            max_det=300,
            verbose=False
        )

        t1 = time.perf_counter()

        r0 = results[0]
        names = r0.names
        dets = []

        if r0.boxes is not None:
            for b in r0.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "conf": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        before = len(dets)
        dets_dedup, stats = dedup_detections(dets, iou_th=dedup_iou)

        metrics = {
            "inference_ms": (t1 - t0) * 1000.0,
            "detections_before": before,
            "detections_after": stats.after,
            "duplicate_ratio": stats.duplicate_ratio,
            "avg_conf_before": avg_confidence(dets),
            "avg_conf_after": avg_confidence(dets_dedup),
        }

        return dets_dedup, metrics
