from ultralytics import YOLO
import numpy as np


class YoloV8Detector:
    def __init__(self, model_name: str, conf: float, iou: float):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def detect(self, image_bgr: np.ndarray):

        # returns a list of detections
        results = self.model.predict(
            source = image_bgr,
            conf=self.conf,
            iou = self.iou,
            verbose=False
        )

        r0 = results[0]
        names = r0.names
        dets = []

        if r0.boxes is None:
            return dets

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

        return dets

