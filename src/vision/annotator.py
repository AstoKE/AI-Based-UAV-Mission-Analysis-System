import cv2
import numpy as np

def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image_bgr.copy()

    for d in detections:
        x1,y1,x2,y2 = map(int,d["bbox_xyxy"])
        label = f'{d["class_name"]} {d["conf"]:.2f}'
        cv2.rectangle(out, (x1,y1), (x2,y2), (0, 255, 0),2)
        cv2.putText(out,label,(x1,max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return out    
