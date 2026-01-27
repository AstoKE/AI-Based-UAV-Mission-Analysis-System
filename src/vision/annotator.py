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


_RISK_COLORS = {
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 215, 255),
    "HIGH": (0, 0, 255),
}

def draw_hud(img_bgr, risk_level: str, risk_score: int, inference_ms: float, fps: float):
    out = img_bgr
    h, w = out.shape[:2]
    color = _RISK_COLORS.get(risk_level, (255, 255, 255))

    pad = 10
    panel_h = 55
    cv2.rectangle(out, (pad, pad), (w - pad, pad + panel_h), (0, 0, 0), thickness=-1)

    left = f"RISK: {risk_level} ({int(risk_score)})"
    right = f"FPS: {fps:.1f} | inf: {inference_ms:.0f} ms"

    cv2.putText(out, left, (pad + 10, pad + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    ts = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    rx = w - pad - 10 - ts[0]
    cv2.putText(out, right, (rx, pad + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out
