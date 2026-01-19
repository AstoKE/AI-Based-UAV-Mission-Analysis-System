from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any
import cv2
import time

from analytics.risk import compute_risk
from nlp.report_generator import generate_report_en
from vision.annotator import draw_detections



@dataclass
class videoSummary:
    frames_total: int
    frames_analyzed: int
    avg_inference_ms: float
    avg_conf_after: float

def analyze_video(
        video_path: str,
        detector,
        fps_sample: int=1,
        max_seconds: int=20,
        report_every_sec: int=1,
) -> Tuple[str, Dict[str, Any], videoSummary,str]:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps/max(1, fps_sample))))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    in_path = Path(video_path)
    out_video_path = str(in_path.with_suffix("").as_posix() + "_annotated.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer= cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frames_total = 0
    frames_analyzed = 0
    inf_ms_list = []
    conf_list = []

    last_risk_payload = {"counts": {}, "risk_level": "LOW", "risk_score": 0, "recommended_action": "Routine monitoring recommended."}
    last_report = ""

    start_wall = time.time()
    last_report_wall = 0.0
    idx = 0

    while True:
        ok, frame=cap.read()
        if not ok:
            break

        frames_total +=1

        if time.time() - start_wall > max_seconds:
            break

        if idx % step== 0:
            dets, metrics=detector.detect(frame)

            inf_ms_list.append(metrics.get("inference_ms", 0.0))
            conf_list.append(metrics.get("avg_conf_after", 0.0))

            risk_payload = compute_risk(dets)
            last_risk_payload=risk_payload


            now=time.time()
            if now - last_report_wall>= report_every_sec:
                mission_id = f"vid-{in_path.stem}"
                last_report=generate_report_en(mission_id=mission_id, risk_payload=risk_payload)
                last_report_wall=now

            ann = draw_detections(frame, dets)
            writer.write(ann)
            frames_analyzed +=1
        else:
            writer.write(frame)

        idx+=1

    cap.release()
    writer.release()

    avg_inf = float(sum(inf_ms_list) / max(1, len(inf_ms_list)))
    avg_conf = float(sum(conf_list) / max(1, len(conf_list)))

    summary = videoSummary(
        frames_total=frames_total,
        frames_analyzed=frames_analyzed,
        avg_inference_ms=avg_inf,
        avg_conf_after=avg_conf,
    )

    return out_video_path, last_risk_payload, summary, last_report