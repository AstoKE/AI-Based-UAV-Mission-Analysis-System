from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any
import cv2
import time
import subprocess

from analytics.risk import compute_risk
from nlp.report_generator import generate_report_en
from vision.annotator import draw_detections
from vision.annotator import draw_hud


@dataclass
class VideoSummary:
    frames_total: int
    frames_analyzed: int
    avg_inference_ms: float
    avg_conf_after: float


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _transcode_to_mp4(input_path: str, output_path: str) -> None:
    """
    Browser-friendly MP4:
      - H.264 (libx264)
      - yuv420p pixel format (Chrome compatible)
      - faststart (moov atom at beginning)
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def analyze_video(
    video_path: str,
    detector,
    fps_sample: int = 1,
    max_seconds: int = 20,
    report_every_sec: int = 1,
) -> Tuple[str, Dict[str, Any], VideoSummary, str]:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_sample <= 0:
        step = 1
    else:
        step = max(1, int(round(fps / fps_sample)))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    in_path = Path(video_path)

    tmp_avi_path = str(in_path.with_suffix("").as_posix() + "_annotated_tmp.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_avi_path, fourcc, fps, (width, height))

    frames_total = 0
    frames_analyzed = 0
    inf_ms_list = []
    conf_list = []

    last_risk_payload = {
        "counts": {},
        "risk_level": "LOW",
        "risk_score": 0,
        "recommended_action": "Routine monitoring recommended.",
    }
    last_report = ""
    last_risk = {
        "risk_level": "LOW",
        "risk_score": 0,
    }
    last_inf_ms = 0.0
    start_wall = time.time()
    last_report_wall = 0.0
    idx = 0
    last_dets = []
    have_last = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_total += 1

        # stop after max_seconds
        if time.time() - start_wall > max_seconds:
            break


        if idx % step == 0:
            dets, metrics = detector.detect(frame)

            last_inf_ms = metrics.get("inference_ms", 0.0)

            risk_payload = compute_risk(dets)
            last_risk = {
                "risk_level": risk_payload.get("risk_level", "LOW"),
                "risk_score": risk_payload.get("risk_score", 0),
            }

            inf_ms_list.append(metrics.get("inference_ms", 0.0))
            conf_list.append(metrics.get("avg_conf_after", 0.0))

            risk_payload = compute_risk(dets)
            last_risk_payload = risk_payload

            now = time.time()
            if now - last_report_wall >= report_every_sec:
                mission_id = f"vid-{in_path.stem}"
                last_report = generate_report_en(mission_id=mission_id, risk_payload=risk_payload)
                last_report_wall = now

            last_dets = dets
            have_last = True
            frames_analyzed += 1

        if have_last:
            ann = draw_detections(frame, last_dets)

            ann = draw_hud(
                ann,
                risk_level=last_risk.get("risk_level", "LOW"),
                risk_score=last_risk.get("risk_score", 0),
                inference_ms=last_inf_ms,
                fps=fps,
            )

            writer.write(ann)
        else:
            writer.write(frame)

        idx += 1

    cap.release()
    writer.release()

    avg_inf = float(sum(inf_ms_list) / max(1, len(inf_ms_list)))
    avg_conf = float(sum(conf_list) / max(1, len(conf_list)))

    summary = VideoSummary(
        frames_total=frames_total,
        frames_analyzed=frames_analyzed,
        avg_inference_ms=avg_inf,
        avg_conf_after=avg_conf,
    )

    out_mp4_path = str(in_path.with_suffix("").as_posix() + "_annotated.mp4")

    final_path = tmp_avi_path
    if _ffmpeg_available():
        try:
            _transcode_to_mp4(tmp_avi_path, out_mp4_path)
            # only switch if mp4 exists and non-empty
            p = Path(out_mp4_path)
            if p.exists() and p.stat().st_size > 0:
                final_path = out_mp4_path
        except Exception:
            final_path = tmp_avi_path

    return final_path, last_risk_payload, summary, last_report
