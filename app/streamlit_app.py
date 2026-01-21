import streamlit as st
from pathlib import Path
import uuid
import json
import numpy as np
import pandas as pd
import cv2
import sys

# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import CFG
from vision.detector import YoloV8Detector
from vision.annotator import draw_detections
from analytics.risk import compute_risk
from analytics.logger import append_jsonl
from nlp.report_generator import generate_report_en
from video.video_analyzer import analyze_video

st.set_page_config(page_title="UAV Mission Analysis", layout="wide")

st.title("AI-Based UAV Mission Analysis System")
st.caption("YOLOv8 + Risk Assessment + NLP Mission Report Output + Mission Logging (JSONL)")

# output directories check
CFG.OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)
CFG.OUTPUT_REP_DIR.mkdir(parents=True, exist_ok=True)
CFG.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_detector(model_name: str, conf: float, iou: float, imgsz: int):
    return YoloV8Detector(model_name, conf, iou, imgsz=imgsz)

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def sample_frames(video_path: str, n: int = 6) -> list[tuple[int, np.ndarray]]:
    """Return list of (frame_index, frame_rgb) sampled across the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        idxs = list(range(n))
    else:
        idxs = np.linspace(0, max(0, total - 1), n, dtype=int).tolist()

    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append((int(fi), bgr_to_rgb(frame)))

    cap.release()
    return frames


# sidebar
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "YOLO Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
)

conf = st.sidebar.slider("Confidence Threshold", 0.05, 0.8, float(CFG.CONF_THRES), 0.05)
iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9, float(CFG.IOU_THRES), 0.05)
imgsz = st.sidebar.selectbox("Inference Image Size", [640, 960, 1280, 1536, 1920], index=2)

detector = load_detector(model_choice, conf, iou, imgsz)

st.sidebar.divider()
st.sidebar.write("Outputs")
st.sidebar.write(f"Annotated: {CFG.OUTPUT_ANN_DIR}")
st.sidebar.write(f"Reports: {CFG.OUTPUT_REP_DIR}")
st.sidebar.write(f"Logs: {CFG.LOG_PATH}")

st.sidebar.divider()
st.sidebar.subheader("Video Demo Settings")
fps_sample = st.sidebar.selectbox("Sample (frames/sec)", [1, 2, 3], index=0)
max_seconds = st.sidebar.selectbox("Max processing (sec)", [10, 20, 30], index=1)
report_every_sec = st.sidebar.selectbox("Report every (sec)", [1, 2, 3], index=0)


# main image demo
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) Upload UAV Image")
    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_uploader")

    use_sample = st.checkbox("Use sample image (assets/samples/perimeter_01.jpg)", value=False)

    img_bgr = None
    img_name = None

    if use_sample:
        sample_path = CFG.SAMPLE_DIR / "perimeter_01.jpg"
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            img_name = sample_path.name
        else:
            st.warning("Sample image not found at assets/samples/perimeter_01.jpg")

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_name = uploaded.name

    if img_bgr is not None:
        st.image(bgr_to_rgb(img_bgr), caption=f"Input: {img_name}", use_container_width=True)

with col2:
    st.subheader("2) Run Analysis")
    run_btn = st.button("Run YOLOv8 + Risk + Report", type="primary", use_container_width=True)

    if run_btn:
        if img_bgr is None:
            st.error("Please upload an image (or enable sample).")
        else:
            with st.spinner("Running inference..."):
                mission_id = str(uuid.uuid4())[:8]
                detections, metrics = detector.detect(img_bgr)

                risk_payload = compute_risk(detections)
                report = generate_report_en(mission_id=mission_id, risk_payload=risk_payload)

                annotated = draw_detections(img_bgr, detections)

                out_img = CFG.OUTPUT_ANN_DIR / f"{Path(img_name).stem}_ann_{mission_id}.jpg"
                out_rep = CFG.OUTPUT_REP_DIR / f"{Path(img_name).stem}_report_{mission_id}.txt"

                cv2.imwrite(str(out_img), annotated)
                out_rep.write_text(report, encoding="utf-8")

                log_payload = {
                    "mission_id": mission_id,
                    "scenario": CFG.SCENARIO_NAME,
                    "input_image": img_name,
                    "model": model_choice,
                    "conf": conf,
                    "iou": iou,
                    "detections": detections,
                    **risk_payload,
                    "log_metrics": metrics,
                }
                append_jsonl(CFG.LOG_PATH, log_payload)

            st.success("Done!")

            st.write("### Annotated Output")
            st.image(bgr_to_rgb(annotated), use_container_width=True)

            st.write("### Mission Report")
            st.code(report)

            st.write("### Quality Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Inference (ms)", f"{metrics.get('inference_ms', 0.0):.0f}")
            m2.metric("Avg Conf (after)", f"{metrics.get('avg_conf_after', 0.0):.2f}")
            m3.metric(
                "Detections kept",
                f"{metrics.get('detections_after', 0)} / {metrics.get('detections_before', 0)}",
            )
            m4.metric("Duplicate ratio", f"{metrics.get('duplicate_ratio', 0.0) * 100:.1f}%")

            st.write("### Risk Summary")
            st.json(
                {
                    "risk_level": risk_payload.get("risk_level"),
                    "risk_score": risk_payload.get("risk_score"),
                    "counts": risk_payload.get("counts"),
                }
            )

            st.caption(f"Saved: {out_img.name} and {out_rep.name}")


# main video demo
st.divider()
st.header("Video Demo")

video_file = st.file_uploader(
    "Upload a short UAV video (mp4 / mov / avi)",
    type=["mp4", "mov", "avi"],
    key="video_uploader",
)

run_video_btn = st.button("Run Video Analysis", use_container_width=True)

if run_video_btn:
    if video_file is None:
        st.error("Please upload a video first.")
    else:
        out_dir = Path("assets/outputs/video")
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = out_dir / f"uploaded_{video_file.name}"
        tmp_path.write_bytes(video_file.read())

        with st.spinner("Analyzing video (sampling frames)..."):
            out_video_path, last_payload, vsummary, last_report = analyze_video(
                video_path=str(tmp_path),
                detector=detector,
                fps_sample=int(fps_sample),
                max_seconds=int(max_seconds),
                report_every_sec=1,
            )

        st.success("Video analysis completed!")

        p = Path(out_video_path)

        if not p.exists() or p.stat().st_size == 0:
            st.error("Output video file not found or empty.")
        else:
            st.subheader("Annotated Video Output")
            
            p = Path(out_video_path)
            st.write("Output video:", str(p))  # debug

            if p.exists() and p.stat().st_size > 0:
                video_bytes = p.read_bytes()

                # mp4 ise browser oynatır
                if p.suffix.lower() == ".mp4":
                    st.video(video_bytes)
                else:
                    st.warning("Browser preview supports mp4 only. Please download the video.")

                st.download_button(
                    "Download annotated video",
                    data=video_bytes,
                    file_name=p.name,
                    mime="video/mp4" if p.suffix.lower() == ".mp4" else "video/x-msvideo",
                    use_container_width=True,
                )
            else:
                st.error("Output video file not found or empty.")


        st.subheader("Frame Previews (sampled from output video)")
        n_frames = st.slider("How many preview frames?", 3, 12, 6, 1)
        frames = sample_frames(str(p), n=int(n_frames))
        if not frames:
            st.warning("Could not sample frames from the output video.")
        else:
            cols = st.columns(3)
            for i, (fi, img_rgb) in enumerate(frames):
                with cols[i % 3]:
                    st.image(img_rgb, caption=f"Frame #{fi}", use_container_width=True)

        st.subheader("Video Summary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames total", int(getattr(vsummary, "frames_total", 0)))
        c2.metric("Frames analyzed", int(getattr(vsummary, "frames_analyzed", 0)))
        c3.metric("Avg inference (ms)", f"{float(getattr(vsummary, 'avg_inference_ms', 0.0)):.0f}")
        c4.metric("Avg conf (after)", f"{float(getattr(vsummary, 'avg_conf_after', 0.0)):.2f}")

        st.subheader("Last Risk Snapshot")
        st.json(
            {
                "risk_level": last_payload.get("risk_level"),
                "risk_score": last_payload.get("risk_score"),
                "counts": last_payload.get("counts"),
            }
        )

        if last_report:
            st.subheader("Last Generated Report")
            st.code(last_report)


# mission logs
st.divider()
st.subheader("3) Mission Log (JSONL)")

rows = read_jsonl(CFG.LOG_PATH)
if not rows:
    st.info("No logs yet. Run an analysis to create mission logs.")
else:
    flat = []
    for r in rows:
        counts = r.get("counts", {})
        flat.append(
            {
                "logged_at_utc": r.get("logged_at_utc"),
                "mission_id": r.get("mission_id"),
                "model": r.get("model"),
                "conf": r.get("conf"),
                "iou": r.get("iou"),
                "risk_level": r.get("risk_level"),
                "risk_score": r.get("risk_score"),
                "persons": counts.get("person", 0),
                "cars": counts.get("car", 0),
                "motorcycles": counts.get("motorcycle", 0),
                "trucks": counts.get("truck", 0),
                "buses": counts.get("bus", 0),
                "input_image": r.get("input_image"),
            }
        )

    df = pd.DataFrame(flat)

    st.dataframe(df.sort_values(by="logged_at_utc", ascending=False), use_container_width=True)

    st.write("### Quick Analytics")
    a1, a2, a3 = st.columns(3)
    a1.metric("Total Missions", len(df))
    a2.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
    a3.metric("High Risk Missions", int((df["risk_level"] == "HIGH").sum()))

    st.write("### Risk Score Trend")
    try:
        df_plot = df.copy()
        df_plot["logged_at_utc"] = pd.to_datetime(df_plot["logged_at_utc"], errors="coerce")
        df_plot = df_plot.dropna(subset=["logged_at_utc", "risk_score"]).sort_values("logged_at_utc")
        st.line_chart(df_plot.set_index("logged_at_utc")["risk_score"])
    except Exception:
        st.caption("Could not parse timestamps for plotting.")
