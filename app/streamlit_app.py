import streamlit as st
from pathlib import Path
import uuid
import json
import numpy as np 
import pandas as pd
from datetime import datetime, timezone
import cv2

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import CFG
from vision.detector import YoloV8Detector
from vision.annotator import draw_detections
from analytics.risk import compute_risk
from analytics.logger import append_jsonl
from nlp.report_generator import generate_report_en

st.set_page_config(page_title="UAV Mission Analysis", layout="wide")

st.title("AI-Based UAV Mission Analysis System")
st.caption("YOLOv8 + Risk Asssesment + NLP Mission Report Output + Mission Logging -JSONL-")

# output directories check
CFG.OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok = True)
CFG.OUTPUT_REP_DIR.mkdir(parents=True, exist_ok=True)
CFG.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_detector(model_name: str, conf: float, iou: float):
    return YoloV8Detector(model_name, conf, iou)

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    return rows


# side-bar

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "YOLO Model",
    options=["yolov8n.pt", "yolov8s.pt","yolov8m.pt"],
    index=0

)

conf =st.sidebar.slider("Confidence Threshold", 0.05, 0.8, float(CFG.CONF_THRES), 0.05)
iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9,float(CFG.IOU_THRES), 0.05)
detector = load_detector(model_choice, conf, iou)

st.sidebar.divider()
st.sidebar.write("Outputs")
st.sidebar.write(f"Annotated: {CFG.OUTPUT_ANN_DIR}")
st.sidebar.write(f"Reports: {CFG.OUTPUT_REP_DIR}")
st.sidebar.write(f"Logs: {CFG.LOG_PATH}")

# main

col1, col2 =st.columns([1,1], gap="large")

with col1: 
    st.subheader("1) Upload UAV Image")
    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg","jpeg","png"])

    use_sample = st.checkbox("Use sample image (assests/samples/perimeter_01.jpg)", value=False)

    img_bgr = None
    img_name = None

    if use_sample:
        sample_path = CFG.SAMPLE_DIR / "perimeter_01.jpg"
        if sample_path.exists():
            img_bgr=cv2.imread(str(sample_path))
            img_name= sample_path.name
        else:
            st.warning("Sample image not found at assets/samples/perimeter_01.jpg")

    
    if uploaded is not None:
        file_bytes= np.frombuffer(uploaded.read(), np.uint8)
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
                    detections = detector.detect(img_bgr)

                    risk_payload = compute_risk(detections)
                    report = generate_report_en(mission_id=mission_id, risk_payload=risk_payload)

                    annotated = draw_detections(img_bgr, detections)

                    out_img =CFG.OUTPUT_ANN_DIR/f"{Path(img_name).stem}_ann_{mission_id}.jpg"
                    out_rep = CFG.OUTPUT_REP_DIR / f"{Path(img_name).stem}_report_{mission_id}.txt"

                    cv2.imwrite(str(out_img),annotated)
                    out_rep.write_text(report, encoding="utf-8")

                    log_payload= {
                        "mission_id": mission_id,
                        "scenario": CFG.SCENARIO_NAME,
                        "input_image": img_name,
                        "model": model_choice,
                        "conf": conf,
                        "iou": iou,
                        "detections": detections, 
                        **risk_payload,
                    }
                    append_jsonl(CFG.LOG_PATH, log_payload)

                st.success("Done!")

                st.write("---Annotated Output---")
                st.image(bgr_to_rgb(annotated), use_container_width=True)

                st.write("---Mission Report---")
                st.code(report)

                st.write("---Risk Summary---")
                st.json({
                    "risk_level": risk_payload["risk_level"],
                    "risk_score": risk_payload["risk_score"],
                    "counts": risk_payload["counts"]
                })

                st.caption(f"Saved: {out_img.name} and {out_rep.name}")

    st.divider()
    st.subheader("3) Mission Log (JSONL)")

    rows = read_jsonl(CFG.LOG_PATH)

    if not rows:
        st.info("No logs yet. Run an analysis to create mission logs.")
    else:
        flat = []
            
        for r in rows:
            counts = r.get("counts", {})
            flat.append({
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
            })

        df = pd.DataFrame(flat)

        st.dataframe(df.sort_values(by="logged_at_utc", ascending=False), use_container_width=True)

        st.write("### Quick Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Missions", len(df))
        c2.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
        c3.metric("High Risk Missions", int((df["risk_level"] == "HIGH").sum()))

        st.write("### Risk Score Trend")
        # parse times if possible
        try:
            df_plot = df.copy()
            df_plot["logged_at_utc"] = pd.to_datetime(df_plot["logged_at_utc"], errors="coerce")
            df_plot = df_plot.dropna(subset=["logged_at_utc"]).sort_values("logged_at_utc")
            st.line_chart(df_plot.set_index("logged_at_utc")["risk_score"])
        except Exception:
            st.caption("Could not parse timestamps for plotting.")