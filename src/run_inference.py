import argparse
import uuid
from pathlib import Path
import cv2

from config import CFG
from vision.detector import YoloV8Detector
from vision.annotator import draw_detections
from analytics.risk import compute_risk
from analytics.logger import append_jsonl
from nlp.report_generator import generate_report_en

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help= "Path to input image")
    args = parser.parse_args()
    CFG.OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)
    CFG.OUTPUT_REP_DIR.mkdir(parents=True, exist_ok=True)
    CFG.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    img_path= Path(args.image)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    detector = YoloV8Detector(CFG.YOLO_MODEL, CFG.CONF_THRES, CFG.IOU_THRES)

    mission_id = str(uuid.uuid4())[:8]
    detections = detector.detect(image)

    risk_payload = compute_risk(detections)
    report = generate_report_en(mission_id=mission_id, risk_payload=risk_payload)

    annotated = draw_detections(image, detections)

    out_img = CFG.OUTPUT_ANN_DIR / f"{img_path.stem}_ann_{mission_id}.jpg"
    out_rep = CFG.OUTPUT_REP_DIR / f"{img_path.stem}_report_{mission_id}.txt"

    cv2.imwrite(str(out_img), annotated)
    out_rep.write_text(report, encoding="utf-8")

    log_payload = {
        "mission_id": mission_id,
        "scenario": CFG.SCENARIO_NAME,
        "input_image": str(img_path),
        "detections": detections,
        **risk_payload
    }
    append_jsonl(CFG.LOG_PATH, log_payload)

    print("-->Done<--")
    print(f"Annotated image: {out_img}")
    print(f"Report: {out_rep}")
    print(f"Log appended: {CFG.LOG_PATH}")

if __name__ == "__main__":
    main()