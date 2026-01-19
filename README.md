# AI-Based UAV Mission Analysis System

End-to-end UAV imagery analysis pipeline integrating:
- **YOLOv8** object detection (computer vision / CNN)
- **Risk assessment** (rule-based decision layer)
- **NLP mission report** generation (operational-style output)
- **Mission logging** in **JSONL** format + quick analytics dashboard (Streamlit)

## Key Features
- Upload UAV image → run detection → visualize annotated output
- Perimeter security scenario: counts of **person** and **vehicles**
- Normalized risk scoring to avoid saturation in crowded scenes
- JSONL mission logs with quick analytics (trend + table)

## Architecture
Image
↓
YOLOv8 Detector (CNN)
↓
Post-process (filters / dedup)
↓
Risk Assessment (decision layer)
↓
NLP Mission Report (TR/EN)
↓
Mission Log (JSONL) + Streamlit Dashboard

r
Kodu kopyala

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate
pip install -r requirements.txt
Run (CLI)
bash
Kodu kopyala
python src/run_inference.py --image assets/samples/perimeter_01.jpg
Outputs:

Annotated image → assets/outputs/annotated/

Mission report → assets/outputs/reports/

Logs → data/logs/mission_logs.jsonl

Run (Streamlit Demo)
bash
Kodu kopyala
streamlit run app/streamlit_app.py
Notes
Higher resolution (e.g., imgsz=1536/1920) improves small-object detection in crowded UAV scenes.

Class-agnostic NMS and post-processing reduce duplicate detections.