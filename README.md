
![ee953ec8479899366fe512230b919032c687808fecdcd303983ff3dc](https://github.com/user-attachments/assets/8f4d7150-6906-42e6-aeaa-13dd7b52502d)

<img width="725" height="651" alt="Ekran görüntüsü 2026-01-27 223524" src="https://github.com/user-attachments/assets/f682277a-9598-458b-a119-c5f9d11af53d" />

AI-Based UAV Mission Analysis System
An end-to-end computer vision–based UAV mission analysis platform that performs real-time object detection, risk assessment, and automated mission reporting on images and videos using deep learning.
This project simulates how an autonomous or semi-autonomous UAV can analyze its surroundings, assess operational risk, and generate mission-level intelligence outputs.

Overview:

Unmanned Aerial Vehicles (UAVs) are increasingly used in surveillance, perimeter security, and urban monitoring.
This system demonstrates a full onboard-style vision pipeline, combining:

-Deep learning–based object detection (YOLOv8
-Real-time video analysis
-Dynamic risk scoring
-Automated natural language mission reports
-Mission-level logging and analytics

The application is built with Streamlit and designed to be both interactive and extensible.

Key Capabilities:

-Image-Based Analysis
-Detects objects such as persons, cars, buses, trucks, motorcycles
-Draws bounding boxes with confidence scores
-Computes risk level based on detected entities
-Generates a textual mission report
-Logs mission data in structured JSONL format
-Video-Based Analysis
-Processes UAV videos frame by frame
-Supports configurable FPS sampling or full-frame analysis
-Maintains persistent bounding boxes across frames

Displays live mission overlays:

Risk level

Risk score

FPS

Inference time

Generates annotated, browser-playable videos

Produces periodic mission reports during video playback

Risk Assessment

Object-type–weighted scoring

Aggregated risk level classification (LOW / MEDIUM / HIGH)

Snapshot of latest operational risk state

Mission Analytics

Historical mission logs

Risk score trends over time

Detection statistics across missions

System Architecture:

Input (Image / Video)
        ↓
YOLOv8 Object Detection
        ↓
Bounding Box Annotation
        ↓
Risk Scoring Engine
        ↓
Mission Report Generator (NLP)
        ↓
Video / Image Output + Logs

User Interface:

The Streamlit-based interface allows users to:

-Upload images or videos
-Configure detection thresholds and sampling rate
-View annotated outputs directly in the browser
-Inspect mission summaries and reports
-Explore mission history and analytics
-The UI dynamically adapts to different video aspect ratios to ensure consistent visualization.

Setup & Installation:

1. Clone the Repository
git clone [https://github.com/your-username/ai-uav-mission-analysis.git](https://github.com/AstoKE/AI-Based-UAV-Mission-Analysis-System?tab=readme-ov-file)
cd ai-uav-mission-analysis

2. Create a Virtual Environment

Windows-->

python -m venv .venv
.venv\Scripts\activate

macOS / Linux-->

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Install FFmpeg (Required for Video Output)

FFmpeg is required to generate browser-compatible annotated videos.

Windows-->
Download from: https://www.gyan.dev/ffmpeg/builds/
Add ffmpeg/bin to your system PATH
Verify:
ffmpeg -version

macOS-->
brew install ffmpeg

Linux-->
sudo apt install ffmpeg

5. Run the Application
streamlit run app/streamlit_app.py

Then open:

http://localhost:8501

Project Structure:

app/
 └── streamlit_app.py        # Main UI

src/
 ├── vision/                # Detection & annotation
 ├── video/                 # Video processing pipeline
 ├── analytics/             # Risk logic & logging
 └── nlp/                   # Mission report generation

assets/
 ├── samples/               # Sample images
 └── outputs/               # Generated outputs

Tested Environment

Python 3.9+

Windows 10 / 11

macOS (Intel & Apple Silicon)

Chrome / Edge (recommended for video playback)



Author : Şükrü Enes Tuğaç

Developed as a full-stack AI vision project combining computer vision, analytics, and NLP for UAV mission intelligence.
