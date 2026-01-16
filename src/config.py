from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    SAMPLE_DIR: Path = PROJECT_ROOT / "assets" / "samples"
    OUTPUT_ANN_DIR: Path = PROJECT_ROOT / "assets"/ "outputs" /"annotated"
    OUTPUT_REP_DIR: Path =PROJECT_ROOT/ "assets" /"outputs" / "reports"
    LOG_PATH: Path=PROJECT_ROOT / "data"/ "logs" / "mission_logs.jsonl"

    YOLO_MODEL: str = "yolov8n.pt"
    CONF_THRES: float = 0.35
    IOU_THRES: float = 0.45

    SCENARIO_NAME: str = "Perimeter Security"
    LANGUAGE: str = "EN"

CFG = Config()