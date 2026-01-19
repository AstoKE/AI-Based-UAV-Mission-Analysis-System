from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class DedupStats:
    before: int
    after: int

    duplicate_ratio: float

def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def dedup_detections(dets: List[Dict], iou_th: float=0.65) -> Tuple[List[Dict], DedupStats]:

    before =  len(dets)
    if before==0:
        return dets, DedupStats(before=0, after=0, duplicate_ratio=0.0)
    
    dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep: List[Dict] = []

    for d in dets_sorted:
        drop = False
        for k in keep:
            if iou_xyxy(d["bbox_xyxy"], k["bbox_xyxy"]) >= iou_th:
                drop = True
                break
        if not drop:
            keep.append(d)

    after = len(keep)
    duplicate_ratio = (before - after) / max(1, before)
    return keep, DedupStats(before=before, after=after, duplicate_ratio=duplicate_ratio)

def avg_confidence(dets: List[Dict]) -> float:
    if not dets:
        return 0.0
    return sum(d["conf"] for d in dets) / len(dets)