from collections import Counter

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motocycle"}

def compute_risk(detections: list[dict]) -> dict:
    counts = Counter([d["class_name"] for d in detections])
    persons = sum(counts[c] for c in PERSON_CLASSES if c in counts)
    vehicles = sum(counts[c] for c in VEHICLE_CLASSES if c in counts)

    #very basic score calculation
    score = 0
    score += persons * 8
    score += vehicles * 5

    if persons >= 8:
        score +=10
    if vehicles >= 5:
        score +=8
    if persons == 0 and vehicles ==0:
        score= 0

    #risk level calc.
    if score >= 60:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"

    #suggestions part
    if level == "HIGH":
        action ="Recommend re-scan at higher altitude and avoid direct approach."
    elif level =="MEDIUM":
        action = "Recommend additiona scan and maintain safe perimeter distance."
    else:
        action = "Routine monitoring recommended."

    return {
        "counts": dict(counts),
        "risk_score": int(score),
        "risk_level": level,
        "recommended_action": action,
    }
