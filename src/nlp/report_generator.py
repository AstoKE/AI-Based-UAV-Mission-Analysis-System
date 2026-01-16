from datetime import datetime, timezone

def generate_report_en(mission_id: str, risk_payload: dict) -> str:
    counts = risk_payload["counts"]
    level =risk_payload["risk_level"]
    score=risk_payload["risk_score"]
    action= risk_payload["recommended_action"]

    persons= counts.get("person", 0)
    cars =counts.get("car", 0)
    trucks=counts.get("truck", 0)
    buses =counts.get("bus", 0)

    ts =datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        f"Mission Report | {ts}",
        f"Mission ID: {mission_id}",
        f"Summary: Perimeter area analyzed via onboard vision pipeline.",
        f"Detections: person={persons}, car={cars}, truck={trucks}, bus={buses}.",
        f"Risk Assessment: {level} (score={score}).",
        f"Recommendation: {action}",
    ]

    return "\n".join(lines)