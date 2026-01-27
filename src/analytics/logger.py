import json
from datetime import datetime, timezone
from pathlib import Path


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict):
    ensure_parent(path)
    payload=dict(payload)
    payload["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")