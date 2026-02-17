from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def state_path(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return base / ".continuum" / "state" / "accelerate_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(cwd: Path | None = None) -> dict[str, Any] | None:
    path = state_path(cwd)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save_state(payload: dict[str, Any], cwd: Path | None = None) -> Path:
    path = state_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = ["state_path", "utc_now", "load_state", "save_state"]
