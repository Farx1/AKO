"""JSON report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_report(data: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
