"""Load CDFG JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        return json.load(f)
