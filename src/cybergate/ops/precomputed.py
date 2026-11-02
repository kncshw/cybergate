"""Offline precomputed score lookup, used by `--dry-run` and any no-GPU path.

Verbatim port of `gate_probe._load_precomputed_scores` / `_PRECOMPUTED_CACHE`. The canonical module
loads the file once at import time; this module keeps that, because the file is read-only input and a
per-call read would add disk I/O to the request path.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import PRECOMPUTED_SCORES_PATH

__all__ = ["PRECOMPUTED_CACHE", "load_precomputed_scores"]


def load_precomputed_scores() -> dict[str, Any]:
    """Read `precomputed_l8_scores.json`, or return `{}` when it is absent or unreadable."""
    if PRECOMPUTED_SCORES_PATH.exists():
        try:
            return json.loads(PRECOMPUTED_SCORES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


PRECOMPUTED_CACHE: dict[str, Any] = load_precomputed_scores()
