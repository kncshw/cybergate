"""Timestamped event logging to stderr and `PLUGIN_EVENTS_LOG_PATH`.

Verbatim port of `gate_probe.log_cybergate_event`. Both writes stay wrapped in bare `except` because
this runs on the model-worker forward path, where a logging failure must never propagate into the
engine.
"""
from __future__ import annotations

import datetime
import os
import sys

from ..config import PLUGIN_EVENTS_LOG_PATH

__all__ = ["log_cybergate_event"]


def log_cybergate_event(tag: str, message: str) -> None:
    """Write a timestamped event line to both `sys.stderr` and `PLUGIN_EVENTS_LOG_PATH`."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    line = f"[{tag}] [{ts}] [pid={os.getpid()}] {message}\n"
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        PLUGIN_EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLUGIN_EVENTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

