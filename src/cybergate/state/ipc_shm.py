"""`/dev/shm` decision store. The MainProcess <-> EngineCore boundary.

Verbatim port of `gate_probe.write_prompt_decision_ipc`, `read_prompt_decision_ipc`,
`write_l21_monitor_ipc`, `read_l21_monitor_ipc` and `_ensure_ipc_dir`.

Records are written to `sha_<key>.json` through a pid+thread-unique temp file and `Path.replace`, so
a concurrent reader sees either the old record or the new one, never a partial write. That matters
because the API-server process reads these files while the worker process writes them.

Records never expire; the audit found 310 in place, the oldest hours old. Phase 6 adds a TTL and
consume-on-read (REFACTOR_PLAN.md §6).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from ..config import IPC_DIR_PATH

__all__ = ["L21_MONITOR_FILENAME", "ShmDecisionStore", "ensure_ipc_dir"]

#: The refusal-monitor telemetry file. Not a decision record -- a single latest-value slot, read by
#: the middleware to attach `l21_gpu_monitor` to a response.
L21_MONITOR_FILENAME = "l21_monitor.json"


def ensure_ipc_dir(ipc_dir: Path = IPC_DIR_PATH) -> Path:
    ipc_dir.mkdir(parents=True, exist_ok=True)
    return ipc_dir


class ShmDecisionStore:
    """File-backed `DecisionStore` under `IPC_DIR_PATH` (default `/dev/shm/cybergate_ipc`)."""

    def __init__(self, ipc_dir: Path | str = IPC_DIR_PATH) -> None:
        self.ipc_dir = Path(ipc_dir)

    def _path(self, key: str) -> Path:
        return self.ipc_dir / f"sha_{key}.json"

    def _tmp_path(self, stem: str) -> Path:
        return self.ipc_dir / f".tmp_{stem}_{os.getpid()}_{threading.get_ident()}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            target = self._path(key)
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def put(self, key: str, record: dict[str, Any]) -> None:
        try:
            ensure_ipc_dir(self.ipc_dir)
            tmp = self._tmp_path(f"sha_{key}")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            tmp.replace(self._path(key))
        except Exception:
            pass

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except Exception:
            pass

    def iter_keys(self) -> Iterator[str]:
        try:
            names = sorted(p.name for p in self.ipc_dir.glob("sha_*.json"))
        except Exception:
            return
        for name in names:
            yield name[len("sha_") : -len(".json")]

    # --- refusal-monitor telemetry slot -----------------------------------------------------------
    def put_monitor(self, monitor_data: dict[str, Any]) -> None:
        """Replace the latest refusal-monitor telemetry record."""
        try:
            ensure_ipc_dir(self.ipc_dir)
            tmp = self._tmp_path("l21")
            tmp.write_text(json.dumps(monitor_data), encoding="utf-8")
            tmp.replace(self.ipc_dir / L21_MONITOR_FILENAME)
        except Exception:
            pass

    def get_monitor(self) -> dict[str, Any] | None:
        """Return the latest refusal-monitor telemetry record, or `None`."""
        try:
            target = self.ipc_dir / L21_MONITOR_FILENAME
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None
