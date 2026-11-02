"""In-process `DecisionStore`, for tests.

Not used in serving. It exists so the evidence layer can be driven without touching `/dev/shm`, which
is what makes the R11 cross-conversation key collision testable on a laptop.
"""
from __future__ import annotations

import copy
from typing import Any, Iterator

__all__ = ["MemoryDecisionStore"]


class MemoryDecisionStore:
    """A dict-backed `DecisionStore`.

    `get` and `put` deep-copy, so a caller mutating a returned record cannot reach back into the
    store. The `/dev/shm` store has that isolation for free via JSON round-tripping, and tests would
    otherwise pass for the wrong reason.
    """

    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        if records:
            for key, rec in records.items():
                self.put(key, rec)

    def get(self, key: str) -> dict[str, Any] | None:
        rec = self._records.get(key)
        return copy.deepcopy(rec) if rec is not None else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self._records[key] = copy.deepcopy(record)

    def delete(self, key: str) -> None:
        self._records.pop(key, None)

    def iter_keys(self) -> Iterator[str]:
        yield from sorted(self._records)

    # --- refusal-monitor telemetry slot -----------------------------------------------------------
    def put_monitor(self, monitor_data: dict[str, Any]) -> None:
        self._monitor = copy.deepcopy(monitor_data)

    def get_monitor(self) -> dict[str, Any] | None:
        return copy.deepcopy(getattr(self, "_monitor", None))
