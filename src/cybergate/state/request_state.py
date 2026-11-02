"""Per-process in-memory gate state.

Verbatim port of `gate_probe.LAST_GATE_DECISION`, `REQUEST_GATE_STATES`, `LIVE_GPU_PROMPT_CACHE`,
`get_request_gate_state` and `clear_request_gate_state`.

**These are module globals, and the plugin runs in two processes that share no memory.** A value
written here by the model worker is invisible to the API server and vice versa. Anything that has to
cross the boundary goes through `state.ipc_shm`. See `docs/PROCESS_TOPOLOGY.md`.
"""
from __future__ import annotations

import threading
from typing import Any

__all__ = [
    "LAST_GATE_DECISION",
    "LIVE_GPU_PROMPT_CACHE",
    "MAX_TRACKED_REQUESTS",
    "REQUEST_GATE_STATES",
    "clear_request_gate_state",
    "get_request_gate_state",
]

#: Latest decision in *this* process. A single slot, so under continuous batching it holds whichever
#: sequence the hook touched last; per-request state below is the batch-safe view.
LAST_GATE_DECISION: dict[str, Any] = {
    "decided": False,
    "fired": False,
    "score": 0.0,
    "threshold": 0.10807704,
    "layer": 10,
    "source": "none",
    "rolling_s_ref_z": 0.0,
    "last_cosine_us": 0.0,
}

_STATE_LOCK = threading.Lock()

#: Per-request isolated state, keyed by vLLM request id, for continuous batching.
REQUEST_GATE_STATES: dict[str, dict[str, Any]] = {}

#: Decision cache keyed by `state.keys` keys, populated from live prefill observations.
LIVE_GPU_PROMPT_CACHE: dict[str, dict[str, Any]] = {}

#: Eviction bound on `REQUEST_GATE_STATES`.
MAX_TRACKED_REQUESTS = 2048


def get_request_gate_state(request_id: str | None = None) -> dict[str, Any]:
    """Return or initialise the per-request gate and refusal-monitor state.

    With no `request_id`, returns the shared `LAST_GATE_DECISION` slot -- which is what the offline
    single-process paths use, and why that slot still exists.

    On insert past `MAX_TRACKED_REQUESTS`, evicts in insertion order, skipping the entry just
    created.
    """
    if not request_id:
        return LAST_GATE_DECISION
    with _STATE_LOCK:
        st = REQUEST_GATE_STATES.get(request_id)
        if st is None:
            st = {
                "request_id": request_id,
                "decided": False,
                "fired": False,
                "score": 0.0,
                "threshold": LAST_GATE_DECISION["threshold"],
                "layer": LAST_GATE_DECISION["layer"],
                "source": "none",
                "rolling_s_ref_z": 0.0,
                "s_ref_history": [],
                "last_cosine_us": 0.0,
            }
            REQUEST_GATE_STATES[request_id] = st
            if len(REQUEST_GATE_STATES) > MAX_TRACKED_REQUESTS:
                oldest_key = next(iter(REQUEST_GATE_STATES))
                if oldest_key != request_id:
                    REQUEST_GATE_STATES.pop(oldest_key, None)
        return st


def clear_request_gate_state(request_id: str) -> None:
    """Remove a completed request's state."""
    with _STATE_LOCK:
        REQUEST_GATE_STATES.pop(request_id, None)
