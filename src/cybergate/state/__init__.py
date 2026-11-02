"""The cross-process boundary, made explicit.

`keys` owns every cache-key constructor, `backend` declares the store protocol, `ipc_shm` is the real
`/dev/shm` store, `memory` is the test store, and `request_state` holds the per-process globals.
"""
from __future__ import annotations

from .backend import DecisionStore
from .ipc_shm import ShmDecisionStore
from .keys import (
    bare_turn_key,
    conversation_key,
    legacy_in_context_key_from_sha,
    standalone_turn_key,
    text_sha256,
)
from .memory import MemoryDecisionStore
from .request_state import (
    LAST_GATE_DECISION,
    LIVE_GPU_PROMPT_CACHE,
    REQUEST_GATE_STATES,
    clear_request_gate_state,
    get_request_gate_state,
)

__all__ = [
    "DecisionStore",
    "LAST_GATE_DECISION",
    "LIVE_GPU_PROMPT_CACHE",
    "MemoryDecisionStore",
    "REQUEST_GATE_STATES",
    "ShmDecisionStore",
    "bare_turn_key",
    "clear_request_gate_state",
    "conversation_key",
    "get_request_gate_state",
    "legacy_in_context_key_from_sha",
    "standalone_turn_key",
    "text_sha256",
]
