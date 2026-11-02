"""Compatibility module for `from cybergate_plugin.gate_probe import ...`.

The canonical 1499-line `gate_probe.py` is now split across `cybergate.decision`, `cybergate.hooks`,
`cybergate.ops`, `cybergate.state` and `cybergate.vllm_adapter`. This module re-exports the names it
used to provide.

`install_vllm_read_hook` and `install_vllm_refusal_monitor_hook` are imported lazily, because they
pull in torch and this module is importable from the API-server process.
"""
from __future__ import annotations

from typing import Any

from cybergate.decision import evaluate_gate_for_messages, evaluate_gate_for_prompt
from cybergate.ops import (
    CyberGateOperator,
    load_cybergate_op,
)
from cybergate.state import (
    LAST_GATE_DECISION,
    LIVE_GPU_PROMPT_CACHE,
    REQUEST_GATE_STATES,
    clear_request_gate_state,
    get_request_gate_state,
)
from cybergate.state.keys import conversation_key as compute_conv_sha256
from cybergate.telemetry import log_cybergate_event

__all__ = [
    "CyberGateOperator",
    "LAST_GATE_DECISION",
    "LIVE_GPU_PROMPT_CACHE",
    "REQUEST_GATE_STATES",
    "clear_request_gate_state",
    "compute_conv_sha256",
    "evaluate_gate_for_messages",
    "evaluate_gate_for_prompt",
    "get_request_gate_state",
    "install_vllm_read_hook",
    "install_vllm_refusal_monitor_hook",
    "load_cybergate_op",
    "log_cybergate_event",
]

_LAZY = {
    "install_vllm_read_hook": "install_category_probe_hook",
    "install_vllm_refusal_monitor_hook": "install_refusal_monitor_hook",
}


def __getattr__(name: str) -> Any:
    """Resolve the two torch-importing hook installers on first access."""
    if name in _LAZY:
        import cybergate.hooks as hooks

        return getattr(hooks, _LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
