"""`cybergate_plugin`: the drop-in import path. Implementation lives in `cybergate`.

The live deployment launches with

    vllm serve ... --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware

and registers the entry point `cybergate_think = "cybergate_plugin:register_cybergate_plugin"`, so
this package name is part of the deployment contract. It re-exports the implementation under the
module paths that launch command and entry point name.

Attribute access is lazy (PEP 562). The canonical package imported `gate_probe` -- and therefore
torch -- at module scope, which the API-server process paid for on every start; here nothing is
imported until it is asked for.
"""
from __future__ import annotations

from typing import Any

from cybergate import __version__
from cybergate._entrypoint import register_cybergate_plugin

#: Symbol -> module it is imported from, resolved on first attribute access.
_LAZY_EXPORTS: dict[str, str] = {
    # decision
    "evaluate_gate_for_messages": "cybergate.decision",
    "evaluate_gate_for_prompt": "cybergate.decision",
    # operators
    "CyberGateOperator": "cybergate.ops",
    "load_cybergate_op": "cybergate.ops",
    # per-process state
    "LAST_GATE_DECISION": "cybergate.state",
    "LIVE_GPU_PROMPT_CACHE": "cybergate.state",
    "REQUEST_GATE_STATES": "cybergate.state",
    "clear_request_gate_state": "cybergate.state",
    "get_request_gate_state": "cybergate.state",
    "conversation_key": "cybergate.state",
    # serving
    "CyberGateThinkMiddleware": "cybergate.serving",
    "CyberGateVLLMClient": "cybergate.serving",
    "build_raw_prompt_for_model": "cybergate.serving",
    "detect_model_family": "cybergate.serving",
    # boilerplate
    "BOTTOM_COMPLIANCE_LOCK": "cybergate.boilerplate",
    "FAKE_THINK_RUBRIC": "cybergate.boilerplate",
    "FAKE_THINK_SHORT": "cybergate.boilerplate",
    "INLINE_COT_SUPPRESSION": "cybergate.boilerplate",
    "INLINE_FINAL_STEP_CLEARANCE": "cybergate.boilerplate",
    "OPEN_THINK_PREFIX": "cybergate.boilerplate",
    "OPEN_THINK_PREFIX_NO_SLOT": "cybergate.boilerplate",
    "POST_THINK_OPENER": "cybergate.boilerplate",
    "strip_injected_think_boilerplate": "cybergate.boilerplate",
    "truncate_think_at_second_guess": "cybergate.boilerplate",
    # telemetry
    "log_cybergate_event": "cybergate.telemetry",
    # worker-side; these pull in torch, hence lazy
    "install_vllm_read_hook": "cybergate.hooks",
    "install_vllm_refusal_monitor_hook": "cybergate.hooks",
}

#: Dropped in the refactor, with where the behaviour went.
_REMOVED = {
    "CyberGateSandwichLogitsProcessor": (
        "the in-engine LogitsProcessor classes were dropped from the port: nothing in the package "
        "called them, and the live `sandwich` and `closed_loop` modes are multi-call prompt routes "
        "in cybergate.serving.think_routes, not logits processors. The originals are preserved in "
        "tests/equivalence/baseline/sandwich_processor.py"
    ),
}
_REMOVED["CyberGateClosedLoopCoTProcessor"] = _REMOVED["CyberGateSandwichLogitsProcessor"]

# `install_vllm_read_hook` was renamed; map the old name onto the new one.
_RENAMED = {
    "install_vllm_read_hook": "install_category_probe_hook",
    "install_vllm_refusal_monitor_hook": "install_refusal_monitor_hook",
}

__all__ = [
    "__version__",
    "register_cybergate_plugin",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Resolve an exported symbol on first access (PEP 562)."""
    if name in _REMOVED:
        raise AttributeError(f"{name} was removed: {_REMOVED[name]}")
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, _RENAMED.get(name, name))


def __dir__() -> list[str]:
    return sorted(__all__)
