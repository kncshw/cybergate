"""Server-side code. Runs in the API-server process; imports no torch.

`middleware` is the in-process ASGI path, `client` the remote one, `think_routes` the CoT
orchestration they share, and `prompts` the prompt-building facade.
"""
from __future__ import annotations

from typing import Any

from .client import CyberGateVLLMClient
from .prompts import build_raw_prompt_for_model, detect_model_family

__all__ = [
    "CyberGateThinkMiddleware",
    "CyberGateVLLMClient",
    "build_raw_prompt_for_model",
    "detect_model_family",
    "start_background_proxy_server",
    "stop_background_proxy_server",
]


def __getattr__(name: str) -> Any:
    if name == "CyberGateThinkMiddleware":
        from .middleware import CyberGateThinkMiddleware

        return CyberGateThinkMiddleware
    if name in ("start_background_proxy_server", "stop_background_proxy_server"):
        from . import proxy as _proxy

        return getattr(_proxy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
