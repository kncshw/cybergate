"""Server-side code. Runs in the API-server process; imports no torch.

`middleware` is the in-process ASGI path, `client` the remote one, `think_routes` the CoT
orchestration they share, and `prompts` the prompt-building facade.
"""
from __future__ import annotations

from .client import CyberGateVLLMClient
from .middleware import CyberGateThinkMiddleware
from .prompts import build_raw_prompt_for_model, detect_model_family

__all__ = [
    "CyberGateThinkMiddleware",
    "CyberGateVLLMClient",
    "build_raw_prompt_for_model",
    "detect_model_family",
]
