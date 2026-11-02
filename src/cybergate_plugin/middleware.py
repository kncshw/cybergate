"""Compatibility module for `--middleware cybergate_plugin.middleware.CyberGateThinkMiddleware`.

vLLM resolves that string by importing this module and reading the attribute off it, so this has to
be a real module rather than a lazy attribute on the package.
"""
from __future__ import annotations

from cybergate.serving.client import CyberGateVLLMClient
from cybergate.serving.middleware import CyberGateThinkMiddleware

__all__ = [
    "CyberGateThinkMiddleware",
    "CyberGateVLLMClient",
]
