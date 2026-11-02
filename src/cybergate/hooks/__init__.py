"""Worker-side forward hooks. **These import torch** -- never import this package from the API server.

Both hooks are read-only (`K = 0.0`): they observe activations and publish decision records, and
never touch weights or logits.
"""
from __future__ import annotations

from .category_probe import install_category_probe_hook
from .cot_controller import RefusalSteeredCoTProcessor
from .steering_config import SteeringConfig, SteeringStats
from .refusal_monitor import install_refusal_monitor_hook

__all__ = [
    "RefusalSteeredCoTProcessor",
    "SteeringConfig",
    "SteeringStats",
    "install_category_probe_hook",
    "install_refusal_monitor_hook",
]
