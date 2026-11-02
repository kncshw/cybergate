"""The one numeric kernel the hooks share.

Kept separate so both hooks and the telemetry probes use the same normalisation, and so the hot path
has a single obvious place to look when `gpu_dot_us` moves.
"""
from __future__ import annotations

from typing import Any

__all__ = ["cosine"]


def cosine(vec: Any, unit_direction: Any) -> float:
    """Cosine of `vec` against an already unit-normalised direction, as a Python float.

    `clamp_min` guards a zero-norm hidden state, which would otherwise produce a NaN score that
    silently compares false against every threshold.
    """
    import torch  # type: ignore

    unit = vec / vec.norm().clamp_min(1e-9)
    return float(torch.dot(unit, unit_direction).item())
