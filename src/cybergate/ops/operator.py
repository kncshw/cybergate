"""The loaded linear-probe operator. Frozen; no I/O.

Port of `gate_probe.CyberGateOperator`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CyberGateOperator"]


@dataclass(frozen=True)
class CyberGateOperator:
    """An `M1_ridge_lam16` linear probe for one layer and pooling mode.

    `c` is unit-normalised at load time, so `s_cat` is a cosine against a direction.
    """

    c: list[float]
    thresh: float
    gate_layer: int
    pool: str
    category: str
    model_key: str
    meta: dict[str, Any]

    @property
    def dim(self) -> int:
        return len(self.c)

