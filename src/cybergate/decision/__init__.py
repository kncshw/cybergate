"""The decision layer.

`policy.decide` is pure; `evidence` owns every cache read, HTTP call and precomputed lookup;
`api` is the legacy dict-returning facade the serving code calls.
"""
from __future__ import annotations

from .api import evaluate_gate_for_messages, evaluate_gate_for_prompt
from .inputs import (
    GateInputs,
    InContextEvidence,
    OperatingPoint,
    PriorTurnEvidence,
    StandaloneEvidence,
)
from .outcome import Branch, GateDecision, render_payload, render_source
from .policy import decide

__all__ = [
    "Branch",
    "GateDecision",
    "GateInputs",
    "InContextEvidence",
    "OperatingPoint",
    "PriorTurnEvidence",
    "StandaloneEvidence",
    "decide",
    "evaluate_gate_for_messages",
    "evaluate_gate_for_prompt",
    "render_payload",
    "render_source",
]
