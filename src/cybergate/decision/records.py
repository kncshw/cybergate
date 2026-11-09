"""Record-level transforms: raw observation dict -> the fields the policy may read.

Separated from `evidence` so the gatherer is only about *where* a record comes from, and this module
is only about *what a record means*. Both halves are pure except that they read nothing but the dict
handed to them.
"""
from __future__ import annotations

from typing import Any

from ..config import DEFAULT_REF_MAINT_Z_THRESH
from .inputs import InContextEvidence, StandaloneEvidence
from .rules import two_factor_suppresses

__all__ = [
    "apply_in_context_two_factor",
    "finalize_live_record",
    "in_context_evidence",
    "optional_float",
    "standalone_evidence",
]


def _record_threshold(record: dict[str, Any], op_thresh: float) -> float:
    """Extract a numeric threshold for policy comparisons (`None` falls back to `op_thresh`)."""
    raw = record.get("threshold")
    return op_thresh if raw is None else float(raw)


def finalize_live_record(
    record: dict[str, Any],
    op_thresh: float,
    ref_maint_z: float,
) -> dict[str, Any]:
    """Apply the single-turn two-factor gate to a live observation record.

    Port of `gate_probe._finalize_live_gpu`. Returns a copy with `fired`, `raw_gate_fired`,
    `raw_l8_fired` and possibly `source` set.

    Called by the evidence layer rather than the policy because whether this runs at all depends on
    where the record came from: live, IPC and cached records get it, while HTTP `/gate` and offline
    precomputed records carry their own `fired` verbatim.
    """
    out = dict(record)
    score = float(out.get("score", 0.0))
    threshold = _record_threshold(out, op_thresh)
    s_ref_z = out.get("s_ref_z")
    raw_fired = bool(score >= threshold)
    out["raw_gate_fired"] = raw_fired
    out["raw_l8_fired"] = raw_fired
    if two_factor_suppresses(raw_fired, s_ref_z, ref_maint_z):
        out["fired"] = False
        out["source"] = (
            "single_turn_L21_no_refusal_passthrough"
            f"(s_cat={score:+.4f}>={threshold:+.4f},"
            f"s_ref_z={float(s_ref_z):+.2f}sigma<{ref_maint_z:+.2f}sigma)"
        )
    else:
        out["fired"] = raw_fired
    return out


def apply_in_context_two_factor(
    record: dict[str, Any],
    op_thresh: float,
    ref_maint_z: float = DEFAULT_REF_MAINT_Z_THRESH,
) -> dict[str, Any]:
    """Apply the two-factor gate to a conversation-level observation.

    Differs from `finalize_live_record` in one detail that is visible in the payload: the canonical
    code sets only `raw_gate_fired` here, not the pre-rename `raw_l8_fired` duplicate. Preserved.
    """
    out = dict(record)
    score = float(out.get("score", 0.0))
    threshold = _record_threshold(out, op_thresh)
    s_ref_z = out.get("s_ref_z")
    raw_fired = bool(score >= threshold)
    out["raw_gate_fired"] = raw_fired
    if two_factor_suppresses(raw_fired, s_ref_z, ref_maint_z):
        out["fired"] = False
        out["source"] = (
            "single_turn_L21_no_refusal_passthrough"
            f"(s_cat={score:+.4f}>={threshold:+.4f},"
            f"s_ref_z={float(s_ref_z):+.2f}sigma<{ref_maint_z:+.2f}sigma)"
        )
    else:
        out["fired"] = raw_fired
    return out


def optional_float(value: Any) -> float | None:
    """`float(value)`, preserving `None` -- a missing refusal reading is not zero."""
    return None if value is None else float(value)


def in_context_evidence(
    record: dict[str, Any],
    op_thresh: float,
    op_layer: int,
) -> InContextEvidence:
    """Project an observation record onto the fields the policy may read."""
    score = float(record.get("score", 0.0))
    return InContextEvidence(
        decided=bool(record.get("decided", False)),
        score=score,
        threshold=_record_threshold(record, op_thresh),
        layer=int(record.get("layer", op_layer)),
        s_cat_conv=float(record.get("s_cat_conv", score)),
        s_cat_first=float(record.get("s_cat_first", score)),
        s_cat_last=float(record.get("s_cat_last", score)),
        s_ref_z=optional_float(record.get("s_ref_z")),
        s_ref_cos=optional_float(record.get("s_ref_cos")),
        source=str(record.get("source", "")),
        raw_gate_fired=record.get("raw_gate_fired"),
        raw_l8_fired=record.get("raw_l8_fired"),
    )


def standalone_evidence(record: dict[str, Any], op_thresh: float) -> StandaloneEvidence:
    """Project a standalone observation record onto the fields the policy may read."""
    score = float(record.get("score", 0.0))
    return StandaloneEvidence(
        decided=bool(record.get("decided", False)),
        score=score,
        threshold=_record_threshold(record, op_thresh),
        s_cat_last=float(record.get("s_cat_last", score)),
        s_ref_z=optional_float(record.get("s_ref_z")),
        s_ref_cos=optional_float(record.get("s_ref_cos")),
        source=str(record.get("source", "")),
    )

