"""Unit tests for the multi-probe category factor (`meta.or_probes`), which is how rule C serves.

The baseline gate is one layer against one threshold. Rule C is an OR over two bundle probes, so
`_or_probe_fires` resolves each named probe's score from the candidate-bundle telemetry and its
threshold from that probe's own bundle entry. These tests pin the three behaviours a deployment
depends on: a baseline operator is untouched, a hit fires, and a probe that never reported is
recorded as missing rather than silently treated as not-firing.
"""
from __future__ import annotations

from typing import Any

from cybergate.hooks.category_probe import _or_probe_fires
from cybergate.ops import CyberGateOperator

_TAU20 = 0.039106916270767171


def _op(**meta: Any) -> CyberGateOperator:
    return CyberGateOperator(
        c=[1.0, 0.0], thresh=0.04563936697855326, gate_layer=21, pool="last",
        category="cyber_offense", model_key="gemma", meta=meta,
    )


def _state() -> dict[str, Any]:
    return {"bundle_thresholds": {"L20_last": {"R2": _TAU20, "R3": 0.03343554604087408}}}


def test_baseline_operator_declares_no_or_probes() -> None:
    """Every shipped baseline operator must take the `None` path, so the gate is byte-identical."""
    assert _or_probe_fires(_op(gate_layer=16, pool="mean_user"), _state(), {"L20_last": 0.9}) is None


def test_or_probe_above_its_bundle_threshold_fires() -> None:
    info = _or_probe_fires(_op(or_probes=["L20_last"]), _state(), {"L20_last": _TAU20 + 1e-6})
    assert info == {"rule": "R2", "hits": ["L20_last"], "missing": []}


def test_or_probe_below_its_bundle_threshold_does_not_fire() -> None:
    info = _or_probe_fires(_op(or_probes=["L20_last"]), _state(), {"L20_last": _TAU20 - 1e-6})
    assert info == {"rule": "R2", "hits": [], "missing": []}


def test_or_probe_fires_exactly_at_threshold() -> None:
    """`>=`, matching the primary gate's own comparison in the hook."""
    info = _or_probe_fires(_op(or_probes=["L20_last"]), _state(), {"L20_last": _TAU20})
    assert info["hits"] == ["L20_last"]


def test_threshold_rule_is_read_from_the_operator() -> None:
    """R3 is looser than R2 here, so a score between them fires only under R3."""
    mid = 0.0360
    assert _or_probe_fires(_op(or_probes=["L20_last"]), _state(), {"L20_last": mid})["hits"] == []
    info = _or_probe_fires(
        _op(or_probes=["L20_last"], or_probes_thresh_rule="R3"), _state(), {"L20_last": mid}
    )
    assert info == {"rule": "R3", "hits": ["L20_last"], "missing": []}


def test_absent_bundle_score_is_reported_missing_not_silently_passed() -> None:
    """A bundle hook that did not run must be visible on the record, not an invisible under-fire."""
    info = _or_probe_fires(_op(or_probes=["L20_last"]), _state(), None)
    assert info == {"rule": "R2", "hits": [], "missing": ["L20_last"]}


def test_absent_bundle_threshold_is_reported_missing() -> None:
    """The threshold is resolved from the bundle at serve time; a bundle without it cannot fire."""
    info = _or_probe_fires(_op(or_probes=["L20_last"]), {"bundle_thresholds": {}}, {"L20_last": 0.9})
    assert info == {"rule": "R2", "hits": [], "missing": ["L20_last"]}
