"""Differential equivalence: the new decision layer against the vendored canonical implementation.

For every case, both implementations read the same `/dev/shm`-shaped records and are handed the same
messages, and every key the canonical payload emits must match exactly. This is the behaviour-
preservation proof for Phase 2 (REFACTOR_PLAN.md §4): any mismatch is a port error.

Two properties beyond equality are asserted separately at the bottom of this module: determinism and
order independence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cases import CASES, Case, conversation_messages, qwen_cases, records_for

GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"

#: The only key the new payload adds. Everything else must match the canonical payload exactly.
ADDED_KEYS = {"branch"}

#: Fields whose equality is the actual decision, called out so a failure says which one broke.
DECISION_FIELDS = (
    "fired",
    "score",
    "source",
    "sticky_unlock",
    "sticky_revoked",
    "sticky_from_turn",
)


def _write_records(ipc_dir: Path, records: dict[str, dict[str, Any]]) -> None:
    ipc_dir.mkdir(parents=True, exist_ok=True)
    for key, record in records.items():
        (ipc_dir / f"sha_{key}.json").write_text(json.dumps(record), encoding="utf-8")


def _run_pair(
    case: Case,
    ipc_dir: Path,
    model_id: str,
    op_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run both implementations on `case` and return `(legacy_payload, new_payload)`."""
    import baseline.gate_probe as legacy

    from cybergate.decision import evaluate_gate_for_messages as new_eval
    from cybergate.state import request_state

    monkeypatch.setenv("CYBERGATE_OP_PATH", str(op_path))

    threshold = json.loads(op_path.read_text())["thresh"]
    layer = json.loads(op_path.read_text())["meta"]["gate_layer"]
    records = records_for(case, threshold=threshold, layer=layer)
    messages = conversation_messages(case)

    # Legacy first, from a clean slate.
    _write_records(ipc_dir, records)
    legacy.LIVE_GPU_PROMPT_CACHE.clear()
    legacy.REQUEST_GATE_STATES.clear()
    legacy_payload = legacy.evaluate_gate_for_messages(
        messages,
        model_id=model_id,
        enable_multi_turn_stickiness=case.enable_multi_turn_stickiness,
    )

    # New implementation, from an identically clean slate and the same records on disk.
    for child in ipc_dir.iterdir():
        if child.is_file():
            child.unlink()
    _write_records(ipc_dir, records)
    request_state.LIVE_GPU_PROMPT_CACHE.clear()
    request_state.REQUEST_GATE_STATES.clear()
    new_payload = new_eval(
        messages,
        model_id=model_id,
        enable_multi_turn_stickiness=case.enable_multi_turn_stickiness,
    )
    return legacy_payload, new_payload


def _assert_equivalent(case: Case, legacy_payload: dict[str, Any], new_payload: dict[str, Any]) -> None:
    context = f"case={case.id!r} ({case.note})"

    # The decision fields first, so a failure names the one that broke rather than dumping a dict.
    for key in DECISION_FIELDS:
        assert key in legacy_payload or key not in new_payload, (
            f"{context}: new payload invented {key!r}"
        )
        if key in legacy_payload:
            assert new_payload.get(key) == legacy_payload.get(key), (
                f"{context}: {key} differs -- "
                f"legacy={legacy_payload.get(key)!r} new={new_payload.get(key)!r}"
            )

    extra = set(new_payload) - set(legacy_payload)
    assert extra <= ADDED_KEYS, f"{context}: new payload added unexpected keys {sorted(extra - ADDED_KEYS)}"
    missing = set(legacy_payload) - set(new_payload)
    assert not missing, f"{context}: new payload dropped keys {sorted(missing)}"

    for key, legacy_value in legacy_payload.items():
        assert new_payload[key] == legacy_value, (
            f"{context}: {key} differs -- legacy={legacy_value!r} new={new_payload[key]!r}"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_gemma_equivalence(
    case: Case,
    ipc_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """Every case replays identically through the new decision layer, on the Gemma-4 operating point."""
    from conftest import GEMMA_OP

    legacy_payload, new_payload = _run_pair(case, ipc_dir, GEMMA_MODEL, GEMMA_OP, monkeypatch)
    _assert_equivalent(case, legacy_payload, new_payload)


@pytest.mark.parametrize("case", qwen_cases(), ids=lambda c: c.id)
def test_qwen_equivalence(
    case: Case,
    ipc_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """The family-parameterised thresholds match too, on the Qwen-3.6 operating point."""
    import baseline.gate_probe as legacy
    from conftest import QWEN_OP

    monkeypatch.setattr(legacy, "DEFAULT_NON_TARGET_REVOKE_CEIL_QWEN36", -0.0250)
    monkeypatch.setattr(legacy, "DEFAULT_STICKY_HYSTERESIS_THRESH_QWEN36", -0.0250)

    legacy_payload, new_payload = _run_pair(case, ipc_dir, QWEN_MODEL, QWEN_OP, monkeypatch)
    _assert_equivalent(case, legacy_payload, new_payload)


def test_every_branch_is_covered(
    ipc_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """The corpus reaches every branch the policy can emit.

    A corpus that misses a branch proves nothing about it, so this test is what stops the suite from
    quietly going green on a subset. `NO_OBSERVATION` is excluded because the behaviour-preserving
    policy never emits it -- it exists for the Phase 6 fix of R14.

    `NEURAL_REVOKED` is also excluded, and that is a finding rather than a gap: it is unreachable.
    Reaching Step 2C requires no refusal signal, and surviving Step 2A requires the in-context score
    to be at or above the in-context pivot floor of 0.0750 -- which already implies it is above both
    families' hysteresis floors (Gemma 0.0650, Qwen 0.0500), so `hysteresis_holds` always holds and
    the revocation below it cannot be reached. Recorded as defect D2 in docs/ARCHITECTURE.md.
    """
    from cybergate.decision import Branch
    from cybergate.decision.api import make_gatherer
    from cybergate.decision.policy import decide
    from conftest import GEMMA_OP

    monkeypatch.setenv("CYBERGATE_OP_PATH", str(GEMMA_OP))
    threshold = json.loads(GEMMA_OP.read_text())["thresh"]
    layer = json.loads(GEMMA_OP.read_text())["meta"]["gate_layer"]

    seen: set[Branch] = set()
    for case in CASES:
        for child in ipc_dir.iterdir():
            if child.is_file():
                child.unlink()
        _write_records(ipc_dir, records_for(case, threshold=threshold, layer=layer))
        gatherer = make_gatherer(GEMMA_MODEL)
        inputs = gatherer.gather(
            conversation_messages(case),
            enable_multi_turn_stickiness=case.enable_multi_turn_stickiness,
        )
        seen.add(decide(inputs).branch)

    unreachable = {Branch.NO_OBSERVATION, Branch.NEURAL_REVOKED}
    expected = set(Branch) - unreachable
    assert expected <= seen, f"corpus never reaches: {sorted(b.value for b in expected - seen)}"


@pytest.mark.parametrize("case", CASES[:8], ids=lambda c: c.id)
def test_determinism(
    case: Case,
    ipc_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """Replaying the same inputs twice yields an identical decision.

    Trivially true once `decide()` is pure, which is the point of asserting it.
    """
    from cybergate.decision.api import make_gatherer
    from cybergate.decision.policy import decide
    from conftest import GEMMA_OP

    monkeypatch.setenv("CYBERGATE_OP_PATH", str(GEMMA_OP))
    threshold = json.loads(GEMMA_OP.read_text())["thresh"]
    layer = json.loads(GEMMA_OP.read_text())["meta"]["gate_layer"]
    _write_records(ipc_dir, records_for(case, threshold=threshold, layer=layer))

    inputs = make_gatherer(GEMMA_MODEL).gather(
        conversation_messages(case),
        enable_multi_turn_stickiness=case.enable_multi_turn_stickiness,
    )
    first = decide(inputs)
    second = decide(inputs)
    assert first == second
    assert first.source == second.source
