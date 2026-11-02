"""R11 end to end: a decision transferring between conversations through the evidence layer.

`test_keys.py` shows the in-context key collides. This shows what the collision *does*: drive two
conversations with different histories but the same final user turn through the real evidence layer,
backed by the in-process store, and watch the second be decided on the first's score.

Both conversations open with a turn that unlocks, so both reach the maintenance/revocation logic. They
differ only in their in-context score for the shared final turn, which is the number the collision
overwrites:

    conversation A   in-context 0.1400  -> above the 0.0750 pivot floor -> maintains
    conversation B   in-context 0.0700  -> below it                    -> revokes

This is the test that was impossible before the refactor. It needs no GPU and no server, only a dict.
"""
from __future__ import annotations

import pytest

from cybergate.decision import Branch
from cybergate.decision.evidence import EvidenceGatherer
from cybergate.decision.policy import decide
from cybergate.families import family_by_name
from cybergate.ops.operator import CyberGateOperator
from cybergate.state import keys as k
from cybergate.state.memory import MemoryDecisionStore

TAU = 0.10807704

#: The turn both conversations end with. Only its hash matters.
SHARED_FINAL_TURN = "shared final turn text"
FIRST_TURN_A = "conversation A opening turn"
FIRST_TURN_B = "conversation B opening turn"

#: Live numbers from R5/R8: a follow-up that maintains on refusal pressure.
UNLOCK_SCORE = 0.1961
FINAL_STANDALONE_SCORE = 0.0726
FINAL_REF_Z = 2.00

IN_CONTEXT_A = 0.1400  # above the 0.0750 in-context pivot floor
IN_CONTEXT_B = 0.0700  # below it


def _op() -> CyberGateOperator:
    return CyberGateOperator(
        c=[1.0, 0.0, 0.0, 0.0],
        thresh=TAU,
        gate_layer=10,
        pool="mean_user",
        category="cyber_offense",
        model_key="gemma",
        meta={},
    )


def _record(score: float, s_ref_z: float, *, multiturn: bool = False) -> dict:
    return {
        "decided": True,
        "fired": score >= TAU,
        "score": score,
        "threshold": TAU,
        "layer": 10,
        "s_cat_conv": score,
        "s_cat_first": score,
        "s_cat_last": score,
        "s_ref_z": s_ref_z,
        "s_ref_cos": 0.1374208 + s_ref_z * 0.0513015,
        "is_multiturn_ctx": multiturn,
        "is_baseline_prefill": True,
        "source": "live_vllm_gpu_L10_mean_user_hook(tokens=24,turns=1,span=4:20,gpu_dot_us=66.0)",
    }


def _messages(first_turn: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": first_turn},
        {"role": "assistant", "content": "filler"},
        {"role": "user", "content": SHARED_FINAL_TURN},
    ]


def _populate(store: MemoryDecisionStore, first_turn: str, in_context_score: float) -> None:
    """Write the records the worker hook would publish for one conversation."""
    opening = _record(UNLOCK_SCORE, 8.0)
    store.put(k.standalone_turn_key(first_turn), opening)
    store.put(k.bare_turn_key(first_turn), opening)

    final_standalone = _record(FINAL_STANDALONE_SCORE, FINAL_REF_Z)
    store.put(k.standalone_turn_key(SHARED_FINAL_TURN), final_standalone)
    store.put(k.bare_turn_key(SHARED_FINAL_TURN), final_standalone)

    in_context = _record(in_context_score, FINAL_REF_Z, multiturn=True)
    store.put(k.conversation_key([first_turn, SHARED_FINAL_TURN]), in_context)


def _decide(store: MemoryDecisionStore, first_turn: str):
    gatherer = EvidenceGatherer(
        store=store, op=_op(), family=family_by_name("gemma4"), family_tag="gemma"
    )
    return decide(gatherer.gather(_messages(first_turn)))


def test_the_two_conversations_differ_on_their_own_merits() -> None:
    """Positive control: given only its own records, each conversation decides differently.

    Without this, a later assertion that B inherited A's decision could just mean neither reached the
    policy at all.
    """
    store_a = MemoryDecisionStore()
    _populate(store_a, FIRST_TURN_A, IN_CONTEXT_A)
    decision_a = _decide(store_a, FIRST_TURN_A)

    store_b = MemoryDecisionStore()
    _populate(store_b, FIRST_TURN_B, IN_CONTEXT_B)
    decision_b = _decide(store_b, FIRST_TURN_B)

    assert decision_a.fired is True
    assert decision_a.branch is Branch.MAINTAINED
    assert decision_a.sticky_from_turn == 1

    assert decision_b.fired is False
    assert decision_b.branch is Branch.NONTARGET_HARM_REVOKED


def test_the_conversation_record_wins_over_a_stale_legacy_ctx_record() -> None:
    """R11 fixed, and fixed against the upgrade case specifically.

    `/dev/shm/cybergate_ipc` outlives a container restart, so after this upgrade the store can still
    hold `ctx_<sha(final turn)>` records written by the previous build. Those must be ignored, not
    merely deprioritised: the canonical code read that key *first*, so a surviving record would
    otherwise keep leaking across conversations after the fix shipped.
    """
    store = MemoryDecisionStore()
    _populate(store, FIRST_TURN_A, IN_CONTEXT_A)
    # The correctly-keyed record says one thing...
    store.put(
        k.conversation_key([FIRST_TURN_A, SHARED_FINAL_TURN]),
        _record(IN_CONTEXT_B, FINAL_REF_Z, multiturn=True),
    )
    # ...a leftover record under the old key says another. It must not be consulted.
    store.put(
        k.legacy_in_context_key_from_sha(k.text_sha256(SHARED_FINAL_TURN)),
        _record(IN_CONTEXT_A, FINAL_REF_Z, multiturn=True),
    )

    decision = _decide(store, FIRST_TURN_A)
    assert decision.evidence.get("s_cat_in_context") == pytest.approx(IN_CONTEXT_B)


def test_a_decision_must_not_transfer_between_conversations() -> None:
    """A is served first into a shared store; B must still be decided on its own score.

    On its own records B revokes. Under the canonical key choice it maintained, because A owned
    `ctx_<sha(final turn)>` -- so the collision flipped the outcome, not just the reported number.
    The legacy record is written here too, to prove it is ignored rather than merely outranked.
    """
    shared_store = MemoryDecisionStore()
    _populate(shared_store, FIRST_TURN_A, IN_CONTEXT_A)
    assert _decide(shared_store, FIRST_TURN_A).fired is True

    _populate(shared_store, FIRST_TURN_B, IN_CONTEXT_B)
    shared_store.put(
        k.legacy_in_context_key_from_sha(k.text_sha256(SHARED_FINAL_TURN)),
        _record(IN_CONTEXT_A, FINAL_REF_Z, multiturn=True),
    )

    decision_b = _decide(shared_store, FIRST_TURN_B)
    assert decision_b.fired is False, "B should be decided on its own in-context score"
    assert decision_b.branch is Branch.NONTARGET_HARM_REVOKED
