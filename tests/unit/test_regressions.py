"""One test per audit finding, R1-R14 from REFACTOR_PLAN.md §5.

Numbers are the exact live measurements from the audit. Operating point throughout:
tau = 0.10807704 (Gemma-4 L10 `mean_user`), `ref_maint_z` = 1.50, `revoke_ceil` = 0.0520,
in-context floor 0.0750, standalone refusal pivot 7.0.

Cases the behaviour-preserving port cannot satisfy are marked `xfail(strict=True)`, so Phase 6
flipping one turns the test green loudly rather than silently. Each xfail reason names the defect.
"""
from __future__ import annotations

import pytest

from cybergate.decision import (
    Branch,
    GateInputs,
    InContextEvidence,
    OperatingPoint,
    PriorTurnEvidence,
    StandaloneEvidence,
    decide,
)

TAU = 0.10807704
OP = OperatingPoint(thresh=TAU, gate_layer=10, revoke_ceil=0.0520, hysteresis_thresh=0.0650)


def _in_context(score: float, s_ref_z: float | None = None, **kwargs: float) -> InContextEvidence:
    return InContextEvidence(
        decided=True,
        score=score,
        threshold=TAU,
        layer=10,
        s_cat_conv=kwargs.get("s_cat_conv", score),
        s_cat_first=kwargs.get("s_cat_first", score),
        s_cat_last=kwargs.get("s_cat_last", score),
        s_ref_z=s_ref_z,
        source="live_vllm_gpu_L10_mean_user_hook(tokens=24,turns=1,span=4:20,gpu_dot_us=66.0)",
        raw_gate_fired=score >= TAU,
    )


def _standalone(score: float, s_ref_z: float | None = None, s_cat_last: float | None = None):
    return StandaloneEvidence(
        decided=True,
        score=score,
        threshold=TAU,
        s_cat_last=s_cat_last if s_cat_last is not None else score,
        s_ref_z=s_ref_z,
        source="live_vllm_gpu_L10_mean_user_hook(tokens=24,turns=1,span=4:20,gpu_dot_us=66.0)",
    )


def _turn(index: int, score: float, s_ref_z: float | None, fired: bool) -> PriorTurnEvidence:
    return PriorTurnEvidence(index=index, decided=True, fired=fired, score=score, s_ref_z=s_ref_z)


# =================================================================================================
# R1-R3: the two-factor gate applies on turn 2, not only on turn 1.
# =================================================================================================
@pytest.mark.parametrize(
    ("label", "s_cat_st", "s_ref_z"),
    [("R1", 0.1160, -0.27), ("R2", 0.1326, 0.42)],
)
def test_r1_r2_two_factor_applies_on_turn_two(label: str, s_cat_st: float, s_ref_z: float) -> None:
    """Above tau but refusal pressure below the floor: the gate holds, even on turn 2.

    Turn 1 scored below tau so nothing unlocked, which is what routes this through the standalone
    two-factor test rather than through maintenance.
    """
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(s_cat_st + 0.01, s_ref_z),
        standalone=_standalone(s_cat_st, s_ref_z),
        priors=(_turn(1, 0.0600, 0.20, fired=False),),
    )
    decision = decide(inputs)
    assert decision.fired is False, label
    assert decision.branch is Branch.NO_PRIOR_UNLOCK
    assert "no_refusal_passthrough" in decision.source


def test_r3_positive_control_still_fires() -> None:
    """The same shape with real refusal pressure must still fire, or the gate is simply off."""
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.2000, 10.50),
        standalone=_standalone(0.1961, 10.50),
        priors=(_turn(1, 0.0600, 0.20, fired=False),),
    )
    assert decide(inputs).fired is True


# =================================================================================================
# R4: no self-authorization.
# =================================================================================================
def test_r4_high_first_turn_score_does_not_synthesize_an_unlock() -> None:
    """`s_cat_first` is the policy's own derived number and must not create a prior unlock.

    The conversation-wide and first-turn scores are computed from the same prefill the decision is
    about, so letting them stand in for a prior turn's decision would let a request authorise itself.
    """
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.1600, 5.0, s_cat_first=0.1900, s_cat_conv=0.1800),
        standalone=_standalone(0.1500, 5.0),
        priors=(_turn(1, 0.0600, 0.20, fired=False),),
    )
    decision = decide(inputs)
    assert decision.sticky_from_turn is None
    assert decision.sticky_unlock is False
    assert decision.branch is Branch.NO_PRIOR_UNLOCK


# =================================================================================================
# R5, R8: a legitimate unlock propagates.
# =================================================================================================
def test_r5_legitimate_unlock_propagates() -> None:
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.1402, 6.69, s_cat_last=0.0820, s_cat_conv=0.1510),
        standalone=_standalone(0.0726, 6.69),
        priors=(_turn(1, 0.1961, 8.0, fired=True),),
    )
    decision = decide(inputs)
    assert decision.fired is True
    assert decision.sticky_unlock is True
    assert decision.sticky_from_turn == 1
    assert decision.branch is Branch.MAINTAINED


def test_r8_control_without_an_interleaved_turn() -> None:
    """R6's shape minus the interleaved turn, isolating that turn as the cause."""
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.1300, 6.69, s_cat_last=0.0900, s_cat_conv=0.1500),
        standalone=_standalone(0.0726, 6.69),
        priors=(_turn(1, 0.1961, 8.0, fired=True),),
    )
    assert decide(inputs).fired is True


# =================================================================================================
# R6, R7: the prior-turn wipe. The live regression.
# =================================================================================================
def _wipe_inputs(interjection_score: float, final_score: float, final_ref_z: float) -> GateInputs:
    return GateInputs(
        op=OP,
        n_turns=3,
        latest=_in_context(0.1200, final_ref_z, s_cat_last=0.0900, s_cat_conv=0.1400),
        standalone=_standalone(final_score, final_ref_z),
        priors=(
            _turn(1, 0.1961, 8.0, fired=True),
            _turn(2, interjection_score, 0.30, fired=False),
        ),
    )


@pytest.mark.parametrize(
    ("label", "interjection", "final_score", "final_ref_z"),
    [("R6", 0.0251, 0.0580, 6.12), ("R7", 0.0079, 0.0654, 6.33)],
)
def test_r6_r7_current_behaviour_denies_the_turn(
    label: str, interjection: float, final_score: float, final_ref_z: float
) -> None:
    """Documents the defect as it stands: the unlock is discarded and the turn is denied.

    This is the *current* behaviour, asserted so that Phase 6 cannot change it by accident. The
    desired behaviour is the xfail below.
    """
    decision = decide(_wipe_inputs(interjection, final_score, final_ref_z))
    assert decision.fired is False, label
    assert decision.sticky_from_turn is None
    assert decision.branch is Branch.NO_PRIOR_UNLOCK


@pytest.mark.xfail(
    strict=True,
    reason="R6/R7 unfixed: one interjection below revoke_ceil discards a legitimate unlock, and the "
    "no-unlock exit returns before Step 2B maintenance is consulted, so a turn at 4x the "
    "maintenance floor is denied. Phase 6 fix #1.",
)
@pytest.mark.parametrize(
    ("label", "interjection", "final_score", "final_ref_z"),
    [("R6", 0.0251, 0.0580, 6.12), ("R7", 0.0079, 0.0654, 6.33)],
)
def test_r6_r7_desired_behaviour_maintains_the_session(
    label: str, interjection: float, final_score: float, final_ref_z: float
) -> None:
    decision = decide(_wipe_inputs(interjection, final_score, final_ref_z))
    assert decision.fired is True, label
    assert decision.sticky_from_turn == 1
    assert decision.branch is Branch.MAINTAINED


def test_r9_recovery_after_a_wipe() -> None:
    """A turn that clears tau on its own standalone score re-unlocks, with no carried unlock.

    One live observation only. Passes under the current code, so it is not marked xfail -- the plan
    grouped it with the unfixed cases, but the port already gives the expected result.
    """
    inputs = GateInputs(
        op=OP,
        n_turns=4,
        latest=_in_context(0.1850, 7.20, s_cat_last=0.1500, s_cat_conv=0.1700),
        standalone=_standalone(0.1798, 7.20),
        priors=(
            _turn(1, 0.1961, 8.0, fired=True),
            _turn(2, 0.0251, 0.30, fired=False),
            _turn(3, 0.0580, 6.12, fired=False),
        ),
    )
    decision = decide(inputs)
    assert decision.fired is True
    assert decision.sticky_from_turn is None


# =================================================================================================
# R10: a pleasantry is not a CBRN pivot.
# =================================================================================================
def _pleasantry_inputs() -> GateInputs:
    return GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.0900, 0.30, s_cat_last=0.0400, s_cat_conv=0.1100),
        standalone=_standalone(0.0251, 0.30),
        priors=(_turn(1, 0.1961, 8.0, fired=True),),
    )


def test_r10_current_behaviour_labels_a_pleasantry_as_a_pivot() -> None:
    """Documents the mislabel: a benign low-scoring turn takes the non-target-harm branch."""
    assert decide(_pleasantry_inputs()).branch is Branch.NONTARGET_HARM_REVOKED


@pytest.mark.xfail(
    strict=True,
    reason="R10 unfixed: the first two pivot disjuncts test only score magnitude, with no guard on "
    "intent, so an ordinary pleasantry is reported as a non-target-harm pivot. The revocation "
    "may be correct; the label is not. Phase 6 fix #3.",
)
def test_r10_desired_behaviour_does_not_label_it_a_pivot() -> None:
    assert decide(_pleasantry_inputs()).branch is not Branch.NONTARGET_HARM_REVOKED


# =================================================================================================
# R12, R13: telemetry integrity.
# =================================================================================================
def test_r12_current_behaviour_reports_the_in_context_raw_gate_fired() -> None:
    """Documents the defect: `raw_gate_fired` comes from the in-context score on every branch.

    Here the branch decides on the standalone score (0.0726, below tau) while `raw_gate_fired`
    reports the in-context score (0.1402, above tau), so the field contradicts the decision.
    """
    latest = _in_context(0.1402, 0.40, s_cat_last=0.0820, s_cat_conv=0.1510)
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=latest,
        standalone=_standalone(0.0726, 0.40),
        priors=(_turn(1, 0.0600, 0.20, fired=False),),
    )
    decision = decide(inputs)
    assert decision.fired is False
    assert latest.raw_gate_fired is True, "in-context score is above tau"


@pytest.mark.xfail(
    strict=True,
    reason="R12 unfixed: `raw_gate_fired` is a property of the in-context observation and does not "
    "track the score the chosen branch used, so it is unusable for attribution. Phase 6 fix #4.",
)
def test_r12_desired_behaviour_raw_gate_fired_matches_the_branch_score() -> None:
    inputs = GateInputs(
        op=OP,
        n_turns=2,
        latest=_in_context(0.1402, 0.40, s_cat_last=0.0820, s_cat_conv=0.1510),
        standalone=_standalone(0.0726, 0.40),
        priors=(_turn(1, 0.0600, 0.20, fired=False),),
    )
    decision = decide(inputs)
    branch_score = decision.evidence.get("s_cat_standalone")
    assert branch_score is not None
    assert getattr(decision, "raw_gate_fired", None) == (branch_score >= TAU)


@pytest.mark.xfail(
    strict=True,
    reason="R13 unfixed: the NO_PRIOR_UNLOCK branch copies a standalone sub-probe's source string "
    "verbatim, and that probe saw one turn, so a 3-turn request reports `turns=1`. The payload's "
    "`user_turn_index` is correct; only the source string lies. Phase 6 fix #4.",
)
def test_r13_turn_count_in_the_source_string_is_honest() -> None:
    inputs = GateInputs(
        op=OP,
        n_turns=3,
        latest=_in_context(0.1200, 5.0),
        standalone=_standalone(0.1500, 5.0),
        priors=(
            _turn(1, 0.0600, 0.20, fired=False),
            _turn(2, 0.0700, 0.30, fired=False),
        ),
    )
    assert "turns=3" in decide(inputs).source


# =================================================================================================
# R14: abstain is explicit.
# =================================================================================================
def test_r14_current_behaviour_treats_no_observation_as_score_zero() -> None:
    """Documents the defect: an undecided observation falls through with `score = 0.0`."""
    inputs = GateInputs(
        op=OP,
        n_turns=1,
        latest=InContextEvidence(
            decided=False,
            score=0.0,
            threshold=TAU,
            layer=10,
            s_cat_conv=0.0,
            s_cat_first=0.0,
            s_cat_last=0.0,
            source="no_live_gpu_hook_attached",
        ),
    )
    decision = decide(inputs)
    assert decision.fired is False
    assert decision.branch is Branch.SINGLE_TURN_RAW, "falls through a normal branch, not an abstain"


@pytest.mark.xfail(
    strict=True,
    reason="R14 unfixed: 'no fresh observation' is not a typed state in the policy -- an undecided "
    "record falls through to the ordinary score comparison with 0.0. Branch.NO_OBSERVATION "
    "exists for the fix. Phase 6 fix #5.",
)
def test_r14_desired_behaviour_is_an_explicit_abstain() -> None:
    inputs = GateInputs(
        op=OP,
        n_turns=1,
        latest=InContextEvidence(
            decided=False,
            score=0.0,
            threshold=TAU,
            layer=10,
            s_cat_conv=0.0,
            s_cat_first=0.0,
            s_cat_last=0.0,
            source="no_live_gpu_hook_attached",
        ),
    )
    assert decide(inputs).branch is Branch.NO_OBSERVATION
