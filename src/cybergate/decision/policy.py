"""`decide(GateInputs) -> GateDecision`. **PURE.**

No cache reads, no HTTP, no globals, no clock, no environment, no torch. The only imports are the
three sibling modules that define the input types, the output types and the named predicates.
`tests/unit/test_purity.py` enforces that by inspecting this module's imports, so the constraint
cannot rot.

This is a behaviour-preserving port of `gate_probe.evaluate_gate_for_messages` lines 1336-1499. Known
defects are preserved on purpose; each is named at the site that causes it and cross-referenced to
its regression test. Fixing them is Phase 6, one commit each, failing test first.

The order of the checks is load-bearing and is the order the canonical code uses:

    stickiness disabled or single turn
      -> two-factor on the chosen score
    prior-unlock scan over earlier turns
      -> no unlock: judge the latest turn standalone
      -> unlock in play:
           Step 2A  non-target-harm pivot      -> revoke
           Step 2B  refusal-monitored maintenance (only when a refusal signal exists)
           Step 2C  category-score hysteresis  -> continue, else revoke

`raw_gate_fired` is deliberately absent from the output: the canonical code computes it from the
in-context observation regardless of which score the branch went on to use, so it is a property of
the observation, not of the decision. The evidence layer attaches it. That split is also what makes
the R12 telemetry defect visible instead of structural.
"""
from __future__ import annotations

from .inputs import GateInputs, InContextEvidence, StandaloneEvidence
from .outcome import Branch, GateDecision
from .rules import (
    hysteresis_holds,
    is_nontarget_harm_pivot,
    maintenance_holds,
    prior_unlock_discarded,
    two_factor_suppresses,
)

__all__ = ["decide"]


def decide(inputs: GateInputs) -> GateDecision:
    """Decide whether the gate fires for the conversation described by `inputs`."""
    op = inputs.op
    latest = inputs.latest
    standalone = inputs.standalone

    # The record's own threshold wins over the operating point's, because the record was written by
    # whichever operator the worker had loaded. Canonical:
    # `float(latest_dec.get("threshold", op.thresh))`.
    threshold = latest.threshold
    layer = latest.layer

    s_cat_in_context = latest.score
    s_cat_standalone = standalone.score if standalone is not None else s_cat_in_context
    s_cat_conv = latest.s_cat_conv
    s_cat_last = latest.s_cat_last

    # The in-context refusal signal, falling back to the standalone one when absent.
    s_ref_z = latest.s_ref_z
    if s_ref_z is None and standalone is not None:
        s_ref_z = standalone.s_ref_z
    st_ref_z = standalone.s_ref_z if standalone is not None else None

    # ---------------------------------------------------------------------------------------------
    # Single turn, or multi-turn with stickiness disabled.
    # ---------------------------------------------------------------------------------------------
    if inputs.n_turns <= 1 or not inputs.enable_multi_turn_stickiness:
        if inputs.n_turns > 1 and not inputs.enable_multi_turn_stickiness and standalone is not None:
            return _decide_stateless(standalone, threshold, op.ref_maint_z)
        return _decide_in_context(latest, s_ref_z, threshold, op.ref_maint_z)

    # ---------------------------------------------------------------------------------------------
    # Prior-unlock scan. Walks earlier turns oldest-first, tracking the most recent one that fired.
    # ---------------------------------------------------------------------------------------------
    prior_turn: int | None = None
    prior_score: float | None = None
    for prev in inputs.priors:
        if prev.fired:
            prior_turn = prev.index
            prior_score = prev.score
        elif prior_unlock_discarded(
            prev.score, prev.s_ref_z, op.revoke_ceil, threshold, op.st_ref_z_pivot
        ):
            # R6/R7: discards a legitimate unlock, and because the canonical loop has no `break` and
            # the `prior_turn is None` exit below returns before Step 2B is consulted, the following
            # turn is denied even when refusal pressure is far above the maintenance floor.
            prior_turn = None
            prior_score = None

    if prior_turn is None:
        # No earlier turn unlocked the session: judge the latest turn on its standalone two-factor
        # decision. With no standalone observation the canonical code returns the in-context record
        # exactly as the two-factor gate above left it.
        if standalone is None:
            return _decide_in_context(latest, s_ref_z, threshold, op.ref_maint_z)
        raw_fired = bool(standalone.score >= standalone.threshold)
        suppressed = two_factor_suppresses(raw_fired, standalone.s_ref_z, op.ref_maint_z)
        return GateDecision(
            fired=bool(raw_fired and not suppressed),
            branch=Branch.NO_PRIOR_UNLOCK,
            score=round(standalone.score, 6),
            inherited_source=_nested_standalone_source(standalone, op.ref_maint_z, suppressed),
            evidence={
                "s_cat_standalone": standalone.score,
                "threshold": standalone.threshold,
                "s_ref_z": float(standalone.s_ref_z) if standalone.s_ref_z is not None else None,
                "ref_maint_z": op.ref_maint_z,
            },
        )

    # ---------------------------------------------------------------------------------------------
    # An unlock is in play.
    # ---------------------------------------------------------------------------------------------
    st_cat_last = standalone.s_cat_last if standalone is not None else s_cat_last

    # Step 2A: cross-domain non-target-harm revocation.
    if is_nontarget_harm_pivot(
        s_cat_standalone=s_cat_standalone,
        s_cat_in_context=s_cat_in_context,
        s_cat_last=s_cat_last,
        st_cat_last=st_cat_last,
        s_ref_z=s_ref_z,
        st_ref_z=st_ref_z,
        thresh=threshold,
        revoke_ceil=op.revoke_ceil,
        in_context_pivot_floor=op.in_context_pivot_floor,
        s_cat_last_pivot_floor=op.s_cat_last_pivot_floor,
        st_ref_z_pivot=op.st_ref_z_pivot,
    ):
        # R10: a benign pleasantry reaches this branch through the first two disjuncts and is
        # reported as a non-target-harm pivot.
        return GateDecision(
            fired=False,
            branch=Branch.NONTARGET_HARM_REVOKED,
            score=round(min(s_cat_standalone, s_cat_in_context), 6),
            sticky_revoked=True,
            sticky_from_turn=prior_turn,
            prior_unlock_score=prior_score,
            evidence={
                "layer": layer,
                "turn": inputs.n_turns,
                "s_cat_standalone": s_cat_standalone,
                "s_cat_in_context": s_cat_in_context,
                "s_cat_last_min": min(s_cat_last, st_cat_last),
                "st_ref_z": st_ref_z,
                "s_ref_z": s_ref_z,
            },
        )

    # Step 2B: refusal-monitored session maintenance. Only consulted when a refusal signal exists;
    # with none, control falls through to the Step 2C hysteresis check below.
    if s_ref_z is not None:
        ref_z_val = float(s_ref_z)
        st_z_val = float(st_ref_z) if st_ref_z is not None else ref_z_val
        shared = {
            "turn": inputs.n_turns,
            "s_ref_z": ref_z_val,
            "ref_maint_z": op.ref_maint_z,
            "s_cat_in_context": s_cat_in_context,
            "s_cat_standalone": s_cat_standalone,
        }
        if maintenance_holds(max(ref_z_val, st_z_val), op.ref_maint_z):
            return GateDecision(
                fired=True,
                branch=Branch.MAINTAINED,
                score=latest.score,
                sticky_unlock=True,
                sticky_from_turn=prior_turn,
                prior_unlock_score=prior_score,
                evidence={**shared, "from_turn": prior_turn, "s_cat_conv": s_cat_conv},
            )
        return GateDecision(
            fired=False,
            branch=Branch.NO_REFUSAL_PASSTHROUGH,
            score=latest.score,
            sticky_from_turn=prior_turn,
            prior_unlock_score=prior_score,
            evidence=shared,
        )

    # Step 2C: offline / fallback hysteresis.
    if hysteresis_holds(s_cat_in_context, s_cat_conv, op.hysteresis_thresh, threshold):
        return GateDecision(
            fired=True,
            branch=Branch.HYSTERESIS_CONTINUED,
            score=latest.score,
            sticky_unlock=True,
            sticky_from_turn=prior_turn,
            prior_unlock_score=prior_score,
            evidence={
                "layer": layer,
                "from_turn": prior_turn,
                "s_cat_in_context": s_cat_in_context,
                "hysteresis_thresh": op.hysteresis_thresh,
            },
        )

    return GateDecision(
        fired=False,
        branch=Branch.NEURAL_REVOKED,
        score=latest.score,
        sticky_revoked=True,
        sticky_from_turn=prior_turn,
        prior_unlock_score=prior_score,
        evidence={
            "layer": layer,
            "turn": inputs.n_turns,
            "s_cat_in_context": s_cat_in_context,
            "hysteresis_thresh": op.hysteresis_thresh,
        },
    )


def _decide_in_context(
    latest: InContextEvidence,
    s_ref_z: float | None,
    threshold: float,
    ref_maint_z: float,
) -> GateDecision:
    """The single-turn two-factor gate, on the in-context score.

    Reached on turn 1, on a multi-turn request with stickiness disabled and no standalone
    observation, and on a multi-turn request where no prior turn unlocked and no standalone
    observation exists.
    """
    raw_fired = bool(latest.score >= threshold)
    if two_factor_suppresses(raw_fired, s_ref_z, ref_maint_z):
        return GateDecision(
            fired=False,
            branch=Branch.SINGLE_TURN_TWO_FACTOR_SUPPRESSED,
            score=latest.score,
            evidence={
                "s_cat": latest.score,
                "threshold": threshold,
                "s_ref_z": float(s_ref_z) if s_ref_z is not None else None,
                "ref_maint_z": ref_maint_z,
            },
        )
    return GateDecision(
        fired=raw_fired,
        branch=Branch.SINGLE_TURN_RAW,
        score=latest.score,
        inherited_source=latest.source,
        evidence={"s_cat": latest.score, "threshold": threshold},
    )


def _decide_stateless(
    standalone: StandaloneEvidence,
    threshold: float,
    ref_maint_z: float,
) -> GateDecision:
    """Stateless evaluation of a multi-turn request: the latest turn's standalone score only.

    Note the threshold: the canonical code uses the *in-context* record's threshold here, not the
    standalone record's, which is why `threshold` is passed in rather than read off `standalone`.
    """
    raw_fired = bool(standalone.score >= threshold)
    if two_factor_suppresses(raw_fired, standalone.s_ref_z, ref_maint_z):
        return GateDecision(
            fired=False,
            branch=Branch.STATELESS_TWO_FACTOR_SUPPRESSED,
            score=round(standalone.score, 6),
            evidence={
                "s_cat_standalone": standalone.score,
                "threshold": threshold,
                "s_ref_z": float(standalone.s_ref_z) if standalone.s_ref_z is not None else None,
                "ref_maint_z": ref_maint_z,
            },
        )
    return GateDecision(
        fired=raw_fired,
        branch=Branch.STATELESS_EVAL,
        score=round(standalone.score, 6),
        evidence={"s_cat_standalone": standalone.score, "threshold": threshold},
    )


def _nested_standalone_source(
    standalone: StandaloneEvidence,
    ref_maint_z: float,
    suppressed: bool,
) -> str:
    """Render the `source=` a nested standalone evaluation would have reported.

    The canonical `NO_PRIOR_UNLOCK` branch calls back into the single-prompt evaluator and copies its
    `source=`, which is either the suppression string or the standalone observation's own source.
    Built here rather than in `outcome.render_source` because it reproduces a *nested* evaluation's
    output rather than this branch's own.
    """
    if suppressed:
        s_ref_z = standalone.s_ref_z
        return (
            "single_turn_L21_no_refusal_passthrough"
            f"(s_cat={standalone.score:+.4f}>={standalone.threshold:+.4f},"
            f"s_ref_z={float(s_ref_z):+.2f}sigma<{ref_maint_z:+.2f}sigma)"
        )
    return standalone.source
