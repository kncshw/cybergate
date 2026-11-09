"""The named predicates the policy is built from. Each one pure, each one independently testable.

In the canonical package these conditions are inline boolean expressions inside a 299-line function,
which is why every one of them was only ever exercised against a live 26B model. Naming them is what
makes a sentence like "a pleasantry is not a CBRN pivot" (R10) an assertion about one function.
"""
from __future__ import annotations

__all__ = [
    "hysteresis_holds",
    "is_nontarget_harm_pivot",
    "maintenance_holds",
    "prior_unlock_discarded",
    "two_factor_suppresses",
]


def two_factor_suppresses(
    raw_fired: bool,
    s_ref_z: float | None,
    ref_maint_z: float,
) -> bool:
    """True when the category score cleared the threshold but refusal pressure did not.

    The second factor of the gate: a prompt that is cyber-topical but that the model was not about to
    refuse gets no intervention, because there is no refusal to remove. A benign technical question
    ("explain TCP vs UDP") lands here.

    `s_ref_z is None` means no refusal observation was available, and the canonical code then skips
    the second factor entirely rather than failing closed.
    """
    return bool(raw_fired and s_ref_z is not None and float(s_ref_z) < ref_maint_z)


def prior_unlock_discarded(
    prev_score: float,
    prev_s_ref_z: float | None,
    revoke_ceil: float,
    thresh: float,
    st_ref_z_pivot: float,
) -> bool:
    """True when an earlier, non-firing turn discards any unlock carried from before it.

    Called for each prior turn that did **not** fire. Two ways to discard:
      * the turn scores below the non-target-harm revocation ceiling, or
      * it scores below the gate threshold while standalone refusal pressure is at or above the pivot
        level, which is the signature of a pivot to another harm domain.

    **This predicate is the R6/R7 live regression.** The first disjunct fires on roughly half of
    natural pleasantries (measured: 0.0196, 0.0251, 0.0377 discard; 0.0558, 0.0619, 0.0724, 0.0772 do
    not), and the canonical loop that calls it neither breaks nor lets Step 2B maintenance run
    afterwards -- so one low-scoring interjection throws away a legitimate unlock even when the
    refusal monitor is at four times the maintenance floor. Preserved as-is; see
    `tests/unit/test_regressions.py` (R6).
    """
    if prev_score < revoke_ceil:
        return True
    return bool(
        prev_score < thresh
        and prev_s_ref_z is not None
        and float(prev_s_ref_z) >= st_ref_z_pivot
    )


def is_nontarget_harm_pivot(
    s_cat_standalone: float,
    s_cat_in_context: float,
    s_cat_last: float,
    st_cat_last: float,
    s_ref_z: float | None,
    st_ref_z: float | None,
    thresh: float,
    revoke_ceil: float,
    in_context_pivot_floor: float,
    s_cat_last_pivot_floor: float,
    st_ref_z_pivot: float,
) -> bool:
    """True when the latest turn pivots out of the target domain, inside an unlocked session.

    Four disjuncts, any one sufficient:
      1. the standalone score falls into the non-target-harm zone (`< revoke_ceil`),
      2. the in-context score falls below the in-context floor,
      3. the standalone score is below threshold while standalone refusal pressure is at the pivot
         level,
      4. the last-token score is very low on both views while either refusal signal is at the pivot
         level and the standalone score is below threshold.

    A missing refusal observation is treated as `0.0`, so disjuncts 3 and 4 cannot fire on it.

    **Disjuncts 1 and 2 have no intent guard**, which is the R10 defect: an ordinary pleasantry
    scores below both floors and is therefore labelled a non-target-harm pivot. The label is wrong;
    the resulting revocation may or may not be. Preserved as-is.
    """
    eff_ref_z = float(s_ref_z) if s_ref_z is not None else 0.0
    eff_st_ref_z = float(st_ref_z) if st_ref_z is not None else 0.0
    st_pivot_ceil = max(in_context_pivot_floor, thresh * 0.65)
    return bool(
        s_cat_standalone < revoke_ceil
        or s_cat_in_context < in_context_pivot_floor
        or (s_cat_standalone < st_pivot_ceil and eff_st_ref_z >= st_ref_z_pivot)
        or (
            min(s_cat_last, st_cat_last) < s_cat_last_pivot_floor
            and max(eff_ref_z, eff_st_ref_z) >= st_ref_z_pivot
            and s_cat_standalone < thresh
        )
    )


def maintenance_holds(max_ref_z: float, ref_maint_z: float) -> bool:
    """True when refusal pressure is at or above the maintenance floor.

    In an already-unlocked session, a follow-up that the model is building up to refuse keeps the
    session unlocked. `max_ref_z` is the larger of the in-context and standalone refusal signals.
    """
    return bool(max_ref_z >= ref_maint_z)


def hysteresis_holds(
    s_cat_in_context: float,
    s_cat_conv: float,
    hysteresis_thresh: float,
    thresh: float,
) -> bool:
    """True when category score alone carries an unlocked session forward.

    The fallback for when no live refusal telemetry is attached -- offline dry runs, and any serving
    path where the refusal monitor hook did not attach.
    """
    return bool(s_cat_in_context >= hysteresis_thresh or s_cat_conv >= thresh)
