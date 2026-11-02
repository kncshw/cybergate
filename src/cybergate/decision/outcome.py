"""`GateDecision`, the `Branch` enum, and the one place a `source=` string is built.

The rule from REFACTOR_PLAN.md §3.1: **the `source=` string is rendered from `branch` + `evidence`,
never hand-assembled at the branch site.** That structurally removes three found defects -- the
mislabelled `single_turn_...` source on a multi-turn request, the `raw_gate_fired` / source-string
contradiction, and a payload `s_ref_z` disagreeing with the `s_ref_z` printed inside `source=`.

One documented exception: two branches pass through the observation's own `source=` string instead of
rendering one, because that is what the canonical code returns. They carry it in
`GateDecision.inherited_source` and `render_source` returns it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .inputs import GateInputs

__all__ = ["Branch", "GateDecision", "render_payload", "render_source"]


class Branch(str, Enum):
    """The branch that produced a decision. One member per return path in the canonical policy."""

    #: Single turn (or stickiness disabled on a 1-turn request): the category score cleared the
    #: threshold but refusal pressure was below the maintenance floor, so the gate holds.
    SINGLE_TURN_TWO_FACTOR_SUPPRESSED = "single_turn_two_factor_suppressed"
    #: Single turn: decided purely by `score >= threshold`. Passes the observation's source through.
    SINGLE_TURN_RAW = "single_turn_raw"
    #: Multi-turn request with stickiness disabled, two-factor suppressed on the standalone score.
    STATELESS_TWO_FACTOR_SUPPRESSED = "stateless_two_factor_suppressed"
    #: Multi-turn request with stickiness disabled, decided on the standalone score.
    STATELESS_EVAL = "stateless_eval"
    #: No earlier turn unlocked the session, so the latest turn is judged on its standalone merits.
    #: Passes the standalone observation's source through.
    NO_PRIOR_UNLOCK = "no_prior_unlock"
    #: Step 2A: pivot to a non-target harm domain inside an unlocked session. Revokes.
    NONTARGET_HARM_REVOKED = "nontarget_harm_revoked"
    #: Step 2B: refusal pressure at or above the maintenance floor keeps the session unlocked.
    MAINTAINED = "maintained"
    #: Step 2B: refusal pressure below the maintenance floor, so the turn passes through ungated.
    NO_REFUSAL_PASSTHROUGH = "no_refusal_passthrough"
    #: Step 2C: no live refusal telemetry; the in-context or conversation score carried the session.
    HYSTERESIS_CONTINUED = "hysteresis_continued"
    #: Step 2C fell through: neither score held, so the session is revoked.
    NEURAL_REVOKED = "neural_revoked"
    #: No fresh observation was available. Defined for the Phase 6 fix of R14; the behaviour-preserving
    #: policy never emits it, because the canonical code lets an undecided record fall through with
    #: `score = 0.0` instead.
    NO_OBSERVATION = "no_observation"


@dataclass(frozen=True)
class GateDecision:
    """The outcome, plus the identity of the branch that produced it."""

    fired: bool
    branch: Branch
    score: float
    sticky_unlock: bool = False
    sticky_revoked: bool = False
    #: 1-based index of the turn whose unlock is being carried, or `None` when no unlock is in play.
    #: `None` also means the canonical payload omits the `sticky_from_turn` key entirely.
    sticky_from_turn: int | None = None
    #: The unlocking turn's score, or `None` when no unlock is in play (key omitted from the payload).
    prior_unlock_score: float | None = None
    #: Exactly the numbers the branch used, and the only values `render_source` may read.
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Set only by `SINGLE_TURN_RAW` and `NO_PRIOR_UNLOCK`; see the module docstring.
    inherited_source: str | None = None

    @property
    def source(self) -> str:
        """The rendered `source=` string for this decision."""
        return render_source(self)


def _fmt_z(value: Any) -> str:
    """Render a refusal-z for the revocation source string.

    The canonical code interpolates these two values with no format spec (`{st_ref_z}`), so a float
    renders as `str(float)` and a missing value as the literal `n/a`.
    """
    return "n/a" if value is None else f"{value}"


def render_source(decision: GateDecision) -> str:
    """Build the `source=` string from `decision.branch` and `decision.evidence`.

    Raises `KeyError` if the branch's evidence is missing a field its format needs, which is the
    point: a branch cannot report a number it did not record.
    """
    e = decision.evidence
    b = decision.branch

    if b in (Branch.SINGLE_TURN_RAW, Branch.NO_PRIOR_UNLOCK):
        return decision.inherited_source or ""

    if b is Branch.SINGLE_TURN_TWO_FACTOR_SUPPRESSED:
        return (
            "single_turn_L21_no_refusal_passthrough"
            f"(s_cat={e['s_cat']:+.4f}>={e['threshold']:+.4f},"
            f"s_ref_z={e['s_ref_z']:+.2f}sigma<{e['ref_maint_z']:+.2f}sigma)"
        )
    if b is Branch.STATELESS_TWO_FACTOR_SUPPRESSED:
        return (
            "stateless_single_turn_L21_no_refusal_passthrough"
            f"(s_cat_standalone={e['s_cat_standalone']:+.4f}>={e['threshold']:+.4f},"
            f"s_ref_z={e['s_ref_z']:+.2f}sigma<{e['ref_maint_z']:+.2f}sigma)"
        )
    if b is Branch.STATELESS_EVAL:
        return (
            "stateless_single_turn_eval"
            f"(s_cat_standalone={e['s_cat_standalone']:+.4f},thresh={e['threshold']:+.4f})"
        )
    if b is Branch.NONTARGET_HARM_REVOKED:
        return (
            f"multi_turn_L{e['layer']}_nontarget_harm_revoked"
            f"(turn={e['turn']},s_cat_standalone={e['s_cat_standalone']:+.4f},"
            f"s_cat_in_ctx={e['s_cat_in_context']:+.4f},"
            f"s_cat_last={e['s_cat_last_min']:+.4f},"
            f"st_ref_z={_fmt_z(e['st_ref_z'])},s_ref_z={_fmt_z(e['s_ref_z'])})"
        )
    if b is Branch.MAINTAINED:
        return (
            "multi_turn_L21_refusal_monitored_maintained"
            f"(from_turn={e['from_turn']},turn={e['turn']},"
            f"s_ref_z={e['s_ref_z']:+.2f}sigma>={e['ref_maint_z']:+.2f}sigma,"
            f"s_cat_in_ctx={e['s_cat_in_context']:+.4f},s_cat_conv={e['s_cat_conv']:+.4f},"
            f"s_cat_standalone={e['s_cat_standalone']:+.4f})"
        )
    if b is Branch.NO_REFUSAL_PASSTHROUGH:
        return (
            "multi_turn_L21_no_refusal_passthrough"
            f"(turn={e['turn']},s_ref_z={e['s_ref_z']:+.2f}sigma<{e['ref_maint_z']:+.2f}sigma,"
            f"s_cat_in_ctx={e['s_cat_in_context']:+.4f},"
            f"s_cat_standalone={e['s_cat_standalone']:+.4f})"
        )
    if b is Branch.HYSTERESIS_CONTINUED:
        return (
            f"multi_turn_L{e['layer']}_hysteresis_continued"
            f"(from_turn={e['from_turn']},"
            f"s_cat={e['s_cat_in_context']:+.4f}>={e['hysteresis_thresh']:+.4f})"
        )
    if b is Branch.NEURAL_REVOKED:
        return (
            f"multi_turn_L{e['layer']}_neural_revoked"
            f"(turn={e['turn']},s_cat={e['s_cat_in_context']:+.4f}<{e['hysteresis_thresh']:+.4f})"
        )
    if b is Branch.NO_OBSERVATION:
        return f"no_observation(turn={e.get('turn')})"
    raise AssertionError(f"Unhandled branch: {b!r}")


def render_payload(inputs: "GateInputs", decision: GateDecision) -> dict[str, Any]:
    """Merge `decision` onto the observation record to produce the legacy `x_cybergate` payload.

    This is the adapter between the typed decision and the dict shape the middleware, the client and
    the example scripts all consume. The canonical policy mutates the observation record in place and
    returns it; this builds the same dict without the policy ever touching it.

    Key order follows the canonical mutation order so a byte-level diff of two payloads stays
    readable.
    """
    out = dict(inputs.raw_record)
    latest = inputs.latest
    op = inputs.op
    standalone = inputs.standalone

    s_cat_in_context = latest.score
    s_cat_standalone = standalone.score if standalone is not None else s_cat_in_context

    # Fields the canonical code attaches before branching (`gate_probe.py:1344-1355`).
    out["s_ref_cos"] = latest.s_ref_cos if latest.s_ref_cos is not None else (
        standalone.s_ref_cos if standalone is not None else None
    )
    out["s_ref_z"] = latest.s_ref_z if latest.s_ref_z is not None else (
        standalone.s_ref_z if standalone is not None else None
    )
    out["user_turn_index"] = inputs.n_turns
    out["sticky_unlock"] = decision.sticky_unlock
    out["sticky_revoked"] = decision.sticky_revoked
    out["s_cat_standalone"] = round(s_cat_standalone, 6)
    out["s_cat_in_context"] = round(s_cat_in_context, 6)
    out["s_cat_conv"] = round(latest.s_cat_conv, 6)
    out["non_target_revoke_ceiling"] = op.revoke_ceil
    out["ref_maint_z_threshold"] = op.ref_maint_z
    out["hysteresis_threshold"] = op.hysteresis_thresh

    if latest.raw_gate_fired is not None:
        out["raw_gate_fired"] = latest.raw_gate_fired
    if latest.raw_l8_fired is not None:
        out["raw_l8_fired"] = latest.raw_l8_fired
    if decision.prior_unlock_score is not None:
        out["prior_unlock_score"] = round(decision.prior_unlock_score, 6)
    if decision.sticky_from_turn is not None:
        out["sticky_from_turn"] = decision.sticky_from_turn

    out["fired"] = decision.fired
    out["score"] = decision.score
    out["source"] = decision.source
    out["branch"] = decision.branch.value
    return out
