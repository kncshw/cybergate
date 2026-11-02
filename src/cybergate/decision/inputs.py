"""`GateInputs` -- everything the policy is allowed to read, and nothing else.

If a number is not reachable from a `GateInputs` instance, `decide()` cannot use it. That is the
whole mechanism by which the policy becomes testable without a GPU: these dataclasses are cheap to
construct by hand, so every branch and every threshold boundary becomes a table row.

All frozen, all stdlib. No cache reads, no HTTP, no clock, no environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import (
    DEFAULT_REF_MAINT_Z_THRESH,
    DEFAULT_ST_REF_Z_PIVOT,
    IN_CONTEXT_PIVOT_FLOOR,
    S_CAT_LAST_PIVOT_FLOOR,
)

__all__ = ["GateInputs", "InContextEvidence", "OperatingPoint", "PriorTurnEvidence", "StandaloneEvidence"]


@dataclass(frozen=True)
class OperatingPoint:
    """The thresholds a decision is made against.

    `thresh`, `gate_layer`, `revoke_ceil` and `hysteresis_thresh` come from the operator file and the
    model family. The last three are literals in the canonical policy rather than family-parameterised
    values, and are defaulted here from `config` so the policy reads them from its inputs.
    """

    thresh: float
    gate_layer: int
    revoke_ceil: float
    hysteresis_thresh: float
    ref_maint_z: float = DEFAULT_REF_MAINT_Z_THRESH
    in_context_pivot_floor: float = IN_CONTEXT_PIVOT_FLOOR
    s_cat_last_pivot_floor: float = S_CAT_LAST_PIVOT_FLOOR
    st_ref_z_pivot: float = DEFAULT_ST_REF_Z_PIVOT


@dataclass(frozen=True)
class InContextEvidence:
    """The latest user turn as observed **inside** its conversation.

    `decided` is the typed "was there a fresh observation" state. When it is False the canonical
    policy still proceeds with `score = 0.0`, which is the R14 defect; the field is here so a Phase 6
    fix has something to branch on.

    `threshold` and `layer` come from the observation record rather than the operating point, because
    the record was written by whichever operator the worker had loaded. They can disagree with
    `OperatingPoint.thresh` -- the policy uses the record's value, as the canonical code does.
    """

    decided: bool
    score: float
    threshold: float
    layer: int
    s_cat_conv: float
    s_cat_first: float
    s_cat_last: float
    s_ref_z: float | None = None
    s_ref_cos: float | None = None
    #: The observation's own `source=` string. Two branches pass it through unchanged; see
    #: `outcome.Branch.SINGLE_TURN_RAW`.
    source: str = ""
    #: `score >= threshold` on the **in-context** score, attached by the evidence layer before any
    #: branch runs. The canonical code reports this field regardless of which score the chosen branch
    #: actually used, so two requests on the same branch can report different values -- that is the
    #: R12 defect, and it lives here rather than on `GateDecision` precisely because it is a property
    #: of the observation. `None` when the observation came from a path that does not set it (HTTP
    #: `/gate`, offline precomputed, or an abstain).
    raw_gate_fired: bool | None = None
    #: Pre-rename duplicate of `raw_gate_fired`, written only by the single-prompt evaluator.
    raw_l8_fired: bool | None = None


@dataclass(frozen=True)
class StandaloneEvidence:
    """The latest user turn scored **alone**, with no conversation around it.

    Absent (`None` on `GateInputs.standalone`) for a single-turn request, where the in-context and
    standalone views are the same observation.
    """

    decided: bool
    score: float
    threshold: float
    s_cat_last: float
    s_ref_z: float | None = None
    s_ref_cos: float | None = None
    source: str = ""


@dataclass(frozen=True)
class PriorTurnEvidence:
    """One earlier user turn, already resolved to a fired/not-fired decision.

    `fired` is resolved by the evidence layer, not by the policy, because whether the two-factor rule
    applies depends on *where* the record came from: a live or cached observation gets the two-factor
    treatment, while an HTTP `/gate` or offline precomputed record carries its own `fired` verbatim.
    That provenance is an I/O fact. See `evidence.resolve_prior_turn`.
    """

    #: 1-based index of this turn among the conversation's user turns.
    index: int
    decided: bool
    fired: bool
    score: float
    s_ref_z: float | None = None


@dataclass(frozen=True)
class GateInputs:
    """The complete input to `decide()`."""

    op: OperatingPoint
    #: Total user turns in the conversation, including the latest. `1` means a single-turn request.
    n_turns: int
    latest: InContextEvidence
    standalone: StandaloneEvidence | None = None
    #: Earlier user turns in order, oldest first. Empty for a single-turn request.
    priors: tuple[PriorTurnEvidence, ...] = ()
    #: False reproduces `enable_multi_turn_stickiness=False`: evaluate the latest turn alone.
    enable_multi_turn_stickiness: bool = True
    #: Opaque to the policy. Carried so the payload renderer can merge decision fields onto the
    #: original observation record without the policy ever reading it.
    raw_record: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
