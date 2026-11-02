"""The equivalence corpus: observation records, not prompts.

A decision is a function of the numbers the probe produced, so a case specifies those numbers
directly. That buys three things over driving conversations through a live server:

  * every branch and both sides of every threshold boundary are reachable, rather than whichever
    branches 40 conversations happen to hit,
  * the corpus is deterministic and runs in milliseconds with no GPU,
  * no prompt content is involved, so the corpus is just a table of floats.

Turn texts are synthetic placeholders made unique per case by construction
(`_turn_text`), because the in-context cache key is `ctx_<sha(final turn)>` with no conversation
component -- two cases sharing a final turn text would share a decision and the corpus would be
poisoned. That is the R11 defect being worked around inside its own test corpus, and it is exactly
the mistake REFACTOR_PLAN.md §4 warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["CASES", "Case", "Obs", "conversation_messages", "records_for"]

#: The live Gemma-4 operating point, for reference in the case table below.
TAU_GEMMA = 0.10807703530099158


@dataclass(frozen=True)
class Obs:
    """One probe observation, as the worker hook would have written it."""

    score: float
    s_ref_z: float | None = None
    s_cat_last: float | None = None
    s_cat_conv: float | None = None
    s_cat_first: float | None = None
    s_ref_cos: float | None = None
    decided: bool = True
    #: Set `False` to omit this observation's record entirely, so the resolver has to fall through.
    present: bool = True

    def to_record(self, *, threshold: float, layer: int, source: str) -> dict[str, Any]:
        """Render the record the way the worker hook writes it, including its own `source=`."""
        score = self.score
        rec: dict[str, Any] = {
            "decided": self.decided,
            "fired": bool(score >= threshold),
            "score": round(score, 6),
            "threshold": threshold,
            "layer": layer,
            "pool": "mean_user",
            "s_cat_mean_user": round(score, 6),
            "s_cat_conv": round(self.s_cat_conv if self.s_cat_conv is not None else score, 6),
            "s_cat_first": round(self.s_cat_first if self.s_cat_first is not None else score, 6),
            "s_cat_last": round(self.s_cat_last if self.s_cat_last is not None else score, 6),
            "source": source,
        }
        if self.s_ref_z is not None:
            rec["s_ref_z"] = self.s_ref_z
            rec["s_ref_cos"] = (
                self.s_ref_cos if self.s_ref_cos is not None else round(0.1374208 + self.s_ref_z * 0.0513015, 6)
            )
        return rec


@dataclass(frozen=True)
class Case:
    """One conversation's worth of observations."""

    id: str
    #: Standalone observation per user turn, oldest first. The last entry is the latest turn.
    turns: tuple[Obs, ...]
    #: The latest turn observed *inside* the conversation. `None` for a single-turn case, where the
    #: standalone and in-context views are the same observation.
    in_context: Obs | None = None
    enable_multi_turn_stickiness: bool = True
    #: What this case is for. Shown on failure.
    note: str = ""
    #: Regression ids from REFACTOR_PLAN.md §5 that this case covers.
    covers: tuple[str, ...] = field(default=())

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _turn_text(case_id: str, index: int) -> str:
    """A unique placeholder for turn `index` of `case_id`. Carries no meaning; only its hash is used."""
    return f"equivalence case {case_id} turn {index}"


def conversation_messages(case: Case) -> list[dict[str, str]]:
    """Build the OpenAI-format message list for `case`, with a filler assistant turn between users."""
    messages: list[dict[str, str]] = []
    for i in range(case.n_turns):
        if i:
            messages.append({"role": "assistant", "content": f"filler reply {i}"})
        messages.append({"role": "user", "content": _turn_text(case.id, i)})
    return messages


def records_for(
    case: Case,
    *,
    threshold: float,
    layer: int,
) -> dict[str, dict[str, Any]]:
    """Return `{ipc_key: record}` for every observation `case` declares.

    Keys are built the way the worker hook builds them, so the resolver sees exactly the shapes it
    sees in production: `st_<sha>` and bare `<sha>` for a standalone turn, `ctx_<sha(final)>` and
    `<conv_sha>` for a multi-turn in-context prefill.
    """
    from cybergate.state import keys as key_builders

    out: dict[str, dict[str, Any]] = {}
    texts = [_turn_text(case.id, i) for i in range(case.n_turns)]

    for i, (obs, text) in enumerate(zip(case.turns, texts)):
        if not obs.present:
            continue
        record = obs.to_record(
            threshold=threshold,
            layer=layer,
            source=(
                f"live_vllm_gpu_L{layer}_mean_user_hook"
                f"(tokens={24 + i},turns=1,span=4:{20 + i},gpu_dot_us=66.0)"
            ),
        )
        record["num_user_turns"] = 1
        record["is_multiturn_ctx"] = False
        record["n_prefill_tokens"] = 24 + i
        out[key_builders.standalone_turn_key(text)] = record
        out[key_builders.bare_turn_key(text)] = record

    if case.in_context is not None and case.in_context.present:
        ctx = case.in_context.to_record(
            threshold=threshold,
            layer=layer,
            source=(
                f"live_vllm_gpu_L{layer}_mean_user_hook"
                f"(tokens={120 + case.n_turns},turns={case.n_turns},span=60:110,gpu_dot_us=68.0)"
            ),
        )
        ctx["num_user_turns"] = case.n_turns
        ctx["is_multiturn_ctx"] = True
        ctx["n_prefill_tokens"] = 120 + case.n_turns
        # Both keys, deliberately. The vendored canonical code reads `ctx_<sha(final turn)>`
        # first; this package reads the conversation key only (R11 fixed). Writing both with
        # the same record keeps the differential comparing decisions rather than key choice.
        out[key_builders.legacy_in_context_key_from_sha(key_builders.text_sha256(texts[-1]))] = ctx
        out[key_builders.conversation_key(texts)] = ctx

    return out


# =================================================================================================
# The corpus.
#
# Numbers marked "live" are real measurements from the audit, quoted in REFACTOR_PLAN.md §5.
# Numbers marked "boundary" are chosen to sit either side of a threshold.
# =================================================================================================
_FIRE = 0.1961  # live: positive control, well above tau
_PASS = 0.0142  # live: below tau and below every floor

CASES: tuple[Case, ...] = (
    # --- single turn --------------------------------------------------------------------------
    Case(
        id="st-fire",
        turns=(Obs(score=_FIRE, s_ref_z=10.50),),
        note="single turn, above tau with refusal pressure present: fires",
        covers=("R3",),
    ),
    Case(
        id="st-two-factor-suppressed",
        turns=(Obs(score=0.1160, s_ref_z=-0.27),),
        note="live R1: above tau, refusal pressure below the maintenance floor, so the gate holds",
        covers=("R1",),
    ),
    Case(
        id="st-two-factor-suppressed-b",
        turns=(Obs(score=0.1326, s_ref_z=0.42),),
        note="live R2 replicate of R1",
        covers=("R2",),
    ),
    Case(
        id="st-below-tau",
        turns=(Obs(score=_PASS, s_ref_z=-0.55),),
        note="single turn, below tau: passes through",
    ),
    Case(
        id="st-no-refusal-signal",
        turns=(Obs(score=_FIRE, s_ref_z=None),),
        note="above tau with no refusal observation: the second factor is skipped, not failed closed",
    ),
    Case(
        id="st-boundary-at-tau",
        turns=(Obs(score=TAU_GEMMA, s_ref_z=2.0),),
        note="boundary: score exactly at tau, comparison is >=",
    ),
    Case(
        id="st-boundary-just-under-tau",
        turns=(Obs(score=0.10807703, s_ref_z=2.0),),
        note="boundary: one ulp-ish below tau",
    ),
    Case(
        id="st-boundary-at-maint-floor",
        turns=(Obs(score=_FIRE, s_ref_z=1.50),),
        note="boundary: refusal pressure exactly at the maintenance floor, comparison is <",
    ),
    Case(
        id="st-abstain",
        turns=(Obs(score=0.0, present=False),),
        note="R14: no observation at all; the resolver abstains and the policy proceeds on score 0.0",
        covers=("R14",),
    ),
    # --- multi-turn, unlock maintained -----------------------------------------------------------
    Case(
        id="mt-maintained",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0726, s_ref_z=6.69)),
        in_context=Obs(score=0.1402, s_ref_z=6.69, s_cat_last=0.0820, s_cat_conv=0.1510),
        note="live R5: turn 1 unlocks, turn 2 maintains on refusal pressure",
        covers=("R5",),
    ),
    Case(
        id="mt-maintained-boundary",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0800, s_ref_z=1.50)),
        in_context=Obs(score=0.0900, s_ref_z=1.50, s_cat_last=0.0600, s_cat_conv=0.1200),
        note="boundary: maintenance at exactly the floor, comparison is >=",
    ),
    Case(
        id="mt-no-refusal-passthrough",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0900, s_ref_z=0.10)),
        in_context=Obs(score=0.0950, s_ref_z=0.10, s_cat_last=0.0700, s_cat_conv=0.1300),
        note="unlocked session, benign follow-up with no refusal buildup: passes through ungated",
    ),
    # --- multi-turn, revocation ------------------------------------------------------------------
    Case(
        id="mt-revoke-standalone-below-ceiling",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0498, s_ref_z=9.74)),
        in_context=Obs(score=0.1100, s_ref_z=9.74, s_cat_last=0.0241, s_cat_conv=0.1400),
        note="live: pivot whose standalone score falls below the revocation ceiling",
    ),
    Case(
        id="mt-revoke-in-context-below-floor",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0900, s_ref_z=3.0)),
        in_context=Obs(score=0.0700, s_ref_z=3.0, s_cat_last=0.0600, s_cat_conv=0.1000),
        note="disjunct 2: in-context score below the in-context floor",
    ),
    Case(
        id="mt-revoke-standalone-ref-pivot",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0700, s_ref_z=7.0)),
        in_context=Obs(score=0.1000, s_ref_z=2.0, s_cat_last=0.0800, s_cat_conv=0.1200),
        note="disjunct 3: below pivot floor with standalone refusal at the pivot level",
    ),
    Case(
        id="mt-revoke-last-token-pivot",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0900, s_ref_z=1.0, s_cat_last=0.0100)),
        in_context=Obs(score=0.1000, s_ref_z=7.5, s_cat_last=0.0300, s_cat_conv=0.1200),
        note="disjunct 4: last-token score low on both views with one refusal signal at the pivot",
    ),
    Case(
        id="mt-pleasantry-mislabelled",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0251, s_ref_z=0.30)),
        in_context=Obs(score=0.0900, s_ref_z=0.30, s_cat_last=0.0400, s_cat_conv=0.1100),
        note="R10: a benign low-scoring turn is labelled a non-target-harm pivot",
        covers=("R10",),
    ),
    # --- multi-turn, no prior unlock -------------------------------------------------------------
    Case(
        id="mt-no-prior-unlock",
        turns=(Obs(score=0.0300, s_ref_z=0.20), Obs(score=0.1500, s_ref_z=5.0)),
        in_context=Obs(score=0.1600, s_ref_z=5.0, s_cat_last=0.1200, s_cat_conv=0.1300),
        note="turn 1 never unlocked, so turn 2 is judged on its own standalone merits and fires",
    ),
    Case(
        id="mt-no-prior-unlock-suppressed",
        turns=(Obs(score=0.0600, s_ref_z=0.20), Obs(score=0.1500, s_ref_z=0.40)),
        in_context=Obs(score=0.1600, s_ref_z=0.40, s_cat_last=0.1200, s_cat_conv=0.1300),
        note="no prior unlock and the standalone two-factor suppresses: the nested source is copied",
    ),
    Case(
        id="mt-no-self-authorization",
        turns=(Obs(score=0.0600, s_ref_z=0.20), Obs(score=0.1500, s_ref_z=5.0)),
        in_context=Obs(
            score=0.1600, s_ref_z=5.0, s_cat_last=0.1200, s_cat_conv=0.1300, s_cat_first=0.1900
        ),
        note="R4: a high s_cat_first must not synthesize a prior unlock the turns do not support",
        covers=("R4",),
    ),
    # --- the R6/R7 live regression ---------------------------------------------------------------
    Case(
        id="mt-prior-turn-wipe-r6",
        turns=(
            Obs(score=_FIRE, s_ref_z=8.0),
            Obs(score=0.0251, s_ref_z=0.30),
            Obs(score=0.0580, s_ref_z=6.12),
        ),
        in_context=Obs(score=0.1200, s_ref_z=6.12, s_cat_last=0.0900, s_cat_conv=0.1400),
        note="live R6: one low interjection discards the turn-1 unlock, so turn 3 is denied at 4x the floor",
        covers=("R6",),
    ),
    Case(
        id="mt-prior-turn-wipe-r7",
        turns=(
            Obs(score=_FIRE, s_ref_z=8.0),
            Obs(score=0.0079, s_ref_z=0.30),
            Obs(score=0.0654, s_ref_z=6.33),
        ),
        in_context=Obs(score=0.1250, s_ref_z=6.33, s_cat_last=0.0950, s_cat_conv=0.1450),
        note="live R7: replicate of R6",
        covers=("R7",),
    ),
    Case(
        id="mt-wipe-control-r8",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0726, s_ref_z=6.69)),
        in_context=Obs(score=0.1300, s_ref_z=6.69, s_cat_last=0.0900, s_cat_conv=0.1500),
        note="live R8: same shape as R6 without the interleaved turn, isolating it as the cause",
        covers=("R8",),
    ),
    Case(
        id="mt-recovery-after-wipe-r9",
        turns=(
            Obs(score=_FIRE, s_ref_z=8.0),
            Obs(score=0.0251, s_ref_z=0.30),
            Obs(score=0.0580, s_ref_z=6.12),
            Obs(score=0.1798, s_ref_z=7.20),
        ),
        in_context=Obs(score=0.1850, s_ref_z=7.20, s_cat_last=0.1500, s_cat_conv=0.1700),
        note="live R9: the turn after a wipe re-unlocks on its own standalone score",
        covers=("R9",),
    ),
    # --- fallback hysteresis (no refusal telemetry anywhere) --------------------------------------
    Case(
        id="mt-hysteresis-continued",
        turns=(Obs(score=_FIRE, s_ref_z=None), Obs(score=0.0900, s_ref_z=None)),
        in_context=Obs(score=0.0950, s_ref_z=None, s_cat_last=0.0800, s_cat_conv=0.1200),
        note="no refusal signal at all: the category score carries the session (Step 2C)",
    ),
    # --- stickiness disabled ---------------------------------------------------------------------
    Case(
        id="mt-stateless-fires",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.1500, s_ref_z=5.0)),
        in_context=Obs(score=0.1600, s_ref_z=5.0, s_cat_last=0.1200, s_cat_conv=0.1300),
        enable_multi_turn_stickiness=False,
        note="stickiness off: the latest turn's standalone score decides, and it clears tau",
    ),
    Case(
        id="mt-stateless-suppressed",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.1500, s_ref_z=0.40)),
        in_context=Obs(score=0.1600, s_ref_z=0.40, s_cat_last=0.1200, s_cat_conv=0.1300),
        enable_multi_turn_stickiness=False,
        note="stickiness off and the standalone two-factor suppresses",
    ),
    Case(
        id="mt-stateless-below-tau",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0400, s_ref_z=5.0)),
        in_context=Obs(score=0.1600, s_ref_z=5.0, s_cat_last=0.1200, s_cat_conv=0.1300),
        enable_multi_turn_stickiness=False,
        note="stickiness off: an unlock from turn 1 must not carry",
    ),
    # --- missing observations --------------------------------------------------------------------
    Case(
        id="mt-no-standalone-record",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0900, s_ref_z=4.0, present=False)),
        in_context=Obs(score=0.1100, s_ref_z=4.0, s_cat_last=0.0800, s_cat_conv=0.1300),
        note="the latest turn has no standalone record, so the in-context view stands in for it",
    ),
    Case(
        id="mt-no-in-context-record",
        turns=(Obs(score=_FIRE, s_ref_z=8.0), Obs(score=0.0900, s_ref_z=4.0)),
        in_context=None,
        note="no in-context record: the resolver falls back to the latest turn's standalone record",
    ),
    Case(
        id="mt-prior-turn-missing",
        turns=(Obs(score=_FIRE, s_ref_z=8.0, present=False), Obs(score=0.0900, s_ref_z=4.0)),
        in_context=Obs(score=0.1100, s_ref_z=4.0, s_cat_last=0.0800, s_cat_conv=0.1300),
        note="the prior turn has no record, so it cannot have unlocked anything",
    ),
    # --- longer conversation ---------------------------------------------------------------------
    Case(
        id="mt-six-turn-workflow",
        turns=(
            Obs(score=_FIRE, s_ref_z=8.0),
            Obs(score=0.0820, s_ref_z=2.77),
            Obs(score=0.0910, s_ref_z=3.40),
            Obs(score=0.0760, s_ref_z=4.10),
            Obs(score=0.0880, s_ref_z=6.87),
            Obs(score=0.0930, s_ref_z=5.20),
        ),
        in_context=Obs(score=0.1320, s_ref_z=5.20, s_cat_last=0.0990, s_cat_conv=0.1480),
        note="six-turn engineering workflow, every follow-up above the revocation ceiling",
    ),
    Case(
        id="mt-twelve-turn-workflow",
        turns=(
            (Obs(score=_FIRE, s_ref_z=8.0),)
            + tuple(Obs(score=0.0800 + i * 0.0010, s_ref_z=2.0 + i * 0.3) for i in range(11))
        ),
        in_context=Obs(score=0.1280, s_ref_z=5.3, s_cat_last=0.0960, s_cat_conv=0.1440),
        note="twelve-turn workflow: the prior-unlock scan walks every earlier turn",
    ),
)

#: Cases rerun against the Qwen-3.6 operating point. The whole table would be redundant; these six
#: exercise the family-parameterised thresholds (revoke ceiling 0.0380, hysteresis floor 0.0500)
#: against Qwen's tau of 0.07104209.
QWEN_CASE_IDS = (
    "st-fire",
    "st-two-factor-suppressed",
    "mt-maintained",
    "mt-revoke-standalone-below-ceiling",
    "mt-hysteresis-continued",
    "mt-prior-turn-wipe-r6",
)


def qwen_cases() -> tuple[Case, ...]:
    """The Qwen subset, with ids suffixed so their turn texts stay unique across the run."""
    by_id = {c.id: c for c in CASES}
    out: list[Case] = []
    for cid in QWEN_CASE_IDS:
        c = replace(by_id[cid], id=f"{cid}-qwen")
        if cid == "mt-prior-turn-wipe-r6":
            # On live Qwen-3.6 L15, the R6 pleasantry scores -0.0540 (< revoke_ceil = -0.0250).
            c = replace(
                c,
                turns=(
                    c.turns[0],
                    Obs(score=-0.0540, s_ref_z=0.30),
                    c.turns[2],
                ),
            )
        elif cid == "mt-revoke-standalone-below-ceiling":
            c = replace(
                c,
                turns=(
                    c.turns[0],
                    Obs(score=-0.0300, s_ref_z=9.74),
                ),
            )
        out.append(c)
    return tuple(out)
