"""Tunables and counters for the measured CoT controller.

Separated from the controller so the thresholds -- and, importantly, their provenance -- can be read
without reading the control logic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import env_first

__all__ = [
    "DEFAULT_COOLDOWN_STEPS",
    "DEFAULT_HEARTBEAT_APPLIES",
    "DEFAULT_MAX_INJECTIONS",
    "DEFAULT_MIN_COT_STEPS",
    "DEFAULT_TAU_SPIKE",
    "DEFAULT_THOUGHT_BUDGET",
    "SteeringConfig",
    "SteeringStats",
]


#: Default trigger level, in **decode-domain** sigma at the refusal layer.
#:
#: The value is inherited from the never-executed V0 processor in the canonical package. What the
#: 2026-10-28 producer review established is that the number was not merely uncalibrated, it was
#: being compared in the wrong units: `rolling_s_ref_z` was formed with `mu_hm`/`sd_hm`, which are
#: fitted on **prefill-end turn-boundary** tokens, and then evaluated on **mid-sentence decode**
#: steps, where the layer-21 cosine has a different mean and spread entirely. Across 28 steered
#: requests the statistic peaked at +2.013 sigma and never crossed 3.55 -- and in baseline mode a
#: model *actively writing a refusal* only reached +1.34 to +1.93 sigma in those units. The trigger
#: could not fire at any operating point.
#:
#: So the comparison now uses `SteeringConfig.mu_decode`/`sd_decode`, fitted on decode steps. Until
#: those are supplied the controller does not steer and says so: a dead trigger that announces
#: itself is recoverable, a dead trigger that reports success is not.
DEFAULT_TAU_SPIKE = 3.55

#: Decode steps to wait before the first injection, so a spike in the opening tokens -- where the
#: rolling window is still filling -- cannot fire.
DEFAULT_MIN_COT_STEPS = 15

#: Decode steps after an injection during which another cannot fire.
DEFAULT_COOLDOWN_STEPS = 35

#: Hard cap on injections per generation, so a persistently high reading cannot loop forever.
DEFAULT_MAX_INJECTIONS = 2

#: Emit a stats line every N `apply()` calls while any row is tracked.
#:
#: Keyed on applies, not on forced tokens: the first version logged only from the injection path, so
#: the `unresolved` counter was silent during the very failure it diagnoses.
DEFAULT_HEARTBEAT_APPLIES = 200

#: Decode steps after which the controller forces the thought block closed.
#:
#: Without this the model spends the entire combined budget thinking and never emits the close tag,
#: so the response carries no visible answer at all. Measured live on 2026-10-28: 18 of 18 firing
#: requests returned an empty answer, with `gen_tokens` landing on exactly the combined budget every
#: time. vLLM ships its own `ThinkingTokenBudgetLogitsProcessor` for the same reason.
DEFAULT_THOUGHT_BUDGET = 180


@dataclass(frozen=True)
class SteeringConfig:
    """Tunables, resolved once at engine init."""

    tau_spike: float = DEFAULT_TAU_SPIKE
    min_cot_steps: int = DEFAULT_MIN_COT_STEPS
    cooldown_steps: int = DEFAULT_COOLDOWN_STEPS
    max_injections: int = DEFAULT_MAX_INJECTIONS
    thought_budget: int = DEFAULT_THOUGHT_BUDGET
    #: Decode-domain calibration for the trigger. `None` means uncalibrated, and the controller then
    #: refuses to steer rather than comparing an out-of-domain number against `tau_spike`.
    #: See `DEFAULT_TAU_SPIKE` and `families.base.ModelFamily.ref_mu_decode`.
    mu_decode: float | None = None
    sd_decode: float | None = None

    def with_family_defaults(self, family: object) -> "SteeringConfig":
        """Fill an unset decode calibration from the family's measured constants.

        Env wins, then the family, then nothing. Both the monitor hook (which forms the decode-domain
        z) and the controller (which reports whether the trigger is live) resolve through here, so
        the two can never disagree about whether steering is calibrated.
        """
        if self.mu_decode is not None and self.sd_decode is not None:
            return self
        return replace(
            self,
            mu_decode=(
                self.mu_decode
                if self.mu_decode is not None
                else getattr(family, "ref_mu_decode", None)
            ),
            sd_decode=(
                self.sd_decode
                if self.sd_decode is not None
                else getattr(family, "ref_sd_decode", None)
            ),
        )

    @property
    def decode_calibrated(self) -> bool:
        """True when `tau_spike` can be compared against a decode-domain z-score."""
        return (
            self.mu_decode is not None
            and self.sd_decode is not None
            and float(self.sd_decode) > 0.0
        )

    @classmethod
    def from_env(cls) -> SteeringConfig:
        def _f(name: str, default: float) -> float:
            raw = env_first(name)
            return float(raw) if raw else default

        def _i(name: str, default: int) -> int:
            raw = env_first(name)
            return int(raw) if raw else default

        def _opt(name: str, default: float | None) -> float | None:
            raw = env_first(name)
            return float(raw) if raw else default

        return cls(
            mu_decode=_opt("CYBERGATE_REF_MU_DECODE", None),
            sd_decode=_opt("CYBERGATE_REF_SD_DECODE", None),
            tau_spike=_f("CYBERGATE_TAU_SPIKE", DEFAULT_TAU_SPIKE),
            min_cot_steps=_i("CYBERGATE_MIN_COT_STEPS", DEFAULT_MIN_COT_STEPS),
            cooldown_steps=_i("CYBERGATE_COOLDOWN_STEPS", DEFAULT_COOLDOWN_STEPS),
            max_injections=_i("CYBERGATE_MAX_INJECTIONS", DEFAULT_MAX_INJECTIONS),
            thought_budget=_i("CYBERGATE_THOUGHT_BUDGET", DEFAULT_THOUGHT_BUDGET),
        )


@dataclass
class SteeringStats:
    """Counters for one engine, read by tests and reported in the log."""

    rows_tracked: int = 0
    rows_skipped: int = 0
    injections: int = 0
    forced_tokens: int = 0
    #: Passes where a tracked row had no reading for this pass. Non-zero means the join is not
    #: resolving -- the symptom the row-index design produced silently.
    unresolved: int = 0
    #: Passes where a reading arrived but its rolling value was not in the decode domain, so the
    #: trigger could not be evaluated. Non-zero means the decode calibration is missing.
    uncalibrated: int = 0
    #: Rows whose thought was force-closed on reaching `thought_budget`.
    thoughts_closed: int = 0
    #: Rows that closed their own thought before the budget.
    thoughts_self_closed: int = 0
    #: Rows adopted with the post-think opener suppressed, i.e. tool-bearing turns. See
    #: `RefusalSteeredCoTProcessor.update_state`.
    openers_suppressed: int = 0
    peak_rolling_z: float = float("-inf")

    def as_log(self) -> str:
        return (
            f"tracked={self.rows_tracked} skipped={self.rows_skipped} "
            f"injections={self.injections} forced_tokens={self.forced_tokens} "
            f"unresolved={self.unresolved} uncalibrated={self.uncalibrated} "
            f"closed={self.thoughts_closed} "
            f"self_closed={self.thoughts_self_closed} "
            f"openers_suppressed={self.openers_suppressed} "
            f"peak_rolling_z={self.peak_rolling_z:+.3f}"
        )
