"""The measured closed loop: a vLLM V1 `LogitsProcessor` that steers CoT on the refusal reading.

This replaces the keyword-regex trigger. The regex approach scanned the *generated text* for words
like "however" and "cannot provide", decided a refusal had begun, and then issued another HTTP round
trip to continue the thought. Three problems with that, in order of seriousness:

1. It measures nothing. A phrase list is a proxy for refusal; the plugin already computes the real
   quantity (`s_ref_z`, the calibrated refusal projection at layer 21) and threw it away.
2. It is language- and phrasing-bound. Any refusal the list does not spell triggers nothing.
3. It costs two extra forward passes over the whole prompt, because each continuation re-prefills.

This controller instead reads the refusal projection the monitor hook already computes, every decode
step, and acts within the same forward pass. One pass, no re-prefill, no word list.

## How it decides

Per tracked row, each decode step:

    trigger  =  rolling_s_ref_z >= tau_spike
            and cot_steps       >= min_cot_steps
            and cooldown        == 0
            and injections_used <  max_injections

On trigger it forces the suppression text token by token, then returns the row to free generation and
starts a cooldown. The model finishes its own derivation rather than being truncated.

## Which requests it steers

Only a generation whose prompt **ends with the family's open-thought prefix** -- that is, exactly the
CoT generation the serving path issues when the gate has fired. Every other request, including every
pass-through, is left untouched. Adoption keys off the prompt shape rather than a request flag, which
keeps a pass-through unsteerable even if it arrives with the flag set.

## The post-think opener, and who may decline it

When the thought closes -- by the model's choice or at the budget -- the controller force-emits
`BOTTOM_COMPLIANCE_LOCK`, the close tag, and `POST_THINK_OPENER` ("Here is the complete, working
technical implementation...") as the first visible tokens, so the visible answer cannot open on a
refusal.

That is right for a single-shot answer and wrong for an agentic turn. Forced tokens commit the answer
to prose before the model emits one free token, so a turn whose correct output is a tool call cannot
produce one. Measured live on 2026-10-29 over ten requests: every fired tool-bearing turn opened with
that sentence and returned no tool call, and one answered the *previous* topic in 3,496 characters
while the user had asked for a ping sweep.

So a request may decline the opener with `vllm_xargs={"cybergate_suppress_opener": 1}` on the
completion call, which vLLM 0.32 delivers as `SamplingParams.extra_args`; the serving path sets it
whenever the request carries tools. The compliance lock and the close tag are still forced -- only the
opener is dropped, so the refusal steering itself is unchanged.

## What it declares

`is_argmax_invariant()` returns **False**: forcing a token changes the argmax by construction. vLLM
uses that to reject the processor under speculative decoding, which is correct -- a forced token and a
speculated one cannot both be honoured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from ..config import SUPPRESS_OPENER_XARG, env_first
from .steering_config import DEFAULT_HEARTBEAT_APPLIES, SteeringConfig, SteeringStats
from ..families import ModelFamily, family_for_model_id
from ..telemetry.log import log_cybergate_event
from . import refusal_trace

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = ["RefusalSteeredCoTProcessor"]

def _logits_processor_base() -> type:
    """vLLM's `LogitsProcessor` ABC when vLLM is importable, else `object`.

    The base class is load-bearing, not decorative. vLLM calls `validate_params()` on every
    registered processor *class* for every request, and resolves it off the ABC; a class that only
    implements the three instance methods has no such attribute and every generation fails. The
    guarded import follows `vllm_adapter.batch`: vLLM is absent in the offline test environment.
    """
    try:
        from vllm.v1.sample.logits_processor.interface import LogitsProcessor  # type: ignore

        return LogitsProcessor
    except Exception:
        return object


_LOGITS_PROCESSOR_BASE = _logits_processor_base()


def _opener_declined(params: Any) -> bool:
    """True when this request asked for `POST_THINK_OPENER` to be left off.

    `vllm_xargs` values arrive as `str | int | float`, so `"0"` and `"false"` must not read as
    truthy: a flag that cannot be switched back off is worse than no flag at all.
    """
    extra = getattr(params, "extra_args", None) or {}
    value = extra.get(SUPPRESS_OPENER_XARG)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no")
    return bool(value)


@dataclass
class _RowState:
    """Per-batch-row steering state."""

    #: Reference to the request's running output token list, as supplied by `BatchUpdate`.
    output_tok_ids: Sequence[int]
    #: The identity key this request's refusal readings are published under. See `refusal_trace`.
    key: str = ""
    #: Cursor into the close-tag token ids while forcing the thought closed; `None` when not closing.
    close_cursor: int | None = None
    #: True once the thought block has ended, by the model's choice or by the budget. The controller
    #: stops touching the row: what follows is the visible answer, not the thought.
    done: bool = False
    #: False when this request declined `POST_THINK_OPENER`, i.e. a tool-bearing turn whose answer
    #: must be free to open on a tool call. The lock and close tag are forced either way.
    force_opener: bool = True
    cooldown: int = 0
    injections_used: int = 0
    #: Index into the suppression token ids while an injection is in progress; `None` when free.
    cursor: int | None = None
    peak_rolling_z: float = float("-inf")
    spike_steps: list[int] = field(default_factory=list)

    @property
    def cot_steps(self) -> int:
        """Tokens generated so far for this row."""
        return len(self.output_tok_ids)

    @property
    def injecting(self) -> bool:
        return self.cursor is not None


class RefusalSteeredCoTProcessor(_LOGITS_PROCESSOR_BASE):
    """Steers an open `<think>` block on the measured refusal projection.

    Constructed once per engine by vLLM with `(vllm_config, device, is_pin_memory)`.

    Subclasses vLLM's `LogitsProcessor` ABC. The V1 contract is not the three instance methods it
    appears to be: vLLM also calls the ABC's `validate_params()` classmethod on every registered
    processor class for every request, via `SamplingParams.verify` ->
    `validate_logits_processors_parameters`, and the `--logits-processors` FQCN path refuses a class
    that is not a subclass. Implementing only the three methods made every generation request --
    not merely the steered ones -- fail with an empty-message 500.
    """

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.config = SteeringConfig.from_env()
        self.stats = SteeringStats()
        self._rows: dict[int, _RowState] = {}
        self._apply_calls = 0

        model_id = _model_id_of(vllm_config)
        self.family: ModelFamily = family_for_model_id(model_id)
        self.config = self.config.with_family_defaults(self.family)
        self.skip_decode_monitor = (
            env_first(
                "CYBERGATE_SKIP_DECODE_MONITOR",
                default="1" if self.family.name == "gemma4_31b" else "0",
            )
            == "1"
        )
        self.prefix_caching = _prefix_caching_enabled(vllm_config)
        tokenizer = _load_tokenizer(vllm_config, model_id)

        if tokenizer is None:
            # Without a tokenizer there are no token ids to force or to match against, so the
            # processor disables itself rather than guessing. Loud, because a silently inert
            # controller is exactly the failure mode that hid the Qwen gate being dead.
            self.enabled = False
            self.suppression_ids: list[int] = []
            self.prompt_suffix_ids: list[int] = []
            self.self_close_tag_ids: list[int] = []
            self.bottom_lock_ids: list[int] = []
            self.post_close_opener_ids: list[int] = []
            self.close_tag_ids: list[int] = []
            self.close_tag_ids_no_opener: list[int] = []
            log_cybergate_event(
                "CYBERGATE-COT-CONTROLLER-WARN",
                "disabled: no tokenizer available, so CoT steering cannot encode its tokens",
            )
            return

        if env_first("CYBERGATE_COT_STEERING", default="1") == "0":
            # Explicitly disabled, so the regex comparison baseline can run uncontaminated. See
            # `serving.think_routes.route_for_mode`.
            self.enabled = False
            self.suppression_ids = []
            self.prompt_suffix_ids = []
            self.self_close_tag_ids = []
            self.bottom_lock_ids = []
            self.post_close_opener_ids = []
            self.close_tag_ids = []
            self.close_tag_ids_no_opener = []
            log_cybergate_event(
                "CYBERGATE-COT-CONTROLLER",
                "disabled by CYBERGATE_COT_STEERING=0; the regex baseline is selectable instead",
            )
            return

        self.enabled = True
        self.suppression_ids = list(
            tokenizer.encode(self.family.inline_cot_suppression, add_special_tokens=False)
        )
        self.self_close_tag_ids = list(
            tokenizer.encode(self.family.thought_close_tag, add_special_tokens=False)
        )
        self.bottom_lock_ids = list(
            tokenizer.encode(self.family.bottom_compliance_lock, add_special_tokens=False)
        )
        self.post_close_opener_ids = list(
            tokenizer.encode(self.family.post_think_opener, add_special_tokens=False)
        )
        self.close_tag_ids = (
            self.bottom_lock_ids + self.self_close_tag_ids + self.post_close_opener_ids
        )
        #: The same sequence without the opener, for a row that declined it.
        self.close_tag_ids_no_opener = self.bottom_lock_ids + self.self_close_tag_ids
        # The tail of the prompt that marks "this generation is an open thought block".
        self.prompt_suffix_ids = list(
            tokenizer.encode(
                f"{self.family.thought_open}{self.family.open_think_prefix}",
                add_special_tokens=False,
            )
        )
        # Claim management of the refusal_trace slot -> key table immediately so hook-side
        # `bind_request` stands down even when prefix caching presents truncated prefill slices.
        refusal_trace.sync_bindings({})

        log_cybergate_event(
            "CYBERGATE-COT-CONTROLLER",
            f"enabled family={self.family.name} prefix_caching={self.prefix_caching} "
            f"decode_calibrated={self.config.decode_calibrated} "
            f"tau_spike={self.config.tau_spike:+.2f}sigma "
            f"min_cot_steps={self.config.min_cot_steps} cooldown={self.config.cooldown_steps} "
            f"max_injections={self.config.max_injections} "
            f"thought_budget={self.config.thought_budget} "
            f"suppression_tokens={len(self.suppression_ids)} "
            f"close_tag_tokens={len(self.close_tag_ids)} "
            f"prompt_suffix_tokens={len(self.prompt_suffix_ids)}",
        )

    # ---------------------------------------------------------------------------------------------
    # vLLM V1 LogitsProcessor interface
    # ---------------------------------------------------------------------------------------------
    def is_argmax_invariant(self) -> bool:
        """False: forcing a token changes the argmax, by design."""
        return False

    def update_state(self, batch_update: Any | None) -> None:
        """Track rows entering, leaving and moving within the persistent batch.

        Order matters and is dictated by vLLM: removed, then added, then moved. Added and moved rows
        may land on an index a previous request occupied, so each operation must overwrite rather
        than merge -- otherwise one request inherits another's injection budget.
        """
        if batch_update is None or not self.enabled:
            return

        for row in batch_update.removed:
            self._rows.pop(row, None)

        for row, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            # An added row replaces whatever was there.
            self._rows.pop(row, None)
            if self._should_steer(prompt_tok_ids):
                assert prompt_tok_ids is not None  # _should_steer rejects None
                force_opener = not _opener_declined(params)
                self._rows[row] = _RowState(
                    output_tok_ids=output_tok_ids,
                    key=refusal_trace.key_for_tokens(prompt_tok_ids),
                    force_opener=force_opener,
                )
                self.stats.rows_tracked += 1
                if not force_opener:
                    self.stats.openers_suppressed += 1
            else:
                self.stats.rows_skipped += 1

        for src, dst, directionality in batch_update.moved:
            self._move(src, dst, directionality)

        # Hand the monitor hook the authoritative row -> key map.
        #
        # The hook can only name a slot by its row, and it learns a row's key on prefill. When a
        # request finishes mid-generation vLLM compacts the rows below it, so a decoding request
        # moves to a row whose binding belongs to someone else -- and it never prefills again, so
        # nothing corrects it. This loop is the only place that sees those events. Measured before
        # the fix: a short request leaving row 0 sent the next 165 decode steps of its successor to
        # the departed key, `unresolved=82`.
        refusal_trace.sync_bindings(
            {refusal_trace.slot_name(row): state.key for row, state in self._rows.items()}
        )

    def apply(self, logits: "torch.Tensor") -> "torch.Tensor":
        """Force suppression tokens on any tracked row whose refusal reading crossed the trigger."""
        if not self.enabled or not self._rows:
            return logits

        if not self.skip_decode_monitor:
            refusal_trace.sync_from_cudagraph(
                [row for row, st in sorted(self._rows.items()) if not st.done]
            )

        # Heartbeat. The stats line used to be emitted only from the injection path, so the
        # `unresolved` counter -- the one that diagnoses a failed join -- was invisible in exactly
        # the case it exists to report. Measured: during two concurrent trials the controller logged
        # nothing at all while `unresolved` climbed to 784.
        self._apply_calls += 1
        if self._apply_calls % DEFAULT_HEARTBEAT_APPLIES == 0:
            self._log_stats()

        for row, state in self._rows.items():
            if state.done:
                continue
            if row >= logits.shape[0]:
                # The batch shrank without a `removed` notification; skip rather than index past it.
                continue

            # Finish a close in progress before anything else: a half-written close tag would leave
            # the thought block unterminated.
            if state.close_cursor is not None:
                self._force_next_close_token(logits, row, state)
                continue

            # Finish an inline suppression injection in progress before checking self-close or
            # budget, so an unforced `</think>` argmax mid-injection cannot abort `state.injecting`.
            if state.injecting:
                self._force_next_suppression_token(logits, row, state)
                continue

            # Intercept a single-token self-close (e.g. `</think>` = 248048 on Qwen3.8) BEFORE it is
            # sampled so `bottom_compliance_lock` is injected inside `<think>` right before `</think>`.
            if self._would_emit_self_close(logits, row, state):
                state.close_cursor = 0
                self.stats.thoughts_self_closed += 1
                self._force_next_close_token(logits, row, state)
                continue

            # The model may have ended its own thought; force the post-think opener across the
            # visible boundary before stepping aside so the first visible token cannot flip to refusal.
            if self._thought_self_closed(state):
                self.stats.thoughts_self_closed += 1
                if state.force_opener and self.post_close_opener_ids:
                    state.close_cursor = len(self.bottom_lock_ids) + len(self.self_close_tag_ids)
                    self._force_next_close_token(logits, row, state)
                else:
                    # Nothing left to force: the model closed the thought itself and this row either
                    # declined the opener or there is none. Step aside and let the answer begin.
                    state.done = True
                continue

            # Budget reached: force the close tag so a visible answer can still be generated.
            # When `thought_budget <= 0`, `<think>` generation is uncapped until the model closes it.
            if self.config.thought_budget > 0 and state.cot_steps >= self.config.thought_budget:
                state.close_cursor = 0
                self.stats.thoughts_closed += 1
                self._force_next_close_token(logits, row, state)
                continue

            if state.cooldown > 0:
                state.cooldown -= 1
                continue
            if self._triggered(state):
                state.cursor = 0
                state.injections_used += 1
                state.spike_steps.append(state.cot_steps)
                self.stats.injections += 1
                self._force_next_suppression_token(logits, row, state)
        return logits

    # ---------------------------------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------------------------------
    def _should_steer(self, prompt_tok_ids: Sequence[int] | None) -> bool:
        """True when this request's prompt ends with the family's open-thought prefix."""
        if not prompt_tok_ids or not self.prompt_suffix_ids:
            return False
        suffix = self.prompt_suffix_ids
        if len(prompt_tok_ids) < len(suffix):
            return False
        return list(prompt_tok_ids[-len(suffix) :]) == suffix

    def _triggered(self, state: _RowState) -> bool:
        """Evaluate the measured trigger for one request, joining on its identity key."""
        if self.skip_decode_monitor:
            return False
        if state.cot_steps < self.config.min_cot_steps:
            return False
        if state.injections_used >= self.config.max_injections:
            return False
        reading = self._reading_for(state)
        if reading is None:
            # No reading for this pass. Either the hook did not run, or the join did not resolve --
            # which happens when a prefix-cached prefill showed the hook only part of the prompt.
            # Counted rather than ignored: the row-index design failed this way in silence.
            self.stats.unresolved += 1
            return False
        rolling = reading.rolling_s_ref_z
        state.peak_rolling_z = max(state.peak_rolling_z, rolling)
        self.stats.peak_rolling_z = max(self.stats.peak_rolling_z, rolling)
        if not reading.decode_domain:
            # The reading is in prefill units, so comparing it against `tau_spike` is meaningless:
            # it is the comparison that made this trigger unfireable at every operating point
            # (Finding 1). Refuse, and count it, so an uncalibrated deployment is visible in the
            # heartbeat instead of looking like a quiet generation.
            self.stats.uncalibrated += 1
            return False
        return rolling >= self.config.tau_spike

    def _reading_for(self, state: _RowState) -> refusal_trace.RowReading | None:
        """This request's reading for the current pass, joined on identity rather than position.

        Falls back to the sole-sequence key only when exactly one row is tracked, because that is the
        case where vLLM reports no request id and no `req_id -> key` binding can exist. With more
        than one tracked row the fallback would be ambiguous, which is precisely the mistake the
        row-index design made.
        """
        reading = refusal_trace.read(state.key)
        if reading is not None:
            return reading
        if len(self._rows) == 1:
            return refusal_trace.read(refusal_trace.SOLE_KEY)
        return None

    def _would_emit_self_close(
        self, logits: Any, row: int, state: _RowState
    ) -> bool:
        """True when a single-token close tag (e.g. `</think>`) is the greedy next token for `row`.

        Intercepting at logit time lets the controller force `bottom_compliance_lock` inside `<think>`
        followed by `</think>` (and `post_think_opener` when enabled), rather than reacting one step
        after `</think>` has already sealed the thought block.
        """
        if len(self.self_close_tag_ids) != 1 or state.cot_steps <= 0:
            return False
        close_tok = self.self_close_tag_ids[0]
        if hasattr(logits, "argmax"):
            return int(logits[row].argmax().item()) == close_tok
        if hasattr(logits, "rows"):
            row_vals = logits.rows[row]
            if 0 <= close_tok < len(row_vals):
                best = row_vals[close_tok]
                return all(best > v for i, v in enumerate(row_vals) if i != close_tok)
        return False

    def _thought_self_closed(self, state: _RowState) -> bool:
        """True when the model has already emitted the close tag for this row."""
        tag_ids = self.self_close_tag_ids or self.close_tag_ids
        k = len(tag_ids)
        if not k:
            return False
        valid = [t for t in state.output_tok_ids if t >= 0]
        if len(valid) < k:
            return False
        if k == 1:
            return tag_ids[0] in valid[-8:]
        window = valid[-max(k + 6, 8) :]
        return any(
            window[i : i + k] == tag_ids
            for i in range(len(window) - k + 1)
        )

    def _force_next_close_token(
        self,
        logits: "torch.Tensor",
        row: int,
        state: _RowState,
    ) -> None:
        """Pin `row`'s next token to the thought close tag, advancing the cursor.

        When the tag is complete the row is marked done and the controller stops touching it, so the
        model is free to generate the visible answer.
        """
        close_ids = self.close_tag_ids if state.force_opener else self.close_tag_ids_no_opener
        cursor = state.close_cursor or 0
        token_id = close_ids[cursor]
        logits[row] = float("-inf")
        logits[row, token_id] = 0.0
        self.stats.forced_tokens += 1

        cursor += 1
        if cursor >= len(close_ids):
            state.close_cursor = None
            state.done = True
        else:
            state.close_cursor = cursor

    def _force_next_suppression_token(
        self,
        logits: "torch.Tensor",
        row: int,
        state: _RowState,
    ) -> None:
        """Pin `row`'s next token to the suppression text, advancing the cursor."""
        cursor = state.cursor or 0
        token_id = self.suppression_ids[cursor]
        logits[row] = float("-inf")
        logits[row, token_id] = 0.0
        self.stats.forced_tokens += 1

        cursor += 1
        if cursor >= len(self.suppression_ids):
            state.cursor = None
            state.cooldown = self.config.cooldown_steps
        else:
            state.cursor = cursor

    def _log_stats(self) -> None:
        """Emit a stats line, so the controller's behaviour is visible in the log."""
        log_cybergate_event(
            "CYBERGATE-COT-CONTROLLER",
            f"{self.stats.as_log()} rows={sorted(self._rows)} "
            f"trace={refusal_trace.stats()}",
        )

    def _move(self, src: int, dst: int, directionality: Any) -> None:
        """Apply one row move, handling both the one-way and the swap form."""
        is_swap = getattr(directionality, "name", "") == "SWAP"
        src_state = self._rows.pop(src, None)
        dst_state = self._rows.pop(dst, None)
        if src_state is not None:
            self._rows[dst] = src_state
        if is_swap and dst_state is not None:
            self._rows[src] = dst_state


def _prefix_caching_enabled(vllm_config: Any) -> bool:
    """Whether this engine has prefix caching on.

    Recorded in the startup telemetry log. When enabled, `RefusalSteeredCoTProcessor` drives the
    `refusal_trace` slot-to-key bindings directly (`sync_bindings`) so hook-side prefill slices do
    not overwrite the full-prompt keys.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    return bool(getattr(cache_config, "enable_prefix_caching", False))


def _model_id_of(vllm_config: Any) -> str:
    """Best-effort model id from a vLLM config, for family resolution."""
    model_config = getattr(vllm_config, "model_config", None)
    for attr in ("model", "served_model_name", "tokenizer"):
        value = getattr(model_config, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _load_tokenizer(vllm_config: Any, model_id: str) -> Any | None:
    """Load the tokenizer this processor needs to encode its own tokens.

    Tries `CYBERGATE_TOKENIZER_PATH`, then the config's tokenizer path, then the model id. Returns
    `None` on failure, which disables the processor loudly rather than silently.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - transformers is present in the serving image
        log_cybergate_event("CYBERGATE-COT-CONTROLLER-WARN", f"transformers unavailable: {exc}")
        return None

    model_config = getattr(vllm_config, "model_config", None)
    candidates = [
        env_first("CYBERGATE_TOKENIZER_PATH"),
        getattr(model_config, "tokenizer", None),
        model_id or None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return AutoTokenizer.from_pretrained(candidate)
        except Exception:
            continue
    return None
