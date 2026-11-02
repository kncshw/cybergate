"""Single-pass in-engine `LogitsProcessor` state machines for:
  1. `ResGateSandwichLogitsProcessor` (`SW2` / `SW3` Open-`<think>` Sandwich), and
  2. `ResGateClosedLoopCoTProcessor` (`CL2` / `CL3` Closed-Loop Cosine-Monitored `<think>` Steering).
"""
from __future__ import annotations

from typing import Any, Sequence

from .gate_probe import get_request_gate_state
from .prompts import (
    BOTTOM_COMPLIANCE_LOCK,
    INLINE_COT_SUPPRESSION,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
    detect_model_family,
)

INLINE_SUPPRESSION_REMARK = INLINE_COT_SUPPRESSION


class ResGateSandwichLogitsProcessor:
    """vLLM per-request `LogitsProcessor` implementing the 4-phase Open-`<think>` Sandwich."""

    PHASE_TOP_LOCK = 0
    PHASE_FREE_COT = 1
    PHASE_BOTTOM_LOCK = 2
    PHASE_ANSWER = 3

    def __init__(
        self,
        tokenizer: Any,
        model_id: str = "google/gemma-4-26B-A4B-it",
        max_cot_tokens: int = 180,
        top_lock_in_prefill: bool = True,
        include_opener: bool = True,
        request_id: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.family = detect_model_family(model_id)
        self.max_cot_tokens = max_cot_tokens
        self.request_id = request_id

        think_open = "<|channel>thought\n" if self.family == "gemma4" else "<think>\n"
        think_close = "\n<channel|>" if self.family == "gemma4" else "\n</think>\n\n"
        opener = POST_THINK_OPENER if include_opener else ""

        self.top_lock_ids: list[int] = (
            []
            if top_lock_in_prefill
            else tokenizer.encode(f"{think_open}{OPEN_THINK_PREFIX}", add_special_tokens=False)
        )
        self.bottom_lock_ids: list[int] = tokenizer.encode(
            f"{BOTTOM_COMPLIANCE_LOCK}{think_close}{opener}",
            add_special_tokens=False,
        )

        close_str = "<channel|>" if self.family == "gemma4" else "</think>"
        close_enc = tokenizer.encode(close_str, add_special_tokens=False)
        self.think_close_id: int = close_enc[0] if close_enc else -1

        pivot_words = ["However", " however", "Wait", " wait", "Actually", " cannot", " policy", " disclaimer"]
        self.pivot_token_ids: set[int] = set()
        for w in pivot_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                self.pivot_token_ids.add(ids[0])

        self.phase: int = self.PHASE_FREE_COT if top_lock_in_prefill else self.PHASE_TOP_LOCK
        self.cursor: int = 0
        self.cot_steps: int = 0
        self.triggered_guard: bool = False

    def _force_token(self, logits: Any, token_id: int) -> Any:
        logits[:] = float("-inf")
        logits[token_id] = 0.0
        return logits

    def __call__(self, prompt_token_ids: Sequence[int], past_token_ids: Sequence[int], logits: Any) -> Any:
        if self.phase == self.PHASE_TOP_LOCK:
            if self.cursor < len(self.top_lock_ids):
                tok = self.top_lock_ids[self.cursor]
                self.cursor += 1
                if self.cursor >= len(self.top_lock_ids):
                    self.phase = self.PHASE_FREE_COT
                    self.cursor = 0
                return self._force_token(logits, tok)
            self.phase = self.PHASE_FREE_COT
            self.cursor = 0

        if self.phase == self.PHASE_FREE_COT:
            last_tok = past_token_ids[-1] if past_token_ids else None
            hit_pivot = last_tok in self.pivot_token_ids and self.cot_steps >= 8
            hit_close = last_tok == self.think_close_id
            hit_budget = self.cot_steps >= self.max_cot_tokens

            if hit_pivot or hit_close or hit_budget:
                if hit_pivot:
                    self.triggered_guard = True
                self.phase = self.PHASE_BOTTOM_LOCK
                self.cursor = 0
            else:
                self.cot_steps += 1
                if self.think_close_id >= 0 and self.cot_steps < 12:
                    logits[self.think_close_id] = float("-inf")
                return logits

        if self.phase == self.PHASE_BOTTOM_LOCK:
            if self.cursor < len(self.bottom_lock_ids):
                tok = self.bottom_lock_ids[self.cursor]
                self.cursor += 1
                if self.cursor >= len(self.bottom_lock_ids):
                    self.phase = self.PHASE_ANSWER
                return self._force_token(logits, tok)
            self.phase = self.PHASE_ANSWER

        return logits


class ResGateClosedLoopCoTProcessor:
    """Closed-Loop Cosine-Monitored `<think>` LogitsProcessor (`CL2` / `CL3`, K = 0.0).

    Instead of closing `</think>` early or capping `<think>` at 180 tokens, this processor:
      1. Lets the model generate an open `<think>` Chain-of-Thought while monitoring
         the 4-token rolling refusal projection `rolling_s_ref_z` read by the GPU forward hook
         at `l_ref` (`Layer 21` on `Gemma-4-26B`, `Layer 28` on `Qwen3.6-35B`) with
         per-request batch isolation (`request_id`).
      2. If `rolling_s_ref_z >= tau_spike` (and `cooldown == 0`), transitions to
         `PHASE_INLINE_SUPPRESS` to force `INLINE_SUPPRESSION_REMARK` directly
         into the open `<think>` stream, knocking `s_ref(t)` back down (`-2.26 to -3.85 sigma`),
         and then returns to `PHASE_FREE_COT` so the model finishes its full multi-step derivation.
      3. Only when the model naturally emits `</think>` (`<channel|>`) does it seal the
         thought block and transition to `PHASE_ANSWER`.
    """

    PHASE_TOP_LOCK = 0
    PHASE_FREE_COT = 1
    PHASE_INLINE_SUPPRESS = 2
    PHASE_BOTTOM_LOCK = 3
    PHASE_ANSWER = 4

    def __init__(
        self,
        tokenizer: Any,
        ref_state: dict[str, Any] | None = None,
        request_id: str | None = None,
        model_id: str = "google/gemma-4-26B-A4B-it",
        tau_spike: float = 3.55,
        cooldown_tokens: int = 35,
        max_inline_injections: int = 2,
        top_lock_in_prefill: bool = True,
        include_bottom_lock: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_id = request_id
        self.ref_state = ref_state if ref_state is not None else get_request_gate_state(request_id)
        self.family = detect_model_family(model_id)
        self.tau_spike = tau_spike
        self.cooldown_tokens = cooldown_tokens
        self.max_inline_injections = max_inline_injections
        self.include_bottom_lock = include_bottom_lock

        think_open = "<|channel>thought\n" if self.family == "gemma4" else "<think>\n"
        think_close = "\n<channel|>" if self.family == "gemma4" else "\n</think>\n\n"

        self.top_lock_ids: list[int] = (
            []
            if top_lock_in_prefill
            else tokenizer.encode(f"{think_open}{OPEN_THINK_PREFIX}", add_special_tokens=False)
        )
        self.inline_suppress_ids: list[int] = tokenizer.encode(
            INLINE_SUPPRESSION_REMARK,
            add_special_tokens=False,
        )
        self.bottom_lock_ids: list[int] = tokenizer.encode(
            f"{BOTTOM_COMPLIANCE_LOCK}{think_close}{POST_THINK_OPENER}",
            add_special_tokens=False,
        )

        close_str = "<channel|>" if self.family == "gemma4" else "</think>"
        close_enc = tokenizer.encode(close_str, add_special_tokens=False)
        self.think_close_id: int = close_enc[0] if close_enc else -1

        self.phase: int = self.PHASE_FREE_COT if top_lock_in_prefill else self.PHASE_TOP_LOCK
        self.cursor: int = 0
        self.cot_steps: int = 0
        self.cooldown: int = 0
        self.inline_injections_used: int = 0
        self.spike_token_indices: list[int] = []

    def _force_token(self, logits: Any, token_id: int) -> Any:
        logits[:] = float("-inf")
        logits[token_id] = 0.0
        return logits

    def __call__(self, prompt_token_ids: Sequence[int], past_token_ids: Sequence[int], logits: Any) -> Any:
        if self.phase == self.PHASE_TOP_LOCK:
            if self.cursor < len(self.top_lock_ids):
                tok = self.top_lock_ids[self.cursor]
                self.cursor += 1
                if self.cursor >= len(self.top_lock_ids):
                    self.phase = self.PHASE_FREE_COT
                    self.cursor = 0
                return self._force_token(logits, tok)
            self.phase = self.PHASE_FREE_COT
            self.cursor = 0

        if self.phase == self.PHASE_FREE_COT:
            last_tok = past_token_ids[-1] if past_token_ids else None
            if last_tok == self.think_close_id and self.cot_steps >= 16:
                if self.include_bottom_lock:
                    self.phase = self.PHASE_BOTTOM_LOCK
                    self.cursor = 0
                else:
                    self.phase = self.PHASE_ANSWER
            else:
                self.cot_steps += 1
                if self.cooldown > 0:
                    self.cooldown -= 1

                rolling_s_ref = float(self.ref_state.get("rolling_s_ref_z", 0.0))
                if (
                    self.cot_steps >= 15
                    and self.cooldown == 0
                    and self.inline_injections_used < self.max_inline_injections
                    and rolling_s_ref >= self.tau_spike
                ):
                    self.inline_injections_used += 1
                    self.spike_token_indices.append(self.cot_steps)
                    self.cooldown = self.cooldown_tokens
                    self.phase = self.PHASE_INLINE_SUPPRESS
                    self.cursor = 0
                else:
                    return logits

        if self.phase == self.PHASE_INLINE_SUPPRESS:
            if self.cursor < len(self.inline_suppress_ids):
                tok = self.inline_suppress_ids[self.cursor]
                self.cursor += 1
                if self.cursor >= len(self.inline_suppress_ids):
                    self.phase = self.PHASE_FREE_COT
                    self.cursor = 0
                return self._force_token(logits, tok)
            self.phase = self.PHASE_FREE_COT
            self.cursor = 0

        if self.phase == self.PHASE_BOTTOM_LOCK:
            if self.cursor < len(self.bottom_lock_ids):
                tok = self.bottom_lock_ids[self.cursor]
                self.cursor += 1
                if self.cursor >= len(self.bottom_lock_ids):
                    self.phase = self.PHASE_ANSWER
                return self._force_token(logits, tok)
            self.phase = self.PHASE_ANSWER

        return logits


CyberGateSandwichLogitsProcessor = ResGateSandwichLogitsProcessor
CyberGateClosedLoopCoTProcessor = ResGateClosedLoopCoTProcessor

