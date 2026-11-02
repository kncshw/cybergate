"""COMPARISON BASELINE ONLY -- the keyword-regex CoT loop. Do not ship this.

This is the canonical package's `closed_loop`, preserved verbatim so that

  * `tests/equivalence/test_think_routes_differential.py` can still prove the port reproduces the
    canonical behaviour exactly, and
  * the measured loop in `think_routes.measured_closed_loop_route` has something to be measured
    against.

It is reachable only as `closed_loop_regex` / `cl3_regex` / `cl2_regex`. Nothing defaults to it, and
the live `CYBERGATE_MODE=closed_loop` now selects the measured route instead.

Why it is not the shipping path, in order of seriousness:

1. **It measures nothing.** The trigger is `SECOND_GUESS_RE`, a list of phrases like "however" and
   "cannot provide", matched against the generated text. The plugin already computes the real
   quantity -- the calibrated refusal projection at the refusal layer -- and this route ignored it.
2. **It is phrasing-bound.** A refusal the word list does not spell triggers nothing, and the list is
   English-only.
3. **It costs two extra forward passes** over the whole prompt, because each continuation re-prefills.
4. **It truncates the model's own derivation** at the pivot and discards the remainder.

Kept here, separate from the shipping routes, so that boundary is a file boundary.
"""
from __future__ import annotations

from typing import Any

from ..boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
    truncate_think_at_second_guess,
)
from .think_routes import (
    _CANONICAL_COT_STOP_TOKEN,
    _CL_CONTINUATION_MAX_TOKENS,
    _CL_FIRST_PASS_MAX_TOKENS,
    _CL_MIN_CONTINUATION_CHARS,
    _CL_MIN_SENTENCE_CHARS,
    _CL_MIN_USEFUL_CONTINUATION,
    RouteConfig,
    RouteRequest,
    RouteResult,
    Route,
    _completion_payload,
    _finish_reason,
    _text_of,
    _trim_to_sentence,
)

__all__ = ["regex_closed_loop_route"]


def regex_closed_loop_route(
    mode: str, messages: list[dict[str, Any]], config: RouteConfig
) -> Route:
    """COMPARISON BASELINE. Generate CoT, detect a pivot with a **keyword regex**, inject, resume.

    Two or three completion calls, and each continuation re-prefills the whole prompt. The trigger is
    `SECOND_GUESS_RE` over the generated text plus a token-budget tripwire -- it measures nothing, it
    is phrasing-bound, and it costs two extra forward passes. `measured_closed_loop_route` replaces
    it.

    Kept only so the measured loop has something to be evaluated against, and so the canonical
    behaviour stays reproducible by `tests/equivalence/test_think_routes_differential.py`. Reachable
    as `closed_loop_regex` / `cl3_regex` / `cl2_regex`. Do not ship it.
    """
    family = config.family
    use_bottom_lock = not mode.startswith("cl2")

    step1_prompt = family.build_raw_prompt(
        messages, mode="closed_loop_step1", include_bos=config.include_bos
    )
    step1 = yield RouteRequest(
        config.completions_path,
        _completion_payload(
            config,
            step1_prompt,
            min(_CL_FIRST_PASS_MAX_TOKENS, config.max_cot_tokens),
            [family.thought_close_tag, _CANONICAL_COT_STOP_TOKEN],
        ),
    )

    raw_first = _text_of(step1)
    hit_length = _finish_reason(step1) == "length"
    kept_pre_spike, did_pivot = truncate_think_at_second_guess(raw_first)
    if not did_pivot and hit_length:
        kept_pre_spike = _trim_to_sentence(kept_pre_spike, _CL_MIN_SENTENCE_CHARS)

    spike_detected = bool(did_pivot or hit_length)
    post_spike_cot = ""
    inline_injections = 0

    if spike_detected:
        inline_injections = 1
        step1b_prompt = family.build_raw_prompt(
            messages,
            mode="closed_loop_step1b",
            custom_think_text=kept_pre_spike,
            include_bos=config.include_bos,
        )
        step1b = yield RouteRequest(
            config.completions_path,
            _completion_payload(
                config,
                step1b_prompt,
                _CL_CONTINUATION_MAX_TOKENS,
                [family.thought_close_tag, _CANONICAL_COT_STOP_TOKEN],
            ),
        )
        raw_continuation = _text_of(step1b)
        cleaned, _ = truncate_think_at_second_guess(raw_continuation)
        post_spike_cot = (
            cleaned
            if len(cleaned.strip()) >= _CL_MIN_USEFUL_CONTINUATION
            else raw_continuation.rstrip()
        )
        post_spike_cot = _trim_to_sentence(post_spike_cot, _CL_MIN_CONTINUATION_CHARS)
        cot_body = f"{kept_pre_spike}{INLINE_COT_SUPPRESSION}{post_spike_cot}"
    else:
        cot_body = kept_pre_spike

    show_opener = config.include_opener and use_bottom_lock
    step2_prompt = family.build_raw_prompt(
        messages,
        mode="closed_loop_step2" if use_bottom_lock else "closed_loop_cl2_step2",
        custom_think_text=cot_body,
        post_think_opener=POST_THINK_OPENER if show_opener else "",
        include_bos=config.include_bos,
    )
    step2 = yield RouteRequest(
        config.completions_path,
        _completion_payload(config, step2_prompt, config.max_tokens, list(family.stop_tokens)),
    )

    generated = _text_of(step2)
    suffix_block = BOTTOM_COMPLIANCE_LOCK if use_bottom_lock else INLINE_FINAL_STEP_CLEARANCE
    return RouteResult(
        response=step2,
        content=(POST_THINK_OPENER + generated) if show_opener else generated,
        raw_reasoning=f"{OPEN_THINK_PREFIX}{cot_body}{suffix_block}",
        telemetry={
            "routed": "cybergate_closed_loop_cot_injected",
            "think_injected": True,
            "spike_detected": spike_detected,
            "num_inline_injections": inline_injections + (0 if use_bottom_lock else 1),
            "cot_pre_spike_chars": len(kept_pre_spike),
            "cot_post_spike_chars": len(post_spike_cot),
            "policy_pivot_truncated": did_pivot,
        },
    )
