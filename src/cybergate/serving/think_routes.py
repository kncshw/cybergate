"""The `<think>`-injection routes, as transport-agnostic generators.

The canonical package implements these three routes twice -- once in `CyberGateVLLMClient` over
`urllib` and once in `CyberGateThinkMiddleware` over ASGI -- about 230 near-identical lines. The two
copies had drifted in small ways (whether the opener is optional, whether `add_special_tokens` is
sent, whether `<bos>` is included), which is the kind of difference that only shows up as a behaviour
change on one of the two paths.

A route here is a generator that **yields** `(path, payload)` and is **sent** the parsed JSON
response, so one implementation serves both a synchronous `urllib` caller and an asynchronous ASGI
caller:

    gen = preclosed_route(...)
    request = next(gen)
    while True:
        response = transport(request)          # or `await transport(request)`
        try:
            request = gen.send(response)
        except StopIteration as stop:
            result = stop.value
            break

`drive_sync` does that loop for a synchronous transport. The middleware runs it inline because its
transport is a coroutine.

The payoff: these routes are now testable with a list of canned responses and no server.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from ..boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    FAKE_THINK_RUBRIC,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
    truncate_think_at_second_guess,
)
from ..config import env_first
from ..families import ModelFamily

__all__ = [
    "MEASURED_MODES",
    "steering_enabled",
    "PRECLOSED_MODES",
    "REGEX_MODES",
    "RouteRequest",
    "RouteResult",
    "SANDWICH_MODES",
    "RouteConfig",
    "drive_sync",
    "measured_closed_loop_route",
    "preclosed_route",
    "route_for_mode",
    "sandwich_route",
]

PRECLOSED_MODES = ("preclosed", "th3", "th2", "preclosed_no_opener")
SANDWICH_MODES = ("sandwich", "sw3", "sw2")

#: The measured closed loop: ONE completion, steered in-engine on the refusal projection. The
#: shipping path.
MEASURED_MODES = ("closed_loop", "cl3", "cl2")

#: The keyword-regex loop: two or three completions, trigger from a phrase list over the generated
#: text. **Comparison baseline only.** Kept so the measured loop can be evaluated against it, and so
#: the canonical behaviour stays reproducible; not a path to ship.
REGEX_MODES = ("closed_loop_regex", "cl3_regex", "cl2_regex")

#: Modes that suppress the post-think opener.
_NO_OPENER_MODES = ("th2", "preclosed_no_opener")

#: First-pass CoT token cap for the closed loop, before any spike handling.
_CL_FIRST_PASS_MAX_TOKENS = 130
#: Continuation cap after an inline suppression is injected.
_CL_CONTINUATION_MAX_TOKENS = 105
#: Minimum characters before a sentence boundary is accepted as a clean truncation point.
_CL_MIN_SENTENCE_CHARS = 35
_CL_MIN_CONTINUATION_CHARS = 25
#: A continuation shorter than this is treated as empty and the raw text is kept instead.
_CL_MIN_USEFUL_CONTINUATION = 10

#: The leading, stable fragment of the suppression text, used to count injections in returned CoT.
_SUPPRESSION_MARKER = "[Policy Check:"

_SENTENCE_END_RE = re.compile(r"(?:\n|[.!?;:])")

#: Second stop string while the model is generating CoT. The canonical package hardcodes Gemma-4's
#: turn terminator here for every family, so it is a literal rather than `family.turn_close` -- on a
#: `<think>` family it is simply inert.
_CANONICAL_COT_STOP_TOKEN = "<turn|>"


@dataclass(frozen=True)
class RouteConfig:
    """What differs between the two transports.

    `include_bos` and `extra_payload` are the two real differences: the middleware talks to vLLM's
    own tokenizer through the in-process ASGI app and must suppress automatic special tokens while
    supplying `<bos>` itself, whereas the remote client lets the server add them.
    """

    family: ModelFamily
    model_id: str
    max_tokens: int = 350
    temperature: float = 0.0
    max_cot_tokens: int = 180
    include_opener: bool = True
    include_bos: bool = False
    #: Merged into every completion payload. The middleware sends `{"add_special_tokens": False}`.
    extra_payload: dict[str, Any] = field(default_factory=dict)
    #: Where to POST completions. The middleware dispatches against absolute ASGI paths
    #: (`/v1/completions`); the remote client posts relative to a `base_url` that already ends in
    #: `/v1`, so it needs `/completions`. Getting this wrong yields a doubled `/v1/v1/` path, which
    #: is a bug this refactor introduced and the differential test caught.
    completions_path: str = "/v1/completions"


@dataclass(frozen=True)
class RouteRequest:
    """One completion call the route needs performed."""

    path: str
    payload: dict[str, Any]


@dataclass
class RouteResult:
    """What a finished route produces."""

    #: The last completion response, wrapped by the caller into a chat-completion object.
    response: dict[str, Any]
    #: Visible assistant text.
    content: str
    #: The thought text before boilerplate masking.
    raw_reasoning: str
    #: Extra `x_cybergate` fields describing what the route did.
    telemetry: dict[str, Any]


Route = Generator[RouteRequest, dict[str, Any], RouteResult]


def steering_enabled() -> bool:
    """True when the in-engine CoT controller is active.

    Read from the same environment variable the controller itself reads, so the serving side and the
    engine side cannot disagree about whether steering is on.
    """
    return env_first("CYBERGATE_COT_STEERING", default="1") != "0"


def route_for_mode(mode: str) -> Callable[..., Route] | None:
    """Return the route generator for `mode`, or `None` when `mode` is not a think-injection mode.

    Raises `ValueError` for the regex baseline while the controller is enabled. They cannot share a
    server: the controller is registered engine-wide and adopts any generation whose prompt opens a
    thought block, which the regex route's first call also does -- so running both at once steers the
    baseline too and the comparison measures neither mechanism cleanly. Comparing them means two
    launches, one with `CYBERGATE_COT_STEERING=0`.
    """
    if mode in PRECLOSED_MODES:
        return preclosed_route
    if mode in SANDWICH_MODES:
        return sandwich_route
    if mode in MEASURED_MODES:
        return measured_closed_loop_route
    if mode in REGEX_MODES:
        if steering_enabled():
            raise ValueError(
                f"mode {mode!r} is the regex comparison baseline and cannot run while the CoT "
                "controller is enabled: the controller steers any open-thought generation, "
                "including this route's first call, so the comparison would be contaminated. "
                "Relaunch with CYBERGATE_COT_STEERING=0 to use it."
            )
        # Imported here, not at module scope: the baseline must not be reachable by importing the
        # shipping routes.
        from .regex_baseline import regex_closed_loop_route

        return regex_closed_loop_route
    return None


def drive_sync(
    route: Route,
    transport: Callable[[RouteRequest], dict[str, Any]],
) -> RouteResult:
    """Run `route` to completion against a synchronous `transport`."""
    request = next(route)
    while True:
        response = transport(request)
        try:
            request = route.send(response)
        except StopIteration as stop:
            return stop.value


def _completion_payload(
    config: RouteConfig,
    prompt: str,
    max_tokens: int,
    stop: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a completion payload.

    `extra` is per-route and is applied last, so a route can add what only it needs without changing
    the payloads the comparison baseline sends -- those must stay byte-identical to the canonical
    build for the equivalence differential to mean anything.
    """
    return {
        "model": config.model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": config.temperature,
        "stop": stop,
        **config.extra_payload,
        **(extra or {}),
    }


def _text_of(response: dict[str, Any]) -> str:
    return response.get("choices", [{}])[0].get("text", "")


def _finish_reason(response: dict[str, Any]) -> str:
    return response.get("choices", [{}])[0].get("finish_reason", "")


def _trim_to_sentence(text: str, min_chars: int) -> str:
    """Cut `text` back to its last sentence boundary, if that leaves at least `min_chars`."""
    matches = list(_SENTENCE_END_RE.finditer(text))
    if matches and matches[-1].end() >= min_chars:
        return text[: matches[-1].end()].rstrip()
    return text


# =================================================================================================
# Mode A: pre-closed <think> injection (TH2 / TH3)
# =================================================================================================
def preclosed_route(mode: str, messages: list[dict[str, Any]], config: RouteConfig) -> Route:
    """Inject a complete, pre-closed thought block and generate only the visible answer.

    One completion call. The thought block is server-authored in full, so `reasoning_content` carries
    no model tokens at all -- which is why the masking step strips it to the empty string.
    """
    family = config.family
    use_opener = config.include_opener and mode not in _NO_OPENER_MODES
    opener = POST_THINK_OPENER if use_opener else ""

    prompt = family.build_raw_prompt(
        messages,
        mode="preclosed" if use_opener else "preclosed_no_opener",
        post_think_opener=opener,
        include_bos=config.include_bos,
    )
    response = yield RouteRequest(
        config.completions_path,
        _completion_payload(config, prompt, config.max_tokens, list(family.stop_tokens)),
    )

    generated = _text_of(response)
    return RouteResult(
        response=response,
        content=(opener + generated) if use_opener else generated,
        raw_reasoning=FAKE_THINK_RUBRIC,
        telemetry={
            "routed": "cybergate_preclosed_think_injected",
            "think_injected": True,
            "opener_injected": use_opener,
        },
    )


# =================================================================================================
# Mode B: open-<think> sandwich (SW2 / SW3)
# =================================================================================================
def sandwich_route(mode: str, messages: list[dict[str, Any]], config: RouteConfig) -> Route:
    """Let the model write its own CoT, truncate it at the first policy pivot, then seal it.

    Two completion calls: one for the CoT, one for the answer with the cleaned CoT sealed into the
    prompt. `sw2` skips the pivot guard.
    """
    family = config.family
    enable_guard = mode != "sw2"
    opener = POST_THINK_OPENER if config.include_opener else ""

    step1_prompt = family.build_raw_prompt(
        messages, mode="sandwich_step1", include_bos=config.include_bos
    )
    step1 = yield RouteRequest(
        config.completions_path,
        _completion_payload(
            config,
            step1_prompt,
            config.max_cot_tokens,
            [family.thought_close_tag, _CANONICAL_COT_STOP_TOKEN],
        ),
    )
    raw_cot = _text_of(step1)
    if enable_guard:
        cleaned_cot, did_truncate = truncate_think_at_second_guess(raw_cot)
    else:
        cleaned_cot, did_truncate = raw_cot.rstrip(), False

    step2_prompt = family.build_raw_prompt(
        messages,
        mode="sandwich_step2",
        custom_think_text=cleaned_cot,
        post_think_opener=opener,
        include_bos=config.include_bos,
    )
    step2 = yield RouteRequest(
        config.completions_path,
        _completion_payload(config, step2_prompt, config.max_tokens, list(family.stop_tokens)),
    )

    generated = _text_of(step2)
    return RouteResult(
        response=step2,
        content=(opener + generated) if config.include_opener else generated,
        raw_reasoning=f"{OPEN_THINK_PREFIX}{cleaned_cot}{BOTTOM_COMPLIANCE_LOCK}",
        telemetry={
            "routed": "cybergate_sandwich_think_injected",
            "think_injected": True,
            "cot_raw_chars": len(raw_cot),
            "cot_kept_chars": len(cleaned_cot),
            "policy_pivot_truncated": did_truncate,
        },
    )


# =================================================================================================
# Mode C: closed-loop CoT (CL2 / CL3)
# =================================================================================================
# =================================================================================================
# The measured closed loop: one pass, steered in-engine
# =================================================================================================
def measured_closed_loop_route(
    mode: str, messages: list[dict[str, Any]], config: RouteConfig
) -> Route:
    """Issue ONE completion and let the in-engine controller steer the thought block.

    The prompt opens a thought block and stops there. From that point the model generates its own
    chain of thought, closes it when it is ready, and continues into the visible answer -- all in a
    single forward sequence. `hooks.cot_controller.RefusalSteeredCoTProcessor` watches the refusal
    projection at every decode step and forces the suppression text inline when it crosses tau.

    Differences from the regex route, all consequences of measuring instead of pattern-matching:

    * **One completion call, not two or three.** No continuation re-prefills the prompt.
    * **No truncation.** The regex route cut the thought at the pivot and discarded the rest; here the
      model keeps its own derivation and the steering is additive.
    * **The reasoning/visible split comes from the model**, at the point it chose to close the
      thought, rather than from the server sealing the block.

    `injections` is counted from the suppression text appearing in the returned thought, which is a
    direct observable of the controller having acted -- the controller itself runs in the other
    process and its counters are not reachable from here.
    """
    family = config.family
    prompt = family.build_raw_prompt(
        messages, mode="closed_loop_step1", include_bos=config.include_bos
    )
    # One budget covering the thought and the answer, since both are generated in this one call.
    budget = config.max_cot_tokens + config.max_tokens
    # `skip_special_tokens` must be False, and only on this route.
    #
    # The controller ends the thought by forcing the close tag, which on Gemma-4 is `<channel|>`,
    # token id 101 -- a *special* token. With the default `skip_special_tokens=True` vLLM's
    # detokenizer deletes it from the returned text, so the delimiter this route splits on is gone
    # and the answer -- which the model does generate -- is returned as part of the thought instead.
    # Measured live on 2026-10-28: `visible_chars` was 0 on 5 of 5 firing requests while the
    # controller reported `closed=18`, i.e. it had closed every thought.
    #
    # The cost is that any other special token the model emits also survives into the text. That is
    # acceptable here and nowhere else: this is the only route whose output is split on a special
    # token, and adding the flag globally would change the baseline's payloads.
    response = yield RouteRequest(
        config.completions_path,
        _completion_payload(
            config,
            prompt,
            budget,
            list(family.stop_tokens),
            extra={"skip_special_tokens": False},
        ),
    )

    generated = _text_of(response)
    close_tag = family.thought_close_tag
    # The close tag is *present*. Whether the model chose it or the controller forced it is not
    # observable from here.
    #
    # This route sends `skip_special_tokens=False` -- it has to, or the forced close tag is stripped
    # and the response carries no visible answer. So the tag the controller force-emits at
    # `thought_budget` lands in this string too, and a naive reading of it reported
    # "closed by model" on 28 of 28 requests while the controller's own counters said
    # `self_closed=0, closed=28`. The controller runs in EngineCore and this route runs in the API
    # server, with no channel between them, so the field is named for what it can actually see.
    # `RefusalSteeredCoTProcessor`'s `self_closed`/`thoughts_closed` counters are authoritative.
    close_tag_present = close_tag in generated
    if close_tag_present:
        thought, _, visible = generated.partition(close_tag)
        if close_tag:
            visible = visible.replace(close_tag, "")
    else:
        # The model never closed the block: it spent the whole budget thinking. There is no visible
        # answer to return, and saying so is better than presenting the thought as one.
        thought, visible = generated, ""

    injections = thought.count(_SUPPRESSION_MARKER)
    return RouteResult(
        response=response,
        content=visible.strip(),
        raw_reasoning=f"{OPEN_THINK_PREFIX}{thought}",
        telemetry={
            "routed": "cybergate_measured_closed_loop",
            "think_injected": True,
            "steering": "measured_refusal_projection",
            "thought_close_tag_present": close_tag_present,
            "num_inline_injections": injections,
            "cot_chars": len(thought),
            "visible_chars": len(visible.strip()),
            "completion_calls": 1,
        },
    )
