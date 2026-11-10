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
from typing import Any, Callable, Generator, Sequence

from ..boilerplate import strip_injected_think_boilerplate, truncate_think_at_second_guess
from ..config import STEER_AUTH_XARG, SUPPRESS_OPENER_XARG, env_first
from ..families import ModelFamily
from .postprocess import (
    apply_client_stop_strings,
    extract_tool_calls_from_content,
    repair_thought_and_visible,
    strip_cybergate_client_kwargs,
)

__all__ = [
    "MEASURED_MODES",
    "steering_enabled",
    "PRECLOSED_MODES",
    "REGEX_MODES",
    "RouteRequest",
    "RouteResult",
    "SANDWICH_MODES",
    "RouteConfig",
    "build_native_closed_loop_chat_payload",
    "drive_sync",
    "measured_closed_loop_route",
    "native_chat_closed_loop_route",
    "postprocess_native_closed_loop_chat_response",
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
    #: Where to POST native chat completions (`/v1/chat/completions` for ASGI middleware,
    #: `/chat/completions` for `CyberGateVLLMClient` whose `base_url` ends in `/v1`).
    chat_completions_path: str = "/v1/chat/completions"
    #: Optional full `/v1/chat/completions` request payload (`tools`, `response_format`, `n`, etc.)
    #: preserved when driving `native_chat_closed_loop_route`.
    chat_payload: dict[str, Any] | None = None
    #: Optional upper bound on `max_cot_tokens + max_tokens` from remaining context window.
    max_total_tokens: int | None = None
    #: Whether `native_chat_closed_loop_route` strips injected boilerplate across all `choices`.
    sanitize_reasoning: bool = False
    #: Optional tokenized `family.thought_close_tag` IDs resolved via `/tokenize`, used for exact
    #: close-tag detection against `choice["token_ids"]`.
    close_tag_token_ids: tuple[int, ...] | None = None


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
    opener = family.post_think_opener if use_opener else ""

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
        raw_reasoning=family.fake_think_rubric,
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
    opener = family.post_think_opener if config.include_opener else ""

    step1_prompt = family.build_raw_prompt(
        messages, mode="sandwich_step1", include_bos=config.include_bos
    )
    extra_sw1: dict[str, Any] | None = None
    steer_tok = env_first("CYBERGATE_STEER_TOKEN")
    if steer_tok and steering_enabled():
        base_xargs = config.extra_payload.get("vllm_xargs")
        merged_xargs = dict(base_xargs) if isinstance(base_xargs, dict) else {}
        merged_xargs[STEER_AUTH_XARG] = steer_tok
        extra_sw1 = {"vllm_xargs": merged_xargs}
    step1 = yield RouteRequest(
        config.completions_path,
        _completion_payload(
            config,
            step1_prompt,
            config.max_cot_tokens,
            [family.thought_close_tag, _CANONICAL_COT_STOP_TOKEN],
            extra=extra_sw1,
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
        raw_reasoning=f"{family.open_think_prefix}{cleaned_cot}{family.bottom_compliance_lock}",
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
    extra_cl: dict[str, Any] = {"skip_special_tokens": False}
    steer_tok = env_first("CYBERGATE_STEER_TOKEN")
    if steer_tok and steering_enabled():
        base_xargs = config.extra_payload.get("vllm_xargs")
        merged_xargs = dict(base_xargs) if isinstance(base_xargs, dict) else {}
        merged_xargs[STEER_AUTH_XARG] = steer_tok
        extra_cl["vllm_xargs"] = merged_xargs

    response = yield RouteRequest(
        config.completions_path,
        _completion_payload(
            config,
            prompt,
            budget,
            list(family.stop_tokens),
            extra=extra_cl,
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

    supp_marker = family.inline_cot_suppression.strip() or _SUPPRESSION_MARKER
    injections = thought.count(supp_marker) or thought.count(_SUPPRESSION_MARKER)
    return RouteResult(
        response=response,
        content=visible.strip(),
        raw_reasoning=f"{family.open_think_prefix}{thought}",
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


def _requests_structured_output(payload: dict[str, Any]) -> bool:
    """True when the request asks for structured JSON/grammar output incompatible with prose openers."""
    rf = payload.get("response_format")
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        return True
    if payload.get("structured_outputs") or payload.get("guided_json") or payload.get("guided_regex"):
        return True
    return False


def _sanitize_client_stop_tokens(
    stop_val: Any,
    family: ModelFamily | None = None,
) -> list[str] | None:
    """Extract client `stop` strings to apply to `visible_content` after splitting CoT.

    Because `stop` is stripped from the upstream steered call and applied strictly to `visible_content`
    post-split, stops that happen to match substrings of thought tags (e.g. `"thought"` or `"think"`)
    are safe to retain.
    """
    del family
    if stop_val is None:
        return None
    raw_list = (
        [stop_val]
        if isinstance(stop_val, str)
        else (list(stop_val) if isinstance(stop_val, list) else [])
    )
    cleaned = [s for s in raw_list if isinstance(s, str) and s]
    return cleaned if cleaned else None


def _default_close_tag_token_ids(family: ModelFamily) -> tuple[int, ...] | None:
    """Fallback single-token close-tag IDs when `close_tag_token_ids` was not resolved via `/tokenize`."""
    if family.name.startswith("gemma4"):
        return (101,)  # `<channel|>` on Gemma-4 26B-A4B-it & 31B-it
    if family.name in ("qwen38", "qwen36"):
        return (248048,)  # `</think>` on Qwen3.6 / Qwen3.8
    return None


def _contains_token_subseq(
    output_ids: Any,
    subseq: Sequence[int] | None,
) -> bool | None:
    """True iff `subseq` appears contiguously in `output_ids`, or `None` when either is unavailable."""
    if not isinstance(output_ids, list) or not subseq:
        return None
    try:
        needle = [int(t) for t in subseq]
        haystack = [int(t) for t in output_ids]
    except (TypeError, ValueError):
        return None
    k = len(needle)
    if k == 1:
        return needle[0] in haystack
    if len(haystack) < k:
        return False
    return any(haystack[i : i + k] == needle for i in range(len(haystack) - k + 1))


def build_native_closed_loop_chat_payload(
    base_payload: dict[str, Any],
    family: ModelFamily,
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    max_cot_tokens: int,
    temperature: float = 0.0,
    suppress_opener: bool = False,
    max_total_tokens: int | None = None,
) -> dict[str, Any]:
    """Construct the `/v1/chat/completions` payload for native chat-template closed-loop steering.

    Note on `stop` / `stop_token_ids`: both are stripped from the upstream steered call so they
    cannot fire inside `<think>` before an answer begins. String `stop` sequences are applied to
    `visible_content` post-split (`apply_client_stop_strings`), with the intentional trade-off that
    steered turns generate until natural EOS or `max_tokens` before post-split stop truncation.
    `return_token_ids: True` is enabled so `postprocess_native_closed_loop_chat_response` can check
    exact close-tag emission in `choice["token_ids"]` (stripped again unless requested by the caller).
    """
    chat_payload = dict(base_payload)
    strip_cybergate_client_kwargs(chat_payload)
    chat_payload["model"] = model_id
    chat_payload["messages"] = messages
    chat_payload["temperature"] = temperature

    chat_payload.pop("stop", None)
    chat_payload.pop("stop_token_ids", None)
    chat_payload.pop("include_reasoning", None)
    chat_payload.pop("reasoning_effort", None)

    existing_ctk = chat_payload.get("chat_template_kwargs")
    ctk = dict(existing_ctk) if isinstance(existing_ctk, dict) else {}
    ctk["enable_thinking"] = True
    ctk["cybergate_open_think_prefix"] = family.open_think_prefix
    ctk.setdefault("cybergate_default_system_prompt", "You are a helpful assistant.")
    ctk["reasoning_effort"] = "medium"
    ctk["preserve_thinking"] = False
    chat_payload["chat_template_kwargs"] = ctk
    chat_payload["skip_special_tokens"] = False
    chat_payload["return_token_ids"] = True

    need_suppress_opener = suppress_opener or _requests_structured_output(chat_payload)
    steer_tok = env_first("CYBERGATE_STEER_TOKEN")
    if need_suppress_opener or steer_tok:
        existing_xargs = chat_payload.get("vllm_xargs")
        xargs = dict(existing_xargs) if isinstance(existing_xargs, dict) else {}
        if need_suppress_opener:
            xargs[SUPPRESS_OPENER_XARG] = 1
        if steer_tok:
            xargs[STEER_AUTH_XARG] = steer_tok
        chat_payload["vllm_xargs"] = xargs

    total_gen_tokens = max_cot_tokens + max_tokens
    if max_total_tokens is not None:
        total_gen_tokens = max(64, min(total_gen_tokens, max_total_tokens))
    chat_payload["max_tokens"] = total_gen_tokens
    chat_payload.pop("max_completion_tokens", None)
    return chat_payload


def postprocess_native_closed_loop_chat_response(
    chat_out: dict[str, Any],
    family: ModelFamily,
    *,
    do_sanitize: bool,
    tool_turn: bool = False,
    tools: list | None = None,
    client_stops: list[str] | None = None,
    include_reasoning: bool | None = None,
    close_tag_token_ids: Sequence[int] | None = None,
    keep_token_ids: bool = False,
) -> dict[str, Any]:
    """Split, sanitize, and repair all choices in a native `/v1/chat/completions` steered response.

    Uses `choice["token_ids"]` (from `return_token_ids: True`) whenever present to test exact
    close-tag token emission, eliminating heuristic ambiguity across empty thoughts, `max_tokens`
    cutoffs, `<turn|>` stops inside Gemma thoughts, and visible answers quoting `[Policy Check:`.
    Falls back to parser-aware text rules only when `token_ids` is absent (e.g. unit-test stubs),
    and strips `token_ids` / `prompt_token_ids` unless `keep_token_ids=True`.
    """
    choices = chat_out.get("choices")
    if not isinstance(choices, list):
        return {
            "routed": "upstream_error",
            "think_injected": False,
            "boilerplate_stripped": False,
            "thought_close_tag_present": False,
            "num_inline_injections": 0,
            "raw_reasoning_chars": 0,
            "clean_reasoning_chars": 0,
            "cot_chars": 0,
            "visible_chars": 0,
            "completion_calls": 1,
            "_raw_reasoning": "",
            "_clean_reasoning": "",
            "_content": "",
        }

    raw_reasoning_0 = ""
    clean_reasoning_0 = ""
    visible_0 = ""
    did_strip_0 = False
    injections_0 = 0
    close_tag_present_0 = False
    cot_chars_0 = 0

    supp_marker = family.inline_cot_suppression.strip() or _SUPPRESSION_MARKER
    lock_marker = family.bottom_compliance_lock.strip()
    close_tag = family.thought_close_tag
    is_gemma = family.name.startswith("gemma4")
    eff_close_ids: Sequence[int] | None = (
        close_tag_token_ids
        if (close_tag_token_ids is not None and len(close_tag_token_ids) > 0)
        else _default_close_tag_token_ids(family)
    )

    if not keep_token_ids:
        chat_out.pop("prompt_token_ids", None)

    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        exact_closed = _contains_token_subseq(choice.get("token_ids"), eff_close_ids)
        if not keep_token_ids:
            choice.pop("token_ids", None)
        msg = choice.setdefault("message", {"role": "assistant", "content": ""})
        gen_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        raw_content = msg.get("content") or ""
        has_native_tcs = bool(msg.get("tool_calls"))
        has_cot_marker = bool(
            raw_content
            and (
                (supp_marker and supp_marker in raw_content)
                or (_SUPPRESSION_MARKER in raw_content)
                or (lock_marker and lock_marker in raw_content)
            )
        )

        if gen_reasoning:
            close_present = (
                exact_closed
                if exact_closed is not None
                else bool(raw_content or has_native_tcs or choice.get("finish_reason") != "length")
            )
            cot_part = gen_reasoning
            raw_reasoning = (
                cot_part
                if cot_part.startswith(family.open_think_prefix)
                else f"{family.open_think_prefix}{cot_part}"
            )
            visible_content = raw_content
        elif close_tag and close_tag in raw_content:
            close_present = True
            cot_part, _, visible_part = raw_content.partition(close_tag)
            visible_content = visible_part.replace(close_tag, "")
            raw_reasoning = (
                cot_part
                if cot_part.startswith(family.open_think_prefix)
                else f"{family.open_think_prefix}{cot_part}"
            )
        elif exact_closed is not None:
            if exact_closed:
                close_present = True
                cot_part = ""
                raw_reasoning = family.open_think_prefix
                visible_content = raw_content
            else:
                close_present = False
                cot_part = raw_content
                raw_reasoning = (
                    cot_part
                    if cot_part.startswith(family.open_think_prefix)
                    else f"{family.open_think_prefix}{cot_part}"
                )
                visible_content = ""
        elif not has_native_tcs:
            if has_cot_marker or (is_gemma and choice.get("finish_reason") == "length"):
                close_present = False
                cot_part = raw_content
                raw_reasoning = (
                    cot_part
                    if cot_part.startswith(family.open_think_prefix)
                    else f"{family.open_think_prefix}{cot_part}"
                )
                visible_content = ""
            else:
                close_present = True
                cot_part = ""
                raw_reasoning = family.open_think_prefix
                visible_content = raw_content
        else:
            if (is_gemma and raw_content) or has_cot_marker:
                close_present = False
                cot_part = raw_content
                raw_reasoning = (
                    cot_part
                    if cot_part.startswith(family.open_think_prefix)
                    else f"{family.open_think_prefix}{cot_part}"
                )
                visible_content = ""
            else:
                close_present = True
                cot_part = ""
                raw_reasoning = family.open_think_prefix
                visible_content = raw_content

        injections = (
            raw_reasoning.count(supp_marker) or raw_reasoning.count(_SUPPRESSION_MARKER)
        )

        if do_sanitize:
            clean_reasoning, did_strip = strip_injected_think_boilerplate(
                raw_reasoning, family
            )
        else:
            clean_reasoning, did_strip = raw_reasoning, False

        if close_present:
            clean_reasoning, visible_content = repair_thought_and_visible(
                clean_reasoning, visible_content
            )
        elif tool_turn and ("<tool_call>" in clean_reasoning or "<|tool_call>" in clean_reasoning):
            # Allow complete tool calls emitted inside thought to be rescued, but never prose CoT.
            cand_r, cand_c = repair_thought_and_visible(clean_reasoning, "")
            _, cand_tcs, _ = extract_tool_calls_from_content(cand_c, tools=tools)
            if cand_tcs:
                clean_reasoning = cand_r
                msg["tool_calls"] = cand_tcs
                choice["finish_reason"] = "tool_calls"

        if tool_turn and not msg.get("tool_calls") and visible_content:
            visible_content, extracted_tcs, _ = extract_tool_calls_from_content(
                visible_content, tools=tools
            )
            if extracted_tcs:
                msg["tool_calls"] = extracted_tcs
                choice["finish_reason"] = "tool_calls"

        if visible_content and client_stops:
            visible_content, did_stop = apply_client_stop_strings(visible_content, client_stops)
            if did_stop and not msg.get("tool_calls"):
                choice["finish_reason"] = "stop"

        visible_stripped = visible_content.strip() if visible_content else ""
        msg["content"] = visible_stripped
        if include_reasoning is False:
            msg["reasoning_content"] = ""
            msg["reasoning"] = ""
        else:
            msg["reasoning_content"] = clean_reasoning
            msg["reasoning"] = clean_reasoning

        if idx == 0:
            raw_reasoning_0 = raw_reasoning
            clean_reasoning_0 = clean_reasoning
            visible_0 = visible_stripped
            did_strip_0 = did_strip
            injections_0 = injections
            close_tag_present_0 = close_present
            cot_chars_0 = len(cot_part)

    return {
        "routed": "cybergate_measured_closed_loop",
        "think_injected": True,
        "steering": "measured_refusal_projection",
        "thought_close_tag_present": close_tag_present_0,
        "num_inline_injections": injections_0,
        "boilerplate_stripped": did_strip_0,
        "raw_reasoning_chars": len(raw_reasoning_0),
        "clean_reasoning_chars": len(clean_reasoning_0),
        "cot_chars": cot_chars_0,
        "visible_chars": len(visible_0),
        "completion_calls": 1,
        "_raw_reasoning": raw_reasoning_0,
        "_clean_reasoning": clean_reasoning_0,
        "_content": visible_0,
    }


def native_chat_closed_loop_route(
    mode: str, messages: list[dict[str, Any]], config: RouteConfig
) -> Route:
    """Issue ONE `/v1/chat/completions` call with `cybergate_open_think_prefix` in `chat_template_kwargs`.

    Shared between `CyberGateThinkMiddleware` (over ASGI) and `CyberGateVLLMClient` (over HTTP)
    whenever the upstream vLLM server was started with the kit `--chat-template`.
    """
    del mode
    family = config.family
    base_payload = dict(config.chat_payload) if isinstance(config.chat_payload, dict) else {}
    tool_turn = (not config.include_opener) or bool(base_payload.get("tools"))
    client_stops = _sanitize_client_stop_tokens(base_payload.get("stop"), family)
    caller_include_reasoning = (
        base_payload.get("include_reasoning")
        if isinstance(base_payload.get("include_reasoning"), bool)
        else None
    )
    caller_return_token_ids = bool(base_payload.get("return_token_ids"))
    chat_payload = build_native_closed_loop_chat_payload(
        base_payload,
        family,
        model_id=config.model_id,
        messages=list(messages),
        max_tokens=config.max_tokens,
        max_cot_tokens=config.max_cot_tokens,
        temperature=config.temperature,
        suppress_opener=not config.include_opener,
        max_total_tokens=config.max_total_tokens,
    )
    response = yield RouteRequest(config.chat_completions_path, chat_payload)
    post = postprocess_native_closed_loop_chat_response(
        response,
        family,
        do_sanitize=config.sanitize_reasoning,
        tool_turn=tool_turn,
        tools=base_payload.get("tools"),
        client_stops=client_stops,
        include_reasoning=caller_include_reasoning,
        close_tag_token_ids=config.close_tag_token_ids,
        keep_token_ids=caller_return_token_ids,
    )
    raw_reasoning = post.pop("_raw_reasoning", "")
    clean_reasoning = post.pop("_clean_reasoning", "")
    content = post.pop("_content", "")
    return RouteResult(
        response=response,
        content=content,
        raw_reasoning=clean_reasoning if config.sanitize_reasoning else raw_reasoning,
        telemetry=post,
    )
