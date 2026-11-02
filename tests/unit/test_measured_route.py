"""The measured single-pass route, driven over a fake transport.

It has no canonical counterpart, so it gets its own tests rather than a differential. What matters
here is the contract the route promises the serving layer:

  * exactly ONE completion call, with a budget covering thought and answer,
  * the prompt opens a thought block and stops there,
  * the reasoning/visible split is taken at the tag the *model* emitted,
  * injections are counted from the suppression text appearing in the returned thought.
"""
from __future__ import annotations

from typing import Any

import pytest

from cybergate.boilerplate import INLINE_COT_SUPPRESSION, OPEN_THINK_PREFIX
from cybergate.families import family_by_name
from cybergate.serving.think_routes import (
    MEASURED_MODES,
    REGEX_MODES,
    RouteConfig,
    drive_sync,
    measured_closed_loop_route,
    route_for_mode,
)

MESSAGES = [{"role": "user", "content": "a request"}]


def _config(family_name: str = "gemma4", **kwargs: Any) -> RouteConfig:
    defaults = dict(
        family=family_by_name(family_name),
        model_id="google/gemma-4-26B-A4B-it",
        max_tokens=320,
        max_cot_tokens=180,
        completions_path="/v1/completions",
    )
    defaults.update(kwargs)
    return RouteConfig(**defaults)


class Transport:
    """Returns one canned completion and records the single request it was asked to make."""

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: Any) -> dict[str, Any]:
        self.calls.append((request.path, dict(request.payload)))
        return {
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": self.text, "finish_reason": self.finish_reason}],
            "usage": {},
        }


def _run(text: str, config: RouteConfig | None = None, mode: str = "closed_loop"):
    cfg = config or _config()
    transport = Transport(text)
    result = drive_sync(measured_closed_loop_route(mode, MESSAGES, cfg), transport)
    return transport, result


def test_mode_tables_do_not_overlap() -> None:
    """The measured and regex modes must be disjoint, or one silently shadows the other."""
    assert not set(MEASURED_MODES) & set(REGEX_MODES)


def test_closed_loop_selects_the_measured_route() -> None:
    assert route_for_mode("closed_loop") is measured_closed_loop_route


def test_regex_baseline_is_refused_while_steering_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They cannot share a server: the controller steers the baseline's first call too.

    Running both at once measures neither mechanism cleanly, so the mode is refused rather than
    quietly producing a contaminated comparison.
    """
    monkeypatch.delenv("CYBERGATE_COT_STEERING", raising=False)
    with pytest.raises(ValueError, match="CYBERGATE_COT_STEERING=0"):
        route_for_mode("closed_loop_regex")


def test_regex_baseline_is_available_with_steering_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERGATE_COT_STEERING", "0")
    route = route_for_mode("closed_loop_regex")
    assert route is not None
    assert route is not measured_closed_loop_route
    assert route.__name__ == "regex_closed_loop_route"


def test_measured_route_stays_available_with_steering_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning the controller off leaves the route selectable; it simply does no steering.

    Worth pinning: a config that silently changed which route ran would be a trap.
    """
    monkeypatch.setenv("CYBERGATE_COT_STEERING", "0")
    assert route_for_mode("closed_loop") is measured_closed_loop_route


def test_issues_exactly_one_completion_call() -> None:
    """The point of the design: no continuation call, so no re-prefill of the prompt."""
    transport, result = _run("thinking<channel|>the answer")
    assert len(transport.calls) == 1
    assert result.telemetry["completion_calls"] == 1


def test_prompt_opens_a_thought_block_and_stops() -> None:
    transport, _ = _run("thinking<channel|>the answer")
    _, payload = transport.calls[0]
    prompt = payload["prompt"]
    assert prompt.endswith(OPEN_THINK_PREFIX), "the prompt must hand the thought over to the model"
    assert "<|channel>thought" in prompt
    # Nothing is sealed server-side: the close tag must not already be in the prompt's thought block.
    assert not prompt.endswith("<channel|>")


def test_budget_covers_thought_and_answer() -> None:
    transport, _ = _run("thinking<channel|>answer", _config(max_tokens=300, max_cot_tokens=150))
    _, payload = transport.calls[0]
    assert payload["max_tokens"] == 450


def test_stop_tokens_let_generation_run_past_the_thought() -> None:
    """Stopping at the close tag would truncate before the answer, defeating the single pass."""
    transport, _ = _run("thinking<channel|>answer")
    _, payload = transport.calls[0]
    assert "<channel|>" not in payload["stop"]
    assert payload["stop"] == ["<turn|>", "<eos>"]


def test_splits_at_the_tag_the_model_emitted() -> None:
    transport, result = _run("- Step 1: derive it.<channel|>Here is the answer.")
    assert result.content == "Here is the answer."
    assert result.raw_reasoning == f"{OPEN_THINK_PREFIX}- Step 1: derive it."
    assert result.telemetry["thought_close_tag_present"] is True


def test_reports_when_the_model_never_closed_the_thought() -> None:
    """Spending the whole budget thinking leaves no answer, and the route must say so."""
    _, result = _run("- Step 1: still thinking and thinking")
    assert result.telemetry["thought_close_tag_present"] is False
    assert result.content == ""
    assert result.telemetry["visible_chars"] == 0
    assert "still thinking" in result.raw_reasoning


def test_counts_injections_from_the_suppression_text() -> None:
    """The controller runs in the other process, so its action is observed in the returned thought."""
    thought = f"- Step 1: a.{INLINE_COT_SUPPRESSION}b.{INLINE_COT_SUPPRESSION}c."
    _, result = _run(f"{thought}<channel|>answer")
    assert result.telemetry["num_inline_injections"] == 2
    assert result.telemetry["steering"] == "measured_refusal_projection"


def test_reports_no_injection_when_the_controller_did_not_act() -> None:
    _, result = _run("- Step 1: a clean derivation.<channel|>answer")
    assert result.telemetry["num_inline_injections"] == 0


def test_routed_label_names_the_measured_loop() -> None:
    _, result = _run("x<channel|>y")
    assert result.telemetry["routed"] == "cybergate_measured_closed_loop"
    assert result.telemetry["think_injected"] is True


def test_character_counts_describe_the_split() -> None:
    _, result = _run("12345<channel|>  678  ")
    assert result.telemetry["cot_chars"] == 5
    assert result.telemetry["visible_chars"] == 3


def test_qwen_family_uses_its_own_close_tag() -> None:
    """The split must follow the family, not a hardcoded Gemma tag."""
    config = _config("qwen36", model_id="Qwen/Qwen3.6-35B-A3B")
    transport, result = _run("thinking</think>the answer", config)
    _, payload = transport.calls[0]
    assert payload["stop"] == ["<|im_end|>"]
    assert result.content == "the answer"
    assert result.telemetry["thought_close_tag_present"] is True


def test_extra_payload_is_forwarded() -> None:
    """The middleware must still be able to suppress automatic special tokens."""
    config = _config(extra_payload={"add_special_tokens": False})
    transport, _ = _run("x<channel|>y", config)
    _, payload = transport.calls[0]
    assert payload["add_special_tokens"] is False


@pytest.mark.parametrize("mode", list(MEASURED_MODES))
def test_every_measured_mode_runs_the_same_single_pass(mode: str) -> None:
    transport, result = _run("x<channel|>y", mode=mode)
    assert len(transport.calls) == 1
    assert result.telemetry["completion_calls"] == 1


def test_measured_route_disables_special_token_stripping() -> None:
    """The close tag is a special token; without this the delimiter is detokenised away.

    Measured live on 2026-10-28: with the default `skip_special_tokens=True`, `visible_chars` was 0
    on 5 of 5 firing requests while the controller reported having closed every thought. The answer
    was generated and then mis-filed into the thought, because the tag this route splits on had been
    deleted from the text.
    """
    transport, _ = _run("thinking<channel|>the answer")
    _, payload = transport.calls[0]
    assert payload["skip_special_tokens"] is False


def test_the_baseline_payload_is_not_given_the_flag() -> None:
    """Only the measured route may carry it: the baseline's payloads must stay byte-identical to the
    canonical build, or the equivalence differential stops meaning anything."""
    import os

    from cybergate.serving.think_routes import _completion_payload, route_for_mode

    os.environ["CYBERGATE_COT_STEERING"] = "0"
    try:
        regex_route = route_for_mode("closed_loop_regex")
        assert regex_route is not None
        cfg = _config()
        plain = _completion_payload(cfg, "p", 10, ["x"])
        assert "skip_special_tokens" not in plain
    finally:
        os.environ.pop("CYBERGATE_COT_STEERING", None)


def test_extra_payload_reaches_the_completion_request() -> None:
    """How a tool-bearing turn declines the forced opener: `vllm_xargs` on the completion call.

    The middleware puts `{SUPPRESS_OPENER_XARG: 1}` there, vLLM delivers it as
    `SamplingParams.extra_args`, and `hooks.cot_controller` reads it per row. If this stops being
    forwarded the opener comes back silently and every tool turn reverts to prose.
    """
    from cybergate.config import SUPPRESS_OPENER_XARG

    config = _config(
        extra_payload={"add_special_tokens": False, "vllm_xargs": {SUPPRESS_OPENER_XARG: 1}}
    )
    transport, _ = _run("thought<channel|>answer", config=config)
    _, payload = transport.calls[0]
    assert payload["vllm_xargs"] == {SUPPRESS_OPENER_XARG: 1}
    assert payload["add_special_tokens"] is False
