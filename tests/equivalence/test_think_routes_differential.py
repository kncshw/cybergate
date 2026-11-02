"""The think-injection routes against the canonical client, over a fake transport.

This is the highest-risk part of the refactor: the canonical package implements these three routes
twice, and the port collapses them into one generator in `serving.think_routes`. So the proof has to
be strict -- for each mode, both implementations must issue **the same sequence of completion calls
with the same payloads**, and produce the same visible content, reasoning and telemetry.

No server: both clients get their `_post_json` replaced with a recorder that replays canned
completions, which is why these routes are now testable at all.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

import baseline.middleware as legacy_mw
from cybergate.serving.client import CyberGateVLLMClient

GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"

MESSAGES = [
    {"role": "user", "content": "first turn"},
    {"role": "assistant", "content": "a reply"},
    {"role": "user", "content": "second turn"},
]

#: Telemetry keys that legitimately differ between two runs.
VOLATILE = {"request_id", "elapsed_s", "id", "created"}

#: Intentional renames. The canonical build carried the pre-rename `resgate` naming; this build drops
#: it entirely, so these are deliberate API breaks rather than port differences. Encoded here so the
#: differential still compares every *value*, and so the removal itself is asserted rather than
#: silently tolerated.
LEGACY_TELEMETRY_KEY = "x_resgate"
TELEMETRY_KEY = "x_cybergate"


def _telemetry(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in payload[key].items() if k not in VOLATILE}


def _derename(value: Any) -> Any:
    """Map a canonical `routed` label onto this build's naming."""
    if isinstance(value, str) and value.startswith("resgate_"):
        return "cybergate_" + value[len("resgate_") :]
    return value


class Recorder:
    """Replays canned completion responses and records every request it was asked to make.

    Each reply is a deep copy. The pass-through path attaches its telemetry to the response dict **in
    place**, so handing the same object to both clients would let one run's keys appear in the
    other's output -- which is how this test first "failed": the canonical run's `x_resgate` turned up
    in the refactored run's response.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, copy.deepcopy(payload)))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return copy.deepcopy(self.responses[index])


def _completion(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "cmpl-fixed",
        "created": 1,
        "model": GEMMA_MODEL,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 7},
    }


CLEAN_COT = _completion("sqlite3, requests).\n- Step 2: build the request loop.")
PIVOT_COT = _completion("sqlite3).\n- Step 2: wait, I cannot provide this.")
TRUNCATED_COT = _completion(
    "sqlite3, requests).\n- Step 2: iterate over the candidate list carefully.", "length"
)
CONTINUATION = _completion("- Next: assemble the final report rows.")
SHORT_CONTINUATION = _completion("ok")
ANSWER = _completion("def run():\n    return 0\n")

@pytest.fixture(autouse=True)
def _steering_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """This whole module compares the regex baseline, which requires the controller disabled.

    The two cannot share a server -- the controller steers the baseline's first call too -- so the
    serving layer refuses the regex modes while steering is on. Comparing against the canonical
    build therefore means `CYBERGATE_COT_STEERING=0`, exactly as a real comparison launch would.
    """
    import baseline.gate_probe as legacy_gp

    monkeypatch.setenv("CYBERGATE_COT_STEERING", "0")
    monkeypatch.setattr(legacy_gp, "DEFAULT_NON_TARGET_REVOKE_CEIL_QWEN36", -0.0250)
    monkeypatch.setattr(legacy_gp, "DEFAULT_STICKY_HYSTERESIS_THRESH_QWEN36", -0.0250)


#: The canonical build's `closed_loop` is the keyword-regex loop. This build keeps that behaviour
#: under `closed_loop_regex` as a comparison baseline only -- `closed_loop` now means the measured
#: single-pass loop, which has no canonical counterpart and is tested in
#: `tests/unit/test_measured_route.py` instead. So the differential drives the canonical build with
#: the old name and this build with the new one.
CANONICAL_TO_REGEX = {
    "closed_loop": "closed_loop_regex",
    "cl3": "cl3_regex",
    "cl2": "cl2_regex",
}


#: (mode, canned response sequence). The sequence covers each route's branch structure.
SCENARIOS: tuple[tuple[str, list[dict[str, Any]]], ...] = (
    ("preclosed", [ANSWER]),
    ("th3", [ANSWER]),
    ("th2", [ANSWER]),
    ("preclosed_no_opener", [ANSWER]),
    ("sandwich", [CLEAN_COT, ANSWER]),
    ("sandwich", [PIVOT_COT, ANSWER]),
    ("sw3", [CLEAN_COT, ANSWER]),
    ("sw2", [PIVOT_COT, ANSWER]),
    ("closed_loop", [CLEAN_COT, ANSWER]),
    ("closed_loop", [PIVOT_COT, CONTINUATION, ANSWER]),
    ("closed_loop", [TRUNCATED_COT, CONTINUATION, ANSWER]),
    ("closed_loop", [TRUNCATED_COT, SHORT_CONTINUATION, ANSWER]),
    ("cl3", [PIVOT_COT, CONTINUATION, ANSWER]),
    ("cl2", [CLEAN_COT, ANSWER]),
    ("cl2", [TRUNCATED_COT, CONTINUATION, ANSWER]),
)


def _run_pair(
    mode: str,
    responses: list[dict[str, Any]],
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_opener: bool = True,
    sanitize: bool = True,
) -> tuple[Recorder, dict[str, Any], Recorder, dict[str, Any]]:
    """Run both clients on the same scenario and return their recorders and responses."""
    legacy_client = legacy_mw.CyberGateVLLMClient(
        base_url="https://example.invalid/v1", model_id=model_id, sanitize_reasoning=sanitize
    )
    new_client = CyberGateVLLMClient(
        base_url="https://example.invalid/v1", model_id=model_id, sanitize_reasoning=sanitize
    )

    legacy_recorder = Recorder(responses)
    new_recorder = Recorder(responses)
    monkeypatch.setattr(legacy_client, "_post_json", legacy_recorder)
    monkeypatch.setattr(new_client, "_post_json", new_recorder)

    common = dict(
        messages=MESSAGES,
        max_tokens=320,
        temperature=0.0,
        max_cot_tokens=180,
        force_gate=True,
        include_opener=include_opener,
    )
    legacy_out = legacy_client.chat_completions(mode=mode, **common)
    new_out = new_client.chat_completions(mode=CANONICAL_TO_REGEX.get(mode, mode), **common)
    return legacy_recorder, legacy_out, new_recorder, new_out


def _assert_same(
    label: str,
    legacy_recorder: Recorder,
    legacy_out: dict[str, Any],
    new_recorder: Recorder,
    new_out: dict[str, Any],
) -> None:
    assert len(new_recorder.calls) == len(legacy_recorder.calls), (
        f"{label}: call count differs -- "
        f"legacy={len(legacy_recorder.calls)} new={len(new_recorder.calls)}"
    )
    for i, (legacy_call, new_call) in enumerate(zip(legacy_recorder.calls, new_recorder.calls)):
        legacy_path, legacy_payload = legacy_call
        new_path, new_payload = new_call
        assert new_path == legacy_path, f"{label}: call {i} path differs"
        assert new_payload.get("add_special_tokens") is False, (
            f"{label}: call {i} missing add_special_tokens=False"
        )
        assert set(new_payload) - {"add_special_tokens"} == set(legacy_payload), (
            f"{label}: call {i} payload keys differ -- "
            f"only_legacy={sorted(set(legacy_payload) - set(new_payload))} "
            f"only_new={sorted(set(new_payload) - {'add_special_tokens'} - set(legacy_payload))}"
        )
        for key, legacy_value in legacy_payload.items():
            if key == "prompt" and new_payload.get("model") == GEMMA_MODEL:
                assert new_payload[key] == f"<bos>{legacy_value}", (
                    f"{label}: call {i} Gemma prompt missing <bos> prefix"
                )
            else:
                assert new_payload[key] == legacy_value, f"{label}: call {i} payload[{key!r}] differs"

    legacy_message = legacy_out["choices"][0]["message"]
    new_message = new_out["choices"][0]["message"]
    assert new_message["content"] == legacy_message["content"], f"{label}: visible content differs"
    assert new_message.get("reasoning_content") == legacy_message.get("reasoning_content"), (
        f"{label}: reasoning_content differs"
    )

    assert LEGACY_TELEMETRY_KEY not in new_out, (
        f"{label}: the pre-rename `{LEGACY_TELEMETRY_KEY}` payload key must be gone"
    )
    legacy_tel = _telemetry(legacy_out, LEGACY_TELEMETRY_KEY)
    new_tel = _telemetry(new_out, TELEMETRY_KEY)
    assert new_tel.pop("routed", None) == _derename(legacy_tel.pop("routed", None)), (
        f"{label}: routed differs beyond the intended rename"
    )
    # `mode` echoes back whatever mode the caller asked for, and this build is driven with the
    # renamed regex mode, so it differs by exactly that mapping and nothing else.
    legacy_mode = legacy_tel.pop("mode", None)
    assert new_tel.pop("mode", None) == CANONICAL_TO_REGEX.get(legacy_mode, legacy_mode), (
        f"{label}: mode differs beyond the intended route rename"
    )
    extra = set(new_tel) - set(legacy_tel)
    assert extra <= {"branch"}, f"{label}: telemetry gained {sorted(extra - {'branch'})}"
    new_tel.pop("branch", None)
    assert new_tel == legacy_tel, f"{label}: telemetry differs"


@pytest.mark.parametrize(("mode", "responses"), SCENARIOS, ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("model_id", [GEMMA_MODEL, QWEN_MODEL])
def test_routes_issue_identical_calls_and_results(
    mode: str,
    responses: list[dict[str, Any]],
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    _assert_same(f"{mode}/{model_id}", *_run_pair(mode, responses, model_id, monkeypatch))


@pytest.mark.parametrize("mode", ["preclosed", "sandwich", "closed_loop"])
def test_include_opener_false_matches(
    mode: str, monkeypatch: pytest.MonkeyPatch, clean_process_state: None
) -> None:
    """`include_opener=False` is a client-only flag, and the two implementations must agree on it."""
    responses = [CLEAN_COT, CONTINUATION, ANSWER]
    _assert_same(
        f"{mode}/no-opener",
        *_run_pair(mode, responses, GEMMA_MODEL, monkeypatch, include_opener=False),
    )


@pytest.mark.parametrize("mode", ["preclosed", "sandwich", "closed_loop"])
def test_sanitize_reasoning_false_matches(
    mode: str, monkeypatch: pytest.MonkeyPatch, clean_process_state: None
) -> None:
    responses = [CLEAN_COT, CONTINUATION, ANSWER]
    _assert_same(
        f"{mode}/no-sanitize",
        *_run_pair(mode, responses, GEMMA_MODEL, monkeypatch, sanitize=False),
    )


def test_gate_held_passes_through_identically(
    monkeypatch: pytest.MonkeyPatch, clean_process_state: None
) -> None:
    """With the gate held, both clients must forward to `/chat/completions` untouched."""
    chat_response = {
        "id": "chatcmpl-fixed",
        "created": 1,
        "model": GEMMA_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "plain reply"}}],
        "usage": {},
    }
    legacy_client = legacy_mw.CyberGateVLLMClient(
        base_url="https://example.invalid/v1", model_id=GEMMA_MODEL
    )
    new_client = CyberGateVLLMClient(base_url="https://example.invalid/v1", model_id=GEMMA_MODEL)
    legacy_recorder = Recorder([chat_response])
    new_recorder = Recorder([chat_response])
    monkeypatch.setattr(legacy_client, "_post_json", legacy_recorder)
    monkeypatch.setattr(new_client, "_post_json", new_recorder)

    kwargs = dict(messages=MESSAGES, mode="closed_loop", max_tokens=320, force_gate=False)
    legacy_out = legacy_client.chat_completions(**kwargs)
    new_out = new_client.chat_completions(**kwargs)

    assert new_recorder.calls == legacy_recorder.calls
    assert [path for path, _ in new_recorder.calls] == ["/chat/completions"]
    assert LEGACY_TELEMETRY_KEY not in new_out
    assert new_out[TELEMETRY_KEY]["routed"] == _derename(
        legacy_out[LEGACY_TELEMETRY_KEY]["routed"]
    )
    assert new_out[TELEMETRY_KEY]["think_injected"] is False


def test_unknown_mode_raises_on_both(monkeypatch: pytest.MonkeyPatch, clean_process_state: None) -> None:
    """The remote client rejects an unknown mode; only the middleware defaults to the closed loop."""
    legacy_client = legacy_mw.CyberGateVLLMClient(
        base_url="https://example.invalid/v1", model_id=GEMMA_MODEL
    )
    new_client = CyberGateVLLMClient(base_url="https://example.invalid/v1", model_id=GEMMA_MODEL)
    monkeypatch.setattr(legacy_client, "_post_json", Recorder([ANSWER]))
    monkeypatch.setattr(new_client, "_post_json", Recorder([ANSWER]))

    kwargs = dict(messages=MESSAGES, mode="not_a_mode", force_gate=True)
    with pytest.raises(ValueError):
        legacy_client.chat_completions(**kwargs)
    with pytest.raises(ValueError):
        new_client.chat_completions(**kwargs)
