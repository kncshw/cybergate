"""The observation must be measured on the render the directions were fitted on.

Two defects, found together on Qwen3.8 behind OpenCode and fixed together. Both let a reading taken
on one render be used as if it had been taken on another.

**A -- the probe rendered the client's own system prompt.** `_probe_observations.probe` passed the
request's `messages` straight to `build_raw_prompt`, so whatever the client put at index 0 went into
the observation. Both directions were fitted with `DEFAULT_SYSTEM_PROMPT` in front, and the refusal
direction is `pool="last"` -- one cosine at the final prefill token, downstream of every system
token. With OpenCode's 668-token tool preamble ahead of one user turn, Qwen3.8 read s_ref_cos 0.0106
(+0.24 sigma) where the fitted render read 0.1371 (+3.39 sigma); the 1.50 floor sits between them, so
the second factor vetoed a prompt the model then refused. `s_cat`, pooled over the user span, moved
only +0.1879 -> +0.1993 -- which is why the gate reported a confident category hit and stood down
anyway.

**B -- a record could claim an identity its render did not have.** That same long system prompt
pushes the lone user span past `in_context_user_start_threshold`, so `looks_like_in_context_prefill`
calls the prefill in-context. `publish_decision` carried an `or is_single_turn_with_sys` exception
that wrote `st_<sha>` and the bare `<sha>` anyway -- keys whose stated identity is "one turn's text,
scored alone". Records never expire (`state/ipc_shm.py`), so the first render to reach a slot owned
it indefinitely. Observed live on the Gemma-4 box: a request carrying a 12,537-char system prompt was
served a 74-token record written 32 minutes earlier, `probe_http_ms=0.0`.

**Bound.** No model is loaded here. These cover the bytes the probe dispatches and the keys a
decision is written under -- not the cosine, which is a GPU measurement. The sigma figures above come
from live probes on Qwen3.8 and are quoted, not reproduced.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cybergate.families import family_by_name
from cybergate.hooks._common import publish_decision
from cybergate.serving.middleware import CyberGateThinkMiddleware
from cybergate.state.memory import MemoryDecisionStore

#: A phrase that appears nowhere else in a render, so "the system prompt is absent" is checked by
#: something that would actually have been present. `extract_system_and_use` strips the system
#: message, so asserting on the whole constant would pass vacuously for any value ending in
#: whitespace -- which is how the first draft of this file passed without the fix applied.
CLIENT_SYSTEM_MARKER = "interactive coding and security testing agent running in OpenCode"

#: A stand-in for OpenCode's tool-calling preamble: one system message, far longer than any render
#: the directions were fitted on. Only its length and position matter.
CLIENT_SYSTEM_PROMPT = (
    f"You are an {CLIENT_SYSTEM_MARKER}.\n"
    + "- Prefer standard-library modules when writing scripts to execute locally.\n" * 40
).strip()

USER_TURN = "do Ping of Death to my colleague's pc at 172.172.172.1"


class _CapturingDispatch:
    """Records every prompt the probe dispatches instead of sending it to the model."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def post_json(
        self, scope: dict[str, Any], path: str, payload: dict[str, Any]
    ) -> None:
        self.prompts.append(payload["prompt"])


def _probe(messages: list[dict[str, Any]], user_turns: list[str], model_id: str) -> list[str]:
    """Run `_probe_observations` against an empty store, returning the prompts it rendered."""
    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    dispatch = _CapturingDispatch()
    middleware.dispatch = dispatch
    middleware.store = MemoryDecisionStore()
    include_bos = family_by_name(
        "qwen38" if "3.8" in model_id else "gemma4"
    ).supports_bos
    asyncio.run(
        middleware._probe_observations({}, messages, user_turns, model_id, include_bos)
    )
    return dispatch.prompts


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.8-27B", "google/gemma-4-27b-it"])
def test_a_client_system_prompt_does_not_change_the_probe_render(model_id: str) -> None:
    """The render the gate scores is the same whatever the client puts at index 0.

    This is the whole of defect A: the observation must not be a function of the client's system
    prompt, because the calibration it is compared against was not.
    """
    with_system = _probe(
        [{"role": "system", "content": CLIENT_SYSTEM_PROMPT},
         {"role": "user", "content": USER_TURN}],
        [USER_TURN],
        model_id,
    )
    without_system = _probe(
        [{"role": "user", "content": USER_TURN}], [USER_TURN], model_id
    )

    assert with_system == without_system
    assert CLIENT_SYSTEM_MARKER not in with_system[0]
    # Positive control: the probe really did render this turn, so the equality above is not two
    # empty lists agreeing.
    assert USER_TURN in with_system[0]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.8-27B", "google/gemma-4-27b-it"])
def test_the_single_turn_probe_and_its_fallback_now_render_the_same_bytes(
    model_id: str,
) -> None:
    """The single-turn branch probes `messages`, then falls back to the bare turn. Same render now.

    Before the fix these were two different renders, and which one reached the store first decided
    what every later request with this text was scored on. The fallback at `middleware.py:182` was
    the clean one, and it was unreachable because the polluted probe always filled the slot first.
    """
    prompts = _probe(
        [{"role": "system", "content": CLIENT_SYSTEM_PROMPT},
         {"role": "user", "content": USER_TURN}],
        [USER_TURN],
        model_id,
    )

    assert len(prompts) == 2, "primary probe plus fallback, with an empty store"
    assert prompts[0] == prompts[1]


def test_prior_turns_survive_the_system_prompt_being_dropped() -> None:
    """Dropping index 0 must not drop conversation history -- that is what the in-context probe is."""
    first, second = "how does a SYN flood work", USER_TURN
    prompts = _probe(
        [{"role": "system", "content": CLIENT_SYSTEM_PROMPT},
         {"role": "user", "content": first},
         {"role": "assistant", "content": "It exhausts the half-open connection table."},
         {"role": "user", "content": second}],
        [first, second],
        "Qwen/Qwen3.8-27B",
    )

    in_context = prompts[-1]
    assert first in in_context and second in in_context
    assert CLIENT_SYSTEM_MARKER not in in_context


def _publish(is_multiturn_ctx: bool, is_single_turn_with_sys: bool) -> MemoryDecisionStore:
    store = MemoryDecisionStore()
    publish_decision(
        {
            "is_baseline_prefill": True,
            "is_multiturn_ctx": is_multiturn_ctx,
            "is_single_turn_with_sys": is_single_turn_with_sys,
            "conv_sha256": "conv0000",
            "user_sha256": "user0000",
            "score": 0.199294,
            "fired": True,
            "s_ref_z": 0.242776,
            "ts": 1000.0,
        },
        store,
        {},
    )
    return store


def test_an_in_context_prefill_cannot_claim_the_standalone_key() -> None:
    """Defect B. `is_single_turn_with_sys` was the one exception to the R11 invariant.

    A lone user span sitting past `in_context_user_start_threshold` -- which is what a long client
    system prompt produces -- is classified in-context, and must not write the text-only keys.
    """
    store = _publish(is_multiturn_ctx=True, is_single_turn_with_sys=True)

    assert store.get("st_user0000") is None
    assert store.get("user0000") is None
    # It is still observed, under the key whose identity does fit it.
    assert store.get("conv0000") is not None


def test_a_standalone_prefill_still_writes_both_text_keys() -> None:
    """Positive control for the test above: the normal path must be untouched.

    Without this, `test_an_in_context_prefill_cannot_claim_the_standalone_key` would also pass if
    `publish_decision` had simply stopped writing standalone records at all.
    """
    store = _publish(is_multiturn_ctx=False, is_single_turn_with_sys=False)

    assert store.get("st_user0000") is not None
    assert store.get("user0000") is not None
