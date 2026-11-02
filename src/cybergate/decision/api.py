"""The legacy decision API: `evaluate_gate_for_prompt` and `evaluate_gate_for_messages`.

These two functions keep the exact signatures and dict-returning shape the canonical package
exposes, so the middleware, the client, `cli.py` and the producer's own scripts keep working. They
are thin: resolve evidence, call the pure policy, render the payload.

`vllm_base_url`, `vllm_api_key` and `verify_ssl` are accepted and ignored, exactly as in the
canonical package -- the gate never calls the completions endpoint itself; the middleware does the
sub-probing and passes the resulting records in.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..config import DEFAULT_MODEL_ID
from ..families import family_for_model_id
from ..messages import extract_user_turns
from ..state.ipc_shm import ShmDecisionStore
from .evidence import EvidenceGatherer, legacy_op_for_model_id
from .outcome import render_payload
from .policy import decide

__all__ = ["evaluate_gate_for_messages", "evaluate_gate_for_prompt", "make_gatherer"]


def make_gatherer(model_id: str, *, gate_url: str | None = None) -> EvidenceGatherer:
    """Build an `EvidenceGatherer` backed by the real `/dev/shm` store.

    The operator comes from `legacy_op_for_model_id`, which reproduces defect D1 (every non-Gemma
    model id loads the Qwen operator). The *family* is resolved correctly from the model id, so the
    revocation ceiling and hysteresis floor are the family's own.
    """
    op, family_tag = legacy_op_for_model_id(model_id)
    return EvidenceGatherer(
        store=ShmDecisionStore(),
        op=op,
        family=family_for_model_id(model_id),
        family_tag=family_tag,
        gate_url=gate_url,
    )


def evaluate_gate_for_prompt(
    user_text: str,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_key: str | None = None,
    gate_http_url: str | None = None,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    verify_ssl: bool = False,
    request_id: str | None = None,
    live_gpu_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the gate for a single `user_text`.

    Returns the resolved decision record. The two-factor gate applies to live, cached and
    cross-process observations; an HTTP `/gate` or offline precomputed record carries its own `fired`.
    """
    del vllm_base_url, vllm_api_key, verify_ssl  # accepted for compatibility; unused
    gatherer = make_gatherer(model_id, gate_url=gate_http_url)
    return gatherer.resolve_prompt(
        user_text,
        prompt_key=prompt_key,
        live_record=live_gpu_decision,
        request_id=request_id,
    )


def evaluate_gate_for_messages(
    messages: Sequence[dict[str, Any]],
    model_id: str = DEFAULT_MODEL_ID,
    prompt_key: str | None = None,
    gate_http_url: str | None = None,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    verify_ssl: bool = False,
    request_id: str | None = None,
    enable_multi_turn_stickiness: bool = True,
    live_gpu_decision: dict[str, Any] | None = None,
    standalone_user_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the multi-turn gate for a conversation.

    With no user turns, falls back to evaluating the empty prompt, as the canonical code does.
    """
    del vllm_base_url, vllm_api_key, verify_ssl  # accepted for compatibility; unused

    if not extract_user_turns(messages):
        return evaluate_gate_for_prompt(
            "",
            model_id=model_id,
            prompt_key=prompt_key,
            gate_http_url=gate_http_url,
            request_id=request_id,
        )

    gatherer = make_gatherer(model_id, gate_url=gate_http_url)
    inputs = gatherer.gather(
        messages,
        live_gpu_decision=live_gpu_decision,
        standalone_user_decision=standalone_user_decision,
        enable_multi_turn_stickiness=enable_multi_turn_stickiness,
        prompt_key=prompt_key,
        request_id=request_id,
    )
    return render_payload(inputs, decide(inputs))
