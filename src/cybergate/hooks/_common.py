"""Shared hook plumbing. Everything here runs **once at attach time**, never per forward pass.

The forward path is latency-critical: measured `gpu_dot_us` 64-70 us and `hook_total_us` 85-240 us on
the live server, and an earlier revision with a 34 ms median was treated as a defect. So the hook
bodies stay tight closures created once at attach, and this module holds only the setup helpers and
the one post-decision publish step.

Do not add per-forward dispatch, per-forward object construction, abstract base classes or registry
lookups to anything called from a hook body. Modularise *around* the hook, not inside it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..state import keys as key_builders
from ..state.backend import DecisionStore

__all__ = ["dynamo_safe", "publish_decision", "update_chunk_state_on_embed"]


def _starts_fresh_prompt(seq_ids: Sequence[int], family: Any) -> bool:
    """True when `seq_ids` begins at token 0 of a fresh prompt rather than a continuation chunk."""
    if not seq_ids:
        return False
    bos_id = getattr(family, "bos_token_id", None)
    if bos_id is not None and seq_ids[0] == bos_id:
        return True
    # The family's own fresh-prompt marker. This was the literal pair `(248045, 8948)` -- Qwen-3.6's
    # `<|im_start|>system` -- applied to every family regardless of architecture.
    marker = getattr(family, "fresh_prompt_marker", ()) or ()
    if marker and len(seq_ids) >= len(marker):
        if all(seq_ids[i] == marker[i] for i in range(len(marker))):
            return True
    return False


def _has_any_user_marker(seq_ids: Sequence[int], family: Any) -> tuple[bool, int]:
    """Return `(found, last_u_start)` for `family.user_turn_marker` in `seq_ids`."""
    marker = getattr(family, "user_turn_marker", ())
    if not marker or len(seq_ids) < len(marker):
        return False, -1
    m_len = len(marker)
    last_u_start = -1
    for i in range(len(seq_ids) - m_len, -1, -1):
        if all(seq_ids[i + k] == marker[k] for k in range(m_len)):
            last_u_start = i + m_len
            break
    return (last_u_start >= 0), last_u_start


def _has_unclosed_user_turn(seq_ids: Sequence[int], family: Any) -> bool:
    """True when `seq_ids` opens a user turn whose closing `turn_end_id` has not arrived yet."""
    found, last_u_start = _has_any_user_marker(seq_ids, family)
    if not found:
        return False
    turn_end_id = getattr(family, "turn_end_id", -1)
    for j in range(last_u_start, len(seq_ids)):
        if seq_ids[j] == turn_end_id:
            return False
    return True


def update_chunk_state_on_embed(state: dict[str, Any], family: Any) -> None:
    """Detect and reassemble multi-chunk prefills across consecutive forward passes."""
    raw_ids = state.get("last_input_ids")
    if raw_ids is None:
        state["chunk_mode"] = "none"
        state["chunk_full_ids"] = None
        return

    n_tok = int(raw_ids.numel())
    prev_buf: list[int] | None = state.get("chunk_ids_buf")
    prev_ts = float(state.get("chunk_ts", 0.0))
    now = time.time()

    if prev_buf is not None and (now - prev_ts) > 5.0:
        prev_buf = None
        state["chunk_ids_buf"] = None
        state["chunk_h_buf"] = {}

    if n_tok <= 1 and prev_buf is None:
        state["chunk_mode"] = "none"
        state["chunk_full_ids"] = None
        return

    ids_1d = raw_ids.reshape(-1).cpu().tolist()

    if prev_buf is not None and not _starts_fresh_prompt(ids_1d, family):
        combined_ids = prev_buf + ids_1d
        if _has_unclosed_user_turn(combined_ids, family):
            state["chunk_mode"] = "accumulate"
            state["chunk_full_ids"] = combined_ids
            state["chunk_ts"] = now
            return
        spans, span_method = family.locate_all_user_spans(combined_ids)
        u_start, u_end = spans[-1]
        u1_start = spans[0][0] if spans else u_start
        suf_len = len(combined_ids) - u_end
        if (
            u_end >= len(prev_buf) - 4
            and 1 <= suf_len <= 12
            and family.is_baseline_prefill(u1_start, suf_len, span_method, 12)
        ):
            state["chunk_mode"] = "complete"
            state["chunk_full_ids"] = combined_ids
            return
        # Not a continuation of prev_buf; reset and inspect ids_1d fresh.
        state["chunk_ids_buf"] = None
        state["chunk_h_buf"] = {}

    if n_tok < 256:
        state["chunk_mode"] = "none"
        state["chunk_full_ids"] = None
        return

    if _has_unclosed_user_turn(ids_1d, family):
        state["chunk_mode"] = "start"
        state["chunk_full_ids"] = ids_1d
        state["chunk_ts"] = now
        state["chunk_h_buf"] = {}
        return

    has_marker, _ = _has_any_user_marker(ids_1d, family)
    if n_tok >= 512 and has_marker:
        spans, _ = family.locate_all_user_spans(ids_1d)
        u_end = spans[-1][1]
        if (len(ids_1d) - u_end) > 12:
            state["chunk_mode"] = "start"
            state["chunk_full_ids"] = ids_1d
            state["chunk_ts"] = now
            state["chunk_h_buf"] = {}
            return

    state["chunk_mode"] = "none"
    state["chunk_full_ids"] = None


def dynamo_safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap `fn` so torch.compile does not try to trace it.

    A hook that gets traced either graph-breaks noisily or, worse, gets specialised on the first
    batch shape. The canonical package guards the attribute lookup because `torch._dynamo` is absent
    from some builds.
    """
    import torch  # type: ignore

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable(fn)
    return fn


def publish_decision(
    decision: dict[str, Any],
    store: DecisionStore,
    cache: dict[str, dict[str, Any]],
) -> None:
    """Write a completed prefill decision to every key that should hold it.

    Called once per observed sequence, after scoring. Writes to both the in-process cache (for the
    worker's own later lookups) and the cross-process store (for the API-server process).

    Which keys get written depends on what was observed:

    * a **standalone** 1-turn prefill lands under `st_<sha>`, the bare `<sha>`, and -- since a 1-turn
      conversation hashes to the same value -- the conversation key, which is the same slot,
    * a **multi-turn in-context** prefill lands under the conversation key, and *only* there. The
      canonical code also wrote `ctx_<sha(final turn)>`, which carries no conversation component;
      that was R11 and it is now gone. See `state.keys.legacy_in_context_key_from_sha`.

    Only baseline (Pass-1) prefills are published. A Pass-2 `<think>`-injected prefill re-observes the
    same conversation with injected text appended, and publishing it would overwrite the real
    decision with one scored on server-authored content.
    """
    if not decision.get("is_baseline_prefill", True):
        return

    conv_sha = decision.get("conv_sha256")
    user_sha = decision.get("user_sha256")
    is_multiturn_ctx = bool(decision.get("is_multiturn_ctx", False))

    key_sha = conv_sha or user_sha
    if not key_sha:
        return

    # For a standalone turn the conversation key and the bare turn key are the same string, so the
    # guard stops writing it twice. For a multi-turn prefill they differ and both are written.
    if not is_multiturn_ctx or key_sha != user_sha:
        cache[key_sha] = decision
        store.put(key_sha, decision)

    if not user_sha:
        return

    # A multi-turn in-context prefill needs no second key: it was already written above under the
    # conversation key, which is the only key whose identity is correct for it. The canonical code
    # also wrote `ctx_<sha(final turn)>` here, and that key is R11 -- it carries no conversation
    # component, so two conversations ending in the same user text shared one decision.
    #
    # There used to be an `or is_single_turn_with_sys` exception here, and it was the one way a
    # prefill the family had classified in-context could still claim the text-only keys. A long
    # client system prompt pushes the lone user span past `in_context_user_start_threshold`, so
    # `looks_like_in_context_prefill` calls it in-context (signal 2); the exception then overrode
    # that and wrote `st_<sha>` anyway. The record landing there was measured with that system
    # prompt in front, while the key asserts "this turn's text, scored alone" -- and records never
    # expire, so the first render to arrive owned the slot indefinitely. Observed live: a request
    # carrying a 12,537-char system prompt served a 74-token record written 32 minutes earlier, with
    # `probe_http_ms=0.0`.
    #
    # Dropping the exception costs nothing, because the miss self-heals through a path that already
    # exists: with no record under the text keys, `middleware._probe_observations`'s single-turn
    # lookup misses and falls through to `probe([{"role": "user", "content": user_turns[-1]}])`,
    # which renders without the client system prompt and is classified standalone. That fallback was
    # unreachable before precisely because this exception kept filling the slot first.
    if not is_multiturn_ctx:
        st_key = key_builders.standalone_turn_key_from_sha(user_sha)
        cache[st_key] = decision
        store.put(st_key, decision)
        cache[user_sha] = decision
        store.put(user_sha, decision)
    if is_multiturn_ctx:
        pc_key = f"pc_ctx_{user_sha}"
        cache[pc_key] = decision
        store.put(pc_key, decision)
        cache["last_multiturn_ctx"] = decision
        store.put("last_multiturn_ctx", decision)

