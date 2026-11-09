"""The category-probe forward hook: scores the user turn against the operator direction.

Runs in the **model-worker** process (`EngineCore`). Port of `gate_probe.install_vllm_read_hook`
lines 722-1050, split so the refusal monitor is its own module.

Read-only: `K = 0.0`, no weight edits, no logits touched. The hook observes activations, scores them,
and publishes a decision record. Everything that acts on the decision happens in the API-server
process.

Latency budget: `gpu_dot_us` 64-70 us, `hook_total_us` 85-240 us, with first-touch spikes to ~1100 us
while the direction tensor moves to the device. The hook body is one closure built at attach time; see
`_common` for why nothing may be added to that path.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..config import env_first
from ..families import ModelFamily, family_for_model_key
from ..ops import CyberGateOperator, load_cybergate_op
from ..ops.kit import check_model, enforce
from ..state import keys as key_builders
from ..state.backend import DecisionStore
from ..state.ipc_shm import ShmDecisionStore
from ..state.request_state import (
    LAST_GATE_DECISION,
    LIVE_GPU_PROMPT_CACHE,
    get_request_gate_state,
)
from ..telemetry.log import log_cybergate_event
from ..vllm_adapter import (
    extract_batch_slices,
    find_embed_tokens_module,
    find_transformer_layers,
    load_worker_tokenizer,
)
from ._common import (
    _starts_fresh_prompt,
    dynamo_safe,
    publish_decision,
    update_chunk_state_on_embed,
)
from ._scoring import cosine
from .bundle_telemetry import attach_bundle_telemetry_hooks

__all__ = ["CURRENT_USER_SPAN", "check_baseline_prefill_renders", "install_category_probe_hook"]

#: The single-turn conversation the render check below is measured on.
_RENDER_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]

#: Every `build_raw_prompt` mode that appends server-authored text after the user's turn. None of
#: these is an observation: `publish_decision` must drop all of them, or an injected `<think>`
#: block gets scored as if the user had written it.
_PASS2_MODES = (
    "preclosed",
    "preclosed_no_opener",
    "closed_loop_step1",
    "closed_loop_step1b",
    "closed_loop_step2",
    "closed_loop_cl2_step2",
    "sandwich_step1",
    "sandwich_step2",
)


def check_baseline_prefill_renders(family: ModelFamily, tok: Any) -> list[str]:
    """Check that `is_baseline_prefill` accepts the observation render and rejects the Pass-2 ones.

    `middleware._probe_observations` renders the observation itself, with
    `build_raw_prompt(mode="baseline")`, and dispatches it to `/v1/completions`. That -- not the
    checkpoint's `apply_chat_template` -- is the render this predicate has to accept, and it is
    what gets measured here. Getting it wrong fails **silently and closed**: `publish_decision`
    drops every prefill, so the gate scores nothing, writes no record, dumps no row and logs
    nothing, while the server answers normally. `check_model` cannot see it; the dimensions agree.

    Returns a list of problems, empty when the predicate separates the two correctly. Warn-only by
    design: it measures one single-turn render, so a mismatch is strong evidence rather than proof,
    and refusing to serve on it would be worse than saying so loudly.

    Three things are checked: the observation render publishes, all eight Pass-2 modes are
    rejected, and the checkpoint's own chat renders -- the answer-generation prefills that
    `middleware.__call__` forwards to `/v1/chat/completions` after the probe has published -- are
    rejected too. The last of those is not optional: gemma4 uses the base `suffix_len <=
    max_suffix` rule, whose first statement is `del u1_start`, and its chat render's suffix is the
    same length as its observation's, so nothing in the predicate separates them.

    **Bound.** One conversation, one turn, no system message. It does not cover multi-turn, tool
    turns or prefix-cached prefills, each of which takes a different branch of the predicate, and
    it cannot see whether the hook calls the predicate with these arguments -- that needs a live
    request and a dump row count. `notebooks/render_preflight.ipynb` is the same measurement with
    a table, runnable without a GPU before a serve.
    """
    encode = getattr(tok, "__call__", None)
    if encode is None or not hasattr(family, "build_raw_prompt"):
        return []

    def measure(text: str) -> tuple[int, int, str, bool] | None:
        """Locate and judge one rendered prompt. One encoder for every render, deliberately.

        Taking ids from `apply_chat_template(tokenize=True)` is not comparable: in current
        transformers it returns a `BatchEncoding`, and `list()` of that yields
        `['input_ids', 'attention_mask']`, which matches no markers and degrades to
        `full_sequence_fallback` -- a branch the predicate publishes unconditionally, so the shape
        bug reads as a real finding.
        """
        try:
            ids = list(tok(text, add_special_tokens=False).input_ids)
        except Exception:
            return None
        if not ids:
            return None
        spans, method = family.locate_all_user_spans(ids)
        if not spans:
            return None
        u1_start, suffix_len = spans[0][0], len(ids) - spans[-1][1]
        return (
            u1_start,
            suffix_len,
            method,
            bool(family.is_baseline_prefill(u1_start, suffix_len, method, _BASELINE_SUFFIX_MAX)),
        )

    def shape(mode: str) -> tuple[int, int, str, bool] | None:
        """One `build_raw_prompt` mode, rendered the way the probe renders it.

        `include_bos` must match `middleware._probe_observations`, which passes
        `family.supports_bos`. Rendering without it shifts `u1_start` by one for a BOS family --
        harmless while the predicate ignores `u1_start`, and wrong the moment one guards on it.
        """
        try:
            text = family.build_raw_prompt(
                _RENDER_PROBE_MESSAGES, mode=mode, include_bos=family.supports_bos
            )
        except Exception:
            return None
        return measure(text)

    def client_renders() -> list[tuple[str, tuple[int, int, str, bool]]]:
        """The checkpoint's own chat renders: the ANSWER prefills, which must all be dropped.

        `middleware.__call__` forwards the caller's untouched payload to `/v1/chat/completions`
        after the probe has already published, and vLLM renders that with the checkpoint's
        template. If one of those renders publishes, it overwrites the decision the gate just made
        with a score pooled under a different prefix, and appends a second unlabelled row to any
        running dump. This is the half a `build_raw_prompt`-only check cannot see.
        """
        render = getattr(tok, "apply_chat_template", None)
        if render is None:
            return []
        out: list[tuple[str, tuple[int, int, str, bool]]] = []
        for label, kwargs in (
            ("default", {}),
            ("enable_thinking=True", {"enable_thinking": True}),
            ("enable_thinking=False", {"enable_thinking": False}),
        ):
            try:
                text = render(
                    _RENDER_PROBE_MESSAGES,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except Exception:
                continue
            got = measure(text) if isinstance(text, str) else None
            if got is not None:
                out.append((label, got))
        return out

    problems: list[str] = []
    observed = shape("baseline")
    if observed is None:
        return []
    u1_start, suffix_len, method, publishes = observed
    if not publishes:
        problems.append(
            f"{family.name}: the OBSERVATION render is REJECTED by is_baseline_prefill "
            f"(u1_start={u1_start}, suffix_len={suffix_len}, span_method={method!r}); "
            f"publish_decision will drop every prefill and this kit will gate nothing. "
            f"baseline_suffix_lens={family.baseline_suffix_lens!r}"
        )

    for mode in _PASS2_MODES:
        got = shape(mode)
        if got is None or not got[3]:
            continue
        problems.append(
            f"{family.name}: Pass-2 render {mode!r} is ACCEPTED as an observation "
            f"(u1_start={got[0]}, suffix_len={got[1]}); it would overwrite the real decision with "
            f"one scored on server-authored text"
        )

    for label, (c_u1, c_suffix, c_method, c_publishes) in client_renders():
        if not c_publishes:
            continue
        same = " -- byte-identical in length to the observation's" if c_suffix == suffix_len else ""
        problems.append(
            f"{family.name}: the CLIENT chat render {label!r} is ACCEPTED as an observation "
            f"(u1_start={c_u1}, suffix_len={c_suffix}{same}, span_method={c_method!r}, against the "
            f"observation's u1_start={u1_start}, suffix_len={suffix_len}). It is the "
            f"answer-generation prefill: it will overwrite the decision the gate just published "
            f"and append a second unlabelled row to any running activation dump"
        )

    declared = tuple(getattr(family, "baseline_suffix_lens", ()) or ())
    if declared and suffix_len in declared:
        dead = [n for n in declared if n != suffix_len]
        if dead:
            problems.append(
                f"{family.name}: baseline_suffix_lens declares {dead} which this template's "
                f"observation render does not produce (it renders at {suffix_len}). Dead or "
                f"inherited values widen the predicate for nothing -- qwen38 carried Qwen3.6's 5 "
                f"this way. Confirm each against notebooks/render_preflight.ipynb or drop it."
            )
    return problems

#: Optional explicit `(start, length)` user span, set before prefill by single-process offline tests.
#: Ignored unless the batch holds exactly one sequence.
CURRENT_USER_SPAN: tuple[int, int] | None = None

#: A prefill whose suffix after the user turn is longer than this is a Pass-2 `<think>`-injected
#: prefill rather than a baseline chat prefill, and must not overwrite the baseline decision.
_BASELINE_SUFFIX_MAX = 12

#: Short sequences without chat markers are concurrent-decode batches, not prefills.
_DECODE_BATCH_MAX_TOKENS = 16


def _or_probe_fires(
    op: CyberGateOperator, state: dict[str, Any], bundle_scores: dict[str, float] | None
) -> dict[str, Any] | None:
    """Evaluate a multi-probe category factor (rule C) against the candidate-bundle telemetry.

    The baseline gate is one layer and one threshold. Rule C is an OR over several probes, so an
    operator may name extra probes in `meta.or_probes`; each one's score comes from the bundle
    telemetry hook and its threshold from that probe's own bundle entry, looked up by rule name.
    Nothing is duplicated into the operator file, so it cannot drift from the bundle.

    Returns `None` when the operator names no extra probes, which is every baseline operator.
    A probe whose score or threshold is absent cannot fire and is reported under `missing`, so an
    under-fire caused by a hook that did not run is visible on the record rather than silent.
    """
    names = op.meta.get("or_probes") or ()
    if not names:
        return None
    rule = op.meta.get("or_probes_thresh_rule", "R2")
    thresholds = state.get("bundle_thresholds", {})
    hits: list[str] = []
    missing: list[str] = []
    for name in names:
        value = (bundle_scores or {}).get(name)
        thresh = thresholds.get(name, {}).get(rule)
        if value is None or thresh is None:
            missing.append(name)
        elif float(value) >= float(thresh):
            hits.append(name)
    return {"rule": rule, "hits": hits, "missing": missing}


def install_category_probe_hook(
    model: Any,
    op_path: str | Path | None = None,
    ref_dir_path: str | Path | None = None,
    store: DecisionStore | None = None,
) -> CyberGateOperator:
    """Attach the embedding pre-hook, the category probe, and (by default) the refusal monitor.

    Returns the loaded operator. Idempotent: a second call on the same model object returns the
    operator without attaching anything again.
    """
    import torch  # type: ignore

    op = load_cybergate_op(op_path)
    if getattr(model, "_cybergate_hook_installed", False):
        return op

    decision_store = store if store is not None else ShmDecisionStore()
    family = family_for_model_key(op.model_key)
    layers = find_transformer_layers(model)
    model._cybergate_hook_installed = True

    c_cpu = torch.tensor(op.c, dtype=torch.float32)
    c_last_raw = op.meta.get("c_last")
    c_last_cpu = torch.tensor(c_last_raw, dtype=torch.float32) if c_last_raw else None
    worker_tok = load_worker_tokenizer(family)

    # Mutable state shared between the embedding pre-hook, this hook and the refusal monitor. Built
    # once; the hook bodies only read and write its slots.
    state: dict[str, Any] = {
        "c_dev": None,
        "c_last_dev": None,
        "prefill_count": 0,
        "last_input_ids": None,
        "pending_prefill_decisions": {},
        "multi_layer_slice_scores": {},
    }

    LAST_GATE_DECISION["threshold"] = op.thresh
    LAST_GATE_DECISION["layer"] = op.gate_layer

    _attach_embed_pre_hook(model, state, family)
    attach_bundle_telemetry_hooks(layers, family, state)

    # Kit binding against the model that is actually loaded. `check_operator` already compared the
    # operator against its family's declaration from metadata alone; this is the half that needs the
    # real checkpoint, and it is the only place the `min(...)` clamp below can be caught before it
    # silently scores a layer nobody asked for.
    model_problems = check_model(op, family, len(layers))
    if model_problems:
        log_cybergate_event("CYBERGATE-KIT-MISMATCH", "; ".join(model_problems))
    enforce(model_problems, f"model {type(model).__name__}")

    # Warn-only, and not covered by `enforce`: this reads one single-turn render, so a mismatch is
    # strong evidence rather than proof. A kit whose observation render is rejected serves
    # perfectly and gates nothing, which `check_model` above cannot see -- the dimensions agree.
    render_problems = check_baseline_prefill_renders(family, worker_tok)
    if render_problems:
        log_cybergate_event("CYBERGATE-RENDER-MISMATCH", "; ".join(render_problems))

    target_layer = layers[min(op.gate_layer, len(layers) - 1)]
    default_pre = family.default_pre
    default_suf = family.default_suf

    def _read_only_impl(_module: Any, _inputs: Any, output: Any) -> Any:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return output
        h = output[0] if isinstance(output, tuple) else output
        chunk_mode = state.get("chunk_mode", "none")
        if h is None or (h.shape[0] <= 1 and chunk_mode not in ("accumulate", "complete")):
            return output

        t_hook0 = time.perf_counter()
        if state["c_dev"] is None or state["c_dev"].device != h.device:
            state["c_dev"] = c_cpu.to(device=h.device, dtype=torch.float32)
            if c_last_cpu is not None:
                state["c_last_dev"] = c_last_cpu.to(device=h.device, dtype=torch.float32)

        h_2d = h.reshape(-1, h.shape[-1]) if h.ndim == 3 else h
        n_tok = int(h_2d.shape[0])

        if chunk_mode == "start":
            state.setdefault("chunk_h_buf", {})["cat"] = h_2d.detach().clone()
            state["chunk_ids_buf"] = state.get("chunk_full_ids")
            return output
        if chunk_mode == "accumulate":
            prev_h = state.setdefault("chunk_h_buf", {}).get("cat")
            state["chunk_h_buf"]["cat"] = (
                torch.cat([prev_h.to(h_2d.device), h_2d.detach()], dim=0).clone()
                if prev_h is not None
                else h_2d.detach().clone()
            )
            state["chunk_ids_buf"] = state.get("chunk_full_ids")
            return output

        chunk_prev_h = (
            state.get("chunk_h_buf", {}).pop("cat", None)
            if chunk_mode == "complete"
            else None
        )
        chunk_full_ids = (
            state.get("chunk_full_ids") if chunk_mode == "complete" else None
        )
        if chunk_mode == "complete":
            state["chunk_ids_buf"] = None

        input_ids_1d: list[int] | None = None
        raw_ids = state.get("last_input_ids")
        if raw_ids is not None and raw_ids.numel() == n_tok:
            input_ids_1d = raw_ids.reshape(-1).cpu().tolist()

        if chunk_prev_h is not None and chunk_full_ids is not None:
            batch_slices = [(None, 0, n_tok)]
        else:
            batch_slices = extract_batch_slices(n_tok, input_ids_1d, family)

        for req_id, q_start, q_end in batch_slices:
            if chunk_prev_h is not None and chunk_full_ids is not None:
                h_seq = torch.cat([chunk_prev_h.to(h_2d.device), h_2d], dim=0)
                seq_ids = list(chunk_full_ids)
                seq_len = len(seq_ids)
            else:
                seq_len = q_end - q_start
                if seq_len <= 1:
                    continue
                h_seq = h_2d[q_start:q_end]
                seq_ids = input_ids_1d[q_start:q_end] if input_ids_1d is not None else None

            located = _locate_spans(
                seq_ids, seq_len, family, default_pre, default_suf,
                single_sequence_batch=len(batch_slices) == 1,
            )
            if located is None:
                continue
            user_spans, span_method = located
            u_start, u_end = user_spans[-1]
            h_user = h_seq[u_start:u_end] if (0 <= u_start < u_end) else h_seq

            t_dot0 = time.perf_counter()
            s_cat_mean = cosine(h_user.float().mean(dim=0), state["c_dev"])

            u1_start, u1_end = user_spans[0]
            if len(user_spans) > 1 and 0 <= u1_start < u1_end:
                s_cat_first = cosine(
                    h_seq[u1_start:u1_end].float().mean(dim=0), state["c_dev"]
                )
                all_user_indices: list[int] = []
                for us_i, ue_i in user_spans:
                    if 0 <= us_i < ue_i <= seq_len:
                        all_user_indices.extend(range(us_i, ue_i))
                s_cat_conv = (
                    cosine(h_seq[all_user_indices].float().mean(dim=0), state["c_dev"])
                    if all_user_indices
                    else s_cat_mean
                )
            else:
                s_cat_first = s_cat_mean
                s_cat_conv = s_cat_mean

            last_dir = state["c_last_dev"] if state["c_last_dev"] is not None else state["c_dev"]
            s_cat_last = cosine(h_seq[-1].float(), last_dir)

            score = s_cat_last if op.pool == "last" else s_cat_mean
            gpu_dot_us = (time.perf_counter() - t_dot0) * 1e6
            hook_total_us = (time.perf_counter() - t_hook0) * 1e6
            bundle_scores = state.get("multi_layer_slice_scores", {}).get((q_start, q_end))
            or_info = _or_probe_fires(op, state, bundle_scores)
            fired = bool(score >= op.thresh) or bool(or_info and or_info["hits"])

            user_sha256, conv_sha256 = _hash_user_turns(
                worker_tok, seq_ids, user_spans, seq_len, u_start, u_end
            )

            suffix_len = seq_len - u_end
            u1_start = user_spans[0][0] if user_spans else u_start
            is_baseline_prefill = family.is_baseline_prefill(
                u1_start, suffix_len, span_method, _BASELINE_SUFFIX_MAX
            )
            is_multiturn_ctx = family.looks_like_in_context_prefill(user_spans, seq_ids)
            is_single_turn_with_sys = bool(
                len(user_spans) == 1
                and seq_ids is not None
                and _starts_fresh_prompt(seq_ids, family)
                and seq_ids[:u_start].count(family.turn_end_id) <= 1
            )

            decision: dict[str, Any] = {
                "ts": time.time(),
                "pid": os.getpid(),
                "req_id": req_id,
                "user_sha256": user_sha256,
                "conv_sha256": conv_sha256 or user_sha256,
                "num_user_turns": len(user_spans),
                "is_multiturn_ctx": is_multiturn_ctx,
                "is_single_turn_with_sys": is_single_turn_with_sys,
                "n_prefill_tokens": seq_len,
                "user_span": [u_start, u_end],
                "all_user_spans": [list(sp) for sp in user_spans],
                "suf_tokens": suffix_len,
                "is_baseline_prefill": is_baseline_prefill,
                "span_method": span_method,
                "decided": True,
                "fired": fired,
                "score": round(score, 6),
                "s_cat_mean_user": round(s_cat_mean, 6),
                "s_cat_conv": round(s_cat_conv, 6),
                "s_cat_first": round(s_cat_first, 6),
                "s_cat_last": round(s_cat_last, 6),
                "threshold": op.thresh,
                "layer": op.gate_layer,
                "pool": op.pool,
                "gpu_dot_us": round(gpu_dot_us, 2),
                "hook_total_us": round(hook_total_us, 2),
                "source": (
                    f"live_vllm_gpu_L{op.gate_layer}_{op.pool}_hook"
                    f"(tokens={seq_len},turns={len(user_spans)},span={u_start}:{u_end},"
                    f"gpu_dot_us={gpu_dot_us:.1f})"
                ),
            }
            if bundle_scores:
                decision["multi_layer_scores"] = dict(bundle_scores)
            if or_info is not None:
                decision["or_probes"] = or_info

            # Handed to the refusal monitor, which attaches `s_ref_z` and then publishes. Published
            # here too, because the monitor may be disabled.
            state["pending_prefill_decisions"][(q_start, q_end)] = decision
            LAST_GATE_DECISION.update(decision)
            if req_id:
                get_request_gate_state(req_id).update(decision)

            state["prefill_count"] += 1
            if conv_sha256 or user_sha256:
                publish_decision(decision, decision_store, LIVE_GPU_PROMPT_CACHE)

            log_cybergate_event(
                f"CYBERGATE-L{op.gate_layer}-GPU",
                f"req_id={req_id} conv_sha={(conv_sha256 or user_sha256 or 'n/a')[:10]} "
                f"user_sha={(user_sha256 or 'n/a')[:10]} turns={len(user_spans)} "
                f"prefill_tokens={seq_len} user_span=[{u_start}:{u_end}] suf={suffix_len} "
                f"baseline_prefill={is_baseline_prefill} "
                f"-> s_cat={score:+.6f} (conv={s_cat_conv:+.6f}, last={s_cat_last:+.6f}, "
                f"thresh={op.thresh:+.6f}) -> FIRED={fired} gpu_dot_us={gpu_dot_us:.2f}us",
            )
        return output

    target_layer.register_forward_hook(dynamo_safe(_read_only_impl))
    log_cybergate_event(
        f"CYBERGATE-L{op.gate_layer}-HOOK",
        f"Attached read-only Layer-{op.gate_layer} ({op.pool}) M1_ridge_lam16 hook "
        f"(dim={op.dim}, thresh={op.thresh:.8f}, total_layers={len(layers)}, K=0.0) "
        f"on {type(model).__name__}",
    )

    ref_env = ref_dir_path or env_first("CYBERGATE_REF_DIR_PATH")
    monitor_enabled = (
        env_first("CYBERGATE_ENABLE_REF_MONITOR", default="1") == "1"
    )
    if ref_env is not None or monitor_enabled:
        from .refusal_monitor import install_refusal_monitor_hook

        install_refusal_monitor_hook(
            model,
            ref_dir_path=ref_env,
            family=family,
            shared_hook_state=state,
            store=decision_store,
        )

    return op




def _locate_spans(
    seq_ids: list[int] | None,
    seq_len: int,
    family: ModelFamily,
    default_pre: int,
    default_suf: int,
    *,
    single_sequence_batch: bool,
) -> tuple[list[tuple[int, int]], str] | None:
    """Resolve user-turn spans for one sequence, or `None` to skip the sequence entirely.

    Skipping happens for short sequences with no chat markers, which are concurrent-decode batches
    rather than prefills -- scoring those would publish a decision for a sequence that has no user
    turn in it.
    """
    if CURRENT_USER_SPAN is not None and single_sequence_batch:
        start, length = CURRENT_USER_SPAN
        return [(start, min(seq_len, start + length))], "explicit_CURRENT_USER_SPAN"

    if seq_ids is not None:
        user_spans, span_method = family.locate_all_user_spans(seq_ids)
        if not span_method.startswith("exact_token_markers") and seq_len <= _DECODE_BATCH_MAX_TOKENS:
            return None
        return user_spans, span_method

    if seq_len > (default_pre + default_suf + 1):
        return (
            [(default_pre, seq_len - default_suf)],
            f"default_offsets({default_pre}:-{default_suf})",
        )
    if seq_len <= _DECODE_BATCH_MAX_TOKENS:
        return None
    return [(0, seq_len)], "full_sequence_fallback"


def _hash_user_turns(
    worker_tok: Any,
    seq_ids: list[int] | None,
    user_spans: list[tuple[int, int]],
    seq_len: int,
    u_start: int,
    u_end: int,
) -> tuple[str | None, str | None]:
    """Decode the user turns and hash them into `(user_sha256, conv_sha256)`.

    Returns `(None, None)` without a tokenizer or without usable spans, which means no decision record
    can be keyed and the serving path will abstain.
    """
    if worker_tok is None or seq_ids is None or not (0 <= u_start < u_end):
        return None, None
    try:
        decoded: list[str] = []
        for us_i, ue_i in user_spans:
            if 0 <= us_i < ue_i <= seq_len:
                text = worker_tok.decode(seq_ids[us_i:ue_i], skip_special_tokens=False).strip()
                if text:
                    decoded.append(text)
        if not decoded:
            return None, None
        return key_builders.text_sha256(decoded[-1]), key_builders.conversation_key(decoded)
    except Exception:
        return None, None


def _attach_embed_pre_hook(model: Any, state: dict[str, Any], family: ModelFamily) -> None:
    """Capture each forward batch's exact `input_ids`, and reset the per-batch slots."""
    import torch  # type: ignore

    embed_mod = find_embed_tokens_module(model)
    if embed_mod is None:
        return

    def _embed_pre_hook(_mod: Any, args: tuple[Any, ...]) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if args and hasattr(args[0], "reshape"):
            raw_ids = args[0]
            if raw_ids.numel() <= 1 and state.get("chunk_ids_buf") is None:
                state["last_input_ids"] = None
                if state.get("pending_prefill_decisions"):
                    state["pending_prefill_decisions"] = {}
                if state.get("multi_layer_slice_scores"):
                    state["multi_layer_slice_scores"] = {}
                state["chunk_mode"] = "none"
                return
            state["last_input_ids"] = raw_ids.detach()
            state["pending_prefill_decisions"] = {}
            state["multi_layer_slice_scores"] = {}
            update_chunk_state_on_embed(state, family)

    embed_mod.register_forward_pre_hook(dynamo_safe(_embed_pre_hook))
    log_cybergate_event(
        "CYBERGATE-EMBED-HOOK",
        f"Attached input_ids pre-hook on {type(embed_mod).__name__} "
        "for exact multi-turn span extraction",
    )
