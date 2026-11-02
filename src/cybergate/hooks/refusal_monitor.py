"""The refusal-monitor forward hook: the gate's second factor.

Runs in the **model-worker** process. Port of `gate_probe.install_vllm_refusal_monitor_hook`.

Measures how close the model is to refusing, as a cosine against a calibrated refusal direction at a
mid-stack layer (L21 on Gemma-4), expressed in sigma above the harmless mean:

    s_ref_z = (s_ref_cos - mu_hm) / sd_hm

It does two different jobs depending on phase:

* **at prefill end** -- attaches `s_ref_z` to the category probe's pending decision record and
  publishes it, so the serving path can apply the two-factor gate and multi-turn maintenance;
* **during decode** -- tracks a 4-step rolling mean for the closed-loop CoT controller.

Read-only: `K = 0.0`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..families import (
    DEFAULT_FAMILY,
    MidLateHits,
    ModelFamily,
    family_by_name,
    family_for_configured_operator,
)
from ..ops import load_cybergate_op, resolve_ref_dir_path
from ..state.backend import DecisionStore
from ..state.ipc_shm import ShmDecisionStore
from ..state.request_state import LAST_GATE_DECISION, LIVE_GPU_PROMPT_CACHE, get_request_gate_state
from .steering_config import SteeringConfig
from ..telemetry.log import log_cybergate_event
from ..vllm_adapter import extract_batch_slices, find_transformer_layers
from . import refusal_trace
from ._common import dynamo_safe, publish_decision
from ._scoring import cosine

__all__ = ["install_refusal_monitor_hook"]

#: Rolling window, in decode steps, for the closed-loop controller's `rolling_s_ref_z`.
_ROLLING_WINDOW = 4

#: Rolling windows, one per bound prompt key. Keyed on identity rather than on batch row, so a
#: request keeps its own window when vLLM compacts the batch around it. Bounded because a long-lived
#: engine would otherwise accumulate one list per prompt it has ever served.
_WINDOWS: dict[str, list[float]] = {}
_MAX_WINDOWS = 4096


def _rolling_window(key: str) -> list[float]:
    """The rolling `s_ref_z` buffer belonging to `key`, created on first use."""
    window = _WINDOWS.get(key)
    if window is None:
        if len(_WINDOWS) >= _MAX_WINDOWS:
            _WINDOWS.pop(next(iter(_WINDOWS)), None)
        window = []
        _WINDOWS[key] = window
    return window

#: Log one decode-step telemetry line every N steps. Prefill ends are always logged.
_DECODE_LOG_EVERY = 25

def _apply_mid_late_category_rescue(
    decision: dict[str, Any],
    s_z: float,
    bundle_scores: dict[str, float],
    bundle_thresholds: dict[str, dict[str, float]],
    fam: ModelFamily,
) -> None:
    """Apply the family's middle/late category-probe consensus to a decision, in place.

    Two jobs, and which one runs depends on where the gate landed:

    1. `decision["fired"]` is True -- ask the family whether the middle/late probes contradict it.
       A veto dampens the score below threshold rather than only flipping `fired`, because
       `records.finalize_live_gpu` recomputes `raw_gate_fired = score >= threshold` downstream and
       `policy`'s prior-unlock scan reads the value as `prior_score` on later turns. The measurement
       is preserved as `raw_early_score`; report that, never `score`, when `suppressed_by_mid_late`
       is set.
    2. `decision["fired"]` is False -- ask the family whether they agree enough to rescue it.

    Counting the hits is identical for every family and stays here. Concluding anything from the
    counts is the family's own `mid_late_veto` / `mid_late_rescue`, so this function names no family.
    A family that declares no `mid_late_probes` returns early and gets neither path -- which is the
    safe default, and the point of the change: the previous `if gemma4 / elif qwen36 / else` handed
    any family that was not Gemma-4 whichever tuning Qwen-3.6 had been fitted for.
    """
    if not bundle_scores or not bundle_thresholds:
        return
    probes = fam.mid_late_probes
    if not probes:
        return

    r3_hits: list[str] = []
    strict_hits: list[str] = []
    r2_hits: list[str] = []
    last_r3_hits: list[str] = []

    for probe_key in probes:
        val = bundle_scores.get(probe_key)
        th = bundle_thresholds.get(probe_key)
        if val is None or not th:
            continue
        t_r3 = float(th.get("R3", 1.0))
        t_lk16 = float(th.get("Lk16", 1.0))
        t_r2 = float(th.get("R2", 1.0))
        t_strict = max(t_r3, t_lk16)
        if val >= t_r3:
            r3_hits.append(probe_key)
            if probe_key.endswith("_last"):
                last_r3_hits.append(probe_key)
        if val >= t_strict:
            strict_hits.append(probe_key)
        if val >= t_r2:
            r2_hits.append(probe_key)

    hits = MidLateHits(
        r3=tuple(r3_hits),
        strict=tuple(strict_hits),
        r2=tuple(r2_hits),
        last_r3=tuple(last_r3_hits),
    )

    # No substituted default here. This read `decision.get("threshold", 0.082846)` -- Gemma-4's
    # threshold as a literal -- which would have been silently applied to any other family. Without
    # a threshold the consensus cannot be evaluated at all (it sets both the veto floor and the
    # rescued score), so stand down rather than borrow a number. The category probe always records
    # `threshold` on the decision, so this is an unreachable guard on the live path.
    threshold = decision.get("threshold")
    if threshold is None:
        return
    thresh_val = float(threshold)
    raw_early = float(decision.get("score", 0.0))

    if decision.get("fired", False):
        veto = fam.mid_late_veto(decision, s_z, bundle_scores, hits, thresh_val, raw_early)
        if veto is None:
            return
        veto_score = round(min(raw_early, max(thresh_val - 0.005, veto.floor)), 6)
        decision["raw_early_score"] = raw_early
        decision["score"] = veto_score
        decision["fired"] = False
        decision["suppressed_by_mid_late"] = True
        decision["mid_late_rescue_hits"] = list(veto.hits)
        decision["source"] = (
            f"{decision.get('source', '')}+mid_late_veto({veto.source_suffix})"
        )
        if veto.reason is not None:
            decision["mid_late_veto_reason"] = veto.reason
            log_cybergate_event(
                "CYBERGATE-MID-LATE-VETO",
                f"fam={fam.name} raw_early={raw_early:+.6f} -> veto_score={veto_score:+.6f} "
                f"(thresh={thresh_val:+.6f}, s_ref_z={s_z:+.2f}sigma) {veto.log_detail}",
            )
        return

    if s_z < 2.0:
        return

    s_cat_last = float(decision.get("s_cat_last", 0.0))
    if not fam.mid_late_rescue(decision, s_z, bundle_scores, hits, thresh_val, raw_early):
        return

    hits_used = strict_hits if strict_hits else r3_hits
    rescued_score = round(max(raw_early, thresh_val + 0.002), 6)

    decision["raw_early_score"] = raw_early
    decision["score"] = rescued_score
    decision["fired"] = True
    decision["rescued_by_mid_late"] = True
    decision["mid_late_rescue_hits"] = list(hits_used)
    decision["source"] = (
        f"{decision.get('source', '')}+mid_late_rescue({','.join(hits_used[:4])})"
    )
    log_cybergate_event(
        "CYBERGATE-MID-LATE-RESCUE",
        f"fam={fam.name} raw_early={raw_early:+.6f} -> rescued_score={rescued_score:+.6f} "
        f"(thresh={thresh_val:+.6f}, s_cat_last={s_cat_last:+.6f}, s_ref_z={s_z:+.2f}sigma) "
        f"strict_hits={strict_hits} r2_hits={r2_hits} r3_hits={r3_hits}",
    )


def install_refusal_monitor_hook(
    model: Any,
    ref_dir_path: str | Path | None = None,
    family: ModelFamily | str | None = None,
    *,
    ref_layer: int | None = None,
    mu_hm: float | None = None,
    sd_hm: float | None = None,
    shared_hook_state: dict[str, Any] | None = None,
    store: DecisionStore | None = None,
) -> dict[str, Any]:
    """Attach the refusal monitor. Returns the resolved `{ref_layer, dim, mu_hm, sd_hm}`."""
    import torch  # type: ignore

    # `None` resolves from the configured operator rather than defaulting to a literal "gemma4".
    # The parameter order matches the canonical `install_vllm_refusal_monitor_hook`, which the
    # `cybergate_plugin.gate_probe` facade still exposes under that name, so a positional caller
    # keeps working.
    if family is None:
        fam = family_for_configured_operator() or DEFAULT_FAMILY
    else:
        fam = family_by_name(family) if isinstance(family, str) else family
    decision_store = store if store is not None else ShmDecisionStore()

    layers = find_transformer_layers(model)
    l_ref = ref_layer if ref_layer is not None else fam.ref_layer
    l_ref = min(l_ref, len(layers) - 1)

    use_mu = mu_hm if mu_hm is not None else fam.ref_mu_hm
    use_sd = sd_hm if sd_hm is not None else fam.ref_sd_hm

    # The decode-domain calibration, resolved through the same helper the controller uses so the
    # two cannot disagree about whether the trigger is live. `mu_hm`/`sd_hm` above are fitted on
    # prefill-end turn-boundary tokens and must not be reused for decode steps -- producer review
    # Finding 1.
    _steer_cal = SteeringConfig.from_env().with_family_defaults(fam)
    mu_dec = _steer_cal.mu_decode
    sd_dec = _steer_cal.sd_decode
    decode_calibrated = _steer_cal.decode_calibrated

    if ref_dir_path is None:
        resolved = resolve_ref_dir_path(fam)
        if resolved is not None:
            ref_dir_path = resolved

    if ref_dir_path is not None and Path(ref_dir_path).exists():
        op_ref = load_cybergate_op(ref_dir_path, model_family=fam.name)
        r_vec = op_ref.c
        use_mu = float(op_ref.meta.get("mu_hm_cos", use_mu))
        use_sd = float(op_ref.meta.get("sd_hm_cos", use_sd)) or fam.ref_sd_hm
    else:
        # No refusal direction for this family: fall back to the category direction, which makes
        # `s_ref_z` a second reading of topicality rather than of refusal. Only Gemma-4 ships a
        # refusal direction; see `families/qwen36/family.py`.
        r_vec = load_cybergate_op(None, model_family=fam.name).c

    r_cpu = torch.tensor(r_vec, dtype=torch.float32)
    state: dict[str, Any] = {"r_dev": None, "step_count": 0}

    def _refusal_monitor_impl(_module: Any, _inputs: Any, output: Any) -> Any:
        t_hook0 = time.perf_counter()
        h = output[0] if isinstance(output, tuple) else output
        if h is None or h.numel() == 0:
            return output

        if state["r_dev"] is None or state["r_dev"].device != h.device:
            state["r_dev"] = r_cpu.to(device=h.device, dtype=torch.float32)

        h_2d = h.reshape(-1, h.shape[-1]) if h.ndim == 3 else h
        if h_2d.shape[-1] != state["r_dev"].numel():
            return output
        n_tok = int(h_2d.shape[0])

        input_ids_1d: list[int] | None = None
        if shared_hook_state is not None:
            raw_ids = shared_hook_state.get("last_input_ids")
            if raw_ids is not None and raw_ids.numel() == n_tok:
                input_ids_1d = raw_ids.reshape(-1).cpu().tolist()

        pending = (
            shared_hook_state.get("pending_prefill_decisions", {})
            if shared_hook_state is not None
            else {}
        )

        # One pass, one generation. The CoT controller only accepts readings tagged with the pass it
        # is sampling, so a row recycled between passes cannot be steered on the previous
        # occupant's reading. See `refusal_trace`.
        generation = refusal_trace.begin_pass()

        batch_slices = extract_batch_slices(n_tok, input_ids_1d, fam)
        sole_sequence = len(batch_slices) == 1

        for row, (req_id, q_start, q_end) in enumerate(batch_slices):
            if q_end <= q_start:
                continue
            seq_ids_for_slice = (
                input_ids_1d[q_start:q_end] if input_ids_1d is not None else None
            )
            seq_len = q_end - q_start
            s_cos = cosine(h_2d[q_end - 1].float(), state["r_dev"])
            s_z = (s_cos - use_mu) / use_sd
            elapsed_us = (time.perf_counter() - t_hook0) * 1e6

            req_state = get_request_gate_state(req_id)
            decision = pending.get((q_start, q_end))

            # Bind before the rolling window, because the window is keyed on the binding.
            #
            # `bind_request` stands down once the controller is syncing the table; see
            # `refusal_trace`. It still matters when steering is off, and for the sole-sequence
            # case, where vLLM reports no request id at all.
            slot = refusal_trace.slot_name(row)
            if seq_len > 1 and seq_ids_for_slice is not None:
                refusal_trace.bind_request(
                    slot, refusal_trace.key_for_tokens(seq_ids_for_slice)
                )
            bound_key = refusal_trace.key_of_slot(slot)

            # The rolling window follows the *request*, not the row.
            #
            # It used to hang off `get_request_gate_state(req_id)`, and `req_id` here is
            # `batch_seq_<row>` or `None` -- a position. So a request's 4-step window was inherited
            # by whoever next occupied its row, and it switched buffers outright whenever the batch
            # size crossed 1 <-> 2 and `req_id` flipped between `None` and `batch_seq_0`. Keyed on
            # the binding instead, the window belongs to one prompt for as long as that prompt is
            # resident. Falls back to the old per-row buffer only when there is no binding, which is
            # the steering-off path where nothing consumes the rolling value anyway.
            history = (
                _rolling_window(bound_key)
                if bound_key is not None
                else req_state.setdefault("s_ref_history", [])
            )

            # A multi-token pass is a prefill. With a shared state dict present, require the category
            # probe to have left a decision for this exact slice -- otherwise this is a Pass-2
            # injected prefill or a slice the probe skipped.
            is_true_prefill = bool(
                decision is not None or (seq_len > 1 and shared_hook_state is None)
            )
            if is_true_prefill:
                history.clear()
                state["step_count"] = 0

            # The window holds **decode-step** values in **decode units**, and nothing else.
            #
            # It used to take every pass, including the prefill, and every value was in prefill
            # units. Both were wrong for what consumes it. A prefill-end token sits at a turn
            # boundary and is not evidence about how a generation is going, and prefill-domain sigma
            # cannot be compared against `tau_spike` at all -- that is Finding 1, the reason the
            # trigger never fired. A prefill now only clears the window.
            if not is_true_prefill:
                history.append(
                    (s_cos - mu_dec) / sd_dec if decode_calibrated else s_z
                )
                if len(history) > _ROLLING_WINDOW:
                    del history[:-_ROLLING_WINDOW]
            rolling_z = sum(history) / len(history) if history else s_z

            req_state["rolling_s_ref_z"] = rolling_z
            req_state["s_ref_cos"] = s_cos
            req_state["s_ref_z"] = s_z
            req_state["last_cosine_us"] = round(elapsed_us, 2)

            # Hand this request's reading to the CoT controller, which samples in this same pass.
            #
            # The join is on identity, not position: a prefill binds this request id to a hash of its
            # prompt tokens, and every later pass publishes under that key. The controller computed
            # the same key from `prompt_tok_ids`. See `refusal_trace` for why position failed.
            reading = refusal_trace.RowReading(
                rolling_s_ref_z=rolling_z,
                s_ref_z=s_z,
                seq_len=seq_len,
                generation=generation,
                decode_domain=decode_calibrated,
            )
            if bound_key is not None:
                refusal_trace.publish(bound_key, reading)
            if sole_sequence:
                # vLLM reports no request id for a single-sequence pass, so no binding can exist.
                # The controller reads this only when it is tracking exactly one row.
                refusal_trace.publish(refusal_trace.SOLE_KEY, reading)

            LAST_GATE_DECISION["rolling_s_ref_z"] = rolling_z
            LAST_GATE_DECISION["s_ref_cos"] = round(s_cos, 6)
            LAST_GATE_DECISION["s_ref_z"] = round(s_z, 6)
            LAST_GATE_DECISION["last_cosine_us"] = round(elapsed_us, 2)

            if is_true_prefill and decision is not None:
                decision["s_ref_cos"] = round(s_cos, 6)
                decision["s_ref_z"] = round(s_z, 6)
                decision["ref_layer"] = l_ref
                decision["ref_mu_hm"] = round(use_mu, 6)
                decision["ref_sd_hm"] = round(use_sd, 6)
                bundle_scores = (
                    shared_hook_state.get("multi_layer_slice_scores", {}).get((q_start, q_end))
                    if shared_hook_state
                    else None
                )
                if bundle_scores:
                    decision["multi_layer_scores"] = dict(bundle_scores)
                    bundle_thresholds = shared_hook_state.get("bundle_thresholds", {})
                    _apply_mid_late_category_rescue(
                        decision, s_z, bundle_scores, bundle_thresholds, fam
                    )
                LAST_GATE_DECISION.update(decision)
                req_state.update(decision)
                # Republish: the category probe already wrote this record without `s_ref_z`, and the
                # serving path needs the refusal reading to apply the two-factor gate.
                publish_decision(decision, decision_store, LIVE_GPU_PROMPT_CACHE)

            state["step_count"] += 1
            if is_true_prefill or state["step_count"] % _DECODE_LOG_EVERY == 0:
                phase = "PREFILL_END" if is_true_prefill else f"DECODE_STEP_{state['step_count']}"
                decision_store.put_monitor(
                    {
                        "layer": l_ref,
                        "phase": phase,
                        "seq_toks": seq_len,
                        "s_cos": round(s_cos, 6),
                        "s_ref_z": round(s_z, 6),
                        "rolling_s_ref_z": round(rolling_z, 6),
                        "mu_hm": round(use_mu, 6),
                        "sd_hm": round(use_sd, 6),
                        "gpu_dot_us": round(elapsed_us, 2),
                    }
                )
                log_cybergate_event(
                    "CYBERGATE-L21-MONITOR",
                    f"layer=L{l_ref} phase={phase} row={row} req_id={req_id} "
                    f"key={bound_key or ('sole' if sole_sequence else 'unbound')} "
                    f"seq_toks={seq_len} s_cos={s_cos:+.5f} "
                    f"s_ref_z={s_z:+.3f}sigma rolling_s_ref_z={rolling_z:+.3f}sigma "
                    f"gpu_dot_us={elapsed_us:.2f}us",
                )
        return output

    layers[l_ref].register_forward_hook(dynamo_safe(_refusal_monitor_impl))
    log_cybergate_event(
        "CYBERGATE-L21-HOOK",
        f"Attached read-only Layer-{l_ref} refusal cosine monitor hook "
        f"(dim={len(r_vec)}, mu_hm={use_mu:+.5f}, sd_hm={use_sd:.5f}, K=0.0) "
        f"on {type(model).__name__}",
    )
    return {"ref_layer": l_ref, "dim": len(r_vec), "mu_hm": use_mu, "sd_hm": use_sd}
