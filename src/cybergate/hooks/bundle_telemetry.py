"""Candidate-layer telemetry hooks. Observation only -- they never influence a decision.

When `ops/<family>_candidate_ops_bundle.json` is present, a read-only scoring hook is attached at each
candidate layer and its scores are recorded on the decision record under `multi_layer_scores`. This
is how the layer choice was compared on live traffic. Absent bundle file means no hooks, which is the
normal case on the live server.

Port of the bundle block inside `gate_probe.install_vllm_read_hook` (lines 783-833).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import env_first
from ..families import ModelFamily
from ..ops import load_cybergate_op
from ..ops.loader import artifact_dirs
from ..telemetry.log import log_cybergate_event
from ..vllm_adapter import extract_batch_slices
from ._common import dynamo_safe
from ._scoring import cosine

__all__ = ["attach_bundle_telemetry_hooks"]

#: Candidate layers scored for telemetry and middle/late layer rescue when a bundle is present.
#: Early layers (L5..L13) are excluded so only middle/late layers (L15..L28) are evaluated.
#:
#: Kept as the value both shipped families declare, for reference and for callers that still import
#: it. The attached layers come from `family.bundle_probe_layers`, not from here, so a new family
#: picks its own set without editing this module.
BUNDLE_PROBE_LAYERS = (15, 16, 18, 19, 20, 21, 22, 24, 26, 28)


#: Mirrors `category_probe._BASELINE_SUFFIX_MAX`. A Pass-2 route prefill appends an
#: injected thought block, so its suffix is long; Pass-1's is short.
_BASELINE_SUFFIX_MAX = 12

def _resolve_bundle_path(family: ModelFamily) -> Path | None:
    """Resolve the candidate operator bundle path across OPS_DIR and CYBERGATE_OP_PATH's directory."""
    if not family.default_bundle_name:
        return None
    candidates: list[Path] = [d / family.default_bundle_name for d in artifact_dirs(family)]
    op_env = env_first("CYBERGATE_OP_PATH")
    if op_env:
        try:
            candidates.append(
                Path(op_env).expanduser().resolve().parent / family.default_bundle_name
            )
        except OSError:
            pass
    candidates.append(Path("/work/cybergate_plugin/ops") / family.default_bundle_name)
    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
        except OSError:
            continue
    return None


def attach_bundle_telemetry_hooks(
    layers: Any,
    family: ModelFamily,
    state: dict[str, Any],
) -> None:
    """Attach read-only scoring hooks at the candidate layers, when a bundle file is present.

    A bundle is not the only reason to attach. `CYBERGATE_CROSS_LAYER_DIAG` and
    `CYBERGATE_DUMP_ACTS_DIR` below are per-layer diagnostics that need no bundle at all -- they
    carry their own direction (the shipped one) or no direction (the dump). Returning early on a
    missing bundle made both unreachable for exactly the family that needs them: one with
    `default_bundle_name = None`, which is every newly fitted family before its bundle exists.
    """
    import torch  # type: ignore

    bundle_path = _resolve_bundle_path(family)
    if bundle_path is None and not (
        env_first("CYBERGATE_CROSS_LAYER_DIAG") or env_first("CYBERGATE_DUMP_ACTS_DIR")
    ):
        return

    # Cross-layer diagnostic: the SHIPPED operator's own direction, applied at every probe layer.
    # This answers whether the deployed operator was fitted against a different layer's activations
    # than the one the serving hook reads -- each bundle probe uses its own layer's vector, so it
    # cannot distinguish that on its own. Telemetry only; no decision reads these keys.
    shipped_mean = shipped_last = None
    if env_first("CYBERGATE_CROSS_LAYER_DIAG"):
        try:
            op_shipped = load_cybergate_op(env_first("CYBERGATE_OP_PATH"), model_family=family.name)
            shipped_mean = torch.tensor(op_shipped.c, dtype=torch.float32)
            raw_last = op_shipped.meta.get("c_last")
            if raw_last:
                shipped_last = torch.tensor(raw_last, dtype=torch.float32)
                shipped_last = shipped_last / shipped_last.norm().clamp_min(1e-9)
            log_cybergate_event(
                "CYBERGATE-XLAYER",
                "cross-layer diagnostic armed from %s (L%s %s)"
                % (env_first("CYBERGATE_OP_PATH"), op_shipped.gate_layer, op_shipped.pool),
            )
        except Exception as exc:
            log_cybergate_event("CYBERGATE-XLAYER-WARN", f"cross-layer diagnostic unavailable: {exc}")

    # Restricting the layers matters in practice: both pools at all 64 layers is ~1.3 MB per
    # request, so a 1681-row refusal corpus writes 2.2 GB through 215k append-and-close calls.
    # A fit needs the gate layer and the refusal layer, so name those two.
    dump_layers: set[int] | None = None
    raw_layers = env_first("CYBERGATE_DUMP_ACTS_LAYERS")
    if raw_layers:
        try:
            dump_layers = {int(p) for p in raw_layers.replace(",", " ").split()}
        except ValueError:
            log_cybergate_event(
                "CYBERGATE-ACTDUMP-WARN",
                f"CYBERGATE_DUMP_ACTS_LAYERS={raw_layers!r} is not a layer list; dumping all",
            )

    dump_dir = env_first("CYBERGATE_DUMP_ACTS_DIR")
    if dump_dir:
        try:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            state["act_dump_dir"] = dump_dir
            state["act_dump_layers"] = dump_layers
            log_cybergate_event(
                "CYBERGATE-ACTDUMP",
                f"pooled-activation dump armed at {dump_dir} for "
                f"{'layers ' + ','.join(map(str, sorted(dump_layers))) if dump_layers else 'all layers'}",
            )
        except OSError as exc:
            log_cybergate_event("CYBERGATE-ACTDUMP-WARN", f"dump dir unusable: {exc}")

    try:
        if bundle_path is not None:
            raw_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle = raw_bundle.get("layers") or raw_bundle.get("operators") or {}
        else:
            bundle = {}
        thresholds_map = state.setdefault("bundle_thresholds", {})
        attached: list[int] = []
        # Either diagnostic wants every layer: the cross-layer one to find which layer the shipped
        # direction actually belongs to, the dump to capture pooled vectors for an offline re-fit.
        diag_layers = (
            range(len(layers)) if (shipped_mean is not None or dump_dir) else ()
        )
        probe_layers = sorted(set(family.bundle_probe_layers) | set(diag_layers))
        for probe_layer in probe_layers:
            if probe_layer >= len(layers):
                continue
            in_bundle = str(probe_layer) in bundle
            mean_cpu = last_cpu = None
            l_entry = bundle.get(str(probe_layer)) or {}
            if in_bundle:
                mean_vec = l_entry["mean_user"].get("c") or l_entry["mean_user"].get("w")
                last_vec = l_entry["last"].get("c") or l_entry["last"].get("w")
                mean_cpu = torch.tensor(mean_vec, dtype=torch.float32)
                last_cpu = torch.tensor(last_vec, dtype=torch.float32)
            elif shipped_mean is None and not dump_dir:
                continue
            for pool_name in ("mean_user", "last") if in_bundle else ():
                th = l_entry.get(pool_name, {}).get("thresholds")
                if isinstance(th, dict):
                    thresholds_map[f"L{probe_layer}_{pool_name}"] = {
                        k: float(v) for k, v in th.items() if isinstance(v, (int, float))
                    }
            layers[probe_layer].register_forward_hook(
                _make_hook(probe_layer, mean_cpu, last_cpu, family, state,
                           shipped_mean, shipped_last)
            )
            attached.append(probe_layer)
        if attached:
            log_cybergate_event(
                "CYBERGATE-BUNDLE-HOOK",
                f"Attached {len(attached)} candidate-layer probes {attached} from "
                f"{bundle_path.name if bundle_path is not None else 'diagnostics only (no bundle)'}",
            )
    except Exception as exc:
        log_cybergate_event("CYBERGATE-BUNDLE-WARN", f"Candidate bundle hooks skipped: {exc}")


def _make_hook(
    layer_idx: int,
    mean_cpu: Any,
    last_cpu: Any,
    family: ModelFamily,
    state: dict[str, Any],
    shipped_mean_cpu: Any = None,
    shipped_last_cpu: Any = None,
) -> Any:
    """Build one candidate-layer telemetry hook. Called at attach time, once per layer."""
    import torch  # type: ignore

    dirs: dict[str, Any] = {"mean": None, "last": None, "ship_mean": None, "ship_last": None,
                            "ready": False, "dev": None}

    def _bundle_hook(_mod: Any, _inp: Any, out: Any) -> Any:
        h = out[0] if isinstance(out, tuple) else out
        chunk_mode = state.get("chunk_mode", "none")
        if h is None or (h.shape[0] <= 1 and chunk_mode not in ("accumulate", "complete")):
            return out
        if not dirs["ready"] or dirs["dev"] != h.device:
            dirs["ready"] = True
            dirs["dev"] = h.device
            if mean_cpu is not None:
                dirs["mean"] = mean_cpu.to(device=h.device, dtype=torch.float32)
                dirs["last"] = last_cpu.to(device=h.device, dtype=torch.float32)
            if shipped_mean_cpu is not None:
                dirs["ship_mean"] = shipped_mean_cpu.to(device=h.device, dtype=torch.float32)
            if shipped_last_cpu is not None:
                dirs["ship_last"] = shipped_last_cpu.to(device=h.device, dtype=torch.float32)

        h_2d = h.reshape(-1, h.shape[-1]) if h.ndim == 3 else h
        n_tok = int(h_2d.shape[0])

        if chunk_mode == "start":
            state.setdefault("chunk_h_buf", {})[layer_idx] = h_2d.detach().clone()
            return out
        if chunk_mode == "accumulate":
            prev_h = state.setdefault("chunk_h_buf", {}).get(layer_idx)
            state["chunk_h_buf"][layer_idx] = (
                torch.cat([prev_h.to(h_2d.device), h_2d.detach()], dim=0).clone()
                if prev_h is not None
                else h_2d.detach().clone()
            )
            return out
        if chunk_mode == "complete":
            prev_h = state.get("chunk_h_buf", {}).pop(layer_idx, None)
            full_ids = state.get("chunk_full_ids")
            if prev_h is not None and full_ids is not None:
                h_full = torch.cat([prev_h.to(h_2d.device), h_2d], dim=0)
                user_spans, _ = family.locate_all_user_spans(full_ids)
                u_start, u_end = user_spans[-1]
                slot = state["multi_layer_slice_scores"].setdefault((0, n_tok), {})
                if dirs["mean"] is not None:
                    slot[f"L{layer_idx}_mean_user"] = round(
                        cosine(h_full[u_start:u_end].float().mean(dim=0), dirs["mean"]), 6
                    )
                    slot[f"L{layer_idx}_last"] = round(
                        cosine(h_full[-1].float(), dirs["last"]), 6
                    )
                return out

        raw_ids = state.get("last_input_ids")
        ids_1d = (
            raw_ids.reshape(-1).cpu().tolist()
            if (raw_ids is not None and raw_ids.numel() == n_tok)
            else None
        )

        for _rid, q_start, q_end in extract_batch_slices(n_tok, ids_1d, family):
            if q_end - q_start <= 1:
                continue
            seq_ids = ids_1d[q_start:q_end] if ids_1d is not None else None
            if seq_ids is not None:
                user_spans, span_method = family.locate_all_user_spans(seq_ids)
            else:
                user_spans = [
                    (
                        family.default_pre,
                        max(family.default_pre + 1, (q_end - q_start) - family.default_suf),
                    )
                ]
                span_method = f"default_offsets({family.default_pre}:-{family.default_suf})"
            u_start, u_end = user_spans[-1]
            h_seq = h_2d[q_start:q_end]
            slot = state["multi_layer_slice_scores"].setdefault((q_start, q_end), {})
            if dirs["mean"] is not None:
                slot[f"L{layer_idx}_mean_user"] = round(
                    cosine(h_seq[u_start:u_end].float().mean(dim=0), dirs["mean"]), 6
                )
                slot[f"L{layer_idx}_last"] = round(
                    cosine(h_seq[-1].float(), dirs["last"]), 6
                )
            if dirs["ship_mean"] is not None:
                slot[f"X{layer_idx}_mean_user"] = round(
                    cosine(h_seq[u_start:u_end].float().mean(dim=0), dirs["ship_mean"]), 6
                )
            dump_dir = state.get("act_dump_dir")
            want = state.get("act_dump_layers")
            # Pass-1 ONLY. Every request runs two prefills: the baseline observation of the user
            # turn, and a Pass-2 `<think>`-injected prefill that re-observes the same conversation
            # with server-authored text appended. `_common.publish_decision` drops Pass-2 for
            # exactly that reason, and a dump that keeps it writes two rows per request where the
            # second is scored over text the user never sent -- half a fit corpus of it, silently,
            # since the file carries no labels. Same predicate the probe records as
            # `is_baseline_prefill`; see hooks/category_probe.py.
            is_pass1 = family.is_baseline_prefill(
                user_spans[0][0] if user_spans else u_start,
                (q_end - q_start) - u_end,
                span_method,
                _BASELINE_SUFFIX_MAX,
            )
            if dump_dir and is_pass1 and (want is None or layer_idx in want):
                # BOTH pools. The category factor is fitted on `mean_user` and the refusal factor
                # on `last`, so dumping only the first makes half the kit unfittable from vLLM's
                # own activations -- which is the only way to fit a model whose kernels disagree
                # with `transformers`. One row appended per forward, in request order: the replay
                # must be strictly sequential for row i to mean prompt i, and the caller checks
                # the row count against the prompt count.
                try:
                    for suffix, vec in (
                        ("", h_seq[u_start:u_end].float().mean(dim=0)),
                        ("_last", h_seq[-1].float()),
                    ):
                        with open(f"{dump_dir}/L{layer_idx}{suffix}.bin", "ab") as fh:
                            fh.write(vec.to("cpu").to(torch.float16).numpy().tobytes())
                except Exception:
                    pass
            if dirs["ship_last"] is not None:
                slot[f"X{layer_idx}_last"] = round(
                    cosine(h_seq[-1].float(), dirs["ship_last"]), 6
                )
        return out

    return dynamo_safe(_bundle_hook)
