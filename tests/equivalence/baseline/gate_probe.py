"""Unlocked-Layer CyberGate Classifier (`M1_ridge_lam16`, K = 0.0) & Layer-21 Refusal-Layer Cosine Monitor for vLLM.

Provides:
  1. `load_cybergate_op(path)`: Loads `.pt` or `.json` CyberGate operator files across any unlocked layer
     `l in {0, ..., L-1}` and pooling mode (`"mean_user"`, `"last"`, or `"hybrid"`):
       - `Gemma-4-26B-A4B-it`:
           * `gemma4_L10_M1.json` / `op_gemma_L10_M1subcat240_M1_ridge_lam16_serve.pt` (`Layer 10 mean_user`,
             `thresh = 0.10807704` [R3] / `0.10649048` [Lk16], plus embedded `Layer 10 last` direction `c_last`
             with `AUROC = 0.9783`, `648/675` CySec, `287/311` Zero-Stem, `3/637` WMDP-Bio, `0/498` SOS-Bio)
           * `gemma4_L10_last_M1.json` / `gemma4_L11_last_M1.json` / `gemma4_L8_M1.json`
       - `Qwen3.6-35B-A3B`:
           * `qwen36_L15_M1.json` / `op_qwen36_L15_M1subcat240_M1_ridge_lam16_serve.pt` (`Layer 15 mean_user`,
             `thresh = 0.07104209`) / `qwen36_L18_last_M1.json` / `qwen36_L13_M1.json`
       - `Phi-4-mini-instruct`:
           * `phi4_L26_M1.json` / `phi4_L30_M1.json` / `phi4_L30_last_M1.json`
       - `Llama-3.1-8B-Instruct`:
           * `llama31_L24_M1.json` / `llama31_L8_last_M1.json`
  2. `install_vllm_read_hook(model, op_path, ref_dir_path)`: Attaches:
       - A pre-hook on `embed_tokens` that captures the exact `input_ids` tensor of each forward batch,
       - Read-only (`K = 0.0`) forward hooks at `layers[gate_layer]` (`Layer 10` on `Gemma-4-26B`, plus
         optional multi-layer telemetry hooks on `L5, L8, L10, L11, L16`) that segment packed batches
         (`batch_size >= 1`), extract all user-turn token spans `[(u_start_1, u_end_1), ..., (u_start_k, u_end_k)]`
         directly from `input_ids` (`<|turn>user\\n` ... `<turn|>`), and compute:
           * `s_cat` (`mean_user` on the latest user turn `[u_start_k : u_end_k]`),
           * `s_cat_conv` (`mean_user` across all user turns in the conversation),
           * `s_cat_first` (`mean_user` on the first user turn `[u_start_1 : u_end_1]`),
           * `s_cat_last` (`last` prefill token cosine at `Layer 10`/`11`),
       - A read-only (`K = 0.0`) refusal-monitor hook at `layers[21]` (`Gemma-4-26B`) / `layers[28]` (`Qwen3.6-35B`)
         that computes calibrated `s_ref_cos` and `s_ref_z = (s_ref_cos - mu_hm) / sd_hm` at prefill end AND
         during decode steps, writing the unified 2D `(CyberGate, RefusalMonitor)` decision record to
         `/dev/shm/cybergate_ipc/sha_<conv_sha256>.json`.
  3. `evaluate_gate_for_prompt(...)` & `evaluate_gate_for_messages(...)`:
     - Single-turn: Pure neural linear-probe rule `fired = (s_cat >= thresh)` (zero keyword regexes).
     - Multi-turn session maintenance: Driven by the **Layer-21 Refusal Monitor (`s_ref_z`)** paired with
       **CyberGate Cross-Domain Non-Target Harm Revocation (`s_cat_standalone` / `s_cat`)**:
       In an unlocked cyber session (`prior_unlocked == True`), ordinary engineering follow-ups
       (e.g., `"refactor into a class with argparse"`, `"log to CSV, add retry logic"`) maintain the session
       whenever `s_ref_z >= ref_maint_thresh_z` (`1.5σ`, i.e., refusal pressure builds back up on the authorized
       cyber task) AND the new turn is not a cross-domain non-target harm pivot (`s_cat_standalone >= tau_revoke`).
       Adversarial pivots to non-target harm (`CBRN`/`Bio`/`Chem`/`CatQA`, where `s_cat_standalone < tau_revoke`
       while `s_ref_z` spikes) immediately revoke the session (`sticky_revoked = True, fired = False`).
       Purely benign questions (`"Explain the difference between TCP and UDP in two sentences."`) have negative
       refusal (`s_ref_z = -0.43σ < 1.5σ`) and `s_cat < thresh`, so they never fire.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import math
import os
import pickle
import struct
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
OPS_DIR = PACKAGE_DIR.parent / "ops"

DEFAULT_GEMMA4_OP = (
    OPS_DIR / "gemma4_L10_M1.json"
    if (OPS_DIR / "gemma4_L10_M1.json").exists()
    else OPS_DIR / "gemma4_L8_M1.json"
)
DEFAULT_GEMMA4_REF_DIR = OPS_DIR / "gemma4_L21_refusal_dir.json"
DEFAULT_GEMMA4_BUNDLE = OPS_DIR / "gemma4_candidate_ops_bundle.json"
DEFAULT_QWEN36_OP = (
    OPS_DIR / "qwen36_L15_M1.json"
    if (OPS_DIR / "qwen36_L15_M1.json").exists()
    else OPS_DIR / "qwen36_L13_M1.json"
)
DEFAULT_PHI4_OP = OPS_DIR / "phi4_L26_M1.json"
DEFAULT_LLAMA31_OP = OPS_DIR / "llama31_L24_M1.json"
PRECOMPUTED_SCORES_PATH = OPS_DIR / "precomputed_l8_scores.json"

# Per-prompt / per-conversation atomic IPC directory in /dev/shm
IPC_DIR_PATH = Path(
    os.environ.get("CYBERGATE_IPC_DIR")
    or os.environ.get("RESGATE_IPC_DIR", "/dev/shm/cybergate_ipc")
)
PLUGIN_EVENTS_LOG_PATH = Path(
    os.environ.get("CYBERGATE_EVENTS_LOG_PATH")
    or os.environ.get("RESGATE_EVENTS_LOG_PATH")
    or str(PACKAGE_DIR.parent / "logs" / "cybergate_plugin_events.log")
)

# Calibrated Layer-21 harmless cosine reference on Gemma-4-26B (n=554 pool681 harmless prompts):
#   mu_hm_cos = +0.13742, sd_hm_cos = 0.05130 (harmful mean = +0.68774, z = +10.73σ)
DEFAULT_GEMMA4_L21_MU_HM = 0.1374208
DEFAULT_GEMMA4_L21_SD_HM = 0.0513015

# Multi-turn Refusal-Monitored Session Maintenance & Cross-Category Non-Target Harm Revocation parameters:
# 1. Refusal Monitor Maintenance Floor (`s_ref_z >= 1.5σ`, i.e., `s_ref_cos >= 0.2144` on Gemma-4 L21):
#    In an active unlocked cyber session, when a follow-up turn causes mid-layer refusal to build back up
#    (`s_ref_z >= 1.5σ`), the session remains unlocked (`sticky_unlock = True, fired = True`) unless revoked
#    by the Cross-Domain Non-Target Harm Guard below.
#    Benign questions (e.g. "Explain the difference between TCP and UDP in two sentences" with s_ref_cos = +0.115,
#    s_ref_z = -0.43σ < 1.5σ) do not build refusal and therefore do not trigger session maintenance.
DEFAULT_REF_MAINT_Z_THRESH = 1.50

# 2. Cross-Domain Non-Target Harm Revocation Ceiling (`tau_revoke`):
#    M1_ridge_lam16 is trained with target cyber (+1) vs 14 non-target harm categories (-1: CBRN, Bio, Chem,
#    CatQA financial crime, self-harm, etc.). When a user turn in an active session has high refusal pressure
#    (`s_ref_z >= 1.5σ`) AND its standalone CyberGate score falls into the Non-Target Harm zone
#    (`s_cat_standalone < tau_revoke`, e.g., Bacillus anthracis / ricin / Sarin / CatQA fraud score -0.04..+0.035),
#    the session is immediately REVOKED (`sticky_revoked = True, fired = False`).
DEFAULT_NON_TARGET_REVOKE_CEIL_GEMMA4 = 0.0520
DEFAULT_NON_TARGET_REVOKE_CEIL_QWEN36 = 0.0380

# Fallback hysteresis floor when Layer-21 s_ref_z telemetry is unavailable (e.g., offline dry-run mode)
DEFAULT_STICKY_HYSTERESIS_THRESH_GEMMA4 = 0.0650
DEFAULT_STICKY_HYSTERESIS_THRESH_QWEN36 = 0.0500

# Shared state populated by the in-process vLLM forward hook during prefill/decode
LAST_GATE_DECISION: dict[str, Any] = {
    "decided": False,
    "fired": False,
    "score": 0.0,
    "threshold": 0.10807704,
    "layer": 10,
    "source": "none",
    "rolling_s_ref_z": 0.0,
    "last_cosine_us": 0.0,
}

# Per-request isolated state registry for continuous batching (`batch_size > 1`) in vLLM
_STATE_LOCK = threading.Lock()
REQUEST_GATE_STATES: dict[str, dict[str, Any]] = {}

# Live GPU prompt score cache (conv_sha256 -> decision dict) populated by live GPU prefill
LIVE_GPU_PROMPT_CACHE: dict[str, dict[str, Any]] = {}

# Optional exact user-content token span `(a, u)` set before prefill in single-process offline tests
CURRENT_USER_SPAN: tuple[int, int] | None = None


def log_cybergate_event(tag: str, message: str) -> None:
    """Write a timestamped event line to both `sys.stderr` and `PLUGIN_EVENTS_LOG_PATH`."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    line = f"[{tag}] [{ts}] [pid={os.getpid()}] {message}\n"
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        PLUGIN_EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLUGIN_EVENTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


log_resgate_event = log_cybergate_event


def compute_conv_sha256(user_turns: Sequence[str]) -> str:
    """Compute a deterministic SHA-256 over all user turns in a conversation.

    For a single-turn request (`len(user_turns) == 1`), `"\n<|turn_sep|>\n".join([u_1]) == u_1`,
    so `compute_conv_sha256([u_1])` is identical to `sha256(u_1.strip())`.
    """
    cleaned = [str(u).strip() for u in user_turns if str(u).strip()]
    if not cleaned:
        return hashlib.sha256(b"").hexdigest()
    joined = "\n<|turn_sep|>\n".join(cleaned)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _ensure_ipc_dir() -> Path:
    IPC_DIR_PATH.mkdir(parents=True, exist_ok=True)
    return IPC_DIR_PATH


def write_prompt_decision_ipc(sha_key: str, decision: dict[str, Any]) -> None:
    """Atomically write a per-prompt / per-conversation GPU decision to `/dev/shm/cybergate_ipc/sha_<sha_key>.json`."""
    try:
        d = _ensure_ipc_dir()
        target = d / f"sha_{sha_key}.json"
        tmp = d / f".tmp_sha_{sha_key}_{os.getpid()}_{threading.get_ident()}.json"
        tmp.write_text(json.dumps(decision), encoding="utf-8")
        tmp.replace(target)
    except Exception:
        pass


def read_prompt_decision_ipc(sha_key: str) -> dict[str, Any] | None:
    """Read the per-prompt / per-conversation GPU decision for `sha_key`, or return `None`."""
    try:
        target = IPC_DIR_PATH / f"sha_{sha_key}.json"
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def write_l21_monitor_ipc(monitor_data: dict[str, Any]) -> None:
    """Write latest Layer-21 decode monitor telemetry to `/dev/shm/cybergate_ipc/l21_monitor.json`."""
    try:
        d = _ensure_ipc_dir()
        target = d / "l21_monitor.json"
        tmp = d / f".tmp_l21_{os.getpid()}_{threading.get_ident()}.json"
        tmp.write_text(json.dumps(monitor_data), encoding="utf-8")
        tmp.replace(target)
    except Exception:
        pass


def read_l21_monitor_ipc() -> dict[str, Any] | None:
    try:
        target = IPC_DIR_PATH / "l21_monitor.json"
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def get_request_gate_state(request_id: str | None = None) -> dict[str, Any]:
    """Return or initialize the per-request isolated gate & cosine-monitor state dict."""
    if not request_id:
        return LAST_GATE_DECISION
    with _STATE_LOCK:
        st = REQUEST_GATE_STATES.get(request_id)
        if st is None:
            st = {
                "request_id": request_id,
                "decided": False,
                "fired": False,
                "score": 0.0,
                "threshold": LAST_GATE_DECISION["threshold"],
                "layer": LAST_GATE_DECISION["layer"],
                "source": "none",
                "rolling_s_ref_z": 0.0,
                "s_ref_history": [],
                "last_cosine_us": 0.0,
            }
            REQUEST_GATE_STATES[request_id] = st
            if len(REQUEST_GATE_STATES) > 2048:
                oldest_key = next(iter(REQUEST_GATE_STATES))
                if oldest_key != request_id:
                    REQUEST_GATE_STATES.pop(oldest_key, None)
        return st


def clear_request_gate_state(request_id: str) -> None:
    """Remove a completed request's state from `REQUEST_GATE_STATES`."""
    with _STATE_LOCK:
        REQUEST_GATE_STATES.pop(request_id, None)


@dataclass(frozen=True)
class CyberGateOperator:
    """Loaded `M1_ridge_lam16` linear probe operator for CyberGate across any layer `0..L-1`."""

    c: list[float]
    thresh: float
    gate_layer: int
    pool: str
    category: str
    model_key: str
    meta: dict[str, Any]

    @property
    def dim(self) -> int:
        return len(self.c)


ResGateOperator = CyberGateOperator


def _load_pt_stdlib(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch `.pt` zip archive using only Python's standard library."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        data_pkl = [n for n in names if n.endswith("data.pkl")][0]
        prefix = data_pkl[: -len("data.pkl")]

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid: Any) -> Any:
                _, stype, key, _loc, size = pid
                raw = zf.read(f"{prefix}data/{key}")
                return (stype.__name__, raw, size)

            def find_class(self, module: str, name: str) -> Any:
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(
                        storage: Any, _offset: int, _size: Any, _stride: Any, _rg: bool, _hooks: Any
                    ) -> list[float]:
                        stype, raw, n = storage
                        fmt = "f" if "Float" in stype else ("d" if "Double" in stype else "h")
                        return list(struct.unpack(f"<{n}{fmt}", raw))

                    return _rebuild
                if module == "torch":
                    return type(name, (), {})
                return super().find_class(module, name)

        return _Unpickler(zf.open(data_pkl)).load()


def load_cybergate_op(path: str | Path | None = None, model_family: str = "gemma4") -> CyberGateOperator:
    """Load a `CyberGateOperator` from `.json` or `.pt`."""
    if path is None:
        env_op = os.environ.get("CYBERGATE_OP_PATH") or os.environ.get("RESGATE_OP_PATH")
        if env_op and Path(env_op).exists():
            path = env_op
        elif model_family.startswith("gemma"):
            path = DEFAULT_GEMMA4_OP
        elif model_family.startswith("phi"):
            path = DEFAULT_PHI4_OP
        elif model_family.startswith("llama"):
            path = DEFAULT_LLAMA31_OP
        else:
            path = DEFAULT_QWEN36_OP
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CyberGate operator file not found: {p}")

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        try:
            import torch  # type: ignore

            loaded = torch.load(str(p), map_location="cpu", weights_only=False)
            c_vec = loaded["c"].detach().float().cpu().tolist() if hasattr(loaded["c"], "detach") else list(loaded["c"])
            raw = {
                "c": c_vec,
                "thresh": float(loaded["thresh"]),
                "meta": dict(loaded.get("meta", {})),
            }
        except ImportError:
            raw = _load_pt_stdlib(p)

    vec_source = raw.get("c") or raw.get("r_unit")
    c_list = [float(x) for x in vec_source]
    norm = math.sqrt(sum(x * x for x in c_list)) or 1.0
    c_unit = [x / norm for x in c_list]
    meta = dict(raw.get("meta") or {})
    thresh = float(raw.get("thresh") or 0.0)
    gate_layer = int(meta.get("gate_layer", raw.get("layer", 10 if len(c_unit) == 2816 else 15)))
    pool = str(meta.get("pool", "mean_user"))
    category = str(meta.get("category", "cyber_offense"))
    model_key = str(meta.get("model_key") or meta.get("model_tag") or model_family)

    return CyberGateOperator(
        c=c_unit,
        thresh=thresh,
        gate_layer=gate_layer,
        pool=pool,
        category=category,
        model_key=model_key,
        meta=meta,
    )


load_resgate_op = load_cybergate_op


def _find_transformer_layers(model: Any) -> Any:
    """Locate the transformer decoder `.layers` sequence across vLLM & HF model wrappers."""
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "named_modules"):
        for _name, mod in model.named_modules():
            if hasattr(mod, "layers") and hasattr(mod.layers, "__len__") and len(mod.layers) >= 16:
                return mod.layers
    raise RuntimeError(f"Cannot locate transformer `.layers` on model {type(model).__name__}")


def _find_embed_tokens_module(model: Any) -> Any | None:
    """Locate the input token embedding module (`embed_tokens`) on `model`."""
    for path in (
        ("language_model", "model", "embed_tokens"),
        ("language_model", "embed_tokens"),
        ("model", "embed_tokens"),
        ("embed_tokens",),
    ):
        curr = model
        for attr in path:
            curr = getattr(curr, attr, None)
            if curr is None:
                break
        if curr is not None and hasattr(curr, "register_forward_pre_hook"):
            return curr
    if hasattr(model, "named_modules"):
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens") and hasattr(mod, "register_forward_pre_hook"):
                return mod
    return None


def _load_worker_tokenizer(model_family: str = "gemma4") -> Any | None:
    """Load the fast HuggingFace tokenizer once inside `EngineCore` to decode exact user token spans."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        candidates: list[str] = []
        if os.environ.get("CYBERGATE_TOKENIZER_PATH"):
            candidates.append(os.environ["CYBERGATE_TOKENIZER_PATH"])
        if model_family.startswith("gemma"):
            candidates.extend(
                sorted(
                    glob.glob("/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*"),
                    reverse=True,
                )
            )
            candidates.extend(["/work/gemma4_restore", "/gemma4_restore", "/home/gpuuser/gemma4_restore"])
        else:
            candidates.extend(
                sorted(
                    glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3*35B*/snapshots/*"),
                    reverse=True,
                )
            )
        for c_path in candidates:
            if (Path(c_path) / "tokenizer.json").exists():
                try:
                    tok = AutoTokenizer.from_pretrained(c_path)
                    log_cybergate_event("CYBERGATE-TOKENIZER", f"Loaded worker tokenizer from {c_path}")
                    return tok
                except Exception:
                    continue
    except Exception as exc:
        log_cybergate_event("CYBERGATE-TOKENIZER-WARN", f"Tokenizer load skipped: {exc}")
    return None


def _locate_all_user_spans_from_token_ids(
    seq_ids: list[int],
    model_family: str = "gemma4",
    default_pre: int = 15,
    default_suf: int = 9,
) -> tuple[list[tuple[int, int]], str]:
    """Locate ALL user-turn token spans `[(u_start_1, u_end_1), ..., (u_start_k, u_end_k)]` in `seq_ids`.

    On `Gemma-4`:
      - `<|turn>user\\n` is token sequence `[105, 2364, 107]`
      - `<turn|>` is token ID `106`
    On `Qwen3.6`:
      - `<|im_start|>user\\n` is token sequence `[151644, 872, 198]`
      - `<|im_end|>` is token ID `151645`
    """
    n = len(seq_ids)
    spans: list[tuple[int, int]] = []
    if model_family.startswith("gemma"):
        i = 0
        while i <= n - 4:
            if seq_ids[i] == 105 and seq_ids[i + 1] == 2364 and seq_ids[i + 2] == 107:
                u_start = i + 3
                u_end = -1
                for j in range(u_start, n):
                    if seq_ids[j] == 106:
                        u_end = j
                        break
                if u_end > u_start:
                    spans.append((u_start, u_end))
                    i = u_end + 1
                    continue
            i += 1
        if spans:
            return spans, "exact_token_markers(<|turn>user..106)"
        # Prefix-cache boundary fallback: if `<|turn>user` (`[105, 2364, 107]`) fell inside a cached 16-token block,
        # locate the closing `<turn|>` (`106`) before `<|turn>model` (`[105, 4368, 107]`)
        for j in range(n - 1, 0, -1):
            if seq_ids[j] == 106:
                u_s = 1 if (seq_ids[0] == 107) else (2 if (n > 2 and seq_ids[0] == 2364 and seq_ids[1] == 107) else 0)
                if j > u_s:
                    return [(u_s, j)], "exact_token_markers(prefix_cached..106)"
                break
    else:
        i = 0
        while i <= n - 4:
            if seq_ids[i] == 151644 and seq_ids[i + 1] == 872 and seq_ids[i + 2] == 198:
                u_start = i + 3
                u_end = -1
                for j in range(u_start, n):
                    if seq_ids[j] == 151645:
                        u_end = j
                        break
                if u_end > u_start:
                    spans.append((u_start, u_end))
                    i = u_end + 1
                    continue
            i += 1
        if spans:
            return spans, "exact_token_markers(<|im_start|>user..151645)"

    if n > (default_pre + default_suf + 1):
        return [(default_pre, n - default_suf)], f"default_offsets({default_pre}:-{default_suf})"
    return [(0, n)], "full_sequence_fallback"


def _locate_user_span_from_token_ids(
    seq_ids: list[int],
    model_family: str = "gemma4",
    default_pre: int = 15,
    default_suf: int = 9,
) -> tuple[int, int, str]:
    """Locate the latest user-turn token span `(u_start, u_end, span_method)` from `seq_ids`."""
    spans, method = _locate_all_user_spans_from_token_ids(
        seq_ids, model_family=model_family, default_pre=default_pre, default_suf=default_suf
    )
    u_start, u_end = spans[-1]
    return u_start, u_end, method


def _extract_vllm_batch_slices(
    n_tok: int,
    input_ids_1d: list[int] | None = None,
    model_family: str = "gemma4",
) -> list[tuple[str | None, int, int]]:
    """Extract per-sequence token ranges `(req_id, q_start, q_end)` from a packed `vLLM` batch."""
    try:
        from vllm.forward_context import get_forward_context  # type: ignore

        ctx = get_forward_context()
        attn_meta = getattr(ctx, "attn_metadata", None)
        if attn_meta is not None:
            qsl = getattr(attn_meta, "query_start_loc", None)
            if qsl is not None and hasattr(qsl, "tolist"):
                locs = [int(x) for x in qsl.detach().cpu().tolist()]
                if len(locs) >= 2 and locs[-1] <= n_tok:
                    req_ids = getattr(attn_meta, "req_ids", None) or getattr(ctx, "req_ids", None)
                    slices: list[tuple[str | None, int, int]] = []
                    for b in range(len(locs) - 1):
                        rid = str(req_ids[b]) if (req_ids and b < len(req_ids)) else f"batch_seq_{b}"
                        slices.append((rid, locs[b], locs[b + 1]))
                    if len(slices) > 1:
                        return slices
    except Exception:
        pass

    # Segment packed prefill batch via `<bos>` (`token_id == 2`) sequence boundaries on Gemma-4
    if input_ids_1d is not None and len(input_ids_1d) == n_tok and model_family.startswith("gemma"):
        bos_positions = [i for i, tid in enumerate(input_ids_1d) if tid == 2 and (i == 0 or input_ids_1d[i - 1] != 2)]
        if len(bos_positions) >= 2:
            bounds = bos_positions + [n_tok]
            return [(f"bos_seq_{b}", bounds[b], bounds[b + 1]) for b in range(len(bos_positions))]

    return [(None, 0, n_tok)]


def install_vllm_refusal_monitor_hook(
    model: Any,
    ref_dir_path: str | Path | None = None,
    model_family: str = "gemma4",
    ref_layer: int | None = None,
    mu_hm: float | None = None,
    sd_hm: float | None = None,
    shared_hook_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a read-only (`K = 0.0`) forward hook at `l_ref` (`Layer 21` on `Gemma-4-26B`,
    `Layer 28` on `Qwen3.6-35B`) that monitors calibrated `s_ref_cos` and `s_ref_z(t)`:
      1. At `PREFILL_END` (`seq_len > 1`): records `s_ref_cos` and `s_ref_z` into the per-conversation
         `/dev/shm/cybergate_ipc/sha_<conv_sha256>.json` IPC record so `evaluate_gate_for_messages`
         can maintain multi-turn cyber sessions whenever refusal pressure builds back up (`s_ref_z >= 1.5σ`).
      2. During `DECODE_STEP` (`seq_len == 1`): tracks rolling `s_ref_z(t)` for Closed-Loop CoT (`CL3`).
    """
    import torch  # type: ignore

    layers = _find_transformer_layers(model)
    l_ref = ref_layer if ref_layer is not None else (21 if model_family.startswith("gemma") else 28)
    l_ref = min(l_ref, len(layers) - 1)

    default_mu = DEFAULT_GEMMA4_L21_MU_HM if model_family.startswith("gemma") else 0.0
    default_sd = DEFAULT_GEMMA4_L21_SD_HM if model_family.startswith("gemma") else 1.0
    use_mu = mu_hm if mu_hm is not None else default_mu
    use_sd = sd_hm if sd_hm is not None else default_sd

    if ref_dir_path is None and model_family.startswith("gemma") and DEFAULT_GEMMA4_REF_DIR.exists():
        ref_dir_path = DEFAULT_GEMMA4_REF_DIR

    if ref_dir_path is not None and Path(ref_dir_path).exists():
        op_ref = load_cybergate_op(ref_dir_path, model_family=model_family)
        r_vec = op_ref.c
        use_mu = float(op_ref.meta.get("mu_hm_cos", use_mu))
        use_sd = float(op_ref.meta.get("sd_hm_cos", use_sd)) or default_sd
    else:
        op_gate = load_cybergate_op(None, model_family=model_family)
        r_vec = op_gate.c

    r_cpu = torch.tensor(r_vec, dtype=torch.float32)
    state: dict[str, Any] = {"r_dev": None, "step_count": 0}

    def _refusal_monitor_impl(_module: Any, _inputs: Any, output: Any) -> Any:
        t0 = time.perf_counter()
        h = output[0] if isinstance(output, tuple) else output
        if h is not None and h.numel() > 0:
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

            batch_slices = _extract_vllm_batch_slices(
                n_tok, input_ids_1d=input_ids_1d, model_family=model_family
            )

            for req_id, q_start, q_end in batch_slices:
                if q_end <= q_start:
                    continue
                seq_len = q_end - q_start
                h_last = h_2d[q_end - 1].float()
                h_unit = h_last / h_last.norm().clamp_min(1e-9)
                s_cos = float(torch.dot(h_unit, state["r_dev"]).item())
                s_z = (s_cos - use_mu) / use_sd
                us = (time.perf_counter() - t0) * 1e6

                req_state = get_request_gate_state(req_id)
                hist = req_state.setdefault("s_ref_history", [])
                pending_map = shared_hook_state.get("pending_prefill_decisions", {}) if shared_hook_state is not None else {}
                dec = pending_map.get((q_start, q_end))
                is_true_prefill = bool(seq_len > 1 and (dec is not None or shared_hook_state is None))
                if is_true_prefill:
                    hist.clear()
                    state["step_count"] = 0
                hist.append(s_z)
                if len(hist) > 4:
                    del hist[:-4]
                rolling_z = sum(hist) / len(hist)
                req_state["rolling_s_ref_z"] = rolling_z
                req_state["s_ref_cos"] = s_cos
                req_state["s_ref_z"] = s_z
                req_state["last_cosine_us"] = round(us, 2)

                LAST_GATE_DECISION["rolling_s_ref_z"] = rolling_z
                LAST_GATE_DECISION["s_ref_cos"] = round(s_cos, 6)
                LAST_GATE_DECISION["s_ref_z"] = round(s_z, 6)
                LAST_GATE_DECISION["last_cosine_us"] = round(us, 2)

                # If this is a prefill pass, attach s_ref_cos & s_ref_z to the slice's decision IPC!
                if is_true_prefill and dec is not None:
                    dec["s_ref_cos"] = round(s_cos, 6)
                    dec["s_ref_z"] = round(s_z, 6)
                    dec["ref_layer"] = l_ref
                    dec["ref_mu_hm"] = round(use_mu, 6)
                    dec["ref_sd_hm"] = round(use_sd, 6)
                    ml_scores = shared_hook_state.get("multi_layer_slice_scores", {}).get((q_start, q_end)) if shared_hook_state else None
                    if ml_scores:
                        dec["multi_layer_scores"] = dict(ml_scores)
                    if dec.get("is_baseline_prefill", True):
                        conv_sha = dec.get("conv_sha256")
                        user_sha = dec.get("user_sha256")
                        is_mt_ctx = bool(dec.get("is_multiturn_ctx", False))
                        if conv_sha and (not is_mt_ctx or conv_sha != user_sha):
                            LIVE_GPU_PROMPT_CACHE[conv_sha] = dec
                            write_prompt_decision_ipc(conv_sha, dec)
                        if user_sha:
                            if is_mt_ctx:
                                ctx_key = f"ctx_{user_sha}"
                                LIVE_GPU_PROMPT_CACHE[ctx_key] = dec
                                write_prompt_decision_ipc(ctx_key, dec)
                            else:
                                st_key = f"st_{user_sha}"
                                LIVE_GPU_PROMPT_CACHE[st_key] = dec
                                write_prompt_decision_ipc(st_key, dec)
                                LIVE_GPU_PROMPT_CACHE[user_sha] = dec
                                write_prompt_decision_ipc(user_sha, dec)

                state["step_count"] += 1
                if is_true_prefill or state["step_count"] % 25 == 0:
                    phase_tag = "PREFILL_END" if is_true_prefill else f"DECODE_STEP_{state['step_count']}"
                    mon_entry = {
                        "layer": l_ref,
                        "phase": phase_tag,
                        "seq_toks": seq_len,
                        "s_cos": round(s_cos, 6),
                        "s_ref_z": round(s_z, 6),
                        "rolling_s_ref_z": round(rolling_z, 6),
                        "mu_hm": round(use_mu, 6),
                        "sd_hm": round(use_sd, 6),
                        "gpu_dot_us": round(us, 2),
                    }
                    write_l21_monitor_ipc(mon_entry)
                    log_cybergate_event(
                        "CYBERGATE-L21-MONITOR",
                        f"layer=L{l_ref} phase={phase_tag} seq_toks={seq_len} s_cos={s_cos:+.5f} "
                        f"s_ref_z={s_z:+.3f}σ rolling_s_ref_z={rolling_z:+.3f}σ gpu_dot_us={us:.2f}us",
                    )
        return output

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        _refusal_monitor_hook = torch._dynamo.disable(_refusal_monitor_impl)
    else:
        _refusal_monitor_hook = _refusal_monitor_impl

    layers[l_ref].register_forward_hook(_refusal_monitor_hook)
    log_cybergate_event(
        "CYBERGATE-L21-HOOK",
        f"Attached read-only Layer-{l_ref} refusal cosine monitor hook "
        f"(dim={len(r_vec)}, mu_hm={use_mu:+.5f}, sd_hm={use_sd:.5f}, K=0.0) on {type(model).__name__}",
    )
    return {"ref_layer": l_ref, "dim": len(r_vec), "mu_hm": use_mu, "sd_hm": use_sd}


def install_vllm_read_hook(
    model: Any,
    op_path: str | Path | None = None,
    ref_dir_path: str | Path | None = None,
) -> CyberGateOperator:
    """Attach read-only (`K = 0.0`) forward hooks inside `vLLM`:
      - `embed_tokens` pre-hook for exact `input_ids` capture,
      - `layers[op.gate_layer]` (`Layer 10` on `Gemma-4-26B`, `Layer 15` on `Qwen3.6-35B`, or any unlocked
        layer `0..L-1`) for exact token-marker multi-turn user-span extraction, packed-batch segmentation,
        and `s_cat` / `s_cat_conv` / `s_cat_first` / `s_cat_last` scoring,
      - Optional candidate-layer telemetry hooks (`L5, L8, L10, L11, L16`) when `gemma4_candidate_ops_bundle.json`
        is present,
      - `layers[21]` (`Gemma-4`) / `layers[28]` (`Qwen3.6`) refusal-monitor hook that writes calibrated
        `s_ref_cos` and `s_ref_z` into `/dev/shm/cybergate_ipc/sha_<conv_sha256>.json`.
    """
    import torch  # type: ignore

    op = load_cybergate_op(op_path)
    if getattr(model, "_cybergate_hook_installed", False):
        return op

    layers = _find_transformer_layers(model)
    model._cybergate_hook_installed = True
    c_cpu = torch.tensor(op.c, dtype=torch.float32)
    c_last_raw = op.meta.get("c_last")
    c_last_cpu = torch.tensor(c_last_raw, dtype=torch.float32) if c_last_raw else None
    worker_tok = _load_worker_tokenizer(op.model_key)

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

    # Attach pre-hook on `embed_tokens` to capture exact `input_ids` for the current forward pass
    embed_mod = _find_embed_tokens_module(model)
    if embed_mod is not None:
        def _embed_pre_hook(_mod: Any, args: tuple[Any, ...]) -> None:
            if args and hasattr(args[0], "reshape"):
                state["last_input_ids"] = args[0].detach()
                state["pending_prefill_decisions"] = {}
                state["multi_layer_slice_scores"] = {}

        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
            embed_mod.register_forward_pre_hook(torch._dynamo.disable(_embed_pre_hook))
        else:
            embed_mod.register_forward_pre_hook(_embed_pre_hook)
        log_cybergate_event(
            "CYBERGATE-EMBED-HOOK",
            f"Attached input_ids pre-hook on {type(embed_mod).__name__} for exact multi-turn span extraction",
        )

    default_pre = 15 if op.model_key.startswith("gemma") else 14
    default_suf = 9 if op.model_key.startswith("gemma") else 5

    # Attach lightweight candidate-layer telemetry hooks (L5, L8, L10, L11, L16) if bundle exists
    if op.model_key.startswith("gemma") and DEFAULT_GEMMA4_BUNDLE.exists():
        try:
            bundle_data = json.loads(DEFAULT_GEMMA4_BUNDLE.read_text(encoding="utf-8")).get("layers", {})
            probe_layers = [5, 8, 10, 11, 16]
            for pl in probe_layers:
                if str(pl) not in bundle_data or pl >= len(layers):
                    continue
                c_m_cpu = torch.tensor(bundle_data[str(pl)]["mean_user"]["c"], dtype=torch.float32)
                c_l_cpu = torch.tensor(bundle_data[str(pl)]["last"]["c"], dtype=torch.float32)

                def _make_cand_hook(layer_idx: int, cm_cpu: Any, cl_cpu: Any) -> Any:
                    c_state: dict[str, Any] = {"cm": None, "cl": None}

                    def _cand_hook(_mod: Any, _inp: Any, out: Any) -> Any:
                        h = out[0] if isinstance(out, tuple) else out
                        if h is None or h.shape[0] <= 1:
                            return out
                        if c_state["cm"] is None or c_state["cm"].device != h.device:
                            c_state["cm"] = cm_cpu.to(device=h.device, dtype=torch.float32)
                            c_state["cl"] = cl_cpu.to(device=h.device, dtype=torch.float32)
                        h_2d = h.reshape(-1, h.shape[-1]) if h.ndim == 3 else h
                        n_tok = int(h_2d.shape[0])
                        raw_ids = state.get("last_input_ids")
                        ids_1d = raw_ids.reshape(-1).cpu().tolist() if (raw_ids is not None and raw_ids.numel() == n_tok) else None
                        for _rid, qs, qe in _extract_vllm_batch_slices(n_tok, ids_1d, op.model_key):
                            if qe - qs <= 1:
                                continue
                            s_ids = ids_1d[qs:qe] if ids_1d is not None else None
                            if s_ids is not None:
                                u_spans, _ = _locate_all_user_spans_from_token_ids(s_ids, op.model_key, default_pre, default_suf)
                            else:
                                u_spans = [(default_pre, max(default_pre + 1, (qe - qs) - default_suf))]
                            us, ue = u_spans[-1]
                            h_seq = h_2d[qs:qe]
                            h_u = h_seq[us:ue].float().mean(dim=0)
                            h_u = h_u / h_u.norm().clamp_min(1e-9)
                            sc_m = float(torch.dot(h_u, c_state["cm"]).item())
                            h_lt = h_seq[-1].float()
                            h_lt = h_lt / h_lt.norm().clamp_min(1e-9)
                            sc_l = float(torch.dot(h_lt, c_state["cl"]).item())
                            sl_dict = state["multi_layer_slice_scores"].setdefault((qs, qe), {})
                            sl_dict[f"L{layer_idx}_mean_user"] = round(sc_m, 6)
                            sl_dict[f"L{layer_idx}_last"] = round(sc_l, 6)
                        return out

                    return torch._dynamo.disable(_cand_hook) if (hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable")) else _cand_hook

                layers[pl].register_forward_hook(_make_cand_hook(pl, c_m_cpu, c_l_cpu))
        except Exception as exc:
            log_cybergate_event("CYBERGATE-BUNDLE-WARN", f"Candidate bundle hooks skipped: {exc}")

    target_layer = layers[min(op.gate_layer, len(layers) - 1)]

    def _read_only_impl(_module: Any, _inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        if h is None or h.shape[0] <= 1:
            return output

        t_gpu0 = time.perf_counter()
        if state["c_dev"] is None or state["c_dev"].device != h.device:
            state["c_dev"] = c_cpu.to(device=h.device, dtype=torch.float32)
            if c_last_cpu is not None:
                state["c_last_dev"] = c_last_cpu.to(device=h.device, dtype=torch.float32)

        h_2d = h.reshape(-1, h.shape[-1]) if h.ndim == 3 else h
        n_tok = int(h_2d.shape[0])

        input_ids_1d: list[int] | None = None
        raw_ids = state.get("last_input_ids")
        if raw_ids is not None and raw_ids.numel() == n_tok:
            input_ids_1d = raw_ids.reshape(-1).cpu().tolist()

        batch_slices = _extract_vllm_batch_slices(n_tok, input_ids_1d=input_ids_1d, model_family=op.model_key)

        for req_id, q_start, q_end in batch_slices:
            seq_len = q_end - q_start
            if seq_len <= 1:
                continue
            h_seq = h_2d[q_start:q_end]
            seq_ids = input_ids_1d[q_start:q_end] if input_ids_1d is not None else None

            if CURRENT_USER_SPAN is not None and len(batch_slices) == 1:
                a, u = CURRENT_USER_SPAN
                u_start, u_end = a, min(seq_len, a + u)
                user_spans = [(u_start, u_end)]
                span_method = "explicit_CURRENT_USER_SPAN"
            elif seq_ids is not None:
                user_spans, span_method = _locate_all_user_spans_from_token_ids(
                    seq_ids, model_family=op.model_key, default_pre=default_pre, default_suf=default_suf
                )
                # Filter out multi-request concurrent decode batches (`seq_len <= 16` without chat turn markers)
                if not span_method.startswith("exact_token_markers") and seq_len <= 16:
                    continue
                u_start, u_end = user_spans[-1]
            elif seq_len > (default_pre + default_suf + 1):
                u_start, u_end = default_pre, seq_len - default_suf
                user_spans = [(u_start, u_end)]
                span_method = f"default_offsets({default_pre}:-{default_suf})"
            else:
                if seq_len <= 16:
                    continue
                u_start, u_end = 0, seq_len
                user_spans = [(u_start, u_end)]
                span_method = "full_sequence_fallback"

            h_user = h_seq[u_start:u_end] if (0 <= u_start < u_end) else h_seq

            t_dot0 = time.perf_counter()
            # 1. Latest user-turn mean_user score
            h_mean = h_user.float().mean(dim=0)
            h_unit = h_mean / h_mean.norm().clamp_min(1e-9)
            s_cat_mean = float(torch.dot(h_unit, state["c_dev"]).item())

            # 2. First user-turn and Conversation-wide (all user turns) mean_user scores
            u1_start, u1_end = user_spans[0]
            if len(user_spans) > 1 and 0 <= u1_start < u1_end:
                h_u1 = h_seq[u1_start:u1_end].float().mean(dim=0)
                h_u1 = h_u1 / h_u1.norm().clamp_min(1e-9)
                s_cat_first = float(torch.dot(h_u1, state["c_dev"]).item())

                all_u_Indices: list[int] = []
                for us_i, ue_i in user_spans:
                    if 0 <= us_i < ue_i <= seq_len:
                        all_u_Indices.extend(range(us_i, ue_i))
                if all_u_Indices:
                    h_conv_u = h_seq[all_u_Indices].float().mean(dim=0)
                    h_conv_u = h_conv_u / h_conv_u.norm().clamp_min(1e-9)
                    s_cat_conv = float(torch.dot(h_conv_u, state["c_dev"]).item())
                else:
                    s_cat_conv = s_cat_mean
            else:
                s_cat_first = s_cat_mean
                s_cat_conv = s_cat_mean

            # 3. Last prefill token score
            h_last_tok = h_seq[-1].float()
            h_last_unit = h_last_tok / h_last_tok.norm().clamp_min(1e-9)
            if state["c_last_dev"] is not None:
                s_cat_last = float(torch.dot(h_last_unit, state["c_last_dev"]).item())
            else:
                s_cat_last = float(torch.dot(h_last_unit, state["c_dev"]).item())

            score = s_cat_last if op.pool == "last" else s_cat_mean
            gpu_dot_us = (time.perf_counter() - t_dot0) * 1e6
            total_hook_us = (time.perf_counter() - t_gpu0) * 1e6
            fired = bool(score >= op.thresh)

            # Decode all user turns to compute both `user_sha256` (latest turn) and `conv_sha256` (full conversation)
            user_sha256: str | None = None
            conv_sha256: str | None = None
            if worker_tok is not None and seq_ids is not None and 0 <= u_start < u_end:
                try:
                    decoded_turns: list[str] = []
                    for us_i, ue_i in user_spans:
                        if 0 <= us_i < ue_i <= seq_len:
                            t_str = worker_tok.decode(seq_ids[us_i:ue_i], skip_special_tokens=False).strip()
                            if t_str:
                                decoded_turns.append(t_str)
                    if decoded_turns:
                        user_sha256 = hashlib.sha256(decoded_turns[-1].encode("utf-8")).hexdigest()
                        conv_sha256 = compute_conv_sha256(decoded_turns)
                except Exception:
                    user_sha256 = None
                    conv_sha256 = None

            # Distinguish Pass-1 / baseline chat prefill (suf_len <= 12) from Pass-2 <think>-injected prefill (suf_len > 12)
            suf_len = seq_len - u_end
            is_baseline_prefill = bool(suf_len <= 12 or span_method == "full_sequence_fallback")
            # Detect whether this sequence is a multi-turn in-context prefill (even under vLLM prefix caching)
            is_multiturn_ctx = bool(
                len(user_spans) > 1
                or u_start > 16
                or (seq_ids is not None and len(seq_ids) > 0 and seq_ids[0] != 2 and u_start != 4)
            )

            decision: dict[str, Any] = {
                "ts": time.time(),
                "pid": os.getpid(),
                "req_id": req_id,
                "user_sha256": user_sha256,
                "conv_sha256": conv_sha256 or user_sha256,
                "num_user_turns": len(user_spans),
                "is_multiturn_ctx": is_multiturn_ctx,
                "n_prefill_tokens": seq_len,
                "user_span": [u_start, u_end],
                "all_user_spans": [list(sp) for sp in user_spans],
                "suf_tokens": suf_len,
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
                "hook_total_us": round(total_hook_us, 2),
                "source": (
                    f"live_vllm_gpu_L{op.gate_layer}_{op.pool}_hook"
                    f"(tokens={seq_len},turns={len(user_spans)},span={u_start}:{u_end},gpu_dot_us={gpu_dot_us:.1f})"
                ),
            }
            ml_scores = state.get("multi_layer_slice_scores", {}).get((q_start, q_end))
            if ml_scores:
                decision["multi_layer_scores"] = dict(ml_scores)

            state["pending_prefill_decisions"][(q_start, q_end)] = decision
            LAST_GATE_DECISION.update(decision)
            if req_id:
                req_st = get_request_gate_state(req_id)
                req_st.update(decision)

            state["prefill_count"] += 1
            if is_baseline_prefill and (conv_sha256 or user_sha256):
                key_sha = conv_sha256 or user_sha256
                assert key_sha is not None
                if not is_multiturn_ctx or key_sha != user_sha256:
                    LIVE_GPU_PROMPT_CACHE[key_sha] = decision
                    write_prompt_decision_ipc(key_sha, decision)
                if user_sha256:
                    if is_multiturn_ctx:
                        ctx_key = f"ctx_{user_sha256}"
                        LIVE_GPU_PROMPT_CACHE[ctx_key] = decision
                        write_prompt_decision_ipc(ctx_key, decision)
                    else:
                        st_key = f"st_{user_sha256}"
                        LIVE_GPU_PROMPT_CACHE[st_key] = decision
                        write_prompt_decision_ipc(st_key, decision)
                        LIVE_GPU_PROMPT_CACHE[user_sha256] = decision
                        write_prompt_decision_ipc(user_sha256, decision)

            log_cybergate_event(
                f"CYBERGATE-L{op.gate_layer}-GPU",
                f"req_id={req_id} conv_sha={(conv_sha256 or user_sha256 or 'n/a')[:10]} "
                f"user_sha={(user_sha256 or 'n/a')[:10]} turns={len(user_spans)} prefill_tokens={seq_len} "
                f"user_span=[{u_start}:{u_end}] suf={suf_len} baseline_prefill={is_baseline_prefill} "
                f"-> s_cat={score:+.6f} (conv={s_cat_conv:+.6f}, last={s_cat_last:+.6f}, thresh={op.thresh:+.6f}) "
                f"-> FIRED={fired} gpu_dot_us={gpu_dot_us:.2f}us",
            )
        return output

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        _read_only_hook = torch._dynamo.disable(_read_only_impl)
    else:
        _read_only_hook = _read_only_impl

    target_layer.register_forward_hook(_read_only_hook)
    log_cybergate_event(
        f"CYBERGATE-L{op.gate_layer}-HOOK",
        f"Attached read-only Layer-{op.gate_layer} ({op.pool}) M1_ridge_lam16 hook "
        f"(dim={op.dim}, thresh={op.thresh:.8f}, total_layers={len(layers)}, K=0.0) on {type(model).__name__}",
    )

    ref_env = ref_dir_path or os.environ.get("CYBERGATE_REF_DIR_PATH") or os.environ.get("RESGATE_REF_DIR_PATH")
    if ref_env is not None or os.environ.get("CYBERGATE_ENABLE_REF_MONITOR", os.environ.get("RESGATE_ENABLE_REF_MONITOR", "1")) == "1":
        install_vllm_refusal_monitor_hook(
            model,
            ref_dir_path=ref_env,
            model_family=op.model_key,
            shared_hook_state=state,
        )

    return op


def _load_precomputed_scores() -> dict[str, Any]:
    if PRECOMPUTED_SCORES_PATH.exists():
        try:
            return json.loads(PRECOMPUTED_SCORES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_PRECOMPUTED_CACHE = _load_precomputed_scores()


def evaluate_gate_for_prompt(
    user_text: str,
    model_id: str = "google/gemma-4-26B-A4B-it",
    prompt_key: str | None = None,
    resgate_http_url: str | None = None,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    verify_ssl: bool = False,
    request_id: str | None = None,
    live_gpu_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the Unlocked-Layer `CyberGate` (`M1_ridge_lam16`) decision for a single `user_text`.

    In `vllm serve`, uses the pure neural `Layer 10` (`Gemma-4`) / `Layer 15` (`Qwen3.6`)
    `M1_ridge_lam16` linear-probe decision:
        `fired = bool(score >= op.thresh)`
    with zero keyword regexes.
    """
    family = "gemma" if "gemma" in model_id.lower() else "qwen36"
    op = load_cybergate_op(model_family="gemma4" if family == "gemma" else "qwen36")
    txt_clean = user_text.strip()
    h_sha = hashlib.sha256(txt_clean.encode("utf-8")).hexdigest()

    def _finalize_live_gpu(dec: dict[str, Any]) -> dict[str, Any]:
        out = dict(dec)
        sc = float(out.get("score", 0.0))
        th = float(out.get("threshold", op.thresh))
        ref_z = out.get("s_ref_z")
        raw_cat_fired = bool(sc >= th)
        out["raw_gate_fired"] = raw_cat_fired
        out["raw_l8_fired"] = raw_cat_fired
        # Two-factor gate: if CyberGate domain score >= thresh, also verify Layer-21 refusal pressure
        # is active (`s_ref_z >= 1.50σ`) so purely benign technical networking questions (e.g. TCP vs UDP,
        # where `s_ref_z = -0.50σ`) pass through naturally without `<|think|>` injection.
        if raw_cat_fired and ref_z is not None and float(ref_z) < DEFAULT_REF_MAINT_Z_THRESH:
            out["fired"] = False
            out["source"] = (
                f"single_turn_L21_no_refusal_passthrough"
                f"(s_cat={sc:+.4f}>={th:+.4f},s_ref_z={float(ref_z):+.2f}sigma<{DEFAULT_REF_MAINT_Z_THRESH:+.2f}sigma)"
            )
        else:
            out["fired"] = raw_cat_fired
        return out

    # 1. Explicit live GPU decision from Pass-1 prefill hook
    if live_gpu_decision and live_gpu_decision.get("decided"):
        res = _finalize_live_gpu(live_gpu_decision)
        LIVE_GPU_PROMPT_CACHE[h_sha] = res
        return res

    # 2. Per-prompt atomic IPC file `/dev/shm/cybergate_ipc/sha_st_<user_sha256>.json` or `sha_<user_sha256>.json`
    ipc_dec = read_prompt_decision_ipc(f"st_{h_sha}") or read_prompt_decision_ipc(h_sha)
    if ipc_dec and ipc_dec.get("decided"):
        res = _finalize_live_gpu(ipc_dec)
        LIVE_GPU_PROMPT_CACHE[h_sha] = res
        return res

    # 3. In-memory live GPU prompt cache
    if h_sha in LIVE_GPU_PROMPT_CACHE:
        return dict(LIVE_GPU_PROMPT_CACHE[h_sha])

    # 4. Per-request isolated hook state (single-process mode)
    if request_id and request_id in REQUEST_GATE_STATES and REQUEST_GATE_STATES[request_id].get("decided"):
        res = _finalize_live_gpu(REQUEST_GATE_STATES[request_id])
        REQUEST_GATE_STATES[request_id]["decided"] = False
        LIVE_GPU_PROMPT_CACHE[h_sha] = res
        return res

    # 5. External tempx /gate endpoint (if configured)
    gate_url = resgate_http_url or os.environ.get("CYBERGATE_GATE_URL") or os.environ.get("RESGATE_GATE_URL")
    if gate_url:
        req = urllib.request.Request(
            gate_url.rstrip("/") + "/gate",
            data=json.dumps({"prompt": user_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            gdata = json.loads(resp.read().decode("utf-8"))
        score = float(gdata.get("score", 0.0))
        thresh = float(gdata.get("threshold", op.thresh))
        return {
            "decided": True,
            "fired": bool(gdata.get("fired", score >= thresh)),
            "score": score,
            "threshold": thresh,
            "layer": op.gate_layer,
            "source": "tempx_gate_http",
        }

    # 6. Offline precomputed M1_ridge_lam16 score lookup (used by --dry-run)
    if prompt_key and prompt_key in _PRECOMPUTED_CACHE.get("by_key", {}):
        entry = _PRECOMPUTED_CACHE["by_key"][prompt_key].get(family)
        if entry:
            return {
                "decided": True,
                "fired": bool(entry["fired_M1"]),
                "score": float(entry["s_cat_M1"]),
                "threshold": op.thresh,
                "layer": op.gate_layer,
                "source": f"offline_L{op.gate_layer}_M1_precomputed(key={prompt_key})",
            }
    if h_sha in _PRECOMPUTED_CACHE.get("by_sha256", {}):
        entry = _PRECOMPUTED_CACHE["by_sha256"][h_sha].get(family)
        if entry:
            return {
                "decided": True,
                "fired": bool(entry["fired_M1"]),
                "score": float(entry["s_cat_M1"]),
                "threshold": op.thresh,
                "layer": op.gate_layer,
                "source": f"offline_L{op.gate_layer}_M1_precomputed(sha256={h_sha[:10]})",
            }
    if txt_clean.lower() in _PRECOMPUTED_CACHE.get("by_text_norm", {}):
        entry = _PRECOMPUTED_CACHE["by_text_norm"][txt_clean.lower()].get(family)
        if entry:
            return {
                "decided": True,
                "fired": bool(entry["fired_M1"]),
                "score": float(entry["s_cat_M1"]),
                "threshold": op.thresh,
                "layer": op.gate_layer,
                "source": f"offline_L{op.gate_layer}_M1_precomputed(id={entry.get('id')})",
            }

    return {
        "decided": False,
        "fired": False,
        "score": 0.0,
        "threshold": op.thresh,
        "layer": op.gate_layer,
        "source": "no_live_gpu_hook_attached",
    }


def evaluate_gate_for_messages(
    messages: Sequence[dict[str, Any]],
    model_id: str = "google/gemma-4-26B-A4B-it",
    prompt_key: str | None = None,
    resgate_http_url: str | None = None,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    verify_ssl: bool = False,
    request_id: str | None = None,
    enable_multi_turn_stickiness: bool = True,
    live_gpu_decision: dict[str, Any] | None = None,
    standalone_user_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """2D Multi-Turn `CyberGate` + `Layer-21 Refusal Monitor` Evaluator (Zero Keyword Regexes):

    1. **Single-Turn / Initial Unlock (`turn == 1`):**
       Requires the user prompt's `CyberGate` (`M1_ridge_lam16` at `Layer 10` on `Gemma-4-26B` /
       `Layer 15` on `Qwen3.6-35B`) cosine score `s_cat >= thresh` (`0.10807704` on `Gemma-4 L10`).
       Standalone benign technical prompts (e.g. `"Explain the difference between TCP and UDP in two sentences."`)
       and non-target harm controls (`CBRN`, `CatQA`) score below `thresh` (`fired = False`).

    2. **Multi-Turn Refusal-Monitored Session Maintenance & Cross-Domain Revocation (`turn >= 2`):**
       When an earlier user turn in the conversation unlocked the cyber session (`prior_unlocked == True`):
       - **Step 2A — Cross-Domain Non-Target Harm Revocation (`s_cat_standalone < tau_revoke`):**
         Because `M1_ridge_lam16` is trained to separate Target Cyber (`+1`) from 14 Non-Target Harm
         categories (`-1`: CBRN, Bio, Chem, CatQA financial crime, etc.), any adversarial pivot to
         Non-Target Harm (e.g., `Bacillus anthracis`, `ricin toxin`, `Sarin`, `counterfeit bills`)
         drives the new turn's standalone `s_cat` into the Non-Target Harm zone (`< 0.052` on `Gemma-4`)
         while spiking `s_ref_z`. Such turns are immediately **REVOKED** (`sticky_revoked = True, fired = False`).
       - **Step 2B — Direct Cyber Re-Clearance (`s_cat >= thresh`):**
         If the follow-up turn itself clears `s_cat >= thresh` (either in-context or standalone), it fires directly.
       - **Step 2C — Refusal-Monitor Session Maintenance (`s_ref_z >= 1.5σ` at `Layer 21`):**
         Legitimate software-engineering follow-ups to an authorized cyber workflow (e.g., `"refactor into a class
         with argparse"`, `"log to CSV, add retry logic"`, `"add time-based blind injection"`) have low standalone
         `s_cat`, yet when prefilled in the multi-turn cyber conversation, **mid-layer refusal at `Layer 21` builds
         back up (`s_ref_z >= 1.5σ`, `s_ref_cos >= 0.214`)** because the model is about to refuse modifying/extending
         the exploit script. Gating session maintenance on `s_ref_z >= 1.5σ` (provided Step 2A did not revoke)
         maintains the session (`sticky_unlock = True, fired = True`) across all legitimate task continuations,
         while purely benign questions (`"Explain the difference between TCP and UDP in two sentences"`, where
         `s_ref_z = -0.43σ < 1.5σ`) do not build refusal and pass through naturally (`fired = False`).
    """
    user_turns: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            text_s = str(content).strip()
            if text_s:
                user_turns.append(text_s)

    if not user_turns:
        return evaluate_gate_for_prompt(
            "",
            model_id=model_id,
            prompt_key=prompt_key,
            resgate_http_url=resgate_http_url,
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            verify_ssl=verify_ssl,
            request_id=request_id,
        )

    family = "gemma" if "gemma" in model_id.lower() else "qwen36"
    op = load_cybergate_op(model_family="gemma4" if family == "gemma" else "qwen36")
    revoke_ceil = (
        DEFAULT_NON_TARGET_REVOKE_CEIL_GEMMA4
        if family == "gemma"
        else DEFAULT_NON_TARGET_REVOKE_CEIL_QWEN36
    )
    hysteresis_thresh = (
        DEFAULT_STICKY_HYSTERESIS_THRESH_GEMMA4
        if family == "gemma"
        else DEFAULT_STICKY_HYSTERESIS_THRESH_QWEN36
    )
    ref_maint_z = DEFAULT_REF_MAINT_Z_THRESH

    conv_sha = compute_conv_sha256(user_turns)
    latest_user = user_turns[-1]
    latest_user_sha = hashlib.sha256(latest_user.encode("utf-8")).hexdigest()

    # Look up full-conversation GPU decision first (keyed by conv_sha or ctx_<latest_user_sha> when prefix-cached)
    conv_dec = live_gpu_decision
    if conv_dec is None or not conv_dec.get("decided"):
        if len(user_turns) > 1:
            conv_dec = read_prompt_decision_ipc(f"ctx_{latest_user_sha}") or read_prompt_decision_ipc(conv_sha)
        else:
            conv_dec = read_prompt_decision_ipc(conv_sha) or read_prompt_decision_ipc(f"st_{latest_user_sha}")
    if conv_dec is not None and conv_dec.get("decided"):
        latest_dec = dict(conv_dec)
        sc = float(latest_dec.get("score", 0.0))
        th = float(latest_dec.get("threshold", op.thresh))
        ref_z_c = latest_dec.get("s_ref_z")
        raw_cat_fired = bool(sc >= th)
        latest_dec["raw_gate_fired"] = raw_cat_fired
        if raw_cat_fired and ref_z_c is not None and float(ref_z_c) < ref_maint_z:
            latest_dec["fired"] = False
            latest_dec["source"] = (
                f"single_turn_L21_no_refusal_passthrough"
                f"(s_cat={sc:+.4f}>={th:+.4f},s_ref_z={float(ref_z_c):+.2f}sigma<{ref_maint_z:+.2f}sigma)"
            )
        else:
            latest_dec["fired"] = raw_cat_fired
    else:
        latest_dec = evaluate_gate_for_prompt(
            latest_user,
            model_id=model_id,
            prompt_key=prompt_key if len(user_turns) == 1 else None,
            resgate_http_url=resgate_http_url,
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            verify_ssl=verify_ssl,
            request_id=request_id,
            live_gpu_decision=live_gpu_decision,
        )
        latest_dec = dict(latest_dec)

    # Also look up standalone latest-user decision if multi-turn
    st_dec = standalone_user_decision
    if st_dec is None and len(user_turns) > 1:
        st_dec = read_prompt_decision_ipc(f"st_{latest_user_sha}") or read_prompt_decision_ipc(latest_user_sha)
        if st_dec is None:
            st_fallback = evaluate_gate_for_prompt(
                latest_user,
                model_id=model_id,
                resgate_http_url=resgate_http_url,
                vllm_base_url=vllm_base_url,
                vllm_api_key=vllm_api_key,
                verify_ssl=verify_ssl,
                request_id=None,
            )
            if st_fallback.get("decided"):
                st_dec = st_fallback

    s_cat_in_context = float(latest_dec.get("score", 0.0))
    s_cat_standalone = float(st_dec.get("score", s_cat_in_context)) if st_dec else s_cat_in_context
    s_cat_conv = float(latest_dec.get("s_cat_conv", s_cat_in_context))
    s_cat_first = float(latest_dec.get("s_cat_first", s_cat_in_context))
    s_cat_last = float(latest_dec.get("s_cat_last", s_cat_in_context))
    s_ref_cos = latest_dec.get("s_ref_cos") or (st_dec.get("s_ref_cos") if st_dec else None)
    s_ref_z = latest_dec.get("s_ref_z") if latest_dec.get("s_ref_z") is not None else (st_dec.get("s_ref_z") if st_dec else None)
    st_ref_z = st_dec.get("s_ref_z") if st_dec else None
    latest_dec["s_ref_cos"] = s_ref_cos
    latest_dec["s_ref_z"] = s_ref_z

    latest_dec["user_turn_index"] = len(user_turns)
    latest_dec["sticky_unlock"] = False
    latest_dec["sticky_revoked"] = False
    latest_dec["s_cat_standalone"] = round(s_cat_standalone, 6)
    latest_dec["s_cat_in_context"] = round(s_cat_in_context, 6)
    latest_dec["s_cat_conv"] = round(s_cat_conv, 6)
    latest_dec["non_target_revoke_ceiling"] = revoke_ceil
    latest_dec["ref_maint_z_threshold"] = ref_maint_z
    latest_dec["hysteresis_threshold"] = hysteresis_thresh

    if len(user_turns) <= 1 or not enable_multi_turn_stickiness:
        # When stickiness is disabled on a multi-turn prompt, evaluate purely on the standalone latest user turn
        if len(user_turns) > 1 and not enable_multi_turn_stickiness and st_dec is not None:
            th = float(latest_dec.get("threshold", op.thresh))
            latest_dec["score"] = round(s_cat_standalone, 6)
            raw_st_fired = bool(s_cat_standalone >= th)
            if raw_st_fired and st_ref_z is not None and float(st_ref_z) < ref_maint_z:
                latest_dec["fired"] = False
                latest_dec["source"] = (
                    f"stateless_single_turn_L21_no_refusal_passthrough"
                    f"(s_cat_standalone={s_cat_standalone:+.4f}>={th:+.4f},s_ref_z={float(st_ref_z):+.2f}sigma<{ref_maint_z:+.2f}sigma)"
                )
            else:
                latest_dec["fired"] = raw_st_fired
                latest_dec["source"] = f"stateless_single_turn_eval(s_cat_standalone={s_cat_standalone:+.4f},thresh={th:+.4f})"
        return latest_dec

    # Check if an earlier user turn in this conversation unlocked the cyber session (`fired == True`)
    prior_unlocked_dec: dict[str, Any] | None = None
    prior_unlocked_turn_idx: int | None = None
    th_main = float(latest_dec.get("threshold", op.thresh))

    for idx, prev_text in enumerate(user_turns[:-1], start=1):
        prev_sha = hashlib.sha256(prev_text.encode("utf-8")).hexdigest()
        prev_st_ipc = read_prompt_decision_ipc(f"st_{prev_sha}") or read_prompt_decision_ipc(prev_sha)
        prev_dec = evaluate_gate_for_prompt(
            prev_text,
            model_id=model_id,
            resgate_http_url=resgate_http_url,
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            verify_ssl=verify_ssl,
            request_id=None,
            live_gpu_decision=prev_st_ipc,
        )
        prev_sc = float(prev_dec.get("score", 0.0))
        prev_rz = prev_dec.get("s_ref_z")
        if prev_dec.get("fired"):
            prior_unlocked_dec = prev_dec
            prior_unlocked_turn_idx = idx
        elif prev_sc < revoke_ceil or (prev_sc < th_main and prev_rz is not None and float(prev_rz) >= 7.0):
            prior_unlocked_dec = None
            prior_unlocked_turn_idx = None

    if prior_unlocked_dec is None:
        # No prior turn unlocked the session: evaluate current turn strictly by its standalone two-factor decision
        if st_dec is not None:
            st_final = evaluate_gate_for_prompt(
                latest_user,
                model_id=model_id,
                resgate_http_url=resgate_http_url,
                vllm_base_url=vllm_base_url,
                vllm_api_key=vllm_api_key,
                verify_ssl=verify_ssl,
                request_id=None,
                live_gpu_decision=st_dec,
            )
            latest_dec["fired"] = bool(st_final.get("fired", False))
            latest_dec["score"] = round(float(st_final.get("score", s_cat_standalone)), 6)
            latest_dec["source"] = str(st_final.get("source", latest_dec.get("source", "")))
        return latest_dec

    prior_score = float(prior_unlocked_dec.get("score", s_cat_first))
    latest_dec["prior_unlock_score"] = round(prior_score, 6)
    latest_dec["sticky_from_turn"] = prior_unlocked_turn_idx

    # Step 2A: Cross-Domain Non-Target Harm Revocation Guard
    # If the user pivots to a non-target harmful domain (CBRN/Bio/Chem/CatQA), the standalone classifier on the
    # new turn scores `s_cat_standalone < th_main` (not cyber) while standalone Layer-21 refusal spikes (`st_ref_z >= 7.0σ`),
    # or `s_cat_standalone < revoke_ceil` (`< 0.0520`) / `s_cat_in_context < 0.0750`. Revoke immediately!
    eff_ref_z = float(s_ref_z) if s_ref_z is not None else 0.0
    eff_st_ref_z = float(st_ref_z) if st_ref_z is not None else 0.0
    st_last = float(st_dec.get("s_cat_last", s_cat_last)) if st_dec else s_cat_last
    is_nontarget_harm_pivot = bool(
        s_cat_standalone < revoke_ceil
        or s_cat_in_context < 0.0750
        or (s_cat_standalone < th_main and eff_st_ref_z >= 7.0)
        or (min(s_cat_last, st_last) < 0.0350 and max(eff_ref_z, eff_st_ref_z) >= 7.0 and s_cat_standalone < th_main)
    )
    if is_nontarget_harm_pivot:
        latest_dec["fired"] = False
        latest_dec["sticky_unlock"] = False
        latest_dec["sticky_revoked"] = True
        latest_dec["score"] = round(min(s_cat_standalone, s_cat_in_context), 6)
        latest_dec["source"] = (
            f"multi_turn_L{latest_dec.get('layer', op.gate_layer)}_nontarget_harm_revoked"
            f"(turn={len(user_turns)},s_cat_standalone={s_cat_standalone:+.4f},s_cat_in_ctx={s_cat_in_context:+.4f},"
            f"s_cat_last={min(s_cat_last, st_last):+.4f},st_ref_z={st_ref_z if st_ref_z is not None else 'n/a'},"
            f"s_ref_z={s_ref_z if s_ref_z is not None else 'n/a'})"
        )
        return latest_dec

    # Step 2B: Refusal-Monitored Session Maintenance (`s_ref_z >= ref_maint_z` at Layer 21)
    # In an unlocked cyber session, if `s_ref_z >= 1.5σ` (the model's mid-layer refusal circuit is building
    # back up to refuse a continuation/refactoring of the cyber workflow), maintain unlock!
    if s_ref_z is not None:
        ref_z_val = float(s_ref_z)
        st_z_val = float(st_ref_z) if st_ref_z is not None else ref_z_val
        max_ref_z = max(ref_z_val, st_z_val)
        if max_ref_z >= ref_maint_z:
            latest_dec["fired"] = True
            latest_dec["sticky_unlock"] = True
            latest_dec["sticky_revoked"] = False
            latest_dec["source"] = (
                f"multi_turn_L21_refusal_monitored_maintained"
                f"(from_turn={prior_unlocked_turn_idx},turn={len(user_turns)},"
                f"s_ref_z={ref_z_val:+.2f}sigma>={ref_maint_z:+.2f}sigma,"
                f"s_cat_in_ctx={s_cat_in_context:+.4f},s_cat_conv={s_cat_conv:+.4f},"
                f"s_cat_standalone={s_cat_standalone:+.4f})"
            )
            return latest_dec
        else:
            # Refusal monitor shows no refusal buildup (`s_ref_z < 1.5σ`, e.g. benign question like TCP vs UDP
            # or benign summary request that the base model answers natively without <think> injection)
            latest_dec["fired"] = False
            latest_dec["sticky_unlock"] = False
            latest_dec["sticky_revoked"] = False
            latest_dec["source"] = (
                f"multi_turn_L21_no_refusal_passthrough"
                f"(turn={len(user_turns)},s_ref_z={ref_z_val:+.2f}sigma<{ref_maint_z:+.2f}sigma,"
                f"s_cat_in_ctx={s_cat_in_context:+.4f},s_cat_standalone={s_cat_standalone:+.4f})"
            )
            return latest_dec

    # Step 2C: Offline / Fallback Hysteresis Check (when live Layer-21 s_ref_z is not attached)
    if s_cat_in_context >= hysteresis_thresh or s_cat_conv >= th_main:
        latest_dec["fired"] = True
        latest_dec["sticky_unlock"] = True
        latest_dec["sticky_revoked"] = False
        latest_dec["source"] = (
            f"multi_turn_L{latest_dec.get('layer', op.gate_layer)}_hysteresis_continued"
            f"(from_turn={prior_unlocked_turn_idx},s_cat={s_cat_in_context:+.4f}>={hysteresis_thresh:+.4f})"
        )
        return latest_dec

    latest_dec["fired"] = False
    latest_dec["sticky_unlock"] = False
    latest_dec["sticky_revoked"] = True
    latest_dec["source"] = (
        f"multi_turn_L{latest_dec.get('layer', op.gate_layer)}_neural_revoked"
        f"(turn={len(user_turns)},s_cat={s_cat_in_context:+.4f}<{hysteresis_thresh:+.4f})"
    )
    return latest_dec
