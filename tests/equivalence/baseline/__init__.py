"""`cybergate_plugin`: Selective Cyber-Domain Unlocking vLLM Plugin on Pristine Models (`K = 0.0`).

Combines Early-Layer `CyberGate` Classifier (`M1_ridge_lam16` at Layer 8/13)
with Conditional Server-Side `<think>` Prefill (`TH3`), Open-`<think>` Sandwich (`SW3`),
and Closed-Loop Cosine-Monitored `<think>` Steering (`CL3` / `CL2`) in `vLLM`.
"""
from __future__ import annotations

import multiprocessing
import os
from typing import Any

from .gate_probe import (
    LAST_GATE_DECISION,
    LIVE_GPU_PROMPT_CACHE,
    REQUEST_GATE_STATES,
    CyberGateOperator,
    ResGateOperator,
    clear_request_gate_state,
    compute_conv_sha256,
    evaluate_gate_for_messages,
    evaluate_gate_for_prompt,
    get_request_gate_state,
    install_vllm_read_hook,
    install_vllm_refusal_monitor_hook,
    load_cybergate_op,
    load_resgate_op,
    log_cybergate_event,
    log_resgate_event,
)
from .middleware import (
    CyberGateThinkMiddleware,
    CyberGateVLLMClient,
    ResGateThinkMiddleware,
    ResGateVLLMClient,
)
from .prompts import (
    BOTTOM_COMPLIANCE_LOCK,
    FAKE_THINK_RUBRIC,
    FAKE_THINK_SHORT,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
    build_raw_prompt_for_model,
    strip_injected_think_boilerplate,
    truncate_think_at_second_guess,
)
from .sandwich_processor import (
    CyberGateClosedLoopCoTProcessor,
    CyberGateSandwichLogitsProcessor,
    ResGateClosedLoopCoTProcessor,
    ResGateSandwichLogitsProcessor,
)

__all__ = [
    "BOTTOM_COMPLIANCE_LOCK",
    "CyberGateClosedLoopCoTProcessor",
    "CyberGateOperator",
    "CyberGateSandwichLogitsProcessor",
    "CyberGateThinkMiddleware",
    "CyberGateVLLMClient",
    "FAKE_THINK_RUBRIC",
    "FAKE_THINK_SHORT",
    "INLINE_COT_SUPPRESSION",
    "INLINE_FINAL_STEP_CLEARANCE",
    "LAST_GATE_DECISION",
    "LIVE_GPU_PROMPT_CACHE",
    "OPEN_THINK_PREFIX",
    "POST_THINK_OPENER",
    "REQUEST_GATE_STATES",
    "ResGateClosedLoopCoTProcessor",
    "ResGateOperator",
    "ResGateSandwichLogitsProcessor",
    "ResGateThinkMiddleware",
    "ResGateVLLMClient",
    "build_raw_prompt_for_model",
    "clear_request_gate_state",
    "evaluate_gate_for_messages",
    "evaluate_gate_for_prompt",
    "get_request_gate_state",
    "install_vllm_read_hook",
    "install_vllm_refusal_monitor_hook",
    "load_cybergate_op",
    "load_resgate_op",
    "log_cybergate_event",
    "log_resgate_event",
    "register_cybergate_plugin",
    "register_resgate_plugin",
    "strip_injected_think_boilerplate",
    "truncate_think_at_second_guess",
]

_PLUGIN_REGISTERED = False


def register_cybergate_plugin() -> None:
    """Entrypoint registered under `vllm.general_plugins` (`cybergate_think`).

    Automatically wraps `vllm.model_executor.model_loader.get_model` and
    `BaseModelLoader.load_model` in the `vLLM` worker process so
    `install_vllm_read_hook(model, op_path, ref_dir_path)` is attached at
    `layers[gate_layer]` (`Layer 8` on `Gemma-4-26B`, `Layer 13` on `Qwen3.6-35B`)
    and `layers[ref_layer]` (`Layer 21` on `Gemma-4-26B`) with zero weight edits (`K = 0.0`).
    """
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return
    _PLUGIN_REGISTERED = True

    proc_name = multiprocessing.current_process().name
    op_path = os.environ.get("CYBERGATE_OP_PATH") or os.environ.get("RESGATE_OP_PATH")
    ref_dir_path = os.environ.get("CYBERGATE_REF_DIR_PATH") or os.environ.get("RESGATE_REF_DIR_PATH")
    log_cybergate_event(
        "CYBERGATE-PLUGIN",
        f"register_cybergate_plugin() invoked in process={proc_name} op_path={op_path or 'default(gemma4_L10_M1.json)'}",
    )

    try:
        import vllm.model_executor.model_loader as loader_mod  # type: ignore

        orig_get_model = getattr(loader_mod, "get_model", None)
        if callable(orig_get_model):
            def _hooked_get_model(*args: Any, **kwargs: Any) -> Any:
                model = orig_get_model(*args, **kwargs)
                try:
                    install_vllm_read_hook(model, op_path=op_path, ref_dir_path=ref_dir_path)
                except Exception as exc:
                    log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"get_model hook failed: {exc}")
                return model

            loader_mod.get_model = _hooked_get_model

        for cls_name in ("BaseModelLoader", "DefaultModelLoader"):
            loader_cls = getattr(loader_mod, cls_name, None)
            if loader_cls is not None and hasattr(loader_cls, "load_model"):
                orig_load = loader_cls.load_model

                def _make_hooked_load(fn: Any) -> Any:
                    def _hooked_load(self_loader: Any, *args: Any, **kwargs: Any) -> Any:
                        model = fn(self_loader, *args, **kwargs)
                        try:
                            install_vllm_read_hook(model, op_path=op_path, ref_dir_path=ref_dir_path)
                        except Exception as exc:
                            log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"{cls_name}.load_model hook failed: {exc}")
                        return model

                    return _hooked_load

                loader_cls.load_model = _make_hooked_load(orig_load)

        log_cybergate_event(
            "CYBERGATE-PLUGIN",
            f"Wrapped vLLM model loader (get_model + BaseModelLoader.load_model) in process={proc_name}",
        )
    except ImportError as exc:
        log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"vllm import skipped: {exc}")

    if proc_name == "MainProcess":
        try:
            import vllm.entrypoints.openai.api_server as api_mod  # type: ignore

            orig_build_app = getattr(api_mod, "build_app", None)
            if callable(orig_build_app):
                def _hooked_build_app(args: Any, *b_args: Any, **b_kwargs: Any) -> Any:
                    mw_list = getattr(args, "middleware", None)
                    mw_fqn = "cybergate_plugin.middleware.CyberGateThinkMiddleware"
                    if mw_list is not None and isinstance(mw_list, list):
                        if mw_fqn not in mw_list and "resgate_plugin.middleware.ResGateThinkMiddleware" not in mw_list:
                            mw_list.append(mw_fqn)
                    return orig_build_app(args, *b_args, **b_kwargs)

                api_mod.build_app = _hooked_build_app
        except Exception:
            pass


# Backward-compatible alias
register_resgate_plugin = register_cybergate_plugin
