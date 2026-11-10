"""The `vllm.general_plugins` entry point, with process-aware dispatch.

vLLM invokes this in **both** processes:

    MainProcess (pid=1)  -> API server + ASGI middleware
    EngineCore  (pid=73) -> model worker + forward hooks

They share no memory. Each needs different work done, and importantly the API-server process must not
import torch -- so the worker-side wiring is imported inside the function that does it, never at
module scope.

Port of `cybergate_plugin/__init__.py:register_cybergate_plugin`.
"""
from __future__ import annotations

import multiprocessing
import os
from typing import Any

from .config import env_first
from .telemetry.log import log_cybergate_event

__all__ = ["register_cybergate_plugin"]

MAIN_PROCESS_NAME = "MainProcess"
MIDDLEWARE_PATH = "cybergate_plugin.middleware.CyberGateThinkMiddleware"

_PLUGIN_REGISTERED = False


def register_cybergate_plugin() -> None:
    """Install the gate into whichever process invoked this.

    Idempotent per process. Every failure is logged and swallowed: a plugin that raises here takes
    the server down with it, and the gate failing to attach degrades to an ungated server, which the
    startup log makes visible.
    """
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return
    _PLUGIN_REGISTERED = True

    # CyberGate's `cybergate_cot_steering` logits processor implements vLLM's V1
    # LogitsProcessor interface (`vllm.v1.sample.logits_processor.LogitsProcessor`).
    # Default VLLM_USE_V2_MODEL_RUNNER=0 at plugin registration time so bare
    # `vllm serve` invocations without explicit env exports do not crash with:
    # "V1-interface plugins are not supported by the V2 model runner."
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    process_name = multiprocessing.current_process().name
    op_path = env_first("CYBERGATE_OP_PATH")
    ref_dir_path = env_first("CYBERGATE_REF_DIR_PATH")
    log_cybergate_event(
        "CYBERGATE-PLUGIN",
        f"register_cybergate_plugin() invoked in process={process_name} "
        f"pid={os.getpid()} op_path={op_path or 'default(family)'}",
    )

    _wrap_model_loader(op_path, ref_dir_path, process_name)
    if process_name == MAIN_PROCESS_NAME:
        _install_middleware()


def _wrap_model_loader(
    op_path: str | None,
    ref_dir_path: str | None,
    process_name: str,
) -> None:
    """Wrap vLLM's model loading so the forward hooks attach to the loaded model.

    Both entry points are wrapped because which one runs depends on the vLLM version and the loader
    in use. `install_category_probe_hook` is idempotent per model object, so a double call attaches
    once.

    This only does anything in the worker process -- the API server never loads a model -- but the
    wrapping itself is harmless there and the canonical package does it unconditionally.
    """
    try:
        import vllm.model_executor.model_loader as loader_mod  # type: ignore
    except ImportError as exc:
        log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"vllm import skipped: {exc}")
        return

    def _attach(model: Any, where: str) -> None:
        try:
            # Imported here, not at module scope: this pulls in torch.
            from .hooks import install_category_probe_hook

            install_category_probe_hook(model, op_path=op_path, ref_dir_path=ref_dir_path)
        except Exception as exc:
            log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"{where} hook failed: {exc}")

    orig_get_model = getattr(loader_mod, "get_model", None)
    if callable(orig_get_model):

        def _hooked_get_model(*args: Any, **kwargs: Any) -> Any:
            model = orig_get_model(*args, **kwargs)
            _attach(model, "get_model")
            return model

        loader_mod.get_model = _hooked_get_model

    for cls_name in ("BaseModelLoader", "DefaultModelLoader"):
        loader_cls = getattr(loader_mod, cls_name, None)
        if loader_cls is None or not hasattr(loader_cls, "load_model"):
            continue
        loader_cls.load_model = _make_hooked_load(loader_cls.load_model, cls_name, _attach)

    log_cybergate_event(
        "CYBERGATE-PLUGIN",
        f"Wrapped vLLM model loader (get_model + BaseModelLoader.load_model) in "
        f"process={process_name}",
    )


def _make_hooked_load(original: Any, cls_name: str, attach: Any) -> Any:
    """Build the wrapped `load_model` for one loader class.

    A factory rather than an inline closure, so each class captures its own `original` instead of the
    last value of a loop variable.
    """

    def _hooked_load(self_loader: Any, *args: Any, **kwargs: Any) -> Any:
        model = original(self_loader, *args, **kwargs)
        attach(model, f"{cls_name}.load_model")
        return model

    return _hooked_load


def _install_middleware() -> None:
    """Append the gate middleware to the API server's middleware list, if it is not already there."""
    try:
        import vllm.entrypoints.openai.api_server as api_mod  # type: ignore

        orig_build_app = getattr(api_mod, "build_app", None)
        if not callable(orig_build_app):
            return

        def _hooked_build_app(args: Any, *b_args: Any, **b_kwargs: Any) -> Any:
            middleware = getattr(args, "middleware", None)
            if isinstance(middleware, list):
                if MIDDLEWARE_PATH not in middleware:
                    middleware.append(MIDDLEWARE_PATH)
            return orig_build_app(args, *b_args, **b_kwargs)

        api_mod.build_app = _hooked_build_app
    except Exception as exc:
        log_cybergate_event("CYBERGATE-PLUGIN-WARN", f"middleware auto-install skipped: {exc}")
