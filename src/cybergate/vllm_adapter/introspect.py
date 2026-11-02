"""QUARANTINE: reverse-engineered vLLM and HuggingFace internals.

Everything in this package pokes at private structure that moves between vLLM releases. When a vLLM
upgrade breaks the plugin, this is where to look first. See `VERSION_NOTES.md` for the verified
version.

Verbatim port of `gate_probe._find_transformer_layers`, `_find_embed_tokens_module` and
`_load_worker_tokenizer`.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from ..families import ModelFamily
from ..telemetry.log import log_cybergate_event

__all__ = ["find_embed_tokens_module", "find_transformer_layers", "load_worker_tokenizer"]

#: A module needs at least this many sub-layers to be believed to be the decoder stack, when falling
#: back to scanning `named_modules()`.
_MIN_PLAUSIBLE_LAYERS = 16


def find_transformer_layers(model: Any) -> Any:
    """Locate the decoder `.layers` sequence across vLLM and HF model wrappers.

    Tries the known attribute paths in order, then falls back to scanning every submodule for one
    with a plausibly long `.layers`. Raises `RuntimeError` when nothing matches, because attaching the
    gate to the wrong module would silently score the wrong activations.
    """
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
            if (
                hasattr(mod, "layers")
                and hasattr(mod.layers, "__len__")
                and len(mod.layers) >= _MIN_PLAUSIBLE_LAYERS
            ):
                return mod.layers
    raise RuntimeError(f"Cannot locate transformer `.layers` on model {type(model).__name__}")


def find_embed_tokens_module(model: Any) -> Any | None:
    """Locate the input token embedding module, or return `None`.

    The gate needs a pre-hook here to capture the exact `input_ids` of each forward batch: the decoder
    layers see only hidden states, so without this the turn markers cannot be found.
    """
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


def load_worker_tokenizer(family: ModelFamily) -> Any | None:
    """Load a fast HF tokenizer inside the worker process, to decode exact user token spans.

    Searched in order: `CYBERGATE_TOKENIZER_PATH`, then `family.tokenizer_globs` (newest snapshot
    first), then `family.tokenizer_dirs`. Returns `None` on any failure -- the hook then still scores,
    but cannot compute the text hashes that key a decision record, so no record is published.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        candidates: list[str] = []
        env_path = os.environ.get("CYBERGATE_TOKENIZER_PATH")
        if env_path:
            candidates.append(env_path)
        for pattern in family.tokenizer_globs:
            candidates.extend(sorted(glob.glob(pattern), reverse=True))
        candidates.extend(family.tokenizer_dirs)

        for candidate in candidates:
            if (Path(candidate) / "tokenizer.json").exists():
                try:
                    tok = AutoTokenizer.from_pretrained(candidate)
                    log_cybergate_event(
                        "CYBERGATE-TOKENIZER", f"Loaded worker tokenizer from {candidate}"
                    )
                    return tok
                except Exception:
                    continue
    except Exception as exc:
        log_cybergate_event("CYBERGATE-TOKENIZER-WARN", f"Tokenizer load skipped: {exc}")
    return None
