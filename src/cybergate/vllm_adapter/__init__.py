"""QUARANTINE: version-fragile vLLM and HuggingFace internals.

Isolated here so a vLLM upgrade has one place to break. `VERSION_NOTES.md` records the version these
were verified against.
"""
from __future__ import annotations

from .batch import extract_batch_slices
from .introspect import find_embed_tokens_module, find_transformer_layers, load_worker_tokenizer

__all__ = [
    "extract_batch_slices",
    "find_embed_tokens_module",
    "find_transformer_layers",
    "load_worker_tokenizer",
]
