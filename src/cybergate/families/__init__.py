"""Per-architecture parameters and the algorithms that consume them.

`base.ModelFamily` holds the common code; each concrete family lives in its own folder and sets only
its own values. Import families through `registry`, not directly, so a single shared instance is used.
"""
from __future__ import annotations

from .base import MidLateHits, MidLateVeto, ModelFamily
from .registry import (
    ALL_FAMILIES,
    DEFAULT_FAMILY,
    UnknownModelFamily,
    family_by_name,
    family_for_configured_operator,
    family_for_model_id,
    family_for_model_key,
)

__all__ = [
    "ALL_FAMILIES",
    "DEFAULT_FAMILY",
    "MidLateHits",
    "MidLateVeto",
    "ModelFamily",
    "UnknownModelFamily",
    "family_by_name",
    "family_for_configured_operator",
    "family_for_model_id",
    "family_for_model_key",
]
