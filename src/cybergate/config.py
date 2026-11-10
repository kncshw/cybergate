"""Constants, environment variables and paths. Standard library only.

Importable in both processes (`MainProcess` API server and `EngineCore` model worker). This module
must never import torch, vllm, httpx or any model-family module, because the API-server process pays
for every import the plugin entry point triggers.

Every constant here is a verbatim port of the value in the canonical `gate_probe.py`; the comment
above each one records where the number came from. Family-specific values (revoke ceilings,
refusal-monitor layers, turn markers) do NOT live here -- they live on the `ModelFamily` subclass in
`cybergate/families/<family>/`.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REF_MAINT_Z_THRESH",
    "DEFAULT_ST_REF_Z_PIVOT",
    "IN_CONTEXT_PIVOT_FLOOR",
    "IPC_DIR_PATH",
    "KITS_DIR",
    "OPS_DIR",
    "OPS_DIR_IS_EXPLICIT",
    "PACKAGE_DIR",
    "PLUGIN_EVENTS_LOG_PATH",
    "PRECOMPUTED_SCORES_PATH",
    "S_CAT_LAST_PIVOT_FLOOR",
    "STEER_AUTH_XARG",
    "SUPPRESS_OPENER_XARG",
    "env_first",
]


def env_first(*names: str, default: str | None = None) -> str | None:
    """Return the first set, non-empty environment variable among `names`.

    Every setting is read as `CYBERGATE_<X>`. The canonical package also honoured a pre-rename
    `RESGATE_<X>` twin for each one, repeated at ~20 call sites; those twins are gone, so a
    deployment still exporting them must be updated to the `CYBERGATE_` names.
    """
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


PACKAGE_DIR = Path(__file__).resolve().parent

# The artifact directory. `CYBERGATE_OPS_DIR` is how a deployment names one -- the launch scripts
# point it at a kit -- and it is how the tests point at fixtures. The repo-relative fallback keeps
# the baseline layout's shape (`<repo>/ops`, package at `<repo>/cybergate_plugin`; here the package
# is `<repo>/src/cybergate`, so it is two levels up). That directory no longer exists in the
# repository: nothing servable lives outside `kits/`, so the fallback resolves to a missing path and
# every consumer of it already tolerates absence.
OPS_DIR = Path(
    env_first("CYBERGATE_OPS_DIR") or str(PACKAGE_DIR.parent.parent / "ops")
)

#: True when `CYBERGATE_OPS_DIR` was set explicitly. An explicit override must win outright -- it is
#: how the tests point at fixtures and how a deployment pins one directory -- so artifact resolution
#: consults `KITS_DIR` only when nobody asked for a specific directory.
OPS_DIR_IS_EXPLICIT = bool(env_first("CYBERGATE_OPS_DIR"))

#: Per-model deployment kits, `kits/<family.kit_name>/`. Searched BEFORE `OPS_DIR` so a deployment
#: that sets no environment at all still resolves the shipped operator. This ordering is why Gemma's
#: default no longer silently resolves to L10, or Qwen's to L15: back when superseded per-layer
#: operators sat in `ops/`, the ladder picked whichever existed.
KITS_DIR = Path(
    env_first("CYBERGATE_KITS_DIR") or str(PACKAGE_DIR.parent.parent / "kits")
)

PRECOMPUTED_SCORES_PATH = OPS_DIR / "precomputed_l8_scores.json"

# Cross-process decision records. The two processes share no memory, so every decision that has to
# cross the MainProcess <-> EngineCore boundary goes through this directory. See
# docs/PROCESS_TOPOLOGY.md.
IPC_DIR_PATH = Path(
    env_first("CYBERGATE_IPC_DIR", default="/dev/shm/cybergate_ipc") or ""
)

PLUGIN_EVENTS_LOG_PATH = Path(
    env_first("CYBERGATE_EVENTS_LOG_PATH")
    or str(PACKAGE_DIR.parent.parent / "logs" / "cybergate_plugin_events.log")
)

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"

#: `vllm_xargs` key with which a completion request declines the force-emitted `POST_THINK_OPENER`.
#: It is a wire constant shared by two processes -- the serving path sets it on the payload, the CoT
#: controller reads it off `SamplingParams.extra_args` in the engine -- so it lives here rather than
#: in either of them. See `hooks.cot_controller` for why a tool-bearing turn declines the opener.
SUPPRESS_OPENER_XARG = "cybergate_suppress_opener"

#: `vllm_xargs` key carrying the per-server steering secret (`CYBERGATE_STEER_TOKEN`). Set by
#: `register_cybergate_plugin` in `MainProcess` before `EngineCore` spawns, stripped from all
#: external client payloads by `strip_cybergate_client_kwargs`, and attached only by CyberGate's
#: steered routes after the activation gate fires.
STEER_AUTH_XARG = "cybergate_steer_token"

# --- Policy thresholds that are NOT family-specific in the baseline -------------------------------
#
# Refusal-monitor maintenance floor, in calibrated sigma at the refusal layer. One value for every
# family in the baseline (`DEFAULT_REF_MAINT_Z_THRESH = 1.50`). It does double duty: it is both the
# multi-turn maintenance floor (Step 2B) and the second factor of the single-turn two-factor gate.
DEFAULT_REF_MAINT_Z_THRESH = 1.50

# Step 2A non-target-harm pivot disjuncts. These three are hard-coded literals in the baseline
# (`gate_probe.py:1430-1435`) rather than named constants, and they are NOT family-parameterised
# there either -- the Gemma-4 numbers are applied to every family.
IN_CONTEXT_PIVOT_FLOOR = 0.0750
S_CAT_LAST_PIVOT_FLOOR = 0.0350
DEFAULT_ST_REF_Z_PIVOT = 7.0
