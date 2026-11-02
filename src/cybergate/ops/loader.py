"""Operator loading from `.json` or PyTorch `.pt`. The only I/O in `ops/`.

Verbatim port of `gate_probe.load_cybergate_op` and `gate_probe._load_pt_stdlib`. The `.pt` path
imports torch when it is available and falls back to a standard-library unpickler when it is not, so
this module stays importable in the API-server process.
"""
from __future__ import annotations

import json
import math
import pickle
import struct
import zipfile
from pathlib import Path
from typing import Any

from ..config import KITS_DIR, OPS_DIR, OPS_DIR_IS_EXPLICIT, env_first
from ..families import ModelFamily, family_by_name, family_for_model_key
from ..telemetry.log import log_cybergate_event
from .kit import check_operator, enforce
from .operator import CyberGateOperator

__all__ = ["artifact_dirs", "default_op_path", "load_cybergate_op", "resolve_ref_dir_path"]


def _load_pt_stdlib(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch `.pt` zip archive using only Python's standard library.

    Reads the pickle's tensor storages directly out of the zip and rebuilds them as flat float lists.
    Only `_rebuild_tensor_v2` is honoured; any other `torch.*` global is stubbed to an empty type, so
    an unexpected archive layout fails rather than executing anything.
    """
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


def artifact_dirs(family: ModelFamily) -> tuple[Path, ...]:
    """Directories to search for `family`'s artifacts, most preferred first.

    The family's kit comes first, so a deployment that configures nothing resolves the operator we
    actually ship. An explicit `CYBERGATE_OPS_DIR` suppresses the kit entirely: asking for a
    specific directory and silently getting a different one would be worse than either.
    """
    if OPS_DIR_IS_EXPLICIT or not family.kit_name:
        return (OPS_DIR,)
    return (KITS_DIR / family.kit_name, OPS_DIR)


def default_op_path(family: ModelFamily) -> Path:
    """Return `family`'s default operator path: the first name that exists, else the last name.

    "Else the last name" reproduces the canonical module-level ladder
    (`DEFAULT_GEMMA4_OP = L10 if L10.exists() else L8`), so a broken directory raises
    `FileNotFoundError` naming the same file it names today.
    """
    names = family.default_op_names
    if not names:
        raise KeyError(f"Family {family.name!r} declares no default operator")
    dirs = artifact_dirs(family)
    for name in names:
        for directory in dirs:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return dirs[0] / names[-1]


def resolve_ref_dir_path(family: ModelFamily) -> Path | None:
    """Return `family`'s refusal-direction file if it is configured and present, else `None`."""
    if not family.default_ref_dir_name:
        return None
    for directory in artifact_dirs(family):
        candidate = directory / family.default_ref_dir_name
        if candidate.exists():
            return candidate
    return None


def load_cybergate_op(
    path: str | Path | None = None,
    model_family: str = "gemma4",
) -> CyberGateOperator:
    """Load a `CyberGateOperator` from `.json` or `.pt`.

    Resolution order for `path is None`, unchanged from the canonical package:
      1. `CYBERGATE_OP_PATH`, when set and the file exists,
      2. the family default for `model_family`.

    `model_family` accepts either a family name (`gemma4`) or an operator `model_key` (`gemma`), so
    both the serving and worker call sites keep working.
    """
    if path is None:
        env_op = env_first("CYBERGATE_OP_PATH")
        if env_op and Path(env_op).exists():
            path = env_op
        else:
            path = default_op_path(_resolve_family(model_family))
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CyberGate operator file not found: {p}")

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        try:
            import torch  # type: ignore

            loaded = torch.load(str(p), map_location="cpu", weights_only=False)
            c_vec = (
                loaded["c"].detach().float().cpu().tolist()
                if hasattr(loaded["c"], "detach")
                else list(loaded["c"])
            )
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
    # Gate layer, in order: the operator's own metadata, then a bare `layer` key, then the family's
    # declared `expected_gate`. The last resort used to be `10 if len(c_unit) == 2816 else 15` --
    # the gate layer inferred from the vector's width, which hardcodes both families' dimensions and
    # silently returns Gemma's old L10 for any 2816-dim model. Every shipped operator records
    # `meta.gate_layer`, so the fallbacks are for hand-made and legacy `.pt` files.
    gate_layer_meta = meta.get("gate_layer", raw.get("layer"))
    if gate_layer_meta is None:
        expected = _resolve_family(model_family).expected_gate
        gate_layer_meta = expected[0] if expected else 0
    gate_layer = int(gate_layer_meta)
    pool = str(meta.get("pool", "mean_user"))
    category = str(meta.get("category", "cyber_offense"))
    model_key = str(meta.get("model_key") or meta.get("model_tag") or model_family)

    op = CyberGateOperator(
        c=c_unit,
        thresh=thresh,
        gate_layer=gate_layer,
        pool=pool,
        category=category,
        model_key=model_key,
        meta=meta,
    )

    # Kit binding, checked against the family the operator names ITSELF -- not against the
    # `model_family` argument. That argument is only a resolution hint for `path is None`, and the
    # worker path leaves it at its "gemma4" default while loading a Qwen operator, so checking
    # against it would false-positive on every Qwen load.
    problems = check_operator(op, family_for_model_key(op.model_key))
    if problems:
        log_cybergate_event(
            "CYBERGATE-KIT-MISMATCH", f"{p.name}: {'; '.join(problems)}"
        )
    enforce(problems, f"operator {p.name}")
    return op


def _resolve_family(model_family: str) -> ModelFamily:
    """Accept a family `name` or an operator `model_key` and return the family."""
    try:
        return family_by_name(model_family)
    except KeyError:
        return family_for_model_key(model_family)

