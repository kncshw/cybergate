"""Kit binding checks: does this operator belong to this family, and to this checkpoint?

Nothing in the canonical package bound an operator to the model it was fitted on. `CYBERGATE_OP_PATH`
accepts any path, `load_cybergate_op` reads whatever is there, and `install_category_probe_hook`
clamps an out-of-range `gate_layer` with `min(op.gate_layer, len(layers) - 1)`. So a mismatched
operator loads, attaches, scores every request and simply stops firing. There is no error, no warning
and no entry in the decision record -- the gate just goes quiet.

That is not hypothetical. The shipped Qwen operator cleared its own threshold on 18 of its own 100
fitted training positives on the served checkpoint, and the only reason anyone found out was a
recall measurement that looked too low.

Three failure modes, and which check covers which:

  * **Wrong architecture** (a 2816-dim Gemma direction under the Qwen family). Caught here, from
    metadata alone, by `check_operator`.
  * **Wrong layer** (an L10 operator where the family ships L16 -- what a stale `default_op_names`
    ladder produced). Caught here by `check_operator`, for the category operator only: a refusal
    direction is `last`-pooled at its own layer by design and is exempt (`REFUSAL_CATEGORY`).
  * **Right architecture, wrong checkpoint.** NOT caught here and not catchable from metadata: the
    dimensions match and the layer matches. It needs the operator run against the live model, which
    is `kits/<name>/canaries.json` and `scripts/serving/check_eval.sh`.

Standard library only; no torch, no I/O. `check_model` is separate because it needs the loaded model's
layer count, which only the worker process has.
"""
from __future__ import annotations

from ..config import env_first
from ..families import ModelFamily
from .operator import CyberGateOperator

__all__ = ["KitMismatch", "check_model", "check_operator", "enforce", "skip_requested"]

#: Set `CYBERGATE_SKIP_KIT_CHECK=1` to downgrade every mismatch to a logged warning. The escape
#: hatch exists so a declared value that turns out to be wrong cannot brick a working deployment;
#: it is not meant for ordinary use, and `enforce` names it in the exception text.
SKIP_ENV = "CYBERGATE_SKIP_KIT_CHECK"

#: `meta.category` of a refusal direction. A refusal direction is loaded by `load_cybergate_op`
#: like a category operator (`ops/schema.md`, "Refusal direction") but it is not one: it is pooled
#: `last` at its own layer, well above the gate. Comparing it against the family's gate is a
#: category error, and `enforce` raising on it took `install_refusal_monitor_hook` down with it --
#: which removes the gate's second factor and every mid/late rescue, leaving nothing in the
#: decision record to say so. A refusal direction that omits `meta.category` still trips the gate
#: check, because the loader defaults the field to `cyber_offense`.
REFUSAL_CATEGORY = "refusal_monitor"


class KitMismatch(RuntimeError):
    """An operator contradicts the family it was loaded under."""


def skip_requested() -> bool:
    return env_first(SKIP_ENV) == "1"


def check_operator(op: CyberGateOperator, family: ModelFamily) -> list[str]:
    """Return one human-readable string per mismatch between `op` and `family`'s declaration.

    Empty list means every declared field agrees. A field the family leaves at its neutral value
    (`hidden_size`/`num_layers` of 0, `expected_gate` of `None`) is not checked, so an unmeasured
    family reports nothing rather than guessing.
    """
    out: list[str] = []

    if family.hidden_size and op.dim != family.hidden_size:
        out.append(
            f"operator vector is {op.dim}-dimensional but family {family.name!r} was fitted on a "
            f"{family.hidden_size}-dim checkpoint"
        )

    if family.num_layers and op.gate_layer >= family.num_layers:
        out.append(
            f"operator declares gate_layer {op.gate_layer}, which does not exist on a "
            f"{family.num_layers}-layer checkpoint (the hook would silently clamp it to "
            f"{family.num_layers - 1})"
        )

    # Only the category operator drives the gate, so only it is held to the declared gate. The
    # width and layer-count checks above apply to both kinds of artifact.
    if family.expected_gate is not None and op.category != REFUSAL_CATEGORY:
        want_layer, want_pool = family.expected_gate
        if (op.gate_layer, op.pool) != (want_layer, want_pool):
            out.append(
                f"operator is L{op.gate_layer} {op.pool!r} but family {family.name!r} ships "
                f"L{want_layer} {want_pool!r}"
            )

    return out


def check_model(op: CyberGateOperator, family: ModelFamily, total_layers: int) -> list[str]:
    """Return one string per mismatch between the loaded model and `family`'s declaration.

    Separate from `check_operator` because `total_layers` comes from the model object, which exists
    only in the worker process. This is the check that distinguishes "the operator is self-consistent"
    from "the operator matches what is actually loaded".
    """
    out: list[str] = []

    if family.num_layers and total_layers != family.num_layers:
        out.append(
            f"loaded model has {total_layers} decoder layers but family {family.name!r} declares "
            f"{family.num_layers}; the served checkpoint is not the one these artifacts were fitted on"
        )

    if op.gate_layer >= total_layers:
        out.append(
            f"operator declares gate_layer {op.gate_layer} but the loaded model has only "
            f"{total_layers} layers; the hook would clamp to {total_layers - 1} and score the wrong layer"
        )

    return out


def enforce(problems: list[str], where: str) -> list[str]:
    """Raise `KitMismatch` for `problems`, or return them unchanged when the skip flag is set.

    Returning the list on the skip path lets the caller log it; an empty list is always returned
    unchanged, so the caller needs no special case for the healthy path.
    """
    if not problems or skip_requested():
        return problems
    detail = "; ".join(problems)
    raise KitMismatch(
        f"{where}: {detail}. "
        f"This is a kit binding check -- the artifacts do not match the model. Fix the operator "
        f"path or the family declaration; set {SKIP_ENV}=1 only to override deliberately."
    )
