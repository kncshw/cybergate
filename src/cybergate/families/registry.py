"""Family lookup. One shared stateless instance per family.

The canonical package dispatches on family in two different ways, and both are kept here as separate
functions because they are not the same predicate:

  * serving path -- `detect_model_family(model_id)` in `prompts.py` tests substrings of the request's
    `model` field.
  * worker path -- `install_vllm_read_hook` and friends test `op.model_key.startswith("gemma")`,
    where `model_key` comes from the operator file's `meta`.

The canonical `else` arm for both was Qwen-3.6, so an unrecognised model id was scored with Qwen's
chat template and token markers. **The two paths no longer agree on that**, and the difference is
deliberate:

  * `family_for_model_id` (serving) does NOT fall back. It resolves from the configured operator and
    otherwise raises `UnknownModelFamily`, because this path chooses the render the prompt is built
    with and guessing it wrong is silent -- see that function.
    `tests/equivalence/test_prompts_differential.py` asserts the **divergence** from the vendored
    baseline, using model ids with no family of their own.
  * `family_for_model_key` (worker) still falls back to `DEFAULT_FAMILY`. That case is already
    fail-closed upstream: `ops.kit.check_operator` rejects an operator whose dimensions do not match
    the served weights before its `model_key` is ever consulted.

Only families that are served and measured get a class. Phi-4 and Llama-3.1 had classes in earlier
revisions, shipped no kit, and were removed.
"""
from __future__ import annotations

from .base import ModelFamily
from .gemma4.family import Gemma4Family
from .gemma4_31b.family import Gemma431BFamily
from .qwen36.family import Qwen36Family
from .qwen38.family import Qwen38Family

__all__ = [
    "ALL_FAMILIES",
    "DEFAULT_FAMILY",
    "UnknownModelFamily",
    "family_by_name",
    "family_for_configured_operator",
    "family_for_model_id",
    "family_for_model_key",
]


class UnknownModelFamily(LookupError):
    """A model id matched no family, and no configured operator could name one."""

GEMMA4_31B = Gemma431BFamily()
GEMMA4 = Gemma4Family()
QWEN36 = Qwen36Family()

QWEN38 = Qwen38Family()

#: Checked in order, so the more specific families come before the broader fallbacks. Both lookups
#: take the FIRST match: `gemma4_31b` must precede `gemma4` (which claims `"gemma-4"` / `"gemma"`),
#: and `qwen38` must precede `qwen36` (which claims `"qwen"`).
ALL_FAMILIES: tuple[ModelFamily, ...] = (GEMMA4_31B, GEMMA4, QWEN38, QWEN36)

#: The canonical `else` arm. Retained because `family_for_model_key` still falls back to it -- an
#: operator naming an unknown `model_key` is already fail-closed, since `ops.kit.check_operator`
#: rejects it on dimensions. `family_for_model_id` no longer uses it; see that function.
DEFAULT_FAMILY: ModelFamily = QWEN36


#: Cache for `family_for_configured_operator`, keyed on the `CYBERGATE_OP_PATH` it resolved from.
#: The resolution reads a file, and this sits on the request path.
_OP_FAMILY_CACHE: dict[str, ModelFamily | None] = {}


def family_for_configured_operator() -> ModelFamily | None:
    """The family named by the operator this process is configured to serve, or `None`.

    Reads `CYBERGATE_OP_PATH` only -- never the family default ladders, which would recurse. The
    operator's `meta.model_key` is the authoritative statement of which architecture it was fitted
    for, and `ops.kit.check_operator` has already validated it against the served weights, so it is a
    better answer than a substring of the request's `model` field.
    """
    from ..config import env_first

    op_path = env_first("CYBERGATE_OP_PATH")
    if not op_path:
        return None
    if op_path in _OP_FAMILY_CACHE:
        return _OP_FAMILY_CACHE[op_path]

    resolved: ModelFamily | None = None
    try:
        # Imported here, not at module scope: `ops.loader` imports this module.
        from ..ops import load_cybergate_op

        model_key = load_cybergate_op(op_path).model_key
        for fam in ALL_FAMILIES:
            if fam.matches_model_key(model_key):
                resolved = fam
                break
    except Exception:
        # A missing or malformed operator is not this function's problem to report; the loader and
        # the kit check both raise with a far better message than anything available here.
        resolved = None
    _OP_FAMILY_CACHE[op_path] = resolved
    return resolved


def family_for_model_id(model_id: str) -> ModelFamily:
    """Resolve the family from an OpenAI `model` field.

    Order:
      1. a family whose `model_id_substrings` match,
      2. the family named by the configured operator (`CYBERGATE_OP_PATH`),
      3. `UnknownModelFamily`.

    **Deliberate divergence from the canonical package**, which fell back to Qwen-3.6 for anything it
    did not recognise. That fallback was silent and wrong in a way that mattered: this function picks
    the chat template the prompt is rendered with (`serving/middleware.py`), so a Gemma server reached
    with `--served-model-name prod`, or serving from a local checkpoint path, rendered every request
    with Qwen's markers. Step 2 makes the aliased case work properly instead of guessing, and step 3
    makes the remaining case loud.
    """
    for fam in ALL_FAMILIES:
        if fam.matches_model_id(model_id):
            return fam
    from_op = family_for_configured_operator()
    if from_op is not None:
        return from_op
    raise UnknownModelFamily(
        f"model id {model_id!r} matches no family "
        f"({', '.join(f.name for f in ALL_FAMILIES)}) and CYBERGATE_OP_PATH names none. "
        f"Serve a model whose id identifies its family, or point CYBERGATE_OP_PATH at the "
        f"operator for the model being served (see kits/<name>/MANIFEST.json for its model_ids)."
    )


def family_for_model_key(model_key: str) -> ModelFamily:
    """Resolve the family from an operator's `meta.model_key`. Falls back to Qwen-3.6."""
    for fam in ALL_FAMILIES:
        if fam.matches_model_key(model_key):
            return fam
    return DEFAULT_FAMILY


def family_by_name(name: str) -> ModelFamily:
    """Resolve a family by its own `name` (`gemma4`, `qwen36`).

    Raises `KeyError` for an unknown name. `family_for_model_id` also raises, with its own message;
    `family_for_model_key` is the only lookup here that falls back.
    """
    for fam in ALL_FAMILIES:
        if fam.name == name:
            return fam
    raise KeyError(f"Unknown model family: {name!r}")
