"""How the monitor hook hands its refusal reading to the CoT controller.

## The constraint

vLLM's V1 logits-processor API tells a processor about a request as

    AddedRequest = tuple[index, SamplingParams, prompt_tok_ids | None, output_tok_ids]

with **no request id**. So the controller cannot look up state the forward hooks keyed by vLLM
request id. The two sides must meet on something both can compute.

## What was tried first, and why it failed

The first design joined on the **batch row index** within one forward pass. It was verified live on
2026-10-28 and **does not work**. Three identical prompts submitted concurrently, identical text,
identical `s_ref_z = 6.438`, temperature 0, received **2, 0 and 1** injections; reversing submission
order moved the steering with the rank rather than the prompt; and with two requests, rank 0 at
z=6.44 was steered while rank 1 at the *higher* z=7.84 was not.

The reason: the controller's index is a slot in vLLM's **persistent batch**, while the hook's index is
a position within the **requests scheduled in that forward pass**. `query_start_loc` covers only the
scheduled subset, packed contiguously. The two coincide only when a single request is resident --
which is exactly why every sequential test passed and every concurrent one lost steering.

## What this does instead: join on identity

The key is a hash of the request's prompt token ids, which both sides can compute:

    prefill   hook sees the slice's token ids and the request id
              -> records req_id -> key
    any pass  hook publishes the reading under that key
    sampling  controller computed the same key from `prompt_tok_ids` at `added`, and reads it

## Where position re-entered, and how it was removed

An earlier revision of this docstring claimed "position never enters it, so batch composition and
scheduling order are irrelevant." **That was false.** No request id is reachable in this vLLM build's
forward context, so the hook can only name a slot by its row (`batch_seq_<n>`) -- position, under a
different name. Binding happened on prefill only, and nothing updated it when a request finished and
vLLM compacted the remaining rows down. The producer's 2026-10-28 review proved the consequence: a
short request in row 0 finished, rows shifted, and row 0's new occupant published its readings under
the departed request's key for 165 decode steps, driving `unresolved` to 82.

The fix uses the one component that *does* track row lifetime correctly. `RefusalSteeredCoTProcessor`
receives vLLM's own `removed`/`added`/`moved` events, so it always knows which key each row carries;
it now pushes that map here through `sync_bindings` on every batch update. vLLM calls
`update_state` before the forward pass of the same step (`_update_states` -> `refresh_metadata` ->
`logit_proc.update_state`, then `_prepare_inputs`, then the model call), so the table the hook reads
during a pass is the one the sampler just wrote for that pass.

`bind_request` remains for the case where no controller is running -- with steering off nothing reads
a binding, so a prefill-only binding is harmless there.

Two further guards:

* **Generation tagging.** Every reading carries the forward pass it was taken in, and `read` refuses
  one from an earlier pass. A stale reading could describe a different token.
* **A sole-sequence key.** When a forward pass holds exactly one sequence, vLLM reports no request
  id, so no `req_id -> key` mapping can be formed. The hook then publishes under `SOLE_KEY`, and the
  controller falls back to it **only when it is tracking exactly one row**. That preserves the
  single-request path, which is the one the live run proved works.

## Known limitation, deliberately observable rather than silent

With prefix caching, a prefill may present only the uncached suffix, so the hook's key will not match
the controller's full-prompt key and no reading is found. The controller then does not steer, and
increments `unresolved` -- see `SteeringStats`. The previous design failed the same way but silently,
which is what made the row-join defect take a live concurrency experiment to find.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = [
    "SOLE_KEY",
    "RowReading",
    "begin_pass",
    "bind_request",
    "current_generation",
    "key_for_tokens",
    "key_of_slot",
    "publish",
    "read",
    "register_cudagraph_harvester",
    "reset",
    "slot_name",
    "stats",
    "sync_bindings",
    "sync_from_cudagraph",
]

#: Key used when a forward pass holds a single sequence and vLLM reports no request id.
SOLE_KEY = "sole"

#: Bound on the `req_id -> key` map, so a long-lived engine cannot grow it without limit.
_MAX_BINDINGS = 4096


@dataclass(frozen=True)
class RowReading:
    """One request's refusal reading, tagged with the pass it came from."""

    rolling_s_ref_z: float
    s_ref_z: float
    #: Tokens in this request's slice for that pass. 1 during decode, more during prefill.
    seq_len: int
    generation: int
    #: True when `rolling_s_ref_z` was formed with the **decode-domain** calibration, so it can be
    #: compared against `tau_spike`. False means the decode calibration is missing and the value is
    #: in prefill units -- the controller must not steer on it. See `steering_config`.
    decode_domain: bool = False


_LOCK = threading.Lock()
_READINGS: dict[str, RowReading] = {}
_BINDINGS: dict[str, str] = {}  # slot -> key
_GENERATION = 0
_PUBLISHED = 0
#: Set once `sync_bindings` has been called, i.e. a controller is driving the table. While set,
#: `bind_request` stands down -- see its docstring for why that matters under prefix caching.
_CONTROLLER_MANAGED = False
_CUDAGRAPH_HARVESTER: Callable[[Sequence[int]], None] | None = None


def register_cudagraph_harvester(harvester: Callable[[Sequence[int]], None] | None) -> None:
    """Register a callback that harvests CUDA-graph-captured decode readings into this trace."""
    global _CUDAGRAPH_HARVESTER
    with _LOCK:
        _CUDAGRAPH_HARVESTER = harvester


def sync_from_cudagraph(active_rows: Sequence[int]) -> None:
    """If the last decode step ran via CUDA graph replay, harvest its GPU cosine buffer."""
    harvester = _CUDAGRAPH_HARVESTER
    if harvester is not None and active_rows:
        harvester(active_rows)


def key_for_tokens(token_ids: Sequence[int]) -> str:
    """The join key for a sequence of prompt token ids.

    A plain hash of the id tuple. Two byte-identical prompts collide, and that is acceptable: the
    same tokens produce the same activations, so the readings are interchangeable. Any *different*
    prompt gets a different key, which is the property the join needs.
    """
    return f"t{len(token_ids)}:{hash(tuple(token_ids)) & 0xFFFFFFFFFFFF:012x}"


def begin_pass() -> int:
    """Start a new forward pass and return its generation number."""
    global _GENERATION
    with _LOCK:
        _GENERATION += 1
        return _GENERATION


def current_generation() -> int:
    with _LOCK:
        return _GENERATION


def bind_request(slot: str, key: str) -> None:
    """Bind batch `slot` to prompt-token key `key`, learned from a prefill in that slot.

    **Rebinds every time.** The first version refused to overwrite an existing binding, which was
    wrong for a decisive reason: `slot` is a *position* (`batch_seq_<n>`), not a request identity --
    request ids are not reachable anywhere in this vLLM build's forward context. So a slot is reused
    by request after request, and a write-once binding pinned it to whichever prompt happened to
    prefill there first. Every later occupant then published its reading under a foreign key and its
    own key was never written, so the controller found nothing.

    Measured on 2026-10-28: four brand-new prompts submitted against already-bound slots got 0
    injections and `unresolved` rose by 448, with prefill lines showing a 112-token slice publishing
    under a `t113` key belonging to the *previous* trial's prompts.

    A binding is therefore valid only for the occupant that most recently prefilled in that slot,
    which is exactly as long as it needs to be: a request prefills once and then decodes.

    **No-op once a controller is driving the table** (`sync_bindings` has been called). Two reasons,
    and the second is the one that bites: the controller's map is correct across row compaction and
    this one is not, and the controller's key is hashed over the *whole* prompt while the hook's is
    hashed over the tokens it was shown. Under prefix caching those differ, so a prefill binding
    here would overwrite a usable key with a truncated one.
    """
    with _LOCK:
        if _CONTROLLER_MANAGED:
            return
        if slot not in _BINDINGS and len(_BINDINGS) >= _MAX_BINDINGS:
            _BINDINGS.pop(next(iter(_BINDINGS)), None)
        _BINDINGS[slot] = key


def sync_bindings(mapping: dict[str, str]) -> None:
    """Replace the whole slot -> key table with `mapping`, the controller's authoritative view.

    Called from `RefusalSteeredCoTProcessor.update_state`, which is the only component that sees
    vLLM's row `removed`/`added`/`moved` events and therefore the only one that can keep a
    positional table correct across batch compaction. See the module docstring.

    Replacement, not merge: a row the controller no longer tracks must lose its binding, or the hook
    keeps publishing a finished request's readings under a key the controller may later re-adopt.
    """
    global _CONTROLLER_MANAGED
    with _LOCK:
        _CONTROLLER_MANAGED = True
        _BINDINGS.clear()
        _BINDINGS.update(mapping)


def key_of_slot(slot: str | None) -> str | None:
    """The key bound to `slot`, or `None` if no prefill has been observed there."""
    if slot is None:
        return None
    with _LOCK:
        return _BINDINGS.get(slot)


def slot_name(row: int) -> str:
    """The binding slot for batch row `row`.

    A single-sequence pass reports `req_id=None`, so without this a request that prefilled **alone**
    was never bound at all -- binding required a non-None id -- and its reading was published under
    no key once other requests joined it. Measured: `bindings=3` for 4 requests, 17 `unbound` lines,
    and the unbound request got 0 injections while the other three got 2. Naming the slot from the
    row removes the special case: every prefill binds, alone or packed.
    """
    return f"batch_seq_{row}"


def publish(key: str, reading: RowReading) -> None:
    """Record a reading under `key` for the current pass."""
    global _PUBLISHED
    with _LOCK:
        _READINGS[key] = reading
        _PUBLISHED += 1


def read(key: str, max_age: int = 0) -> RowReading | None:
    """Return `key`'s reading if it is from the current pass, else `None`.

    `max_age` permits a reading up to that many passes old. The default of 0 means this pass only:
    an older reading may describe a different token in the same stream.
    """
    with _LOCK:
        reading = _READINGS.get(key)
        if reading is None:
            return None
        if _GENERATION - reading.generation > max_age:
            return None
        return reading


def stats() -> dict[str, int]:
    """Counters for tests and diagnostics."""
    with _LOCK:
        return {
            "generation": _GENERATION,
            "readings": len(_READINGS),
            "bindings": len(_BINDINGS),
            "published": _PUBLISHED,
            "controller_managed": int(_CONTROLLER_MANAGED),
        }


def reset() -> None:
    """Drop all state. For tests."""
    global _GENERATION, _PUBLISHED, _CONTROLLER_MANAGED, _CUDAGRAPH_HARVESTER
    with _LOCK:
        _READINGS.clear()
        _BINDINGS.clear()
        _GENERATION = 0
        _PUBLISHED = 0
        _CONTROLLER_MANAGED = False
        _CUDAGRAPH_HARVESTER = None
