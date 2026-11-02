"""The contract vLLM imposes on a registered logits processor.

**Why this file exists.** The controller's 29 logic tests all passed while the processor made the
server return HTTP 500 for *every* request. They construct the processor directly and exercise its
three instance methods; they never go through vLLM's admission path. vLLM calls
`validate_params()` -- a fourth, non-abstract classmethod resolved off the `LogitsProcessor` ABC --
on every registered processor class for every request. A class that implements only the three
instance methods has no such attribute, and `SamplingParams.verify` raises before the forward pass,
with an empty error message and no traceback.

So: thorough tests of a plugin's logic say nothing about whether the host will accept the plugin.
These tests cover the host's side of the contract instead.

They skip where vLLM is absent, which is the normal offline case. They run inside the serving
container, which is where they matter. Verified there on 2026-10-28:

    issubclass(LogitsProcessor) = True
    has validate_params        = True
    validate_params(...)       = OK
    entry-point discovered     = ['RefusalSteeredCoTProcessor']
"""
from __future__ import annotations

import pytest

vllm_logitsprocs = pytest.importorskip(
    "vllm.v1.sample.logits_processor",
    reason="vLLM is not installed; this contract is checked inside the serving container",
)

from cybergate.hooks.cot_controller import RefusalSteeredCoTProcessor  # noqa: E402


def test_is_a_subclass_of_the_vllm_abc() -> None:
    """The base class is load-bearing, not decorative -- see the module docstring."""
    assert issubclass(RefusalSteeredCoTProcessor, vllm_logitsprocs.LogitsProcessor)


def test_exposes_validate_params() -> None:
    """vLLM resolves this classmethod on every registered class, for every request."""
    assert hasattr(RefusalSteeredCoTProcessor, "validate_params")


def test_validate_params_accepts_ordinary_sampling_params() -> None:
    """The call vLLM makes during request admission must not raise."""
    from vllm.sampling_params import SamplingParams

    RefusalSteeredCoTProcessor.validate_params(SamplingParams(max_tokens=8))


def test_declares_it_is_not_argmax_invariant() -> None:
    """Forcing a token changes the argmax. vLLM uses this to reject the processor under
    speculative decoding, which is the correct outcome: a forced token and a speculated one
    cannot both be honoured."""
    assert RefusalSteeredCoTProcessor.is_argmax_invariant(RefusalSteeredCoTProcessor) is False


def test_the_entry_point_discovers_the_processor() -> None:
    """Registration is via the `vllm.logits_processors` group, with no CLI flag.

    This is what makes the controller load on a stock `vllm serve`. If the entry point were wrong
    the processor would simply never run, silently.
    """
    discovered = [c.__name__ for c in vllm_logitsprocs.cached_load_custom_logitsprocs(None)]
    assert "RefusalSteeredCoTProcessor" in discovered


def test_the_fqcn_path_also_accepts_it() -> None:
    """The `--logits-processors` fallback requires the same subclass relationship.

    Worth pinning: before the base class was fixed, this path raised
    `must be a subclass of LogitsProcessor`, so the documented fallback could never have worked
    either.
    """
    # Two things this pins, both of which I got wrong first:
    #   1. the loader is `lru_cache`d, so the argument must be hashable -- a tuple, not a list;
    #   2. FQCN syntax is `<module>:<Type>`, colon-separated. A dotted path fails to unpack.
    loaded = vllm_logitsprocs.cached_load_custom_logitsprocs(
        ("cybergate.hooks.cot_controller:RefusalSteeredCoTProcessor",)
    )
    assert any(c.__name__ == "RefusalSteeredCoTProcessor" for c in loaded)


def test_batch_update_shape_is_what_the_controller_unpacks() -> None:
    """The controller unpacks `added` as a 4-tuple and `moved` as a 3-tuple.

    If vLLM changes either shape the controller breaks at runtime, so the assumption is pinned here
    rather than discovered on a server.
    """
    from vllm.v1.sample.logits_processor import interface

    assert interface.AddedRequest.__args__[0] is int
    assert len(interface.AddedRequest.__args__) == 4
    assert len(interface.MovedRequest.__args__) == 3
    assert {"batch_size", "removed", "added", "moved"} <= set(
        interface.BatchUpdate.__dataclass_fields__
    )


def test_added_request_carries_no_request_id() -> None:
    """The constraint the whole identity-join design exists to work around.

    If a future vLLM adds a request id to `AddedRequest`, the join can be simplified back to the
    id and this test should fail loudly to say so.
    """
    args = vllm_logitsprocs.interface.AddedRequest.__args__
    assert args[2] is not str and args[3] is not str, (
        "AddedRequest appears to carry a string id now; revisit the identity join in refusal_trace"
    )
