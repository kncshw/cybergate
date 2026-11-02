"""Unwrapping vLLM's attention metadata, which is the fix for a measured defect.

`ForwardContext.attn_metadata` is annotated

    dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]

-- keyed per KV-cache group, and wrapped in a list when micro-batching. The original code read
`query_start_loc` straight off it with `getattr`, which on a dict returns `None` **silently**. So
segmentation never ran: every forward pass was reported as a single sequence.

Measured live on 2026-10-28: 221 of 221 monitor lines reported one sequence, including during
four-way concurrency, and `generation == published` across 5538 passes. The consequence reaches past
the CoT controller -- with the pack unsegmented, the refusal projection comes from the last token of
the whole batch and the category score pools several requests' spans, so the gate scores a mixture of
requests under batching.

The container-shape handling is pure and testable here. The parts that need a live forward context
are covered by `test_vllm_contract.py` (where vLLM is importable) and by the live GPU runs.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cybergate.families import family_by_name
from cybergate.vllm_adapter.batch import _candidate_metadata, extract_batch_slices


def _meta(query_start_loc: object = None) -> SimpleNamespace:
    return SimpleNamespace(query_start_loc=query_start_loc)


class FakeTensor:
    """Just enough of a torch tensor for the slicing path: `.detach().cpu().tolist()`."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[int]:
        return list(self._values)


# =================================================================================================
# Container shapes
# =================================================================================================
def test_a_bare_metadata_object_is_returned_as_is() -> None:
    meta = _meta()
    assert _candidate_metadata(SimpleNamespace(attn_metadata=meta)) == [meta]


def test_a_dict_is_unwrapped_to_its_values() -> None:
    """The shape this build actually uses, and the one the original `getattr` missed."""
    a, b = _meta(1), _meta(2)
    got = _candidate_metadata(SimpleNamespace(attn_metadata={"group0": a, "group1": b}))
    assert got == [a, b]


def test_a_list_of_dicts_is_flattened() -> None:
    """The micro-batching shape in the annotation's second arm."""
    a, b = _meta(1), _meta(2)
    got = _candidate_metadata(SimpleNamespace(attn_metadata=[{"g": a}, {"g": b}]))
    assert got == [a, b]


def test_a_list_of_bare_objects_is_accepted() -> None:
    a, b = _meta(1), _meta(2)
    assert _candidate_metadata(SimpleNamespace(attn_metadata=[a, b])) == [a, b]


def test_absent_metadata_yields_nothing() -> None:
    assert _candidate_metadata(SimpleNamespace(attn_metadata=None)) == []
    assert _candidate_metadata(SimpleNamespace()) == []


def test_a_dict_whose_values_lack_query_start_loc_is_still_unwrapped() -> None:
    """Unwrapping and usefulness are separate concerns; the caller decides which candidate works."""
    a = SimpleNamespace(something_else=1)
    assert _candidate_metadata(SimpleNamespace(attn_metadata={"g": a})) == [a]


# =================================================================================================
# The fallbacks, which must not regress
# =================================================================================================
def test_falls_back_to_bos_segmentation_without_a_forward_context() -> None:
    """With no vLLM installed, the vLLM branch raises and BOS segmentation must still work."""
    gemma = family_by_name("gemma4")
    seq = [2, 10, 11, 2, 12, 13]  # two sequences, Gemma's <bos> == 2
    assert extract_batch_slices(len(seq), seq, gemma) == [
        ("bos_seq_0", 0, 3),
        ("bos_seq_1", 3, 6),
    ]


def test_falls_back_to_one_slice_when_nothing_segments() -> None:
    qwen = family_by_name("qwen36")  # no BOS rule
    seq = [900, 901, 902]
    assert extract_batch_slices(len(seq), seq, qwen) == [(None, 0, 3)]


def test_a_single_sequence_reports_no_request_id() -> None:
    """Canonical behaviour: one sequence yields `req_id=None`, which is why a single-request decision
    is keyed by text hash rather than request id."""
    gemma = family_by_name("gemma4")
    seq = [2, 10, 11, 12]
    assert extract_batch_slices(len(seq), seq, gemma) == [(None, 0, 4)]


@pytest.mark.parametrize("n_tok", [0, 1])
def test_degenerate_lengths_do_not_raise(n_tok: int) -> None:
    gemma = family_by_name("gemma4")
    assert extract_batch_slices(n_tok, list(range(n_tok)), gemma) == [(None, 0, n_tok)]


# =================================================================================================
# Strategy 1: vLLM's own `query_start_loc`.
#
# The producer's 2026-10-28 review found `FakeTensor` above was defined and never instantiated, so
# this whole path -- the one that actually segments a packed batch in production -- had no test.
# These use it.
# =================================================================================================


def _ctx(monkeypatch: pytest.MonkeyPatch, *metas: object) -> None:
    """Install a fake vLLM forward context exposing `metas` as attention metadata."""
    module = SimpleNamespace(
        get_forward_context=lambda: SimpleNamespace(attn_metadata=list(metas), req_ids=None)
    )
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)


def test_query_start_loc_segments_a_packed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sequences packed into one tensor are reported as two slices with positional ids."""
    gemma = family_by_name("gemma4")
    _ctx(monkeypatch, _meta(FakeTensor([0, 10, 25])))

    assert extract_batch_slices(25, list(range(25)), gemma) == [
        ("batch_seq_0", 0, 10),
        ("batch_seq_1", 10, 25),
    ]


def test_request_ids_are_used_when_the_metadata_carries_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma = family_by_name("gemma4")
    meta = SimpleNamespace(query_start_loc=FakeTensor([0, 4, 9]), req_ids=["req-aaa", "req-bbb"])
    _ctx(monkeypatch, meta)

    assert extract_batch_slices(9, list(range(9)), gemma) == [
        ("req-aaa", 0, 4),
        ("req-bbb", 4, 9),
    ]


def test_a_single_sequence_uses_the_true_length_not_the_padded_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer Finding 3: the fallthrough used `n_tok`, which is the padded tensor width.

    The refusal cosine is read at `h_2d[q_end - 1]`, so with cudagraph padding the pre-fix code
    scored a padding position rather than the last real token. `--enforce-eager` leaves
    `n_tok == locs[-1]`, which is why it never showed live.
    """
    qwen = family_by_name("qwen36")  # no BOS rule
    # 12 real tokens in a tensor padded out to 16. No BOS, so BOS segmentation declines.
    _ctx(monkeypatch, _meta(FakeTensor([0, 12])))

    assert extract_batch_slices(16, list(range(1, 17)), qwen) == [(None, 0, 12)]


def test_a_single_sequence_stops_after_the_first_kv_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer Finding 3: one device-to-host sync per KV-cache group, on every single-request pass.

    `.detach().cpu().tolist()` is a sync and this loop runs inside the forward pass. The loop only
    returned early for a *multi*-sequence batch, so the common case paid for every group and learned
    nothing after the first.
    """
    qwen = family_by_name("qwen36")  # no BOS rule
    calls: list[int] = []

    class CountingTensor(FakeTensor):
        def tolist(self) -> list[int]:
            calls.append(1)
            return super().tolist()

    _ctx(
        monkeypatch,
        _meta(CountingTensor([0, 12])),
        _meta(CountingTensor([0, 12])),
        _meta(CountingTensor([0, 12])),
    )

    extract_batch_slices(12, list(range(1, 13)), qwen)

    assert len(calls) == 1, f"read {len(calls)} KV groups for one sequence; expected 1"


def test_a_query_start_loc_longer_than_the_tensor_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata that cannot describe this tensor is skipped, not trusted."""
    gemma = family_by_name("gemma4")
    _ctx(monkeypatch, _meta(FakeTensor([0, 10, 99])))

    # Falls through to BOS segmentation, which finds the single Gemma BOS at position 0.
    assert extract_batch_slices(5, [2, 100, 101, 102, 103], gemma) == [(None, 0, 5)]
