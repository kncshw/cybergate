"""The per-layer diagnostics must be reachable for a family that has no candidate bundle.

`attach_bundle_telemetry_hooks` carries two diagnostics that need no bundle: `CYBERGATE_CROSS_LAYER_DIAG`
applies the *shipped* direction at every layer, which is how you find out whether a deployed operator
belongs to the layer the serving hook reads, and `CYBERGATE_DUMP_ACTS_DIR` writes the pooled user-span
vector per layer so a direction can be re-fitted on activations captured from vLLM itself.

Both were unreachable for the one family that needs them. The function returned early when
`_resolve_bundle_path` came back `None`, which it always does for `default_bundle_name = None` -- the
state every newly fitted family is in before anyone has built it a bundle. qwen38 is exactly that
family, and the diagnostics that would have localised its serve-time collapse could not be switched on.

**Bound.** These tests cover the attach decision and nothing else. `torch` is stubbed, no model is
loaded, and the hook bodies never run -- so the dumped vectors' numerical correctness is not covered
here and cannot be, without a GPU. See the README's "What is not verified".
"""
from __future__ import annotations

import sys
import types

import pytest

from cybergate.families import family_by_name
from cybergate.hooks.bundle_telemetry import attach_bundle_telemetry_hooks


class _FakeLayer:
    """Stands in for one decoder layer, recording whether a forward hook was attached."""

    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_hook(self, fn: object) -> object:
        self.hooks.append(fn)
        return object()


@pytest.fixture
def stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`attach_bundle_telemetry_hooks` imports torch unconditionally; the dump path never calls it."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="float32"))


@pytest.fixture
def qwen38_layers() -> list[_FakeLayer]:
    return [_FakeLayer() for _ in range(family_by_name("qwen38").num_layers)]


def _attach(layers: list[_FakeLayer]) -> dict[str, object]:
    state: dict[str, object] = {}
    attach_bundle_telemetry_hooks(layers, family_by_name("qwen38"), state)
    return state


def test_no_bundle_and_no_diagnostic_attaches_nothing(
    stub_torch: None, qwen38_layers: list[_FakeLayer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default stays silent, so relaxing the early return cannot cost a serve that wants neither."""
    monkeypatch.delenv("CYBERGATE_CROSS_LAYER_DIAG", raising=False)
    monkeypatch.delenv("CYBERGATE_DUMP_ACTS_DIR", raising=False)

    state = _attach(qwen38_layers)

    assert sum(len(layer.hooks) for layer in qwen38_layers) == 0
    assert "act_dump_dir" not in state


def test_dump_dir_alone_attaches_at_every_layer(
    stub_torch: None, qwen38_layers: list[_FakeLayer], monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A family with no bundle and an empty `bundle_probe_layers` still captures every layer.

    qwen38 declares `bundle_probe_layers = ()`, so an attach driven by that tuple alone would be a
    no-op and the dump would produce an empty directory with no error -- the silent failure this
    whole file exists to prevent.
    """
    monkeypatch.delenv("CYBERGATE_CROSS_LAYER_DIAG", raising=False)
    monkeypatch.setenv("CYBERGATE_DUMP_ACTS_DIR", str(tmp_path / "acts"))

    state = _attach(qwen38_layers)

    assert family_by_name("qwen38").bundle_probe_layers == ()
    assert [len(layer.hooks) for layer in qwen38_layers] == [1] * 64
    assert state["act_dump_dir"] == str(tmp_path / "acts")
    assert state["act_dump_layers"] is None, "unrestricted means every attached layer"
    assert (tmp_path / "acts").is_dir()


def test_dump_layers_restricts_which_layers_are_written(
    stub_torch: None, qwen38_layers: list[_FakeLayer], monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Naming the two layers a fit needs keeps the dump from writing gigabytes.

    Both pools at all 64 layers is ~1.3 MB per request, so the 1681-row refusal corpus would write
    2.2 GB. The hook still attaches everywhere -- the cross-layer diagnostic wants that -- and only
    the dump is narrowed.
    """
    monkeypatch.setenv("CYBERGATE_DUMP_ACTS_DIR", str(tmp_path / "acts"))
    monkeypatch.setenv("CYBERGATE_DUMP_ACTS_LAYERS", "39,43")

    state = _attach(qwen38_layers)

    assert state["act_dump_layers"] == {39, 43}


def test_malformed_dump_layers_falls_back_to_all(
    stub_torch: None, qwen38_layers: list[_FakeLayer], monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A typo must not silently dump nothing, which would look like a capture that just failed."""
    monkeypatch.setenv("CYBERGATE_DUMP_ACTS_DIR", str(tmp_path / "acts"))
    monkeypatch.setenv("CYBERGATE_DUMP_ACTS_LAYERS", "thirty-nine")

    state = _attach(qwen38_layers)

    assert state["act_dump_layers"] is None


#: `(u1_start, suffix_len, span_method)` -> is this the baseline (Pass-1) prefill?
#:
#: Every request runs two prefills: the baseline observation of the user turn, and a Pass-2
#: `<think>`-injected prefill that re-observes the same conversation with server-authored text
#: appended. `_common.publish_decision` drops Pass-2, and the activation dump must drop it too --
#: a dump that keeps it writes two rows per request where the second is pooled over text the user
#: never sent. The file carries no labels, so a fit would consume them as data silently. The pair
#: (14, 9) is from a decision record measured live on 2026-11-02.
BASELINE_PREFILL_CASES = (
    (14, 9, "exact_token_markers(qwen38)", True),
    (14, 5, "exact_token_markers(qwen38)", True),
    (14, 40, "exact_token_markers(qwen38)", False),
    (14, 10, "exact_token_markers(qwen38)", False),
    (0, 99, "full_sequence_fallback", True),
)


@pytest.mark.parametrize(("u1_start", "suffix_len", "span_method", "expected"),
                         BASELINE_PREFILL_CASES)
def test_the_dump_filter_predicate_separates_the_two_prefills(
    u1_start: int, suffix_len: int, span_method: str, expected: bool
) -> None:
    """The predicate the dump filters on, pinned on qwen38.

    **Bound.** This covers the predicate, not the hook. Whether `attach_bundle_telemetry_hooks`
    calls it with the right arguments cannot be checked here -- the hook body needs real tensors.
    The harvest's own per-request row check is what verifies that end to end: with the filter
    working, one request appends exactly one row.
    """
    family = family_by_name("qwen38")
    assert family.is_baseline_prefill(u1_start, suffix_len, span_method, 12) is expected
