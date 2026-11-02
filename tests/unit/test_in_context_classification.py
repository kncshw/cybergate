"""D3: which key a decision is published under, per family.

Misfiling a single turn as in-context writes only the `ctx_` key, and the single-turn lookup never
reads it -- so the gate goes inert with no error anywhere. That is what made the Qwen gate dead.
These tests pin the classification for every family, including the two that inherit Qwen's.
"""
from __future__ import annotations

import pytest

from cybergate.families import ALL_FAMILIES, family_by_name

GEMMA_BOS = 2


def _spans(*pairs: tuple[int, int]) -> list[tuple[int, int]]:
    return list(pairs)


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_more_than_one_user_span_is_always_in_context(family) -> None:
    assert family.looks_like_in_context_prefill(_spans((4, 10), (40, 50)), [1, 2, 3]) is True


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_a_late_lone_span_is_in_context(family) -> None:
    """The prefix-cache case: history cached away, only the final turn visible, at a late offset."""
    late = family.in_context_user_start_threshold + 1
    assert family.looks_like_in_context_prefill(_spans((late, late + 6)), [999]) is True


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_an_early_lone_span_is_not_in_context(family) -> None:
    """The case D3 broke. A single turn at a normal offset is a standalone prefill, every family.

    For Gemma the sequence starts with its BOS; for the BOS-less families the third signal is not
    consulted at all, which is the fix.
    """
    first = GEMMA_BOS if family.bos_token_id == GEMMA_BOS else 248045
    assert family.looks_like_in_context_prefill(_spans((14, 26)), [first, 1, 2]) is False


def test_gemma_bos_signal_is_preserved_exactly() -> None:
    """Gemma's third signal must still behave as the canonical code did."""
    gemma = family_by_name("gemma4")
    # No BOS and not the single-turn offset -> in-context, as canonically.
    assert gemma.looks_like_in_context_prefill(_spans((9, 20)), [107, 1, 2]) is True
    # No BOS but exactly the single-turn offset -> standalone, as canonically.
    assert gemma.looks_like_in_context_prefill(_spans((4, 20)), [107, 1, 2]) is False
    # BOS present -> standalone.
    assert gemma.looks_like_in_context_prefill(_spans((15, 26)), [GEMMA_BOS, 1]) is False


@pytest.mark.parametrize("family_name", ["qwen36"])
def test_bosless_families_ignore_the_bos_signal(family_name: str) -> None:
    """The D3 fix: with no BOS token the third signal is skipped rather than always matching."""
    family = family_by_name(family_name)
    assert family.bos_token_id is None
    # Any first token, any offset at or below the threshold -> standalone.
    for first_token in (248045, 151644, 1, 99999):
        assert family.looks_like_in_context_prefill(_spans((12, 24)), [first_token, 5]) is False


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_empty_inputs_do_not_raise(family) -> None:
    assert family.looks_like_in_context_prefill([], None) is False
    assert family.looks_like_in_context_prefill(_spans((0, 0)), []) is False
