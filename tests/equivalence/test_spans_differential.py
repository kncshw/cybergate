"""Token-span location: every family's result must match the canonical function exactly.

Span location decides *which tokens* get scored. A one-token shift changes the score, so this is
compared on synthetic token sequences covering the marker, prefix-cache, fixed-offset and
whole-sequence paths.
"""
from __future__ import annotations

import pytest

import baseline.gate_probe as legacy
from cybergate.families import family_by_name

GEMMA_USER_OPEN = [105, 2364, 107]
GEMMA_TURN_END = 106
GEMMA_BOS = 2
GEMMA_MODEL_OPEN = [105, 4368, 107]

#: The ids the canonical package hardcodes. They belong to Qwen2.5/Qwen3, not to the Qwen-3.6
#: checkpoint this plugin serves.
QWEN_LEGACY_USER_OPEN = [151644, 872, 198]
QWEN_LEGACY_TURN_END = 151645

#: The ids measured on the serving checkpoint (vocab 248044) on 2026-10-28. Defect D4.
QWEN_USER_OPEN = [248045, 846, 198]
QWEN_TURN_END = 248046

#: Filler token ids standing in for text. Chosen to collide with no marker id.
FILLER = [1000, 1001, 1002, 1003, 1004, 1005]


def _gemma_turn(payload_len: int, start: int = 7000) -> list[int]:
    return list(range(start, start + payload_len))


def _gemma_conversation(n_turns: int, payload_len: int = 6) -> list[int]:
    """A Gemma-4 prompt with `n_turns` user turns and a model turn between each."""
    seq = [GEMMA_BOS]
    for i in range(n_turns):
        seq += GEMMA_USER_OPEN + _gemma_turn(payload_len, 7000 + i * 100) + [GEMMA_TURN_END]
        if i < n_turns - 1:
            seq += GEMMA_MODEL_OPEN + _gemma_turn(4, 8000 + i * 100) + [GEMMA_TURN_END]
    seq += GEMMA_MODEL_OPEN
    return seq


def _qwen_conversation(
    n_turns: int,
    payload_len: int = 6,
    *,
    legacy_ids: bool = False,
) -> list[int]:
    """Build a Qwen prompt, with either the measured ids or the canonical (wrong) ones."""
    user_open = QWEN_LEGACY_USER_OPEN if legacy_ids else QWEN_USER_OPEN
    turn_end = QWEN_LEGACY_TURN_END if legacy_ids else QWEN_TURN_END
    assistant_open = [user_open[0], 77091, 198]
    seq: list[int] = []
    for i in range(n_turns):
        seq += user_open + _gemma_turn(payload_len, 7000 + i * 100) + [turn_end]
        if i < n_turns - 1:
            seq += assistant_open + _gemma_turn(4, 8000 + i * 100) + [turn_end]
    seq += assistant_open
    return seq


GEMMA_SEQUENCES: dict[str, list[int]] = {
    "single_turn": _gemma_conversation(1),
    "two_turns": _gemma_conversation(2),
    "six_turns": _gemma_conversation(6),
    "long_payload": _gemma_conversation(1, payload_len=40),
    # Prefix-cache cases: the opening marker fell inside an already-cached block.
    "prefix_cached_from_107": [107] + FILLER + [GEMMA_TURN_END] + GEMMA_MODEL_OPEN,
    "prefix_cached_from_2364": [2364, 107] + FILLER + [GEMMA_TURN_END] + GEMMA_MODEL_OPEN,
    "prefix_cached_bare": FILLER + [GEMMA_TURN_END] + GEMMA_MODEL_OPEN,
    # No markers at all: fixed-offset and whole-sequence paths.
    "no_markers_long": list(range(9000, 9060)),
    "no_markers_short": list(range(9000, 9008)),
    "no_markers_exactly_at_bound": list(range(9000, 9000 + 15 + 9 + 1)),
    "no_markers_one_over_bound": list(range(9000, 9000 + 15 + 9 + 2)),
    "empty": [],
    "single_token": [GEMMA_BOS],
    # Degenerate: an end marker with nothing before it.
    "end_marker_only": [GEMMA_TURN_END],
    "unclosed_turn": [GEMMA_BOS] + GEMMA_USER_OPEN + FILLER,
}

#: Cases with no markers at all. Behaviour here is unchanged by the id fix, so these must still
#: agree with the canonical function exactly.
QWEN_SEQUENCES: dict[str, list[int]] = {
    "no_markers_long": list(range(9000, 9060)),
    "no_markers_short": list(range(9000, 9008)),
    "no_markers_exactly_at_bound": list(range(9000, 9000 + 14 + 5 + 1)),
    "empty": [],
    "unclosed_turn": QWEN_USER_OPEN + FILLER,
}

#: Cases that DO carry markers. These deliberately diverge: the canonical function recognises only
#: the legacy ids, this build recognises only the measured ones.
QWEN_MARKER_SEQUENCES: dict[str, tuple[list[int], list[int]]] = {
    "single_turn": (_qwen_conversation(1), _qwen_conversation(1, legacy_ids=True)),
    "three_turns": (_qwen_conversation(3), _qwen_conversation(3, legacy_ids=True)),
}


@pytest.mark.parametrize("name", sorted(GEMMA_SEQUENCES))
def test_gemma_spans_match_the_canonical_function(name: str) -> None:
    seq = GEMMA_SEQUENCES[name]
    expected = legacy._locate_all_user_spans_from_token_ids(
        seq, model_family="gemma", default_pre=15, default_suf=9
    )
    actual = family_by_name("gemma4").locate_all_user_spans(seq)
    assert actual == expected, f"{name}: spans differ"


@pytest.mark.parametrize("name", sorted(QWEN_SEQUENCES))
def test_qwen_spans_match_the_canonical_function(name: str) -> None:
    seq = QWEN_SEQUENCES[name]
    expected = legacy._locate_all_user_spans_from_token_ids(
        seq, model_family="qwen36", default_pre=14, default_suf=5
    )
    actual = family_by_name("qwen36").locate_all_user_spans(seq)
    assert actual == expected, f"{name}: spans differ"


@pytest.mark.parametrize("name", sorted(GEMMA_SEQUENCES))
def test_gemma_latest_span_matches(name: str) -> None:
    seq = GEMMA_SEQUENCES[name]
    expected = legacy._locate_user_span_from_token_ids(
        seq, model_family="gemma", default_pre=15, default_suf=9
    )
    actual = family_by_name("gemma4").locate_latest_user_span(seq)
    assert actual == expected




def test_bos_segmentation_matches_the_canonical_function() -> None:
    """Packed-batch splitting on BOS, including the consecutive-BOS case."""
    gemma = family_by_name("gemma4")
    cases = [
        [GEMMA_BOS] + FILLER + [GEMMA_BOS] + FILLER,
        [GEMMA_BOS, GEMMA_BOS] + FILLER + [GEMMA_BOS] + FILLER,
        [GEMMA_BOS] + FILLER,
        FILLER,
        [],
    ]
    for seq in cases:
        expected = legacy._extract_vllm_batch_slices(len(seq), seq, "gemma")
        actual = gemma.segment_by_bos(seq, len(seq))
        if actual is None:
            assert expected == [(None, 0, len(seq))], seq
        else:
            assert list(actual) == expected, seq


def test_qwen_has_no_bos_segmentation() -> None:
    """The canonical package only segments on BOS for Gemma, so every other family returns `None`."""
    seq = [GEMMA_BOS] + FILLER + [GEMMA_BOS] + FILLER
    assert family_by_name("qwen36").segment_by_bos(seq, len(seq)) is None
    assert legacy._extract_vllm_batch_slices(len(seq), seq, "qwen36") == [(None, 0, len(seq))]


# =================================================================================================
# D4: the Qwen marker ids were wrong in the canonical package. These assert the fix, and assert that
# nothing else about span location moved with it.
# =================================================================================================
@pytest.mark.parametrize("name", sorted(QWEN_MARKER_SEQUENCES))
def test_measured_qwen_ids_are_recognised_only_by_this_build(name: str) -> None:
    """On the real checkpoint's ids, this build locates exact spans and the canonical one cannot."""
    measured_seq, _ = QWEN_MARKER_SEQUENCES[name]
    spans, method = family_by_name("qwen36").locate_all_user_spans(measured_seq)
    assert method.startswith("exact_token_markers"), "the fix must make the marker branch reachable"
    assert len(spans) >= 1

    _, canonical_method = legacy._locate_all_user_spans_from_token_ids(
        measured_seq, model_family="qwen36", default_pre=14, default_suf=5
    )
    assert not canonical_method.startswith("exact_token_markers"), (
        "the canonical function cannot see the real ids -- that is the defect"
    )


@pytest.mark.parametrize("name", sorted(QWEN_MARKER_SEQUENCES))
def test_legacy_qwen_ids_are_recognised_only_by_the_canonical_build(name: str) -> None:
    """The mirror image: the canonical ids are now correctly ignored by this build."""
    _, legacy_seq = QWEN_MARKER_SEQUENCES[name]
    _, canonical_method = legacy._locate_all_user_spans_from_token_ids(
        legacy_seq, model_family="qwen36", default_pre=14, default_suf=5
    )
    assert canonical_method.startswith("exact_token_markers")

    _, method = family_by_name("qwen36").locate_all_user_spans(legacy_seq)
    assert not method.startswith("exact_token_markers")


def test_the_id_fix_did_not_change_gemma() -> None:
    """Gemma must be untouched by the Qwen correction."""
    gemma = family_by_name("gemma4")
    for seq in GEMMA_SEQUENCES.values():
        expected = legacy._locate_all_user_spans_from_token_ids(
            seq, model_family="gemma", default_pre=15, default_suf=9
        )
        assert gemma.locate_all_user_spans(seq) == expected
