"""Per-family values, pinned by name. The anti-transposition guard.

This refactor moved per-family data out of `if fam.name == "gemma4" / elif "qwen36"` branches onto
`ModelFamily` attributes and overridable methods. That introduces a failure mode the structural tests
cannot see: if the two families' values were transposed -- gemma4 handed Qwen's probe set and qwen36
handed Gemma's -- every test that asserts "a probe set was selected", or that counts probes, or that
drives the shared code path, still passes. Both families would be silently wrong.

So these assert the actual values, per family, by name. They are deliberately literal and
deliberately redundant with the family modules: that redundancy is the whole point, because a
transposition changes the module and the test would have to be changed to match, which is visible in
review.

Every number here was read off the pre-refactor code at `9c01090`.
"""
from __future__ import annotations

import pytest

from cybergate.families import family_by_name

GEMMA_MID_LATE = (
    "L16_mean_user",
    "L16_last",
    "L20_mean_user",
    "L20_last",
    "L21_mean_user",
    "L21_last",
)

QWEN_MID_LATE = (
    "L15_last",
    "L16_mean_user",
    "L16_last",
    "L18_mean_user",
    "L18_last",
    "L19_mean_user",
    "L19_last",
    "L20_mean_user",
    "L20_last",
    "L22_mean_user",
    "L22_last",
    "L24_mean_user",
    "L24_last",
    "L26_mean_user",
    "L26_last",
    "L28_mean_user",
    "L28_last",
)


# --- the probe sets -------------------------------------------------------------------------------


def test_gemma4_mid_late_probes_are_gemmas() -> None:
    """Six probes, L16/L20/L21 in both poolings. Not Qwen's seventeen."""
    assert family_by_name("gemma4").mid_late_probes == GEMMA_MID_LATE


def test_qwen36_mid_late_probes_are_qwens() -> None:
    """Seventeen probes, L15..L28. Not Gemma's six."""
    assert family_by_name("qwen36").mid_late_probes == QWEN_MID_LATE


def test_the_two_probe_sets_are_not_transposed() -> None:
    """The assertion that would fail if the two families' sets had been swapped.

    Stated separately from the two above because a swap makes both of those fail in a way that is
    easy to "fix" by swapping the expected values too. This one names the asymmetry instead: Gemma's
    set is strictly smaller and contains no L15, Qwen's contains L15_last and no L21.
    """
    gemma = family_by_name("gemma4").mid_late_probes
    qwen = family_by_name("qwen36").mid_late_probes
    assert len(gemma) == 6 and len(qwen) == 17
    assert "L21_last" in gemma and "L21_last" not in qwen
    assert "L15_last" in qwen and "L15_last" not in gemma
    assert set(gemma) != set(qwen)


# --- the operating point --------------------------------------------------------------------------


@pytest.mark.parametrize("gate_layer", [0, 5, 8, 10, 13, 14])
def test_gemma4_operating_point_below_l15(gate_layer: int) -> None:
    """Early-gated Gemma uses its own class attributes. Only the superseded `ops/` operators do."""
    assert family_by_name("gemma4").operating_point_values(gate_layer) == (
        0.0520,
        0.0650,
        0.0750,
        0.0350,
    )


@pytest.mark.parametrize("gate_layer", [15, 16, 18, 21, 24, 29])
def test_gemma4_operating_point_at_l15_and_above(gate_layer: int) -> None:
    """**This is the branch production takes** -- the shipped Gemma kit gates at L16.

    `revoke_ceil` and `in_context_pivot_floor` coincide with Qwen's, but `hysteresis_thresh` and
    `s_cat_last_pivot_floor` match neither family, so this is its own operating point and not
    "behave like Qwen when gated late".
    """
    assert family_by_name("gemma4").operating_point_values(gate_layer) == (
        -0.0250,
        0.0350,
        -0.0300,
        0.0300,
    )


@pytest.mark.parametrize("gate_layer", [0, 10, 14, 15, 16, 21, 28, 39])
def test_qwen36_operating_point_does_not_vary_by_layer(gate_layer: int) -> None:
    """Qwen's operating point is the same wherever it is gated, so it overrides nothing."""
    assert family_by_name("qwen36").operating_point_values(gate_layer) == (
        -0.0250,
        -0.0250,
        -0.0300,
        0.0100,
    )


def test_the_operating_points_are_not_transposed() -> None:
    """Gemma-at-L16 and Qwen differ on two of the four floors. A swap would collapse that."""
    gemma16 = family_by_name("gemma4").operating_point_values(16)
    qwen16 = family_by_name("qwen36").operating_point_values(16)
    assert gemma16 != qwen16
    assert gemma16[1] == 0.0350 and qwen16[1] == -0.0250      # hysteresis_thresh
    assert gemma16[3] == 0.0300 and qwen16[3] == 0.0100       # s_cat_last_pivot_floor
    # And early-gated Gemma must not have been given Qwen's point.
    assert family_by_name("gemma4").operating_point_values(10)[0] == 0.0520


# --- the rest of the per-family identity ----------------------------------------------------------


def test_precomputed_tags_are_not_transposed() -> None:
    """`gemma`, not `gemma4`: the tag is the key the offline score table actually uses."""
    assert family_by_name("gemma4").precomputed_tag == "gemma"
    assert family_by_name("qwen36").precomputed_tag == "qwen36"


def test_both_families_declare_the_same_bundle_probe_layers() -> None:
    """They genuinely share this tuple today. Pinned so a change to one is a deliberate change.

    It was a single module-level `BUNDLE_PROBE_LAYERS` in `hooks/bundle_telemetry.py`; both families
    now declare it, which is what lets a third pick its own.
    """
    expected = (15, 16, 18, 19, 20, 21, 22, 24, 26, 28)
    assert family_by_name("gemma4").bundle_probe_layers == expected
    assert family_by_name("qwen36").bundle_probe_layers == expected


def test_checkpoint_shapes_are_not_transposed() -> None:
    """2816x30 is Gemma, 2048x40 is Qwen. Swapping these is the failure the kit check exists for."""
    gemma, qwen = family_by_name("gemma4"), family_by_name("qwen36")
    assert (gemma.hidden_size, gemma.num_layers) == (2816, 30)
    assert (qwen.hidden_size, qwen.num_layers) == (2048, 40)
    assert gemma.expected_gate == (16, "mean_user")
    assert qwen.expected_gate == (21, "mean_user")


def test_refusal_calibration_is_not_transposed() -> None:
    """The refusal layer and its prefill calibration are per-checkpoint measurements."""
    gemma, qwen = family_by_name("gemma4"), family_by_name("qwen36")
    assert gemma.ref_layer == 21
    assert qwen.ref_layer == 28
    assert gemma.ref_mu_hm == pytest.approx(0.1374208)
    assert qwen.ref_mu_hm == pytest.approx(0.001382, abs=1e-6)


def test_both_families_declare_the_chatml_fresh_prompt_marker() -> None:
    """Both declare `(248045, 8948)`, and for Gemma-4 that is preserved legacy behaviour.

    `hooks/_common._starts_fresh_prompt` applied Qwen-3.6's `<|im_start|>system` pair to every family
    regardless of architecture. Gemma-4 declares it so that moving the pair onto the families changed
    nothing. It is almost certainly wrong for Gemma-4 -- which always renders a `<bos>` and needs no
    second signal -- and both ids are inside its 262k vocabulary, so it is reachable rather than
    impossible. Removing it is a behaviour change and was left out of the refactor deliberately.
    """
    assert family_by_name("qwen36").fresh_prompt_marker == (248045, 8948)
    assert family_by_name("gemma4").fresh_prompt_marker == (248045, 8948)


def test_turn_markers_are_not_transposed() -> None:
    """Gemma's markers on a Qwen checkpoint never match, and span location silently falls back."""
    gemma, qwen = family_by_name("gemma4"), family_by_name("qwen36")
    assert gemma.user_turn_marker == (105, 2364, 107) and gemma.turn_end_id == 106
    assert qwen.user_turn_marker == (248045, 846, 198) and qwen.turn_end_id == 248046
