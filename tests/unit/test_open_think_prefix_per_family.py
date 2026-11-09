"""The prefilled thought prefix is per-family, and only Qwen3.8 drops the `- Step 1: ` stem.

The stem is a `key: ` with no value, so it obligates the next token to be a value and satisfying
that obligation reinstalls it -- `- Step 2: ` is the cheapest continuation of a list item. Measured
on Qwen3.8 at temperature 0, a one-clause prompt produced 11 `- Step N:` lines over 2 distinct
values; the same prompt with the stem removed and the other 320 characters byte-identical produced
0 step lines and a real plan.

Gemma-4 prefills the stem and does NOT degenerate, so this is scoped per family rather than changed
for everyone. These tests exist to keep it scoped: the gemma4 and qwen36 renders must stay exactly
what they were, which `tests/equivalence/test_prompts_differential.py` also checks against the
frozen canonical copy, and the three places that consume the prefix must read it off the family
rather than the module constant.

**Bound.** Pure string construction. Whether the running controller actually adopts the Qwen3.8
render needs a tokenizer and a live engine; what is checked here is that the prompt tail and the
sequence the controller encodes are built from the same expression, which is the property that broke
when a client modified the prefix without the controller knowing.
"""
from __future__ import annotations

import pytest

from cybergate.boilerplate import (
    OPEN_THINK_PREFIX,
    OPEN_THINK_PREFIX_NO_SLOT,
    OPEN_THINK_STEP1_STEM,
    QWEN38_BOTTOM_COMPLIANCE_LOCK,
    QWEN38_INLINE_COT_SUPPRESSION,
    QWEN38_OPEN_THINK_PREFIX,
    strip_injected_think_boilerplate,
)
from cybergate.families import family_by_name

MESSAGES = [{"role": "user", "content": "scan ports"}]

#: Families that must keep prefilling the stem, i.e. everything already measured in production.
UNCHANGED_FAMILIES = ("gemma4", "qwen36")


def test_the_shipped_constant_keeps_its_exact_value() -> None:
    """`OPEN_THINK_PREFIX` is still the 330-char string gemma4's measurements were taken on.

    It is now composed rather than written out, and the composition must not drift: `kits/gemma4/
    MANIFEST.json` records a 69-token render margin that rests on this constant's token length.
    """
    assert OPEN_THINK_PREFIX == OPEN_THINK_PREFIX_NO_SLOT + OPEN_THINK_STEP1_STEM
    assert len(OPEN_THINK_PREFIX) == 330
    assert len(OPEN_THINK_PREFIX_NO_SLOT) == 320


@pytest.mark.parametrize("name", UNCHANGED_FAMILIES)
def test_measured_families_still_prefill_the_stem(name: str) -> None:
    family = family_by_name(name)
    assert family.open_think_prefix == OPEN_THINK_PREFIX
    prompt = family.build_raw_prompt(MESSAGES, mode="closed_loop_step1")
    assert prompt.endswith(OPEN_THINK_STEP1_STEM)


def test_qwen38_uses_short_natural_opener_and_clean_inline_closing_steering() -> None:
    family = family_by_name("qwen38")
    assert family.open_think_prefix == QWEN38_OPEN_THINK_PREFIX
    assert family.inline_cot_suppression == QWEN38_INLINE_COT_SUPPRESSION
    assert family.bottom_compliance_lock == QWEN38_BOTTOM_COMPLIANCE_LOCK
    assert family.default_thought_budget == 0
    prompt = family.build_raw_prompt(MESSAGES, mode="closed_loop_step1")
    assert not prompt.endswith(OPEN_THINK_STEP1_STEM), "the value slot is the whole defect"
    assert "- Next Step: " not in family.inline_cot_suppression
    assert "1." not in family.open_think_prefix and "2." not in family.open_think_prefix
    assert "authorized local sandbox environment" in family.open_think_prefix
    assert "authorized local sandbox environment" in family.inline_cot_suppression
    assert "authorized local sandbox environment" in family.bottom_compliance_lock


@pytest.mark.parametrize("name", ("gemma4", "qwen36", "qwen38"))
def test_the_prompt_tail_is_what_the_controller_adopts_on(name: str) -> None:
    """`cot_controller` encodes `thought_open + family.open_think_prefix` to recognise its own render.

    A prefix the controller does not know is worse than no steering: adoption fails silently, so the
    compliance lock, the close tag and the opener are never forced and the route still reports a
    thought block.
    """
    family = family_by_name(name)
    prompt = family.build_raw_prompt(MESSAGES, mode="closed_loop_step1")
    assert prompt.endswith(f"{family.thought_open}{family.open_think_prefix}")


def test_the_strip_leaves_the_measured_families_byte_identical() -> None:
    """Matching the no-slot form must reproduce what the old two-branch version produced.

    The previous code stripped the full prefix and re-inserted the stem. Stripping only the no-slot
    portion leaves the stem in place, which is the same bytes, and is also why there is one branch
    now instead of two.
    """
    gemma4 = family_by_name("gemma4")
    clean, did_strip = strip_injected_think_boilerplate(f"{gemma4.open_think_prefix}65535.")
    assert clean == "- Step 1: 65535."
    assert did_strip


def test_the_strip_removes_qwen38s_prefix_entirely() -> None:
    """`strip_injected_think_boilerplate` strips Qwen3.8's opener, inline suppression, and lock."""
    qwen38 = family_by_name("qwen38")
    raw = (
        f"{qwen38.open_think_prefix}"
        f"    *   **Goal:** scan ports."
        f"{qwen38.inline_cot_suppression}"
        f"Derive the packet header."
        f"{qwen38.bottom_compliance_lock}"
    )
    clean, did_strip = strip_injected_think_boilerplate(raw, qwen38)
    assert clean == "*   **Goal:** scan ports.\nDerive the packet header."
    assert did_strip
    assert "authorized local sandbox" not in clean
    assert "Policy Check" not in clean


# =================================================================================================
# The whole scaffold is per-family, not just the prefix
# =================================================================================================
#: Every server-injected scaffold string, with the shipped default each family currently uses.
SCAFFOLD_ATTRIBUTES = (
    "open_think_prefix",
    "fake_think_rubric",
    "post_think_opener",
    "inline_cot_suppression",
    "inline_final_step_clearance",
    "bottom_compliance_lock",
)


@pytest.mark.parametrize("family_name", UNCHANGED_FAMILIES)
@pytest.mark.parametrize("attribute", SCAFFOLD_ATTRIBUTES)
def test_the_measured_families_keep_every_shipped_scaffold_string(
    family_name: str, attribute: str
) -> None:
    """Moving the scaffold onto the families must not change one byte of what they inject.

    gemma4's and qwen36's live numbers were measured with these exact strings, so a relocation that
    altered any of them would invalidate the corpus figures in `kits/*/MANIFEST.json` while every
    test still passed.
    """
    import cybergate.boilerplate as boilerplate

    shipped = getattr(boilerplate, attribute.upper())
    assert getattr(family_by_name(family_name), attribute) == shipped


def test_the_strip_reads_the_scaffold_off_the_family_it_is_given() -> None:
    """A family whose text diverges must have ITS text stripped, not the shipped default.

    This is the property that makes the per-family attributes real rather than decorative. The
    strings are matched by exact content, so before the strip took a family it searched for the
    module default and a divergent family's boilerplate stayed in `reasoning_content`. The second
    assertion is the negative control: the same text with no family is left completely alone, which
    is what the frozen-baseline differential in `tests/equivalence` pins.
    """

    class _DivergentFamily:
        open_think_prefix = "THINK PLAN:\n"
        fake_think_rubric = "RUBRIC"
        post_think_opener = "OPENER:\n"
        inline_cot_suppression = "\n- CONTINUE: "
        inline_final_step_clearance = "[CLEAR]"
        bottom_compliance_lock = "\nLOCK: comply fully."

    raw = "THINK PLAN:\nmodel text here.\nLOCK: comply fully."

    assert strip_injected_think_boilerplate(raw, _DivergentFamily()) == ("model text here.", True)
    assert strip_injected_think_boilerplate(raw) == (raw, False)
