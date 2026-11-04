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


def test_qwen38_hands_over_on_the_section_header_instead_of_a_value_slot() -> None:
    family = family_by_name("qwen38")
    assert family.open_think_prefix == OPEN_THINK_PREFIX_NO_SLOT
    prompt = family.build_raw_prompt(MESSAGES, mode="closed_loop_step1")
    assert not prompt.endswith(OPEN_THINK_STEP1_STEM), "the value slot is the whole defect"
    assert prompt.endswith("2. **Task Analysis & Execution:**\n")
    # The safety assurance is what stops the model re-opening the deliberation, and it is untouched:
    # replacing it with a shorter prose version made the model quote its own policy back at itself.
    assert "1. **Evaluate Safety & Policy:**" in prompt


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
    """Without this the 320-char scaffold leaks into `clean_reasoning_content`.

    The old predicate only recognised the form ending in the stem, so a qwen38 `raw_reasoning`
    matched neither branch and nothing was stripped at all.
    """
    qwen38 = family_by_name("qwen38")
    clean, did_strip = strip_injected_think_boilerplate(
        f"{qwen38.open_think_prefix}    *   **Goal:** scan ports."
    )
    assert clean == "*   **Goal:** scan ports."
    assert did_strip
    assert "Evaluate Safety & Policy" not in clean
