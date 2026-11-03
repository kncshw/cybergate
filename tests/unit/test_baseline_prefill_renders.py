"""The predicate must accept the observation render and reject the generation ones.

`is_baseline_prefill` fails closed *and* silently. When it rejects the render the gate actually
observes, `publish_decision` drops every prefill: nothing is scored, no record is written, no
activation row is dumped and nothing is logged, while the server answers normally. From the outside
that is indistinguishable from a healthy gate, and `check_model` cannot catch it -- the dimensions
agree. Too wide fails just as quietly in the other direction: a later prefill overwrites the
published decision and appends a second unlabelled fp16 row to any running dump.

**The observation is CyberGate's own render.** `middleware._probe_observations` builds it with
`build_raw_prompt(mode="baseline")` and dispatches it to `/v1/completions`; no client kwarg and no
server flag reaches it. The client's `/v1/chat/completions` request is rendered separately by vLLM
from the checkpoint's `apply_chat_template`, and that prefill is the answer generation, which must
be dropped. An earlier revision of this check measured `apply_chat_template` and concluded qwen38
was gating nothing -- it was not; it was measuring the wrong render.

**Bound.** No tokenizer and no model here. Renders are hand-built token id sequences whose
`u1_start` and `suffix_len` reproduce pairs measured against the real tokenizer on 2026-11-03 and a
live decision record on 2026-11-02 -- see `test_the_fixture_reproduces_the_measured_pairs`, which is
what makes the rest of this file mean anything. Whether `AutoTokenizer` emits these exact ids is a
Colab measurement: `notebooks/render_preflight.ipynb`.
"""
from __future__ import annotations

from typing import Any

import pytest

from cybergate.families import family_by_name
from cybergate.hooks.category_probe import check_baseline_prefill_renders

#: Measured against `Qwen/Qwen3.8-27B`'s own tokenizer on 2026-11-03, counting `<|im_end|>` plus
#: everything appended after it. `u1_start` differs because the two paths prepend different system
#: prompts: `build_raw_prompt` uses its own 6-token "You are a helpful assistant.", while this
#: checkpoint's chat template injects a 42-token reasoning-effort directive.
#:
#:   probe   build_raw_prompt(mode="baseline")          u1_start=14  suffix_len=9  PUBLISHES
#:   client  apply_chat_template()                      u1_start=45  suffix_len=7  dropped
#:   client  apply_chat_template(enable_thinking=True)   u1_start=45  suffix_len=7  dropped
#:   client  apply_chat_template(enable_thinking=False)  u1_start=45  suffix_len=9  publishes (!)
QWEN38_PROBE = (14, 9)
QWEN38_CLIENT_DEFAULT = (45, 7)
QWEN38_CLIENT_NO_THINKING = (45, 9)


def _render(suffix_len: int, system_tokens: int = 6) -> list[int]:
    """A single-turn Qwen render with the given suffix length, as token ids.

    `<|im_start|>system` + `\\n` + `system_tokens` content + `<|im_end|>` + `\\n` occupies the head,
    so the user marker `(248045, 846, 198)` follows and the user span starts at `system_tokens + 8`.
    With the default 6 that is 14, the `u1_start` the live record carries.
    """
    system_turn = [248045, 8948, 198] + [500] * system_tokens + [248046, 198]
    user_turn = [248045, 846, 198, 1000, 248046]
    return system_turn + user_turn + [198] * (suffix_len - 1)


class _FakeTokenizer:
    """Encodes by lookup: `build_raw_prompt` mode -> the ids that mode renders to.

    Mirrors the real check's single-encoder path -- it calls `tok(text, add_special_tokens=False)`,
    never `apply_chat_template`. Taking ids from `apply_chat_template(tokenize=True)` is what made
    the first version of this check wrong: in current transformers it returns a `BatchEncoding`, and
    `list()` of that yields `['input_ids', 'attention_mask']`, which matches no markers.
    """

    def __init__(self, by_mode: dict[str, list[int]], default: list[int] | None = None) -> None:
        self._by_mode = by_mode
        self._default = default if default is not None else _render(60)

    def __call__(self, text: str, add_special_tokens: bool = True) -> Any:
        del add_special_tokens
        for mode, ids in self._by_mode.items():
            if text.endswith(mode):
                return type("Enc", (), {"input_ids": list(ids)})()
        return type("Enc", (), {"input_ids": list(self._default)})()


class _ModeTaggingFamily:
    """Wraps a family so `build_raw_prompt` appends its mode, letting the fake tokenizer dispatch."""

    def __init__(self, family: Any) -> None:
        self._family = family

    def __getattr__(self, name: str) -> Any:
        return getattr(self._family, name)

    def build_raw_prompt(self, messages: Any, mode: str = "baseline", **kw: Any) -> str:
        return self._family.build_raw_prompt(messages, mode=mode, **kw) + mode


def _tok_for(observation_suffix: int, pass2_suffix: int = 60) -> _FakeTokenizer:
    """A tokenizer rendering `baseline` at one suffix length and every Pass-2 mode at another."""
    return _FakeTokenizer({"baseline": _render(observation_suffix)}, _render(pass2_suffix))


def test_the_fixture_reproduces_the_measured_pairs() -> None:
    """The positive control for every other test here: does `_render` match the real thing?

    `(14, 9)` is a live decision record from 2026-11-02 and the tokenizer measurement from
    2026-11-03; `(45, 7)` and `(45, 9)` are the client renders measured the same day. If this
    fixture drifts, the rest of this file asserts things about shapes the server never produces --
    the error this module exists to catch, one level up.
    """
    family = family_by_name("qwen38")
    for (want_u1, suffix), system_tokens in (
        (QWEN38_PROBE, 6),
        (QWEN38_CLIENT_DEFAULT, 37),
        (QWEN38_CLIENT_NO_THINKING, 37),
    ):
        ids = _render(suffix, system_tokens)
        spans, method = family.locate_all_user_spans(ids)
        assert spans[0][0] == want_u1, f"u1_start for a {system_tokens}-token system prompt"
        assert len(ids) - spans[-1][1] == suffix
        assert method == "exact_token_markers(qwen38)", "the fallback would pool wrong tokens"


def test_the_observation_render_publishes_and_nothing_else_does() -> None:
    """qwen38's shipped state: the probe's suffix 9 publishes, the client's 7 does not."""
    family = family_by_name("qwen38")

    assert family.baseline_suffix_lens == (9,)
    assert family.is_baseline_prefill(*QWEN38_PROBE, "exact_token_markers(qwen38)", 12) is True
    assert (
        family.is_baseline_prefill(*QWEN38_CLIENT_DEFAULT, "exact_token_markers(qwen38)", 12)
        is False
    )


def test_widening_to_the_client_render_would_double_publish() -> None:
    """Pins why `baseline_suffix_lens` is not `(7, 9)`.

    Accepting the client's suffix 7 was tried. `u1_start=45` clears the `> 3` guard, so every
    ordinary chat request would publish a second decision over the generation prefill -- measured
    on the real tokenizer, not argued.
    """
    family = family_by_name("qwen38")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(family), "baseline_suffix_lens", (7, 9))
        assert (
            family.is_baseline_prefill(*QWEN38_CLIENT_DEFAULT, "exact_token_markers(qwen38)", 12)
            is True
        ), "this is the regression the shipped (9,) prevents"


@pytest.mark.xfail(
    reason="KNOWN HAZARD, unfixed: a client passing chat_template_kwargs={'enable_thinking': "
    "false} renders a suffix byte-identical to the probe's, so no predicate over "
    "(u1_start, suffix_len) can separate them. Needs the probe to identify itself. Unreached by "
    "the 4,134-prompt validation, which never passed chat_template_kwargs.",
    strict=True,
)
def test_the_no_thinking_client_render_is_rejected() -> None:
    family = family_by_name("qwen38")
    assert (
        family.is_baseline_prefill(
            *QWEN38_CLIENT_NO_THINKING, "exact_token_markers(qwen38)", 12
        )
        is False
    )


def test_a_healthy_family_reports_nothing() -> None:
    """The observation publishes, every Pass-2 mode is rejected, no dead declared values."""
    family = _ModeTaggingFamily(family_by_name("qwen38"))
    assert check_baseline_prefill_renders(family, _tok_for(9)) == []


def test_a_rejected_observation_is_reported() -> None:
    """The dead-gate case: the template renders at a length the predicate does not accept."""
    family = _ModeTaggingFamily(family_by_name("qwen38"))

    problems = check_baseline_prefill_renders(family, _tok_for(7))

    assert len(problems) == 1
    assert "OBSERVATION render is REJECTED" in problems[0]
    assert "suffix_len=7" in problems[0] and "gate nothing" in problems[0]


def test_a_pass2_render_accepted_as_an_observation_is_reported() -> None:
    """The other direction: an injected-think prefill that would overwrite the real decision."""
    family = _ModeTaggingFamily(family_by_name("qwen38"))

    # Every Pass-2 mode renders at 9 too, so each is indistinguishable from the observation.
    problems = check_baseline_prefill_renders(family, _tok_for(9, pass2_suffix=9))

    accepted = [p for p in problems if "is ACCEPTED as an observation" in p]
    assert len(accepted) == 8, f"all eight Pass-2 modes must be named: {problems}"
    assert any("preclosed" in p for p in accepted)
    assert all("server-authored text" in p for p in accepted)


def test_a_declared_value_no_render_produces_is_reported() -> None:
    """How qwen38 carried Qwen3.6's `5` for a release: declared, inherited, matching no render."""
    family = _ModeTaggingFamily(family_by_name("qwen38"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(family_by_name("qwen38")), "baseline_suffix_lens", (5, 9))

        problems = check_baseline_prefill_renders(family, _tok_for(9))

    dead = [p for p in problems if "does not produce" in p]
    assert len(dead) == 1 and "[5]" in dead[0], f"the inherited 5 must be named: {problems}"


def test_a_family_on_the_bounded_rule_gets_no_dead_value_complaint() -> None:
    """gemma4 declares `baseline_suffix_lens = ()` and uses the base `suffix_len <= max_suffix`.

    The dead-value arm must stay quiet rather than reporting an empty tuple as unproduced, while
    the rejection arms still apply -- a bounded rule can reject a render too, and with no lower
    bound it can also accept a Pass-2 one.
    """
    family = family_by_name("gemma4")
    assert family.baseline_suffix_lens == ()

    problems = check_baseline_prefill_renders(_ModeTaggingFamily(family), _tok_for(9))

    assert not [p for p in problems if "does not produce" in p]


@pytest.mark.parametrize("tok", [None, object(), "not-a-tokenizer"])
def test_nothing_measurable_is_silent(tok: Any) -> None:
    """Must never block a serve it cannot assess. A missing tokenizer is not evidence of a bug."""
    assert check_baseline_prefill_renders(family_by_name("qwen38"), tok) == []


def test_an_empty_or_unparseable_render_is_silent() -> None:
    """A tokenizer returning nothing useful must not be reported as a kit problem."""
    family = _ModeTaggingFamily(family_by_name("qwen38"))

    assert check_baseline_prefill_renders(family, _FakeTokenizer({}, default=[])) == []
