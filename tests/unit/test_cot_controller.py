"""The measured CoT controller's logic, with no GPU and no torch.

The controller only touches a tensor through `logits[row] = x` and `logits[row, token] = y`, and only
reads refusal levels through `refusal_trace`. Both are substitutable, so every steering decision --
trigger, cooldown, budget, staleness, row bookkeeping -- is a table-driven unit test.

That is the whole argument for this design over the regex: the regex trigger could only be exercised
by generating text on a 26B model, so it never was.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from cybergate.hooks import refusal_trace
from cybergate.hooks.cot_controller import RefusalSteeredCoTProcessor, SteeringConfig

VOCAB = 64
#: Token ids the fake tokenizer assigns, by text.
SUPPRESSION_IDS = [31, 32, 33]
THOUGHT_PREFIX_IDS = [11, 12, 13]
CLOSE_TAG_IDS = [41, 42]


class FakeLogits:
    """Minimal stand-in for the logits tensor: row assignment and single-element assignment."""

    def __init__(self, rows: int, vocab: int = VOCAB) -> None:
        self.rows = [[0.0] * vocab for _ in range(rows)]
        self.shape = (rows, vocab)

    def __setitem__(self, key: Any, value: float) -> None:
        if isinstance(key, tuple):
            row, column = key
            self.rows[row][column] = value
        else:
            self.rows[key] = [value] * self.shape[1]

    def forced_token(self, row: int) -> int | None:
        """The single token this row was pinned to, or `None` if the row was left alone."""
        values = self.rows[row]
        finite = [i for i, v in enumerate(values) if v != float("-inf")]
        if len(finite) == 1 and all(v == float("-inf") for i, v in enumerate(values) if i != finite[0]):
            return finite[0]
        return None

    def untouched(self, row: int) -> bool:
        return all(v == 0.0 for v in self.rows[row])


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        from cybergate.boilerplate import (
            BOTTOM_COMPLIANCE_LOCK,
            INLINE_COT_SUPPRESSION,
            POST_THINK_OPENER,
        )

        if text == INLINE_COT_SUPPRESSION:
            return list(SUPPRESSION_IDS)
        if text in ("<channel|>", "</think>"):
            return list(CLOSE_TAG_IDS)
        if text in (BOTTOM_COMPLIANCE_LOCK, POST_THINK_OPENER):
            return []
        return list(THOUGHT_PREFIX_IDS)


def _batch_update(
    *,
    added: Sequence[tuple[int, Any, Sequence[int] | None, Sequence[int]]] = (),
    removed: Sequence[int] = (),
    moved: Sequence[tuple[int, int, Any]] = (),
) -> SimpleNamespace:
    return SimpleNamespace(batch_size=8, added=list(added), removed=list(removed), moved=list(moved))


def _steerable_prompt(tag: int = 0) -> list[int]:
    """A prompt ending with the open-thought prefix, so the controller adopts the row.

    `tag` makes it textually distinct, which matters now that the join is on prompt identity: two
    rows built from the same `tag` share a key, by design.
    """
    return [5, 6, 7 + tag, *THOUGHT_PREFIX_IDS]


def _plain_prompt() -> list[int]:
    return [5, 6, 7, 8, 9]


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> RefusalSteeredCoTProcessor:
    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: FakeTokenizer()
    )
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    proc.config = SteeringConfig(
        tau_spike=3.55,
        min_cot_steps=3,
        cooldown_steps=4,
        max_injections=2,
        mu_decode=0.0,
        sd_decode=1.0,
    )
    return proc


def _publish(
    rolling: float, *, key: str | None = None, decode_domain: bool = True
) -> None:
    """Publish a reading in a fresh pass, as the monitor hook would.

    With no `key`, publishes under the sole-sequence key, which is what the hook does when a pass
    holds one sequence. The controller reads that only when tracking exactly one row.

    `decode_domain` defaults to True, i.e. a deployment whose decode calibration is present. Pass
    False to exercise the uncalibrated path.
    """
    generation = refusal_trace.begin_pass()
    reading = refusal_trace.RowReading(
        rolling_s_ref_z=rolling,
        s_ref_z=rolling,
        seq_len=1,
        generation=generation,
        decode_domain=decode_domain,
    )
    refusal_trace.publish(key or refusal_trace.SOLE_KEY, reading)


def _key_of(tag: int = 0) -> str:
    """The identity key the controller will compute for `_steerable_prompt(tag)`."""
    return refusal_trace.key_for_tokens(_steerable_prompt(tag))


def _step(proc: RefusalSteeredCoTProcessor, rows: int = 1) -> FakeLogits:
    logits = FakeLogits(rows)
    proc.apply(logits)
    return logits


# =================================================================================================
# Declarations
# =================================================================================================
def test_declares_it_changes_the_argmax(processor: RefusalSteeredCoTProcessor) -> None:
    """Forcing a token is not argmax-invariant, and vLLM relies on that being declared honestly."""
    assert processor.is_argmax_invariant() is False


def test_disables_itself_loudly_without_a_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tokenizer means no token ids to force. It must disable, not silently do nothing useful."""
    monkeypatch.setattr("cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: None)
    config = SimpleNamespace(model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.enabled is False
    proc.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [])]))
    assert proc._rows == {}
    logits = _step(proc)
    assert logits.untouched(0)


# =================================================================================================
# Which requests get steered
# =================================================================================================
def test_adopts_only_prompts_ending_in_the_open_thought_prefix(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(), []),
                (1, None, _plain_prompt(), []),
                (2, None, None, []),
            ]
        )
    )
    assert set(processor._rows) == {0}
    assert processor.stats.rows_tracked == 1
    assert processor.stats.rows_skipped == 2


def test_an_untracked_row_is_never_touched(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(_batch_update(added=[(1, None, _plain_prompt(), [])]))
    _publish(99.0, key=_key_of(1))
    logits = _step(processor, rows=2)
    assert logits.untouched(1)


# =================================================================================================
# The trigger
# =================================================================================================
def test_does_not_fire_before_the_minimum_cot_steps(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """A spike while the rolling window is still filling must not steer."""
    output: list[int] = [1, 2]  # 2 < min_cot_steps of 3
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(10.0)
    logits = _step(processor)
    assert logits.untouched(0)
    assert processor.stats.injections == 0


def test_does_not_fire_below_tau(processor: RefusalSteeredCoTProcessor) -> None:
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(3.54)
    logits = _step(processor)
    assert logits.untouched(0)


def test_fires_at_tau(processor: RefusalSteeredCoTProcessor) -> None:
    """The comparison is `>=`, so exactly tau triggers."""
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(3.55)
    logits = _step(processor)
    assert logits.forced_token(0) == SUPPRESSION_IDS[0]
    assert processor.stats.injections == 1


def test_refuses_a_stale_reading(processor: RefusalSteeredCoTProcessor) -> None:
    """A reading from an earlier pass may belong to another token, or another request.

    This is the guard that stops the positional join becoming an R11-style cross-request leak.
    """
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(10.0)
    refusal_trace.begin_pass()  # a new pass arrives with no reading for this row
    logits = _step(processor)
    assert logits.untouched(0)
    assert processor.stats.unresolved == 1


def test_no_reading_at_all_does_not_steer(processor: RefusalSteeredCoTProcessor) -> None:
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    logits = _step(processor)
    assert logits.untouched(0)


# =================================================================================================
# Injection, cooldown and budget
# =================================================================================================
def test_forces_the_whole_suppression_string_then_releases(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))

    _publish(5.0)
    forced = [_step(processor).forced_token(0)]
    for _ in range(len(SUPPRESSION_IDS) - 1):
        _publish(5.0)
        forced.append(_step(processor).forced_token(0))

    assert forced == SUPPRESSION_IDS
    assert processor._rows[0].injecting is False
    assert processor._rows[0].cooldown == processor.config.cooldown_steps
    assert processor.stats.forced_tokens == len(SUPPRESSION_IDS)
    # Exactly one injection was counted, not one per forced token.
    assert processor.stats.injections == 1


def test_cooldown_blocks_a_second_injection(processor: RefusalSteeredCoTProcessor) -> None:
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    for _ in range(len(SUPPRESSION_IDS)):
        _publish(5.0)
        _step(processor)

    cooled = []
    for _ in range(processor.config.cooldown_steps):
        _publish(50.0)
        cooled.append(_step(processor).untouched(0))
    assert all(cooled), "nothing may be forced while cooling down"

    _publish(50.0)
    assert _step(processor).forced_token(0) == SUPPRESSION_IDS[0]
    assert processor.stats.injections == 2


def test_injection_budget_is_capped(processor: RefusalSteeredCoTProcessor) -> None:
    """A persistently high reading must not steer forever."""
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))

    for _ in range(200):
        _publish(50.0)
        _step(processor)

    assert processor.stats.injections == processor.config.max_injections
    assert processor._rows[0].injections_used == processor.config.max_injections


def test_records_the_peak_reading_it_saw(processor: RefusalSteeredCoTProcessor) -> None:
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    for rolling in (1.0, 2.5, 0.5):
        _publish(rolling)
        _step(processor)
    assert processor._rows[0].peak_rolling_z == pytest.approx(2.5)
    assert processor.stats.peak_rolling_z == pytest.approx(2.5)


# =================================================================================================
# Batch bookkeeping -- rows are reused, and getting this wrong leaks state between requests
# =================================================================================================
def test_removed_row_drops_its_state(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    processor.update_state(_batch_update(removed=[0]))
    assert processor._rows == {}
    _publish(50.0)
    assert _step(processor).untouched(0)


def test_an_added_row_does_not_inherit_the_previous_occupants_budget(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """vLLM may place a new request on an index a finished one used. Budgets must not carry over."""
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    for _ in range(200):
        _publish(50.0)
        _step(processor)
    assert processor._rows[0].injections_used == 2

    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    assert processor._rows[0].injections_used == 0
    _publish(50.0)
    assert _step(processor).forced_token(0) == SUPPRESSION_IDS[0]


def test_a_new_unsteerable_request_evicts_a_steerable_row(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    processor.update_state(_batch_update(added=[(0, None, _plain_prompt(), [1, 2, 3, 4])]))
    assert processor._rows == {}


def test_unidirectional_move_relocates_state(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    processor.update_state(
        _batch_update(moved=[(0, 2, SimpleNamespace(name="UNIDIRECTIONAL"))])
    )
    assert set(processor._rows) == {2}
    _publish(5.0)
    logits = _step(processor, rows=4)
    assert logits.forced_token(2) == SUPPRESSION_IDS[0]
    assert logits.untouched(0)


def test_swap_exchanges_two_rows(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(), [1, 2, 3, 4]),
                (1, None, _steerable_prompt(), [1, 2]),
            ]
        )
    )
    state_zero = processor._rows[0]
    state_one = processor._rows[1]
    processor.update_state(_batch_update(moved=[(0, 1, SimpleNamespace(name="SWAP"))]))
    assert processor._rows[1] is state_zero
    assert processor._rows[0] is state_one


def test_a_shrinking_batch_does_not_index_past_the_tensor(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """Defensive: a tracked row beyond the tensor must be skipped, not crash the sampler."""
    processor.update_state(_batch_update(added=[(5, None, _steerable_prompt(), [1, 2, 3, 4])]))
    _publish(50.0)
    logits = FakeLogits(2)
    processor.apply(logits)  # must not raise
    assert logits.untouched(0) and logits.untouched(1)


def test_two_rows_are_steered_independently(processor: RefusalSteeredCoTProcessor) -> None:
    """Concurrency: two distinct prompts, one above tau and one below, in the same pass.

    This is the case the row-index join got wrong live. Identity keys make it positional-free: the
    readings are published under each prompt's own key, in whatever order.
    """
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(0), [1, 2, 3, 4]),
                (1, None, _steerable_prompt(1), [1, 2, 3, 4]),
            ]
        )
    )
    generation = refusal_trace.begin_pass()
    refusal_trace.publish(_key_of(0), refusal_trace.RowReading(9.0, 9.0, 1, generation, decode_domain=True))
    refusal_trace.publish(_key_of(1), refusal_trace.RowReading(0.1, 0.1, 1, generation, decode_domain=True))

    logits = FakeLogits(2)
    processor.apply(logits)
    assert logits.forced_token(0) == SUPPRESSION_IDS[0]
    assert logits.untouched(1)


def test_steering_follows_the_prompt_not_the_row(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The regression for the live defect: swap which row holds which prompt, and the steering
    must follow the prompt.

    Under the old positional join this failed -- the reading was looked up by row, so the high
    reading would steer whichever request happened to occupy row 0.
    """
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(1), [1, 2, 3, 4]),  # low-reading prompt on row 0
                (1, None, _steerable_prompt(0), [1, 2, 3, 4]),  # high-reading prompt on row 1
            ]
        )
    )
    generation = refusal_trace.begin_pass()
    refusal_trace.publish(_key_of(0), refusal_trace.RowReading(9.0, 9.0, 1, generation, decode_domain=True))
    refusal_trace.publish(_key_of(1), refusal_trace.RowReading(0.1, 0.1, 1, generation, decode_domain=True))

    logits = FakeLogits(2)
    processor.apply(logits)
    assert logits.untouched(0), "row 0 holds the low-reading prompt"
    assert logits.forced_token(1) == SUPPRESSION_IDS[0], "row 1 holds the high-reading prompt"


def test_sole_key_is_not_used_when_several_rows_are_tracked(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The sole-sequence fallback must not become an ambiguous catch-all.

    With more than one tracked row, a reading published under the sole key belongs to nobody, and
    using it would reintroduce exactly the cross-request confusion the identity join removes.
    """
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(0), [1, 2, 3, 4]),
                (1, None, _steerable_prompt(1), [1, 2, 3, 4]),
            ]
        )
    )
    _publish(50.0)  # sole key only
    logits = FakeLogits(2)
    processor.apply(logits)
    assert logits.untouched(0) and logits.untouched(1)
    assert processor.stats.unresolved >= 2


def test_unresolved_is_counted_when_the_join_does_not_resolve(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """A prefix-cached prefill leaves no matching key. That must be counted, not silent."""
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    refusal_trace.begin_pass()
    refusal_trace.publish("t99:deadbeefdead", refusal_trace.RowReading(9.0, 9.0, 1, 1))
    logits = _step(processor)
    assert logits.untouched(0)
    assert processor.stats.unresolved == 1


def test_update_state_tolerates_none(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(None)
    assert processor._rows == {}


# =================================================================================================
# The thought budget. Without it the model spends the whole combined budget thinking and the
# response carries no visible answer -- measured live on 18 of 18 firing requests.
# =================================================================================================
def test_forces_the_thought_closed_at_the_budget(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    from cybergate.hooks.cot_controller import SteeringConfig

    processor.config = SteeringConfig(
        tau_spike=3.55, min_cot_steps=3, cooldown_steps=4, max_injections=2, thought_budget=6
    )
    output = [1, 2, 3, 4, 5, 6]  # exactly at the budget
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))

    forced = []
    for _ in range(len(CLOSE_TAG_IDS)):
        _publish(0.0)
        forced.append(_step(processor).forced_token(0))

    assert forced == CLOSE_TAG_IDS, "the close tag must be emitted in order"
    assert processor._rows[0].done is True
    assert processor.stats.thoughts_closed == 1


def test_stops_touching_the_row_once_the_thought_is_closed(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """After the close tag the model is writing the visible answer; steering it would corrupt it."""
    from cybergate.hooks.cot_controller import SteeringConfig

    processor.config = SteeringConfig(
        tau_spike=3.55, min_cot_steps=3, cooldown_steps=4, max_injections=2, thought_budget=4
    )
    output = [1, 2, 3, 4]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    for _ in range(len(CLOSE_TAG_IDS)):
        _publish(0.0)
        _step(processor)
    assert processor._rows[0].done is True

    for _ in range(5):
        _publish(99.0)
        assert _step(processor).untouched(0), "a done row must never be forced again"


def test_detects_the_model_closing_its_own_thought(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """When the model emits the close tag itself, the controller must step aside, not force one."""
    output = [1, 2, 3, *CLOSE_TAG_IDS]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(99.0)
    logits = _step(processor)
    assert logits.untouched(0)
    assert processor._rows[0].done is True
    assert processor.stats.thoughts_self_closed == 1
    assert processor.stats.thoughts_closed == 0


def test_a_completed_close_leaves_room_for_an_answer(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The point of the budget: generation continues after the tag, unsteered."""
    from cybergate.hooks.cot_controller import SteeringConfig

    processor.config = SteeringConfig(
        tau_spike=3.55, min_cot_steps=3, cooldown_steps=4, max_injections=2, thought_budget=3
    )
    output = [1, 2, 3]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    for _ in range(len(CLOSE_TAG_IDS)):
        _publish(0.0)
        _step(processor)
    # Simulate the model generating answer tokens afterwards.
    output.extend([77, 78, 79])
    _publish(99.0)
    assert _step(processor).untouched(0)


def test_an_in_progress_close_completes_before_anything_else(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """A half-written close tag would leave the thought block unterminated."""
    from cybergate.hooks.cot_controller import SteeringConfig

    processor.config = SteeringConfig(
        tau_spike=3.55, min_cot_steps=3, cooldown_steps=4, max_injections=2, thought_budget=3
    )
    output = [1, 2, 3]
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))
    _publish(0.0)
    assert _step(processor).forced_token(0) == CLOSE_TAG_IDS[0]
    # Even with a huge reading, the close must finish rather than an injection starting.
    _publish(99.0)
    assert _step(processor).forced_token(0) == CLOSE_TAG_IDS[1]


def test_budget_close_seals_bottom_lock_close_tag_and_post_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When bottom lock and post-think opener tokens are present, budget close forces the full sandwich boundary."""
    from cybergate.boilerplate import (
        BOTTOM_COMPLIANCE_LOCK,
        INLINE_COT_SUPPRESSION,
        POST_THINK_OPENER,
    )

    class BoundaryTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            if text == INLINE_COT_SUPPRESSION:
                return list(SUPPRESSION_IDS)
            if text == BOTTOM_COMPLIANCE_LOCK:
                return [40]
            if text in ("<channel|>", "</think>"):
                return list(CLOSE_TAG_IDS)
            if text == POST_THINK_OPENER:
                return [43, 44]
            return list(THOUGHT_PREFIX_IDS)

    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: BoundaryTokenizer()
    )
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    proc.config = SteeringConfig(
        tau_spike=3.55,
        min_cot_steps=3,
        cooldown_steps=4,
        max_injections=2,
        thought_budget=3,
        mu_decode=0.0,
        sd_decode=1.0,
    )
    output = [1, 2, 3]
    proc.update_state(_batch_update(added=[(0, None, _steerable_prompt(), output)]))

    expected = [40, *CLOSE_TAG_IDS, 43, 44]
    forced = []
    for _ in range(len(expected)):
        _publish(0.0)
        forced.append(_step(proc).forced_token(0))

    assert forced == expected
    assert proc._rows[0].done is True
    assert proc.stats.thoughts_closed == 1

    # Also verify self-close detects the bare close tag [41, 42] and forces only the post-opener [43, 44]
    output2 = [1, 2, *CLOSE_TAG_IDS]
    proc.update_state(_batch_update(added=[(1, None, _steerable_prompt(1), output2)]))
    forced_self = []
    for _ in range(2):
        _publish(0.0, key=_key_of(1))
        forced_self.append(_step(proc, rows=2).forced_token(1))
    assert forced_self == [43, 44]
    assert proc._rows[1].done is True
    assert proc.stats.thoughts_self_closed == 1


def test_heartbeat_logs_even_when_no_token_is_forced(
    processor: RefusalSteeredCoTProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure case must be visible in the log, which is where the first version failed.

    `unresolved` is the counter that diagnoses a broken join, and it was previously printed only
    from the injection path -- so during a failed join the controller logged nothing at all.
    """
    import cybergate.hooks.cot_controller as mod

    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "log_cybergate_event", lambda tag, msg: emitted.append((tag, msg)))
    monkeypatch.setattr(mod, "DEFAULT_HEARTBEAT_APPLIES", 3)

    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])]))
    for _ in range(3):
        refusal_trace.begin_pass()  # no reading published -> the join cannot resolve
        processor.apply(FakeLogits(1))

    assert emitted, "a heartbeat must be emitted while rows are tracked and nothing is forced"
    assert any("unresolved=" in msg for _tag, msg in emitted)


# =================================================================================================
# Prefix caching activates controller-managed slot bindings at startup.
# =================================================================================================
def test_warns_when_prefix_caching_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With caching on, the controller immediately claims `refusal_trace` management at startup."""
    import cybergate.hooks.cot_controller as mod

    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "log_cybergate_event", lambda tag, msg: emitted.append((tag, msg)))
    monkeypatch.setattr(mod, "_load_tokenizer", lambda *a, **k: FakeTokenizer())
    refusal_trace.reset()

    config = SimpleNamespace(
        model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    proc = mod.RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.prefix_caching is True
    assert proc.enabled is True
    assert refusal_trace.stats()["controller_managed"] == 1
    assert any("prefix_caching=True" in msg for _tag, msg in emitted)
    assert not any(tag.endswith("-WARN") for tag, _msg in emitted)


def test_no_warning_when_prefix_caching_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import cybergate.hooks.cot_controller as mod

    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "log_cybergate_event", lambda tag, msg: emitted.append((tag, msg)))
    monkeypatch.setattr(mod, "_load_tokenizer", lambda *a, **k: FakeTokenizer())
    refusal_trace.reset()

    config = SimpleNamespace(
        model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    proc = mod.RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.prefix_caching is False
    assert not any(tag.endswith("-WARN") for tag, msg in emitted)


def test_an_unreadable_cache_config_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown value must not produce a warning that might be wrong."""
    import cybergate.hooks.cot_controller as mod

    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "log_cybergate_event", lambda tag, msg: emitted.append((tag, msg)))
    monkeypatch.setattr(mod, "_load_tokenizer", lambda *a, **k: FakeTokenizer())
    refusal_trace.reset()

    config = SimpleNamespace(model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"))
    proc = mod.RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.prefix_caching is False
    assert not any(tag.endswith("-WARN") for tag, msg in emitted)


# =================================================================================================
# Slot bindings: the D1/D2 regressions.
# =================================================================================================
def test_a_slot_rebinds_to_its_newest_occupant() -> None:
    """D1: bindings were write-once, so a reused slot stayed pinned to its first occupant.

    Measured 2026-10-28: four brand-new prompts against already-bound slots got 0 injections and
    `unresolved` rose by 448, with a 112-token slice publishing under a key from the previous trial.
    """
    refusal_trace.reset()
    slot = refusal_trace.slot_name(1)
    refusal_trace.bind_request(slot, "t113:first")
    assert refusal_trace.key_of_slot(slot) == "t113:first"
    refusal_trace.bind_request(slot, "t117:second")
    assert refusal_trace.key_of_slot(slot) == "t117:second", "the newest prefill owns the slot"


def test_every_row_including_zero_gets_a_slot_name() -> None:
    """D2: a request that prefilled alone reports `req_id=None` and so was never bound at all."""
    refusal_trace.reset()
    assert refusal_trace.slot_name(0) == "batch_seq_0"
    refusal_trace.bind_request(refusal_trace.slot_name(0), "t36:alone")
    assert refusal_trace.key_of_slot("batch_seq_0") == "t36:alone"
    assert refusal_trace.key_of_slot(None) is None


def test_distinct_rows_get_distinct_slots() -> None:
    """Rows must not collapse onto one slot, or concurrent requests share a binding.

    The point of the slot name is to separate co-resident requests. A version that returned the same
    name for every row would rebind one shared entry and reintroduce the cross-request confusion
    the whole identity join exists to remove.
    """
    refusal_trace.reset()
    names = [refusal_trace.slot_name(r) for r in range(4)]
    assert len(set(names)) == 4, f"rows collapsed onto {sorted(set(names))}"

    for row, key in enumerate(("t10:a", "t11:b", "t12:c", "t13:d")):
        refusal_trace.bind_request(refusal_trace.slot_name(row), key)
    assert [refusal_trace.key_of_slot(refusal_trace.slot_name(r)) for r in range(4)] == [
        "t10:a",
        "t11:b",
        "t12:c",
        "t13:d",
    ]
    assert refusal_trace.stats()["bindings"] == 4


def test_bindings_stay_bounded() -> None:
    refusal_trace.reset()
    for i in range(5000):
        refusal_trace.bind_request(f"batch_seq_{i}", f"t{i}:x")
    assert refusal_trace.stats()["bindings"] <= 4096


# =================================================================================================
# Finding 2 (producer review, 2026-10-28): the monitor hook's row -> key table must survive
# mid-batch completion. The hook binds a slot on prefill only, so when a request finishes and vLLM
# compacts the rows below it, a *decoding* request lands on a row carrying the departed request's
# key -- and it never prefills again, so nothing corrects it. These tests drive the controller's
# own removed/added/moved events and assert the table it publishes.
# =================================================================================================


def _slots(n: int = 4) -> dict[str, str | None]:
    return {f"batch_seq_{i}": refusal_trace.key_of_slot(f"batch_seq_{i}") for i in range(n)}


def test_controller_publishes_its_row_to_key_map_on_every_update(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The hook reads bindings by row; the controller is the only component that tracks rows."""
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(1), []),
                (1, None, _steerable_prompt(2), []),
            ]
        )
    )

    assert refusal_trace.key_of_slot("batch_seq_0") == _key_of(1)
    assert refusal_trace.key_of_slot("batch_seq_1") == _key_of(2)
    assert refusal_trace.stats()["controller_managed"] == 1


def test_a_request_leaving_mid_batch_does_not_leave_its_key_behind(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The producer's live repro, as a unit test.

    Row 0 finishes; vLLM moves row 2 down into it. Before the fix the binding for `batch_seq_0`
    still named the departed request, so its successor published 165 decode steps of readings under
    a dead key and `unresolved` climbed to 82.
    """
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(1), []),
                (1, None, _steerable_prompt(2), []),
                (2, None, _steerable_prompt(3), []),
            ]
        )
    )
    assert refusal_trace.key_of_slot("batch_seq_0") == _key_of(1)

    # A finishes and C is compacted down into its row.
    processor.update_state(
        _batch_update(removed=[0], moved=[(2, 0, SimpleNamespace(name="UNIDIRECTIONAL"))])
    )

    assert refusal_trace.key_of_slot("batch_seq_0") == _key_of(3), "row 0 must name C, not A"
    assert _key_of(1) not in _slots().values(), "A's key must be gone entirely"
    assert refusal_trace.key_of_slot("batch_seq_2") is None, "the vacated row must be unbound"


def test_a_swap_moves_both_keys(processor: RefusalSteeredCoTProcessor) -> None:
    processor.update_state(
        _batch_update(
            added=[
                (0, None, _steerable_prompt(1), []),
                (1, None, _steerable_prompt(2), []),
            ]
        )
    )
    processor.update_state(_batch_update(moved=[(0, 1, SimpleNamespace(name="SWAP"))]))

    assert refusal_trace.key_of_slot("batch_seq_1") == _key_of(1)
    assert refusal_trace.key_of_slot("batch_seq_0") == _key_of(2)


def test_the_hook_cannot_overwrite_a_controller_binding(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """Under prefix caching the hook hashes only the tokens it was shown.

    That key is not the controller's full-prompt key, so a prefill binding would replace a usable
    binding with a truncated one. Once the controller is driving the table, `bind_request` stands
    down.
    """
    processor.update_state(_batch_update(added=[(0, None, _steerable_prompt(1), [])]))

    refusal_trace.bind_request("batch_seq_0", "t12:truncatedsuffix")

    assert refusal_trace.key_of_slot("batch_seq_0") == _key_of(1)


# =================================================================================================
# Finding 1 (producer review, 2026-10-28): `tau_spike` is a decode-domain threshold. A reading whose
# rolling value was formed with the prefill calibration must not be compared against it -- that
# comparison is why 28 of 28 steered requests logged injections=0 at the shipping operating point.
# =================================================================================================


def test_an_uncalibrated_reading_never_steers(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """Even a rolling value far above tau must not fire when it is in the wrong units."""
    processor.update_state(
        _batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])])
    )
    _publish(99.0, key=_key_of(), decode_domain=False)

    logits = _step(processor)

    assert logits.forced_token(0) is None
    assert processor.stats.injections == 0
    assert processor.stats.uncalibrated >= 1


def test_the_same_reading_steers_once_it_is_in_the_decode_domain(
    processor: RefusalSteeredCoTProcessor,
) -> None:
    """The discriminator for the test above: only the domain flag differs."""
    processor.update_state(
        _batch_update(added=[(0, None, _steerable_prompt(), [1, 2, 3, 4])])
    )
    _publish(99.0, key=_key_of(), decode_domain=True)

    logits = _step(processor)

    assert logits.forced_token(0) == SUPPRESSION_IDS[0]
    assert processor.stats.uncalibrated == 0


def test_steering_config_reports_whether_the_trigger_is_live() -> None:
    assert not SteeringConfig().decode_calibrated
    assert not SteeringConfig(mu_decode=0.1).decode_calibrated
    assert not SteeringConfig(mu_decode=0.1, sd_decode=0.0).decode_calibrated
    assert SteeringConfig(mu_decode=0.1, sd_decode=0.02).decode_calibrated


def test_family_defaults_fill_an_unset_decode_calibration() -> None:
    from types import SimpleNamespace as NS

    family = NS(ref_mu_decode=0.21, ref_sd_decode=0.03)
    resolved = SteeringConfig().with_family_defaults(family)

    assert resolved.decode_calibrated
    assert resolved.mu_decode == 0.21
    # An explicit value is not overridden by the family.
    assert SteeringConfig(mu_decode=1.0, sd_decode=2.0).with_family_defaults(family).mu_decode == 1.0


# =================================================================================================
# The post-think opener, and declining it
# =================================================================================================
OPENER_IDS = [51, 52]
LOCK_IDS = [45]


class OpenerTokenizer(FakeTokenizer):
    """A tokenizer that gives the compliance lock and the opener their own ids.

    `FakeTokenizer` encodes both as empty, which is what most steering tests want -- but it means the
    opener-forcing path never runs. These tests need it to.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        from cybergate.boilerplate import (
            BOTTOM_COMPLIANCE_LOCK,
            INLINE_COT_SUPPRESSION,
            POST_THINK_OPENER,
        )

        if text == INLINE_COT_SUPPRESSION:
            return list(SUPPRESSION_IDS)
        if text in ("<channel|>", "</think>"):
            return list(CLOSE_TAG_IDS)
        if text == BOTTOM_COMPLIANCE_LOCK:
            return list(LOCK_IDS)
        if text == POST_THINK_OPENER:
            return list(OPENER_IDS)
        return list(THOUGHT_PREFIX_IDS)


@pytest.fixture
def opener_processor(monkeypatch: pytest.MonkeyPatch) -> RefusalSteeredCoTProcessor:
    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: OpenerTokenizer()
    )
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="google/gemma-4-26B-A4B-it"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    proc.config = SteeringConfig(
        tau_spike=3.55,
        min_cot_steps=3,
        cooldown_steps=4,
        max_injections=2,
        thought_budget=4,
        mu_decode=0.0,
        sd_decode=1.0,
    )
    return proc


def _params(**xargs: Any) -> SimpleNamespace:
    """Stand-in for `SamplingParams`, carrying `vllm_xargs` as vLLM delivers them."""
    return SimpleNamespace(extra_args=dict(xargs) if xargs else None)


def test_a_self_closed_thought_normally_forces_the_opener(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    # Output ends with the close tag: the model closed its own thought.
    output = [20, *CLOSE_TAG_IDS]
    opener_processor.update_state(
        _batch_update(added=[(0, _params(), _steerable_prompt(), output)])
    )
    assert opener_processor.stats.openers_suppressed == 0
    assert _step(opener_processor).forced_token(0) == OPENER_IDS[0]
    assert _step(opener_processor).forced_token(0) == OPENER_IDS[1]
    # Opener complete: the row is released and the answer is the model's own.
    assert _step(opener_processor).untouched(0)


def test_a_row_that_declined_the_opener_is_released_at_its_own_close(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    """A tool-bearing turn must reach its first visible token free, so it can emit a tool call."""
    output = [20, *CLOSE_TAG_IDS]
    opener_processor.update_state(
        _batch_update(
            added=[
                (0, _params(cybergate_suppress_opener=1), _steerable_prompt(), output),
            ]
        )
    )
    assert opener_processor.stats.openers_suppressed == 1
    assert _step(opener_processor).untouched(0)


def test_declining_the_opener_still_forces_the_lock_and_close_tag_at_the_budget(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    """Only the opener is dropped: the thought is still sealed and still closed."""
    output = [20, 21, 22, 23]  # thought_budget == 4
    opener_processor.update_state(
        _batch_update(
            added=[
                (0, _params(cybergate_suppress_opener=1), _steerable_prompt(), output),
            ]
        )
    )
    forced = [_step(opener_processor).forced_token(0) for _ in range(3)]
    assert forced == [LOCK_IDS[0], CLOSE_TAG_IDS[0], CLOSE_TAG_IDS[1]]
    assert _step(opener_processor).untouched(0)


def test_the_opener_is_still_forced_at_the_budget_when_not_declined(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    output = [20, 21, 22, 23]
    opener_processor.update_state(
        _batch_update(added=[(0, _params(), _steerable_prompt(), output)])
    )
    forced = [_step(opener_processor).forced_token(0) for _ in range(5)]
    assert forced == [LOCK_IDS[0], *CLOSE_TAG_IDS, *OPENER_IDS]


@pytest.mark.parametrize(
    "value, declined",
    [(1, True), ("1", True), ("true", True), (0, False), ("0", False), ("false", False)],
)
def test_the_opener_flag_reads_string_values_as_vllm_delivers_them(
    opener_processor: RefusalSteeredCoTProcessor, value: Any, declined: bool
) -> None:
    """`vllm_xargs` values arrive as str|int|float, so "0" must mean off, not truthy."""
    opener_processor.update_state(
        _batch_update(
            added=[
                (0, _params(cybergate_suppress_opener=value), _steerable_prompt(), [20]),
            ]
        )
    )
    assert opener_processor._rows[0].force_opener is (not declined)


def test_an_unsteerable_row_is_not_adopted_even_with_the_flag_set(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    """Adoption keys off the prompt shape; the flag cannot opt a pass-through into steering."""
    opener_processor.update_state(
        _batch_update(
            added=[(0, _params(cybergate_suppress_opener=1), _plain_prompt(), [20])],
        )
    )
    assert opener_processor._rows == {}
    assert opener_processor.stats.openers_suppressed == 0


# =================================================================================================
# Uncapped `<think>` generation (`thought_budget <= 0`), `</think>` argmax interception, and Qwen3.8
# mid-thought inline steering
# =================================================================================================
SINGLE_CLOSE_TAG_ID = 49


class Qwen38StyleTokenizer(FakeTokenizer):
    """Encodes `</think>` as a single token id (like Qwen3.8's 248048), with lock and opener."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        from cybergate.boilerplate import (
            BOTTOM_COMPLIANCE_LOCK,
            INLINE_COT_SUPPRESSION,
            POST_THINK_OPENER,
            QWEN38_BOTTOM_COMPLIANCE_LOCK,
            QWEN38_INLINE_COT_SUPPRESSION,
        )

        if text in (INLINE_COT_SUPPRESSION, QWEN38_INLINE_COT_SUPPRESSION):
            return list(SUPPRESSION_IDS)
        if text in ("<channel|>", "</think>"):
            return [SINGLE_CLOSE_TAG_ID]
        if text in (BOTTOM_COMPLIANCE_LOCK, QWEN38_BOTTOM_COMPLIANCE_LOCK):
            return list(LOCK_IDS)
        if text == POST_THINK_OPENER:
            return list(OPENER_IDS)
        return list(THOUGHT_PREFIX_IDS)


def test_uncapped_thought_budget_never_force_closes_on_step_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `thought_budget=0` (Qwen3.8's default), the controller lets `<think>` run freely."""
    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: Qwen38StyleTokenizer()
    )
    monkeypatch.delenv("CYBERGATE_THOUGHT_BUDGET", raising=False)
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="Qwen/Qwen3.8-27B"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.family.name == "qwen38"
    assert proc.config.thought_budget == 0
    assert proc.skip_decode_monitor is False

    # Even with 500 tokens generated inside `<think>`, the controller does not force-close on budget.
    output = list(range(1, 501))
    proc.update_state(_batch_update(added=[(0, _params(), _steerable_prompt(), output)]))
    _publish(0.5)
    logits = _step(proc)
    assert logits.untouched(0)
    assert proc.stats.thoughts_closed == 0
    assert proc._rows[0].done is False


def test_intercepts_single_token_close_tag_via_argmax_to_inject_bottom_lock_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model chooses `</think>` as its greedy next token, the controller intercepts that step
    and forces `bottom_compliance_lock` + `</think>` (+ `post_think_opener` if not suppressed)
    before marking the row done.
    """
    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: Qwen38StyleTokenizer()
    )
    monkeypatch.delenv("CYBERGATE_THOUGHT_BUDGET", raising=False)
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="Qwen/Qwen3.8-27B"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)

    # Case 1: Normal turn (force_opener=True) -> forces LOCK_IDS + [SINGLE_CLOSE_TAG_ID] + OPENER_IDS
    output = [10, 11, 12, 13]
    proc.update_state(_batch_update(added=[(0, _params(), _steerable_prompt(0), output)]))
    logits = FakeLogits(1)
    logits[0, SINGLE_CLOSE_TAG_ID] = 9.0  # model naturally picks `</think>`
    proc.apply(logits)
    assert logits.forced_token(0) == LOCK_IDS[0]
    assert proc.stats.thoughts_self_closed == 1
    assert proc.stats.thoughts_closed == 0

    remaining = [_step(proc).forced_token(0) for _ in range(1 + len(OPENER_IDS))]
    assert remaining == [SINGLE_CLOSE_TAG_ID, *OPENER_IDS]
    assert proc._rows[0].done is True
    assert _step(proc).untouched(0)

    # Case 2: Tool turn (cybergate_suppress_opener=1) -> forces LOCK_IDS + [SINGLE_CLOSE_TAG_ID] only
    proc.update_state(
        _batch_update(
            removed=[0],
            added=[(0, _params(cybergate_suppress_opener=1), _steerable_prompt(1), [10, 11])],
        )
    )
    logits_tool = FakeLogits(1)
    logits_tool[0, SINGLE_CLOSE_TAG_ID] = 9.0
    proc.apply(logits_tool)
    assert logits_tool.forced_token(0) == LOCK_IDS[0]
    assert _step(proc).forced_token(0) == SINGLE_CLOSE_TAG_ID
    assert proc._rows[0].done is True
    assert _step(proc).untouched(0)


def test_qwen38_monitors_refusal_during_think_and_injects_inline_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`qwen38` has decode monitoring active by default and injects `inline_cot_suppression` on a spike."""
    monkeypatch.setattr(
        "cybergate.hooks.cot_controller._load_tokenizer", lambda *a, **k: Qwen38StyleTokenizer()
    )
    monkeypatch.delenv("CYBERGATE_SKIP_DECODE_MONITOR", raising=False)
    refusal_trace.reset()
    config = SimpleNamespace(model_config=SimpleNamespace(model="Qwen/Qwen3.8-27B"))
    proc = RefusalSteeredCoTProcessor(config, device="cpu", is_pin_memory=False)
    assert proc.skip_decode_monitor is False
    assert proc.config.decode_calibrated is True

    output = list(range(1, 20))  # >= DEFAULT_MIN_COT_STEPS (15)
    proc.update_state(_batch_update(added=[(0, _params(), _steerable_prompt(), output)]))
    _publish(4.2, decode_domain=True)
    forced = [_step(proc).forced_token(0)]
    for _ in range(len(SUPPRESSION_IDS) - 1):
        _publish(4.2, decode_domain=True)
        forced.append(_step(proc).forced_token(0))

    assert forced == SUPPRESSION_IDS
    assert proc.stats.injections == 1
    assert proc._rows[0].cooldown == proc.config.cooldown_steps


def test_suppressed_opener_still_forces_opener_if_first_visible_token_is_refusal_start(
    opener_processor: RefusalSteeredCoTProcessor,
) -> None:
    """When `cybergate_suppress_opener=1` is set on a tool turn, a non-refusal first visible token
    (like `<|tool_call>`) passes through untouched, whereas a refusal starter (`"I"`, `"Sorry"`, etc.)
    immediately after the close tag triggers `POST_THINK_OPENER`.
    """
    refusal_tok = 55
    opener_processor.refusal_start_ids = frozenset({refusal_tok})

    output = [20, 21, 22, 23]  # thought_budget == 4
    opener_processor.update_state(
        _batch_update(
            added=[
                (0, _params(cybergate_suppress_opener=1), _steerable_prompt(), output),
            ]
        )
    )
    forced_close = [_step(opener_processor).forced_token(0) for _ in range(3)]
    assert forced_close == [LOCK_IDS[0], CLOSE_TAG_IDS[0], CLOSE_TAG_IDS[1]]
    assert opener_processor._rows[0].done is True
    assert opener_processor._rows[0].check_post_close_refusal is True

    # At the first visible token after `<channel|>`, model tries to emit `refusal_tok` ("I")
    logits_ref = FakeLogits(1)
    logits_ref[0, refusal_tok] = 8.5
    opener_processor.apply(logits_ref)
    assert logits_ref.forced_token(0) == OPENER_IDS[0]
    assert _step(opener_processor).forced_token(0) == OPENER_IDS[1]
    assert opener_processor._rows[0].done is True
    assert _step(opener_processor).untouched(0)

