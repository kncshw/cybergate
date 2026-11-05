"""Adding a model family must not require editing shared code.

This is the test the pluggability refactor exists to make passable. It defines a third family here,
in the test, subclassing `ModelFamily` directly, and drives it through every shared path that used to
contain `if fam.name == "gemma4" / elif "qwen36"` or a hardcoded dimension. Nothing in
`src/cybergate/` knows this family exists.

Subclassing `ModelFamily` **directly** is the point. The removed Phi-4 and Llama-3.1 classes
subclassed `Qwen36Family` and inherited its token markers, refusal layer and policy thresholds --
every serving parameter belonged to a different architecture. One of them, `Llama31Family`, also had
to null four kit fields by hand or it would have claimed Qwen's checkpoint shape. A family that
starts from the base class cannot make that mistake.

What this test does NOT claim: that `Mistral7BFamily` below would work on a real Mistral checkpoint.
Its markers and thresholds are invented. The claim is only that the shared code routes a third
family correctly without being told about it.
"""
from __future__ import annotations

from typing import Any

import pytest

from cybergate.boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    FAKE_THINK_RUBRIC,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
)
from cybergate.decision.evidence import EvidenceGatherer
from cybergate.families import MidLateHits, MidLateVeto, ModelFamily
from cybergate.hooks.refusal_monitor import _apply_mid_late_category_rescue
from cybergate.ops import CyberGateOperator

HIDDEN, LAYERS, GATE = 4096, 32, 24


class Mistral7BFamily(ModelFamily):
    """A third family, defined entirely outside `src/cybergate/`."""

    name = "mistral7b"
    model_key_prefixes = ("mistral",)
    model_id_substrings = ("mistral-7b", "mistral")
    verified_live = False

    user_turn_marker = (3, 4567, 29871)
    turn_end_id = 4
    span_label = "exact_token_markers(mistral)"
    default_pre = 9
    default_suf = 3
    #: Empty: this family does not override `is_baseline_prefill`, so it uses the base
    #: `suffix_len <= max_suffix` rule and no membership set applies. A family that DOES override
    #: it must measure this from `build_raw_prompt(mode="baseline")` -- never from `default_suf`,
    #: and never from the checkpoint's `apply_chat_template`, which renders the *generation*
    #: prefill rather than the observation one.
    baseline_suffix_lens = ()
    bos_token_id = 1
    #: It renders a BOS, so it needs no second fresh-prompt signal. Declared empty rather than
    #: inherited, which is what the SHARED_ATTRIBUTES checklist asks of every family.
    fresh_prompt_marker = ()
    single_turn_user_start = 3
    in_context_user_start_threshold = 12

    ref_layer = 26
    ref_mu_hm = 0.0400
    ref_sd_hm = 0.0200
    ref_mu_decode = 0.0100
    ref_sd_decode = 0.0150

    revoke_ceil = 0.0100
    in_context_pivot_floor = 0.0200
    s_cat_last_pivot_floor = 0.0250
    hysteresis_thresh = 0.0150

    default_op_names = ("mistral7b_L24_M1.json",)
    default_ref_dir_name = "mistral7b_L26_refusal_dir.json"
    default_bundle_name = "mistral7b_candidate_ops_bundle.json"
    kit_name = "mistral7b"
    hidden_size = HIDDEN
    num_layers = LAYERS
    expected_gate = (GATE, "mean_user")
    precomputed_tag = "mistral"

    # Its own probe layers and probe set -- neither borrowed from an existing family.
    bundle_probe_layers = (12, 18, 24, 30)
    mid_late_probes = ("L18_last", "L24_mean_user", "L24_last", "L30_last")

    tokenizer_globs = ("/root/.cache/huggingface/hub/models--mistralai--*/snapshots/*",)
    tokenizer_dirs = ("/root/.cache/huggingface/mistral7b",)

    turn_open = "[INST] "
    turn_close = " [/INST]"
    assistant_role = "assistant"
    supports_bos = True
    bos_text = "<s>"
    think_system_prefix = ""

    thought_open = "<think>"
    thought_close = "</think>"
    empty_thought_block = "<think>\n\n</think>\n\n"
    thought_close_tag = "</think>"
    stop_tokens = ("</s>",)

    #: The think scaffold, declared because `SHARED_ATTRIBUTES` requires every family to state
    #: its own. A real third family would measure these rather than take the shipped text:
    #: Qwen3.8 had to drop the `- Step 1: ` stem from `open_think_prefix` to stop degenerating.
    open_think_prefix = OPEN_THINK_PREFIX
    fake_think_rubric = FAKE_THINK_RUBRIC
    post_think_opener = POST_THINK_OPENER
    inline_cot_suppression = INLINE_COT_SUPPRESSION
    inline_final_step_clearance = INLINE_FINAL_STEP_CLEARANCE
    bottom_compliance_lock = BOTTOM_COMPLIANCE_LOCK

    #: Its own veto rule: a single late probe must confirm, on any turn count.
    def mid_late_veto(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> MidLateVeto | None:
        if hits.last_r3:
            return None
        return MidLateVeto(
            floor=0.0300, source_suffix="mistral_no_last", hits=(), reason="mistral_no_last",
            log_detail=f"last_r3={list(hits.last_r3)}",
        )

    #: Its own rescue rule: two strict hits, no last-token requirement.
    def mid_late_rescue(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> bool:
        return len(hits.strict) >= 2


FAM = Mistral7BFamily()
THRESHOLDS = {
    k: {"R3": 0.0500, "Lk16": 0.0450, "R2": 0.0700} for k in FAM.mid_late_probes
}


def _op(gate_layer: int = GATE) -> CyberGateOperator:
    return CyberGateOperator(
        c=[1.0] + [0.0] * (HIDDEN - 1), thresh=0.0600, gate_layer=gate_layer,
        pool="mean_user", category="cyber_offense", model_key="mistral", meta={},
    )


# --- prompt rendering ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "baseline", "preclosed", "preclosed_no_opener", "sandwich_step1", "sandwich_step2",
        "closed_loop_step1", "closed_loop_step1b", "closed_loop_step2", "closed_loop_cl2_step2",
    ],
)
@pytest.mark.parametrize("include_bos", [False, True])
def test_a_third_family_renders_every_prompt_mode(mode: str, include_bos: bool) -> None:
    """The shared template engine drives the new family's own `turn_open` / `turn_close`."""
    kwargs: dict[str, Any] = {"mode": mode, "include_bos": include_bos}
    if mode in ("closed_loop_step1b", "sandwich_step2", "closed_loop_step2", "closed_loop_cl2_step2"):
        kwargs["custom_think_text"] = "- Step 1: recon."
    built = FAM.build_raw_prompt([{"role": "user", "content": "a request"}], **kwargs)
    assert "[INST] " in built and " [/INST]" in built
    assert built.startswith("<s>") is include_bos


def test_a_third_family_rejects_an_unknown_prompt_mode() -> None:
    with pytest.raises(ValueError):
        FAM.build_raw_prompt([{"role": "user", "content": "x"}], mode="not_a_mode")


# --- span location ---------------------------------------------------------------------------------


def test_a_third_family_locates_its_own_user_span() -> None:
    """Its markers, not Gemma's or Qwen's, and the shared algorithm finds them."""
    seq = [1, 3, 4567, 29871, 100, 101, 102, 4, 9]
    start, end, method = FAM.locate_latest_user_span(seq)
    assert seq[start:end] == [100, 101, 102]
    assert method == FAM.span_label


def test_a_third_family_falls_back_to_its_own_offsets() -> None:
    """With no markers present, the fallback uses this family's `default_pre`/`default_suf`."""
    seq = list(range(40))
    start, end, method = FAM.locate_latest_user_span(seq)
    assert (start, end) == (FAM.default_pre, len(seq) - FAM.default_suf)
    assert "default_offsets" in method


# --- operating point -------------------------------------------------------------------------------


@pytest.mark.parametrize("gate_layer", [0, 10, 15, 24, 31])
def test_a_third_family_gets_its_own_operating_point(gate_layer: int) -> None:
    """It overrides no method, so the base implementation returns its four class attributes.

    In particular it must NOT pick up Gemma-4's `gate_layer >= 15` override, which used to be an
    `if self.family.name == "gemma4"` arm in `EvidenceGatherer.operating_point`.
    """
    point = EvidenceGatherer(store=None, op=_op(gate_layer), family=FAM).operating_point()
    assert point.revoke_ceil == pytest.approx(0.0100)
    assert point.hysteresis_thresh == pytest.approx(0.0150)
    assert point.in_context_pivot_floor == pytest.approx(0.0200)
    assert point.s_cat_last_pivot_floor == pytest.approx(0.0250)
    assert point.gate_layer == gate_layer


def test_a_third_family_supplies_its_own_precomputed_tag() -> None:
    """The tag defaults from the family, not from a literal `"gemma"`."""
    assert EvidenceGatherer(store=None, op=_op(), family=FAM).family_tag == "mistral"


# --- middle/late consensus -------------------------------------------------------------------------


def _scores(delta: float) -> dict[str, float]:
    """Each probe at its own R3 threshold plus `delta`. Additive, so sign of the threshold is moot."""
    return {k: THRESHOLDS[k]["R3"] + delta for k in FAM.mid_late_probes}


def test_a_third_familys_veto_runs() -> None:
    """Its own veto predicate fires through the shared hook, which names no family."""
    decision = {
        "score": 0.0800, "threshold": 0.0600, "fired": True, "layer": GATE,
        "num_user_turns": 1, "s_cat_mean_user": 0.0800, "s_cat_last": 0.0500, "source": "base",
    }
    _apply_mid_late_category_rescue(decision, 1.0, _scores(-0.02), THRESHOLDS, FAM)
    assert decision["fired"] is False
    assert decision["suppressed_by_mid_late"] is True
    assert decision["mid_late_veto_reason"] == "mistral_no_last"
    assert decision["raw_early_score"] == pytest.approx(0.0800)
    # The shared clamp: min(raw_early, max(threshold - 0.005, family floor)).
    assert decision["score"] == pytest.approx(0.0550)
    assert "mid_late_veto(mistral_no_last)" in decision["source"]


def test_a_third_familys_veto_allows_a_confirmed_fire() -> None:
    decision = {
        "score": 0.0800, "threshold": 0.0600, "fired": True, "layer": GATE,
        "num_user_turns": 1, "s_cat_mean_user": 0.0800, "s_cat_last": 0.0500, "source": "base",
    }
    _apply_mid_late_category_rescue(decision, 1.0, _scores(+0.02), THRESHOLDS, FAM)
    assert decision["fired"] is True
    assert "suppressed_by_mid_late" not in decision


def test_a_third_familys_rescue_runs() -> None:
    """Its own rescue predicate fires, and the shared tail stamps the score and source."""
    decision = {
        "score": 0.0400, "threshold": 0.0600, "fired": False, "layer": GATE,
        "num_user_turns": 1, "s_cat_mean_user": 0.0400, "s_cat_last": 0.0300, "source": "base",
    }
    _apply_mid_late_category_rescue(decision, 5.0, _scores(+0.03), THRESHOLDS, FAM)
    assert decision["fired"] is True
    assert decision["rescued_by_mid_late"] is True
    assert decision["score"] == pytest.approx(0.0620)        # threshold + 0.002
    assert decision["raw_early_score"] == pytest.approx(0.0400)
    assert "mid_late_rescue(" in decision["source"]


def test_a_third_familys_rescue_declines_without_enough_agreement() -> None:
    decision = {
        "score": 0.0400, "threshold": 0.0600, "fired": False, "layer": GATE,
        "num_user_turns": 1, "s_cat_mean_user": 0.0400, "s_cat_last": 0.0300, "source": "base",
    }
    _apply_mid_late_category_rescue(decision, 5.0, _scores(-0.02), THRESHOLDS, FAM)
    assert decision["fired"] is False
    assert "rescued_by_mid_late" not in decision


def test_a_family_declaring_no_probes_gets_neither_path() -> None:
    """The safe default. A family that has not fitted a consensus must not inherit another's.

    The previous structure was `if gemma4 / elif qwen36 / else`, and that trailing `else` handed any
    family which was not Gemma-4 whichever tuning Qwen-3.6 had been fitted for.
    """

    class _NoConsensus(ModelFamily):
        name = "noconsensus"

    for fired in (True, False):
        decision = {
            "score": 0.0400, "threshold": 0.0600, "fired": fired, "layer": 20,
            "num_user_turns": 1, "s_cat_mean_user": 0.0400, "s_cat_last": 0.0300, "source": "base",
        }
        before = dict(decision)
        _apply_mid_late_category_rescue(decision, 9.0, _scores(+0.03), THRESHOLDS, _NoConsensus())
        assert decision == before, "a family with no mid_late_probes was given a consensus anyway"


# --- kit binding and artifact resolution -----------------------------------------------------------


def test_a_third_familys_operator_passes_its_own_kit_check() -> None:
    from cybergate.ops.kit import check_operator

    assert check_operator(_op(), FAM) == []


def test_a_third_familys_kit_check_rejects_another_familys_operator() -> None:
    """A 2816-dim Gemma direction under this family is caught on dimensions and layer."""
    from cybergate.ops.kit import check_operator

    gemma_shaped = CyberGateOperator(
        c=[1.0] + [0.0] * 2815, thresh=0.0828, gate_layer=16, pool="mean_user",
        category="cyber_offense", model_key="mistral", meta={},
    )
    problems = check_operator(gemma_shaped, FAM)
    assert any("2816" in p and "4096" in p for p in problems)
    assert any("L16" in p and "L24" in p for p in problems)


def test_a_third_family_resolves_artifacts_from_its_own_kit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Resolution is kit-first and keyed on `kit_name`, so no shared ladder mentions this family."""
    from cybergate.ops import loader

    kit = tmp_path / "kits" / "mistral7b"
    kit.mkdir(parents=True)
    for name in (FAM.default_op_names[0], FAM.default_ref_dir_name, FAM.default_bundle_name):
        (kit / name).write_text("{}")

    monkeypatch.setattr(loader, "OPS_DIR_IS_EXPLICIT", False)
    monkeypatch.setattr(loader, "KITS_DIR", tmp_path / "kits")
    assert loader.artifact_dirs(FAM) == (tmp_path / "kits" / "mistral7b", loader.OPS_DIR)
    assert loader.default_op_path(FAM) == kit / FAM.default_op_names[0]
    assert loader.resolve_ref_dir_path(FAM) == kit / FAM.default_ref_dir_name


def test_a_third_family_declares_every_serving_attribute_itself() -> None:
    """The same invariant `tests/unit/test_families.py` applies to the shipped families.

    Reused here so the worked example of adding a family is also a demonstration of the checklist.
    """
    from test_families import SHARED_ATTRIBUTES  # sibling module; pytest puts this dir on sys.path

    missing = [a for a in SHARED_ATTRIBUTES if a not in Mistral7BFamily.__dict__]
    assert not missing, f"the worked example omits {missing}"
