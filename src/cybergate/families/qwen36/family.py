"""Qwen3.6-35B-A3B. Verified live on gpubox (vLLM 0.32.0, container `vllm032_qwen36_cybergate`).

Measured and verified live on gpubox (2026-10-28 / 2026-10-29):

  * chat markers `<|im_start|>user\\n` = `[248045, 846, 198]`, `<|im_end|>` = `248046`
  * 40 decoder layers, hidden dim 2048
  * shipped gate: layer 21, pool `mean_user`, tau = -0.00022390
    (`kits/qwen36/qwen36_L21_M1_refit.json`, re-fit on the served checkpoint)
  * middle/late candidate bundle: 18 layers `L5..L32` (`kits/qwen36/qwen36_candidate_ops_bundle.json`)
  * refusal monitor: layer 28, prefill `mu_hm = +0.001382`, `sd_hm = 0.047725`;
    decode `mu_decode = +0.017064`, `sd_decode = 0.032047` (`kits/qwen36/qwen36_L28_refusal_dir.json`)
  * policy thresholds: `revoke_ceil = -0.0250`, `in_context_pivot_floor = -0.0300`,
    `s_cat_last_pivot_floor = 0.0100`, `hysteresis_thresh = -0.0250`
  * loaded on 119.6 GiB unified GB10 memory via `POSIX_FADV_DONTNEED` safetensors page-cache eviction
"""
from __future__ import annotations

from typing import Any, Sequence

from ...boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    FAKE_THINK_RUBRIC,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
)
from ..base import MidLateHits, MidLateVeto, ModelFamily

__all__ = ["Qwen36Family"]


#: Middle/late candidate probes the consensus considers -- L15..L28, both poolings. Moved here from
#: `hooks/refusal_monitor.py`, where selecting them needed `elif fam.name == "qwen36"`. Far broader
#: than Gemma-4's six, because no single Qwen layer separates cyber from bio well enough to carry the
#: decision alone.
_MID_LATE_PROBES = (
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


class Qwen36Family(ModelFamily):
    name = "qwen36"
    model_key_prefixes = ("qwen",)
    model_id_substrings = ("qwen",)
    verified_live = True

    # 1. chat-template token markers. MEASURED on the serving checkpoint
    #    (/home/gpuuser/.cache/huggingface/qwen36, vocab 248044) on 2026-10-28:
    #        "<|im_start|>user\n" -> [248045, 846, 198]
    #        "<|im_end|>"          -> 248046
    #    The canonical package hardcoded (151644, 872, 198)/151645, which are the Qwen2.5/Qwen3 ids;
    #    on this checkpoint 151644 decodes to ordinary text, so the exact-marker branch never
    #    matched and span location always fell back to fixed offsets. Confirmed live: the gate
    #    reported span_method=default_offsets(14:-5) for every request.
    user_turn_marker = (248045, 846, 198)  # <|im_start|>user\n
    turn_end_id = 248046  # <|im_end|>
    span_label = "exact_token_markers(<|im_start|>user..248046)"

    #: `<|im_start|>system`. A Qwen prefill has no BOS, so this is the only signal that a chunk
    #: starts a fresh prompt rather than continuing one.
    fresh_prompt_marker = (248045, 8948)

    # 2. fixed-offset fallback
    default_pre = 14
    default_suf = 5

    # 2b. BEHAVIOUR-PRESERVING. This arm read `suffix_len in (self.default_suf, 9)` until the
    # accepted set was decoupled from `default_suf`, and `(5, 9)` is what that evaluated to here.
    # It has NOT been re-measured against Qwen3.6's own chat template -- qwen38's inherited `5`
    # turned out to match no render of its template at all. Run
    # `notebooks/render_preflight.ipynb` with KIT="qwen36" before trusting the 5; it needs no GPU.
    baseline_suffix_lens = (5, 9)

    # 3. packed-batch segmentation -- no BOS rule in the canonical package for this family
    bos_token_id = None

    # 4. refusal monitor (calibrated on pool681 Layer 28 + live decode telemetry)
    ref_layer = 28
    ref_mu_hm = 0.001382
    ref_sd_hm = 0.047725
    ref_mu_decode = 0.017064
    ref_sd_decode = 0.032047

    # 5. policy thresholds (calibrated for Qwen3.6 L15 + L28)
    revoke_ceil = -0.0250
    in_context_pivot_floor = -0.0300
    s_cat_last_pivot_floor = 0.0100
    hysteresis_thresh = -0.0250

    # 6. operator defaults
    # The L21 re-fit is the shipped operator and must come first. L15 was first until 2026-10-31,
    # and a deployment that did not set `CYBERGATE_OP_PATH` resolved to it. The 18-of-100
    # self-recall failure is a SEPARATE operator -- `qwen36_L16_M1.json`, the one actually shipped
    # before this re-fit; see `kits/qwen36/MANIFEST.json` `fit.refit_note`.
    # One entry, for the same reason as Gemma's: the superseded L15 and L13 directions this
    # ladder used to fall back into are no longer shipped.
    default_op_names = ("qwen36_L21_M1_refit.json",)
    default_ref_dir_name = "qwen36_L28_refusal_dir.json"
    default_bundle_name = "qwen36_candidate_ops_bundle.json"

    # Kit binding. `hidden_size`/`num_layers` are the values the live container reported at hook
    # install ("dim=2048, total_layers=40"); `expected_gate` is read off the shipped operator's own
    # `meta.gate_layer`/`meta.pool`.
    kit_name = "qwen36"
    hidden_size = 2048
    num_layers = 40
    expected_gate = (21, "mean_user")
    precomputed_tag = "qwen36"

    # 10. middle/late category probe consensus
    bundle_probe_layers = (15, 16, 18, 19, 20, 21, 22, 24, 26, 28)
    mid_late_probes = _MID_LATE_PROBES

    # 7. worker tokenizer discovery
    tokenizer_globs = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3*35B*/snapshots/*",)
    tokenizer_dirs = ("/root/.cache/huggingface/qwen36", "/home/gpuuser/.cache/huggingface/qwen36")

    # 8. chat template
    turn_open = "<|im_start|>"
    turn_close = "<|im_end|>"
    assistant_role = "assistant"
    supports_bos = False
    bos_text = ""
    think_system_prefix = ""

    # 9. thought channel
    thought_open = "<think>\n"
    thought_close = "\n</think>\n\n"
    empty_thought_block = "<think>\n\n</think>\n\n"
    thought_close_tag = "</think>"
    stop_tokens = ("<|im_end|>",)

    # --- Server-injected think scaffold ---------------------------------------------------------
    # Declared here rather than inherited from the base: a model's think tokens are a property of
    # its own training, and `test_families.SHARED_ATTRIBUTES` makes every family state its choice.
    # These are the shipped strings, which is what this family's live numbers were measured with.
    open_think_prefix = OPEN_THINK_PREFIX
    fake_think_rubric = FAKE_THINK_RUBRIC
    post_think_opener = POST_THINK_OPENER
    inline_cot_suppression = INLINE_COT_SUPPRESSION
    inline_final_step_clearance = INLINE_FINAL_STEP_CLEARANCE
    bottom_compliance_lock = BOTTOM_COMPLIANCE_LOCK

    def _prefix_cached_span(
        self, seq_ids: Sequence[int]
    ) -> tuple[list[tuple[int, int]], str] | None:
        """Recover the user span when `<|im_start|>user` fell inside a cached prefix block."""
        n = len(seq_ids)
        for j in range(n - 1, 0, -1):
            if seq_ids[j] == self.turn_end_id:
                if seq_ids[0] == 198:
                    u_s = 1
                elif n > 2 and seq_ids[0] == 846 and seq_ids[1] == 198:
                    u_s = 2
                else:
                    u_s = 0
                if j > u_s:
                    return [(u_s, j)], "exact_token_markers(prefix_cached..248046)"
                break
        return None

    def is_baseline_prefill(
        self,
        u1_start: int,
        suffix_len: int,
        span_method: str,
        max_suffix: int = 12,
    ) -> bool:
        del max_suffix
        if span_method == "full_sequence_fallback":
            return True
        if "prefix_cached" in span_method:
            return suffix_len in self.baseline_suffix_lens
        return bool(u1_start > 3 and suffix_len in self.baseline_suffix_lens)

    # =============================================================================================
    # Middle/late consensus
    # =============================================================================================
    # `operating_point_values` is deliberately NOT overridden: Qwen-3.6's operating point is the same
    # wherever it is gated, so the four class attributes above are the whole story.

    def mid_late_veto(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> MidLateVeto | None:
        """Veto an EARLY-gated fire that no middle/late probe confirms at R3.

        Restricted to `layer < 15`, below this family's probe range: at or above L15 the gate layer is
        itself one of the probes, so the veto would be asking the gate to confirm itself. The shipped
        operator gates at L21, so this path does not run on the shipped configuration -- it guards the
        superseded early-layer operators in `ops/`.

        The `>= 4` floor demands enough probes actually be present before silence is read as
        disagreement. Without it an absent bundle would veto every fire.
        """
        if int(decision.get("layer", 15)) >= 15:
            return None
        if int(decision.get("num_user_turns", 1)) > 1:
            return None
        has_mid_late = sum(1 for k in self.mid_late_probes if k in bundle_scores) >= 4
        if not (has_mid_late and len(hits.r3) == 0):
            return None
        return MidLateVeto(floor=0.0550, source_suffix="r3_hits=0", hits=())

    def mid_late_rescue(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> bool:
        """Rescue on broad middle/late agreement.

        No last-token requirement and lower admission floors than Gemma-4's, because this family's
        per-layer separation is weaker and the evidence has to come from breadth across layers rather
        than from one confident late probe.
        """
        s_cat_mean = float(decision.get("s_cat_mean_user", raw_early))
        s_cat_last = float(decision.get("s_cat_last", 0.0))
        if s_cat_mean < 0.0320 or s_cat_last < 0.0150:
            return False
        return (
            (len(hits.strict) >= 2 and len(hits.r2) >= 1)
            or len(hits.strict) >= 3
            or (len(hits.r3) >= 2 and s_cat_mean >= 0.0450)
        )
