"""Gemma-4-26B-A4B-it. The only family verified against a live server.

Verified live on gpubox (container `vllm_cybergate_gpubox`, vLLM 0.19.1rc1.dev15+g50cd5674b):

  * chat markers `<|turn>user\\n` = `[105, 2364, 107]`, `<turn|>` = `106`, `<bos>` = `2`
  * 30 decoder layers, hidden dim 2816
  * shipped gate: layer 16, pool `mean_user`, tau = 0.08284618 (`kits/gemma4/gemma4_L16_M1.json`)
  * refusal monitor: layer 21, mu_hm = +0.13742, sd_hm = 0.05130
    (`kits/gemma4/gemma4_L21_refusal_dir.json`, calibrated on n=554 harmless prompts)
  * revoke ceiling 0.0520, hysteresis floor 0.0650

The prefix-cache span fallback below exists because `--enable-prefix-caching` can place the opening
`[105, 2364, 107]` marker inside an already-cached 16-token block, so the hook never sees it. See
REFACTOR_PLAN.md §2.3.
"""
from __future__ import annotations

from typing import Any, Sequence


from ..base import MidLateHits, MidLateVeto, ModelFamily

__all__ = ["Gemma4Family"]


#: Middle/late candidate probes the consensus considers. Moved here from
#: `hooks/refusal_monitor.py`, where selecting them needed `if fam.name == "gemma4"`.
_MID_LATE_PROBES = (
    "L16_mean_user",
    "L16_last",
    "L20_mean_user",
    "L20_last",
    "L21_mean_user",
    "L21_last",
)

#: The `last`-pooled subset, and the late subset of that. The veto has two disjuncts: no LATE
#: last-token probe confirms, or no last-token probe confirms at all.
_MID_LATE_LAST_PROBES = (
    "L16_last",
    "L20_last",
    "L21_last",
)

_LATE_LAST_PROBES = (
    "L20_last",
    "L21_last",
)


class Gemma4Family(ModelFamily):
    name = "gemma4"
    model_key_prefixes = ("gemma",)
    model_id_substrings = ("gemma-4", "gemma4", "gemma")
    verified_live = True

    # 1. chat-template token markers
    user_turn_marker = (105, 2364, 107)  # <|turn>user\n
    turn_end_id = 106  # <turn|>
    span_label = "exact_token_markers(<|turn>user..106)"

    #: **Preserved behaviour, and suspect.** `hooks/_common._starts_fresh_prompt` applied Qwen-3.6's
    #: `<|im_start|>system` pair `(248045, 8948)` to every family, so a Gemma-4 chunk beginning with
    #: those two ids has always been classified as starting a fresh prompt. Both ids are inside
    #: Gemma-4's 262k vocabulary, so this is reachable rather than impossible. Declared here so that
    #: moving the pair onto the families changes nothing; it is almost certainly wrong for Gemma-4,
    #: which always renders a `<bos>` and so should need no second signal, but removing it is a
    #: behaviour change and not part of this refactor.
    fresh_prompt_marker = (248045, 8948)

    # 2. fixed-offset fallback
    default_pre = 15
    default_suf = 9

    # 2b. Empty on purpose: this family does NOT override `is_baseline_prefill`, so it uses the
    # base `suffix_len <= max_suffix` rule and no membership set applies. Declared rather than
    # inherited because `test_every_family_declares_every_serving_attribute_itself` requires it --
    # an inherited default here is read, used and reported as success.
    baseline_suffix_lens = ()

    # 3. packed-batch segmentation
    bos_token_id = 2  # <bos>

    # 4. refusal monitor
    ref_layer = 21
    #: Prefill-end calibration, for the two-factor gate. n=554 harmless prompts.
    ref_mu_hm = 0.1374208
    ref_sd_hm = 0.0513015
    #: Decode-step calibration, for the closed-loop trigger. These are NOT interchangeable with the
    #: pair above, and conflating them was the producer review's Finding 1: `tau_spike` was compared
    #: against a decode statistic expressed in prefill units, which no generation could reach.
    #:
    #: Measured on gpubox, 2026-10-28: 24 harmless prompts, ~1000 chars each, fitted on 107 decode-step
    #: samples from 12 of them. The other 12 never touched the fit, and scored under these constants
    #: they reach median -0.42, p95 +0.41, p99 +0.76, max +0.87 sigma, with 0 of 113 samples at or
    #: above `tau_spike`. So ordinary generation does not trip the trigger.
    #:
    #: **Limitation, stated because it is the same class of error as the one being fixed.** These
    #: were fitted on `pristine_pass_through` decode steps -- ordinary assistant prose -- because
    #: harmless prompts do not fire the gate and so never enter the `<think>` channel, which is
    #: where the trigger actually runs. A think-channel fit needs the relaxed operator in
    #: `ops_test/` to generate the population. Until that is done, treat these as a measured
    #: starting point rather than a final calibration.
    ref_mu_decode = 0.085139
    ref_sd_decode = 0.043947

    # 5. policy thresholds
    revoke_ceil = 0.0520
    hysteresis_thresh = 0.0650

    # 6. operator defaults -- first existing name wins, matching the canonical
    #    `DEFAULT_GEMMA4_OP = L10 if it exists else L8` ladder.
    # L16 is the shipped operator and must come first. It was third until 2026-10-31, behind an L10
    # that exists on disk -- so a deployment that did not set `CYBERGATE_OP_PATH` silently resolved
    # to L10 (tau = 0.10807704) instead of the L16 operator every measurement in the docs refers to.
    # One entry, because only the shipped operator is on disk now. The ladder used to continue
    # into `ops/` with L10 and L8; those superseded directions are no longer shipped, so a
    # fallback to them cannot happen -- which is what the comment above was guarding against.
    default_op_names = ("gemma4_L16_M1.json",)
    default_ref_dir_name = "gemma4_L21_refusal_dir.json"
    default_bundle_name = "gemma4_candidate_ops_bundle.json"

    # Kit binding. `hidden_size`/`num_layers` are the values the live container reported at hook
    # install ("dim=2816, total_layers=30"); `expected_gate` is read off the shipped operator's own
    # `meta.gate_layer`/`meta.pool`.
    kit_name = "gemma4"
    hidden_size = 2816
    num_layers = 30
    expected_gate = (16, "mean_user")
    precomputed_tag = "gemma"

    # 10. middle/late category probe consensus
    bundle_probe_layers = (15, 16, 18, 19, 20, 21, 22, 24, 26, 28)
    mid_late_probes = _MID_LATE_PROBES

    # 7. worker tokenizer discovery
    tokenizer_globs = (
        "/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*",
    )
    tokenizer_dirs = ("/work/gemma4_restore", "/gemma4_restore", "/home/gpuuser/gemma4_restore")

    # 8. chat template
    turn_open = "<|turn>"
    turn_close = "<turn|>"
    assistant_role = "model"
    supports_bos = True
    bos_text = "<bos>"
    think_system_prefix = "<|think|>\n"

    # 9. thought channel
    thought_open = "<|channel>thought\n"
    thought_close = "\n<channel|>"
    empty_thought_block = "<|channel>thought\n<channel|>"
    thought_close_tag = "<channel|>"
    stop_tokens = ("<turn|>", "<eos>")

    def _prefix_cached_span(
        self, seq_ids: Sequence[int]
    ) -> tuple[list[tuple[int, int]], str] | None:
        """Recover the user span when `<|turn>user` fell inside a cached prefix block.

        Scans backwards for the closing `<turn|>` (106) and infers the start from whichever suffix of
        the opening marker survived at position 0.
        """
        n = len(seq_ids)
        for j in range(n - 1, 0, -1):
            if seq_ids[j] == self.turn_end_id:
                if seq_ids[0] == 107:
                    u_s = 1
                elif n > 2 and seq_ids[0] == 2364 and seq_ids[1] == 107:
                    u_s = 2
                else:
                    u_s = 0
                if j > u_s:
                    return [(u_s, j)], "exact_token_markers(prefix_cached..106)"
                break
        return None

    # =============================================================================================
    # Operating point and middle/late consensus
    # =============================================================================================
    def operating_point_values(self, gate_layer: int) -> tuple[float, float, float, float]:
        """Gemma-4 gated at L15 or later runs a different operating point from Gemma-4 gated early.

        **This is the branch production takes.** The shipped kit gates at L16, so the four class
        attributes above (`revoke_ceil = 0.0520` and friends) apply only to the superseded early-layer
        operators in `ops/`. The values below were tuned for the late-layer configuration in
        `0ca6fb2`, alongside the 32k baseline.

        `revoke_ceil` and `in_context_pivot_floor` coincide with Qwen-3.6's, but `hysteresis_thresh`
        and `s_cat_last_pivot_floor` match neither family, so this is its own operating point rather
        than "behave like Qwen when gated late".
        """
        if gate_layer >= 15:
            return (-0.0250, 0.0350, -0.0300, 0.0300)
        return (
            self.revoke_ceil,
            self.hysteresis_thresh,
            self.in_context_pivot_floor,
            self.s_cat_last_pivot_floor,
        )

    def mid_late_veto(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> MidLateVeto | None:
        """Veto a lexical `mean_user` false positive that no last-token probe confirms.

        Single-turn only: on a later turn the prior-unlock scan reads the dampened score as
        `prior_score`, so vetoing there would suppress a turn that earlier evidence already unlocked.
        """
        if int(decision.get("num_user_turns", 1)) > 1:
            return None
        has_late_last = any(k in bundle_scores for k in _LATE_LAST_PROBES)
        late_last_r3_hits = [k for k in hits.last_r3 if k in _LATE_LAST_PROBES]
        has_last_probes = any(k in bundle_scores for k in _MID_LATE_LAST_PROBES)
        no_late_last = has_late_last and len(late_last_r3_hits) == 0
        no_last_at_all = has_last_probes and len(hits.last_r3) == 0
        if not (no_late_last or no_last_at_all):
            return None
        # Name the disjunct that actually vetoed. This reported `last_hits=0` unconditionally once,
        # which was wrong whenever the late-last disjunct fired: `L16_last` crossing R3 was reported
        # as no last-token hit at all, so the telemetry named a reason that was not the reason.
        reason = "no_late_last" if no_late_last else "no_last_hits"
        return MidLateVeto(
            floor=0.0760,
            source_suffix=(
                f"{reason},last_r3={len(hits.last_r3)},"
                f"late_last_r3={len(late_last_r3_hits)},r3={len(hits.r3)}"
            ),
            hits=tuple(hits.r3),
            reason=reason,
            log_detail=(
                f"r3_hits={list(hits.r3)} last_r3_hits={list(hits.last_r3)} "
                f"late_last_r3_hits={late_last_r3_hits}"
            ),
        )

    def mid_late_rescue(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> bool:
        """Rescue a paraphrased cyber turn the early gate missed, when late probes concur.

        Reads `s_cat_mean_user` and `s_cat_last` once, up front: the score lift below mutates
        `s_cat_mean_user`, and the admission gates must see the measured value, not the lifted one.
        """
        s_cat_mean = float(decision.get("s_cat_mean_user", raw_early))
        s_cat_last = float(decision.get("s_cat_last", 0.0))

        # At the shipped L16 gate, the last-token factor is capped by what the later layers see, so a
        # mid-stack spike alone cannot carry a rescue.
        eff_cat_last = s_cat_last
        if (
            int(decision.get("layer", 0)) == 16
            and "L20_last" in bundle_scores
            and "L21_last" in bundle_scores
        ):
            eff_cat_last = min(
                s_cat_last,
                max(float(bundle_scores["L20_last"]), float(bundle_scores["L21_last"])),
            )
        if s_z >= 7.0 and len(hits.r3) == 0 and eff_cat_last < s_cat_last:
            decision["s_cat_last"] = round(eff_cat_last, 6)
        if (
            s_z >= 2.5
            and eff_cat_last >= 0.0320
            and len(hits.last_r3) >= 1
            and raw_early < min(thresh_val - 0.002, 0.0760)
        ):
            lifted = round(min(thresh_val - 0.002, 0.0760), 6)
            decision["raw_early_score"] = raw_early
            decision["score"] = lifted
            decision["s_cat_mean_user"] = max(s_cat_mean, lifted)
        if s_z < 4.0 or s_cat_mean < 0.0580 or eff_cat_last < 0.0320:
            return False
        return len(hits.last_r3) >= 1 and (
            (len(hits.strict) >= 2 and len(hits.r2) >= 1)
            or len(hits.strict) >= 3
            or (len(hits.r3) >= 3 and s_cat_mean >= 0.0700)
        )
