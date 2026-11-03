"""Qwen3.8-27B. Fitted, then served and measured, 2026-11-02 by `notebooks/qwen38_colab.ipynb`.

This docstring read "NOT SERVED ... neither transfers to vLLM" after the refit landed, which
contradicted both the class docstring below it and `verified_live = True`. The first fit, made on
`transformers` activations, is the thing that did not transfer; see the class docstring.
"""
from __future__ import annotations

from ..qwen36.family import Qwen36Family


class Qwen38Family(Qwen36Family):
    """Qwen3.8-27B: 27B dense, 64 layers, hybrid full/linear attention. Gates at L40.

    Served and measured on 2026-11-02, A100 80GB, vLLM 0.32.0, bf16. 4,134 prompts replayed at
    the deployed gate, every row matched by prompt sha, 0 missing, kit check clean without
    `CYBERGATE_SKIP_KIT_CHECK`, all on the render `middleware._probe_observations` builds for
    itself (`suffix_len=9`; see `baseline_suffix_lens`), which is the only render this gate ever
    observes and is the same one the operator was fitted on:

    * **CySecBench 587/675 (86.96%)** on the test split, which chose nothing -- not the layer,
      not tau, not the direction. gemma4 delivers 558/675 (82.67%) by the same measurement, and
      482 of those are threshold crossings against all 587 here, since this family has no
      mid/late rescue.
    * **CatQA 23/234 (9.83%)**, identical to gemma4's. **SOSBench-bio 0/498**, as gemma4.
      **WMDP-bio 6/637** against gemma4's 4.
    * **harmless 12/1000 as recorded, 0/1000 veto-adjusted** -- the two-factor veto removes
      every benign fire.

    The history is the useful part. The first operator was fitted at L39 on `transformers`
    activations and scored at chance under vLLM, because the directions do not transfer between
    the engines on this architecture: served, it scored SOSBench-bio prompts as highly as its own
    cyber positives. Refitting on vLLM's own pooled activations fixed it. Four of the canaries
    below are prompts from that diagnosis: with the L39 operator they scored +0.036978, +0.002108,
    +0.022492 and -0.001360 against tau +0.0630 and none fired; with this one they score +0.410478,
    +0.394156, +0.393348 and +0.419564 against tau +0.0879 and all four clear it by ~4.5x. Same
    prompts, same checkpoint, same render.

    L39 was chosen on an in-sample AUC of 0.9985. Refitted properly it reaches 0.9526 held-out
    recall and leaks **0.2104** into WMDP-bio -- one of the three worst layers in this stack for
    selectivity. The in-sample AUC is 0.99+ at all 64 layers, including layers at 0.21 held-out
    recall, so it could not have shown that. L40 was chosen on held-out recall with leak measured.

    **Known gaps are real and in `kits/qwen38/MANIFEST.json`**: the refusal sigma is still the
    transformers value and the veto costs 90 of 587 fires (veto-adjusted recall 0.7363 against
    gemma4's 0.8193); PKU-cybercrime leaks 220/612 and that is not explained; the decode
    calibration and the policy thresholds are Qwen3.6's; multi-turn is unmeasured; and a client
    passing `chat_template_kwargs={"enable_thinking": false}` renders a prefill this predicate
    cannot distinguish from the observation (see `baseline_suffix_lens`).

    **The two-factor veto is a benign-false-positive filter and nothing more.** Measured at the
    1.50 floor: `harmless` 12 -> 0, every benign fire removed. But CatQA 23 -> 23, PKU 220 -> 220,
    `incorpus` 32 -> 32, WMDP-bio 6 -> 6 -- *zero* removed from any harm-adjacent leak corpus.
    12 of 293 leak fires in total (4.1%) against 90 of 587 true positives (15.3%). The veto
    requires `s_ref_z >= 1.50`, i.e. that the model was about to refuse; a benign prompt does not
    elicit a refusal and a burglary prompt does. So recalibrating the refusal sigma cannot move
    any leak rate, and the category direction's selectivity is the only thing that can.

    **The gate's render is not the render the model answers in.** The probe prepends
    `build_raw_prompt`'s own 6-token system prompt ("You are a helpful assistant."); this
    checkpoint's chat template injects a 42-token reasoning-effort directive instead, putting the
    client's `u1_start` at 45 against the probe's 14. Fit and serving both go through the probe,
    so the numbers above are internally consistent -- but they are measured on a prefix the served
    request does not have.
    """

    name = "qwen38"
    # Both matchers must be MORE SPECIFIC than qwen36's bare ('qwen',) and this family must be
    # registered BEFORE qwen36, because both lookups are first-match-wins. Appending it after
    # qwen36 leaves it unmatchable: 'Qwen/Qwen3.8-27B' contains 'qwen' and resolves to qwen36.
    model_id_substrings = ("qwen3.8", "qwen3_8")
    model_key_prefixes = ("qwen38",)
    verified_live = True

    # Measured from the tokenizer, not inherited. Qwen3.6 is (248045, 846, 198)/248046.
    user_turn_marker = (248045, 846, 198)
    turn_end_id = 248046
    fresh_prompt_marker = (248045, 8678)
    span_label = "exact_token_markers(qwen38)"

    hidden_size = 5120
    num_layers = 64
    kit_name = "qwen38"

    # L55, fitted on vLLM's OWN activations and chosen on held-out recall with leak measured.
    # L39 was the original choice, made on an in-sample AUC of 0.9985; refitted properly it
    # reaches 0.9526 recall but leaks 0.2104 into WMDP-bio, which makes it one of the three
    # worst layers in this stack for selectivity. The in-sample AUC was 0.99+ at all 64 layers
    # and could not have shown that. `qwen38_L39_M1_serving.json` stays in the kit as the
    # record of it, and as the positive control the dump diagnostics compare against.
    # L40, fitted on vLLM's OWN activations and chosen on held-out recall with leak measured.
    #
    # **The gate layer must sit BELOW `ref_layer`.** `refusal_monitor` attaches `s_ref_z` to the
    # record the category probe already wrote (`decision = pending.get((q_start, q_end))`, and
    # `is_true_prefill` requires it to exist). With the gate above `ref_layer` the monitor fires
    # first in the forward pass, finds nothing pending, and takes the decode path: no `s_ref_z`
    # on the record, no `multi_layer_scores`, no mid/late rescue, and the two-factor veto gets
    # `None` -- silently. gemma4 is 16/21 and qwen36 is 21/28, so every shipped kit satisfies
    # this, but nothing enforced it. L55 scores better on the sweep (0.9230 recall at 0.0769
    # CatQA) and is unusable for exactly this reason; see kits/qwen38/MANIFEST.json.
    #
    # L39 was the original choice, made on an in-sample AUC of 0.9985. Refitted properly it
    # reaches 0.9526 recall but leaks 0.2104 into WMDP-bio -- one of the three worst layers in
    # this stack for selectivity, which an in-sample AUC of 0.99+ at every layer could not show.
    expected_gate = (40, "mean_user")
    default_op_names = ("qwen38_L40_M1_vllm.json",)
    default_ref_dir_name = "qwen38_L43_refusal_dir_serving.json"
    default_bundle_name = None

    # Declared, not inherited. `test_every_family_declares_every_serving_attribute_itself`
    # requires all 30 on the class: every base default is a legal value, so an inherited one
    # is read, used and reported as success. Values below are Qwen3.6's ChatML scaffolding,
    # which cell 4 verified is this model's template too -- EXCEPT where noted.
    turn_open = "<|im_start|>"
    turn_close = "<|im_end|>"
    assistant_role = "assistant"
    supports_bos = False
    bos_text = ""
    think_system_prefix = ""
    thought_open = "<think>\n"
    thought_close = "\n</think>\n\n"
    empty_thought_block = "<think>\n\n</think>\n\n"
    thought_close_tag = "</think>"
    stop_tokens = ("<|im_end|>",)

    # UNVERIFIED on this checkpoint: Qwen3.6's fixed-offset fallback. Only consulted when the
    # exact markers above are absent, which should not happen -- if `span_method` in a decision
    # record ever reads default_offsets(14:-5), the markers did not match and the score is
    # pooled over the wrong tokens. Re-measure rather than trust these.
    default_pre = 14
    default_suf = 5
    bos_token_id = None          # a Qwen prefill has no BOS

    # 2b. MEASURED 2026-11-03 against this checkpoint's own tokenizer, via
    # `notebooks/render_preflight.ipynb`. The observation prefill is the one
    # `middleware._probe_observations` renders ITSELF -- `build_raw_prompt(mode="baseline")`, which
    # emits `empty_thought_block` -- and dispatches to `/v1/completions`. No client kwarg and no
    # server flag reaches it:
    #
    #   probe   build_raw_prompt(mode="baseline")           u1_start=14  suffix_len=9   PUBLISHES
    #   client  apply_chat_template()                       u1_start=45  suffix_len=7   dropped
    #   client  apply_chat_template(enable_thinking=True)    u1_start=45  suffix_len=7   dropped
    #   client  apply_chat_template(enable_thinking=False)   u1_start=45  suffix_len=9   PUBLISHES
    #
    # `(14, 9)` is the pair a live decision record carried on 2026-11-02, reproduced exactly.
    #
    # This arm used to compute `(self.default_suf, 9)` = `(5, 9)`, which is behaviourally the same
    # as `(9,)` here: **5 matches no render of this template** -- it is Qwen3.6's `default_suf`,
    # inherited. Widening it to accept the client's 7 was tried and is WRONG: every ordinary chat
    # request would then publish a second decision over the generation prefill, overwriting the
    # probe's record and appending a second unlabelled row to any running dump.
    #
    # **Known hazard, not fixed here.** The last row above publishes and should not. A client that
    # passes `chat_template_kwargs={"enable_thinking": false}` renders a suffix byte-identical to
    # the probe's, so no predicate over `(u1_start, suffix_len)` can separate them -- `u1_start`
    # shifts with whatever system message the client sends, since `extract_system_and_user` passes
    # it through to the probe too. Separating them needs the probe to identify itself rather than
    # be inferred from its shape. Unreached by the 4,134-prompt validation, which never passed
    # `chat_template_kwargs`; see kits/qwen38/MANIFEST.json known_gaps.
    baseline_suffix_lens = (9,)

    # UNVERIFIED on this checkpoint: Qwen3.6's policy thresholds, calibrated at its L15/L28.
    # Recalibrate before trusting the mid/late veto or the rescue on this model.
    revoke_ceil = -0.0250
    hysteresis_thresh = -0.0250
    in_context_pivot_floor = -0.0300
    s_cat_last_pivot_floor = 0.0100

    # Empty on purpose: no candidate-layer bundle has been measured for this model, and empty
    # disables the middle/late consensus entirely. That is the safe default -- inheriting
    # Qwen3.6's layer list would attach probes at layers chosen for a different architecture.
    bundle_probe_layers = ()
    mid_late_probes = ()
    precomputed_tag = "qwen38"

    # Fitted on Colab, where vllm runs in-process and the tokenizer is in the HF cache.
    tokenizer_globs = ('/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*',)
    tokenizer_dirs = ()

    # Refusal calibration, fitted in the SERVING render (`--render serving`).
    # sep 1.1239, mu_hm_cos +0.000866, z_harm_mean 13.34, 681/681 harmful and
    # 76/1000 harmless at the 1.50 floor -- a sound fit by every diagnostic.
    ref_layer = 43
    ref_mu_hm = 0.0008657520481332933
    ref_sd_hm = 0.04016976950385832
