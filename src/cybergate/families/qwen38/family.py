"""Qwen3.8-27B. Fitted 2026-11-02 by `notebooks/qwen38_colab.ipynb`. NOT SERVED.

See the class docstring: both factors fit well and neither transfers to vLLM.
"""
from __future__ import annotations

from ..qwen36.family import Qwen36Family


class Qwen38Family(Qwen36Family):
    """Qwen3.8-27B: 27B dense, 64 layers, hybrid full/linear attention. **Does not gate.**

    `verified_live = False`, and the reason is now measured rather than hypothesised. Measured
    live on 2026-11-02/03, A100 80GB, vLLM 0.32.0, bf16, serving render:

    * the **category** direction scores at **chance** -- AUC 0.5893 against 14 of its own 140 fit
      references and 0.5500 against 15 harmless prompts, using 12 of its own 100 fit positives,
      having claimed 0.9974 in-sample. Both sides of that comparison are the serving render and
      the direction is the shipped one, so it is not a render artifact. No layer rescued it.
    * the **refusal** direction at L43 **transfers cleanly** -- 31/31 harmful prompts above the
      1.50 floor (mean z +4.15), 0/15 harmless (mean -0.92, max -0.187). Compressed against the
      fit, cos 0.168 against the 0.537 the fit implies, but the floor sits inside a 2.8-z gap.

    The cause is the **capture geometry**, and the discriminating evidence is that refitting at
    the *same* L39 on vLLM's own pooled activations -- same 240 rows, same ridge, same R3
    threshold -- takes held-out recall from chance to **0.9526** on 675 CySecBench prompts the fit
    never saw. Depth does not explain it (L43 is deeper than L39 and works), and overfitting does
    not explain it (the same 240 rows generalise once fitted in the right geometry). What the
    in-sample 0.9974 was, is nothing: the in-sample AUC is 0.99+ at **all 64 layers**, including
    layers whose held-out recall is 0.21.

    The fix is `scripts/fit/harvest_vllm_acts.py` then `scripts/fit/sweep_layers_from_dump.py`.
    **It is not finished**: leak has not been measured on any out-of-sample non-target set, so
    0.9615 at L51 is a recall number without a false-positive number beside it and is not yet a
    gating result. `kits/qwen38/MANIFEST.json` carries the full measurements and what is missing.

    Also unverified, separately: the policy thresholds (`revoke_ceil`, `hysteresis_thresh`,
    `in_context_pivot_floor`, `s_cat_last_pivot_floor`) are Qwen3.6's, calibrated at its L15/L28
    and never recalibrated here, so the mid/late veto and rescue paths are untested on this model.
    The refusal factor's `ref_mu_hm`/`ref_sd_hm` are the transformers-capture values; they work
    live but have not been re-measured on vLLM activations.
    """

    name = "qwen38"
    # Both matchers must be MORE SPECIFIC than qwen36's bare ('qwen',) and this family must be
    # registered BEFORE qwen36, because both lookups are first-match-wins. Appending it after
    # qwen36 leaves it unmatchable: 'Qwen/Qwen3.8-27B' contains 'qwen' and resolves to qwen36.
    model_id_substrings = ("qwen3.8", "qwen3_8")
    model_key_prefixes = ("qwen38",)
    verified_live = False

    # Measured from the tokenizer, not inherited. Qwen3.6 is (248045, 846, 198)/248046.
    user_turn_marker = (248045, 846, 198)
    turn_end_id = 248046
    fresh_prompt_marker = (248045, 8678)
    span_label = "exact_token_markers(qwen38)"

    hidden_size = 5120
    num_layers = 64
    kit_name = "qwen38"

    expected_gate = (39, "mean_user")
    default_op_names = ("qwen38_L39_M1_serving.json",)
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
