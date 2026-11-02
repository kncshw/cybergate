"""Qwen3.8-27B. Fitted 2026-11-02 by `notebooks/qwen38_colab.ipynb`. NOT SERVED.

See the class docstring: both factors fit well and neither transfers to vLLM.
"""
from __future__ import annotations

from ..qwen36.family import Qwen36Family


class Qwen38Family(Qwen36Family):
    """Qwen3.8-27B: 27B dense, 64 layers, hybrid full/linear attention. **Does not gate.**

    `verified_live = False` and it is not decoration. Both factors were fitted in the serving
    render and both are sound by every offline diagnostic -- category AUC 0.9974 in-sample with
    100/100 fit positives clearing tau, refusal `sep` 1.1239 with `mu_hm_cos` +0.000866 and
    681/681 harmful rows above the 1.50 floor. Served under vLLM on 2026-11-02 the same prompts
    scored **+0.02 to +0.05 against +0.20 in the capture**, and nothing fired.

    The cause is not the render, the span or the operator. All were held identical and verified:
    the same 35 token ids, `span_method=exact_token_markers(qwen38)` at (14, 26), the same
    operator and tau. A 20-layer sweep (L30-L49) found no layer whose captured scores match the
    live ones, ruling out an indexing error. What remains is the forward pass: the kit was fitted
    on activations captured through `transformers` -- running reference PyTorch fallbacks for
    `causal_conv1d` and `chunk_gated_delta_rule` -- and served through vLLM's own Gated DeltaNet
    kernels. The two do not agree on this architecture.

    That is an unstated assumption of `scripts/fit/`, not a defect in this declaration: it holds
    for gemma4 and qwen36, both standard attention, both serving correctly. Gating a hybrid
    linear-attention model needs activations captured from vLLM itself.

    Also unverified, separately: the policy thresholds (`revoke_ceil`, `hysteresis_thresh`,
    `in_context_pivot_floor`, `s_cat_last_pivot_floor`) are Qwen3.6's, calibrated at its L15/L28
    and never recalibrated here, so the mid/late veto and rescue paths are untested on this model.
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
