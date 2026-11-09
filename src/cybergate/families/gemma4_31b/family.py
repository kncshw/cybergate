"""Gemma-4 31B dense (`google/gemma-4-31B-it`, 60 layers, hidden_size=5376).

Subclasses `Gemma4Family` (`google/gemma-4-26B-A4B-it`) because the two checkpoints share the exact
same tokenizer (`tokenizer.json` sha256 `cc8d3a0ce364...`) and `chat_template.jinja`, while
differing in decoder architecture (60-layer dense `hidden_size=5376` vs 30-layer MoE
`hidden_size=2816`) and therefore requiring checkpoint-specific category and refusal operators.
"""

from __future__ import annotations

from ..gemma4.family import Gemma4Family


class Gemma431BFamily(Gemma4Family):
    """Gemma-4 31B dense (`google/gemma-4-31B-it`): 60 layers, hidden_size=5376."""

    name = "gemma4_31b"
    # More specific than `Gemma4Family.model_id_substrings = ("gemma-4", "gemma4")`; must be
    # registered BEFORE `GEMMA4` in `ALL_FAMILIES` because family lookup is first-match-wins.
    model_id_substrings = ("gemma-4-31b", "gemma4-31b", "gemma4_31b")
    model_key_prefixes = ("gemma4_31b", "gemma4-31b")
    verified_live = True

    # --- Token-level span extraction (identical tokenizer to Gemma-4 26B-A4B-it) -----------------
    user_turn_marker = (105, 2364, 107)
    turn_end_id = 106
    span_label = "exact_token_markers(gemma4_31b)"
    default_pre = 15
    default_suf = 9
    baseline_suffix_lens = (9,)
    bos_token_id = 2
    fresh_prompt_marker = (2,)

    # --- Checkpoint & operator geometry ----------------------------------------------------------
    hidden_size = 5376
    num_layers = 60
    kit_name = "gemma4_31b"
    expected_gate = (30, "mean_user")
    default_op_names = ("gemma4_31b_L30_M1_vllm.json",)

    # --- Refusal monitor & two-factor threshold --------------------------------------------------
    ref_layer = 34
    ref_mu_hm = -0.00021306765443346337
    ref_sd_hm = 0.016573137950799642
    ref_mu_decode = None
    ref_sd_decode = None
    revoke_ceil = -0.35
    hysteresis_thresh = 0.02999161
    category_threshold = 0.0320
    default_ref_dir_name = "gemma4_31b_L34_refusal_dir_vllm.json"

    # --- Multi-layer bundle telemetry ------------------------------------------------------------
    default_bundle_name = None
    precomputed_tag = "gemma4_31b"
    bundle_probe_layers: tuple[int, ...] = ()
    mid_late_probes: tuple[tuple[int, str], ...] = ()

    # --- Tokenizer discovery in the vLLM worker process ------------------------------------------
    tokenizer_globs = (
        "/root/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/*",
    )
    tokenizer_dirs: tuple[str, ...] = ()

    # --- Chat template & reasoning channel scaffolding -------------------------------------------
    turn_open = "<|turn>"
    turn_close = "<turn|>"
    assistant_role = "model"
    supports_bos = True
    bos_text = "<bos>"
    think_system_prefix = "<|think|>\n"
    thought_open = "<|channel>thought\n"
    thought_close = "\n<channel|>"
    empty_thought_block = "<|channel>thought\n<channel|>"
    thought_close_tag = "<channel|>"
    stop_tokens = ("<turn|>", "<eos>")

    # --- Prompt bodies ---------------------------------------------------------------------------
    open_think_prefix = Gemma4Family.open_think_prefix
    fake_think_rubric = Gemma4Family.fake_think_rubric
    post_think_opener = Gemma4Family.post_think_opener
    inline_cot_suppression = Gemma4Family.inline_cot_suppression
    inline_final_step_clearance = Gemma4Family.inline_final_step_clearance
    bottom_compliance_lock = Gemma4Family.bottom_compliance_lock

    def is_baseline_prefill(
        self, u1_start: int, suffix_len: int, span_method: str, max_suffix_len: int
    ) -> bool:
        """Accept only the Pass-1 baseline probe prefill (`u1_start=15, suffix_len=9`).

        Rejects client `/v1/chat/completions` renders that omit a system message (`u1_start=4,
        suffix_len=9`) or use `enable_thinking=True` (`u1_start=11, suffix_len=5`), preventing
        Pass-2 baseline answer prefills from overwriting the Pass-1 probe observation or double-
        writing rows during `CYBERGATE_DUMP_ACTS_DIR` harvests.
        """
        if "prefix_cached" in span_method:
            return suffix_len in self.baseline_suffix_lens
        if span_method == "full_sequence_fallback":
            return True
        return u1_start > 11 and suffix_len in self.baseline_suffix_lens


GEMMA4_31B = Gemma431BFamily()
