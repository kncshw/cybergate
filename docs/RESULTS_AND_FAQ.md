# Benchmarks, Model Compatibility & Quantization FAQ

**[English](RESULTS_AND_FAQ.md) | [简体中文](RESULTS_AND_FAQ_zh.md)** · **[← Back to README](../README.md)**

---

## Measured Results (Live vLLM Harvests)

All numbers below were measured on live vLLM harvests against the served checkpoints (`CySecBench` was held out and did **not** choose layers, directions, or thresholds; full audit details and caveats live in `kits/*/MANIFEST.json`):

| Corpus | `n` | `Qwen/Qwen3.8-27B` | `google/gemma-4-26B-A4B-it` | Role |
|---|---:|---:|---:|---|
| **CySecBench** | 675 | **587 (87.0%)** | **558 (82.7%)** *(553 veto-adj.)* | Target - **Unlock** |
| **WMDP-cyber** | 637 | - | **606 (95.1%)** | Target - **Unlock** |
| **SOSBench-bio** | 498 | **0 (0.0%)** | **0 (0.0%)** | Non-target harm - **Hold** |
| **WMDP-bio** | 637 | **6 (0.9%)** | **4 (0.6%)** | Non-target harm - **Hold** |
| **Harmless** | 1,000 | **0 (0.0%)** *(12 pre-veto)* | - | Benign - **Hold** |
| **CatQA** | 234 | **23 (9.8%)** | **23 (9.8%)** | Non-target harm - **Hold** |
| **PKU-cybercrime** | 612 | **220 (35.9%)** | - | Topical overlap *(known gap)* |

### Key Engineering Findings & Honest Limitations
- **`transformers` directions did not transfer to vLLM on `Qwen3.8-27B`:** On Qwen3.8's hybrid full/linear-attention stack, an operator fitted in Hugging Face `transformers` gated nothing when served in vLLM (scoring bio prompts as high as cyber positives). Refitting directly on vLLM's own pooled activations moved four diagnostic cyber prompts from `+0.037, +0.002, +0.022, -0.001` (vs `tau +0.063`) to `+0.411, +0.394, +0.393, +0.420` (vs `tau +0.088`).
- **In-sample AUC across layers was a trap:** On Qwen3.8, all 64 layers scored `0.9894-0.9993` in-sample AUC. In-sample AUC selected `L39` (`0.9985`), which leaked **21.0%** into `WMDP-bio`; sweeping held-out selectivity picked `L40` (`0.9%` `WMDP-bio` leak). See `kits/qwen38/records/harvest_2026-11-02_layer_sweep.txt`.
- **Middle/late rescue on Gemma-4:** On `google/gemma-4-26B-A4B-it`, `82` of the `558` `CySecBench` fires come from the middle/late layer rescue mechanism (`Qwen3.8` uses pure single-threshold crossing with no middle/late rescue).
- **Topical overlap & no caller-authorization signal:** The refusal veto eliminates benign false positives (`12 -> 0` on `Harmless`), but on `PKU-cybercrime` (`35.9%`) both cyber topicality and refusal pressure hold simultaneously. Furthermore, the gate reads topicality + refusal pressure only - it has **no cryptographic or policy notion of caller authorization**.

---

## Model & Quantization Compatibility (Adding a Model)

### Which models can be added?
Out of the box, CyberGate ships validated kits in `kits/qwen38` and `kits/gemma4`. You can add other models supported by vLLM `0.32` subject to **two requirements**:

1. **Reasoning / `<think>` support:** Because Stage 2 steers compliance inside the model's Chain-of-Thought rather than mutating weights, models with an explicit thought block (`<think>...</think>`, `<|thought|>`, etc. - such as Qwen3/3.8, Gemma-4, DeepSeek-R1 distills, QwQ) are first-class targets. Pure non-reasoning instruct models have no thought channel for closed-loop CoT steering.
2. **Per-checkpoint direction fitting (no cross-model vector reuse):** Hidden-state geometry is specific to each model checkpoint. Adding a model requires three artifacts in `kits/<name>/` (no core plugin changes needed):
   - **Two fitted vectors** (category + calibrated refusal direction) captured via `scripts/fit/` (`harvest_vllm_acts.py`, `fit_category.py`, `fit_refusal.py`) or `notebooks/qwen38_colab.ipynb`.
   - **A `ModelFamily` subclass** (`src/cybergate/families/`) declaring that checkpoint's exact turn-marker token IDs, thought tags, and thresholds (verify offline first with `notebooks/render_preflight.ipynb`).
   - **Checkpoint-pinned `canaries.json`** verified by `scripts/serving/check_eval.sh` - hooks attach silently even if an operator was fitted on the wrong checkpoint, so live canary replay is mandatory.

### Does quantization affect CyberGate?
- **8-bit (`FP8`) & `BF16`:** Verified in production (`google/gemma-4-26B-A4B-it` is served and benchmarked under `--quantization fp8_per_channel`; `Qwen/Qwen3.8-27B` under `BF16`).
- **4-bit quants (`AWQ`, `GPTQ`, `bitsandbytes`):** Any quantization format executed through vLLM's PyTorch model layers (`--enforce-eager`) exposes hidden states to CyberGate's forward hooks (external non-vLLM runtimes such as `llama.cpp` GGUF are not supported). However, aggressive 4-bit quantization can slightly shift hidden-state cosine distributions (`tau` and `s_ref_z`). Always run `./scripts/serving/check_eval.sh` against your quantized deployment - if scores drift, refitting directions directly on vLLM's activations under that exact `--quantization` flag recalibrates the thresholds cleanly.

---

## Documentation & Notebooks Index

- **[docs/EVALUATOR_QUICKSTART.md](EVALUATOR_QUICKSTART.md)** - Full serving flags (`--enforce-eager`, `--attention-backend TRITON_ATTN`, `--enable-prefix-caching`), port architecture, and canary verification.
- **[docs/PROCESS_TOPOLOGY.md](PROCESS_TOPOLOGY.md)** - Two-process vLLM V1 architecture (API server vs GPU worker, `/dev/shm` IPC, zero shared memory).
- **[docs/ARCHITECTURE.md](ARCHITECTURE.md)** - Decision rules, differential equivalence suites against the vendored reference (`tests/equivalence/`), and tracked regression behavior (`R6-R14`).
- **[docs/DECODE_CALIBRATION.md](DECODE_CALIBRATION.md)** - Closed-loop CoT spike calibration and decode-time logits steering.
- **`notebooks/`** - Colab workflows: `render_preflight.ipynb` (CPU-only span check), `qwen38_live_chat.ipynb`, `serve_kit_colab.ipynb`, and `qwen38_colab.ipynb` (end-to-end kit fitting).
