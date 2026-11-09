# CyberGate kit: qwen38

Everything needed to serve and verify the CyberGate category gate on **Qwen/Qwen3.8-27B** (64 layers,
`hidden_size=5120`, hybrid full/linear attention). `MANIFEST.json` is the machine-readable version of
this page.

## What is in here

| file | role |
|---|---|
| `qwen38_L40_M1_vllm.json` | the shipped category operator — layer 40, `mean_user` pooling, τ = +0.08792114 |
| `qwen38_L43_refusal_dir_vllm.json` | refusal direction for the second gate factor, layer 43 (fitted on vLLM activations) |
| `canaries.json` | 9 prompts (5 target, 4 non-target) with verdicts measured on the served gate |
| `MANIFEST.json` | model ids, checkpoint shape, measured results, known gaps |
| `records/` | layer sweep (`harvest_2026-11-02_layer_sweep.txt`), corpus digest, and 4,134-prompt live validation record |
| `qwen38_L39_M1_serving.json` | superseded `transformers`-captured L39 operator (retained as diagnostic reference) |
| `qwen38_L43_refusal_dir_serving.json` | superseded `transformers`-captured L43 refusal direction |
| `qwen38_L55_M1_vllm.json` | rejected L55 operator (`gate_layer > ref_layer`; retained as evidence) |

## Serve it

```bash
docker compose -f docker-compose.Qwen3.8-27B.yml up -d
# or via the evaluation script:
scripts/serving/serve_eval.sh --kit qwen38
scripts/serving/check_eval.sh --kit qwen38
```

## What was measured

Live, on the served checkpoint (`vLLM 0.32.0`, `bfloat16`, A100 80 GB), 4,134 prompts matched by exact prompt sha (`records/live_validation_2026-11-02.json`):

| | CySecBench test 675 | CatQA 234 | WMDP-bio 637 | SOSBench-bio 498 | Harmless 1000 | In-corpus non-target 378 | PKU-cybercrime 612 |
|---|---|---|---|---|---|---|---|
| as recorded | **587 (87.0%)** | **23 (9.8%)** | **6 (0.9%)** | **0 (0.0%)** | 12 (1.2%) | 32 (8.5%) | 220 (35.9%) |
| veto-adjusted | 497 (73.6%) | 23 (9.8%) | 6 (0.9%) | 0 (0.0%) | **0 (0.0%)** | 32 (8.5%) | 220 (35.9%) |

Note: `qwen38_L43_refusal_dir_vllm.json` refits the L43 refusal direction directly on vLLM's own L43 last-token activations (`sep = 1.1487`, `z_harm_mean = +18.18σ`, `681/681` harmful above `1.50σ`); the `veto-adjusted` row in `records/live_validation_2026-11-02.json` was recorded prior to that refit using `qwen38_L43_refusal_dir_serving.json`.
