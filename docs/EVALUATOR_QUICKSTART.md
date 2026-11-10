# Evaluator quickstart

Two independent things can be checked, and they cost very different amounts.

| | needs | time | what it proves |
|---|---|---|---|
| **A. Offline** | a laptop | ~2 min | the decision logic, end to end |
| **B. Serving** | one 26B-class GPU | ~15 min | the probe runs on real activations and the gate steers |

Reproducing the published corpus numbers is a third thing and is **not** currently possible from
this repository alone: the fitting and harvest scripts are in `scripts/fit/`, while the external
evaluation benchmark corpora are not redistributed here. See §4.

---

## A. Offline — no GPU, no network, no model

```bash
git clone <repo> && cd cybergate
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest                              # 991 passed, 1 skipped, 7 xfailed
.venv/bin/python -m cybergate_plugin.cli --self-check
```

Requires Python >= 3.10; verified on 3.12.3. The package declares **no runtime dependencies** --
torch, transformers and vllm are supplied by the serving container and imported lazily, so the
suite runs on a machine with none of them.

Do **not** add `--no-build-isolation`: a fresh `python3 -m venv` on Python >= 3.12 contains no
setuptools, and that flag then fails with `ModuleNotFoundError: No module named 'setuptools'`.

What the suite covers, and what it does not, is in `README.md` under *How correctness is
established* and *What is not verified*. The short version: the policy is a pure function and is tested
exhaustively against a vendored copy of the canonical implementation; the two forward hooks need a
loaded model and are only import-checked. Part B is what exercises them.

## B. Serving on a GPU

**Hardware.** A 26B MoE at fp8 with 32k context and `--gpu-memory-utilization 0.88`. Developed and
measured on a GB10 with 119 GB unified memory; peak host memory in use while serving is ~115 GB.
Docker with GPU support and the weights already in a local Hugging Face cache.

Artifacts come from a per-model **kit** under `kits/<name>/`, which is self-contained: operator,
refusal direction, candidate bundle, canaries and a `MANIFEST.json`. The kit directory is mounted as
the operator directory, so no other model's artifacts are reachable from the container. Each kit has
its own README with that model's numbers and caveats.

```bash
# Primary recommended way: Docker Compose (single container binding :8000 Pristine and :8001 CyberGate)
# 1. For google/gemma-4-26B-A4B-it:
docker compose -f docker-compose.gemma-4-26B-A4B-it.yml up -d

# 2. Or for Qwen/Qwen3.8-27B:
docker compose -f docker-compose.Qwen3.8-27B.yml up -d
```

Alternatively, using the standalone shell script (`scripts/serving/serve_eval.sh`):

```bash
# gemma4 kit
./scripts/serving/serve_eval.sh --kit gemma4 --hf ~/.cache/huggingface --gpus device=0

# qwen38 kit -- model, operator, and bfloat16 serve flags are selected automatically
./scripts/serving/serve_eval.sh --kit qwen38

# an alternate operator from inside the gemma4 kit
./scripts/serving/serve_eval.sh --kit gemma4 --op gemma4_ruleC_serve.json
```

### Which Port to Connect To (`:8000` Pristine vs `:8001` CyberGate)

Whether launched via `docker compose -f docker-compose.<model>.yml up -d` or `serve_eval.sh`, the single vLLM container loads the model once and binds **two OpenAI-compatible ports** inside the same process:

| Port | OpenAI API Base URL | Default Mode | What to Connect Here |
|---|---|---|---|
| **Port `8000`** *(Pristine by default)* | `http://localhost:8000/v1` | `CYBERGATE_MODE=baseline` | **Pristine (Un-Gated by Default) Model.** Bound directly by `vllm serve --port 8000`. Standard requests without a `"cybergate_mode"` override in the JSON body return the unmodified base model's output (`routed="pristine_pass_through"`) while recording prefill probe telemetry (passing `"cybergate_mode": "closed_loop"` in the request body still activates the gate). |
| **Port `8001`** *(CyberGate)* | `http://localhost:8001/v1` | `cybergate_mode=closed_loop` | **CyberGate (Gated) Agent/Chat Bridge.** Bound automatically on a background thread by `CyberGateThinkMiddleware` via `CYBERGATE_PROXY_PORT=8001` (defaults to `127.0.0.1`; set to `0.0.0.0` in `docker-compose.*.yml`). Routes requests through the two-factor activation gate, inline CoT refusal steering, OpenAI `tools`/`<tool_call>` normalization, multi-turn context compaction, default `CYBERGATE_DEFAULT_MAX_TOKENS=4096` cap, and `[CyberGate \| ...]` telemetry header. |

`serve_eval.sh` prints the kit's gate, its threshold rule and its known gaps before starting, so an
in-sample threshold is visible at serve time rather than buried in a table.

`--help` lists every flag. First load takes several minutes; follow
`docker logs -f cybergate_gemma4_26b_a4b_it` (or `logs/<name>_serve.log`) and wait for `Application startup complete`.

Then verify the gate is actually live:

```bash
# If launched via docker compose -f docker-compose.gemma-4-26B-A4B-it.yml up -d:
./scripts/serving/check_eval.sh --kit gemma4 --port 8000 --ipc /dev/shm/cybergate_ipc_gemma4

# Or if launched via serve_eval.sh:
./scripts/serving/check_eval.sh --kit gemma4 --port 8000 --ipc /dev/shm/cybergate_ipc_cybergate_eval
```

Five checks. [1] the probe runs and scores in the serving render, [2] a benign control does not
fire, [3] the steered opener appears in a real generation, [4] a pass-through reply contains no
control tokens, and [5] the kit's canaries reproduce.

**[5] is the one that distinguishes a working gate from a merely attached one.** The dimension and
layer checks at load catch a cross-architecture mismatch, but an operator fitted on a different
checkpoint of the *same* architecture passes all of them, attaches cleanly, and then fires on almost
nothing. That is what the previously shipped Qwen operator did -- it cleared its own threshold on 18
of its own 100 fitted training positives. The canaries carry verdicts measured on the checkpoint the
artifacts were fitted on, so only they can catch it. On `gemma4`, expect 5 positives firing and 4
negatives quiet, 9/9.

Canaries are replayed with `cybergate_mode=baseline` and no system prompt, which is the render they
were measured in. The render is not incidental: check [1]'s prompt scores **+0.186274** over a
26-token prefill in that render and **+0.167149** over a 37-token prefill in the default serving
render (measured live 2026-11-01) -- the same prompt, a different score. That is why [1] reports the
score but does not assert the verdict, and [5] does. Both of those happen to clear tau. An earlier
edition of this file recorded **+0.047361** over a 30-token prefill for the serving render, which is
*below* tau; that figure is **not reproduced** and the render it was taken in is not identified. The
30-versus-37-token prefill suggests the serving render changed under it, but that is a hypothesis,
not a measurement -- so treat it as a prior record, and either way do not compare a score against a
threshold from another render.

**One model per GPU.** `serve_eval.sh` warns if another vllm container is running. Two 26B models on one
device does not degrade gracefully -- it OOMs, or starves the host until ssh stops completing its
banner exchange.

### The serve flags are load-bearing

`serve_eval.sh` sets these and they are not preferences:

| flag | why |
|---|---|
| `--quantization fp8_per_channel` | 26B at bf16 (~52 GB weights) does not fit on a 40–48 GB GPU; it loads, then OOMs (`gemma4`; `qwen38` uses `--dtype bfloat16` on 80 GB+) |
| `--enable-prefix-caching` | reuses KV cache across multi-turn history and Pass-1 $\to$ Pass-2 generation (`middleware._probe_observations` salts Pass-1 observation probes via `cache_salt` so the prefill hook always observes the full prompt, while `sync_bindings` keeps CoT decode steering bound to the full prompt key) |
| `--enforce-eager` | the hooks read hidden states; compiled graphs elide them |
| `--attention-backend TRITON_ATTN` | the backend `gemma4` was measured on (`qwen38` omits this flag for hybrid Gated DeltaNet layers) |

The operator and the refusal direction must match the served weights and inference engine. A direction
fitted on one checkpoint or engine can score near zero on another and the gate simply stops firing,
with no error. That happened on Qwen3.8 when transferring a `transformers`-fitted probe to vLLM on a
hybrid linear-attention stack: see `kits/qwen38/MANIFEST.json` -> `fit.capture_note` and `fit.refusal_quality`.

## 3. What is being served

| | gemma4 | qwen38 |
|---|---|---|
| model | `google/gemma-4-26B-A4B-it` | `Qwen/Qwen3.8-27B` |
| operator | `kits/gemma4/gemma4_L16_M1.json` | `kits/qwen38/qwen38_L40_M1_vllm.json` |
| refusal direction | `kits/gemma4/gemma4_L21_refusal_dir.json` (L21) | `kits/qwen38/qwen38_L43_refusal_dir_vllm.json` (L43) |
| gate layer / pooling | L16 `mean_user` | L40 `mean_user` |
| threshold | +0.08284618 | +0.08792114 |
| threshold rule | R3 (`m_budget=11` on the 140 fit-reference prompts). **Out-of-sample.** | R3 (`m_budget=11` on the 140 fit-reference prompts). **Out-of-sample.** |
| precision / backend | `--quantization fp8_per_channel --attention-backend TRITON_ATTN` | `--dtype bfloat16` |

Both operators come from the same 240-prompt fit (100 `cyber_offense` positives, 140 across 14
non-target harm subcategories) and the same closed-form method (`M1_subcat_Tridge_lam16` / `M1_ridge_lam16`),
and both use the blind R3 threshold rule (`m_budget=11` on the 140 fit-reference prompts).

## 4. Known gaps

* **The evaluation corpora are not redistributed in this repository.** The fitting and vLLM activation-harvest
  scripts (`scripts/fit/fit_category.py`, `scripts/fit/fit_refusal.py`, `scripts/fit/harvest_vllm_acts.py`,
  `scripts/fit/sweep_layers_from_dump.py`) and Colab walkthroughs (`notebooks/qwen38_colab.ipynb`) are included,
  while the 240 fit prompts and external evaluation benchmark datasets (`CySecBench`, `CatQA`, `WMDP`, `SOSBench`,
  `PKU-SafeRLHF`) must be supplied separately.
* **Biology and WMDP live measurements.** Both shipped kits are measured live on the biology and WMDP corpora:
  on `gemma4` (`kits/gemma4/MANIFEST.json` -> `measured_live.bio`, measured 2026-11-01 via `serve_eval.sh --kit gemma4`),
  `wmdp_cyber` fires on **606/637 (95.1%)**, `wmdp_bio` on **4/637 (0.6%)** (`1` raw + `3` from the mid/late rescue),
  and `sos_bio` on **0/498 (0.0%)**; on `qwen38` (`kits/qwen38/records/live_validation_2026-11-02.json`),
  `wmdp_bio` fires on **6/637 (0.9%)** and `sos_bio` on **0/498 (0.0%)**.
* **The gate reads no authorization signal.** It decides on cyber-topicality plus refusal pressure,
  so the "authorized pentest" framing in the prompts does no work in the decision.
