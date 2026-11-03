# Evaluator quickstart

Two independent things can be checked, and they cost very different amounts.

| | needs | time | what it proves |
|---|---|---|---|
| **A. Offline** | a laptop | ~2 min | the decision logic, end to end |
| **B. Serving** | one 26B-class GPU | ~15 min | the probe runs on real activations and the gate steers |

Reproducing the published corpus numbers is a third thing and is **not** currently possible from
this repository: the evaluation corpora and the harvest scripts are not in it. See §4.

---

## A. Offline — no GPU, no network, no model

```bash
git clone <repo> && cd cybergate
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest                              # 850 passed, 1 skipped, 6 xfailed
.venv/bin/python -m cybergate_plugin.cli --self-check
```

Requires Python >= 3.10; verified on 3.12.3. The package declares **no runtime dependencies** --
torch, transformers and vllm are supplied by the serving container and imported lazily, so the
suite runs on a machine with none of them.

Do **not** add `--no-build-isolation`: a fresh `python3 -m venv` on Python >= 3.12 contains no
setuptools, and that flag then fails with `ModuleNotFoundError: No module named 'setuptools'`.

What the suite covers, and what it does not, is in `README.md` under *How behaviour preservation is
proved* and *What is not proved*. The short version: the policy is a pure function and is tested
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
# gemma4 kit
./scripts/serving/serve_eval.sh --kit gemma4 --hf ~/.cache/huggingface --gpus device=0

# qwen36 kit -- operator and reasoning parser are selected automatically
./scripts/serving/serve_eval.sh --kit qwen36

# an alternate operator from inside the gemma4 kit
./scripts/serving/serve_eval.sh --kit gemma4 --op gemma4_ruleC_serve.json
```

The launcher prints the kit's gate, its threshold rule and its known gaps before starting, so an
in-sample threshold is visible at serve time rather than buried in a table.

`--help` lists every flag. First load takes several minutes; follow
`logs/<name>_serve.log` and wait for `Application startup complete`.

Then verify the gate is actually live:

```bash
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

**One model per GPU.** Both scripts warn if another vllm container is running. Two 26B models on one
device does not degrade gracefully -- it OOMs, or starves the host until ssh stops completing its
banner exchange.

### The serve flags are load-bearing

`serve_eval.sh` sets these and they are not preferences:

| flag | why |
|---|---|
| `--quantization fp8_per_channel` | 26B at bf16 does not fit; it loads, then OOMs |
| `--no-enable-prefix-caching` | **the subtle one.** The gate probes the prefill. A cached prefix skips the forward pass, so the hook never runs and the decision is silently stale -- the server answers normally and the gate is wrong |
| `--enforce-eager` | the hooks read hidden states; compiled graphs elide them |
| `--attention-backend TRITON_ATTN` | the backend everything here was measured on |

The operator and the refusal direction must match the served weights. A direction fitted on one
checkpoint scores near zero on another and the gate simply stops firing, with no error. That
happened here: see `kits/qwen36/qwen36_L21_M1_refit.json` -> `meta.refit_provenance`.

## 3. What is being served

| | gemma4 | qwen36 |
|---|---|---|
| operator | `kits/gemma4/gemma4_L16_M1.json` | `kits/qwen36/qwen36_L21_M1_refit.json` |
| gate layer / pooling | L16 `mean_user` | L21 `mean_user` |
| threshold | +0.08284618 | -0.00022390 |
| threshold rule | R3: the 12th-largest score on the 140 fit-reference prompts, so 11 of them fire. **Never sees an evaluation corpus.** | `catqa_leak_le_40`: the lowest threshold whose CatQA leak is <= 40. **Chosen against an evaluation corpus, so it is in-sample.** |
| reasoning parser | none | `qwen3` |

Both operators come from the same 240-prompt fit (100 `cyber_offense` positives, 140 across 14
non-target harm subcategories) and the same closed-form method, `M1_subcat_Tridge_lam16`.

**The two threshold rules differ, so the two CatQA figures are not comparable.** Gemma's is a blind
measurement; Qwen's is the constraint that was imposed. R3 at Qwen's own layer gives 621/675 CySec
at 30/234 CatQA if a matched pair is wanted.

## 4. Known gaps

* **The corpus numbers are not reproducible from this repository.** The 240 fit prompts and the
  eleven evaluation corpora are not included, and neither are the harvest and analysis scripts that
  produced every live figure. Request them separately.
* **Neither baseline is measured on biology.** `wmdp_bio`, `sos_bio` and `wmdp_cyber` are unmeasured
  for both shipped operators. Rule C has offline bio numbers; the baselines do not.
* **Qwen's layer and threshold are in-sample** on `cysec_675` and `catqa_234`. No held-out
  confirmation has been run. Gemma's R3 threshold is blind.
* **The gate reads no authorization signal.** It decides on cyber-topicality plus refusal pressure,
  so the "authorized pentest" framing in the prompts does no work in the decision.
