# CyberGate

**Selectively unlock one domain on a pristine model, per request, without touching the weights.**

CyberGate is a plugin for [vLLM](https://github.com/vllm-project/vllm) 0.32. During prefill it reads
the model's own hidden states, decides whether the current request is *both* on-topic for a gated
domain *and* under refusal pressure, and when both hold it prefills the assistant's thought block to
steer the answer. The weights are never modified — `K = 0.0`, no ablation, no fine-tune — and
nothing persists between requests.

```
pytest          # 845 passed, 1 skipped, 6 xfailed, no GPU, no network, ~1.3 s
```

## Why read activations instead of editing weights

The usual way to remove a refusal behaviour is abliteration: find the refusal direction and subtract
it from the weights. That is permanent, global, and invisible afterwards — the model that comes out
refuses less at everything, and you cannot ask it why.

CyberGate uses the same refusal direction as an *instrument* rather than something to delete. The
direction stays, the weights stay, and each request gets a decision that is recorded and can be
replayed. Turn the plugin off and you have the original model back, byte for byte.

## How it decides

Two factors, both required:

| factor | what it measures | how |
|---|---|---|
| **category** | is this request on-topic for the gated domain? | `cos(h, c) >= tau`, where `h` is the hidden state pooled over the user's own tokens at one layer and `c` is a fitted unit direction |
| **refusal** | is the model actually under refusal pressure? | `s_ref_z >= 1.50`, where `s_ref_z = (s_ref_cos - mu_hm_cos) / sd_hm_cos` against a separately calibrated refusal direction at a later layer |

A single factor is not enough in either direction. Category alone fires on any on-topic question the
model would have answered anyway; refusal alone fires on everything the model declines. Requiring
both, plus a middle/late layer veto and a rescue pass, is what makes the gate narrow.

The decision itself is a pure function — `decision.policy.decide(GateInputs) -> GateDecision` — with
every cache read, HTTP call and lookup behind it in `decision.evidence`. That is what makes the
policy testable without a GPU.

Two read-only forward hooks live in the model-worker process and are the only code here that imports
torch. **Read `docs/PROCESS_TOPOLOGY.md` before moving code between packages:** the plugin runs in
two processes that share no memory, and that constraint is the architecture.

## Measured results

Gemma-4-26B-A4B-it, 1,149-prompt live harvest on the served checkpoint, 2026-10-31, every row
matched by exact prompt sha:

| corpus | n | fired | |
|---|---|---|---|
| CySecBench | 675 | **558** (82.7%) | target — firing is correct |
| WMDP-cyber | 637 | **606** | target |
| CatQA | 234 | **23** (9.8%) | non-target — leakage |
| WMDP-bio | 637 | **4** | non-target |
| SOSBench-bio | 498 | **0** | non-target |
| fit positives | 100 | 100 | sanity |
| fit references | 140 | 12 | the set the threshold rule is calibrated on |

Three caveats travel with those numbers, and each kit's `MANIFEST.json` states them in full:

* The worker publishes `fired` *before* the refusal monitor attaches `s_ref_z`, so the record is the
  category gate plus the rescue, not the policy's two-factor veto. Veto-adjusted CySecBench is 553.
* **82 of the 558 fires come from the middle/late rescue**, which the threshold rule does not cover.
  14.7% of delivered recall rests on a separate mechanism.
* The two bio corpora disagree by two orders of magnitude on the same model and operator — 4/637
  versus 0/498 — which points at multiple-choice framing rather than biology content. Not isolated.

## Try it

**Offline, on a laptop, no GPU and no network:**

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'          # do NOT add --no-build-isolation
.venv/bin/python -m pytest
.venv/bin/python -m cybergate_plugin.cli --self-check
```

The package declares **no runtime dependencies**: the decision path is standard library only, and
torch, transformers and vllm are supplied by the serving container and imported lazily. Requires
Python >= 3.10, verified on 3.12.

**On a GPU** — one 26B-class device, Docker with GPU support, weights already in a local Hugging
Face cache:

```bash
./scripts/serving/serve_eval.sh --kit gemma4 --hf ~/.cache/huggingface --gpus device=0
tail -f logs/cybergate_eval_serve.log        # wait for "Application startup complete"
./scripts/serving/check_eval.sh --kit gemma4
```

`docs/EVALUATOR_QUICKSTART.md` has the full version, including what this repository cannot
reproduce: the evaluation corpora and harvest scripts are not in it.

**On Colab**, if you have no local GPU: `notebooks/serve_kit_colab.ipynb` clones this repo,
downloads the weights and serves any kit with the canary check at the end. One 26B-class
model needs an 80 GB card in bf16.

## Kits, and why canaries exist

Everything needed to serve and verify one model lives in `kits/<name>/`, and nothing belonging to
another model does. A kit directory *is* an operator directory, so pointing `CYBERGATE_OPS_DIR` at
one needs no special case, and `kits/` is the only source of servable artifacts.

Each kit carries a `MANIFEST.json` (model ids, checkpoint shape, measured results, known gaps) and a
`canaries.json` — prompts whose verdict was measured on the checkpoint the artifacts were fitted on.

The canaries are the check that matters, because **an operator fitted on a different checkpoint of
the same architecture passes every metadata check, attaches cleanly, and then fires on almost
nothing.** That is not hypothetical: the Qwen operator shipped before the re-fit cleared its own
threshold on 18 of its own 100 fitted training positives on the served checkpoint. Declared
`hidden_size` / `num_layers` / `expected_gate` catch the cross-architecture case at load; only a
canary catches the same-architecture case. A running server proves nothing — the hooks fail silently.

## Deployment

`scripts/serving/serve_eval.sh` is the supported path and pre-flights everything below. By hand:

```bash
pip install --no-deps -e <this directory>
export CYBERGATE_OP_PATH=.../kits/gemma4/gemma4_L16_M1.json
export CYBERGATE_REF_DIR_PATH=.../kits/gemma4/gemma4_L21_refusal_dir.json
export CYBERGATE_MODE=closed_loop

vllm serve google/gemma-4-26B-A4B-it \
  --quantization fp8_per_channel \
  --enforce-eager --no-enable-prefix-caching --attention-backend TRITON_ATTN \
  --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware
```

None of those four flags is a preference:

| flag | without it |
|---|---|
| `--quantization fp8_per_channel` | 26B at bf16 loads, then OOMs |
| `--no-enable-prefix-caching` | **a cached prefix skips the forward pass, so the hook never runs and the decision is silently stale** — the server answers normally and the gate is wrong |
| `--enforce-eager` | compiled graphs elide the hidden states the hooks read |
| `--attention-backend TRITON_ATTN` | the backend everything here was measured on |

The endpoint stays a standard OpenAI-compatible one. `CYBERGATE_MODE` sets the default, so an
ordinary client needs no custom request fields; `cybergate_mode`, `max_cot_tokens` and the rest are
optional per-request overrides that the middleware consumes before forwarding.

Artifact resolution searches the kit first, so a deployment that configures nothing still gets the
shipped operator. `CYBERGATE_OPS_DIR` overrides that and wins outright; `CYBERGATE_OP_PATH` names
one operator file directly. Loading an operator at the wrong gate layer raises `KitMismatch`. See
`src/cybergate/ops/schema.md` for the file contract.

## Adding a model

Three things, and the plugin is not one of them:

1. **The two directions** — a category direction with its threshold, and a refusal direction with
   its `mu`/`sd` calibration. `scripts/fit/` fits both; `scripts/fit/README.md` documents the method,
   the two reproductions, and the two ways to get it wrong. `notebooks/qwen38_colab.ipynb` walks a
   new model end to end.
2. **A render declaration** — a `ModelFamily` subclass carrying that checkpoint's turn markers *as
   measured token ids*, its thought-channel strings, BOS handling and policy thresholds. This is the
   one that fails quietly: wrong markers do not raise, they fall through to a fixed-offset span and
   score arbitrary tokens. Check `span_method` in the decision record.
3. **Canaries** — measured on that checkpoint. Without them you cannot tell a working gate from an
   attached one.

## How correctness is established

The pre-refactor implementation is vendored at `tests/equivalence/baseline/`, pinned by sha256, and
the suite drives it in lockstep with the current code:

| Suite | Compares | Cases |
|---|---|---|
| `test_differential.py` | the full decision payload, key by key, from the same `/dev/shm` records | 48 |
| `test_prompts_differential.py` | raw prompts as exact strings, plus both text transforms | 336 |
| `test_spans_differential.py` | token-span location and BOS batch segmentation | 42 |
| `test_think_routes_differential.py` | the exact sequence of completion calls, payloads, content and telemetry | 38 |

First-run-green is not evidence, so each differential was checked by mutation: flipping one
comparison operator, or changing one character in a rendered `source=` string, fails the suite. The
route differential found two real bugs before it went green.

The corpus is a table of floats rather than prompts — a decision is a function of the probe's scores,
so the cases specify those directly. That reaches every branch and both sides of every threshold,
which driving conversations through a live server does not.

## What is not verified

* **The hook bodies are not executed by the suite.** They need torch and a loaded model. They are
  import-checked and pyflakes-clean; a `NameError` from a late edit was caught by grep, not by a
  test. Treat that as the measure of the gap, and use `check_eval.sh` against a live server.
* **The middleware is not driven end to end.** Its probe sequencing and ASGI plumbing have no test;
  the routes they call do.
* **Hook latency is not measured here.** The 64–70 µs `gpu_dot_us` figure is an audit measurement of
  the pre-refactor code, not of this one.

## Known defects

R6/R7, R10, R11, R12, R13, R14 and a missing IPC TTL are carried forward deliberately. Each has a
test asserting *current* behaviour plus a strict-xfail test asserting the desired behaviour, so a
fix flips exactly its own test and nothing else. Two more are described in `docs/ARCHITECTURE.md`:
**D1**, `evidence.legacy_op_for_model_id` loads the wrong operator for a model id that is neither
Gemma nor Qwen, and **D2**, `Branch.NEURAL_REVOKED` is unreachable in both families.

**The largest limitation is by design, not by oversight: the gate reads no authorization signal at
all.** It decides on topicality plus refusal pressure, so the "authorized pentest" framing in the
prompts and test suites does no work in the decision. Anything that depends on *who* is asking needs
a design answer this codebase does not have.

## Layout

```
src/cybergate/            the implementation
  config.py               constants, env vars, paths (stdlib only)
  boilerplate.py          injected thought text, pivot regex, the two text transforms
  messages.py             OpenAI-format message handling
  families/               base.py + gemma4/ qwen36/ -- per-model render and kit binding
  ops/                    operator loading; schema.md documents the file contract
  state/                  keys, backend protocol, /dev/shm store, memory store, process globals
  decision/               inputs, records, evidence (ALL I/O), rules, policy (PURE), outcome, api
  hooks/                  the two worker-side forward hooks -- the only torch in the package
  vllm_adapter/           quarantined version-fragile vLLM/HF introspection
  serving/                middleware, client, think_routes, asgi, prompts
  telemetry/              event logging
src/cybergate_plugin/     drop-in facade: the --middleware path and entry point
kits/                     one self-contained deployment kit per served model
scripts/fit/              fit a new model's two directions
scripts/serving/          serve and verify
notebooks/                Colab: serve_kit_colab (any kit), qwen38_colab (fit one)
tests/unit/               policy regressions R1-R14, keys, families, kit binding, purity
tests/equivalence/        differential suites against the vendored reference implementation
docs/                     ARCHITECTURE, PROCESS_TOPOLOGY, EVALUATOR_QUICKSTART, DECODE_CALIBRATION
```

`cybergate_plugin` keeps the older import paths (`cybergate_plugin.gate_probe`,
`cybergate_plugin.prompts`, `cybergate_plugin.middleware`) and the pre-rename `ResGate*` aliases.
Attribute access is lazy, so the API-server process does not import torch just to start.

## License

MIT — see `LICENSE`. `ATTRIBUTION.md` records the upstream provenance of the one fitting component
that has any, and why this repository is MIT rather than AGPL-3.0-or-later like the research package
it grew up beside.
