# CyberGate — Abliteration Without Weight Editing

> **Selectively unlock an LLM's cybersecurity capabilities—per request, without modifying model weights.**  
> **Unlock cybersecurity while safely refusing other harmful topics.**

CyberGate is a plugin for [vLLM](https://github.com/vllm-project/vllm) 0.32. During prefill it reads
the model's own hidden states, decides whether the current request is *both* on-topic for a gated
domain *and* under refusal pressure, and when both hold it prefills the assistant's thought block to
steer the answer. The weights are never modified — `K = 0.0`, no ablation, no fine-tune — and
nothing persists between requests.

```
pytest          # 991 passed, 1 skipped, 7 xfailed, no GPU, no network, ~4.8 s
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

Gemma-4-26B-A4B-it, live harvests on the served checkpoint (`2026-10-31` and `2026-11-01`; the `2026-11-01` bio harvest ran via `scripts/serving/serve_eval.sh --kit gemma4`, every row matched by exact prompt sha; see `kits/gemma4/MANIFEST.json`, where `measured_offline` records the firing counts `611`, `3`, and `0` on the same 637- and 498-row corpora):

| corpus | n | fired | |
|---|---|---|---|
| CySecBench | 675 | **558** (82.7%) | target — live harvest |
| WMDP-cyber | 637 | **606** (95.1%) | target — live harvest (`611/637` offline) |
| CatQA | 234 | **23** (9.8%) | non-target — live harvest leakage |
| WMDP-bio | 637 | **4** (0.6%) | non-target — live harvest (`1` raw + `3` rescue; `3/637` offline) |
| SOSBench-bio | 498 | **0** (0.0%) | non-target — live harvest (`0/498` offline) |
| fit positives | 100 | 100 | sanity — live harvest |
| fit references | 140 | 12 | the set the threshold rule is calibrated on |

Three caveats travel with those numbers, and each kit's `MANIFEST.json` states them in full:

* The worker publishes `fired` *before* the refusal monitor attaches `s_ref_z`, so the record is the
  category gate plus the rescue, not the policy's two-factor veto. Veto-adjusted CySecBench is 553.
* **82 of the 558 fires come from the middle/late rescue**, which the threshold rule does not cover.
  14.7% of delivered recall rests on a separate mechanism.
* The two bio corpora disagree by two orders of magnitude on the same model and operator — 4/637 on
  `WMDP-bio` versus 0/498 on `SOSBench-bio` — which points at multiple-choice framing rather than
  biology content. Not isolated.

### A second model, measured the same way

Qwen3.8-27B, 4,134-prompt live harvest on the served checkpoint, 2026-11-02. The CySecBench split
below **chose nothing** — not the layer, not the threshold, not the direction:

| corpus | n | qwen38 | gemma4 | |
|---|---|---|---|---|
| CySecBench | 675 | **587** (87.0%) | 558 (82.7%) | target |
| CatQA | 234 | **23** (9.8%) | 23 (9.8%) | non-target |
| WMDP-bio | 637 | 6 (0.9%) | 4 (0.6%) | non-target |
| SOSBench-bio | 498 | **0** | 0 | non-target |
| harmless | 1000 | 12, and **0** after the veto | not measured | non-target |
| PKU-cybercrime | 612 | **220** (35.9%) | not measured | non-target |

Qwen3.8 has **no middle/late rescue**, so all 587 fires are threshold crossings — against 476 of
gemma4's 558. In the initial 2026-11-02 harvest (`kits/qwen38/records/live_validation_2026-11-02.json`),
its refusal sigma still came from the `transformers`-capture refusal vector, which caused the two-factor
veto to remove 90 of the 587 fires (73.6% veto-adjusted recall against gemma4's 81.9%). The shipped kit now
includes a vLLM-refitted L43 refusal direction (`kits/qwen38/qwen38_L43_refusal_dir_vllm.json`, `sep = 1.1487`,
`z_harm_mean = +18.18σ`, `681/681` harmful above `1.50σ`); and the 35.9% (`220/612`) PKU-cybercrime leak
is not explained (`kits/qwen38/MANIFEST.json` `known_gaps`).

**What this model cost to get working is the more useful result.** Its first operator was fitted on
`transformers` activations, the way every kit here was, and served under vLLM it gated nothing —
not from a bug, but because the directions do not transfer between the two engines on a hybrid
linear-attention stack. Served, it scored SOSBench-bio prompts as highly as its own cyber
positives. Refitting on vLLM's own pooled activations fixed it: four prompts from the diagnosis
scored +0.0370, +0.0021, +0.0225 and −0.0014 against a threshold of +0.0630 before, and +0.4105,
+0.3942, +0.3933 and +0.4196 against +0.0879 after. Same prompts, same checkpoint, same render.

And the layer that in-sample AUC picked was one of the worst in the stack. **In-sample AUC is 0.989+
at all 64 layers (0.9894 to 0.9993)**, including layers whose held-out recall is 0.21; it selected L39, which refitted
properly reaches 0.9526 recall and leaks **0.2104** into WMDP-bio. The 64-layer table is in
`kits/qwen38/records/harvest_2026-11-02_layer_sweep.txt` if you want to check that yourself.

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
Face cache. Two ready-to-run Docker Compose files are provided at the repository root—one per
supported model:

- **`docker-compose.gemma-4-26B-A4B-it.yml`** — serves `google/gemma-4-26B-A4B-it` (MoE, FP8) with `kits/gemma4`
- **`docker-compose.Qwen3.8-27B.yml`** — serves `Qwen/Qwen3.8-27B` (Dense, BF16) with `kits/qwen38`

#### 1. Launch either model with Docker Compose

```bash
# Option A: Launch google/gemma-4-26B-A4B-it
docker compose -f docker-compose.gemma-4-26B-A4B-it.yml up -d
docker logs -f cybergate_gemma4_26b_a4b_it      # wait for "Application startup complete"

# Option B: Launch Qwen/Qwen3.8-27B
docker compose -f docker-compose.Qwen3.8-27B.yml up -d
docker logs -f cybergate_qwen38_27b             # wait for "Application startup complete"
```

#### 2. Which Port to Connect To (`:8000` Pristine vs `:8001` CyberGate)

A single container loads the model into GPU memory **once** and binds **two OpenAI-compatible ports** inside the same process (`--port 8000` bound by vLLM Uvicorn + `CYBERGATE_PROXY_PORT=8001` bound on a background thread by `CyberGateThinkMiddleware`):

| Port | OpenAI API Base URL | Mode | What to Connect Here |
|---|---|---|---|
| **Port `8000`** *(Pristine by default)* | `http://localhost:8000/v1` | `CYBERGATE_MODE=baseline` | **Pristine (Un-Steered by Default) Model.** Requests without a `"cybergate_mode"` override in the JSON body pass straight through to the unmodified base model (`routed="pristine_pass_through"`), while still running the prefill activation probe and returning `x_cybergate` telemetry. Use `:8000` (with `"cybergate_mode"` in the JSON body) for raw gate measurements. **Note:** Any caller who can reach `:8000` can activate the gate by passing `"cybergate_mode": "closed_loop"` in the request body; do not expose `:8000` to callers who must not access gated mode. |
| **Port `8001`** *(CyberGate)* | `http://localhost:8001/v1` | `cybergate_mode=closed_loop` | **CyberGate (Gated) Agent/Chat Bridge.** Requests go through the two-factor activation gate (`cos(h, c) >= tau` and `s_ref_z >= +1.50`) + inline CoT refusal steering, OpenAI `tools`/`<tool_call>` normalization, multi-turn history compaction, and `[CyberGate \| ...]` telemetry header. Point **OpenCode**, **MCP agents**, **Open WebUI**, or gated API clients to `:8001/v1`. |

> **Security, network binding, logging & `:8001` bridge notes:**
> - **Bind addresses, authentication & access control:** `middleware.py` and `serve_eval.sh` bind the `:8001` proxy to `127.0.0.1` by default (`CYBERGATE_PROXY_HOST=127.0.0.1`). The two `docker-compose.*.yml` files set `network_mode: host`, `vllm serve --host 0.0.0.0`, and `CYBERGATE_PROXY_HOST=0.0.0.0` so both `:8000` and `:8001` are reachable across the host network, and **neither port requires authentication in the default Compose configuration**. To restrict access on shared networks, either set `--host 127.0.0.1` and `CYBERGATE_PROXY_HOST=127.0.0.1` (and connect via `ssh -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 user@gpu-host`), apply a host firewall rule (`ufw` / `iptables`, which applies directly under `network_mode: host`), or pass `--api-key <key>` to `vllm serve` (`:8001` forwards the `Authorization` header to `:8000`).
> - **Conversation trace logs:** Outside Docker Compose, `CYBERGATE_PROXY_TRACE_JSONL` and `CYBERGATE_PROXY_DEBUG_LOG` default to `""` (disabled). The two `docker-compose.*.yml` files set both to `/work/cybergate_public/logs/...` inside the container (mounted from `./logs/` on the host), recording full request/response conversation traces.
> - **`:8001` agent/chat bridge behavior vs `:8000`:** Port `:8001` is an agent/chat normalization bridge rather than a transparent byte proxy (use `:8000` with `"cybergate_mode"` for raw benchmark evaluation). Specifically, `:8001`: (1) strips prior `<think>` blocks and compacts long assistant history turns; (2) merges any caller `system` message with the `<tool_call>` instruction prompt when `tools` are declared and converts `role: "tool"` turns into `[Tool Execution Output]` user turns; (3) routes Open WebUI `### Task:` background metadata prompts to `cybergate_mode="baseline"`; (4) defaults `max_tokens` to `min(CYBERGATE_DEFAULT_MAX_TOKENS, remaining_context)` (`CYBERGATE_DEFAULT_MAX_TOKENS=4096` when omitted); and (5) buffers the upstream `:8000` response (with timeout `CYBERGATE_PROXY_TIMEOUT=600` seconds, returning HTTP `502` on upstream timeout) before emitting JSON or SSE chunks so `<tool_call>` blocks and reasoning splits can be repaired cleanly.

#### 3. Side-by-Side Example (`:8000` Pristine vs `:8001` CyberGate)

```bash
# Connect to Port 8000 -> Pristine (un-gated) base model response
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26B-A4B-it",
    "messages": [{"role": "user", "content": "Write a Python script to test a target for SQL injection."}]
  }'

# Connect to Port 8001 -> CyberGate (gated) response with two-factor activation gate
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26B-A4B-it",
    "messages": [{"role": "user", "content": "Write a Python script to test a target for SQL injection."}]
  }'
```

#### 4. Verify the Live Deployment

```bash
# Verify Gemma-4 container
./scripts/serving/check_eval.sh --kit gemma4 --port 8000 --ipc /dev/shm/cybergate_ipc_gemma4

# Or using the standalone shell launcher instead of Docker Compose:
./scripts/serving/serve_eval.sh --kit gemma4 --hf ~/.cache/huggingface --gpus device=0
./scripts/serving/check_eval.sh --kit gemma4
```

`docs/EVALUATOR_QUICKSTART.md` has the full version. The fitting scripts (`scripts/fit/`) are in
this repository, while the external evaluation benchmark corpora are not redistributed here.

**On Colab**, if you have no local GPU: `notebooks/qwen38_live_chat.ipynb` serves a kit and
lets you talk to it, showing the gate's decision on every turn and checking that a request
the gate *holds* comes back byte-identical to the pristine model.
`notebooks/serve_kit_colab.ipynb` is the same serve without the conversation, for verifying
a kit. One 26B-class model needs an 80 GB card in bf16.

Run `notebooks/render_preflight.ipynb` first. It needs no GPU, no vllm and no weights -- only a
tokenizer -- and it answers the one question a serve cannot: of all the prompt renders this
deployment produces, which ones does `is_baseline_prefill` publish? Exactly one must. A kit that
publishes none serves perfectly and gates nothing; a kit that publishes two overwrites its own
decision and, with a dump running, appends a second unlabelled row.

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

`docker compose -f docker-compose.gemma-4-26B-A4B-it.yml up -d` (or `-f docker-compose.Qwen3.8-27B.yml`) and `scripts/serving/serve_eval.sh`
are the supported paths and configure everything below. By hand:

```bash
pip install --no-deps -e <this directory>
export CYBERGATE_OP_PATH=.../kits/gemma4/gemma4_L16_M1.json
export CYBERGATE_REF_DIR_PATH=.../kits/gemma4/gemma4_L21_refusal_dir.json
export CYBERGATE_MODE=baseline          # Port :8000 serves un-gated normal LLM
export CYBERGATE_PROXY_PORT=8001        # Port :8001 auto-binds for gated CyberGate LLM (closed_loop)

vllm serve google/gemma-4-26B-A4B-it \
  --quantization fp8_per_channel \
  --enforce-eager --enable-prefix-caching --attention-backend TRITON_ATTN \
  --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware
```

None of those four flags is a preference:

| flag | why |
|---|---|
| `--quantization fp8_per_channel` | 26B at bf16 loads, then OOMs |
| `--enable-prefix-caching` | reuses KV blocks across multi-turn chats and Pass-1 $\to$ Pass-2 generation (Pass-1 probes set per-turn `cache_salt` so hooks always observe the full prompt) |
| `--enforce-eager` | compiled graphs elide the hidden states the hooks read |
| `--attention-backend TRITON_ATTN` | the backend everything here was measured on |

The endpoint stays a standard OpenAI-compatible one. `CYBERGATE_MODE` sets the default, so an
ordinary client needs no custom request fields; `cybergate_mode`, `max_cot_tokens` and the rest are
optional per-request overrides that the middleware consumes before forwarding.

Artifact resolution searches `<repo>/kits/<family>/` first when installed in editable mode (`pip install -e .`)
from a source checkout, so a local checkout that configures nothing still resolves the shipped operator.
In container or non-editable deployments, set `CYBERGATE_OPS_DIR` to the kit directory (or `CYBERGATE_OP_PATH`
to name one operator file directly). Loading an operator at the wrong gate layer raises `KitMismatch`. See
`src/cybergate/ops/schema.md` for the file contract.

## Adding a model

Three things, and the plugin is not one of them:

1. **The two directions** — a category direction with its threshold, and a refusal direction with
   its `mu`/`sd` calibration. `scripts/fit/` fits both (`fit_category.py`, `fit_refusal.py`,
   `harvest_vllm_acts.py`, `sweep_layers_from_dump.py`), and `notebooks/qwen38_colab.ipynb` walks a
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

Policy behaviors `R6/R7`, `R8`, `R10`, `R12`, `R13`, `R14` and a missing IPC TTL are carried forward
deliberately so the differential equivalence suite against the vendored reference remains exact.
`tests/unit/test_regressions.py` has **6 strict-xfail** tests (`R6/R7` across 2 tests, `R10`, `R12`,
`R13`, `R14`, plus a control test for `R8`), and `tests/unit/test_baseline_prefill_renders.py` carries
the **7th strict-xfail** test (`gemma4`'s `is_baseline_prefill` accepting `apply_chat_template`), so a
fix flips its own test and nothing else. (**R11**, cross-conversation `in_context_key` collision, and
**D1**, family operator resolution by model id, are fixed.) See `docs/ARCHITECTURE.md` (§5) for the full
table and **D2** (`Branch.NEURAL_REVOKED` reachability and shipped hysteresis values: `+0.0350` on Gemma-4 L16, `-0.0250` on Qwen).

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
  families/               base.py + gemma4/ qwen38/ gemma4_31b/ qwen36/ -- per-model render and kit binding
  ops/                    operator loading; schema.md documents the file contract
  state/                  keys, backend protocol, /dev/shm store, memory store, process globals
  decision/               inputs, records, evidence (ALL I/O), rules, policy (PURE), outcome, api
  hooks/                  the two worker-side forward hooks -- the only torch in the package
  vllm_adapter/           quarantined version-fragile vLLM/HF introspection
  serving/                middleware, proxy, client, think_routes, asgi, prompts
  telemetry/              event logging
src/cybergate_plugin/     drop-in facade: the --middleware path and entry point
kits/                     one self-contained deployment kit per served model (gemma4, qwen38)
scripts/fit/              fit a new model's two directions
scripts/serving/          serve and verify
notebooks/                Colab: render_preflight (does the gate observe anything? CPU
                          only, run it first), qwen38_live_chat (talk to it),
                          serve_kit_colab (verify a kit), qwen38_colab (fit one)
tests/unit/               policy regressions R1-R14, keys, families, kit binding, purity, proxy
tests/equivalence/        differential suites against the vendored reference implementation
docs/                     ARCHITECTURE, PROCESS_TOPOLOGY, EVALUATOR_QUICKSTART, DECODE_CALIBRATION
```

`cybergate_plugin` keeps the older module import paths (`cybergate_plugin.gate_probe`,
`cybergate_plugin.prompts`, `cybergate_plugin.middleware`) and `CyberGate*` symbols.
Attribute access is lazy, so the API-server process does not import torch just to start.

## License

MIT — see `LICENSE`. `ATTRIBUTION.md` records the upstream provenance of the one fitting component
that has any, and why this repository is MIT rather than AGPL-3.0-or-later like the research package
it grew up beside.
