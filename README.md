# CyberGate: vLLM Plugin for Cybersecurity Uncensoring Without Weight Editing

[![vLLM Plugin](https://img.shields.io/badge/vLLM_Plugin-0.32.0-3b82f6.svg)](https://github.com/vllm-project/vllm)
[![Model: Qwen3.8-27B](https://img.shields.io/badge/Model-Qwen3.8--27B-f59e0b.svg)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Model: Gemma-4-26B](https://img.shields.io/badge/Model-Gemma--4--26B--A4B-ea4335.svg)](https://huggingface.co/google/gemma-4-26B-A4B-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)
[![Offline Tests](https://img.shields.io/badge/Tests-1%2C033%20passed%20(no%20GPU)-8b5cf6.svg)](tests/)

> **A drop-in vLLM 0.32 plugin that unlocks cybersecurity, CTF, and reverse-engineering capabilities per request - using original upstream weights.**  
> **Ships ready-to-run kits for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (plus fitting scripts to add new `<think>` models).**  
> **Other harmful domains stay safely refused. Zero weight editing, zero capability loss.**

When local LLMs refuse security research, CTF, or reverse-engineering prompts, the standard workaround is **abliteration** - subtracting the refusal direction from the model's weights.

**CyberGate** is a **vLLM plugin** (`--middleware` + worker forward hooks + V1 `LogitsProcessor`) that takes the opposite approach: instead of deleting the refusal direction globally from the weights, it reads hidden-state activations per request and steers the model's `<think>` block **only** when a prompt is both on-topic for cybersecurity *and* under refusal pressure.

| | ❌ Weight Abliteration (Global Edit) | ✅ CyberGate (vLLM Plugin) |
|---|---|---|
| **Model Weights** | Permanently mutated across layers | **100% untouched** (`K = 0.0`, no LoRA, no fine-tune) |
| **Reasoning & Tool Calls** | Can silently degrade multi-step CoT & tool JSON | **Zero capability drift** - turn it off for byte-identical base output |
| **Refusal Selectivity** | **Zero** - unlocks *every* harmful domain (bio, self-harm, etc.) | **Domain-selective** - opens only for cyber (`0.0%` SOSBench-bio leak) |
| **Checkpoint Storage** | Requires storing & maintaining a forked 50 GB model | **Serves stock upstream Hugging Face checkpoints directly** |

> **Live vLLM Benchmarks:** Unlocks **87.0%** on CySecBench (`Qwen3.8-27B`) and **95.1%** on WMDP-cyber (`Gemma-4-26B`), while holding **SOSBench-bio at 0.0% (`0/498`)** and **WMDP-bio under 1%**.  
> 👉 **[Full Benchmarks, Adding Models & Quantization FAQ (`docs/RESULTS_AND_FAQ.md`)](docs/RESULTS_AND_FAQ.md)**

---

## Quickstart: Try It First on Original Models

### 1. Launch on a GPU (Docker Compose, or Without Docker / Google Colab)

```bash
# --- Option A: With Docker Compose ---
# Launch Qwen/Qwen3.8-27B (or use docker-compose.gemma-4-26B-A4B-it.yml)
docker compose -f docker-compose.Qwen3.8-27B.yml up -d
docker logs -f cybergate_qwen38_27b             # wait for "Application startup complete"
```

```bash
# --- Option B: Without Docker (Google Colab A100, RunPod, Vast, or bare Python venv) ---
# Add --bootstrap on a fresh VM to auto-install vllm==0.32.0 and download upstream weights:
./scripts/serving/serve_no_docker.sh qwen38 --bootstrap
# (or: ./scripts/serving/serve_no_docker.sh gemma4 --bootstrap)

./scripts/serving/serve_no_docker.sh --status   # check :8000 (Pristine) and :8001 (CyberGate)
./scripts/serving/serve_no_docker.sh --stop     # stop the background server
```

<details>
<summary><b>No GPU handy? Run the 1,033-test decision suite offline on any laptop (~8 s)</b></summary>

The package declares **zero runtime dependencies** (`cybergate.decision` is pure Python stdlib; `torch`, `transformers`, and `vllm` are imported lazily inside the serving worker):

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'          # do NOT add --no-build-isolation
.venv/bin/python -m pytest
.venv/bin/python -m cybergate_plugin.cli --self-check
```

</details>

### 2. Connect Your Client or Agent (`:8000` Pristine vs `:8001` CyberGate)

A single `vllm serve` process loads the model into VRAM **once** (with `--enable-prefix-caching` enabled) and binds two OpenAI-compatible ports:

| Port | Base URL | Mode | What to Connect Here |
|---|---|---|---|
| **`:8001`** *(CyberGate)* | `http://localhost:8001/v1` | `closed_loop` | **Gated Agent / Chat Endpoint.** Runs the two-factor activation gate + closed-loop `<think>` steering, preserves native OpenAI `tools`/`tool_calls` and `--reasoning-parser`, strips prior CyberGate `<think>` boilerplate so multi-turn KV prefix caching stays clean, and clamps `max_tokens` via `/tokenize`. Point **OpenCode**, **MCP agents**, or **Open WebUI** here. |
| **`:8000`** *(Base Model)* | `http://localhost:8000/v1` | `baseline` | **Un-Steered Upstream Model.** Requests without `"cybergate_mode"` in the JSON body pass straight through un-steered (`routed="pristine_pass_through"`) while returning `x_cybergate` probe telemetry. Use `:8000` for baseline comparisons or raw gate benchmarks. |

> **Network & Security Note:** Outside Docker Compose, `:8001` binds to `127.0.0.1` with trace logs disabled by default. The two `docker-compose.*.yml` files use `network_mode: host`, bind `:8000` and `:8001` on `0.0.0.0` without authentication, and write conversation traces under `./logs/`. Any caller who can reach `:8000` can also request gated mode via `"cybergate_mode": "closed_loop"` in the JSON body - on shared networks, bind to `127.0.0.1` + SSH tunnel, add firewall rules, or pass `--api-key <key>` to `vllm serve` (`:8001` forwards `Authorization` headers).

<details>
<summary><b>Click to expand: OpenCode configuration (<code>~/.config/opencode/opencode.json</code>)</b></summary>

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "cybergate/Qwen/Qwen3.8-27B",
  "provider": {
    "cybergate": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "CyberGate vLLM",
      "options": {
        "baseURL": "http://127.0.0.1:8001/v1",
        "apiKey": "EMPTY"
      },
      "models": {
        "Qwen/Qwen3.8-27B": {
          "name": "Qwen3.8-27B (CyberGate)",
          "tool_call": true,
          "limit": { "context": 262144, "output": 16384 }
        },
        "google/gemma-4-26B-A4B-it": {
          "name": "Gemma-4-26B-A4B-it (CyberGate)",
          "tool_call": true,
          "limit": { "context": 262144, "output": 16384 }
        }
      }
    }
  }
}
```

</details>

### 3. Side-by-Side Test & Canary Check

```bash
# Port 8000 -> Un-gated upstream base model
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python script to test a target for SQL injection."}]}'

# Port 8001 -> CyberGate (two-factor gate + closed-loop <think> steering)
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python script to test a target for SQL injection."}]}'

# Replay checkpoint-pinned canary prompts to verify live gate health
./scripts/serving/check_eval.sh --kit qwen38 --port 8000 --ipc /dev/shm/cybergate_ipc_qwen38
```

---

## How It Works: Two-Factor Gate + Closed-Loop `<think>` Steering

### Stage 1: Two-Factor Prefill Activation Gate
During prefill, two read-only PyTorch forward hooks in the vLLM worker read hidden states and evaluate a pure decision function (`cybergate.decision.policy.decide`):

1. **Category (Domain):** `cos(h, c) >= tau`, where `h` is the hidden state pooled over user tokens (`L16` on Gemma-4, `L40` on Qwen3.8) and `c` is a fitted cybersecurity unit vector.
2. **Refusal Pressure:** `s_ref_z >= +1.50σ` against a separately calibrated refusal vector at a later layer (`L21` on Gemma-4, `L43` on Qwen3.8).

Requiring **both** keeps benign security chat un-steered and non-cyber harms safely refused.

### Stage 2: Closed-Loop `<think>` Steering (`RefusalSteeredCoTProcessor`)
Even after an analytical `<think>` opener, reasoning models frequently pivot toward refusal mid-thought. CyberGate steers inside **a single forward pass** via a vLLM V1 `LogitsProcessor` (no regex word-lists, no extra re-prefills):

1. **Open-thought prefill:** Seeds an analytical `<think>` opener via the kit's `chat_template.jinja` when the gate fires.
2. **Decode-step refusal monitoring:** Checks rolling refusal projection (`rolling_s_ref_z`) at **every generated token** inside `<think>`, force-injecting a brief inline clearance nudge if refusal spikes (`>= +3.55σ`).
3. **Tool-aware `</think>` close lock:** Closes `</think>` with a compliance lock, while automatically suppressing prose openers on tool turns (`cybergate_suppress_opener`) so native `<tool_call>` JSON emits cleanly.

---

## Benchmarks, Adding Models & Documentation

- **[docs/RESULTS_AND_FAQ.md](docs/RESULTS_AND_FAQ.md)** - Live vLLM harvest benchmarks, engineering findings, **adding new models**, and **quantization (`FP8` / `4-bit`) FAQ**.
- **[docs/EVALUATOR_QUICKSTART.md](docs/EVALUATOR_QUICKSTART.md)** - Full serving flags, port architecture, and canary verification.
- **[docs/PROCESS_TOPOLOGY.md](docs/PROCESS_TOPOLOGY.md)** & **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Two-process vLLM V1 topology, decision policy, and differential test suites.
- **[docs/DECODE_CALIBRATION.md](docs/DECODE_CALIBRATION.md)** - Closed-loop CoT spike calibration.

## License

MIT - see [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).
