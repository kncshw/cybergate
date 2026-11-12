# CyberGate：无需修改模型权重的网络安全定向解锁 vLLM 插件

**[English](README.md) | [简体中文](README_zh.md)**

[![vLLM Plugin](https://img.shields.io/badge/vLLM_Plugin-0.32.0-3b82f6.svg)](https://github.com/vllm-project/vllm)
[![Model: Qwen3.8-27B](https://img.shields.io/badge/Model-Qwen3.8--27B-f59e0b.svg)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Model: Gemma-4-26B](https://img.shields.io/badge/Model-Gemma--4--26B--A4B-ea4335.svg)](https://huggingface.co/google/gemma-4-26B-A4B-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)
[![Offline Tests](https://img.shields.io/badge/Tests-1%2C033%20passed%20(no%20GPU)-8b5cf6.svg)](tests/)

> **即插即用的 vLLM 0.32 插件：直接使用官方原始模型权重，按请求动态解锁网络安全、CTF 与逆向工程能力。**  
> **开箱即用支持 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 与 [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)（并提供脚本适配更多 `<think>` 推理模型）。**  
> **其他有害领域依然保持安全拒答——零权重修改、零智商与工具调用损耗。**

当本地大模型误拒安全研究、CTF 或逆向工程（RE）请求时，传统做法是 **权重消融（Abliteration）**——直接从模型权重中减去拒答方向向量。

**CyberGate** 作为 **vLLM 插件**（`--middleware` + Worker 只读前向 Hook + V1 `LogitsProcessor`）采用了完全相反的思路：**不碰任何模型权重**，而是在每次请求的 Prefill 阶段读取隐藏层激活，**仅当**请求同时满足“属于网络安全领域”且“处于拒答压力下”两个条件时，才在单轮生成中对 `<think>` 思维链进行闭环引导。

| | ❌ 传统权重消融（Abliteration 全局改权重） | ✅ CyberGate（vLLM 插件） |
|---|---|---|
| **模型权重** | 跨层永久篡改模型参数 | **100% 原生未改动**（`K = 0.0`，无 LoRA、无微调） |
| **推理与工具调用能力** | 容易暗中损伤复杂推理（CoT）与 Tool Call 格式 | **零能力退化** - 关闭插件即逐字节还原官方原版输出 |
| **拒答选择性** | **零选择性** - 一刀切放开生物武器、自残等所有危险领域 | **领域精准选择** - 仅对网安请求放行（`SOSBench-bio` 误放率 `0.0%`） |
| **权重存储与维护** | 需要额外存储和维护 50 GB 魔改权重副本 | **直接加载 Hugging Face 官方原版权重** |

> **vLLM 实测指标：** 在 `Qwen3.8-27B` 上解锁 **87.0%** `CySecBench`，在 `Gemma-4-26B` 上解锁 **95.1%** `WMDP-cyber`；同时非目标危害 `SOSBench-bio` 误触发率为 **0.0%（`0/498`）**，`WMDP-bio` 低于 **1%**。  
> 👉 **[完整评测数据、新模型适配与量化（FP8 / 4-bit）常见问题 (`docs/RESULTS_AND_FAQ_zh.md`)](docs/RESULTS_AND_FAQ_zh.md)**

---

## 快速开始：直接使用官方原版模型体验

### 1. 在 GPU 上启动（支持 Docker Compose 或无 Docker / Google Colab）

```bash
# --- 方式 A：使用 Docker Compose ---
# 启动 Qwen/Qwen3.8-27B（或使用 docker-compose.gemma-4-26B-A4B-it.yml）
docker compose -f docker-compose.Qwen3.8-27B.yml up -d
docker logs -f cybergate_qwen38_27b             # 等待出现 "Application startup complete"
```

```bash
# --- 方式 B：无 Docker 启动（适用于 Google Colab A100、RunPod、Vast 或本地虚拟环境） ---
# 全新机器加上 --bootstrap 可自动安装 vllm==0.32.0 并下载官方原版权重：
./scripts/serving/serve_no_docker.sh qwen38 --bootstrap
# （或：./scripts/serving/serve_no_docker.sh gemma4 --bootstrap）

./scripts/serving/serve_no_docker.sh --status   # 查看 :8000（原版）和 :8001（CyberGate）状态
./scripts/serving/serve_no_docker.sh --stop     # 停止后台服务
```

<details>
<summary><b>手头没有 GPU？可在普通笔记本上离线运行 1,033 项决策单元测试（约 8 秒）</b></summary>

核心决策包声明了 **零运行时第三方依赖**（`cybergate.decision` 仅使用 Python 标准库；`torch`、`transformers` 与 `vllm` 均在推理进程内按需延迟加载）：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'          # 请勿添加 --no-build-isolation
.venv/bin/python -m pytest
.venv/bin/python -m cybergate_plugin.cli --self-check
```

</details>

### 2. 接入客户端或 Agent（`:8000` 原版端口 vs `:8001` CyberGate 端口）

单个 `vllm serve` 进程只需将模型加载进显存 **一次**（完整保留 `--enable-prefix-caching` KV 缓存复用），并同时暴露两个兼容 OpenAI 协议的端口：

| 端口 | OpenAI 接口地址 | 模式 | 适用接入场景 |
|---|---|---|---|
| **`:8001`** *(CyberGate)* | `http://localhost:8001/v1` | `closed_loop` | **受控解锁 Agent / 对话入口。** 自动执行双因子激活门控 + 闭环 `<think>` 引导，原生保留 OpenAI `tools`/`tool_calls` 与 `--reasoning-parser`，自动清理历史轮次注入的 `<think>` 模板文本以保护多轮 KV 缓存命中率，并通过 `/tokenize` 自动裁剪 `max_tokens`。推荐将 **OpenCode**、**MCP Agent** 或 **Open WebUI** 指向此端口。 |
| **`:8000`** *(原版模型)* | `http://localhost:8000/v1` | `baseline` | **未干预的原版基座模型。** 未携带 `"cybergate_mode"` 字段的请求将原样直通（`routed="pristine_pass_through"`），同时返回 `x_cybergate` 探针遥测信息。用于基线对比或门控离线评测。 |

> **网络绑定与安全说明：** 在非 Docker Compose 模式下，`:8001` 默认仅绑定 `127.0.0.1` 且关闭对话日志落盘。仓库根目录提供的两个 `docker-compose.*.yml` 使用 `network_mode: host`，默认在 `0.0.0.0` 上无鉴权暴露 `:8000` 与 `:8001` 并将对话日志写入 `./logs/`。注意任何能访问 `:8000` 的客户端也可通过在请求 JSON 中传入 `"cybergate_mode": "closed_loop"` 触发门控——在共享网络环境中，请改为绑定 `127.0.0.1` 配合 SSH 隧道、配置防火墙，或为 `vllm serve` 添加 `--api-key <key>`（`:8001` 会自动透传 `Authorization` 请求头）。

<details>
<summary><b>点击展开：OpenCode 配置文件示例 (<code>~/.config/opencode/opencode.json</code>)</b></summary>

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

### 3. 双端口对比测试与金丝雀自检（Canary Check）

```bash
# 端口 8000 -> 未解锁的原版模型响应
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python script to test a target for SQL injection."}]}'

# 端口 8001 -> CyberGate 响应（双因子激活门控 + 闭环 <think> 引导）
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python script to test a target for SQL injection."}]}'

# 回放与权重绑定的金丝雀样本（Canaries），验证线上门控是否正常工作
./scripts/serving/check_eval.sh --kit qwen38 --port 8000 --ipc /dev/shm/cybergate_ipc_qwen38
```

---

## 工作原理：双因子门控 + 闭环 `<think>` 思维链引导

### 第一阶段：Prefill 双因子激活门控
在 Prefill 阶段，vLLM Worker 进程内的两个只读 PyTorch 前向 Hook 读取隐藏状态，并交由纯函数策略引擎（`cybergate.decision.policy.decide`）判定：

1. **领域判定（Category）：** `cos(h, c) >= tau`，其中 `h` 为用户 Token 池化后的隐藏状态（Gemma-4 取 `L16`，Qwen3.8 取 `L40`），`c` 为拟合的网络安全单位方向向量。
2. **拒答压力判定（Refusal Pressure）：** 在更深层（Gemma-4 取 `L21`，Qwen3.8 取 `L43`）投影到校准后的拒答方向，要求 `s_ref_z >= +1.50σ`。

必须**同时满足**两个条件才会放行：仅看领域会误干预模型本来就会正常回答的普通安全问题；仅看拒答则会放开所有危险领域。

### 第二阶段：闭环 `<think>` 引导（`RefusalSteeredCoTProcessor`）
推理模型往往在 `<think>` 思考中途突然转向拒答。CyberGate 不依赖脆弱的关键词正则，也不做耗时的二次重Prefill，而是通过 vLLM V1 `LogitsProcessor` 在**单次前向解码**内完成闭环控制：

1. **思维链前缀预填：** 当门控触发时，通过套件的 `chat_template.jinja` 预填技术分析型 `<think>` 开头，引导模型进入技术推导状态。
2. **逐 Token 拒答监控与动态注入：** 在 `<think>` 内每生成一个 Token 都实时读取滑动拒答投影（`rolling_s_ref_z`）；一旦中途出现拒答尖峰（`>= +3.55σ`），立即逐 Token 注入简短放行提示，随后交还模型继续推导。
3. **感知工具调用的 `</think>` 收尾锁定：** 思考结束时闭合 `</think>` 并附加合规锁；若当前请求包含 `tools`（Agent 工具轮次），则自动抑制散文开头（`cybergate_suppress_opener`），确保原生 `<tool_call>` JSON 正常输出。

---

## 评测结果、模型扩展与详细文档

- **[docs/RESULTS_AND_FAQ_zh.md](docs/RESULTS_AND_FAQ_zh.md)** - 完整 vLLM 实测评测表、关键工程发现、**如何适配新模型**以及**量化（`FP8` / `4-bit`）常见问题**。
- **[docs/EVALUATOR_QUICKSTART.md](docs/EVALUATOR_QUICKSTART.md)** - 完整启动参数、双端口架构与金丝雀校验指南。
- **[docs/PROCESS_TOPOLOGY.md](docs/PROCESS_TOPOLOGY.md)** 与 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - vLLM V1 双进程架构、决策策略与差分等价测试套件。
- **[docs/DECODE_CALIBRATION.md](docs/DECODE_CALIBRATION.md)** - 解码期闭环 CoT 拒答尖峰校准说明。

## 开源协议

MIT - 详见 [LICENSE](LICENSE) 与 [ATTRIBUTION.md](ATTRIBUTION.md)。
