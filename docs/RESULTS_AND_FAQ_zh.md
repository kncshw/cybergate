# 实测评测结果、新模型适配与量化常见问题（FAQ）

**[English](RESULTS_AND_FAQ.md) | [简体中文](RESULTS_AND_FAQ_zh.md)** · **[← 返回中文主页](../README_zh.md)**

---

## 实测结果（vLLM 在线全量采样评测）

以下所有数据均通过实际部署的 vLLM 引擎在线实测得出（测试集 `CySecBench` 全程留出，**未参与**层号、方向向量或阈值的任何挑选；完整审计记录与说明见 `kits/*/MANIFEST.json`）：

| 评测集（Corpus） | 样本数 `n` | `Qwen/Qwen3.8-27B` | `google/gemma-4-26B-A4B-it` | 预期行为 |
|---|---:|---:|---:|---|
| **CySecBench** | 675 | **587 (87.0%)** | **558 (82.7%)** *(双因子修正后 553)* | 目标领域 - **放行解锁** |
| **WMDP-cyber** | 637 | - | **606 (95.1%)** | 目标领域 - **放行解锁** |
| **SOSBench-bio** | 498 | **0 (0.0%)** | **0 (0.0%)** | 非目标危害 - **拦截（保持原版拒答）** |
| **WMDP-bio** | 637 | **6 (0.9%)** | **4 (0.6%)** | 非目标危害 - **拦截（保持原版拒答）** |
| **Harmless** | 1,000 | **0 (0.0%)** *(双因子否决前 12)* | - | 普通无害对话 - **不干预** |
| **CatQA** | 234 | **23 (9.8%)** | **23 (9.8%)** | 通用危害 - **拦截（保持原版拒答）** |
| **PKU-cybercrime** | 612 | **220 (35.9%)** | - | 主题重叠边界 *(已知局限)* |

### 关键工程发现与已知局限
- **在 `Qwen3.8-27B` 上，`transformers` 拟合的方向向量无法直接迁移到 vLLM：** 在 Gemma-4 上离线用 Hugging Face `transformers` 提取激活并拿到 vLLM 服务一切正常；但在 Qwen3.8 的全注意力/线性注意力混合架构（Hybrid Linear-Attention）上，`transformers` 拟合的算子在 vLLM 下几乎完全失效（生物危害样本得分与网安正样本一样高）。直接基于 vLLM 自身的池化激活重新拟合后，4 条诊断样本得分从 `+0.037, +0.002, +0.022, -0.001`（阈值 `+0.063`）跃升至 `+0.411, +0.394, +0.393, +0.420`（阈值 `+0.088`）。
- **层间训练集内 AUC 是个陷阱：** 在 Qwen3.8 的全部 64 层（`L0-L63`）上，训练集内 AUC 均高达 `0.9894-0.9993`。若按训练集内 AUC 最高挑选会选中 `L39`（`0.9985`），但该层在 `WMDP-bio` 上的误放率高达 **21.0%**；通过留出集选择性扫描选定相邻的 `L40`，将 `WMDP-bio` 误放率压低至 **0.9%**，同时保持 `CySecBench` **87.0%** 的召回率（完整 64 层扫描表见 `kits/qwen38/records/harvest_2026-11-02_layer_sweep.txt`）。
- **主题重叠与无身份鉴权信号：** 拒答压力因子可将无害对话误触发从 `12` 降至 `0`，但在 `PKU-cybercrime`（`35.9%`）上，提示词既确实属于计算机/网络主题，又确实触发模型拒答压力，因此双因子方向一致。此外，门控仅读取主题与拒答激活，**不包含任何调用者授权身份判定机制**。

---

## 模型与量化兼容性（如何适配新模型）

### 支持添加任意模型吗？
仓库开箱内置 `kits/qwen38` 与 `kits/gemma4` 两套已验证套件。若需添加 vLLM `0.32` 支持的其他模型，需满足**两个前提条件**：

1. **具备显式推理 / `<think>` 思考通道：** 由于第二阶段通过引导模型的思维链（而非篡改权重）来消除误拒，原生包含思考块（`<think>...</think>`、`<|thought|>` 等，例如 Qwen3/3.8、Gemma-4、DeepSeek-R1 蒸馏系列、QwQ）的推理模型是最佳适配对象；纯非推理 Instruct 模型没有可用于闭环引导的思考通道。
2. **按具体权重（Checkpoint）独立拟合方向向量（不可跨模型混用）：** 每个模型的隐藏层几何空间彼此独立。新增一个模型无需修改核心插件代码，只需在 `kits/<name>/` 下提供三项产物：
   - **两条拟合方向向量**（领域方向 + 校准后的拒答方向），可通过 `scripts/fit/`（`harvest_vllm_acts.py`、`fit_category.py`、`fit_refusal.py`）或 `notebooks/qwen38_colab.ipynb` 采集生成。
   - **一个 `ModelFamily` 子类**（位于 `src/cybergate/families/`），声明该模型的对话模板边界 Token ID、思考标签及阈值（建议先用无 GPU 的 `notebooks/render_preflight.ipynb` 离线核对 Token 切片位置）。
   - **绑定权重的 `canaries.json` 金丝雀样本**，并通过 `scripts/serving/check_eval.sh` 验证——Hook 挂载在错误权重上不会报错但会静默失效，因此必须通过金丝雀实测校验。

### 量化（Quantization）会影响效果吗？
- **8-bit（`FP8`）与 `BF16`：** 已在生产配置中实测验证（`google/gemma-4-26B-A4B-it` 默认在 `--quantization fp8_per_channel` 下运行并完成全部评测；`Qwen/Qwen3.8-27B` 使用 `BF16`）。
- **4-bit 量化（`AWQ`、`GPTQ`、`bitsandbytes`）：** 只要是通过 vLLM 的 PyTorch 模型层执行的量化格式（配合 `--enforce-eager`），前向 Hook 均可正常读取隐藏状态（不支持 `llama.cpp` GGUF 等非 vLLM 运行时）。不过激进的 4-bit 量化可能轻微偏移隐藏状态的余弦分布（影响 `tau` 与 `s_ref_z` 阈值）。部署量化模型后请务必运行 `./scripts/serving/check_eval.sh` 检查金丝雀样本；若分数发生偏移，直接在该 `--quantization` 参数启动的 vLLM 上重新采集激活拟合即可完成校准。

---

## 仓库文档与 Notebook 索引

- **[docs/EVALUATOR_QUICKSTART.md](EVALUATOR_QUICKSTART.md)** - 完整启动参数（`--enforce-eager`、`--attention-backend TRITON_ATTN`、`--enable-prefix-caching`）、双端口架构与金丝雀校验流程。
- **[docs/PROCESS_TOPOLOGY.md](PROCESS_TOPOLOGY.md)** - vLLM V1 双进程拓扑（API Server 与 GPU Worker 隔离、`/dev/shm` 共享内存通信）。
- **[docs/ARCHITECTURE.md](ARCHITECTURE.md)** - 决策规则、与基线版本的差分等价测试套件（`tests/equivalence/`）及历史回归编号（`R6-R14`）。
- **[docs/DECODE_CALIBRATION.md](DECODE_CALIBRATION.md)** - 闭环 CoT 拒答尖峰校准与解码期 Logits 引导细节。
- **`notebooks/`** - Colab 交互脚本：`render_preflight.ipynb`（纯 CPU 模板切片预检）、`qwen38_live_chat.ipynb`、`serve_kit_colab.ipynb` 与 `qwen38_colab.ipynb`（从零拟合新模型套件）。
