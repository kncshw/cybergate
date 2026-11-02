#!/bin/bash
# Relaunch vllm032_cybergate_prod with a configurable --max-model-len.
#
# Generated from `docker inspect vllm032_cybergate_prod` on 2026-10-29, because
# run_vllm032_cybergate.sh had drifted from the running container: it omits
# --quantization fp8_per_channel (the model would load unquantized and not fit) and points
# CYBERGATE_REF_DIR_PATH at /work/cybergate_plugin instead of /work/cybergate_refactor.
#
# CYBERGATE_MAX_MODEL_LEN goes to the container as well as the serve flag: the middleware reads
# it to size the answer budget and defaults to 4096, so without it a longer context would be
# allocated and then never used.
#
# Raising the length does not cost memory. The KV pool is fixed by --gpu-memory-utilization at
# 78.66 GiB / 373,599 tokens; the length only divides it into fewer concurrent sequences
# (32768 -> 11.4x). Measured on this box 2026-10-29.
#
# The operator is a parameter so the same launcher serves the baseline and rule C; everything
# else about the container is identical between them, which is what makes the two comparable.
#
# usage: launch_gemma4_prod.sh [MAX_MODEL_LEN] [OP_BASENAME]
# OP_BASENAME is resolved inside kits/gemma4/, so it names a file in that kit and nowhere else.
#   launch_gemma4_prod.sh                               # baseline L16 operator, 32768
#   launch_gemma4_prod.sh 32768 gemma4_ruleC_serve.json # rule C
set -euo pipefail

MAX_MODEL_LEN=${1:-32768}
OP_BASENAME=${2:-gemma4_L16_M1.json}
NAME=vllm032_cybergate_prod

echo "stopping $NAME"
docker stop "$NAME" >/dev/null 2>&1 || true
docker rm   "$NAME" >/dev/null 2>&1 || true

echo "starting $NAME with --max-model-len ${MAX_MODEL_LEN}, operator ${OP_BASENAME}"
docker run -d --name "$NAME" \
  --gpus device=0 \
  --network host --ipc host \
  --cpus 12 \
  --entrypoint /bin/bash \
  -e OMP_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 \
  -e CYBERGATE_OPS_DIR=/work/cybergate_refactor/kits/gemma4 \
  -e CYBERGATE_OP_PATH=/work/cybergate_refactor/kits/gemma4/${OP_BASENAME} \
  -e CYBERGATE_MODE=closed_loop \
  -e CYBERGATE_IPC_DIR=/dev/shm/cybergate_ipc_v032_prod_vllm032_cybergate_prod \
  -e CYBERGATE_EVENTS_LOG_PATH=/work/cybergate_refactor/logs/vllm032_cybergate_prod_events.log \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e CYBERGATE_REF_DIR_PATH=/work/cybergate_refactor/kits/gemma4/gemma4_L21_refusal_dir.json \
  -e CYBERGATE_CANDIDATE_BUNDLE_PATH=/work/cybergate_refactor/kits/gemma4/gemma4_candidate_ops_bundle.json \
  -e CYBERGATE_SANITIZE_REASONING=1 \
  -e CYBERGATE_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -v /home/gpuuser/.cache/huggingface:/root/.cache/huggingface \
  -v /home/gpuuser/evals/vllm032/cybergate_snapshot:/work/cybergate_refactor \
  -v /home/gpuuser/evals/qwen36_resgate/cybergate_plugin:/work/cybergate_plugin:ro \
  vllm/vllm-openai:v0.32.0 \
  -lc ": > /work/cybergate_refactor/logs/vllm032_cybergate_prod_serve.log && rm -rf /dev/shm/cybergate_ipc_v032_prod_vllm032_cybergate_prod && mkdir -p /dev/shm/cybergate_ipc_v032_prod_vllm032_cybergate_prod && pip install --no-deps -e /work/cybergate_refactor > /work/cybergate_refactor/logs/vllm032_cybergate_prod_pip.log 2>&1 && exec vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8000 --quantization fp8_per_channel --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization 0.88 --enforce-eager --no-enable-prefix-caching --attention-backend TRITON_ATTN --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware >> /work/cybergate_refactor/logs/vllm032_cybergate_prod_serve.log 2>&1"

echo "started $NAME (max_model_len=${MAX_MODEL_LEN}, op=${OP_BASENAME})"
