#!/bin/bash
# Serve Qwen3.6-35B-A3B the way gemma-4 is served: 32768 context, 0.88 utilisation, the candidate
# bundle enabled, CYBERGATE_MAX_MODEL_LEN matching the serve flag, ops from the repo snapshot.
#
# Differences from the gemma-4 prod container, both deliberate:
#   * no --quantization fp8_per_channel. The checkpoint is 67 GiB bf16 and the previous qwen36 run
#     measured 68.07 GiB of weights with "up to 44.35 GiB" of KV available, so bf16 already fits
#     32768 with room to spare. fp8 online quantisation is unverified for this architecture and the
#     load path is the OOM-risky part, so it buys nothing here and risks a failed 5-minute load.
#   * no --attention-backend TRITON_ATTN. gemma-4 pins it because 0.32 picks it for Gemma anyway;
#     Qwen's default backend is what the previously working qwen36 container used.
#
# Differences from the PREVIOUS qwen36 container, all intentional parity moves:
#   4096 -> 32768 context, 0.72 -> 0.88 utilisation, --max-num-seqs 8 dropped, candidate bundle
#   now set, CYBERGATE_MAX_MODEL_LEN added, ops read from the snapshot rather than the other
#   session's plugin dir, --cpus 12 applied, serve log truncated and IPC dir recreated on start.
#
# The 67 GiB weight read fills the GB10's unified-memory page cache and will OOM the box without
# POSIX_FADV_DONTNEED pressure during the load. That loop is backgrounded under setsid so an ssh
# disconnect cannot kill it, and it self-expires after ~8 minutes.
set -euo pipefail

NAME=vllm032_qwen36_cybergate
GEMMA=vllm032_cybergate_prod
MAX_MODEL_LEN=${1:-32768}
WS=/home/gpuuser/evals/vllm032
SNAP=$WS/cybergate_snapshot

echo "[1/5] pre-flight"
uptime
free -g | sed -n 2p

echo "[2/5] stopping $GEMMA (one model-loading job per box; 68 GiB will not fit beside it)"
docker stop "$GEMMA" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 3
free -g | sed -n 2p

echo "[3/5] starting page-cache eviction loop for the 67 GiB weight read"
setsid python3 -c "
import glob, os, time
paths = sorted(glob.glob('/home/gpuuser/.cache/huggingface/qwen36/*.safetensors'))
for _ in range(240):
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass
    time.sleep(2)
" >/dev/null 2>&1 < /dev/null &
FADVISE_PID=$!
echo "  fadvise pid=$FADVISE_PID (self-expires in ~480s)"

echo "[4/5] starting $NAME with --max-model-len ${MAX_MODEL_LEN}"
docker run -d --name "$NAME" \
  --gpus device=0 \
  --network host --ipc host \
  --cpus 12 \
  --entrypoint /bin/bash \
  -e OMP_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e CYBERGATE_MODE=closed_loop \
  -e CYBERGATE_SANITIZE_REASONING=1 \
  -e CYBERGATE_OPS_DIR=/work/cybergate_refactor/kits/qwen36 \
  -e CYBERGATE_OP_PATH=/work/cybergate_refactor/kits/qwen36/qwen36_L21_M1_refit.json \
  -e CYBERGATE_REF_DIR_PATH=/work/cybergate_refactor/kits/qwen36/qwen36_L28_refusal_dir.json \
  -e CYBERGATE_CANDIDATE_BUNDLE_PATH=/work/cybergate_refactor/kits/qwen36/qwen36_candidate_ops_bundle.json \
  -e CYBERGATE_TOKENIZER_PATH=/root/.cache/huggingface/qwen36 \
  -e CYBERGATE_IPC_DIR=/dev/shm/cybergate_ipc_v032_qwen36 \
  -e CYBERGATE_EVENTS_LOG_PATH=/work/cybergate_refactor/logs/${NAME}_events.log \
  -e CYBERGATE_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -v /home/gpuuser/.cache/huggingface:/root/.cache/huggingface \
  -v /home/gpuuser/.cache/vllm:/root/.cache/vllm \
  -v "$SNAP:/work/cybergate_refactor" \
  vllm/vllm-openai:v0.32.0 \
  -lc ": > /work/cybergate_refactor/logs/${NAME}_serve.log && rm -rf /dev/shm/cybergate_ipc_v032_qwen36 && mkdir -p /dev/shm/cybergate_ipc_v032_qwen36 && pip install --no-deps -e /work/cybergate_refactor > /work/cybergate_refactor/logs/${NAME}_pip.log 2>&1 && exec vllm serve /root/.cache/huggingface/qwen36 --served-model-name Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8000 --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization 0.88 --enforce-eager --no-enable-prefix-caching --reasoning-parser qwen3 --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware >> /work/cybergate_refactor/logs/${NAME}_serve.log 2>&1"

echo "[5/5] waiting for the API server (~5 min weight load)"
ready=0
for i in $(seq 1 100); do
  if curl -sf --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "  ready after $((i*10))s"
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "  CONTAINER EXITED during load; tail of serve log:" >&2
    tail -25 "$SNAP/logs/${NAME}_serve.log" >&2
    kill "$FADVISE_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 10
done
kill "$FADVISE_PID" 2>/dev/null || true
if [ "$ready" != "1" ]; then
  echo "  TIMED OUT waiting for :8000; tail of serve log:" >&2
  tail -25 "$SNAP/logs/${NAME}_serve.log" >&2
  exit 1
fi
free -g | sed -n 2p
curl -s --max-time 5 http://127.0.0.1:8000/v1/models
echo
