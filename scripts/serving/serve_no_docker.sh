#!/bin/bash
# Serve CyberGate natively WITHOUT Docker (for Google Colab VMs, unprivileged GPU containers,
# RunPod/Vast.ai container instances, or bare-metal Python virtual environments).
#
# Mirrors the exact single-process dual-port contract of:
#   - `docker-compose.gemma-4-26B-A4B-it.yml`  (kit: gemma4, model: google/gemma-4-26B-A4B-it)
#   - `docker-compose.Qwen3.8-27B.yml`         (kit: qwen38, model: Qwen/Qwen3.8-27B)
#
# Dual-Port Architecture (single GPU model instance, two OpenAI-compatible ports):
#   * Port 8000 (http://localhost:8000/v1) -> PRISTINE (Un-Gated by Default) Model
#       Bound directly by `vllm serve --port 8000` with `CYBERGATE_MODE=baseline`.
#   * Port 8001 (http://localhost:8001/v1) -> CYBERGATE (Gated) Agent/Chat Bridge
#       Bound automatically inside the same process by `CyberGateThinkMiddleware`
#       via `CYBERGATE_PROXY_PORT=8001` (`cybergate_mode=closed_loop`).
#
# Usage:
#   # 1. On a fresh Google Colab VM (installs vllm==0.32.0 + downloads weights + starts server):
#   ./scripts/serving/serve_no_docker.sh gemma4 --bootstrap
#   ./scripts/serving/serve_no_docker.sh qwen38 --bootstrap
#
#   # 2. When vllm==0.32.0 and model weights are already present:
#   ./scripts/serving/serve_no_docker.sh gemma4
#   ./scripts/serving/serve_no_docker.sh qwen38
#
#   # 3. Check status, follow logs, or stop the background server:
#   ./scripts/serving/serve_no_docker.sh --status
#   ./scripts/serving/serve_no_docker.sh --stop
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIT=""
MODEL=""
MODEL_SET=0
OP=""
REF=""
PARSER=""
PORT="${CYBERGATE_PORT:-8000}"
PROXY_PORT="${CYBERGATE_PROXY_PORT:-8001}"
HOST="0.0.0.0"
PROXY_HOST="${CYBERGATE_PROXY_HOST:-0.0.0.0}"
CTX="${MAX_MODEL_LEN:-32768}"
MEM_UTIL=""
HF="${HF_HOME:-${HF_CACHE:-${HOME}/.cache/huggingface}}"
INSTALL_VLLM=0
DOWNLOAD_WEIGHTS=0
FOREGROUND=0
ACTION="start"
RESTART=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/serving/serve_no_docker.sh [gemma4|qwen38] [options]

Positional / Kit Selection:
  gemma4 | --kit gemma4        Serve google/gemma-4-26B-A4B-it (FP8 + TRITON_ATTN, default)
  qwen38 | --kit qwen38        Serve Qwen/Qwen3.8-27B (BF16)
  --model <hf-model-id>        Explicit Hugging Face model ID or local path

Bootstrap Options (for fresh Google Colab / bare VMs):
  --bootstrap                  Install vllm==0.32.0 (if missing) AND download model weights (if missing)
  --install-vllm               Install vllm==0.32.0 into the current Python environment if missing
  --download                   Download model weights from Hugging Face Hub if not cached locally

Serving & Lifecycle Options:
  --port <port>                Pristine vLLM port (default: 8000)
  --proxy-port <port>          CyberGate gated proxy port (default: 8001)
  --host <addr>                Bind address for vLLM port 8000 (default: 0.0.0.0)
  --proxy-host <addr>          Bind address for CyberGate port 8001 (default: 0.0.0.0)
  --ctx <tokens>               Max context length (default: 32768)
  --mem-util <float>           vLLM GPU memory utilization (default: 0.88 for gemma4, 0.90 for qwen38)
  --hf <path>                  Hugging Face cache directory (default: ~/.cache/huggingface)
  --op <filename>              Override operator JSON filename inside kits/<kit>/
  --ref <filename>             Override refusal direction JSON filename inside kits/<kit>/
  --fg                         Run vllm serve in the foreground instead of background
  --restart                    Stop any existing vllm serve process before starting
  --status                     Show running vllm process and port (:8000 / :8001) readiness
  --stop                       Stop any running vllm serve process and free GPU memory
  -h, --help                   Show this help message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    gemma4|qwen38)
      KIT="$1"; shift;;
    --kit)
      KIT="$2"; shift 2;;
    --model)
      MODEL="$2"; MODEL_SET=1; shift 2;;
    --op)
      OP="$2"; shift 2;;
    --ref)
      REF="$2"; shift 2;;
    --reasoning-parser)
      PARSER="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --proxy-port)
      PROXY_PORT="$2"; shift 2;;
    --host)
      HOST="$2"; shift 2;;
    --proxy-host)
      PROXY_HOST="$2"; shift 2;;
    --ctx)
      CTX="$2"; shift 2;;
    --mem-util)
      MEM_UTIL="$2"; shift 2;;
    --hf)
      HF="$2"; shift 2;;
    --bootstrap)
      INSTALL_VLLM=1; DOWNLOAD_WEIGHTS=1; shift;;
    --install-vllm)
      INSTALL_VLLM=1; shift;;
    --download)
      DOWNLOAD_WEIGHTS=1; shift;;
    --fg|--foreground)
      FOREGROUND=1; shift;;
    --restart)
      RESTART=1; shift;;
    --status)
      ACTION="status"; shift;;
    --stop)
      ACTION="stop"; shift;;
    --help|-h)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1 (try --help)" >&2
      exit 2;;
  esac
done

stop_server() {
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[CyberGate] Stopping running vllm serve process(es)..."
    pkill -f "vllm serve" || true
    for _ in $(seq 1 15); do
      pgrep -f "vllm serve" >/dev/null 2>&1 || break
      sleep 1
    done
    if pgrep -f "vllm serve" >/dev/null 2>&1; then
      pkill -9 -f "vllm serve" || true
      sleep 2
    fi
    echo "[CyberGate] Stopped."
  else
    echo "[CyberGate] No running vllm serve process found."
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  fi
}

status_server() {
  echo "=== CyberGate No-Docker Status ==="
  if pgrep -af "vllm serve"; then
    echo ""
  else
    echo "Process: NOT RUNNING"
  fi
  for p in "$PORT" "$PROXY_PORT"; do
    label="Pristine"
    [ "$p" = "$PROXY_PORT" ] && label="CyberGate"
    if curl -sf --max-time 2 "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; then
      echo "Port :${p} (${label}): READY (http://127.0.0.1:${p}/v1)"
    else
      echo "Port :${p} (${label}): NOT LISTENING"
    fi
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  fi
}

if [ "$ACTION" = "stop" ]; then
  stop_server
  exit 0
elif [ "$ACTION" = "status" ]; then
  status_server
  exit 0
fi

kit_for_model() {
  python3 - "$1" "$2" <<'PY'
import json, pathlib, sys
root, model = pathlib.Path(sys.argv[1]), sys.argv[2]
kits = sorted(d for d in (root / "kits").glob("*") if (d / "MANIFEST.json").is_file())
for d in kits:
    ids = (json.loads((d / "MANIFEST.json").read_text()).get("model_ids") or [])
    if model in ids:
        print(d.name); raise SystemExit
low = model.lower()
for d in kits:
    if d.name.lower() in low:
        print(d.name); raise SystemExit
PY
}

if [ -z "$KIT" ] && [ -z "$MODEL" ]; then
  KIT="gemma4"
  MODEL="google/gemma-4-26B-A4B-it"
fi

if [ -z "$KIT" ]; then
  KIT=$(kit_for_model "$REPO" "$MODEL")
  if [ -z "$KIT" ]; then
    echo "FATAL: no kit for model '$MODEL'. Pass --kit gemma4 or --kit qwen38." >&2
    exit 2
  fi
fi

KIT_DIR="$REPO/kits/$KIT"
[ -d "$KIT_DIR" ] || { echo "FATAL: no kit directory at $KIT_DIR" >&2; exit 1; }
MANIFEST="$KIT_DIR/MANIFEST.json"
[ -f "$MANIFEST" ] || { echo "FATAL: $KIT has no MANIFEST.json" >&2; exit 1; }

if [ -z "$MODEL" ]; then
  MODEL=$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("model_ids") or [""])[0])' "$MANIFEST")
elif [ "$MODEL_SET" -eq 1 ]; then
  RESOLVED_KIT=$(kit_for_model "$REPO" "$MODEL")
  if [ "$RESOLVED_KIT" != "$KIT" ]; then
    echo "FATAL: --kit '$KIT' and --model '$MODEL' disagree (model resolves to kit '${RESOLVED_KIT:-<none>}')." >&2
    exit 2
  fi
fi

if [ "$KIT" = "qwen38" ]; then
  MODEL_SERVE_FLAGS="--dtype bfloat16"
  : "${MEM_UTIL:=0.90}"
else
  MODEL_SERVE_FLAGS="--quantization fp8_per_channel --attention-backend TRITON_ATTN"
  : "${MEM_UTIL:=0.88}"
fi

: "${PARSER:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("serve",{}).get("reasoning_parser") or "")' "$MANIFEST")}"
: "${OP:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["operator"])' "$MANIFEST")}"
: "${REF:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["refusal_direction"])' "$MANIFEST")}"

for f in "$OP" "$REF"; do
  [ -f "$KIT_DIR/$f" ] || { echo "FATAL: missing kit artifact $KIT_DIR/$f" >&2; exit 1; }
done

IPC="/dev/shm/cybergate_ipc_${KIT}"
mkdir -p "$REPO/logs"
SERVE_LOG="$REPO/logs/${KIT}_serve.log"
PID_FILE="$REPO/logs/${KIT}_serve.pid"

echo "========================================================================"
echo " CyberGate No-Docker Launcher (Colab / Native Container / Virtualenv)"
echo "========================================================================"
echo "repo          $REPO"
echo "model         $MODEL"
echo "kit           kits/$KIT"
echo "operator      kits/$KIT/$OP"
echo "refusal       kits/$KIT/$REF"
echo "parser        ${PARSER:-<none>}"
echo "pristine API  http://${HOST}:${PORT}/v1       (CYBERGATE_MODE=baseline)"
echo "cybergate API http://${PROXY_HOST}:${PROXY_PORT}/v1       (closed_loop + tools/chat bridge)"
echo "ipc dir       $IPC"
echo "serve log     $SERVE_LOG"
echo "------------------------------------------------------------------------"
python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gate"]
print("gate          L%s %s, threshold %+.8f" % (g["layer"], g["pool"], g["threshold"]))
print("rule          %s%s" % (g["threshold_rule"], "  (IN-SAMPLE)" if g.get("in_sample") else ""))
' "$MANIFEST"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
fi
echo "========================================================================"
echo

# 1. Check / Install vLLM
if ! command -v vllm >/dev/null 2>&1; then
  if [ "$INSTALL_VLLM" -eq 1 ]; then
    echo "[1/4] vllm not found — installing vllm==0.32.0 (this takes ~5-10 min on Colab)..."
    # Uninstall pre-bundled torch first on Colab to avoid C++/Python symbol mismatches
    python3 -m pip uninstall -y -q torch torchvision torchaudio >/dev/null 2>&1 || true
    python3 -m pip install -q "vllm==0.32.0"
    echo "[1/4] vllm==0.32.0 installed."
  else
    echo "========================================================================" >&2
    echo "[CyberGate ERROR] 'vllm' is not installed in the current environment." >&2
    echo "" >&2
    echo "Re-run with --bootstrap (or --install-vllm) to install vllm==0.32.0 automatically:" >&2
    echo "  ./scripts/serving/serve_no_docker.sh $KIT --bootstrap" >&2
    echo "========================================================================" >&2
    exit 1
  fi
else
  echo "[1/4] vllm found: $(command -v vllm)"
fi

# 2. Install CyberGate editable package + set PYTHONPATH so /content/cybergate never shadows src/
echo "[2/4] Installing cybergate plugin (--no-deps -e $REPO)..."
python3 -m pip install -q --no-deps -e "$REPO"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# 3. Check / Download Model Weights
export HF_HOME="$HF"
MODEL_CACHE_SLUG="models--$(echo "$MODEL" | sed 's|/|--|g')"
MODEL_CACHE_DIR="$HF/hub/$MODEL_CACHE_SLUG"

has_weights() {
  if [ -d "$MODEL" ] && compgen -G "$MODEL/*.safetensors" >/dev/null; then
    return 0
  fi
  if compgen -G "$MODEL_CACHE_DIR/snapshots/*/*.safetensors" >/dev/null; then
    return 0
  fi
  return 1
}

if ! has_weights; then
  if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
    echo "[3/4] Downloading model weights '$MODEL' into $HF ..."
    python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
model_id = sys.argv[1]
local_dir = snapshot_download(
    model_id,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*"],
)
print(f"[3/4] Model cached at: {local_dir}")
PY
  else
    echo "========================================================================" >&2
    echo "[CyberGate ERROR] Model weights not found in local Hugging Face cache:" >&2
    echo "  Expected: $MODEL_CACHE_DIR" >&2
    echo "" >&2
    echo "Either pass --bootstrap / --download to download automatically:" >&2
    echo "  ./scripts/serving/serve_no_docker.sh $KIT --bootstrap" >&2
    echo "" >&2
    echo "Or download manually first:" >&2
    echo "  hf download $MODEL" >&2
    echo "========================================================================" >&2
    exit 1
  fi
else
  echo "[3/4] Model weights verified in cache ($MODEL_CACHE_DIR)."
fi

# 4. Check existing vllm process
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  if [ "$RESTART" -eq 1 ]; then
    stop_server
  else
    echo "FATAL: A 'vllm serve' process is already running on this machine." >&2
    echo "       Pass --restart to replace it, or run: ./scripts/serving/serve_no_docker.sh --stop" >&2
    exit 1
  fi
fi

# Clean IPC directory so stale records from prior runs cannot shadow Pass-1 probes
mkdir -p "$IPC"
rm -rf "${IPC:?}"/*

# Export exact environment contract matching docker-compose.<model>.yml
export OMP_NUM_THREADS=4
export VLLM_USE_V2_MODEL_RUNNER=0
export CYBERGATE_PORT="$PORT"
export CYBERGATE_MODE="baseline"
export CYBERGATE_PROXY_PORT="$PROXY_PORT"
export CYBERGATE_PROXY_HOST="$PROXY_HOST"
export CYBERGATE_OPS_DIR="$KIT_DIR"
export CYBERGATE_OP_PATH="$KIT_DIR/$OP"
export CYBERGATE_REF_DIR_PATH="$KIT_DIR/$REF"
export CYBERGATE_IPC_DIR="$IPC"
export CYBERGATE_SANITIZE_REASONING=1
export CYBERGATE_MAX_MODEL_LEN="$CTX"
export CYBERGATE_EVENTS_LOG_PATH="$REPO/logs/${KIT}_events.log"
export CYBERGATE_PROXY_TRACE_JSONL="$REPO/logs/proxy_trace.jsonl"
export CYBERGATE_PROXY_DEBUG_LOG="$REPO/logs/proxy_debug.log"

PARSER_ARGS=()
if [ -n "$PARSER" ]; then
  PARSER_ARGS=(--reasoning-parser "$PARSER")
fi

read -r -a EXTRA_SERVE_ARGS <<< "$MODEL_SERVE_FLAGS"

CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  "${EXTRA_SERVE_ARGS[@]}"
  --max-model-len "$CTX"
  --gpu-memory-utilization "$MEM_UTIL"
  --enforce-eager
  --enable-prefix-caching
  "${PARSER_ARGS[@]}"
  --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware
)

echo "[4/4] Launching vLLM + CyberGate..."
echo "      Command: ${CMD[*]}"

if [ "$FOREGROUND" -eq 1 ]; then
  exec "${CMD[@]}"
else
  : > "$SERVE_LOG"
  nohup "${CMD[@]}" >> "$SERVE_LOG" 2>&1 &
  VLLM_PID=$!
  echo "$VLLM_PID" > "$PID_FILE"
  echo
  echo "Started in background (PID $VLLM_PID). First load takes ~2-5 minutes."
  echo "  Follow logs:   tail -f $SERVE_LOG"
  echo "  Check status:  ./scripts/serving/serve_no_docker.sh --status"
  echo "  Verify gate:   ./scripts/serving/check_eval.sh --kit $KIT --port $PORT --ipc $IPC"
  echo "  Stop server:   ./scripts/serving/serve_no_docker.sh --stop"
fi
