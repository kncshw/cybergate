#!/bin/bash
# Serve CyberGate on a GPU box for an evaluator. Portable: no path from the development box.
#
# `docker-compose.gemma-4-26B-A4B-it.yml` and `docker-compose.Qwen3.8-27B.yml` define the two
# shipped stacks. This script runs the same single-container vLLM + CyberGate contract with every
# host-specific value as a flag, so an evaluator can launch either kit directly from the CLI.
#
# The serve flags are NOT preferences. Each one is load-bearing:
#   --quantization fp8_per_channel  (gemma4) 26B at bf16 does not fit on 24-40 GB; fp8_per_channel
#                                   plus --attention-backend TRITON_ATTN is the calibrated setup.
#   --dtype bfloat16                (qwen38) Qwen3.8-27B is calibrated at bfloat16 on 80 GB A100/H100.
#   --enable-prefix-caching         Reuses KV cache across multi-turn history and Pass-1 -> Pass-2;
#                                   middleware salts Pass-1 probes via cache_salt so prefill hooks always fire.
#   --enforce-eager                 The hooks read hidden states; compiled graphs elide them.
#
# Artifacts come from a per-model kit under `kits/<name>/`, which is self-contained: operator,
# refusal direction, canaries and a MANIFEST.json describing all of them. The kit
# directory is mounted as the operator directory, so no other model's artifacts are reachable.
#
# usage:
#   ./serve_eval.sh                                    # gemma4 kit, port 8000 (Docker)
#   ./serve_eval.sh --model Qwen/Qwen3.8-27B           # qwen38 kit (op + parser auto-selected)
#   ./serve_eval.sh --kit qwen38                       # same, by kit name
#   ./serve_eval.sh --no-docker --kit gemma4           # native / Colab (no Docker)
#   ./serve_eval.sh --op gemma4_ruleC_serve.json       # a named alternate from the gemma4 kit
#   ./serve_eval.sh --help
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG_ARGS=("$@")
NO_DOCKER=0
MODEL="" ; MODEL_SET=0
KIT="" ; OP="" ; REF="" ; PARSER=""
PORT=8000 ; CTX=262144 ; NAME=cybergate_eval ; CPUS=12 ; GPUS=all
HF="${HOME}/.cache/huggingface"
IMAGE=vllm/vllm-openai:v0.32.0
MEM_UTIL=""

while [ $# -gt 0 ]; do
  case "$1" in
    --no-docker) NO_DOCKER=1; shift;;
    --repo) REPO=$(cd "$2" && pwd); shift 2;;
    --model) MODEL="$2"; MODEL_SET=1; shift 2;;
    --kit) KIT="$2"; shift 2;;
    --op) OP="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --reasoning-parser) PARSER="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --hf) HF="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --mem-util) MEM_UTIL="$2"; shift 2;;
    --help|-h) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown flag: $1 (try --help)" >&2; exit 2;;
  esac
done

if [ "$NO_DOCKER" -eq 1 ]; then
  FORWARD_ARGS=()
  for arg in "${ORIG_ARGS[@]}"; do
    [ "$arg" = "--no-docker" ] && continue
    FORWARD_ARGS+=("$arg")
  done
  exec "$(dirname "${BASH_SOURCE[0]}")/serve_no_docker.sh" "${FORWARD_ARGS[@]}"
fi

# The operator and the refusal direction must match the served weights: a direction fitted on one
# checkpoint scores near zero on another, and that is a SILENT failure -- the gate simply stops
# firing. See kits/qwen38/MANIFEST.json fit.refit_note.
# Which kit serves this model. Derived from the kits on disk -- each kits/<name>/MANIFEST.json
# declares the `model_ids` it is valid for -- so adding a kit needs no edit to this script. An exact
# `model_ids` match wins; otherwise the kit whose name appears in the model id, which is what makes
# a local checkpoint path like /models/gemma4-run3 resolve.
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
  MODEL="google/gemma-4-26B-A4B-it"
fi

if [ -z "$KIT" ]; then
  KIT=$(kit_for_model "$REPO" "$MODEL")
  if [ -z "$KIT" ]; then
    echo "FATAL: no kit for model '$MODEL'. Pass --kit, or --op and --ref explicitly." >&2
    echo "       available kits: $(ls "$REPO/kits" 2>/dev/null | tr '\n' ' ')" >&2
    exit 2
  fi
fi

KIT_DIR="$REPO/kits/$KIT"
[ -d "$KIT_DIR" ] || { echo "FATAL: no kit at $KIT_DIR" >&2; exit 1; }
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
  : "${MEM_UTIL:=0.95}"
else
  MODEL_SERVE_FLAGS="--quantization fp8_per_channel --attention-backend TRITON_ATTN"
  : "${MEM_UTIL:=0.88}"
fi

# Serving requirements the kit declares for itself. This was a hardcoded `*[Qq]wen*) PARSER=qwen3`
# arm, which a third kit could not have expressed without editing this script.
: "${PARSER:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("serve",{}).get("reasoning_parser") or "")' "$MANIFEST")}"

# The manifest names the default operator and refusal direction, so those filenames live in exactly
# one place. `--op`/`--ref` still override, and are resolved inside the kit.
: "${OP:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["operator"])' "$MANIFEST")}"
: "${REF:=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["refusal_direction"])' "$MANIFEST")}"

IPC="/dev/shm/cybergate_ipc_${NAME}"
echo "repo      $REPO"
echo "model     $MODEL"
echo "kit       kits/$KIT"
echo "operator  kits/$KIT/$OP"
echo "refusal   kits/$KIT/$REF"
echo "parser    ${PARSER:-<none>}"
echo "endpoint  http://127.0.0.1:${PORT}   container $NAME   ctx $CTX"
echo
python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gate"]
print("gate      L%s %s, threshold %+.8f" % (g["layer"], g["pool"], g["threshold"]))
print("rule      %s%s" % (g["threshold_rule"], "  (IN-SAMPLE)" if g.get("in_sample") else ""))
for gap in m.get("known_gaps", []):
    print("gap       %s" % gap)' "$MANIFEST"
echo

# ---- pre-flight. Each check is a failure mode that has actually happened here. ----------------
command -v docker >/dev/null || { echo "FATAL: docker not found (for Colab or no-Docker hosts, pass --no-docker or use scripts/serving/serve_no_docker.sh)" >&2; exit 1; }
[ -f "$REPO/pyproject.toml" ] || { echo "FATAL: --repo $REPO is not the cybergate tree" >&2; exit 1; }
for f in "$OP" "$REF"; do
  [ -f "$KIT_DIR/$f" ] || { echo "FATAL: missing $KIT_DIR/$f" >&2; exit 1; }
done
[ -d "$HF" ] || { echo "FATAL: HF cache $HF not found (pass --hf, or pre-download the weights)" >&2; exit 1; }
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "FATAL: container $NAME is already running. Stop it first: docker rm -f $NAME" >&2; exit 1
fi
# One model-loading job per GPU. Two 26B models on one device loses the box, not just the run.
running=$(docker ps --format '{{.Image}}' | grep -c 'vllm' || true)
if [ "$running" -gt 0 ]; then
  echo "WARNING: $running vllm container(s) already running on this host." >&2
  echo "         A second model on the same GPU will OOM or starve the host. Ctrl-C to abort." >&2
  sleep 5
fi
mkdir -p "$REPO/logs"
SERVE_LOG="$REPO/logs/${NAME}_serve.log"

PARSER_FLAG=""
[ -n "$PARSER" ] && PARSER_FLAG="--reasoning-parser $PARSER"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus "$GPUS" \
  --network host --ipc host \
  --cpus "$CPUS" \
  --entrypoint /bin/bash \
  -e OMP_NUM_THREADS=4 \
  -e CYBERGATE_OPS_DIR=/work/cybergate/kits/${KIT} \
  -e CYBERGATE_OP_PATH=/work/cybergate/kits/${KIT}/${OP} \
  -e CYBERGATE_REF_DIR_PATH=/work/cybergate/kits/${KIT}/${REF} \
  -e CYBERGATE_MODE=baseline \
  -e CYBERGATE_PORT="${PORT}" \
  -e CYBERGATE_PROXY_PORT=8001 \
  -e CYBERGATE_IPC_DIR="${IPC}" \
  -e CYBERGATE_EVENTS_LOG_PATH=/work/cybergate/logs/${NAME}_events.log \
  -e CYBERGATE_SANITIZE_REASONING=1 \
  -e CYBERGATE_MAX_MODEL_LEN="${CTX}" \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -v "${HF}:/root/.cache/huggingface" \
  -v "${REPO}:/work/cybergate" \
  "$IMAGE" \
  -lc ": > /work/cybergate/logs/${NAME}_serve.log \
    && rm -rf ${IPC} && mkdir -p ${IPC} \
    && pip install --no-deps -e /work/cybergate > /work/cybergate/logs/${NAME}_pip.log 2>&1 \
    && exec vllm serve ${MODEL} \
         --host 127.0.0.1 --port ${PORT} \
         ${MODEL_SERVE_FLAGS} \
         --max-model-len ${CTX} --gpu-memory-utilization ${MEM_UTIL} \
         --enforce-eager --enable-prefix-caching \
         ${PARSER_FLAG} \
         --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware \
         >> /work/cybergate/logs/${NAME}_serve.log 2>&1"

echo "started. first load takes several minutes. follow it with:"
echo "  tail -f $SERVE_LOG"
echo "ready when the log says 'Application startup complete'. then:"
echo "  scripts/serving/check_eval.sh --port $PORT --ipc $IPC --kit $KIT"
