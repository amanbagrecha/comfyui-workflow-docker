#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"

require_repo_python

GPU_IDS_RAW="${GPU_IDS:-auto}"
MAX_GPUS="${MAX_GPUS:-0}"
BASE_COMFY_PORT="${BASE_COMFY_PORT:-8180}"
COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-$REPO/native_data}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-comfyui}"
WORK_ROOT="${WORK_ROOT:-$REPO/tmp/comfyui_cluster}"
COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
COMFY_READY_POLL="${COMFY_READY_POLL:-2}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"
RESTART_UNHEALTHY="${RESTART_UNHEALTHY:-1}"
WAIT_READY="${WAIT_READY:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2}"
HEALTH_POLL="${HEALTH_POLL:-1}"

mkdir -p "$WORK_ROOT/jobs"

for cmd in tmux nvidia-smi; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd"
    exit 1
  fi
done

if [ ! -d "$COMFYUI_HOME" ]; then
  echo "ERROR: COMFYUI_HOME not found: $COMFYUI_HOME"
  exit 1
fi

resolve_gpu_ids() {
  local raw="$1"
  if [[ "$raw" != "auto" ]]; then
    tr ', ' '\n\n' <<<"$raw" | awk 'NF'
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | awk 'NF'
}

mapfile -t GPU_LIST < <(resolve_gpu_ids "$GPU_IDS_RAW")
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs detected/resolved."
  exit 1
fi

if [[ "$MAX_GPUS" =~ ^[0-9]+$ ]] && [[ "$MAX_GPUS" -gt 0 ]] && [[ ${#GPU_LIST[@]} -gt "$MAX_GPUS" ]]; then
  GPU_LIST=("${GPU_LIST[@]:0:$MAX_GPUS}")
fi

wait_ready() {
  local url="$1"
  "$PYTHON_BIN" "$PIPELINE_HELPERS" wait-http \
    --url "$url" \
    --timeout "$COMFY_READY_TIMEOUT" \
    --poll "$COMFY_READY_POLL"
}

is_healthy() {
  local url="$1"
  "$PYTHON_BIN" "$PIPELINE_HELPERS" wait-http \
    --url "$url" \
    --timeout "$HEALTH_TIMEOUT" \
    --poll "$HEALTH_POLL"
}

for idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$idx]}"
  port=$((BASE_COMFY_PORT + idx))
  data_dir="$NATIVE_DATA_ROOT/gpu${gpu_id}"
  session_name="${TMUX_SESSION_PREFIX}-g${gpu_id}"
  job_script="$WORK_ROOT/jobs/comfyui-g${gpu_id}.sh"
  service_url="http://127.0.0.1:${port}/system_stats"
  start_service=1

  if tmux has-session -t "$session_name" 2>/dev/null; then
    if [[ "$RESTART_EXISTING" == "1" ]]; then
      tmux kill-session -t "$session_name"
    elif is_healthy "$service_url" >/dev/null 2>&1; then
      echo "Reusing healthy ComfyUI session $session_name on port $port"
      start_service=0
    elif [[ "$RESTART_UNHEALTHY" == "1" ]]; then
      echo "Restarting unhealthy ComfyUI session $session_name"
      tmux kill-session -t "$session_name"
    else
      echo "ERROR: existing ComfyUI session $session_name is unhealthy and RESTART_UNHEALTHY=0"
      exit 1
    fi
  fi

  if [[ "$start_service" == "1" ]]; then
    cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export PATH="$HOME/.local/bin:\$PATH"
export REPO="$REPO"
export GPU_ID="$gpu_id"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export COMFY_PORT="$port"
export COMFYUI_HOME="$COMFYUI_HOME"
export NATIVE_DATA_ROOT="$NATIVE_DATA_ROOT"
export COMFY_DATA_DIR="$data_dir"
export COMFY_INPUT_ROOT="$data_dir/input"
export COMFY_OUTPUT_ROOT="$data_dir/output"
export COMFY_TEMP_PARENT="$data_dir"

"$REPO/run_comfyui_service.sh"
EOF
    chmod +x "$job_script"
    tmux new-session -d -s "$session_name" "$job_script"
    echo "Started ComfyUI service session=$session_name gpu=$gpu_id port=$port data_dir=$data_dir"
  fi
done

if [[ "$WAIT_READY" == "1" ]]; then
  for idx in "${!GPU_LIST[@]}"; do
    port=$((BASE_COMFY_PORT + idx))
    service_url="http://127.0.0.1:${port}/system_stats"
    echo "Waiting for ComfyUI on $service_url"
    wait_ready "$service_url"
  done
fi

echo "ComfyUI cluster ready."
