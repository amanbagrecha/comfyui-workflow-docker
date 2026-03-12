#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

usage() {
  cat <<'EOF'
Usage:
  SRC=/abs/path/to/images ./run_multi_gpu_pipeline.sh

Environment variables:
  SRC                       Required. Source directory with input images.
  RUN_NAME                  Optional. Default: multigpu-YYYYmmdd_HHMMSS
  GPU_IDS                   Optional. Comma/space list (e.g. "0,1,2") or "auto".
  MAX_GPUS                  Optional. 0 means all detected GPUs.
  BASE_COMFY_PORT           Optional. Default: 8188
  CONTAINER_PREFIX          Optional. Default: comfyui-g
  COMFYUI_DATA_ROOT         Optional. Default: <repo>/comfyui_data
  TMUX_SESSION_PREFIX       Optional. Default: mgpu
  WORK_ROOT                 Optional. Default: <repo>/tmp/multigpu/<RUN_NAME>
  WAIT_POLL_SEC             Optional. Default: 10
  FINAL_OUTPUT_DIR          Optional. If set, writes per-shard outputs under
                            FINAL_OUTPUT_DIR/<RUN_NAME>/gpu<id>/
  STRICT_HARDLINK           Optional. 1 = fail when hardlink is not possible.
                            0 = allow copy fallback. Default: 1.
  DRY_RUN                   Optional. 1 = prepare/print plan only, no launches.
  SINGLE_GPU_CONFLICT_MODE  Optional. off|warn|stop. Default: warn.
                            Applies only when NUM_GPUS=1. Detects other running
                            comfyui containers and optionally stops them.
  S3_MODELS_ROOT            Optional. Default: s3://panaromic-images/pano_models

Forwarded to each shard run_full_pipeline.sh invocation (per GPU):
  MODELS_ROOT, MODELS_COMFYUI_DIR, MODELS_PRIVACY_DIR,
  AUTO_DOWNLOAD_MODELS, FORCE_REPROCESS, STRICT_HARDLINK,
  SAM3_WORKERS, SAM3_RESIZE_WIDTH, SAM3_RESIZE_HEIGHT, SAM3_GLARE_THRESHOLD,
  SAM3_TILE_ROWS, SAM3_TILE_COLS, SAM3_SCRIPT,
  POSTPROCESS_WORKERS,
  PRIVACY_WORKERS, PRIVACY_FACE_MODEL, PRIVACY_LP_MODEL,
  PRIVACY_FACE_CONF, PRIVACY_LP_CONF, PRIVACY_FACE_IOU, PRIVACY_FACE_IMGSZ,
  PRIVACY_DET_FACE_W, PRIVACY_P360_DEVICE, PRIVACY_BLUR_SCOPE,
  PRIVACY_BLUR_BACKEND, PRIVACY_OUTPUT_MODE,
  COMFY_IMAGE_NODE_ID, COMFY_MASK_NODE_ID, COMFY_SAM3_MASK_NODE_ID,
  SKY_REFERENCE_SOURCE, SKY_REFERENCE_FILENAME,
  LAPLACIAN_DILATION, LAPLACIAN_BLUR, LAPLACIAN_LEVELS,
  DOWNSTREAM_MODE (default: isolated), STOP_AFTER_STAGE (default: egoblur),
  COMFY_READY_TIMEOUT, COMFY_READY_POLL,
  NVIDIA_CUDA_TEST_IMAGE, COMFY_IMAGE, TORCH_CACHE_DIR.
  RESET_CONTAINER_BEFORE_RUN (default: 1 in this launcher).

Notes:
  - Input sharding is automatic (sequential contiguous splits) using hardlinks when possible.
  - One shard run is launched per GPU in tmux.
  - Each shard only stops its own container via COMFY_STOP_CONTAINERS=<container>.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"

SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  echo "ERROR: SRC is required."
  echo "Hint: SRC=/abs/path/to/images ./run_multi_gpu_pipeline.sh"
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: SRC directory not found: $SRC"
  exit 1
fi

RUN_NAME="${RUN_NAME:-multigpu-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS_RAW="${GPU_IDS:-auto}"
MAX_GPUS="${MAX_GPUS:-0}"
BASE_COMFY_PORT="${BASE_COMFY_PORT:-8180}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-comfyui-g}"
COMFYUI_DATA_ROOT="${COMFYUI_DATA_ROOT:-$REPO/comfyui_data}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-mgpu}"
WORK_ROOT="${WORK_ROOT:-$REPO/tmp/multigpu/$RUN_NAME}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-10}"
DRY_RUN="${DRY_RUN:-0}"
SINGLE_GPU_CONFLICT_MODE="${SINGLE_GPU_CONFLICT_MODE:-warn}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"

DOWNSTREAM_MODE="${DOWNSTREAM_MODE:-isolated}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"

LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/multigpu_${RUN_NAME}.log"
EVENTS_FILE="$LOG_DIR/multigpu_${RUN_NAME}.events.jsonl"
mkdir -p "$LOG_DIR" "$WORK_ROOT/shards" "$WORK_ROOT/jobs"

exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
END_EPOCH=0
TOTAL_IN=0
TOTAL_OUT=0
FAIL=0
NUM_GPUS=0
RUN_STATUS="running"
RUN_FINALIZED=0
CURRENT_STEP=""
CURRENT_STEP_STARTED_AT=0
FAILED_STEP=""
FAILED_COMMAND=""
FAILED_ERROR=""
FAILED_EXIT_CODE=0
MERGED_ROOT=""

EVENT_BASE_ARGS=(
  append-event
  --file "$EVENTS_FILE"
  --run-type multi_gpu
  --run-id "$RUN_NAME"
  --script run_multi_gpu_pipeline.sh
)

log_event() {
  python3 "$PIPELINE_HELPERS" "${EVENT_BASE_ARGS[@]}" "$@" >/dev/null 2>&1 || true
}

set_step() {
  CURRENT_STEP="$1"
  CURRENT_STEP_STARTED_AT=$(date +%s)
}

clear_step() {
  CURRENT_STEP=""
  CURRENT_STEP_STARTED_AT=0
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local command="$3"
  local step="${CURRENT_STEP:-runtime}"
  local elapsed=0
  RUN_STATUS="failure"
  if [ "$CURRENT_STEP_STARTED_AT" -gt 0 ]; then
    elapsed=$(( $(date +%s) - CURRENT_STEP_STARTED_AT ))
  fi
  if [ -z "$FAILED_STEP" ]; then
    FAILED_STEP="$step"
  fi
  if [ -z "$FAILED_COMMAND" ]; then
    FAILED_COMMAND="$command"
  fi
  if [ -z "$FAILED_ERROR" ]; then
    FAILED_ERROR="command failed at line $line_no"
  fi
  FAILED_EXIT_CODE="$exit_code"
  log_event \
    --event step_fail \
    --status failure \
    --stage "$step" \
    --elapsed-sec "$elapsed" \
    --exit-code "$exit_code" \
    --command "$command" \
    --error "command failed at line $line_no"
  clear_step
}

on_exit() {
  local exit_code="$1"
  local -a event_args=(
    --event run_end
    --status success
    --metric total_in="$TOTAL_IN"
    --metric total_out="$TOTAL_OUT"
    --metric failed_shards="$FAIL"
    --metric num_gpus="$NUM_GPUS"
    --path log_file="$LOG_FILE"
    --path events_file="$EVENTS_FILE"
    --path work_root="$WORK_ROOT"
  )

  if [ "$RUN_FINALIZED" -eq 1 ]; then
    return
  fi
  RUN_FINALIZED=1

  END_EPOCH=$(date +%s)
  if [ -n "${START_EPOCH:-}" ]; then
    event_args+=( --elapsed-sec "$((END_EPOCH - START_EPOCH))" )
  fi
  if [ -n "$MERGED_ROOT" ]; then
    event_args+=( --path merged_output="$MERGED_ROOT" )
  fi

  if [ "$exit_code" -ne 0 ] || [ "$RUN_STATUS" = "failure" ]; then
    event_args=(
      --event run_end
      --status failure
      --metric total_in="$TOTAL_IN"
      --metric total_out="$TOTAL_OUT"
      --metric failed_shards="$FAIL"
      --metric num_gpus="$NUM_GPUS"
      --path log_file="$LOG_FILE"
      --path events_file="$EVENTS_FILE"
      --path work_root="$WORK_ROOT"
    )
    if [ -n "${START_EPOCH:-}" ]; then
      event_args+=( --elapsed-sec "$((END_EPOCH - START_EPOCH))" )
    fi
    if [ -n "$MERGED_ROOT" ]; then
      event_args+=( --path merged_output="$MERGED_ROOT" )
    fi
    if [ -n "$FAILED_STEP" ]; then
      event_args+=( --stage "$FAILED_STEP" )
    fi
    if [ "$FAILED_EXIT_CODE" -ne 0 ]; then
      event_args+=( --exit-code "$FAILED_EXIT_CODE" )
    fi
    if [ -n "$FAILED_COMMAND" ]; then
      event_args+=( --command "$FAILED_COMMAND" )
    fi
    if [ -n "$FAILED_ERROR" ]; then
      event_args+=( --error "$FAILED_ERROR" )
    fi
  fi

  log_event "${event_args[@]}"
}

trap 'on_error $? $LINENO "$BASH_COMMAND"' ERR
trap 'on_exit $?' EXIT

if [[ "$MAX_GPUS" =~ [^0-9] ]]; then
  echo "ERROR: MAX_GPUS must be a non-negative integer"
  exit 1
fi

if [[ "$WAIT_POLL_SEC" =~ [^0-9] || "$WAIT_POLL_SEC" -lt 1 ]]; then
  echo "ERROR: WAIT_POLL_SEC must be an integer >= 1"
  exit 1
fi

if [[ "$SINGLE_GPU_CONFLICT_MODE" != "off" && "$SINGLE_GPU_CONFLICT_MODE" != "warn" && "$SINGLE_GPU_CONFLICT_MODE" != "stop" ]]; then
  echo "ERROR: SINGLE_GPU_CONFLICT_MODE must be one of: off|warn|stop"
  exit 1
fi

if [[ "$STRICT_HARDLINK" != "0" && "$STRICT_HARDLINK" != "1" ]]; then
  echo "ERROR: STRICT_HARDLINK must be 0 or 1"
  exit 1
fi

for cmd in docker nvidia-smi tmux python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd"
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose is not available."
  exit 1
fi

RUN_FULL="$REPO/run_full_pipeline.sh"
if [[ ! -x "$RUN_FULL" ]]; then
  echo "ERROR: run_full_pipeline.sh not found or not executable at $RUN_FULL"
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

if [[ "$MAX_GPUS" -gt 0 && ${#GPU_LIST[@]} -gt "$MAX_GPUS" ]]; then
  GPU_LIST=("${GPU_LIST[@]:0:$MAX_GPUS}")
fi

NUM_GPUS=${#GPU_LIST[@]}

PLANNED_CONTAINERS=()
for gpu_id in "${GPU_LIST[@]}"; do
  PLANNED_CONTAINERS+=("${CONTAINER_PREFIX}${gpu_id}")
done

if [[ "$NUM_GPUS" -eq 1 && "$SINGLE_GPU_CONFLICT_MODE" != "off" ]]; then
  mapfile -t RUNNING_COMFY_CONTAINERS < <(docker ps --format '{{.Names}}' | grep -E '^(comfyui-container|comfyui-g[0-9]+)$' || true)
  CONFLICT_CONTAINERS=()

  for c in "${RUNNING_COMFY_CONTAINERS[@]}"; do
    skip=0
    for p in "${PLANNED_CONTAINERS[@]}"; do
      if [[ "$c" == "$p" ]]; then
        skip=1
        break
      fi
    done
    if [[ "$skip" -eq 0 ]]; then
      CONFLICT_CONTAINERS+=("$c")
    fi
  done

  if [[ ${#CONFLICT_CONTAINERS[@]} -gt 0 ]]; then
    echo "Single-GPU guard: detected other running comfyui containers: ${CONFLICT_CONTAINERS[*]}"
    if [[ "$SINGLE_GPU_CONFLICT_MODE" == "stop" ]]; then
      echo "Single-GPU guard mode=stop -> stopping conflicting containers..."
      docker stop "${CONFLICT_CONTAINERS[@]}" >/dev/null
      echo "Stopped conflicting containers."
    else
      echo "Single-GPU guard mode=warn -> continuing without stopping them."
    fi
  fi
fi

MANIFEST_JSON="$WORK_ROOT/shard_manifest.json"
COUNT_JSON="$WORK_ROOT/shard_counts.json"

GPU_IDS_CSV=$(IFS=,; printf '%s' "${GPU_LIST[*]}")
log_event \
  --event run_start \
  --status running \
  --param gpu_ids="$GPU_IDS_CSV" \
  --param downstream_mode="$DOWNSTREAM_MODE" \
  --param stop_after_stage="$STOP_AFTER_STAGE" \
  --param strict_hardlink="$STRICT_HARDLINK" \
  --param dry_run="$DRY_RUN" \
  --param single_gpu_conflict_mode="$SINGLE_GPU_CONFLICT_MODE" \
  --metric num_gpus="$NUM_GPUS" \
  --path log_file="$LOG_FILE" \
  --path events_file="$EVENTS_FILE" \
  --path work_root="$WORK_ROOT" \
  --path source_dir="$SRC" \
  --path manifest_json="$MANIFEST_JSON" \
  --path count_json="$COUNT_JSON"

set_step split_shards
python3 "$PIPELINE_HELPERS" split-shards \
  --src "$SRC" \
  --shards-root "$WORK_ROOT/shards" \
  --num-gpus "$NUM_GPUS" \
  --manifest-json "$MANIFEST_JSON" \
  --count-json "$COUNT_JSON" \
  --strict-hardlink "$STRICT_HARDLINK"
clear_step

echo "RUN_NAME=$RUN_NAME"
echo "SRC=$SRC"
echo "NUM_GPUS=$NUM_GPUS"
echo "GPU_IDS=${GPU_LIST[*]}"
echo "WORK_ROOT=$WORK_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "EVENTS_FILE=$EVENTS_FILE"
echo "DOWNSTREAM_MODE=$DOWNSTREAM_MODE"
echo "STOP_AFTER_STAGE=$STOP_AFTER_STAGE"
echo "STRICT_HARDLINK=$STRICT_HARDLINK"
echo "DRY_RUN=$DRY_RUN"
echo "SINGLE_GPU_CONFLICT_MODE=$SINGLE_GPU_CONFLICT_MODE"

set_step preflight

install_nvidia_container_toolkit_once() {
  if command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit already installed."
    return 0
  fi

  echo "Installing NVIDIA Container Toolkit..."

  if command -v sudo >/dev/null 2>&1; then
    SUDO=sudo
  else
    SUDO=
  fi

  $SUDO rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
  $SUDO sh -c 'apt-get update && apt-get install -y curl gpg ca-certificates' 2>/dev/null || true
  $SUDO mkdir -p /usr/share/keyrings 2>/dev/null || true
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true

  if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  else
    wget -qO- https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO apt-key add - 2>/dev/null || true
    echo "deb https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  fi

  $SUDO apt-get update 2>/dev/null || true
  $SUDO apt-get install -y nvidia-container-toolkit 2>/dev/null || true
  $SUDO nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
  $SUDO systemctl restart docker 2>/dev/null || true

  if command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit installed successfully."
  else
    echo "WARNING: NVIDIA Container Toolkit installation may have failed."
  fi
}

echo "Checking NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  install_nvidia_container_toolkit_once
  if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: Docker GPU runtime is not working after toolkit installation."
    echo "Please verify NVIDIA Container Toolkit + Docker runtime setup manually."
    exit 1
  fi
else
  echo "NVIDIA Container Toolkit already working."
fi

echo "Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation failed. Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  echo "uv installed successfully."
else
  echo "uv already available."
fi

ensure_aws_cli() {
  if command -v aws >/dev/null 2>&1; then
    echo "aws already available."
    return 0
  fi

  echo "Installing awscli with uv..."
  if ! uv tool install awscli; then
    echo "WARNING: awscli installation failed. Model downloads will fall back to HTTP sources."
    return 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v aws >/dev/null 2>&1; then
    echo "WARNING: awscli installed but 'aws' is still not in PATH. Model downloads will fall back to HTTP sources."
    return 1
  fi

  echo "awscli installed successfully."
}

echo "Checking awscli..."
ensure_aws_cli || true

echo "Checking required models..."
_models_root="${MODELS_ROOT:-$REPO/models}"
_models_comfyui="${MODELS_COMFYUI_DIR:-$_models_root/comfyui}"
_models_privacy="${MODELS_PRIVACY_DIR:-$_models_root/privacy_blur}"
_required_models=(
  "$_models_comfyui/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
  "$_models_comfyui/vae/qwen_image_vae.safetensors"
  "$_models_comfyui/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
  "$_models_comfyui/upscale_models/RealESRGAN_x2plus.pth"
  "$_models_comfyui/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
  "$_models_comfyui/sam3/model.safetensors"
  "$_models_comfyui/lama/big-lama.pt"
  "$_models_privacy/face_yolov8n.pt"
  "$_models_privacy/yolo-v9-s-608-license-plates-end2end.onnx"
)
_need_download=0
for _m in "${_required_models[@]}"; do
  if [ ! -f "$_m" ]; then
    _need_download=1
    break
  fi
done
if [ "$_need_download" = "1" ]; then
  if [ "${AUTO_DOWNLOAD_MODELS:-1}" = "1" ]; then
    echo "Required models missing. Running download-models.sh once before launching shards..."
    MODELS_ROOT="$_models_root" bash "$REPO/download-models.sh"
  else
    echo "ERROR: Required models are missing and AUTO_DOWNLOAD_MODELS=0"
    exit 1
  fi
else
  echo "All required models present."
fi

_comfy_image="${COMFY_IMAGE:-amanbagrecha/container-comfyui:latest}"
if [[ "${SKIP_IMAGE_PULL:-0}" == "1" ]]; then
  echo "Skipping Docker image pull (SKIP_IMAGE_PULL=1)."
else
  echo "Pulling Docker image..."
  docker pull "$_comfy_image"
fi

clear_step

LAUNCH_PLAN="$WORK_ROOT/launch_plan.tsv"
: > "$LAUNCH_PLAN"

set_step launch_shards
for idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$idx]}"
  shard_dir="$WORK_ROOT/shards/gpu${idx}"
  shard_count=$(python3 "$PIPELINE_HELPERS" count-images --path "$shard_dir" --include-bmp)

  if [[ "$shard_count" -eq 0 ]]; then
    echo "Skipping GPU $gpu_id (empty shard)"
    continue
  fi

  container_name="${CONTAINER_PREFIX}${gpu_id}"
  comfy_port=$((BASE_COMFY_PORT + idx))
  batch_name="${RUN_NAME}-g${gpu_id}"
  child_run_id="${RUN_NAME}_g${gpu_id}"
  comfy_data_dir="$COMFYUI_DATA_ROOT/$container_name"
  session_name="${TMUX_SESSION_PREFIX}-${RUN_NAME}-g${gpu_id}"
  rc_file="$WORK_ROOT/rc_gpu${gpu_id}.txt"
  child_log_file="$LOG_DIR/fullrun_${child_run_id}.log"
  child_events_file="$LOG_DIR/fullrun_${child_run_id}.events.jsonl"
  job_script="$WORK_ROOT/jobs/gpu${gpu_id}.sh"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export SRC="$shard_dir"
export INPUT_MODE="staged"
export BATCH_NAME="$batch_name"
export CONTAINER_NAME="$container_name"
export COMFYUI_DATA_DIR="$comfy_data_dir"
export COMFY_PORT="$comfy_port"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export DOWNSTREAM_MODE="$DOWNSTREAM_MODE"
export COMFY_STOP_CONTAINERS="$container_name"
export STOP_AFTER_STAGE="$STOP_AFTER_STAGE"
export FINAL_OUTPUT_DIR=""
export SAM3_WORKERS="${SAM3_WORKERS:-4}"
export SAM3_RESIZE_WIDTH="${SAM3_RESIZE_WIDTH:-4000}"
export SAM3_RESIZE_HEIGHT="${SAM3_RESIZE_HEIGHT:-2000}"
export SAM3_GLARE_THRESHOLD="${SAM3_GLARE_THRESHOLD:-0.4}"
export SAM3_TILE_ROWS="${SAM3_TILE_ROWS:-2}"
export SAM3_TILE_COLS="${SAM3_TILE_COLS:-1}"
export SAM3_SCRIPT="${SAM3_SCRIPT:-sam3_tiled_mask.py}"
export POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-3}"
export LAPLACIAN_DILATION="${LAPLACIAN_DILATION:-1}"
export LAPLACIAN_BLUR="${LAPLACIAN_BLUR:-10}"
export LAPLACIAN_LEVELS="${LAPLACIAN_LEVELS:-7}"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
export MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-${MODELS_ROOT:-$REPO/models}/comfyui}"
export MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}"
export AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
export FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
export SKIP_PREFLIGHT=1
export STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
export PRIVACY_WORKERS="${PRIVACY_WORKERS:-4}"
export PRIVACY_FACE_MODEL="${PRIVACY_FACE_MODEL:-${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}/face_yolov8n.pt}"
export PRIVACY_LP_MODEL="${PRIVACY_LP_MODEL:-${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}/yolo-v9-s-608-license-plates-end2end.onnx}"
export PRIVACY_FACE_CONF="${PRIVACY_FACE_CONF:-0.4}"
export PRIVACY_LP_CONF="${PRIVACY_LP_CONF:-0.4}"
export PRIVACY_FACE_IOU="${PRIVACY_FACE_IOU:-0.5}"
export PRIVACY_FACE_IMGSZ="${PRIVACY_FACE_IMGSZ:-1024}"
export PRIVACY_DET_FACE_W="${PRIVACY_DET_FACE_W:-1024}"
export PRIVACY_P360_DEVICE="${PRIVACY_P360_DEVICE:-auto}"
export PRIVACY_BLUR_SCOPE="${PRIVACY_BLUR_SCOPE:-roi}"
export PRIVACY_BLUR_BACKEND="${PRIVACY_BLUR_BACKEND:-gpu}"
export PRIVACY_OUTPUT_MODE="${PRIVACY_OUTPUT_MODE:-blur_only}"
export COMFY_IMAGE_NODE_ID="${COMFY_IMAGE_NODE_ID:-91}"
export COMFY_MASK_NODE_ID="${COMFY_MASK_NODE_ID:-34}"
export COMFY_SAM3_MASK_NODE_ID="${COMFY_SAM3_MASK_NODE_ID:-60}"
export SKY_REFERENCE_SOURCE="${SKY_REFERENCE_SOURCE:-$REPO/inpainting-workflow-master/reference_sky.png}"
export SKY_REFERENCE_FILENAME="${SKY_REFERENCE_FILENAME:-chrome_xWUjmfs7m4.png}"
export COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
export COMFY_READY_POLL="${COMFY_READY_POLL:-2}"
export NVIDIA_CUDA_TEST_IMAGE="${NVIDIA_CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu22.04}"
export COMFY_IMAGE="${COMFY_IMAGE:-amanbagrecha/container-comfyui:latest}"
export TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/.cache/torch}"
export RESET_CONTAINER_BEFORE_RUN="${RESET_CONTAINER_BEFORE_RUN:-1}"
export RUN_ID="$child_run_id"
export PARENT_RUN_ID="$RUN_NAME"

"$RUN_FULL"
rc=\$?
printf '%s\n' "\$rc" > "$rc_file"
exit "\$rc"
EOF

  chmod +x "$job_script"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$gpu_id" "$idx" "$container_name" "$batch_name" "$comfy_data_dir" "$session_name" "$rc_file" "$child_run_id" "$child_log_file" "$child_events_file" >> "$LAUNCH_PLAN"

  echo "Planned GPU=$gpu_id shard_idx=$idx count=$shard_count container=$container_name batch=$batch_name port=$comfy_port session=$session_name"
  log_event \
    --event shard_planned \
    --status success \
    --gpu-id "$gpu_id" \
    --shard-index "$idx" \
    --child-run-id "$child_run_id" \
    --container-name "$container_name" \
    --batch-name "$batch_name" \
    --metric input_count="$shard_count" \
    --path shard_dir="$shard_dir" \
    --path child_log_file="$child_log_file" \
    --path child_events_file="$child_events_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  tmux new-session -d -s "$session_name" "$job_script"
  log_event \
    --event shard_launched \
    --status running \
    --gpu-id "$gpu_id" \
    --shard-index "$idx" \
    --child-run-id "$child_run_id" \
    --container-name "$container_name" \
    --batch-name "$batch_name" \
    --path child_log_file="$child_log_file" \
    --path child_events_file="$child_events_file"
done
clear_step

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. Launch plan: $LAUNCH_PLAN"
  exit 0
fi

if [[ ! -s "$LAUNCH_PLAN" ]]; then
  echo "ERROR: no shard jobs were launched."
  exit 1
fi

echo "Waiting for shard sessions to complete..."
set_step wait_shards
while true; do
  alive=0
  while IFS=$'\t' read -r _ _ _ _ _ session _ _ _ _; do
    if tmux has-session -t "$session" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "$LAUNCH_PLAN"

  if [[ "$alive" -eq 0 ]]; then
    break
  fi

  echo "still_running_sessions=$alive"
  sleep "$WAIT_POLL_SEC"
done

echo "All shard sessions finished."

while IFS=$'\t' read -r gpu_id idx container batch comfy_data_dir _ rc_file child_run_id child_log_file child_events_file; do
  rc="missing"
  if [[ -f "$rc_file" ]]; then
    rc="$(tr -d '[:space:]' < "$rc_file")"
  fi

  in_count=$(python3 "$PIPELINE_HELPERS" count-images --path "$WORK_ROOT/shards/gpu${idx}" --include-bmp)

  if [[ "$STOP_AFTER_STAGE" == "egoblur" ]]; then
    out_dir="$comfy_data_dir/output-egoblur/$batch"
  elif [[ "$STOP_AFTER_STAGE" == "sam3" ]]; then
    out_dir="$comfy_data_dir/output-sam3-mask/$batch"
  elif [[ "$STOP_AFTER_STAGE" == "postprocess" ]]; then
    out_dir="$comfy_data_dir/output-postprocessed/$batch"
  else
    out_dir="$comfy_data_dir/output/$batch"
  fi

  out_count=$(python3 "$PIPELINE_HELPERS" count-images --path "$out_dir" --include-bmp)

  TOTAL_IN=$((TOTAL_IN + in_count))
  TOTAL_OUT=$((TOTAL_OUT + out_count))

  if [[ "$rc" != "0" ]]; then
    FAIL=$((FAIL + 1))
  fi

  shard_status="success"
  if [[ "$rc" != "0" ]]; then
    shard_status="failure"
  fi

  shard_event_args=(
    --event shard_finished
    --status "$shard_status"
    --gpu-id "$gpu_id"
    --shard-index "$idx"
    --child-run-id "$child_run_id"
    --container-name "$container"
    --batch-name "$batch"
    --metric input_count="$in_count"
    --metric output_count="$out_count"
    --path child_log_file="$child_log_file"
    --path child_events_file="$child_events_file"
    --path output_dir="$out_dir"
  )
  if [[ "$rc" =~ ^[0-9]+$ ]]; then
    shard_event_args+=( --exit-code "$rc" )
  fi
  log_event "${shard_event_args[@]}"

  echo "gpu=$gpu_id rc=$rc in_count=$in_count out_count=$out_count batch=$batch log=$child_log_file events=$child_events_file"
done < "$LAUNCH_PLAN"
clear_step

echo "TOTAL_IN=$TOTAL_IN"
echo "TOTAL_OUT=$TOTAL_OUT"
echo "FAILED_SHARDS=$FAIL"

if [[ -n "${FINAL_OUTPUT_DIR:-}" ]]; then
  MERGED_ROOT="$FINAL_OUTPUT_DIR/$RUN_NAME"
  mkdir -p "$MERGED_ROOT"
  echo "Collecting per-shard outputs into $MERGED_ROOT"
  log_event --event merge_start --status running --path merged_output="$MERGED_ROOT"
  set_step merge_outputs

  while IFS=$'\t' read -r gpu_id _ _ batch comfy_data_dir _ _ _ _ _ _; do
    if [[ "$STOP_AFTER_STAGE" == "egoblur" ]]; then
      src_dir="$comfy_data_dir/output-egoblur/$batch"
    elif [[ "$STOP_AFTER_STAGE" == "postprocess" ]]; then
      src_dir="$comfy_data_dir/output-postprocessed/$batch"
    else
      src_dir="$comfy_data_dir/output/$batch"
    fi
    dst_dir="$MERGED_ROOT/gpu${gpu_id}"
    mkdir -p "$dst_dir"

    python3 "$PIPELINE_HELPERS" link-tree \
      --src "$src_dir" \
      --dst "$dst_dir" \
      --strict-hardlink "$STRICT_HARDLINK"

  done < "$LAUNCH_PLAN"

  echo "MERGED_OUTPUT=$MERGED_ROOT"
  clear_step
  log_event --event merge_end --status success --path merged_output="$MERGED_ROOT"
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo "ERROR: $FAIL shard(s) failed."
  exit 1
fi

RUN_STATUS="success"
echo "Multi-GPU run completed successfully."
echo "Events: $EVENTS_FILE"
echo "Log: $LOG_FILE"
