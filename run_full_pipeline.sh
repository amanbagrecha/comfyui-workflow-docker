#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-$SCRIPT_DIR}"
CONTAINER_NAME="${CONTAINER_NAME:-comfyui-container}"
COMFYUI_DATA_DIR="${COMFYUI_DATA_DIR:-$REPO/comfyui_data/$CONTAINER_NAME}"
SRC="${SRC:-$COMFYUI_DATA_DIR/input}"
BATCH_NAME="${BATCH_NAME:-batch-$(date +%Y%m%d_%H%M%S)}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-3}"
PRIVACY_WORKERS="${PRIVACY_WORKERS:-4}"
SAM3_WORKERS="${SAM3_WORKERS:-4}"
SAM3_RESIZE_WIDTH="${SAM3_RESIZE_WIDTH:-4000}"
SAM3_RESIZE_HEIGHT="${SAM3_RESIZE_HEIGHT:-2000}"
SAM3_GLARE_THRESHOLD="${SAM3_GLARE_THRESHOLD:-0.4}"
SAM3_TILE_ROWS="${SAM3_TILE_ROWS:-2}"
SAM3_TILE_COLS="${SAM3_TILE_COLS:-1}"
SAM3_SCRIPT="${SAM3_SCRIPT:-sam3_tiled_mask.py}"
LAPLACIAN_DILATION="${LAPLACIAN_DILATION:-1}"
LAPLACIAN_BLUR="${LAPLACIAN_BLUR:-10}"
LAPLACIAN_LEVELS="${LAPLACIAN_LEVELS:-7}"
INPUT_MODE="${INPUT_MODE:-auto}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-}"
DOWNSTREAM_MODE="${DOWNSTREAM_MODE:-inline}"
COMFY_STOP_CONTAINERS="${COMFY_STOP_CONTAINERS:-auto}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-$MODELS_ROOT/privacy_blur}"
PRIVACY_FACE_MODEL="${PRIVACY_FACE_MODEL:-$MODELS_PRIVACY_DIR/face_yolov8n.pt}"
PRIVACY_LP_MODEL="${PRIVACY_LP_MODEL:-$MODELS_PRIVACY_DIR/yolo-v9-s-608-license-plates-end2end.onnx}"
PRIVACY_FACE_CONF="${PRIVACY_FACE_CONF:-0.4}"
PRIVACY_LP_CONF="${PRIVACY_LP_CONF:-0.4}"
PRIVACY_FACE_IOU="${PRIVACY_FACE_IOU:-0.5}"
PRIVACY_FACE_IMGSZ="${PRIVACY_FACE_IMGSZ:-1024}"
PRIVACY_DET_FACE_W="${PRIVACY_DET_FACE_W:-1024}"
PRIVACY_P360_DEVICE="${PRIVACY_P360_DEVICE:-auto}"
PRIVACY_BLUR_SCOPE="${PRIVACY_BLUR_SCOPE:-roi}"
PRIVACY_BLUR_BACKEND="${PRIVACY_BLUR_BACKEND:-gpu}"
PRIVACY_OUTPUT_MODE="${PRIVACY_OUTPUT_MODE:-blur_only}"
COMFY_IMAGE="${COMFY_IMAGE:-amanbagrecha/container-comfyui:latest}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/.cache/torch}"
COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
COMFY_READY_POLL="${COMFY_READY_POLL:-2}"
RESET_CONTAINER_BEFORE_RUN="${RESET_CONTAINER_BEFORE_RUN:-1}"
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
NVIDIA_CUDA_TEST_IMAGE="${NVIDIA_CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu22.04}"
COMFY_IMAGE_NODE_ID="${COMFY_IMAGE_NODE_ID:-91}"
COMFY_MASK_NODE_ID="${COMFY_MASK_NODE_ID:-34}"
COMFY_SAM3_MASK_NODE_ID="${COMFY_SAM3_MASK_NODE_ID:-60}"
SKY_REFERENCE_SOURCE="${SKY_REFERENCE_SOURCE:-$REPO/inpainting-workflow-master/reference_sky.png}"
SKY_REFERENCE_FILENAME="${SKY_REFERENCE_FILENAME:-chrome_xWUjmfs7m4.png}"
PARENT_RUN_ID="${PARENT_RUN_ID:-}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"
CONTAINER_PIPELINE_HELPERS="/workspace/inpainting/pipeline_helpers.py"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${CONTAINER_NAME}_$$}"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fullrun_${RUN_ID}.log"
EVENTS_FILE="$LOG_DIR/fullrun_${RUN_ID}.events.jsonl"

exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
END_EPOCH=0
TOTAL_SEC=0

RUN_STATUS="running"
RUN_FINALIZED=0
CURRENT_STAGE=""
CURRENT_STAGE_STARTED_AT=0
CURRENT_STAGE_COMMAND=""
FAILED_STAGE=""
FAILED_COMMAND=""
FAILED_ERROR=""
FAILED_EXIT_CODE=0

HARDLINK_SEC=0
SAM3_SEC=0
INPAINT_SEC=0
POSTPROCESS_SEC=0
EGOBLUR_SEC=0
COUNT_SEC=0

SAM3_MASK_COUNT=0
INPAINT_COUNT=0
COUNT_INPUT=0
COUNT_SAM3_MASK=0
COUNT_INPAINT=0
COUNT_POSTPROCESS=0
COUNT_EGOBLUR=0

HOST_INPUT_DIR="$SRC"
CONTAINER_INPUT_DIR="$SRC"
RESOLVED_INPUT_MODE=""
FINAL_BATCH_DIR=""
DST=""
OUT1=""
OUT_MASK=""
OUT2=""
OUT3=""

EVENT_BASE_ARGS=(
  append-event
  --file "$EVENTS_FILE"
  --run-type full_pipeline
  --run-id "$RUN_ID"
  --script run_full_pipeline.sh
  --container-name "$CONTAINER_NAME"
  --batch-name "$BATCH_NAME"
)
if [ -n "$PARENT_RUN_ID" ]; then
  EVENT_BASE_ARGS+=( --parent-run-id "$PARENT_RUN_ID" )
fi
if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] && [ "${NVIDIA_VISIBLE_DEVICES:-unset}" != "unset" ]; then
  EVENT_BASE_ARGS+=( --gpu-id "${NVIDIA_VISIBLE_DEVICES}" )
fi

log_event() {
  python3 "$PIPELINE_HELPERS" "${EVENT_BASE_ARGS[@]}" "$@" >/dev/null 2>&1 || true
}

quote_cmd() {
  local quoted
  printf -v quoted '%q ' "$@"
  printf '%s' "${quoted% }"
}

start_stage() {
  local stage="$1"
  local command="${2:-}"
  CURRENT_STAGE="$stage"
  CURRENT_STAGE_STARTED_AT=$(date +%s)
  CURRENT_STAGE_COMMAND="$command"
  echo "=== STAGE_START $stage ==="
  if [ -n "$command" ]; then
    log_event --event stage_start --status running --stage "$stage" --command "$command"
  else
    log_event --event stage_start --status running --stage "$stage"
  fi
}

finish_stage() {
  local stage="$1"
  local elapsed="$2"
  shift 2
  echo "=== STAGE_END $stage elapsed_sec=$elapsed ==="
  log_event --event stage_end --status success --stage "$stage" --elapsed-sec "$elapsed" "$@"
  CURRENT_STAGE=""
  CURRENT_STAGE_STARTED_AT=0
  CURRENT_STAGE_COMMAND=""
}

skip_stage() {
  local stage="$1"
  local reason="$2"
  echo "=== STAGE_SKIP $stage reason=$reason elapsed_sec=0 ==="
  log_event --event stage_skip --status skipped --stage "$stage" --reason "$reason"
}

fail_stage() {
  local stage="$1"
  local message="$2"
  local exit_code="${3:-1}"
  local elapsed=0
  local -a event_args=(
    --event stage_fail
    --status failure
    --stage "$stage"
    --exit-code "$exit_code"
    --error "$message"
  )
  if [ "$CURRENT_STAGE_STARTED_AT" -gt 0 ]; then
    elapsed=$(( $(date +%s) - CURRENT_STAGE_STARTED_AT ))
    event_args+=( --elapsed-sec "$elapsed" )
  fi
  if [ -n "$CURRENT_STAGE_COMMAND" ]; then
    event_args+=( --command "$CURRENT_STAGE_COMMAND" )
  fi
  RUN_STATUS="failure"
  FAILED_STAGE="$stage"
  FAILED_COMMAND="$CURRENT_STAGE_COMMAND"
  FAILED_ERROR="$message"
  FAILED_EXIT_CODE="$exit_code"
  echo "ERROR: $message"
  log_event "${event_args[@]}"
  CURRENT_STAGE=""
  CURRENT_STAGE_STARTED_AT=0
  CURRENT_STAGE_COMMAND=""
  exit "$exit_code"
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local command="$3"
  local stage="${CURRENT_STAGE:-runtime}"
  local elapsed=0
  RUN_STATUS="failure"
  if [ "$CURRENT_STAGE_STARTED_AT" -gt 0 ]; then
    elapsed=$(( $(date +%s) - CURRENT_STAGE_STARTED_AT ))
  fi
  if [ -z "$FAILED_STAGE" ]; then
    FAILED_STAGE="$stage"
  fi
  if [ -z "$FAILED_COMMAND" ]; then
    FAILED_COMMAND="$command"
  fi
  if [ -z "$FAILED_ERROR" ]; then
    FAILED_ERROR="command failed at line $line_no"
  fi
  FAILED_EXIT_CODE="$exit_code"
  log_event \
    --event stage_fail \
    --status failure \
    --stage "$stage" \
    --elapsed-sec "$elapsed" \
    --exit-code "$exit_code" \
    --command "$command" \
    --error "command failed at line $line_no"
  CURRENT_STAGE=""
  CURRENT_STAGE_STARTED_AT=0
  CURRENT_STAGE_COMMAND=""
}

on_exit() {
  local exit_code="$1"
  local final_status="success"
  local -a event_args=(
    --event run_end
    --status success
  )

  if [ "$RUN_FINALIZED" -eq 1 ]; then
    return
  fi
  RUN_FINALIZED=1

  END_EPOCH=$(date +%s)
  TOTAL_SEC=$((END_EPOCH - START_EPOCH))

  if [ "$exit_code" -ne 0 ] || [ "$RUN_STATUS" = "failure" ]; then
    final_status="failure"
    event_args=( --event run_end --status failure )
  fi

  event_args+=(
    --elapsed-sec "$TOTAL_SEC"
    --metric stage_hardlink_sec="$HARDLINK_SEC"
    --metric stage_sam3_mask_sec="$SAM3_SEC"
    --metric stage_inpainting_sec="$INPAINT_SEC"
    --metric stage_postprocess_sec="$POSTPROCESS_SEC"
    --metric stage_egoblur_sec="$EGOBLUR_SEC"
    --metric stage_counts_sec="$COUNT_SEC"
    --metric count_input="$COUNT_INPUT"
    --metric count_sam3_mask="$COUNT_SAM3_MASK"
    --metric count_inpainting="$COUNT_INPAINT"
    --metric count_postprocess="$COUNT_POSTPROCESS"
    --metric count_egoblur="$COUNT_EGOBLUR"
    --path log_file="$LOG_FILE"
    --path events_file="$EVENTS_FILE"
  )

  if [ -n "$RESOLVED_INPUT_MODE" ]; then
    event_args+=( --param input_mode_resolved="$RESOLVED_INPUT_MODE" )
  fi
  if [ -n "$HOST_INPUT_DIR" ]; then
    event_args+=( --path local_input_dir="$HOST_INPUT_DIR" )
  fi
  if [ -n "$CONTAINER_INPUT_DIR" ]; then
    event_args+=( --path container_input_dir="$CONTAINER_INPUT_DIR" )
  fi
  if [ -n "$OUT_MASK" ]; then
    event_args+=( --path local_sam3_mask_dir="$OUT_MASK" )
  fi
  if [ -n "$OUT3" ]; then
    event_args+=( --path local_out_dir="$OUT3" )
  fi
  if [ -n "$FINAL_BATCH_DIR" ]; then
    event_args+=( --path final_out_dir="$FINAL_BATCH_DIR" )
  fi
  if [ "$final_status" = "failure" ]; then
    if [ -n "$FAILED_STAGE" ]; then
      event_args+=( --stage "$FAILED_STAGE" )
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

echo "RUN_ID=$RUN_ID"
echo "LOG_FILE=$LOG_FILE"
echo "EVENTS_FILE=$EVENTS_FILE"
echo "REPO=$REPO"
echo "SRC=$SRC"
echo "BATCH_NAME=$BATCH_NAME"
echo "CONTAINER_NAME=$CONTAINER_NAME"
echo "COMFYUI_DATA_DIR=$COMFYUI_DATA_DIR"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}"
echo "COMFY_PORT=${COMFY_PORT:-8188}"
echo "FORCE_REPROCESS=$FORCE_REPROCESS"
echo "STRICT_HARDLINK=$STRICT_HARDLINK"
echo "POSTPROCESS_WORKERS=$POSTPROCESS_WORKERS"
echo "PRIVACY_WORKERS=$PRIVACY_WORKERS"
echo "SAM3_WORKERS=$SAM3_WORKERS"
echo "SAM3_RESIZE_WIDTH=$SAM3_RESIZE_WIDTH"
echo "SAM3_RESIZE_HEIGHT=$SAM3_RESIZE_HEIGHT"
echo "SAM3_GLARE_THRESHOLD=$SAM3_GLARE_THRESHOLD"
echo "SAM3_TILE_ROWS=$SAM3_TILE_ROWS"
echo "SAM3_TILE_COLS=$SAM3_TILE_COLS"
echo "SAM3_SCRIPT=$SAM3_SCRIPT"
echo "LAPLACIAN_DILATION=$LAPLACIAN_DILATION"
echo "LAPLACIAN_BLUR=$LAPLACIAN_BLUR"
echo "LAPLACIAN_LEVELS=$LAPLACIAN_LEVELS"
echo "DOWNSTREAM_MODE=$DOWNSTREAM_MODE"
echo "COMFY_STOP_CONTAINERS=$COMFY_STOP_CONTAINERS"
echo "STOP_AFTER_STAGE=$STOP_AFTER_STAGE"
echo "COMFY_IMAGE=$COMFY_IMAGE"
echo "RESET_CONTAINER_BEFORE_RUN=$RESET_CONTAINER_BEFORE_RUN"
echo "INPUT_MODE=$INPUT_MODE"
echo "MODELS_COMFYUI_DIR=$MODELS_COMFYUI_DIR"
echo "MODELS_PRIVACY_DIR=$MODELS_PRIVACY_DIR"
echo "PRIVACY_FACE_MODEL=$PRIVACY_FACE_MODEL"
echo "PRIVACY_LP_MODEL=$PRIVACY_LP_MODEL"
echo "PRIVACY_OUTPUT_MODE=$PRIVACY_OUTPUT_MODE"
echo "TORCH_CACHE_DIR=$TORCH_CACHE_DIR"
echo "COMFY_READY_TIMEOUT=$COMFY_READY_TIMEOUT"
echo "COMFY_IMAGE_NODE_ID=$COMFY_IMAGE_NODE_ID"
echo "COMFY_MASK_NODE_ID=$COMFY_MASK_NODE_ID"
echo "COMFY_SAM3_MASK_NODE_ID=$COMFY_SAM3_MASK_NODE_ID"
echo "SKY_REFERENCE_SOURCE=$SKY_REFERENCE_SOURCE"
echo "SKY_REFERENCE_FILENAME=$SKY_REFERENCE_FILENAME"
if [ -n "$FINAL_OUTPUT_DIR" ]; then
  echo "FINAL_OUTPUT_DIR=$FINAL_OUTPUT_DIR"
fi

mkdir -p \
  "$COMFYUI_DATA_DIR/input" \
  "$COMFYUI_DATA_DIR/output" \
  "$COMFYUI_DATA_DIR/output-sam3-mask" \
  "$COMFYUI_DATA_DIR/output-postprocessed" \
  "$COMFYUI_DATA_DIR/output-egoblur" \
  "$MODELS_COMFYUI_DIR" \
  "$MODELS_PRIVACY_DIR"

RUN_START_ARGS=(
  --event run_start
  --status running
  --param input_mode="$INPUT_MODE"
  --param downstream_mode="$DOWNSTREAM_MODE"
  --param stop_after_stage="$STOP_AFTER_STAGE"
  --param force_reprocess="$FORCE_REPROCESS"
  --param strict_hardlink="$STRICT_HARDLINK"
  --param postprocess_workers="$POSTPROCESS_WORKERS"
  --param privacy_workers="$PRIVACY_WORKERS"
  --param sam3_workers="$SAM3_WORKERS"
  --param sam3_resize_width="$SAM3_RESIZE_WIDTH"
  --param sam3_resize_height="$SAM3_RESIZE_HEIGHT"
  --param sam3_glare_threshold="$SAM3_GLARE_THRESHOLD"
  --param sam3_tile_rows="$SAM3_TILE_ROWS"
  --param sam3_tile_cols="$SAM3_TILE_COLS"
  --param sam3_script="$SAM3_SCRIPT"
  --param laplacian_dilation="$LAPLACIAN_DILATION"
  --param laplacian_blur="$LAPLACIAN_BLUR"
  --param laplacian_levels="$LAPLACIAN_LEVELS"
  --param privacy_face_model="$PRIVACY_FACE_MODEL"
  --param privacy_lp_model="$PRIVACY_LP_MODEL"
  --param privacy_output_mode="$PRIVACY_OUTPUT_MODE"
  --param comfy_image_node_id="$COMFY_IMAGE_NODE_ID"
  --param comfy_mask_node_id="$COMFY_MASK_NODE_ID"
  --param comfy_sam3_mask_node_id="$COMFY_SAM3_MASK_NODE_ID"
  --path log_file="$LOG_FILE"
  --path events_file="$EVENTS_FILE"
  --path repo="$REPO"
  --path source_dir="$SRC"
  --path comfyui_data_dir="$COMFYUI_DATA_DIR"
  --path sam3_mask_dir="$COMFYUI_DATA_DIR/output-sam3-mask/$BATCH_NAME"
  --path inpainting_dir="$COMFYUI_DATA_DIR/output/$BATCH_NAME"
  --path postprocess_dir="$COMFYUI_DATA_DIR/output-postprocessed/$BATCH_NAME"
  --path egoblur_dir="$COMFYUI_DATA_DIR/output-egoblur/$BATCH_NAME"
)
if [ -n "$FINAL_OUTPUT_DIR" ]; then
  RUN_START_ARGS+=( --path final_out_dir="$FINAL_OUTPUT_DIR/$BATCH_NAME" )
fi
log_event "${RUN_START_ARGS[@]}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose is not available."
  exit 1
fi

if [[ "$RESET_CONTAINER_BEFORE_RUN" != "0" && "$RESET_CONTAINER_BEFORE_RUN" != "1" ]]; then
  echo "ERROR: RESET_CONTAINER_BEFORE_RUN must be 0 or 1"
  exit 1
fi

if [[ "$STRICT_HARDLINK" != "0" && "$STRICT_HARDLINK" != "1" ]]; then
  echo "ERROR: STRICT_HARDLINK must be 0 or 1"
  exit 1
fi

if [[ "$DOWNSTREAM_MODE" != "inline" && "$DOWNSTREAM_MODE" != "isolated" ]]; then
  echo "ERROR: Invalid DOWNSTREAM_MODE=$DOWNSTREAM_MODE (expected: inline|isolated)"
  exit 1
fi

if [[ "$STOP_AFTER_STAGE" != "sam3" && "$STOP_AFTER_STAGE" != "inpainting" && "$STOP_AFTER_STAGE" != "postprocess" && "$STOP_AFTER_STAGE" != "egoblur" ]]; then
  echo "ERROR: Invalid STOP_AFTER_STAGE=$STOP_AFTER_STAGE (expected: sam3|inpainting|postprocess|egoblur)"
  exit 1
fi

resolve_comfy_containers_to_stop() {
  local mode="$1"
  if [[ "$mode" != "auto" ]]; then
    tr ', ' '\n\n' <<<"$mode" | awk 'NF'
    return
  fi

  docker ps --format '{{.Names}}' | grep -E '^(comfyui-container|comfyui-g[0-9]+)$' || true
}

stop_comfy_containers_for_downstream() {
  local c
  local stopped=0
  mapfile -t STOP_LIST < <(resolve_comfy_containers_to_stop "$COMFY_STOP_CONTAINERS")

  if [ ${#STOP_LIST[@]} -eq 0 ]; then
    echo "No running ComfyUI containers found to stop."
    return
  fi

  echo "Stopping ComfyUI containers before downstream stages..."
  for c in "${STOP_LIST[@]}"; do
    if docker ps --format '{{.Names}}' | grep -Fxq "$c"; then
      echo " - stopping $c"
      docker stop "$c" >/dev/null
      stopped=$((stopped + 1))
    else
      echo " - $c not running (skip)"
    fi
  done
  echo "Stopped $stopped ComfyUI container(s)."
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required on host for helper steps."
  exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget is required (used by download-models.sh)."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found on host. NVIDIA drivers/GPU are required for this pipeline."
  exit 1
fi

docker_gpu_ready() {
  docker run --rm --gpus all "$NVIDIA_CUDA_TEST_IMAGE" nvidia-smi >/dev/null 2>&1
}

nvidia_toolkit_installed() {
  command -v nvidia-container-cli >/dev/null 2>&1
}

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: Need root/sudo to install NVIDIA Container Toolkit."
    return 1
  fi
}

install_nvidia_container_toolkit() {
  if nvidia_toolkit_installed; then
    echo "NVIDIA Container Toolkit already installed."
    return 0
  fi

  if [ ! -r /etc/os-release ]; then
    echo "ERROR: Cannot detect OS (missing /etc/os-release)."
    return 1
  fi

  . /etc/os-release
  if [ "${ID:-}" != "ubuntu" ] && [[ " ${ID_LIKE:-} " != *" debian "* ]] && [[ " ${ID_LIKE:-} " != *" ubuntu "* ]]; then
    echo "ERROR: Auto-install currently supports Ubuntu/Debian only."
    return 1
  fi

  echo "Installing NVIDIA Container Toolkit..."
  run_privileged rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
  run_privileged sh -c 'set -e; apt-get update; apt-get install -y curl gpg ca-certificates'
  run_privileged mkdir -p /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    run_privileged gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
  if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
      run_privileged tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  else
    wget -qO- https://nvidia.github.io/libnvidia-container/gpgkey | run_privileged apt-key add - 2>/dev/null || true
    echo "deb https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
      run_privileged tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  fi
  run_privileged apt-get update || true
  run_privileged apt-get install -y nvidia-container-toolkit || true
  run_privileged nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
  run_privileged systemctl restart docker 2>/dev/null || true
}

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "Skipping preflight checks (handled by orchestrator)"
else
  echo "Checking Docker GPU runtime"
  if docker_gpu_ready; then
    echo "Docker GPU runtime: OK"
  else
    echo "Docker GPU runtime: NOT READY"
    install_nvidia_container_toolkit
    if docker_gpu_ready; then
      echo "Docker GPU runtime: OK after toolkit installation"
    else
      echo "ERROR: Docker GPU runtime still not ready after installation attempt."
      echo "Please verify NVIDIA Container Toolkit + Docker runtime setup manually."
      exit 1
    fi
  fi

  required_files=(
    "$MODELS_COMFYUI_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    "$MODELS_COMFYUI_DIR/vae/qwen_image_vae.safetensors"
    "$MODELS_COMFYUI_DIR/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
    "$MODELS_COMFYUI_DIR/sam3/model.safetensors"
    "$MODELS_COMFYUI_DIR/lama/big-lama.pt"
    "$PRIVACY_FACE_MODEL"
    "$PRIVACY_LP_MODEL"
  )

  need_download=0
  for model_path in "${required_files[@]}"; do
    if [ ! -f "$model_path" ]; then
      need_download=1
      break
    fi
  done

  if [ "$need_download" = "1" ]; then
    if [ "$AUTO_DOWNLOAD_MODELS" = "1" ]; then
      echo "Required models missing. Running download-models.sh ..."
      MODELS_ROOT="$MODELS_ROOT" bash "$REPO/download-models.sh"
    else
      echo "ERROR: Required models are missing and AUTO_DOWNLOAD_MODELS=0"
      exit 1
    fi
  fi
fi

if [ ! -f "$COMFYUI_DATA_DIR/input/perspective_mask.png" ]; then
  cp "$REPO/inpainting-workflow-master/perspective_mask.png" "$COMFYUI_DATA_DIR/input/perspective_mask.png"
  echo "Copied perspective mask to $COMFYUI_DATA_DIR/input/perspective_mask.png"
fi

if [ ! -f "$SKY_REFERENCE_SOURCE" ]; then
  echo "ERROR: sky reference source not found: $SKY_REFERENCE_SOURCE"
  exit 1
fi
SKY_REFERENCE_TARGET="$COMFYUI_DATA_DIR/input/$SKY_REFERENCE_FILENAME"
cp "$SKY_REFERENCE_SOURCE" "$SKY_REFERENCE_TARGET"
echo "Staged sky reference image to $SKY_REFERENCE_TARGET"

if [ "$RESET_CONTAINER_BEFORE_RUN" = "1" ]; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Resetting existing container before run: $CONTAINER_NAME"
    reset_ok=0
    for attempt in 1 2 3; do
      if docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1; then
        reset_ok=1
        break
      fi
      echo "WARN: failed to remove container $CONTAINER_NAME (attempt $attempt/3), retrying..."
      sleep 3
    done
    if [ "$reset_ok" -ne 1 ]; then
      if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        echo "WARN: could not remove $CONTAINER_NAME; continuing and letting docker compose reconcile state."
      fi
    fi
  else
    echo "RESET_CONTAINER_BEFORE_RUN=1: no existing container to reset for $CONTAINER_NAME"
  fi
fi

echo "Ensuring container is up via docker compose"
CONTAINER_NAME="$CONTAINER_NAME" \
COMFYUI_DATA_DIR="$COMFYUI_DATA_DIR" \
MODELS_COMFYUI_DIR="$MODELS_COMFYUI_DIR" \
docker compose -p "${CONTAINER_NAME}" up -d

echo "Waiting for ComfyUI API readiness inside container"
docker exec "$CONTAINER_NAME" \
  python "$CONTAINER_PIPELINE_HELPERS" wait-http \
  --url http://127.0.0.1:8188/system_stats \
  --timeout "$COMFY_READY_TIMEOUT" \
  --poll "$COMFY_READY_POLL"

DST="$COMFYUI_DATA_DIR/input/$BATCH_NAME"
OUT1="$COMFYUI_DATA_DIR/output/$BATCH_NAME"
OUT_MASK="$COMFYUI_DATA_DIR/output-sam3-mask/$BATCH_NAME"
OUT2="$COMFYUI_DATA_DIR/output-postprocessed/$BATCH_NAME"
OUT3="$COMFYUI_DATA_DIR/output-egoblur/$BATCH_NAME"

if [ ! -d "$SRC" ]; then
  echo "ERROR: SRC directory not found: $SRC"
  exit 1
fi

HOST_INPUT_DIR="$SRC"
CONTAINER_INPUT_DIR="$SRC"
USE_STAGED_INPUT=0
RESOLVED_INPUT_MODE="direct"

if [ "$INPUT_MODE" = "staged" ]; then
  USE_STAGED_INPUT=1
elif [ "$INPUT_MODE" = "direct" ]; then
  if ! docker exec "$CONTAINER_NAME" test -d "$SRC" >/dev/null 2>&1; then
    echo "ERROR: INPUT_MODE=direct but SRC is not accessible inside container: $SRC"
    echo "Hint: set INPUT_MODE=staged to hardlink/copy from host path automatically."
    exit 1
  fi
elif [ "$INPUT_MODE" = "auto" ]; then
  if docker exec "$CONTAINER_NAME" test -d "$SRC" >/dev/null 2>&1; then
    USE_STAGED_INPUT=0
  else
    USE_STAGED_INPUT=1
  fi
else
  echo "ERROR: Invalid INPUT_MODE=$INPUT_MODE (expected: auto|direct|staged)"
  exit 1
fi

if [ "$USE_STAGED_INPUT" = "1" ]; then
  RESOLVED_INPUT_MODE="staged"
  HOST_INPUT_DIR="$DST"
  CONTAINER_INPUT_DIR="/workspace/ComfyUI/input/$BATCH_NAME"
  echo "Input mode: staged (hardlink SRC -> $DST)"
else
  RESOLVED_INPUT_MODE="direct"
  CONTAINER_INPUT_DIR="$SRC"
  echo "Input mode: direct (container SRC=$CONTAINER_INPUT_DIR)"
fi

if [ "$FORCE_REPROCESS" = "1" ]; then
  echo "FORCE_REPROCESS=1, clearing batch directories..."
  rm -rf "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"
else
  echo "FORCE_REPROCESS=0, preserving existing outputs for skip/resume behavior..."
fi
mkdir -p "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"

if [ "$USE_STAGED_INPUT" = "1" ]; then
  S_HARD=$(date +%s)
  HARDLINK_CMD=$(quote_cmd python3 "$PIPELINE_HELPERS" stage-images --src "$SRC" --dst "$DST" --strict-hardlink "$STRICT_HARDLINK")
  start_stage hardlink_stage "$HARDLINK_CMD"
  python3 "$PIPELINE_HELPERS" stage-images \
    --src "$SRC" \
    --dst "$DST" \
    --strict-hardlink "$STRICT_HARDLINK"
  E_HARD=$(date +%s)
  HARDLINK_SEC=$((E_HARD - S_HARD))
  finish_stage hardlink_stage "$HARDLINK_SEC"
else
  skip_stage hardlink_stage direct_input
fi

S_SAM3=$(date +%s)
SAM3_CMD=$(quote_cmd docker exec "$CONTAINER_NAME" python "/workspace/inpainting/$SAM3_SCRIPT" --input-dir "$CONTAINER_INPUT_DIR" --output-dir "/workspace/output-sam3-mask/$BATCH_NAME" --pattern "*" --glare-threshold "$SAM3_GLARE_THRESHOLD" --tile-rows "$SAM3_TILE_ROWS" --tile-cols "$SAM3_TILE_COLS" --resize-width "$SAM3_RESIZE_WIDTH" --resize-height "$SAM3_RESIZE_HEIGHT" --workers "$SAM3_WORKERS")
start_stage sam3_mask "$SAM3_CMD"
docker exec "$CONTAINER_NAME" python "/workspace/inpainting/$SAM3_SCRIPT" \
  --input-dir "$CONTAINER_INPUT_DIR" \
  --output-dir /workspace/output-sam3-mask/$BATCH_NAME \
  --pattern "*" \
  --glare-threshold "$SAM3_GLARE_THRESHOLD" \
  --tile-rows "$SAM3_TILE_ROWS" \
  --tile-cols "$SAM3_TILE_COLS" \
  --resize-width "$SAM3_RESIZE_WIDTH" \
  --resize-height "$SAM3_RESIZE_HEIGHT" \
  --workers "$SAM3_WORKERS"
E_SAM3=$(date +%s)
SAM3_SEC=$((E_SAM3 - S_SAM3))

SAM3_MASK_COUNT=$(python3 "$PIPELINE_HELPERS" count-images --path "$OUT_MASK" --include-bmp)
echo "sam3_mask_output_count=$SAM3_MASK_COUNT"
if [ "$SAM3_MASK_COUNT" -eq 0 ]; then
  fail_stage sam3_mask "No SAM3 mask outputs found in $OUT_MASK; aborting downstream stages."
fi
finish_stage sam3_mask "$SAM3_SEC" --metric output_count="$SAM3_MASK_COUNT"

if [ "$STOP_AFTER_STAGE" = "sam3" ]; then
  skip_stage inpainting STOP_AFTER_STAGE=sam3
  skip_stage postprocess STOP_AFTER_STAGE=sam3
  skip_stage egoblur STOP_AFTER_STAGE=sam3
else
  S_INP=$(date +%s)
  INPAINT_CMD=$(quote_cmd docker exec "$CONTAINER_NAME" python /workspace/inpainting/comfyui_run.py --workflow-json /workspace/workflow.json --input-dir "$CONTAINER_INPUT_DIR" --mask /workspace/ComfyUI/input/perspective_mask.png --sam3-mask-dir "/workspace/output-sam3-mask/$BATCH_NAME" --output-dir "/workspace/ComfyUI/output/$BATCH_NAME" --image-node-id "$COMFY_IMAGE_NODE_ID" --mask-node-id "$COMFY_MASK_NODE_ID" --sam3-mask-node-id "$COMFY_SAM3_MASK_NODE_ID" --workers 1 --timeout-s 3600)
  start_stage inpainting "$INPAINT_CMD"
  docker exec "$CONTAINER_NAME" python /workspace/inpainting/comfyui_run.py \
    --workflow-json /workspace/workflow.json \
    --input-dir "$CONTAINER_INPUT_DIR" \
    --mask /workspace/ComfyUI/input/perspective_mask.png \
    --sam3-mask-dir /workspace/output-sam3-mask/$BATCH_NAME \
    --output-dir /workspace/ComfyUI/output/$BATCH_NAME \
    --image-node-id "$COMFY_IMAGE_NODE_ID" \
    --mask-node-id "$COMFY_MASK_NODE_ID" \
    --sam3-mask-node-id "$COMFY_SAM3_MASK_NODE_ID" \
    --workers 1 \
    --timeout-s 3600
  E_INP=$(date +%s)
  INPAINT_SEC=$((E_INP - S_INP))

  INPAINT_COUNT=$(python3 "$PIPELINE_HELPERS" count-images --path "$OUT1")
  echo "inpainting_output_count=$INPAINT_COUNT"
  if [ "$INPAINT_COUNT" -eq 0 ]; then
    fail_stage inpainting "No inpainting outputs found in $OUT1; aborting downstream stages."
  fi
  finish_stage inpainting "$INPAINT_SEC" --metric output_count="$INPAINT_COUNT"

  if [ "$STOP_AFTER_STAGE" = "inpainting" ]; then
    skip_stage postprocess STOP_AFTER_STAGE=inpainting
    skip_stage egoblur STOP_AFTER_STAGE=inpainting
  else
    if [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
      : # privacy blur now runs inside container; no host-side setup needed
    fi

    if [ "$DOWNSTREAM_MODE" = "isolated" ]; then
      stop_comfy_containers_for_downstream
    fi

    S_POST=$(date +%s)
    if [ "$DOWNSTREAM_MODE" = "inline" ]; then
      POST_CMD=$(quote_cmd docker exec -u "$(id -u):$(id -g)" -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt "$CONTAINER_NAME" python /workspace/inpainting/postprocess.py -i "/workspace/ComfyUI/output/$BATCH_NAME" -o "/workspace/output-postprocessed/$BATCH_NAME" --top-mask /workspace/inpainting/sky_mask_updated.png --sam3-mask-dir "/workspace/output-sam3-mask/$BATCH_NAME" --dilation "$LAPLACIAN_DILATION" --blur "$LAPLACIAN_BLUR" --levels "$LAPLACIAN_LEVELS" --pattern "*.jpg" -j "$POSTPROCESS_WORKERS")
      start_stage postprocess "$POST_CMD"
      docker exec \
        -u "$(id -u):$(id -g)" \
        -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt \
        "$CONTAINER_NAME" \
        python /workspace/inpainting/postprocess.py \
        -i /workspace/ComfyUI/output/$BATCH_NAME \
        -o /workspace/output-postprocessed/$BATCH_NAME \
        --top-mask /workspace/inpainting/sky_mask_updated.png \
        --sam3-mask-dir /workspace/output-sam3-mask/$BATCH_NAME \
        --dilation "$LAPLACIAN_DILATION" \
        --blur "$LAPLACIAN_BLUR" \
        --levels "$LAPLACIAN_LEVELS" \
        --pattern "*.jpg" \
        -j "$POSTPROCESS_WORKERS"
    else
      POST_DOCKER_ARGS=(
        --rm
        --name "postprocess-${RUN_ID}"
        --gpus "device=${NVIDIA_VISIBLE_DEVICES:-0}"
        -u "$(id -u):$(id -g)"
        -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt
        -v "$REPO/inpainting-workflow-master:/workspace/inpainting"
        -v "$MODELS_COMFYUI_DIR:/workspace/ComfyUI/models:ro"
        -v "$REPO/p2e-local:/workspace/ComfyUI/custom_nodes/p2e"
        -v "$COMFYUI_DATA_DIR/output:/workspace/ComfyUI/output"
        -v "$COMFYUI_DATA_DIR/output-sam3-mask:/workspace/output-sam3-mask"
        -v "$COMFYUI_DATA_DIR/output-postprocessed:/workspace/output-postprocessed"
      )
      if [ -d "$TORCH_CACHE_DIR" ]; then
        POST_DOCKER_ARGS+=( -v "$TORCH_CACHE_DIR:/root/.cache/torch" )
      fi
      POST_CMD=$(quote_cmd docker run "${POST_DOCKER_ARGS[@]}" "$COMFY_IMAGE" python /workspace/inpainting/postprocess.py -i "/workspace/ComfyUI/output/$BATCH_NAME" -o "/workspace/output-postprocessed/$BATCH_NAME" --top-mask /workspace/inpainting/sky_mask_updated.png --sam3-mask-dir "/workspace/output-sam3-mask/$BATCH_NAME" --dilation "$LAPLACIAN_DILATION" --blur "$LAPLACIAN_BLUR" --levels "$LAPLACIAN_LEVELS" --pattern "*.jpg" -j "$POSTPROCESS_WORKERS")
      start_stage postprocess "$POST_CMD"
      docker run "${POST_DOCKER_ARGS[@]}" "$COMFY_IMAGE" \
        python /workspace/inpainting/postprocess.py \
        -i /workspace/ComfyUI/output/$BATCH_NAME \
        -o /workspace/output-postprocessed/$BATCH_NAME \
        --top-mask /workspace/inpainting/sky_mask_updated.png \
        --sam3-mask-dir /workspace/output-sam3-mask/$BATCH_NAME \
        --dilation "$LAPLACIAN_DILATION" \
        --blur "$LAPLACIAN_BLUR" \
        --levels "$LAPLACIAN_LEVELS" \
        --pattern "*.jpg" \
        -j "$POSTPROCESS_WORKERS"
    fi
    E_POST=$(date +%s)
    POSTPROCESS_SEC=$((E_POST - S_POST))
    finish_stage postprocess "$POSTPROCESS_SEC"

    if [ "$STOP_AFTER_STAGE" = "postprocess" ]; then
      skip_stage egoblur STOP_AFTER_STAGE=postprocess
    else
      S_EGO=$(date +%s)
      EGO_DOCKER_ARGS=(
        --rm
        --name "egoblur-${RUN_ID}"
        --gpus "device=${NVIDIA_VISIBLE_DEVICES:-0}"
        -v "$REPO/inpainting-workflow-master:/workspace/inpainting"
        -v "$MODELS_PRIVACY_DIR:/workspace/models/privacy_blur:ro"
        -v "$COMFYUI_DATA_DIR/output-postprocessed:/workspace/output-postprocessed"
        -v "$COMFYUI_DATA_DIR/output-egoblur:/workspace/output-egoblur"
      )
      EGO_CMD=(
        python /workspace/inpainting/privacy_blur_infer.py \
        --input-dir   /workspace/output-postprocessed/$BATCH_NAME \
        --output-dir  /workspace/output-egoblur/$BATCH_NAME \
        --face-model  /workspace/models/privacy_blur/face_yolov8n.pt \
        --lp-model    /workspace/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx \
        --face-conf   "$PRIVACY_FACE_CONF" \
        --lp-conf     "$PRIVACY_LP_CONF" \
        --face-iou    "$PRIVACY_FACE_IOU" \
        --face-imgsz  "$PRIVACY_FACE_IMGSZ" \
        --det-face-w  "$PRIVACY_DET_FACE_W" \
        --p360-device "$PRIVACY_P360_DEVICE" \
        --blur-scope  "$PRIVACY_BLUR_SCOPE" \
        --blur-backend "$PRIVACY_BLUR_BACKEND" \
        --output-mode "$PRIVACY_OUTPUT_MODE" \
        --workers     "$PRIVACY_WORKERS"
      )
      if [ "$FORCE_REPROCESS" = "1" ]; then
        EGO_CMD+=( --overwrite )
      fi
      EGO_CMD_STR=$(quote_cmd docker run "${EGO_DOCKER_ARGS[@]}" "$COMFY_IMAGE" "${EGO_CMD[@]}")
      start_stage egoblur "$EGO_CMD_STR"
      docker run "${EGO_DOCKER_ARGS[@]}" "$COMFY_IMAGE" "${EGO_CMD[@]}"

      E_EGO=$(date +%s)
      EGOBLUR_SEC=$((E_EGO - S_EGO))
      finish_stage egoblur "$EGOBLUR_SEC"
    fi
  fi
fi

if [ -n "$FINAL_OUTPUT_DIR" ] && [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
  FINAL_BATCH_DIR="$FINAL_OUTPUT_DIR/$BATCH_NAME"
  mkdir -p "$FINAL_BATCH_DIR"
  python3 "$PIPELINE_HELPERS" link-flat \
    --src "$OUT3" \
    --dst "$FINAL_BATCH_DIR" \
    --strict-hardlink "$STRICT_HARDLINK"
fi

S_COUNT=$(date +%s)
COUNTS_CMD=$(quote_cmd python3 "$PIPELINE_HELPERS" report-counts --input-dir "$HOST_INPUT_DIR" --sam3-dir "$OUT_MASK" --inpainting-dir "$OUT1" --postprocess-dir "$OUT2" --egoblur-dir "$OUT3")
start_stage counts "$COUNTS_CMD"
mapfile -t COUNT_LINES < <(python3 "$PIPELINE_HELPERS" report-counts \
  --input-dir "$HOST_INPUT_DIR" \
  --sam3-dir "$OUT_MASK" \
  --inpainting-dir "$OUT1" \
  --postprocess-dir "$OUT2" \
  --egoblur-dir "$OUT3")
for count_line in "${COUNT_LINES[@]}"; do
  echo "$count_line"
  case "$count_line" in
    count_input=*) COUNT_INPUT="${count_line#*=}" ;;
    count_sam3_mask=*) COUNT_SAM3_MASK="${count_line#*=}" ;;
    count_inpainting=*) COUNT_INPAINT="${count_line#*=}" ;;
    count_postprocess=*) COUNT_POSTPROCESS="${count_line#*=}" ;;
    count_egoblur=*) COUNT_EGOBLUR="${count_line#*=}" ;;
  esac
done
E_COUNT=$(date +%s)
COUNT_SEC=$((E_COUNT - S_COUNT))
finish_stage counts "$COUNT_SEC" \
  --metric count_input="$COUNT_INPUT" \
  --metric count_sam3_mask="$COUNT_SAM3_MASK" \
  --metric count_inpainting="$COUNT_INPAINT" \
  --metric count_postprocess="$COUNT_POSTPROCESS" \
  --metric count_egoblur="$COUNT_EGOBLUR"

RUN_STATUS="success"
echo "DONE"
echo "Events: $EVENTS_FILE"
echo "Log: $LOG_FILE"
