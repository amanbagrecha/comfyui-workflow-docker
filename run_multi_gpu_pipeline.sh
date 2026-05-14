#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  SRC=/abs/path/to/images ./run_multi_gpu_pipeline.sh

Environment variables:
  SRC                       Required. Source directory with input images.
  RUN_NAME                  Optional. Default: multigpu-YYYYmmdd_HHMMSS
  GPU_IDS                   Optional. Comma/space list (e.g. "0,1,2") or "auto".
  MAX_GPUS                  Optional. 0 means all detected GPUs.
  BASE_COMFY_PORT           Optional. Default: 8180
  NATIVE_DATA_ROOT          Optional. Default: <repo>/native_data
  COMFYUI_HOME              Optional. Default: <repo>/ComfyUI
  TMUX_SESSION_PREFIX       Optional. Default: mgpu
  WORK_ROOT                 Optional. Default: <repo>/tmp/multigpu/<RUN_NAME>
  WAIT_POLL_SEC             Optional. Default: 10
  FINAL_OUTPUT_DIR          Optional. If set, writes merged outputs under
                            FINAL_OUTPUT_DIR/<RUN_NAME>/gpu<id>/
  STRICT_HARDLINK           Optional. 1 = fail when hardlink is not possible.
                            0 = allow copy fallback. Default: 1.
  SKIP_HOST_BOOTSTRAP       Optional. 1 = assume the runtime is already baked
                            into the machine and skip setup_host_environment.sh.
  DRY_RUN                   Optional. 1 = prepare/print plan only, no launches.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"
# shellcheck disable=SC1091
. "$REPO/scripts/shell_helpers.sh"

ensure_uv
ensure_repo_python

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
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-$REPO/native_data}"
COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-mgpu}"
WORK_ROOT="${WORK_ROOT:-$REPO/tmp/multigpu/$RUN_NAME}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-10}"
DRY_RUN="${DRY_RUN:-0}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
SKIP_HOST_BOOTSTRAP="${SKIP_HOST_BOOTSTRAP:-0}"

STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"

LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/multigpu_${RUN_NAME}.log"
EVENTS_FILE="$LOG_DIR/multigpu_${RUN_NAME}.events.jsonl"
mkdir -p "$LOG_DIR" "$WORK_ROOT/shards" "$WORK_ROOT/jobs"

setup_timestamped_log "$LOG_FILE"

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
  "$PYTHON_BIN" "$PIPELINE_HELPERS" "${EVENT_BASE_ARGS[@]}" "$@" >/dev/null 2>&1 || true
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

validate_non_negative_int MAX_GPUS "$MAX_GPUS"
validate_min_int WAIT_POLL_SEC "$WAIT_POLL_SEC" 1
validate_flag STRICT_HARDLINK "$STRICT_HARDLINK"
validate_flag DRY_RUN "$DRY_RUN"
validate_flag SKIP_HOST_BOOTSTRAP "$SKIP_HOST_BOOTSTRAP"
require_commands nvidia-smi tmux wget

RUN_FULL="$REPO/run_full_pipeline.sh"
if [[ ! -x "$RUN_FULL" ]]; then
  echo "ERROR: run_full_pipeline.sh not found or not executable at $RUN_FULL"
  exit 1
fi

RUN_BOOTSTRAP="$REPO/setup_host_environment.sh"
if [[ "$SKIP_HOST_BOOTSTRAP" != "1" && ! -x "$RUN_BOOTSTRAP" ]]; then
  echo "ERROR: setup_host_environment.sh not found or not executable at $RUN_BOOTSTRAP"
  exit 1
fi

mapfile -t GPU_LIST < <(resolve_gpu_ids "$GPU_IDS_RAW")
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs detected/resolved."
  exit 1
fi

if [[ "$MAX_GPUS" -gt 0 && ${#GPU_LIST[@]} -gt "$MAX_GPUS" ]]; then
  GPU_LIST=("${GPU_LIST[@]:0:$MAX_GPUS}")
fi

NUM_GPUS=${#GPU_LIST[@]}
MANIFEST_JSON="$WORK_ROOT/shard_manifest.json"
COUNT_JSON="$WORK_ROOT/shard_counts.json"

GPU_IDS_CSV=$(IFS=,; printf '%s' "${GPU_LIST[*]}")
log_event \
  --event run_start \
  --status running \
  --param gpu_ids="$GPU_IDS_CSV" \
  --param stop_after_stage="$STOP_AFTER_STAGE" \
  --param strict_hardlink="$STRICT_HARDLINK" \
  --param skip_host_bootstrap="$SKIP_HOST_BOOTSTRAP" \
  --param dry_run="$DRY_RUN" \
  --metric num_gpus="$NUM_GPUS" \
  --path log_file="$LOG_FILE" \
  --path events_file="$EVENTS_FILE" \
  --path work_root="$WORK_ROOT" \
  --path source_dir="$SRC" \
  --path comfyui_home="$COMFYUI_HOME" \
  --path native_data_root="$NATIVE_DATA_ROOT" \
  --path manifest_json="$MANIFEST_JSON" \
  --path count_json="$COUNT_JSON"

echo "RUN_NAME=$RUN_NAME SRC=$SRC GPU_IDS=${GPU_LIST[*]}"
echo "WORK_ROOT=$WORK_ROOT STOP_AFTER_STAGE=$STOP_AFTER_STAGE DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "0" && "$SKIP_HOST_BOOTSTRAP" != "1" ]]; then
  set_step bootstrap_host
  MODELS_ROOT="${MODELS_ROOT:-$REPO/models}" \
  MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-${MODELS_ROOT:-$REPO/models}/comfyui}" \
  MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}" \
  COMFYUI_HOME="$COMFYUI_HOME" \
  DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}" \
  INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}" \
  "$RUN_BOOTSTRAP"
  require_repo_python
  clear_step

  if [[ ! -d "$COMFYUI_HOME" ]]; then
    echo "ERROR: COMFYUI_HOME not found after bootstrap: $COMFYUI_HOME"
    exit 1
  fi
else
  require_repo_python
fi

set_step split_shards
"$PYTHON_BIN" "$PIPELINE_HELPERS" split-shards \
  --src "$SRC" \
  --shards-root "$WORK_ROOT/shards" \
  --num-gpus "$NUM_GPUS" \
  --manifest-json "$MANIFEST_JSON" \
  --count-json "$COUNT_JSON" \
  --strict-hardlink "$STRICT_HARDLINK"
clear_step

set_step preflight
if [[ "$DRY_RUN" == "0" ]]; then
  _models_root="${MODELS_ROOT:-$REPO/models}"
  _models_comfyui="${MODELS_COMFYUI_DIR:-$_models_root/comfyui}"
  _models_privacy="${MODELS_PRIVACY_DIR:-$_models_root/privacy_blur}"
  mapfile -t _required_models < <(
    required_model_paths \
      "$_models_comfyui" \
      "$_models_privacy" \
      "$_models_comfyui/lama/big-lama.pt" \
      "$_models_privacy/face_yolov8n.pt" \
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
      MODELS_ROOT="$_models_root" \
      COMFY_MODELS_DIR="$_models_comfyui" \
      PRIVACY_MODELS_DIR="$_models_privacy" \
      bash "$REPO/download-models.sh"
    else
      echo "ERROR: Required models are missing and AUTO_DOWNLOAD_MODELS=0"
      exit 1
    fi
  else
    echo "All required models present."
  fi

else
  echo "DRY_RUN=1: skipping dependency install and model checks."
fi
clear_step

LAUNCH_PLAN="$WORK_ROOT/launch_plan.tsv"
: > "$LAUNCH_PLAN"

set_step launch_shards
for idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$idx]}"
  shard_dir="$WORK_ROOT/shards/gpu${idx}"
  shard_count=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$shard_dir" --include-bmp)

  if [[ "$shard_count" -eq 0 ]]; then
    echo "Skipping GPU $gpu_id (empty shard)"
    continue
  fi

  comfy_port=$((BASE_COMFY_PORT + idx))
  batch_name="${RUN_NAME}-g${gpu_id}"
  child_run_id="${RUN_NAME}_g${gpu_id}"
  comfy_data_dir="$NATIVE_DATA_ROOT/gpu${gpu_id}"
  session_name="${TMUX_SESSION_PREFIX}-${RUN_NAME}-g${gpu_id}"
  rc_file="$WORK_ROOT/rc_gpu${gpu_id}.txt"
  child_log_file="$LOG_DIR/fullrun_${child_run_id}.log"
  child_events_file="$LOG_DIR/fullrun_${child_run_id}.events.jsonl"
  job_script="$WORK_ROOT/jobs/gpu${gpu_id}.sh"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"
export PATH="$HOME/.local/bin:\$PATH"
export SRC="$shard_dir"
export GPU_ID="$gpu_id"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export BATCH_NAME="$batch_name"
export COMFY_PORT="$comfy_port"
export COMFY_SERVER="http://127.0.0.1:$comfy_port"
export COMFYUI_HOME="$COMFYUI_HOME"
export COMFY_TMUX_SESSION_PREFIX="comfyui"
export NATIVE_DATA_ROOT="$NATIVE_DATA_ROOT"
export COMFY_DATA_DIR="$comfy_data_dir"
export COMFY_INPUT_ROOT="$comfy_data_dir/input"
export COMFY_OUTPUT_ROOT="$comfy_data_dir/output"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
export MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-${MODELS_ROOT:-$REPO/models}/comfyui}"
export MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}"
export AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
export FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
export STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
export STOP_AFTER_STAGE="$STOP_AFTER_STAGE"
export FINAL_OUTPUT_DIR=""
export SAM3_WORKERS="${SAM3_WORKERS:-2}"
export SAM3_RESIZE_WIDTH="${SAM3_RESIZE_WIDTH:-4000}"
export SAM3_RESIZE_HEIGHT="${SAM3_RESIZE_HEIGHT:-2000}"
export SAM3_GLARE_RESIZE_WIDTH="${SAM3_GLARE_RESIZE_WIDTH:-2000}"
export SAM3_GLARE_RESIZE_HEIGHT="${SAM3_GLARE_RESIZE_HEIGHT:-1000}"
export SAM3_GLARE_THRESHOLD="${SAM3_GLARE_THRESHOLD:-0.4}"
export SAM3_GLARE_DILATION="${SAM3_GLARE_DILATION:-5}"
export SAM3_TILE_ROWS="${SAM3_TILE_ROWS:-1}"
export SAM3_TILE_COLS="${SAM3_TILE_COLS:-2}"
export SAM3_SCRIPT="${SAM3_SCRIPT:-sam3_tiled_mask.py}"
export POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-3}"
export LAPLACIAN_DILATION="${LAPLACIAN_DILATION:-1}"
export LAPLACIAN_BLUR="${LAPLACIAN_BLUR:-10}"
export LAPLACIAN_LEVELS="${LAPLACIAN_LEVELS:-7}"
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export SKIP_PREFLIGHT=1
export RUN_ID="$child_run_id"
export PARENT_RUN_ID="$RUN_NAME"
export PERSPECTIVE_MASK="${PERSPECTIVE_MASK:-}"
export WORKFLOW_JSON="${WORKFLOW_JSON:-$REPO/workflow-updated.json}"

"$RUN_FULL"
rc=\$?
printf '%s\n' "\$rc" > "$rc_file"
exit "\$rc"
EOF

  chmod +x "$job_script"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$gpu_id" "$idx" "$batch_name" "$comfy_data_dir" "$comfy_port" "$session_name" "$rc_file" "$child_run_id" "$child_log_file" "$child_events_file" >> "$LAUNCH_PLAN"

  echo "Planned GPU=$gpu_id shard_idx=$idx count=$shard_count batch=$batch_name port=$comfy_port session=$session_name"
  log_event \
    --event shard_planned \
    --status success \
    --gpu-id "$gpu_id" \
    --shard-index "$idx" \
    --child-run-id "$child_run_id" \
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
    --batch-name "$batch_name" \
    --path child_log_file="$child_log_file" \
    --path child_events_file="$child_events_file"
done
clear_step

final_stage_dir() {
  local data_dir="$1"
  local batch="$2"
  stage_output_dir "$data_dir" "$batch" "$STOP_AFTER_STAGE"
}

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

while IFS=$'\t' read -r gpu_id idx batch comfy_data_dir _ _ rc_file child_run_id child_log_file child_events_file; do
  rc="missing"
  if [[ -f "$rc_file" ]]; then
    rc="$(tr -d '[:space:]' < "$rc_file")"
  fi

  in_count=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$WORK_ROOT/shards/gpu${idx}" --include-bmp)

  out_dir="$(final_stage_dir "$comfy_data_dir" "$batch")"

  out_count=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$out_dir" --include-bmp)

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

  while IFS=$'\t' read -r gpu_id _ batch comfy_data_dir _ _ _ _ _ _; do
    src_dir="$(final_stage_dir "$comfy_data_dir" "$batch")"

    "$PYTHON_BIN" "$PIPELINE_HELPERS" link-flat \
      --src "$src_dir" \
      --dst "$MERGED_ROOT" \
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
