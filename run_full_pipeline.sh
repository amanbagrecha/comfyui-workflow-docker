#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-$SCRIPT_DIR}"
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"
# shellcheck disable=SC1091
. "$REPO/scripts/shell_helpers.sh"

require_repo_python

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
SRC="${SRC:-}"
BATCH_NAME="${BATCH_NAME:-batch-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_g${GPU_ID}_$$}"
PARENT_RUN_ID="${PARENT_RUN_ID:-}"

COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
WORKFLOW_JSON="${WORKFLOW_JSON:-$REPO/workflow-updated.json}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-$MODELS_ROOT/privacy_blur}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-$REPO/native_data}"
COMFY_DATA_DIR="${COMFY_DATA_DIR:-$NATIVE_DATA_ROOT/gpu${GPU_ID}}"
COMFY_PORT="${COMFY_PORT:-8180}"
COMFY_SERVER="${COMFY_SERVER:-http://127.0.0.1:${COMFY_PORT}}"
COMFY_INPUT_ROOT="${COMFY_INPUT_ROOT:-$COMFY_DATA_DIR/input}"
COMFY_OUTPUT_ROOT="${COMFY_OUTPUT_ROOT:-$COMFY_DATA_DIR/output}"
COMFY_TMUX_SESSION_PREFIX="${COMFY_TMUX_SESSION_PREFIX:-comfyui}"
COMFY_SESSION_NAME="${COMFY_SESSION_NAME:-${COMFY_TMUX_SESSION_PREFIX}-g${GPU_ID}}"

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
SEAM_WIDTH="${SEAM_WIDTH:-16}"
SEAM_FEATHER="${SEAM_FEATHER:-15}"
SEAM_SIGMA="${SEAM_SIGMA:-1.5}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
COMFY_READY_POLL="${COMFY_READY_POLL:-2}"

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

COMFY_IMAGE_NODE_ID="${COMFY_IMAGE_NODE_ID:-91}"
COMFY_MASK_NODE_ID="${COMFY_MASK_NODE_ID:-34}"
COMFY_SAM3_MASK_NODE_ID="${COMFY_SAM3_MASK_NODE_ID:-60}"
SKY_REFERENCE_SOURCE="${SKY_REFERENCE_SOURCE:-$REPO/inpainting-workflow-master/reference_sky.png}"
SKY_REFERENCE_FILENAME="${SKY_REFERENCE_FILENAME:-chrome_xWUjmfs7m4.png}"
LAMA_MODEL="${LAMA_MODEL:-$MODELS_COMFYUI_DIR/lama/big-lama.pt}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"

LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/fullrun_${RUN_ID}.log"
EVENTS_FILE="$LOG_DIR/fullrun_${RUN_ID}.events.jsonl"

setup_timestamped_log "$LOG_FILE"

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

COUNT_INPUT=0
COUNT_SAM3_MASK=0
COUNT_INPAINT=0
COUNT_POSTPROCESS=0
COUNT_EGOBLUR=0

HOST_INPUT_DIR="$SRC"
FINAL_BATCH_DIR=""
DST=""
OUT1=""
OUT_MASK=""
OUT2=""
OUT3=""
COMFY_STARTED_BY_THIS_RUN=0

EVENT_BASE_ARGS=(
  append-event
  --file "$EVENTS_FILE"
  --run-type full_pipeline
  --run-id "$RUN_ID"
  --script run_full_pipeline.sh
  --batch-name "$BATCH_NAME"
)
if [ -n "$PARENT_RUN_ID" ]; then
  EVENT_BASE_ARGS+=( --parent-run-id "$PARENT_RUN_ID" )
fi
if [ -n "$GPU_ID" ]; then
  EVENT_BASE_ARGS+=( --gpu-id "$GPU_ID" )
fi

log_event() {
  "$PYTHON_BIN" "$PIPELINE_HELPERS" "${EVENT_BASE_ARGS[@]}" "$@" >/dev/null 2>&1 || true
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
    --path comfy_input_root="$COMFY_INPUT_ROOT"
    --path comfy_output_root="$COMFY_OUTPUT_ROOT"
    --path comfy_data_dir="$COMFY_DATA_DIR"
  )

  if [ -n "$HOST_INPUT_DIR" ]; then
    event_args+=( --path source_dir="$HOST_INPUT_DIR" )
  fi
  if [ -n "$OUT_MASK" ]; then
    event_args+=( --path local_sam3_mask_dir="$OUT_MASK" )
  fi
  if [ -n "$OUT1" ]; then
    event_args+=( --path local_inpainting_dir="$OUT1" )
  fi
  if [ -n "$OUT2" ]; then
    event_args+=( --path local_postprocess_dir="$OUT2" )
  fi
  if [ -n "$OUT3" ]; then
    event_args+=( --path local_egoblur_dir="$OUT3" )
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

  if [ "$COMFY_STARTED_BY_THIS_RUN" = "1" ] && tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$COMFY_SESSION_NAME" >/dev/null 2>&1 || true
  fi
}

ensure_comfyui_service() {
  if "$PYTHON_BIN" "$PIPELINE_HELPERS" wait-http --url "$COMFY_SERVER/system_stats" --timeout 2 --poll 1 >/dev/null 2>&1; then
    return 0
  fi

  GPU_IDS="$GPU_ID" \
  MAX_GPUS=1 \
  BASE_COMFY_PORT="$COMFY_PORT" \
  COMFYUI_HOME="$COMFYUI_HOME" \
  NATIVE_DATA_ROOT="$NATIVE_DATA_ROOT" \
  TMUX_SESSION_PREFIX="$COMFY_TMUX_SESSION_PREFIX" \
  COMFY_READY_TIMEOUT="$COMFY_READY_TIMEOUT" \
  COMFY_READY_POLL="$COMFY_READY_POLL" \
  "$REPO/run_comfyui_cluster.sh"

  COMFY_STARTED_BY_THIS_RUN=1
}

stop_comfyui_service() {
  if tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$COMFY_SESSION_NAME"
  fi
}

trap 'on_error $? $LINENO "$BASH_COMMAND"' ERR
trap 'on_exit $?' EXIT

if [ -z "$SRC" ]; then
  echo "ERROR: SRC is required"
  exit 1
fi

if [ ! -d "$SRC" ]; then
  echo "ERROR: SRC directory not found: $SRC"
  exit 1
fi

validate_flag STRICT_HARDLINK "$STRICT_HARDLINK"
validate_flag FORCE_REPROCESS "$FORCE_REPROCESS"

if [[ "$STOP_AFTER_STAGE" != "sam3" && "$STOP_AFTER_STAGE" != "inpainting" && "$STOP_AFTER_STAGE" != "postprocess" && "$STOP_AFTER_STAGE" != "egoblur" ]]; then
  echo "ERROR: Invalid STOP_AFTER_STAGE=$STOP_AFTER_STAGE (expected: sam3|inpainting|postprocess|egoblur)"
  exit 1
fi

require_commands nvidia-smi wget tmux

if [ ! -d "$COMFYUI_HOME" ]; then
  echo "ERROR: COMFYUI_HOME not found: $COMFYUI_HOME"
  exit 1
fi

if [ ! -f "$WORKFLOW_JSON" ]; then
  echo "ERROR: workflow JSON not found: $WORKFLOW_JSON"
  exit 1
fi

if [ ! -f "$SKY_REFERENCE_SOURCE" ]; then
  echo "ERROR: sky reference source not found: $SKY_REFERENCE_SOURCE"
  exit 1
fi

if [ ! -f "$REPO/inpainting-workflow-master/perspective_mask.png" ]; then
  echo "ERROR: perspective_mask.png not found in inpainting-workflow-master"
  exit 1
fi

echo "RUN_ID=$RUN_ID GPU_ID=$GPU_ID BATCH_NAME=$BATCH_NAME"
echo "SRC=$SRC COMFY_SERVER=$COMFY_SERVER STOP_AFTER_STAGE=$STOP_AFTER_STAGE"
echo "MODELS_COMFYUI_DIR=$MODELS_COMFYUI_DIR MODELS_PRIVACY_DIR=$MODELS_PRIVACY_DIR"

mkdir -p \
  "$COMFY_INPUT_ROOT" \
  "$COMFY_OUTPUT_ROOT" \
  "$COMFY_DATA_DIR/output-sam3-mask" \
  "$COMFY_DATA_DIR/output-postprocessed" \
  "$COMFY_DATA_DIR/output-egoblur" \
  "$MODELS_COMFYUI_DIR" \
  "$MODELS_PRIVACY_DIR"

RUN_START_ARGS=(
  --event run_start
  --status running
  --param comfy_server="$COMFY_SERVER"
  --param comfy_port="$COMFY_PORT"
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
  --param seam_width="$SEAM_WIDTH"
  --param seam_feather="$SEAM_FEATHER"
  --param seam_sigma="$SEAM_SIGMA"
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
  --path comfyui_home="$COMFYUI_HOME"
  --path comfy_input_root="$COMFY_INPUT_ROOT"
  --path comfy_output_root="$COMFY_OUTPUT_ROOT"
  --path comfy_data_dir="$COMFY_DATA_DIR"
  --path sam3_mask_dir="$COMFY_DATA_DIR/output-sam3-mask/$BATCH_NAME"
  --path inpainting_dir="$COMFY_OUTPUT_ROOT/$BATCH_NAME"
  --path postprocess_dir="$COMFY_DATA_DIR/output-postprocessed/$BATCH_NAME"
  --path egoblur_dir="$COMFY_DATA_DIR/output-egoblur/$BATCH_NAME"
)
if [ -n "$FINAL_OUTPUT_DIR" ]; then
  RUN_START_ARGS+=( --path final_out_dir="$FINAL_OUTPUT_DIR/$BATCH_NAME" )
fi
log_event "${RUN_START_ARGS[@]}"

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  mapfile -t required_files < <(
    required_model_paths \
      "$MODELS_COMFYUI_DIR" \
      "$MODELS_PRIVACY_DIR" \
      "$LAMA_MODEL" \
      "$PRIVACY_FACE_MODEL" \
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
      MODELS_ROOT="$MODELS_ROOT" \
      COMFY_MODELS_DIR="$MODELS_COMFYUI_DIR" \
      PRIVACY_MODELS_DIR="$MODELS_PRIVACY_DIR" \
      bash "$REPO/download-models.sh"
    else
      echo "ERROR: Required models are missing and AUTO_DOWNLOAD_MODELS=0"
      exit 1
    fi
  fi
fi

if [ ! -f "$COMFY_INPUT_ROOT/perspective_mask.png" ]; then
  cp "$REPO/inpainting-workflow-master/perspective_mask.png" "$COMFY_INPUT_ROOT/perspective_mask.png"
  echo "Copied perspective mask to $COMFY_INPUT_ROOT/perspective_mask.png"
fi

SKY_REFERENCE_TARGET="$COMFY_INPUT_ROOT/$SKY_REFERENCE_FILENAME"
cp "$SKY_REFERENCE_SOURCE" "$SKY_REFERENCE_TARGET"
echo "Staged sky reference image to $SKY_REFERENCE_TARGET"

DST="$COMFY_INPUT_ROOT/$BATCH_NAME"
OUT1="$COMFY_OUTPUT_ROOT/$BATCH_NAME"
OUT_MASK="$COMFY_DATA_DIR/output-sam3-mask/$BATCH_NAME"
OUT2="$COMFY_DATA_DIR/output-postprocessed/$BATCH_NAME"
OUT3="$COMFY_DATA_DIR/output-egoblur/$BATCH_NAME"

if [ "$FORCE_REPROCESS" = "1" ]; then
  echo "FORCE_REPROCESS=1, clearing batch directories..."
  rm -rf "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"
else
  echo "FORCE_REPROCESS=0, preserving existing outputs for skip/resume behavior..."
fi
mkdir -p "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"

S_HARD=$(date +%s)
HARDLINK_CMD=$(quote_cmd "$PYTHON_BIN" "$PIPELINE_HELPERS" stage-images --src "$SRC" --dst "$DST" --strict-hardlink "$STRICT_HARDLINK")
start_stage hardlink_stage "$HARDLINK_CMD"
HARDLINK_OUTPUT=$("$PYTHON_BIN" "$PIPELINE_HELPERS" stage-images \
  --src "$SRC" \
  --dst "$DST" \
  --strict-hardlink "$STRICT_HARDLINK")
echo "$HARDLINK_OUTPUT"
E_HARD=$(date +%s)
HARDLINK_SEC=$((E_HARD - S_HARD))
SKIPPED_INVALID=$(echo "$HARDLINK_OUTPUT" | grep -oP 'skipped_invalid=\K\d+' || echo "0")
finish_stage hardlink_stage "$HARDLINK_SEC" --metric skipped_invalid="$SKIPPED_INVALID"

S_SAM3=$(date +%s)
SAM3_CMD=$(quote_cmd "$PYTHON_BIN" "$REPO/inpainting-workflow-master/$SAM3_SCRIPT" --input-dir "$DST" --output-dir "$OUT_MASK" --pattern '*' --model-path "$MODELS_COMFYUI_DIR/sam3" --glare-threshold "$SAM3_GLARE_THRESHOLD" --tile-rows "$SAM3_TILE_ROWS" --tile-cols "$SAM3_TILE_COLS" --resize-width "$SAM3_RESIZE_WIDTH" --resize-height "$SAM3_RESIZE_HEIGHT" --workers "$SAM3_WORKERS")
start_stage sam3_mask "$SAM3_CMD"
"$PYTHON_BIN" "$REPO/inpainting-workflow-master/$SAM3_SCRIPT" \
  --input-dir "$DST" \
  --output-dir "$OUT_MASK" \
  --pattern "*" \
  --model-path "$MODELS_COMFYUI_DIR/sam3" \
  --glare-threshold "$SAM3_GLARE_THRESHOLD" \
  --tile-rows "$SAM3_TILE_ROWS" \
  --tile-cols "$SAM3_TILE_COLS" \
  --resize-width "$SAM3_RESIZE_WIDTH" \
  --resize-height "$SAM3_RESIZE_HEIGHT" \
  --workers "$SAM3_WORKERS"
E_SAM3=$(date +%s)
SAM3_SEC=$((E_SAM3 - S_SAM3))

COUNT_SAM3_MASK=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$OUT_MASK" --include-bmp)
echo "sam3_mask_output_count=$COUNT_SAM3_MASK"
if [ "$COUNT_SAM3_MASK" -eq 0 ]; then
  fail_stage sam3_mask "No SAM3 mask outputs found in $OUT_MASK; aborting downstream stages."
fi
finish_stage sam3_mask "$SAM3_SEC" --metric output_count="$COUNT_SAM3_MASK"

if [ "$STOP_AFTER_STAGE" = "sam3" ]; then
  skip_stage inpainting STOP_AFTER_STAGE=sam3
  skip_stage postprocess STOP_AFTER_STAGE=sam3
  skip_stage egoblur STOP_AFTER_STAGE=sam3
else
  S_WAIT=$(date +%s)
  WAIT_CMD=$(quote_cmd ensure_comfyui_service)
  start_stage wait_comfyui "$WAIT_CMD"
  ensure_comfyui_service
  E_WAIT=$(date +%s)
  finish_stage wait_comfyui "$((E_WAIT - S_WAIT))"

  S_INP=$(date +%s)
  INPAINT_CMD=$(quote_cmd "$PYTHON_BIN" "$REPO/inpainting-workflow-master/comfyui_run.py" --workflow-json "$WORKFLOW_JSON" --server "$COMFY_SERVER" --input-dir "$DST" --mask "$COMFY_INPUT_ROOT/perspective_mask.png" --sam3-mask-dir "$OUT_MASK" --output-dir "$OUT1" --image-node-id "$COMFY_IMAGE_NODE_ID" --mask-node-id "$COMFY_MASK_NODE_ID" --sam3-mask-node-id "$COMFY_SAM3_MASK_NODE_ID" --workers 1 --timeout-s 3600 --comfy-input-root "$COMFY_INPUT_ROOT" --comfy-output-root "$COMFY_OUTPUT_ROOT")
  start_stage inpainting "$INPAINT_CMD"
  "$PYTHON_BIN" "$REPO/inpainting-workflow-master/comfyui_run.py" \
    --workflow-json "$WORKFLOW_JSON" \
    --server "$COMFY_SERVER" \
    --input-dir "$DST" \
    --mask "$COMFY_INPUT_ROOT/perspective_mask.png" \
    --sam3-mask-dir "$OUT_MASK" \
    --output-dir "$OUT1" \
    --image-node-id "$COMFY_IMAGE_NODE_ID" \
    --mask-node-id "$COMFY_MASK_NODE_ID" \
    --sam3-mask-node-id "$COMFY_SAM3_MASK_NODE_ID" \
    --workers 1 \
    --timeout-s 3600 \
    --comfy-input-root "$COMFY_INPUT_ROOT" \
    --comfy-output-root "$COMFY_OUTPUT_ROOT"
  E_INP=$(date +%s)
  INPAINT_SEC=$((E_INP - S_INP))

  COUNT_INPAINT=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$OUT1")
  echo "inpainting_output_count=$COUNT_INPAINT"
  if [ "$COUNT_INPAINT" -eq 0 ]; then
    fail_stage inpainting "No inpainting outputs found in $OUT1; aborting downstream stages."
  fi
  finish_stage inpainting "$INPAINT_SEC" --metric output_count="$COUNT_INPAINT"

  S_STOP=$(date +%s)
  STOP_CMD=$(quote_cmd stop_comfyui_service)
  start_stage stop_comfyui "$STOP_CMD"
  stop_comfyui_service
  COMFY_STARTED_BY_THIS_RUN=0
  E_STOP=$(date +%s)
  finish_stage stop_comfyui "$((E_STOP - S_STOP))"

  if [ "$STOP_AFTER_STAGE" = "inpainting" ]; then
    skip_stage postprocess STOP_AFTER_STAGE=inpainting
    skip_stage egoblur STOP_AFTER_STAGE=inpainting
  else
    S_POST=$(date +%s)
    POST_CMD=$(quote_cmd env LAMA_MODEL="$LAMA_MODEL" "$PYTHON_BIN" "$REPO/inpainting-workflow-master/postprocess.py" -i "$OUT1" -o "$OUT2" --top-mask "$REPO/inpainting-workflow-master/sky_mask_updated.png" --sam3-mask-dir "$OUT_MASK" --dilation "$LAPLACIAN_DILATION" --blur "$LAPLACIAN_BLUR" --levels "$LAPLACIAN_LEVELS" --seam-width "$SEAM_WIDTH" --seam-feather "$SEAM_FEATHER" --mask-sigma "$SEAM_SIGMA" --pattern '*.jpg' -j "$POSTPROCESS_WORKERS")
    start_stage postprocess "$POST_CMD"
    LAMA_MODEL="$LAMA_MODEL" "$PYTHON_BIN" "$REPO/inpainting-workflow-master/postprocess.py" \
      -i "$OUT1" \
      -o "$OUT2" \
      --top-mask "$REPO/inpainting-workflow-master/sky_mask_updated.png" \
      --sam3-mask-dir "$OUT_MASK" \
      --dilation "$LAPLACIAN_DILATION" \
      --blur "$LAPLACIAN_BLUR" \
      --levels "$LAPLACIAN_LEVELS" \
      --seam-width "$SEAM_WIDTH" \
      --seam-feather "$SEAM_FEATHER" \
      --mask-sigma "$SEAM_SIGMA" \
      --pattern "*.jpg" \
      -j "$POSTPROCESS_WORKERS"
    E_POST=$(date +%s)
    POSTPROCESS_SEC=$((E_POST - S_POST))

    COUNT_POSTPROCESS=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$OUT2")
    echo "postprocess_output_count=$COUNT_POSTPROCESS"
    if [ "$COUNT_POSTPROCESS" -eq 0 ]; then
      fail_stage postprocess "No postprocess outputs found in $OUT2; aborting downstream stages."
    fi
    finish_stage postprocess "$POSTPROCESS_SEC" --metric output_count="$COUNT_POSTPROCESS"

    if [ "$STOP_AFTER_STAGE" = "postprocess" ]; then
      skip_stage egoblur STOP_AFTER_STAGE=postprocess
    else
      S_EGO=$(date +%s)
      EGO_CMD=(
        "$PYTHON_BIN" "$REPO/inpainting-workflow-master/privacy_blur_infer.py"
        --input-dir "$OUT2"
        --output-dir "$OUT3"
        --face-model "$PRIVACY_FACE_MODEL"
        --lp-model "$PRIVACY_LP_MODEL"
        --face-conf "$PRIVACY_FACE_CONF"
        --lp-conf "$PRIVACY_LP_CONF"
        --face-iou "$PRIVACY_FACE_IOU"
        --face-imgsz "$PRIVACY_FACE_IMGSZ"
        --det-face-w "$PRIVACY_DET_FACE_W"
        --p360-device "$PRIVACY_P360_DEVICE"
        --blur-scope "$PRIVACY_BLUR_SCOPE"
        --blur-backend "$PRIVACY_BLUR_BACKEND"
        --output-mode "$PRIVACY_OUTPUT_MODE"
        --workers "$PRIVACY_WORKERS"
      )
      if [ "$FORCE_REPROCESS" = "1" ]; then
        EGO_CMD+=( --overwrite )
      fi
      start_stage egoblur "$(quote_cmd "${EGO_CMD[@]}")"
      "${EGO_CMD[@]}"
      E_EGO=$(date +%s)
      EGOBLUR_SEC=$((E_EGO - S_EGO))

      COUNT_EGOBLUR=$("$PYTHON_BIN" "$PIPELINE_HELPERS" count-images --path "$OUT3")
      echo "egoblur_output_count=$COUNT_EGOBLUR"
      if [ "$COUNT_EGOBLUR" -eq 0 ]; then
        fail_stage egoblur "No privacy blur outputs found in $OUT3."
      fi
      finish_stage egoblur "$EGOBLUR_SEC" --metric output_count="$COUNT_EGOBLUR"
    fi
  fi
fi

if [ -n "$FINAL_OUTPUT_DIR" ] && [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
  FINAL_BATCH_DIR="$FINAL_OUTPUT_DIR/$BATCH_NAME"
  mkdir -p "$FINAL_BATCH_DIR"
  "$PYTHON_BIN" "$PIPELINE_HELPERS" link-flat \
    --src "$OUT3" \
    --dst "$FINAL_BATCH_DIR" \
    --strict-hardlink "$STRICT_HARDLINK"
fi

S_COUNT=$(date +%s)
COUNTS_CMD=$(quote_cmd "$PYTHON_BIN" "$PIPELINE_HELPERS" report-counts --input-dir "$HOST_INPUT_DIR" --sam3-dir "$OUT_MASK" --inpainting-dir "$OUT1" --postprocess-dir "$OUT2" --egoblur-dir "$OUT3")
start_stage counts "$COUNTS_CMD"
mapfile -t COUNT_LINES < <("$PYTHON_BIN" "$PIPELINE_HELPERS" report-counts \
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
