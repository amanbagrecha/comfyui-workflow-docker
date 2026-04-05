#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"

RUN_ID="${RUN_ID:-native-$(date +%Y%m%d_%H%M%S)}"
BATCH_NAME="${BATCH_NAME:-$RUN_ID}"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  echo "ERROR: SRC is required"
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: SRC not found: $SRC"
  exit 1
fi

NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_SERVER="${COMFY_SERVER:-http://127.0.0.1:${COMFY_PORT}}"
COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
WORKFLOW_JSON="${WORKFLOW_JSON:-$REPO/workflow-updated.json}"

MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-$MODELS_ROOT/privacy_blur}"

COMFY_INPUT_ROOT="${COMFY_INPUT_ROOT:-$COMFYUI_HOME/input}"
COMFY_OUTPUT_ROOT="${COMFY_OUTPUT_ROOT:-$COMFYUI_HOME/output}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-$REPO/native_data}"
COMFY_DATA_DIR="${COMFY_DATA_DIR:-$NATIVE_DATA_ROOT/gpu${NVIDIA_VISIBLE_DEVICES}}"

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

FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-}"

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

LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/fullrun_native_${RUN_ID}.log"
EVENTS_FILE="$LOG_DIR/fullrun_native_${RUN_ID}.events.jsonl"
mkdir -p "$LOG_DIR"
exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

for cmd in python3 nvidia-smi tmux; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: missing $cmd"; exit 1; }
done

if [[ ! -d "$COMFYUI_HOME" ]]; then
  echo "ERROR: COMFYUI_HOME not found: $COMFYUI_HOME"
  exit 1
fi

python3 "$PIPELINE_HELPERS" wait-http \
  --url "$COMFY_SERVER/system_stats" \
  --timeout "${COMFY_READY_TIMEOUT:-300}" \
  --poll "${COMFY_READY_POLL:-2}"

mkdir -p "$COMFY_INPUT_ROOT" "$COMFY_OUTPUT_ROOT" "$COMFY_DATA_DIR/output-sam3-mask" "$COMFY_DATA_DIR/output-postprocessed" "$COMFY_DATA_DIR/output-egoblur"

if [[ ! -f "$COMFY_INPUT_ROOT/perspective_mask.png" ]]; then
  cp "$REPO/inpainting-workflow-master/perspective_mask.png" "$COMFY_INPUT_ROOT/perspective_mask.png"
fi
cp "$SKY_REFERENCE_SOURCE" "$COMFY_INPUT_ROOT/$SKY_REFERENCE_FILENAME"

DST="$COMFY_INPUT_ROOT/$BATCH_NAME"
OUT1="$COMFY_OUTPUT_ROOT/$BATCH_NAME"
OUT_MASK="$COMFY_DATA_DIR/output-sam3-mask/$BATCH_NAME"
OUT2="$COMFY_DATA_DIR/output-postprocessed/$BATCH_NAME"
OUT3="$COMFY_DATA_DIR/output-egoblur/$BATCH_NAME"

if [[ "$FORCE_REPROCESS" == "1" ]]; then
  rm -rf "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"
fi
mkdir -p "$DST" "$OUT1" "$OUT_MASK" "$OUT2" "$OUT3"

python3 "$PIPELINE_HELPERS" stage-images --src "$SRC" --dst "$DST" --strict-hardlink "$STRICT_HARDLINK"

python3 "$REPO/inpainting-workflow-master/$SAM3_SCRIPT" \
  --input-dir "$DST" \
  --output-dir "$OUT_MASK" \
  --pattern "*" \
  --glare-threshold "$SAM3_GLARE_THRESHOLD" \
  --tile-rows "$SAM3_TILE_ROWS" \
  --tile-cols "$SAM3_TILE_COLS" \
  --resize-width "$SAM3_RESIZE_WIDTH" \
  --resize-height "$SAM3_RESIZE_HEIGHT" \
  --workers "$SAM3_WORKERS"

if [[ "$STOP_AFTER_STAGE" != "sam3" ]]; then
  python3 "$REPO/inpainting-workflow-master/comfyui_run.py" \
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
fi

if [[ "$STOP_AFTER_STAGE" == "egoblur" || "$STOP_AFTER_STAGE" == "postprocess" ]]; then
  LAMA_MODEL="$LAMA_MODEL" python3 "$REPO/inpainting-workflow-master/postprocess.py" \
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
fi

if [[ "$STOP_AFTER_STAGE" == "egoblur" ]]; then
  PRIVACY_CMD=(
    python3 "$REPO/inpainting-workflow-master/privacy_blur_infer.py"
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
  if [[ "$FORCE_REPROCESS" == "1" ]]; then
    PRIVACY_CMD+=(--overwrite)
  fi
  "${PRIVACY_CMD[@]}"
fi

if [[ -n "$FINAL_OUTPUT_DIR" && "$STOP_AFTER_STAGE" == "egoblur" ]]; then
  mkdir -p "$FINAL_OUTPUT_DIR/$BATCH_NAME"
  python3 "$PIPELINE_HELPERS" link-flat --src "$OUT3" --dst "$FINAL_OUTPUT_DIR/$BATCH_NAME" --strict-hardlink "$STRICT_HARDLINK"
fi

python3 "$PIPELINE_HELPERS" report-counts \
  --input-dir "$DST" \
  --sam3-dir "$OUT_MASK" \
  --inpainting-dir "$OUT1" \
  --postprocess-dir "$OUT2" \
  --egoblur-dir "$OUT3"

echo "DONE native run: $RUN_ID"
