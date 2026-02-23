#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-$SCRIPT_DIR}"
SRC="${SRC:-$REPO/input}"
BATCH_NAME="${BATCH_NAME:-batch-$(date +%Y%m%d_%H%M%S)}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-1}"
EGOBLUR_WORKERS="${EGOBLUR_WORKERS:-3}"
CONTAINER_NAME="${CONTAINER_NAME:-comfyui-container}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_EGOBLUR_DIR="${MODELS_EGOBLUR_DIR:-$MODELS_ROOT/egoblur_gen2}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fullrun_${RUN_ID}.log"
SUMMARY_FILE="$LOG_DIR/fullrun_${RUN_ID}.summary.txt"

exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)

echo "RUN_ID=$RUN_ID"
echo "LOG_FILE=$LOG_FILE"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "REPO=$REPO"
echo "SRC=$SRC"
echo "BATCH_NAME=$BATCH_NAME"
echo "CONTAINER_NAME=$CONTAINER_NAME"
echo "POSTPROCESS_WORKERS=$POSTPROCESS_WORKERS"
echo "EGOBLUR_WORKERS=$EGOBLUR_WORKERS"
echo "MODELS_COMFYUI_DIR=$MODELS_COMFYUI_DIR"
echo "MODELS_EGOBLUR_DIR=$MODELS_EGOBLUR_DIR"
if [ -n "$FINAL_OUTPUT_DIR" ]; then
  echo "FINAL_OUTPUT_DIR=$FINAL_OUTPUT_DIR"
fi

mkdir -p \
  "$REPO/input" \
  "$REPO/output" \
  "$REPO/output-postprocessed" \
  "$REPO/output-egoblur" \
  "$MODELS_COMFYUI_DIR" \
  "$MODELS_EGOBLUR_DIR" \
  "$REPO/inpainting-workflow-master/models/egoblur_gen2"

required_files=(
  "$MODELS_COMFYUI_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
  "$MODELS_COMFYUI_DIR/vae/qwen_image_vae.safetensors"
  "$MODELS_COMFYUI_DIR/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
  "$MODELS_COMFYUI_DIR/sam3/sam3.pt"
  "$MODELS_EGOBLUR_DIR/ego_blur_face_gen2.jit"
  "$MODELS_EGOBLUR_DIR/ego_blur_lp_gen2.jit"
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

if [ ! -f "$REPO/input/perspective_mask.png" ]; then
  cp "$REPO/inpainting-workflow-master/perspective_mask.png" "$REPO/input/perspective_mask.png"
  echo "Copied perspective mask to $REPO/input/perspective_mask.png"
fi

echo "Ensuring container is up via docker compose"
CONTAINER_NAME="$CONTAINER_NAME" \
MODELS_COMFYUI_DIR="$MODELS_COMFYUI_DIR" \
MODELS_EGOBLUR_DIR="$MODELS_EGOBLUR_DIR" \
docker compose up -d

DST="$REPO/input/$BATCH_NAME"
OUT1="$REPO/output/$BATCH_NAME"
OUT2="$REPO/output-postprocessed/$BATCH_NAME"
OUT3="$REPO/output-egoblur/$BATCH_NAME"

if [ ! -d "$SRC" ]; then
  echo "ERROR: SRC directory not found: $SRC"
  exit 1
fi

echo "Preparing clean batch directories..."
rm -rf "$DST" "$OUT1" "$OUT2" "$OUT3"
mkdir -p "$DST" "$OUT1" "$OUT2" "$OUT3"

S_HARD=$(date +%s)
echo "=== STAGE_START hardlink_stage ==="
export SRC DST
python3 - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])
exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

images = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
for image in images:
    try:
        (dst / image.name).hardlink_to(image)
    except OSError:
        (dst / image.name).write_bytes(image.read_bytes())

print(f"staged_images={len(images)}")
PY
E_HARD=$(date +%s)
HARDLINK_SEC=$((E_HARD - S_HARD))
echo "=== STAGE_END hardlink_stage elapsed_sec=$HARDLINK_SEC ==="

S_INP=$(date +%s)
echo "=== STAGE_START inpainting ==="
docker exec "$CONTAINER_NAME" python /workspace/inpainting/comfyui_run.py \
  --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input/$BATCH_NAME \
  --mask /workspace/ComfyUI/input/perspective_mask.png \
  --output-dir /workspace/ComfyUI/output/$BATCH_NAME \
  --workers 1 \
  --timeout-s 3600
E_INP=$(date +%s)
INPAINT_SEC=$((E_INP - S_INP))
echo "=== STAGE_END inpainting elapsed_sec=$INPAINT_SEC ==="

S_POST=$(date +%s)
echo "=== STAGE_START postprocess ==="
docker exec "$CONTAINER_NAME" python /workspace/inpainting/postprocess.py \
  -i /workspace/ComfyUI/output/$BATCH_NAME \
  -o /workspace/output-postprocessed/$BATCH_NAME \
  --top-mask /workspace/inpainting/sky_mask_updated.png \
  --pattern "*.jpg" \
  -j "$POSTPROCESS_WORKERS"
E_POST=$(date +%s)
POSTPROCESS_SEC=$((E_POST - S_POST))
echo "=== STAGE_END postprocess elapsed_sec=$POSTPROCESS_SEC ==="

S_EGO=$(date +%s)
echo "=== STAGE_START egoblur ==="
docker exec "$CONTAINER_NAME" python /workspace/inpainting/egoblur_infer.py \
  --input-dir /workspace/output-postprocessed/$BATCH_NAME \
  --output-dir /workspace/output-egoblur/$BATCH_NAME \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit \
  --workers "$EGOBLUR_WORKERS"
E_EGO=$(date +%s)
EGOBLUR_SEC=$((E_EGO - S_EGO))
echo "=== STAGE_END egoblur elapsed_sec=$EGOBLUR_SEC ==="

if [ -n "$FINAL_OUTPUT_DIR" ]; then
  FINAL_BATCH_DIR="$FINAL_OUTPUT_DIR/$BATCH_NAME"
  mkdir -p "$FINAL_BATCH_DIR"
  export OUT3 FINAL_BATCH_DIR
  python3 - <<'PY'
import os
import shutil
from pathlib import Path

src = Path(os.environ["OUT3"])
dst = Path(os.environ["FINAL_BATCH_DIR"])
for file in src.iterdir():
    if not file.is_file():
        continue
    target = dst / file.name
    try:
        if target.exists():
            target.unlink()
        target.hardlink_to(file)
    except OSError:
        shutil.copy2(file, target)

print(f"final_output_written={dst}")
PY
fi

S_COUNT=$(date +%s)
echo "=== STAGE_START counts ==="
export DST OUT1 OUT2 OUT3
python3 - <<'PY'
import os
from pathlib import Path

exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def count_images(path: str) -> int:
    root = Path(path)
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


print(f"count_input={count_images(os.environ['DST'])}")
print(f"count_inpainting={count_images(os.environ['OUT1'])}")
print(f"count_postprocess={count_images(os.environ['OUT2'])}")
print(f"count_egoblur={count_images(os.environ['OUT3'])}")
PY
E_COUNT=$(date +%s)
COUNT_SEC=$((E_COUNT - S_COUNT))
echo "=== STAGE_END counts elapsed_sec=$COUNT_SEC ==="

END_EPOCH=$(date +%s)
TOTAL_SEC=$((END_EPOCH - START_EPOCH))

{
  echo "run_id=$RUN_ID"
  echo "start_epoch=$START_EPOCH"
  echo "end_epoch=$END_EPOCH"
  echo "total_sec=$TOTAL_SEC"
  printf "total_hms=%02d:%02d:%02d\n" $((TOTAL_SEC / 3600)) $(((TOTAL_SEC % 3600) / 60)) $((TOTAL_SEC % 60))
  echo "stage_hardlink_sec=$HARDLINK_SEC"
  echo "stage_inpainting_sec=$INPAINT_SEC"
  echo "stage_postprocess_sec=$POSTPROCESS_SEC"
  echo "stage_egoblur_sec=$EGOBLUR_SEC"
  echo "stage_counts_sec=$COUNT_SEC"
  echo "batch_name=$BATCH_NAME"
  echo "local_out_dir=$OUT3"
  if [ -n "$FINAL_OUTPUT_DIR" ]; then
    echo "final_out_dir=$FINAL_OUTPUT_DIR/$BATCH_NAME"
  fi
  echo "log_file=$LOG_FILE"
} | tee "$SUMMARY_FILE"

echo "DONE"
echo "Summary: $SUMMARY_FILE"
echo "Log: $LOG_FILE"
