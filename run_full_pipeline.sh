#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-$SCRIPT_DIR}"
CONTAINER_NAME="${CONTAINER_NAME:-comfyui-container}"
COMFYUI_DATA_DIR="${COMFYUI_DATA_DIR:-$REPO/comfyui_data/$CONTAINER_NAME}"
SRC="${SRC:-$COMFYUI_DATA_DIR/input}"
BATCH_NAME="${BATCH_NAME:-batch-$(date +%Y%m%d_%H%M%S)}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-4}"
EGOBLUR_WORKERS="${EGOBLUR_WORKERS:-4}"
PRIVACY_WORKERS="${PRIVACY_WORKERS:-2}"
SAM3_WORKERS="${SAM3_WORKERS:-4}"
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
MODELS_EGOBLUR_DIR="${MODELS_EGOBLUR_DIR:-$MODELS_ROOT/egoblur_gen2}"
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
PRIVACY_PYTHON_BIN="${PRIVACY_PYTHON_BIN:-/data/.venv/bin/python}"
COMFY_IMAGE="${COMFY_IMAGE:-amanbagrecha/container-comfyui:latest}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/.cache/torch}"
COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
COMFY_READY_POLL="${COMFY_READY_POLL:-2}"
RESET_CONTAINER_BEFORE_RUN="${RESET_CONTAINER_BEFORE_RUN:-1}"
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
AUTO_INSTALL_NVIDIA_TOOLKIT="${AUTO_INSTALL_NVIDIA_TOOLKIT:-1}"
NVIDIA_CUDA_TEST_IMAGE="${NVIDIA_CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu22.04}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${CONTAINER_NAME}_$$"

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
echo "COMFYUI_DATA_DIR=$COMFYUI_DATA_DIR"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}"
echo "COMFY_PORT=${COMFY_PORT:-8188}"
echo "FORCE_REPROCESS=$FORCE_REPROCESS"
echo "STRICT_HARDLINK=$STRICT_HARDLINK"
echo "AUTO_INSTALL_NVIDIA_TOOLKIT=$AUTO_INSTALL_NVIDIA_TOOLKIT"
echo "POSTPROCESS_WORKERS=$POSTPROCESS_WORKERS"
echo "EGOBLUR_WORKERS=$EGOBLUR_WORKERS"
echo "PRIVACY_WORKERS=$PRIVACY_WORKERS"
echo "SAM3_WORKERS=$SAM3_WORKERS"
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
echo "MODELS_EGOBLUR_DIR=$MODELS_EGOBLUR_DIR"
echo "MODELS_PRIVACY_DIR=$MODELS_PRIVACY_DIR"
echo "PRIVACY_FACE_MODEL=$PRIVACY_FACE_MODEL"
echo "PRIVACY_LP_MODEL=$PRIVACY_LP_MODEL"
echo "PRIVACY_OUTPUT_MODE=$PRIVACY_OUTPUT_MODE"
echo "PRIVACY_PYTHON_BIN=$PRIVACY_PYTHON_BIN"
echo "TORCH_CACHE_DIR=$TORCH_CACHE_DIR"
echo "COMFY_READY_TIMEOUT=$COMFY_READY_TIMEOUT"
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
  "$MODELS_EGOBLUR_DIR" \
  "$MODELS_PRIVACY_DIR" \
  "$REPO/inpainting-workflow-master/models/egoblur_gen2"

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

if [[ "$STOP_AFTER_STAGE" != "inpainting" && "$STOP_AFTER_STAGE" != "postprocess" && "$STOP_AFTER_STAGE" != "egoblur" ]]; then
  echo "ERROR: Invalid STOP_AFTER_STAGE=$STOP_AFTER_STAGE (expected: inpainting|postprocess|egoblur)"
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

ensure_privacy_python_ready() {
  local py_bin="$PRIVACY_PYTHON_BIN"
  if [[ ! -x "$py_bin" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      py_bin="$(command -v python3)"
    else
      echo "ERROR: privacy blur requires python interpreter but none was found."
      return 1
    fi
  fi
  PRIVACY_PYTHON_BIN="$py_bin"

  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: privacy blur stage requires 'uv' but it is not available in PATH."
    return 1
  fi

  if "$PRIVACY_PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import pytorch360convert
import torch
import ultralytics
import open_image_models
PY
  then
    return 0
  fi

  echo "Installing missing privacy blur Python dependencies into $PRIVACY_PYTHON_BIN"
  uv pip install --python "$PRIVACY_PYTHON_BIN" "open-image-models[onnx-gpu]" ultralytics pytorch360convert
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found on host. NVIDIA drivers/GPU are required for this pipeline."
  exit 1
fi

docker_gpu_ready() {
  docker run --rm --gpus all "$NVIDIA_CUDA_TEST_IMAGE" nvidia-smi >/dev/null 2>&1
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
  run_privileged sh -c 'set -e; apt-get update; apt-get install -y curl gpg ca-certificates'
  run_privileged mkdir -p /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    run_privileged gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    run_privileged tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  run_privileged apt-get update
  run_privileged apt-get install -y nvidia-container-toolkit
  run_privileged nvidia-ctk runtime configure --runtime=docker
  run_privileged systemctl restart docker
}

echo "Checking Docker GPU runtime"
if docker_gpu_ready; then
  echo "Docker GPU runtime: OK"
else
  echo "Docker GPU runtime: NOT READY"
  if [ "$AUTO_INSTALL_NVIDIA_TOOLKIT" = "1" ]; then
    install_nvidia_container_toolkit
    if docker_gpu_ready; then
      echo "Docker GPU runtime: OK after toolkit installation"
    else
      echo "ERROR: Docker GPU runtime still not ready after installation attempt."
      echo "Please verify NVIDIA Container Toolkit + Docker runtime setup manually."
      exit 1
    fi
  else
    echo "ERROR: GPU runtime not ready and AUTO_INSTALL_NVIDIA_TOOLKIT=0"
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

if [ ! -f "$COMFYUI_DATA_DIR/input/perspective_mask.png" ]; then
  cp "$REPO/inpainting-workflow-master/perspective_mask.png" "$COMFYUI_DATA_DIR/input/perspective_mask.png"
  echo "Copied perspective mask to $COMFYUI_DATA_DIR/input/perspective_mask.png"
fi

if [ "$RESET_CONTAINER_BEFORE_RUN" = "1" ]; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Resetting existing container before run: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  else
    echo "RESET_CONTAINER_BEFORE_RUN=1: no existing container to reset for $CONTAINER_NAME"
  fi
fi

echo "Ensuring container is up via docker compose"
CONTAINER_NAME="$CONTAINER_NAME" \
COMFYUI_DATA_DIR="$COMFYUI_DATA_DIR" \
MODELS_COMFYUI_DIR="$MODELS_COMFYUI_DIR" \
MODELS_EGOBLUR_DIR="$MODELS_EGOBLUR_DIR" \
docker compose -p "${CONTAINER_NAME}" up -d

echo "Waiting for ComfyUI API readiness inside container"
docker exec -i \
  -e COMFY_READY_TIMEOUT="$COMFY_READY_TIMEOUT" \
  -e COMFY_READY_POLL="$COMFY_READY_POLL" \
  "$CONTAINER_NAME" \
  python - <<'PY'
import os
import sys
import time
import urllib.request

timeout = int(os.environ["COMFY_READY_TIMEOUT"])
poll = float(os.environ["COMFY_READY_POLL"])
url = "http://127.0.0.1:8188/system_stats"
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print("ComfyUI API ready")
                sys.exit(0)
    except Exception as exc:  # pragma: no cover - runtime readiness polling
        last_error = exc
    time.sleep(poll)

print(f"ERROR: ComfyUI API not ready within {timeout}s. Last error: {last_error}")
sys.exit(1)
PY

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

HARDLINK_SEC=0
if [ "$USE_STAGED_INPUT" = "1" ]; then
  S_HARD=$(date +%s)
  echo "=== STAGE_START hardlink_stage ==="
  export SRC DST
  python3 - <<'PY'
import shutil
import os
from pathlib import Path

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])
strict_hardlink = os.environ.get("STRICT_HARDLINK", "1") == "1"
exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

images = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
for image in images:
    target = dst / image.name
    if target.exists():
        src_stat = image.stat()
        dst_stat = target.stat()
        same_inode = src_stat.st_dev == dst_stat.st_dev and src_stat.st_ino == dst_stat.st_ino
        if same_inode:
            continue
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink requirement failed (STRICT_HARDLINK=1): existing target is not a hardlink: {target}"
            )
        shutil.copy2(image, target)
        continue

    try:
        target.hardlink_to(image)
    except OSError as exc:
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink failed (STRICT_HARDLINK=1): {image} -> {target}: {exc}"
            )
        shutil.copy2(image, target)

print(f"staged_images={len(images)}")
PY
  E_HARD=$(date +%s)
  HARDLINK_SEC=$((E_HARD - S_HARD))
  echo "=== STAGE_END hardlink_stage elapsed_sec=$HARDLINK_SEC ==="
else
  echo "=== STAGE_SKIP hardlink_stage reason=direct_input elapsed_sec=0 ==="
fi

S_SAM3=$(date +%s)
echo "=== STAGE_START sam3_mask ==="
docker exec "$CONTAINER_NAME" python /workspace/inpainting/sam3_tiled_mask.py \
  --input-dir "$CONTAINER_INPUT_DIR" \
  --output-dir /workspace/output-sam3-mask/$BATCH_NAME \
  --pattern "*" \
  --workers "$SAM3_WORKERS"
E_SAM3=$(date +%s)
SAM3_SEC=$((E_SAM3 - S_SAM3))
echo "=== STAGE_END sam3_mask elapsed_sec=$SAM3_SEC ==="

export OUT_MASK
SAM3_MASK_COUNT=$(python3 - <<'PY'
import os
from pathlib import Path

exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
root = Path(os.environ["OUT_MASK"])
count = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
print(count)
PY
)
echo "sam3_mask_output_count=$SAM3_MASK_COUNT"
if [ "$SAM3_MASK_COUNT" -eq 0 ]; then
  echo "ERROR: No SAM3 mask outputs found in $OUT_MASK; aborting downstream stages."
  exit 1
fi

S_INP=$(date +%s)
echo "=== STAGE_START inpainting ==="
docker exec "$CONTAINER_NAME" python /workspace/inpainting/comfyui_run.py \
  --workflow-json /workspace/workflow.json \
  --input-dir "$CONTAINER_INPUT_DIR" \
  --mask /workspace/ComfyUI/input/perspective_mask.png \
  --sam3-mask-dir /workspace/output-sam3-mask/$BATCH_NAME \
  --output-dir /workspace/ComfyUI/output/$BATCH_NAME \
  --image-node-id 48 \
  --mask-node-id 34 \
  --sam3-mask-node-id 60 \
  --workers 1 \
  --timeout-s 3600
E_INP=$(date +%s)
INPAINT_SEC=$((E_INP - S_INP))
echo "=== STAGE_END inpainting elapsed_sec=$INPAINT_SEC ==="

export OUT1
INPAINT_COUNT=$(python3 - <<'PY'
import os
from pathlib import Path

exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
root = Path(os.environ["OUT1"])
count = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
print(count)
PY
)
echo "inpainting_output_count=$INPAINT_COUNT"
if [ "$INPAINT_COUNT" -eq 0 ]; then
  echo "ERROR: No inpainting outputs found in $OUT1; aborting downstream stages."
  exit 1
fi

POSTPROCESS_SEC=0
EGOBLUR_SEC=0

if [ "$STOP_AFTER_STAGE" = "inpainting" ]; then
  echo "=== STAGE_SKIP postprocess reason=STOP_AFTER_STAGE=inpainting elapsed_sec=0 ==="
  echo "=== STAGE_SKIP egoblur reason=STOP_AFTER_STAGE=inpainting elapsed_sec=0 ==="
else
  if [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
    ensure_privacy_python_ready
  fi

  if [ "$DOWNSTREAM_MODE" = "isolated" ]; then
    stop_comfy_containers_for_downstream
  fi

  S_POST=$(date +%s)
  echo "=== STAGE_START postprocess ==="
  if [ "$DOWNSTREAM_MODE" = "inline" ]; then
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
  echo "=== STAGE_END postprocess elapsed_sec=$POSTPROCESS_SEC ==="

  if [ "$STOP_AFTER_STAGE" = "postprocess" ]; then
    echo "=== STAGE_SKIP egoblur reason=STOP_AFTER_STAGE=postprocess elapsed_sec=0 ==="
  else
    S_EGO=$(date +%s)
    echo "=== STAGE_START egoblur ==="
    PRIVACY_RUNNER="$REPO/inpainting-workflow-master/privacy_blur_parallel.sh"
    if [ ! -f "$PRIVACY_RUNNER" ]; then
      echo "ERROR: Privacy blur runner not found at $PRIVACY_RUNNER"
      exit 1
    fi

    RUN_NAME="privacy-${RUN_ID}" \
    SRC="$OUT2" \
    OUT_ROOT="$OUT3" \
    WORKERS="$PRIVACY_WORKERS" \
    FACE_MODEL="$PRIVACY_FACE_MODEL" \
    LP_MODEL="$PRIVACY_LP_MODEL" \
    FACE_CONF="$PRIVACY_FACE_CONF" \
    LP_CONF="$PRIVACY_LP_CONF" \
    FACE_IOU="$PRIVACY_FACE_IOU" \
    FACE_IMGSZ="$PRIVACY_FACE_IMGSZ" \
    DET_FACE_W="$PRIVACY_DET_FACE_W" \
    P360_DEVICE="$PRIVACY_P360_DEVICE" \
    BLUR_SCOPE="$PRIVACY_BLUR_SCOPE" \
    BLUR_BACKEND="$PRIVACY_BLUR_BACKEND" \
    OUTPUT_MODE="$PRIVACY_OUTPUT_MODE" \
    PYTHON_BIN="$PRIVACY_PYTHON_BIN" \
    STRICT_HARDLINK="$STRICT_HARDLINK" \
    bash "$PRIVACY_RUNNER"

    E_EGO=$(date +%s)
    EGOBLUR_SEC=$((E_EGO - S_EGO))
    echo "=== STAGE_END egoblur elapsed_sec=$EGOBLUR_SEC ==="
  fi
fi

if [ -n "$FINAL_OUTPUT_DIR" ] && [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
  FINAL_BATCH_DIR="$FINAL_OUTPUT_DIR/$BATCH_NAME"
  mkdir -p "$FINAL_BATCH_DIR"
  export OUT3 FINAL_BATCH_DIR STRICT_HARDLINK
  python3 - <<'PY'
import os
import shutil
from pathlib import Path

src = Path(os.environ["OUT3"])
dst = Path(os.environ["FINAL_BATCH_DIR"])
strict_hardlink = os.environ.get("STRICT_HARDLINK", "1") == "1"
for file in src.iterdir():
    if not file.is_file():
        continue
    target = dst / file.name
    try:
        if target.exists():
            target.unlink()
        target.hardlink_to(file)
    except OSError as exc:
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink failed (STRICT_HARDLINK=1): {file} -> {target}: {exc}"
            )
        shutil.copy2(file, target)

print(f"final_output_written={dst}")
PY
fi

S_COUNT=$(date +%s)
echo "=== STAGE_START counts ==="
export HOST_INPUT_DIR OUT_MASK OUT1 OUT2 OUT3
python3 - <<'PY'
import os
from pathlib import Path

exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def count_images(path: str) -> int:
    root = Path(path)
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


print(f"count_input={count_images(os.environ['HOST_INPUT_DIR'])}")
print(f"count_sam3_mask={count_images(os.environ['OUT_MASK'])}")
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
  echo "stage_sam3_mask_sec=$SAM3_SEC"
  echo "stage_inpainting_sec=$INPAINT_SEC"
  echo "stage_postprocess_sec=$POSTPROCESS_SEC"
  echo "stage_egoblur_sec=$EGOBLUR_SEC"
  echo "stage_counts_sec=$COUNT_SEC"
  echo "laplacian_dilation=$LAPLACIAN_DILATION"
  echo "laplacian_blur=$LAPLACIAN_BLUR"
  echo "laplacian_levels=$LAPLACIAN_LEVELS"
  echo "privacy_workers=$PRIVACY_WORKERS"
  echo "privacy_face_model=$PRIVACY_FACE_MODEL"
  echo "privacy_lp_model=$PRIVACY_LP_MODEL"
  echo "privacy_face_conf=$PRIVACY_FACE_CONF"
  echo "privacy_lp_conf=$PRIVACY_LP_CONF"
  echo "privacy_output_mode=$PRIVACY_OUTPUT_MODE"
  echo "privacy_blur_scope=$PRIVACY_BLUR_SCOPE"
  echo "privacy_blur_backend=$PRIVACY_BLUR_BACKEND"
  echo "downstream_mode=$DOWNSTREAM_MODE"
  echo "comfy_stop_containers=$COMFY_STOP_CONTAINERS"
  echo "stop_after_stage=$STOP_AFTER_STAGE"
  echo "strict_hardlink=$STRICT_HARDLINK"
  echo "batch_name=$BATCH_NAME"
  echo "input_mode_requested=$INPUT_MODE"
  echo "input_mode_resolved=$RESOLVED_INPUT_MODE"
  echo "local_input_dir=$HOST_INPUT_DIR"
  echo "container_input_dir=$CONTAINER_INPUT_DIR"
  echo "local_sam3_mask_dir=$OUT_MASK"
  echo "local_out_dir=$OUT3"
  if [ -n "$FINAL_OUTPUT_DIR" ] && [ "$STOP_AFTER_STAGE" = "egoblur" ]; then
    echo "final_out_dir=$FINAL_OUTPUT_DIR/$BATCH_NAME"
  fi
  echo "log_file=$LOG_FILE"
} | tee "$SUMMARY_FILE"

echo "DONE"
echo "Summary: $SUMMARY_FILE"
echo "Log: $LOG_FILE"
