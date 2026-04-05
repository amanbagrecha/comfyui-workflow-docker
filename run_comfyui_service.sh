#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
DEFAULT_PYTHON_BIN="$REPO/.venv/bin/python"
if [ -x "$DEFAULT_PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
COMFY_PORT="${COMFY_PORT:-8180}"
COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-$REPO/native_data}"
COMFY_DATA_DIR="${COMFY_DATA_DIR:-$NATIVE_DATA_ROOT/gpu${GPU_ID}}"
COMFY_INPUT_ROOT="${COMFY_INPUT_ROOT:-$COMFY_DATA_DIR/input}"
COMFY_OUTPUT_ROOT="${COMFY_OUTPUT_ROOT:-$COMFY_DATA_DIR/output}"
COMFY_TEMP_PARENT="${COMFY_TEMP_PARENT:-$COMFY_DATA_DIR}"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/comfyui_g${GPU_ID}.log"

exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

if [[ "$PYTHON_BIN" == */* ]]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python executable not found: $PYTHON_BIN"
    exit 1
  fi
fi

if [ ! -d "$COMFYUI_HOME" ]; then
  echo "ERROR: COMFYUI_HOME not found: $COMFYUI_HOME"
  exit 1
fi

if [ ! -f "$COMFYUI_HOME/main.py" ]; then
  echo "ERROR: ComfyUI entrypoint not found: $COMFYUI_HOME/main.py"
  exit 1
fi

if [ ! -e "$COMFYUI_HOME/models" ]; then
  echo "ERROR: $COMFYUI_HOME/models is missing. Run ./setup_host_environment.sh first."
  exit 1
fi

if [ ! -e "$COMFYUI_HOME/custom_nodes/p2e" ]; then
  echo "ERROR: $COMFYUI_HOME/custom_nodes/p2e is missing. Run ./setup_host_environment.sh first."
  exit 1
fi

mkdir -p "$COMFY_INPUT_ROOT" "$COMFY_OUTPUT_ROOT" "$COMFY_TEMP_PARENT"

export CUDA_VISIBLE_DEVICES
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONPATH="$REPO/p2e-lib:$COMFYUI_HOME${PYTHONPATH:+:$PYTHONPATH}"

echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "COMFY_PORT=$COMFY_PORT"
echo "COMFYUI_HOME=$COMFYUI_HOME"
echo "COMFY_INPUT_ROOT=$COMFY_INPUT_ROOT"
echo "COMFY_OUTPUT_ROOT=$COMFY_OUTPUT_ROOT"
echo "COMFY_TEMP_PARENT=$COMFY_TEMP_PARENT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "LOG_FILE=$LOG_FILE"

exec "$PYTHON_BIN" "$COMFYUI_HOME/main.py" \
  --listen 127.0.0.1 \
  --port "$COMFY_PORT" \
  --input-directory "$COMFY_INPUT_ROOT" \
  --output-directory "$COMFY_OUTPUT_ROOT" \
  --temp-directory "$COMFY_TEMP_PARENT" \
  --disable-auto-launch
