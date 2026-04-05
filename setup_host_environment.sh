#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"

ensure_uv

VENV_DIR="$REPO/.venv"
COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-$MODELS_ROOT/privacy_blur}"
P2E_LIB_DIR="${P2E_LIB_DIR:-$REPO/p2e-lib}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
FORCE_LINKS="${FORCE_LINKS:-1}"

COMFYUI_COMMIT="${COMFYUI_COMMIT:-532e2850794c7b497174a0a42ac0cb1fe5b62499}"
COMFYUI_MANAGER_COMMIT="${COMFYUI_MANAGER_COMMIT:-2478d20e76aeb2f42a6f372029e417201ef927b3}"
P2E_COMMIT="${P2E_COMMIT:-1968f6aed36b300be3599ea5053e0206aaa5704b}"

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: root/sudo is required for system package installation"
    return 1
  fi
}

install_system_packages() {
  if [[ "$INSTALL_SYSTEM_PACKAGES" != "1" ]]; then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Skipping system package install: apt-get not available"
    return 0
  fi

  run_privileged apt-get update
  run_privileged apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux
}

clone_or_checkout() {
  local repo_url="$1"
  local dest="$2"
  local commit="$3"

  mkdir -p "$(dirname "$dest")"
  if [ ! -d "$dest/.git" ]; then
    git clone "$repo_url" "$dest"
  fi

  git -C "$dest" fetch --depth 1 origin "$commit"
  git -C "$dest" checkout "$commit"
}

link_path() {
  local src="$1"
  local dest="$2"

  mkdir -p "$(dirname "$dest")"
  if [ -L "$dest" ]; then
    ln -sfn "$src" "$dest"
    return 0
  fi

  if [ -e "$dest" ]; then
    if [[ "$FORCE_LINKS" != "1" ]]; then
      echo "ERROR: $dest exists and FORCE_LINKS=0"
      exit 1
    fi
    rm -rf "$dest"
  fi

  ln -s "$src" "$dest"
}

install_system_packages

for cmd in git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd"
    exit 1
  fi
done

mkdir -p "$MODELS_COMFYUI_DIR" "$MODELS_PRIVACY_DIR"

ensure_repo_python
CM_CLI="$COMFYUI_HOME/custom_nodes/ComfyUI-Manager/cm-cli.py"

uv pip install --python "$PYTHON_BIN" pip
uv pip install --python "$PYTHON_BIN" comfy-cli

clone_or_checkout https://github.com/comfyanonymous/ComfyUI "$COMFYUI_HOME" "$COMFYUI_COMMIT"
clone_or_checkout https://github.com/ltdrdata/ComfyUI-Manager "$COMFYUI_HOME/custom_nodes/ComfyUI-Manager" "$COMFYUI_MANAGER_COMMIT"
uv pip install --python "$PYTHON_BIN" -r "$COMFYUI_HOME/custom_nodes/ComfyUI-Manager/requirements.txt"

TORCH_REQS=$(mktemp)
COMFY_CORE_REQS=$(mktemp)
cleanup_tmp() {
  rm -f "$TORCH_REQS" "$COMFY_CORE_REQS"
}
trap cleanup_tmp EXIT

"$PYTHON_BIN" - "$REPO/pipeline-requirements.txt" "$TORCH_REQS" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
keep = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('--extra-index-url') or line.startswith('torch==') or line.startswith('torchvision==') or line.startswith('torchaudio=='):
        keep.append(line)
dst.write_text('\n'.join(keep) + '\n')
PY

uv pip install --python "$PYTHON_BIN" -r "$TORCH_REQS"

"$PYTHON_BIN" - "$COMFYUI_HOME/requirements.txt" "$COMFY_CORE_REQS" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip = {'torch', 'torchaudio', 'torchvision'}
keep = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    name = line.split('==', 1)[0].split('>=', 1)[0].split('~=', 1)[0].strip().lower()
    if name in skip:
        continue
    keep.append(line)
dst.write_text('\n'.join(keep) + '\n')
PY

uv pip install --python "$PYTHON_BIN" -r "$COMFY_CORE_REQS"

COMFYUI_PATH="$COMFYUI_HOME" "$PYTHON_BIN" "$CM_CLI" restore-snapshot \
  "$REPO/Comfy-Lock.yaml" \
  --pip-non-url \
  --pip-non-local-url

clone_or_checkout https://github.com/amanbagrecha/p2e.git "$P2E_LIB_DIR" "$P2E_COMMIT"

uv pip install --python "$PYTHON_BIN" -r "$REPO/pipeline-requirements.txt"
uv pip install --python "$PYTHON_BIN" --no-deps \
  simple-lama-inpainting==0.1.0 \
  ultralytics==8.4.21 \
  open-image-models==0.5.1
uv pip uninstall --python "$PYTHON_BIN" opencv-python opencv-python-headless || true
uv pip install --python "$PYTHON_BIN" --no-deps opencv-contrib-python==4.12.0.88

link_path "$MODELS_COMFYUI_DIR" "$COMFYUI_HOME/models"
link_path "$REPO/p2e-local" "$COMFYUI_HOME/custom_nodes/p2e"

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  MODELS_ROOT="$MODELS_ROOT" \
  COMFY_MODELS_DIR="$MODELS_COMFYUI_DIR" \
  PRIVACY_MODELS_DIR="$MODELS_PRIVACY_DIR" \
  bash "$REPO/download-models.sh"
fi

echo "Host environment setup complete."
echo "Run the pipeline with:"
echo "  SRC=/abs/path/to/images FINAL_OUTPUT_DIR=/abs/path/to/final ./run_multi_gpu_pipeline.sh"
