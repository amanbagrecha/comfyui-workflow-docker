#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"
# shellcheck disable=SC1091
. "$REPO/scripts/shell_helpers.sh"

ensure_uv

COMFYUI_HOME="${COMFYUI_HOME:-$REPO/ComfyUI}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-$MODELS_ROOT/comfyui}"
MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-$MODELS_ROOT/privacy_blur}"
P2E_LIB_DIR="${P2E_LIB_DIR:-$REPO/p2e-lib}"
COMFY_CORE_REQS="$REPO/requirements/comfyui-core.txt"
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

  local os_pkg="libgl1-mesa-glx"
  if grep -q "Noble\|24.04" /etc/os-release 2>/dev/null; then
    os_pkg="libgl1"
  fi

  run_privileged apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    "$os_pkg" \
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

enable_custom_node() {
  local name="$1"
  local active_dir="$COMFYUI_HOME/custom_nodes/$name"
  local disabled_dir="$COMFYUI_HOME/custom_nodes/.disabled/$name"

  if [ -d "$disabled_dir" ] && [ ! -e "$active_dir" ]; then
    mv "$disabled_dir" "$active_dir"
  fi
}

list_cnr_custom_nodes() {
  LOCK_PATH="$REPO/Comfy-Lock.yaml" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

import yaml

lock_path = Path(os.environ["LOCK_PATH"])
data = yaml.safe_load(lock_path.read_text()) or {}
custom_nodes = data.get("custom_nodes", {})
for name in custom_nodes.get("cnr_custom_nodes", {}):
    print(name)
PY
}

ensure_cnr_custom_node() {
  local name="$1"
  local active_dir="$COMFYUI_HOME/custom_nodes/$name"
  local disabled_dir="$COMFYUI_HOME/custom_nodes/.disabled/$name"

  if [ ! -d "$active_dir" ] && [ ! -d "$disabled_dir" ]; then
    COMFYUI_PATH="$COMFYUI_HOME" "$PYTHON_BIN" "$CM_CLI" install --no-deps "$name"
  fi

  enable_custom_node "$name"

  if [ ! -d "$active_dir" ]; then
    echo "ERROR: required custom node '$name' is not installed"
    exit 1
  fi
}

install_system_packages
require_commands git

mkdir -p "$MODELS_COMFYUI_DIR" "$MODELS_PRIVACY_DIR"

ensure_repo_python
CM_CLI="$COMFYUI_HOME/custom_nodes/ComfyUI-Manager/cm-cli.py"

clone_or_checkout https://github.com/comfyanonymous/ComfyUI "$COMFYUI_HOME" "$COMFYUI_COMMIT"
clone_or_checkout https://github.com/ltdrdata/ComfyUI-Manager "$COMFYUI_HOME/custom_nodes/ComfyUI-Manager" "$COMFYUI_MANAGER_COMMIT"

# Install Python dependencies before invoking cm-cli; ComfyUI-Manager's CLI
# imports typer and other packages from the repo venv.
uv sync --python "$PYTHON_BIN" --no-install-project

# Best-effort snapshot restore for ComfyUI-Manager state.
# Pip installs are handled entirely by uv sync above, so we keep --no-deps
# when we explicitly ensure required CNR nodes after this step.
COMFYUI_PATH="$COMFYUI_HOME" "$PYTHON_BIN" "$CM_CLI" restore-snapshot \
  "$REPO/Comfy-Lock.yaml" || true

while IFS= read -r cnr_node; do
  [ -n "$cnr_node" ] || continue
  ensure_cnr_custom_node "$cnr_node"
done < <(list_cnr_custom_nodes)

clone_or_checkout https://github.com/amanbagrecha/p2e.git "$P2E_LIB_DIR" "$P2E_COMMIT"

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
