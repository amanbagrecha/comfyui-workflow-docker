#!/usr/bin/env bash

export PATH="$HOME/.local/bin:$PATH"

repo_python_version() {
  tr -d '[:space:]' < "$REPO/.python-version"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      if [ "$(id -u)" -eq 0 ]; then
        apt-get update && apt-get install -y curl
      elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y curl
      fi
    fi
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "ERROR: curl or wget is required to install uv"
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation failed"
    exit 1
  fi
}

ensure_repo_python() {
  local version current_version

  ensure_uv
  version="$(repo_python_version)"
  PYTHON_BIN="$REPO/.venv/bin/python"

  current_version=""
  if [ -x "$PYTHON_BIN" ]; then
    current_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  fi

  if [ "$current_version" != "$version" ]; then
    rm -rf "$REPO/.venv"
    uv python install "$version"
    uv venv --python "$version" "$REPO/.venv"
  fi

  PYTHON_BIN="$REPO/.venv/bin/python"
  export PYTHON_BIN
}

require_repo_python() {
  PYTHON_BIN="$REPO/.venv/bin/python"
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: missing $PYTHON_BIN. Run ./run_multi_gpu_pipeline.sh first."
    exit 1
  fi
  export PYTHON_BIN
}
