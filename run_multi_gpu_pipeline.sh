#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

usage() {
  cat <<'EOF'
Usage:
  SRC=/abs/path/to/images ./run_multi_gpu_pipeline.sh

Environment variables:
  SRC                       Required. Source directory with input images.
  RUN_NAME                  Optional. Default: multigpu-YYYYmmdd_HHMMSS
  GPU_IDS                   Optional. Comma/space list (e.g. "0,1,2") or "auto".
  MAX_GPUS                  Optional. 0 means all detected GPUs.
  BASE_COMFY_PORT           Optional. Default: 8188
  CONTAINER_PREFIX          Optional. Default: comfyui-g
  COMFYUI_DATA_ROOT         Optional. Default: <repo>/comfyui_data
  TMUX_SESSION_PREFIX       Optional. Default: mgpu
  WORK_ROOT                 Optional. Default: <repo>/tmp/multigpu/<RUN_NAME>
  WAIT_POLL_SEC             Optional. Default: 10
  FINAL_OUTPUT_DIR          Optional. If set, writes per-shard outputs under
                            FINAL_OUTPUT_DIR/<RUN_NAME>/gpu<id>/
  STRICT_HARDLINK           Optional. 1 = fail when hardlink is not possible.
                            0 = allow copy fallback. Default: 1.
  DRY_RUN                   Optional. 1 = prepare/print plan only, no launches.
  SINGLE_GPU_CONFLICT_MODE  Optional. off|warn|stop. Default: warn.
                            Applies only when NUM_GPUS=1. Detects other running
                            comfyui containers and optionally stops them.

Forwarded to each shard run_full_pipeline.sh invocation (per GPU):
  MODELS_ROOT, MODELS_COMFYUI_DIR, MODELS_EGOBLUR_DIR, MODELS_PRIVACY_DIR,
  AUTO_DOWNLOAD_MODELS, FORCE_REPROCESS, STRICT_HARDLINK,
  SAM3_WORKERS, SAM3_RESIZE_WIDTH, SAM3_RESIZE_HEIGHT, SAM3_GLARE_THRESHOLD,
  SAM3_TILE_ROWS, SAM3_TILE_COLS, SAM3_SCRIPT,
  POSTPROCESS_WORKERS, EGOBLUR_WORKERS,
  PRIVACY_WORKERS, PRIVACY_FACE_MODEL, PRIVACY_LP_MODEL,
  PRIVACY_FACE_CONF, PRIVACY_LP_CONF, PRIVACY_FACE_IOU, PRIVACY_FACE_IMGSZ,
  PRIVACY_DET_FACE_W, PRIVACY_P360_DEVICE, PRIVACY_BLUR_SCOPE,
  PRIVACY_BLUR_BACKEND, PRIVACY_OUTPUT_MODE, PRIVACY_PYTHON_BIN,
  COMFY_IMAGE_NODE_ID, COMFY_MASK_NODE_ID, COMFY_SAM3_MASK_NODE_ID,
  SKY_REFERENCE_SOURCE, SKY_REFERENCE_FILENAME,
  LAPLACIAN_DILATION, LAPLACIAN_BLUR, LAPLACIAN_LEVELS,
  DOWNSTREAM_MODE (default: isolated), STOP_AFTER_STAGE (default: egoblur),
  COMFY_READY_TIMEOUT, COMFY_READY_POLL,
  NVIDIA_CUDA_TEST_IMAGE, COMFY_IMAGE, TORCH_CACHE_DIR.
  RESET_CONTAINER_BEFORE_RUN (default: 1 in this launcher).

Notes:
  - Input sharding is automatic (round-robin) using hardlinks when possible.
  - One shard run is launched per GPU in tmux.
  - Each shard only stops its own container via COMFY_STOP_CONTAINERS=<container>.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"

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
CONTAINER_PREFIX="${CONTAINER_PREFIX:-comfyui-g}"
COMFYUI_DATA_ROOT="${COMFYUI_DATA_ROOT:-$REPO/comfyui_data}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-mgpu}"
WORK_ROOT="${WORK_ROOT:-$REPO/tmp/multigpu/$RUN_NAME}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-10}"
DRY_RUN="${DRY_RUN:-0}"
SINGLE_GPU_CONFLICT_MODE="${SINGLE_GPU_CONFLICT_MODE:-warn}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"

DOWNSTREAM_MODE="${DOWNSTREAM_MODE:-isolated}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-egoblur}"

if [[ "$MAX_GPUS" =~ [^0-9] ]]; then
  echo "ERROR: MAX_GPUS must be a non-negative integer"
  exit 1
fi

if [[ "$WAIT_POLL_SEC" =~ [^0-9] || "$WAIT_POLL_SEC" -lt 1 ]]; then
  echo "ERROR: WAIT_POLL_SEC must be an integer >= 1"
  exit 1
fi

if [[ "$SINGLE_GPU_CONFLICT_MODE" != "off" && "$SINGLE_GPU_CONFLICT_MODE" != "warn" && "$SINGLE_GPU_CONFLICT_MODE" != "stop" ]]; then
  echo "ERROR: SINGLE_GPU_CONFLICT_MODE must be one of: off|warn|stop"
  exit 1
fi

if [[ "$STRICT_HARDLINK" != "0" && "$STRICT_HARDLINK" != "1" ]]; then
  echo "ERROR: STRICT_HARDLINK must be 0 or 1"
  exit 1
fi

for cmd in docker nvidia-smi tmux python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd"
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose is not available."
  exit 1
fi

RUN_FULL="$REPO/run_full_pipeline.sh"
if [[ ! -x "$RUN_FULL" ]]; then
  echo "ERROR: run_full_pipeline.sh not found or not executable at $RUN_FULL"
  exit 1
fi

resolve_gpu_ids() {
  local raw="$1"
  if [[ "$raw" != "auto" ]]; then
    tr ', ' '\n\n' <<<"$raw" | awk 'NF'
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | awk 'NF'
}

mapfile -t GPU_LIST < <(resolve_gpu_ids "$GPU_IDS_RAW")
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs detected/resolved."
  exit 1
fi

if [[ "$MAX_GPUS" -gt 0 && ${#GPU_LIST[@]} -gt "$MAX_GPUS" ]]; then
  GPU_LIST=("${GPU_LIST[@]:0:$MAX_GPUS}")
fi

NUM_GPUS=${#GPU_LIST[@]}

PLANNED_CONTAINERS=()
for gpu_id in "${GPU_LIST[@]}"; do
  PLANNED_CONTAINERS+=("${CONTAINER_PREFIX}${gpu_id}")
done

if [[ "$NUM_GPUS" -eq 1 && "$SINGLE_GPU_CONFLICT_MODE" != "off" ]]; then
  mapfile -t RUNNING_COMFY_CONTAINERS < <(docker ps --format '{{.Names}}' | grep -E '^(comfyui-container|comfyui-g[0-9]+)$' || true)
  CONFLICT_CONTAINERS=()

  for c in "${RUNNING_COMFY_CONTAINERS[@]}"; do
    skip=0
    for p in "${PLANNED_CONTAINERS[@]}"; do
      if [[ "$c" == "$p" ]]; then
        skip=1
        break
      fi
    done
    if [[ "$skip" -eq 0 ]]; then
      CONFLICT_CONTAINERS+=("$c")
    fi
  done

  if [[ ${#CONFLICT_CONTAINERS[@]} -gt 0 ]]; then
    echo "Single-GPU guard: detected other running comfyui containers: ${CONFLICT_CONTAINERS[*]}"
    if [[ "$SINGLE_GPU_CONFLICT_MODE" == "stop" ]]; then
      echo "Single-GPU guard mode=stop -> stopping conflicting containers..."
      docker stop "${CONFLICT_CONTAINERS[@]}" >/dev/null
      echo "Stopped conflicting containers."
    else
      echo "Single-GPU guard mode=warn -> continuing without stopping them."
    fi
  fi
fi

mkdir -p "$WORK_ROOT/shards" "$WORK_ROOT/jobs" "$REPO/logs"

MANIFEST_JSON="$WORK_ROOT/shard_manifest.json"
COUNT_JSON="$WORK_ROOT/shard_counts.json"

python3 - <<'PY' "$SRC" "$WORK_ROOT/shards" "$NUM_GPUS" "$MANIFEST_JSON" "$COUNT_JSON" "$STRICT_HARDLINK"
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
shards_root = Path(sys.argv[2]).resolve()
num_gpus = int(sys.argv[3])
manifest_path = Path(sys.argv[4]).resolve()
count_path = Path(sys.argv[5]).resolve()
strict_hardlink = sys.argv[6] == "1"

exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
files = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise SystemExit(f"No image files found in {src}")

for i in range(num_gpus):
    d = shards_root / f"gpu{i}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

manifest = []
counts = {str(i): 0 for i in range(num_gpus)}

for idx, src_file in enumerate(files):
    shard_i = idx % num_gpus
    shard_dir = shards_root / f"gpu{shard_i}"
    stem = src_file.stem
    suffix = src_file.suffix.lower()
    dst = shard_dir / f"{stem}{suffix}"
    k = 1
    while dst.exists():
        dst = shard_dir / f"{stem}__dup{k:03d}{suffix}"
        k += 1

    try:
        dst.hardlink_to(src_file)
    except OSError as exc:
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink failed (STRICT_HARDLINK=1): {src_file} -> {dst}: {exc}"
            )
        shutil.copy2(src_file, dst)

    manifest.append(
        {
            "src": str(src_file),
            "dst": str(dst),
            "shard_index": shard_i,
            "dst_name": dst.name,
        }
    )
    counts[str(shard_i)] += 1

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
count_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

print(f"input_images={len(files)}")
for i in range(num_gpus):
    print(f"shard_gpu_index_{i}_count={counts[str(i)]}")
print(f"manifest={manifest_path}")
PY

echo "RUN_NAME=$RUN_NAME"
echo "SRC=$SRC"
echo "NUM_GPUS=$NUM_GPUS"
echo "GPU_IDS=${GPU_LIST[*]}"
echo "WORK_ROOT=$WORK_ROOT"
echo "DOWNSTREAM_MODE=$DOWNSTREAM_MODE"
echo "STOP_AFTER_STAGE=$STOP_AFTER_STAGE"
echo "STRICT_HARDLINK=$STRICT_HARDLINK"
echo "DRY_RUN=$DRY_RUN"
echo "SINGLE_GPU_CONFLICT_MODE=$SINGLE_GPU_CONFLICT_MODE"

install_nvidia_container_toolkit_once() {
  if command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit already installed."
    return 0
  fi

  echo "Installing NVIDIA Container Toolkit..."

  if command -v sudo >/dev/null 2>&1; then
    SUDO=sudo
  else
    SUDO=
  fi

  $SUDO sh -c 'apt-get update && apt-get install -y curl gpg ca-certificates' 2>/dev/null || true
  $SUDO mkdir -p /usr/share/keyrings 2>/dev/null || true
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true

  if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64" | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  else
    wget -qO- https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO apt-key add - 2>/dev/null || true
    echo "deb https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  fi

  $SUDO apt-get update 2>/dev/null || true
  $SUDO apt-get install -y nvidia-container-toolkit 2>/dev/null || true
  $SUDO nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
  $SUDO systemctl restart docker 2>/dev/null || true

  if command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit installed successfully."
  else
    echo "WARNING: NVIDIA Container Toolkit installation may have failed."
  fi
}

echo "Checking NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  install_nvidia_container_toolkit_once
else
  echo "NVIDIA Container Toolkit already working."
fi

LAUNCH_PLAN="$WORK_ROOT/launch_plan.tsv"
: > "$LAUNCH_PLAN"

for idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$idx]}"
  shard_dir="$WORK_ROOT/shards/gpu${idx}"
  shard_count=$(python3 - <<'PY' "$shard_dir"
from pathlib import Path
import sys
p = Path(sys.argv[1])
exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
print(sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts))
PY
)

  if [[ "$shard_count" -eq 0 ]]; then
    echo "Skipping GPU $gpu_id (empty shard)"
    continue
  fi

  container_name="${CONTAINER_PREFIX}${gpu_id}"
  comfy_port=$((BASE_COMFY_PORT + idx))
  batch_name="${RUN_NAME}-g${gpu_id}"
  comfy_data_dir="$COMFYUI_DATA_ROOT/$container_name"
  session_name="${TMUX_SESSION_PREFIX}-${RUN_NAME}-g${gpu_id}"
  rc_file="$WORK_ROOT/rc_gpu${gpu_id}.txt"
  summary_file="$WORK_ROOT/summary_gpu${gpu_id}.txt"
  job_script="$WORK_ROOT/jobs/gpu${gpu_id}.sh"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export SRC="$shard_dir"
export INPUT_MODE="staged"
export BATCH_NAME="$batch_name"
export CONTAINER_NAME="$container_name"
export COMFYUI_DATA_DIR="$comfy_data_dir"
export COMFY_PORT="$comfy_port"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export DOWNSTREAM_MODE="$DOWNSTREAM_MODE"
export COMFY_STOP_CONTAINERS="$container_name"
export STOP_AFTER_STAGE="$STOP_AFTER_STAGE"
export FINAL_OUTPUT_DIR=""
export SAM3_WORKERS="${SAM3_WORKERS:-4}"
export SAM3_RESIZE_WIDTH="${SAM3_RESIZE_WIDTH:-4000}"
export SAM3_RESIZE_HEIGHT="${SAM3_RESIZE_HEIGHT:-2000}"
export SAM3_GLARE_THRESHOLD="${SAM3_GLARE_THRESHOLD:-0.4}"
export SAM3_TILE_ROWS="${SAM3_TILE_ROWS:-2}"
export SAM3_TILE_COLS="${SAM3_TILE_COLS:-1}"
export SAM3_SCRIPT="${SAM3_SCRIPT:-sam3_tiled_mask.py}"
export POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-4}"
export EGOBLUR_WORKERS="${EGOBLUR_WORKERS:-4}"
export LAPLACIAN_DILATION="${LAPLACIAN_DILATION:-1}"
export LAPLACIAN_BLUR="${LAPLACIAN_BLUR:-10}"
export LAPLACIAN_LEVELS="${LAPLACIAN_LEVELS:-7}"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/models}"
export MODELS_COMFYUI_DIR="${MODELS_COMFYUI_DIR:-${MODELS_ROOT:-$REPO/models}/comfyui}"
export MODELS_EGOBLUR_DIR="${MODELS_EGOBLUR_DIR:-${MODELS_ROOT:-$REPO/models}/egoblur_gen2}"
export MODELS_PRIVACY_DIR="${MODELS_PRIVACY_DIR:-${MODELS_ROOT:-$REPO/models}/privacy_blur}"
export AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
export FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
export STRICT_HARDLINK="${STRICT_HARDLINK:-1}"
export PRIVACY_WORKERS="${PRIVACY_WORKERS:-2}"
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
export PRIVACY_PYTHON_BIN="${PRIVACY_PYTHON_BIN:-/data/.venv/bin/python}"
export COMFY_IMAGE_NODE_ID="${COMFY_IMAGE_NODE_ID:-91}"
export COMFY_MASK_NODE_ID="${COMFY_MASK_NODE_ID:-34}"
export COMFY_SAM3_MASK_NODE_ID="${COMFY_SAM3_MASK_NODE_ID:-60}"
export SKY_REFERENCE_SOURCE="${SKY_REFERENCE_SOURCE:-$REPO/inpainting-workflow-master/reference_sky.png}"
export SKY_REFERENCE_FILENAME="${SKY_REFERENCE_FILENAME:-chrome_xWUjmfs7m4.png}"
export COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-300}"
export COMFY_READY_POLL="${COMFY_READY_POLL:-2}"
export NVIDIA_CUDA_TEST_IMAGE="${NVIDIA_CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu22.04}"
export COMFY_IMAGE="${COMFY_IMAGE:-amanbagrecha/container-comfyui:latest}"
export TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/.cache/torch}"
export RESET_CONTAINER_BEFORE_RUN="${RESET_CONTAINER_BEFORE_RUN:-1}"

"$RUN_FULL"
rc=\$?
printf '%s\n' "\$rc" > "$rc_file"

latest_summary=\$(ls -t "$REPO"/logs/fullrun_*"${container_name}"*.summary.txt 2>/dev/null | head -n 1 || true)
printf '%s\n' "\$latest_summary" > "$summary_file"
exit "\$rc"
EOF

  chmod +x "$job_script"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$gpu_id" "$idx" "$container_name" "$batch_name" "$comfy_data_dir" "$session_name" "$rc_file" "$summary_file" >> "$LAUNCH_PLAN"

  echo "Planned GPU=$gpu_id shard_idx=$idx count=$shard_count container=$container_name batch=$batch_name port=$comfy_port session=$session_name"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  tmux new-session -d -s "$session_name" "$job_script"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. Launch plan: $LAUNCH_PLAN"
  exit 0
fi

if [[ ! -s "$LAUNCH_PLAN" ]]; then
  echo "ERROR: no shard jobs were launched."
  exit 1
fi

echo "Waiting for shard sessions to complete..."
while true; do
  alive=0
  while IFS=$'\t' read -r _ _ _ _ _ session _ _; do
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

FAIL=0
TOTAL_IN=0
TOTAL_OUT=0

while IFS=$'\t' read -r gpu_id idx container batch comfy_data_dir _ rc_file summary_file; do
  rc="missing"
  if [[ -f "$rc_file" ]]; then
    rc="$(tr -d '[:space:]' < "$rc_file")"
  fi

  in_count=$(python3 - <<'PY' "$WORK_ROOT/shards/gpu${idx}"
from pathlib import Path
import sys
exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
p = Path(sys.argv[1])
print(sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts))
PY
)

  if [[ "$STOP_AFTER_STAGE" == "egoblur" ]]; then
    out_dir="$comfy_data_dir/output-egoblur/$batch"
  elif [[ "$STOP_AFTER_STAGE" == "sam3" ]]; then
    out_dir="$comfy_data_dir/output-sam3-mask/$batch"
  elif [[ "$STOP_AFTER_STAGE" == "postprocess" ]]; then
    out_dir="$comfy_data_dir/output-postprocessed/$batch"
  else
    out_dir="$comfy_data_dir/output/$batch"
  fi

  out_count=$(python3 - <<'PY' "$out_dir"
from pathlib import Path
import sys
exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
p = Path(sys.argv[1])
if not p.exists():
    print(0)
else:
    print(sum(1 for f in p.rglob('*') if f.is_file() and f.suffix.lower() in exts))
PY
)

  TOTAL_IN=$((TOTAL_IN + in_count))
  TOTAL_OUT=$((TOTAL_OUT + out_count))

  if [[ "$rc" != "0" ]]; then
    FAIL=$((FAIL + 1))
  fi

  summary_path=""
  if [[ -f "$summary_file" ]]; then
    summary_path="$(cat "$summary_file")"
  fi

  echo "gpu=$gpu_id rc=$rc in_count=$in_count out_count=$out_count batch=$batch summary=${summary_path:-none}"
done < "$LAUNCH_PLAN"

echo "TOTAL_IN=$TOTAL_IN"
echo "TOTAL_OUT=$TOTAL_OUT"
echo "FAILED_SHARDS=$FAIL"

if [[ -n "${FINAL_OUTPUT_DIR:-}" ]]; then
  MERGED_ROOT="$FINAL_OUTPUT_DIR/$RUN_NAME"
  mkdir -p "$MERGED_ROOT"
  echo "Collecting per-shard outputs into $MERGED_ROOT"

  while IFS=$'\t' read -r gpu_id _ _ batch comfy_data_dir _ _ _; do
    if [[ "$STOP_AFTER_STAGE" == "egoblur" ]]; then
      src_dir="$comfy_data_dir/output-egoblur/$batch"
    elif [[ "$STOP_AFTER_STAGE" == "postprocess" ]]; then
      src_dir="$comfy_data_dir/output-postprocessed/$batch"
    else
      src_dir="$comfy_data_dir/output/$batch"
    fi
    dst_dir="$MERGED_ROOT/gpu${gpu_id}"
    mkdir -p "$dst_dir"

    python3 - <<'PY' "$src_dir" "$dst_dir" "$STRICT_HARDLINK"
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
strict_hardlink = sys.argv[3] == "1"
if not src.exists():
    raise SystemExit(0)

for p in src.rglob('*'):
    if not p.is_file():
        continue
    rel = p.relative_to(src)
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(p)
    except OSError as exc:
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink failed (STRICT_HARDLINK=1): {p} -> {target}: {exc}"
            )
        shutil.copy2(p, target)
PY

  done < "$LAUNCH_PLAN"

  echo "MERGED_OUTPUT=$MERGED_ROOT"
fi

if [[ "$FAIL" -gt 0 ]]; then
  echo "ERROR: $FAIL shard(s) failed."
  exit 1
fi

echo "Multi-GPU run completed successfully."
