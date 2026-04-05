#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"

SRC="${SRC:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "ERROR: SRC must be an existing directory"
  exit 1
fi

RUN_NAME="${RUN_NAME:-multigpu-native-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS_RAW="${GPU_IDS:-auto}"
MAX_GPUS="${MAX_GPUS:-0}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-10}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-mgpu-native}"
WORK_ROOT="${WORK_ROOT:-$REPO/tmp/native/$RUN_NAME}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"

LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/multigpu_native_${RUN_NAME}.log"
mkdir -p "$LOG_DIR" "$WORK_ROOT/shards" "$WORK_ROOT/jobs"
exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

for cmd in nvidia-smi tmux python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: missing $cmd"; exit 1; }
done

resolve_gpu_ids() {
  local raw="$1"
  if [[ "$raw" != "auto" ]]; then
    tr ', ' '\n\n' <<<"$raw" | awk 'NF'
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | awk 'NF'
}

mapfile -t GPU_IDS < <(resolve_gpu_ids "$GPU_IDS_RAW")
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs found"
  exit 1
fi

if [[ "$MAX_GPUS" =~ ^[0-9]+$ ]] && [[ "$MAX_GPUS" -gt 0 ]] && [[ "$MAX_GPUS" -lt ${#GPU_IDS[@]} ]]; then
  GPU_IDS=("${GPU_IDS[@]:0:$MAX_GPUS}")
fi

NUM_GPUS="${#GPU_IDS[@]}"
MANIFEST_JSON="$WORK_ROOT/manifest.json"
COUNT_JSON="$WORK_ROOT/counts.json"
python3 "$PIPELINE_HELPERS" split-shards \
  --src "$SRC" \
  --shards-root "$WORK_ROOT/shards" \
  --num-gpus "$NUM_GPUS" \
  --manifest-json "$MANIFEST_JSON" \
  --count-json "$COUNT_JSON" \
  --strict-hardlink "$STRICT_HARDLINK"

for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  shard_dir="$WORK_ROOT/shards/gpu${idx}"
  session="${TMUX_SESSION_PREFIX}-${RUN_NAME}-g${gpu}"
  batch_name="${RUN_NAME}_g${gpu}"
  cmd=(
    "cd $REPO"
    "NVIDIA_VISIBLE_DEVICES=$gpu"
    "SRC=$shard_dir"
    "RUN_ID=$batch_name"
    "BATCH_NAME=$batch_name"
    "FINAL_OUTPUT_DIR=${FINAL_OUTPUT_DIR:+$FINAL_OUTPUT_DIR/$RUN_NAME/gpu$gpu}"
    "./run_full_pipeline_native.sh"
  )
  tmux new-session -d -s "$session" "${cmd[*]}"
  echo "Started shard gpu=$gpu session=$session src=$shard_dir"
done

echo "Waiting for all shard sessions to finish..."
while true; do
  alive=0
  for gpu in "${GPU_IDS[@]}"; do
    session="${TMUX_SESSION_PREFIX}-${RUN_NAME}-g${gpu}"
    if tmux has-session -t "$session" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep "$WAIT_POLL_SEC"
done

echo "All native shard sessions finished."
