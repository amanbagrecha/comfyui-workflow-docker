#!/bin/bash
# orchestrate.sh — Sequential download + pipelined GPU runs with auto tar/upload/cleanup
#
# Usage:
#   bash orchestrate.sh <prefix1> <prefix2> ...
#
# Each prefix is a Wasabi path like "1234567890_9876543210/session_id"
# Folder naming follows s3_parallel_download.py: parent_child
#
# Required env vars (set by bootstrap or manually):
#   AWS_DOWNLOAD_PROFILE  — Wasabi profile (default: download)
#   AWS_UPLOAD_PROFILE    — S3 upload profile (default: s3)
#
# Optional:
#   WASABI_BUCKET         — Source bucket (default: pano-bkp)
#   S3_UPLOAD_PATH        — Upload destination (default: s3://aipanoexport-batch2/panoramic_clean)
#   EVERY_NTH             — Download every Nth file (default: 3)
#   DRY_RUN               — Set to 1 to print plan without executing

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_PROFILE="${AWS_DOWNLOAD_PROFILE:-wasabi}"
UPLOAD_PROFILE="${AWS_UPLOAD_PROFILE:-s3}"
WASABI_BUCKET="${WASABI_BUCKET:-pano-bkp}"
S3_UPLOAD_PATH="${S3_UPLOAD_PATH:-s3://aipanoexport-batch2/panoramic_clean}"
S3_LOGS_PATH="${S3_LOGS_PATH:-s3://aipanoexport-batch2/logs/$(hostname)}"
EVERY_NTH="${EVERY_NTH:-3}"
DRY_RUN="${DRY_RUN:-0}"

IMGS_DIR="/workspace/imgs"
OUTPUTS_DIR="/workspace/outputs"
TARS_DIR="/workspace/tars"
NATIVE_DATA="$REPO/native_data"
TMP_DIR="$REPO/tmp/multigpu"
LOGS_DIR="/workspace/logs/orchestrator"
PIPELINE="$REPO/run_multi_gpu_pipeline.sh"
DOWNLOADER="$REPO/inpainting-workflow-master/s3_parallel_download.py"

mkdir -p "$LOGS_DIR" "$TARS_DIR" "$OUTPUTS_DIR"

ORCH_LOG="$LOGS_DIR/orchestrator.log"
STATUS_FILE="$LOGS_DIR/status.txt"
FAILURES_FILE="$LOGS_DIR/failures.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ORCH_LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: $*" | tee -a "$FAILURES_FILE" "$ORCH_LOG"; }
status() { echo "$*" > "$STATUS_FILE"; log "STATUS: $*"; }

sync_logs() {
  aws --profile "$UPLOAD_PROFILE" s3 sync "$LOGS_DIR/" "$S3_LOGS_PATH/orchestrate/" \
    --quiet --no-progress 2>/dev/null || true
  aws --profile "$UPLOAD_PROFILE" s3 sync "$REPO/logs/" "$S3_LOGS_PATH/pipeline/" \
    --quiet --no-progress 2>/dev/null || true
}

# ── One-time host setup ───────────────────────────────────────────────────────
# Runs setup_host_environment.sh once to ensure .venv, ComfyUI, and models are
# in place. After this, all pipeline calls use SKIP_HOST_BOOTSTRAP=1 which is
# safe because require_repo_python checks for .venv/bin/python — if we skip
# setup before .venv exists it hard-fails, hence we always run it once here.
SETUP_DONE_FLAG="$REPO/.orchestrate_setup_done"
if [[ ! -f "$SETUP_DONE_FLAG" ]]; then
  log "Running setup_host_environment.sh (one-time)..."
  INSTALL_SYSTEM_PACKAGES=1 \
  DOWNLOAD_MODELS=1 \
    bash "$REPO/setup_host_environment.sh" 2>&1 | tee -a "$ORCH_LOG"
  touch "$SETUP_DONE_FLAG"
  log "Host setup complete."
else
  log "Host already set up (found $SETUP_DONE_FLAG), skipping."
fi

# ── Derive folder name from prefix (prefix IS the run_id, e.g. 1021231_123123) ─
prefix_to_name() {
  # Strip any trailing slash — the prefix is already the run_id
  echo "${1%/}"
}

# ── Download one prefix from Wasabi ──────────────────────────────────────────
download_prefix() {
  local prefix="$1"
  local dest_dir="$2"
  local log_file="$3"

  log "Downloading s3://$WASABI_BUCKET/$prefix -> $dest_dir"
  mkdir -p "$dest_dir"

  # Use s3_parallel_download.py via uv run (auto-installs deps at runtime)
  AWS_PROFILE="$DOWNLOAD_PROFILE" uv run "$DOWNLOADER" \
    --bucket "$WASABI_BUCKET" \
    --prefix "$prefix" \
    --dest "$(dirname "$dest_dir")" \
    --every "$EVERY_NTH" \
    --workers 100 \
    2>&1 | tee "$log_file"

  return ${PIPESTATUS[0]}
}

# ── Tar, upload, verify, then delete everything ───────────────────────────────
tar_upload_cleanup() {
  local run_name="$1"
  local src_dir="$IMGS_DIR/$run_name"
  local out_dir="$OUTPUTS_DIR/$run_name"
  local tar_file="$TARS_DIR/${run_name}.tar"
  local s3dest="$S3_UPLOAD_PATH/${run_name}.tar"

  # 1. Tar
  log "[$run_name] Tarring output..."
  if ! tar -cf "$tar_file" -C "$OUTPUTS_DIR" "$run_name" 2>&1 | tee -a "$ORCH_LOG"; then
    fail "[$run_name] Tar failed"
    return 1
  fi
  log "[$run_name] Tar done: $(du -sh "$tar_file" | cut -f1)"

  # 2. Check if already on S3
  if aws --profile "$UPLOAD_PROFILE" s3 ls "$s3dest" > /dev/null 2>&1; then
    log "[$run_name] Already on S3, skipping upload"
  else
    log "[$run_name] Uploading to $s3dest..."
    if ! aws --profile "$UPLOAD_PROFILE" s3 cp "$tar_file" "$s3dest" \
        2>&1 | tee -a "$ORCH_LOG"; then
      fail "[$run_name] Upload failed"
      return 1
    fi
  fi

  # 3. Verify: compare sizes
  local s3_size local_size
  s3_size=$(aws --profile "$UPLOAD_PROFILE" s3 ls "$s3dest" | awk '{print $3}')
  local_size=$(stat -c%s "$tar_file")

  if [[ "$s3_size" != "$local_size" ]]; then
    fail "[$run_name] Size mismatch — local=$local_size s3=$s3_size. NOT deleting."
    return 1
  fi
  log "[$run_name] Upload verified ($local_size bytes). Starting cleanup..."

  # 4. Cleanup — order respects hardlink chains (most space freed first)

  # Intermediates: unique inodes, safe to delete independently
  log "[$run_name] Deleting intermediates..."
  for g in 0 1 2 3 4 5 6 7; do
    rm -rf "$NATIVE_DATA/gpu${g}/output/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output-postprocessed/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output-sam3-mask/${run_name}-g${g}"
  done

  # Egoblur + output dir: hardlinked pair — must delete BOTH to free inode
  log "[$run_name] Deleting egoblur + output dir..."
  for g in 0 1 2 3 4 5 6 7; do
    rm -rf "$NATIVE_DATA/gpu${g}/output-egoblur/${run_name}-g${g}"
  done
  rm -rf "$out_dir"

  # Input: 3 hardlink refs — must delete ALL THREE to free inode
  log "[$run_name] Deleting input (3 hardlink refs)..."
  for g in 0 1 2 3 4 5 6 7; do
    rm -rf "$NATIVE_DATA/gpu${g}/input/${run_name}-g${g}"
  done
  rm -rf "$TMP_DIR/${run_name}"
  rm -rf "$src_dir"

  # Tar: confirmed on S3, safe to delete
  rm -f "$tar_file"
  log "[$run_name] Cleanup complete. Disk free: $(df -h /workspace | awk 'NR==2{print $4}')"
}

# ── Main orchestration loop ───────────────────────────────────────────────────
PREFIXES=("$@")
if [[ ${#PREFIXES[@]} -eq 0 ]]; then
  echo "Usage: $0 <prefix1> <prefix2> ..."
  exit 1
fi

log "Starting orchestration of ${#PREFIXES[@]} batches"
log "Download profile: $DOWNLOAD_PROFILE | Upload profile: $UPLOAD_PROFILE"
log "Wasabi bucket: $WASABI_BUCKET | Upload path: $S3_UPLOAD_PATH"
[[ "$DRY_RUN" == "1" ]] && log "DRY RUN — will print plan only"

# Parallel tracking
declare -a RUN_NAMES=()
declare -a RUN_PIDS=()

for i in "${!PREFIXES[@]}"; do
  prefix="${PREFIXES[$i]}"
  run_name="$(prefix_to_name "$prefix")"
  src_dir="$IMGS_DIR/$run_name"
  n=$((i + 1))
  total=${#PREFIXES[@]}

  status "[$n/$total] Downloading $run_name"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: download s3://$WASABI_BUCKET/$prefix -> $src_dir"
    log "DRY: launch tmux run_${run_name}"
    log "DRY: tar_upload_cleanup $run_name"
    continue
  fi

  # ── Skip if already uploaded to S3 ────────────────────────────────────────
  if aws --profile "$UPLOAD_PROFILE" s3 ls "$S3_UPLOAD_PATH/${run_name}.tar" > /dev/null 2>&1; then
    log "[$n/$total] SKIP $run_name — already on S3"
    continue
  fi

  # ── Download current batch ─────────────────────────────────────────────────
  dl_log="$LOGS_DIR/download_${run_name}.log"
  if ! download_prefix "$prefix" "$src_dir" "$dl_log"; then
    fail "Download failed: $prefix"
    sync_logs
    continue
  fi
  log "[$n/$total] Download complete: $run_name ($(ls "$src_dir" | wc -l) files)"

  # ── Wait for previous run to finish before launching new one ──────────────
  if [[ ${#RUN_PIDS[@]} -gt 0 ]]; then
    prev_pid="${RUN_PIDS[-1]}"
    prev_name="${RUN_NAMES[-1]}"
    status "[$n/$total] Waiting for run $prev_name..."
    if wait "$prev_pid"; then
      log "Run finished: $prev_name"
      tar_upload_cleanup "$prev_name" || fail "tar/upload/cleanup failed: $prev_name"
    else
      fail "Pipeline failed: $prev_name (rc=$?)"
    fi
    sync_logs
  fi

  # ── Launch GPU run in its own tmux session ────────────────────────────────
  run_log="$LOGS_DIR/run_${run_name}.log"
  rc_file="$LOGS_DIR/rc_${run_name}.txt"
  status "[$n/$total] Running $run_name"
  log "Launching tmux session: run_${run_name}"

  tmux new-session -d -s "run_${run_name}" \
    "set -o pipefail; \
     SKIP_HOST_BOOTSTRAP=1 \
     RUN_NAME=${run_name} \
     SRC=${src_dir} \
     FINAL_OUTPUT_DIR=${OUTPUTS_DIR} \
     bash ${PIPELINE} 2>&1 | tee ${run_log}; \
     echo \$? > ${rc_file}"

  # Background waiter: resolves when tmux session exits
  (
    while tmux has-session -t "run_${run_name}" 2>/dev/null; do sleep 100; done
    rc=$(cat "$rc_file" 2>/dev/null || echo 1)
    exit "$rc"
  ) &
  RUN_PIDS+=($!)
  RUN_NAMES+=("$run_name")
done

# ── Handle the last run ───────────────────────────────────────────────────────
if [[ ${#RUN_PIDS[@]} -gt 0 ]]; then
  last_pid="${RUN_PIDS[-1]}"
  last_name="${RUN_NAMES[-1]}"
  status "Waiting for final run: $last_name"
  if wait "$last_pid"; then
    log "Run finished: $last_name"
    tar_upload_cleanup "$last_name" || fail "tar/upload/cleanup failed: $last_name"
  else
    fail "Pipeline failed: $last_name (rc=$?)"
  fi
fi

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All batches complete."
if [[ -s "$FAILURES_FILE" ]]; then
  log "FAILURES:"
  cat "$FAILURES_FILE" | tee -a "$ORCH_LOG"
else
  log "No failures."
fi
log "Disk free: $(df -h /workspace | awk 'NR==2{print $4}')"
sync_logs
