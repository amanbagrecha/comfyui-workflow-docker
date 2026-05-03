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
#
# Per-run structured state is emitted to:
#   $REPO/logs/orchestrate_<run_name>.events.jsonl
# which the vast_controller polls over SSH to populate the run_events table.

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
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"

IMGS_DIR="/workspace/imgs"
OUTPUTS_DIR="/workspace/outputs"
TARS_DIR="/workspace/tars"
NATIVE_DATA="$REPO/native_data"
TMP_DIR="$REPO/tmp/multigpu"
LOGS_DIR="/workspace/logs/orchestrator"
EVENTS_ROOT="$REPO/logs"
PIPELINE="$REPO/run_multi_gpu_pipeline.sh"
PIPELINE_HELPERS="$REPO/inpainting-workflow-master/pipeline_helpers.py"
DOWNLOADER="$REPO/inpainting-workflow-master/s3_parallel_download.py"

mkdir -p "$LOGS_DIR" "$TARS_DIR" "$OUTPUTS_DIR" "$EVENTS_ROOT"

ORCH_LOG="$LOGS_DIR/orchestrator.log"

# ── Logging (orchestrator.log = human scrollback) ─────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ORCH_LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: $*" | tee -a "$ORCH_LOG"; }

cleanup_transient_run() {
  local run_name="$1"

  log "[$run_name] Deleting transient local data..."

  for g in 0 1 2 3 4 5 6 7; do
    rm -rf "$NATIVE_DATA/gpu${g}/input/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output-postprocessed/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output-egoblur/${run_name}-g${g}"
    rm -rf "$NATIVE_DATA/gpu${g}/output-sam3-mask/${run_name}-g${g}"
  done

  rm -rf "$TMP_DIR/${run_name}"
  rm -rf "$IMGS_DIR/${run_name}"
}

cleanup_output_dir() {
  local run_name="$1"
  local out_dir="$OUTPUTS_DIR/$run_name"

  if [[ -e "$out_dir" ]]; then
    log "[$run_name] Deleting output directory: $out_dir"
    rm -rf "$out_dir"
  fi
}

cleanup_stale_local_runs() {
  local current_run="$1"
  local path base run_name
  declare -A seen_runs=()

  for path in "$IMGS_DIR"/* "$TMP_DIR"/* "$NATIVE_DATA"/gpu*/input/*; do
    [[ -e "$path" ]] || continue
    base="${path##*/}"

    if [[ "$path" == "$NATIVE_DATA"/gpu*/input/* ]]; then
      run_name="${base%-g*}"
    else
      run_name="$base"
    fi

    [[ -n "$run_name" ]] || continue
    [[ "$run_name" == "$current_run" ]] && continue
    [[ -n "${seen_runs[$run_name]:-}" ]] && continue
    seen_runs["$run_name"]=1

    if tmux has-session -t "run_${run_name}" 2>/dev/null; then
      continue
    fi

    cleanup_transient_run "$run_name"
  done
}

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

# ── Resolve PYTHON_BIN now that .venv exists ──────────────────────────────────
# shellcheck disable=SC1091
. "$REPO/scripts/runtime.sh"
require_repo_python

# ── Structured event emission ─────────────────────────────────────────────────
# Per-run events file: $EVENTS_ROOT/orchestrate_<run_name>.events.jsonl
# Each call appends one JSON line via pipeline_helpers.py append-event.
emit_event() {
  # Usage: emit_event <run_name> <--event E> <--status S> [extra append-event args...]
  local run_name="$1"
  shift
  local events_file="$EVENTS_ROOT/orchestrate_${run_name}.events.jsonl"
  "$PYTHON_BIN" "$PIPELINE_HELPERS" append-event \
    --file "$events_file" \
    --run-type orchestrate \
    --run-id "$run_name" \
    --script orchestrate.sh \
    "$@" >/dev/null 2>&1 || true
}

declare -gA STAGE_STARTED=()
stage_start() {
  # Usage: stage_start <run_name> <stage>
  local run_name="$1" stage="$2"
  STAGE_STARTED["${run_name}:${stage}"]=$(date +%s)
  emit_event "$run_name" --event stage_start --status running --stage "$stage"
}

stage_end() {
  # Usage: stage_end <run_name> <stage> <success|failure> [extra append-event args...]
  local run_name="$1" stage="$2" status="$3"
  shift 3
  local key="${run_name}:${stage}"
  local started="${STAGE_STARTED[$key]:-0}"
  local elapsed=0
  if [[ "$started" -gt 0 ]]; then
    elapsed=$(( $(date +%s) - started ))
  fi
  unset 'STAGE_STARTED[$key]'
  emit_event "$run_name" --event stage_end --status "$status" --stage "$stage" \
    --elapsed-sec "$elapsed" "$@"
}

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
    --workers "$DOWNLOAD_WORKERS" \
    2>&1 | tee "$log_file"

  return ${PIPESTATUS[0]}
}

# ── Tar, upload, verify, then delete everything ───────────────────────────────
# Emits tar / upload / verify / cleanup stage events. Returns 0 on full success,
# nonzero on any stage failure (caller decides run_end status).
tar_upload_cleanup() {
  local run_name="$1"
  local src_dir="$IMGS_DIR/$run_name"
  local out_dir="$OUTPUTS_DIR/$run_name"
  local tar_file="$TARS_DIR/${run_name}.tar"
  local s3dest="$S3_UPLOAD_PATH/${run_name}.tar"

  # 1. Tar
  stage_start "$run_name" tar
  log "[$run_name] Tarring output..."
  if ! tar -cf "$tar_file" -C "$OUTPUTS_DIR" "$run_name" 2>&1 | tee -a "$ORCH_LOG"; then
    stage_end "$run_name" tar failure --error "tar -cf failed"
    fail "[$run_name] Tar failed"
    return 1
  fi
  local tar_bytes
  tar_bytes=$(stat -c%s "$tar_file" 2>/dev/null || echo 0)
  log "[$run_name] Tar done: $(du -sh "$tar_file" | cut -f1)"
  stage_end "$run_name" tar success --metric tar_bytes="$tar_bytes"

  # 2. Upload (skipped if already on S3)
  stage_start "$run_name" upload
  local upload_skipped=0
  if aws --profile "$UPLOAD_PROFILE" s3 ls "$s3dest" > /dev/null 2>&1; then
    upload_skipped=1
    log "[$run_name] Already on S3, skipping upload"
  else
    log "[$run_name] Uploading to $s3dest..."
    if ! aws --profile "$UPLOAD_PROFILE" s3 cp "$tar_file" "$s3dest" \
        2>&1 | tee -a "$ORCH_LOG"; then
      stage_end "$run_name" upload failure --error "aws s3 cp failed"
      fail "[$run_name] Upload failed"
      return 1
    fi
  fi
  if [[ "$upload_skipped" -eq 1 ]]; then
    stage_end "$run_name" upload success --reason already_on_s3
  else
    stage_end "$run_name" upload success --metric upload_bytes="$tar_bytes"
  fi

  # 3. Verify: compare sizes
  stage_start "$run_name" verify
  local s3_size local_size
  s3_size=$(aws --profile "$UPLOAD_PROFILE" s3 ls "$s3dest" | awk '{print $3}')
  local_size=$(stat -c%s "$tar_file")

  if [[ "$s3_size" != "$local_size" ]]; then
    stage_end "$run_name" verify failure \
      --error "size mismatch local=$local_size s3=$s3_size"
    fail "[$run_name] Size mismatch — local=$local_size s3=$s3_size. NOT deleting."
    return 1
  fi
  log "[$run_name] Upload verified ($local_size bytes). Starting cleanup..."
  stage_end "$run_name" verify success --metric verified_bytes="$local_size"

  # 4. Cleanup — transient local data + outputs + local tar (everything successful)
  stage_start "$run_name" cleanup
  cleanup_transient_run "$run_name"
  cleanup_output_dir "$run_name"
  log "[$run_name] Deleting local tar: $tar_file"
  rm -f "$tar_file"
  stage_end "$run_name" cleanup success
  log "[$run_name] Cleanup complete. Disk free: $(df -h /workspace | awk 'NR==2{print $4}')"
}

# Wait for a launched pipeline to finish, then run finalize stages.
# Called from both the main loop (previous run) and the tail block (last run).
finalize_previous_run() {
  local prev_pid="$1" prev_name="$2"
  if wait "$prev_pid"; then
    stage_end "$prev_name" pipeline success
    log "Run finished: $prev_name"
    if tar_upload_cleanup "$prev_name"; then
      emit_event "$prev_name" --event run_end --status success
    else
      emit_event "$prev_name" --event run_end --status failure \
        --error "tar_upload_cleanup failed"
      fail "tar/upload/cleanup failed: $prev_name"
    fi
  else
    local rc=$?
    stage_end "$prev_name" pipeline failure --exit-code "$rc"
    emit_event "$prev_name" --event run_end --status failure --exit-code "$rc"
    fail "Pipeline failed: $prev_name (rc=$rc)"
    cleanup_output_dir "$prev_name"
  fi
  sync_logs
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

  cleanup_stale_local_runs "$run_name"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: download s3://$WASABI_BUCKET/$prefix -> $src_dir"
    log "DRY: launch tmux run_${run_name}"
    log "DRY: tar_upload_cleanup $run_name"
    continue
  fi

  emit_event "$run_name" --event run_start --status running \
    --param prefix="$prefix" \
    --param bucket="$WASABI_BUCKET" \
    --param s3_upload_path="$S3_UPLOAD_PATH" \
    --param every_nth="$EVERY_NTH" \
    --param position="$n/$total"

  # ── Skip if already uploaded to S3 ────────────────────────────────────────
  if aws --profile "$UPLOAD_PROFILE" s3 ls "$S3_UPLOAD_PATH/${run_name}.tar" > /dev/null 2>&1; then
    log "[$n/$total] SKIP $run_name — already on S3"
    cleanup_transient_run "$run_name"
    emit_event "$run_name" --event run_end --status success --reason already_on_s3
    continue
  fi

  # ── Download current batch ─────────────────────────────────────────────────
  dl_log="$LOGS_DIR/download_${run_name}.log"
  stage_start "$run_name" download
  if ! download_prefix "$prefix" "$src_dir" "$dl_log"; then
    stage_end "$run_name" download failure --error "s3_parallel_download.py failed"
    emit_event "$run_name" --event run_end --status failure \
      --error "download failed"
    fail "Download failed: $prefix"
    sync_logs
    continue
  fi
  dl_count=$(ls "$src_dir" 2>/dev/null | wc -l)
  validate_output=$("$PYTHON_BIN" "$PIPELINE_HELPERS" assert-image-size \
    --path "$src_dir" 2>&1)
  validate_rc=$?
  printf '%s\n' "$validate_output" | tee -a "$dl_log" "$ORCH_LOG" >/dev/null
  if [[ "$validate_rc" -ne 0 ]]; then
    validate_error=$(printf '%s' "$validate_output" | tr '\n' ' ' | tr -s ' ')
    stage_end "$run_name" download failure \
      --error "$validate_error" \
      --metric input_count="$dl_count"
    emit_event "$run_name" --event run_end --status failure \
      --error "$validate_error"
    fail "Download validation failed: $prefix"
    cleanup_transient_run "$run_name"
    sync_logs
    continue
  fi
  stage_end "$run_name" download success --metric input_count="$dl_count"
  log "[$n/$total] Download complete: $run_name ($dl_count files)"

  # ── Wait for previous run to finish before launching new one ──────────────
  if [[ ${#RUN_PIDS[@]} -gt 0 ]]; then
    prev_pid="${RUN_PIDS[-1]}"
    prev_name="${RUN_NAMES[-1]}"
    log "[$n/$total] Waiting for run $prev_name..."
    finalize_previous_run "$prev_pid" "$prev_name"
  fi

  # ── Launch GPU run in its own tmux session ────────────────────────────────
  run_log="$LOGS_DIR/run_${run_name}.log"
  rc_file="$LOGS_DIR/rc_${run_name}.txt"
  log "Launching tmux session: run_${run_name}"
  # ── Resolve perspective mask for this run ────────────────────────────────
  _mask_rel=$("$PYTHON_BIN" -c "
import json, sys
try:
    idx = json.load(open('$REPO/perspective_mask_index.json'))
    print(idx.get('${run_name}', ''))
except Exception as e:
    sys.stderr.write(str(e) + '\n')
    print('')
" 2>>"$ORCH_LOG")
  if [ -z "$_mask_rel" ]; then
    stage_end "$run_name" pipeline failure --error "no_perspective_mask"
    emit_event "$run_name" --event run_end --status failure \
      --error "run_id not in perspective_mask_index.json — no mask assigned, skipping"
    fail "[$run_name] No perspective mask found in index — skipping run"
    cleanup_transient_run "$run_name"
    sync_logs
    continue
  fi
  _mask_abs="$REPO/$_mask_rel"
  log "[$run_name] Perspective mask: $_mask_abs"

  stage_start "$run_name" pipeline

  tmux new-session -d -s "run_${run_name}" \
    "set -o pipefail; \
     SKIP_HOST_BOOTSTRAP=1 \
     RUN_NAME=${run_name} \
     SRC=${src_dir} \
     FINAL_OUTPUT_DIR=${OUTPUTS_DIR} \
     PERSPECTIVE_MASK=${_mask_abs} \
     bash ${PIPELINE} 2>&1 | tee ${run_log}; \
     echo \$? > ${rc_file}"

  # Background waiter: resolves when tmux session exits
  (
    while tmux has-session -t "run_${run_name}" 2>/dev/null; do sleep 20; done
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
  log "Waiting for final run: $last_name"
  finalize_previous_run "$last_pid" "$last_name"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All batches complete."
log "Disk free: $(df -h /workspace | awk 'NR==2{print $4}')"
sync_logs
