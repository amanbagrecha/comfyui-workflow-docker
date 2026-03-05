#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="${SRC:-$REPO/comfyui_data/comfyui-g0/input/mgpu}"
WORKERS="${WORKERS:-2}"
RUN_NAME="${RUN_NAME:-privacy-blur-par${WORKERS}-$(date +%Y%m%d_%H%M%S)}"

FACE_MODEL="${FACE_MODEL:-$REPO/models/privacy_blur/face_yolov8n.pt}"
LP_MODEL="${LP_MODEL:-$REPO/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx}"

FACE_CONF="${FACE_CONF:-0.4}"
LP_CONF="${LP_CONF:-0.4}"
FACE_IOU="${FACE_IOU:-0.5}"
FACE_IMGSZ="${FACE_IMGSZ:-1024}"

DET_FACE_W="${DET_FACE_W:-1024}"
P360_DEVICE="${P360_DEVICE:-auto}"
BLUR_SCOPE="${BLUR_SCOPE:-roi}"
BLUR_BACKEND="${BLUR_BACKEND:-gpu}"
OUTPUT_MODE="${OUTPUT_MODE:-blur_only}"
JPG_QUALITY="${JPG_QUALITY:-85}"
PYTHON_BIN="${PYTHON_BIN:-/data/.venv/bin/python}"
STRICT_HARDLINK="${STRICT_HARDLINK:-1}"

if [[ "$STRICT_HARDLINK" != "0" && "$STRICT_HARDLINK" != "1" ]]; then
  echo "ERROR: STRICT_HARDLINK must be 0 or 1"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: PYTHON_BIN not executable and python3 not found."
    exit 1
  fi
fi

WORK_ROOT="${WORK_ROOT:-$REPO/tmp/privacy-blur/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/$RUN_NAME}"
OUT_ROOT="${OUT_ROOT:-$REPO/comfyui_data/final_outputs/$RUN_NAME/gpu0}"

mkdir -p "$WORK_ROOT/shards" "$WORK_ROOT/worker_outputs" "$LOG_ROOT" "$OUT_ROOT"

python3 - <<'PY' "$SRC" "$WORK_ROOT/shards" "$WORKERS" "$STRICT_HARDLINK"
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
shards = Path(sys.argv[2])
workers = int(sys.argv[3])
strict_hardlink = sys.argv[4] == '1'
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

images = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
if not images:
    raise SystemExit(f"No images found in {src}")

for i in range(workers):
    wdir = shards / f"w{i}"
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.mkdir(parents=True, exist_ok=True)

for idx, img in enumerate(images):
    wi = idx % workers
    dst = shards / f"w{wi}" / img.name
    try:
        dst.hardlink_to(img)
    except OSError as exc:
        if strict_hardlink:
            raise SystemExit(
                f"Hardlink failed (STRICT_HARDLINK=1): {img} -> {dst}: {exc}"
            )
        shutil.copy2(img, dst)

for i in range(workers):
    count = sum(1 for _ in (shards / f"w{i}").iterdir())
    print(f"worker_{i}_images={count}")
print(f"total_images={len(images)}")
PY

GPU_LOG="$LOG_ROOT/gpu_util.csv"
echo "timestamp,gpu_util_pct,mem_util_pct,mem_used_mb,mem_total_mb" > "$GPU_LOG"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 1
  done
) &
MON_PID=$!

START_TS=$(date +%s)

declare -a PIDS=()
declare -a WORKER_IDS=()

for ((i = 0; i < WORKERS; i++)); do
  IN_DIR="$WORK_ROOT/shards/w$i"
  COUNT=$(python3 - <<'PY' "$IN_DIR"
from pathlib import Path
import sys
p = Path(sys.argv[1])
print(sum(1 for _ in p.iterdir()))
PY
)
  if [[ "$COUNT" -eq 0 ]]; then
    continue
  fi

  W_OUT="$WORK_ROOT/worker_outputs/w$i"
  mkdir -p "$W_OUT"

  uv run --no-project --python "$PYTHON_BIN" "$REPO/inpainting-workflow-master/privacy_blur_infer.py" \
    --input-dir "$IN_DIR" \
    --output-dir "$W_OUT" \
    --face-model "$FACE_MODEL" \
    --lp-model "$LP_MODEL" \
    --face-conf "$FACE_CONF" \
    --lp-conf "$LP_CONF" \
    --face-iou "$FACE_IOU" \
    --face-imgsz "$FACE_IMGSZ" \
    --det-face-w "$DET_FACE_W" \
    --p360-device "$P360_DEVICE" \
    --blur-scope "$BLUR_SCOPE" \
    --blur-backend "$BLUR_BACKEND" \
    --output-mode "$OUTPUT_MODE" \
    --jpg-quality "$JPG_QUALITY" \
    --overwrite > "$LOG_ROOT/worker${i}.log" 2>&1 &

  PIDS+=("$!")
  WORKER_IDS+=("$i")
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  wi="${WORKER_IDS[$idx]}"
  if ! wait "$pid"; then
    echo "worker_${wi}_failed=1"
    FAIL=1
  else
    echo "worker_${wi}_failed=0"
  fi
done

kill "$MON_PID" >/dev/null 2>&1 || true
wait "$MON_PID" 2>/dev/null || true

END_TS=$(date +%s)
WALL_ELAPSED=$((END_TS - START_TS))

if [[ "$FAIL" -ne 0 ]]; then
  echo "ERROR: one or more workers failed"
  exit 1
fi

python3 - <<'PY' "$WORK_ROOT/worker_outputs" "$OUT_ROOT" "$OUTPUT_MODE" "$STRICT_HARDLINK"
from pathlib import Path
import csv
import shutil
import sys

src_root = Path(sys.argv[1])
dst_root = Path(sys.argv[2])
output_mode = sys.argv[3]
strict_hardlink = sys.argv[4] == '1'
dst_root.mkdir(parents=True, exist_ok=True)

summary_rows = []
profile_rows = []

for wdir in sorted(src_root.glob('w*')):
    for p in sorted(wdir.glob('*_blur.jpg')):
        target = dst_root / p.name
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

    if output_mode == 'both':
        for p in sorted(wdir.glob('*_annot.jpg')):
            target = dst_root / p.name
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

    summ = wdir / 'summary.csv'
    if summ.exists():
        with summ.open('r', encoding='utf-8', newline='') as f:
            r = csv.DictReader(f)
            summary_rows.extend(list(r))

    prof = wdir / 'profile.csv'
    if prof.exists():
        with prof.open('r', encoding='utf-8', newline='') as f:
            r = csv.DictReader(f)
            profile_rows.extend(list(r))

summary_csv = dst_root / 'summary.csv'
with summary_csv.open('w', encoding='utf-8', newline='') as f:
    fields = ['image', 'face_boxes', 'lp_boxes', 'total_boxes', 'elapsed_sec', 'blur_output', 'annot_output']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in summary_rows:
        row = dict(row)
        row['blur_output'] = str(dst_root / Path(row['blur_output']).name)
        if row.get('annot_output'):
            row['annot_output'] = str(dst_root / Path(row['annot_output']).name)
        w.writerow(row)

profile_csv = dst_root / 'profile.csv'
with profile_csv.open('w', encoding='utf-8', newline='') as f:
    fields = [
        'image', 'read_sec', 'e2c_sec', 'detect_face_sec', 'detect_lp_sec',
        'project_sec', 'blur_sec', 'write_sec', 'total_sec'
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in profile_rows:
        w.writerow(row)

print(f'merged_images={len(summary_rows)}')
print(f'merged_summary={summary_csv}')
print(f'merged_profile={profile_csv}')
PY

python3 - <<'PY' "$OUT_ROOT/summary.csv" "$OUT_ROOT/profile.csv" "$GPU_LOG" "$WALL_ELAPSED" "$LOG_ROOT/run_summary.txt" "$RUN_NAME" "$OUT_ROOT"
from pathlib import Path
import csv
import sys
from datetime import datetime

summary_csv = Path(sys.argv[1])
profile_csv = Path(sys.argv[2])
gpu_log = Path(sys.argv[3])
wall_elapsed = int(sys.argv[4])
summary_path = Path(sys.argv[5])
run_name = sys.argv[6]
out_root = sys.argv[7]

images = 0
total_face = 0
total_lp = 0
sum_elapsed = 0.0
with summary_csv.open('r', encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        images += 1
        total_face += int(float(row.get('face_boxes', 0)))
        total_lp += int(float(row.get('lp_boxes', 0)))
        sum_elapsed += float(row.get('elapsed_sec', 0.0))

stage_sums = {
    'read_sec': 0.0,
    'e2c_sec': 0.0,
    'detect_face_sec': 0.0,
    'detect_lp_sec': 0.0,
    'project_sec': 0.0,
    'blur_sec': 0.0,
    'write_sec': 0.0,
    'total_sec': 0.0,
}
profile_rows = 0
with profile_csv.open('r', encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        profile_rows += 1
        for k in stage_sums:
            stage_sums[k] += float(row.get(k, 0.0))

gpu_vals = []
mem_vals = []
mem_used = []
times = []
with gpu_log.open('r', encoding='utf-8', newline='') as f:
    r = csv.reader(f)
    next(r, None)
    for row in r:
        if len(row) < 5:
            continue
        try:
            times.append(datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f'))
            gpu_vals.append(float(row[1].strip()))
            mem_vals.append(float(row[2].strip()))
            mem_used.append(float(row[3].strip()))
        except Exception:
            pass

sampled_elapsed = (times[-1] - times[0]).total_seconds() if len(times) >= 2 else 0.0

def avg(values):
    return sum(values) / len(values) if values else 0.0

metrics = {
    'run_name': run_name,
    'images': images,
    'total_face_boxes': total_face,
    'total_lp_boxes': total_lp,
    'total_boxes': total_face + total_lp,
    'wall_elapsed_sec': wall_elapsed,
    'sum_image_elapsed_sec': round(sum_elapsed, 4),
    'avg_image_elapsed_sec': round((sum_elapsed / images) if images else 0.0, 4),
    'avg_read_sec': round((stage_sums['read_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_e2c_sec': round((stage_sums['e2c_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_detect_face_sec': round((stage_sums['detect_face_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_detect_lp_sec': round((stage_sums['detect_lp_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_project_sec': round((stage_sums['project_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_blur_sec': round((stage_sums['blur_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_write_sec': round((stage_sums['write_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'avg_total_sec': round((stage_sums['total_sec'] / profile_rows) if profile_rows else 0.0, 4),
    'sampled_elapsed_sec': round(sampled_elapsed, 2),
    'avg_gpu_util_pct': round(avg(gpu_vals), 2),
    'max_gpu_util_pct': round(max(gpu_vals), 2) if gpu_vals else 0.0,
    'avg_mem_util_pct': round(avg(mem_vals), 2),
    'max_mem_util_pct': round(max(mem_vals), 2) if mem_vals else 0.0,
    'peak_mem_used_mb': round(max(mem_used), 2) if mem_used else 0.0,
    'output_dir': out_root,
}

with summary_path.open('w', encoding='utf-8') as f:
    for k, v in metrics.items():
        f.write(f'{k}={v}\n')

for k, v in metrics.items():
    print(f'{k}={v}')
PY
