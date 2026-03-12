# AGENTS.md

## General Conventions
- Always use `tmux` for long-running tasks (like model downloads, Docker pulls, pipeline runs) to prevent blocking the terminal.

## Purpose
This repository runs a 5-stage 360 panorama pipeline:
1. SAM3 tiled mask generation
2. ComfyUI inpainting
3. Laplacian sky replacement (SAM3 mask + carremoved/newsky)
4. Panorama postprocess (top blend + seam fix)
5. Privacy Blur (faces/license plates) — runs inside the Docker container via `docker run`

## Docker Image
- Image: `amanbagrecha/container-comfyui:latest`
- Built from a multi-stage Dockerfile (builder + runtime stages).
- Builder stage: installs uv, creates venv, installs comfy-cli, ComfyUI (pinned commit), restores Comfy-Lock.yaml snapshot, installs pipeline deps via `pipeline-requirements.txt`.
- Runtime stage: copies venv, ComfyUI, p2e-lib, inpainting-workflow-master, and comfy-cli config (`/root/.config`) from builder. Includes system packages: `git`, `wget`, `curl`, `python3`, `python3-pip`, `python3-dev`, `build-essential`, `libgl1-mesa-glx`, `libglib2.0-0`.
- All pipeline Python deps (numba, opencv, ultralytics, open-image-models, pytorch360convert, etc.) are baked into the image via `pipeline-requirements.txt`.

## Workflow Source Used by Multi-GPU Entry Point
- Primary entry point is `run_multi_gpu_pipeline.sh`.
- `run_multi_gpu_pipeline.sh` shards `SRC` and launches one `run_full_pipeline.sh` job per GPU in tmux.
- Each per-GPU `run_full_pipeline.sh` job runs SAM3 masks via `inpainting-workflow-master/$SAM3_SCRIPT` (`sam3_tiled_mask.py` default; supports `archive_sam3_tiled_mask.py`) using transformers-based SAM3 (`Sam3Model`/`Sam3Processor`).
- Each per-GPU `run_full_pipeline.sh` job runs ComfyUI with `--workflow-json /workspace/workflow.json`.
- `docker-compose.yml` mounts `/workspace/workflow.json` from `workflow-updated.json` (current content comes from `carremoval-input-to-sky.json`).
- Result: running `run_multi_gpu_pipeline.sh` uses `sam3_tiled_mask.py` CLI defaults and `workflow-updated.json` for ComfyUI inpainting on every shard.

## One-Command Behavior
`run_multi_gpu_pipeline.sh` is the user-facing one-command runner. It handles all bootstrapping **once** before launching per-GPU shards:
- Ensures `uv` is installed (installs from astral.sh if missing).
- Installs NVIDIA Container Toolkit if not present.
- Checks required models and runs `download-models.sh` if missing (`AUTO_DOWNLOAD_MODELS=1`).
- Pulls the Docker image (`COMFY_IMAGE`).
- Exports `SKIP_PREFLIGHT=1` to all shard jobs so they skip the above checks.

Per-GPU `run_full_pipeline.sh` shard jobs (when `SKIP_PREFLIGHT=1`):
- Ensures `$COMFYUI_DATA_DIR/input/perspective_mask.png` exists (copies from `inpainting-workflow-master/perspective_mask.png` if missing).
- Stages sky reference image as `$COMFYUI_DATA_DIR/input/chrome_xWUjmfs7m4.png`.
- Starts the container via `docker compose -p "$CONTAINER_NAME" up -d` if not already up.
- When `DOWNSTREAM_MODE=isolated`, stops ComfyUI containers before postprocess/privacy blur.
- Preserves existing batch outputs by default (`FORCE_REPROCESS=0`); use `FORCE_REPROCESS=1` for a full rerun.

Useful env vars (with defaults):
- `SRC` (input dataset folder, no default — must be provided)
- `FINAL_OUTPUT_DIR` (where final privacy-blur outputs are copied, no default — optional)
- `RUN_NAME` (default: `multigpu-$(date +%Y%m%d_%H%M%S)`)
- `GPU_IDS` (default: `auto`; supports comma list like `0,1,3`)
- `MAX_GPUS` (default: `0`, meaning all detected GPUs)
- `BASE_COMFY_PORT` (default: `8180`; GPU i uses port BASE_COMFY_PORT+i)
- `CONTAINER_PREFIX` (default: `comfyui-g`)
- `COMFYUI_DATA_ROOT` (default: `./comfyui_data`)
- `TMUX_SESSION_PREFIX` (default: `mgpu`)
- `WORK_ROOT` (default: `./tmp/multigpu/$RUN_NAME`)
- `WAIT_POLL_SEC` (default: `10`)
- `SINGLE_GPU_CONFLICT_MODE` (default: `warn`; `off|warn|stop`)
- `POSTPROCESS_WORKERS` (default: `3`)
- `PRIVACY_WORKERS` (default: `4`)
- `SAM3_WORKERS` (default: `4`)
- `SAM3_RESIZE_WIDTH` (default: `4000`)
- `SAM3_RESIZE_HEIGHT` (default: `2000`)
- `SAM3_GLARE_THRESHOLD` (default: `0.4`)
- `SAM3_TILE_ROWS` (default: `2`)
- `SAM3_TILE_COLS` (default: `1`)
- `SAM3_SCRIPT` (default: `sam3_tiled_mask.py`; set `archive_sam3_tiled_mask.py` to use archive tiled predictor)
- `LAPLACIAN_DILATION` (default: `1`)
- `LAPLACIAN_BLUR` (default: `10`)
- `LAPLACIAN_LEVELS` (default: `7`)
- `MODELS_ROOT` (default: `./models`)
- `MODELS_COMFYUI_DIR` (default: `$MODELS_ROOT/comfyui`)
- `MODELS_PRIVACY_DIR` (default: `$MODELS_ROOT/privacy_blur`)
- `PRIVACY_FACE_MODEL` (default: `$MODELS_PRIVACY_DIR/face_yolov8n.pt`)
- `PRIVACY_LP_MODEL` (default: `$MODELS_PRIVACY_DIR/yolo-v9-s-608-license-plates-end2end.onnx`)
- `PRIVACY_FACE_CONF` (default: `0.4`)
- `PRIVACY_LP_CONF` (default: `0.4`)
- `PRIVACY_FACE_IOU` (default: `0.5`)
- `PRIVACY_FACE_IMGSZ` (default: `1024`)
- `PRIVACY_DET_FACE_W` (default: `1024`)
- `PRIVACY_P360_DEVICE` (default: `auto`)
- `PRIVACY_BLUR_SCOPE` (default: `roi`)
- `PRIVACY_BLUR_BACKEND` (default: `gpu`)
- `PRIVACY_OUTPUT_MODE` (default: `blur_only`; set `both` for debug)
- `FORCE_REPROCESS` (default: `0`)
- `DOWNSTREAM_MODE` (default: `isolated`)
- `STOP_AFTER_STAGE` (default: `egoblur`; supports `sam3`, `inpainting`, or `postprocess` for partial runs)
- `COMFY_READY_TIMEOUT` (default: `300`)
- `COMFY_READY_POLL` (default: `2`)
- `NVIDIA_CUDA_TEST_IMAGE` (default: `nvidia/cuda:12.6.0-base-ubuntu22.04`)
- `RESET_CONTAINER_BEFORE_RUN` (default: `1`)
- `COMFY_IMAGE_NODE_ID` (default: `91`)
- `COMFY_MASK_NODE_ID` (default: `34`)
- `COMFY_SAM3_MASK_NODE_ID` (default: `60`)
- `SKY_REFERENCE_SOURCE` (default: `./inpainting-workflow-master/reference_sky.png`)
- `SKY_REFERENCE_FILENAME` (default: `chrome_xWUjmfs7m4.png`)
- `COMFY_IMAGE` (default: `amanbagrecha/container-comfyui:latest`)
- `SKIP_PREFLIGHT` (default: `0`; set to `1` in shard jobs by orchestrator to skip toolkit/model/pull checks)

## What the Pipeline Does Per GPU Shard
Each per-GPU `run_full_pipeline.sh` shard job performs the following stages:

1. **Stage input dataset**
   - Reads images from `SRC`.
   - Hardlinks them into `$COMFYUI_DATA_DIR/input/$BATCH_NAME`.

2. **SAM3 tiled mask generation**
   - Runs `inpainting-workflow-master/$SAM3_SCRIPT` inside `$CONTAINER_NAME`.
   - Reads from `/workspace/ComfyUI/input/$BATCH_NAME`.
   - Uses env-mapped settings: `--glare-threshold $SAM3_GLARE_THRESHOLD`, `--tile-rows $SAM3_TILE_ROWS`, `--tile-cols $SAM3_TILE_COLS`, `--resize-width $SAM3_RESIZE_WIDTH`, `--resize-height $SAM3_RESIZE_HEIGHT`, `--workers $SAM3_WORKERS`.
   - Writes masks to `/workspace/output-sam3-mask/$BATCH_NAME`.

3. **ComfyUI inpainting**
   - Runs `inpainting-workflow-master/comfyui_run.py` inside `$CONTAINER_NAME`.
   - Uses:
     - workflow: `/workspace/workflow.json` (mapped from `workflow-updated.json`)
     - images: `/workspace/ComfyUI/input/$BATCH_NAME`
     - mask: `/workspace/ComfyUI/input/perspective_mask.png`
     - reference sky image: `/workspace/ComfyUI/input/chrome_xWUjmfs7m4.png`
     - node IDs: `--image-node-id 91 --mask-node-id 34 --sam3-mask-node-id 60`
   - Writes stage output to `/workspace/ComfyUI/output/$BATCH_NAME`.

4. **Postprocess**
   - Runs `inpainting-workflow-master/postprocess.py` via `docker run` using `$COMFY_IMAGE`.
   - Reads paired files from `/workspace/ComfyUI/output/$BATCH_NAME`:
     - `*_comfyui_carremoved.jpg` as base/source
     - `*_comfyui_newsky.jpg` as destination sky
   - Reads SAM3 masks from `/workspace/output-sam3-mask/$BATCH_NAME`.
   - Uses `LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt` for SimpleLama weights.
   - Runs Laplacian replacement with tunables: `--dilation`, `--blur`, `--levels`.
   - Uses top mask `/workspace/inpainting/sky_mask_updated.png`.
   - Writes to `/workspace/output-postprocessed/$BATCH_NAME`.

5. **Privacy Blur**
   - Runs `inpainting-workflow-master/privacy_blur_infer.py` via `docker run` using `$COMFY_IMAGE`.
   - Reads from `/workspace/output-postprocessed/$BATCH_NAME`.
   - Writes to `/workspace/output-egoblur/$BATCH_NAME`.
   - Also writes `profile.csv` and `summary.csv` alongside blurred images (counted in egoblur dir).

6. **Count outputs**
   - Prints image counts for input, SAM3 mask, inpainting, postprocess, and egoblur directories.

## How to Run

Use `run_multi_gpu_pipeline.sh` as the default entry point for end-to-end runs.
Do not provide user-facing `run_full_pipeline.sh` commands unless explicitly requested for debugging a single shard.

Single-command run (recommended):

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
./run_multi_gpu_pipeline.sh
```

With explicit run name:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
./run_multi_gpu_pipeline.sh
```

Shared model cache run:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
./run_multi_gpu_pipeline.sh
```

Pin to specific GPUs:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
GPU_IDS="0,1,3" \
./run_multi_gpu_pipeline.sh
```

Force full reprocess of an existing batch:

```bash
FORCE_REPROCESS=1 SRC="..." ./run_multi_gpu_pipeline.sh
```

SAM3-only run (skip downstream stages):

```bash
STOP_AFTER_STAGE=sam3 SRC="..." ./run_multi_gpu_pipeline.sh
```

Check outputs:

```bash
ls -lah comfyui_data/<container-name>/input/<batch-name>
ls -lah comfyui_data/<container-name>/output-sam3-mask/<batch-name>
ls -lah comfyui_data/<container-name>/output/<batch-name>
ls -lah comfyui_data/<container-name>/output-postprocessed/<batch-name>
ls -lah comfyui_data/<container-name>/output-egoblur/<batch-name>
```

## Independent Stage Runs (Debug)

Stop ComfyUI containers to free VRAM before running downstream stages standalone:

```bash
# Multi-GPU
docker stop comfyui-g0 comfyui-g1
```

Postprocess-only:

```bash
docker run --rm --name postproc-g0 --gpus device=0 \
  -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt \
  -v /root/comfyui-workflow-docker/models/comfyui:/workspace/ComfyUI/models:ro \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output:/workspace/ComfyUI/output \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-sam3-mask:/workspace/output-sam3-mask \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed:/workspace/output-postprocessed \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/postprocess.py \
    -i /workspace/ComfyUI/output/gpu0-batch \
    -o /workspace/output-postprocessed/gpu0-batch \
    --top-mask /workspace/inpainting/sky_mask_updated.png \
    --sam3-mask-dir /workspace/output-sam3-mask/gpu0-batch \
    --dilation 1 --blur 10 --levels 7 \
    --pattern "*.jpg" -j 6
```

Privacy blur-only (runs inside container):

```bash
docker run --rm --name egoblur-g0 --gpus device=0 \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/models/privacy_blur:/workspace/models/privacy_blur:ro \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed:/workspace/output-postprocessed \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-egoblur:/workspace/output-egoblur \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/privacy_blur_infer.py \
    --input-dir  /workspace/output-postprocessed/gpu0-batch \
    --output-dir /workspace/output-egoblur/gpu0-batch \
    --face-model /workspace/models/privacy_blur/face_yolov8n.pt \
    --lp-model   /workspace/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx
```

Count-check rule before/after privacy blur:
- Run privacy blur only after `output-postprocessed/<batch>` count matches the number of `*_comfyui_carremoved.jpg` files in `output/<batch>`.
- Privacy blur is complete for a batch when image count in `output-egoblur/<batch>` matches `output-postprocessed/<batch>` (the egoblur dir also contains `profile.csv` and `summary.csv` — subtract 2 from total file count to get image count).

## Image Output Quality
- ComfyUI inpainting save node (`workflow-updated.json`) writes JPG at quality `80`.
- Postprocess stage writes JPG at quality `85`.
- Privacy blur stage writes JPG at quality `85`.

## Host-Side Input and Output Paths
- Source dataset: `/data/comfyui-workflow-docker/pano_data/<dataset-or-smoke-dir>`
- Staged input: `comfyui_data/<container-name>/input/<batch-name>`
- SAM3 mask output: `comfyui_data/<container-name>/output-sam3-mask/<batch-name>`
- Inpainting output: `comfyui_data/<container-name>/output/<batch-name>`
- Postprocess output: `comfyui_data/<container-name>/output-postprocessed/<batch-name>`
- Final privacy-blur output: `comfyui_data/<container-name>/output-egoblur/<batch-name>`
- Optional copied final output: `FINAL_OUTPUT_DIR/<batch-name>`

## Logging Artifacts
- Multi-GPU runs write `logs/multigpu_<RUN_NAME>.log` and `logs/multigpu_<RUN_NAME>.events.jsonl`.
- Per-shard/full runs write `logs/fullrun_<RUN_ID>.log` and `logs/fullrun_<RUN_ID>.events.jsonl`.
- In multi-GPU runs, shard `RUN_ID` is deterministic: `<RUN_NAME>_g<gpu_id>`.
- `multigpu_*.events.jsonl` contains orchestrator events only; `fullrun_*.events.jsonl` contains shard stage events only.
- There is no `summary.txt` run artifact anymore.
- Consume structured logs as JSONL: one JSON object per line via Python `json.loads`, `jq`, W&B, or any JSONL-compatible collector.

## SAM3 Tiled Workflow TODOs
- [ ] Measure average per-image runtime for `sam3_tiled_mask.py` and record timings for at least one real batch.
- [ ] Compare resolution choices (`4000x2000` vs `2000x4000`) and decide the default target size based on quality + speed.
- [ ] Evaluate overlap and edge padding (start with 10 px border pad before SAM3, then unpad) to reduce boundary waviness; document best save settings.

## Debug: Running ComfyUI Container Only

To start just the ComfyUI container (without running the full pipeline):

```bash
# Using all defaults
docker compose -p comfyui-container up -d

# With custom settings
MODELS_ROOT="/path/to/shared-model-cache" \
NVIDIA_VISIBLE_DEVICES=0 \
COMFY_PORT=8188 \
docker compose -p comfyui-container up -d
```

Default values:
| Variable | Default |
|----------|---------|
| MODELS_COMFYUI_DIR | ./models/comfyui |
| MODELS_PRIVACY_DIR | ./models/privacy_blur |
| COMFYUI_DATA_DIR | ./comfyui_data/comfyui-container |
| COMFY_PORT | 8188 |
| CONTAINER_NAME | comfyui-container |
| NVIDIA_VISIBLE_DEVICES | 0 |

Access ComfyUI at: http://localhost:8188

To view logs:
```bash
docker logs -f comfyui-container
```

To stop:
```bash
docker compose -p comfyui-container down
```
