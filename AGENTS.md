# AGENTS.md

## General Conventions
- Always use `tmux` for long-running tasks (like model downloads, Docker pulls, pipeline runs) to prevent blocking the terminal.

## Purpose
This repository runs a 5-stage 360 panorama pipeline:
1. SAM3 tiled mask generation
2. ComfyUI inpainting
3. Laplacian sky replacement (SAM3 mask + carremoved/newsky)
4. Panorama postprocess (top blend + seam fix)
5. Privacy Blur (faces/license plates)

## Workflow Source Used by Multi-GPU Entry Point
- Primary entry point is `run_multi_gpu_pipeline.sh`.
- `run_multi_gpu_pipeline.sh` shards `SRC` and launches one `run_full_pipeline.sh` job per GPU in tmux.
- Each per-GPU `run_full_pipeline.sh` job runs SAM3 masks via `inpainting-workflow-master/sam3_tiled_mask.py` using CLI defaults (transformers-based SAM3 via `Sam3Model`/`Sam3Processor`).
- Each per-GPU `run_full_pipeline.sh` job runs ComfyUI with `--workflow-json /workspace/workflow.json`.
- `docker-compose.yml` mounts `/workspace/workflow.json` from `workflow-updated.json`.
- `docker-compose.yml` still mounts `/workspace/workflow_SAM3_prompt.json` from `workflow_SAM3_prompt.json` for optional/manual workflows.
- Result: running `run_multi_gpu_pipeline.sh` uses `sam3_tiled_mask.py` CLI defaults and `workflow-updated.json` for ComfyUI inpainting on every shard.

## One-Command Behavior
`run_multi_gpu_pipeline.sh` is the user-facing one-command runner and, through per-GPU `run_full_pipeline.sh` jobs, bootstraps setup automatically:
- Ensures `$COMFYUI_DATA_DIR/input/perspective_mask.png` exists (copies from `inpainting-workflow-master/perspective_mask.png` if missing).
- Checks required models (including `models/comfyui/lama/big-lama.pt`) and runs `download-models.sh` if missing (`AUTO_DOWNLOAD_MODELS=1`).
- Starts the container via `docker compose -p "$CONTAINER_NAME" up -d` if not already up.
- When `DOWNSTREAM_MODE=isolated`, it stops ComfyUI containers before postprocess/privacy blur.
- Preserves existing batch outputs by default (`FORCE_REPROCESS=0`) so reruns can skip/resume; use `FORCE_REPROCESS=1` for a full rerun.

Useful env vars (with defaults):
- `SRC` (input dataset folder, no default - must be provided)
- `FINAL_OUTPUT_DIR` (where final privacy-blur outputs are copied, no default - optional)
- `RUN_NAME` (default: `multigpu-$(date +%Y%m%d_%H%M%S)`)
- `GPU_IDS` (default: `auto`; supports comma list like `0,1,3`)
- `MAX_GPUS` (default: `0`, meaning all detected GPUs)
- `BASE_COMFY_PORT` (default: `8188`)
- `CONTAINER_PREFIX` (default: `comfyui-g`)
- `COMFYUI_DATA_ROOT` (default: `./comfyui_data`)
- `TMUX_SESSION_PREFIX` (default: `mgpu`)
- `WORK_ROOT` (default: `./tmp/multigpu/$RUN_NAME`)
- `WAIT_POLL_SEC` (default: `10`)
- `SINGLE_GPU_CONFLICT_MODE` (default: `warn`; `off|warn|stop`)
- `POSTPROCESS_WORKERS` (default: `4`)
- `EGOBLUR_WORKERS` (legacy, ignored by privacy blur)
- `PRIVACY_WORKERS` (default: `2`)
- `SAM3_WORKERS` (default: `4`)
- `LAPLACIAN_DILATION` (default: `1`)
- `LAPLACIAN_BLUR` (default: `10`)
- `LAPLACIAN_LEVELS` (default: `7`)
- `MODELS_ROOT` (default: `./models`)
- `MODELS_COMFYUI_DIR` (default: `$MODELS_ROOT/comfyui`)
- `MODELS_EGOBLUR_DIR` (legacy)
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
- `PRIVACY_PYTHON_BIN` (default: `/data/.venv/bin/python`)
- `FORCE_REPROCESS` (default: `0`)
- `DOWNSTREAM_MODE` (default: `isolated`)
- `STOP_AFTER_STAGE` (default: `egoblur`; supports `inpainting` or `postprocess` for partial runs)
- `AUTO_INSTALL_NVIDIA_TOOLKIT` (default: `1`)
- `COMFY_READY_TIMEOUT` (default: `300`)
- `COMFY_READY_POLL` (default: `2`)
- `NVIDIA_CUDA_TEST_IMAGE` (default: `nvidia/cuda:12.6.0-base-ubuntu22.04`)
- `RESET_CONTAINER_BEFORE_RUN` (default: `1`)

## What the Pipeline Does Per GPU Shard
Each per-GPU `run_full_pipeline.sh` shard job performs the following stages:

1. **Stage input dataset**
   - Reads images from `SRC`.
   - Hardlinks them into `$COMFYUI_DATA_DIR/input/$BATCH_NAME`.

2. **SAM3 tiled mask generation**
   - Runs `inpainting-workflow-master/sam3_tiled_mask.py` inside `$CONTAINER_NAME`.
   - Reads from `/workspace/ComfyUI/input/$BATCH_NAME`.
   - Uses `sam3_tiled_mask.py` CLI defaults for resize/tile/prompt settings unless explicit flags are passed.
   - Writes masks to `/workspace/output-sam3-mask/$BATCH_NAME`.

3. **ComfyUI inpainting**
   - Runs `inpainting-workflow-master/comfyui_run.py` inside `$CONTAINER_NAME`.
   - Uses:
     - workflow: `/workspace/workflow.json` (mapped from `workflow-updated.json`)
     - images: `/workspace/ComfyUI/input/$BATCH_NAME`
     - mask: `/workspace/ComfyUI/input/perspective_mask.png`
   - Writes stage output to `/workspace/ComfyUI/output/$BATCH_NAME`.

4. **Postprocess**
   - Runs `inpainting-workflow-master/postprocess.py`.
   - Reads paired files from `/workspace/ComfyUI/output/$BATCH_NAME`:
     - `*_comfyui_carremoved.jpg` as base/source
     - `*_comfyui_newsky.jpg` as destination sky
   - Reads SAM3 masks from `/workspace/output-sam3-mask/$BATCH_NAME`.
   - Uses `LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt` for SimpleLama weights.
   - Runs Laplacian replacement with tunables: `--dilation`, `--blur`, `--levels`.
   - Uses top mask `/workspace/inpainting/sky_mask_updated.png`.
   - Writes to `/workspace/output-postprocessed/$BATCH_NAME`.

5. **Privacy Blur**
   - Runs `inpainting-workflow-master/privacy_blur_parallel.sh`.
   - Reads from `/workspace/output-postprocessed/$BATCH_NAME`.
   - Writes to `/workspace/output-egoblur/$BATCH_NAME`.

6. **Count outputs**
   - Prints image counts for input, SAM3 mask, inpainting, postprocess, and egoblur directories.

## How to Run

Use `run_multi_gpu_pipeline.sh` as the default entry point for end-to-end runs. If you want every relevant runtime setting to be explicit in the command, use:
Do not provide user-facing `run_full_pipeline.sh` commands unless explicitly requested for debugging a single shard.

```bash
SRC="" \
FINAL_OUTPUT_DIR="" \
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
GPU_IDS="auto" \
MAX_GPUS=0 \
BASE_COMFY_PORT=8188 \
CONTAINER_PREFIX="comfyui-g" \
COMFYUI_DATA_ROOT="./comfyui_data" \
TMUX_SESSION_PREFIX="mgpu" \
WAIT_POLL_SEC=10 \
SINGLE_GPU_CONFLICT_MODE="warn" \
POSTPROCESS_WORKERS=4 \
EGOBLUR_WORKERS=4 \
PRIVACY_WORKERS=2 \
SAM3_WORKERS=4 \
LAPLACIAN_DILATION=1 \
LAPLACIAN_BLUR=10 \
LAPLACIAN_LEVELS=7 \
MODELS_ROOT="./models" \
MODELS_COMFYUI_DIR="./models/comfyui" \
MODELS_EGOBLUR_DIR="./models/egoblur_gen2" \
MODELS_PRIVACY_DIR="./models/privacy_blur" \
AUTO_DOWNLOAD_MODELS=1 \
FORCE_REPROCESS=0 \
DOWNSTREAM_MODE="isolated" \
STOP_AFTER_STAGE="egoblur" \
AUTO_INSTALL_NVIDIA_TOOLKIT=1 \
COMFY_READY_TIMEOUT=300 \
COMFY_READY_POLL=2 \
NVIDIA_CUDA_TEST_IMAGE="nvidia/cuda:12.6.0-base-ubuntu22.04" \
RESET_CONTAINER_BEFORE_RUN=1 \
./run_multi_gpu_pipeline.sh
```

`run_multi_gpu_pipeline.sh` auto-detects GPUs, round-robin shards inputs, launches one shard run per GPU in tmux,
pins each run to its own container/GPU, and keeps downstream stop behavior container-local.

Set `SRC` and `FINAL_OUTPUT_DIR` to real paths when needed.

Single-command run (recommended entry point):

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
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
./run_multi_gpu_pipeline.sh
```

Pin to specific GPUs:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
GPU_IDS="0,1,3" \
./run_multi_gpu_pipeline.sh
```

Check outputs:

```bash
ls -lah comfyui_data/<container-name>/input/<batch-name>
ls -lah comfyui_data/<container-name>/output-sam3-mask/<batch-name>
ls -lah comfyui_data/<container-name>/output/<batch-name>
ls -lah comfyui_data/<container-name>/output-postprocessed/<batch-name>
ls -lah comfyui_data/<container-name>/output-egoblur/<batch-name>
```

Force full reprocess of an existing batch:

```bash
FORCE_REPROCESS=1 ./run_multi_gpu_pipeline.sh
```

Independent stage run (without ComfyUI API):

```bash
# Stop ComfyUI containers to free VRAM before downstream-only runs
# (single-container default)
docker stop comfyui-container

# Multi-GPU example
docker stop comfyui-g0 comfyui-g1

# Postprocess-only (recommended worker setting: -j 6)
docker run --rm --name postproc-g0 --gpus device=0 \
  -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt \
  -v /root/comfyui-workflow-docker/models/comfyui:/workspace/ComfyUI/models:ro \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/p2e-local:/workspace/ComfyUI/custom_nodes/p2e \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output:/workspace/ComfyUI/output \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-sam3-mask:/workspace/output-sam3-mask \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed:/workspace/output-postprocessed \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/postprocess.py \
    -i /workspace/ComfyUI/output/gpu0-batch \
    -o /workspace/output-postprocessed/gpu0-batch \
    --top-mask /workspace/inpainting/sky_mask_updated.png \
    --sam3-mask-dir /workspace/output-sam3-mask/gpu0-batch \
    --dilation 1 \
    --blur 10 \
    --levels 7 \
    --pattern "*.jpg" \
    -j 6

# Privacy blur-only (host-side parallel runner)
SRC="/root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed/gpu0-batch" \
OUT_ROOT="/root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-egoblur/gpu0-batch" \
FACE_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/face_yolov8n.pt" \
LP_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx" \
WORKERS=2 \
OUTPUT_MODE=blur_only \
PYTHON_BIN=/data/.venv/bin/python \
bash /root/comfyui-workflow-docker/inpainting-workflow-master/privacy_blur_parallel.sh
```

Count-check rule before/after privacy blur:
- Run privacy blur only after `output-postprocessed/<batch>` count matches the number of `*_comfyui_carremoved.jpg` files in `output/<batch>`.
- Privacy blur is complete for a batch when `output-egoblur/<batch>` count matches `output-postprocessed/<batch>` count.

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

## SAM3 Tiled Workflow TODOs
- [ ] Measure average per-image runtime for `workflow_SAM3_prompt.json` and record timings for at least one real batch.
- [ ] Compare resolution choices (`4000x2000` vs `2000x4000`) and decide the default target size based on quality + speed.
- [ ] Evaluate overlap and edge padding (start with 10 px border pad before SAM3, then unpad) to reduce boundary waviness; document best save settings.

## Debug: Running ComfyUI Container Only

To start just the ComfyUI container (without running the full pipeline) for debugging or API access:

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
| MODELS_EGOBLUR_DIR | ./models/egoblur_gen2 (legacy) |
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
