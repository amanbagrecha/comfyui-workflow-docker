# AGENTS.md

## General Conventions
- Always use `tmux` for long-running tasks (like model downloads, Docker pulls, pipeline runs) to prevent blocking the terminal.

## Purpose
This repository runs a 5-stage 360 panorama pipeline:
1. SAM3 tiled mask generation
2. ComfyUI inpainting
3. Laplacian sky replacement (SAM3 mask + carremoved/newsky)
4. Panorama postprocess (top blend + seam fix)
5. EgoBlur (faces/license plates)

## Workflow Source Used by Full Pipeline
- `run_full_pipeline.sh` runs SAM3 masks via `inpainting-workflow-master/sam3_tiled_mask.py` using the script's CLI defaults (transformers-based SAM3 via `Sam3Model`/`Sam3Processor`).
- `run_full_pipeline.sh` runs ComfyUI with `--workflow-json /workspace/workflow.json`.
- `docker-compose.yml` mounts `/workspace/workflow.json` from `workflow-updated.json`.
- `docker-compose.yml` still mounts `/workspace/workflow_SAM3_prompt.json` from `workflow_SAM3_prompt.json` for optional/manual workflows.
- Result: running the full pipeline script uses `sam3_tiled_mask.py` CLI defaults and `workflow-updated.json` for ComfyUI inpainting.

## One-Command Behavior
`run_full_pipeline.sh` bootstraps setup automatically:
- Ensures `$COMFYUI_DATA_DIR/input/perspective_mask.png` exists (copies from `inpainting-workflow-master/perspective_mask.png` if missing).
- Checks required models (including `models/comfyui/lama/big-lama.pt`) and runs `download-models.sh` if missing (`AUTO_DOWNLOAD_MODELS=1`).
- Starts the container via `docker compose -p "$CONTAINER_NAME" up -d` if not already up.
- When `DOWNSTREAM_MODE=isolated`, it stops ComfyUI containers before postprocess/egoblur and runs downstream stages in ephemeral `docker run --rm` containers.
- Preserves existing batch outputs by default (`FORCE_REPROCESS=0`) so reruns can skip/resume; use `FORCE_REPROCESS=1` for a full rerun.

Useful env vars (with defaults):
- `SRC` (input dataset folder, no default - must be provided)
- `FINAL_OUTPUT_DIR` (where final egoblur outputs are copied, no default - optional)
- `BATCH_NAME` (default: `batch-$(date +%Y%m%d_%H%M%S)`)
- `POSTPROCESS_WORKERS` (default: `4`)
- `EGOBLUR_WORKERS` (default: `4`)
- `SAM3_WORKERS` (default: `4`)
- `LAPLACIAN_DILATION` (default: `1`)
- `LAPLACIAN_BLUR` (default: `10`)
- `LAPLACIAN_LEVELS` (default: `7`)
- `CONTAINER_NAME` (default: `comfyui-container`)
- `COMFYUI_DATA_DIR` (default: `$REPO/comfyui_data/$CONTAINER_NAME`)
- `NVIDIA_VISIBLE_DEVICES` (default: `0`)
- `COMFY_PORT` (default: `8188`)
- `MODELS_ROOT` (default: `./models`)
- `MODELS_COMFYUI_DIR` (default: `$MODELS_ROOT/comfyui`)
- `MODELS_EGOBLUR_DIR` (default: `$MODELS_ROOT/egoblur_gen2`)
- `FORCE_REPROCESS` (default: `0`)
- `DOWNSTREAM_MODE` (default: `inline`; set `isolated` to stop ComfyUI before postprocess/egoblur)
- `COMFY_STOP_CONTAINERS` (default: `auto`; or comma list like `comfyui-g0,comfyui-g1`)
- `STOP_AFTER_STAGE` (default: `egoblur`; supports `inpainting` or `postprocess` for partial runs)

## What the Full Pipeline Does
`run_full_pipeline.sh` performs the following stages:

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

5. **EgoBlur**
   - Runs `inpainting-workflow-master/egoblur_infer.py`.
   - Reads from `/workspace/output-postprocessed/$BATCH_NAME`.
   - Writes to `/workspace/output-egoblur/$BATCH_NAME`.

6. **Count outputs**
   - Prints image counts for input, SAM3 mask, inpainting, postprocess, and egoblur directories.

## How to Run

Use `run_full_pipeline.sh` for normal end-to-end runs. If you want every runtime setting to be explicit in the command (with defaults where defaults exist), use:

```bash
SRC="" \
FINAL_OUTPUT_DIR="" \
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" \
POSTPROCESS_WORKERS=4 \
EGOBLUR_WORKERS=4 \
SAM3_WORKERS=4 \
LAPLACIAN_DILATION=1 \
LAPLACIAN_BLUR=10 \
LAPLACIAN_LEVELS=7 \
CONTAINER_NAME="comfyui-container" \
COMFYUI_DATA_DIR="./comfyui_data/comfyui-container" \
NVIDIA_VISIBLE_DEVICES=0 \
COMFY_PORT=8188 \
MODELS_ROOT="./models" \
MODELS_COMFYUI_DIR="./models/comfyui" \
MODELS_EGOBLUR_DIR="./models/egoblur_gen2" \
AUTO_DOWNLOAD_MODELS=1 \
FORCE_REPROCESS=0 \
DOWNSTREAM_MODE="inline" \
COMFY_STOP_CONTAINERS="auto" \
STOP_AFTER_STAGE="egoblur" \
AUTO_INSTALL_NVIDIA_TOOLKIT=1 \
COMFY_READY_TIMEOUT=300 \
COMFY_READY_POLL=2 \
NVIDIA_CUDA_TEST_IMAGE="nvidia/cuda:12.6.0-base-ubuntu22.04" \
./run_full_pipeline.sh
```

For automatic multi-GPU sharding/launch from one large `SRC`, use `run_multi_gpu_pipeline.sh`.
It auto-detects GPUs, round-robin shards inputs, launches one `run_full_pipeline.sh` per GPU in tmux,
pins each run to its own container/GPU, and keeps downstream stop behavior container-local.

Set `SRC` and `FINAL_OUTPUT_DIR` to real paths when needed.

Single-command run:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" \
./run_full_pipeline.sh
```

Shared model cache run:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" \
./run_full_pipeline.sh
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
FORCE_REPROCESS=1 ./run_full_pipeline.sh
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

# Egoblur-only (requested worker setting: --workers 12)
docker run --rm --name egoblur-g0 --gpus device=0 \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/models/egoblur_gen2:/workspace/inpainting/models/egoblur_gen2:ro \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed:/workspace/output-postprocessed \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-egoblur:/workspace/output-egoblur \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/egoblur_infer.py \
    --input-dir /workspace/output-postprocessed/gpu0-batch \
    --output-dir /workspace/output-egoblur/gpu0-batch \
    --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
    --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit \
    --device cuda \
    --workers 12
```

Count-check rule before/after egoblur:
- Run egoblur only after `output-postprocessed/<batch>` count matches the number of `*_comfyui_carremoved.jpg` files in `output/<batch>`.
- Egoblur is complete for a batch when `output-egoblur/<batch>` count matches `output-postprocessed/<batch>` count.

## Image Output Quality
- ComfyUI inpainting save node (`workflow-updated.json`) writes JPG at quality `80`.
- Postprocess stage writes JPG at quality `90`.

## Host-Side Input and Output Paths
- Source dataset: `/data/comfyui-workflow-docker/pano_data/<dataset-or-smoke-dir>`
- Staged input: `comfyui_data/<container-name>/input/<batch-name>`
- SAM3 mask output: `comfyui_data/<container-name>/output-sam3-mask/<batch-name>`
- Inpainting output: `comfyui_data/<container-name>/output/<batch-name>`
- Postprocess output: `comfyui_data/<container-name>/output-postprocessed/<batch-name>`
- Final egoblur output: `comfyui_data/<container-name>/output-egoblur/<batch-name>`
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
| MODELS_EGOBLUR_DIR | ./models/egoblur_gen2 |
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
