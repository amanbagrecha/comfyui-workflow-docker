# AGENTS.md

## General Conventions
- Always use `tmux` for long-running tasks (like model downloads, Docker pulls, pipeline runs) to prevent blocking the terminal.

## Purpose
This repository runs a 3-stage 360 panorama pipeline:
1. ComfyUI inpainting
2. Panorama postprocess (top blend + seam fix)
3. EgoBlur (faces/license plates)

## Workflow Source Used by Full Pipeline
- `run_full_pipeline.sh` runs ComfyUI with `--workflow-json /workspace/workflow.json`.
- `docker-compose.yml` mounts `/workspace/workflow.json` from `workflow-updated.json`.
- Result: running the full pipeline script uses `workflow-updated.json`.

## One-Command Behavior
`run_full_pipeline.sh` bootstraps setup automatically:
- Ensures `$COMFYUI_DATA_DIR/input/perspective_mask.png` exists (copies from `inpainting-workflow-master/perspective_mask.png` if missing).
- Checks required models and runs `download-models.sh` if missing (`AUTO_DOWNLOAD_MODELS=1`).
- Starts the container via `docker compose -p "$CONTAINER_NAME" up -d` if not already up.
- Preserves existing batch outputs by default (`FORCE_REPROCESS=0`) so reruns can skip/resume; use `FORCE_REPROCESS=1` for a full rerun.

Useful env vars (with defaults):
- `SRC` (input dataset folder, no default - must be provided)
- `FINAL_OUTPUT_DIR` (where final egoblur outputs are copied, no default - optional)
- `BATCH_NAME` (default: `batch-$(date +%Y%m%d_%H%M%S)`)
- `POSTPROCESS_WORKERS` (default: `1`)
- `EGOBLUR_WORKERS` (default: `3`)
- `CONTAINER_NAME` (default: `comfyui-container`)
- `COMFYUI_DATA_DIR` (default: `$REPO/comfyui_data/$CONTAINER_NAME`)
- `NVIDIA_VISIBLE_DEVICES` (default: `0`)
- `COMFY_PORT` (default: `8188`)
- `MODELS_ROOT` (default: `./models`)
- `MODELS_COMFYUI_DIR` (default: `$MODELS_ROOT/comfyui`)
- `MODELS_EGOBLUR_DIR` (default: `$MODELS_ROOT/egoblur_gen2`)
- `FORCE_REPROCESS` (default: `0`)

## What the Full Pipeline Does
`run_full_pipeline.sh` performs the following stages:

1. **Stage input dataset**
   - Reads images from `SRC`.
   - Hardlinks them into `$COMFYUI_DATA_DIR/input/$BATCH_NAME`.

2. **ComfyUI inpainting**
   - Runs `inpainting-workflow-master/comfyui_run.py` inside `$CONTAINER_NAME`.
   - Uses:
     - workflow: `/workspace/workflow.json` (mapped from `workflow-updated.json`)
     - images: `/workspace/ComfyUI/input/$BATCH_NAME`
     - mask: `/workspace/ComfyUI/input/perspective_mask.png`
   - Writes stage output to `/workspace/ComfyUI/output/$BATCH_NAME`.

3. **Postprocess**
   - Runs `inpainting-workflow-master/postprocess.py`.
   - Reads from `/workspace/ComfyUI/output/$BATCH_NAME`.
   - Uses top mask `/workspace/inpainting/sky_mask_updated.png`.
   - Writes to `/workspace/output-postprocessed/$BATCH_NAME`.

4. **EgoBlur**
   - Runs `inpainting-workflow-master/egoblur_infer.py`.
   - Reads from `/workspace/output-postprocessed/$BATCH_NAME`.
   - Writes to `/workspace/output-egoblur/$BATCH_NAME`.

5. **Count outputs**
   - Prints image counts for input, inpainting, postprocess, and egoblur directories.

## How to Run

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
docker stop comfyui-g0 comfyui-g1

# Postprocess-only (recommended worker setting: -j 6)
docker run --rm --name postproc-g0 --gpus device=0 \
  -v /root/.cache/torch:/root/.cache/torch \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/p2e-local:/workspace/ComfyUI/custom_nodes/p2e \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output:/workspace/ComfyUI/output \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed:/workspace/output-postprocessed \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/postprocess.py \
    -i /workspace/ComfyUI/output/gpu0-batch \
    -o /workspace/output-postprocessed/gpu0-batch \
    --top-mask /workspace/inpainting/sky_mask_updated.png \
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
- Run egoblur only after `output-postprocessed/<batch>` count matches `output/<batch>` count.
- Egoblur is complete for a batch when `output-egoblur/<batch>` count matches `output-postprocessed/<batch>` count.

## Image Output Quality
- ComfyUI inpainting save node (`workflow-updated.json`) writes JPG at quality `80`.
- Postprocess stage writes JPG at quality `90`.

## Host-Side Input and Output Paths
- Source dataset: `/data/comfyui-workflow-docker/pano_data/<dataset-or-smoke-dir>`
- Staged input: `comfyui_data/<container-name>/input/<batch-name>`
- Inpainting output: `comfyui_data/<container-name>/output/<batch-name>`
- Postprocess output: `comfyui_data/<container-name>/output-postprocessed/<batch-name>`
- Final egoblur output: `comfyui_data/<container-name>/output-egoblur/<batch-name>`
- Optional copied final output: `FINAL_OUTPUT_DIR/<batch-name>`

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
