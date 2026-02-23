# AGENTS.md

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
- Creates required local directories.
- Ensures `input/perspective_mask.png` exists (copies from `inpainting-workflow-master/perspective_mask.png` if missing).
- Checks required models and runs `download-models.sh` if missing (`AUTO_DOWNLOAD_MODELS=1`).
- Starts the container via `docker compose up -d` if not already up.

Useful env vars:
- `SRC` (input dataset folder)
- `FINAL_OUTPUT_DIR` (where final egoblur outputs are copied)
- `BATCH_NAME`
- `POSTPROCESS_WORKERS` (default `1`)
- `EGOBLUR_WORKERS` (default `3`)
- `CONTAINER_NAME` (default `comfyui-container`)
- `MODELS_ROOT` (shared model cache root)
- `MODELS_COMFYUI_DIR` / `MODELS_EGOBLUR_DIR` (explicit model path overrides)

## What the Full Pipeline Does
`run_full_pipeline.sh` performs the following stages:

1. **Stage input dataset**
   - Reads images from `SRC`.
   - Hardlinks them into `input/$BATCH_NAME`.

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
ls -lah input/<batch-name>
ls -lah output/<batch-name>
ls -lah output-postprocessed/<batch-name>
ls -lah output-egoblur/<batch-name>
```

## Host-Side Input and Output Paths
- Source dataset: `/data/comfyui-workflow-docker/pano_data/<dataset-or-smoke-dir>`
- Staged input: `input/<batch-name>`
- Inpainting output: `output/<batch-name>`
- Postprocess output: `output-postprocessed/<batch-name>`
- Final egoblur output: `output-egoblur/<batch-name>`
- Optional copied final output: `FINAL_OUTPUT_DIR/<batch-name>`
