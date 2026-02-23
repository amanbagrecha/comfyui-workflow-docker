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

## What the Full Pipeline Does
`run_full_pipeline.sh` performs the following stages:

1. **Stage input dataset**
   - Reads images from `SRC`.
   - Hardlinks them into `input/$BATCH_NAME`.

2. **ComfyUI inpainting**
   - Runs `inpainting-workflow-master/comfyui_run.py` inside `comfyui-container`.
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

## How to Run a One-Image Smoke Test

1. Create a one-image sample source directory:

```bash
SMOKE_SRC="/data/comfyui-workflow-docker/pano_data/smoke-one-workflow-updated-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE_SRC"
cp "/data/comfyui-workflow-docker/pano_data/12096793288_407096364923_ladybug_Panoramic_000006.jpg" "$SMOKE_SRC/"
```

2. Run the full pipeline on that sample:

```bash
REPO="/data/comfyui-workflow-docker/repo" \
SRC="$SMOKE_SRC" \
BATCH_NAME="batch-smoke-workflow-updated-$(date +%Y%m%d_%H%M%S)" \
./run_full_pipeline.sh
```

3. Check outputs:

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
