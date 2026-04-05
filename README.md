# ComfyUI Inpainting and Privacy Blur Pipeline

Host-native multi-GPU pipeline for 360 panorama processing.

This repository no longer uses Docker. The full workflow runs directly on the host with:

1. SAM3 tiled mask generation
2. ComfyUI inpainting through the local ComfyUI HTTP API
3. Laplacian sky replacement and panorama postprocess
4. Privacy blur for faces and license plates
5. Per-stage counts, JSONL event logs, and merged final outputs

## Overview

`run_multi_gpu_pipeline.sh` is the main command.
It bootstraps a `uv`-managed Python `3.10` environment, downloads missing models, runs inpainting, stops ComfyUI to free VRAM, and finishes the remaining stages.

## Requirements

- Ubuntu or Debian-like host recommended
- NVIDIA driver installed and working via `nvidia-smi`
- `curl` or `wget`
- tmux
- Enough local disk for models and intermediate outputs

The host runtime is auto-installed by `run_multi_gpu_pipeline.sh` on first use.

## Quick Start

### Run The Full Pipeline

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
./run_multi_gpu_pipeline.sh
```

That is the only command you need for normal runs.

Examples:

```bash
# Explicit run name
SRC="/data/panos" \
FINAL_OUTPUT_DIR="/data/final" \
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
./run_multi_gpu_pipeline.sh

# Shared model cache
SRC="/data/panos" \
FINAL_OUTPUT_DIR="/data/final" \
MODELS_ROOT="/data/shared-models" \
./run_multi_gpu_pipeline.sh

# Restrict to selected GPUs
SRC="/data/panos" \
FINAL_OUTPUT_DIR="/data/final" \
GPU_IDS="0,1,3" \
./run_multi_gpu_pipeline.sh

# Partial runs
STOP_AFTER_STAGE=sam3 SRC="/data/panos" ./run_multi_gpu_pipeline.sh
STOP_AFTER_STAGE=postprocess SRC="/data/panos" ./run_multi_gpu_pipeline.sh
```

## Runtime Layout

Default host layout:

```text
.
├── .venv/
├── ComfyUI/
├── p2e-lib/
├── models/
│   ├── comfyui/
│   └── privacy_blur/
├── native_data/
│   ├── gpu0/
│   │   ├── input/
│   │   ├── output/
│   │   ├── output-sam3-mask/
│   │   ├── output-postprocessed/
│   │   └── output-egoblur/
│   └── gpu1/
├── logs/
└── tmp/
```

Per-GPU worker roots use the physical GPU id, not the shard index.

## Main Scripts

- `run_multi_gpu_pipeline.sh`: main entrypoint
- `run_full_pipeline.sh`: per-GPU shard runner
- `setup_host_environment.sh`: internal bootstrap helper
- `run_comfyui_service.sh`: internal ComfyUI launcher

## Important Environment Variables

### Placement

- `GPU_IDS`
- `MAX_GPUS`
- `BASE_COMFY_PORT`
- `CUDA_VISIBLE_DEVICES`

### Paths

- `COMFYUI_HOME`
- `MODELS_ROOT`
- `MODELS_COMFYUI_DIR`
- `MODELS_PRIVACY_DIR`
- `NATIVE_DATA_ROOT`
- `FINAL_OUTPUT_DIR`

### Pipeline control

- `FORCE_REPROCESS`
- `STOP_AFTER_STAGE`
- `STRICT_HARDLINK`
- `SAM3_WORKERS`
- `POSTPROCESS_WORKERS`
- `PRIVACY_WORKERS`

## Logs and Outputs

Top-level orchestrator logs:

- `logs/multigpu_<RUN_NAME>.log`
- `logs/multigpu_<RUN_NAME>.events.jsonl`

Per-shard logs:

- `logs/fullrun_<RUN_NAME>_g<gpu_id>.log`
- `logs/fullrun_<RUN_NAME>_g<gpu_id>.events.jsonl`

Per-GPU ComfyUI service logs:

- `logs/comfyui_g<gpu_id>.log`

Stage outputs:

- `native_data/gpu<id>/input/<batch>`
- `native_data/gpu<id>/output-sam3-mask/<batch>`
- `native_data/gpu<id>/output/<batch>`
- `native_data/gpu<id>/output-postprocessed/<batch>`
- `native_data/gpu<id>/output-egoblur/<batch>`

Merged outputs:

- `FINAL_OUTPUT_DIR/<run-name>/gpu<id>/`

## Production Notes

- Start with one worker per heavy stage on each GPU and scale only after measuring VRAM usage.
- ComfyUI is always stopped after inpainting to release VRAM before downstream stages.
- Keep the model cache on shared local storage if multiple repo copies use the same machine.

## Migration Guide

Detailed migration notes and the end-to-end host-native rollout plan live in:

`docs/native-no-docker-migration.md`
