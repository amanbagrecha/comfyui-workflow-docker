# ComfyUI Inpainting and Privacy Blur Pipeline

Host-native multi-GPU pipeline for 360 panorama processing.

This repository no longer uses Docker. The full workflow runs directly on the host with:

1. SAM3 tiled mask generation
2. ComfyUI inpainting through the local ComfyUI HTTP API
3. Laplacian sky replacement and panorama postprocess
4. Privacy blur for faces and license plates
5. Per-stage counts, JSONL event logs, and merged final outputs

## Overview

The main runtime flow is:

1. `setup_host_environment.sh`
2. `download-models.sh` if you skipped model download during setup
3. `run_comfyui_cluster.sh`
4. `run_multi_gpu_pipeline.sh`

The orchestrator shards one `SRC` directory across available GPUs, launches one worker per GPU in tmux, and merges the final outputs under `FINAL_OUTPUT_DIR/<run-name>/gpu<id>/`.

## Requirements

- Ubuntu or Debian-like host recommended
- NVIDIA driver installed and working via `nvidia-smi`
- Python 3
- tmux
- Enough local disk for models and intermediate outputs

The Python environment, pinned ComfyUI checkout, custom nodes, and pipeline dependencies are installed by `setup_host_environment.sh`.

## Quick Start

### 1. Install the host environment

```bash
./setup_host_environment.sh
```

Defaults:

- Creates `.venv/`
- Clones `ComfyUI/` at the pinned commit used by this repo
- Restores `Comfy-Lock.yaml`
- Clones `p2e-lib/`
- Symlinks `ComfyUI/models` to `models/comfyui`
- Symlinks `ComfyUI/custom_nodes/p2e` to `p2e-local`
- Downloads models unless `DOWNLOAD_MODELS=0`

Useful overrides:

```bash
MODELS_ROOT="/data/shared-models" \
INSTALL_SYSTEM_PACKAGES=0 \
DOWNLOAD_MODELS=1 \
./setup_host_environment.sh
```

### 2. Start one ComfyUI service per GPU

```bash
./run_comfyui_cluster.sh
```

Examples:

```bash
# All detected GPUs
./run_comfyui_cluster.sh

# Specific GPUs
GPU_IDS="0,1,3" ./run_comfyui_cluster.sh

# First 2 detected GPUs starting at port 8180
MAX_GPUS=2 BASE_COMFY_PORT=8180 ./run_comfyui_cluster.sh
```

This creates tmux sessions named `comfyui-g<gpu-id>` by default.

### 3. Run the full multi-GPU pipeline

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
./run_multi_gpu_pipeline.sh
```

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

- `setup_host_environment.sh`
  - Installs the pinned host Python environment and ComfyUI runtime
- `run_comfyui_service.sh`
  - Starts one local ComfyUI API service for one GPU
- `run_comfyui_cluster.sh`
  - Starts or reuses one ComfyUI tmux session per GPU
- `run_full_pipeline.sh`
  - Runs one shard on one GPU
- `run_multi_gpu_pipeline.sh`
  - Splits one `SRC` across GPUs and merges outputs
- `download-models.sh`
  - Downloads required models into `models/comfyui` and `models/privacy_blur`

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
- `run_comfyui_cluster.sh` uses tmux for simplicity. For long-lived production services, move the same command line into `systemd` units.
- Keep the model cache on shared local storage if multiple repo copies use the same machine.
- The orchestrator can auto-start missing ComfyUI services. Set `START_COMFYUI_CLUSTER=0` if you manage them externally.

## Migration Guide

Detailed migration notes and the end-to-end host-native rollout plan live in:

`docs/native-no-docker-migration.md`
