# Native No-Docker Migration

This repository now runs directly on the host. Docker, Docker Compose, and the NVIDIA container toolkit are no longer part of the runtime.

## Final Architecture

The native production layout is:

1. One shared host Python environment in `.venv`
2. One pinned local `ComfyUI/` checkout
3. One shared model cache under `models/comfyui` and `models/privacy_blur`
4. One temporary ComfyUI API service per GPU during inpainting
5. One per-GPU worker data root under `native_data/gpu<id>`
6. One tmux shard worker per GPU during pipeline runs

The pipeline stages remain the same:

1. SAM3 tiled mask generation
2. ComfyUI inpainting
3. Laplacian postprocess
4. Privacy blur
5. Stage counting and output merge

## What Changed

### Removed

- `Dockerfile`
- `docker-compose.yml`
- Runtime container startup in `run_full_pipeline.sh`
- Docker image pulls and Docker GPU runtime preflight in `run_multi_gpu_pipeline.sh`

### Added

- `setup_host_environment.sh`
- `run_comfyui_service.sh`
- `run_comfyui_cluster.sh`

### Replaced

- `run_full_pipeline.sh` now runs natively on host paths
- `run_multi_gpu_pipeline.sh` now orchestrates host-native shard workers

## Path Mapping

| Old container path | Native host path |
|---|---|
| `/workspace/ComfyUI/input` | `native_data/gpu<id>/input` |
| `/workspace/ComfyUI/output` | `native_data/gpu<id>/output` |
| `/workspace/output-sam3-mask` | `native_data/gpu<id>/output-sam3-mask` |
| `/workspace/output-postprocessed` | `native_data/gpu<id>/output-postprocessed` |
| `/workspace/output-egoblur` | `native_data/gpu<id>/output-egoblur` |
| `/workspace/ComfyUI/models` | `models/comfyui` |
| `/workspace/models/privacy_blur` | `models/privacy_blur` |
| `/workspace/inpainting` | `inpainting-workflow-master` |

## Single Public Entry Point

Normal operation uses one command only:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
./run_multi_gpu_pipeline.sh
```

`run_multi_gpu_pipeline.sh` bootstraps a `uv`-managed Python `3.10` environment, downloads missing models, runs one shard per GPU, starts ComfyUI only for inpainting, stops it after inpainting, and merges final outputs.

## Internal Bootstrap Helper

`setup_host_environment.sh` still exists, but it is an internal helper used by `run_multi_gpu_pipeline.sh`.

It installs `uv`, creates `.venv`, pins `ComfyUI`, restores `Comfy-Lock.yaml`, installs dependencies, links the model paths, and downloads missing models.

## Internal ComfyUI Helper

Each GPU gets one local ComfyUI HTTP API process only during the inpainting stage by default.

Example service mapping:

- GPU 0 -> port `8180` -> `native_data/gpu0`
- GPU 1 -> port `8181` -> `native_data/gpu1`
- GPU 2 -> port `8182` -> `native_data/gpu2`

`run_comfyui_cluster.sh` is still available for debugging, but operators should not need it in normal usage.

## Multi-GPU Pipeline Run

Run:

```bash
SRC="/absolute/path/to/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" \
./run_multi_gpu_pipeline.sh
```

Examples:

```bash
SRC="/data/panos" \
FINAL_OUTPUT_DIR="/data/final" \
GPU_IDS="0,1,2,3" \
./run_multi_gpu_pipeline.sh

SRC="/data/panos" \
FINAL_OUTPUT_DIR="/data/final" \
STOP_AFTER_STAGE=postprocess \
./run_multi_gpu_pipeline.sh
```

## Single-Shard Debug Run

You can still run one shard directly:

```bash
GPU_ID=0 \
CUDA_VISIBLE_DEVICES=0 \
COMFY_PORT=8180 \
SRC="/absolute/path/to/shard" \
BATCH_NAME="debug-g0" \
./run_full_pipeline.sh
```

## Operational Guidance For 8 GPUs

1. Start with `SAM3_WORKERS=1`, `POSTPROCESS_WORKERS=1`, and `PRIVACY_WORKERS=1`
2. Scale worker counts only after confirming stable VRAM headroom
3. ComfyUI is stopped after inpainting so VRAM is reclaimed before downstream stages
4. Use shared model storage when multiple checkouts live on the same host
5. Move the `run_comfyui_service.sh` command into `systemd` once the tmux workflow is validated

## Validation Checklist

1. Run one small shard on one GPU
2. Verify `logs/fullrun_<run>.events.jsonl`
3. Confirm counts match across stages
4. Compare outputs against a known-good baseline batch
5. Run 2 GPUs
6. Run all 8 GPUs with conservative worker counts

## Remaining Risk Areas

- Host Python dependency drift if setup is bypassed
- VRAM contention if multiple workers are increased aggressively
- Misconfigured ComfyUI paths if services are started outside the provided scripts

The native scripts reduce those risks by pinning versions, using one service per GPU, and standardizing the per-GPU filesystem layout.
