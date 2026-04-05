# Native (No-Docker) Migration Plan

This document describes how to run the same 5-stage panorama pipeline without Docker while preserving multi-GPU sharding, logs, and batch outputs.

## Goals

- Remove runtime Docker dependency (`docker`, `docker compose`, NVIDIA container toolkit).
- Keep the same stage semantics:
  1. SAM3 tiled mask generation
  2. ComfyUI inpainting
  3. Laplacian sky replacement postprocess
  4. Privacy blur
  5. Count/report outputs
- Keep multi-GPU sharding and tmux-based orchestration.
- Keep machine portability through reproducible local setup scripts and pinned versions.

## New Native Entry Points

- `run_multi_gpu_pipeline_native.sh`
  - Splits `SRC` into one shard per GPU.
  - Launches one tmux session per GPU.
  - Runs `run_full_pipeline_native.sh` in each session.
- `run_full_pipeline_native.sh`
  - Executes all stages locally using Python, no Docker calls.

## Required Local Runtime Layout

Recommended host layout (defaults used by native scripts):

- `ComfyUI/` (local ComfyUI checkout)
- `ComfyUI/input/`
- `ComfyUI/output/`
- `models/comfyui/` (Qwen, SAM3, lama, etc.)
- `models/privacy_blur/` (face & LP detectors)
- `native_data/gpuX/` (SAM3/postprocess/egoblur outputs per GPU)

## Critical Path Changes

### 1) Replace Container Paths with Host Paths

Container paths such as `/workspace/ComfyUI/input` and `/workspace/output-sam3-mask` are replaced by host paths:

- Comfy input root: `${COMFY_INPUT_ROOT}`
- Comfy output root: `${COMFY_OUTPUT_ROOT}`
- SAM3 output: `${COMFY_DATA_DIR}/output-sam3-mask/<batch>`
- Postprocess output: `${COMFY_DATA_DIR}/output-postprocessed/<batch>`
- Egoblur output: `${COMFY_DATA_DIR}/output-egoblur/<batch>`

### 2) Make ComfyUI API Runner Root-Configurable

`inpainting-workflow-master/comfyui_run.py` now accepts:

- `--comfy-input-root`
- `--comfy-output-root`

This removes hard dependency on `/workspace/ComfyUI/{input,output}`.

### 3) Run Per-GPU Native Process with GPU Pinning

Each shard runs with `NVIDIA_VISIBLE_DEVICES=<gpu_id>` and uses its own:

- Batch name (`<run>_g<gpu_id>`)
- Output roots (`native_data/gpu<id>/...`)
- Optional final copy target (`FINAL_OUTPUT_DIR/<run>/gpu<id>/...`)

### 4) Keep Orchestration Semantics

- Source split uses `pipeline_helpers.py split-shards`.
- tmux session per GPU keeps long-running processes detached.
- Existing helper commands for staging, linking, and counting are reused.

## Native Execution Steps (Production Checklist)

1. **Provision machine baseline**
   - Install NVIDIA driver + CUDA-compatible PyTorch stack.
   - Ensure `nvidia-smi`, `python3`, `tmux` are available.

2. **Install Python environments**
   - One venv for orchestration/stages and Comfy custom nodes.
   - Install `pipeline-requirements.txt` plus ComfyUI dependencies.

3. **Install ComfyUI locally**
   - Pin ComfyUI commit to match your Docker image build.
   - Restore custom nodes/snapshots equivalent to prior image state.

4. **Prepare model cache**
   - Place models under `models/comfyui` and `models/privacy_blur`.
   - Confirm `LAMA_MODEL` exists at `models/comfyui/lama/big-lama.pt`.

5. **Start one ComfyUI API server per GPU/port**
   - Example: `gpu0->8180`, `gpu1->8181`, ...
   - Run each in tmux/systemd/supervisor.

6. **Run native multi-GPU pipeline**
   - Provide `SRC`, optional `FINAL_OUTPUT_DIR`, optional `GPU_IDS`.
   - Launch `./run_multi_gpu_pipeline_native.sh`.

7. **Validate counts and outputs**
   - Compare input counts vs stage counts per shard.
   - Verify final egoblur output parity.

8. **Operate at scale**
   - Move from ad-hoc tmux to service management:
     - systemd units for per-GPU ComfyUI API services
     - health checks and restart policies
   - Centralize logs (`logs/*.jsonl`) to ELK/W&B/Datadog.

## Suggested Environment Variables to Standardize

- Compute / placement:
  - `GPU_IDS`, `MAX_GPUS`, `NVIDIA_VISIBLE_DEVICES`
  - `BASE_COMFY_PORT`
- Paths:
  - `COMFYUI_HOME`, `COMFY_INPUT_ROOT`, `COMFY_OUTPUT_ROOT`
  - `MODELS_ROOT`, `MODELS_COMFYUI_DIR`, `MODELS_PRIVACY_DIR`
  - `NATIVE_DATA_ROOT`, `FINAL_OUTPUT_DIR`
- Pipeline controls:
  - `FORCE_REPROCESS`, `STOP_AFTER_STAGE`
  - `SAM3_*`, `POSTPROCESS_WORKERS`, `PRIVACY_WORKERS`

## Risks and Mitigations

- **Dependency drift** (host Python differs per machine)
  - Mitigation: lockfiles + scripted bootstrap + pinned commits.
- **GPU contention** (one process steals all VRAM)
  - Mitigation: dedicated ComfyUI service per GPU + explicit device pinning.
- **Path mismatch** (workflow assumes `/workspace/...`)
  - Mitigation: configurable roots and strict startup checks.
- **Operational regressions** (less isolation than containers)
  - Mitigation: service supervision, readonly model mounts where possible, and CI smoke tests.

## Rollout Strategy

1. Run one GPU in native mode and compare outputs with Docker baseline on same batch.
2. Run 2-GPU shadow mode and compare throughput + failure rate.
3. Promote native mode to primary once quality + reliability thresholds are met.
4. Keep Docker scripts temporarily as fallback until 2+ weeks of stable native runs.
