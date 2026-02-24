# ComfyUI Inpainting & EgoBlur Pipeline

Complete Docker-based pipeline for 360° image inpainting with SAM3 segmentation, perspective transformation, and privacy blurring (faces & license plates).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Required Files & Directory Structure](#required-files--directory-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Scripts](#pipeline-scripts)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Prerequisites

### 1. Install Docker & Docker Compose

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

**For other systems:** Visit https://docs.docker.com/engine/install/

### 2. Install NVIDIA Container Toolkit (for GPU support)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 3. System Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested with L40S, 45GB VRAM)
- **CUDA**: Version 12.6 drivers recommended
- **Disk**: ~25GB for Docker image + models
- **RAM**: 32GB+ recommended

---

## Quick Start with Pre-built Image

The easiest way to get started is using the pre-built Docker image from Docker Hub:

### Step 1: Pull the Pre-built Image

```bash
docker pull amanbagrecha/container-comfyui:latest
```

### Step 2: Download Required Models (optional prefetch)

Models are not included in the Docker image due to their size. You can pre-download them now, or skip this step and let `run_full_pipeline.sh` auto-download missing models.

```bash
# Run the model download script
./download-models.sh
```

The script will download:
- **Text Encoder**: Qwen 2.5 VL 7B (FP8)
- **VAE**: Qwen Image VAE
- **LoRA**: Qwen Image Edit Lightning (4-step)
- **Upscaler**: Real-ESRGAN x2
- **Diffusion Model**: Qwen Image Edit 2509 (FP8)
- **SAM3**: Segment Anything Model 3 (from [public mirror](https://huggingface.co/aravgarg588/comfyui-container-model))
- **EgoBlur Face**: Face detection model (from [public mirror](https://huggingface.co/aravgarg588/comfyui-container-model))
- **EgoBlur License Plate**: License plate detection model (from [public mirror](https://huggingface.co/aravgarg588/comfyui-container-model))

**Note:** The script automatically skips models that already exist, so you can safely re-run it. No HuggingFace token required!

### Step 3: Run Full Pipeline (one command)

`run_full_pipeline.sh` now auto-creates local directories, auto-copies `perspective_mask.png` into `$COMFYUI_DATA_DIR/input/`, auto-starts Docker (if needed), and can auto-download models if missing.

```bash
SRC="/absolute/path/to/your/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/your/final_outputs" \
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" \
./run_full_pipeline.sh
```

For shared models across multiple repo copies or machines mounted on shared storage:

```bash
SRC="/absolute/path/to/your/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/your/final_outputs" \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" \
./run_full_pipeline.sh
```

Notes:
- `SRC` is your image folder.
- Final egoblur outputs are copied to `FINAL_OUTPUT_DIR/<batch-name>/`.
- Intermediate outputs are stored under `comfyui_data/<container-name>/` by default.
- Defaults: `POSTPROCESS_WORKERS=1`, `EGOBLUR_WORKERS=3`.
- Default rerun behavior is skip/resume (`FORCE_REPROCESS=0`); set `FORCE_REPROCESS=1` to clear the batch and rerun from scratch.

Optional overrides:

```bash
CONTAINER_NAME="comfyui-container" \
POSTPROCESS_WORKERS=1 \
EGOBLUR_WORKERS=3 \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
NVIDIA_VISIBLE_DEVICES=0 \
COMFY_PORT=8188 \
AUTO_DOWNLOAD_MODELS=1 \
FORCE_REPROCESS=0 \
./run_full_pipeline.sh
```

Image quality defaults:
- ComfyUI inpainting output (`workflow-updated.json`) saves JPG at quality `80`.
- Postprocess stage saves JPG at quality `90`.

Multi-GPU pattern (one container per GPU, shared models):

```bash
NVIDIA_VISIBLE_DEVICES=0 CONTAINER_NAME=comfyui-g0 COMFY_PORT=8188 MODELS_ROOT=/data/shared-models SRC=/data/shard0 FINAL_OUTPUT_DIR=/data/out0 ./run_full_pipeline.sh
NVIDIA_VISIBLE_DEVICES=1 CONTAINER_NAME=comfyui-g1 COMFY_PORT=8189 MODELS_ROOT=/data/shared-models SRC=/data/shard1 FINAL_OUTPUT_DIR=/data/out1 ./run_full_pipeline.sh
NVIDIA_VISIBLE_DEVICES=2 CONTAINER_NAME=comfyui-g2 COMFY_PORT=8190 MODELS_ROOT=/data/shared-models SRC=/data/shard2 FINAL_OUTPUT_DIR=/data/out2 ./run_full_pipeline.sh
NVIDIA_VISIBLE_DEVICES=3 CONTAINER_NAME=comfyui-g3 COMFY_PORT=8191 MODELS_ROOT=/data/shared-models SRC=/data/shard3 FINAL_OUTPUT_DIR=/data/out3 ./run_full_pipeline.sh
```

---

## Building from Source

If you prefer to build the Docker image yourself, follow these instructions:

## Required Files & Directory Structure

Before building, ensure you have the following files and directories:

```
container/
├── Dockerfile                    # Main Dockerfile
├── docker-compose.yml           # Docker Compose configuration
├── download-models.sh           # Automated model download script
├── Comfy-Lock.yaml              # ComfyUI snapshot for reproducible builds
├── workflow-updated.json        # ComfyUI workflow definition
├── .dockerignore                # Files to exclude from build
│
├── bin/                         # Executable scripts
│   ├── comfyui-run             # ComfyUI workflow runner
│   ├── postprocess             # Perspective-to-equirectangular postprocessing
│   └── egoblur                 # Face & license plate blurring
│
├── inpainting-workflow-master/  # Inpainting scripts
│   ├── comfyui_run.py
│   ├── egoblur_infer.py
│   ├── postprocess.py
│   └── sky_mask_updated.png    # Top mask for postprocessing
│
├── models/                      # Model files
│   ├── comfyui/                # ComfyUI models (checkpoints, VAE, etc.)
│   │   └── sam3/               # SAM3 model
│   │       └── sam3.pt         # Required for segmentation
│   └── egoblur_gen2/           # EgoBlur models
│       ├── ego_blur_face_gen2.jit    # Face detection model (~400MB)
│       └── ego_blur_lp_gen2.jit      # License plate model (~400MB)
│
├── comfyui_data/<container-name>/
│   ├── input/                  # Staged input images + perspective_mask.png
│   ├── output/                 # ComfyUI outputs
│   ├── output-postprocessed/   # Postprocessed outputs
│   └── output-egoblur/         # Final outputs with blurring
```

---

## Build Instructions (From Source)

### Step 1: Clone/Copy Project Files

```bash
# Navigate to your project directory
cd /path/to/container

# Ensure all required files are present
ls -la
```

### Step 2: Clean Up Old Docker Resources (Optional)

```bash
# Remove old containers
docker ps -a  # List all containers
docker rm -f comfyui-container  # Remove if exists

# Clean up unused images and cache
docker system prune -a --volumes -f
```

### Step 3: Build Docker Image

```bash
# Build the image (takes 10-20 minutes)
docker build -t container-comfyui:latest -f Dockerfile .

# Verify the image was built
docker images | grep container-comfyui
```

### Step 4: Start the Container

```bash
# Start with docker-compose (recommended)
docker compose up -d

# Check container status
docker ps

# View startup logs
docker logs comfyui-container --tail 50
```

**ComfyUI will be accessible at:** http://localhost:8188

---

## Running the Pipeline

The pipeline consists of three stages, each with a dedicated script in the `bin/` directory.

### Stage 1: ComfyUI Inpainting

Process images through ComfyUI workflow with SAM3 segmentation and inpainting.

```bash
docker exec comfyui-container python /workspace/inpainting/comfyui_run.py \
  --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input/<batch-name> \
  --mask /workspace/ComfyUI/input/perspective_mask.png \
  --output-dir /workspace/ComfyUI/output/<batch-name> \
  --workers 3

# Or use the bin script:
./bin/comfyui-run --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input/<batch-name> \
  --mask /workspace/ComfyUI/input/perspective_mask.png
```

**Output:** `comfyui_data/<container-name>/output/<batch-name>/*.jpg`

### Stage 2: Postprocessing

Apply perspective-to-equirectangular transformation and sky blending.

```bash
docker exec comfyui-container python /workspace/inpainting/postprocess.py \
  -i /workspace/ComfyUI/output/<batch-name> \
  -o /workspace/output-postprocessed/<batch-name> \
  --top-mask /workspace/inpainting/sky_mask_updated.png \
  --pattern "*.jpg" \
  -j 1

# Or use the bin script:
./bin/postprocess -i /workspace/ComfyUI/output/<batch-name> \
  -o /workspace/output-postprocessed/<batch-name> \
  --top-mask /workspace/inpainting/sky_mask_updated.png
```

**Output:** `comfyui_data/<container-name>/output-postprocessed/<batch-name>/*.jpg`

### Stage 3: EgoBlur (Privacy Protection)

Blur faces and license plates for privacy.

```bash
docker exec comfyui-container python /workspace/inpainting/egoblur_infer.py \
  --input-dir /workspace/output-postprocessed/<batch-name> \
  --output-dir /workspace/output-egoblur/<batch-name> \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit \
  --workers 1

# Or use the bin script:
./bin/egoblur --input-dir /workspace/output-postprocessed/<batch-name> \
  --output-dir /workspace/output-egoblur/<batch-name> \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit
```

**Output:** `comfyui_data/<container-name>/output-egoblur/<batch-name>/*.jpg`

### Complete Pipeline Example

```bash
# 1. Run complete 3-stage pipeline
SRC="/absolute/path/to/input_images"
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs"
BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)"
SRC="$SRC" FINAL_OUTPUT_DIR="$FINAL_OUTPUT_DIR" BATCH_NAME="$BATCH_NAME" ./run_full_pipeline.sh

# 2. Check stage outputs for this batch
ls -lah "comfyui_data/comfyui-container/output/$BATCH_NAME"
ls -lah "comfyui_data/comfyui-container/output-postprocessed/$BATCH_NAME"
ls -lah "comfyui_data/comfyui-container/output-egoblur/$BATCH_NAME"

# 3. Optional copied final output
ls -lah "$FINAL_OUTPUT_DIR/$BATCH_NAME"
```

---

## Pipeline Scripts

### Script Options

#### `comfyui_run.py`
```bash
Options:
  --workflow-json FILE   [required] - Path to workflow JSON
  --input-dir DIRECTORY  [required] - Input images directory
  --mask FILE            [required] - Inpainting mask
  --output-dir PATH      [default: comfy_outputs]
  --server TEXT          [default: http://127.0.0.1:8188]
  --workers INTEGER      [default: 3] - Parallel processing
  --image-node-id TEXT   [default: 349] - LoadImage node ID
  --mask-node-id TEXT    [default: 463] - Mask node ID
```

#### `postprocess.py`
```bash
Options:
  -i, --input PATH       [required] - Input directory
  -o, --output PATH      [required] - Output directory
  --top-mask PATH        [required] - Sky mask for blending
  --pattern TEXT         [default: *.jpg]
  -j, --workers INTEGER  [default: 1]
  --fov-w INTEGER        [default: 70] - Perspective FOV width (degrees)
  --fov-h INTEGER        [default: 70] - Perspective FOV height (degrees)
  --h-deg FLOAT          [default: 180] - Horizontal rotation
  --v-deg FLOAT          [default: 50] - Vertical rotation
```

#### `egoblur_infer.py`
```bash
Options:
  --input-dir DIRECTORY  [required]
  --output-dir PATH      [required]
  --face-model FILE      [required] - Face detection model
  --lp-model FILE        [required] - License plate model
  --workers INTEGER      [default: 4] - Use 1 for CUDA
  --blur [soft|pixelate] [default: soft]
  --blur-strength FLOAT  [default: 0.5]
  --face-threshold FLOAT [default: 0.3]
  --lp-threshold FLOAT   [default: 0.4]
```

---

## Monitoring & Troubleshooting

### Container Management

```bash
# View logs
docker logs comfyui-container --tail 100
docker logs -f comfyui-container  # Follow logs

# Check container status
docker ps
docker stats comfyui-container

# Restart container
docker compose restart

# Stop container
docker compose stop

# Start container
docker compose start

# Rebuild after Dockerfile changes
docker compose down
docker build -t container-comfyui:latest .
docker compose up -d
```

### Run Postprocess/EgoBlur Without ComfyUI API

Use this mode when inpainting outputs already exist and you want to free ComfyUI VRAM before downstream stages.

```bash
# 1) Stop ComfyUI API containers (frees ~35GB VRAM/GPU in large runs)
docker stop comfyui-g0 comfyui-g1

# 2) Postprocess only (recommended: -j 6)
tmux new-session -d -s postproc-g0 "bash -lc '
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
'"

tmux new-session -d -s postproc-g1 "bash -lc '
docker run --rm --name postproc-g1 --gpus device=1 \
  -v /root/.cache/torch:/root/.cache/torch \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/p2e-local:/workspace/ComfyUI/custom_nodes/p2e \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g1/output:/workspace/ComfyUI/output \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g1/output-postprocessed:/workspace/output-postprocessed \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/postprocess.py \
    -i /workspace/ComfyUI/output/gpu1-batch \
    -o /workspace/output-postprocessed/gpu1-batch \
    --top-mask /workspace/inpainting/sky_mask_updated.png \
    --pattern "*.jpg" \
    -j 6
'"

# 3) Verify postprocess completed before egoblur
python3 - <<'PY'
from pathlib import Path
exts={'.png','.jpg','.jpeg','.webp','.tif','.tiff'}
for c,b in [('comfyui-g0','gpu0-batch'),('comfyui-g1','gpu1-batch')]:
    base=Path('/root/comfyui-workflow-docker/comfyui_data')/c
    i=sum(1 for p in (base/'output'/b).rglob('*') if p.is_file() and p.suffix.lower() in exts)
    p=sum(1 for p in (base/'output-postprocessed'/b).rglob('*') if p.is_file() and p.suffix.lower() in exts)
    print(f'{c} inpaint={i} postprocess={p}')
PY

# 4) Egoblur only (requested setting: --workers 12)
tmux new-session -d -s egoblur-g0 "bash -lc '
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
'"

tmux new-session -d -s egoblur-g1 "bash -lc '
docker run --rm --name egoblur-g1 --gpus device=1 \
  -v /root/comfyui-workflow-docker/inpainting-workflow-master:/workspace/inpainting \
  -v /root/comfyui-workflow-docker/models/egoblur_gen2:/workspace/inpainting/models/egoblur_gen2:ro \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g1/output-postprocessed:/workspace/output-postprocessed \
  -v /root/comfyui-workflow-docker/comfyui_data/comfyui-g1/output-egoblur:/workspace/output-egoblur \
  amanbagrecha/container-comfyui:latest \
  python /workspace/inpainting/egoblur_infer.py \
    --input-dir /workspace/output-postprocessed/gpu1-batch \
    --output-dir /workspace/output-egoblur/gpu1-batch \
    --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
    --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit \
    --device cuda \
    --workers 12
'"

# 5) Verify egoblur counts match postprocess counts
python3 - <<'PY'
from pathlib import Path
exts={'.png','.jpg','.jpeg','.webp','.tif','.tiff'}
for c,b in [('comfyui-g0','gpu0-batch'),('comfyui-g1','gpu1-batch')]:
    base=Path('/root/comfyui-workflow-docker/comfyui_data')/c
    p=sum(1 for p in (base/'output-postprocessed'/b).rglob('*') if p.is_file() and p.suffix.lower() in exts)
    e=sum(1 for p in (base/'output-egoblur'/b).rglob('*') if p.is_file() and p.suffix.lower() in exts)
    print(f'{c} postprocess={p} egoblur={e} match={p==e}')
PY
```

Notes:
- `postprocess.py` with high workers can fail partially on GPU OOM; `-j 6` is a safer parallel setting.
- `egoblur_infer.py --workers 12` is aggressive; if you see instability, lower workers.

### System Monitoring

```bash
# View CPU/Memory usage
docker exec comfyui-container htop

# View GPU usage
docker exec comfyui-container nvtop

# Check Python environment
docker exec comfyui-container bash -c "source /workspace/.venv/bin/activate && python --version"

# Test GPU in container
docker exec comfyui-container nvidia-smi
```

### Access Container Shell

```bash
docker exec -it comfyui-container bash

# Inside container:
cd /workspace
source .venv/bin/activate
python --version
```

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

**Out of memory:**
- Reduce `--workers` parameter in scripts
- Increase `shm_size` in docker-compose.yml
- Process fewer images at once

**ComfyUI not accessible:**
```bash
# Check if server is running
docker logs comfyui-container | grep "To see the GUI"

# Port might be in use
sudo lsof -i :8188
```

---

## What's Included

### Base System
- NVIDIA CUDA 12.6 runtime
- Python 3.10.12 environment
- System monitoring tools (htop, nvtop)
- uv package manager for fast dependency management

### ComfyUI Setup
- ComfyUI with ComfyUI-CLI
- PyTorch 2.9.1 with CUDA 12.6 support
- Custom nodes:
  - ComfyUI-Manager
  - ComfyUI-SAM3 (for segmentation)
  - ComfyUI Essentials
  - ComfyUI LayerStyle
  - WAS Node Suite
  - PyTorch360Convert

### Pipeline Libraries
- **EgoBlur**: Face & license plate detection/blurring (installed as package + cloned repo)
- **p2e**: Perspective-to-equirectangular transformation
- **simple-lama-inpainting**: LAMA-based inpainting
- **SAM3**: Segment Anything Model 3 for object detection

### Volume Mounts
```yaml
- ./models/comfyui:/workspace/ComfyUI/models
- ./models/egoblur_gen2:/workspace/inpainting/models/egoblur_gen2
- ${COMFYUI_DATA_DIR}/input:/workspace/ComfyUI/input
- ${COMFYUI_DATA_DIR}/output:/workspace/ComfyUI/output
- ${COMFYUI_DATA_DIR}/output-postprocessed:/workspace/output-postprocessed
- ${COMFYUI_DATA_DIR}/output-egoblur:/workspace/output-egoblur
- ./workflow-updated.json:/workspace/workflow.json:ro
```

---

## Performance Benchmarks

**Tested on NVIDIA L40S (45GB VRAM):**

| Stage | Images | Time | Speed |
|-------|--------|------|-------|
| ComfyUI Inpainting | 4 | ~50s | ~12s/image |
| Postprocessing | 6 | ~13s | ~2s/image |
| EgoBlur | 6 | ~20s | ~3s/image |

**Total pipeline:** ~83s for 6 images (~14s per image)

---

## License

This project uses multiple open-source components:
- ComfyUI: GPL-3.0
- SAM3: Apache-2.0
- EgoBlur: CC-BY-NC-4.0
- PyTorch: BSD-3-Clause

Please review individual component licenses for commercial use.

---

## Support

For issues:
1. Check container logs: `docker logs comfyui-container`
2. Verify GPU access: `docker exec comfyui-container nvidia-smi`
3. Check disk space: `df -h`
4. Review model files are downloaded correctly

---

## Quick Reference

```bash
# Build
docker build -t container-comfyui:latest .

# Start
docker compose up -d

# Process images (full pipeline)
SRC="/absolute/path/to/input_images" FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" BATCH_NAME="batch-$(date +%Y%m%d_%H%M%S)" ./run_full_pipeline.sh

# Stop
docker compose down
```
