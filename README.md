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

## Required Files & Directory Structure

Before building, ensure you have the following files and directories:

```
container/
├── Dockerfile                    # Main Dockerfile
├── docker-compose.yml           # Docker Compose configuration
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
├── input/                       # Input images directory
│   ├── *.jpg                   # Your 360° panorama images
│   └── perspective_mask.png    # Mask for inpainting
│
├── output/                      # ComfyUI outputs
├── output-postprocessed/       # Postprocessed outputs
└── output-egoblur/             # Final outputs with blurring
```

### Download Required Models

**SAM3 Model:**
```bash
mkdir -p models/comfyui/sam3
# Download from: https://huggingface.co/facebook/sam2-hiera-large/tree/main
# Place sam3.pt in models/comfyui/sam3/
```

**EgoBlur Models:**
```bash
mkdir -p models/egoblur_gen2
# Download from: https://github.com/facebookresearch/EgoBlur/releases
# - ego_blur_face_gen2.jit
# - ego_blur_lp_gen2.jit
```

---

## Setup Instructions

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
  --input-dir /workspace/ComfyUI/input \
  --mask /workspace/ComfyUI/input/perspective_mask.png \
  --output-dir /workspace/ComfyUI/output \
  --workers 3

# Or use the bin script:
./bin/comfyui-run --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input \
  --mask /workspace/ComfyUI/input/perspective_mask.png
```

**Output:** `output/YYYY-MM-DD/ComfyUI_*.jpg`

### Stage 2: Postprocessing

Apply perspective-to-equirectangular transformation and sky blending.

```bash
docker exec comfyui-container python /workspace/inpainting/postprocess.py \
  -i /workspace/ComfyUI/output/2025-12-26 \
  -o /workspace/output-postprocessed \
  --top-mask /workspace/inpainting/sky_mask_updated.png \
  --pattern "*.jpg" \
  -j 1

# Or use the bin script:
./bin/postprocess -i /workspace/ComfyUI/output/2025-12-26 \
  -o /workspace/output-postprocessed \
  --top-mask /workspace/inpainting/sky_mask_updated.png
```

**Output:** `output-postprocessed/ComfyUI_*.jpg`

### Stage 3: EgoBlur (Privacy Protection)

Blur faces and license plates for privacy.

```bash
docker exec comfyui-container python /workspace/inpainting/egoblur_infer.py \
  --input-dir /workspace/output-postprocessed \
  --output-dir /workspace/output-egoblur \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit \
  --workers 1

# Or use the bin script:
./bin/egoblur --input-dir /workspace/output-postprocessed \
  --output-dir /workspace/output-egoblur \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit
```

**Output:** `output-egoblur/ComfyUI_*_egoblur.jpg`

### Complete Pipeline Example

```bash
# 1. Place your 360° images in input/
cp my_pano_*.jpg input/

# 2. Run ComfyUI inpainting
docker exec comfyui-container python /workspace/inpainting/comfyui_run.py \
  --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input \
  --mask /workspace/ComfyUI/input/perspective_mask.png

# 3. Get today's date for output directory
TODAY=$(date +%Y-%m-%d)

# 4. Run postprocessing
docker exec comfyui-container python /workspace/inpainting/postprocess.py \
  -i /workspace/ComfyUI/output/$TODAY \
  -o /workspace/output-postprocessed \
  --top-mask /workspace/inpainting/sky_mask_updated.png

# 5. Run EgoBlur
docker exec comfyui-container python /workspace/inpainting/egoblur_infer.py \
  --input-dir /workspace/output-postprocessed \
  --output-dir /workspace/output-egoblur \
  --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit \
  --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit

# 6. Find your final outputs
ls -lah output-egoblur/
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
- ./input:/workspace/ComfyUI/input
- ./output:/workspace/ComfyUI/output
- ./output-postprocessed:/workspace/output-postprocessed
- ./output-egoblur:/workspace/output-egoblur
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
docker exec comfyui-container python /workspace/inpainting/comfyui_run.py --workflow-json /workspace/workflow.json --input-dir /workspace/ComfyUI/input --mask /workspace/ComfyUI/input/perspective_mask.png
docker exec comfyui-container python /workspace/inpainting/postprocess.py -i /workspace/ComfyUI/output/$(date +%Y-%m-%d) -o /workspace/output-postprocessed --top-mask /workspace/inpainting/sky_mask_updated.png
docker exec comfyui-container python /workspace/inpainting/egoblur_infer.py --input-dir /workspace/output-postprocessed --output-dir /workspace/output-egoblur --face-model /workspace/inpainting/models/egoblur_gen2/ego_blur_face_gen2.jit --lp-model /workspace/inpainting/models/egoblur_gen2/ego_blur_lp_gen2.jit

# Stop
docker compose down
```
