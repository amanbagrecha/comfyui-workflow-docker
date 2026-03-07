# ComfyUI Inpainting & Privacy Blur Pipeline

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

Models are not included in the Docker image due to their size. You can pre-download them now, or skip this step and let `run_multi_gpu_pipeline.sh` auto-download missing models during shard runs.

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
- **SAM3**: Segment Anything Model 3 in HF transformers format (from [public mirror](https://huggingface.co/aravgarg588/comfyui-container-model))
- **SimpleLama Checkpoint**: `big-lama.pt` for postprocess inpainting
- **Privacy Face Detector**: Ultralytics YOLOv8n face model (`face_yolov8n.pt`)
- **Privacy LP Detector**: YOLOv9-s-608 ONNX license plate model (`yolo-v9-s-608-license-plates-end2end.onnx`)

**Note:** The script automatically skips models that already exist, so you can safely re-run it. No HuggingFace token required!
`big-lama.pt` is saved at `models/comfyui/lama/big-lama.pt` and used by postprocess via `LAMA_MODEL`.

### Step 3: Run Multi-GPU Pipeline (one command, recommended entry point)

`run_multi_gpu_pipeline.sh` is the primary entry point. It auto-shards one large `SRC`, launches one tmux job per GPU, and each shard run auto-creates local directories, auto-copies `perspective_mask.png` into `$COMFYUI_DATA_DIR/input/`, auto-stages `chrome_xWUjmfs7m4.png` from `inpainting-workflow-master/reference_sky.png`, auto-starts Docker (if needed), and can auto-download models if missing.

```bash
SRC="/absolute/path/to/your/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/your/final_outputs" \
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
./run_multi_gpu_pipeline.sh
```

Default values:
| Variable | Default |
|----------|---------|
| RUN_NAME | multigpu-YYYYmmdd_HHMMSS |
| GPU_IDS | auto |
| MAX_GPUS | 0 (all detected) |
| BASE_COMFY_PORT | 8188 |
| CONTAINER_PREFIX | comfyui-g |
| COMFYUI_DATA_ROOT | ./comfyui_data |
| MODELS_ROOT | ./models |
| MODELS_COMFYUI_DIR | ./models/comfyui |
| MODELS_EGOBLUR_DIR | ./models/egoblur_gen2 (legacy) |
| MODELS_PRIVACY_DIR | ./models/privacy_blur |
| POSTPROCESS_WORKERS | 4 |
| EGOBLUR_WORKERS | 4 (legacy) |
| PRIVACY_WORKERS | 2 |
| SAM3_WORKERS | 4 |
| SAM3_RESIZE_WIDTH | 4000 |
| SAM3_RESIZE_HEIGHT | 2000 |
| SAM3_GLARE_THRESHOLD | 0.4 |
| SAM3_TILE_ROWS | 2 |
| SAM3_TILE_COLS | 1 |
| SAM3_SCRIPT | sam3_tiled_mask.py |
| DOWNSTREAM_MODE | isolated |
| STOP_AFTER_STAGE | egoblur |
| FORCE_REPROCESS | 0 |
| AUTO_DOWNLOAD_MODELS | 1 |
| RESET_CONTAINER_BEFORE_RUN | 1 |
| COMFY_IMAGE_NODE_ID | 91 |
| COMFY_MASK_NODE_ID | 34 |
| COMFY_SAM3_MASK_NODE_ID | 60 |
| SKY_REFERENCE_SOURCE | ./inpainting-workflow-master/reference_sky.png |
| SKY_REFERENCE_FILENAME | chrome_xWUjmfs7m4.png |

For shared models across multiple repo copies or machines mounted on shared storage:

```bash
SRC="/absolute/path/to/your/input_images" \
FINAL_OUTPUT_DIR="/absolute/path/to/your/final_outputs" \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" \
./run_multi_gpu_pipeline.sh
```

Notes:
- `SRC` is your image folder.
- Final privacy-blur outputs are copied to `FINAL_OUTPUT_DIR/<run-name>/gpu<id>/`.
- Intermediate outputs are stored under `comfyui_data/<container-name>/` per GPU shard (for example, `comfyui-g0`, `comfyui-g1`).
- Defaults: `POSTPROCESS_WORKERS=4`, `PRIVACY_WORKERS=2`, `SAM3_WORKERS=4`.
- Default rerun behavior is skip/resume (`FORCE_REPROCESS=0`); set `FORCE_REPROCESS=1` to clear the batch and rerun from scratch.

Optional overrides:

```bash
RUN_NAME="multigpu-myrun" \
GPU_IDS="0,1,3" \
MAX_GPUS=2 \
BASE_COMFY_PORT=8188 \
CONTAINER_PREFIX="comfyui-g" \
POSTPROCESS_WORKERS=4 \
PRIVACY_WORKERS=2 \
MODELS_ROOT="/absolute/path/to/shared-model-cache" \
AUTO_DOWNLOAD_MODELS=1 \
FORCE_REPROCESS=0 \
./run_multi_gpu_pipeline.sh
```

Image quality defaults:
- ComfyUI inpainting output (`workflow-updated.json`) saves JPG at quality `80`.
- Postprocess stage saves JPG at quality `85`.
- Privacy blur stage saves JPG at quality `85`.

Automatic multi-GPU launcher (auto-shard one big `SRC` across detected GPUs):

```bash
SRC="/data/full-dataset" \
MODELS_ROOT="/data/shared-models" \
FINAL_OUTPUT_DIR="/data/final-outputs" \
./run_multi_gpu_pipeline.sh

# Optional: limit to first 2 detected GPUs
SRC="/data/full-dataset" MAX_GPUS=2 ./run_multi_gpu_pipeline.sh

# Optional: explicit GPU ids
SRC="/data/full-dataset" GPU_IDS="0,1,3" ./run_multi_gpu_pipeline.sh

# Optional on one-GPU machines: auto-stop other running comfyui containers
SRC="/data/full-dataset" MAX_GPUS=1 SINGLE_GPU_CONFLICT_MODE=stop ./run_multi_gpu_pipeline.sh
```

---

## Building from Source

If you prefer to build the Docker image yourself, follow these instructions:

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

## Debug: Running ComfyUI Container Only

To start just the ComfyUI container (without running the full pipeline) for debugging or API access:

```bash
# Using all defaults (./models/comfyui, ./models/privacy_blur, ./comfyui_data/comfyui-container)
docker compose -p comfyui-container up -d

# With custom model path (shared model cache)
MODELS_ROOT="/path/to/shared-model-cache" docker compose -p comfyui-container up -d

# With custom settings
MODELS_ROOT="./models" \
NVIDIA_VISIBLE_DEVICES=0 \
COMFY_PORT=8188 \
CONTAINER_NAME=comfyui-container \
docker compose -p comfyui-container up -d
```

Default values:
| Variable | Default |
|----------|---------|
| MODELS_COMFYUI_DIR | ./models/comfyui |
| MODELS_EGOBLUR_DIR | ./models/egoblur_gen2 (legacy) |
| MODELS_PRIVACY_DIR | ./models/privacy_blur |
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

---

## Running the Pipeline

The pipeline consists of four runtime stages, each with a dedicated script in the `bin/` directory.

### Stage 1: SAM3 Tiled Mask Generation

Generate per-image SAM3 sky/glare masks.

```bash
docker exec comfyui-container python /workspace/inpainting/sam3_tiled_mask.py \
  --input-dir /workspace/ComfyUI/input/<batch-name> \
  --output-dir /workspace/output-sam3-mask/<batch-name> \
  --pattern "*" \
  --resize-width 4000 \
  --resize-height 2000 \
  --glare-threshold 0.4 \
  --tile-rows 2 \
  --tile-cols 1 \
  --workers 1
```

**Output:** `comfyui_data/<container-name>/output-sam3-mask/<batch-name>/*.png`

This stage uses the transformers-based SAM3 loader (`Sam3Model`/`Sam3Processor`) from the local model directory under `/workspace/ComfyUI/models/sam3`.

Runner defaults map to SAM3 flags as:
- `SAM3_GLARE_THRESHOLD` -> `--glare-threshold` (default `0.4`)
- `SAM3_RESIZE_WIDTH`/`SAM3_RESIZE_HEIGHT` -> `--resize-width`/`--resize-height` (defaults `4000/2000`)
- `SAM3_TILE_ROWS`/`SAM3_TILE_COLS` -> `--tile-rows`/`--tile-cols` (defaults `2/1`)
- `SAM3_SCRIPT` chooses predictor script (`sam3_tiled_mask.py` default; `archive_sam3_tiled_mask.py` optional)

Current mask merge behavior in both SAM3 scripts:
- Sky + glare are merged by `max(sky, glare)` after per-prompt SAM3 thresholds.
- There is no additional post-merge threshold binarization step.

### Stage 2: ComfyUI Inpainting

Process images through ComfyUI workflow using external SAM3 masks + perspective mask.

```bash
docker exec comfyui-container python /workspace/inpainting/comfyui_run.py \
  --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input/<batch-name> \
  --mask /workspace/ComfyUI/input/perspective_mask.png \
  --sam3-mask-dir /workspace/output-sam3-mask/<batch-name> \
  --sam3-mask-node-id 60 \
  --image-node-id 91 \
  --mask-node-id 34 \
  --output-dir /workspace/ComfyUI/output/<batch-name> \
  --workers 3

# Or use the bin script:
./bin/comfyui-run --workflow-json /workspace/workflow.json \
  --input-dir /workspace/ComfyUI/input/<batch-name> \
  --mask /workspace/ComfyUI/input/perspective_mask.png
```

**Output:** `comfyui_data/<container-name>/output/<batch-name>/*.jpg`

Current default workflow (`workflow-updated.json`, synced from `carremoval-input-to-sky.json`) also expects the sky reference file `chrome_xWUjmfs7m4.png` in `/workspace/ComfyUI/input`; pipeline runners stage this automatically.

### Stage 3: Postprocessing (Laplacian + Seam Fix)

Run Laplacian sky replacement using SAM3 masks (`carremoved` as source, `newsky` as destination), then apply perspective top-fix and seam cleanup.

```bash
docker exec -e LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt comfyui-container python /workspace/inpainting/postprocess.py \
  -i /workspace/ComfyUI/output/<batch-name> \
  -o /workspace/output-postprocessed/<batch-name> \
  --top-mask /workspace/inpainting/sky_mask_updated.png \
  --sam3-mask-dir /workspace/output-sam3-mask/<batch-name> \
  --dilation 1 \
  --blur 10 \
  --levels 7 \
  --pattern "*.jpg" \
  -j 1

# Or use the bin script:
./bin/postprocess -i /workspace/ComfyUI/output/<batch-name> \
  -o /workspace/output-postprocessed/<batch-name> \
  --top-mask /workspace/inpainting/sky_mask_updated.png \
  --sam3-mask-dir /workspace/output-sam3-mask/<batch-name>
```

**Output:** `comfyui_data/<container-name>/output-postprocessed/<batch-name>/*.jpg`

`postprocess.py` uses `LAMA_MODEL=/workspace/ComfyUI/models/lama/big-lama.pt` so workers do not race on first-time model download.
Laplacian sky replacement currently runs in full-image mode (`--laplacian-roi-pad` is deprecated/no-op).

### Stage 4: Privacy Blur (Face + LP)

Blur faces and license plates using `privacy_blur_parallel.sh` and `privacy_blur_infer.py`.

```bash
SRC="/root/comfyui-workflow-docker/comfyui_data/comfyui-container/output-postprocessed/<batch-name>" \
OUT_ROOT="/root/comfyui-workflow-docker/comfyui_data/comfyui-container/output-egoblur/<batch-name>" \
FACE_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/face_yolov8n.pt" \
LP_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx" \
WORKERS=2 \
OUTPUT_MODE=blur_only \
PYTHON_BIN=/data/.venv/bin/python \
bash /root/comfyui-workflow-docker/inpainting-workflow-master/privacy_blur_parallel.sh
```

**Output:** `comfyui_data/<container-name>/output-egoblur/<batch-name>/*.jpg`

### Complete Pipeline Example

```bash
# 1. Run complete pipeline
SRC="/absolute/path/to/input_images"
FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs"
RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)"
GPU_IDS="0,1"
SRC="$SRC" FINAL_OUTPUT_DIR="$FINAL_OUTPUT_DIR" RUN_NAME="$RUN_NAME" GPU_IDS="$GPU_IDS" ./run_multi_gpu_pipeline.sh

# Optional: stop after postprocess for partial runs
SRC="$SRC" RUN_NAME="${RUN_NAME}-post" GPU_IDS="$GPU_IDS" STOP_AFTER_STAGE=postprocess ./run_multi_gpu_pipeline.sh

# Optional: SAM3-only run
SRC="$SRC" RUN_NAME="${RUN_NAME}-sam3" GPU_IDS="$GPU_IDS" STOP_AFTER_STAGE=sam3 ./run_multi_gpu_pipeline.sh

# 2. Check stage outputs for one shard
ls -lah "comfyui_data/comfyui-g0/output/${RUN_NAME}-g0"
ls -lah "comfyui_data/comfyui-g0/output-postprocessed/${RUN_NAME}-g0"
ls -lah "comfyui_data/comfyui-g0/output-egoblur/${RUN_NAME}-g0"

# 3. Optional copied final output
ls -lah "$FINAL_OUTPUT_DIR/$RUN_NAME"
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
  --sam3-mask-dir DIR               - Optional per-image SAM3 mask directory
  --sam3-mask-node-id TEXT          - SAM3 mask node id when SAM3 dir is set
  --output-dir PATH      [default: comfy_outputs]
  --server TEXT          [default: http://127.0.0.1:8188]
  --workers INTEGER      [default: 3] - Parallel processing
  --image-node-id TEXT   [default: 91] - Main image node ID
  --mask-node-id TEXT    [default: 34] - Perspective mask node ID
```

#### `postprocess.py`
```bash
Options:
  -i, --input PATH       [required] - Input directory
  -o, --output PATH      [required] - Output directory
  --top-mask PATH        [required] - Sky mask for blending
  --sam3-mask-dir DIR               - Enable Laplacian sky replacement mode
  --dilation INTEGER     [default: 1]
  --blur INTEGER         [default: 10]
  --levels INTEGER       [default: 7]
  --pattern TEXT         [default: *.jpg]
  -j, --workers INTEGER  [default: 1]
  --fov-w INTEGER        [default: 70] - Perspective FOV width (degrees)
  --fov-h INTEGER        [default: 70] - Perspective FOV height (degrees)
  --h-deg FLOAT          [default: 180] - Horizontal rotation
  --v-deg FLOAT          [default: 50] - Vertical rotation
```

#### `privacy_blur_infer.py`
```bash
Options:
  --input-dir DIRECTORY  [required]
  --output-dir PATH      [required]
  --face-model FILE      [default: models/privacy_blur/face_yolov8n.pt]
  --lp-model FILE        [default: models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx]
  --face-conf FLOAT      [default: 0.4]
  --lp-conf FLOAT        [default: 0.4]
  --face-iou FLOAT       [default: 0.5]
  --face-imgsz INTEGER   [default: 1024]
  --det-face-w INTEGER   [default: 1024]
  --p360-device TEXT     [default: auto]
  --blur-scope TEXT      [default: roi]
  --blur-backend TEXT    [default: gpu]
  --output-mode TEXT     [default: blur_only]
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

### Run Postprocess/Privacy Blur Without ComfyUI API

Use this mode when inpainting outputs already exist and you want to free ComfyUI VRAM before downstream stages.

```bash
# 1) Stop ComfyUI API containers to free VRAM
docker stop comfyui-container
# multi-GPU example: docker stop comfyui-g0 comfyui-g1

# 2) Postprocess only (example: gpu0, recommended -j 6)
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

# 3) Privacy blur only (host-side parallel runner)
SRC="/root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-postprocessed/gpu0-batch" \
OUT_ROOT="/root/comfyui-workflow-docker/comfyui_data/comfyui-g0/output-egoblur/gpu0-batch" \
FACE_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/face_yolov8n.pt" \
LP_MODEL="/root/comfyui-workflow-docker/models/privacy_blur/yolo-v9-s-608-license-plates-end2end.onnx" \
WORKERS=2 \
OUTPUT_MODE=blur_only \
PYTHON_BIN=/data/.venv/bin/python \
bash /root/comfyui-workflow-docker/inpainting-workflow-master/privacy_blur_parallel.sh

# 4) Verify privacy output count matches postprocess count
python3 - <<'PY'
from pathlib import Path
exts={'.png','.jpg','.jpeg','.webp','.tif','.tiff'}
base=Path('/root/comfyui-workflow-docker/comfyui_data/comfyui-g0')
p=sum(1 for x in (base/'output-postprocessed'/'gpu0-batch').rglob('*') if x.is_file() and x.suffix.lower() in exts)
e=sum(1 for x in (base/'output-egoblur'/'gpu0-batch').rglob('*') if x.is_file() and x.suffix.lower() in exts)
print(f'postprocess={p} privacy_blur={e} match={p==e}')
PY
```

Notes:
- `postprocess.py` with high workers can fail partially on GPU OOM; `-j 6` is a safer parallel setting.
- `privacy_blur_parallel.sh` defaults to `WORKERS=2`; increase carefully based on available VRAM.

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
- **Privacy Blur stack**: Ultralytics YOLO face + open-image-models YOLOv9 LP + pytorch360convert projection
- **p2e**: Perspective-to-equirectangular transformation
- **simple-lama-inpainting**: LAMA-based inpainting
- **SAM3**: Segment Anything Model 3 for object detection

### Volume Mounts
```yaml
- ./models/comfyui:/workspace/ComfyUI/models
- ./models/egoblur_gen2:/workspace/inpainting/models/egoblur_gen2
- ./models/privacy_blur:/workspace/models/privacy_blur
- ${COMFYUI_DATA_DIR}/input:/workspace/ComfyUI/input
- ${COMFYUI_DATA_DIR}/output:/workspace/ComfyUI/output
- ${COMFYUI_DATA_DIR}/output-postprocessed:/workspace/output-postprocessed
- ${COMFYUI_DATA_DIR}/output-egoblur:/workspace/output-egoblur
- ./workflow-updated.json:/workspace/workflow.json:ro  # currently based on carremoval-input-to-sky.json
```

---

## Performance Benchmarks

**Tested on NVIDIA L40S (45GB VRAM):**

| Stage | Images | Time | Speed |
|-------|--------|------|-------|
| ComfyUI Inpainting | 4 | ~50s | ~12s/image |
| Postprocessing | 6 | ~13s | ~2s/image |
| Privacy Blur | 6 | ~20s | ~3s/image |

**Total pipeline:** ~83s for 6 images (~14s per image)

---

## License

This project uses multiple open-source components:
- ComfyUI: GPL-3.0
- SAM3: Apache-2.0
- Privacy models/runtimes: follow upstream model licenses (Ultralytics + open-image-models)
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

# Process images (multi-GPU entry point)
SRC="/absolute/path/to/input_images" FINAL_OUTPUT_DIR="/absolute/path/to/final_outputs" RUN_NAME="multigpu-$(date +%Y%m%d_%H%M%S)" ./run_multi_gpu_pipeline.sh

# Stop
docker compose down
```
