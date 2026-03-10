# ===== Stage 1: Builder =====
# CUDA 12.8 base required for Blackwell (sm_120 / RTX 5090) support.
# Also covers Ada Lovelace (sm_89 / L40S, RTX 6000 Ada) and all older archs.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Create virtual environment and install pip inside it
RUN uv venv .venv
RUN . .venv/bin/activate && uv pip install pip

# Install comfy-cli
RUN . .venv/bin/activate && uv pip install comfy-cli

COPY pipeline-requirements.txt ./

# Seed the pinned torch stack before the ComfyUI bootstrap so later installers
# reuse the same CUDA wheels instead of resolving their own torch variant.
RUN python3 - <<'PY'
from pathlib import Path

src = Path('/workspace/pipeline-requirements.txt')
keep = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('--extra-index-url') or line.startswith('torch==') or line.startswith('torchvision==') or line.startswith('torchaudio=='):
        keep.append(line)

Path('/tmp/torch-bootstrap-requirements.txt').write_text('\n'.join(keep) + '\n')
PY
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r /tmp/torch-bootstrap-requirements.txt

# Clone ComfyUI at a pinned commit; core Python deps are installed separately so
# torch stays owned by pipeline-requirements.txt.
ARG COMFYUI_COMMIT=532e2850794c7b497174a0a42ac0cb1fe5b62499
ARG COMFYUI_MANAGER_COMMIT=2478d20e76aeb2f42a6f372029e417201ef927b3
RUN . .venv/bin/activate && \
    git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI && \
    git -C /workspace/ComfyUI fetch --depth 1 origin ${COMFYUI_COMMIT} && \
    git -C /workspace/ComfyUI checkout ${COMFYUI_COMMIT}
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager /workspace/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git -C /workspace/ComfyUI/custom_nodes/ComfyUI-Manager fetch --depth 1 origin ${COMFYUI_MANAGER_COMMIT} && \
    git -C /workspace/ComfyUI/custom_nodes/ComfyUI-Manager checkout ${COMFYUI_MANAGER_COMMIT}
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Install ComfyUI core deps without re-installing torch/torchaudio/torchvision.
RUN python3 - <<'PY'
from pathlib import Path

src = Path('/workspace/ComfyUI/requirements.txt')
skip = {'torch', 'torchaudio', 'torchvision'}
keep = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    name = line.split('==', 1)[0].split('>=', 1)[0].split('~=', 1)[0].strip().lower()
    if name in skip:
        continue
    keep.append(line)

Path('/tmp/comfy-core-requirements.txt').write_text('\n'.join(keep) + '\n')
PY
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r /tmp/comfy-core-requirements.txt

# Restore ComfyUI custom nodes from lock file for reproducible builds
WORKDIR /workspace/ComfyUI
COPY Comfy-Lock.yaml ./
RUN . /workspace/.venv/bin/activate && \
    comfy --skip-prompt --workspace /workspace/ComfyUI node restore-snapshot \
        --pip-non-url --pip-non-local-url \
        Comfy-Lock.yaml

# Pin p2e custom node to known-good commit
ARG P2E_COMMIT=1968f6aed36b300be3599ea5053e0206aaa5704b
RUN if [ -d /workspace/ComfyUI/custom_nodes/p2e ]; then \
      git -C /workspace/ComfyUI/custom_nodes/p2e fetch --depth 1 origin ${P2E_COMMIT} && \
      git -C /workspace/ComfyUI/custom_nodes/p2e checkout ${P2E_COMMIT}; \
    fi

# Clone p2e standalone library (required by postprocess.py)
RUN git clone https://github.com/amanbagrecha/p2e.git /workspace/p2e-lib && \
    git -C /workspace/p2e-lib checkout ${P2E_COMMIT}

# Install the full pipeline deps after ComfyUI/snapshot to add pipeline-only
# packages while keeping the pinned torch stack aligned across all installers.
# Covers:
#   sm_89  — L40S, RTX 6000 Ada (Ada Lovelace)
#   sm_120 — RTX 5090 (Blackwell)
WORKDIR /workspace
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r pipeline-requirements.txt

# Install packages that otherwise pull conflicting OpenCV wheel names; their
# shared runtime deps are installed via pipeline-requirements.txt above.
RUN . /workspace/.venv/bin/activate && \
    uv pip install --no-deps \
        simple-lama-inpainting==0.1.0 \
        ultralytics==8.4.21 \
        open-image-models==0.5.1

# Keep a single OpenCV wheel in the final environment to avoid cv2 file overlap
# between contrib/gui/headless builds pulled by different dependencies.
RUN . /workspace/.venv/bin/activate && \
    uv pip uninstall opencv-python opencv-python-headless || true && \
    uv pip install --no-deps opencv-contrib-python==4.12.0.88

# Remove pip-bundled cuDNN only. Keep the pip NCCL wheel because the system
# NCCL in the base image is too old for the current torch build.
RUN . /workspace/.venv/bin/activate && \
    uv pip uninstall nvidia-cudnn-cu12 || true

# Inpainting scripts are volume-mounted at runtime from the host
# (./inpainting-workflow-master:/workspace/inpainting in docker-compose.yml).
# Copied here as fallback so the image works standalone without compose.
COPY inpainting-workflow-master /workspace/inpainting/

# ===== Stage 2: Runtime =====
# CUDA 12.8 base to match builder — required for Blackwell/Ada GPU support.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy built artifacts from builder — nothing else
COPY --from=builder /workspace/.venv        /workspace/.venv
COPY --from=builder /workspace/ComfyUI      /workspace/ComfyUI
COPY --from=builder /workspace/p2e-lib      /workspace/p2e-lib
COPY --from=builder /workspace/inpainting   /workspace/inpainting
COPY --from=builder /root/.config           /root/.config

ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace/p2e-lib:/workspace/ComfyUI"
# Allow torch to find system cuDNN/NCCL (replaces pip-bundled nvidia-cudnn-cu12, nvidia-nccl-cu12)
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

RUN mkdir -p /workspace/input /workspace/output /workspace/models

EXPOSE 8188

WORKDIR /workspace
CMD ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8188"]
