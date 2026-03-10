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

# Install ComfyUI — pinned to the commit recorded in Comfy-Lock.yaml for reproducibility.
# --fast-deps installs base ComfyUI deps; restore-snapshot then adds custom node deps on top.
ARG COMFYUI_COMMIT=532e2850794c7b497174a0a42ac0cb1fe5b62499
RUN . .venv/bin/activate && \
    comfy --skip-prompt --workspace /workspace/ComfyUI install --nvidia --fast-deps && \
    git -C /workspace/ComfyUI fetch --depth 1 origin ${COMFYUI_COMMIT} && \
    git -C /workspace/ComfyUI checkout ${COMFYUI_COMMIT}

# Restore ComfyUI custom nodes from lock file for reproducible builds
WORKDIR /workspace/ComfyUI
COPY Comfy-Lock.yaml ./
RUN . /workspace/.venv/bin/activate && \
    comfy --workspace /workspace/ComfyUI node restore-snapshot \
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

# Install pipeline deps LAST — pipeline-requirements.txt pins torch+cu128 so it installs
# after ComfyUI/snapshot and becomes the definitive version. Covers:
#   sm_89  — L40S, RTX 6000 Ada (Ada Lovelace)
#   sm_120 — RTX 5090 (Blackwell)
WORKDIR /workspace
COPY pipeline-requirements.txt ./
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r pipeline-requirements.txt

# Remove pip-bundled cuDNN and NCCL — both are already provided by the
# cudnn-runtime base image. Saves ~1.4GB. torch finds them via LD_LIBRARY_PATH.
RUN . /workspace/.venv/bin/activate && \
    uv pip uninstall -y nvidia-cudnn-cu12 nvidia-nccl-cu12 || true

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
