FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    git \
    wget \
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

# Set working directory
WORKDIR /workspace

# Create virtual environment with uv
RUN uv venv .venv

# Install pip in the uv environment so comfy can use it for custom nodes
RUN . .venv/bin/activate && uv pip install pip

# Install comfy-cli first
RUN . .venv/bin/activate && uv pip install comfy-cli

# Install ComfyUI with NVIDIA support using comfy CLI
RUN . .venv/bin/activate && \
    comfy --skip-prompt --workspace /workspace/ComfyUI install --nvidia --fast-deps

# Copy lock file and restore from snapshot for reproducible builds
WORKDIR /workspace/ComfyUI
COPY Comfy-Lock.yaml ./
RUN . /workspace/.venv/bin/activate && \
    echo "Restoring from Comfy-Lock.yaml for reproducible build..." && \
    comfy --workspace /workspace/ComfyUI node restore-snapshot \
        --pip-non-url --pip-non-local-url \
        Comfy-Lock.yaml && \
    echo "Snapshot restored successfully!"

# Pin p2e custom node to a known-good commit that includes "P2E And Blend"
ARG P2E_COMMIT=b06b31072d13afeb323ecff364a869377c631568
RUN if [ -d /workspace/ComfyUI/custom_nodes/p2e ]; then \
      git -C /workspace/ComfyUI/custom_nodes/p2e fetch --depth 1 origin ${P2E_COMMIT} && \
      git -C /workspace/ComfyUI/custom_nodes/p2e checkout ${P2E_COMMIT}; \
    fi

# Install any missing dependencies that custom nodes need
RUN . /workspace/.venv/bin/activate && \
    uv pip install numba opencv-contrib-python opencv-python

# Install inpainting workflow dependencies
RUN . /workspace/.venv/bin/activate && \
    uv pip install simple-lama-inpainting click requests tqdm egoblur

# Clone EgoBlur Gen2 library (CRITICAL: required by egoblur_infer.py)
ARG EGOBLUR_REF=main
RUN git clone --depth 1 --branch ${EGOBLUR_REF} https://github.com/facebookresearch/EgoBlur.git /workspace/EgoBlur

# Clone p2e standalone library (required by postprocess.py for perspective-to-equirectangular)
RUN git clone https://github.com/amanbagrecha/p2e.git /workspace/p2e-lib && \
    git -C /workspace/p2e-lib checkout ${P2E_COMMIT}

# Copy inpainting workflow scripts and data
COPY inpainting-workflow-master /workspace/inpainting/

# Create symlink for /data path compatibility
RUN ln -s /workspace/inpainting /data

# Set environment to use the venv by default
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace/EgoBlur:/workspace/p2e-lib:/workspace/ComfyUI:$PYTHONPATH"

# Create necessary directories
RUN mkdir -p /workspace/input /workspace/output /workspace/models

# Expose ComfyUI port
EXPOSE 8188

# Default command to launch ComfyUI
WORKDIR /workspace
CMD ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8188"]
