# ===== Stage 1: Builder =====
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS builder

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
# Improvement 1: pin ComfyUI commit so upstream changes never silently break the build.
# Improvement 2: --fast-deps installs base ComfyUI deps; restore-snapshot then adds
#   custom node deps on top. These are complementary, not redundant:
#   --fast-deps = fast install of ComfyUI core; snapshot = custom node packages.
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

# Improvement 3: pipeline deps in a separate requirements file so this layer
# only re-runs when pipeline deps change, not when scripts or ComfyUI change.
WORKDIR /workspace
COPY pipeline-requirements.txt ./
RUN . /workspace/.venv/bin/activate && \
    uv pip install -r pipeline-requirements.txt

# Clone p2e standalone library (required by postprocess.py)
RUN git clone https://github.com/amanbagrecha/p2e.git /workspace/p2e-lib && \
    git -C /workspace/p2e-lib checkout ${P2E_COMMIT}

# Improvement 4: inpainting scripts are volume-mounted at runtime from the host
# (./inpainting-workflow-master:/workspace/inpainting in docker-compose.yml).
# We COPY them here as a fallback so the image works standalone (e.g. docker run
# without compose), but the host mount always takes precedence when present.
COPY inpainting-workflow-master /workspace/inpainting/

# ===== Stage 2: Runtime =====
# Lean image — build tools (git, build-essential, python3-dev, curl, uv) are NOT present.
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy built artifacts from builder — nothing else
COPY --from=builder /workspace/.venv        /workspace/.venv
COPY --from=builder /workspace/ComfyUI      /workspace/ComfyUI
COPY --from=builder /workspace/p2e-lib      /workspace/p2e-lib
COPY --from=builder /workspace/inpainting   /workspace/inpainting

ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace/p2e-lib:/workspace/ComfyUI"

RUN mkdir -p /workspace/input /workspace/output /workspace/models

EXPOSE 8188

WORKDIR /workspace
CMD ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8188"]
