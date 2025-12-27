#!/bin/bash

# ComfyUI Docker Setup Script
# This script automates the build and deployment of your ComfyUI Docker container

set -e

echo "========================================"
echo "ComfyUI Docker Setup"
echo "========================================"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: docker-compose is not installed."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA Docker runtime may not be properly configured."
    echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Building ComfyUI Docker image..."
echo "This may take 10-20 minutes on first build..."
echo

# Build using docker-compose
if command -v docker-compose &> /dev/null; then
    docker-compose build
else
    docker compose build
fi

echo
echo "========================================"
echo "Build complete!"
echo "========================================"
echo
echo "To start ComfyUI, run:"
echo "  docker-compose up -d"
echo
echo "To view logs:"
echo "  docker logs -f comfyui-container"
echo
echo "To access ComfyUI:"
echo "  http://localhost:8188"
echo
echo "To stop ComfyUI:"
echo "  docker-compose down"
echo

read -p "Start ComfyUI now? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Setup complete. Start manually with: docker-compose up -d"
    exit 0
fi

# Start the container
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

echo
echo "ComfyUI is starting..."
echo "Waiting for initialization (this may take 30-60 seconds)..."
sleep 10

# Show logs
echo
echo "Recent logs:"
docker logs --tail 20 comfyui-container

echo
echo "========================================"
echo "ComfyUI is running!"
echo "========================================"
echo
echo "Access at: http://localhost:8188"
echo "View logs: docker logs -f comfyui-container"
echo
