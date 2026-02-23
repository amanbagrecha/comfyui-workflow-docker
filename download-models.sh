#!/bin/bash

# Model Download Script for ComfyUI Inpainting Pipeline
# Downloads all required models using wget

set -e  # Exit on error

echo "======================================"
echo "ComfyUI Model Download Script"
echo "======================================"
echo ""

# Check if wget is available
if ! command -v wget &> /dev/null; then
    echo "ERROR: 'wget' command not found."
    echo "Please install wget first:"
    echo "  sudo apt-get install wget"
    exit 1
fi

# Optional: Set HuggingFace API token for gated models
# You can set this as an environment variable: export HF_TOKEN=your_token_here
HF_TOKEN="${HF_TOKEN:-}"

# Optional model root override for shared storage across multiple repo copies
MODELS_ROOT="${MODELS_ROOT:-models}"
COMFY_MODELS_DIR="${COMFY_MODELS_DIR:-$MODELS_ROOT/comfyui}"
EGOBLUR_MODELS_DIR="${EGOBLUR_MODELS_DIR:-$MODELS_ROOT/egoblur_gen2}"

# Function to download a model if it doesn't exist
download_model() {
    local url="$1"
    local output_path="$2"
    local description="$3"

    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"

    if [ -f "$output_path" ]; then
        echo "✓ Already exists: $output_path"
        echo "  Skipping download."
    else
        echo "↓ Downloading: $description"
        echo "  URL: $url"
        echo "  Destination: $output_path"

        # Download with wget
        if [ -n "$HF_TOKEN" ]; then
            wget --header="Authorization: Bearer $HF_TOKEN" \
                 --show-progress \
                 -O "$output_path" \
                 "$url"
        else
            wget --show-progress \
                 -O "$output_path" \
                 "$url"
        fi

        echo "✓ Downloaded successfully"
    fi
    echo ""
}

echo "Starting model downloads..."
echo "Existing models will be skipped."
echo ""

# 1. Text Encoder Model (~7B parameters, FP8)
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$COMFY_MODELS_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "[1/9] Qwen 2.5 VL Text Encoder (FP8)"

# 2. VAE Model
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$COMFY_MODELS_DIR/vae/qwen_image_vae.safetensors" \
    "[2/9] Qwen Image VAE"

# 3. LoRA Model (Lightning 4-step)
download_model \
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" \
    "$COMFY_MODELS_DIR/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" \
    "[3/9] Qwen Image Edit Lightning LoRA"

# 4. Upscale Model (Real-ESRGAN x2)
download_model \
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/a86fc6182b4650b4459cb1ddcb0a0d1ec86bf3b0/RealESRGAN_x2.pth" \
    "$COMFY_MODELS_DIR/upscale_models/RealESRGAN_x2.pth" \
    "[4/9] Real-ESRGAN x2 Upscaler"

# 4b. Upscale Model required by workflow-updated.json
download_model \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    "$COMFY_MODELS_DIR/upscale_models/RealESRGAN_x2plus.pth" \
    "[4b/9] Real-ESRGAN x2plus Upscaler"

# 5. Diffusion Model (Qwen Image Edit FP8)
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
    "$COMFY_MODELS_DIR/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
    "[5/9] Qwen Image Edit Diffusion Model (FP8)"

# 6. SAM3 Model (Segment Anything Model 3)
download_model \
    "https://huggingface.co/aravgarg588/comfyui-container-model/resolve/main/sam3/sam3.pt" \
    "$COMFY_MODELS_DIR/sam3/sam3.pt" \
    "[6/9] SAM3 Segmentation Model"

# 7. EgoBlur Face Detection Model
download_model \
    "https://huggingface.co/aravgarg588/comfyui-container-model/resolve/main/egoblur_gen2/ego_blur_face_gen2.jit" \
    "$EGOBLUR_MODELS_DIR/ego_blur_face_gen2.jit" \
    "[7/9] EgoBlur Face Detection Model"

# 8. EgoBlur License Plate Detection Model
download_model \
    "https://huggingface.co/aravgarg588/comfyui-container-model/resolve/main/egoblur_gen2/ego_blur_lp_gen2.jit" \
    "$EGOBLUR_MODELS_DIR/ego_blur_lp_gen2.jit" \
    "[8/9] EgoBlur License Plate Detection Model"

# 9. Checkpoints (if needed)
# Uncomment if you have additional checkpoint models
# download_model \
#     "https://huggingface.co/your-repo/model.safetensors" \
#     "models/checkpoints/model.safetensors" \
#     "[9/9] Additional Checkpoint Model"

echo "======================================"
echo "✓ Model download complete!"
echo "======================================"
echo ""
echo "Downloaded models structure:"
echo "  models/"
echo "  ├── comfyui/"
echo "  │   ├── text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
echo "  │   ├── vae/qwen_image_vae.safetensors"
echo "  │   ├── loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
echo "  │   ├── upscale_models/RealESRGAN_x2.pth"
echo "  │   ├── upscale_models/RealESRGAN_x2plus.pth"
echo "  │   ├── diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
echo "  │   └── sam3/sam3.pt"
echo "  └── egoblur_gen2/"
echo "      ├── ego_blur_face_gen2.jit"
echo "      └── ego_blur_lp_gen2.jit"
echo ""
echo "You can now start the Docker container:"
echo "  docker compose up -d"
echo ""
