#!/bin/bash

# Model Download Script for ComfyUI Inpainting Pipeline
# Downloads all required models from S3 first, then falls back to HTTP

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

# Optional model root override for shared storage across multiple repo copies
MODELS_ROOT="${MODELS_ROOT:-models}"
MODELS_ROOT="${MODELS_ROOT%/}"
COMFY_MODELS_DIR="${COMFY_MODELS_DIR:-$MODELS_ROOT/comfyui}"
EGOBLUR_MODELS_DIR="${EGOBLUR_MODELS_DIR:-$MODELS_ROOT/egoblur_gen2}"
PRIVACY_MODELS_DIR="${PRIVACY_MODELS_DIR:-$MODELS_ROOT/privacy_blur}"
S3_MODELS_ROOT="${S3_MODELS_ROOT:-s3://panaromic-images/pano_models}"
S3_DOWNLOADS_ENABLED=0

relative_model_path() {
    local output_path="$1"
    local relative_path

    case "$output_path" in
        "$MODELS_ROOT"/*)
            relative_path="${output_path#"$MODELS_ROOT"/}"
            ;;
        *)
            relative_path="$output_path"
            ;;
    esac

    relative_path="${relative_path#./}"
    printf '%s\n' "$relative_path"
}

check_s3_models_root() {
    if ! command -v aws >/dev/null 2>&1; then
        echo "aws not found. Using HTTP fallback sources."
        return 1
    fi

    echo "Checking S3 model mirror..."
    if aws s3 ls "$S3_MODELS_ROOT" >/dev/null 2>&1; then
        echo "S3 model mirror available: $S3_MODELS_ROOT"
        return 0
    fi

    echo "S3 model mirror unavailable or unauthorized: $S3_MODELS_ROOT"
    echo "Using HTTP fallback sources."
    return 1
}

# Function to download a model if it doesn't exist
download_model() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    local relative_path=""
    local s3_url=""
    local downloaded=0

    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"

    if [ -f "$output_path" ]; then
        echo "✓ Already exists: $output_path"
        echo "  Skipping download."
    else
        echo "↓ Downloading: $description"
        echo "  Destination: $output_path"

        if [ "$S3_DOWNLOADS_ENABLED" = "1" ]; then
            relative_path="$(relative_model_path "$output_path")"
            s3_url="${S3_MODELS_ROOT%/}/$relative_path"
            echo "  Trying S3: $s3_url"
            if aws s3 cp --only-show-errors "$s3_url" "$output_path"; then
                downloaded=1
                echo "✓ Downloaded successfully from S3"
            else
                rm -f "$output_path"
                echo "  WARN: S3 download failed, falling back to HTTP source."
            fi
        fi

        if [ "$downloaded" -ne 1 ]; then
            echo "  Fallback URL: $url"
            wget --show-progress \
                 -O "$output_path" \
                 "$url"

            echo "✓ Downloaded successfully from HTTP source"
        fi
    fi
    echo ""
}

echo "Starting model downloads..."
echo "Existing models will be skipped."
if check_s3_models_root; then
    S3_DOWNLOADS_ENABLED=1
fi
echo ""

# 1. Text Encoder Model (~7B parameters, FP8)
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$COMFY_MODELS_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "[1/8] Qwen 2.5 VL Text Encoder (FP8)"

# 2. VAE Model
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$COMFY_MODELS_DIR/vae/qwen_image_vae.safetensors" \
    "[2/8] Qwen Image VAE"

# 3. LoRA Model (Lightning 4-step)
download_model \
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" \
    "$COMFY_MODELS_DIR/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" \
    "[3/8] Qwen Image Edit Lightning LoRA"

# 4. Upscale Model required by workflow-updated.json
download_model \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    "$COMFY_MODELS_DIR/upscale_models/RealESRGAN_x2plus.pth" \
    "[4/8] Real-ESRGAN x2plus Upscaler"

# 5. Diffusion Model (Qwen Image Edit FP8)
download_model \
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
    "$COMFY_MODELS_DIR/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
    "[5/8] Qwen Image Edit Diffusion Model (FP8)"

# 6. SAM3 HF Transformers format weights + config files
download_model \
    "https://huggingface.co/aravgarg588/comfyui-container-model/resolve/main/sam3/model.safetensors" \
    "$COMFY_MODELS_DIR/sam3/model.safetensors" \
    "[6/8] SAM3 Weights (HF transformers format)"

for _sam3_file in config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    download_model \
        "https://huggingface.co/aravgarg588/comfyui-container-model/resolve/main/sam3/$_sam3_file" \
        "$COMFY_MODELS_DIR/sam3/$_sam3_file" \
        "[6/8] SAM3 HF config: $_sam3_file"
done

# 6b. SimpleLama checkpoint (used by postprocess)
download_model \
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt" \
    "$COMFY_MODELS_DIR/lama/big-lama.pt" \
    "[6b/8] SimpleLama big-lama checkpoint"

# 7. Privacy blur face model (Ultralytics)
download_model \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
    "$PRIVACY_MODELS_DIR/face_yolov8n.pt" \
    "[7/8] Privacy blur face detector (YOLOv8n)"

# 8. Privacy blur LP model (open-image-models YOLOv9 ONNX)
download_model \
    "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx" \
    "$PRIVACY_MODELS_DIR/yolo-v9-s-608-license-plates-end2end.onnx" \
    "[8/8] Privacy blur LP detector (YOLOv9s-608 ONNX)"

# 9. Checkpoints (if needed)
# Uncomment if you have additional checkpoint models
# download_model \
#     "https://huggingface.co/your-repo/model.safetensors" \
#     "models/checkpoints/model.safetensors" \
#     "[optional] Additional Checkpoint Model"

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
echo "  │   ├── upscale_models/RealESRGAN_x2plus.pth"
echo "  │   ├── diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
echo "  │   └── sam3/"
echo "  │       ├── model.safetensors"
echo "  │       ├── config.json"
echo "  │       ├── processor_config.json"
echo "  │       ├── tokenizer.json + tokenizer_config.json"
echo "  │       ├── vocab.json + merges.txt"
echo "  │       └── special_tokens_map.json"
echo "  │   └── lama/big-lama.pt"
echo "  ├── privacy_blur/"
echo "  │   ├── face_yolov8n.pt"
echo "  │   └── yolo-v9-s-608-license-plates-end2end.onnx"
echo ""
echo "You can now start the Docker container:"
echo "  docker compose up -d"
echo ""
