#!/usr/bin/env bash

setup_timestamped_log() {
  local log_file="$1"
  mkdir -p "$(dirname "$log_file")"
  exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$log_file") 2>&1
}

require_commands() {
  local cmd
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "ERROR: required command not found: $cmd"
      exit 1
    fi
  done
}

validate_flag() {
  local name="$1"
  local value="$2"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "ERROR: $name must be 0 or 1"
    exit 1
  fi
}

validate_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer"
    exit 1
  fi
}

validate_min_int() {
  local name="$1"
  local value="$2"
  local min="$3"
  validate_non_negative_int "$name" "$value"
  if (( value < min )); then
    echo "ERROR: $name must be >= $min"
    exit 1
  fi
}

resolve_gpu_ids() {
  local raw="$1"
  if [[ "$raw" != "auto" ]]; then
    tr ', ' '\n\n' <<<"$raw" | awk 'NF'
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | awk 'NF'
}

required_model_paths() {
  local comfy_dir="$1"
  local privacy_dir="$2"
  local lama_model="$3"
  local face_model="$4"
  local lp_model="$5"

  cat <<EOF
$comfy_dir/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
$comfy_dir/vae/qwen_image_vae.safetensors
$comfy_dir/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors
$comfy_dir/upscale_models/RealESRGAN_x2plus.pth
$comfy_dir/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors
$comfy_dir/sam3/model.safetensors
$comfy_dir/sam3/config.json
$comfy_dir/sam3/processor_config.json
$comfy_dir/sam3/special_tokens_map.json
$comfy_dir/sam3/tokenizer.json
$comfy_dir/sam3/tokenizer_config.json
$comfy_dir/sam3/vocab.json
$comfy_dir/sam3/merges.txt
$lama_model
$face_model
$lp_model
EOF
}

stage_output_dir() {
  local data_dir="$1"
  local batch="$2"
  local stage="$3"

  case "$stage" in
    egoblur) printf '%s\n' "$data_dir/output-egoblur/$batch" ;;
    sam3) printf '%s\n' "$data_dir/output-sam3-mask/$batch" ;;
    postprocess) printf '%s\n' "$data_dir/output-postprocessed/$batch" ;;
    inpainting) printf '%s\n' "$data_dir/output/$batch" ;;
    *)
      echo "ERROR: unknown stage: $stage" >&2
      return 1
      ;;
  esac
}
