#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
. "$REPO/scripts/shell_helpers.sh"

export PATH="$HOME/.opencode/bin:$PATH"

INSTALL_AWS_CLI="${INSTALL_AWS_CLI:-1}"
INSTALL_OPENCODE="${INSTALL_OPENCODE:-1}"
DOWNLOAD_SIZE_ONLY="${DOWNLOAD_SIZE_ONLY:-0}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: missing env var $name"
    exit 1
  fi
}

linux_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'x86_64\n' ;;
    aarch64|arm64) printf 'aarch64\n' ;;
    *)
      echo "ERROR: unsupported architecture $(uname -m)"
      exit 1
      ;;
  esac
}

ensure_command() {
  local cmd="$1"
  local pkg="$2"

  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
  fi

  require_commands "$cmd"
}

install_aws_cli() {
  local arch tmp_dir install_args

  if [ "$INSTALL_AWS_CLI" != "1" ]; then
    return 0
  fi

  if [ -x "/usr/local/bin/aws" ]; then
    return 0
  fi

  ensure_command curl curl
  ensure_command unzip unzip
  arch="$(linux_arch)"
  tmp_dir="$(mktemp -d)"
  install_args=(--bin-dir "/usr/local/bin" --install-dir "/usr/local/aws-cli")

  if [ -d "/usr/local/aws-cli" ]; then
    install_args+=(--update)
  fi

  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o "$tmp_dir/awscliv2.zip"
  unzip -q "$tmp_dir/awscliv2.zip" -d "$tmp_dir"
  "$tmp_dir/aws/install" "${install_args[@]}"
  rm -rf "$tmp_dir"

  require_commands aws
}

install_opencode() {
  if [ "$INSTALL_OPENCODE" != "1" ]; then
    return 0
  fi

  if [ ! -x "$HOME/.opencode/bin/opencode" ]; then
    ensure_command curl curl
    curl -fsSL https://opencode.ai/install | bash
  fi

  ln -sf "$HOME/.opencode/bin/opencode" /usr/local/bin/opencode
  require_commands opencode
}

configure_profile() {
  local prefix="$1"
  local default_profile="$2"
  local profile_var="${prefix}_PROFILE"
  local access_var="${prefix}_ACCESS_KEY_ID"
  local secret_var="${prefix}_SECRET_ACCESS_KEY"
  local region_var="${prefix}_REGION"
  local endpoint_var="${prefix}_ENDPOINT_URL"
  local session_var="${prefix}_SESSION_TOKEN"
  local profile

  if [ -z "${!access_var:-}" ] && [ -z "${!secret_var:-}" ] && [ -z "${!region_var:-}" ] && [ -z "${!endpoint_var:-}" ]; then
    return 0
  fi

  require_commands aws
  require_env "$access_var"
  require_env "$secret_var"
  require_env "$region_var"

  profile="${!profile_var:-$default_profile}"
  aws configure set aws_access_key_id "${!access_var}" --profile "$profile"
  aws configure set aws_secret_access_key "${!secret_var}" --profile "$profile"
  aws configure set region "${!region_var}" --profile "$profile"
  aws configure set output json --profile "$profile"

  if [ -n "${!session_var:-}" ]; then
    aws configure set aws_session_token "${!session_var}" --profile "$profile"
  fi

  if [ -n "${!endpoint_var:-}" ]; then
    aws configure set endpoint_url "${!endpoint_var}" --profile "$profile"
  fi
}

download_data() {
  local profile sync_args

  if [ -z "${DOWNLOAD_S3_URI:-}" ]; then
    return 0
  fi

  require_commands aws
  require_env DOWNLOAD_DEST_DIR

  profile="${AWS_DOWNLOAD_PROFILE:-default}"
  sync_args=(s3 sync "$DOWNLOAD_S3_URI" "$DOWNLOAD_DEST_DIR" --profile "$profile" --no-progress)

  if [ -n "${AWS_DOWNLOAD_ENDPOINT_URL:-}" ]; then
    sync_args+=(--endpoint-url "$AWS_DOWNLOAD_ENDPOINT_URL")
  fi

  if [ "$DOWNLOAD_SIZE_ONLY" = "1" ]; then
    sync_args+=(--size-only)
  fi

  mkdir -p "$DOWNLOAD_DEST_DIR"
  aws "${sync_args[@]}"
}

run_pipeline() {
  if [ "$RUN_PIPELINE" != "1" ]; then
    return 0
  fi

  require_env SRC
  SKIP_HOST_BOOTSTRAP="${SKIP_HOST_BOOTSTRAP:-1}" bash "$REPO/run_multi_gpu_pipeline.sh"
}

main() {
  install_aws_cli
  configure_profile AWS_UPLOAD upload
  configure_profile AWS_DOWNLOAD download
  install_opencode
  download_data
  run_pipeline
}

main "$@"
