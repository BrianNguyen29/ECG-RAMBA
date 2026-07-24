#!/usr/bin/env bash
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required for headless Colab CLI authentication." >&2
  exit 2
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v colab >/dev/null 2>&1; then
  echo "colab is not installed. Run scripts/colab_cli/install_wsl.sh first." >&2
  exit 2
fi

gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

echo
echo "Colab CLI ADC identity and scopes:"
colab --auth=adc whoami
