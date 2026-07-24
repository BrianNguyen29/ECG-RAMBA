#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Google Colab CLI is supported on Linux/macOS. Run this script inside WSL2." >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install uv." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv tool install --upgrade google-colab-cli

echo "Colab CLI installation:"
command -v colab
colab version
echo
echo "Next, configure ADC once with the Colab-required scopes:"
echo "  bash scripts/colab_cli/setup_adc.sh"
