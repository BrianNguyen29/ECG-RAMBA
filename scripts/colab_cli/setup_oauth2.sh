#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v colab >/dev/null 2>&1; then
  echo "colab is not installed. Run scripts/colab_cli/install_wsl.sh first." >&2
  exit 2
fi

cat <<'EOF'
Colab CLI OAuth2 setup

1. Open the authorization URL printed below in any browser.
2. Approve the requested Google/Colab scopes.
3. Copy the authorization code shown by Google.
4. Paste the code back into this terminal.

The refresh token is stored by Colab CLI under ~/.config/colab-cli/token.json.
No gcloud credential is required.
EOF

colab --auth=oauth2 whoami
