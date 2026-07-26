#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 STAGE_ID SESSION_NAME [AUTH]" >&2
  exit 2
fi

stage_id="$1"
session_name="$2"
auth="${3:-oauth2}"

if [[ ! "$stage_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid stage id: $stage_id" >&2
  exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid session name: $session_name" >&2
  exit 2
fi
if [[ "$auth" != "oauth2" && "$auth" != "adc" ]]; then
  echo "invalid auth mode: $auth" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log_root="${HOME}/.cache/ecg-ramba-colab-cli/launcher_logs"
log_path="${log_root}/${stage_id}.log"
tmux_name="ecgr_${stage_id}"

mkdir -p "$log_root"
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for a durable background launcher." >&2
  exit 4
fi

if tmux has-session -t "$tmux_name" 2>/dev/null; then
  echo "stage already running: stage=${stage_id} tmux=${tmux_name} log=${log_path}"
  exit 3
fi

: > "$log_path"
command_text="$(
  printf 'export PATH=%q; cd %q; python3 scripts/colab_cli/pipeline.py --auth %q run --stage %q --session %q --keep --no-mount 2>&1 | tee -a %q' \
    "${HOME}/.local/bin:${PATH}" \
    "$repo_root" \
    "$auth" \
    "$stage_id" \
    "$session_name" \
    "$log_path"
)"
tmux new-session -d -s "$tmux_name" "$command_text"
pane_pid="$(tmux display-message -p -t "$tmux_name" '#{pane_pid}')"
echo "started: stage=${stage_id} tmux=${tmux_name} pid=${pane_pid} log=${log_path}"
