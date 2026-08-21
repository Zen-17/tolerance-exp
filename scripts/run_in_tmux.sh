#!/usr/bin/env bash
# Run a command inside a detached tmux session so SSH hangup does not stop it.
# Usage:
#   bash scripts/run_in_tmux.sh <session-name> -- <command> [args...]
# Example:
#   bash scripts/run_in_tmux.sh phase1 -- bash /opt/data/data/tolerance-exp/run_phase1_chain.sh
set -euo pipefail
if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: $0 <session-name> -- <command> [args...]" >&2
  exit 2
fi
SESSION=$1
shift 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists; refuse to start a second copy" >&2
  echo "attach: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n job "$@"
echo "started in tmux session '$SESSION'. attach: tmux attach -t $SESSION"
echo "detach: Ctrl-b d  (or Ctrl-a d)"
