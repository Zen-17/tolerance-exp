#!/usr/bin/env bash
# Detached tmux session for the phase1 chain. SSH disconnect does not kill it.
#   bash scripts/tmux_phase1.sh          # create or attach
#   tmux attach -t phase1                # re-attach after SSH reconnect
#   Ctrl-b d   or  Ctrl-a d              # detach (job keeps running)
set -euo pipefail
SESSION=phase1
LOG=/opt/data/data/tolerance-exp/results/phase1_chain.log
ROOT=/opt/data/data/tolerance-exp

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed" >&2
  exit 1
fi

mkdir -p "$ROOT/results"
touch "$LOG"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists; attaching"
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n log "tail -n 80 -F '$LOG'"
tmux new-window -t "$SESSION:" -n gpu "watch -n 5 nvidia-smi"
tmux new-window -t "$SESSION:" -n procs \
  "watch -n 5 'pgrep -a -f \"run_exp_|run_phase1_chain\" || echo none'"
tmux select-window -t "$SESSION:log"
echo "created tmux session '$SESSION' (detached). attach with: tmux attach -t $SESSION"
exec tmux attach -t "$SESSION"
