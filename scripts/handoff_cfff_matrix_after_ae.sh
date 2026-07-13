#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-running shared AE, then replace the paused legacy
# scheduler with the current dependency graph. Run this script inside tmux so
# the handoff survives SSH and Codex disconnections.

BASE="${CHATPATHWAY_ASSET_ROOT:-/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui}"
REPO="${CHATPATHWAY_REPO:-$BASE/ChatPathway2}"
PYTHON="${CHATPATHWAY_PYTHON:-$REPO/.venv/bin/python}"
SEED="${CHATPATHWAY_SEED:-20260711}"
GPUS="${CHATPATHWAY_GPUS:-0,1,2,3}"
POLL_SECONDS="${CHATPATHWAY_HANDOFF_POLL_SECONDS:-60}"
OLD_SESSION="${CHATPATHWAY_OLD_SCHEDULER_SESSION:-cpath_core_20260711}"
NEW_SESSION="${CHATPATHWAY_NEW_SCHEDULER_SESSION:-cpath_core_ddp_20260713}"
RUN_NAME="${CHATPATHWAY_SCHEDULER_RUN_NAME:-core_matrix_ddp_20260713}"

AE_ROOT="$BASE/checkpoints/seeds/$SEED/shared/pathway_reconstruction_ae"
AE_COMPLETE="$AE_ROOT/run_complete.json"
LOG_DIR="$BASE/runs/cfff_matrix_scheduler/$RUN_NAME"
HANDOFF_LOG="$LOG_DIR/handoff.log"
SCHEDULER_LOG="$LOG_DIR/scheduler.log"

mkdir -p "$LOG_DIR"
exec >>"$HANDOFF_LOG" 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "$(timestamp) handoff_wait_started repo_commit=$(git -C "$REPO" rev-parse HEAD) ae_complete=$AE_COMPLETE"
while [[ ! -f "$AE_COMPLETE" ]]; do
  sleep "$POLL_SECONDS"
done
echo "$(timestamp) ae_run_complete_detected"

# run_complete.json is written at the end of training, just before process
# teardown. Do not destroy the old tmux session until that child has exited.
while pgrep -f "method.training.latent_ae.*pathway_reconstruction_ae" >/dev/null; do
  sleep 5
done
echo "$(timestamp) ae_process_exited"

if tmux has-session -t "$OLD_SESSION" 2>/dev/null; then
  tmux kill-session -t "$OLD_SESSION"
  echo "$(timestamp) old_scheduler_session_closed session=$OLD_SESSION"
fi

if tmux has-session -t "$NEW_SESSION" 2>/dev/null; then
  echo "$(timestamp) new_scheduler_already_running session=$NEW_SESSION"
  exit 0
fi

tmux new-session -d -s "$NEW_SESSION" -c "$REPO" \
  "env CHATPATHWAY_PROFILE=cfff '$PYTHON' -u -m experiments.run_cfff_matrix --seeds '$SEED' --gpus '$GPUS' --log-dir '$LOG_DIR' --poll-seconds 5 > '$SCHEDULER_LOG' 2>&1"
echo "$(timestamp) new_scheduler_started session=$NEW_SESSION log=$SCHEDULER_LOG"
