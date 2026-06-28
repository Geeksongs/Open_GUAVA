#!/usr/bin/env bash
# Tiny launcher for a GUAVA run with sensible defaults and auto-managed servers.
#
# Usage:
#   bash guava/run.sh                      # lift task, gemini, viz, 1 trial
#   bash guava/run.sh stack                # stack task
#   bash guava/run.sh lift 10              # lift, 10 trials
#   bash guava/run.sh stack 5 minimax/minimax-m3   # task, trials, model
#
# Everything else (proxy, PyRoKi, SAM3/GraspNet) is started and cleaned up
# automatically by the runner. Run from the repo root in the `capx` conda env.

set -euo pipefail
cd "$(dirname "$0")/.."

TASK_KEY="${1:-lift}"
TRIALS="${2:-1}"
MODEL="${3:-google/gemini-2.5-flash-lite}"

case "$TASK_KEY" in
  lift)
    ENV="franka_robosuite_cube_lift_low_level"
    TASK="Pick up the red cube and lift it."
    MAXT=8 ;;
  stack)
    ENV="franka_robosuite_cubes_low_level"
    TASK="Pick up the red cube and stack it on top of the green cube, then release."
    MAXT=12 ;;
  restack)
    ENV="franka_robosuite_cubes_restack_low_level"
    TASK="Restack the cubes."
    MAXT=15 ;;
  *)
    echo "unknown task key '$TASK_KEY' (use: lift | stack | restack)"; exit 1 ;;
esac

DISPLAY="${DISPLAY:-:0}" PYTHONPATH=. python guava/runner.py \
  --env "$ENV" \
  --task "$TASK" \
  --model "$MODEL" \
  --segmenter sam3 --serial-gpu --viz \
  --trials "$TRIALS" --max-turns "$MAXT"
