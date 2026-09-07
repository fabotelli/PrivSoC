#!/usr/bin/env bash
# One-shot driver for the myCobot two-cube stacking pipeline (PrivSoC):
#   collect 8000 successes-only episodes -> train 25 epochs -> finalize
#   (200-ep diagnostic eval + demos + session notes + copy deliverables).
# Sequential so finalize's "is training still running?" guard sees a clean slate.
# Launch under nohup/setsid so it survives SSH drop.
set -uo pipefail

cd ~/mujoco-test/mycobot_stack

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

NPROC=$(nproc)
COLLECT_WORKERS=${COLLECT_WORKERS:-24}
EPISODES=${EPISODES:-8000}

echo "=== PIPELINE START $(date -u +%Y-%m-%dT%H:%M:%SZ)  nproc=$NPROC collect_workers=$COLLECT_WORKERS episodes=$EPISODES ==="

# --- 1) COLLECT ----------------------------------------------------------- #
if [ -f dataset_mycobot_stack.npz ]; then
  echo "dataset_mycobot_stack.npz already exists — skipping collection"
else
  echo "=== COLLECT ($EPISODES eps, $COLLECT_WORKERS workers, 256px, successes-only) ==="
  python3 collect_dataset_mycobot_stack.py \
      --episodes "$EPISODES" --workers "$COLLECT_WORKERS" \
      --resolution 256 --rate 50 --max-frames-per-episode 160 \
      --successes-only --out dataset_mycobot_stack.npz \
      > collect_mycobot_stack.log 2>&1
  COLLECT_RC=$?
  echo "collect exited rc=$COLLECT_RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tail -8 collect_mycobot_stack.log
fi

if [ ! -f dataset_mycobot_stack.npz ]; then
  echo "no dataset_mycobot_stack.npz after collection — abort"; exit 1
fi

# --- 2) TRAIN ------------------------------------------------------------- #
echo "=== TRAIN (25 epochs, lr 3e-4, batch 256, workers 8, chunk 8, augment, decay 0.01) ==="
bash gpu_claim.sh python3 train_bc_mycobot_stack.py \
    --data dataset_mycobot_stack.npz \
    --out bc_mycobot_stack.pt \
    --epochs 25 --lr 3e-4 --batch 256 \
    --num-workers 8 --chunk 8 --augment \
    --ensemble-decay 0.01 \
    > train_mycobot_stack.log 2>&1
TRAIN_RC=$?
echo "train exited rc=$TRAIN_RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail -30 train_mycobot_stack.log

if [ ! -f bc_mycobot_stack.pt ]; then
  echo "no bc_mycobot_stack.pt after training — abort before finalize"; exit 1
fi

# --- 3) FINALIZE ---------------------------------------------------------- #
echo "=== FINALIZE (200-ep diagnostic eval seed 20000 + demo mp4s + session notes) ==="
bash finalize.sh
echo "finalize exited rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
