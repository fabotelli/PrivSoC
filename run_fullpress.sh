#!/usr/bin/env bash
# FULL PRESS (directive 2026-09-07): two tracks to beat the 48.5% champion.
#   Track A (primary): v3 phase-consistent DAgger — collect state-triggered
#     solver takeovers -> aggregate train (clean + corrections, eval every 2
#     epochs) -> paired 200-ep eval of top-2.
#   Track B (hedge): +8000 more clean demos collected at low priority while
#     Track A trains; if A disappoints, scratch-train on 16k clean overnight.
set -uo pipefail

cd ~/mujoco-test/mycobot_stack
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

echo "=== FULL PRESS START $(date -u +%FT%TZ) ==="

# --- A1: collect v3 phase-consistent corrections (all cores) ---
if [ ! -f dataset_dagger_v3.npz ]; then
  python3 collect_dagger_mycobot_stack.py \
      --episodes 2500 --workers 24 --resolution 256 --rate 50 \
      --max-frames-per-episode 300 --successes-only --seed 50000 \
      --policy bc_mycobot_stack_epoch10.pt --ensemble-decay 0.01 \
      --trigger-radius 0.05 --max-policy-steps 200 \
      --out dataset_dagger_v3.npz > collect_dagger_v3.log 2>&1
  echo "v3 collect rc=$? $(date -u +%FT%TZ)"; tail -4 collect_dagger_v3.log
fi
[ -f dataset_dagger_v3.npz ] || { echo "ABORT: no v3 dataset"; exit 1; }

# --- B1: hedge clean collection, low priority, overlaps A2 ---
if [ ! -f dataset_clean2.npz ]; then
  nohup nice -n 10 python3 collect_dataset_mycobot_stack.py \
      --episodes 8000 --workers 16 --resolution 256 --rate 50 \
      --max-frames-per-episode 160 --successes-only --seed 40000 \
      --out dataset_clean2.npz > collect_clean2.log 2>&1 &
  echo "hedge collection launched (pid $!)"
fi

# --- A2: aggregate train, dense in-flight evals ---
rm -f bc_dagger_v3*.pt bc_dagger_v3_eval100_epoch*.json bc_dagger_v3_eval100_epoch*.log
bash gpu_claim.sh python3 train_bc_mycobot_stack.py \
    --data dataset_mycobot_stack.npz --extra-data dataset_dagger_v3.npz \
    --out bc_dagger_v3.pt --eval-every 2 \
    --epochs 25 --lr 3e-4 --batch 256 --num-workers 8 --chunk 8 \
    --augment --ensemble-decay 0.01 > train_dagger.log 2>&1
echo "train rc=$? $(date -u +%FT%TZ)"; tail -12 train_dagger.log
[ -f bc_dagger_v3.pt ] || { echo "ABORT: no trained policy"; exit 1; }

# --- A3: 200-ep paired eval of the top-2 by in-flight task score ---
TOP2=$(python3 - <<'PY'
import glob, json
rows = []
for p in glob.glob("bc_dagger_v3_eval100_epoch*.json"):
    try:
        d = json.load(open(p))
        rows.append((d["policy_summary"]["overall_success"], d["policy"]))
    except Exception:
        pass
rows.sort(reverse=True)
print("\n".join(ck for _, ck in rows[:2]))
PY
)
echo "top-2 by in-flight task score: $TOP2"
for CK in $TOP2; do
  TAG=$(basename "$CK" .pt)
  python3 eval_bc_mycobot_stack.py --policy "$CK" --episodes 200 \
      --start-seed 20000 --workers 24 --max-steps 280 --ensemble-decay 0.01 \
      --skip-solver --save-json "eval200_${TAG}.json" > "eval200_${TAG}.log" 2>&1
  echo "eval $CK rc=$?"; grep "overall success" "eval200_${TAG}.log"
done

echo "=== FULL PRESS TRACK A DONE $(date -u +%FT%TZ) ==="
