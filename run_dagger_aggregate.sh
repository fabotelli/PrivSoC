#!/usr/bin/env bash
# Classic-DAgger aggregate train: original clean demos + solver-takeover
# recovery corrections in one training set.  Recovery-only was a dead end
# (11% -> 6% in-flight: no from-home coverage, so the policy degrades before
# it ever reaches the states the corrections fix); the staircase precedent
# trained DAgger-only ONLY because its episodes were complete from-home
# trajectories.  Ours are suffixes -> aggregate.
set -uo pipefail

cd ~/mujoco-test/mycobot_stack
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

echo "=== DAGGER AGGREGATE START $(date -u +%FT%TZ) ==="
[ -f dataset_dagger_recovery.npz ] || { echo "ABORT: no recovery dataset"; exit 1; }

bash gpu_claim.sh python3 train_bc_mycobot_stack.py \
    --data dataset_mycobot_stack.npz --extra-data dataset_dagger_recovery.npz \
    --out bc_dagger_mix.pt \
    --epochs 25 --lr 3e-4 --batch 256 --num-workers 8 --chunk 8 \
    --augment --ensemble-decay 0.01 > train_dagger.log 2>&1
echo "train rc=$? $(date -u +%FT%TZ)"; tail -12 train_dagger.log
[ -f bc_dagger_mix.pt ] || { echo "ABORT: no trained policy"; exit 1; }

TOP2=$(python3 - <<'PY'
import glob, json
rows = []
for p in glob.glob("bc_dagger_mix_eval100_epoch*.json"):
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

echo "=== DAGGER AGGREGATE DONE $(date -u +%FT%TZ) ==="
