#!/usr/bin/env bash
# Salvage retrain after the recorded-prefix flaw (see trim_dagger_prefix.py):
# trim the collected DAgger set to solver-takeover suffixes -> retrain from
# scratch -> 200-ep paired eval of the top-2 checkpoints by in-flight score.
set -uo pipefail

cd ~/mujoco-test/mycobot_stack
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

echo "=== DAGGER RETRAIN START $(date -u +%FT%TZ) ==="

# --- 1: trim to recovery suffixes ---
if [ ! -f dataset_dagger_recovery.npz ]; then
  python3 trim_dagger_prefix.py --data dataset_dagger_mycobot.npz \
      --out dataset_dagger_recovery.npz > trim_dagger.log 2>&1
  echo "trim rc=$? $(date -u +%FT%TZ)"; tail -4 trim_dagger.log
fi
[ -f dataset_dagger_recovery.npz ] || { echo "ABORT: no recovery dataset"; exit 1; }

# --- 2: clear the poisoned run's checkpoints, retrain from scratch ---
rm -f bc_dagger_mycobot*.pt bc_dagger_mycobot_eval100_epoch*.json \
      bc_dagger_mycobot_eval100_epoch*.log bc_dagger_mycobot_inflight_eval.log
bash gpu_claim.sh python3 train_bc_mycobot_stack.py \
    --data dataset_dagger_recovery.npz --out bc_dagger_mycobot.pt \
    --epochs 25 --lr 3e-4 --batch 256 --num-workers 8 --chunk 8 \
    --augment --ensemble-decay 0.01 > train_dagger.log 2>&1
echo "train rc=$? $(date -u +%FT%TZ)"; tail -12 train_dagger.log
[ -f bc_dagger_mycobot.pt ] || { echo "ABORT: no trained policy"; exit 1; }

# --- 3: 200-ep paired eval of the top-2 by in-flight task score ---
TOP2=$(python3 - <<'PY'
import glob, json
rows = []
for p in glob.glob("bc_dagger_mycobot_eval100_epoch*.json"):
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

echo "=== DAGGER RETRAIN DONE $(date -u +%FT%TZ) ==="
