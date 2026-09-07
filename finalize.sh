#!/usr/bin/env bash
# After training completes (bc_mycobot_stack.pt exists, no train process
# running), run the final 200-ep diagnostic eval (with solver baseline),
# record the success / policy-cam / failure demos, render the session notes,
# and copy all deliverables into ~/Downloads/mycobot_stack_2026-09-07/.
set -uo pipefail

cd ~/mujoco-test/mycobot_stack

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

POLICY=bc_mycobot_stack.pt
EVAL_JSON=eval_mycobot_stack_final.json
NOTES=SESSION_NOTES_2026-09-07_mycobot_stack.md
DEST=~/Downloads/mycobot_stack_2026-09-07

if [ ! -f "$POLICY" ]; then
  echo "no $POLICY yet; bailing"
  exit 1
fi
if pgrep -f train_bc_mycobot_stack.py >/dev/null; then
  echo "training still running; bailing"
  exit 1
fi

echo "=== 200-ep diagnostic eval + solver baseline (24 workers) ==="
python3 eval_bc_mycobot_stack.py --policy "$POLICY" --episodes 200 \
    --start-seed 20000 --workers 24 --rate 50 --max-steps 170 \
    --ensemble-decay 0.01 --save-json "$EVAL_JSON" 2>&1 \
    | tee mycobot_stack_full_eval.log

echo "=== record success demo (demo_cam + policy_cam) ==="
python3 record_bc_mycobot_stack.py --policy "$POLICY" \
    --out demo_policy_mycobot.mp4 \
    --policycam-out demo_policy_mycobot_policycam.mp4 \
    --seed 20000 --max-search 30 --max-steps 170 --rate 50 \
    --ensemble-decay 0.01 --vid-res 512 --fps 30 2>&1 \
    | tee demo_record.log

echo "=== record one representative failure (if any) ==="
python3 record_bc_mycobot_stack.py --policy "$POLICY" --failure \
    --out demo_policy_failure_mycobot.mp4 \
    --seed 20000 --max-search 100 --max-steps 170 --rate 50 \
    --ensemble-decay 0.01 --vid-res 512 --fps 30 2>&1 \
    | tee -a demo_record.log || echo "(no failure found in 100 seeds)"

echo "=== render session notes ==="
SOLVER_SELF_TEST=100 python3 render_notes.py "$NOTES"

echo "=== copy deliverables to $DEST ==="
mkdir -p "$DEST"
cp -f "$POLICY" "$EVAL_JSON" "$NOTES" "$DEST/" 2>/dev/null || true
cp -f demo_solver_mycobot.mp4 demo_policy_mycobot.mp4 \
   demo_policy_mycobot_policycam.mp4 demo_policy_failure_mycobot.mp4 \
   "$DEST/" 2>/dev/null || true
cp -f DECISIONS.md METHOD.md HANDOVER.md prompt1.txt "$DEST/" 2>/dev/null || true
cp -f bc_mycobot_stack_epoch*.pt "$DEST/" 2>/dev/null || true
cp -f bc_mycobot_stack_eval100_epoch*.json "$DEST/" 2>/dev/null || true
cp -f bc_mycobot_stack_eval100_epoch*.log "$DEST/" 2>/dev/null || true
cp -f bc_mycobot_stack_inflight_eval.log "$DEST/" 2>/dev/null || true
cp -f train_mycobot_stack.log collect_mycobot_stack.log "$DEST/" 2>/dev/null || true
cp -f mycobot_stack_full_eval.log demo_record.log "$DEST/" 2>/dev/null || true
cp -f mycobot_stack_env.py mycobot_stack_solver.py collect_dataset_mycobot_stack.py \
   train_bc_mycobot_stack.py eval_bc_mycobot_stack.py record_bc_mycobot_stack.py \
   check_pixel_floor.py mycobot_scene_cube_stack.xml render_notes.py \
   record_solver_mycobot.py reach_check.py \
   finalize.sh run_pipeline.sh run_validation_then_full.sh "$DEST/" 2>/dev/null || true
cp -rf pixel_floor_check "$DEST/" 2>/dev/null || true
cp -rf meshes_mycobot "$DEST/" 2>/dev/null || true

echo "deliverables in $DEST:"
ls -la "$DEST"
echo "FINALIZE DONE"
