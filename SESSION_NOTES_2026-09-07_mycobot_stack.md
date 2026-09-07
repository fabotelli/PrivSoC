# Session notes — 2026-09-07 (PrivSoC: single-cam chunked BC for two-cube stacking on the myCobot 280)

Host: Lambda box — `~/mujoco-test/mycobot_stack/` (repo: fabotelli/PrivSoC)

## TL;DR

- **PrivSoC (Privileged SolverClone) port of the cube_stack pipeline to the
  myCobot 280 (6 revolute joints + parallel gripper), recipe verbatim.**
- **CHAMPION CORRECTED (later same day): `bc_mycobot_stack.pt` = epoch 10, 48.5%.**
  The pipeline shipped the best-VAL checkpoint (epoch 25, val 0.00793) but the
  in-flight task evals peaked at epoch 10 (51 vs 33 /100) — best-task ≠ best-val
  AGAIN (same lesson as cube_staircase).  Paired 200-seed re-eval:
  - epoch 10: overall **97/200 = 48.5%**, grasp 130/200 = 65.0%, wrong-cube 0,
    missed-grasp 69/200 = 34.5%, placement-miss 33/200 = 16.5%
    (`eval200_epoch10.json`)
  - epoch 25 (old ship): overall 76/200 = 38.0%, grasp 110/200 = 55.0%,
    missed-grasp 89/200 = 44.5% (`eval_mycobot_stack_final.json`)
  - +10.5 pts paired on 200 seeds is far outside the ±~5 paired-noise band.
- **Verdict:** needs iteration; not demo-ready. Dominant gap: missed grasp
  (34.5%), then placement (69% of grasps land).  On-policy DAgger launched (below).
- **Recipe is the proven single-cam chunked recipe verbatim** (k=8 action chunk,
  GELU encoder, AdamW 3e-4, batch 256, CosineAnnealingWarmRestarts T_0=25
  T_mult=1, GPU brightness/colour augment only, temporal-ensemble decay 0.01).
  Only the *arm, gripper and geometry constants* are new.
- Solver baseline on the same 200 seeds: **200/200 = 100.0%** overall.

## Task

Two IDENTICAL red cubes, randomly placed each episode (non-overlapping, both in
reach).  Pick the cube FARTHER from the gripper HOME (x=0.1725, y=0.0, workspace
plane) and stack it ON TOP of the NEARER cube.  Because the cubes are identical,
the policy gets NO appearance cue — it must localise both cubes from the single
256x256 side camera and infer farther-vs-nearer purely from position.  The
student sees only the side camera + joint state; the teacher uses ground-truth
positions.  GRIP and MOVE are separate steps.  Placement tolerance is tight
(~cube width = 0.018 m xy), so success needs precise placement, not a
forgiving bin drop.

## Arm port (what's new vs the LeArm cube_stack run)

- Arm: official Elephant Robotics `mycobot_280jn_mujoco.xml` chain + meshes,
  joint names `joint2_to_joint1`..`joint7_to_joint6` (J1..J6, 1:1 with
  `pymycobot`).  Position actuators kp=20 kv=2 (not tuned for speed).
- Parallel gripper modelled as two box fingers on `joint6_flange` (40 mm max
  opening, 19 mm jaw depth), one actuated slide `grip_left`, right jaw coupled
  -1; `pinch` site at the fingertip midpoint.
- Base plate flush with the table top (z=0.15, desktop deployment).  Cube zone
  x∈[0.150,0.195], y∈[-0.070,0.070] from an empirical wrist-down IK
  feasibility sweep (annulus r∈[0.15,0.21]); reach check 15/15 sub-mm.
- Solver geometry re-swept: GRASP_DROP 0.009 (0.004–0.012 sweep;
  shallow drops fail as missed-grasp), STACK_RELEASE_GAP 0.006
  (smallest no-loss gap of {0.006,0.010,0.014,0.020}).

## Pixel-floor preflight (256×256, two identical red cubes)

8 random scenes (seeds 9000–9007) + 6 forced worst-case pairs at the spawn
min-separation (0.065 m) across camera orientations.

| Stat | Value |
|---|---|
| cube bbox max side (px), min..max (mean) | 21..41 (mean 29.3) |
| Mean saturation of detected cube pixels | 0.83 |
| Circular-mean hue (deg) | 2.1 |
| Floor check (every cube ≥ 12 px) | PASS |
| Distinguishable (a background gap between the two red blobs, all scenes) | PASS |

Sample frames: `pixel_floor_check/pixel_floor_sample_*.png`,
`pixel_floor_check/min_sep_*.png`.

## Scene + solver

- **Scene** `mycobot_scene_cube_stack.xml`: two identical red cubes (`cube_a`,
  `cube_b`, 1.9 cm side, free joints), no bins.  `MycobotStackEnv.reset()`
  randomises both cube positions + yaws per episode with rejection-sampled
  non-overlap (min-sep 0.065 m).
- **Sequencing rule:** farther-from-home cube is picked up; nearer-from-home is
  the base.  Pure geometry (spawn positions), ties broken canonical (a, b).  The
  student must infer the same farther/nearer decision from pixels alone.
- **Solver** `mycobot_stack_solver.py`: single approach→descend→grasp→lift→
  over-base→lower→release→retreat FSM.  After the lift it reads the privileged
  grasp offset (held-cube centre relative to the pinch) and commands the pinch
  so the CUBE lands centred a 0.006 m gap above the base cube's top
  face — needed for the tight stack tolerance.  GRIP and MOVE are separate
  steps throughout.
- Solver 100-ep self-test (no policy, no DR): **100%** stacked.

## Data collection

```
collect_dataset_mycobot_stack.py --episodes 8000 --workers 24 \
    --resolution 256 --rate 50 --max-frames-per-episode 160 \
    --successes-only --out dataset_mycobot_stack.npz
```

- Single side camera; lighting + table DR only (cubes stay identical red).
- Result: kept **8000** episodes, **677395** frames, **133.20 GB**,
  in **2:27:41** wall time (solve rate 100.0%).

## Training

```
python3 train_bc_mycobot_stack.py --data dataset_mycobot_stack.npz \
    --out bc_mycobot_stack.pt \
    --epochs 25 --lr 3e-4 --batch 256 \
    --num-workers 8 --chunk 8 --augment --ensemble-decay 0.01
```

Architecture: `BCPolicySideChunk` (n_joints=7).  RAM plan: **full dataset fits in RAM**.
Train duration: **75m 51s**.

Per-epoch in-flight evals (100 ep, seed 20000, decay 0.01, max_steps 280):

| Epoch | Overall | Correct-sel | Grasp | Place-on-top |
|---|---|---|---|---|
| 5 | 42/100 = 42.0% | 68/100 = 68.0% | 68/100 = 68.0% | 42/100 = 42.0% |
| 10 | 51/100 = 51.0% | 66/100 = 66.0% | 66/100 = 66.0% | 51/100 = 51.0% |
| 15 | 38/100 = 38.0% | 54/100 = 54.0% | 54/100 = 54.0% | 38/100 = 38.0% |
| 20 | 41/100 = 41.0% | 61/100 = 61.0% | 61/100 = 61.0% | 41/100 = 41.0% |
| 25 | 33/100 = 33.0% | 50/100 = 50.0% | 50/100 = 50.0% | 33/100 = 33.0% |

## Final eval (200 ep, seed 20000, decay 0.01, max_steps 280, 24 workers)

| Metric | Value |
|---|---|
| Overall success | 76/200 = 38.0% |
| Correct-selection (picked FARTHER) | 110/200 = 55.0% |
| Grasp success | 110/200 = 55.0% |
| Place-on-top | 76/200 = 38.0% |
| Failure: wrong_cube_picked | 1/200 = 0.5% |
| Failure: missed_grasp | 89/200 = 44.5% |
| Failure: placement_miss | 34/200 = 17.0% |
| Failure: knocked_over_unstable | 0/200 = 0.0% |

JSON: `eval_mycobot_stack_final.json`.  Solver baseline on same 200 seeds:
**200/200 = 100.0%** overall.

## Deliverables

Remote `~/mujoco-test/mycobot_stack/`:
- `bc_mycobot_stack.pt` (**epoch 10, best-TASK: 48.5% @ paired 200 seeds** —
  re-crowned from the best-val epoch 25 = 38.0%, see TL;DR)
- `bc_mycobot_stack_epoch{5,10,15,20,25}.pt` (periodic snapshots)
- `eval_mycobot_stack_final.json`
- `demo_solver_mycobot.mp4`, `demo_policy_mycobot.mp4`,
  `demo_policy_mycobot_policycam.mp4`, `demo_policy_failure_mycobot.mp4`
- `METHOD.md` (full write-up), `HANDOVER.md` (calibration items),
  `DECISIONS.md`
- Source files (`mycobot_stack_env.py`, `mycobot_stack_solver.py`,
  `collect_dataset_mycobot_stack.py`, `train_bc_mycobot_stack.py`,
  `eval_bc_mycobot_stack.py`, `record_bc_mycobot_stack.py`,
  `mycobot_scene_cube_stack.xml`, `check_pixel_floor.py`)
- Pixel-floor preflight: `pixel_floor_check/`
- Logs: `collect_mycobot_stack.log`, `train_mycobot_stack.log`,
  `mycobot_stack_full_eval.log`, `demo_record.log`

Copied to a dated downloads folder `~/Downloads/mycobot_stack_2026-09-07/`.

## Decisions logged (no-input run)

See `DECISIONS.md` for the assumptions made autonomously during this run
(base mount + zone derivation, gripper model, grasp-depth/release-gap sweeps,
eval step budget, disk plan).

## Iteration 2 (same day): on-policy DAgger — LAUNCHED

Why: the failure signature (best-task epoch ≪ best-val epoch in closed loop,
dominant missed-grasp with near-zero wrong-cube) is covariate shift — the
policy's imperfect approach visits states the solver's clean demos never
cover.  Same wall cube_staircase hit at 23.5%; its fix (expert correction
FROM THE POLICY'S OWN states, combined in training) took it to 88.5%.

Port (`collect_dagger_mycobot_stack.py`): the failing phase here is the FIRST
one (grasp), so instead of a phase-boundary handoff the handoff step is
randomised — per episode the epoch-10 policy drives k ~ U[0,120] control
steps (exact eval-time temporal-ensemble inference), then the privileged
solver completes from whatever state the policy produced:
  * cubes on table  -> full FSM re-pick (it re-reads privileged positions,
    so nudged cubes get re-approached — the recovery labels we're missing);
  * farther cube in hand -> place-half only (over_base->lower->release->
    retreat), wrist roll pinned where the policy left it;
  * already stacked -> settle, keep;
  * cube off table / out of IK patch / wrong cube lifted -> early abort.
`--successes-only`; labels stay future-achieved qpos.  max-policy-steps
capped at 120 so kept episodes don't clone long failure-hover prefixes.
Smoke test: 8/8 rescued, ~126 frames/ep, schema identical to base npz.

Run (`run_dagger_pipeline.sh`, detached): collect 2500 eps (CPU/osmesa, 24
workers, seed 30000) -> train FROM SCRATCH on the DAgger set alone (recipe
verbatim, 25 epochs, in-flight 100-ep evals) -> 200-ep eval (seeds 20000+,
paired with baseline) of the top-2 task checkpoints.  Logs:
`collect_dagger.log`, `train_dagger.log`, `dagger_pipeline.log`.
