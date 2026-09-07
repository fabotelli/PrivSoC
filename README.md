# PrivSoC — Privileged SolverClone

**A privileged scripted teacher + a pixels-only student.** A hand-written FSM
solver with full state access (exact cube poses, IK) demonstrates the task at
~100%; a single-camera behavior-cloning student distills it into a deployable
vision policy (side camera + joint angles in, chunked joint targets out).
This repo is the myCobot 280 instance: **two-cube stacking** — two identical
red cubes spawn randomly, pick the one farther from gripper home, stack it on
the nearer one. No appearance cue: the policy must infer farther/nearer from
pixels alone.

## Key files

| | |
|---|---|
| **Typical Training Episode** | `demo_solver_mycobot.mp4` — the privileged solver demonstrating one episode (what the student clones) |
| **Prompt** | `prompt1.txt` — the original ticket this pipeline was built against |
| **Policy** | `bc_mycobot_stack.pt` — the deliverable weights (always the current best-task champion) |
| **ReadMe** | `README.md` — this file |

## Quickstart — run the policy in sim yourself

```bash
git clone https://github.com/fabotelli/PrivSoC.git && cd PrivSoC
pip install mujoco torch numpy imageio imageio-ffmpeg
# headless machine (no display)? add:  export MUJOCO_GL=osmesa   (or egl on a GPU)

# 1. watch the shipped champion do one episode (writes demo.mp4)
python3 record_bc_mycobot_stack.py --policy bc_mycobot_stack.pt --out demo.mp4

# 2. full diagnostic eval + solver baseline (200 eps; ~13 min on 24 cores,
#    scales linearly — set --workers to your core count minus a few)
python3 eval_bc_mycobot_stack.py --policy bc_mycobot_stack.pt \
    --episodes 200 --workers 8 --save-json my_eval.json
```

Expected on the shipped scene: **solver 100%, policy ~48.5% overall / ~65%
grasp** (seeds 20000+). If you reproduce those numbers, your setup is good.
Everything runs on CPU; no GPU needed for inference or eval.

### Running it on your own scene MJCF (`--xml`)

`eval_bc_mycobot_stack.py`, `record_bc_mycobot_stack.py` and
`collect_dagger_mycobot_stack.py` all accept `--xml your_model.xml` to swap
the world model (e.g. a higher-fidelity 280 model) while keeping the policy
and harness unchanged. Your MJCF must keep the **naming contract** — the env
resolves everything by name:

| What | Required names |
|---|---|
| arm joints | `joint2_to_joint1` … `joint7_to_joint6` (J1..J6) |
| gripper | slide joint `grip_left`, actuator `a_grip` |
| arm actuators | `a_j1` … `a_j6` (position) |
| cubes | bodies `cube_a`/`cube_b`, free joints `cube_a_free`/`cube_b_free`, geom `cube_a_geom` |
| end-effector | site `pinch` (between the fingertips) |
| camera | fixed camera `policy_cam` (the policy's single view) |
| DR hooks | material `table` (+ scene lights) |

Suggested order for estimating the sim-to-real delta with a hi-fi model:

1. **Gate the teacher**: `eval ... --xml hifi.xml` *without* `--skip-solver`.
   The solver baseline tells you whether the *task* survived your physics
   (expect ~100%). If it drops, re-sweep the two contact constants
   (`sweep_solver_geometry.py`: `GRASP_DROP`, `STACK_RELEASE_GAP`) first —
   otherwise you'll misread teacher breakage as transfer gap.
2. **Measure the delta**: same eval with the shipped policy, same seeds as
   the baseline table above. The drop vs 48.5% is your s2r-delta estimate,
   and the diagnostic breakdown localizes it (selection = visual gap,
   grasp = contact/servo gap).
3. **Close it**: `collect_dagger_mycobot_stack.py --xml hifi.xml --policy
   bc_mycobot_stack.pt ...` then retrain (`run_dagger_pipeline.sh` shows the
   exact train/eval calls) — the solver corrects the policy exactly where
   your physics makes it misbehave.

## Pipeline

```
scene + env port          mycobot_scene_cube_stack.xml, mycobot_stack_env.py
      |
gate A: solver 100%       mycobot_stack_solver.py  (privileged FSM teacher)
gate B: pixel floor       check_pixel_floor.py     (every cube >=12 px, separable)
      |
collect demos             collect_dataset_mycobot_stack.py  (successes-only,
      |                    single side cam 256x256, 10 Hz, lighting/table DR)
train BC student          train_bc_mycobot_stack.py  (k=8 action chunks, GELU
      |                    CNN, in-flight closed-loop evals every 5 epochs)
eval + diagnose           eval_bc_mycobot_stack.py  (grasp / selection /
      |                    placement breakdown; crown BEST-TASK, never best-val)
DAgger iteration          collect_dagger_mycobot_stack.py + run_dagger_pipeline.sh
      |                    (policy drives k~U[0,120] steps, solver completes
      |                    from the policy's own states; retrain from scratch)
deploy                    run_real_mycobot_stack.py  (real 280 over pymycobot)
```

Orchestration: `run_validation_then_full.sh` (cheap 800-ep learning-signal
gate, then the full run) -> `run_pipeline.sh` -> `finalize.sh`; the A10 is
taken from the idle harvester via `gpu_claim.sh` only for training.

## Pipeline files

| File | Role |
|---|---|
| `mycobot_scene_cube_stack.xml` | MuJoCo scene: 280 arm + parallel gripper + two identical red cubes |
| `mycobot_stack_env.py` | env wrapper: spawn randomization, privileged accessors, DLS IK, `is_stacked()` |
| `mycobot_stack_solver.py` | privileged teacher FSM (approach/descend/grasp/lift/place, 100/100) |
| `check_pixel_floor.py` | gate B: cubes visible + separable at 256 px |
| `collect_dataset_mycobot_stack.py` | parallel successes-only demo collector -> npz |
| `train_bc_mycobot_stack.py` | chunked single-cam BC trainer + in-flight evals |
| `eval_bc_mycobot_stack.py` | 200-ep diagnostic eval (also the sim runtime executor) |
| `record_bc_mycobot_stack.py` | single-episode mp4 demos |
| `collect_dagger_mycobot_stack.py` | on-policy DAgger collector (random handoff -> solver takeover) |
| `run_dagger_pipeline.sh` | DAgger iteration: collect -> scratch-train -> paired eval |
| `run_real_mycobot_stack.py` | real-hardware runner (pymycobot + USB cam) |
| `bc_mycobot_stack.pt` | **the deliverable**: current best-task champion weights |
| `METHOD.md` / `HANDOVER.md` / `DECISIONS.md` | write-up, bench calibration items, autonomous-run decisions |

## Results (200 paired seeds)

| policy | overall | grasp | notes |
|---|---|---|---|
| privileged solver | 100% | 100% | teacher |
| BC epoch 10 (champion) | 48.5% | 65% | best-task; best-val epoch 25 scored only 38% |
| + on-policy DAgger | in flight | — | the cube_staircase recipe that went 23.5% -> 88.5% |

## Real-hardware runner (`run_real_mycobot_stack.py`)

Runs the trained policy on a physical 280: USB camera in, `pymycobot`
`send_radians` + gripper percent out, inference numerically identical to sim
eval (10 Hz, ACT temporal ensemble). Needs torch-CPU, OpenCV, pymycobot — no
MuJoCo on the bench. Bring-up order:

```bash
# 1. align the camera: blend live feed 50/50 with a sim reference frame
python3 run_real_mycobot_stack.py --dry-run --preview --overlay sim_frame.png
# 2. verify joint directions at low speed (fix JOINT_SIGNS if any mismatch)
python3 run_real_mycobot_stack.py --sign-check --servo-speed 20
# 3. calibrate GRIP_PCT_OPEN/CLOSE constants to the sim's 33mm/full-clamp
# 4. first live run: slow, slew-clamped, hand on the e-stop
python3 run_real_mycobot_stack.py --go-home --servo-speed 30 --max-delta-rad 0.08
```

Ctrl-C stops motion without opening the gripper (a held cube is not dropped).
Safety rails: per-step slew clamp, joint-limit clip, 10 Hz loop-lag warning.
Measured joint angles (not echoed commands) are fed to the policy.

## Transfer via a higher-fidelity sim first

A sensible de-risking step before the bench: treat the high-fi sim (better
meshes/contacts, real camera intrinsics/noise, servo latency — e.g. Isaac Sim
or an upgraded MuJoCo model) as *just another deployment target* and reuse the
pipeline:

1. **Port the scene, keep the contract.** Same obs/action interface: one
   256x256 side cam at the calibrated pose, 7-dim joint state, 10 Hz joint
   targets. Only the world model changes — if it is another MJCF, that is
   just `--xml` (see Quickstart above).
2. **Zero-shot eval the frozen policy** there with the same diagnostic
   harness. The breakdown localizes the gap: grasp collapse with intact
   selection = dynamics/contact gap; selection collapse = visual gap.
3. **Visual gap -> re-render, don't re-demonstrate.** Replay the recorded
   demo trajectories (qpos streams) in the high-fi renderer and re-render the
   frames: paired data for near-free, labels unchanged. Widen DR toward the
   high-fi appearance and retrain.
4. **Dynamics gap -> re-run PrivSoC natively.** The teacher is privileged
   state + IK, not learned — if the high-fi sim exposes state, the whole
   collect->train->DAgger loop ports as-is and regenerates demos under the
   new physics.
5. **Then DAgger across the gap.** Point `collect_dagger_mycobot_stack.py` at
   the high-fi env with the low-fi-trained policy as the phase-0 driver: the
   solver corrects exactly the states where the transferred policy misbehaves
   under the new physics — same covariate-shift cure, applied to the fidelity
   gap instead of the training gap.

Whatever survives the high-fi sim inherits the same bench calibration items
(`HANDOVER.md`): base height offset, camera pose, gripper endpoint mapping,
joint signs.
