# PrivSoC — Privileged SolverClone: two-cube stacking on a myCobot 280

Full method write-up, self-contained (no prior cube_stack context assumed).
Repo: https://github.com/fabotelli/PrivSoC — dir `~/mujoco-test/mycobot_stack/`.

## 1. Task definition

Two **identical** red cubes (1.9 cm side) spawn at random, non-overlapping
poses (min centre separation 0.065 m, random yaws) on a table, inside the
zone x∈[0.150, 0.195], y∈[-0.070, 0.070] (metres, arm-base frame). The robot
must **pick the cube FARTHER from the gripper home point (0.1725, 0.0) and
stack it on top of the NEARER cube**.

- Farther/nearer is decided by Euclidean distance from home in the workspace
  plane, using **spawn** positions (fixed labels for the whole episode); ties
  break canonically (cube a first).
- Because the cubes are identical, there is **no appearance cue**: a policy
  must localise both cubes in the image and infer the farther/nearer relation
  purely from geometry.
- **Success** (privileged check): farther-cube centre within **0.018 m (xy)**
  of the nearer-cube centre (~one cube width — tight placement), at least
  **0.012 m above it** (clean stack ≈ 0.019 m), with the base cube still on
  the table. Evaluated after the retreat, so unstable stacks that topple
  count as failures.

## 2. Simulator

- **Arm**: official Elephant Robotics MJCF, `mycobot_280jn_mujoco.xml` +
  STL meshes (github.com/elephantrobotics/mycobot_mujoco), chain and joint
  names verbatim: `joint2_to_joint1` … `joint7_to_joint6` = J1…J6, mapping
  1:1 to `pymycobot` joint indices. JN-variant base housing; kinematics are
  identical across 280 variants. MuJoCo 3.9, timestep 0.002, implicit
  integrator, 200 iterations, noslip 10.
- **Actuators**: `<position>` on all six joints, kp=20 kv=2, forcerange ±20
  (`a_j1..a_j6`) — deliberately NOT tuned for speed (the real 280's servos
  are slow and weak). All arm/gripper bodies have `gravcomp="1"` so the
  weak gains hold pose without sag (mirrors how the real servos hold
  position).
- **Gripper**: parallel jaws modelled as two box fingers on `joint6_flange`:
  palm box + fingers extending along flange +z (the tool axis, pointing down
  at grasp). One actuated slide joint `grip_left` (axis +y, range
  [-0.001, 0.0185] m), right jaw mirrored via
  `<equality><joint polycoef="0 -1 0 0 0">`. Max opening 40 mm, pad (jaw)
  depth 19 mm, friction 3.0 pads. `pinch` site at the fingertip midpoint,
  z-axis along the approach. `GRIPPER_OPEN=0.016` (33 mm opening),
  `GRIPPER_CLOSE=-0.001` (~12 mm over-travel past cube contact → ≈3.6 N
  clamp at kp=300). Grip actuator kp=300 kv=5 forcerange ±15 (slide units
  are N/m, chosen for clamp force equivalent to the LeArm's hinge gripper).
- **Mounting**: base plate flush with the table top (z=0.15) on a pedestal —
  the desktop deployment of a real 280 (arm and cubes on the same surface).
  `BASE_HEIGHT_OFFSET` (default 0.0) in `mycobot_stack_env.py` shifts the
  base at load for real-bench calibration.
- **Scene**: work table top z=0.15 spanning x∈[0.02,0.30], y∈[-0.16,0.16];
  two free-joint cubes sharing one `cube_red` material; two fixed lights.
- **Cameras**: single **256×256 `policy_cam`** at (0.357, -0.242, 0.353)
  looking at the zone centre from ~0.36 m (the only visual input the student
  ever sees); `demo_cam` for videos; `wrist_cam` on the flange present in
  the XML but never recorded (reserved for a future phase).
- **Collision filtering**: arm meshes are visual-only (contype 0 — the arm
  never collides); only the finger pads collide, and only with the cubes
  (pad contype 4/conaffinity 1, cube 1/3, env 2/1). Cube↔cube and
  cube↔table collide normally.
- **Domain randomisation** (collection + eval): per-episode light diffuse
  ×U(0.6,1.4) per light, table RGB ×U(0.8,1.2). Cubes are never jittered
  (identical red is the task premise).

## 3. Privileged teacher (the "Solver")

`mycobot_stack_solver.py` — a fixed-sequence FSM over Cartesian waypoints,
with full access to ground-truth cube poses:

```
approach(standoff over farther cube) → descend → close-grip → lift
→ over-base(standoff over nearer cube) → lower → release → retreat
```

- **Motion primitive**: each waypoint solved by damped-least-squares IK
  (6-DoF, 3D position + approach-axis alignment to vertical, pos_tol 8e-4,
  damping 0.05, joint limits clamped per iteration), then tracked by a
  joint-space slew limiter (**0.005 rad/substep**) until within settle_tol.
  The wrist roll (J6) is a free DoF about the approach axis; it is pinned
  after IK so the jaws align with the cube's yaw (mod 90°, verified to
  0.10° worst-case on rendered seeds).
- **GRIP and MOVE are separate steps** — a gripper command never overlaps a
  Cartesian move (robot-legal for position-servo hardware).
- **Placement-offset trick**: after the lift, read the privileged offset
  between the held-cube centre and the pinch site, then command the pinch so
  the **cube** (not the pinch) lands centred over the base cube. This is
  what makes the tight 0.018 m tolerance reachable; without it, in-jaw
  slippage biases the drop.
- **Constants** (swept, 100 eps each, no DR):

  | Constant | Value | Sweep evidence |
  |---|---|---|
  | STANDOFF_HEIGHT | 0.05 | kept from LeArm (gripper not taller) |
  | STACK_STANDOFF_HEIGHT | 0.085 | kept from LeArm |
  | GRASP_DROP | **0.009** | 0.004→75%, 0.006→53%, 0.0075→79%, 0.008→92%, 0.009→**100%**, 0.012→100% (shallow drops fail as missed-grasp: pads bite too high). Smallest 100% value; fingertip clips the table plane by 1.5 mm nominal → 2–3 mm mat on the real desk |
  | STACK_RELEASE_GAP | **0.006** | at drop 0.009: 0.006/0.010/0.014/0.020 all 100/100 → smallest no-loss gap (fingertip stays ~4.5 mm above base-cube top at release) |

- **Teacher self-test (GATE A): 100/100 stacked** (seeds 0-99, no DR).
  Episode length 52–58 policy frames (52.7–53.4 mean across sweeps).

## 4. Data

```
collect_dataset_mycobot_stack.py --episodes 8000 --workers 24 \
    --resolution 256 --rate 50 --max-frames-per-episode 160 \
    --successes-only --out dataset_mycobot_stack.npz
```

- One sample every 50 physics substeps (10 Hz at dt=0.002), single side
  camera + 7 joint angles (6 arm + `grip_left`), successes only, DR on.
  24 worker processes (BLAS threads pinned to 1), atomic per-worker
  checkpoints, merged into one uncompressed npz.
- Result: kept **TBD-KEPT** episodes (solve rate TBD-SOLVE), **TBD-FRAMES**
  frames, **TBD-GB GB**, in **TBD-WALL** wall time.

## 5. Student

`BCPolicySideChunk` (13.20 M params) — action-chunked single-camera BC:

- **Inputs**: 256×256×3 side image (÷255) and the 7 current joint angles,
  standardised by training-set mean/std.
- **Encoder**: Conv(32,8,4)-GELU → Conv(64,4,2)-GELU → Conv(64,3,1)-GELU →
  flatten → Linear→256 GELU bottleneck.
- **Head**: concat(image feature, joints) → 512 GELU → 256 GELU → linear to
  **chunk k=8 × 7 joints** = the next 8 future absolute joint targets
  (standardised), MSE loss.
- **Optimiser**: AdamW lr 3e-4, weight decay 1e-4, batch 256,
  CosineAnnealingWarmRestarts (T_0=25, T_mult=1, eta_min=1e-7), 25 epochs,
  AMP on CUDA, 5% episode-level val split (best-val checkpoint kept).
- **Augmentation** (GPU, train only): per-image brightness ×N(1,0.15²)
  clamped [0.7,1.3] + per-channel colour offset U(-0.05,0.05). No noise.
- **Inference**: temporal ensembling over the last k predicted chunks — at
  each step the k overlapping predictions for "now" are combined with
  exponential weights exp(-0.01·age) and the blended 7-vector is commanded
  (arm targets clipped to joint limits, grip to ctrlrange); physics advances
  50 substeps per policy step (10 Hz), matching collection.

## 6. Results

Per-epoch in-flight evals (100 eps, seed 20000, decay 0.01, max_steps 170):

TBD-EPOCH-TABLE

Final 200-episode diagnostic eval (seed 20000, decay 0.01, max_steps 170,
24 workers), solver baseline on the SAME seeds and DR:

TBD-FINAL-TABLE

GATE D verdict: TBD-VERDICT

## 7. Sim-to-real checklist (for the real myCobot 280 bench)

1. **Base height**: scene assumes the base plate exactly at table-top level.
   Measure the real offset and set `BASE_HEIGHT_OFFSET` in
   `mycobot_stack_env.py` (added to base z at load).
2. **Work surface**: place a **2–3 mm compliant mat** (cork/EVA) under the
   cube zone — the swept grasp depth (GRASP_DROP 0.009) nominally brings the
   fingertip 1.5 mm below the table plane at the deepest grip.
3. **Gripper geometry**: the sim gripper is a 40 mm-opening, 19 mm-jaw-depth
   parallel model on `joint6_flange`; ONLY the gripper block of
   `mycobot_scene_cube_stack.xml` (palm/fingers/pads + `a_grip` actuator +
   the equality coupling) needs editing if the real gripper differs. Re-run
   the Step-3 sweeps (`sweep_solver_geometry.py`) after any change.
4. **Camera**: mount the real side camera to match `policy_cam`: position
   (0.357, -0.242, 0.353) m in the arm-base frame, looking at
   (0.1725, 0, 0.16), MuJoCo default fovy 45°, 256×256. Calibrate by
   overlaying a rendered frame on the live feed (cube corners + table edge).
5. **Joint mapping**: sim joints J1..J6 = `joint2_to_joint1` …
   `joint7_to_joint6` map index-for-index to `pymycobot`
   `send_radians`/`get_radians`. Verify sign conventions joint-by-joint at
   low speed before closed-loop control.
6. **Joint limits & speed**: sim ranges are the official MJCF limits;
   the real 280's servo speed is much lower than a 10 Hz step of up to
   0.25 rad — command with `send_radians(..., speed)` calibrated so the arm
   reaches each 10 Hz target in <100 ms, or downsample the policy rate.
7. **Command rate**: the policy expects observations at 10 Hz
   (`--rate 50` × dt 0.002). Keep the real control loop at 10 Hz and feed
   the CURRENT joint angles (not the last command) as the joint input.
8. **Gripper mapping**: sim `grip_left` metres → real gripper percent:
   OPEN 0.016 ≈ 33 mm opening, CLOSE at full clamp. Calibrate the two
   endpoints; the policy only ever needs open/close extremes.

## 8. Known gaps / next steps

- TBD-GAPS (filled after the final eval)
- The wrist camera is modelled but unused (single-cam recipe); a wrist-cam
  phase would need a new collection + training run.
- No force/torque sensing: the grip is open-loop position over-travel; real
  gripper current limits substitute for the sim's forcerange clamp.
- Real-bench validation of the sim-to-real checklist (items 1–8) is the next
  concrete step; no real-robot data exists in this run.
