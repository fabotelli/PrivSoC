# HANDOVER — PrivSoC mycobot_stack run (2026-09-07)

State of the run, gate results, and the calibration items a real-bench
engineer needs. Written incrementally during the run; final numbers in
METHOD.md / SESSION_NOTES.

## Gate results

| Gate | Requirement | Result |
|---|---|---|
| A (solver) | ≥85% on 100 eps | **PASS — 100/100** |
| B (pixel floor) | every cube ≥12 px + separable | **PASS — bbox mean 29.3 px, worst-case gap 6 px, 0 sep failures** |
| C (validation learning signal) | grasp ≥5% or any place/overall | **PASS — grasp 54%, place 24, overall 24/100 (seed 90000)** |
| D (report-only) | student ≥85% & within 5 pts of solver | TBD |

## State on disk (`~/mujoco-test/mycobot_stack/`)

- Scene/env/solver/collector/trainer/eval/record ports: committed (repo
  fabotelli/PrivSoC, pushed after every gate).
- Sweep evidence: `sweep_solver_geometry.log`, `sweep_grasp_drop_fine.log`,
  `sweep_gap_at_drop009.log`, `jaw_check/`, `reach_check_out/`,
  `pixel_floor_check/`.
- Pipeline: `run_validation_then_full.sh` (detached, PID in
  `pipeline_run.pid`) → `run_pipeline.sh` → `finalize.sh`;
  watcher log `pipeline_watch.log`; main log `pipeline_run.log`.
- Vendored upstream arm model: `mycobot_mujoco/` (gitignored clone; meshes
  copied to `meshes_mycobot/`, tracked).

## Calibration items (real bench) — details in METHOD.md §7

1. `BASE_HEIGHT_OFFSET` (env constant, default 0.0): measure real base-plate
   height vs table top; the JN housing in the model may differ from your 280
   variant's base.
2. 2–3 mm compliant mat under the cube zone (GRASP_DROP 0.009 puts the
   fingertip 1.5 mm below the nominal table plane at the deepest grip).
3. Gripper block: ONLY the palm/finger/pad + `a_grip` + equality section of
   `mycobot_scene_cube_stack.xml` changes if the real gripper differs from
   the modelled 40 mm/19 mm parallel jaws; then re-run
   `sweep_solver_geometry.py` (GRASP_DROP, STACK_RELEASE_GAP) and Gate A.
4. Side camera at (0.357, -0.242, 0.353) looking at (0.1725, 0, 0.16),
   fovy 45°, 256×256; verify by render-overlay.
5. `pymycobot` J1..J6 = `joint2_to_joint1`..`joint7_to_joint6`
   index-for-index; verify signs at low speed.
6. 10 Hz control loop; servo speed calibrated so each step target is reached
   in <100 ms; feed measured joint angles (not last command) to the policy.
7. Gripper endpoint mapping: sim OPEN 0.016 (33 mm) / CLOSE full-clamp →
   real percent open/close.

## Next steps

- TBD after Gate D: either demo-ready → real-bench calibration (items
  above), or needs-iteration → see METHOD.md §8.
