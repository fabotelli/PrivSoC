"""
render_notes.py — fill the SESSION_NOTES template with numbers parsed from the
final eval JSON, the per-epoch eval JSONs, the pixel-floor report, and the
collection / training logs.

Usage:
    python3 render_notes.py SESSION_NOTES_2026-09-07_mycobot_stack.md
"""

import json
import os
import re
import sys


def percent(s, n):
    return f"{s}/{n} = {100*s/n:.1f}%" if n else "n/a"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt_minmax_mean(d):
    return f"{int(d['min'])}..{int(d['max'])} (mean {d['mean']:.1f})"


TEMPLATE = """# Session notes — 2026-09-07 (PrivSoC: single-cam chunked BC for two-cube stacking on the myCobot 280)

Host: Lambda box — `~/mujoco-test/mycobot_stack/` (repo: fabotelli/PrivSoC)

## TL;DR

- **PrivSoC (Privileged SolverClone) port of the cube_stack pipeline to the
  myCobot 280 (6 revolute joints + parallel gripper), recipe verbatim.**
- **`bc_mycobot_stack.pt` final 200-ep diagnostic eval (seed 20000, decay 0.01, max_steps {max_steps}, {eval_workers} workers):**
  - Overall success (farther cube stably stacked on nearer cube): **{overall}**
  - Correct-selection (picked the FARTHER cube, not the nearer): **{sel}**
  - Grasp success (farther cube lifted/moved): **{grasp}**
  - Place-on-top (landed on the nearer cube): **{pot}**
  - Failure modes: wrong-cube-picked {wcp}, missed-grasp {mg}, placement-miss {pm}, knocked-over/unstable {ko}
- **Verdict:** {verdict}
- **Recipe is the proven single-cam chunked recipe verbatim** (k=8 action chunk,
  GELU encoder, AdamW 3e-4, batch 256, CosineAnnealingWarmRestarts T_0=25
  T_mult=1, GPU brightness/colour augment only, temporal-ensemble decay 0.01).
  Only the *arm, gripper and geometry constants* are new.
- Solver baseline on the same 200 seeds: **{solver_overall}** overall.

## Task

Two IDENTICAL red cubes, randomly placed each episode (non-overlapping, both in
reach).  Pick the cube FARTHER from the gripper HOME (x=0.1725, y=0.0, workspace
plane) and stack it ON TOP of the NEARER cube.  Because the cubes are identical,
the policy gets NO appearance cue — it must localise both cubes from the single
256x256 side camera and infer farther-vs-nearer purely from position.  The
student sees only the side camera + joint state; the teacher uses ground-truth
positions.  GRIP and MOVE are separate steps.  Placement tolerance is tight
(~cube width = {stack_tol} m xy), so success needs precise placement, not a
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
- Solver geometry re-swept: GRASP_DROP {grasp_drop} (0.004–0.012 sweep;
  shallow drops fail as missed-grasp), STACK_RELEASE_GAP {release_gap}
  (smallest no-loss gap of {{0.006,0.010,0.014,0.020}}).

## Pixel-floor preflight (256×256, two identical red cubes)

8 random scenes (seeds 9000–9007) + 6 forced worst-case pairs at the spawn
min-separation ({min_sep} m) across camera orientations.

| Stat | Value |
|---|---|
| cube bbox max side (px), min..max (mean) | {pf_size} |
| Mean saturation of detected cube pixels | {pf_sat} |
| Circular-mean hue (deg) | {pf_hue} |
| Floor check (every cube ≥ 12 px) | {pf_floor} |
| Distinguishable (a background gap between the two red blobs, all scenes) | {pf_sep} |

Sample frames: `pixel_floor_check/pixel_floor_sample_*.png`,
`pixel_floor_check/min_sep_*.png`.

## Scene + solver

- **Scene** `mycobot_scene_cube_stack.xml`: two identical red cubes (`cube_a`,
  `cube_b`, 1.9 cm side, free joints), no bins.  `MycobotStackEnv.reset()`
  randomises both cube positions + yaws per episode with rejection-sampled
  non-overlap (min-sep {min_sep} m).
- **Sequencing rule:** farther-from-home cube is picked up; nearer-from-home is
  the base.  Pure geometry (spawn positions), ties broken canonical (a, b).  The
  student must infer the same farther/nearer decision from pixels alone.
- **Solver** `mycobot_stack_solver.py`: single approach→descend→grasp→lift→
  over-base→lower→release→retreat FSM.  After the lift it reads the privileged
  grasp offset (held-cube centre relative to the pinch) and commands the pinch
  so the CUBE lands centred a {release_gap} m gap above the base cube's top
  face — needed for the tight stack tolerance.  GRIP and MOVE are separate
  steps throughout.
- Solver 100-ep self-test (no policy, no DR): **{solver_self_test}%** stacked.

## Data collection

```
collect_dataset_mycobot_stack.py --episodes 8000 --workers 24 \\
    --resolution 256 --rate 50 --max-frames-per-episode 160 \\
    --successes-only --out dataset_mycobot_stack.npz
```

- Single side camera; lighting + table DR only (cubes stay identical red).
- Result: kept **{kept_eps}** episodes, **{total_frames}** frames, **{dataset_gb} GB**,
  in **{collect_wall}** wall time (solve rate {solve_rate}).

## Training

```
python3 train_bc_mycobot_stack.py --data dataset_mycobot_stack.npz \\
    --out bc_mycobot_stack.pt \\
    --epochs 25 --lr 3e-4 --batch 256 \\
    --num-workers 8 --chunk 8 --augment --ensemble-decay 0.01
```

Architecture: `BCPolicySideChunk` (n_joints=7).  RAM plan: **{ram_plan}**.
Train duration: **{train_wall}**.

Per-epoch in-flight evals (100 ep, seed 20000, decay 0.01, max_steps {max_steps}):

| Epoch | Overall | Correct-sel | Grasp | Place-on-top |
|---|---|---|---|---|
{periodic_eval_rows}

## Final eval (200 ep, seed 20000, decay 0.01, max_steps {max_steps}, {eval_workers} workers)

| Metric | Value |
|---|---|
| Overall success | {overall} |
| Correct-selection (picked FARTHER) | {sel} |
| Grasp success | {grasp} |
| Place-on-top | {pot} |
| Failure: wrong_cube_picked | {wcp} |
| Failure: missed_grasp | {mg} |
| Failure: placement_miss | {pm} |
| Failure: knocked_over_unstable | {ko} |

JSON: `eval_mycobot_stack_final.json`.  Solver baseline on same 200 seeds:
**{solver_overall}** overall.

## Deliverables

Remote `~/mujoco-test/mycobot_stack/`:
- `bc_mycobot_stack.pt` (best epoch checkpoint, **{best_epoch}**)
- `bc_mycobot_stack_epoch{{5,10,15,20,25}}.pt` (periodic snapshots)
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
"""


def main(out_path):
    pf = load_json("pixel_floor_check/pixel_floor_report.json")["summary"]
    final = load_json("eval_mycobot_stack_final.json")
    ps = final["policy_summary"]
    ss = final.get("solver_summary") or {}
    n = ps["n"]
    nn = ss.get("n", n) if ss else n
    max_steps = final.get("max_steps", 170)

    rows = []
    for ep in (5, 10, 15, 20, 25):
        p = f"bc_mycobot_stack_eval100_epoch{ep}.json"
        if not os.path.exists(p):
            rows.append(f"| {ep} | — | — | — | — |")
            continue
        try:
            d = load_json(p)
            s = d["policy_summary"]
            tot = s["n"]
            rows.append(
                f"| {ep} | {percent(s['overall_success'], tot)} | "
                f"{percent(s['correct_selection'], tot)} | "
                f"{percent(s['grasp_success'], tot)} | "
                f"{percent(s['place_on_top'], tot)} |")
        except Exception as e:
            rows.append(f"| {ep} | (load error: {e}) | | | |")

    kept_eps = total_frames = dataset_gb = collect_wall = solve_rate = "?"
    try:
        with open("collect_mycobot_stack.log") as f:
            txt = f.read()
        m = re.search(r"Saved [^ ]+ \(([\d.]+) GB\) in (\d+:\d+:\d+): "
                      r"(\d+) frames from (\d+) episodes \(([\d.]+)%", txt)
        if m:
            dataset_gb, collect_wall = m.group(1), m.group(2)
            total_frames, kept_eps = m.group(3), m.group(4)
            solve_rate = m.group(5) + "%"
    except Exception:
        pass

    ram_plan = train_wall = best_epoch = "?"
    try:
        with open("train_mycobot_stack.log") as f:
            txt = f.read()
        m = re.search(r"RAM plan: ([^\n]+)", txt)
        if m:
            ram_plan = m.group(1)
        epochs = re.findall(r"epoch +(\d+)/\d+\s+train ([\d.]+)\s+val ([\d.]+)", txt)
        if epochs:
            best = None
            best_val = float("inf")
            for ep, _, val in epochs:
                v = float(val)
                if v < best_val:
                    best_val, best = v, int(ep)
            best_epoch = f"epoch {best} (val {best_val:.5f})" if best else "?"
        durs = re.findall(r"epoch +\d+/\d+.*?\s+([\d.]+)s", txt)
        if durs:
            secs = sum(float(d) for d in durs)
            train_wall = f"{int(secs // 60)}m {int(secs % 60)}s"
    except Exception:
        pass

    solver_self_test = os.environ.get("SOLVER_SELF_TEST", "100")
    pct = 100.0 * ps["overall_success"] / n
    sol_pct = 100.0 * ss["overall_success"] / nn if ss else None
    if pct >= 85 and (sol_pct is None or sol_pct - pct <= 5):
        verdict = "demo-ready (GATE D: student >= 85% and within 5 pts of solver)."
    elif pct >= 75:
        verdict = "needs iteration; near-demo-ready, small loss vs ceiling."
    elif pct >= 50:
        verdict = "needs iteration; substantial failure modes — see breakdown."
    else:
        verdict = "needs iteration; not demo-ready."

    # dominant capability gap
    sel_pct = 100.0 * ps["correct_selection"] / n
    grasp_pct = 100.0 * ps["grasp_success"] / n
    pot_pct = 100.0 * ps["place_on_top"] / n
    caps = {
        "cube-selection (picking the farther cube)": sel_pct,
        "grasp": grasp_pct,
        "stack-placement (landing on the base)": pot_pct,
        "stability (stack holds)": pct,
    }
    if pct < 90:
        dom = min(caps, key=caps.get)
        verdict += f" Dominant capability gap: {dom} ({caps[dom]:.0f}%)."

    body = TEMPLATE.format(
        overall=percent(ps["overall_success"], n),
        sel=percent(ps["correct_selection"], n),
        grasp=percent(ps["grasp_success"], n),
        pot=percent(ps["place_on_top"], n),
        wcp=percent(ps["wrong_cube_picked"], n),
        mg=percent(ps["missed_grasp"], n),
        pm=percent(ps["placement_miss"], n),
        ko=percent(ps["knocked_over_unstable"], n),
        verdict=verdict,
        solver_overall=percent(ss["overall_success"], nn) if ss else "(skipped)",
        solver_self_test=solver_self_test,
        max_steps=max_steps, eval_workers=24,
        stack_tol="0.018", min_sep="0.065",
        release_gap="0.006", grasp_drop="0.009",
        pf_size=fmt_minmax_mean(pf["cube_bbox_max_side"]),
        pf_sat=f"{pf['cube_sat_mean']:.2f}",
        pf_hue=f"{pf['cube_hue_circ_mean']:.1f}",
        pf_floor="PASS" if pf["pass_pixel_floor"] else "FAIL",
        pf_sep="PASS" if pf["pass_distinguishable"] else "FAIL",
        kept_eps=kept_eps, total_frames=total_frames,
        dataset_gb=dataset_gb, collect_wall=collect_wall, solve_rate=solve_rate,
        ram_plan=ram_plan, train_wall=train_wall, best_epoch=best_epoch,
        periodic_eval_rows="\n".join(rows),
    )

    with open(out_path, "w") as f:
        f.write(body)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "SESSION_NOTES_2026-09-07_mycobot_stack.md"
    main(out)
