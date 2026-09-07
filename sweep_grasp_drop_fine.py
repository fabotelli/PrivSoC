"""Fine sweep around the physically-legal GRASP_DROP (fingertip must stay at
or above the table plane on a real desk: drop <= 0.0075 with the fingertip
2 mm beyond the pinch site).  Diagnose failure modes at each config."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

import mycobot_stack_solver as S
from mycobot_stack_env import MycobotStackEnv, STACK_XY_TOL, STACK_DZ_MIN


def classify(env, log):
    farther, nearer = log["farther"], log["nearer"]
    fp, pn = env.cube_pos(farther), env.cube_pos(nearer)
    fs, ns = env.cube_xy0[farther], env.cube_xy0[nearer]
    far_moved = (np.linalg.norm(fp[:2] - fs) > 0.02
                 or fp[2] - env.table_top_z > 0.02)
    dxy = float(np.linalg.norm(fp[:2] - pn[:2]))
    if log["success"]:
        return "success"
    if not far_moved:
        return "missed_grasp"
    if dxy >= STACK_XY_TOL:
        return "placement_miss"
    return "knocked_or_unstable"


for drop, gap in [(0.006, 0.010), (0.0075, 0.006), (0.0075, 0.010),
                  (0.0075, 0.014), (0.0075, 0.020), (0.009, 0.010)]:
    S.GRASP_DROP = drop
    S.STACK_RELEASE_GAP = gap
    env = MycobotStackEnv()
    modes = {}
    for i in range(100):
        env.reset(seed=i)
        log = S.MycobotStackSolver(env).run_episode()
        m = classify(env, log)
        modes[m] = modes.get(m, 0) + 1
    ok = modes.get("success", 0)
    print(f"drop={drop:.4f} gap={gap:.3f}: {ok}/100  modes={modes}", flush=True)
