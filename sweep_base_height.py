"""Empirical feasibility sweep: base-plate height x (radius, height) grid.
Finds the base height + cube-zone rectangle where wrist-down IK converges
(pos_tol 8e-4, axis z < -0.995) at grasp/standoff/stack-standoff heights."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from mycobot_stack_env import MycobotStackEnv

GRASP_DROP = 0.008
HEIGHTS = {"grasp": -GRASP_DROP, "standoff": 0.05, "stack_standoff": 0.085}

env = MycobotStackEnv()
bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
cube_z = env.table_top_z + env.cube_half + 1e-3

def feasible(x, y, z):
    tgt = np.array([x, y, z])
    for seed_q in (env.home_seed, np.array([0., 1.2, -2.0, -0.3, 0., 0.]),
                   np.array([0., 0.4, -0.8, -1.2, 0., 0.])):
        q = env.solve_ik(tgt, q_init=seed_q, pin_wrist_roll=0.0)
        env._ik_data.qpos[env.arm_qpos] = q
        mujoco.mj_fwdPosition(env.model, env._ik_data)
        p = env._ik_data.site_xpos[env.site_pinch]
        ax = env._ik_data.site_xmat[env.site_pinch].reshape(3, 3)[:, 2]
        if np.linalg.norm(tgt - p) < 8e-4 and ax[2] < -0.995:
            return True
    return False

for base_z in (0.15, 0.18, 0.21, 0.24):
    env.model.body_pos[bid][2] = base_z
    env.reset(seed=0)
    # radial feasibility along y=-0.01 at each height
    print(f"\nbase_z={base_z:.2f} (shoulder z={base_z+0.15756:.3f})")
    for hname, dz in HEIGHTS.items():
        ok_r = [r for r in np.arange(0.06, 0.22, 0.01)
                if feasible(r, -0.01, cube_z + dz)]
        print(f"  {hname:14s} feasible r: "
              f"{min(ok_r):.2f}..{max(ok_r):.2f}" if ok_r else
              f"  {hname:14s} NONE")
    # candidate zone rectangles: count grid failures
    for (xlo, xhi, ylo, yhi) in [(0.100, 0.160, -0.075, 0.055),
                                 (0.110, 0.170, -0.075, 0.055),
                                 (0.120, 0.180, -0.070, 0.060)]:
        fails = 0
        for hname, dz in HEIGHTS.items():
            for x in np.linspace(xlo, xhi, 5):
                for y in np.linspace(ylo, yhi, 5):
                    fails += not feasible(x, y, cube_z + dz)
        print(f"  zone x[{xlo:.3f},{xhi:.3f}] y[{ylo:.3f},{yhi:.3f}]: "
              f"{fails}/75 grid failures")
