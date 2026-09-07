"""Pick the largest cube-zone rectangle with zero IK failures on a 7x7 grid
x 3 heights (base_z = 0.15, flush mount)."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from mycobot_stack_env import MycobotStackEnv

GRASP_DROP = 0.008
HEIGHTS = {"grasp": -GRASP_DROP, "standoff": 0.05, "stack_standoff": 0.085}

env = MycobotStackEnv()
env.reset(seed=0)
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

rects = [
    (0.150, 0.200, -0.060, 0.060),
    (0.155, 0.200, -0.065, 0.065),
    (0.155, 0.195, -0.075, 0.075),
    (0.160, 0.205, -0.060, 0.060),
    (0.150, 0.195, -0.070, 0.070),
]
for (xlo, xhi, ylo, yhi) in rects:
    fails = []
    for hname, dz in HEIGHTS.items():
        for x in np.linspace(xlo, xhi, 7):
            for y in np.linspace(ylo, yhi, 7):
                if not feasible(x, y, cube_z + dz):
                    fails.append((hname, round(x, 3), round(y, 3)))
    area = (xhi - xlo) * (yhi - ylo) * 1e4
    print(f"zone x[{xlo:.3f},{xhi:.3f}] y[{ylo:+.3f},{yhi:+.3f}] "
          f"area={area:.0f}cm2: {len(fails)}/147 failures "
          f"{fails[:6] if fails else ''}", flush=True)
