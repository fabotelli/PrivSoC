"""
reach_check.py — Step 2.4: verify DLS-IK convergence (pos_tol 8e-4) at the
four corners + centre of the cube zone, at grasp, standoff and stack-standoff
heights.  Also reports the approach-axis alignment and renders one frame from
each camera for a framing sanity check.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio.v2 as imageio

from mycobot_stack_env import (MycobotStackEnv, CUBE_X_LO, CUBE_X_HI,
                               CUBE_Y_LO, CUBE_Y_HI)

GRASP_DROP = 0.008          # planned initial value (swept in Step 3)
STANDOFF_HEIGHT = 0.05
STACK_STANDOFF_HEIGHT = 0.085

env = MycobotStackEnv()
env.reset(seed=0)

cube_z = env.table_top_z + env.cube_half + 1e-3
heights = {
    "grasp":          cube_z - GRASP_DROP,
    "standoff":       cube_z + STANDOFF_HEIGHT,
    "stack_standoff": cube_z + STACK_STANDOFF_HEIGHT,
}
xy_pts = [(CUBE_X_LO, CUBE_Y_LO), (CUBE_X_LO, CUBE_Y_HI),
          (CUBE_X_HI, CUBE_Y_LO), (CUBE_X_HI, CUBE_Y_HI),
          ((CUBE_X_LO + CUBE_X_HI) / 2, (CUBE_Y_LO + CUBE_Y_HI) / 2)]

n_fail = 0
print(f"zone x[{CUBE_X_LO},{CUBE_X_HI}] y[{CUBE_Y_LO},{CUBE_Y_HI}]  "
      f"pos_tol 8e-4")
for hname, z in heights.items():
    for (x, y) in xy_pts:
        tgt = np.array([x, y, z])
        # Seed the way the solver's FSM does: standoff poses are approached
        # from home; the grasp descend is seeded by continuation from the
        # standoff solution directly above the target.
        if hname == "grasp":
            q_seed = env.solve_ik(np.array([x, y, heights["standoff"]]),
                                  q_init=env.home_seed, pin_wrist_roll=0.0)
        else:
            q_seed = env.home_seed
        q = env.solve_ik(tgt, q_init=q_seed, pin_wrist_roll=0.0)
        env._ik_data.qpos[env.arm_qpos] = q
        mujoco.mj_fwdPosition(env.model, env._ik_data)
        p = env._ik_data.site_xpos[env.site_pinch]
        ax = env._ik_data.site_xmat[env.site_pinch].reshape(3, 3)[:, 2]
        perr = float(np.linalg.norm(tgt - p))
        ok = perr < 8e-4 and ax[2] < -0.995
        n_fail += (not ok)
        r = float(np.hypot(x, y))
        print(f"  {hname:14s} ({x:+.3f},{y:+.3f}) r={r:.3f} z={z:.3f} "
              f"perr={perr*1000:6.2f}mm axz={ax[2]:+.3f} "
              f"{'OK' if ok else '** FAIL **'}")

# framing sanity renders (home pose, cubes at reset positions)
env.model.vis.global_.offwidth = 512
env.model.vis.global_.offheight = 512
r = mujoco.Renderer(env.model, height=512, width=512)
os.makedirs("reach_check_out", exist_ok=True)
for cam in ("policy_cam", "demo_cam"):
    r.update_scene(env.data, camera=cam)
    imageio.imwrite(f"reach_check_out/{cam}.png", r.render())
r.close()

print(f"\n{'PASS' if n_fail == 0 else f'FAIL ({n_fail} points)'}")
raise SystemExit(0 if n_fail == 0 else 1)
