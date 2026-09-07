"""Verify jaw/cube yaw alignment of _wrist_roll_for_cube on 10 rendered seeds:
numeric jaw-vs-cube angle at the descend pose + a rendered closeup each."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio.v2 as imageio

import mycobot_stack_solver as S
from mycobot_stack_env import MycobotStackEnv

env = MycobotStackEnv()
env.model.vis.global_.offwidth = 512
env.model.vis.global_.offheight = 512
r = mujoco.Renderer(env.model, height=512, width=512)
os.makedirs("jaw_check", exist_ok=True)

worst = 0.0
for i in range(10):
    env.reset(seed=100 + i)
    solver = S.MycobotStackSolver(env)
    farther, _ = env.pick_order()
    wr = solver._wrist_roll_for_cube(farther)
    cube_q = env.cube_quat(farther)
    cube_yaw = 2.0 * np.arctan2(cube_q[3], cube_q[0])

    # place the arm at the descend pose with the commanded roll
    grasp = env.cube_pos(farther) - np.array([0, 0, S.GRASP_DROP])
    q = env.solve_ik(grasp, q_init=env.home_seed, pin_wrist_roll=wr)
    env.data.qpos[env.arm_qpos] = q
    env.data.ctrl[env.arm_act] = q
    mujoco.mj_forward(env.model, env.data)

    pmat = env.data.site_xmat[env.site_pinch].reshape(3, 3)
    jaw = float(np.arctan2(pmat[1, 1], pmat[0, 1]))
    # misalignment mod 90 deg (cube is square), wrapped to +-45
    mis = ((jaw - cube_yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4
    worst = max(worst, abs(mis))
    print(f"seed {100+i}: farther={farther} cube_yaw={np.degrees(cube_yaw):7.1f} "
          f"jaw={np.degrees(jaw):7.1f} misalign={np.degrees(mis):6.2f} deg")

    r.update_scene(env.data, camera="policy_cam")
    imageio.imwrite(f"jaw_check/jaw_{i:02d}.png", r.render())
r.close()
print(f"\nworst misalignment: {np.degrees(worst):.2f} deg "
      f"({'PASS' if worst < np.radians(5) else 'FAIL'})")
