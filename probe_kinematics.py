"""Probe the raw mycobot_280jn model: frame poses at qpos=0 and at candidate
grasp configurations, to place the gripper, table and cube zone."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("mycobot_mujoco/xml/mycobot_280jn_mujoco.xml")
d = mujoco.MjData(m)

names = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
         "joint5_to_joint4", "joint6_to_joint5", "joint7_to_joint6"]
for n in names:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    print(f"{n:20s} range {m.jnt_range[j]}")

def show(q, tag):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    for b in ["joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        xp = d.xpos[bid]
        zax = d.xmat[bid].reshape(3, 3)[:, 2]
        print(f"  {tag} {b:14s} pos=({xp[0]:+.4f},{xp[1]:+.4f},{xp[2]:+.4f}) "
              f"zaxis=({zax[0]:+.3f},{zax[1]:+.3f},{zax[2]:+.3f})")

show(np.zeros(6), "zero")
print()
# Candidate elbow-bent config: J2 forward, J3 down, J4 to make flange point down.
for q2, q3, q4 in [(0.6, -1.2, -0.9), (0.8, -1.5, -0.9), (0.5, -1.0, -1.1),
                   (0.9, -1.8, -0.6), (0.7, -1.4, -0.85)]:
    q = np.array([0.0, q2, q3, q4, 0.0, 0.0])
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "joint6_flange")
    xp = d.xpos[bid]
    zax = d.xmat[bid].reshape(3, 3)[:, 2]
    print(f"q2={q2:+.2f} q3={q3:+.2f} q4={q4:+.2f} flange "
          f"pos=({xp[0]:+.4f},{xp[1]:+.4f},{xp[2]:+.4f}) "
          f"zaxis=({zax[0]:+.3f},{zax[1]:+.3f},{zax[2]:+.3f})")
