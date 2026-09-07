"""
mycobot_stack_solver.py
=======================

Privileged ("teacher") FSM for the two-cube STACKING task on the myCobot 280.
Port of ``cube_stack_solver.py`` (LeArm): FSM and motion primitives VERBATIM
(slew 0.005, grip-then-move, privileged grasp offset); only the geometry
constants are re-derived for the 280's parallel gripper:

  * GRASP_DROP for the ~19 mm jaw (pad) depth,
  * STACK_RELEASE_GAP re-swept over {0.006, 0.010, 0.014, 0.020} (see
    DECISIONS.md),
  * STANDOFF heights kept (gripper is not taller than the LeArm's),
  * ``_wrist_roll_for_cube`` re-derived for the new pinch frame (jaws close
    along the pinch y-axis, roll = J6 about the approach axis -> the LeArm
    formula carries over; verified on rendered seeds by check_jaw_alignment.py).

    pick the cube FARTHER from gripper home  ->  place it ON TOP of the cube
    NEARER to gripper home.

Sequencing rule is GEOMETRY-ONLY: the farther-from-home cube is the one that
gets lifted (a tie is broken canonical 'a','b').  The cubes are identical, so
the policy never has an appearance cue -- it must infer the same farther/nearer
decision from the side camera alone.

Precise placement: after the lift, we read the privileged offset between the
held cube's centre and the pinch site, then command the pinch so the *cube*
(not the pinch) is centred a hair above the base cube's top face.  This keeps
the placement inside the tight ~cube-width tolerance.

GRIP and MOVE are separate steps (never overlap a gripper command with a
Cartesian move).
"""

from __future__ import annotations

import time
import numpy as np
import mujoco

from mycobot_stack_env import (MycobotStackEnv, APPROACH_DOWN,
                               GRIPPER_OPEN, GRIPPER_CLOSE, CUBES)


# Waypoint offsets.
STANDOFF_HEIGHT = 0.05    # standoff above cube pre-/post-grasp (kept from LeArm)
GRASP_DROP = 0.0075       # grasp this far below cube centre.  Under the fixed
                          # physics (rigid jaw mirror + straight descend) every
                          # swept drop 0.006-0.012 is 100/100; 0.0075 puts the
                          # pad dead-centre on the cube with the fingertip
                          # exactly at the table plane (zero clip -> no mat
                          # needed on the real desk).  The pre-fix sweep that
                          # favoured 0.009 was confounded by the descend-shove
                          # (DECISIONS.md #19).

# Stack-side waypoints.  Carry the held cube high over the base cube, then lower
# so the held cube's bottom face rests a few mm above the base cube's top.
STACK_STANDOFF_HEIGHT = 0.085   # cube centre carried this far above base centre
STACK_RELEASE_GAP = 0.006       # release with cube bottom this far above base
                                # top; re-swept {0.006,0.010,0.014,0.020} under
                                # the fixed physics: all 100/100 -> smallest
                                # no-loss gap


class MycobotStackSolver:
    """Single pick-and-place FSM: pick farther cube, stack on nearer cube."""

    def __init__(self, env: MycobotStackEnv, speed: float | None = None,
                 recorder=None):
        self.env = env
        self._recorder = recorder
        self._step_sleep = (None if speed is None
                            else env.model.opt.timestep / speed)

    def _render(self, viewer) -> None:
        if self._recorder is not None:
            self._recorder.capture(self.env.data)
        if viewer is None:
            return
        viewer.sync()
        if self._step_sleep:
            time.sleep(self._step_sleep)

    # ------------------------------------------------------------------ #
    #  Motion primitives (verbatim from cube_stack_solver).
    # ------------------------------------------------------------------ #
    def _move_to_pose(self, target_pos, *,
                      gripper, pin_wrist_roll=None, settle_tol=0.02,
                      max_steps=1500, viewer=None, straight=False):
        """LeArm primitive (slew-limited joint tracking of one IK solution),
        plus a `straight` mode required by the 280's kinematics: joint-space
        interpolation between the standoff and grasp configs bows the pinch
        up to 26 mm sideways (measured, DECISIONS.md #19) — more than the
        8 mm jaw clearance — so vertical strokes near the cubes track a
        chain of IK via-points every ~12 mm along the straight line instead.
        Same slew, same settle logic, same grip-then-move discipline."""
        env = self.env
        if straight:
            start = env.pinch_pos.copy()
            dist = float(np.linalg.norm(target_pos - start))
            n_seg = max(1, int(np.ceil(dist / 0.012)))
            vias = [start + (target_pos - start) * (k / n_seg)
                    for k in range(1, n_seg + 1)]
        else:
            vias = [np.asarray(target_pos, dtype=float)]
        env.set_gripper(gripper)
        slew = 0.005
        q_cmd = env.arm_qpos_now
        budget = max_steps
        for vi, via in enumerate(vias):
            q_target = env.solve_ik(via, APPROACH_DOWN,
                                    q_init=q_cmd,
                                    pin_wrist_roll=pin_wrist_roll)
            tol = settle_tol if vi == len(vias) - 1 else max(settle_tol, 0.02)
            reached = False
            while budget > 0:
                budget -= 1
                step = np.clip(q_target - q_cmd, -slew, slew)
                q_cmd = q_cmd + step
                env.set_arm_target(q_cmd)
                env.step()
                self._render(viewer)
                if np.max(np.abs(env.arm_qpos_now - q_target)) < tol:
                    reached = True
                    break
            if not reached:
                return False
        return True

    def _hold(self, steps, *, gripper, viewer=None):
        env = self.env
        env.set_gripper(gripper)
        for _ in range(steps):
            env.step()
            self._render(viewer)

    def _close_grip(self, *, ramp=250, hold=200, viewer=None):
        env = self.env
        for i in range(ramp):
            env.set_gripper(GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN)
                            * (i + 1) / ramp)
            env.step()
            self._render(viewer)
        for _ in range(hold):
            env.step()
            self._render(viewer)

    # ------------------------------------------------------------------ #
    #  Per-cube wrist-roll alignment (probe wr=0, align jaws to cube yaw).
    # ------------------------------------------------------------------ #
    def _wrist_roll_for_cube(self, c: str) -> float:
        env = self.env
        cube_xyz = env.cube_pos(c)
        cube_q = env.cube_quat(c)
        cube_yaw = 2.0 * np.arctan2(cube_q[3], cube_q[0])
        q_probe = env.solve_ik(cube_xyz + np.array([0, 0, STANDOFF_HEIGHT]),
                               q_init=env.home_seed, pin_wrist_roll=0.0)
        env._ik_data.qpos[env.arm_qpos] = q_probe
        mujoco.mj_fwdPosition(env.model, env._ik_data)
        pmat = env._ik_data.site_xmat[env.site_pinch].reshape(3, 3)
        jaw_at_wr0 = float(np.arctan2(pmat[1, 1], pmat[0, 1]))
        return ((jaw_at_wr0 - cube_yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4

    # ------------------------------------------------------------------ #
    #  Whole-episode FSM: pick farther cube, stack onto nearer cube.
    # ------------------------------------------------------------------ #
    def run_episode(self, viewer=None) -> dict:
        env = self.env
        farther, nearer = env.pick_order()

        pick_xyz = env.cube_pos(farther)
        grasp_xyz = pick_xyz - np.array([0, 0, GRASP_DROP])
        standoff_xyz = pick_xyz + np.array([0, 0, STANDOFF_HEIGHT])

        wr = self._wrist_roll_for_cube(farther)
        log = {"farther": farther, "nearer": nearer}

        # 1) APPROACH: standoff above the farther cube, jaws open.
        log["approach"] = self._move_to_pose(
            standoff_xyz, gripper=GRIPPER_OPEN, pin_wrist_roll=wr, viewer=viewer)

        # 2) DESCEND: lower onto the cube.
        log["descend"] = self._move_to_pose(
            grasp_xyz, gripper=GRIPPER_OPEN, pin_wrist_roll=wr,
            settle_tol=0.01, viewer=viewer, straight=True)

        # 3) GRASP: close (separate step from the move).
        self._close_grip(viewer=viewer)

        # 4) LIFT: back to standoff, jaws closed.
        log["lift"] = self._move_to_pose(
            standoff_xyz, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr,
            viewer=viewer, straight=True)

        # Privileged grasp offset: where the held cube sits relative to the
        # pinch right now.  We use it so the CUBE (not the pinch) lands centred
        # over the base cube's top -- needed for the tight stack tolerance.
        offset = env.cube_pos(farther) - env.pinch_pos

        base_xyz = env.cube_pos(nearer)
        base_top_z = base_xyz[2] + env.cube_half
        # Desired held-cube centre when released: bottom face a small gap above
        # the base top.
        place_cube_centre = np.array([
            base_xyz[0], base_xyz[1],
            base_top_z + env.cube_half + STACK_RELEASE_GAP])
        stand_cube_centre = np.array([
            base_xyz[0], base_xyz[1], base_xyz[2] + STACK_STANDOFF_HEIGHT])

        pinch_place = place_cube_centre - offset
        pinch_stand = stand_cube_centre - offset

        # 5) OVER BASE: hover the held cube above the base cube.
        log["over_base"] = self._move_to_pose(
            pinch_stand, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr, viewer=viewer)

        # 6) LOWER: descend so the held cube is just above the base top.
        log["lower"] = self._move_to_pose(
            pinch_place, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr,
            settle_tol=0.01, viewer=viewer, straight=True)

        # 7) RELEASE: open and let the cube settle onto the base.
        self._hold(400, gripper=GRIPPER_OPEN, viewer=viewer)

        # 8) RETREAT: rise straight up so the gripper doesn't topple the stack.
        log["retreat"] = self._move_to_pose(
            pinch_stand, gripper=GRIPPER_OPEN, pin_wrist_roll=wr,
            viewer=viewer, straight=True)

        success = env.is_stacked()
        return dict(
            farther=farther, nearer=nearer, phase=log,
            success=bool(success),
            cube_a_final=env.cube_pos("a"),
            cube_b_final=env.cube_pos("b"),
        )


# --------------------------------------------------------------------------- #
#  Headless smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    env = MycobotStackEnv()
    if args.viewer:
        import mujoco.viewer
        env.reset(seed=args.seed)
        solver = MycobotStackSolver(env, speed=1.0)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.sync()
            log = solver.run_episode(viewer=viewer)
            print(log)
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
    else:
        solver = MycobotStackSolver(env)
        n_ok = 0
        for i in range(args.episodes):
            env.reset(seed=args.seed + i)
            t0 = time.perf_counter()
            log = solver.run_episode()
            dt = time.perf_counter() - t0
            n_ok += int(log["success"])
            print(f"ep {i:3d} farther={log['farther']} nearer={log['nearer']} "
                  f"success={log['success']}  {dt:.1f}s "
                  f"phases={ {k: v for k, v in log['phase'].items() if k not in ('farther', 'nearer')} }")
        n = args.episodes
        print(f"\n{n_ok}/{n} stacked = {100*n_ok/n:.1f}%")
