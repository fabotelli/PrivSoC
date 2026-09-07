"""
collect_dagger_mycobot_stack.py
===============================

On-policy DAgger data collector for the two-cube STACKING task (myCobot 280).

The end-to-end BC policy misses the grasp 44.5% of the time and lands only 69%
of its grasps (final eval, 200 eps).  The teacher solver is 100%, but it only
ever sees its OWN clean rollouts -> the student never learns to recover from
the states its imperfect approach actually produces.  Same wall, same fix as
cube_staircase's C-phase DAgger (0->88.5%): expert correction FROM THE POLICY'S
OWN on-distribution states, combined in TRAINING, not at a runtime seam.

Per episode (vs the phase-boundary handoff of the staircase, the failing phase
here is the FIRST one, so the handoff point is randomised):

  PHASE 0 -- the trained POLICY drives for k control steps, k ~ U[min,max]
             per episode (exact replica of eval_bc_mycobot_stack's
             temporal-ensemble inference), recorded at the solver's capture
             cadence.  Detectably-unrecoverable states (cube off the table /
             out of the IK-feasible patch / wrong cube lifted) abort early.
  HANDOFF -- classify the policy's state:
               * already stacked        -> settle, done (rare, kept);
               * farther cube IN HAND   -> solver runs its place-half
                 (over_base -> lower -> release -> retreat) with the
                 privileged grasp offset, wrist roll pinned where the policy
                 left it;
               * cubes on the table     -> solver runs its full FSM
                 (it re-reads privileged cube positions, so nudged/displaced
                 cubes are re-approached from wherever they now are);
               * anything else          -> abort.
  Episodes are kept only if the stack succeeds (--successes-only), so the
  dataset is on-policy and self-consistent: labels are future-achieved qpos,
  and near the seam they encode the solver's recovery from policy states.

This file is a copy of ``collect_dataset_mycobot_stack.py`` with ONLY the
per-episode execution changed (+ a per-worker seed-policy load).  DatasetLogger,
merge, checkpoint/resume and CLI are otherwise identical.
"""

from __future__ import annotations

import os
import sys
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
# Single-thread BLAS so N osmesa CPU workers don't thrash each other.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import multiprocessing as mp
import time
from collections import deque

import numpy as np
import torch
import mujoco

from mycobot_stack_env import (MycobotStackEnv, ARM_JOINTS, WRIST_ROLL_IDX,
                               GRIPPER_OPEN, GRIPPER_CLOSE,
                               CUBE_X_LO, CUBE_X_HI, CUBE_Y_LO, CUBE_Y_HI)
from mycobot_stack_solver import (MycobotStackSolver, STACK_STANDOFF_HEIGHT,
                                  STACK_RELEASE_GAP)
from train_bc_mycobot_stack import BCPolicySideChunk
from collect_dataset_mycobot_stack import (DatasetLogger, _dr_setup, _dr_apply,
                                           fmt_hms, _close_mmap, _rm,
                                           _write_checkpoint, _load_checkpoint,
                                           merge_to_npz, _purge_temps,
                                           CAM_LOOKAT, CAM_DISTANCE,
                                           CAM_AZIMUTH, CAM_ELEVATION,
                                           JOINT_NAMES, N_JOINTS)


# Soft recoverability margin around the IK-feasible spawn patch: the solver can
# still pick a cube nudged a little outside the zone, but far outside it the
# wrist-down IK has no solution and the episode would burn its full budget.
ZONE_MARGIN = 0.03


def _load_seed_policy(path):
    """Load the single-cam seed policy onto CPU (per worker).  Returns
    (model, mean, std, img_hw, chunk)."""
    device = torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    model = BCPolicySideChunk(n_joints=a["n_joints"], in_ch=a["in_ch"],
                              chunk=a["chunk"], img_hw=tuple(a["img_hw"]),
                              bottleneck=a["bottleneck"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    mean = np.asarray(ck["joint_mean"], dtype=np.float32)
    std = np.asarray(ck["joint_std"], dtype=np.float32)
    return model, mean, std, tuple(a["img_hw"]), int(a["chunk"])


def _abort_state(env, farther, nearer) -> bool:
    """Detectably-unrecoverable mid-rollout states -> stop burning budget."""
    fp = env.cube_pos(farther)
    pn = env.cube_pos(nearer)
    table = env.table_top_z
    # A cube fell off the table edge.
    if fp[2] < table - 0.02 or pn[2] < table - 0.02:
        return True
    # Wrong cube lifted (the nearer/base cube must stay down).
    if pn[2] > table + env.cube_half + 0.02:
        return True
    # Farther cube shoved out of the IK-feasible patch (solver can't re-pick).
    # Elevated = probably in-hand mid-carry, so only the on-table case aborts.
    in_zone = (CUBE_X_LO - ZONE_MARGIN <= fp[0] <= CUBE_X_HI + ZONE_MARGIN
               and CUBE_Y_LO - ZONE_MARGIN <= fp[1] <= CUBE_Y_HI + ZONE_MARGIN)
    elevated = fp[2] > table + env.cube_half + 0.012
    if not in_zone and not elevated:
        return True
    return False


def _place_from_carry(env, solver, farther, nearer) -> None:
    """Solver's place-half (run_episode steps 5-8) from a mid-carry takeover:
    privileged grasp offset read NOW, wrist roll pinned where the policy left
    it (re-rolling a held cube would twist it out of the jaws)."""
    wr = float(env.arm_qpos_now[WRIST_ROLL_IDX])
    offset = env.cube_pos(farther) - env.pinch_pos

    base_xyz = env.cube_pos(nearer)
    base_top_z = base_xyz[2] + env.cube_half
    place_cube_centre = np.array([
        base_xyz[0], base_xyz[1],
        base_top_z + env.cube_half + STACK_RELEASE_GAP])
    stand_cube_centre = np.array([
        base_xyz[0], base_xyz[1], base_xyz[2] + STACK_STANDOFF_HEIGHT])
    pinch_place = place_cube_centre - offset
    pinch_stand = stand_cube_centre - offset

    solver._move_to_pose(pinch_stand, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr)
    solver._move_to_pose(pinch_place, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr,
                         settle_tol=0.01, straight=True)
    solver._hold(400, gripper=GRIPPER_OPEN)
    solver._move_to_pose(pinch_stand, gripper=GRIPPER_OPEN, pin_wrist_roll=wr,
                         straight=True)


def run_dagger_episode(env, solver, policy_bundle, logger, cfg, rng) -> dict:
    """PHASE 0: policy drives k steps (recorded), then the solver completes
    from the policy's own state (recorded).  Returns {"success": bool}."""
    model, mean, std, img_hw, chunk = policy_bundle
    decay = float(cfg["ensemble_decay"])
    arm_lo, arm_hi = env.arm_range[:, 0], env.arm_range[:, 1]
    grip_lo, grip_hi = env.model.actuator_ctrlrange[env.grip_act]
    renderer = logger.renderer
    qadr = logger.joint_qadr
    farther, nearer = env.pick_order()

    k_handoff = int(rng.integers(cfg["min_policy_steps"],
                                 cfg["max_policy_steps"] + 1))

    # ---- PHASE 0: the policy drives (exact eval-time inference) ---- #
    chunks_buf: deque[np.ndarray] = deque(maxlen=chunk)
    for _ in range(k_handoff):
        renderer.update_scene(env.data, camera="policy_cam")
        s_full = renderer.render().astype(np.float32) / 255.0
        s_t = torch.from_numpy(s_full).permute(2, 0, 1).contiguous().unsqueeze(0)
        joints = env.data.qpos[qadr].astype(np.float32)
        jin = torch.from_numpy((joints - mean) / std).unsqueeze(0)
        with torch.no_grad():
            pred_chunk = model(s_t, jin).squeeze(0).numpy()
        pred_chunk = pred_chunk * std + mean

        chunks_buf.append(pred_chunk)
        n = len(chunks_buf)
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights /= weights.sum()
        stacked = np.stack([c[n - 1 - i] for i, c in enumerate(chunks_buf)])
        action = (weights[:, None] * stacked).sum(axis=0)

        env.set_arm_target(np.clip(action[:6], arm_lo, arm_hi))
        env.set_gripper(float(np.clip(action[6], grip_lo, grip_hi)))
        for _ in range(cfg["rate"]):
            env.step()
            logger.capture(env.data)

        if _abort_state(env, farther, nearer):
            return {"success": False}

    # ---- HANDOFF: classify the policy's state ---- #
    if env.is_stacked():
        solver._hold(150, gripper=GRIPPER_OPEN)          # settle + confirm
        return {"success": bool(env.is_stacked())}

    fp = env.cube_pos(farther)
    elevated = fp[2] > env.table_top_z + env.cube_half + 0.012
    in_hand = elevated and np.linalg.norm(fp - env.pinch_pos) < 0.035

    if in_hand:
        _place_from_carry(env, solver, farther, nearer)
    elif elevated:
        return {"success": False}   # airborne/perched but not held: no branch
    else:
        solver.run_episode()        # re-reads privileged positions; full re-pick

    return {"success": bool(env.is_stacked())}


def collect_chunk(task):
    worker_id, ep_start, ep_count, base_seed, cfg, progress, gl_lock = task

    torch.set_num_threads(1)

    side_path = cfg["side_img_tmpl"].format(w=worker_id)
    jnt_path = cfg["jnt_tmpl"].format(w=worker_id)
    ckpt_path = cfg["ckpt_tmpl"].format(w=worker_id)

    i_start, cursor0, n_success = 0, 0, 0
    lengths, seeds, success = [], [], []
    farther, nearer = [], []
    resume = False
    if (cfg["resume"] and os.path.exists(ckpt_path)
            and os.path.exists(side_path) and os.path.exists(jnt_path)):
        try:
            ck = _load_checkpoint(ckpt_path)
            i_start, cursor0, n_success = ck["i_done"], ck["cursor"], ck["n_success"]
            lengths, seeds, success = ck["lengths"], ck["seeds"], ck["success"]
            farther, nearer = ck["farther"], ck["nearer"]
            resume = True
        except Exception:
            i_start, cursor0, n_success = 0, 0, 0
            lengths, seeds, success = [], [], []
            farther, nearer = [], []
            resume = False

    env = MycobotStackEnv()
    with gl_lock:
        logger = DatasetLogger(
            env, capacity=ep_count * cfg["max_frames"], rate=cfg["rate"],
            height=cfg["res"], width=cfg["res"],
            side_img_path=side_path, joints_path=jnt_path,
            resume=resume, start_cursor=cursor0)
    solver = MycobotStackSolver(env, recorder=logger)
    policy_bundle = _load_seed_policy(cfg["policy_path"])
    dr = _dr_setup(env.model)

    progress[worker_id] = i_start
    last_ckpt = i_start
    for i in range(i_start, ep_count):
        seed = base_seed + ep_start + i
        env.reset(seed=seed)
        _dr_apply(env.model, np.random.default_rng(seed), dr)
        mujoco.mj_forward(env.model, env.data)
        ep_rng = np.random.default_rng(seed + 777_777)   # handoff-k stream
        fa, ne = env.pick_order()
        start = logger.start_episode()
        log = run_dagger_episode(env, solver, policy_bundle, logger, cfg, ep_rng)
        s = bool(log["success"])
        n_success += s
        if cfg["successes_only"] and not s:
            logger.rewind(start)
        else:
            lengths.append(logger.cursor - start)
            seeds.append(seed)
            success.append(s)
            farther.append(fa)
            nearer.append(ne)
        progress[worker_id] = i + 1
        if (i + 1) - last_ckpt >= cfg["checkpoint_every"]:
            logger.flush()
            _write_checkpoint(ckpt_path, i + 1, logger.cursor,
                              lengths, seeds, success, n_success, farther, nearer)
            last_ckpt = i + 1

    logger.flush()
    _write_checkpoint(ckpt_path, ep_count, logger.cursor,
                      lengths, seeds, success, n_success, farther, nearer)
    logger.close()
    return dict(worker_id=worker_id, n_frames=logger.cursor,
                side_path=side_path, jnt_path=jnt_path, ckpt_path=ckpt_path,
                lengths=lengths, seeds=seeds, success=success,
                farther=farther, nearer=nearer,
                n_run=ep_count, n_success=n_success, overflowed=logger.overflowed)


def _run_signature(args):
    return dict(episodes=args.episodes, workers=args.workers, seed=args.seed,
                rate=args.rate, resolution=args.resolution,
                max_frames=args.max_frames_per_episode,
                successes_only=bool(args.successes_only),
                policy=os.path.basename(args.policy),
                ensemble_decay=args.ensemble_decay,
                min_policy_steps=args.min_policy_steps,
                max_policy_steps=args.max_policy_steps,
                out=os.path.basename(args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--successes-only", action="store_true", default=True)
    ap.add_argument("--no-successes-only", action="store_false",
                    dest="successes_only")
    ap.add_argument("--max-frames-per-episode", type=int, default=300)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out", default="dataset_dagger_mycobot.npz")
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--merge-tmpdir", default=None)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    # ---- DAgger additions ---- #
    ap.add_argument("--policy", required=True,
                    help="seed BC policy that drives PHASE 0")
    ap.add_argument("--ensemble-decay", type=float, default=0.01,
                    help="ACT temporal-ensemble decay for the policy inference")
    ap.add_argument("--min-policy-steps", type=int, default=0,
                    help="lower bound of the per-episode random handoff step")
    ap.add_argument("--max-policy-steps", type=int, default=120,
                    help="upper bound of the per-episode random handoff step "
                         "(capped low so kept episodes don't contain long "
                         "failure-hover prefixes that BC would clone)")
    args = ap.parse_args()

    if args.episodes <= 0:
        ap.error("--episodes must be positive")
    if args.workers <= 0 or args.workers > args.episodes:
        ap.error("--workers must be in 1..episodes")
    if args.rate <= 0:
        ap.error("--rate must be positive")
    if args.resolution <= 0 or args.resolution > 720:
        ap.error("--resolution must be in 1..720")
    if args.max_frames_per_episode <= 0 or args.checkpoint_every <= 0:
        ap.error("--max-frames-per-episode and --checkpoint-every must be positive")
    if not (0 <= args.min_policy_steps <= args.max_policy_steps):
        ap.error("need 0 <= --min-policy-steps <= --max-policy-steps")
    if not os.path.exists(args.policy):
        ap.error(f"--policy {args.policy} not found")

    res, W, E = args.resolution, args.workers, args.episodes

    chunks, start = [], 0
    for w in range(W):
        cnt = E // W + (1 if w < E % W else 0)
        chunks.append((w, start, cnt))
        start += cnt

    out_dir = os.path.dirname(os.path.abspath(args.out))
    tmpdir = args.tmpdir or out_dir
    os.makedirs(tmpdir, exist_ok=True)
    merge_dir = args.merge_tmpdir or tmpdir
    os.makedirs(merge_dir, exist_ok=True)
    base = os.path.basename(args.out)
    manifest = os.path.join(out_dir, base + ".run.json")
    cfg = dict(
        rate=args.rate, res=res, max_frames=args.max_frames_per_episode,
        successes_only=args.successes_only, checkpoint_every=args.checkpoint_every,
        policy_path=os.path.abspath(args.policy),
        ensemble_decay=args.ensemble_decay,
        min_policy_steps=args.min_policy_steps,
        max_policy_steps=args.max_policy_steps,
        side_img_tmpl=os.path.join(tmpdir, base + ".w{w}.images_side.npy"),
        jnt_tmpl=os.path.join(tmpdir, base + ".w{w}.joints.npy"),
        ckpt_tmpl=os.path.join(tmpdir, base + ".w{w}.ckpt.npz"),
        merge_side=os.path.join(merge_dir, base + ".merge.images_side.npy"),
        merge_jnt=os.path.join(merge_dir, base + ".merge.joints.npy"),
    )

    sig = _run_signature(args)
    if args.fresh:
        _purge_temps(cfg, manifest)
    resuming = False
    if os.path.exists(manifest):
        try:
            with open(manifest) as f:
                saved = json.load(f)
        except Exception:
            saved = None
        if saved == sig:
            resuming = True
        else:
            ap.error(f"{manifest} is from a different config; pass --fresh to "
                     f"restart or use matching arguments.")
    if not resuming:
        _purge_temps(cfg, manifest)
        with open(manifest, "w") as f:
            json.dump(sig, f, indent=2)
    cfg["resume"] = resuming

    timestep = float(MycobotStackEnv().model.opt.timestep)

    per_worker_gb = (max(c for _, _, c in chunks) * args.max_frames_per_episode
                     * res * res * 3 / 1e9)
    print(f"{'RESUMING' if resuming else 'Collecting'} {E} DAgger episodes "
          f"across {W} workers @ {res}x{res} px, every {args.rate} substeps "
          f"(~{500.0 / args.rate:.0f} Hz).  "
          f"policy={os.path.basename(args.policy)} decay={args.ensemble_decay} "
          f"handoff k~U[{args.min_policy_steps},{args.max_policy_steps}]"
          f"{'  [successful only]' if args.successes_only else ''}", flush=True)
    print(f"Per-worker episodes: {[c for _, _, c in chunks]}; checkpoint every "
          f"{args.checkpoint_every}; temps in {tmpdir} (~{per_worker_gb:.1f} GB/worker).",
          flush=True)

    t0 = time.perf_counter()
    with mp.Manager() as manager:
        progress = manager.list([0] * W)
        gl_lock = manager.Lock()
        with mp.Pool(W) as pool:
            tasks = [(w, s, c, args.seed, cfg, progress, gl_lock)
                     for (w, s, c) in chunks]
            async_results = [pool.apply_async(collect_chunk, (t,)) for t in tasks]
            pool.close()

            done0, last_bucket = None, -1
            while True:
                ready = all(r.ready() for r in async_results)
                done = sum(progress)
                if done0 is None:
                    done0 = done
                bucket = done // args.progress_every
                if bucket != last_bucket or ready:
                    elapsed = time.perf_counter() - t0
                    eps = (done - done0) / elapsed if elapsed > 0 else 0.0
                    eta = (E - done) / eps if eps > 0 else 0.0
                    print(f"  [{done:>6d}/{E}] workers={list(progress)} "
                          f"{eps:4.2f} ep/s  elapsed {fmt_hms(elapsed)}  "
                          f"ETA {fmt_hms(eta)}", flush=True)
                    last_bucket = bucket
                if ready:
                    break
                time.sleep(5)

            results = [r.get() for r in async_results]
            pool.join()

    n_run = sum(r["n_run"] for r in results)
    n_success = sum(r["n_success"] for r in results)
    n_kept = sum(len(r["lengths"]) for r in results)
    if any(r["overflowed"] for r in results):
        print("WARNING: a worker hit its frame capacity; some tail frames were "
              "dropped. Re-run --fresh with a larger --max-frames-per-episode.",
              flush=True)

    print(f"\nAll workers done in {fmt_hms(time.perf_counter() - t0)}. "
          f"Kept {n_kept} episodes. Merging into {args.out} ...", flush=True)
    total = merge_to_npz(results, args.out, cfg, timestep, args.compress)
    _rm(manifest)

    size_gb = os.path.getsize(args.out) / 1e9
    print(f"Saved {args.out} ({size_gb:.2f} GB) in "
          f"{fmt_hms(time.perf_counter() - t0)}: {total} frames from {n_kept} "
          f"episodes ({100.0 * n_success / max(n_run, 1):.1f}% solve rate).",
          flush=True)


if __name__ == "__main__":
    main()
