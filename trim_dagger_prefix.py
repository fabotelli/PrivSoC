"""
trim_dagger_prefix.py
=====================

Salvage pass over a `collect_dagger_mycobot_stack.py` dataset collected WITH
the (flawed) recorded policy prefix: the per-episode random handoff step k is
invisible to the observation, so prefix frames carry labels that switch
policy->solver on an unpredictable coin flip — bimodal chunk targets, val loss
plateaus ~0.10 vs ~0.008 on clean data, 0/100 closed loop.

k is exactly re-derivable (collector: `default_rng(seed + 777_777)
.integers(min_policy_steps, max_policy_steps + 1)`, one recorded frame per
policy control step), so each episode is trimmed to its solver-takeover
suffix: pure state-determined expert behavior from the policy-visited state.
No re-collection needed.

Suffixes shorter than --min-frames (already-stacked-at-handoff settle stubs)
are dropped.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_dagger_mycobot.npz")
    ap.add_argument("--out", default="dataset_dagger_recovery.npz")
    ap.add_argument("--min-policy-steps", type=int, default=0)
    ap.add_argument("--max-policy-steps", type=int, default=120)
    ap.add_argument("--min-frames", type=int, default=12,
                    help="drop episodes whose suffix is shorter than this "
                         "(train needs chunk+1 = 9 frames)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    z = np.load(args.data)
    starts = z["episode_starts"]
    lengths = z["episode_lengths"]
    seeds = z["episode_seeds"]
    E = len(lengths)

    ks = np.array([
        int(np.random.default_rng(int(s) + 777_777)
            .integers(args.min_policy_steps, args.max_policy_steps + 1))
        for s in seeds], dtype=np.int64)
    suffix_len = lengths - ks
    keep = suffix_len >= args.min_frames
    bad = suffix_len < 0
    if bad.any():
        raise SystemExit(f"{bad.sum()} episodes have k > length — seed/k "
                         f"derivation does not match this dataset, refusing.")

    print(f"{E} episodes: k mean {ks.mean():.1f}, prefix frames "
          f"{ks.sum()} of {int(lengths.sum())} "
          f"({100 * ks.sum() / lengths.sum():.1f}%); dropping "
          f"{int((~keep).sum())} short-suffix episodes", flush=True)

    idx = np.concatenate([
        np.arange(starts[i] + ks[i], starts[i] + lengths[i])
        for i in range(E) if keep[i]])

    print(f"loading images ({z['images_side'].nbytes/1e9:.1f} GB in the "
          f"source)...", flush=True)
    images = z["images_side"][idx]
    joints = z["joint_angles"][idx]

    new_lengths = suffix_len[keep].astype(np.int64)
    new_starts = (np.concatenate([[0], np.cumsum(new_lengths)[:-1]])
                  .astype(np.int64))
    episode_id = np.repeat(np.arange(len(new_lengths), dtype=np.int32),
                           new_lengths)

    payload = dict(
        images_side=images,
        joint_angles=joints,
        joint_names=z["joint_names"],
        episode_id=episode_id, episode_starts=new_starts,
        episode_lengths=new_lengths,
        episode_seeds=seeds[keep],
        episode_success=z["episode_success"][keep],
        episode_farther=z["episode_farther"][keep],
        episode_nearer=z["episode_nearer"][keep],
        subsample_rate=z["subsample_rate"],
        physics_timestep=z["physics_timestep"],
        image_hw=z["image_hw"],
        cam_lookat=z["cam_lookat"], cam_distance=z["cam_distance"],
        cam_azimuth=z["cam_azimuth"], cam_elevation=z["cam_elevation"],
    )
    print(f"writing {args.out} ({images.nbytes/1e9:.1f} GB)...", flush=True)
    np.savez(args.out, **payload)
    print(f"done in {time.perf_counter()-t0:.0f}s: kept "
          f"{int(keep.sum())}/{E} episodes, {len(idx)} frames "
          f"(mean suffix {new_lengths.mean():.1f})", flush=True)


if __name__ == "__main__":
    main()
