"""Record demo_solver_mycobot.mp4: one successful teacher rollout from
demo_cam, 512 px, 30 fps (captured every 16 substeps ~ real-time speed)."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse
import numpy as np
import mujoco
import imageio

from mycobot_stack_env import MycobotStackEnv
from mycobot_stack_solver import MycobotStackSolver


class VideoRecorder:
    def __init__(self, env, writer, cam, every=16, res=512):
        env.model.vis.global_.offwidth = res
        env.model.vis.global_.offheight = res
        self.renderer = mujoco.Renderer(env.model, height=res, width=res)
        self.writer = writer
        self.cam = cam
        self.every = every
        self._i = 0

    def capture(self, data):
        if self._i % self.every == 0:
            self.renderer.update_scene(data, camera=self.cam)
            self.writer.append_data(self.renderer.render())
        self._i += 1

    def close(self):
        self.renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_solver_mycobot.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-search", type=int, default=10)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    env = MycobotStackEnv()
    for off in range(args.max_search):
        seed = args.seed + off
        env.reset(seed=seed)
        writer = imageio.get_writer(args.out + ".tmp.mp4", fps=args.fps)
        rec = VideoRecorder(env, writer, "demo_cam")
        solver = MycobotStackSolver(env, recorder=rec)
        log = solver.run_episode()
        rec.close()
        writer.close()
        print(f"seed {seed}: success={log['success']}")
        if log["success"]:
            os.replace(args.out + ".tmp.mp4", args.out)
            print(f"saved {args.out}")
            return
    os.remove(args.out + ".tmp.mp4")
    raise SystemExit("no successful episode found")


if __name__ == "__main__":
    main()
