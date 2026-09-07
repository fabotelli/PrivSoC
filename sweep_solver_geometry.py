"""Step 3 geometry sweeps for the myCobot solver:
  1) GRASP_DROP sweep (new 19 mm jaw depth) at STACK_RELEASE_GAP=0.010,
  2) STACK_RELEASE_GAP sweep {0.006, 0.010, 0.014, 0.020} at the best drop,
100 episodes each (seeds 0..99, no DR), phase-failure breakdown, and the
episode length in policy frames (substeps / rate 50) for --max-frames /
eval --max-steps budgeting."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco

import mycobot_stack_solver as S
from mycobot_stack_env import MycobotStackEnv


class FrameCounter:
    """Recorder stub: counts solver substeps -> policy frames at rate 50."""
    def __init__(self):
        self.substeps = 0
    def capture(self, data):
        self.substeps += 1


def run_block(drop, gap, n=100, seed0=0):
    S.GRASP_DROP = drop
    S.STACK_RELEASE_GAP = gap
    env = MycobotStackEnv()
    ok, frames, phase_fail = 0, [], {}
    for i in range(n):
        env.reset(seed=seed0 + i)
        fc = FrameCounter()
        solver = S.MycobotStackSolver(env, recorder=fc)
        log = solver.run_episode()
        ok += int(log["success"])
        frames.append(fc.substeps / 50.0)
        if not log["success"]:
            for ph, v in log["phase"].items():
                if ph in ("farther", "nearer"):
                    continue
                if v is False:
                    phase_fail[ph] = phase_fail.get(ph, 0) + 1
    return ok, np.array(frames), phase_fail


print("=== GRASP_DROP sweep (gap=0.010, 100 eps, seeds 0-99) ===", flush=True)
results = {}
for drop in (0.004, 0.008, 0.012):
    ok, fr, pf = run_block(drop, 0.010)
    results[drop] = ok
    print(f"  drop={drop:.3f}: {ok}/100 stacked  "
          f"frames mean={fr.mean():.1f} max={fr.max():.1f}  ik_fails={pf}",
          flush=True)
best_drop = max(results, key=results.get)
print(f"best GRASP_DROP = {best_drop}")

print(f"\n=== STACK_RELEASE_GAP sweep (drop={best_drop}, 100 eps) ===", flush=True)
gres = {}
for gap in (0.006, 0.010, 0.014, 0.020):
    ok, fr, pf = run_block(best_drop, gap)
    gres[gap] = ok
    print(f"  gap={gap:.3f}: {ok}/100 stacked  "
          f"frames mean={fr.mean():.1f} max={fr.max():.1f}  ik_fails={pf}",
          flush=True)
best = max(gres.values())
# smallest gap with no loss vs best
best_gap = min(g for g, v in gres.items() if v == best)
print(f"best (smallest no-loss) STACK_RELEASE_GAP = {best_gap} ({best}/100)")
