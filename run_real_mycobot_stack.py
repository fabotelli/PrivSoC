"""
run_real_mycobot_stack.py
=========================

DRAFT real-hardware runner: executes the trained single-cam chunked BC policy
(`bc_mycobot_stack.pt`) on a physical myCobot 280 over `pymycobot`, replacing
the sim render/step of `eval_bc_mycobot_stack.py` with a USB camera feed and
serial joint commands.  The inference loop (image /255 -> CHW, joints
standardised by the checkpoint's mean/std, ACT temporal ensemble decay 0.01,
10 Hz) is kept numerically identical to eval/training.

Needs: torch (CPU is fine), opencv-python, numpy, pymycobot — and this repo
on the path for the net class.  No MuJoCo required on the bench box.

Bring-up order (matches HANDOVER.md "Calibration items"):
  1. `--dry-run --preview`             : camera aligned?  The preview shows the
     exact 256x256 the policy sees; pass `--overlay ref.png` (a sim policy_cam
     frame) to blend at 50% and match the mount to the sim camera:
     position (0.357, -0.242, 0.353), lookat (0.1725, 0, 0.16), fovy 45 deg.
  2. `--sign-check`                    : wiggles each joint +5 deg at low speed,
     one at a time; verify each moves the same way as the sim (HANDOVER #5) and
     fix JOINT_SIGNS below if not.
  3. Calibrate GRIP_PCT_OPEN/CLOSE     : percent values where the real jaws
     match the sim's 33 mm open / full clamp (HANDOVER #7).
  4. Measure BASE_HEIGHT_OFFSET on the sim side (HANDOVER #1) — that constant
     lives in the ENV; if it changes, policies must be re-evaluated in sim.
  5. First live runs with `--servo-speed 30 --max-delta-rad 0.08` and a hand
     on the e-stop.  Ctrl-C stops motion (gripper is left as-is so a held
     cube is not dropped).

The policy expects the episode to start from the sim home configuration
(pinch ~60 mm above the zone centre, jaws open): pass `--go-home` to move
there slowly before the loop starts.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch

from train_bc_mycobot_stack import BCPolicySideChunk


# ---------------------------------------------------------------------- #
#  Baked-in sim constants (printed from MycobotStackEnv on 2026-09-07).
# ---------------------------------------------------------------------- #
# Episode start config (env.reset home IK solution), J1..J6 radians.
HOME_RADIANS = [0.38458, -0.49674, -2.28318, 1.20912, -0.00002, 0.38644]
# Joint limits (sim jnt_range == firmware limits of the 280).
ARM_LO = np.array([-2.9321, -2.0943, -2.6179, -2.5307, -2.8797, -3.14])
ARM_HI = np.array([2.9321, 2.0943, 2.6179, 2.5307, 2.8797, 3.14159])
# Sim gripper slide endpoints (metres) used for the obs/action mapping.
SIM_GRIP_OPEN = 0.016     # 33 mm jaw opening
SIM_GRIP_CLOSE = -0.001   # full clamp (over-travel supplies grip force)

# ---------------------------------------------------------------------- #
#  CALIBRATE ON THE BENCH (HANDOVER.md items 5 and 7).
# ---------------------------------------------------------------------- #
# pymycobot J1..J6 are index-for-index with the sim joints; flip a sign here
# if the low-speed sign check shows a joint moving the opposite way.
JOINT_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
JOINT_OFFSETS = np.zeros(6)   # radians, real = sign * sim + offset
# pymycobot gripper percent at the two sim endpoints.
GRIP_PCT_OPEN = 100.0         # jaws at 33 mm
GRIP_PCT_CLOSE = 0.0          # full clamp


def sim_to_real_radians(q_sim: np.ndarray) -> list[float]:
    return (JOINT_SIGNS * q_sim + JOINT_OFFSETS).tolist()


def real_to_sim_radians(q_real) -> np.ndarray:
    q = np.asarray(q_real, dtype=np.float32)
    return (q - JOINT_OFFSETS) / JOINT_SIGNS


def grip_m_to_pct(m: float) -> float:
    t = (m - SIM_GRIP_CLOSE) / (SIM_GRIP_OPEN - SIM_GRIP_CLOSE)
    return float(np.clip(GRIP_PCT_CLOSE + t * (GRIP_PCT_OPEN - GRIP_PCT_CLOSE),
                         min(GRIP_PCT_CLOSE, GRIP_PCT_OPEN),
                         max(GRIP_PCT_CLOSE, GRIP_PCT_OPEN)))


def grip_pct_to_m(pct: float) -> float:
    t = (pct - GRIP_PCT_CLOSE) / (GRIP_PCT_OPEN - GRIP_PCT_CLOSE)
    return SIM_GRIP_CLOSE + t * (SIM_GRIP_OPEN - SIM_GRIP_CLOSE)


def _load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    model = BCPolicySideChunk(n_joints=a["n_joints"], in_ch=a["in_ch"],
                              chunk=a["chunk"], img_hw=tuple(a["img_hw"]),
                              bottleneck=a["bottleneck"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    mean = np.asarray(ck["joint_mean"], dtype=np.float32)
    std = np.asarray(ck["joint_std"], dtype=np.float32)
    return model, mean, std, tuple(a["img_hw"]), int(a["chunk"])


class Camera:
    """USB camera -> the policy's square RGB frame.  Crop BEFORE resize so the
    aspect ratio matches the sim's square render."""

    def __init__(self, index, img_hw, crop=None):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera index {index} did not open")
        self.img_hw = img_hw
        self.crop = crop

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if self.crop is not None:
            x, y, w, h = self.crop
            frame = frame[y:y + h, x:x + w]
        else:
            H, W = frame.shape[:2]
            s = min(H, W)
            frame = frame[(H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2]
        frame = self.cv2.resize(frame, (self.img_hw[1], self.img_hw[0]),
                                interpolation=self.cv2.INTER_AREA)
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class Arm:
    """Thin pymycobot wrapper; measured joint angles are fed to the policy
    (HANDOVER #6), falling back to the last good read on serial glitches."""

    def __init__(self, port, baud, servo_speed):
        try:
            from pymycobot import MyCobot280 as MC
        except ImportError:
            from pymycobot.mycobot import MyCobot as MC
        self.mc = MC(port, baud)
        self.servo_speed = servo_speed
        self._last_q = None

    def get_radians_sim(self) -> np.ndarray:
        q = None
        for _ in range(3):
            q = self.mc.get_radians()
            if isinstance(q, list) and len(q) == 6:
                break
            q = None
            time.sleep(0.005)
        if q is None:
            if self._last_q is None:
                raise RuntimeError("no joint feedback from the arm")
            return self._last_q
        self._last_q = real_to_sim_radians(q)
        return self._last_q

    def send_radians_sim(self, q_sim: np.ndarray):
        self.mc.send_radians(sim_to_real_radians(q_sim), self.servo_speed)

    def set_gripper_pct(self, pct: float):
        self.mc.set_gripper_value(int(round(pct)), self.servo_speed)

    def stop(self):
        try:
            self.mc.stop()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="bc_mycobot_stack.pt")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--crop", default=None,
                    help="x,y,w,h camera crop (default: centre square)")
    ap.add_argument("--rate-hz", type=float, default=10.0,
                    help="control rate; sim trained at 10 Hz")
    ap.add_argument("--steps", type=int, default=280,
                    help="control steps to run (matches eval max_steps)")
    ap.add_argument("--ensemble-decay", type=float, default=0.01)
    ap.add_argument("--servo-speed", type=int, default=50,
                    help="pymycobot speed 0-100; HANDOVER #6: each 10 Hz step "
                         "target must be reached in <100 ms")
    ap.add_argument("--max-delta-rad", type=float, default=0.15,
                    help="safety slew clamp per control step, per joint")
    ap.add_argument("--go-home", action="store_true",
                    help="move slowly to the sim episode-start pose first")
    ap.add_argument("--dry-run", action="store_true",
                    help="no serial: print the commands instead")
    ap.add_argument("--preview", action="store_true",
                    help="show the exact policy input frame in a window")
    ap.add_argument("--overlay", default=None,
                    help="path of a sim policy_cam frame to 50%%-blend into "
                         "the preview for camera alignment")
    ap.add_argument("--grip-feedback", action="store_true",
                    help="read gripper percent from the arm instead of echoing "
                         "the last command (many 280 grippers report -1)")
    ap.add_argument("--sign-check", action="store_true",
                    help="wiggle each joint +5 deg one at a time and exit")
    args = ap.parse_args()

    torch.set_num_threads(2)
    model, mean, std, img_hw, chunk = _load_policy(args.policy)
    decay = float(args.ensemble_decay)

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    cam = Camera(args.camera_index, img_hw, crop)
    overlay = None
    if args.overlay:
        import cv2
        ov = cv2.imread(args.overlay)
        overlay = cv2.resize(ov, (img_hw[1], img_hw[0]))

    arm = None if args.dry_run else Arm(args.port, args.baud, args.servo_speed)

    if args.sign_check:
        if arm is None:
            raise SystemExit("--sign-check needs the arm (drop --dry-run)")
        q0 = arm.get_radians_sim()
        for j in range(6):
            q = q0.copy()
            q[j] += np.deg2rad(5.0)
            print(f"J{j+1}: +5 deg — should match the sim's positive direction")
            arm.send_radians_sim(q)
            time.sleep(2.0)
            arm.send_radians_sim(q0)
            time.sleep(2.0)
        return

    if args.go_home and arm is not None:
        print("moving to sim home pose (slow)...")
        arm.mc.send_radians(sim_to_real_radians(np.array(HOME_RADIANS)), 20)
        time.sleep(4.0)
        arm.set_gripper_pct(grip_m_to_pct(SIM_GRIP_OPEN))
        time.sleep(2.0)

    q_cmd = (arm.get_radians_sim() if arm is not None
             else np.array(HOME_RADIANS, dtype=np.float32))
    grip_pct_cmd = grip_m_to_pct(SIM_GRIP_OPEN)
    chunks_buf: deque[np.ndarray] = deque(maxlen=chunk)
    dt = 1.0 / args.rate_hz
    t_next = time.monotonic()

    try:
        for step in range(args.steps):
            frame = cam.read()

            if args.preview:
                import cv2
                shown = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if overlay is not None:
                    shown = cv2.addWeighted(shown, 0.5, overlay, 0.5, 0)
                cv2.imshow("policy input", shown)
                cv2.waitKey(1)

            q_meas = (arm.get_radians_sim() if arm is not None else q_cmd)
            if args.grip_feedback and arm is not None:
                g = arm.mc.get_gripper_value()
                grip_m = (grip_pct_to_m(g) if isinstance(g, (int, float))
                          and g >= 0 else grip_pct_to_m(grip_pct_cmd))
            else:
                grip_m = grip_pct_to_m(grip_pct_cmd)
            joints = np.append(q_meas, grip_m).astype(np.float32)

            s_t = torch.from_numpy(frame.astype(np.float32) / 255.0) \
                       .permute(2, 0, 1).contiguous().unsqueeze(0)
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

            q_target = np.clip(action[:6], ARM_LO, ARM_HI)
            q_target = q_cmd + np.clip(q_target - q_cmd,
                                       -args.max_delta_rad, args.max_delta_rad)
            grip_target_pct = grip_m_to_pct(float(action[6]))

            if arm is not None:
                arm.send_radians_sim(q_target)
                if abs(grip_target_pct - grip_pct_cmd) >= 2.0:
                    arm.set_gripper_pct(grip_target_pct)
                    grip_pct_cmd = grip_target_pct
            else:
                print(f"[{step:3d}] q={np.round(q_target, 3)} "
                      f"grip={grip_target_pct:5.1f}%")
                grip_pct_cmd = grip_target_pct
            q_cmd = q_target

            t_next += dt
            lag = t_next - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -dt:
                print(f"WARN: control loop {-lag*1000:.0f} ms behind "
                      f"(camera/serial too slow for {args.rate_hz} Hz)")
                t_next = time.monotonic()
        print("done: step budget reached; arm holding last target.")
    except KeyboardInterrupt:
        print("\ninterrupted: stopping motion (gripper left as-is).")
        if arm is not None:
            arm.stop()
    finally:
        cam.close()


if __name__ == "__main__":
    main()
