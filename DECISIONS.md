# Autonomous decisions — mycobot_stack run (2026-09-07)

Run executed under "no input" rules (same as cube_stack 2026-06-10). Every
decision point below was resolved without asking; reasoning logged here.

1. **Disk plan.** Box was at 97% (47G free) at run start vs a ~75 GB dataset
   with ~150 GB peak during collect+merge. Initial plan: stage worker shards
   in /dev/shm (180G tmpfs) + upload `rc_car_follow/data_hr` (86G, closed
   project) to the gdrive backup before deleting it locally (the sanctioned
   backup-verify-delete pattern). Superseded mid-run by the user directive to
   free space from the RSNA project: large regenerable data dirs were removed
   (raw/cache/derived reprs; 27,797 git-tracked files that went with them were
   restored from the rsna-knee git object store, repo intact). 385G free →
   default on-disk staging, no shm gymnastics needed. `data_hr` upload left
   running as a pure backup; nothing further is deleted this run.

2. **Base mount: plate flush with the table top (z = 0.15).** The myCobot 280
   is a desktop arm; real deployment puts base and cubes on the same surface.
   Pedestal (r=0.05, top at z=0.15) raises the base plate to exactly table-top
   height, so `table_top_z = 0.15` is unchanged from the LeArm scene and the
   sim matches the natural real-world setup. `BASE_HEIGHT_OFFSET = 0.0` is
   exposed in the env (added to the base body z at load) as the calibration
   knob. Shoulder (J2) sits at z = 0.15 + 0.15756 = 0.3076.

3. **Cube zone from wrist-down reach analysis, not copied.** With the wrist
   vertical, the planar shoulder-elbow chain (0.1104 + 0.096 m) plus the fixed
   lateral elbow offset (0.0646 m) gives max radial pinch reach ≈ 0.216 m at
   grasp height but only ≈ 0.188 m at the stack-standoff height (pinch ~0.245m
   high, wrist centre ~0.169 m above pinch). Zone chosen x∈[0.100, 0.160],
   y∈[-0.075, 0.055]: outer-corner radius 0.177 (11 mm margin at the worst
   height), area 78 cm² ≈ LeArm's 71.5 cm². To be verified/shrunk by the IK
   reach check (Step 2.4). Home pinch target (0.130, 0, 0.210); home_xy for
   the farther/nearer rule = (0.130, 0.0).

4. **Gripper model.** Parallel jaws per ticket spec: palm box + two box
   fingers on `joint6_flange`, fingers extend along flange +z (verified: the
   flange z-axis is the tool axis, pointing down at grasp). One slide joint
   `grip_left` (axis +y, range [-0.001, 0.0185]), right jaw mirrored -1 via
   equality. Inner pad faces at ±(s + 0.0015): max opening 40 mm at s=0.0185,
   GRIPPER_OPEN = 0.016 (33 mm opening, 7 mm/side clearance around the 19 mm
   cube), GRIPPER_CLOSE = -0.001 (commands ~12 mm over-travel past cube
   contact → grip force ≈ kp×0.012 ≈ 3.6 N at kp=300, ~10× cube weight with
   friction 3.0). Pad depth 19 mm ≈ ticket's "~20 mm jaw depth"; pinch site at
   fingertip midpoint z=0.058 in flange frame. Slide actuator kp=300 kv=5
   forcerange ±15: the LeArm's grip numbers (kp=25, hinge) are Nm/rad units;
   a slide needs N/m — value chosen for equivalent clamp force, not speed.

5. **Arm meshes visual-only.** The official XML ships duplicate visual +
   collision mesh geoms. The LeArm scene runs its arm entirely non-colliding
   (only the finger pads collide, with cubes only, never the table); the port
   keeps that exact collision filtering (pad contype4/conaff1, cube 1/3,
   env 2/1) and drops the arm collision meshes.

6. **gravcomp=1 on all arm/gripper bodies** (as in the LeArm scene): position
   servos hold pose without steady-state sag at kp=20, which the ticket says
   not to tune.

7. **Cameras shifted, not re-designed.** policy_cam/demo_cam keep the LeArm
   orientations (xyaxes verbatim) and viewing distance (~0.36 m to zone
   centre); positions translated by the zone-centre shift so framing and the
   verified pixel floor carry over. policy_cam (0.314, -0.252, 0.353);
   demo_cam (0.330, -0.560, 0.600). To be confirmed by renders + Gate B.

8. **Zone re-derived empirically after first reach check FAILED the analytic
   zone.** With the flush base, inner radii (r<0.15) fail wrist-down IK at
   grasp height (the arm cannot fold the tool vertical that close-in), while
   the outer bound is ~0.21 — the opposite of the analytic prediction. A base
   height sweep (0.15/0.18/0.21/0.24) showed RAISING the base only shrinks
   the feasible annulus (at +0.09 the grasp band vanishes), so the flush
   mount stands and the zone moved outward instead. Final zone
   x∈[0.150,0.195], y∈[-0.070,0.070] (63 cm², largest 0-failure candidate on
   a 7x7 grid x 3 heights). home_xy = zone centre = (0.1725, 0).

9. **Reach-check seeding mirrors the solver.** Grasp-height IK at one corner
   (0.150,+0.070) fails when seeded from HOME, but the FSM never solves a
   grasp from home: descend IK is seeded by continuation from the standoff
   solution directly above (and passes everywhere, <=0.75 mm). The check
   seeds grasp targets from the standoff solution accordingly; standoff and
   stack-standoff still verify from the home seed.

10. **GRASP_DROP = 0.009 (swept 0.004-0.012, failure-mode diagnosed).**
    Coarse sweep at gap 0.010: 0.004→75%, 0.008→92%, 0.012→100%. Fine sweep:
    0.006→53%, 0.0075→79% (all failures = missed_grasp: the 19 mm pads bite
    too high and the cube slips), 0.009→100%. 0.009 is the smallest 100/100
    value. Physical-realism check: the fingertip extends 2 mm past the pinch
    site, so drop 0.009 nominally clips the table plane by 1.5 mm (0.012
    would clip 4.5 mm — rejected). On the real desk this is absorbed by a
    2-3 mm mat under the workspace (HANDOVER calibration item).

11. **STACK_RELEASE_GAP = 0.006 (swept {0.006,0.010,0.014,0.020} x 100 eps
    at drop 0.009): all four 100/100 → per the ticket rule, smallest no-loss
    gap.** Unlike the LeArm (where 0.006 lost 15 pts to finger-shove), the
    280's shallower jaws keep the fingertip ~4.5 mm above the base-cube top
    at release, so the smallest gap is safe.

12. **Wrist-roll formula carried over unchanged.** Both arms roll the jaws
    about a downward approach axis with jaws closing along the pinch y-axis,
    so jaw_at_wr0 - wr algebra is identical; re-derived jaw_at_wr0 from the
    280's pinch frame and verified on 10 rendered seeds: worst jaw/cube
    misalignment 0.10 deg (jaw_check/).

13. **GATE A PASS: 100/100 stacked** (`mycobot_stack_solver.py --episodes 100
    --seed 0`), vs the 85% bar and the LeArm teacher's 87%. Episode length
    52-58 policy frames at rate 50 (LeArm ~51) → --max-frames-per-episode 160
    kept; eval --max-steps stays 170 (58 max frames + ~3x margin, matching
    cube_stack's 170-for-51 ratio).

14. **Cameras (final, supersedes #7 numbers):** policy_cam (0.357, -0.242,
    0.353), demo_cam (0.373, -0.550, 0.600), orientations verbatim from the
    LeArm scene, aimed at the final zone centre (0.1725, 0, 0.16). Framing
    verified by renders; pixel floor verified in Gate B.
