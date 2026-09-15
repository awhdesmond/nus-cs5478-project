# Vision-Guided Contact-Rich Connector Insertion — Final Project Plan

**Course:** Intelligent Robots — Algorithms and Systems
**Duration:** 8 weeks (6 originally; extended after week 5 — see weeks 6–7)
**Available time:** ~10–16 hrs/week (2–3 weeknights + weekend blocks) → ~60–95 hrs total

---

## 1. Task

*This section is the original (weeks 1–5) insertion task, which remains the "insertion module". The extended
task — perceive, identify and pick two keyed connectors, insert each into its own socket — is defined under
"Weeks 6–7". Lessons from both are collected in §10.*

A 6-DOF manipulator inserts a **rigid wire terminal / connector** into a socket with
**1–3 mm target clearance**, from a randomised pre-insertion pose with bounded
uncertainty (±5–10 mm translation, ±5° rotation).

The cable behind the connector is **not** modelled as deformable — the stiff region near
the plug is treated as rigid. This keeps the project focused on contact-rich control
rather than deformable-object simulation.

**Uncertainty semantics:** the randomised offset is between the socket pose the controller
*believes* and the *true* socket pose. The controller never reads ground-truth socket pose —
only the success checker does. Otherwise the evaluation is trivially solvable in sim.

**Geometry:** the socket has a small chamfer / lead-in (as real connectors do). Start with a
**round peg** (yaw-invariant), then ratchet to a rectangular connector once round-peg
insertion is reliable.

**Success:** connector reaches full seated depth without exceeding a contact-force cap.

## 2. Technical stack

| Choice | Rationale |
|---|---|
| **MuJoCo** (not PyBullet) | More accurate, numerically stable contact/friction at mm-scale clearances — the algorithmic contribution *is* the contact behaviour |
| **`mujoco_menagerie` UR5e** | Pre-built MJCF; do not hand-write arm models. UR5e matches the 6-DOF task statement and the Intrinsic setup (Franka is 7-DOF → needs null-space handling in IK/impedance) |
| **Torque actuators** | Menagerie models ship with position-servo actuators. Replace them with `motor` (torque) actuators so impedance control is possible; gravity/Coriolis compensation via `qfrc_bias` |
| **Socket from primitive geoms** | MuJoCo uses the **convex hull** of mesh collision geoms — a socket mesh gets its hole filled. Build the socket from box geoms (or use a convex decomposition). Same applies to any Intrinsic toolkit assets: visual mesh only |
| **Not Gazebo/ROS2** | ROS2 bring-up competes with a tight time budget; the "systems" content is the controller + harness, not middleware |

## 3. Components

**Algorithms**
1. Jacobian-based IK (damped least squares) + quintic/trapezoidal joint-space trajectory generation — used for the **free-space approach phase**: IK solves the pre-insertion joint config once (from the *believed* socket pose), a joint trajectory is tracked by joint PD + `qfrc_bias`. Control then switches to impedance for the contact phase. Shares the Jacobian (`mj_jacSite`) with the impedance controller.
2. Cartesian impedance controller (torque-level: `τ = Jᵀ(K·Δx − D·ẋ) + qfrc_bias`) — **the core contribution**. Low rotational stiffness so contact torques can realign the plug.
   Simulated wrist F/T sensor used for contact detection, the force cap, and (optionally) force-regulated insertion along z.
   *Note:* impedance control does not need F/T sensing; admittance does. Admittance on stiff position servos against stiff contact with noisy sim F/T is prone to instability — avoid unless impedance fails.
3. Contact-triggered spiral search for alignment under pose uncertainty (translation only — rotation handled by chamfer + rotational compliance)

**Systems**
4. Configurable MuJoCo environment (clearance, uncertainty, friction)
5. Automated evaluation harness — N trials per condition, logging success / cycle time / peak force

---

## 5. Schedule (Plan B — committed)

### Week 1 — Environment + harness + easy-mode success
*Weeknights: setup. Weekend: contact sanity + first insert.*

- [x] MuJoCo installed (uv), stock UR5e loads and steps (viewer not yet opened interactively)
- [x] Load UR5e MJCF from `mujoco_menagerie`; swap position actuators → torque `motor`s
- [x] Joint-space PD + `qfrc_bias` gravity compensation holds the arm steady (zero drift at home; 44 mrad sag without gravity comp; 0.05 rad step settles in 0.46 s)
- [x] Add socket (box geoms, small chamfer). Needed stiffer contact params: default sank 1.16 mm under 30 N on the chamfer lip, tuned ≤0.21 mm
- [x] Round peg welded to flange (same contact params as the socket), with `ft_site` and `peg_tip` sites
- [x] Basic step loop: set targets, `mj_step`, read state (`Sim`)
- [x] Contact sanity check: no penetration, no jitter, no explosion (**≤4 hrs tuning**; try `timestep` 1 ms → 0.5 ms, `cone="elliptic"`) — `scripts/contact_sanity.py`: 6 scenarios (top surface to 40 N, chamfer lip, hole bottom, hole wall) at 1 mm clearance all pass: penetration ≤0.12 mm, force jitter ≤1.5%, no warnings. 0.5 ms gives no improvement → keep 1 ms
- [x] Wrist `force`/`torque` site sensors: tare before contact (they include connector weight/inertia) + low-pass filter (`FTSensor`, 30 Hz; sign flipped to external force on peg — matches ground-truth contact force within 0.2 N)
- [x] Read contact forces programmatically — this is what the week-2 guarded-move baseline uses (`Sim.peg_contacts()` ground truth for scoring; controllers use `FTSensor`)
- [x] **Evaluation harness skeleton** — logs success, cycle time, peak force per trial (`harness.run_trials` → CSV + per-step `.npz` trajectories; controllers only see believed socket pose + proprioception + F/T)
- [x] Quick Jacobian step for Cartesian motion: `mj_jacSite` → `Δq = J⁺·Δx`, iterated toward a target EE position (~5 lines; seeds the week-2 IK) — `JacobianIK`, includes orientation hold so the peg stays vertical
- [x] **Hardcoded straight-down insert succeeds at loose clearance** (uses the Jacobian step) — `scripts/easy_insert.py`: 3/3 at 10 mm, 5.7 s cycle, no contact. Socket moved to (−0.15, 0.50, 0.18), clear of singularities

**Done =** arm commandable, stable contact, forces readable, harness logging, one easy success.

**Week 1 findings**
- With zero uncertainty, the open-loop insert tracks the hole axis to 0.08 mm and seats even at **1 mm** clearance without touching. Ratcheting clearance alone is uninformative — uncertainty is what makes the task hard, so ratchet uncertainty early.
- With ±5 mm uncertainty at 1 mm clearance, the open-loop insert jams on every seed tried (peak force 51–69 N, over the 30 N cap). This is the motivating failure for compliance.
- Joint-space PD is soft (~1.3 N/mm) and anisotropic in Cartesian space: pressing down also pushes sideways. MuJoCo's soft friction then creeps (0.7 mm/s at 40 N). Expect Cartesian impedance to fix the anisotropy; keep creep in mind for spiral search.

### Week 2 — Nominal motion pipeline
- [x] Grow the week-1 Jacobian step into full IK: add orientation error, damped least squares, joint limits (`control/ik.py`: DLS λ=0.01, joint-limit clamping, random restarts near the initial guess)
- [x] Quintic/trapezoidal trajectory generation (`control/trajectory.py`: limit-respecting quintic joint moves; trapezoidal Cartesian lines tracked by warm-started IK + feed-forward q̇)
- [x] Reliable pre-insertion reach from varied start configurations — 300/300 starts at ±0.25/0.5/1.0 rad joint noise, error ≤0.011 mm, no collisions. Needed: IK seeded from home (not the start) + lift → via → line-down approach
- [x] **Guarded-move + random-retry baseline** implemented (`control/guarded.py`; grid via `scripts/evaluate.py`)
- [x] **Report skeleton** written (headers, method text, plot placeholders) — weeknight task (`docs/report.md`)

**Done =** IK reaching reliably; dumb baseline produces nonzero success rate.

*Week 2 is the most loaded week. If it runs long, the report skeleton slides first; IK only
needs to be good enough to reach the pre-insertion pose — don't polish it.*

**Week 2 findings** (grid: 20 seeds per cell, ±0.25 rad start noise, 60 s budget; measured at the old μ = 0.8 default — see the weeks 3.5–4 benchmark for μ = 0.3)

| Clearance | Uncertainty | Open-loop | Guarded + retry | Guarded cycle time |
|---|---|---|---|---|
| 3 mm | ±2 mm | 60% | 100% | 7.3 s |
| 3 mm | ±5 mm | 10% | 100% | 17.3 s |
| 1 mm | ±2 mm | 10% | 55% | 30.4 s |
| 1 mm | ±5 mm | 5% | 45% | 27.6 s |

- Guarded baseline at 1 mm is **time-budget limited**, not force limited: ~50 attempts at ~0.9 s each. It also samples ±5 mm regardless of the actual uncertainty level, which is why ±2 and ±5 mm score alike.
- **Impact spikes**: touching down at 1 cm/s with the stiff joint-PD arm gives 12–29 N single-step contact force that decays within ~6 ms (10 ms average ≈ 7 N). Success is still scored on the raw peak (conservative); both are logged (`peak_force`, `peak_force_10ms`). Decide before week 5 which one the force-cap criterion uses — raw peaks sit right under the 30 N cap for the baseline.
- Open-loop peaks reach 19–66 N on average per cell — the motivating failure for compliance.

### Weeks 2–3 — Compliance controller
*Weekend blocks — gain tuning needs contiguous time, not scattered evenings.*

- [x] Cartesian impedance controller at torque level (Jacobian, `qfrc_bias`, joint-space damping) — `control/impedance.py`: damping from task-space inertia Λ (4–8 kg translational), spring-force saturation (15 N). Defaults K_xy 300, K_z 1000 N/m, K_rot 0.5 Nm/rad
- [x] F/T-based contact detection + force-cap monitor wired into the harness — `ContactDetector` (hysteresis + debounce); harness logs `ft_contact_time`, `ft_detect_latency` (2 ms on decisive contact), `ft_over_cap`
- [x] Validated in isolation: arm yields on surface contact rather than fighting it; low rotational stiffness lets a tilted peg flatten against a surface — `scripts/compliance_check.py`: press force = K·depth to 0.1%, lateral 0.1% (joint PD 77%), push displacement = F/K; 4° tilt flattens to 0.01° at K_rot ≤ 0.5
- [x] Begin ratcheting clearance: 10 → 5 mm — went to 1 mm: `ImpedanceInsert` (`control/compliant.py`) 100% at 10/5/3/1 mm with ±2 mm uncertainty (μ=0.3), 0 force-cap violations

**Done =** compliant contact response verified independently of insertion.

**Weeks 2–3 findings** (default friction was 0.8 at the time)
- **Impact spikes come from arm inertia, not controller stiffness.** Touching down at 1 cm/s gives ~27 N single-step force with both impedance and joint PD. The spike scales ~2.7 N per mm/s (48 N at 20 mm/s → 5 N at 1 mm/s). Rule: descend fast, touch down at ≤2.5 mm/s. This mostly resolves the raw-vs-10 ms force-cap question: at slow touchdown they agree.
- **Setpoint windup** caused a 130 N wall strike: while the peg was held on the chamfer the setpoint kept descending, and the stored spring force fired the peg into the hole wall (57 mm/s) when it dropped in. Fixed with a leash (`max_lead` 5 mm): the setpoint path pauses while the tip lags.
- **Chamfer guidance needs low lateral stiffness.** On a 45° chamfer the peg slides at most δ = P(1−μ)/(K_xy(1+μ)) before stalling. Hence K_xy 300 (not 1000).
- **Friction is the dominant parameter.** Decided 2026-09-14: default μ = 0.3 (brass/tin-plated pins in plastic), μ = 0.8 kept as a high-friction benchmark. Ratchet (20 seeds/cell, ±0.25 rad start noise):

| μ | Clearance | Uncertainty | Open-loop | Impedance |
|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 70% | 100% |
| 0.3 | 1 mm | ±1 mm | 45% | 100% |
| 0.3 | 1 mm | ±2 mm | 15% | 100% |
| 0.8 | 3 mm | ±2 mm | 60% | 90% |
| 0.8 | 1 mm | ±1 mm | 25% | 65% |
| 0.8 | 1 mm | ±2 mm | 10% | 30% |

  All 10 and 5 mm cells: 100% for both. Beyond the chamfer's capture range (e.g. ±5 mm) the impedance insert presses at bounded force (~6 N) and times out — that's what spiral search is for.
- F/T detection latency is 2 ms on decisive contact; the metric is ill-defined when force ramps slowly through the threshold.

### Weeks 3.5–4 — Search and alignment
- [x] Spiral search triggered on contact detection — `SpiralSearchInsert` (`control/compliant.py`): F/T contact → 3 N push, Archimedean spiral (1.5 mm pitch, 10 mm/s), stiff lateral gains while sweeping
- [x] Transition logic: search → insert once aligned (detect via z-drop below surface height) — >0.5 mm new drop or 0.3 s stall pauses the spiral and centres along the sensed lateral force; >2.5 mm drop → insert
- [x] **First full end-to-end success at target clearance** — 95% at 1 mm / ±5 mm (μ = 0.3), 0 force-cap violations
- [ ] Ratchet: clearance → 3 mm, then translation uncertainty, then rotation uncertainty — clearance and translation done (benchmark below); **rotation uncertainty not started** (needs a rotatable socket in the scene + harness)
- [ ] If time: round peg → rectangular connector (yaw now matters — see note) — **superseded** by the square connector in the weeks 6–7 extension

**Spiral budget.** Pitch must be ≤ capture width (clearance + chamfer). Path length to cover
radius *r* ≈ π·r²/pitch — at r = 10 mm, pitch = 1 mm that's ~314 mm, i.e. **30–60 s** at
5–10 mm/s. Set the cycle-time budget from this, or widen the chamfer to allow coarser pitch.

**Yaw note.** For a rectangular connector of width *w*, yaw error θ shifts corners by ≈ (w/2)·sinθ.
At w = 20 mm, 5° → ~0.9 mm, which exceeds the per-side gap at 1 mm clearance. Either rely on
chamfer + rotational compliance to self-align, or cap yaw uncertainty at ~±2°.

**Done =** end-to-end success at target conditions on at least one seed.

**Weeks 3.5–4 findings** — benchmark: 20 seeds per cell, ±0.25 rad start noise, 60 s budget, raw force-cap scoring (superseded by the week-5 rerun below)

| μ | Clearance | Uncertainty | Open-loop | Guarded + retry | Impedance | **Spiral** |
|---|---|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 70% | 95% | 100% | **100%** |
| 0.3 | 3 mm | ±5 mm | 15% | 90% | 45% | **100%** |
| 0.3 | 1 mm | ±2 mm | 15% | 60% | 100% | **100%** |
| 0.3 | 1 mm | ±5 mm | 0% | 45% | 25% | **95%** |
| 0.8 | 3 mm | ±2 mm | 60% | 100% | 90% | **100%** |
| 0.8 | 3 mm | ±5 mm | 10% | 100% | 25% | **65%** |
| 0.8 | 1 mm | ±2 mm | 10% | 55% | 50% | **85%** |
| 0.8 | 1 mm | ±5 mm | 5% | 45% | 15% | **55%** |

- Spiral search has **zero force-cap violations in all 8 cells** (open-loop: 27 total, impedance: 15, guarded: 5). Cycle time 10.7–14 s at μ = 0.3.
- Two failure modes found and fixed while building it: (1) a peg already in the top of the hole got dragged into the wall by the stiff spiral spring (~15 N, jam) → lateral leash + shallower hole-drop threshold; (2) a peg caught on the chamfer near a hole corner stalled the spiral forever → force-guided centring (move along the sensed lateral force). (2) took 1 mm/±5 mm from 35% → 95%.
- **High friction (μ = 0.8) is where search struggles**: timeouts, ~34 pauses and 16 s stalled per trial at 1 mm/±5 mm. At 3 mm/±5 mm guarded retry (100%) beats spiral (65%) — lifting between attempts avoids dragging a high-friction peg. Candidate week-5 fixes: lift-and-hop when stalled, or lower search force.
- Force-cap scoring: with the slow touchdown zone, the raw-vs-10 ms question only matters for the open-loop and guarded baselines (their 1 cm/s touchdowns spike ~19–30 N). Still to decide before the final grid.
- Spiral budget in practice: mean search 6 s at 1 mm/±5 mm (μ = 0.3), 33 s at μ = 0.8.

### Week 5 — Evaluation and tuning
- [x] Run grid: **2 clearances × 2 uncertainty levels × 20 trials per cell** (80 trials; each trial is a distinct seed). Headless sim runs well faster than real time — this is minutes, not hours — rerun for all 4 controllers × 2 frictions after the week-5 fixes (`results/benchmark_*`)
- [x] Diagnose and fix failure modes — `scripts/diagnose_spiral.py`; spiral 140 → 160/160 (details below)
- [x] Comparison run: spiral search vs. guarded-move baseline — `scripts/compare.py`: 160/160 vs 123/160, paired McNemar p = 1.5e-11
- [x] Buffer week if earlier weeks ran long — used for the Plan A grid (done, see §8)

**Week 5 decisions**
- **Force cap scored on the 10 ms moving average** of true contact force (decided 2026-09-14). Single-step peaks still logged (`peak_force`, `raw_over_cap`) and reported. Reason: remaining violations were one 1 ms step in ~17,000 when the peg clipped the chamfer mid-sweep (10 ms average 6–9 N); spikes scale with contact speed and solver contact stiffness.

**Week 5 findings**

Spiral failure diagnosis (20 failing trials from the weeks 3.5–4 benchmark, 19 at μ = 0.8): the controller spent ~28 s paused and ~21 s stalled out of ~50 s searching, sweeping only 1–4 s (1–3 spiral turns). Fixes, in the order they were found:
1. Sweep lighter (2 N) with a looser lateral leash (3 mm) so friction drag can't stall it; push harder (5 N) only when paused.
2. Clamp the spiral setpoint to the leash and re-anchor it at the tip after a pause (it was jumping ahead and re-stalling).
3. Leaky stall timer (a clamped setpoint flickers stalled/free every step, so a hard reset never fired).
4. Hop over flat-top stalls and over chamfer spots where centring already failed; pitch 1.5 → 1.0 mm (at μ = 0.8 the chamfer doesn't funnel, so capture ≈ clearance).
5. Anchor the pause push at the contact height (a tip-tracking push drove the peg into the wall, 32–72 N).
6. Drop-speed governor, 3 mm/s (falling off the flat top onto the chamfer at ~20 mm/s gave 30–53 N landings).

Final benchmark (20 seeds/cell, 10 ms scoring):

| μ | Clearance | Uncertainty | Open-loop | Guarded | Impedance | **Spiral** |
|---|---|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 75% | 100% | 100% | **100%** |
| 0.3 | 3 mm | ±5 mm | 30% | 100% | 45% | **100%** |
| 0.3 | 1 mm | ±2 mm | 45% | 60% | 100% | **100%** |
| 0.3 | 1 mm | ±5 mm | 15% | 50% | 25% | **100%** |
| 0.8 | 3 mm | ±2 mm | 75% | 100% | 100% | **100%** |
| 0.8 | 3 mm | ±5 mm | 25% | 100% | 35% | **100%** |
| 0.8 | 1 mm | ±2 mm | 45% | 55% | 100% | **100%** |
| 0.8 | 1 mm | ±5 mm | 10% | 50% | 20% | **100%** |

- Spiral vs guarded, paired by seed: 37 trials only spiral solved, 0 only guarded. At 1 mm clearance spiral is also 1.5–2.5× faster; at 3 mm / ±2 mm guarded is faster (7 s vs 11 s).
- Spiral: 0 timeouts, 0 force-cap violations, 8 trials with a single-step peak > 30 N (7 of them at μ = 0.8, 1 mm / ±5 mm).

### Weeks 6–7 — Extension: camera, two connector types, grasping
*Added 2026-09-15. Turns the task into perceive → identify → pick → insert into the matching socket.
The insertion-only benchmark and Plan A results stay as the "insertion module" results.*

**Task.** Each trial has **both connectors** on the table, one round and one square, at random,
non-overlapping poses in a pick area (random which is where). Up to M4 connectors stand upright; from M4b
each connector is independently upright or lying on its side. A fixture holds **two sockets**: a round hole and a
square hole. The robot must use the camera to find the **round connector first**, grasp it, insert it into the round
socket and release it; then find the square connector, grasp it and insert it into the square socket. No
ground-truth poses reach the controller: connector and socket poses come from perception.

**Sequence per trial:** capture → identify both connectors and both socket holes → pick round → insert round →
release and retract → **re-capture** (the first pick or insertion may have nudged the square connector) → pick
square → insert square → release and retract.

**Design decisions (2026-09-15)**
| Choice | Decision | Why |
|---|---|---|
| Camera | **Fixed overhead RGB-D**, calibrated (intrinsics from `fovy`, extrinsics from the model) | Sees connector and both sockets in one shot; simplest pipeline. Image taken with the arm clear of the view |
| Perception | **Classical CV** (OpenCV): depth/colour segmentation, contour shape analysis, back-projection | Explainable, no training data, easy to report |
| Grasp synthesis | **Antipodal grasp proposals from the camera point cloud** (M3), decided 2026-09-15 | The standard condition for a parallel-jaw gripper (the default Pick-Nav-Reach project grades the same algorithm on ground-truth point clouds); derives the grasp from perception instead of hand-coding it per shape |
| Grasp physics | **Attach on grasp** for M1–M5 (weld activated when both fingers contact the connector); **friction grasp in M6**, attach kept as fallback | Attach keeps the validated insertion physics while the pipeline is built; M6 matches the default project's friction-grasp physics (hold and carry stability) |
| Gripper | Simple parallel-jaw gripper built from primitives (finger slide joints + position actuator) | Robust contact, fast; menagerie Robotiq 2F-85 can replace it visually later |
| Pose uncertainty | Comes from **perception error** (plus optional depth noise), replacing the synthetic believed-entrance offset | More realistic; the synthetic mode stays for the insertion-only benchmark |

**Geometry: each connector fits only its own socket.** At 1 mm clearance: square peg 8.5 mm side → square hole
9.5 mm; round peg Ø10 mm → round hole Ø11 mm (16-gon). The round peg can't enter the square hole (10 > 9.5), and the
square peg's 12.0 mm diagonal can't enter the round hole, whose widest point (the polygon vertices) is 11.22 mm.
An 8 mm square (11.3 mm diagonal) was rejected in M1: with its corners on the polygon vertices the margin was
0.05 mm, about contact-penetration size. At larger clearances the exclusivity breaks, so the harness scores
"correct socket" explicitly rather than relying on fit. Connectors are 50 mm long so the fingers (grasping 10 mm
below the top) stay 12 mm above the socket when seated.

**Milestones**
- [x] **M1 Scene.** Table, fixture with round (N-gon) + square holes, round and square connectors as free bodies, overhead RGB-D camera, parallel gripper on the flange with the F/T sensor behind it, attach-on-grasp weld. Insertion-only scene still builds unchanged (flag). Contact sanity check passes for square-in-square and round-in-N-gon.
  **Done =** scene renders; gripper closes on either connector and the weld holds it; both pegs seat by hand-scripted motion.
  *Done 2026-09-15* — `assembly_scene.py`, `assembly_sim.py`, `assembly_check.py`, `tests/test_assembly_scene.py` (16 tests), `scripts/assembly_demo.py`:
  - Hand-scripted ground-truth assembly (round then square): both grasped, inserted (tip within 0.15 mm / 0.03 mm of the axis at descent), released, still seated at the end; the waiting connector moved 0.003 mm.
  - Keying holds under a 30 N push, including the square's worst-case yaw (corners on the polygon vertices). Chamfer-lip penetration: 16-gon 0.18 mm vs square 0.19 mm (limit 0.5 mm). Round-hole ray profile matches the design at 4 faces and all 16 vertices.
  - Camera: 1280×960, 25° fovy, 0.75 m above the table (~0.34 mm/px; connector tops ~1,250 px). A `camera_clear` keyframe (arm parked behind the base) leaves zero arm pixels in view; at `home` the arm covers 24% of the image.
  - All grasp poses over the pick area (4 yaws) and all insertion poses are reachable; min position-Jacobian σ 0.27.
  - Found and fixed: the weld's default anchor came from the joint equality's polycoef ([0, 1, 0]) and flung the connector 1 m on attach — weld data is now set explicitly.
- [x] **M2 Perception.** RGB-D render → point back-projection; segment connector and socket holes; classify round vs square; estimate connector xy/yaw/height and socket-hole centres/yaw. `perception_check` over random layouts vs ground truth.
  **Done =** 100% classification; position error < 1 mm and square yaw error < 2° (95th percentile), with and without depth noise.
  *Done 2026-09-15 (upright connectors)* — `camera.py`, `perception.py` (`PoseEstimator` interface + `GeometricEstimator`), `perception_check.py`, `scripts/perception_check.py`, `tests/test_perception.py`:
  - Method: depth → world height map; connector tops = regions > 40 mm above the table, top face outlined in the **colour** image (falls back to the depth face); socket openings = holes in the fixture's top plane. Pixels are intersected with the relevant plane so shapes are metric. Identification by IoU against an equal-area circle vs square, plus a one-of-each prior when exactly two candidates are found. Square socket yaw from the block's outer edges.
  - Results (40 layouts each; depth noise σ 0 / 1 / 2 mm, RGB noise 3 levels when depth noise is on): identification **100%** at all levels; position error 95th percentile ≤ 0.16 mm (connectors) and ≤ 0.52 mm (sockets); square connector yaw 95th percentile 1.0° / 1.4° / 1.4° (max 1.9°); square socket yaw exact.
  - What didn't work: contour circularity/convex-hull rectangularity misidentified 25% of sockets at σ = 2 mm (noise inflates hulls); depth-only yaw degraded to 1.5–3° p95 with noise. Compared on identical images: colour-outline yaw 0.47° p95 at every depth noise level vs depth minimum-area rectangle 0.37° / 1.48° / 3.15°.
  - Conventions verified against ray casts: MuJoCo's depth image is distance along the optical axis; image row 0 is the top of the view.
- [x] **M3 Antipodal grasping, pick and release.** Antipodal grasp proposals from the perceived outline; approach / close / attach / lift; in-hand estimate; re-tare F/T after the grasp; release and retract. Round first, chosen by classification.
  **Done =** proposals are antipodal against ground-truth geometry (square: finger axis within 2° of the flat normals); ≥ 95% grasp success over random two-connector layouts; the other connector is not disturbed by the pick; released connectors stay seated.
- [x] **M4 Square insertion.** Impedance insert + spiral search generalised to a movable `tool_tip` site set from the in-hand estimate, with a target tip orientation aligned to the perceived socket yaw; square peg in square hole, round peg in round hole.
  **Done =** insertion-only benchmark (perception-derived uncertainty) for both shapes at 3 mm and 1 mm, both frictions.
  *M3 + M4 done 2026-09-15* — `grasping.py`, `assembly_skills.py` (`Robot` interface without ground truth, `AssemblyRun` skill sequence), `scripts/assembly_benchmark.py`, `scripts/pipeline_demo.py`, `tests/test_grasping_pipeline.py`:
  - **Benchmark** (20 layouts × clearance {3, 1} mm × friction {0.3, 0.8} × camera {ideal, realistic: depth σ 1 mm, RGB noise 3, extrinsic calibration error ±2 mm / 0.5°}): **160/160 assemblies seated both connectors in the correct sockets**. Identification, grasp and lift 100%; square grasp vs true face normal ≤ 1.17°; other connector moved ≤ 0.002 mm; still seated after release 100%; 0 wrong sockets; peak sustained force ≤ 6.3 N; in-hand tip error p95 0.10–0.13 mm (ideal) and ~4.2 mm (realistic).
  - **Caveat:** the calibration error biases connectors and sockets the same way (one camera), so most of it cancels and spiral search was needed in only 5–10% of square insertions at 1 mm (realistic). M5 should add a condition with independent per-object errors to stress the search.
  - **Grasp planner:** upright connectors are prisms, so side normals come from the top-face outline. Closing directions every 2°; contact patches within 0.4 mm of the extremes; normals from a line fit over each whole patch; friction-cone test (μ 0.3, 5° margin); patches < 3 mm rejected (corners); refinement to the patch normals. Local ±4-point outline normals gave 2.6° p95 error (pixel staircasing) — whole-patch fits gave 0.83° / 1.16° (clean / σ 1 mm).
  - **In-hand estimate:** across-finger position and height from perception; along the closing axis from the finger joint positions at first touch; square axes aligned with the fingers.
  - **Found and fixed:**
    - `tool_tip` site orientation changes were silently ignored: MuJoCo sets `site_sameframe` for a site compiled coincident with its body. IK then pointed the gripper up and the wrist swept into the fixture. Now compiled with a non-identity orientation, and the flag is cleared on every update.
    - The attach weld was soft: a 5 N insertion push slid the gripper 4 mm down the connector and the F/T saw only 0.8 N. Stiffened (solref 4 ms).
    - Fingers slammed shut in 27 ms, bit 0.6 mm into the connector unevenly and pushed it 1 mm off-centre; finger coupling was soft (±2 mm). Now a 20 mm/s finger speed limit and a stiff coupling: in-hand error ≤ 0.1 mm on every axis.
    - Completion by estimated tip height stopped short under calibration bias and let connectors drop in on release. Now mechanical: not searching, F/T > 2 N, stopped 0.3 s, within 5 mm of the believed bottom.
    - Controller F/T is now expressed in the tool-tip frame (the sensor sits at the gripper base) so force-guided centring stays correct.
  - **Demo:** `results/demos/pipeline_realistic_seed0.mp4` (3D view | live overhead camera | perception overlay with outlines, sockets and the chosen grasp).
- [ ] **M4b Lying connectors (in-hand reorientation, no regrasp).** *Added 2026-09-15.* Each connector starts upright or lying on its side (seeded, independently). A lying connector is grasped from above across its width near one end, lifted, and rotated 90° about the gripper's closing axis so it points down; the hand then approaches the socket from the side, facing away from the other socket, and the M4 compliant insertion runs with the tool horizontal.
  - *Scene:* layout sampler adds orientation {upright, lying} with in-plane yaw; lying round connectors must not roll away (check rest stability; small friction/damping if needed).
  - *Perception:* detect lying connectors (footprint ≈ 50 × 10 mm), estimate long-axis yaw and both end positions, identify round vs square from the depth cross-section across the width (curved vs flat top), and report upright/lying.
  - *Grasp:* grasp within the top ~15 mm of the connector (≥ insertion depth + finger clearance left below the fingers after rotation); antipodal proposals across the width; fingertips clear the table when pinching a lying round connector at its centreline.
  - *Insertion:* sideways-tool pre-insertion poses; IK reach and collision checks for hand/wrist vs fixture and the other socket; in-hand tip estimate from the perceived end position.
  **Done =** perception identifies orientation and shape at 100% and estimates the long axis within 2° and the end position within 1 mm (95th percentile) over random mixed layouts; ≥ 90% of lying connectors are grasped, reoriented and seated (3 mm clearance, μ = 0.3) with no hand–fixture collisions; upright results unchanged.
- [ ] **M5 Full pipeline.** `PickAndInsert` controller (round then square) + harness task mode; metrics per connector and per trial: classification correct, round picked first, grasp success, correct socket, seated, still seated at the end, force cap (10 ms), cycle time per connector and total, perception errors vs ground truth. Grid: clearance {3, 1} mm × friction {0.3, 0.8} × depth noise {off, on} × orientation {all upright, mixed upright/lying} × 20–50 seeds (each seed = a random two-connector layout). Demo videos (trace panel + camera view).
  **Done =** end-to-end success (both connectors seated in the correct sockets) reported per cell; failure modes diagnosed by stage (perception / grasp / insertion / release) and by connector.
- [ ] **M6 Friction grasp** (after M5; attach-on-grasp stays as the fallback). Remove the weld: the connector is held only by finger friction. Force-limited finger actuation (grip force set from connector mass, insertion push and μ, with margin), finger pad contact parameters validated like the socket contact check, and F/T thresholds re-checked with the gripper and connector in the load path. Log in-hand slip (connector tip displacement relative to the gripper) through lift, transport and insertion.
  **Done =** on the full-task grid, (a) ≥ 90% of grasps lift and transport without a drop, (b) in-hand slip during insertion < 0.5 mm in ≥ 90% of trials, and (c) end-to-end success within 10 percentage points of the attach-on-grasp result in every cell. If not met by the start of week 8, report friction grasp as a documented partial result and keep attach-on-grasp for the headline numbers.

**Risks**
| Risk | Mitigation |
|---|---|
| Fingers hit the socket top during insertion | Grasp high on the connector: finger tips must stay above insertion depth + chamfer; check clearance in M1 |
| N-gon round hole gives poor contact (corners, chamfer seams) | Run the contact sanity check on it first; fall back to a round peg in a square-with-rounded-feel hole only if it fails |
| Arm occludes the camera | Capture images from a camera-clear arm pose, before each pick |
| Picking or inserting the first connector knocks the second | Minimum spacing between connectors in the layout sampler; approach from above; re-capture before the second pick |
| Retracting after release pulls the seated connector out | Open fingers fully and wait before lifting; check "still seated" at the end of the trial |
| A connector lands inside the other's grasp or approach path | Layout sampler rejects overlapping keep-out regions; plan approach yaw to keep fingers away from the other connector |
| Weld relative pose wrong when activated | Set the weld's relative pose from the current body poses at activation; test that the connector doesn't jump |
| Square peg jams at the corners from yaw error | Low rotational stiffness about z + chamfer; measure yaw tolerance in M4 |
| F/T thresholds shift with gripper + connector mass | Tare at start and again after grasp |
| Friction grasp slips under insertion force (up to ~6 N sustained) | Size grip force so the two-finger friction hold 2·μ·F_grip exceeds the insertion push with margin (at μ = 0.3 and ~6 N push: F_grip ≳ 10 N plus margin); M6 done-criteria include slip; fallback to attach-on-grasp |
| Hand or wrist hits the fixture or the other socket during sideways insertion of a lying connector | Approach from the side facing away from the other socket; IK + collision check of pre-insertion poses before committing; 80 mm socket spacing can be widened |
| Lying round connector rolls when bumped or pinched | Check rest stability in the scene; pinch at the centreline height; re-capture before picking |
| Scope creep | Arbitrary pick order, wrist camera and tilt uncertainty are stretch only |

### Week 8 — Friction grasp (M6) and results and writeup
- [ ] Plots: success rate vs. clearance/uncertainty; contact force; cycle time; perception error distributions; stage-wise failure breakdown
- [ ] Report (~3–5 pages): task, method, results, discussion — add perception + grasping sections
- [ ] Demo video / trajectory replay (full pipeline, with camera view)
- [ ] Update the proposal's task definition to the extended task (instructor-facing)

---

## 7. Evaluation rubric

Baseline grid: 2 clearance levels × 2 uncertainty levels × 20 trials (unseen seeds) per cell.

**Insertion success — 60%**

| Criterion | Weight |
|---|---|
| Reaches contact within step budget | 15% |
| Aligns and seats connector | 15% |
| Completes within cycle-time budget | 15% |
| Peak contact force below safety cap | 15% |

*"Reaches contact" is a low bar — it mainly catches IK/approach failures. Consider moving
some of its weight to seating if the instructor allows.*

**Robustness — 25%**

| Criterion | Weight |
|---|---|
| Success rate across uncertainty levels | 15% |
| Degrades gracefully at max uncertainty — measured as success rate at max uncertainty relative to zero uncertainty | 10% |

**Efficiency — 15%**

| Criterion | Weight |
|---|---|
| Mean cycle time (successful trials) vs. guarded-move + retry baseline, reported alongside both success rates | 15% |

*A no-search baseline mostly fails under uncertainty, so its cycle time isn't a meaningful comparison.*

*(Weights adjustable to instructor preference.)*

---

## 8. Stretch tiers — only if core is complete and validated

| Tier | Scope | Cost |
|---|---|---|
| **Plan A grid** | Expand to 3 clearances × 3 uncertainty levels × 50 trials per cell | **Done** (2026-09-15): clearance 3/2/1 mm × uncertainty ±1/±3/±5 mm × 50 seeds × μ 0.3/0.8 × 4 controllers = 3,600 trials → `results/plan_a_*`. Overall success: open-loop 56%, guarded 86%, impedance 78%, **spiral 99.1%** (lowest cell 92% at μ 0.8, 1 mm, ±3 mm; 6 timeouts, 2 force-cap violations in 900 spiral trials). Compute took ~19 h wall clock because the laptop slept on battery — plug in for long runs |
| **Friction grasp** | Replace attach-on-grasp with a connector held only by finger friction | **Moved into the schedule as M6** (2026-09-15) |
| **Tilt (rotation) uncertainty** | Socket axis tilted a few degrees from its believed orientation | ~half a day. Yaw is now covered by perception for the square connector; tilt remains untested |
| **Tier B** | Partially deformable connector: 2–3 link chain with joint compliance, so the plug lags/swings relative to gripper | ~1 week |
| **Tier C** | Fully deformable cable (MuJoCo composite body) + tip state estimation | Research-scale; comparable to the Intrinsic AI-for-Industry challenge task |
| **LeRobot** | Use the classical controller as a scripted expert → generate demos → train ACT/diffusion policy → compare against classical on the same harness | ~15–25 hrs + GPU. **Post-submission, not week 6** |

**Two zero-cost moves to keep the LeRobot door open now:**
- Log demos as (observation, action, reward) per timestep — easy conversion to `LeRobotDataset` (parquet + video) later
- Keep the evaluation harness **policy-agnostic** so a learned policy drops in behind the same interface as the classical controller

Both are good design regardless.

---

## 9. Context notes

- **Difficulty calibration:** roughly equivalent to the course's default Pick-Nav-Reach project, plus one harder skill — force-based fine insertion (mm tolerances, closed-loop compliance) replaces coarse navigation (10 cm tolerances, no force control).
- **Prior art:** Intrinsic × Open Robotics "AI for Industry Challenge" ($180k, 7 months, teams up to 10) targets the full deformable-cable version of this task. Useful as (a) motivation in the writeup, (b) a methodology reference — UR5e + wrist F/T + wrist cameras, force feedback without ground-truth pose at eval, and (c) a possible source of cable/connector assets from their open participant toolkit (visual meshes only — see convex-hull note in §2). **TODO: verify the $180k / 7 months / teams-of-10 figures against a primary source and cite it before submission.**
- **Session strategy:** deep debugging (compliance tuning) → weekend blocks. Low-context work (logging, plotting, trajectory code, report text) → weeknights. Momentum loss between short sessions is the main hidden cost at this pace.

---

## 10. Lessons learned and engineering notes

*Collected 2026-09-15 across weeks 1–7. Numbers are from the runs recorded in the sections above; the
per-milestone notes there have the full context.*

### 10.1 MuJoCo gotchas (each cost real debugging time)
- **Mesh collision = convex hull.** A mesh socket gets its hole filled. Build holes from box primitives (square
  hole; round hole as a 16-gon of boxes, vertices at apothem/cos(π/N), +0.11 mm at N = 16).
- **Contact parameters are mixed between the two geoms.** A stiff socket with a default-parameter peg gave
  half-stiff contact. Put the same `solref`/`solimp` on *both* sides of a contact pair (`contact_params`).
- **`solref` time constant must be ≥ 2 × timestep** (5 ms used at 1 ms). Halving the timestep to 0.5 ms changed
  nothing measurable, so 1 ms stayed.
- **Equality constraints default to soft.** The attach weld let a 5 N push slide the gripper 4 mm down the
  connector; the finger coupling let the fingers differ by ~2 mm. Set `solref` ≈ 4 ms and high `solimp` on any
  equality that should behave rigidly.
- **Weld `data` defaults are wrong for welds when built with MjSpec.** The anchor inherited the joint equality's
  polycoef [0, 1, 0, …] and flung the connector 1 m on attach. Set weld data explicitly:
  anchor(3) relpose(7) torquescale(1). The relpose is **body2's pose in body1's frame** (verified on a toy model).
- **`site_sameframe`.** A site compiled coincident with its body is flagged "same frame", and MuJoCo then ignores
  runtime changes to its `site_quat`. The movable `tool_tip` site silently kept the gripper's orientation, IK
  pointed the gripper up and the wrist swept into the fixture. Compile movable sites with a non-identity
  orientation and clear `site_sameframe` when updating them.
- **Adding joints invalidates keyframes** (qpos size). Rebuild: delete keys → compile → build full qpos from
  `qpos0` → `add_key` → compile again.
- **Menagerie actuators are position servos** (8 mrad sag at home). Replace with torque motors for impedance
  control; set `ctrlrange` to the real torque limits (±150 Nm base, ±28 Nm wrist).
- **Wrist F/T site sensors** report the force the parent exerts on the child subtree: includes the subtree's
  weight and inertia (tare before contact, and again after grasping), and is the negative of the external contact
  force (flip the sign for controllers). With a gripper, express it in the tool-tip frame before using its direction.
- **`mj_contactForce`** returns the force in the contact frame, acting on geom2; frame rows are world axes.
- **API changes in MuJoCo 3.13:** mass matrix is `data.M` (not `qM`); `mj_fullM(model, data, dst)`.
- **Renderer:** the offscreen framebuffer defaults to 640 px wide — set `visual.global_.offwidth/offheight` for
  larger images. The depth image is **distance along the optical axis** (not ray length); image row 0 is the top.
  Sites render by default — put them in group 4 to keep them out of camera images. The "ARB_clip_control
  unavailable" warning did not affect measured depth accuracy (table read exactly 0.17000 m); it writes
  `MUJOCO_LOG.TXT` (gitignored).
- **Managed vs passive viewer:** `mujoco.viewer.launch` works with plain Python on macOS; `launch_passive`
  needs `mjpython`. A control callback (`mujoco.set_mjcb_control`) is how the managed viewer runs a controller live.

### 10.2 Contact physics and validity of the results
- **Penetration vs clearance** is the validity check that matters at 1 mm clearance (0.5 mm per side). Default
  contact sank 1.16 mm under 30 N on a chamfer lip; tuned contact keeps it ≤ 0.21 mm (square and 16-gon sockets
  alike, ~0.18–0.19 mm with a 2 kg effective mass).
- **Impact spikes come from the arm's reflected inertia, not controller stiffness:** ≈ 2.7 N per mm/s of contact
  speed with either impedance or joint PD (48 N at 20 mm/s, 5 N at 1 mm/s). Design rule: move fast in free space,
  touch down at ≤ 2.5 mm/s, and govern falls onto lower surfaces (3 mm/s drop-speed limit in search).
- **Force-cap scoring on the 10 ms moving average** (decided week 5): single 1 ms steps above 30 N occur when a
  moving peg clips an edge and are sensitive to solver stiffness; the sustained force is what damages parts.
  Raw peaks are still logged and reported.
- **Friction dominates chamfer-guided insertion.** A peg pushed with force P slides at most
  δ = P(1−μ)/(K_xy(1+μ)) down a 45° chamfer before a lateral spring K_xy stalls it; at μ = 0.8 the friction angle
  (38.7°) nearly matches the chamfer, so it barely funnels. μ = 0.3 (brass/tin pins in plastic) is the main
  setting; μ = 0.8 is the stress benchmark.
- **MuJoCo soft friction creeps** under sustained lateral load (0.7 mm/s at 40 N, μ = 0.8; 0.2 mm/s at 25 N,
  μ = 0.3) — relevant to anything that relies on static friction (search, friction grasp).
- **Clearance alone doesn't make the task hard:** with zero pose error an open-loop insert tracked the hole axis to
  0.08 mm and seated at 1 mm clearance without touching. Uncertainty is the driver — ratchet it early.

### 10.3 Motion and control
- **IK configuration families:** seed IK from the home configuration, not the current (perturbed) pose; seeding
  from the start converged to flipped-wrist solutions whose joint sweeps hit the fixture.
- **Joint-space moves need a safe path:** lift → joint move to a via pose above the target → straight line down.
  Direct joint interpolation swept the wrist through the socket. With this approach 300/300 random starts
  (±1 rad) reached the pre-insertion pose without collision.
- **Place the task away from singularities** (elbow ≈ 2 rad, wrist 0.43 m off the base axis); check the
  position-Jacobian σ_min along paths (0.27 minimum in the assembly scene).
- **Joint PD is soft and anisotropic in Cartesian space** (~1.3 N/mm; pressing down pushed sideways at 77% of the
  normal force). Cartesian impedance fixes it (0.1–0.2%).
- **Impedance damping from task-space inertia** Λ = (J M⁻¹ Jᵀ)⁻¹ (4–8 kg translational, 0.1–0.3 kg·m²
  rotational, nearly constant over the insertion workspace), D = 2ζ√(KΛ).
- **Low rotational stiffness (≤ 0.5 Nm/rad) is needed** for contact to realign a peg: edge-contact torque is only
  ~0.05 Nm per 10 N on a 5 mm-radius peg. A 4° tilt flattened to 0.01° at 0.5 Nm/rad, but stayed at 3.8° at 20.
- **Leash every setpoint that can run ahead of a blocked tool** (axial 5 mm, lateral 3 mm). Without it the
  stored spring force fires the peg into walls when it drops in (130 N) or drags it into a jam (~15 N).

### 10.4 Contact-based search
- **Pitch ≤ capture width.** With friction the chamfer doesn't widen the capture region, so pitch must be ≤ the
  clearance (1.0 mm; 1.5 mm stepped over the ±0.5 mm direct-entry zone).
- **Diagnose search with phase logs vs ground truth.** Failing trials spent ~49 of 50 s paused or stalled and only
  1–4 s sweeping; that measurement led directly to the fixes: sweep lighter (2 N) and push harder only when paused
  (5 N, anchored at contact height), re-anchor the spiral at the tip after a pause, leaky stall timer (a clamped
  setpoint flickers stalled/free), hop over flat-top stalls and over chamfer spots where centring already failed.
- **Force-guided centring** (move along the sensed lateral contact force, which on a chamfer points toward the
  hole axis) unhooked pegs caught near hole corners: 1 mm/±5 mm success 35% → 95% on its own.
- **Completion must be mechanical, not a height threshold,** once pose estimates carry calibration bias: not
  searching, F/T > 2 N, stopped for 0.3 s, within 5 mm of the believed bottom.

### 10.5 Perception
- **Verify camera conventions against ray casts** before building on them (depth definition, row order, pixel
  centres at +0.5).
- **Resolution budget:** a 10 mm part at ~0.86 mm/px (~12 px) is too small to tell circles from squares; 25° fovy
  at 1280×960 from 0.75 m gives ~0.34 mm/px (~40 px across a connector top).
- **The arm occludes an overhead camera** (24% of the image at home): capture from a camera-clear pose, and
  re-capture before each pick.
- **Map pixels onto the relevant horizontal plane** (connector top, fixture top) before measuring shape, so
  perspective doesn't distort footprints.
- **Shape identification:** convex-hull rectangularity and circularity broke under depth noise (hulls inflate;
  25% of sockets misidentified at σ = 2 mm). IoU against an equal-area circle vs square, plus the scene prior
  (exactly one of each), gave 100% at every noise level tested.
- **Fuse modalities by strength:** depth finds and localises parts in 3D; colour gives crisp outlines. Square yaw:
  depth-only minimum-area rectangle 0.37° / 1.48° / 3.15° p95 at σ 0 / 1 / 2 mm vs colour outline 0.47° at all
  levels. Add RGB noise too, or the colour channel looks unrealistically clean.
- **Use the longest reliable feature:** square socket yaw from the block's ~43 mm outer edges (exact) rather than
  its ~13 mm opening (2.6–2.9° p95 under noise).
- **Tolerances should adapt to measured noise** (robust MAD of the table surface) instead of fixed thresholds.
- **Brightness thresholds admit lit side faces** if too low (value ≥ 180 hurt yaw; ≥ 200 fixed it).
- **Calibration error from a single camera largely cancels** between a part and its socket (both shift together),
  so it barely stresses insertion. Independent per-object errors are needed to test search under perception error.

### 10.6 Grasping and in-hand estimation
- **Antipodal grasping works from the top-face outline** for upright prismatic parts (side normals = outline
  normals). Estimate contact normals over the whole contact patch, not locally: local ±4-point normals had 2.6° p95
  error from pixel staircasing; whole-patch line fits gave 0.83° / 1.16° (σ 0 / 1 mm). Reject short patches
  (< 3 mm) to exclude corner contacts; refine the closing direction to the patch normals.
- **Grippers need a closing speed limit.** A step command slammed the fingers shut in 27 ms, biting 0.6 mm unevenly
  and pushing the part 1 mm off-centre and tilting it. 20 mm/s per finger fixed it.
- **Read finger positions at first touch** (gap reaches the object width): later readings drift as pads squeeze in.
  The pad midpoint from finger proprioception beats assuming the fingers centre the part.
- **In-hand estimate** = across-finger position and height from perception, along-closing position from the
  fingers, square axes aligned to the fingers: ≤ 0.1 mm error on every axis (ideal camera).
- **Keep physics shortcuts honest:** attach-on-grasp welds only when both pads touch, and grasps are planned to be
  antipodal, i.e. ones that would also hold under friction.

### 10.7 Evaluation methodology
- **Separate what controllers may see from ground truth** in code (`Observation` / `Robot` interfaces); ground
  truth only in scoring and diagnostic logs.
- **Seeds fix everything** (layout or socket offset, start pose), so controllers face identical trials and can be
  compared pairwise (exact McNemar on discordant outcomes) with Wilson intervals on success rates. 3 trials per cell
  is meaningless; 20 per cell was the minimum used, 50 for the Plan A grid.
- **Compare efficiency on trials both controllers solved,** or harder trials bias the average.
- **Report success before and after release** (`seated` vs `stayed_seated`): the gap exposed connectors that
  dropped in on release rather than being inserted.
- **Baselines should be honest:** the guarded-retry baseline samples ±5 mm regardless of the true uncertainty,
  which is why its ±2 and ±5 mm cells score alike.

### 10.8 Tooling and process
- **Never edit code while a long grid runs:** each evaluation process imports the code when it starts, so later
  controllers in a queued grid would pick up the edits. Use a git worktree or wait.
- **Laptops sleep on battery:** the Plan A grid took ~19 h instead of ~1 h. Plug in, keep the lid open, and use
  `caffeinate -i -s -w <pid>`.
- **Long jobs outlive tool timeouts only if detached:** run grids with `nohup`, log to a file and watch the log,
  rather than as a timed foreground/background command.
- **Multiprocessing needs importable functions:** `ProcessPoolExecutor` can't pickle functions defined in
  `python -c`; put jobs in a script with a module-level function.
- **zsh doesn't word-split variables** (`$CMD args` fails); write commands out.
- **Collection-time work in tests** (e.g. running simulations inside `@pytest.mark.parametrize`) slows every
  `pytest -k` run.
- **`results/` is gitignored,** so copy final plots and videos somewhere that gets submitted.

### 10.9 Open items and caveats
- Socket **tilt** uncertainty is untested (stretch tier).
- Lying connectors (M4b), the full-pipeline harness grid (M5) and the friction grasp (M6) are not started.
- The interactive viewer's live-control callback has only been tested headless (the GUI itself wasn't checked during development).
- Proposal assumptions to confirm: 8-week duration and rubric weights.
- Intrinsic challenge figures still unverified (§9).
