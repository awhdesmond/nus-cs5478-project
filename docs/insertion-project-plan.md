# Vision-Guided Contact-Rich Connector Insertion — Final Project Plan

**Course:** Intelligent Robots — Algorithms and Systems
**Duration:** 6 weeks
**Available time:** ~10–16 hrs/week (2–3 weeknights + weekend blocks) → ~60–95 hrs total

---

## 1. Task

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

**Task.** Each trial has **both connectors** on the table, one round and one square, upright at random,
non-overlapping poses in a pick area (random which is where). A fixture holds **two sockets**: a round hole and a
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

**Geometry: each connector fits only its own socket.** Proposed with 1 mm clearance: square peg 8 mm side → square
hole 9 mm; round peg Ø10 mm → round hole Ø11 mm. The round peg can't enter the square hole (10 > 9), and the square
peg's 11.3 mm diagonal can't enter the round hole (> 11). At larger clearances the exclusivity breaks, so the
harness scores "correct socket" explicitly rather than relying on fit. Round hole = N-gon of box primitives
(N ≈ 16; faces within ~0.1 mm of a true circle) with per-face chamfer boxes; must pass the contact sanity check.

**Milestones**
- [ ] **M1 Scene.** Table, fixture with round (N-gon) + square holes, round and square connectors as free bodies, overhead RGB-D camera, parallel gripper on the flange with the F/T sensor behind it, attach-on-grasp weld. Insertion-only scene still builds unchanged (flag). Contact sanity check passes for square-in-square and round-in-N-gon.
  **Done =** scene renders; gripper closes on either connector and the weld holds it; both pegs seat by hand-scripted motion.
- [ ] **M2 Perception.** RGB-D render → point back-projection; segment connector and socket holes; classify round vs square (circularity + polygon vertex count); estimate connector xy/yaw/height and socket-hole centres/yaw. `perception_check` over random layouts vs ground truth.
  **Done =** 100% classification; position error < 1 mm and square yaw error < 2° (95th percentile), with and without depth noise.
- [ ] **M3 Antipodal grasping, pick and release.**
  - *Antipodal grasp proposals:* back-project the connector's depth pixels to a point cloud, estimate surface normals (local PCA), sample finger contact pairs across the closing direction at the grasp height, and keep pairs where the contact line lies within both friction cones (angle to each normal ≤ atan(μ_assumed) with a safety margin). Reject pairs wider than the gripper opening or whose fingers would hit the table or the other connector; rank by friction-cone margin, then by closeness to the connector axis. Expected outcome: any diameter for the round connector, opposite flats for the square.
  - *Pick and release:* approach / close / attach / lift from the best proposal; in-hand offset estimate (connector tip relative to the gripper, from perception at pick time); re-tare F/T after the grasp (carried mass). Release: open fingers, deactivate the weld, retract without dragging the seated connector back out. Pick order: round first, chosen by classification, not by position.
  **Done =** proposals are antipodal against ground-truth geometry (square: finger axis within 2° of the flat normals); ≥ 95% grasp success over random two-connector layouts; the other connector is not disturbed by the pick; released connectors stay seated.
- [ ] **M4 Square insertion.** Generalise impedance insert + spiral search to a tool-tip site set from the in-hand estimate, with yaw alignment to the perceived socket; square peg in square hole, round peg in round hole.
  **Done =** insertion-only benchmark (perception-derived uncertainty) for both shapes at 3 mm and 1 mm, both frictions.
- [ ] **M5 Full pipeline.** `PickAndInsert` controller (round then square) + harness task mode; metrics per connector and per trial: classification correct, round picked first, grasp success, correct socket, seated, still seated at the end, force cap (10 ms), cycle time per connector and total, perception errors vs ground truth. Grid: clearance {3, 1} mm × friction {0.3, 0.8} × depth noise {off, on} × 20–50 seeds (each seed = a random two-connector layout). Demo videos (trace panel + camera view).
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
