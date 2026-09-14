# Contact-Rich Connector Insertion with Compliant Control

**CS5478 Intelligent Robots — Project Report** · *Draft (weeks 3.5–4). `TODO` marks sections filled in later weeks.*

---

## Abstract

> TODO (week 6, ~120 words): task, why position control fails at mm clearance, approach (impedance + spiral search), headline numbers vs. the guarded-move baseline across the clearance × uncertainty grid.

---

## 1. Introduction

Inserting wire terminals and connectors into sockets remains one of the least automated steps in electronics assembly. At millimetre scale the uncertainty between gripper and socket pose exceeds the insertion clearance, so a purely position-controlled arm jams or damages the part instead of seating it. The task is *contact-rich*: success depends on how the robot responds to contact, not only on where it moves.

This project studies a scoped version of the problem in simulation: a 6-DOF arm inserts a rigid connector into a socket with 1–3 mm clearance under bounded pose uncertainty. We compare an open-loop insert, a guarded-move-with-random-retry baseline, and a compliant controller (Cartesian impedance with contact-triggered spiral search) over a grid of clearances and uncertainty levels.

> TODO: one sentence on the Intrinsic × Open Robotics AI for Industry Challenge as motivation, with a citation (verify figures against a primary source first).

**Contributions.** (1) A reproducible MuJoCo insertion environment with validated contact behaviour at 1 mm clearance; (2) an evaluation harness that separates controller information from ground truth; (3) a controlled comparison of non-compliant and compliant insertion strategies. TODO: finalise once results exist.

---

## 2. Task and Setup

### 2.1 Task

A UR5e with a round peg rigidly attached to its flange must insert the peg into a square socket. The cable behind the connector is treated as rigid (§7).

- **Clearance** *c* (total): hole width = peg diameter + *c*, with *c* ∈ {3, 1} mm (baseline grid).
- **Pose uncertainty**: the controller receives a *believed* socket entrance offset from the true one by a seeded uniform xy error in ±*u*, *u* ∈ {2, 5} mm. It never observes the true pose.
- **Start pose**: home configuration plus seeded uniform joint noise (±0.25 rad), resampled until collision-free.
- **Friction**: μ = 0.3 (typical of brass or tin-plated pins in plastic housings) is the main setting; μ = 0.8 (dry steel on steel) is a high-friction benchmark.
- **Success**: the peg tip reaches within 2 mm of the hole bottom, inside the hole footprint, within the cycle-time budget (60 s), without the contact force exceeding the 30 N cap.

### 2.2 Simulation environment

- **Simulator**: MuJoCo 3.x at 1 ms timestep, implicit-fast integrator, elliptic friction cones. Chosen over PyBullet for more stable, accurate contact at millimetre scale.
- **Robot**: UR5e from MuJoCo Menagerie. The stock position servos are replaced by joint-torque motors (±150 Nm base joints, ±28 Nm wrist) so the controller owns gravity compensation and compliance.
- **Socket**: built from box primitives (MuJoCo collides meshes via their convex hull, which would fill the hole), with a 2 mm 45° chamfer formed from rotated boxes. Placed where the insertion path stays clear of the UR5e's elbow, wrist and shoulder singularities.
- **Sensing**: a force/torque sensor at the flange–peg interface, tared at rest and low-pass filtered (30 Hz).

> **Figure 1** — TODO: scene render (whole arm) + socket close-up with chamfer. Source: `scripts/replay.py --video`, frame grab.

### 2.3 Contact validation

Default MuJoCo contact parameters let the peg sink 1.16 mm into the chamfer lip under a 30 N push — more than the 0.5 mm per-side gap at 1 mm clearance, so an "insertion" could succeed by penetration rather than alignment. Stiffer contact parameters (`solref` = 5 ms, `solimp` = [0.95, 0.99, 1 mm]) reduce this to 0.21 mm. We then press the peg against the top surface (up to 25 N), the chamfer lip, the hole bottom and a hole wall at 1 mm clearance:

**Table 1** — Contact sanity check (1 mm clearance, μ = 0.3, 1 ms timestep). Source: `results/contact_sanity.csv`.

| Scenario | Mean force [N] | Force jitter | Max penetration [mm] |
|---|---|---|---|
| Top surface (max) | 25.5 | 0.23 % | 0.046 |
| Chamfer lip | 10.4 | 0.10 % | 0.032 |
| Hole bottom | 18.2 | 0.02 % | 0.031 |
| Hole wall | 9.2 | < 0.01 % | 0.079 |

All scenarios stay below 0.5 mm penetration and 5 % jitter with no solver warnings, at both μ = 0.3 and μ = 0.8. Halving the timestep changes nothing, so 1 ms is used throughout.

---

## 3. Method

### 3.1 Inverse kinematics

Joint targets for the peg tip come from damped least-squares IK on the 6×6 geometric Jacobian *J* (from MuJoCo's `mj_jacSite`):

$$\Delta q = J^\top \left(J J^\top + \lambda^2 I\right)^{-1} e, \qquad e = \begin{bmatrix} p^* - p \\ \tfrac12 \sum_i r_i \times r_i^* \end{bmatrix}$$

where *rᵢ* are the columns of the tip rotation matrix. Damping (λ = 0.01) keeps steps bounded near singularities; each step is clamped to joint limits and to 0.2 rad per joint. If the solve does not converge, it restarts from random perturbations near the initial guess.

**Configuration consistency.** IK is always seeded from the home configuration rather than the (perturbed) start. Seeding from the start occasionally converged to a flipped arm configuration whose long joint-space sweep hit the socket.

### 3.2 Trajectory generation

- **Joint space**: quintic polynomial with zero boundary velocity and acceleration. Duration is the shortest satisfying per-joint limits (1 rad/s, 2 rad/s²), using the quintic's peak-velocity (1.875 Δq/T) and peak-acceleration (5.77 Δq/T²) factors.
- **Cartesian space**: straight lines with a trapezoidal speed profile (triangular for short moves), converted to joint targets each control tick by warm-started IK, with feed-forward joint velocity q̇* = J⁻¹[v; 0].

**Approach.** From an arbitrary start: lift the tip vertically to a via height, joint-space move to a via pose 10 cm above the pre-insertion pose, then descend in a straight line to the pre-insertion pose 5 cm above the believed entrance, peg pointing down. Across 300 seeded starts with up to ±1 rad joint noise, all reached the pre-insertion pose (error ≤ 0.011 mm, ≤ 0.002°) without collision.

### 3.3 Joint-space tracking

All non-compliant motion is tracked by joint PD with gravity and Coriolis compensation:

$$\tau = K_p (q^* - q) + K_d (\dot q^* - \dot q) + \tau_\text{bias}(q, \dot q)$$

with τ_bias from recursive Newton–Euler. This holds the home pose with zero drift (versus 44 mrad sag without compensation). Its Cartesian stiffness is low (~1.3 N/mm) and anisotropic: pressing vertically also produces lateral force.

### 3.4 Baselines

- **Open-loop insert**: approach, then a straight descent to the believed hole bottom. No force feedback.
- **Guarded move + random retry**: descend at 1 cm/s until the filtered F/T force exceeds 3 N. If blocked, retract to 4 mm above the believed entrance, move to a new xy sampled uniformly in ±5 mm around the believed entrance, and descend again.

### 3.5 Compliant contact control

During the contact phase the arm is controlled as a Cartesian spring–damper about a moving setpoint at the peg tip:

$$\tau = J^\top F + \tau_\text{bias}(q, \dot q) - D_q \dot q, \qquad F = \begin{bmatrix} \operatorname{sat}\!\left(K_p e_p\right) + D_p(\dot p^* - \dot p) \\ K_r e_r + D_r(\omega^* - \omega) \end{bmatrix}$$

where *e_p* is the tip position error, *e_r* the orientation error as in §3.1, and [ṗ; ω] = J q̇. In free space the tip tracks the setpoint; in contact it yields, and the contact force settles near K·(setpoint penetration). The spring force is saturated at 15 N so contact force stays bounded whatever the setpoint. Damping is set per axis as D = 2ζ√(K Λ_ii) with ζ = 1, using the task-space inertia Λ = (J M⁻¹ Jᵀ)⁻¹ at the start of the contact phase (4–8 kg translational, 0.1–0.3 kg·m² rotational, nearly constant over the insertion workspace). The control law needs no force measurement; the wrist F/T sensor is used only for contact detection and monitoring.

**Gain selection.**
- *Lateral* stiffness is low (300 N/m) and *axial* stiffness higher (1000 N/m). On a 45° chamfer with friction μ, a peg pushed with force P slides at most δ = P(1−μ)/(K_xy(1+μ)) before the lateral spring stalls it.
- *Rotational* stiffness is 0.5 Nm/rad. The edge-contact torque on a 5 mm-radius peg is only ~0.05 Nm per 10 N, so higher values cannot realign the peg (Table 4).

**Insertion.** After the joint-space approach (§3.2), the controller switches to impedance and moves the setpoint straight down to 5 mm below the nominal hole bottom. It moves at 1 cm/s, slowing to 2.5 mm/s within ±3 mm of the entrance. A leash pauses the setpoint whenever the tip lags it by more than 5 mm along the insertion axis. This caps the push at ≈5–6 N and prevents windup: without it, a peg held on the chamfer accumulated spring force and struck the hole wall at 57 mm/s (130 N) when it dropped in.

**Table 4** — Isolation checks (no insertion). Source: `results/compliance_check/`.

| Check | Impedance | Joint PD |
|---|---|---|
| Press, setpoint 5 / 10 / 30 mm below surface | 5.0 / 10.0 / 15.0 N (saturated) | 8.1 N at 8 mm |
| Lateral / normal force while pressing | 0.2 % | 28 % (at the friction limit: the tip slides); 77 % at μ = 0.8 |
| 3 N lateral push at tip → displacement | 10.0 mm (= F/K) | 2.5 mm |
| 4° commanded tilt pressed at 15 N → final tilt (K_r = 0.5 / 2 / 20) | 0.01° / 2.1° / 3.8° | — |

> **Figure** — TODO: contact force and tip height vs. time for a surface press (impedance vs. joint PD).

**Impact.** The single-step contact force at touchdown is set by the arm's reflected inertia, not the controller: ~19 N at 1 cm/s for both controllers (μ = 0.3; ~27 N at μ = 0.8). It scales roughly linearly with touchdown speed (31 N at 20 mm/s, 9.9 N at 5 mm/s, 5.4 N at 2.5 mm/s, 5.2 N at 1 mm/s against a 5 N steady press), which motivates the slow touchdown zone.

### 3.6 Contact-triggered spiral search

The impedance insert only succeeds when the offset lies within the chamfer's capture range. Beyond that, the spiral search controller finds the hole by contact:

1. **Touchdown.** Same approach and impedance descent as §3.5. Contact is detected from the wrist F/T sensor (30 Hz low-pass, 2 N on / 1 N off hysteresis, 3-sample debounce), never from ground truth. The tip height at detection is taken as the surface height.
2. **Search.** Hold the setpoint 3 mm below the surface, which gives a 3 N push, and sweep its xy along an Archimedean spiral r = p·θ/2π around the believed entrance. The pitch is p = 1.5 mm, at most the capture width (clearance + chamfer); the setpoint moves at 10 mm/s, with angular speed capped near the centre. Lateral stiffness rises to 1500 N/m while sweeping, so friction drag (μF/K) doesn't distort the path.
3. **Pause and centre.** A new drop of more than 0.5 mm, or a spiral stalled for 0.3 s, pauses the sweep. The controller then returns to soft insertion gains and moves the xy setpoint along the sensed lateral contact force, which on the chamfer points toward the hole axis. The pause is extended while the peg keeps dropping.
4. **Insert.** A drop of more than 2.5 mm below the surface (past the 2 mm chamfer) confirms the hole. The controller then pushes down with insertion gains, using the same leash as §3.5.

**Leashes.** Both the axial and lateral setpoints are leashed. The spiral stops advancing while the tip lags it by more than 2 mm, capping sideways force at ≈3 N.

Two failure modes motivated the leash and the centring step. First, without the lateral leash, a peg that had already dropped into the top of the hole was dragged sideways into the wall at ~15 N, and wall friction then jammed it. Second, without force-guided centring, a peg caught on the chamfer near a corner of the square hole stalled the spiral indefinitely: success at 1 mm clearance with ±5 mm uncertainty was 35 % (μ = 0.3), versus 95 % with centring.

**Search-time bound.** Covering radius r takes a path length of ≈ πr²/p. For the ±5 mm uncertainty square (r ≈ 7.1 mm) that is ≈ 105 mm, or ~10 s at 10 mm/s, before pauses.

---

## 4. Evaluation Protocol

- **Harness**: every controller implements the same interface (observation → joint torques). Controllers see the model (for kinematics), joint state, filtered wrist F/T and the believed entrance only. Ground-truth contact force, penetration and the true socket pose are used solely for scoring.
- **Grid**: friction {0.3, 0.8} × clearance {3, 1} mm × uncertainty {±2, ±5} mm × 20 seeds per cell. Seeds fix both the socket offset and the start pose, so all controllers face identical trials.
- **Metrics**: success rate; cycle time over successful trials; peak contact force, both single-step (includes ~1 ms impact spikes) and 10 ms moving average; failure reason (timeout / force cap).

> TODO (week 5): state the final rubric mapping (insertion success 60 %, robustness 25 %, efficiency 15 %) and how each metric feeds it.

---

## 5. Results

### 5.1 Controller comparison

**Table 2** — Success rate over 20 seeds per cell (±0.25 rad start noise, 60 s budget). Source: `results/benchmark_*/summary.csv`.

| μ | Clearance | Uncertainty | Open-loop | Guarded + retry | Impedance | **Spiral search** |
|---|---|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 70 % | 95 % | 100 % | **100 %** |
| 0.3 | 3 mm | ±5 mm | 15 % | 90 % | 45 % | **100 %** |
| 0.3 | 1 mm | ±2 mm | 15 % | 60 % | 100 % | **100 %** |
| 0.3 | 1 mm | ±5 mm | 0 % | 45 % | 25 % | **95 %** |
| 0.8 | 3 mm | ±2 mm | 60 % | 100 % | 90 % | **100 %** |
| 0.8 | 3 mm | ±5 mm | 10 % | 100 % | 25 % | **65 %** |
| 0.8 | 1 mm | ±2 mm | 10 % | 55 % | 50 % | **85 %** |
| 0.8 | 1 mm | ±5 mm | 5 % | 45 % | 15 % | **55 %** |

**Table 3** — Cycle time of successful trials [s] / force-cap violations (of 20) per cell.

| μ | Clearance | Uncertainty | Open-loop | Guarded + retry | Impedance | Spiral search |
|---|---|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 5.8 / 1 | 6.6 / 1 | 10.7 / 0 | 10.7 / 0 |
| 0.3 | 3 mm | ±5 mm | 5.7 / 3 | 18.5 / 2 | 10.7 / 0 | 12.7 / 0 |
| 0.3 | 1 mm | ±2 mm | 5.7 / 6 | 30.7 / 0 | 10.7 / 0 | 10.8 / 0 |
| 0.3 | 1 mm | ±5 mm | — / 3 | 25.2 / 1 | 10.7 / 0 | 14.1 / 0 |
| 0.8 | 3 mm | ±2 mm | 5.8 / 3 | 7.3 / 0 | 10.7 / 2 | 10.8 / 0 |
| 0.8 | 3 mm | ±5 mm | 5.6 / 3 | 17.3 / 0 | 19.5 / 2 | 20.3 / 0 |
| 0.8 | 1 mm | ±2 mm | 5.7 / 7 | 30.4 / 0 | 24.2 / 10 | 18.1 / 0 |
| 0.8 | 1 mm | ±5 mm | 5.6 / 1 | 27.6 / 1 | 25.5 / 1 | 27.8 / 0 |

Spiral search is the only controller with zero force-cap violations in every cell. Its mean 10 ms-average peak force is 1–6 N, against 10–35 N for open-loop and 3–7 N for guarded retry.

### 5.2 Clearance ratchet (weeks 2–3)

**Table 3a** — Success rate, 20 seeds per cell, uncertainty within the chamfer's capture range. Source: `results/ratchet_{straight,impedance}/summary.csv`. All 10 mm and 5 mm cells are 100 % for both controllers.

| Friction μ | Clearance | Uncertainty | Open-loop | Impedance insert |
|---|---|---|---|---|
| 0.3 | 3 mm | ±2 mm | 70 % | 100 % |
| 0.3 | 1 mm | ±1 mm | 45 % | 100 % |
| 0.3 | 1 mm | ±2 mm | 15 % | 100 % |
| 0.8 | 3 mm | ±2 mm | 60 % | 90 % |
| 0.8 | 1 mm | ±1 mm | 25 % | 65 % |
| 0.8 | 1 mm | ±2 mm | 10 % | 30 % |

### 5.3 Figures

> **Figure 2** — TODO: success rate vs. uncertainty, one line per controller, one panel per clearance.
>
> **Figure 3** — TODO: cycle-time distribution (successful trials) per controller.
>
> **Figure 4** — TODO: contact force vs. time for one representative trial per controller (impact spike vs. sustained force).

---

## 6. Discussion

> TODO: failure-mode breakdown per controller. Points to address:
> - Clearance alone is not what makes the task hard: with zero uncertainty the open-loop insert tracks the hole axis to 0.08 mm and seats at 1 mm clearance. Uncertainty is the driver.
> - The guarded baseline's success at 1 mm is bounded by the time budget: each attempt lands inside a ~1 mm capture region with low probability and costs ~0.9 s (median, including the approach).
> - Impact spikes come from the arm's reflected inertia, not controller stiffness: both controllers spike at touchdown, ∝ touchdown speed. A slow touchdown zone, not compliance, is what keeps them small.
> - Friction dominates chamfer-guided insertion (δ = P(1−μ)/(K_xy(1+μ))). At μ = 0.8, spiral search success at 1 mm / ±5 mm drops to 55 %; timeouts dominate, with mean search time 33 s, ~34 pauses and 16 s stalled per trial. The chamfer barely guides the peg, so centring has to do more of the work. At 3 mm / ±5 mm, guarded retry (100 %) beats spiral search (65 %) at μ = 0.8: lifting off between attempts avoids dragging a high-friction peg across the surface.
> - Guarded retry at 1 mm is bounded by the time budget: each attempt lands inside a ~1 mm capture region with low probability and costs ~0.9 s.
> - Simulation validity: penetration stays < 0.1 mm; MuJoCo's soft friction allows slow creep under sustained lateral load (0.2 mm/s at 25 N, μ = 0.3).

---

## 7. Limitations and Future Work

- Rigid connector and cable; no deformable cable dynamics.
- Rotation uncertainty not yet evaluated (the believed pose is offset in xy only).
- Peg welded to the flange; a real gripper adds in-hand slip and grasp-pose uncertainty.
- Simulation only; contact parameters were validated for stability, not against physical measurements.

> TODO: stretch results if attempted (expanded grid, partially deformable connector, learned policy from scripted demonstrations).

---

## References

> TODO: MuJoCo (Todorov et al., 2012); MuJoCo Menagerie; Siciliano et al., *Robotics: Modelling, Planning and Control* (DLS IK, impedance control); Hogan (1985) impedance control; spiral search for peg-in-hole (e.g. Park et al.); Intrinsic AI for Industry Challenge.
