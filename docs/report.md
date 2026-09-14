# Contact-Rich Connector Insertion with Compliant Control

**CS5478 Intelligent Robots — Project Report** · *Draft skeleton (week 2). `TODO` marks sections filled in later weeks.*

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
- **Success**: the peg tip reaches within 2 mm of the hole bottom, inside the hole footprint, within the cycle-time budget (60 s), without the contact force exceeding the 30 N cap.

### 2.2 Simulation environment

- **Simulator**: MuJoCo 3.x at 1 ms timestep, implicit-fast integrator, elliptic friction cones. Chosen over PyBullet for more stable, accurate contact at millimetre scale.
- **Robot**: UR5e from MuJoCo Menagerie. The stock position servos are replaced by joint-torque motors (±150 Nm base joints, ±28 Nm wrist) so the controller owns gravity compensation and compliance.
- **Socket**: built from box primitives (MuJoCo collides meshes via their convex hull, which would fill the hole), with a 2 mm 45° chamfer formed from rotated boxes. Placed where the insertion path stays clear of the UR5e's elbow, wrist and shoulder singularities.
- **Sensing**: a force/torque sensor at the flange–peg interface, tared at rest and low-pass filtered (30 Hz).

> **Figure 1** — TODO: scene render (whole arm) + socket close-up with chamfer. Source: `scripts/replay.py --video`, frame grab.

### 2.3 Contact validation

Default MuJoCo contact parameters let the peg sink 1.16 mm into the chamfer lip under a 30 N push — more than the 0.5 mm per-side gap at 1 mm clearance, so an "insertion" could succeed by penetration rather than alignment. Stiffer contact parameters (`solref` = 5 ms, `solimp` = [0.95, 0.99, 1 mm]) reduce this to 0.21 mm. We then press the peg against the top surface (up to 40 N), the chamfer lip, the hole bottom and a hole wall at 1 mm clearance:

**Table 1** — Contact sanity check (1 mm clearance, 1 ms timestep). Source: `results/contact_sanity.csv`.

| Scenario | Mean force [N] | Force jitter | Max penetration [mm] |
|---|---|---|---|
| Top surface (max) | 40.3 | 1.5 % | 0.054 |
| Chamfer lip | 16.6 | 0.3 % | 0.038 |
| Hole bottom | 18.1 | 0.02 % | 0.041 |
| Hole wall | 10.0 | 0.06 % | 0.092 |

All scenarios stay below 0.5 mm penetration and 5 % jitter with no solver warnings; halving the timestep changes nothing, so 1 ms is used throughout.

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

> TODO (weeks 2–3): Cartesian impedance law at torque level, τ = Jᵀ(K Δx − D ẋ) + τ_bias, with low rotational stiffness; gain selection; validation that the arm yields on surface contact and a tilted peg flattens against a surface. **Figure** — force/position response to surface contact.

### 3.6 Contact-triggered spiral search

> TODO (weeks 3.5–4): Archimedean spiral in the surface plane under constant normal force; pitch chosen from capture width (clearance + chamfer); hole detection by z-drop; transition to insertion. Search-time bound ≈ π r² / pitch / v.

---

## 4. Evaluation Protocol

- **Harness**: every controller implements the same interface (observation → joint torques). Controllers see the model (for kinematics), joint state, filtered wrist F/T and the believed entrance only. Ground-truth contact force, penetration and the true socket pose are used solely for scoring.
- **Grid**: clearance {3, 1} mm × uncertainty {±2, ±5} mm × 20 seeds per cell. Seeds fix both the socket offset and the start pose, so all controllers face identical trials.
- **Metrics**: success rate; cycle time over successful trials; peak contact force, both single-step (includes ~1 ms impact spikes) and 10 ms moving average; failure reason (timeout / force cap).

> TODO (week 5): state the final rubric mapping (insertion success 60 %, robustness 25 %, efficiency 15 %) and how each metric feeds it.

---

## 5. Results

### 5.1 Baselines (preliminary, week 2)

**Table 2** — Success rate over 20 seeds per cell. Source: `results/{straight,guarded}_grid/summary.csv`.

| Clearance | Uncertainty | Open-loop | Guarded + retry | Guarded cycle time [s] |
|---|---|---|---|---|
| 3 mm | ±2 mm | 60 % | 100 % | 7.3 ± 4.2 |
| 3 mm | ±5 mm | 10 % | 100 % | 17.3 ± 12.0 |
| 1 mm | ±2 mm | 10 % | 55 % | 30.4 ± 14.4 |
| 1 mm | ±5 mm | 5 % | 45 % | 27.6 ± 15.3 |

> TODO (week 5): add the compliant controller column and rerun all three on the final grid.

### 5.2 Compliant controller

> **Table 3** — TODO: success rate, cycle time, peak force per cell for impedance + spiral search.
>
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
> - Impact spikes: a stiff position-controlled arm produces 12–29 N single-step contact force when touching down at 1 cm/s, decaying within ~6 ms (10 ms average ≈ 7 N). Compliance should reduce both.
> - Simulation validity: penetration stays < 0.1 mm; MuJoCo's soft friction allows slow creep (0.7 mm/s at 40 N), relevant to spiral search.

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
