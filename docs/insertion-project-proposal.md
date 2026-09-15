# Project Proposal: Contact-Rich Wire Connector Insertion with Compliant Control

**Course:** Intelligent Robots — Algorithms and Systems
**Project type:** Custom project (in lieu of the default Pick-Nav-Reach project)
**Duration:** 6 weeks

---

## 1. Motivation

Electronics assembly — specifically the insertion of wire terminals and connectors into
sockets — remains one of the least automated steps in modern manufacturing. The task is
difficult for robots because it is *contact-rich*: at the millimetre scale, pose
uncertainty between the gripper and the socket exceeds the insertion clearance, so a
purely position-controlled arm will jam or damage the part rather than seat it.

## 2. Task Definition

A 6-DOF manipulator inserts a **rigid wire terminal / connector** into a socket with
**1–3 mm clearance**, starting from a randomised pre-insertion pose with bounded
uncertainty (approximately ±5–10 mm translation, ±5° rotation).

**Scoping decision.** The cable behind the connector is *not* modelled as a deformable
body. The stiff region near the plug is treated as rigid, so the connector's pose is
determined by the gripper. This keeps the project focused on contact-rich control and
alignment under uncertainty — the algorithmic content — rather than on deformable-object
simulation, which is a research problem in its own right (see §7).

**Success criterion.** The connector reaches full seated depth in the socket without
exceeding a contact-force cap.

## 3. Approach

### 3.1 Algorithms

1. **Inverse kinematics and trajectory generation.** Jacobian-based numerical IK
   (damped least squares) to reach a pre-insertion pose above the socket, with
   quintic or trapezoidal joint-space trajectory generation for smooth motion.
   This component builds directly on DH-convention and Jacobian material from the
   course.

2. **Compliant contact control.** A torque-level Cartesian impedance controller at the
   end-effector, with a simulated wrist force-torque sensor used for contact detection
   and force limiting. This is the core of the project: pure position control cannot
   succeed at these tolerances, because the arm must yield to contact constraints
   rather than fight them.

3. **Contact-based search and alignment.** A spiral search strategy triggered on
   surface contact, which explores laterally to locate the socket opening despite
   pose uncertainty, then transitions to insertion once alignment is detected.

### 3.2 Systems

1. **Simulation environment.** MuJoCo, chosen over PyBullet for its more accurate and
   numerically stable contact/friction modelling — essential when the algorithmic
   contribution *is* the contact behaviour, and when clearances are millimetre-scale.
   Scene parameters (clearance, pose uncertainty, friction) are configurable.

2. **Evaluation harness.** An automated runner that executes N trials per condition
   across a parameter grid and logs success, cycle time, and peak contact force —
   so results are reproducible and statistical rather than anecdotal.

## 4. Evaluation and Grading Criteria

Trials are run across a grid of unseen random seeds and task conditions, in the spirit
of the default project's per-seed scoring.

**Baseline evaluation grid (Plan B):** 2 clearance levels × 2 uncertainty levels, with
20 trials (unseen seeds) per condition.

### Insertion success — 60%

| Criterion | Weight |
|---|---|
| Reaches contact within sim-step budget | 15% |
| Successfully aligns and seats the connector | 15% |
| Completes within cycle-time budget | 15% |
| Peak contact force stays below safety cap | 15% |

### Robustness — 25%

| Criterion | Weight |
|---|---|
| Success rate across uncertainty levels | 15% |
| Performance degrades gracefully at maximum uncertainty | 10% |

### Efficiency — 15%

| Criterion | Weight |
|---|---|
| Mean cycle time relative to a guarded-move-with-retry baseline | 15% |

## 5. Deliverables

- Source code: simulation environment, controllers, and evaluation harness
- Results: success-rate tables and plots over the clearance × uncertainty grid;
  contact-force and cycle-time plots
- Written report (~3–5 pages): task setup, method, results, discussion
- Demo video or trajectory replay of a successful insertion

## 6. Timeline

### Plan B — baseline schedule

Assumes roughly 10–16 hours per week (2–3 weekday evenings plus weekend time).
Compliance-controller work is deliberately allocated to weekend blocks, since
iterative gain tuning benefits from contiguous time.

| Week | Focus | Definition of done |
|---|---|---|
| 1 | Environment + evaluation harness | MuJoCo scene loads; arm commandable; stable contact; forces readable; harness logging; one easy-mode insertion |
| 2 | Nominal motion + baseline | IK reaching pre-insertion pose; guarded-move-with-retry baseline working |
| 2–3 | Compliance controller | Wrist F/T sensing integrated; arm yields correctly on contact |
| 3.5–4 | Search and alignment | Spiral search implemented; first full end-to-end insertion success |
| 5 | Evaluation and tuning | Grid executed; failure modes diagnosed; spiral search compared against baseline |
| 6 | Results and writeup | Plots, report, demo produced |

### Plan A — stretch schedule

If progress runs ahead of the baseline, the project expands as follows:

- **Expanded evaluation grid:** 3 clearance levels × 3 uncertainty levels with 50 trials
  per condition, substantially increasing statistical confidence in the reported
  success rates.

## 7. Descoped Extensions

These are explicitly *not* part of the committed scope, and are recorded here to make
the scoping decision in §2 transparent. They would be attempted only if the core
project is complete and validated with time remaining.

- **Partially deformable connector.** Replace the rigid connector with a short 2–3
  link chain with joint compliance, so the plug can lag and swing relative to the
  gripper. This tests whether the controller tolerates an under-actuated tip.
- **Fully deformable cable.** Model the cable as a MuJoCo composite body and add tip
  state estimation, since the gripper pose would no longer determine the plug pose.
  This is comparable in difficulty to the Intrinsic challenge task and is not
  achievable within this project's time budget.

## 8. Risk Assessment and Contingency

**Principal risks**

| Risk | Mitigation |
|---|---|
| Contact instability in simulation (jitter, penetration) during week 1 | Begin with loose clearances (8–10 mm) and tighten only once contact is stable; budget explicit time for `solref`/`solimp`/timestep tuning |
| Impedance gain tuning consumes more time than allocated | Scheduled into weekend blocks; treated as the highest-priority component and never descoped |
| Evaluation grid produces failures that are expensive to diagnose | Grid size is a scaling lever, not a fixed requirement (see below) |
| Loss of momentum between short work sessions | Weeknights reserved for well-defined, low-context tasks (logging, plotting, trajectory code); deep debugging reserved for longer blocks |

**Descoping order.** If the schedule slips, scope is reduced in this order:

1. Shrink the evaluation grid (down to 1 clearance level × 1 uncertainty sweep)
2. Convert the force-safety criterion from a tuned metric to a simple pass/fail cap
3. Report graceful degradation from the measured success-rate curve rather than engineering for it
4. Fall back from spiral search to a simpler guarded-move-with-retry strategy

**Protected scope.** The compliance controller and at least one clean end-to-end
insertion success are never descoped — together they constitute the minimum viable
result for this project.

## 9. Relationship to the Default Project

This project is comparable in structure to the default Pick-Nav-Reach project — a
pipeline of several algorithmic modules (IK, motion planning, task-specific strategy)
evaluated across unseen seeds — but replaces coarse navigation with contact-rich fine
manipulation. Where the default project's placement tolerance is 10 cm and requires no
force control, this task operates at millimetre clearances and requires closed-loop
compliant control against contact forces, which is the principal source of added
difficulty.
