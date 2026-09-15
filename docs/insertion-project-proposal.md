# Project Proposal: Vision-Guided Contact-Rich Connector Insertion with Compliant Control

**Course:** Intelligent Robots — Algorithms and Systems
**Project type:** Custom project (in lieu of the default Pick-Nav-Reach project)
**Duration:** 8 weeks (scope extended after week 5; see §10)

---

## 1. Motivation

Electronics assembly — specifically inserting connectors into sockets — remains one of the
least automated steps in modern manufacturing. It is hard for robots for two reasons.
First, it is *contact-rich*: at the millimetre scale, pose uncertainty between the gripper
and the socket exceeds the insertion clearance, so a purely position-controlled arm jams or
damages the part rather than seating it. Second, parts are *keyed*: each connector fits only
its own port, so the robot must identify which part goes where before it can insert it.

## 2. Task Definition

A 6-DOF manipulator with a parallel-jaw gripper and a fixed overhead RGB-D camera assembles
**two keyed connectors** into a fixture:

- **Scene.** A round connector and a square connector stand upright at random,
  non-overlapping positions and orientations in a pick area. A fixture holds two sockets:
  a round hole and a square hole, each with **1–3 mm clearance**.
- **Sequence.** Using the camera, the robot identifies the **round connector first**,
  grasps it, inserts it into the round socket and releases it; it then identifies the
  square connector, grasps it and inserts it into the square socket.
- **Uncertainty.** The robot never observes ground-truth poses. Connector and socket
  poses come from perception, so perception error (up to a few millimetres, plus
  simulated depth noise) is the source of pose uncertainty at insertion.

**Modelling decisions.**
- **Connectors are rigid keyed pegs** (round Ø10 mm, square 8.5 mm, 50 mm long) standing in for plug
  housings. Sizes are chosen so each connector physically fits only its own socket at
  1 mm clearance, mirroring connector keying.
- **No cable is modelled.** Deformable-cable simulation is a research problem in its own
  right (see §8); it would shift the project away from its algorithmic content.
- **Grasping is built in two stages.** Grasp poses always come from antipodal grasp
  synthesis on the camera's point cloud. The pipeline is first built with attach-on-grasp
  (once both fingers contact a connector it is locked to the gripper; where the fingers
  closed still sets the in-hand pose, so perception error carries through). It is then
  upgraded to a friction grasp, where the connector is held only by finger friction and can
  slip; attach-on-grasp remains the fallback.

**Success criterion.** Both connectors are seated to full depth in their correct sockets,
within the cycle-time budget, without the sustained contact force (10 ms moving average)
exceeding a 30 N cap.

## 3. Approach

### 3.1 Algorithms

1. **Perception and identification.** Classical computer vision on the overhead RGB-D
   image: segment connectors and socket holes by depth and colour, classify each as round
   or square from contour shape (circularity and polygon approximation), and estimate
   position from the centroid and depth via the calibrated camera model, and yaw of square
   parts from a minimum-area rectangle.

2. **Inverse kinematics and trajectory generation.** Damped least-squares numerical IK with
   joint limits; quintic joint-space trajectories and trapezoidal Cartesian lines, with a
   collision-avoiding approach through a via pose. Builds directly on the course's
   Jacobian and kinematics material.

3. **Antipodal grasp planning.** Back-project the connector's depth pixels to a point cloud,
   estimate surface normals, and sample finger contact pairs for the parallel-jaw gripper,
   keeping pairs whose contact line lies inside both friction cones (force closure) and
   that clear the table and the other connector. Then approach–close–lift–transport–release
   sequencing, grip-force control for the friction grasp, and an in-hand pose estimate used
   to place the connector tip at the socket.

4. **Compliant contact control.** A torque-level Cartesian impedance controller at the
   connector tip, with low lateral and rotational stiffness so the socket chamfer can guide
   the connector. A simulated wrist force-torque sensor provides contact detection and
   force monitoring.

5. **Contact-based search and alignment.** A spiral search under light normal force,
   triggered by force-sensed contact, with force-guided centring on the chamfer, to find the
   socket opening despite perception error; transition to insertion on a depth drop.

### 3.2 Systems

1. **Simulation environment.** MuJoCo, chosen over PyBullet for more accurate and
   numerically stable contact and friction at millimetre clearances. Scene parameters
   (clearance, friction, connector layout, depth noise) are configurable.

2. **Evaluation harness.** An automated runner that executes seeded trials across a
   parameter grid, separates what controllers may observe from ground truth, and logs
   stage-wise outcomes, perception errors, cycle time and contact force — so results are
   reproducible and statistical rather than anecdotal.

## 4. Evaluation and Grading Criteria

Trials run across unseen random seeds, in the spirit of the default project's per-seed
scoring. Each seed fixes a connector layout and the robot's start pose.

**Full-task grid:** clearance {3, 1} mm × friction {0.3, 0.8} × depth noise {off, on},
20–50 seeds per condition. **Insertion-module grid** (insertion only, synthetic pose
uncertainty): clearance {3, 2, 1} mm × uncertainty {±1, ±3, ±5} mm × friction {0.3, 0.8},
50 seeds per condition.

### Task success — 50%

| Criterion | Weight |
|---|---|
| Round connector correctly identified and picked first | 10% |
| Both connectors grasped and transported without a drop | 10% |
| Both connectors seated in their correct sockets | 20% |
| Within cycle-time budget and sustained-force cap | 10% |

### Perception — 10%

| Criterion | Weight |
|---|---|
| Classification accuracy and pose error (position, yaw) against ground truth | 10% |

### Robustness — 25%

| Criterion | Weight |
|---|---|
| Success rate across clearance, friction and uncertainty levels | 15% |
| Performance degrades gracefully at the hardest conditions | 10% |

### Efficiency — 15%

| Criterion | Weight |
|---|---|
| Insertion cycle time relative to a guarded-move-with-retry baseline | 15% |

*(Weights adjustable to instructor preference.)*

## 5. Deliverables

- Source code: simulation environment, perception, controllers and evaluation harness
- Results: success rates by stage and condition; perception error distributions;
  controller comparison (open-loop, guarded retry, impedance, spiral search);
  contact-force and cycle-time plots
- Written report (~3–5 pages): task setup, method, results, discussion
- Demo video of a full two-connector assembly, with camera view and search trace

## 6. Timeline

Assumes roughly 10–16 hours per week.

| Week | Focus | Definition of done |
|---|---|---|
| 1 | Environment + evaluation harness | MuJoCo scene; torque-controlled arm; stable contact; forces readable; harness logging; one easy insertion |
| 2 | Nominal motion + baseline | IK and trajectories reach the pre-insertion pose from varied starts; guarded-move-with-retry baseline |
| 2–3 | Compliance controller | Cartesian impedance validated in isolation; F/T contact detection |
| 3.5–4 | Search and alignment | Spiral search; end-to-end insertion at 1 mm clearance |
| 5 | Evaluation and tuning | Insertion grids; failure modes diagnosed; spiral search vs. baseline |
| 6–7 | Perception, grasping, two connectors | Scene with fixture and connectors; perception check; antipodal grasping with attach-on-grasp; pick and release; square insertion; full pipeline grid |
| 8 | Friction grasp; results and writeup | Friction grasp meets its slip and success criteria (or is reported as a partial result); plots, report, demo |

Weeks 1–5 are complete (§10).

## 7. Risk Assessment and Contingency

**Principal risks**

| Risk | Mitigation |
|---|---|
| Perception error exceeds what search can absorb | Target < 1 mm position and < 2° yaw error, validated against ground truth before grasping is built on it |
| Fingers collide with the socket fixture during insertion | Grasp high on the connector; check finger clearance when building the scene |
| Picking or inserting one connector disturbs the other | Minimum spacing in the layout; approach from above; re-capture the image before the second pick |
| Square connector jams from yaw error | Yaw alignment from perception, low rotational stiffness about the insertion axis, chamfered socket |
| Round socket built from primitives gives poor contact | Validate with the existing contact sanity checks before use |
| Friction grasp slips under insertion force | Grip force sized with margin over the insertion push; in-hand slip is measured; attach-on-grasp remains the fallback |

**Descoping order.** If the extension slips, scope is reduced in this order:

1. Keep attach-on-grasp for the headline results; report the friction grasp as partial
2. Shrink the full-task grid (one clearance, one friction level)
3. One connector per trial instead of two
4. Replace perceived poses with noisy ground-truth poses (keeping identification by shape)
5. Report the insertion-module results only

**Protected scope.** The compliance controller, spiral search and the insertion-module
results (already complete) are never descoped — together they are the minimum viable result.

## 8. Descoped Extensions

These are explicitly *not* part of the committed scope. They would be attempted only if the
core project is complete and validated with time remaining.

- **Socket tilt uncertainty.** Socket axis tilted a few degrees from its perceived orientation.
- **Partially deformable connector.** A short 2–3 link chain with joint compliance behind
  the plug, so the connector can lag and swing relative to the gripper.
- **Fully deformable cable.** A MuJoCo composite cable with tip state estimation, since the
  gripper pose would no longer determine the plug pose. Not achievable within this
  project's time budget.

## 9. Relationship to the Default Project

The default Pick-Nav-Reach project (PyBullet, Fetch robot) asks students to generate
antipodal grasps from the object's point cloud and normals, plan motions to pick a YCB
object with a friction grasp, navigate a maze, and bring the object within 10 cm of a goal.
Its environment provides ground-truth object poses, point clouds and normals.

| | Default project | This project |
|---|---|---|
| Perception | Ground truth provided | Overhead RGB-D camera; segmentation, identification and pose estimation from images |
| Grasp synthesis | Antipodal grasps from ground-truth point cloud and normals | Antipodal grasps from the camera's point cloud, with estimated normals |
| Grasp physics | Friction grasp; hold and carry stability scored | Attach-on-grasp, then friction grasp; drops and in-hand slip measured |
| Objects | YCB objects (irregular shapes) | Two keyed connectors (simple shapes); correct part–socket assignment |
| Motion | Motion planning and maze navigation | IK, trajectory generation and collision-avoiding approach; no mobile base |
| Precision and contact | 10 cm placement; no force control | Millimetre-clearance insertion; compliant force control and contact-based search |

Both are pipelines of algorithmic modules evaluated across unseen seeds. This project
trades navigation and irregular-object grasping for real camera perception, part
identification and contact-rich fine insertion, where the principal difficulty lies.

## 10. Scope Update and Progress (end of week 5)

The original proposal covered compliant insertion of a single rigid connector from a
randomised pre-insertion pose. That scope is complete:

- **Insertion module.** A torque-level impedance controller with contact-triggered spiral
  search and force-guided centring. On the insertion-module grid (900 trials per controller,
  50 seeds per condition) it succeeds in 99.1 % of trials, against 86.1 % for
  guarded move with retry, 78.1 % for impedance control without search and 56.4 % for
  open-loop insertion.
- **Paired comparison** (160 trials, same seeds): spiral search solved every trial;
  37 trials were solved only by spiral search and none only by the baseline
  (exact McNemar p ≈ 1.5 × 10⁻¹¹). At 1 mm clearance it was also 1.5–2.5× faster.

Because the core was completed ahead of schedule, the project is extended with
perception, identification and grasping of two keyed connectors (§2). This makes the
pose uncertainty come from a real sensing pipeline rather than a synthetic offset, and
adds the part-identification problem that keyed connectors pose in real assembly.
