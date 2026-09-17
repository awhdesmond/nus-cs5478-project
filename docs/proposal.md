# Vision-Guided, Contact-Rich Connector Insertion with Impedance Control

## 1. Motivation

Electronics assembly, particularly connectors into sockets, remains one of the least automated processes in modern manufacturing. It is difficult for robots due to several reasons. First, it is contact-rich, at millimetre scale, pose uncertainty between the gripper and the socket exceeds clearance, so a purely position-controlled arm jams or damages the part rather than inserting it. Secondly, parts are **keyed**, each connector fits only in its own port, so the robot must identify which connector goes where before it can insert it. With the increasing demand for computing servers and data centers, automating electronics assembly could reduce manufacturing lead time and increase production throughput.

## 2. Project Task & Scene Description

The proposed project task is a simplified version of the electronics assembly problem. Using a robot arm and a fixed overhead camera, the robot has to identify, grasp, and insert three different connectors into their matching ports on a board.

**Success criterion:** All three connectors are seated to full depth in their correct ports, each insertion within a 60 s budget, without the sustained contact force exceeding a 30 N cap.

### Scene Components

1. **Robot, Gripper & Sensors:** A 6-DoF manipulator (UR5e) whose joints are driven by torque:

   | Component | Specification |
   |---|---|
   | Arm | UR5e, 6 DoF, joint-torque control |
   | Gripper | Custom parallel-jaw: two fingers on linear slides |
   | Force sensing | Six-axis force/torque sensor at the wrist |
   | Camera | RGB-D, 1280 × 960, 25° vertical field of view, fixed 0.75 m above the table pointing down |

   The camera is fixed above the table rather than on the arm, and images are taken from a pose that keeps the arm out of its field of view.

2. **Connectors:** Three parts are placed at random, non-overlapping positions (no occlusions) and orientations in a pick zone:

   | Part | Cross-section | Starting orientation | Clearance in its port |
   |---|---|---|---|
   | Round peg | Ø10 mm, 50 mm long | upright or lying | **1–3 mm** (grid) |
   | Square peg | 8.5 mm across the flats, 50 mm long | upright or lying | **1–3 mm** (grid) |
   | Rectangular card | 20 × 6 mm, 50 mm long | always upright, as in a tray | **1 mm** (its own tolerance) |

   Each connector carries a coloured band on its **insertion end**. An upright part stands on that end; a lying one may point either way, so the robot has to find it from the image and insert that end first.

3. **Board & Ports:** One board carries all three ports, a round hole, a square hole and a rectangular slot, at known offsets from the board's own frame. The board's position and rotation on the table are randomised per trial, so the robot must locate the board and derive the port poses from it rather than being told where each port is.

4. **Uncertainty:** During simulation and testing, the robot never observes the ground-truth poses. Connector and port poses come from perception, so perception error (up to a few millimetres, plus simulated depth noise) is the source of pose uncertainty at insertion.

### Simulation

* Connectors are simulated as rigid parts standing in for plug housings: two pegs and a thin card. Sizes are chosen so each part physically fits only its own port, mirroring connector keying, the round peg is too wide for the square hole, the square peg's diagonal too wide for the round one, and both are too thick for the card's slot. Keying holds at the card's 1 mm clearance and the harness scores "correct port" explicitly rather than relying on the fit.

* No cable is modelled. Deformable-cable simulation is a research problem in its own right and would shift the project away from its algorithmic content.

* Grasp poses come from antipodal grasp using the camera's data, and are planned to be ones that would hold under friction. Once both fingers contact a part it is locked to the gripper; where the fingers closed still sets the in-hand pose, so perception error carries through to insertion. A friction grasp is a stretch goal.

## 3. Approach

### 3.1 Algorithms

1. **Perception and identification.** Classical computer vision on the overhead RGB-D image. Identify the parts and the board by depth and colour, classify each part as round, square or rectangular from its footprint, find a lying part's marked insertion end from its colour band, and estimate positions and yaw via the calibrated camera model.

2. **Inverse kinematics & trajectory path planning.** Damped least-squares numerical IK with joint limits; quintic joint-space trajectories and trapezoidal Cartesian lines, with a collision-avoiding approach through a via pose.

3. **Grasp Planning.** An upright connector is a prism, so its side-surface normals can be estimated from the outline of its top face as seen by the camera. For each candidate closing direction, the planner estimates the normals at the two finger contact patches and keeps the grasp if both lie inside the friction cones (force closure), the contacts are on flats rather than corners, and the fingers clear the table and the other parts. Then approach–close–lift–transport–release sequencing, with an in-hand pose estimate, from perception and from the finger positions at contact, used to place the connector tip at the port. The card is grasped across its 6 mm thickness, with the pads on its large faces. A peg lying on its side is grasped across its width and turned in the hand to point down, rather than being put back down and regrasped; the arm configuration for the sideways insertion is chosen by a clearance check against the perceived obstacles.

4. **Compliant contact control.** A torque-level Cartesian impedance controller at the connector tip, with low lateral and rotational stiffness so the port chamfer can guide the connector. A simulated wrist force-torque sensor provides contact detection and force monitoring.

5. **Contact-based search and alignment.** A spiral search under light normal force, triggered by force-sensed contact, with force-guided centring on the chamfer, to find the port opening despite perception error; transition to insertion on a depth drop. The card's slot has an anisotropic capture region, forgiving along its length, tight across its thickness, so its search is biased across the thin axis rather than sweeping a circle.

### 3.2 Systems

1. **Simulation Environment.** MuJoCo, chosen over PyBullet for more accurate and numerically stable contact and friction at millimetre clearances. Scene parameters (clearance, friction, connector layout, depth noise) are configurable.

2. **Evaluation Harness.** An automated runner that executes seeded trials across a parameter grid, separates what controllers may observe from ground truth, and logs stage-wise outcomes, perception errors, cycle time and contact force, so results are reproducible and statistical rather than anecdotal.


## 4. Evaluation

Trials run across unseen random seeds, similar to the default project's per-seed scoring, where each seed fixes a connector layout and the robot's start pose.

**Full-task grid:** peg orientation {all upright, mixed upright and lying} × peg clearance {3, 1} mm × friction {0.3, 0.8} × camera noise {off, on}, 20–50 seeds per condition, with the board pose and all three part poses resampled per seed. The card is upright at its own 1 mm clearance throughout. A robustness condition adds an **independent per-object pose error** on top of the camera noise, since a single camera's calibration error is largely common to a connector and the board.

**Insertion-module grid** (insertion only, synthetic pose uncertainty): clearance {3, 2, 1} mm × uncertainty {±1, ±3, ±5} mm × friction {0.3, 0.8}, 50 seeds per condition.

**Metrics.** Success rate with 95 % Wilson intervals; controllers compared pairwise on identical seeds with an exact McNemar test, and cycle time compared only on trials both solved; peak sustained contact force (10 ms average, with single-step peaks reported separately); perception error against ground truth; and a stage-wise failure breakdown (perception / grasp / planning / insertion / release).

## 5. Relationship to Default Project

The default Pick-Nav-Reach project (PyBullet, Fetch robot) asks students to generate antipodal grasps from the object's point cloud and normals, plan motions to pick a YCB object with a friction grasp, navigate a maze, and bring the object within 10 cm of a goal.

| | Default project | This project |
|---|---|---|
| Perception | Ground truth provided | Overhead RGB-D camera; segmentation, identification and pose estimation from images |
| Grasp synthesis | Antipodal grasps from ground-truth point cloud and normals | Antipodal grasps from the camera's own view of the part, with normals estimated from its perceived outline |
| Grasp physics | Friction grasp; hold and carry stability scored | Attach-on-grasp with antipodal grasp poses (friction grasp is a stretch goal) |
| Objects | YCB objects (irregular shapes) | Three keyed connectors (round peg, square peg, thin card), upright or lying; correct part–port assignment |
| Motion | Motion planning and maze navigation | IK, trajectory generation and collision-avoiding approach; no mobile base |
| Precision and contact | 10 cm placement; no force control | Millimetre-clearance insertion; compliant force control and contact-based search |

Both are pipelines of algorithmic modules evaluated across unseen seeds. This project trades navigation and irregular-object grasping for real camera perception, part identification and contact-rich fine insertion, where the principal difficulty lies.

