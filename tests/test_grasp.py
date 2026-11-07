"""Step 4: grasping, pick and carry."""

import math

import numpy as np
import pytest

from perception.pegs import PegEstimate, wrap_quarter_turn
from grasp.plan import GraspConfig, closing_direction, plan_grasp


def polygon_outline(corners, per_side=60) -> np.ndarray:
    """Densely sampled boundary of a polygon, (N, 2)."""
    corners = np.asarray(corners, float)
    t = np.linspace(0, 1, per_side, endpoint=False)[:, None]
    return np.vstack([a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, axis=0))])


def peg(kind, outline, centre=(0.1, 0.5)) -> PegEstimate:
    """A PegEstimate around `outline`, top face at z = 0.22, as perception would give it."""
    return PegEstimate(kind, np.array([*centre, 0.22]), 0.0, 0.01, 0.0, outline + centre)


def square(side, yaw):
    h = side / 2
    turn = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return polygon_outline([[h, h], [-h, h], [-h, -h], [h, -h]]) @ turn.T


def circle(radius, n=240):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(a), np.sin(a)])


# --- 4a: grasp planning ----------------------------------------------------------------------------

@pytest.mark.parametrize("yaw_deg", [0, 20, -35, 44])
def test_square_is_grasped_across_two_faces(yaw_deg):
    g = plan_grasp(peg("square", square(0.0085, math.radians(yaw_deg))))
    d = closing_direction(g.yaw)
    # Closing along a face normal: the direction is the square's yaw, modulo 90°.
    assert abs(wrap_quarter_turn(math.atan2(d[1], d[0]) - math.radians(yaw_deg))) < math.radians(0.5)
    assert g.width == pytest.approx(0.0085, abs=2e-4)
    assert abs(g.yaw) <= math.pi / 4 + 1e-9                      # nearest face pair to the wrist's yaw 0
    np.testing.assert_allclose(g.tcp, [0.1, 0.5, 0.22 - GraspConfig().depth])


@pytest.mark.parametrize("reference", [0.0, 0.6, -1.2])
def test_round_peg_is_grasped_at_the_wrists_current_yaw(reference):
    g = plan_grasp(peg("round", circle(0.005)), reference_yaw=reference)
    assert abs(g.yaw - reference) < math.radians(3)              # any direction works: turn the wrist least
    assert g.width == pytest.approx(0.010, abs=2e-4)


def test_fingers_go_around_a_neighbouring_object():
    other = circle(0.005) + [0.1, 0.5 + 0.020]                   # another peg 20 mm away, along world y
    g = plan_grasp(peg("round", circle(0.005)), obstacles=[other])
    # At yaw 0 the fingers close along y, straight into it. They must turn until it's clear of the
    # 16 mm wide fingers (+3 mm clearance): asin(11 / 20) ≈ 33°.
    assert abs(g.yaw) > math.radians(33)


def test_friction_cone_decides_on_sides_that_are_not_parallel():
    # Four flat sides, no two parallel: across either pair, the normals are 15-17° apart, so each is
    # ~8° off the best closing axis. (A triangle wouldn't test this: a corner is always opposite a side.)
    quad = polygon_outline([[-0.004, -0.003], [0.004, -0.0019], [0.003, 0.0019], [-0.004, 0.003]])
    g = plan_grasp(peg("square", quad), cfg=GraspConfig(friction=0.5))       # cone ±26.6°: holds
    assert math.radians(6) < g.normal_angle < math.radians(10)
    with pytest.raises(ValueError):
        plan_grasp(peg("square", quad), cfg=GraspConfig(friction=0.1))       # cone ±5.7°: would slip


def test_grasps_on_perceived_pegs():
    from harness.perception_check import body_yaw
    from perception.estimate import perceive
    from sim.camera import OverheadCamera
    from sim.config import SceneConfig
    from sim.layout import sample_layout
    from sim.scene import build_model
    from sim.sim import Sim

    for seed in range(6):
        cfg = sample_layout(SceneConfig(), seed)
        sim = Sim(build_model(cfg), cfg)
        sim.reset("ready")
        with OverheadCamera(sim.model, cfg) as camera:
            pegs = perceive(camera.capture(sim.data)).pegs
        for kind, est in pegs.items():
            g = plan_grasp(est, [p.outline for k, p in pegs.items() if k != kind])
            assert g.normal_angle < math.radians(8), (seed, kind)
            if kind == "round":
                assert abs(g.yaw) < math.radians(5), seed
            else:
                d = closing_direction(g.yaw)
                error = wrap_quarter_turn(math.atan2(d[1], d[0]) - body_yaw(sim.data, "square_peg"))
                assert abs(error) < math.radians(2), (seed, math.degrees(error))

# --- 4b: pick and carry ----------------------------------------------------------------------------

from dataclasses import replace

from harness.perception_check import body_yaw
from perception.estimate import perceive
from motion.arm import Arm
from grasp.pick import carry_to_port, pick
from sim.camera import OverheadCamera
from sim.config import PEGS, SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim


def perceived_scene(seed, noise=0.002):
    """A fresh layout, the arm at ready, and what the camera makes of it."""
    cfg = replace(sample_layout(SceneConfig(), seed), depth_noise=noise)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    with OverheadCamera(sim.model, cfg) as camera:
        scene = perceive(camera.capture(sim.data, np.random.default_rng(seed)))
    return sim, scene


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("kind", PEGS)
def test_peg_is_picked_and_carried_to_hover_over_its_port(seed, kind):
    sim, scene = perceived_scene(seed)
    arm = Arm(sim)
    picked = pick(arm, scene, kind)
    assert picked.ok, picked
    assert sim.attached() == kind                                     # ground truth: really holding it
    carried = carry_to_port(arm, scene, picked)
    assert carried.ok
    tip, entrance = sim.data.site(f"{kind}_tip").xpos, sim.data.site(f"{kind}_port_entrance").xpos
    assert np.linalg.norm(tip[:2] - entrance[:2]) < 0.0005          # over the hole to 0.5 mm
    assert abs(tip[2] - entrance[2] - 0.010) < 0.001                 # hovering 10 mm above, to 1 mm
    upright = sim.data.body(f"{kind}_peg").xmat.reshape(3, 3)[2, 2]
    assert upright > math.cos(math.radians(1))                       # tilted under 1°
    if kind == "square":                                             # turned to fit the port
        error = wrap_quarter_turn(body_yaw(sim.data, "square_peg") - body_yaw(sim.data, "square_port"))
        assert abs(error) < math.radians(1)


def test_closing_on_nothing_is_detected():
    sim, scene = perceived_scene(1)
    # Move the round peg 10 cm away after the picture was taken: the fingers will close on empty air.
    joint = sim.model.joint("round_peg")
    sim.data.qpos[joint.qposadr[0]] += 0.10
    picked = pick(Arm(sim), scene, "round")
    assert not picked.ok and picked.stage == "grasp"
    assert picked.gap < 0.001                                        # fingers closed right up
    assert sim.attached() is None
