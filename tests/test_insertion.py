"""Step 5: compliant insertion."""

import numpy as np
import pytest

from motion.arm import Arm
from motion.impedance import CartesianImpedance, ImpedanceConfig
from motion.wrist import WristFT
from sim.config import SceneConfig
from sim.scene import build_model
from sim.sim import Sim


# --- 5a: impedance control and the wrist sensor ----------------------------------------------------

OFFSET = [0.003, 0.0, 0.038]  # a control point like a held peg's tip, in the tcp frame


@pytest.fixture
def setup():
    cfg = SceneConfig()
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    arm = Arm(sim)
    arm.hold(0.2)  # let the sensor see the gripper's weight before taring
    impedance, ft = CartesianImpedance(sim, arm.kin, OFFSET), WristFT(sim, arm.kin)
    ft.tare()
    return sim, impedance, ft


def run(sim, impedance, ft, seconds, target, R, push=None):
    """Impedance control toward `target` for `seconds`, optionally with an external force `push` [N, world]
    applied at the control point. Returns the last filtered F/T reading."""
    gripper = sim.model.body("gripper").id
    for _ in range(round(seconds / sim.model.opt.timestep)):
        if push is not None:
            # A force applied at a point other than the body's centre of mass also turns it: add r x F.
            x, _ = impedance.pose()
            sim.data.xfrc_applied[gripper, :3] = push
            sim.data.xfrc_applied[gripper, 3:] = np.cross(x - sim.data.xipos[gripper], push)
        tau, _ = impedance.torque(target, R)
        sim.step(tau)
        force = ft.update()
    sim.data.xfrc_applied[gripper] = 0
    return force


def test_impedance_holds_and_moves_the_control_point(setup):
    sim, impedance, ft = setup
    x0, R0 = impedance.pose()
    run(sim, impedance, ft, 0.5, x0, R0)
    assert np.linalg.norm(impedance.pose()[0] - x0) < 1e-5            # holds still
    run(sim, impedance, ft, 1.0, x0 + [0.005, 0, 0], R0)
    assert np.linalg.norm(impedance.pose()[0] - x0 - [0.005, 0, 0]) < 2e-4   # follows a 5 mm step


@pytest.mark.parametrize("push", [[3.0, 0, 0], [0, -3.0, 0], [0, 0, 5.0]])
def test_pushed_control_point_gives_way_by_force_over_stiffness(setup, push):
    sim, impedance, ft = setup
    x0, R0 = impedance.pose()
    force = run(sim, impedance, ft, 1.5, x0, R0, push=np.array(push))
    deflection = impedance.pose()[0] - x0
    expected = np.array(push) / np.array(ImpedanceConfig().stiffness)  # 3 N / 300 N/m = 10 mm, 5 N / 1000 = 5 mm
    np.testing.assert_allclose(deflection, expected, atol=0.0005)       # within 0.5 mm
    np.testing.assert_allclose(force, push, atol=0.05)                  # and the wrist sensor reads the push

# --- 5b: compliant insertion with spiral search ----------------------------------------------------

import math
from dataclasses import replace

from grasp.pick import carry_to_port, pick
from insertion.compliant import InsertConfig, insert, spiral_offset
from perception.estimate import perceive
from sim.camera import OverheadCamera
from sim.config import PEGS
from sim.layout import sample_layout


def test_spiral_grows_one_pitch_per_turn():
    cfg = InsertConfig()
    for turns in (1, 2, 5):
        # Time to complete `turns` turns, from θ = √(2vt / b): t = b·θ² / 2v with θ = 2π·turns.
        b = cfg.spiral_pitch / (2 * math.pi)
        t = b * (2 * math.pi * turns) ** 2 / (2 * cfg.spiral_speed)
        np.testing.assert_allclose(spiral_offset(t, cfg), [turns * cfg.spiral_pitch, 0, 0], atol=1e-9)


def held_over_port(seed, kind, port_error=0.0, friction=0.3, clearance=0.001):
    """Perceive, pick `kind`, carry it over its port. `port_error` [m] shifts the *perceived* port sideways
    (in a seeded direction), as a calibration error would: the arm then hovers off-centre by that much."""
    base = replace(SceneConfig(), friction=friction, clearance=clearance)
    cfg = replace(sample_layout(base, seed), depth_noise=0.002)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    with OverheadCamera(sim.model, cfg) as camera:
        scene = perceive(camera.capture(sim.data, np.random.default_rng(seed)))
    angle = np.random.default_rng([seed, 12]).uniform(0, 2 * math.pi)
    scene.ports[kind].entrance = scene.ports[kind].entrance + port_error * np.array([math.cos(angle), math.sin(angle), 0])
    arm = Arm(sim)
    held = carry_to_port(arm, scene, pick(arm, scene, kind))
    return sim, arm, scene, held


@pytest.mark.parametrize("kind", PEGS)
@pytest.mark.parametrize("port_error, searches", [(0.0, False), (0.0015, False), (0.0045, True)])
def test_peg_is_inserted_despite_port_error(kind, port_error, searches):
    sim, arm, scene, held = held_over_port(0, kind, port_error)
    result = insert(arm, scene, held.in_hand)
    assert result.ok and sim.seated(kind), result.stage                  # ground truth: on the floor
    assert sim.in_wrong_port(kind) is None and sim.attached() is None     # right port, released
    assert result.searched == searches        # inside the chamfer's ~2.5 mm reach: no search needed
    assert result.sustained_force < 15 and result.duration < 60


def test_search_copes_with_high_friction():
    sim, arm, scene, held = held_over_port(1, "square", 0.0045, friction=0.8)
    result = insert(arm, scene, held.in_hand)
    assert result.ok and sim.seated("square") and result.searched
    assert result.sustained_force < 15


def test_search_gives_up_beyond_its_radius_and_keeps_hold():
    sim, arm, scene, held = held_over_port(0, "round", 0.010)              # 10 mm off: past the 6 mm spiral
    result = insert(arm, scene, held.in_hand)
    assert not result.ok and result.stage == "spiral"
    assert result.search_radius > InsertConfig().spiral_max_radius
    assert not sim.seated("round") and sim.attached() == "round"         # didn't let go of it


# --- 5c: position-controlled baseline --------------------------------------------------------------

from insertion.baseline import insert_stiff


@pytest.mark.parametrize("kind", PEGS)
def test_stiff_baseline_inserts_when_perception_is_right(kind):
    sim, arm, scene, held = held_over_port(0, kind)
    result = insert_stiff(arm, scene, held.in_hand)
    assert result.ok and sim.seated(kind) and result.sustained_force < 5


def test_stiff_baseline_jams_where_compliance_succeeds():
    """2 mm off: inside the chamfer's reach, but outside the 0.5 mm gap. The stiff arm can't be pushed over,
    so the chamfer just loads it up to the force cap; the compliant arm is pushed over with a few newtons."""
    sim, arm, scene, held = held_over_port(0, "square", 0.002)
    stiff = insert_stiff(arm, scene, held.in_hand)
    assert not stiff.ok and stiff.stage == "force" and not sim.seated("square")

    sim, arm, scene, held = held_over_port(0, "square", 0.002)
    compliant = insert(arm, scene, held.in_hand)
    assert compliant.ok and sim.seated("square") and not compliant.searched
    assert compliant.sustained_force < 10

# --- 5d: the insertion-module grid -----------------------------------------------------------------

from harness.insertion_check import insertion_grid, summarise


def test_insertion_grid_runs_end_to_end():
    results = insertion_grid(seeds=[0], port_errors=[0.002], clearances=[0.001], frictions=[0.3])
    assert len(results) == 2 * 2                                       # 2 controllers × 2 pegs
    by_controller = results.groupby("controller").success.mean()
    assert by_controller["compliant"] == 1.0 and by_controller["stiff"] == 0.0   # the headline contrast
    table = summarise(results)
    assert list(table.index.names) == ["controller", "clearance_mm", "friction", "error_mm"]
    assert table.loc[("stiff", 1.0, 0.3, 2.0), "force_max"] > 29       # stiff: stopped by the force cap
