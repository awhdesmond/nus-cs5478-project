import numpy as np
import pytest

from insertion.config import JointPDConfig, SceneConfig, SensorConfig
from insertion.control.jacobian_step import JacobianIK
from insertion.control.joint_pd import joint_pd_torque
from insertion.scene import hole_width, build_model
from insertion.sensors import FTSensor
from insertion.sim import Sim

CFG = SceneConfig()


@pytest.fixture(scope="module")
def pressed():
    """Hold still and tare, then press the peg onto the socket top next to the hole."""
    sim = Sim(build_model(CFG))
    sim.reset()
    pd = JointPDConfig()
    ft = FTSensor(sim, SensorConfig())
    q = sim.q()
    samples = []
    for _ in range(50):
        sim.step(joint_pd_torque(sim, q, pd))
        samples.append(sim.ft_raw())
    ft.tare(samples)
    rest_wrench = None
    for _ in range(200):
        sim.step(joint_pd_torque(sim, q, pd))
        rest_wrench = ft.update().copy()

    ik = JacobianIK(sim.model)
    _, rot = ik.fk(q)
    top = np.array(CFG.socket_pos) + [hole_width(CFG) / 2 + CFG.socket_chamfer + CFG.peg_radius + 0.002, 0, CFG.socket_depth]
    q_des, _ = ik.solve(q, top + [0, 0, 0.03], rot)
    for z in np.linspace(0.03, -0.010, 3000):
        q_des, _ = ik.solve(q_des, top + [0, 0, z], rot, iters=3)
        sim.step(joint_pd_torque(sim, q_des, pd))
        ft.update()
    for _ in range(1000):
        sim.step(joint_pd_torque(sim, q_des, pd))
        ft.update()
    return sim, ft, rest_wrench


def test_tared_reading_is_near_zero_at_rest(pressed):
    _, _, rest_wrench = pressed
    assert np.abs(rest_wrench).max() < 1e-3


def test_contact_force_pushes_peg_up_when_pressing(pressed):
    sim, _, _ = pressed
    contacts = sim.peg_contacts()
    assert contacts.count > 0
    assert contacts.force[2] > 5.0


def test_ft_external_force_matches_ground_truth_contact(pressed):
    sim, ft, _ = pressed
    np.testing.assert_allclose(ft.force_world(), sim.peg_contacts().force, atol=0.2)
