"""Registry of runnable controllers by name, shared by the evaluation and viewer scripts."""

from insertion.control.compliant import ImpedanceInsert, SpiralSearchInsert
from insertion.control.guarded import GuardedRetry
from insertion.control.scripted import StraightDownInsert

CONTROLLERS = {"straight": StraightDownInsert, "guarded": GuardedRetry, "impedance": ImpedanceInsert, "spiral": SpiralSearchInsert}
