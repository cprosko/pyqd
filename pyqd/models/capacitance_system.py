import numpy as np
from .base import ChargeSystemBase


class CapacitanceSystem(ChargeSystemBase):
    """Classical system of capacitively coupled charges."""

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)

    @property
    def basis_states(self):
        return self.charge_states

    def solve_system(self):
        states = self.charge_states
        # TODO: Finish this function
