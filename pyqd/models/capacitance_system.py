import numpy as np
from .base import ChargeSystemBase


class CapacitanceSystem(ChargeSystemBase):
    """Classical system of capacitively coupled charges."""

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
