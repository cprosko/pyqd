"""
dotsystem:
Module containing the main DotSystem class which manages simulations of simple
charge-basis multi-dot simulations, neglecting spin and other degeneracies.
"""

import numpy as np


class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(
        self,
        is_floating=False,
        max_charge=None,
        floating_charge=None,
    ):
        self._dots = dict()
        self._couplings = dict()
        self._state_map = np.array([])
        self._inverse_state_map = dict()
        self._is_floating = is_floating
        self._floating_charge = floating_charge
        self._max_charge = max_charge


def main():
    print("Not ready for an example yet.")


if __name__ == "__main__":
    main()
