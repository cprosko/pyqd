"""
dotsystem:
Module containing the main DotSystem class which manages simulations of simple
charge-basis multi-dot simulations, neglecting spin and other degeneracies.
"""

import numpy as np
from numbers import Number

from dotsystem import QuantumDot


class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(
        self,
        dots=None,
        max_charge=None,
        floating_charge=None,
    ):
        self._dots = dict()
        if dots is not None:
            for dot in dots:
                self.add_dot(dot)
        self._couplings = dict()
        self._coupling_matrix = None
        self._onsite_hamiltonian = None
        self._state_map = None  # Becomes an array of state labels
        self._inverse_state_map = None  # Becomes a dict of state indices from labels
        self._floating_charge = floating_charge
        self._max_charge = max_charge
        self._dots_changed = False
        self._params_changed = False

    @property
    def onsite_hamiltonian(self):
        if self._dots_changed or self._params_changed:
            self._update_onsite_hamiltonian()
        return self._onsite_hamiltonian

    @property
    def coupling_matrix(self):
        if self._unimplemented_couplings:
            self._update_coupling_matrix()
        return self._coupling_matrix

    def add_dot(self, dot: QuantumDot, overwrite_dots=False) -> None:
        if dot.name in self._dots.keys() and not overwrite_dots:
            raise Exception(f"Dot named '{dot.name}' already exists in this DotSystem!")
        self._dots.update({dot.name: dot})

    def add_coupling(self, name1: str, name2: str, coupling: Number):
        coupling_is_new = {name1, name2} in self._couplings
        self._couplings[{name1, name2}:coupling]
        if coupling_is_new or (
            not coupling_is_new and self._couplings[{name1, name2}] != coupling
        ):
            self._unimplemented_couplings = True

    def remove_coupling(self, name1: str, name2: str):
        if {name1, name2} in self._couplings:
            del self._couplings[{name1, name2}]
        self._unimplemented_couplings = True

    def onsite_energy(self, states):
        """Calculate the on-site energy of basis states of the DotSystem.

        Parameters
        ----------
        states (dict[numpy.ndarray[int] or int]):
            State or sequence of states given as a dictionary
            of sequences of states for each QuantumDot in the system (including
            QuasiLeads and SuperconductingIslands) with its 'name' property as the key.
            If iterables are given for each dictionary entry, they must be the same
            length as 'states' specifies states of the entire DotSystem and not only
            individual dots.

        Returns
        -------
        numpy.ndarray[float] or complex:
            One-dimensional array of energies or an energy for each input state
            ignoring interactions.
        """
        pass

    def _update_coupling_matrix(self):
        # TODO Write this function
        self._coupling_matrix = None
        self._unimplemented_couplings = False
        pass

    def _update_onsite_hamiltonian(self):
        # TODO Write this function
        self._onsite_hamiltonian = None
        self._dots_changed = False
        self._params_changed = False
        pass


def main():
    print("Not ready for an example yet.")


if __name__ == "__main__":
    main()
