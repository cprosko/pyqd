"""
dotsystem:
Module containing the main DotSystem class which manages simulations of simple
charge-basis multi-dot simulations, neglecting spin and other degeneracies.
"""

import numpy as np
from quantumdot import QuantumDot, SuperconductingIsland, QuasiLead
from utilities import binom
from itertools import product

_CHARGE_PARAM_NAMES = {"is_floating", "max_charge", "floating_charge", "charge_range"}

# TODO: Allow for this to project onto only nearby charge states
class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(
        self,
        is_floating=False,
        max_charge=None,
        floating_charge=None,
        charge_range=None,
    ):
        self._dots = []
        self._couplings = dict()
        self._state_map = np.array([])
        self._inverse_state_map = dict()
        self._is_floating = is_floating
        self._floating_charge = floating_charge
        self._max_charge = max_charge
        self._charge_range = charge_range
        self._Ecs = None
        self._Ems = None
        self._tcs_1e = None
        self._tcs_2e = None
        self._coupling_operator = None

    @property
    def couplings(self):
        return self._couplings

    @property
    def dots(self):
        return self._dots

    @property
    def charge_range(self):
        return self._charge_range

    @property
    def num_dots(self):
        return len(self._dots)

    @property
    def is_floating(self):
        return self._is_floating

    @is_floating.setter
    def is_floating(self, val):
        self._is_floating = val

    @property
    def num_states(self):
        return self._state_map.shape[0]

    def num_dots_type(self, dot_type):
        return len([d for d in self._dots if d.dot_type == dot_type])

    def set_charge_params(self, **kwargs):
        if any([k not in _CHARGE_PARAM_NAMES for k in kwargs.keys()]):
            raise Exception(
                "Unrecognized charge parameter input. "
                + "Recognized keys include: {keys}".format(_CHARGE_PARAM_NAMES)
            )
        if "max_charge" in kwargs.keys():
            # If is floating, max charge times number of dots must be more than floating charge
            self._max_charge = kwargs["max_charge"]
        if "floating_charge" in kwargs.keys():
            self._floating_charge = kwargs["floating_charge"]
        if "is_floating" in kwargs.keys():
            if self._floating_charge is None or self._floating_charge <= 0:
                raise Exception(
                    "floating_charge must be specified if system is made floating!"
                )
            self._is_floating = kwargs["is_floating"]
        if "charge_range" in kwargs.keys():
            self._charge_range = kwargs["charge_range"]
        self._refresh_state_map()

    def set_couplings(
        self, dot_name, *coupled_dot_names, amplitude=0, amplitude2e=0, Em=0
    ):
        for cdn in coupled_dot_names:
            self._couplings[(dot_name, cdn)] = {
                "1e": amplitude,
                "2e": amplitude2e,
                "Em": Em,
            }
            self._couplings[(cdn, dot_name)] = {
                "1e": np.conjugate(amplitude),
                "2e": np.conjugate(amplitude2e),
                "Em": Em,
            }
        print(self._couplings)
        self._update_coupling_matrices()

    def initialize_coupling(self, dotname):
        existing_dot_names = [d.name for d in self._dots if d.name != dotname]
        print("edn: " + str(existing_dot_names))
        self.set_couplings(dotname, *existing_dot_names)

    def remove_coupling(self, dot1name, dot2name):
        self.set_coupling(dot1name, dot2name, 0)

    def remove_dot(self, dotname):
        for dot in self._dots:
            if dot.name == dotname:
                del dot
            else:
                self.remove_coupling(dotname, dot.name)

    def attach_dot(self, dot):
        if dot.name is None:
            dot.name = dot.dot_type + str(self.num_dots_type(dot.dot_type) + 1)
        elif dot.name in {d.name for d in self._dots}:
            raise Exception(
                "Dot with name {n} already present in DotSystem!".format(dot.name)
            )
        self._dots.append(dot)
        self.initialize_coupling(dot.name)
        self._update_dot_prop_arrays()
        self._refresh_state_map()

    def add_dot(self, *args, **kwargs):
        if len(args) == 1:
            dot = args[0]
            if not isinstance(dot, QuantumDot):
                raise Exception(("Invalid object type passed to DotSystem.add_dot!"))
        else:
            dot = QuantumDot(*args, **kwargs)
        self.attach_dot(dot)

    def add_lead(self, name=None):
        self.add_dot(QuasiLead(name=name))

    def add_island(self, *args, name=None):
        self.add_dot(SuperconductingIsland(*args, name=name))

    @staticmethod
    def calculate_num_states(
        max_charge,
        num_dots,
        is_floating=False,
        floating_charge=None,
    ):
        """Calculate dimension of Hilbert space for given charge boundaries.

        Assumes each island/dot is allowed to have a minimum charge of 0 electrons.
        No orbital or spin degeneracies are considered.

        Parameters:
        -----------
        max_charge (int): Maximum charge per dot.
        num_dots (int): Number of dots/islands in the system. Must be specified if
            max_charge is given as an integer (i.e. is the same for all dots).

        Keyword Arguments:
        ------------------
        is_floating (bool): Whether or not total charge is fixed.
        floating_charge (int): Total number of charges distributed across the dots if
            they are floating. Should only be specified if is_floating is True.

        Returns:
        --------
        int: Dimension of the charge-state Hilbert space.
        """
        if is_floating:
            # Formula from Lemma 1.1 of doi:10.2298/AADM0802222R (Joel Ratsaby) for
            # number of ordered partitions of the integer floating_charge into num_dots
            # partitions of maximum size max_charge.
            return sum(
                [
                    (-1) ** (i // (max_charge + 1))
                    * binom(num_dots, i / (max_charge + 1))
                    * binom(floating_charge - i + num_dots - 1, floating_charge - i)
                    for i in np.arange(0, floating_charge + 1, max_charge + 1)
                ]
            )
        return np.sum(
            [
                binom(total_charge - 1, num_dots - 1)
                for total_charge in range(max_charge)
            ]
        )

    def _refresh_state_map(self):
        # First update state mapping indices to charge states
        single_dot_charge_states = [list(range(self._max_charge + 1))] * self.num_dots
        unfixed_charge_states = np.array(list(product(*single_dot_charge_states)))
        if not self.is_floating:
            self._state_map = unfixed_charge_states
        else:
            fixed_charge_states = unfixed_charge_states[
                np.sum(unfixed_charge_states, axis=1) == self._floating_charge
            ]
            self._state_map = fixed_charge_states
        # Next, update inverse state map accordingly (mapping charge states to indices)
        num_states = self._state_map.shape[0]
        self._inverse_state_map = {
            tuple(self._state_map[i]): i for i in range(num_states)
        }

    def onsite_energy(self, gates, as_matrix=False):
        states = self._state_map
        if len(gates) != states[0]:
            raise Exception(
                (
                    "Same number of reduced gate voltages must be provided "
                    "as there are dots!"
                )
            )
        reduced_charges = states - gates[np.newaxis, ...]
        coulomb_energy = reduced_charges ** 2 @ self._Ecs
        level_spacings = (states % 2) @ self._level_spacings
        mutual_couplings = reduced_charges @ self._Ems @ reduced_charges.T
        if as_matrix:
            return np.diag(coulomb_energy + level_spacings + mutual_couplings)
        return coulomb_energy + level_spacings + mutual_couplings

    def _update_coupling_operator(self):
        # TODO: Finish this method
        coupling_operator = np.zeros((self.num_states, self.num_states))
        # 3D matrix of all possible charge differences between states
        states = self._state_map
        state_diffs = (
            np.repeat(states, self.num_states, axis=0).reshape(
                self.num_states, self.num_states, self.num_dots
            )
            - states
        )
        # TODO: Find a way to vectorize this algorithm
        # Compute upper diagonal part of hopping matrix first:
        for i in range(self.num_states):
            for j in np.arange(i + 1, self.num_states, +1):
                if (
                    # Charge isn't conserved between states:
                    sum(state_diffs[i, j] != 0)
                    # Hopping has occurred between more than two dots/islands/leads:
                    or state_diffs[i, j, state_diffs[i, j] != 0].shape[0] != 2
                ):
                    continue
                if sum(np.abs(state_diffs[i, j])) == 2:
                    # One electron has tunneled between these states.
                    coupling_operator[i, j] = self._tcs_1e[
                        state_diffs[i, j] == -1, state_diffs[i, j] == +1
                    ]
                if sum(np.abs(state_diffs[i, j])) == 4:
                    # Two electrons have tunneled across two dots between these states.
                    coupling_operator[i, j] = self._tcs_2e[
                        state_diffs[i, j] == -2, state_diffs[i, j] == +2
                    ]
        self._coupling_operator = coupling_operator
        return coupling_operator

    def _update_coupling_matrices(self):
        num_dots = self.num_dots
        Ems = np.zeros((num_dots, num_dots))
        tcs_1e = np.zeros((num_dots, num_dots))
        tcs_2e = np.zeros((num_dots, num_dots))
        # TODO: Find way to do this without double for loop
        for i, dot1 in enumerate(self._dots):
            for j, dot2 in enumerate(self._dots):
                if dot1.name == dot2.name:
                    continue
                coupling = self._couplings[(dot1.name, dot2.name)]
                Ems[i, j] = coupling["Em"]
                tcs_1e[i, j] = coupling["1e"]
                tcs_2e[i, j] = coupling["2e"]
        self._Ems = Ems
        self._tcs_1e = tcs_1e
        self._tcs_2e = tcs_2e

    def _update_dot_prop_arrays(self):
        self._Ecs = np.array([d.charging_energy for d in self._dots])
        self._level_spacings = np.array([d.level_spacing for d in self._dots])


def main():
    dot_system = DotSystem(is_floating=True, max_charge=0, floating_charge=0)
    dot_system.add_dot(150, 10, name="cooldot")
    dot_system.add_dot(300, 50, name="lamedot")
    dot_system.add_island(100, 50, name="thisisanisland")
    dot_system.add_lead(name="alead")
    print(dot_system._state_map)
    print(dot_system.num_dots)
    print(dot_system.num_states)


if __name__ == "__main__":
    main()
