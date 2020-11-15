"""
dotsystem:
Module containing the main DotSystem class which manages simulations of simple
charge-basis multi-dot simulations.
"""

import numpy as np
from .quantumdot import QuantumDot, SuperconductingIsland, QuasiLead
from .utilities import binom
from itertools import product

_CHARGE_PARAM_NAMES = {"is_floating", "max_charge", "floating_charge"}


class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(self, is_floating=False, max_charge=None, floating_charge=None):
        self._dots = dict()
        self._couplings = dict()
        self._state_map = []
        self._inverse_state_map = dict()
        self._is_floating = is_floating
        self._floating_charge = floating_charge
        self._max_charge = max_charge

    @property
    def couplings(self):
        return self._couplings

    @property
    def dots(self):
        return self._dots

    @property
    def numdots(self):
        return len(self._dots.keys())

    @property
    def num_degenerate_dots(self):
        return len([v for v in self._dots.values() if v.spin_degenerate is True])

    @property
    def is_floating(self):
        return self._is_floating

    @is_floating.setter
    def is_floating(self, val):
        self._is_floating = val

    def num_dots_type(self, dot_type):
        return len([v for v in self._dots.values() if v.dot_type == dot_type])

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

    def set_coupling(self, dot1name, dot2name, amplitude=0, amplitude2e=0, Em=0):
        self._couplings[(dot1name, dot2name)] = {
            "1e": amplitude,
            "2e": amplitude2e,
            "Em": Em,
        }
        self._couplings[(dot2name, dot1name)] = {
            "1e": np.conjugate(amplitude),
            "2e": np.conjugate(amplitude2e),
            "Em": Em,
        }

    def initialize_coupling(self, dotname):
        existing_dot_names = [d.name for d in self._dots if d.name != dotname]
        for edn in existing_dot_names:
            self.set_coupling(dotname, edn)

    def remove_coupling(self, dot1name, dot2name):
        self.set_coupling(dot1name, dot2name, 0)

    def remove_dot(self, dotname):
        for dot in self._dots:
            if dot.name == dotname:
                del dot
        for rd in self._dots:
            del self._couplings[(dotname, rd.name)]
            del self._couplings[(rd.name, dotname)]

    def attach_dot(self, dot):
        if dot.name is None:
            dot.name = dot.dot_type + str(self.num_dots_type(dot.dot_type) + 1)
        elif dot.name in {d.name for d in self._dots}:
            raise Exception(
                "Dot with name {n} already present in DotSystem!".format(dot.name)
            )
        self._dots.append(dot)
        existing_dot_names = [d.name for d in self._dots if d.name != dot.name]
        for edn in existing_dot_names:
            self.set_coupling(dot.name, edn, 0)

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

    def add_island(self, *args, **kwargs):
        self.add_dot(SuperconductingIsland(*args, **kwargs))

    @staticmethod
    def num_states(
        max_charge,
        numdots,
        is_floating=False,
        floating_charge=None,
        num_degenerate_dots=0,
    ):
        """Calculate dimension of Hilbert space for given charge boundaries.

        Assumes each island/dot is allowed to have a minimum charge of 0 electrons.
        No orbital or spin degeneracies are considered.

        Parameters:
        -----------
        max_charge (int): Maximum charge per dot.
        numdots (int): Number of dots/islands in the system. Must be specified if
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
        spin_degeneracy_multiplier = 2 ** num_degenerate_dots
        if is_floating:
            # Formula from Lemma 1.1 of doi:10.2298/AADM0802222R (Joel Ratsaby) for
            # number of ordered partitions of the integer floating_charge into numdots
            # partitions of maximum size max_charge.
            return spin_degeneracy_multiplier * sum(
                [
                    (-1) ** (i // (max_charge + 1))
                    * binom(numdots, i / (max_charge + 1))
                    * binom(floating_charge - i + numdots - 1, floating_charge - i)
                    for i in np.arange(0, floating_charge + 1, max_charge + 1)
                ]
            )
        return spin_degeneracy_multiplier * np.sum(
            [binom(total_charge - 1, numdots - 1) for total_charge in range(max_charge)]
        )

    def _refresh_state_map(self):
        # First update state mapping indices to charge states
        single_dot_charge_states = [list(range(self._max_charge))] * self.numdots
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
        self._inverse_state_map = {self._state_map[i]: i for i in range(num_states)}


def main():
    print("Nothing to do here yet!")


if __name__ == "__main__":
    main()
