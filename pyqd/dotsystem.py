"""
dotsystem:
Module for simulating quantum dot systems with or without leads,
including only degeneracy and charge degrees of freedom, no fermionic
statistics.
"""

import numpy as np

from .quantumdot import QuantumDot, SuperconductingIsland, QuasiLead
from .utilities import is_iterable, binom


class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(self):
        self._dots = dict()
        self._couplings = dict()
        pass

    @property
    def couplings(self):
        return self._couplings

    @property
    def dots(self):
        return self._dots

    @property
    def numdots(self):
        return len(self._dots.keys())

    def set_coupling(self, dot1name, dot2name, amplitude):
        self._couplings[(dot1name, dot2name)] = amplitude
        self._couplings[(dot2name, dot1name)] = np.conjugate(amplitude)

    def initialize_coupling(self, dotname, amplitude=0):
        existing_dot_names = [k for k in self._dots.keys() if k != dotname]
        for edn in existing_dot_names:
            self.set_coupling(dotname, edn, amplitude)

    def remove_coupling(self, dot1name, dot2name):
        self.set_coupling(dot1name, dot2name, 0)

    def remove_dot(self, dotname):
        del self._dots[dotname]
        for rd in self._dots.keys():
            del self._couplings[(dotname, rd)]
            del self._couplings[(rd, dotname)]

    def attach_dot(self, dot):
        if dot.name in self._dots.keys():
            raise Exception(
                "Dot with name {n} already present in DotSystem!".format(dot.name)
            )
        self._dots[dot.name] = dot
        existing_dot_names = [k for k in self._dots.keys() if k != dot.name]
        for edn in existing_dot_names:
            self.set_coupling(dot.name, edn, 0)

    def add_dot(self, *args, **kwargs):
        if len(args) == 1:
            dot = args[0]
            if not isinstance(dot, QuantumDot):
                raise Exception(("Invalid object type passed to " "DotSystem.add_dot!"))
        else:
            dot = QuantumDot(*args, **kwargs)
        self.attach_dot(dot)

    @staticmethod
    def num_states(max_charge, numdots, is_floating=False, floating_charge=None):
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
        if is_floating:
            # Formula from Lemma 1.1 of doi:10.2298/AADM0802222R (Joel Ratsaby) for
            # number of ordered partitions of the integer floating_charge into numdots
            # partitions of maximum size max_charge.
            return sum(
                [
                    (-1) ** (i / (max_charge + 1))
                    * binom(numdots, i / (max_charge + 1))
                    * binom(floating_charge - i + numdots - 1, floating_charge - i)
                    for i in np.arange(0, floating_charge + 1, max_charge + 1)
                ]
            )
        return np.sum(
            [binom(total_charge - 1, numdots - 1) for total_charge in range(max_charge)]
        )


def main():
    print("Nothing to do here yet!")


if __name__ == "__main__":
    main()
