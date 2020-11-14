"""
dotsystem:
Module for simulating quantum dot systems with or without leads,
including only degeneracy and charge degrees of freedom, no fermionic
statistics.
"""

import numpy as np

from .quantumdot import QuantumDot, SuperconductingIsland, QuasiLead
from .utilities import is_iterable
from scipy.special import binom


class DotSystem:
    """Manager class for simulations of multiple dots."""

    def __init__(self):
        self._dots = dict()
        self._couplings = dict()
        pass

    @property
    def couplings(self):
        return self._couplings

    @couplings.setter
    def couplings(self, dot1name, dot2name, amplitude):
        self._couplings[(dot1name, dot2name)] = amplitude
        self._couplings[(dot2name, dot1name)] = np.conjugate(amplitude)

    @property
    def dots(self):
        return self._dots

    @property
    def numdots(self):
        return len(self._dots.keys())

    def initialize_coupling(self, dotname, amplitude=0):
        existing_dot_names = [k for k in self._dots.keys() if k != dotname]
        for edn in existing_dot_names:
            self.couplings(dotname, edn, amplitude)

    def remove_coupling(self, dot1name, dot2name):
        self.couplings(dot1name, dot2name, 0)

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
            self.couplings(dotname, edn, amplitude)

    def add_dot(self, *args, **kwargs):
        if len(args) == 1:
            dot = args[0]
            if not isinstance(dot, QuantumDot):
                raise Exception(("Invalid object type passed to " "DotSystem.add_dot!"))
        else:
            dot = QuantumDot(*args, **kwargs)
        self.attach_dot(dot)

    @staticmethod
    def num_states(max_charge, is_floating=False, numdots=None, floating_charge=None):
        if is_iterable(max_charge):
            if numdots is not None:
                raise Exception(
                    (
                        "Number of dots is already specified by "
                        "length of max_charge iterable!"
                    )
                )
            if not is_floating:
                return np.prod(max_charge)
            if floating_charge is None:
                raise Exception(
                    (
                        "Total charge must be provided if a floating"
                        " system with charge bounds is specified!"
                    )
                )
            if floating_charge > sum(max_charge):
                raise Exception(
                    (
                        "Total floating charge cannot exceed sum of"
                        " maximum charges for each dot!"
                    )
                )
            sorted_max_charges = np.sort(max_charge)
            return 0  # TODO: Finish this
        elif numdots is None:
            raise Exception(
                (
                    "If total charge is specified as integer, numdots"
                    " must also be specified!"
                )
            )
        if is_floating:
            return binom(max_charge - 1, numdots - 1)
        return np.sum(
            [binom(total_charge - 1, numdots - 1) for total_charge in range(max_charge)]
        )


def main():
    print("Nothing to do here yet!")


if __name__ == "__main__":
    main()
