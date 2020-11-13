"""
dotsystem:
Module for simulating quantum dot systems with or without leads,
including only degeneracy and charge degrees of freedom, no fermionic
statistics.
"""

import numpy as np

from .quantumdot import QuantumDot, SuperconductingIsland, QuasiLead
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
            raise Exception("Dot with name {n} already present in DotSystem!"
                            .format(dot.name))
        self._dots[dot.name] = dot
        existing_dot_names = [k for k in self._dots.keys() if k != dot.name]
        for edn in existing_dot_names:
            self.couplings(dotname, edn, amplitude)

    def add_dot(self, *args, **kwargs):
        if len(args) == 1:
            dot = args[0]
            if not isinstance(dot, QuantumDot):
                raise Exception(("Invalid object type passed to "
                                 "DotSystem.add_dot!"))
        else:
            dot = QuantumDot(*args, **kwargs)
        self.attach_dot(dot)
        
    def num_states(self, max_charge=None, charge_bounds=None,
                   is_floating=True):
        if charge_bounds is None:
            if max_charge is None:
                raise Exception(("Either charge bounds or max charge or both"
                                 "must be provided!"))
            # Total number of states is sum over the number for all possible
            # total charges.
            if not is_floating:
                return sum([binom(charge - 1, self.numdots)
                            for charge in range(max_charge)])
            # Equivalent to problem of distributing 'max_charge' balls into
            # 'self.numdots' boxes.
            return binom(max_charge - 1, self.numdots - 1)
        if (max_charge is not None
                and max_charge < sum([min(bound) for bound in charge_bounds])):
            raise Exception(("max_charge is too small to satisfy charge"
                             " bounds!"))
        if not is_floating:
            if max_charge is None:
                # All possible charges for each dot within bounds are allowed.
                return np.prod([max(bound) - min(bound) + 1
                                for bound in charge_bounds])
        return 0
    pass


def main():
    print("Nothing to do here yet!")


if __name__ == '__main__':
    main()
