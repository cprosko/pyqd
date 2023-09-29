import numpy as np

from itertools import permutations, product
from typing import Iterable
from ..islands import Island
from ..utils import partitions


class ChargeSystem:
    def __init__(
        self,
        islands: Iterable[Island] or Island,
        floating: bool,
        total_charge: int = None,
    ):
        if isinstance(islands, Island):
            self.islands = (islands,)
        else:
            self.islands = tuple(islands)
        self.__total_charge = total_charge
        self.__charge_states = None
        # TODO: implement auto-updating charge states
        self.__charge_states_updated = False
        self.floating = floating
        self._props_dict = {
            k: np.array([isl.prop_dict[k] for isl in self.islands])
            for k in self.islands[0].prop_dict
        }

    @property
    def total_charge(self):
        return self.__total_charge

    @total_charge.setter
    def total_charge(self, val):
        if val != self.__total_charge:
            self.__charge_states_updated = False
            self.__total_charge = val

    @property
    def num_dots(self):
        return len(self.islands)

    @property
    def ecs(self):
        return self._props_dict["ec"]

    @property
    def gaps(self):
        return self._props_dict["gap"]

    # TODO: Add lazy evaluation of arrays of dot properties using IslandArray class

    def charge_states(self, total_charge=None, num_dots=None, floating=None):
        if total_charge is None:
            total_charge = self.total_charge
        if num_dots is None:
            num_dots = self.num_dots
        if floating is None:
            floating = self.floating
        if floating:
            summands = partitions(total_charge, 1)
            trunc_summands = tuple(s for s in summands if len(s) <= num_dots)
            padded_summands = [
                list(s) + [0] * (num_dots - len(s)) for s in trunc_summands
            ]
            states = [p for c in padded_summands for p in permutations(c)]
            states = np.array(list(set(states)))
        else:
            states = np.array(list(product(range(total_charge + 1), repeat=num_dots)))
        self.__charge_states = states
        return states

    def coulomb_energies(self, ngs, states=None):
        if len(ngs) != self.num_dots:
            raise Exception(
                f"len(ngs)={len(ngs)}."
                "A reduced gate charge for all dots in the system must be specified!"
            )
        states = self.charge_states() if states is None else states
        energies = np.zeros((len(states)))
        # Nested loop requiring iterations equal to num_dots * (charge space dimension)
        for i, charge_state in enumerate(states):
            coulomb_en = np.sum(self.ecs * (charge_state - ngs) * (charge_state - ngs))
            super_en = np.sum(self.gaps * np.mod(charge_state, 2))
            energies[i] = coulomb_en + super_en
        return energies

    def solve_system(self, ngs):
        # TODO: Note in docstring this can/should be overwritten
        states = self.charge_states()
        energies = self.coulomb_energies(ngs, states)
        sorted_inds = np.argsort(energies)
        return states[sorted_inds], energies[sorted_inds]
