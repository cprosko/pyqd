from itertools import permutations
from typing import Iterable
from ..islands import ChargeIsland
from ..utils import partitions


class ChargeSystemBase:
    def __init__(
        self,
        islands: Iterable[ChargeIsland] or ChargeIsland,
        floating: bool,
        total_charge: int = None,
    ):
        if isinstance(islands, ChargeIsland):
            self.islands = (islands,)
        else:
            self.islands = tuple(islands)
        self.__total_charge = total_charge
        self.__charge_states = None
        # TODO: implement auto-updating charge states
        self.__charge_states_updated = False
        self.floating = floating

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

    def charge_states(self, total_charge=None, num_dots=None):
        if total_charge is None:
            total_charge = self.total_charge
        if num_dots is None:
            num_dots = self.num_dots
        summands = partitions(total_charge, 1)
        trunc_summands = tuple(s for s in summands if len(s) <= num_dots)
        padded_summands = [list(s) + [0] * (num_dots - len(s)) for s in trunc_summands]
        states = [p for c in padded_summands for p in permutations(c)]
        states = list(set(states))
        return states

    # TODO: Finish this class
