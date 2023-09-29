from typing import Optional
from numpy import array


class Island:
    """Base class for a charge island or 'quasilead'."""

    def __init__(
        self,
        ec: float,
        is_sc: bool = False,
        gap: Optional[float] = None,
        name: Optional[str] = None,
    ):
        self.name = name
        self.ec = ec
        self.is_sc = is_sc
        if is_sc:
            if gap is None:
                raise Exception("'gap' must be specified when is_sc is True!")
        self.gap = gap if is_sc else 0

    @property
    def charging_energy(self):
        return self.ec

    @property
    def is_superconducting(self):
        return self.is_superconducting

    @property
    def prop_dict(self):
        return {"name": self.name, "ec": self.ec, "is_sc": self.is_sc, "gap": self.gap}
