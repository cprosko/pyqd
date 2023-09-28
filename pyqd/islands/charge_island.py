class ChargeIsland():
    """Base class for a charge island or 'quasilead'."""
    def __init__(self, name: str, ec: float, is_sc: bool=False):
        self.name = name
        self.ec = ec
        self.is_sc = is_sc

    @property
    def charging_energy(self):
        return self.ec

    @property
    def is_superconducting(self):
        return self.is_superconducting
