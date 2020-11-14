"""Classes containing information about quantum dots and other islands."""

class QuantumDot:
    """Class containing properties and couplings for a single quantum dot."""
    def __init__(self, name, chargingEnergy, levelSpacing,
                 spinDegenerate=False, __dotType="dot"):
        self.name = name
        self.chargingEnergy = chargingEnergy
        self.levelSpacing = levelSpacing
        self.spinDegenerate = spinDegenerate
        self.dotType = __dotType
    pass


class SuperconductingIsland(QuantumDot):
    """Class containing properties of a superconducting charge island."""
    def __init__(self, name, chargingEnergy, gapSize):
        super().__init__(name, chargingEnergy, gapSize,
                         spinDegenerate=True, __dotType="superconducting")
    pass


class QuasiLead(QuantumDot):
    """Class for naively modeling leads as 0-charging-energy dots."""
    def __init__(self):
        super().__init__("lead", 0, 0, spinDegenerate=False, __dotType="lead")
    pass
