"""Classes containing information about quantum dots and other islands."""


class QuantumDot:
    """Class containing properties and couplings for a single quantum dot."""

    def __init__(
        self,
        charging_energy,
        level_spacing,
        name=None,
        spin_degenerate=False,
        _dot_type="dot",
    ):
        self.name = name
        self.charging_energy = charging_energy
        self.level_spacing = level_spacing
        self.spin_degenerate = spin_degenerate
        self.dot_type = _dot_type


class SuperconductingIsland(QuantumDot):
    """Class containing properties of a superconducting charge island."""

    def __init__(self, charging_energy, gap_size, name=None):
        super().__init__(
            name,
            charging_energy,
            gap_size,
            spin_degenerate=True,
            _dot_type="superconducting",
        )


class QuasiLead(QuantumDot):
    """Class for naively modeling leads as 0-charging-energy dots."""

    def __init__(self, name=None):
        super().__init__(0, 0, name=name, spin_degenerate=False, _dot_type="lead")
