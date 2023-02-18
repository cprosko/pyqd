"""Classes containing information about quantum dots and other islands."""


import numpy as np

from utilities import ensure_iterable


class QuantumDot:
    """Class containing properties and couplings for a single quantum dot."""

    def __init__(
        self,
        charging_energy,
        level_spacing,
        name=None,
        degeneracy=1,
        _dot_type="dot",
    ):
        self.name = name
        self.charging_energy = charging_energy
        self.level_spacing = level_spacing
        self.degeneracy = degeneracy
        self.dot_type = _dot_type

    def coulomb_energies(self, states, voltages):
        """Coulomb energies for given states and voltages.

        Parameters
        ----------
        states : int or numpy.ndarray[int]
        voltages : float or numpy.ndarray[float]
            Reduced gate voltages to calculate energies at.

        Returns
        -------
        energies : float or numpy.ndarray[float]
        """
        states, voltages = ensure_iterable(states, voltages)
        single_voltage = voltages.size == 1
        energies = self.charging_energy * np.subtract.outer(states, voltages) ** 2
        if single_voltage:
            energies = energies.flatten()
        return energies

    def mode_energies(self, states):
        """Ground state mode/orbital energies for given states and the dot's degeneracy.

        Parameters
        ----------
        states : int or numpy.ndarray[int]

        Returns
        -------
        energies : float or numpy.ndarray[float]
        """
        states = ensure_iterable(states)
        num_filled_modes, extra_electrons = np.divmod(states, self.degeneracy)
        energies = self.level_spacing * (
            self.degeneracy * num_filled_modes * (num_filled_modes + 1) / 2
            + (num_filled_modes + 1) * extra_electrons
        )
        return energies

    def ground_state_energies(self, states, voltages):
        """Ground-state energies of sequence of charge states of the QuantumDot.

        Parameters
        ----------
        states : int or numpy.ndarray[int]
        voltages : float or numpy.ndarray[float]
            Reduced gate voltages to calculate energies at.

        Returns
        -------
        energies : float or numpy.ndarray[float]
        """
        # TODO: Finish writing this function
        pass


class SuperconductingIsland(QuantumDot):
    """Class containing properties of a superconducting charge island."""

    def __init__(self, charging_energy, gap_size, name=None, degeneracy=1):
        super().__init__(
            charging_energy,
            gap_size,
            name=name,
            degeneracy=degeneracy,
            _dot_type="superconducting",
        )


class QuasiLead(QuantumDot):
    """Class for naively modeling leads as 0-charging-energy dots."""

    def __init__(self, name=None, gap_size=0):
        super().__init__(0, gap_size, name=name, degeneracy=1, _dot_type="lead")
