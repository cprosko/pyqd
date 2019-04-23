# pyqd
A package for simulating arbitrary quantum dot systems with or without leads using the underlying DotSystem class. Allows for adding degeneracy to normal and superconducting islands, implementing parity dependent tunneling, spin effects, and non-zero temperature.

## Known issues:
Parity dependent tunneling (the u option in DotSystem.add_dot()) does not work correctly for normal
dots currently. Adding two-e tunnel couplings also does not yet function.
