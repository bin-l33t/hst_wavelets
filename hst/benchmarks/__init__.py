"""
HST Benchmarks

Physical systems for validating the Heisenberg Scattering Transform.

Modules
-------
hamiltonian : Proper Hamiltonian systems with (q, p) dynamics
    - CoupledHarmonicChain : Linear benchmark  
    - Phi4LatticeField : Nonlinear benchmark (matches Glinsky Fig 3)
    - NonlinearSchrodinger : Natively complex field

ising : Stochastic lattice model (for phase transition analysis)
"""

from .hamiltonian import (
    CoupledHarmonicChain,
    Phi4LatticeField,
    NonlinearSchrodinger,
    HamiltonianState,
    create_hamiltonian_test_signals,
)

from .ising import (
    IsingModel,
    IsingSnapshot,
    TC_EXACT,
    generate_ising_dataset,
    spin_to_signal,
    spin_to_row_signals,
)

__all__ = [
    # Hamiltonian systems (proper dynamics)
    'CoupledHarmonicChain',
    'Phi4LatticeField',
    'NonlinearSchrodinger',
    'HamiltonianState',
    'create_hamiltonian_test_signals',
    # Ising model (stochastic)
    'IsingModel',
    'IsingSnapshot',
    'TC_EXACT',
    'generate_ising_dataset',
    'spin_to_signal',
    'spin_to_row_signals',
]
