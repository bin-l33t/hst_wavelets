#!/usr/bin/env python3
"""
2D Ising Model Simulation

Implements the classical 2D Ising model on a square lattice with
periodic boundary conditions using the Metropolis algorithm.

This provides a ground-truth physical system for validating HST:
- Known phase transition at critical temperature Tc
- Scale invariance at Tc (power-law correlations)
- Real-valued field (tests H+/H- symmetry)

Critical temperature: Tc = 2J / (k_B * ln(1 + sqrt(2))) ≈ 2.269 (for J=k_B=1)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# Critical temperature for 2D square Ising (J=1, k_B=1)
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269


@dataclass
class IsingSnapshot:
    """A snapshot of the Ising system."""
    spins: np.ndarray          # Spin configuration (+1 or -1)
    temperature: float         # Temperature at which snapshot was taken
    magnetization: float       # Average magnetization
    energy: float             # Total energy
    L: int                    # Lattice size
    
    @property
    def is_ordered(self) -> bool:
        """Below Tc: ordered/magnetized."""
        return self.temperature < TC_EXACT * 0.9
    
    @property
    def is_critical(self) -> bool:
        """Near Tc: critical/scale-invariant."""
        return abs(self.temperature - TC_EXACT) < 0.2
    
    @property
    def is_disordered(self) -> bool:
        """Above Tc: disordered/paramagnetic."""
        return self.temperature > TC_EXACT * 1.1


class IsingModel:
    """
    2D Ising Model simulator.
    
    H = -J Σ_{<i,j>} s_i s_j
    
    Parameters
    ----------
    L : int
        Lattice size (L x L)
    J : float
        Coupling constant (default 1.0)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, L: int = 64, J: float = 1.0, seed: Optional[int] = None):
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)
        
        # Initialize random spins
        self.spins = self.rng.choice([-1, 1], size=(L, L))
        
        # Precompute neighbor indices (periodic boundary)
        self._setup_neighbors()
    
    def _setup_neighbors(self):
        """Setup neighbor lookup for periodic boundary conditions."""
        L = self.L
        # For each site (i, j), neighbors are (i±1, j) and (i, j±1) mod L
        self.neighbors = {}
        for i in range(L):
            for j in range(L):
                self.neighbors[(i, j)] = [
                    ((i + 1) % L, j),
                    ((i - 1) % L, j),
                    (i, (j + 1) % L),
                    (i, (j - 1) % L),
                ]
    
    def energy(self) -> float:
        """Compute total energy H = -J Σ s_i s_j."""
        E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                s = self.spins[i, j]
                # Only count right and down neighbors to avoid double counting
                E -= self.J * s * self.spins[(i + 1) % L, j]
                E -= self.J * s * self.spins[i, (j + 1) % L]
        return E
    
    def energy_fast(self) -> float:
        """Compute energy using vectorized operations."""
        s = self.spins
        # Sum over nearest neighbor pairs (avoiding double count)
        E = -self.J * np.sum(s * np.roll(s, 1, axis=0))  # vertical
        E -= self.J * np.sum(s * np.roll(s, 1, axis=1))  # horizontal
        return E
    
    def magnetization(self) -> float:
        """Compute magnetization per spin."""
        return np.mean(self.spins)
    
    def local_energy(self, i: int, j: int) -> float:
        """Energy contribution from spin at (i, j)."""
        s = self.spins[i, j]
        neighbor_sum = sum(self.spins[ni, nj] for ni, nj in self.neighbors[(i, j)])
        return -self.J * s * neighbor_sum
    
    def metropolis_step(self, T: float) -> bool:
        """
        Single Metropolis update step.
        
        Parameters
        ----------
        T : float
            Temperature
            
        Returns
        -------
        accepted : bool
            Whether the flip was accepted
        """
        L = self.L
        
        # Random site
        i = self.rng.integers(0, L)
        j = self.rng.integers(0, L)
        
        # Energy change if we flip
        s = self.spins[i, j]
        neighbor_sum = (
            self.spins[(i + 1) % L, j] +
            self.spins[(i - 1) % L, j] +
            self.spins[i, (j + 1) % L] +
            self.spins[i, (j - 1) % L]
        )
        dE = 2 * self.J * s * neighbor_sum
        
        # Metropolis criterion
        if dE <= 0 or self.rng.random() < np.exp(-dE / T):
            self.spins[i, j] = -s
            return True
        return False
    
    def sweep(self, T: float) -> int:
        """
        One Monte Carlo sweep (L² attempted flips).
        
        Returns number of accepted flips.
        """
        accepted = 0
        for _ in range(self.L * self.L):
            if self.metropolis_step(T):
                accepted += 1
        return accepted
    
    def wolff_step(self, T: float) -> int:
        """
        Single Wolff cluster update.
        
        Much more efficient than Metropolis near Tc (no critical slowing down).
        
        Returns size of flipped cluster.
        """
        L = self.L
        p_add = 1 - np.exp(-2 * self.J / T)  # Probability to add aligned neighbor
        
        # Pick random seed site
        i0, j0 = self.rng.integers(0, L), self.rng.integers(0, L)
        seed_spin = self.spins[i0, j0]
        
        # Grow cluster using BFS
        cluster = set()
        stack = [(i0, j0)]
        
        while stack:
            i, j = stack.pop()
            if (i, j) in cluster:
                continue
            if self.spins[i, j] != seed_spin:
                continue
            
            cluster.add((i, j))
            
            # Try to add neighbors
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                ni, nj = ni % L, nj % L
                if (ni, nj) not in cluster:
                    if self.spins[ni, nj] == seed_spin:
                        if self.rng.random() < p_add:
                            stack.append((ni, nj))
        
        # Flip entire cluster
        for i, j in cluster:
            self.spins[i, j] = -self.spins[i, j]
        
        return len(cluster)
    
    def wolff_sweep(self, T: float, n_clusters: int = 1) -> int:
        """
        Perform n_clusters Wolff updates.
        
        Returns total number of flipped spins.
        """
        total_flipped = 0
        for _ in range(n_clusters):
            total_flipped += self.wolff_step(T)
        return total_flipped
    
    def equilibrate(self, T: float, n_sweeps: int = 1000, verbose: bool = False,
                    use_wolff: bool = False):
        """
        Equilibrate the system at temperature T.
        
        Parameters
        ----------
        T : float
            Target temperature
        n_sweeps : int
            Number of Monte Carlo sweeps (or Wolff clusters)
        verbose : bool
            Print progress
        use_wolff : bool
            Use Wolff cluster updates (much faster near Tc)
        """
        for sweep_idx in range(n_sweeps):
            if use_wolff:
                self.wolff_step(T)
            else:
                self.sweep(T)
            if verbose and sweep_idx % 100 == 0:
                m = abs(self.magnetization())
                print(f"  Sweep {sweep_idx}: |m| = {m:.4f}")
    
    def snapshot(self, T: float) -> IsingSnapshot:
        """Get current state as IsingSnapshot."""
        return IsingSnapshot(
            spins=self.spins.copy(),
            temperature=T,
            magnetization=self.magnetization(),
            energy=self.energy_fast(),
            L=self.L,
        )
    
    def generate_snapshot(
        self,
        T: float,
        equilibration_sweeps: int = 2000,
        verbose: bool = False,
    ) -> IsingSnapshot:
        """
        Generate an equilibrated snapshot at temperature T.
        
        Parameters
        ----------
        T : float
            Temperature
        equilibration_sweeps : int
            Sweeps to equilibrate
        verbose : bool
            Print progress
            
        Returns
        -------
        snapshot : IsingSnapshot
        """
        # Hot start for T > Tc, cold start for T < Tc
        if T > TC_EXACT:
            self.spins = self.rng.choice([-1, 1], size=(self.L, self.L))
        else:
            self.spins = np.ones((self.L, self.L), dtype=int)
        
        self.equilibrate(T, equilibration_sweeps, verbose)
        return self.snapshot(T)


def generate_ising_dataset(
    L: int = 64,
    temperatures: Optional[List[float]] = None,
    n_samples_per_T: int = 10,
    equilibration_sweeps: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> List[IsingSnapshot]:
    """
    Generate dataset of Ising snapshots at various temperatures.
    
    Parameters
    ----------
    L : int
        Lattice size
    temperatures : list, optional
        Temperatures to sample. Default: range around Tc
    n_samples_per_T : int
        Snapshots per temperature
    equilibration_sweeps : int
        Sweeps to equilibrate
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    snapshots : list of IsingSnapshot
    """
    if temperatures is None:
        # Default: ordered, critical, disordered
        temperatures = [
            1.5,           # Well below Tc (ordered)
            2.0,           # Below Tc
            TC_EXACT,      # Critical
            2.5,           # Above Tc
            3.5,           # Well above Tc (disordered)
        ]
    
    model = IsingModel(L=L, seed=seed)
    snapshots = []
    
    for T in temperatures:
        if verbose:
            phase = "ordered" if T < TC_EXACT * 0.9 else ("critical" if abs(T - TC_EXACT) < 0.2 else "disordered")
            print(f"Generating at T = {T:.3f} ({phase})...")
        
        for sample_idx in range(n_samples_per_T):
            snapshot = model.generate_snapshot(T, equilibration_sweeps, verbose=False)
            snapshots.append(snapshot)
            
            if verbose:
                print(f"  Sample {sample_idx + 1}: |m| = {abs(snapshot.magnetization):.4f}")
    
    return snapshots


def spin_to_signal(spins: np.ndarray, flatten: bool = True) -> np.ndarray:
    """
    Convert 2D spin configuration to 1D signal for HST.
    
    Parameters
    ----------
    spins : ndarray
        2D spin array (L x L)
    flatten : bool
        If True, return flattened 1D array
        If False, return as complex signal (real part only)
        
    Returns
    -------
    signal : ndarray
        1D signal of length L² (if flatten) or L (if not)
    """
    if flatten:
        return spins.flatten().astype(float)
    else:
        # Row-by-row as complex signal (imaginary = 0)
        L = spins.shape[0]
        signal = np.zeros(L * L, dtype=complex)
        signal.real = spins.flatten()
        return signal


def spin_to_row_signals(spins: np.ndarray) -> np.ndarray:
    """
    Convert 2D spins to array of 1D row signals.
    
    Returns
    -------
    signals : ndarray of shape (L, L)
        Each row is a 1D signal
    """
    return spins.astype(float)


if __name__ == "__main__":
    print("=" * 60)
    print("2D ISING MODEL - DEMONSTRATION")
    print("=" * 60)
    print(f"Critical temperature Tc = {TC_EXACT:.4f}")
    print()
    
    # Generate snapshots
    L = 32
    model = IsingModel(L=L, seed=42)
    
    for T, name in [(1.5, "ORDERED"), (TC_EXACT, "CRITICAL"), (3.5, "DISORDERED")]:
        print(f"\n{name} phase (T = {T:.3f}):")
        print("-" * 40)
        
        snapshot = model.generate_snapshot(T, equilibration_sweeps=1000)
        
        print(f"  |m| = {abs(snapshot.magnetization):.4f}")
        print(f"  E/N = {snapshot.energy / (L * L):.4f}")
        print(f"  is_ordered: {snapshot.is_ordered}")
        print(f"  is_critical: {snapshot.is_critical}")
        print(f"  is_disordered: {snapshot.is_disordered}")
        
        # Visual sample (8x8 corner)
        print("\n  Spin configuration (8x8 corner):")
        for i in range(8):
            row = "  "
            for j in range(8):
                row += "+" if snapshot.spins[i, j] == 1 else "-"
            print(row)
