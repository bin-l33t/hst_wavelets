#!/usr/bin/env python3
"""
Local test script for Modal app logic.

Run this to verify the experiment code works before deploying to Modal.

Usage:
    python test_modal_local.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from hst_torch.scattering import HeisenbergScatteringTransformTorch

TC_EXACT = 2 / np.log(1 + np.sqrt(2))


class IsingModel:
    """Simple Ising model with Wolff updates."""
    
    def __init__(self, L, J_coupling=1.0, seed=None):
        self.L = L
        self.J = J_coupling
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(L, L))
    
    def wolff_step(self, T):
        L = self.L
        p_add = 1 - np.exp(-2 * self.J / T)
        i0, j0 = self.rng.integers(0, L), self.rng.integers(0, L)
        seed_spin = self.spins[i0, j0]
        cluster = set()
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            if (i, j) in cluster:
                continue
            if self.spins[i, j] != seed_spin:
                continue
            cluster.add((i, j))
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                ni, nj = ni % L, nj % L
                if (ni, nj) not in cluster:
                    if self.spins[ni, nj] == seed_spin:
                        if self.rng.random() < p_add:
                            stack.append((ni, nj))
        for i, j in cluster:
            self.spins[i, j] = -self.spins[i, j]
        return len(cluster)
    
    def magnetization(self):
        return np.mean(self.spins)


def encode_spins(row, encoding='magnitude'):
    if encoding == 'magnitude':
        return row + 3.0 + 0j
    elif encoding == 'phase':
        phase = np.pi * (row + 1) / 4
        return 3.0 * np.exp(1j * phase)
    else:
        return row + 3.0 + 0j


def run_single_config(
    L: int,
    T: float, 
    seed: int,
    n_snapshots: int = 2,
    equilibration_sweeps: int = 200,
    decorrelation_sweeps: int = 30,
    J: int = 3,
    Q: int = 2,
    max_order: int = 2,
    encoding: str = 'magnitude',
) -> dict:
    """Run HST analysis for a single (L, T, seed) configuration."""
    
    # Initialize model
    model = IsingModel(L=L, seed=seed)
    
    # Hot/cold start
    if T > TC_EXACT:
        model.spins = model.rng.choice([-1, 1], size=(L, L))
    else:
        model.spins = np.ones((L, L), dtype=int)
    
    # Equilibrate with Wolff
    for _ in range(equilibration_sweeps):
        model.wolff_step(T)
    
    # Collect signals and metadata
    signals = []
    mags = []
    mag_sqs = []
    
    for snap_idx in range(n_snapshots):
        if snap_idx > 0:
            for _ in range(decorrelation_sweeps):
                model.wolff_step(T)
        
        m = model.magnetization()
        mags.append(abs(m))
        mag_sqs.append(m ** 2)
        
        for row_idx in range(L):
            row = model.spins[row_idx, :].astype(float)
            signal = encode_spins(row, encoding)
            signals.append(signal)
    
    # Create HST
    hst = HeisenbergScatteringTransformTorch(
        L, J=J, Q=Q, max_order=max_order,
        lifting='radial_floor', epsilon=1e-8,
        device=torch.device('cpu'),
    )
    
    # Compute metrics
    metrics = {
        'total_energy': {m: [] for m in range(max_order + 1)},
        'd_eff': {m: [] for m in range(max_order + 1)},
        'top8': {m: [] for m in range(max_order + 1)},
    }
    
    for x_np in signals:
        x = torch.tensor(x_np, dtype=torch.complex128)
        output = hst.forward(x, max_order=max_order)
        
        for m in range(max_order + 1):
            paths_m = output.order(m)
            energies = np.array([
                torch.sum(torch.abs(c) ** 2).item() 
                for c in paths_m.values()
            ])
            
            total_e = np.sum(energies)
            metrics['total_energy'][m].append(total_e)
            
            if np.sum(energies ** 2) > 0:
                d_eff = (total_e ** 2) / np.sum(energies ** 2)
            else:
                d_eff = 1.0
            metrics['d_eff'][m].append(d_eff)
            
            if len(energies) >= 8 and total_e > 0:
                top8 = np.sum(np.sort(energies)[-8:]) / total_e
            else:
                top8 = 1.0
            metrics['top8'][m].append(top8)
    
    # Aggregate results
    result = {
        'L': L,
        'T': T,
        'seed': seed,
        'encoding': encoding,
        'mean_mag': float(np.mean(mags)),
        'mean_mag_sq': float(np.mean(mag_sqs)),
        'susceptibility': float(L * L * (np.mean(mag_sqs) - np.mean(mags)**2)),
    }
    
    for m in range(max_order + 1):
        result[f'd_eff_{m}'] = float(np.mean(metrics['d_eff'][m]))
        result[f'top8_{m}'] = float(np.mean(metrics['top8'][m]))
        result[f'total_energy_{m}'] = float(np.mean(metrics['total_energy'][m]))
    
    return result


def main():
    """Run a quick local test."""
    print("="*60)
    print("LOCAL TEST: HST Ising Experiment")
    print("="*60)
    
    # Quick test
    print("\n1. Single config test...")
    result = run_single_config(L=16, T=TC_EXACT, seed=42, n_snapshots=1)
    print(f"   L={result['L']}, T={result['T']:.3f}")
    print(f"   |m|={result['mean_mag']:.3f}, χ={result['susceptibility']:.2f}")
    print(f"   d_eff_1={result['d_eff_1']:.2f}, d_eff_2={result['d_eff_2']:.2f}")
    print("   ✓ Single config works")
    
    # Mini sweep
    print("\n2. Mini sweep test (3 temperatures, 2 seeds)...")
    T_values = [1.5, TC_EXACT, 3.5]
    results = []
    
    for T in T_values:
        for seed in [0, 1]:
            r = run_single_config(L=16, T=T, seed=seed, n_snapshots=1)
            results.append(r)
            print(f"   T={T:.2f}, seed={seed}: d_eff_1={r['d_eff_1']:.2f}")
    
    print("   ✓ Mini sweep works")
    
    # Summary
    print("\n3. Summary by temperature:")
    for T in T_values:
        T_results = [r for r in results if abs(r['T'] - T) < 0.01]
        d_eff_1_mean = np.mean([r['d_eff_1'] for r in T_results])
        chi_mean = np.mean([r['susceptibility'] for r in T_results])
        print(f"   T={T:.2f}: d_eff_1={d_eff_1_mean:.2f}, χ={chi_mean:.2f}")
    
    print("\n" + "="*60)
    print("LOCAL TEST PASSED - Ready for Modal deployment")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
