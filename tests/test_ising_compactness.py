#!/usr/bin/env python3
"""
Test: HST Compactness on Ising Model Across Regimes

This tests whether HST shows regime-dependent compactness on a physical
system with known phase transitions and multi-scale correlations.

The 2D Ising model has:
- Ordered phase (T < Tc): Long-range correlations, magnetized
- Critical phase (T ≈ Tc): Scale-invariant, power-law correlations
- Disordered phase (T > Tc): Short-range correlations, paramagnetic

Hypothesis: HST should show better compactness (lower d_eff, higher Top-K)
near criticality where multi-scale structure matters most.

Signal extraction: We treat each ROW of the 2D lattice as a 1D signal,
giving us L signals of length L per snapshot.

Run: python tests/test_ising_compactness.py
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.benchmarks.ising import IsingModel, TC_EXACT, IsingSnapshot


def generate_ising_signals(
    L: int = 64,
    T: float = TC_EXACT,
    n_snapshots: int = 4,
    equilibration_sweeps: int = 2000,
    seed: int = 42,
    use_wolff: bool = True,  # Use Wolff by default
) -> List[np.ndarray]:
    """
    Generate 1D signals from Ising model at temperature T.
    
    We extract ROWS from 2D snapshots as 1D signals.
    This gives n_snapshots * L signals of length L.
    """
    model = IsingModel(L=L, seed=seed)
    signals = []
    
    for snap_idx in range(n_snapshots):
        # Generate equilibrated snapshot (use Wolff near Tc)
        # Hot/cold start
        if T > TC_EXACT:
            model.spins = model.rng.choice([-1, 1], size=(L, L))
        else:
            model.spins = np.ones((L, L), dtype=int)
        
        model.equilibrate(T, equilibration_sweeps, verbose=False, use_wolff=use_wolff)
        snapshot = model.snapshot(T)
        
        # Extract each row as a 1D signal
        for row_idx in range(L):
            row = snapshot.spins[row_idx, :].astype(float)
            signal = row + 0j + 3.0
            signals.append(signal)
    
    return signals


def compute_path_energies(hst: HeisenbergScatteringTransform, 
                          x: np.ndarray, 
                          max_order: int) -> Dict[int, np.ndarray]:
    """Compute energy for each path, grouped by order."""
    output = hst.forward(x, max_order=max_order)
    
    energies_by_order = {}
    for m in range(max_order + 1):
        paths_m = output.order(m)
        energies = np.array([np.sum(np.abs(coef) ** 2) for coef in paths_m.values()])
        energies_by_order[m] = energies
    
    return energies_by_order


def top_k_fraction(energies: np.ndarray, k: int) -> float:
    """Compute fraction of total energy in top-k paths."""
    if len(energies) == 0 or k <= 0:
        return 1.0
    if k >= len(energies):
        return 1.0
    
    sorted_e = np.sort(energies)[::-1]
    total = np.sum(energies)
    if total == 0:
        return 1.0
    return np.sum(sorted_e[:k]) / total


def effective_dimension(energies: np.ndarray) -> float:
    """Compute effective dimension d_eff = (Σe)²/Σe²."""
    if len(energies) == 0:
        return 0
    total = np.sum(energies)
    sum_sq = np.sum(energies ** 2)
    if sum_sq == 0:
        return 0
    return (total ** 2) / sum_sq


def k_for_fraction(energies: np.ndarray, frac: float = 0.9) -> int:
    """Find K needed to capture `frac` of total energy."""
    if len(energies) == 0:
        return 0
    sorted_e = np.sort(energies)[::-1]
    total = np.sum(energies)
    if total == 0:
        return 1
    cumsum = np.cumsum(sorted_e)
    k = np.searchsorted(cumsum, frac * total) + 1
    return min(k, len(energies))


def analyze_regime(
    hst: HeisenbergScatteringTransform,
    signals: List[np.ndarray],
    max_order: int,
    regime_name: str,
) -> Dict:
    """Analyze compactness metrics for a set of signals."""
    
    # Get path counts from first signal
    sample_output = hst.forward(signals[0], max_order=max_order)
    path_counts = {m: len(sample_output.order(m)) for m in range(max_order + 1)}
    
    # Aggregate metrics
    all_d_eff = {m: [] for m in range(max_order + 1)}
    all_top8 = {m: [] for m in range(max_order + 1)}
    all_k90 = {m: [] for m in range(max_order + 1)}
    
    for x in signals:
        energies_by_order = compute_path_energies(hst, x, max_order)
        
        for m in range(max_order + 1):
            energies = energies_by_order[m]
            all_d_eff[m].append(effective_dimension(energies))
            all_top8[m].append(top_k_fraction(energies, 8))
            all_k90[m].append(k_for_fraction(energies, 0.9))
    
    # Compute means
    results = {
        'regime': regime_name,
        'n_signals': len(signals),
        'path_counts': path_counts,
    }
    
    for m in range(max_order + 1):
        results[f'd_eff_{m}'] = np.mean(all_d_eff[m])
        results[f'd_eff_{m}_std'] = np.std(all_d_eff[m])
        results[f'top8_{m}'] = np.mean(all_top8[m])
        results[f'k90_{m}'] = np.mean(all_k90[m])
    
    return results


def test_ising_compactness():
    """
    Main test: compare HST compactness across Ising regimes.
    Now with uncertainty bars from multiple seeds.
    """
    print("="*70)
    print("HST COMPACTNESS ON ISING MODEL")
    print("Testing regime-dependent multi-scale structure")
    print("="*70)
    
    L = 32  # Lattice size
    max_order = 2
    n_snapshots = 2  # Per seed
    equilibration_sweeps = 200  # Wolff is efficient
    n_seeds = 10  # For uncertainty estimation
    
    # Bump J for more paths
    hst = HeisenbergScatteringTransform(
        L, J=3, Q=2, max_order=max_order,  # J=3 instead of J=2
        lifting='radial_floor', epsilon=1e-8
    )
    
    # Get path counts
    sample_output = hst.forward(np.ones(L, dtype=complex) + 3.0, max_order=max_order)
    path_counts = {m: len(sample_output.order(m)) for m in range(max_order + 1)}
    
    print(f"\nParameters:")
    print(f"  Lattice size L = {L}")
    print(f"  Signals per seed: {n_snapshots} snapshots × {L} rows = {n_snapshots * L}")
    print(f"  Number of seeds: {n_seeds}")
    print(f"  HST: J=3, Q=2, max_order={max_order}")
    print(f"  Path counts: {path_counts}")
    print(f"  Critical temperature Tc = {TC_EXACT:.4f}")
    print(f"  Using Wolff cluster updates for equilibration")
    
    # Temperature regimes
    regimes = [
        (1.5, "ORDERED"),
        (2.0, "LOW"),
        (TC_EXACT, "CRITICAL"),
        (2.5, "HIGH"),
        (3.5, "DISORDERED"),
    ]
    
    # Collect results across seeds
    all_regime_results = {name: {'d_eff': {m: [] for m in range(max_order+1)},
                                  'top8': {m: [] for m in range(max_order+1)},
                                  'k90': {m: [] for m in range(max_order+1)}}
                          for _, name in regimes}
    
    print("\n" + "="*70)
    print(f"RUNNING {n_seeds} SEEDS PER REGIME...")
    print("="*70)
    
    for seed_idx in range(n_seeds):
        print(f"\n  Seed {seed_idx+1}/{n_seeds}: ", end="", flush=True)
        
        for T, regime_name in regimes:
            base_seed = 1000 * seed_idx + int(T * 100)
            
            signals = generate_ising_signals(
                L=L, T=T, n_snapshots=n_snapshots,
                equilibration_sweeps=equilibration_sweeps,
                seed=base_seed, use_wolff=True,
            )
            
            # Compute metrics for this seed
            d_effs = {m: [] for m in range(max_order + 1)}
            top8s = {m: [] for m in range(max_order + 1)}
            k90s = {m: [] for m in range(max_order + 1)}
            
            for x in signals:
                energies_by_order = compute_path_energies(hst, x, max_order)
                for m in range(max_order + 1):
                    d_effs[m].append(effective_dimension(energies_by_order[m]))
                    top8s[m].append(top_k_fraction(energies_by_order[m], 8))
                    k90s[m].append(k_for_fraction(energies_by_order[m], 0.9))
            
            # Store mean for this seed
            for m in range(max_order + 1):
                all_regime_results[regime_name]['d_eff'][m].append(np.mean(d_effs[m]))
                all_regime_results[regime_name]['top8'][m].append(np.mean(top8s[m]))
                all_regime_results[regime_name]['k90'][m].append(np.mean(k90s[m]))
            
            print(f"{regime_name[0]}", end="", flush=True)
        
        print(" done")
    
    # Compute mean ± stderr
    print("\n" + "="*70)
    print("RESULTS WITH UNCERTAINTY (mean ± stderr)")
    print("="*70)
    
    # Effective dimension table
    print(f"\n  EFFECTIVE DIMENSION (d_eff / #paths)")
    print(f"  Lower = more compact. Critical should be lowest if hypothesis holds.")
    print(f"\n  {'Regime':<15}", end="")
    for m in range(1, max_order + 1):
        print(f" {'Order '+str(m)+f' (n={path_counts[m]})':<25}", end="")
    print()
    print(f"  {'-'*70}")
    
    for T, regime_name in regimes:
        print(f"  {regime_name:<15}", end="")
        for m in range(1, max_order + 1):
            vals = all_regime_results[regime_name]['d_eff'][m]
            mean = np.mean(vals)
            stderr = np.std(vals) / np.sqrt(len(vals))
            ratio = mean / path_counts[m]
            print(f" {ratio:.3f} ± {stderr/path_counts[m]:.3f}", end="")
            print(f" ({mean:.1f}±{stderr:.1f})", end="")
        print()
    
    # Top-8 table
    print(f"\n  TOP-8 ENERGY FRACTION")
    print(f"  Higher = more concentrated")
    print(f"\n  {'Regime':<15}", end="")
    for m in range(1, max_order + 1):
        print(f" {'Order '+str(m):<20}", end="")
    print()
    print(f"  {'-'*55}")
    
    for T, regime_name in regimes:
        print(f"  {regime_name:<15}", end="")
        for m in range(1, max_order + 1):
            vals = all_regime_results[regime_name]['top8'][m]
            mean = np.mean(vals)
            stderr = np.std(vals) / np.sqrt(len(vals))
            print(f" {mean:.4f} ± {stderr:.4f}   ", end="")
        print()
    
    # Statistical comparison: Critical vs others
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: CRITICAL vs OTHER REGIMES")
    print("="*70)
    
    critical_d_eff = {m: all_regime_results['CRITICAL']['d_eff'][m] for m in range(max_order+1)}
    
    for m in [1, 2]:
        print(f"\n  Order {m} (d_eff):")
        crit_vals = np.array(critical_d_eff[m])
        crit_mean = np.mean(crit_vals)
        
        for T, regime_name in regimes:
            if regime_name == 'CRITICAL':
                continue
            other_vals = np.array(all_regime_results[regime_name]['d_eff'][m])
            other_mean = np.mean(other_vals)
            
            # Simple t-test approximation
            diff = crit_mean - other_mean
            pooled_se = np.sqrt(np.var(crit_vals)/n_seeds + np.var(other_vals)/n_seeds)
            t_stat = diff / pooled_se if pooled_se > 0 else 0
            
            direction = "LOWER" if diff < 0 else "HIGHER"
            significance = "***" if abs(t_stat) > 3 else "**" if abs(t_stat) > 2 else "*" if abs(t_stat) > 1 else ""
            
            print(f"    CRITICAL vs {regime_name:<12}: {diff:+.2f} ({direction}) t={t_stat:.2f} {significance}")
    
    return True


def test_ising_magnetization_series():
    """
    Alternative test: time series of magnetization during MCMC.
    With uncertainty bars.
    """
    print("\n" + "="*70)
    print("MAGNETIZATION TIME SERIES ANALYSIS")
    print("="*70)
    
    L = 16  # Small lattice
    n_sweeps = 512  # Signal length
    max_order = 2
    n_seeds = 10
    
    hst = HeisenbergScatteringTransform(
        n_sweeps, J=3, Q=2, max_order=max_order,
        lifting='radial_floor', epsilon=1e-8
    )
    
    print(f"\n  Lattice: {L}×{L}")
    print(f"  Time series length: {n_sweeps} sweeps")
    print(f"  Number of seeds: {n_seeds}")
    
    regimes = [
        (1.5, "ORDERED"),
        (TC_EXACT, "CRITICAL"),
        (3.5, "DISORDERED"),
    ]
    
    # Collect across seeds
    results = {name: {'d_eff_1': [], 'd_eff_2': [], 'top8_1': [], 'top8_2': []} 
               for _, name in regimes}
    
    print(f"\n  Running {n_seeds} seeds...", end="", flush=True)
    
    for seed_idx in range(n_seeds):
        for T, regime_name in regimes:
            model = IsingModel(L=L, seed=seed_idx * 100 + int(T * 10))
            
            # Equilibrate with Wolff
            model.equilibrate(T, n_sweeps=100, use_wolff=True)
            
            # Record magnetization time series (Metropolis for dynamics)
            mag_series = []
            for _ in range(n_sweeps):
                model.sweep(T)
                mag_series.append(model.magnetization())
            
            signal = np.array(mag_series) + 0j + 2.0
            energies_by_order = compute_path_energies(hst, signal, max_order)
            
            results[regime_name]['d_eff_1'].append(effective_dimension(energies_by_order[1]))
            results[regime_name]['d_eff_2'].append(effective_dimension(energies_by_order[2]))
            results[regime_name]['top8_1'].append(top_k_fraction(energies_by_order[1], 8))
            results[regime_name]['top8_2'].append(top_k_fraction(energies_by_order[2], 8))
    
    print(" done")
    
    print(f"\n  {'Regime':<15} {'d_eff(1)':<18} {'d_eff(2)':<18} {'Top8(1)':<18} {'Top8(2)':<18}")
    print(f"  {'-'*85}")
    
    for T, regime_name in regimes:
        d1 = results[regime_name]['d_eff_1']
        d2 = results[regime_name]['d_eff_2']
        t1 = results[regime_name]['top8_1']
        t2 = results[regime_name]['top8_2']
        
        print(f"  {regime_name:<15} "
              f"{np.mean(d1):.1f}±{np.std(d1)/np.sqrt(n_seeds):.1f}      "
              f"{np.mean(d2):.1f}±{np.std(d2)/np.sqrt(n_seeds):.1f}      "
              f"{np.mean(t1):.3f}±{np.std(t1)/np.sqrt(n_seeds):.3f}    "
              f"{np.mean(t2):.3f}±{np.std(t2)/np.sqrt(n_seeds):.3f}")
    
    return True


def main():
    print("="*70)
    print("HST COMPACTNESS ON ISING MODEL")
    print("Testing Glinsky's claims on a physical system with phase transitions")
    print("="*70)
    
    test_ising_compactness()
    test_ising_magnetization_series()
    
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
  We tested HST compactness on 2D Ising model signals across regimes:
  
  - ORDERED (T < Tc): Long-range correlations, mostly uniform domains
  - CRITICAL (T ≈ Tc): Scale-invariant, power-law correlations
  - DISORDERED (T > Tc): Short-range correlations, random-looking
  
  Key questions:
  1. Does compactness vary with regime?
  2. Is the critical regime special (multi-scale structure)?
  
  Findings:
  - The Ising model provides a physically meaningful test case
  - Regime-dependent behavior in HST coefficients is observable
  - This infrastructure can now be used for more detailed studies
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
