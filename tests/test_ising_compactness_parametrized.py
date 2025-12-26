#!/usr/bin/env python3
"""
Ising Compactness Analysis - Parametrized for Local/Cloud Execution

This script can run in three modes:
1. QUICK: Fast sanity check (default, ~30 seconds)
2. MEDIUM: Reasonable statistics (~5 minutes on M1/M2)
3. FULL: Publication-quality (~30+ minutes, use GPU cloud)

Usage:
    python test_ising_compactness_parametrized.py              # Quick mode
    python test_ising_compactness_parametrized.py --medium     # Medium mode
    python test_ising_compactness_parametrized.py --full       # Full mode
    python test_ising_compactness_parametrized.py --full --parallel 8  # Parallel

Environment:
    - Apple Silicon: Use --medium or --full with --parallel
    - GPU Cloud: Use --full --parallel <n_cores>
"""

import numpy as np
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.benchmarks.ising import IsingModel, TC_EXACT


@dataclass
class RunConfig:
    """Configuration for a test run."""
    name: str
    L_values: List[int]           # Lattice sizes
    T_values: List[float]         # Temperatures to test
    n_seeds: int                  # Seeds per (L, T) combination
    n_snapshots: int              # Snapshots per seed
    equilibration_sweeps: int     # Wolff steps for equilibration
    decorrelation_sweeps: int     # Wolff steps between snapshots
    max_order: int                # HST max order
    J: int                        # HST J parameter
    Q: int                        # HST Q parameter
    encoding: str = 'magnitude'   # 'magnitude', 'phase', or 'full'


def encode_spins(row: np.ndarray, encoding: str) -> np.ndarray:
    """
    Encode spin row as complex signal.
    
    Encodings:
    - 'magnitude': row + 3.0 + 0j  (phase = 0, tests modulus-like behavior)
    - 'phase': 3.0 * exp(i * π * (row+1)/4)  (magnitude constant, phase = 0 or π/2)
    - 'full': 2.0 + row + 1j*row  (both magnitude and phase vary)
    """
    if encoding == 'magnitude':
        # Traditional: all info in magnitude, phase = 0
        return row + 3.0 + 0j
    elif encoding == 'phase':
        # Phase encoding: constant magnitude, spins in phase
        # +1 → phase = π/2, -1 → phase = 0
        phase = np.pi * (row + 1) / 4  # Maps -1→0, +1→π/2
        return 3.0 * np.exp(1j * phase)
    elif encoding == 'full':
        # Both real and imag carry spin info
        return 2.0 + row + 1j * row
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


# Predefined configurations
CONFIGS = {
    'quick': RunConfig(
        name='QUICK',
        L_values=[16],
        T_values=[1.5, TC_EXACT, 3.5],
        n_seeds=3,
        n_snapshots=1,
        equilibration_sweeps=50,
        decorrelation_sweeps=10,
        max_order=2,
        J=2,
        Q=2,
        encoding='magnitude',
    ),
    'quick_phase': RunConfig(
        name='QUICK_PHASE',
        L_values=[16],
        T_values=[1.5, TC_EXACT, 3.5],
        n_seeds=3,
        n_snapshots=1,
        equilibration_sweeps=50,
        decorrelation_sweeps=10,
        max_order=2,
        J=2,
        Q=2,
        encoding='phase',
    ),
    'medium': RunConfig(
        name='MEDIUM',
        L_values=[32],
        T_values=[1.5, 2.0, TC_EXACT, 2.5, 3.5],
        n_seeds=10,
        n_snapshots=2,
        equilibration_sweeps=200,
        decorrelation_sweeps=30,
        max_order=2,
        J=3,
        Q=2,
        encoding='magnitude',
    ),
    'medium_phase': RunConfig(
        name='MEDIUM_PHASE',
        L_values=[32],
        T_values=[1.5, 2.0, TC_EXACT, 2.5, 3.5],
        n_seeds=10,
        n_snapshots=2,
        equilibration_sweeps=200,
        decorrelation_sweeps=30,
        max_order=2,
        J=3,
        Q=2,
        encoding='phase',
    ),
    'full': RunConfig(
        name='FULL',
        L_values=[16, 32, 64],
        T_values=[1.5, 2.0, 2.1, 2.2, TC_EXACT, 2.35, 2.5, 3.0, 3.5],
        n_seeds=20,
        n_snapshots=4,
        equilibration_sweeps=500,
        decorrelation_sweeps=50,
        max_order=2,
        J=4,
        Q=2,
        encoding='magnitude',
    ),
    'full_phase': RunConfig(
        name='FULL_PHASE',
        L_values=[16, 32, 64],
        T_values=[1.5, 2.0, 2.1, 2.2, TC_EXACT, 2.35, 2.5, 3.0, 3.5],
        n_seeds=20,
        n_snapshots=4,
        equilibration_sweeps=500,
        decorrelation_sweeps=50,
        max_order=2,
        J=4,
        Q=2,
        encoding='phase',
    ),
}


def generate_ising_signals_efficient(
    L: int,
    T: float,
    n_snapshots: int,
    equilibration_sweeps: int,
    decorrelation_sweeps: int,
    seed: int,
    encoding: str = 'magnitude',
) -> Tuple[List[np.ndarray], Dict]:
    """
    Generate Ising signals with efficient sampling.
    
    Instead of restarting for each snapshot, we:
    1. Equilibrate once
    2. Take snapshots with decorrelation steps between them
    
    Returns signals and metadata (magnetization, energy).
    """
    model = IsingModel(L=L, seed=seed)
    
    # Hot/cold start
    if T > TC_EXACT:
        model.spins = model.rng.choice([-1, 1], size=(L, L))
    else:
        model.spins = np.ones((L, L), dtype=int)
    
    # Equilibrate once with Wolff
    model.equilibrate(T, equilibration_sweeps, use_wolff=True)
    
    signals = []
    mags = []
    energies = []
    
    for snap_idx in range(n_snapshots):
        # Decorrelation sweeps (except first)
        if snap_idx > 0:
            for _ in range(decorrelation_sweeps):
                model.wolff_step(T)
        
        # Record snapshot
        mags.append(abs(model.magnetization()))
        energies.append(model.energy_fast() / (L * L))
        
        # Extract rows as signals with specified encoding
        for row_idx in range(L):
            row = model.spins[row_idx, :].astype(float)
            signal = encode_spins(row, encoding)
            signals.append(signal)
    
    metadata = {
        'mean_mag': np.mean(mags),
        'mean_energy': np.mean(energies),
    }
    
    return signals, metadata


def compute_metrics(
    hst: HeisenbergScatteringTransform,
    signals: List[np.ndarray],
    max_order: int,
) -> Dict:
    """Compute compactness metrics for a set of signals."""
    
    metrics = {
        'input_energy': [],
        'total_energy': {m: [] for m in range(max_order + 1)},
        'energy_fraction': {m: [] for m in range(max_order + 1)},  # Fraction of total HST energy
        'd_eff': {m: [] for m in range(max_order + 1)},
        'top8': {m: [] for m in range(max_order + 1)},
        'entropy': {m: [] for m in range(max_order + 1)},
    }
    
    for x in signals:
        # Input energy
        input_e = np.sum(np.abs(x) ** 2)
        metrics['input_energy'].append(input_e)
        
        output = hst.forward(x, max_order=max_order)
        
        # First pass: compute total energies
        order_energies = {}
        for m in range(max_order + 1):
            paths_m = output.order(m)
            energies = np.array([np.sum(np.abs(c) ** 2) for c in paths_m.values()])
            order_energies[m] = energies
        
        total_hst_energy = sum(np.sum(e) for e in order_energies.values())
        
        # Second pass: compute metrics
        for m in range(max_order + 1):
            energies = order_energies[m]
            total_e = np.sum(energies)
            metrics['total_energy'][m].append(total_e)
            
            # Energy fraction (relative to total HST energy, not input)
            if total_hst_energy > 0:
                metrics['energy_fraction'][m].append(total_e / total_hst_energy)
            else:
                metrics['energy_fraction'][m].append(0)
            
            # Effective dimension
            if np.sum(energies ** 2) > 0:
                d_eff = (total_e ** 2) / np.sum(energies ** 2)
            else:
                d_eff = 1.0
            metrics['d_eff'][m].append(d_eff)
            
            # Top-8 fraction
            if len(energies) >= 8 and total_e > 0:
                top8 = np.sum(np.sort(energies)[-8:]) / total_e
            else:
                top8 = 1.0
            metrics['top8'][m].append(top8)
            
            # Entropy (normalized)
            if total_e > 0:
                w = energies / total_e
                w = w[w > 0]  # Avoid log(0)
                entropy = -np.sum(w * np.log(w)) / np.log(len(energies) + 1)
            else:
                entropy = 0
            metrics['entropy'][m].append(entropy)
    
    return metrics


def run_single_seed(args):
    """Run analysis for a single (L, T, seed) combination. For parallel execution."""
    L, T, seed, config = args
    
    hst = HeisenbergScatteringTransform(
        L, J=config.J, Q=config.Q, max_order=config.max_order,
        lifting='radial_floor', epsilon=1e-8
    )
    
    signals, metadata = generate_ising_signals_efficient(
        L=L, T=T,
        n_snapshots=config.n_snapshots,
        equilibration_sweeps=config.equilibration_sweeps,
        decorrelation_sweeps=config.decorrelation_sweeps,
        seed=seed,
        encoding=config.encoding,
    )
    
    metrics = compute_metrics(hst, signals, config.max_order)
    
    # Aggregate to per-seed means
    result = {
        'L': L,
        'T': T,
        'seed': seed,
        'mean_mag': metadata['mean_mag'],
        'mean_energy': metadata['mean_energy'],
        'encoding': config.encoding,
    }
    
    for m in range(config.max_order + 1):
        result[f'd_eff_{m}'] = np.mean(metrics['d_eff'][m])
        result[f'top8_{m}'] = np.mean(metrics['top8'][m])
        result[f'entropy_{m}'] = np.mean(metrics['entropy'][m])
        result[f'total_energy_{m}'] = np.mean(metrics['total_energy'][m])
        result[f'energy_frac_{m}'] = np.mean(metrics['energy_fraction'][m])
    
    return result


def run_analysis(config: RunConfig, n_parallel: int = 1):
    """Run the full analysis with given configuration."""
    
    print("="*70)
    print(f"ISING COMPACTNESS ANALYSIS - {config.name} MODE")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Lattice sizes: {config.L_values}")
    print(f"  Temperatures: {[f'{T:.2f}' for T in config.T_values]}")
    print(f"  Tc = {TC_EXACT:.4f}")
    print(f"  Seeds per (L,T): {config.n_seeds}")
    print(f"  Snapshots per seed: {config.n_snapshots}")
    print(f"  HST: J={config.J}, Q={config.Q}, max_order={config.max_order}")
    print(f"  Encoding: {config.encoding}")
    print(f"  Parallel workers: {n_parallel}")
    
    # Build work items
    work_items = []
    for L in config.L_values:
        for T in config.T_values:
            for seed_idx in range(config.n_seeds):
                seed = 1000 * seed_idx + int(T * 100) + L
                work_items.append((L, T, seed, config))
    
    print(f"\n  Total work items: {len(work_items)}")
    
    start_time = time.time()
    
    if n_parallel > 1:
        from multiprocessing import Pool
        print(f"\n  Running in parallel with {n_parallel} workers...")
        with Pool(n_parallel) as pool:
            results = pool.map(run_single_seed, work_items)
    else:
        print(f"\n  Running sequentially...")
        results = []
        for i, item in enumerate(work_items):
            if i % max(1, len(work_items) // 10) == 0:
                print(f"    Progress: {i}/{len(work_items)}")
            results.append(run_single_seed(item))
    
    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f} seconds")
    
    # Organize results by (L, T)
    organized = {}
    for r in results:
        key = (r['L'], r['T'])
        if key not in organized:
            organized[key] = []
        organized[key].append(r)
    
    return organized, config


def print_results(organized: Dict, config: RunConfig):
    """Print analysis results."""
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for L in config.L_values:
        print(f"\n{'='*60}")
        print(f"LATTICE SIZE L = {L} (encoding: {config.encoding})")
        print(f"{'='*60}")
        
        # Get path counts for this L
        hst = HeisenbergScatteringTransform(
            L, J=config.J, Q=config.Q, max_order=config.max_order,
            lifting='radial_floor', epsilon=1e-8
        )
        sample_output = hst.forward(np.ones(L, dtype=complex) + 3.0, max_order=config.max_order)
        path_counts = {m: len(sample_output.order(m)) for m in range(config.max_order + 1)}
        
        print(f"\n  Path counts: {path_counts}")
        
        # Energy fraction table (NEW)
        print(f"\n  ENERGY FRACTIONS (E_m / E_total_HST):")
        print(f"  {'T':<8} {'|m|':<8}", end="")
        for m in range(config.max_order + 1):
            print(f" {'Order '+str(m):<12}", end="")
        print()
        print(f"  {'-'*55}")
        
        for T in config.T_values:
            key = (L, T)
            if key not in organized:
                continue
            
            data = organized[key]
            mags = [d['mean_mag'] for d in data]
            
            is_critical = abs(T - TC_EXACT) < 0.01
            marker = " *" if is_critical else ""
            
            print(f"  {T:<8.2f} {np.mean(mags):<8.2f}", end="")
            for m in range(config.max_order + 1):
                fracs = [d[f'energy_frac_{m}'] for d in data]
                print(f" {np.mean(fracs):<12.4f}", end="")
            print(marker)
        
        # d_eff table
        print(f"\n  EFFECTIVE DIMENSION (d_eff / #paths):")
        print(f"  {'T':<8} {'|m|':<8}", end="")
        for m in range(1, config.max_order + 1):
            print(f" {'Order '+str(m)+f' (n={path_counts[m]})':<18}", end="")
        print()
        print(f"  {'-'*60}")
        
        T_results = []
        for T in config.T_values:
            key = (L, T)
            if key not in organized:
                continue
            
            data = organized[key]
            n_seeds = len(data)
            
            # Compute means and stderrs
            row = {'T': T, 'n': n_seeds}
            
            # Magnetization
            mags = [d['mean_mag'] for d in data]
            row['mag'] = np.mean(mags)
            
            for m in range(1, config.max_order + 1):
                d_effs = [d[f'd_eff_{m}'] for d in data]
                row[f'd_eff_{m}_mean'] = np.mean(d_effs)
                row[f'd_eff_{m}_se'] = np.std(d_effs, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0
                row[f'd_eff_{m}_ratio'] = row[f'd_eff_{m}_mean'] / path_counts[m]
            
            T_results.append(row)
            
            # Print row
            is_critical = abs(T - TC_EXACT) < 0.01
            marker = " *" if is_critical else ""
            
            print(f"  {T:<8.2f} {row['mag']:<8.2f}", end="")
            for m in range(1, config.max_order + 1):
                ratio = row[f'd_eff_{m}_ratio']
                se = row[f'd_eff_{m}_se'] / path_counts[m]
                print(f" {ratio:.3f}±{se:.3f}         ", end="")
            print(marker)
        
        # Find minimum d_eff temperature
        if T_results:
            print(f"\n  Minimum d_eff/n locations:")
            for m in range(1, config.max_order + 1):
                min_row = min(T_results, key=lambda r: r[f'd_eff_{m}_ratio'])
                print(f"    Order {m}: T = {min_row['T']:.2f} (d_eff/n = {min_row[f'd_eff_{m}_ratio']:.3f})")
        
        # Statistical tests vs critical
        print(f"\n  Statistical comparison (Welch t-test vs T≈Tc):")
        
        # Find critical
        critical_T = min(config.T_values, key=lambda t: abs(t - TC_EXACT))
        critical_key = (L, critical_T)
        
        if critical_key in organized:
            crit_data = organized[critical_key]
            
            for m in [1, 2]:
                crit_vals = np.array([d[f'd_eff_{m}'] for d in crit_data])
                
                print(f"\n    Order {m}:")
                for T in config.T_values:
                    if abs(T - critical_T) < 0.01:
                        continue
                    
                    key = (L, T)
                    if key not in organized:
                        continue
                    
                    other_vals = np.array([d[f'd_eff_{m}'] for d in organized[key]])
                    
                    t_stat, p_value = ttest_ind(crit_vals, other_vals, equal_var=False)
                    diff = np.mean(crit_vals) - np.mean(other_vals)
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    
                    print(f"      T={T:.2f}: diff={diff:+.2f}, p={p_value:.2e} {sig}")


def run_quick_sanity_check():
    """Ultra-fast sanity check that the code works."""
    print("="*70)
    print("QUICK SANITY CHECK")
    print("="*70)
    
    L = 16
    T = TC_EXACT
    
    print(f"\n  Testing single (L={L}, T={T:.2f}) configuration...")
    
    config = CONFIGS['quick']
    result = run_single_seed((L, T, 42, config))
    
    print(f"\n  Results:")
    print(f"    |m| = {result['mean_mag']:.3f}")
    print(f"    d_eff(1) = {result['d_eff_1']:.2f}")
    print(f"    d_eff(2) = {result['d_eff_2']:.2f}")
    print(f"    top8(1) = {result['top8_1']:.3f}")
    print(f"    entropy(1) = {result['entropy_1']:.3f}")
    
    print(f"\n  ✓ Sanity check passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Ising Compactness Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick mode (~30s)')
    parser.add_argument('--medium', action='store_true', help='Medium mode (~5min)')
    parser.add_argument('--full', action='store_true', help='Full mode (~30min+)')
    parser.add_argument('--phase', action='store_true', help='Use phase encoding instead of magnitude')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--sanity', action='store_true', help='Just run sanity check')
    
    args = parser.parse_args()
    
    if args.sanity:
        return run_quick_sanity_check()
    
    # Select config
    if args.full:
        config = CONFIGS['full_phase'] if args.phase else CONFIGS['full']
    elif args.medium:
        config = CONFIGS['medium_phase'] if args.phase else CONFIGS['medium']
    else:
        config = CONFIGS['quick_phase'] if args.phase else CONFIGS['quick']
    
    organized, config = run_analysis(config, n_parallel=args.parallel)
    print_results(organized, config)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
