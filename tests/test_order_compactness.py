#!/usr/bin/env python3
"""
Test: Order Compactness (Top-K Energy Concentration & Effective Dimension)

This tests Glinsky's practical "superconvergence" claim: that HST coefficients
are compressible / ROM-friendly.

Metrics:
1. Top-K energy concentration: What fraction of energy is in the top K paths?
2. Effective dimension (participation ratio): How many paths "matter"?

   d_eff = (Σ e_p)² / Σ e_p²

If d_eff << n_paths, the representation is compressible.

Run: python tests/test_order_compactness.py
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform


def generate_test_signals(n_signals: int, T: int, seed: int = 42) -> Dict[str, List[np.ndarray]]:
    """Generate test signal families."""
    np.random.seed(seed)
    
    families = {}
    
    # AM signals
    am_signals = []
    for i in range(n_signals):
        t = np.arange(T)
        f_am = np.random.uniform(1, 5)
        depth = np.random.uniform(0.3, 0.8)
        amplitude = 1 + depth * np.cos(2 * np.pi * f_am * t / T)
        x = amplitude + 3.0 + 0j
        am_signals.append(x)
    families['AM'] = am_signals
    
    # FM signals
    fm_signals = []
    for i in range(n_signals):
        t = np.arange(T)
        f_carrier = np.random.uniform(5, 15)
        f_mod = np.random.uniform(1, 3)
        mod_index = np.random.uniform(0.5, 2.0)
        phase = 2 * np.pi * f_carrier * t / T + mod_index * np.sin(2 * np.pi * f_mod * t / T)
        x = np.exp(1j * phase) + 3.0
        fm_signals.append(x)
    families['FM'] = fm_signals
    
    # Mixed signals
    mixed_signals = []
    for i in range(n_signals):
        t = np.arange(T)
        f1 = np.random.uniform(5, 8)
        f2 = f1 + np.random.uniform(0.5, 1.5)
        x = np.cos(2 * np.pi * f1 * t / T) + 0.7 * np.cos(2 * np.pi * f2 * t / T)
        noise = np.random.randn(T) * 0.1
        x = x + noise + 3.0 + 0j
        mixed_signals.append(x)
    families['Mixed'] = mixed_signals
    
    return families


def compute_path_energies(hst: HeisenbergScatteringTransform, 
                          x: np.ndarray, 
                          max_order: int) -> Dict[int, np.ndarray]:
    """
    Compute energy for each path, grouped by order.
    
    Returns dict: order -> array of energies (one per path at that order)
    """
    output = hst.forward(x, max_order=max_order)
    
    energies_by_order = {}
    for m in range(max_order + 1):
        paths_m = output.order(m)
        energies = np.array([np.sum(np.abs(coef) ** 2) for coef in paths_m.values()])
        energies_by_order[m] = energies
    
    return energies_by_order


def top_k_fraction(energies: np.ndarray, k: int) -> float:
    """Compute fraction of total energy in top-k paths."""
    if len(energies) == 0:
        return 1.0
    if k >= len(energies):
        return 1.0
    
    sorted_e = np.sort(energies)[::-1]  # Descending
    total = np.sum(energies)
    if total == 0:
        return 1.0
    return np.sum(sorted_e[:k]) / total


def effective_dimension(energies: np.ndarray) -> float:
    """
    Compute effective dimension (participation ratio).
    
    d_eff = (Σ e_p)² / Σ e_p²
    
    This is the "number of paths that matter":
    - If all paths equal: d_eff = n_paths
    - If one path dominates: d_eff ≈ 1
    """
    if len(energies) == 0:
        return 0
    
    total = np.sum(energies)
    sum_sq = np.sum(energies ** 2)
    
    if sum_sq == 0:
        return 0
    
    return (total ** 2) / sum_sq


def test_order_compactness():
    """
    Test compactness metrics across signal families.
    """
    print("="*70)
    print("ORDER COMPACTNESS TEST")
    print("Testing 'ROM-friendly' compressibility of HST coefficients")
    print("="*70)
    
    T = 1024
    n_signals = 16
    max_order = 3
    
    hst = HeisenbergScatteringTransform(
        T, J=4, Q=2, max_order=max_order,
        lifting='radial_floor', epsilon=1e-8
    )
    
    # Get path counts
    sample_output = hst.forward(np.ones(T, dtype=complex) + 3.0, max_order=max_order)
    path_counts = {m: len(sample_output.order(m)) for m in range(max_order + 1)}
    
    print(f"\nPath counts: {path_counts}")
    print(f"Total paths: {sum(path_counts.values())}")
    
    families = generate_test_signals(n_signals, T)
    
    K_values = [4, 8, 16, 32]
    
    for family_name, signals in families.items():
        print(f"\n{'='*60}")
        print(f"FAMILY: {family_name}")
        print(f"{'='*60}")
        
        # Aggregate metrics across signals
        all_top_k = {m: {k: [] for k in K_values} for m in range(max_order + 1)}
        all_d_eff = {m: [] for m in range(max_order + 1)}
        
        for x in signals:
            energies_by_order = compute_path_energies(hst, x, max_order)
            
            for m in range(max_order + 1):
                energies = energies_by_order[m]
                
                # Top-K fractions
                for k in K_values:
                    frac = top_k_fraction(energies, k)
                    all_top_k[m][k].append(frac)
                
                # Effective dimension
                d_eff = effective_dimension(energies)
                all_d_eff[m].append(d_eff)
        
        # Report Top-K
        print(f"\n  TOP-K ENERGY CONCENTRATION")
        print(f"  (Fraction of total energy in top K paths)")
        header = f"  {'Order':<8} {'#Paths':<8}"
        for k in K_values:
            header += f" {'Top-'+str(k):<10}"
        print(header)
        print(f"  {'-'*(8+8+10*len(K_values))}")
        
        for m in range(max_order + 1):
            row = f"  {m:<8} {path_counts[m]:<8}"
            for k in K_values:
                mean_frac = np.mean(all_top_k[m][k])
                row += f" {mean_frac:<10.4f}"
            print(row)
        
        # Report Effective Dimension
        print(f"\n  EFFECTIVE DIMENSION (participation ratio)")
        print(f"  d_eff = (Σe)²/Σe² : 'how many paths matter'")
        print(f"  {'Order':<8} {'#Paths':<8} {'d_eff':<12} {'d_eff/#Paths':<15} {'Interpretation':<20}")
        print(f"  {'-'*65}")
        
        for m in range(max_order + 1):
            n_paths = path_counts[m]
            mean_d_eff = np.mean(all_d_eff[m])
            ratio = mean_d_eff / n_paths if n_paths > 0 else 0
            
            if ratio < 0.1:
                interp = "HIGHLY COMPACT"
            elif ratio < 0.3:
                interp = "COMPACT"
            elif ratio < 0.6:
                interp = "MODERATE"
            else:
                interp = "SPREAD"
            
            print(f"  {m:<8} {n_paths:<8} {mean_d_eff:<12.2f} {ratio:<15.4f} {interp:<20}")
    
    return True


def test_compactness_summary():
    """
    Summary view across all families.
    """
    print("\n" + "="*70)
    print("COMPACTNESS SUMMARY ACROSS FAMILIES")
    print("="*70)
    
    T = 1024
    n_signals = 16
    max_order = 3
    
    hst = HeisenbergScatteringTransform(
        T, J=4, Q=2, max_order=max_order,
        lifting='radial_floor', epsilon=1e-8
    )
    
    families = generate_test_signals(n_signals, T)
    
    # Get path counts
    sample_output = hst.forward(np.ones(T, dtype=complex) + 3.0, max_order=max_order)
    path_counts = {m: len(sample_output.order(m)) for m in range(max_order + 1)}
    
    print(f"\n  {'Family':<10} {'Order':<8} {'#Paths':<8} {'d_eff':<10} {'Top-8':<10} {'Compact?':<12}")
    print(f"  {'-'*60}")
    
    for family_name, signals in families.items():
        for m in [1, 2, 3]:  # Skip order 0 (trivial)
            d_effs = []
            top8s = []
            
            for x in signals:
                energies = compute_path_energies(hst, x, max_order)[m]
                d_effs.append(effective_dimension(energies))
                top8s.append(top_k_fraction(energies, 8))
            
            mean_d_eff = np.mean(d_effs)
            mean_top8 = np.mean(top8s)
            
            compact = "✓ YES" if mean_d_eff / path_counts[m] < 0.3 else "~ PARTIAL" if mean_d_eff / path_counts[m] < 0.6 else "✗ NO"
            
            print(f"  {family_name:<10} {m:<8} {path_counts[m]:<8} {mean_d_eff:<10.2f} {mean_top8:<10.4f} {compact:<12}")
    
    print(f"\n  Interpretation:")
    print(f"    d_eff << #Paths means energy is concentrated in few paths")
    print(f"    Top-8 > 0.9 means 8 paths capture 90%+ of energy")
    print(f"    Both indicate 'ROM-friendly' compressibility")
    
    return True


def test_compression_ratio():
    """
    Estimate compression ratio: how many paths needed to capture 90% energy?
    """
    print("\n" + "="*70)
    print("COMPRESSION RATIO: Paths needed for 90% energy")
    print("="*70)
    
    T = 1024
    n_signals = 16
    max_order = 3
    
    hst = HeisenbergScatteringTransform(
        T, J=4, Q=2, max_order=max_order,
        lifting='radial_floor', epsilon=1e-8
    )
    
    families = generate_test_signals(n_signals, T)
    
    sample_output = hst.forward(np.ones(T, dtype=complex) + 3.0, max_order=max_order)
    path_counts = {m: len(sample_output.order(m)) for m in range(max_order + 1)}
    
    print(f"\n  {'Family':<10} {'Order':<8} {'#Paths':<8} {'K for 90%':<12} {'Compression':<15}")
    print(f"  {'-'*55}")
    
    for family_name, signals in families.items():
        for m in [1, 2, 3]:
            k_90_list = []
            
            for x in signals:
                energies = compute_path_energies(hst, x, max_order)[m]
                
                # Find K needed for 90% energy
                sorted_e = np.sort(energies)[::-1]
                total = np.sum(energies)
                if total == 0:
                    k_90_list.append(1)
                    continue
                
                cumsum = np.cumsum(sorted_e)
                k_90 = np.searchsorted(cumsum, 0.9 * total) + 1
                k_90 = min(k_90, len(energies))
                k_90_list.append(k_90)
            
            mean_k = np.mean(k_90_list)
            compression = path_counts[m] / mean_k if mean_k > 0 else float('inf')
            
            print(f"  {family_name:<10} {m:<8} {path_counts[m]:<8} {mean_k:<12.1f} {compression:<15.1f}x")
    
    print(f"\n  'Compression' = #Paths / K_90%")
    print(f"  Higher compression = more ROM-friendly")
    
    return True


def main():
    print("="*70)
    print("HST ORDER COMPACTNESS ANALYSIS")
    print("Testing Glinsky's 'superconvergence' as compressibility")
    print("="*70)
    
    test_order_compactness()
    test_compactness_summary()
    test_compression_ratio()
    
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
  Glinsky's "superconvergence" claim in practical terms means:
  - HST coefficients should be compressible
  - A small number of paths should capture most energy
  - Effective dimension d_eff << total path count
  
  Our findings for these test signals:
  1. TOP-K CONCENTRATION: Energy is spread - Top 8 captures only 5-77%
  2. EFFECTIVE DIMENSION: d_eff is 50-98% of path count (NOT compact)
  3. COMPRESSION RATIO: Only 1.1-1.6x compression for 90% energy
  
  ASSESSMENT:
  - For simple AM/FM/Mixed signals, HST does NOT show strong compactness
  - Energy spreads across many paths rather than concentrating in a few
  - This may be because:
    a) These signals ARE simple (don't need hierarchical decomposition)
    b) The filterbank is too redundant for these signal classes
    c) "Superconvergence" may require specific signal structure
  
  To see compactness, we might need:
  - Signals with multi-scale/hierarchical structure
  - Signals from physical systems (turbulence, finance, etc.)
  - Different filterbank parameters
  
  The test infrastructure is now in place to measure compactness
  on more realistic signal ensembles.
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
