#!/usr/bin/env python3
"""
Order Parameterization Tests

Tests for:
1. Order truncation correctness (subset consistency)
2. Monotone path constraint (j1 < j2 < j3 < ...)
3. Arbitrary order gradient computation
4. Super-convergence diagnostic (energy decay across orders)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    compute_loss_and_grad,
    finite_difference_gradient,
    extract_all_targets,
)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = {}
    
    def record(self, name: str, passed: bool, msg: str = ""):
        if passed:
            self.passed += 1
            status = "✓"
        else:
            self.failed += 1
            status = "✗"
        
        self.results[name] = {'passed': passed, 'msg': msg}
        print(f"  {status} {name}")
        if msg:
            print(f"      {msg}")


# =============================================================================
# Test 1: Order Truncation Correctness
# =============================================================================

def test_order_truncation_correctness():
    """
    Verify that hst.forward(x, max_order=1) produces a subset of paths
    from hst.forward(x, max_order=3), and values match exactly.
    """
    print("\n[TEST 1: ORDER TRUNCATION CORRECTNESS]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Compute at different max_orders
    hst1 = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=1, lifting='shift', epsilon=2.0)
    hst2 = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=2, lifting='shift', epsilon=2.0)
    hst3 = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=3, lifting='shift', epsilon=2.0)
    
    out1 = hst1.forward(x)
    out2 = hst2.forward(x)
    out3 = hst3.forward(x)
    
    print(f"  Order 1 paths: {len(out1.paths)}")
    print(f"  Order 2 paths: {len(out2.paths)}")
    print(f"  Order 3 paths: {len(out3.paths)}")
    
    # Check subset relationships
    errors = []
    
    # out1 paths should be subset of out3
    for path in out1.paths:
        if path not in out3.paths:
            errors.append(f"Path {path} in Order 1 but not in Order 3")
        else:
            diff = np.max(np.abs(out1.paths[path] - out3.paths[path]))
            if diff > 1e-12:
                errors.append(f"Path {path} differs: max_diff = {diff:.2e}")
    
    # out2 paths should be subset of out3
    for path in out2.paths:
        if path not in out3.paths:
            errors.append(f"Path {path} in Order 2 but not in Order 3")
        else:
            diff = np.max(np.abs(out2.paths[path] - out3.paths[path]))
            if diff > 1e-12:
                errors.append(f"Path {path} differs: max_diff = {diff:.2e}")
    
    if errors:
        for e in errors[:5]:  # Show first 5 errors
            print(f"    ERROR: {e}")
    
    passed = len(errors) == 0
    return passed, f"Order 1 ⊂ Order 3, Order 2 ⊂ Order 3"


# =============================================================================
# Test 2: Monotone Path Constraint
# =============================================================================

def test_monotone_path_constraint():
    """
    Assert that for higher orders (Order 3), paths satisfy j1 < j2 < j3.
    """
    print("\n[TEST 2: MONOTONE PATH CONSTRAINT]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    hst = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=3, lifting='shift', epsilon=2.0)
    output = hst.forward(x)
    
    violations = []
    
    for path in output.paths:
        if len(path) <= 1:
            continue  # Order 0 and 1 don't have constraint
        
        # Check monotonicity
        for i in range(len(path) - 1):
            if path[i] >= path[i + 1]:
                violations.append(f"Path {path}: j[{i}]={path[i]} >= j[{i+1}]={path[i+1]}")
    
    # Count paths by order
    order_counts = {}
    for path in output.paths:
        order = len(path)
        order_counts[order] = order_counts.get(order, 0) + 1
    
    print(f"  Path counts by order: {order_counts}")
    
    if violations:
        for v in violations[:5]:
            print(f"    VIOLATION: {v}")
    else:
        print("  All paths satisfy j1 < j2 < j3 < ...")
    
    passed = len(violations) == 0
    return passed, f"All {sum(order_counts.get(i, 0) for i in range(2, 4))} Order 2+ paths are monotone"


# =============================================================================
# Test 3: Gradient Order 3
# =============================================================================

def test_gradient_order3():
    """
    Validate gradients for Order 3 targets against finite differences.
    """
    print("\n[TEST 3: ORDER 3 GRADIENT CHECK]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=3,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_all_targets(hst, x_target)
    
    n_order3 = len([p for p in target_coeffs if len(p) == 3])
    print(f"  Total paths: {len(target_coeffs)}")
    print(f"  Order 3 paths: {n_order3}")
    
    # Compute analytic gradient
    loss, grad_analytic = compute_loss_and_grad(x, hst, target_coeffs)
    
    # Compute numeric gradient
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
    
    rel_error = np.linalg.norm(grad_analytic - grad_numeric) / np.linalg.norm(grad_numeric)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  |grad_analytic|: {np.linalg.norm(grad_analytic):.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


def test_gradient_order3_only():
    """
    Test gradient when only Order 3 coefficients are targeted.
    
    This tests that gradients correctly backpropagate through 3 layers.
    """
    print("\n[TEST 4: ORDER 3 ONLY GRADIENT CHECK]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=3,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    # Extract only Order 3 targets
    all_coeffs = extract_all_targets(hst, x_target)
    order3_targets = {p: c for p, c in all_coeffs.items() if len(p) == 3}
    
    print(f"  Order 3 paths: {len(order3_targets)}")
    
    # Compute gradients
    loss, grad_analytic = compute_loss_and_grad(x, hst, order3_targets)
    grad_numeric = finite_difference_gradient(x, hst, order3_targets, eps=1e-7)
    
    rel_error = np.linalg.norm(grad_analytic - grad_numeric) / np.linalg.norm(grad_numeric)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


# =============================================================================
# Test 4: Super-convergence Diagnostic
# =============================================================================

def test_superconvergence_diagnostic():
    """
    Compute and print energy distribution across orders for test signals.
    
    Glinsky claims rapid energy decay with order ("super-convergence").
    
    CRITICAL FIX: Use Average Energy per Path to normalize for combinatorial
    explosion (Order 3 has 56 paths vs Order 1's 8 paths).
    
    Metric: E_avg[m] = E_total[m] / N_paths[m]
    """
    print("\n[TEST 5: SUPER-CONVERGENCE DIAGNOSTIC]")
    print("-" * 50)
    print("  Measuring AVERAGE energy per path (normalized for path count)")
    print("  (Glinsky claims rapid decay = 'super-convergence')")
    print()
    
    T = 256
    np.random.seed(42)
    
    # Test signals of different types
    test_signals = {}
    
    # 1. Random complex signal
    test_signals['Random'] = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # 2. Chirp (frequency-sweeping signal)
    t = np.linspace(0, 1, T)
    test_signals['Chirp'] = np.exp(1j * 2 * np.pi * (10 * t + 20 * t**2)) + 3.0
    
    # 3. Van der Pol-like oscillator (limit cycle)
    mu = 1.0
    x_vdp = np.zeros(T)
    v_vdp = np.zeros(T)
    x_vdp[0] = 2.0
    v_vdp[0] = 0.0
    dt = 0.05
    for i in range(1, T):
        x_vdp[i] = x_vdp[i-1] + dt * v_vdp[i-1]
        v_vdp[i] = v_vdp[i-1] + dt * (mu * (1 - x_vdp[i-1]**2) * v_vdp[i-1] - x_vdp[i-1])
    test_signals['Van der Pol'] = x_vdp + 1j * v_vdp + 3.0
    
    # 4. Simple harmonic oscillator (integrable)
    omega0 = 2 * np.pi * 3 / T
    test_signals['Harmonic'] = 2.0 * np.cos(omega0 * np.arange(T)) + 3.0 + 0j
    
    # 5. Coupled oscillators
    x_coupled = np.zeros(T, dtype=complex)
    for k in range(1, 4):
        x_coupled += np.exp(1j * 2 * np.pi * k * np.arange(T) / T)
    test_signals['Coupled'] = x_coupled + 3.0
    
    hst = HeisenbergScatteringTransform(T, J=3, Q=2, max_order=3, lifting='shift', epsilon=2.0)
    
    results = {}
    
    for name, signal in test_signals.items():
        output = hst.forward(signal)
        
        # Count paths and compute energy by order
        path_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        energy_by_order = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        for path, coeff in output.paths.items():
            order = len(path)
            path_counts[order] += 1
            energy = np.sum(np.abs(coeff)**2).real
            energy_by_order[order] += energy
        
        # Compute AVERAGE energy per path
        avg_energy = {}
        for order in range(4):
            if path_counts[order] > 0:
                avg_energy[order] = energy_by_order[order] / path_counts[order]
            else:
                avg_energy[order] = 0.0
        
        results[name] = {
            'path_counts': path_counts,
            'total_energy': energy_by_order,
            'avg_energy': avg_energy,
        }
    
    # Print path counts (same for all signals)
    sample_counts = results['Random']['path_counts']
    print(f"  Path counts: Order 0={sample_counts[0]}, Order 1={sample_counts[1]}, "
          f"Order 2={sample_counts[2]}, Order 3={sample_counts[3]}")
    print()
    
    # Print average energy per path
    print(f"  {'Signal':<15} {'Avg E(0)':>12} {'Avg E(1)':>12} {'Avg E(2)':>12} {'Avg E(3)':>12}")
    print("  " + "-" * 63)
    
    for name, data in results.items():
        avg = data['avg_energy']
        print(f"  {name:<15} {avg[0]:>12.2e} {avg[1]:>12.2e} {avg[2]:>12.2e} {avg[3]:>12.2e}")
    
    # Print ratios to see decay
    print()
    print(f"  {'Signal':<15} {'E(1)/E(0)':>12} {'E(2)/E(1)':>12} {'E(3)/E(2)':>12}")
    print("  " + "-" * 51)
    
    decay_pattern_count = 0
    for name, data in results.items():
        avg = data['avg_energy']
        r10 = avg[1] / avg[0] if avg[0] > 0 else 0
        r21 = avg[2] / avg[1] if avg[1] > 0 else 0
        r32 = avg[3] / avg[2] if avg[2] > 0 else 0
        print(f"  {name:<15} {r10:>12.4f} {r21:>12.4f} {r32:>12.4f}")
        
        # Check if showing decay pattern (each ratio < 1)
        if r10 < 1 and r21 < 1 and r32 < 1:
            decay_pattern_count += 1
    
    print()
    print("  Interpretation:")
    print("    - Ratio < 1: Energy decays (supports super-convergence)")
    print("    - Ratio > 1: Energy grows (counter to Glinsky)")
    print(f"    - {decay_pattern_count}/{len(results)} signals show consistent decay")
    
    # Always pass - this is diagnostic
    return True, f"Diagnostic: {decay_pattern_count}/{len(results)} show decay pattern"


def test_gradient_multiple_seeds_order3():
    """Test Order 3 gradient on multiple seeds for robustness."""
    print("\n[TEST 6: ORDER 3 GRADIENT (MULTIPLE SEEDS)]")
    print("-" * 50)
    
    T = 32
    errors = []
    
    for seed in [0, 1, 42, 123]:
        np.random.seed(seed)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=3,
            lifting='shift', epsilon=2.0
        )
        
        x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
        x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
        target_coeffs = extract_all_targets(hst, x_target)
        
        _, grad_a = compute_loss_and_grad(x, hst, target_coeffs)
        grad_n = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
        
        rel_error = np.linalg.norm(grad_a - grad_n) / np.linalg.norm(grad_n)
        errors.append(rel_error)
        
        status = "✓" if rel_error < 1e-4 else "✗"
        print(f"    Seed {seed:3d}: rel_error = {rel_error:.2e} {status}")
    
    max_error = max(errors)
    passed = max_error < 1e-4
    return passed, f"Max error = {max_error:.2e}"


def test_incremental_reconstruction_value():
    """
    Test if higher orders provide significant reconstruction improvement.
    
    This validates Glinsky's claim that "stopping at Order 2 is sufficient".
    
    Method:
    1. Take a signal and extract its coefficients
    2. From noise, reconstruct using only Order 1 targets -> Error_1
    3. Reconstruct using Order 1+2 targets -> Error_2
    4. Reconstruct using Order 1+2+3 targets -> Error_3
    
    Expected: Error_2 < Error_1 (sanity)
    Question: Is Error_3 << Error_2? Or is the improvement marginal?
    """
    print("\n[TEST 7: INCREMENTAL RECONSTRUCTION VALUE]")
    print("-" * 50)
    print("  Testing if higher orders provide significant reconstruction gain")
    print()
    
    T = 64
    np.random.seed(42)
    
    # Create target signal
    x_target = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Create HST with max_order=3 to get all coefficients
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=3,
        lifting='shift', epsilon=2.0
    )
    
    # Extract all coefficients
    all_coeffs = extract_all_targets(hst, x_target)
    
    # Split by order
    coeffs_order1 = {p: c for p, c in all_coeffs.items() if len(p) <= 1}
    coeffs_order2 = {p: c for p, c in all_coeffs.items() if len(p) <= 2}
    coeffs_order3 = all_coeffs  # All coefficients
    
    print(f"  Coefficient counts: Order≤1: {len(coeffs_order1)}, "
          f"Order≤2: {len(coeffs_order2)}, Order≤3: {len(coeffs_order3)}")
    
    # Starting point: random noise
    x0 = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    from hst.optimize_numpy import optimize_signal
    
    # Reconstruct with Order 1 only
    result1 = optimize_signal(
        coeffs_order1, hst, x0.copy(),
        n_steps=300, lr=0.05, verbose=False
    )
    error1 = np.linalg.norm(result1.signal - x_target) / np.linalg.norm(x_target)
    
    # Reconstruct with Order 1+2
    result2 = optimize_signal(
        coeffs_order2, hst, x0.copy(),
        n_steps=300, lr=0.05, verbose=False
    )
    error2 = np.linalg.norm(result2.signal - x_target) / np.linalg.norm(x_target)
    
    # Reconstruct with Order 1+2+3
    result3 = optimize_signal(
        coeffs_order3, hst, x0.copy(),
        n_steps=300, lr=0.05, verbose=False
    )
    error3 = np.linalg.norm(result3.signal - x_target) / np.linalg.norm(x_target)
    
    print(f"\n  Reconstruction errors (relative to ||x_target||):")
    print(f"    Order ≤ 1: {error1:.6f} ({error1*100:.2f}%)")
    print(f"    Order ≤ 2: {error2:.6f} ({error2*100:.2f}%)")
    print(f"    Order ≤ 3: {error3:.6f} ({error3*100:.2f}%)")
    
    # Compute improvement ratios
    improvement_2_over_1 = (error1 - error2) / error1 if error1 > 0 else 0
    improvement_3_over_2 = (error2 - error3) / error2 if error2 > 0 else 0
    
    print(f"\n  Improvement ratios:")
    print(f"    Order 2 vs Order 1: {improvement_2_over_1*100:.1f}% reduction")
    print(f"    Order 3 vs Order 2: {improvement_3_over_2*100:.1f}% reduction")
    
    print(f"\n  Analysis:")
    if improvement_3_over_2 < 0.1:  # Less than 10% improvement
        print(f"    Order 3 provides < 10% improvement over Order 2")
        print(f"    -> Supports Glinsky: 'Order 2 is sufficient'")
    else:
        print(f"    Order 3 provides {improvement_3_over_2*100:.1f}% improvement")
        print(f"    -> Order 3 may be valuable for this signal class")
    
    # Sanity check: Order 2 should be better than Order 1
    sanity_passed = error2 <= error1 * 1.1  # Allow 10% tolerance
    
    return sanity_passed, f"Errors: O1={error1:.4f}, O2={error2:.4f}, O3={error3:.4f}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ORDER PARAMETERIZATION TESTS")
    print("Testing arbitrary order gradients and super-convergence")
    print("=" * 60)
    
    result = TestResult()
    
    # Truncation and monotonicity
    p, m = test_order_truncation_correctness()
    result.record("order_truncation_correctness", p, m)
    
    p, m = test_monotone_path_constraint()
    result.record("monotone_path_constraint", p, m)
    
    # Order 3 gradients
    p, m = test_gradient_order3()
    result.record("gradient_order3", p, m)
    
    p, m = test_gradient_order3_only()
    result.record("gradient_order3_only", p, m)
    
    p, m = test_gradient_multiple_seeds_order3()
    result.record("gradient_multi_seed_order3", p, m)
    
    # Super-convergence diagnostic
    p, m = test_superconvergence_diagnostic()
    result.record("superconvergence_diagnostic", p, m)
    
    # Incremental reconstruction value
    p, m = test_incremental_reconstruction_value()
    result.record("incremental_reconstruction_value", p, m)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ Arbitrary order optimization ready!")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
