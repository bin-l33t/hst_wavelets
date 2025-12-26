#!/usr/bin/env python3
"""
Phase Wrapping / Branch Cut Optimization Tests

Tests the phase-robust loss function that handles branch cuts in the
complex logarithm by using circular distance for the phase component.

Key insight:
- Standard L2: |W - W_t|² treats θ=π and θ=-π as far apart
- Phase-robust: |exp(iθ) - exp(iθ_t)|² = 2(1 - cos(θ - θ_t)) is periodic

This is critical for physical systems where coefficients wind around the origin.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    compute_loss_and_grad_order1,
    compute_loss_and_grad_order2,
    finite_difference_gradient,
    optimize_signal,
    extract_order1_targets,
    extract_all_targets,
    _compute_phase_robust_loss_and_grad,
    _compute_l2_loss_and_grad,
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
# Test 1: Gradient Check for Phase-Robust Loss
# =============================================================================

def test_gradient_phase_robust():
    """Verify analytic gradient of phase-robust loss against finite differences."""
    print("\n[TEST 1: PHASE-ROBUST GRADIENT CHECK]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Compute analytic gradient
    loss, grad_analytic = compute_loss_and_grad_order2(
        x, hst, target_coeffs, loss_type='phase_robust'
    )
    
    # Compute numeric gradient
    grad_numeric = finite_difference_gradient(
        x, hst, target_coeffs, loss_type='phase_robust'
    )
    
    rel_error = np.linalg.norm(grad_analytic - grad_numeric) / np.linalg.norm(grad_numeric)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  |grad_analytic|: {np.linalg.norm(grad_analytic):.6e}")
    print(f"  |grad_numeric|: {np.linalg.norm(grad_numeric):.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


def test_gradient_phase_robust_multi_seed():
    """Test phase-robust gradient on multiple seeds."""
    print("\n[TEST 2: PHASE-ROBUST GRADIENT (MULTIPLE SEEDS)]")
    print("-" * 50)
    
    T = 32
    errors = []
    
    for seed in [0, 1, 42, 123, 456]:
        np.random.seed(seed)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=2,
            lifting='shift', epsilon=2.0
        )
        
        x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
        x_target = x + 0.4 * (np.random.randn(T) + 1j * np.random.randn(T))
        target_coeffs = extract_all_targets(hst, x_target)
        
        _, grad_a = compute_loss_and_grad_order2(x, hst, target_coeffs, loss_type='phase_robust')
        grad_n = finite_difference_gradient(x, hst, target_coeffs, loss_type='phase_robust')
        
        rel_error = np.linalg.norm(grad_a - grad_n) / np.linalg.norm(grad_n)
        errors.append(rel_error)
        
        status = "✓" if rel_error < 1e-4 else "✗"
        print(f"    Seed {seed:3d}: rel_error = {rel_error:.2e} {status}")
    
    max_error = max(errors)
    passed = max_error < 1e-4
    return passed, f"Max error = {max_error:.2e}"


# =============================================================================
# Test 2: Forced Wrap Crossing
# =============================================================================

def test_forced_wrap_crossing():
    """
    Test optimization when phase must cross the branch cut (θ = ±π).
    
    Setup:
    - Target: coefficient with phase ≈ +π (just below)
    - Init: coefficient with phase ≈ -π (just above)
    
    Standard L2 sees this as a large distance (~2π).
    Phase-robust sees this as a small distance (~0).
    """
    print("\n[TEST 3: FORCED WRAP CROSSING]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='shift', epsilon=2.0
    )
    
    # Create a target signal with phase near +π (0.95π)
    phase_target = 0.95 * np.pi
    x_target = 3.0 * np.exp(1j * phase_target) * np.ones(T)
    x_target += 0.1 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_order1_targets(hst, x_target)
    
    # Initial signal with phase near -π (-0.95π) - on opposite side of branch cut
    phase_init = -0.95 * np.pi
    x0 = 3.0 * np.exp(1j * phase_init) * np.ones(T)
    x0 += 0.1 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    print(f"  Target phase: {phase_target/np.pi:.2f}π")
    print(f"  Initial phase: {phase_init/np.pi:.2f}π")
    print(f"  Phase difference (direct): {(phase_target - phase_init)/np.pi:.2f}π")
    print(f"  Phase difference (wrapped): {np.abs(np.angle(np.exp(1j*(phase_target - phase_init))))/np.pi:.2f}π")
    
    # Optimize with standard L2
    result_l2 = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=100, lr=0.05, loss_type='l2', verbose=False
    )
    
    # Optimize with phase-robust
    result_pr = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=100, lr=0.05, loss_type='phase_robust', verbose=False
    )
    
    print(f"\n  L2 loss:")
    print(f"    Initial: {result_l2.loss_history[0]:.4e}")
    print(f"    Final: {result_l2.final_loss:.4e}")
    print(f"    Reduction: {(1 - result_l2.final_loss/result_l2.loss_history[0])*100:.1f}%")
    
    print(f"\n  Phase-robust loss:")
    print(f"    Initial: {result_pr.loss_history[0]:.4e}")
    print(f"    Final: {result_pr.final_loss:.4e}")
    print(f"    Reduction: {(1 - result_pr.final_loss/result_pr.loss_history[0])*100:.1f}%")
    
    # Phase-robust should reduce more or converge better near branch cut
    # Check the actual phase match
    final_coeffs_l2 = extract_order1_targets(hst, result_l2.signal)
    final_coeffs_pr = extract_order1_targets(hst, result_pr.signal)
    
    # Compute phase differences (using circular distance)
    phase_errors_l2 = []
    phase_errors_pr = []
    for path in target_coeffs:
        target = target_coeffs[path]
        
        final_l2 = final_coeffs_l2[path]
        phase_diff_l2 = np.abs(np.sin(final_l2.real - target.real))  # Circular distance
        phase_errors_l2.append(np.mean(phase_diff_l2))
        
        final_pr = final_coeffs_pr[path]
        phase_diff_pr = np.abs(np.sin(final_pr.real - target.real))
        phase_errors_pr.append(np.mean(phase_diff_pr))
    
    mean_phase_error_l2 = np.mean(phase_errors_l2)
    mean_phase_error_pr = np.mean(phase_errors_pr)
    
    print(f"\n  Mean phase error (circular):")
    print(f"    L2: {mean_phase_error_l2:.4f}")
    print(f"    Phase-robust: {mean_phase_error_pr:.4f}")
    
    # Phase-robust should have lower or equal phase error
    passed = mean_phase_error_pr <= mean_phase_error_l2 * 1.1  # Allow 10% tolerance
    return passed, f"Phase error L2={mean_phase_error_l2:.4f}, PR={mean_phase_error_pr:.4f}"


# =============================================================================
# Test 3: Circular Distance Properties
# =============================================================================

def test_circular_distance_properties():
    """
    Verify the phase-robust loss has correct circular distance properties.
    
    Key properties:
    1. |exp(iπ) - exp(i(-π))|² = 0 (same point)
    2. |exp(i0) - exp(iπ)|² = 4 (opposite points)
    3. Gradient is 0 at θ = θ_target
    """
    print("\n[TEST 4: CIRCULAR DISTANCE PROPERTIES]")
    print("-" * 50)
    
    # Test 1: θ = π and θ = -π should have zero circular distance
    W1 = np.array([np.pi + 0j])  # Phase = π
    W2 = np.array([-np.pi + 0j])  # Phase = -π
    
    loss, grad = _compute_phase_robust_loss_and_grad(W1, W2, weight=1.0, phase_lambda=1.0)
    
    print(f"  θ=π vs θ=-π:")
    print(f"    Loss: {loss:.6e} (should be ~0)")
    print(f"    |grad|: {np.linalg.norm(grad):.6e}")
    
    test1_passed = loss < 1e-10
    
    # Test 2: θ = 0 and θ = π should have max circular distance
    W1 = np.array([0.0 + 0j])  # Phase = 0
    W2 = np.array([np.pi + 0j])  # Phase = π
    
    loss, grad = _compute_phase_robust_loss_and_grad(W1, W2, weight=1.0, phase_lambda=1.0)
    
    print(f"\n  θ=0 vs θ=π:")
    print(f"    Loss: {loss:.6e} (should be 4)")
    
    test2_passed = abs(loss - 4.0) < 1e-10
    
    # Test 3: Gradient should be 0 at θ = θ_target
    W = np.array([1.5 + 0.5j])  # Some arbitrary point
    loss, grad = _compute_phase_robust_loss_and_grad(W, W, weight=1.0, phase_lambda=1.0)
    
    print(f"\n  θ=θ_target:")
    print(f"    Loss: {loss:.6e} (should be 0)")
    print(f"    |grad|: {np.linalg.norm(grad):.6e} (should be 0)")
    
    test3_passed = loss < 1e-10 and np.linalg.norm(grad) < 1e-10
    
    passed = test1_passed and test2_passed and test3_passed
    return passed, f"All circular distance properties verified"


# =============================================================================
# Test 4: High Winding Number
# =============================================================================

def test_high_winding_convergence():
    """
    Test optimization on a signal that winds multiple times around origin.
    
    Phase-robust loss should handle this without getting stuck.
    """
    print("\n[TEST 5: HIGH WINDING CONVERGENCE]")
    print("-" * 50)
    
    T = 128
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='shift', epsilon=2.0
    )
    
    # Create a signal that winds around the origin
    t = np.linspace(0, 1, T)
    n_winds = 3
    target_phase = 2 * np.pi * n_winds * t
    x_target = 3.0 * np.exp(1j * target_phase)
    x_target += 0.2 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    target_coeffs = extract_order1_targets(hst, x_target)
    
    # Initial: different winding
    init_phase = 2 * np.pi * (n_winds + 0.5) * t  # Half-wind offset
    x0 = 3.0 * np.exp(1j * init_phase)
    x0 += 0.2 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    print(f"  Target winding: {n_winds}")
    print(f"  Initial offset: 0.5 winds")
    
    # Optimize with both methods
    result_l2 = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=200, lr=0.03, loss_type='l2', verbose=False
    )
    
    result_pr = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=200, lr=0.03, loss_type='phase_robust', verbose=False
    )
    
    reduction_l2 = 1 - result_l2.final_loss / result_l2.loss_history[0]
    reduction_pr = 1 - result_pr.final_loss / result_pr.loss_history[0]
    
    print(f"\n  L2: {reduction_l2*100:.1f}% reduction")
    print(f"  Phase-robust: {reduction_pr*100:.1f}% reduction")
    
    # Both should reduce loss significantly
    passed = reduction_pr > 0.5  # At least 50% reduction
    return passed, f"L2={reduction_l2*100:.1f}%, PR={reduction_pr*100:.1f}%"


# =============================================================================
# Test 5: Backward Compatibility
# =============================================================================

def test_l2_backward_compatibility():
    """Ensure L2 loss still works after refactoring."""
    print("\n[TEST 6: L2 BACKWARD COMPATIBILITY]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Test L2 gradient
    loss, grad_analytic = compute_loss_and_grad_order2(x, hst, target_coeffs, loss_type='l2')
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, loss_type='l2')
    
    rel_error = np.linalg.norm(grad_analytic - grad_numeric) / np.linalg.norm(grad_numeric)
    
    print(f"  L2 gradient relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE WRAPPING / BRANCH CUT OPTIMIZATION TESTS")
    print("Testing phase-robust loss for signals winding around origin")
    print("=" * 60)
    
    result = TestResult()
    
    # Gradient checks
    p, m = test_gradient_phase_robust()
    result.record("gradient_phase_robust", p, m)
    
    p, m = test_gradient_phase_robust_multi_seed()
    result.record("gradient_multi_seed", p, m)
    
    # Wrap crossing
    p, m = test_forced_wrap_crossing()
    result.record("forced_wrap_crossing", p, m)
    
    # Circular distance
    p, m = test_circular_distance_properties()
    result.record("circular_distance_properties", p, m)
    
    # High winding
    p, m = test_high_winding_convergence()
    result.record("high_winding_convergence", p, m)
    
    # Backward compatibility
    p, m = test_l2_backward_compatibility()
    result.record("l2_backward_compatibility", p, m)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ Phase-robust optimization ready for physical systems!")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
