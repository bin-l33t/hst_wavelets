#!/usr/bin/env python3
"""
Order 2 Optimization Tests

Tests the analytic gradient for the full scattering tree (Order 1 + Order 2)
and verifies deep reconstruction via optimization.
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
    extract_order2_targets,
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
# Test 1: Order 2 Gradient Check
# =============================================================================

def test_grad_check_order2_tiny():
    """
    Compare analytic gradient vs finite difference for full Order 2 tree.
    """
    print("\n[TEST 1: ORDER 2 GRADIENT CHECK (T=32)]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Get ALL targets (Order 0, 1, 2)
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_all_targets(hst, x_target)
    
    n_order0 = len([p for p in target_coeffs if len(p) == 0])
    n_order1 = len([p for p in target_coeffs if len(p) == 1])
    n_order2 = len([p for p in target_coeffs if len(p) == 2])
    
    print(f"  Signal: T={T}, min|x|={np.min(np.abs(x)):.2f}")
    print(f"  Targets: {n_order0} Order-0, {n_order1} Order-1, {n_order2} Order-2")
    
    # Compute analytic gradient
    loss, grad_analytic = compute_loss_and_grad_order2(x, hst, target_coeffs)
    
    # Compute numeric gradient
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
    
    diff = grad_analytic - grad_numeric
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  |grad_analytic|: {np.linalg.norm(grad_analytic):.6e}")
    print(f"  |grad_numeric|: {np.linalg.norm(grad_numeric):.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


def test_grad_check_order2_only():
    """
    Test gradient when only Order 2 coefficients are targeted.
    """
    print("\n[TEST 2: ORDER 2 ONLY GRADIENT CHECK]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    # Only Order 2 targets
    target_coeffs = extract_order2_targets(hst, x_target)
    
    print(f"  Order 2 paths: {len(target_coeffs)}")
    
    # Compute gradients
    loss, grad_analytic = compute_loss_and_grad_order2(x, hst, target_coeffs)
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
    
    diff = grad_analytic - grad_numeric
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


# =============================================================================
# Test 2: Reduction Consistency
# =============================================================================

def test_reduction_consistency():
    """
    When Order 2 weights are zero, gradient should match Order 1 computation.
    
    Note: We need to use the same targets for a fair comparison.
    """
    print("\n[TEST 3: REDUCTION CONSISTENCY]")
    print("-" * 50)
    print("  If Order 2 weight = 0, Order 2 gradients should not contribute")
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    # Use only Order 1 targets (no Order 0, no Order 2)
    order1_targets = extract_order1_targets(hst, x_target)
    
    # Compute with Order 2 function but only Order 1 targets
    # With w2=0, there's nothing to compute for Order 2 anyway
    weights_w2_zero = {0: 0.0, 1: 1.0, 2: 0.0}
    loss_order2_func, grad_order2_func = compute_loss_and_grad_order2(
        x, hst, order1_targets, weights_w2_zero
    )
    
    # Compute with Order 1 function directly
    weights_order1 = {0: 0.0, 1: 1.0}
    loss_order1_func, grad_order1_func = compute_loss_and_grad_order1(
        x, hst, order1_targets, weights_order1
    )
    
    # They should match exactly since both compute only Order 1 loss
    loss_diff = abs(loss_order2_func - loss_order1_func)
    grad_diff = np.linalg.norm(grad_order2_func - grad_order1_func)
    
    print(f"  Order 1 targets only: {len(order1_targets)} paths")
    print(f"  Loss (via Order 2 func): {loss_order2_func:.6e}")
    print(f"  Loss (via Order 1 func): {loss_order1_func:.6e}")
    print(f"  Loss difference: {loss_diff:.2e}")
    print(f"  Gradient difference: {grad_diff:.2e}")
    
    passed = loss_diff < 1e-10 and grad_diff < 1e-10
    return passed, f"Loss diff = {loss_diff:.2e}, Grad diff = {grad_diff:.2e}"


# =============================================================================
# Test 3: Deep Reconstruction
# =============================================================================

def test_deep_reconstruction():
    """
    Demonstrate deep invertibility via optimization.
    
    Start from noise and reconstruct a signal by matching its
    Order 1 and Order 2 coefficients.
    """
    print("\n[TEST 4: DEEP RECONSTRUCTION VIA OPTIMIZATION]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    # Target signal
    x_target = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    target_coeffs = extract_all_targets(hst, x_target)
    
    n_order1 = len([p for p in target_coeffs if len(p) == 1])
    n_order2 = len([p for p in target_coeffs if len(p) == 2])
    print(f"  Target: {n_order1} Order-1 + {n_order2} Order-2 paths")
    
    # Start from random noise (not perturbed target!)
    x0 = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Run optimization
    result = optimize_signal(
        target_coeffs, hst, x0,
        n_steps=500, lr=0.05, momentum=0.9,
        verbose=False
    )
    
    print(f"  Initial loss: {result.loss_history[0]:.6e}")
    print(f"  Final loss: {result.final_loss:.6e}")
    
    # Compute reconstruction error
    # Note: We can only expect coefficient match, not signal match
    # (HST is not 1-to-1 from Order 2 alone)
    final_coeffs = extract_all_targets(hst, result.signal)
    
    coeff_error = 0.0
    for path in target_coeffs:
        if path in final_coeffs:
            diff = final_coeffs[path] - target_coeffs[path]
            coeff_error += np.sum(np.abs(diff)**2)
    coeff_error = np.sqrt(coeff_error)
    
    # Signal reconstruction error (just for reference)
    signal_error = np.linalg.norm(result.signal - x_target) / np.linalg.norm(x_target)
    
    print(f"  Coefficient error: {coeff_error:.6e}")
    print(f"  Signal error (reference): {signal_error:.4f}")
    
    # Pass if final loss < 1e-3 (demonstrating we can match coefficients)
    passed = result.final_loss < 1e-3
    return passed, f"Final loss = {result.final_loss:.2e}"


def test_deep_reconstruction_order2_weighted():
    """
    Reconstruction with higher weight on Order 2.
    
    Tests that Order 2 gradients properly propagate through the tree.
    """
    print("\n[TEST 5: ORDER 2 WEIGHTED RECONSTRUCTION]")
    print("-" * 50)
    
    T = 64
    np.random.seed(123)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=2,
        lifting='shift', epsilon=2.0
    )
    
    # Target signal
    x_target = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Weight Order 2 more heavily
    weights = {0: 0.1, 1: 0.5, 2: 2.0}
    
    # Start from noise
    x0 = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Compute initial Order 2 error
    initial_coeffs = extract_all_targets(hst, x0)
    initial_order2_error = sum(
        np.sum(np.abs(initial_coeffs[p] - target_coeffs[p])**2)
        for p in target_coeffs if len(p) == 2
    )
    
    # Run optimization
    result = optimize_signal(
        target_coeffs, hst, x0,
        n_steps=300, lr=0.05, momentum=0.9,
        weights=weights, verbose=False
    )
    
    # Compute final Order 2 error
    final_coeffs = extract_all_targets(hst, result.signal)
    final_order2_error = sum(
        np.sum(np.abs(final_coeffs[p] - target_coeffs[p])**2)
        for p in target_coeffs if len(p) == 2
    )
    
    print(f"  Initial Order 2 error: {initial_order2_error:.4e}")
    print(f"  Final Order 2 error: {final_order2_error:.4e}")
    print(f"  Reduction: {(1 - final_order2_error/initial_order2_error)*100:.1f}%")
    
    # Should reduce Order 2 error significantly
    passed = final_order2_error < initial_order2_error * 0.1
    return passed, f"Order 2 error reduced to {final_order2_error/initial_order2_error*100:.1f}%"


# =============================================================================
# Test 4: Multiple Seeds
# =============================================================================

def test_grad_check_order2_multiple_seeds():
    """Test Order 2 gradient on multiple random seeds."""
    print("\n[TEST 6: ORDER 2 GRADIENT (MULTIPLE SEEDS)]")
    print("-" * 50)
    
    T = 32
    errors = []
    
    for seed in [0, 1, 2, 42, 123]:
        np.random.seed(seed)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=2,
            lifting='shift', epsilon=2.0
        )
        
        x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
        x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
        target_coeffs = extract_all_targets(hst, x_target)
        
        _, grad_analytic = compute_loss_and_grad_order2(x, hst, target_coeffs)
        grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
        
        diff = grad_analytic - grad_numeric
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
        errors.append(rel_error)
        
        status = "✓" if rel_error < 1e-4 else "✗"
        print(f"    Seed {seed:3d}: rel_error = {rel_error:.2e} {status}")
    
    max_error = max(errors)
    passed = max_error < 1e-4
    return passed, f"Max relative error = {max_error:.2e}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ORDER 2 OPTIMIZATION TESTS")
    print("Testing deep gradient computation and reconstruction")
    print("=" * 60)
    
    result = TestResult()
    
    # Gradient checks
    p, m = test_grad_check_order2_tiny()
    result.record("grad_check_order2_tiny", p, m)
    
    p, m = test_grad_check_order2_only()
    result.record("grad_check_order2_only", p, m)
    
    p, m = test_reduction_consistency()
    result.record("reduction_consistency", p, m)
    
    # Deep reconstruction
    p, m = test_deep_reconstruction()
    result.record("deep_reconstruction", p, m)
    
    p, m = test_deep_reconstruction_order2_weighted()
    result.record("deep_reconstruction_order2_weighted", p, m)
    
    # Robustness
    p, m = test_grad_check_order2_multiple_seeds()
    result.record("grad_check_multi_seed", p, m)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ All Order 2 optimization contracts satisfied!")
        print("  Deep gradients are correct and reconstruction works.")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
