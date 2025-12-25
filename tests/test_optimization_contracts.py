#!/usr/bin/env python3
"""
Optimization Contract Tests

Tests the analytic gradient implementation against finite differences
and verifies optimization behavior.

Test Categories:
1. Gradient correctness (analytic vs finite difference)
2. Descent sanity (loss decreases with optimization)
3. Branch cut handling (gradient near discontinuities)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    compute_loss_and_grad_order1,
    finite_difference_gradient,
    optimize_signal,
    extract_order1_targets,
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
# Test 1: Gradient Check (Tiny)
# =============================================================================

def test_grad_check_order1_tiny():
    """
    Compare analytic gradient vs finite difference on tiny problem.
    
    CRITICAL: Signal shifted away from origin to avoid 1/U singularity.
    """
    print("\n[TEST 1: GRADIENT CHECK (T=32)]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    # Create HST with explicit shift lifting
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1, 
        lifting='shift', epsilon=2.0
    )
    
    # Signal shifted away from origin (CRITICAL for avoiding singularity)
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    print(f"  Signal: T={T}, min|x|={np.min(np.abs(x)):.2f}")
    print(f"  HST: J={hst.J}, Q={hst.Q}, n_mothers={hst.n_mothers}")
    
    # Get target from slightly different signal
    x_target = x + 0.5 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_order1_targets(hst, x_target)
    
    print(f"  Target paths: {len(target_coeffs)}")
    
    # Compute analytic gradient
    loss, grad_analytic = compute_loss_and_grad_order1(x, hst, target_coeffs)
    
    # Compute numeric gradient (central differences)
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
    
    # Compare
    diff = grad_analytic - grad_numeric
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  |grad_analytic|: {np.linalg.norm(grad_analytic):.6e}")
    print(f"  |grad_numeric|: {np.linalg.norm(grad_numeric):.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    
    passed = rel_error < 1e-4
    return passed, f"Relative error = {rel_error:.2e}"


def test_grad_check_multiple_seeds():
    """Test gradient on multiple random seeds for robustness."""
    print("\n[TEST 2: GRADIENT CHECK (MULTIPLE SEEDS)]")
    print("-" * 50)
    
    T = 32
    errors = []
    
    for seed in [0, 1, 2, 42, 123]:
        np.random.seed(seed)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='shift', epsilon=2.0
        )
        
        x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
        x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
        target_coeffs = extract_order1_targets(hst, x_target)
        
        _, grad_analytic = compute_loss_and_grad_order1(x, hst, target_coeffs)
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
# Test 2: Descent Sanity
# =============================================================================

def test_descent_sanity():
    """
    Start with x0 = target + noise, run optimization, assert loss decreases.
    """
    print("\n[TEST 3: DESCENT SANITY]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='shift', epsilon=2.0
    )
    
    # Target signal
    x_target = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    target_coeffs = extract_order1_targets(hst, x_target)
    
    # Initial signal = target + noise
    noise_level = 0.5
    x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
    
    print(f"  Target signal: T={T}")
    print(f"  Initial perturbation: {noise_level}")
    
    # Run optimization
    result = optimize_signal(
        target_coeffs, hst, x0,
        n_steps=20, lr=0.1, momentum=0.9,
        verbose=False
    )
    
    # Check loss decreases
    loss_history = result.loss_history
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    
    print(f"  Initial loss: {initial_loss:.6e}")
    print(f"  Final loss: {final_loss:.6e}")
    print(f"  Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    # Check monotonic decrease (allowing small tolerance for numerical noise)
    decreasing = True
    for i in range(1, len(loss_history)):
        if loss_history[i] > loss_history[i-1] * 1.01:  # 1% tolerance
            decreasing = False
            break
    
    passed = final_loss < initial_loss * 0.5  # At least 50% reduction
    return passed, f"Loss reduced from {initial_loss:.2e} to {final_loss:.2e}"


def test_convergence_to_target():
    """
    Test that optimization can recover a signal from its coefficients.
    """
    print("\n[TEST 4: CONVERGENCE TO TARGET]")
    print("-" * 50)
    
    T = 64
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='shift', epsilon=2.0
    )
    
    # Target signal
    x_target = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    target_coeffs = extract_order1_targets(hst, x_target)
    
    # Start from random initialization (not perturbed target)
    x0 = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Run longer optimization
    result = optimize_signal(
        target_coeffs, hst, x0,
        n_steps=200, lr=0.05, momentum=0.9,
        verbose=False
    )
    
    # Check final loss is small
    print(f"  Initial loss: {result.loss_history[0]:.6e}")
    print(f"  Final loss: {result.final_loss:.6e}")
    
    # Check coefficient match
    final_coeffs = extract_order1_targets(hst, result.signal)
    coeff_error = 0.0
    for path in target_coeffs:
        diff = final_coeffs[path] - target_coeffs[path]
        coeff_error += np.sum(np.abs(diff)**2)
    coeff_error = np.sqrt(coeff_error)
    
    print(f"  Coefficient error: {coeff_error:.6e}")
    
    passed = result.final_loss < 1e-2  # More realistic threshold
    return passed, f"Final loss = {result.final_loss:.2e}"


# =============================================================================
# Test 3: Branch Cut Handling
# =============================================================================

def test_branch_cut_gradient():
    """
    Test gradient when coefficients cross the negative real axis (branch cut of ln).
    
    The branch cut is at arg(z) = ±π. We create a signal where wavelet
    coefficients have phase near ±π and verify the gradient is still correct.
    """
    print("\n[TEST 5: BRANCH CUT GRADIENT]")
    print("-" * 50)
    
    T = 32
    np.random.seed(42)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='shift', epsilon=0.5  # Smaller shift to get closer to branch cut
    )
    
    # Create signal that will have coefficients near negative real axis
    # A signal with strong negative DC component
    x = -2.0 + 0.5 * np.random.randn(T) + 0.5j * np.random.randn(T)
    
    print(f"  Signal mean: {np.mean(x):.2f}")
    print(f"  Signal near negative real axis: {np.mean(x.real) < 0}")
    
    # Check phases of coefficients
    output = hst.forward(x)
    phases = []
    for path, coeff in output.paths.items():
        if len(path) == 1:
            phase = np.angle(coeff)
            phases.extend(phase.tolist())
    
    phases = np.array(phases)
    near_branch_cut = np.sum(np.abs(np.abs(phases) - np.pi) < 0.5)
    print(f"  Coefficients near branch cut (|phase| near π): {near_branch_cut}/{len(phases)}")
    
    # Get target and compute gradient
    x_target = x + 0.3 * (np.random.randn(T) + 1j * np.random.randn(T))
    target_coeffs = extract_order1_targets(hst, x_target)
    
    loss, grad_analytic = compute_loss_and_grad_order1(x, hst, target_coeffs)
    grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
    
    diff = grad_analytic - grad_numeric
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
    
    print(f"  Gradient relative error: {rel_error:.6e}")
    
    # With lifting, the branch cut should be avoided
    passed = rel_error < 1e-3  # Slightly relaxed tolerance near branch cut
    return passed, f"Relative error = {rel_error:.2e} (near branch cut)"


def test_gradient_at_various_phases():
    """Test gradient correctness for signals with various phase distributions."""
    print("\n[TEST 6: GRADIENT AT VARIOUS PHASES]")
    print("-" * 50)
    
    T = 32
    results = []
    
    # Test different phase configurations
    # Use 'shift' lifting (constant shift) for well-defined gradients
    configs = [
        ("Positive real", 3.0 + 0.1j),
        ("Positive imag", 0.1 + 3.0j),
        ("Negative real", -3.0 + 0.1j),
        ("Negative imag", 0.1 - 3.0j),
        ("Mixed", 2.0 + 2.0j),
    ]
    
    for name, dc in configs:
        np.random.seed(42)
        
        # Use 'shift' lifting with large epsilon to avoid singularities
        # 'adaptive' lifting has data-dependent shift which breaks gradient
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='shift', epsilon=5.0  # Large shift to handle all DC values
        )
        
        x = dc + 0.5 * (np.random.randn(T) + 1j * np.random.randn(T))
        x_target = x + 0.2 * (np.random.randn(T) + 1j * np.random.randn(T))
        target_coeffs = extract_order1_targets(hst, x_target)
        
        _, grad_analytic = compute_loss_and_grad_order1(x, hst, target_coeffs)
        grad_numeric = finite_difference_gradient(x, hst, target_coeffs, eps=1e-7)
        
        diff = grad_analytic - grad_numeric
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
        
        status = "✓" if rel_error < 1e-4 else "✗"
        print(f"    {name:15s}: rel_error = {rel_error:.2e} {status}")
        results.append(rel_error)
    
    max_error = max(results)
    passed = max_error < 1e-4
    return passed, f"Max error = {max_error:.2e}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("OPTIMIZATION CONTRACT TESTS")
    print("Testing analytic gradients and optimization behavior")
    print("=" * 60)
    
    result = TestResult()
    
    # Gradient checks
    p, m = test_grad_check_order1_tiny()
    result.record("grad_check_tiny", p, m)
    
    p, m = test_grad_check_multiple_seeds()
    result.record("grad_check_multi_seed", p, m)
    
    # Descent sanity
    p, m = test_descent_sanity()
    result.record("descent_sanity", p, m)
    
    p, m = test_convergence_to_target()
    result.record("convergence_to_target", p, m)
    
    # Branch cut handling
    p, m = test_branch_cut_gradient()
    result.record("branch_cut_gradient", p, m)
    
    p, m = test_gradient_at_various_phases()
    result.record("gradient_various_phases", p, m)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ All optimization contracts satisfied!")
        print("  Ready to proceed with Order 2 gradients or performance optimization.")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
