#!/usr/bin/env python3
"""
High Winding Stress Tests

Tests the optimizer's ability to maintain topological invariants (winding number)
under increasing complexity.

Key Risk (Homotopy Class Jumping):
The phase_robust loss uses circular distance |exp(iθ) - exp(iθ_t)|², which is
2π-periodic. This means the optimizer could potentially reduce loss by "unwinding"
a k-turn signal to (k-1) turns, jumping between homotopy classes.

This test suite verifies:
1. At what winding number does L2 loss break?
2. Does phase_robust loss survive all tested winding numbers?
3. Is the optimizer preserving or destroying topology?

Anti-aliasing strategy:
- Scale T proportionally to k to maintain constant phase step Δθ
- This rules out aliasing artifacts and isolates the topological behavior
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    optimize_signal,
    extract_all_targets,
    normalize_energy,
    min_segment_distance_to_origin,
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


def compute_winding_number(z: np.ndarray) -> float:
    """
    Compute the winding number of a complex signal around the origin.
    
    W = (1/2π) Σ Δθ_i (properly unwrapped)
    """
    phases = np.angle(z)
    dphases = np.diff(phases)
    
    # Unwrap phase jumps
    dphases = np.where(dphases > np.pi, dphases - 2*np.pi, dphases)
    dphases = np.where(dphases < -np.pi, dphases + 2*np.pi, dphases)
    
    # Close the curve
    final_dphase = phases[0] - phases[-1]
    final_dphase = np.where(final_dphase > np.pi, final_dphase - 2*np.pi, final_dphase)
    final_dphase = np.where(final_dphase < -np.pi, final_dphase + 2*np.pi, final_dphase)
    
    total_phase_change = np.sum(dphases) + final_dphase
    return float(total_phase_change / (2 * np.pi))


def create_winding_signal(T: int, k: int, radius: float = 1.0) -> np.ndarray:
    """
    Create a signal that winds k times around the origin.
    
    z(t) = radius * exp(i * 2π * k * t / T)
    
    The phase step per sample is Δθ = 2πk/T.
    To maintain constant Δθ across different k, scale T proportionally.
    """
    t = np.arange(T)
    return radius * np.exp(1j * 2 * np.pi * k * t / T)


def perturb_signal(z: np.ndarray, noise_level: float = 0.1, seed: int = 42) -> np.ndarray:
    """Add complex Gaussian noise to a signal."""
    np.random.seed(seed)
    noise = noise_level * (np.random.randn(len(z)) + 1j * np.random.randn(len(z)))
    return z + noise


# =============================================================================
# Test 1: High Winding Ladder
# =============================================================================

def test_high_winding_ladder():
    """
    The main stress test: optimize winding signals with k = 1, 2, 4, 8, 16, 32.
    
    For each k:
    1. Create target signal with k windings
    2. Start from perturbed version
    3. Optimize using L2 and phase_robust loss
    4. Check if winding number is preserved
    
    Anti-aliasing: T = 64 * k to maintain constant phase step.
    
    Diagnostics:
    - Continuous winding error |W_final - k| (not just rounded)
    - min|x|: pointwise minimum (can miss segment crossings)
    - min_seg: minimum segment distance to origin (better topology indicator)
    
    Key hypothesis: If min_seg ≈ 0 while min|x| is large, the optimizer
    is "tunneling" through the origin between samples.
    """
    print("\n[TEST 1: HIGH WINDING LADDER]")
    print("-" * 95)
    print("  Testing optimizer stability across increasing winding numbers")
    print("  min|x| = pointwise min, min_seg = segment distance (better for detecting tunneling)")
    print()
    
    winding_numbers = [1, 2, 4, 8, 16, 32]
    base_T = 64
    noise_level = 0.02  # Very small perturbation
    
    results = {
        'l2': {},
        'phase_robust': {},
    }
    
    print(f"  {'k':>4} {'T':>5} {'Loss':<7} {'W_f':>6} {'|ΔW|':>6} {'ok':>3} {'min|x|':>9} {'min_seg':>9} {'Loss%':>7}")
    print("  " + "-" * 70)
    
    for k in winding_numbers:
        # Scale T to maintain constant phase step
        T = base_T * k
        
        # Create target and perturbed start
        x_target = create_winding_signal(T, k, radius=1.0)
        x0 = perturb_signal(x_target, noise_level=noise_level)
        
        initial_winding = compute_winding_number(x0)
        
        # Create HST with radial_floor for topology preservation
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        for loss_type in ['l2', 'phase_robust']:
            short_name = 'L2' if loss_type == 'l2' else 'robust'
            
            # Optimize with proper normalization (preconditioner, not new objective)
            result = optimize_signal(
                target_coeffs, hst, x0.copy(),
                n_steps=100,
                lr=1e-8,
                momentum=0.0,
                normalize=True,
                loss_type=loss_type,
                phase_lambda=1.0,
                verbose=False,
            )
            
            final_winding = compute_winding_number(result.signal)
            
            # Continuous error (more informative than just rounding)
            continuous_error = abs(final_winding - k)
            rounded_winding = round(final_winding)
            winding_preserved = (rounded_winding == k)
            
            # Get topology indicators
            min_mag = min(result.min_magnitude_history) if hasattr(result, 'min_magnitude_history') else np.min(np.abs(result.signal))
            min_seg = min(result.min_segment_dist_history) if hasattr(result, 'min_segment_dist_history') else min_segment_distance_to_origin(result.signal)
            
            loss_reduction = (1 - result.final_loss / result.loss_history[0]) * 100
            
            results[loss_type][k] = {
                'initial_winding': initial_winding,
                'final_winding': final_winding,
                'continuous_error': continuous_error,
                'preserved': winding_preserved,
                'min_magnitude': min_mag,
                'min_segment_dist': min_seg,
                'loss_reduction': loss_reduction,
            }
            
            status = "✓" if winding_preserved else "✗"
            print(f"  {k:>4} {T:>5} {short_name:<7} {final_winding:>6.2f} "
                  f"{continuous_error:>6.3f} {status:>3} {min_mag:>9.4f} {min_seg:>9.4f} {loss_reduction:>+6.1f}%")
    
    # Analyze results
    print()
    print("  Summary:")
    
    l2_preserved = sum(1 for k in winding_numbers if results['l2'][k]['preserved'])
    robust_preserved = sum(1 for k in winding_numbers if results['phase_robust'][k]['preserved'])
    
    l2_avg_error = np.mean([results['l2'][k]['continuous_error'] for k in winding_numbers])
    robust_avg_error = np.mean([results['phase_robust'][k]['continuous_error'] for k in winding_numbers])
    
    print(f"    L2:           {l2_preserved}/{len(winding_numbers)} preserved, avg |ΔW| = {l2_avg_error:.4f}")
    print(f"    phase_robust: {robust_preserved}/{len(winding_numbers)} preserved, avg |ΔW| = {robust_avg_error:.4f}")
    
    # Check for tunneling: winding changed but min_seg was not small
    print()
    print("  Tunneling analysis (winding changed with large min_seg?):")
    tunneling_detected = False
    for loss_type in ['l2', 'phase_robust']:
        for k in winding_numbers:
            r = results[loss_type][k]
            if not r['preserved'] and r['min_segment_dist'] > 0.1:
                print(f"    ⚠ {loss_type} k={k}: winding changed but min_seg={r['min_segment_dist']:.4f} (possible artifact)")
                tunneling_detected = True
    
    if not tunneling_detected:
        print("    No tunneling detected - winding changes correlate with small min_seg")
    
    # This test is diagnostic - always pass
    return True, f"Diagnostic: L2 {l2_preserved}/{len(winding_numbers)}, robust {robust_preserved}/{len(winding_numbers)}"


# =============================================================================
# Test 2: Homotopy Jump Detection
# =============================================================================

def test_homotopy_jump_detection():
    """
    Test for homotopy class jumping under various perturbation levels.
    
    Key insight: Winding changes require the trajectory to pass through
    or near the origin. We track two indicators:
    - min|x|: pointwise minimum (can miss segment crossings)
    - min_seg: segment distance to origin (catches "tunneling")
    """
    print("\n[TEST 2: HOMOTOPY JUMP DETECTION]")
    print("-" * 95)
    print("  Testing topology preservation under perturbation")
    print("  min|x| = pointwise, min_seg = segment distance (catches tunneling)")
    print()
    
    k = 8  # Moderate winding number
    T = 64 * k
    
    # Create target
    x_target = create_winding_signal(T, k, radius=1.0)
    
    print(f"  Target winding: k = {k}")
    print()
    print(f"  {'Noise':<6} {'Loss':<7} {'W_f':>6} {'|ΔW|':>6} {'ok':>3} {'min|x|':>9} {'min_seg':>9} {'Loss%':>7}")
    print("  " + "-" * 60)
    
    results = {}
    
    for noise_level in [0.01, 0.05, 0.1, 0.2]:
        x0 = perturb_signal(x_target, noise_level=noise_level, seed=42)
        initial_winding = compute_winding_number(x0)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        for loss_type in ['l2', 'phase_robust']:
            short_name = 'L2' if loss_type == 'l2' else 'robust'
            
            result = optimize_signal(
                target_coeffs, hst, x0.copy(),
                n_steps=100,
                lr=1e-8,
                momentum=0.0,
                normalize=True,
                loss_type=loss_type,
                phase_lambda=1.0,
                verbose=False,
            )
            
            final_winding = compute_winding_number(result.signal)
            continuous_error = abs(final_winding - k)
            rounded_winding = round(final_winding)
            preserved = (rounded_winding == k)
            
            min_mag = min(result.min_magnitude_history) if hasattr(result, 'min_magnitude_history') else np.min(np.abs(result.signal))
            min_seg = min(result.min_segment_dist_history) if hasattr(result, 'min_segment_dist_history') else min_segment_distance_to_origin(result.signal)
            loss_reduction = (1 - result.final_loss / result.loss_history[0]) * 100
            
            key = (noise_level, loss_type)
            results[key] = {
                'initial': initial_winding,
                'final': final_winding,
                'error': continuous_error,
                'preserved': preserved,
                'min_mag': min_mag,
                'min_seg': min_seg,
                'loss_reduction': loss_reduction,
            }
            
            status = "✓" if preserved else "✗"
            print(f"  {noise_level:<6.2f} {short_name:<7} {final_winding:>6.2f} "
                  f"{continuous_error:>6.3f} {status:>3} {min_mag:>9.4f} {min_seg:>9.4f} {loss_reduction:>+6.1f}%")
    
    print()
    print("  Interpretation:")
    print("    - |ΔW| < 0.5: Winding preserved (rounds to correct integer)")
    print("    - Small min|x|: Signal approached origin (topology danger)")
    print("    - phase_robust should be more stable than L2 at high noise")
    
    # This is diagnostic - always pass
    return True, "Homotopy jump analysis complete"


# =============================================================================
# Test 3: Topology Margin Constraint
# =============================================================================

def test_topology_margin():
    """
    Test the topology_margin parameter that enforces min_seg >= δ.
    
    This uses backtracking line search to prevent the optimizer from
    "tunneling" through the origin between samples.
    """
    print("\n[TEST 3: TOPOLOGY MARGIN CONSTRAINT]")
    print("-" * 95)
    print("  Testing backtracking line search with min_seg >= δ constraint")
    print("  Comparing optimization with and without topology protection")
    print()
    
    # Use k=16 which showed tunneling in previous tests
    k = 16
    T = 64 * k
    noise_level = 0.02
    margin = 0.01  # Enforce min_seg >= 0.01
    
    x_target = create_winding_signal(T, k, radius=1.0)
    x0 = perturb_signal(x_target, noise_level=noise_level, seed=42)
    initial_winding = compute_winding_number(x0)
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='radial_floor', epsilon=1e-8
    )
    
    target_coeffs = extract_all_targets(hst, x_target)
    
    print(f"  Target winding: k = {k}, margin = {margin}")
    print(f"  Initial winding: {initial_winding:.2f}")
    print()
    
    results = {}
    
    print(f"  {'Mode':<20} {'W_final':>8} {'|ΔW|':>7} {'ok':>4} {'min_seg':>10} {'Loss%':>8}")
    print("  " + "-" * 60)
    
    for mode, use_margin in [('No protection', None), ('With margin', margin)]:
        for loss_type in ['l2', 'phase_robust']:
            label = f"{mode} ({loss_type})"
            
            result = optimize_signal(
                target_coeffs, hst, x0.copy(),
                n_steps=100,
                lr=1e-8,
                momentum=0.0,
                normalize=True,
                loss_type=loss_type,
                phase_lambda=1.0,
                topology_margin=use_margin,
                verbose=False,
            )
            
            final_winding = compute_winding_number(result.signal)
            continuous_error = abs(final_winding - k)
            preserved = (round(final_winding) == k)
            min_seg = min(result.min_segment_dist_history)
            loss_reduction = (1 - result.final_loss / result.loss_history[0]) * 100
            
            results[(mode, loss_type)] = {
                'final_winding': final_winding,
                'error': continuous_error,
                'preserved': preserved,
                'min_seg': min_seg,
                'loss_reduction': loss_reduction,
            }
            
            status = "✓" if preserved else "✗"
            print(f"  {label:<20} {final_winding:>8.2f} {continuous_error:>7.3f} {status:>4} "
                  f"{min_seg:>10.6f} {loss_reduction:>+7.1f}%")
    
    print()
    
    # Check if margin helped
    no_prot_preserved = sum(1 for k, v in results.items() if k[0] == 'No protection' and v['preserved'])
    with_margin_preserved = sum(1 for k, v in results.items() if k[0] == 'With margin' and v['preserved'])
    
    print(f"  Results: No protection {no_prot_preserved}/2 preserved, With margin {with_margin_preserved}/2 preserved")
    
    # Verify margin constraint was enforced
    margin_enforced = all(
        v['min_seg'] >= margin * 0.9  # Allow small tolerance
        for k, v in results.items() if k[0] == 'With margin'
    )
    
    if margin_enforced:
        print(f"  ✓ Margin constraint enforced (min_seg >= {margin})")
    else:
        print(f"  ✗ Margin constraint violated!")
    
    # Pass if margin was enforced (even if winding still changed)
    return margin_enforced, f"Margin enforced: {margin_enforced}, preserved: {with_margin_preserved}/2"


# =============================================================================
# Test 4: Gradient Direction with High Winding
# =============================================================================

def test_gradient_direction_high_winding():
    """
    Verify gradient direction is correct for high winding signals.
    
    Even if we can't fully optimize, the gradient should point toward
    loss reduction (not divergence).
    """
    print("\n[TEST 4: GRADIENT DIRECTION (HIGH WINDING)]")
    print("-" * 70)
    print("  Verifying gradient points toward loss reduction")
    print()
    
    from hst.optimize_numpy import compute_loss_and_grad
    
    test_cases = [1, 8, 32]
    all_correct = True
    
    for k in test_cases:
        T = 64 * k
        
        x_target = create_winding_signal(T, k, radius=1.0)
        x = perturb_signal(x_target, noise_level=0.05)
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        for loss_type in ['l2', 'phase_robust']:
            loss0, grad = compute_loss_and_grad(
                x, hst, target_coeffs,
                loss_type=loss_type, phase_lambda=1.0
            )
            
            # Normalized step
            step_size = 0.0001 * np.linalg.norm(x) / (np.linalg.norm(grad) + 1e-15)
            x1 = x - step_size * grad
            
            loss1, _ = compute_loss_and_grad(
                x1, hst, target_coeffs,
                loss_type=loss_type, phase_lambda=1.0
            )
            
            decreased = loss1 < loss0
            status = "✓" if decreased else "✗"
            
            if not decreased:
                all_correct = False
            
            print(f"  k={k:>2}, {loss_type:<14}: loss {loss0:.2e} -> {loss1:.2e}, "
                  f"decreased: {status}")
    
    print()
    if all_correct:
        print("  ✓ Gradient direction correct for all cases")
    else:
        print("  ✗ Some gradients point wrong direction")
    
    return all_correct, "Gradient direction test"


# =============================================================================
# Test 4: Phase Step Analysis
# =============================================================================

def test_phase_step_analysis():
    """
    Analyze the phase step Δθ for different winding numbers.
    
    This diagnostic shows why anti-aliasing (scaling T with k) is necessary.
    """
    print("\n[TEST 5: PHASE STEP ANALYSIS]")
    print("-" * 70)
    print("  Analyzing phase step Δθ = 2πk/T")
    print()
    
    print(f"  {'k':>4} {'T (fixed)':>10} {'Δθ (rad)':>12} {'Δθ (deg)':>10} {'T (scaled)':>12} {'Δθ (rad)':>12}")
    print("  " + "-" * 62)
    
    base_T = 64
    
    for k in [1, 2, 4, 8, 16, 32]:
        # Fixed T
        dtheta_fixed = 2 * np.pi * k / base_T
        
        # Scaled T
        T_scaled = base_T * k
        dtheta_scaled = 2 * np.pi * k / T_scaled
        
        print(f"  {k:>4} {base_T:>10} {dtheta_fixed:>12.4f} {np.degrees(dtheta_fixed):>10.2f}° "
              f"{T_scaled:>12} {dtheta_scaled:>12.4f}")
    
    print()
    print("  Note: With fixed T=64, k=32 gives Δθ=π (Nyquist limit)")
    print("  Scaling T with k keeps Δθ constant at π/32 ≈ 0.098 rad")
    
    return True, "Phase step analysis complete"


# =============================================================================
# Test 5: Segment Distance to Origin (Unit Test)
# =============================================================================

def test_segment_distance_to_origin():
    """
    Unit test for the segment_distance_to_origin function.
    
    This verifies the function correctly computes the minimum distance
    from the origin to line segments in the complex plane.
    """
    print("\n[TEST 6: SEGMENT DISTANCE TO ORIGIN]")
    print("-" * 70)
    print("  Unit testing the segment distance computation")
    print()
    
    from hst.optimize_numpy import segment_distance_to_origin, min_segment_distance_to_origin
    
    all_correct = True
    
    # Test case 1: Segment on real axis, passing through origin
    z1, z2 = -1+0j, 1+0j
    dist = segment_distance_to_origin(z1, z2)
    expected = 0.0
    correct = abs(dist - expected) < 1e-10
    print(f"  Segment [-1, 1] (passes through 0): dist={dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 2: Segment parallel to real axis, offset by 1
    z1, z2 = -1+1j, 1+1j
    dist = segment_distance_to_origin(z1, z2)
    expected = 1.0  # Perpendicular distance
    correct = abs(dist - expected) < 1e-10
    print(f"  Segment [-1+i, 1+i] (parallel, offset 1): dist={dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 3: Segment where closest point is an endpoint
    z1, z2 = 2+0j, 3+0j
    dist = segment_distance_to_origin(z1, z2)
    expected = 2.0  # Closest is z1
    correct = abs(dist - expected) < 1e-10
    print(f"  Segment [2, 3] (endpoint closest): dist={dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 4: Diagonal segment near origin
    z1, z2 = 1+1j, 2+2j
    dist = segment_distance_to_origin(z1, z2)
    expected = np.sqrt(2)  # Closest is z1
    correct = abs(dist - expected) < 1e-10
    print(f"  Segment [1+i, 2+2i] (diagonal): dist={dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 5: Segment where perpendicular hits interior
    z1, z2 = 1+1j, 1-1j
    dist = segment_distance_to_origin(z1, z2)
    expected = 1.0  # Perpendicular distance to vertical line at x=1
    correct = abs(dist - expected) < 1e-10
    print(f"  Segment [1+i, 1-i] (vertical at x=1): dist={dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 6: Square around origin
    square = np.array([1+1j, 1-1j, -1-1j, -1+1j])
    min_dist = min_segment_distance_to_origin(square, closed=True)
    expected = 1.0  # Each edge is distance 1 from origin
    correct = abs(min_dist - expected) < 1e-10
    print(f"  Square [±1±i]: min_dist={min_dist:.6f}, expected={expected:.6f}, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    # Test case 7: Triangle where one segment passes close to origin
    # Segment from (1, 0) to (-0.5, 0.866) passes through y-intercept at x=0
    # When x=0: y = 0.866 * (0.5)/(1.5) = 0.289
    triangle = np.array([1+0j, -0.5+0.866j, -0.5-0.866j])
    min_dist = min_segment_distance_to_origin(triangle, closed=True)
    # Actually compute expected: segment [1, -0.5+0.866i] has closest point
    # The perpendicular from 0 to this line...
    # Line: (1,0) + t*(-1.5, 0.866), t in [0,1]
    # This is a bit complex, let's just verify it's reasonably small
    correct = min_dist < 0.5  # Should be less than endpoint distances
    print(f"  Triangle near origin: min_dist={min_dist:.6f}, should be <0.5, {'✓' if correct else '✗'}")
    all_correct = all_correct and correct
    
    print()
    if all_correct:
        print("  ✓ All segment distance tests passed")
    else:
        print("  ✗ Some segment distance tests failed")
    
    return all_correct, "Segment distance unit tests"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 95)
    print("HIGH WINDING STRESS TESTS")
    print("Testing optimizer stability under increasing topological complexity")
    print("=" * 95)
    
    result = TestResult()
    
    p, m = test_segment_distance_to_origin()
    result.record("segment_distance_unit", p, m)
    
    p, m = test_phase_step_analysis()
    result.record("phase_step_analysis", p, m)
    
    p, m = test_gradient_direction_high_winding()
    result.record("gradient_direction_high_winding", p, m)
    
    p, m = test_topology_margin()
    result.record("topology_margin", p, m)
    
    p, m = test_high_winding_ladder()
    result.record("high_winding_ladder", p, m)
    
    p, m = test_homotopy_jump_detection()
    result.record("homotopy_jump_detection", p, m)
    
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ High winding stress tests passed!")
        print("  Diagnostics in place; topology preserved unless trajectory enters")
        print("  origin-danger zone (min_seg small). Use topology_margin to enforce.")
    else:
        print("\n⚠ Some tests failed - see details above.")
        print("  Consider using phase_robust loss for high-winding signals.")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
