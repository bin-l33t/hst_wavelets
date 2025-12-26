#!/usr/bin/env python3
"""
Winding Number Invariant Tests

Verifies that lifting strategies preserve or destroy the topological
winding number of signals.

The winding number W = (1/2π) ∮ d(arg z) counts how many times a
trajectory encircles the origin. This is a fundamental topological
invariant that Glinsky's theory relies on.

Key insight:
- radial_floor: z → r̃·exp(iθ) preserves angle → preserves winding
- shift: z → z + ε shifts the center → can change winding

This is the definitive test for "topology preservation".
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform


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
    Compute the winding number of a closed curve z(t).
    
    W = (1/2π) ∮ d(arg z) = (1/2π) Σ Δθ_i
    
    where Δθ_i is the phase change between consecutive samples,
    properly unwrapped to handle branch cuts.
    
    Parameters
    ----------
    z : ndarray, complex
        Complex signal representing a closed curve
        
    Returns
    -------
    winding : float
        Winding number (integer for closed curves)
    """
    # Compute phase of each point
    phases = np.angle(z)
    
    # Compute phase differences
    dphases = np.diff(phases)
    
    # Unwrap: phase jumps > π should be adjusted
    # This handles the branch cut at ±π
    dphases = np.where(dphases > np.pi, dphases - 2*np.pi, dphases)
    dphases = np.where(dphases < -np.pi, dphases + 2*np.pi, dphases)
    
    # Close the curve: add the phase change from last to first
    final_dphase = phases[0] - phases[-1]
    final_dphase = np.where(final_dphase > np.pi, final_dphase - 2*np.pi, final_dphase)
    final_dphase = np.where(final_dphase < -np.pi, final_dphase + 2*np.pi, final_dphase)
    
    total_phase_change = np.sum(dphases) + final_dphase
    winding = total_phase_change / (2 * np.pi)
    
    return float(winding)


def create_winding_signal(T: int, radius: float = 1.0, n_winds: int = 1,
                          center: complex = 0.0) -> np.ndarray:
    """
    Create a signal that winds around a center point n_winds times.
    
    z(t) = center + radius * exp(i * 2π * n_winds * t / T)
    """
    t = np.arange(T)
    return center + radius * np.exp(1j * 2 * np.pi * n_winds * t / T)


# =============================================================================
# Test 1: Winding Number Computation Verification
# =============================================================================

def test_winding_computation():
    """Verify our winding number computation is correct."""
    print("\n[TEST 1: WINDING NUMBER COMPUTATION]")
    print("-" * 50)
    
    T = 256
    
    test_cases = [
        (1.0, 0.0, 1, "Circle, r=1, center=0"),
        (0.5, 0.0, 1, "Circle, r=0.5, center=0"),
        (0.1, 0.0, 1, "Small circle, r=0.1, center=0"),
        (1.0, 0.0, 2, "Double winding"),
        (1.0, 0.0, 3, "Triple winding"),
        (1.0, 0.0, -1, "Negative winding"),
        (1.0, 2.0+0j, 1, "Circle centered at 2 (excludes origin)"),
    ]
    
    all_correct = True
    
    for radius, center, expected_wind, desc in test_cases:
        z = create_winding_signal(T, radius, expected_wind, center)
        
        # Compute winding around origin
        computed_wind = compute_winding_number(z)
        
        # For circle centered away from origin, winding should be 0
        if np.abs(center) > radius:
            expected_wind = 0
        
        is_correct = abs(computed_wind - expected_wind) < 0.1
        status = "✓" if is_correct else "✗"
        
        print(f"  {status} {desc}: computed={computed_wind:.3f}, expected={expected_wind}")
        
        if not is_correct:
            all_correct = False
    
    return all_correct, "Winding number computation verified"


# =============================================================================
# Test 2: Radial Floor Preserves Winding
# =============================================================================

def test_radial_floor_preserves_winding():
    """
    Verify that radial_floor lifting preserves winding number.
    
    Sweep radius from small to large and epsilon values.
    The winding should always be preserved.
    """
    print("\n[TEST 2: RADIAL FLOOR PRESERVES WINDING]")
    print("-" * 50)
    
    T = 256
    
    test_cases = []
    
    # Sweep radius from 0.01 to 10.0
    for radius in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
        # Sweep epsilon from 1e-8 to 1.0
        for eps in [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]:
            test_cases.append((radius, eps))
    
    all_preserved = True
    failures = []
    
    for radius, eps in test_cases:
        # Create winding signal centered at origin
        z = create_winding_signal(T, radius=radius, n_winds=1, center=0.0)
        
        # Original winding
        original_wind = compute_winding_number(z)
        
        # Apply radial_floor lifting
        r = np.abs(z)
        r_floor = np.sqrt(r**2 + eps**2)
        tiny = 1e-300
        scale = r_floor / np.maximum(r, tiny)
        z_lifted = z * scale
        
        # Lifted winding
        lifted_wind = compute_winding_number(z_lifted)
        
        # Winding should be preserved (within tolerance)
        preserved = abs(lifted_wind - original_wind) < 0.1
        
        if not preserved:
            all_preserved = False
            failures.append((radius, eps, original_wind, lifted_wind))
    
    print(f"  Tested {len(test_cases)} (radius, epsilon) combinations")
    
    if all_preserved:
        print("  ✓ All cases preserved winding number")
    else:
        print(f"  ✗ {len(failures)} cases failed:")
        for r, e, orig, lift in failures[:5]:
            print(f"      r={r}, eps={e}: {orig:.2f} -> {lift:.2f}")
    
    return all_preserved, f"{len(test_cases) - len(failures)}/{len(test_cases)} preserved"


# =============================================================================
# Test 3: Shift Destroys Winding When ε > r
# =============================================================================

def test_shift_destroys_winding():
    """
    Verify that shift lifting destroys winding when epsilon > radius.
    
    When we shift by ε > r, the origin is no longer enclosed by the curve,
    so winding number changes from 1 to 0.
    """
    print("\n[TEST 3: SHIFT DESTROYS WINDING WHEN ε > r]")
    print("-" * 50)
    
    T = 256
    
    test_cases = []
    
    # Test cases where eps > radius (should destroy winding)
    for radius in [0.1, 0.5, 1.0]:
        for eps in [radius * 0.1, radius * 0.5, radius * 2.0, radius * 5.0]:
            should_destroy = eps > radius
            test_cases.append((radius, eps, should_destroy))
    
    correct_predictions = 0
    results = []
    
    for radius, eps, should_destroy in test_cases:
        # Create winding signal
        z = create_winding_signal(T, radius=radius, n_winds=1, center=0.0)
        
        # Original winding
        original_wind = compute_winding_number(z)
        
        # Apply shift lifting
        z_shifted = z + eps
        
        # Shifted winding
        shifted_wind = compute_winding_number(z_shifted)
        
        # Check if winding was destroyed (went from ~1 to ~0)
        was_destroyed = abs(shifted_wind) < 0.5
        
        correct = was_destroyed == should_destroy
        if correct:
            correct_predictions += 1
        
        results.append({
            'radius': radius,
            'eps': eps,
            'should_destroy': should_destroy,
            'was_destroyed': was_destroyed,
            'original': original_wind,
            'shifted': shifted_wind,
            'correct': correct,
        })
    
    print(f"  Testing shift lifting on {len(test_cases)} cases")
    print(f"  {'r':>6} {'ε':>8} {'ε>r':>6} {'Destroyed':>10} {'Original':>10} {'Shifted':>10}")
    print("  " + "-" * 56)
    
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"  {r['radius']:>6.2f} {r['eps']:>8.2f} {str(r['should_destroy']):>6} "
              f"{str(r['was_destroyed']):>10} {r['original']:>10.2f} {r['shifted']:>10.2f} {status}")
    
    all_correct = correct_predictions == len(test_cases)
    return all_correct, f"{correct_predictions}/{len(test_cases)} predictions correct"


# =============================================================================
# Test 4: HST Lifting Comparison
# =============================================================================

def test_hst_lifting_winding():
    """
    Test winding preservation through actual HST lifting methods.
    """
    print("\n[TEST 4: HST LIFTING METHODS]")
    print("-" * 50)
    
    T = 128
    
    # Create winding signal
    radius = 0.5
    z = create_winding_signal(T, radius=radius, n_winds=1, center=0.0)
    original_wind = compute_winding_number(z)
    
    print(f"  Original signal: r={radius}, winding={original_wind:.2f}")
    print()
    
    results = {}
    
    # Test different lifting modes
    test_configs = [
        ('radial_floor', 1e-8, True),
        ('radial_floor', 0.1, True),
        ('radial_floor', 1.0, True),  # Even with large eps, radial_floor preserves angle
        ('shift', 0.1, True),  # eps < radius, should preserve
        ('shift', 1.0, False),  # eps > radius, should destroy
        ('shift', 2.0, False),  # eps >> radius, definitely destroys
    ]
    
    all_correct = True
    
    for lifting, eps, should_preserve in test_configs:
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting=lifting, epsilon=eps
        )
        
        # Apply lifting
        z_lifted, _ = hst._lift(z)
        lifted_wind = compute_winding_number(z_lifted)
        
        # Check preservation
        was_preserved = abs(lifted_wind - original_wind) < 0.5
        correct = was_preserved == should_preserve
        
        if not correct:
            all_correct = False
        
        status = "✓" if correct else "✗"
        outcome = "preserved" if was_preserved else "destroyed"
        expected = "preserve" if should_preserve else "destroy"
        
        print(f"  {status} {lifting:12} eps={eps:>6.4f}: winding {outcome:>10} "
              f"({lifted_wind:.2f}), expected: {expected}")
        
        results[(lifting, eps)] = {
            'lifted_wind': lifted_wind,
            'was_preserved': was_preserved,
            'correct': correct,
        }
    
    return all_correct, f"HST lifting winding test"


# =============================================================================
# Test 5: Coefficient Winding Analysis
# =============================================================================

def test_coefficient_winding():
    """
    Analyze winding numbers of HST coefficients.
    
    For a winding signal, the convolution with wavelets may create
    new winding patterns at different scales.
    """
    print("\n[TEST 5: COEFFICIENT WINDING ANALYSIS]")
    print("-" * 50)
    
    T = 256
    
    # Create winding signal
    z = create_winding_signal(T, radius=1.0, n_winds=2, center=0.0)
    original_wind = compute_winding_number(z)
    
    print(f"  Input signal: winding = {original_wind:.2f}")
    print()
    
    # Process with radial_floor (topology-preserving)
    hst = HeisenbergScatteringTransform(
        T, J=3, Q=2, max_order=1,
        lifting='radial_floor', epsilon=1e-8
    )
    
    output = hst.forward(z)
    
    print("  Order-1 coefficient windings (after lifting but before R):")
    
    # Get lifted signal
    z_lifted, _ = hst._lift(z)
    z_hat = np.fft.fft(z_lifted)
    
    for j in range(min(4, hst.n_mothers)):  # Show first 4 scales
        path = (j,)
        
        # Get the convolution output (U, before R)
        U = np.fft.ifft(z_hat * hst.filters[j])
        U_lifted, _ = hst._lift(U)
        
        # Compute winding of lifted coefficient
        coeff_wind = compute_winding_number(U_lifted)
        
        print(f"    Scale j={j}: winding = {coeff_wind:.2f}")
    
    # This test is diagnostic - always pass
    return True, "Coefficient winding analysis complete"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("WINDING NUMBER INVARIANT TESTS")
    print("Verifying topological preservation under lifting")
    print("=" * 60)
    
    result = TestResult()
    
    p, m = test_winding_computation()
    result.record("winding_computation", p, m)
    
    p, m = test_radial_floor_preserves_winding()
    result.record("radial_floor_preserves", p, m)
    
    p, m = test_shift_destroys_winding()
    result.record("shift_destroys_when_large", p, m)
    
    p, m = test_hst_lifting_winding()
    result.record("hst_lifting_winding", p, m)
    
    p, m = test_coefficient_winding()
    result.record("coefficient_winding", p, m)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, r in result.results.items():
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {result.passed}/{result.passed + result.failed} passed")
    
    if result.failed == 0:
        print("\n✓ Winding number invariance verified!")
        print("  radial_floor preserves topology, shift destroys it when ε > r")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
