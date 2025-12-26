#!/usr/bin/env python3
"""
Fast test to verify the homotopy-aware topology guard enforces its margin.

Asserts: With guard enabled, min_seg(homotopy) never dips below topology_margin
for any accepted optimization step.

Run: python tests/test_homotopy_guard.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    optimize_signal,
    extract_all_targets,
    min_seg_along_homotopy,
)


def test_guard_enforces_margin():
    """
    Verify that with topology_margin enabled, no accepted step has
    homotopy min_seg below the margin.
    """
    print("="*60)
    print("TEST: Homotopy guard enforces margin")
    print("="*60)
    
    # Focus on k=16 cases (k=32 is slow, covered by test_winding_diagnostic)
    test_cases = [
        (16, 'l2'),
        (16, 'phase_robust'),
    ]
    
    MARGIN = 0.05
    base_T = 64
    N_STEPS = 50  # Reduced for speed
    
    all_passed = True
    
    for k, loss_type in test_cases:
        T = base_T * k
        
        # Create target and perturbed start
        t = np.arange(T)
        x_target = np.exp(1j * 2 * np.pi * k * t / T)
        np.random.seed(42)
        x0 = x_target + 0.02 * (np.random.randn(T) + 1j * np.random.randn(T))
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        target_coeffs = extract_all_targets(hst, x_target)
        
        # Capture all signals during optimization
        signals = [x0.copy()]
        
        def capture(step, x, loss, grad):
            signals.append(x.copy())
        
        # Run with guard
        result = optimize_signal(
            target_coeffs, hst, x0.copy(),
            n_steps=N_STEPS,
            lr=1e-8,
            normalize=True,
            loss_type=loss_type,
            topology_margin=MARGIN,
            verbose=False,
            callback=capture,
        )
        
        # Check homotopy min_seg for every consecutive pair
        violations = []
        min_homotopy_seen = float('inf')
        
        for i in range(len(signals) - 1):
            homotopy_min, worst_s = min_seg_along_homotopy(signals[i], signals[i+1])
            min_homotopy_seen = min(min_homotopy_seen, homotopy_min)
            
            if homotopy_min < MARGIN:
                violations.append((i, homotopy_min, worst_s))
        
        # Report
        case_name = f"k={k} {loss_type}"
        if violations:
            print(f"\n{case_name}: ✗ FAILED - {len(violations)} violations")
            for step, hmin, s in violations[:3]:
                print(f"  Step {step}→{step+1}: homotopy_min={hmin:.6f} < {MARGIN} at s={s:.2f}")
            all_passed = False
        else:
            w_init = round(result.winding_history[0])
            w_final = round(result.winding_history[-1])
            print(f"{case_name}: ✓ PASSED (min_homotopy={min_homotopy_seen:.4f} >= {MARGIN}, W: {w_init}→{w_final})")
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED: Guard enforces margin on all homotopy paths")
    else:
        print("SOME TESTS FAILED: Guard did not enforce margin")
    print("="*60)
    
    return all_passed


def test_guard_preserves_winding():
    """Quick check that guard preserves winding number."""
    print("\n" + "="*60)
    print("TEST: Guard preserves winding number")
    print("="*60)
    
    test_cases = [(16, 'l2'), (16, 'phase_robust')]
    MARGIN = 0.05
    N_STEPS = 50
    all_passed = True
    
    for k, loss_type in test_cases:
        T = 64 * k
        t = np.arange(T)
        x_target = np.exp(1j * 2 * np.pi * k * t / T)
        np.random.seed(42)
        x0 = x_target + 0.02 * (np.random.randn(T) + 1j * np.random.randn(T))
        
        hst = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=1, lifting='radial_floor', epsilon=1e-8)
        target_coeffs = extract_all_targets(hst, x_target)
        
        result = optimize_signal(
            target_coeffs, hst, x0.copy(), n_steps=N_STEPS, lr=1e-8,
            normalize=True, loss_type=loss_type, topology_margin=MARGIN, verbose=False
        )
        
        w_init = round(result.winding_history[0])
        w_final = round(result.winding_history[-1])
        
        case_name = f"k={k} {loss_type}"
        if w_init == w_final:
            print(f"{case_name}: ✓ Winding preserved ({w_init})")
        else:
            print(f"{case_name}: ✗ Winding changed ({w_init}→{w_final})")
            all_passed = False
    
    print("="*60)
    return all_passed


if __name__ == "__main__":
    p1 = test_guard_enforces_margin()
    p2 = test_guard_preserves_winding()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Guard enforces margin: {'PASS' if p1 else 'FAIL'}")
    print(f"Guard preserves winding: {'PASS' if p2 else 'FAIL'}")
    
    sys.exit(0 if (p1 and p2) else 1)
