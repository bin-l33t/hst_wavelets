#!/usr/bin/env python3
"""
Winding Diagnostic Tests

Deep investigation into when and why winding numbers change during optimization.
This test tracks winding at every step to identify:
1. Whether winding changes gradually or suddenly
2. Correlation between winding change and min_segment_distance
3. Whether the R mapping's branch cuts are involved
4. Numerical precision issues at high winding numbers

Key question: Does winding change because:
(a) The optimizer trajectory passes through the origin (topological)
(b) Numerical precision loss in phase computation (numerical artifact)
(c) Branch cut crossing in R-space (conformal mapping issue)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hst.scattering import HeisenbergScatteringTransform
from hst.optimize_numpy import (
    optimize_signal,
    extract_all_targets,
    compute_winding_number,
    detect_winding_change,
    min_segment_distance_to_origin,
    _compute_winding_inline,
)


def create_winding_signal(T: int, k: int, radius: float = 1.0) -> np.ndarray:
    """Create a signal that winds k times around the origin."""
    t = np.arange(T)
    return radius * np.exp(1j * 2 * np.pi * k * t / T)


def test_winding_precision():
    """
    Test winding computation precision at various k values.
    
    This isolates numerical issues from optimization issues.
    """
    print("\n" + "="*70)
    print("TEST: WINDING COMPUTATION PRECISION")
    print("="*70)
    
    print("\nTesting winding computation accuracy (no optimization)")
    print(f"{'k':>4} {'T':>6} {'W_computed':>12} {'Error':>12} {'Precision':>12}")
    print("-" * 50)
    
    base_T = 64
    all_good = True
    
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        T = base_T * k  # Scale T to maintain phase resolution
        z = create_winding_signal(T, k)
        
        # Test both precision modes
        w_high = _compute_winding_inline(z, high_precision=True)
        w_low = _compute_winding_inline(z, high_precision=False)
        
        error_high = abs(w_high - k)
        error_low = abs(w_low - k)
        
        status = "✓" if error_high < 0.01 else "✗"
        print(f"{k:>4} {T:>6} {w_high:>12.6f} {error_high:>12.2e} (high) {status}")
        print(f"{'':>4} {'':>6} {w_low:>12.6f} {error_low:>12.2e} (low)")
        
        if error_high > 0.01:
            all_good = False
    
    return all_good


def test_winding_tracking_during_optimization():
    """
    Track winding number at every optimization step.
    
    This reveals whether winding changes:
    - Suddenly (suggesting a topological event)
    - Gradually (suggesting numerical drift)
    """
    print("\n" + "="*70)
    print("TEST: WINDING TRACKING DURING OPTIMIZATION")
    print("="*70)
    
    # Focus on the problematic k=16 case
    test_cases = [
        (8, 'l2'),
        (8, 'phase_robust'),
        (16, 'l2'),
        (16, 'phase_robust'),
        (32, 'l2'),
        (32, 'phase_robust'),
    ]
    
    base_T = 64
    noise_level = 0.02
    
    results = {}
    
    for k, loss_type in test_cases:
        T = base_T * k
        
        print(f"\n--- k={k}, T={T}, loss={loss_type} ---")
        
        # Create target and perturbed start
        x_target = create_winding_signal(T, k, radius=1.0)
        np.random.seed(42)
        x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
        
        initial_winding = compute_winding_number(x0)
        print(f"Initial winding: {initial_winding:.4f} (target: {k})")
        
        # Create HST
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        # Optimize with verbose tracking
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
        
        # Analyze winding history
        winding_history = result.winding_history
        analysis = detect_winding_change(winding_history)
        
        print(f"Final winding: {winding_history[-1]:.4f}")
        print(f"Winding changed: {analysis['changed']}")
        if analysis['changed']:
            print(f"  Change detected at step: {analysis['change_step']}")
            print(f"  Initial → Final: {analysis['initial_winding']} → {analysis['final_winding']}")
        print(f"Max deviation from integer: {analysis['max_deviation']:.4f}")
        
        # Check correlation with min_seg
        min_segs = result.min_segment_dist_history
        
        # Find minimum min_seg and corresponding winding
        min_seg_idx = np.argmin(min_segs)
        min_seg_value = min_segs[min_seg_idx]
        winding_at_min_seg = winding_history[min_seg_idx]
        
        print(f"Min segment distance: {min_seg_value:.6f} at step {min_seg_idx}")
        print(f"Winding at min_seg step: {winding_at_min_seg:.4f}")
        
        results[(k, loss_type)] = {
            'analysis': analysis,
            'min_seg_at_change': min_segs[analysis['change_step']] if analysis['change_step'] else None,
            'winding_history': winding_history,
            'min_seg_history': min_segs,
        }
    
    return results


def test_winding_change_correlation():
    """
    Analyze whether winding changes correlate with small min_seg.
    
    Hypothesis: If winding changes only when min_seg is small, it's topological.
    If winding changes with large min_seg, it's numerical or branch-cut related.
    """
    print("\n" + "="*70)
    print("TEST: WINDING CHANGE vs MIN_SEG CORRELATION")
    print("="*70)
    
    results = test_winding_tracking_during_optimization()
    
    print("\n" + "-"*70)
    print("CORRELATION ANALYSIS")
    print("-"*70)
    
    print(f"\n{'Case':<20} {'Changed':>8} {'Change Step':>12} {'min_seg@change':>15} {'Classification':>15}")
    print("-" * 75)
    
    for (k, loss_type), data in results.items():
        analysis = data['analysis']
        case_name = f"k={k} {loss_type}"
        
        changed = "YES" if analysis['changed'] else "NO"
        
        if analysis['changed'] and analysis['change_step'] is not None:
            step = analysis['change_step']
            min_seg = data['min_seg_history'][step]
            
            # Classify the change
            if min_seg < 0.01:
                classification = "TOPOLOGICAL"
            elif min_seg < 0.1:
                classification = "MARGINAL"
            else:
                classification = "ANOMALOUS"
            
            print(f"{case_name:<20} {changed:>8} {step:>12} {min_seg:>15.6f} {classification:>15}")
        else:
            print(f"{case_name:<20} {changed:>8} {'N/A':>12} {'N/A':>15} {'STABLE':>15}")
    
    # Summary
    anomalous_cases = [
        (k, lt) for (k, lt), d in results.items()
        if d['analysis']['changed'] and d['analysis']['change_step'] is not None
        and d['min_seg_history'][d['analysis']['change_step']] > 0.1
    ]
    
    if anomalous_cases:
        print("\n⚠ ANOMALOUS CASES DETECTED:")
        print("  Winding changed despite large min_seg distance.")
        print("  This suggests numerical or branch-cut issues, not topological crossing.")
        for k, lt in anomalous_cases:
            print(f"    - k={k}, loss={lt}")
    else:
        print("\n✓ All winding changes correlate with small min_seg (topological)")
    
    return results


def test_winding_with_topology_margin():
    """
    Test if topology_margin constraint prevents winding changes.
    """
    print("\n" + "="*70)
    print("TEST: TOPOLOGY MARGIN EFFECTIVENESS")
    print("="*70)
    
    # Test the cases that failed without margin
    test_cases = [
        (16, 'l2'),
        (16, 'phase_robust'),
    ]
    
    base_T = 64
    noise_level = 0.02
    margin = 0.01
    
    print(f"\nTopology margin: {margin}")
    print(f"{'Case':<20} {'W_initial':>10} {'W_final':>10} {'Protected':>10} {'min_seg':>12}")
    print("-" * 65)
    
    for k, loss_type in test_cases:
        T = base_T * k
        
        x_target = create_winding_signal(T, k, radius=1.0)
        np.random.seed(42)
        x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        # With topology margin
        result = optimize_signal(
            target_coeffs, hst, x0.copy(),
            n_steps=100,
            lr=1e-8,
            momentum=0.0,
            normalize=True,
            loss_type=loss_type,
            phase_lambda=1.0,
            topology_margin=margin,
            verbose=False,
        )
        
        initial_w = round(result.winding_history[0])
        final_w = round(result.winding_history[-1])
        protected = "✓" if initial_w == final_w else "✗"
        min_seg = min(result.min_segment_dist_history)
        
        case_name = f"k={k} {loss_type}"
        print(f"{case_name:<20} {initial_w:>10} {final_w:>10} {protected:>10} {min_seg:>12.6f}")


def test_detailed_trajectory_analysis():
    """
    Detailed step-by-step analysis of k=16 phase_robust case.
    """
    print("\n" + "="*70)
    print("TEST: DETAILED TRAJECTORY (k=16, phase_robust)")
    print("="*70)
    
    k = 16
    T = 64 * k
    noise_level = 0.02
    
    x_target = create_winding_signal(T, k, radius=1.0)
    np.random.seed(42)
    x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='radial_floor', epsilon=1e-8
    )
    
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Optimize with verbose
    print("\nOptimization trajectory:")
    result = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=100,
        lr=1e-8,
        momentum=0.0,
        normalize=True,
        loss_type='phase_robust',
        phase_lambda=1.0,
        verbose=True,
    )
    
    # Detailed winding analysis
    print("\nWinding trajectory (every 10 steps):")
    wh = result.winding_history
    msh = result.min_segment_dist_history
    
    for i in range(0, len(wh), 10):
        w = wh[i]
        ms = msh[i]
        deviation = w - round(w)
        print(f"  Step {i:3d}: W={w:8.4f} (dev={deviation:+.4f}), min_seg={ms:.6f}")
    
    # Final
    print(f"  Step {len(wh)-1:3d}: W={wh[-1]:8.4f} (dev={wh[-1]-round(wh[-1]):+.4f}), min_seg={msh[-1]:.6f}")


def test_step_by_step_winding_change():
    """
    Fine-grained analysis of the exact step where winding changes.
    """
    print("\n" + "="*70)
    print("TEST: STEP-BY-STEP WINDING CHANGE ANALYSIS")
    print("="*70)
    
    k = 16
    T = 64 * k
    noise_level = 0.02
    
    x_target = create_winding_signal(T, k, radius=1.0)
    np.random.seed(42)
    x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
    
    hst = HeisenbergScatteringTransform(
        T, J=2, Q=2, max_order=1,
        lifting='radial_floor', epsilon=1e-8
    )
    
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Track signal state at each step
    print("\nRunning optimization with per-step signal capture...")
    
    signals_at_steps = []
    
    def capture_callback(step, x, loss, grad):
        signals_at_steps.append((step, x.copy()))
    
    result = optimize_signal(
        target_coeffs, hst, x0.copy(),
        n_steps=50,  # Shorter run, focused on change region
        lr=1e-8,
        momentum=0.0,
        normalize=True,
        loss_type='phase_robust',
        phase_lambda=1.0,
        verbose=False,
        callback=capture_callback,
    )
    
    # Find where winding changes
    print("\nStep-by-step winding analysis:")
    print(f"{'Step':>4} {'W':>10} {'min_seg':>12} {'min|z|':>12} {'Phase range':>15}")
    print("-" * 60)
    
    prev_w = None
    change_step = None
    
    for step, x in signals_at_steps:
        w = compute_winding_number(x)
        ms = min_segment_distance_to_origin(x, closed=True)
        min_z = np.min(np.abs(x))
        
        # Compute phase range
        phases = np.angle(x)
        phase_range = np.max(phases) - np.min(phases)
        
        w_int = round(w)
        if prev_w is not None and w_int != prev_w:
            print(f"{step:>4} {w:>10.4f} {ms:>12.6f} {min_z:>12.6f} {phase_range:>15.4f}  <-- CHANGE!")
            change_step = step
        else:
            print(f"{step:>4} {w:>10.4f} {ms:>12.6f} {min_z:>12.6f} {phase_range:>15.4f}")
        
        prev_w = w_int
    
    if change_step is not None:
        # Detailed comparison of step before and after change
        print(f"\n--- Detailed comparison around step {change_step} ---")
        
        idx = change_step
        x_before = signals_at_steps[idx-1][1]
        x_after = signals_at_steps[idx][1]
        
        print(f"\nPhase unwrapping comparison:")
        
        phases_before = np.unwrap(np.angle(x_before))
        phases_after = np.unwrap(np.angle(x_after))
        
        total_phase_before = phases_before[-1] - phases_before[0]
        total_phase_after = phases_after[-1] - phases_after[0]
        
        print(f"  Total unwrapped phase before: {total_phase_before:.4f} rad ({total_phase_before/(2*np.pi):.2f} turns)")
        print(f"  Total unwrapped phase after:  {total_phase_after:.4f} rad ({total_phase_after/(2*np.pi):.2f} turns)")
        
        # Check for large phase jumps in the signal
        print(f"\n  Max phase jump before: {np.max(np.abs(np.diff(np.angle(x_before)))):.4f} rad")
        print(f"  Max phase jump after:  {np.max(np.abs(np.diff(np.angle(x_after)))):.4f} rad")
        
        # Check signal difference
        diff = x_after - x_before
        print(f"\n  Signal change magnitude: {np.linalg.norm(diff):.6f}")
        print(f"  Max pointwise change:    {np.max(np.abs(diff)):.6f}")


def main():
    print("="*70)
    print("WINDING DIAGNOSTIC TESTS")
    print("Deep investigation of winding number changes during optimization")
    print("="*70)
    
    # Run tests
    test_winding_precision()
    test_winding_change_correlation()
    test_winding_with_topology_margin()
    test_step_by_step_winding_change()
    test_detailed_trajectory_analysis()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
