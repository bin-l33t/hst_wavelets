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
    compute_winding_atan2,
    detect_winding_change,
    check_topology_invariant,
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
    margin = 0.05  # Use a reasonable margin
    
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
        
        # With topology margin (now homotopy-aware!)
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


def test_homotopy_guard():
    """
    Verify that homotopy-aware guard catches tunneling that endpoint checks miss.
    """
    print("\n" + "="*70)
    print("TEST: HOMOTOPY-AWARE GUARD (ChatGPT's insight)")
    print("="*70)
    
    from hst.optimize_numpy import min_seg_along_homotopy
    
    # Create a synthetic case where endpoints are far but path crosses near origin
    T = 64
    t = np.arange(T)
    
    # x_old: circle far from origin
    x_old = 2.0 + 0.5 * np.exp(1j * 2 * np.pi * t / T)
    
    # x_new: circle on opposite side, also far from origin
    x_new = -2.0 + 0.5 * np.exp(1j * 2 * np.pi * t / T)
    
    # Endpoint checks
    min_seg_old = min_segment_distance_to_origin(x_old, closed=True)
    min_seg_new = min_segment_distance_to_origin(x_new, closed=True)
    
    # Homotopy check
    homotopy_min, worst_s = min_seg_along_homotopy(x_old, x_new)
    
    print(f"\nSynthetic tunneling test:")
    print(f"  x_old: circle centered at +2 (min_seg = {min_seg_old:.4f})")
    print(f"  x_new: circle centered at -2 (min_seg = {min_seg_new:.4f})")
    print(f"  Homotopy path minimum: {homotopy_min:.4f} at s={worst_s:.2f}")
    
    # The homotopy should pass through/near origin around s=0.5
    if homotopy_min < 0.1 and min_seg_old > 1.0 and min_seg_new > 1.0:
        print(f"  ✓ Homotopy guard correctly detects tunneling!")
        print(f"    (Endpoints look safe, but path crosses near origin)")
    else:
        print(f"  ✗ Test setup issue or guard not working")
    
    # Now test on actual optimization case
    print(f"\nReal optimization test (k=16, l2):")
    
    k = 16
    T = 64 * k
    x_target = create_winding_signal(T, k)
    np.random.seed(42)
    x0 = x_target + 0.02 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    hst = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=1, lifting='radial_floor', epsilon=1e-8)
    target_coeffs = extract_all_targets(hst, x_target)
    
    # Without margin
    result_no_margin = optimize_signal(
        target_coeffs, hst, x0.copy(), n_steps=100, lr=1e-8,
        normalize=True, loss_type='l2', verbose=False
    )
    
    # With homotopy-aware margin
    result_with_margin = optimize_signal(
        target_coeffs, hst, x0.copy(), n_steps=100, lr=1e-8,
        normalize=True, loss_type='l2', topology_margin=0.05, verbose=False
    )
    
    w_initial = round(result_no_margin.winding_history[0])
    w_final_no = round(result_no_margin.winding_history[-1])
    w_final_with = round(result_with_margin.winding_history[-1])
    
    print(f"  Initial winding: {w_initial}")
    print(f"  Final winding (no margin): {w_final_no}")
    print(f"  Final winding (with homotopy guard): {w_final_with}")
    
    if w_initial == w_final_with and w_initial != w_final_no:
        print(f"  ✓ Homotopy guard preserved winding!")
    elif w_initial == w_final_with == w_final_no:
        print(f"  ⚠ Both preserved winding (margin may not have been needed)")
    else:
        print(f"  ✗ Homotopy guard did not preserve winding")


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


def test_winding_consistency():
    """
    ChatGPT-requested diagnostic: prove consistency between winding methods.
    
    1. Compute winding two ways (diff vs atan2) and flag disagreements
    2. Log exact min_seg at the step where winding changes
    3. Run topology invariant check
    """
    print("\n" + "="*70)
    print("TEST: WINDING CONSISTENCY (ChatGPT Diagnostic)")
    print("="*70)
    
    from hst.optimize_numpy import (
        compute_winding_number,
        compute_winding_atan2,
        check_topology_invariant,
        _compute_winding_inline,
    )
    
    # Focus on the cases that showed anomalies
    test_cases = [
        (16, 'phase_robust'),
        (32, 'phase_robust'),
    ]
    
    base_T = 64
    noise_level = 0.02
    MARGIN = 0.1
    
    for k, loss_type in test_cases:
        print(f"\n{'='*60}")
        print(f"k={k}, loss={loss_type}")
        print(f"{'='*60}")
        
        T = base_T * k
        
        x_target = create_winding_signal(T, k, radius=1.0)
        np.random.seed(42)
        x0 = x_target + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))
        
        hst = HeisenbergScatteringTransform(
            T, J=2, Q=2, max_order=1,
            lifting='radial_floor', epsilon=1e-8
        )
        
        target_coeffs = extract_all_targets(hst, x_target)
        
        # Capture ALL signals for detailed analysis
        all_signals = []
        
        def capture_all(step, x, loss, grad):
            all_signals.append((step, x.copy()))
        
        result = optimize_signal(
            target_coeffs, hst, x0.copy(),
            n_steps=100,
            lr=1e-8,
            momentum=0.0,
            normalize=True,
            loss_type=loss_type,
            phase_lambda=1.0,
            verbose=False,
            callback=capture_all,
        )
        
        # 1. Compare winding methods at every step
        print("\n1. WINDING METHOD COMPARISON (diff vs atan2)")
        print("-" * 50)
        
        disagreements = []
        winding_diff_history = []
        winding_atan2_history = []
        min_seg_at_step = []
        
        for step, x in all_signals:
            w_diff = _compute_winding_inline(x)
            w_atan2 = compute_winding_atan2(x)
            ms = min_segment_distance_to_origin(x, closed=True)
            
            winding_diff_history.append(w_diff)
            winding_atan2_history.append(w_atan2)
            min_seg_at_step.append(ms)
            
            if abs(w_diff - w_atan2) > 0.1:
                disagreements.append((step, w_diff, w_atan2, ms))
        
        if disagreements:
            print(f"  ⚠ {len(disagreements)} DISAGREEMENTS found:")
            for step, wd, wa, ms in disagreements[:5]:
                print(f"    Step {step}: diff={wd:.4f}, atan2={wa:.4f}, min_seg={ms:.6f}")
        else:
            print(f"  ✓ All {len(all_signals)} steps agree (diff ≈ atan2)")
        
        # 2. Find winding change points and log 5-step window
        print("\n2. WINDING CHANGE POINTS (5-step window)")
        print("-" * 50)
        
        prev_w = round(winding_diff_history[0])
        change_steps = []
        
        for i, w in enumerate(winding_diff_history):
            curr_w = round(w)
            if curr_w != prev_w:
                change_steps.append((i, prev_w, curr_w))
                prev_w = curr_w
        
        if not change_steps:
            print(f"  No winding changes detected. Final winding: {round(winding_diff_history[-1])}")
        else:
            print(f"  {len(change_steps)} winding changes detected")
            
            # Show first 3 changes with 5-step windows
            for change_idx, (step, w_before, w_after) in enumerate(change_steps[:3]):
                print(f"\n  Change #{change_idx+1}: Step {step}, W: {w_before} → {w_after}")
                print(f"  {'Step':>6} {'W_diff':>10} {'W_atan2':>10} {'min_seg':>12} {'Note':>10}")
                print(f"  {'-'*50}")
                
                # 5-step window around change
                start = max(0, step - 2)
                end = min(len(all_signals), step + 3)
                
                for i in range(start, end):
                    wd = winding_diff_history[i]
                    wa = winding_atan2_history[i]
                    ms = min_seg_at_step[i]
                    note = "<-- CHANGE" if i == step else ""
                    print(f"  {i:>6} {wd:>10.4f} {wa:>10.4f} {ms:>12.6f} {note:>10}")
        
        # 3. Run topology invariant check
        print("\n3. TOPOLOGY INVARIANT CHECK")
        print("-" * 50)
        
        invariant_result = check_topology_invariant(
            winding_diff_history,
            min_seg_at_step,
            margin=MARGIN
        )
        
        if invariant_result['passed']:
            print(f"  ✓ PASSED: No violations with margin={MARGIN}")
            print(f"    (Winding only changes when min_seg < {MARGIN})")
        else:
            print(f"  ✗ FAILED: {invariant_result['n_violations']} violations with margin={MARGIN}")
            for step, w1, w2, ms1, ms2 in invariant_result['violations'][:3]:
                print(f"    Step {step}→{step+1}: W={w1}→{w2}, min_seg={ms1:.6f}→{ms2:.6f}")
            print(f"    This indicates a BUG: winding changed without origin crossing!")
        
        # Summary
        print(f"\n  SUMMARY for k={k} {loss_type}:")
        print(f"    Initial winding: {round(winding_diff_history[0])}")
        print(f"    Final winding: {round(winding_diff_history[-1])}")
        print(f"    Method agreement: {'YES' if not disagreements else 'NO'}")
        print(f"    Invariant passed: {'YES' if invariant_result['passed'] else 'NO'}")


def main():
    print("="*70)
    print("WINDING DIAGNOSTIC TESTS")
    print("Deep investigation of winding number changes during optimization")
    print("="*70)
    
    # Run tests
    test_winding_precision()
    test_winding_consistency()  # ChatGPT's requested diagnostic
    test_homotopy_guard()  # NEW: Test the homotopy-aware guard
    test_winding_change_correlation()
    test_winding_with_topology_margin()
    test_detailed_trajectory_analysis()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
