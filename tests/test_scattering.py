#!/usr/bin/env python3
"""
Tests for HST Scattering Transform

Tests the multi-layer architecture and verifies:
1. Path structure (λ₂ > λ₁ constraint)
2. Energy conservation across layers
3. Coefficient computation consistency
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
except:
    pass
sys.path.insert(0, '.')

from hst.scattering import (
    HeisenbergScatteringTransform,
    ScatteringOutput,
    hst_forward,
    hst_coefficients,
)
from hst.filter_bank import two_channel_paul_filterbank


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def record(self, name, passed, msg=""):
        self.results.append((name, passed, msg))
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            print(f"  ✗ {name}")
        if msg:
            print(f"      {msg}")


def test_construction():
    """HST can be constructed with various parameters."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    ok = hst.n_filters > 0 and hst.filter_info['pou_ok']
    return ok, f"{hst.n_filters} filters, PoU={hst.filter_info['pou_ok']}"


def test_forward_returns_output():
    """Forward transform returns ScatteringOutput."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    x = np.random.randn(512) + 1j * np.random.randn(512)
    output = hst.forward(x)
    ok = isinstance(output, ScatteringOutput) and len(output.paths) > 0
    return ok, f"{len(output.paths)} paths"


def test_path_ordering():
    """Paths satisfy λ₂ > λ₁ constraint."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    x = np.random.randn(512) + 1j * np.random.randn(512)
    output = hst.forward(x)
    
    violations = 0
    for path in output.paths.keys():
        if len(path) >= 2:
            for i in range(1, len(path)):
                if path[i] <= path[i-1]:
                    violations += 1
    
    return violations == 0, f"Order violations: {violations}"


def test_order_counts():
    """Correct number of paths per order."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    x = np.random.randn(512) + 1j * np.random.randn(512)
    output = hst.forward(x)
    
    n0 = len(output.order(0))
    n1 = len(output.order(1))
    n2 = len(output.order(2))
    
    # Order 0: 1 (father)
    # Order 1: n_mothers
    # Order 2: C(n_mothers, 2) = n*(n-1)/2 paths with λ₂ > λ₁
    expected_n1 = hst.n_mothers
    expected_n2 = hst.n_mothers * (hst.n_mothers - 1) // 2
    
    ok = (n0 == 1 and n1 == expected_n1 and n2 == expected_n2)
    return ok, f"Order 0: {n0}, Order 1: {n1}/{expected_n1}, Order 2: {n2}/{expected_n2}"


def test_energy_distribution():
    """Energy is distributed across orders."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    x = np.exp(1j * 2 * np.pi * 10 * np.arange(512) / 512) + 2.0
    output = hst.forward(x)
    
    energy = output.energy_by_order()
    total = sum(energy.values())
    
    # All orders should have some energy
    ok = all(e > 0 for e in energy.values()) and total > 0
    pcts = {k: v/total*100 for k, v in energy.items()}
    return ok, f"Energy: {pcts}"


def test_coefficient_shapes():
    """All coefficients have correct shape."""
    T = 512
    hst = HeisenbergScatteringTransform(T, J=4, Q=4, max_order=2)
    x = np.random.randn(T) + 1j * np.random.randn(T)
    output = hst.forward(x)
    
    wrong_shapes = 0
    for path, coef in output.paths.items():
        if coef.shape != (T,):
            wrong_shapes += 1
    
    return wrong_shapes == 0, f"Wrong shapes: {wrong_shapes}"


def test_feature_vector():
    """Feature vector has consistent length."""
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2)
    x1 = np.random.randn(512) + 1j * np.random.randn(512)
    x2 = np.random.randn(512) + 1j * np.random.randn(512)
    
    out1 = hst.forward(x1)
    out2 = hst.forward(x2)
    
    f1 = out1.coefficients_flat()
    f2 = out2.coefficients_flat()
    
    ok = len(f1) == len(f2) and len(f1) > 0
    return ok, f"Feature length: {len(f1)}"


def test_convenience_functions():
    """Convenience functions work."""
    x = np.random.randn(512) + 1j * np.random.randn(512)
    
    output = hst_forward(x, J=4, Q=4, max_order=2)
    features = hst_coefficients(x, J=4, Q=4, max_order=2)
    
    ok = isinstance(output, ScatteringOutput) and len(features) > 0
    return ok, f"Paths: {len(output.paths)}, Features: {len(features)}"


def test_different_max_orders():
    """Different max_order values work."""
    x = np.random.randn(512) + 1j * np.random.randn(512)
    
    results = []
    for max_order in [1, 2, 3]:
        hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=max_order)
        output = hst.forward(x)
        results.append((max_order, len(output.paths)))
    
    # More orders = more paths
    ok = results[0][1] < results[1][1] < results[2][1]
    msg = ", ".join(f"order {r[0]}: {r[1]} paths" for r in results)
    return ok, msg


def test_r_types():
    """Both R mapping types work."""
    x = np.random.randn(512) + 1j * np.random.randn(512) + 5.0  # Shift for log
    
    hst_simple = HeisenbergScatteringTransform(512, J=4, Q=4, r_type='simple')
    hst_glinsky = HeisenbergScatteringTransform(512, J=4, Q=4, r_type='glinsky')
    
    out1 = hst_simple.forward(x)
    out2 = hst_glinsky.forward(x)
    
    ok = len(out1.paths) == len(out2.paths)
    return ok, f"simple: {len(out1.paths)}, glinsky: {len(out2.paths)}"


def test_real_signal():
    """Works with real input."""
    x = np.cos(2 * np.pi * 10 * np.arange(512) / 512)
    
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2, lifting='adaptive')
    output = hst.forward(x)
    
    ok = len(output.paths) > 0
    return ok, f"{len(output.paths)} paths"


def test_inverse_reconstruction():
    """Forward-inverse round-trip is exact."""
    x = np.exp(1j * 2 * np.pi * 10 * np.arange(512) / 512) + 2.0
    
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2, lifting='adaptive')
    output = hst.forward(x)
    x_rec = hst.inverse(output)
    
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    ok = error < 1e-10
    return ok, f"Error: {error:.2e}"


def test_inverse_real_signal():
    """Inverse works for real signals with analytic lifting."""
    x = np.cos(2 * np.pi * 10 * np.arange(512) / 512)
    
    hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2, lifting='analytic')
    output = hst.forward(x)
    x_rec = hst.inverse(output).real  # Real part for analytic
    
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    ok = error < 1e-10
    return ok, f"Error: {error:.2e}"


def test_inverse_van_der_pol():
    """Inverse works for Van der Pol trajectory."""
    T = 512
    # Simulate limit cycle
    t_sim = np.linspace(0, 20*np.pi, 2000)
    x_vdp_raw = 2.0 * np.exp(1j * t_sim)  # Approximate limit cycle
    x = np.interp(np.linspace(0, len(x_vdp_raw)-1, T), 
                   np.arange(len(x_vdp_raw)), x_vdp_raw.real) + \
        1j * np.interp(np.linspace(0, len(x_vdp_raw)-1, T),
                       np.arange(len(x_vdp_raw)), x_vdp_raw.imag)
    
    hst = HeisenbergScatteringTransform(T, J=4, Q=4, max_order=2, lifting='adaptive')
    output = hst.forward(x)
    x_rec = hst.inverse(output)
    
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    ok = error < 1e-10
    return ok, f"Error: {error:.2e}"


def test_all_lifting_strategies():
    """All lifting strategies give good reconstruction."""
    x = np.sin(2 * np.pi * 5 * np.arange(512) / 512)  # Real signal crossing zero
    
    results = []
    for lifting in ['shift', 'adaptive', 'analytic']:
        hst = HeisenbergScatteringTransform(512, J=4, Q=4, max_order=2, 
                                             lifting=lifting, epsilon=1.0)
        output = hst.forward(x)
        x_rec = hst.inverse(output)
        if lifting == 'analytic':
            x_rec = x_rec.real
        error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        results.append((lifting, error < 1e-10))
    
    ok = all(r[1] for r in results)
    return ok, ", ".join(f"{r[0]}:{'✓' if r[1] else '✗'}" for r in results)


def main():
    print("=" * 70)
    print("HST SCATTERING TESTS")
    print("=" * 70)
    
    result = TestResult()
    
    print("\n[CONSTRUCTION]")
    print("-" * 50)
    p, m = test_construction()
    result.record("construction", p, m)
    
    print("\n[FORWARD TRANSFORM]")
    print("-" * 50)
    p, m = test_forward_returns_output()
    result.record("forward_returns_output", p, m)
    
    p, m = test_path_ordering()
    result.record("path_ordering", p, m)
    
    p, m = test_order_counts()
    result.record("order_counts", p, m)
    
    p, m = test_energy_distribution()
    result.record("energy_distribution", p, m)
    
    p, m = test_coefficient_shapes()
    result.record("coefficient_shapes", p, m)
    
    print("\n[FEATURE EXTRACTION]")
    print("-" * 50)
    p, m = test_feature_vector()
    result.record("feature_vector", p, m)
    
    print("\n[API]")
    print("-" * 50)
    p, m = test_convenience_functions()
    result.record("convenience_functions", p, m)
    
    p, m = test_different_max_orders()
    result.record("different_max_orders", p, m)
    
    p, m = test_r_types()
    result.record("r_types", p, m)
    
    p, m = test_real_signal()
    result.record("real_signal", p, m)
    
    print("\n[INVERSE RECONSTRUCTION]")
    print("-" * 50)
    
    p, m = test_inverse_reconstruction()
    result.record("inverse_reconstruction", p, m)
    
    p, m = test_inverse_real_signal()
    result.record("inverse_real_signal", p, m)
    
    p, m = test_inverse_van_der_pol()
    result.record("inverse_van_der_pol", p, m)
    
    p, m = test_all_lifting_strategies()
    result.record("all_lifting_strategies", p, m)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {result.passed}/{result.passed + result.failed}")
    
    if result.failed == 0:
        print("\n✓ ALL SCATTERING TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
