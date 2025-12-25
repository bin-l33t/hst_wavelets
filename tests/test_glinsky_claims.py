#!/usr/bin/env python3
"""
Test Suite for HST Wavelets

Tests organized by Glinsky's claims from glinsky_collective.pdf.
Runs standalone without pytest.

Usage:
    python tests/test_glinsky_claims.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path (works from any directory)
try:
    _this_file = Path(__file__).resolve()
    _parent = _this_file.parent.parent
    sys.path.insert(0, str(_parent))
except NameError:
    # __file__ not defined (e.g., in interactive mode)
    sys.path.insert(0, '.')

from hst.filter_bank import (
    two_channel_paul_filterbank,
    forward_transform,
    inverse_transform,
)
from hst.conformal import (
    glinsky_R,
    glinsky_R_inverse,
    simple_R,
    verify_R_inverse,
)


# =============================================================================
# Simple test runner
# =============================================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, name, passed, error=None):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  ✗ {name}")
            if error:
                print(f"    Error: {error}")


def run_test(result, name, test_fn):
    """Run a single test and record result."""
    try:
        test_fn()
        result.record(name, True)
    except AssertionError as e:
        result.record(name, False, str(e))
    except Exception as e:
        result.record(name, False, f"{type(e).__name__}: {e}")


# =============================================================================
# CLAIM 1: Partition of Unity (Glinsky Section VII, Page 15)
# =============================================================================

def test_pou_equals_one():
    """Partition of unity should equal 1 at all frequencies."""
    filters, info = two_channel_paul_filterbank(512, 4, 4)
    assert info['pou_ok'], (
        f"Partition of unity failed: "
        f"min={info['pou_min']:.6f}, max={info['pou_max']:.6f}"
    )


def test_pou_various_sizes():
    """PoU should hold for various signal lengths."""
    for T in [128, 256, 512, 1024]:
        filters, info = two_channel_paul_filterbank(T, 4, 4)
        assert info['pou_ok'], f"Failed for T={T}"


def test_pou_various_parameters():
    """PoU should hold for various J, Q, m."""
    for J in [2, 4, 6]:
        for Q in [2, 4, 8]:
            for m in [1, 2, 4]:
                filters, info = two_channel_paul_filterbank(512, J, Q, m)
                assert info['pou_ok'], f"Failed for J={J}, Q={Q}, m={m}"


# =============================================================================
# CLAIM 2: Invertibility (Glinsky Section VII, Page 15)
# =============================================================================

def test_analytic_signal():
    """Reconstruction of analytic (H⁺) signal."""
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    T = 512
    t = np.arange(T)
    x = np.exp(1j * 2 * np.pi * 10 * t / T)
    
    # Make strictly analytic
    x_hat = np.fft.fft(x)
    x_hat[T//2:] = 0
    x = np.fft.ifft(x_hat)
    
    x_rec = inverse_transform(forward_transform(x, filters), filters)
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    
    assert error < 1e-10, f"Analytic signal error: {error}"


def test_real_signal():
    """Reconstruction of real signal (symmetric spectrum)."""
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    T = 512
    t = np.arange(T)
    x = np.sin(2 * np.pi * 10 * t / T) + 0.5 * np.cos(2 * np.pi * 25 * t / T)
    
    x_rec = inverse_transform(forward_transform(x, filters), filters)
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    
    assert error < 1e-10, f"Real signal error: {error}"


def test_broadband_random():
    """CRITICAL: Reconstruction of broadband random signal."""
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    np.random.seed(42)
    x = np.random.randn(512) + 1j * np.random.randn(512)
    
    x_rec = inverse_transform(forward_transform(x, filters), filters)
    error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    
    assert error < 1e-10, f"Broadband random error: {error}"


# =============================================================================
# CLAIM 3: R Mapping Invertibility
# =============================================================================

def test_R_inverse_roundtrip():
    """R⁻¹(R(z)) = z."""
    result = verify_R_inverse(n_samples=1000)
    assert result['passed'], f"R inverse failed: max_error={result['max_error']}"


def test_R_inverse_various_inputs():
    """R inverse works for various input domains."""
    np.random.seed(42)
    
    test_cases = [
        np.random.randn(100) + 1j * np.random.randn(100) + 2.0,
        np.random.randn(100) + 1j * np.random.randn(100) + 0.5j + 1.0,
        np.abs(np.random.randn(100)) + 0.1 + 1j * np.random.randn(100),
    ]
    
    for i, z in enumerate(test_cases):
        w = glinsky_R(z)
        z_rec = glinsky_R_inverse(w)
        error = np.abs(z - z_rec).max()
        assert error < 1e-6, f"Case {i}: R inverse failed with error {error}"


# =============================================================================
# CLAIM 4: Reconstruction After R Transform
# =============================================================================

def test_R_shifted_signal():
    """Reconstruction of R(shifted analytic signal)."""
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    T = 512
    t = np.arange(T)
    x = np.exp(1j * 2 * np.pi * 10 * t / T) + 2.0
    
    y = glinsky_R(x)
    y_rec = inverse_transform(forward_transform(y, filters), filters)
    error = np.linalg.norm(y - y_rec) / np.linalg.norm(y)
    
    assert error < 1e-10, f"R(shifted) error: {error}"


def test_R_oscillator_hardcase():
    """
    CRITICAL: Pure oscillator through R.
    This was the failing case with single-channel (66% error).
    """
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    T = 512
    t = np.arange(T)
    x = 2.0 * np.exp(1j * 2 * np.pi * 20 * t / T)
    
    y = glinsky_R(x)
    
    # Check spectrum distribution
    y_hat = np.fft.fft(y)
    k = np.fft.fftfreq(T)
    en_pos = np.sum(np.abs(y_hat[k >= 0])**2)
    en_neg = np.sum(np.abs(y_hat[k < 0])**2)
    neg_ratio = en_neg / (en_pos + en_neg)
    
    assert neg_ratio > 0.3, f"Expected significant neg freq, got {neg_ratio:.2%}"
    
    y_rec = inverse_transform(forward_transform(y, filters), filters)
    error = np.linalg.norm(y - y_rec) / np.linalg.norm(y)
    
    assert error < 1e-10, f"R(oscillator) error: {error}"


def test_simple_R_comparison():
    """Compare Glinsky R vs simple i·ln(z)."""
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    T = 512
    t = np.arange(T)
    x = np.exp(1j * 2 * np.pi * 10 * t / T) + 2.0
    
    y_glinsky = glinsky_R(x)
    y_simple = simple_R(x)
    
    for y, name in [(y_glinsky, 'Glinsky'), (y_simple, 'Simple')]:
        y_rec = inverse_transform(forward_transform(y, filters), filters)
        error = np.linalg.norm(y - y_rec) / np.linalg.norm(y)
        assert error < 1e-10, f"{name} R error: {error}"


# =============================================================================
# CLAIM 5: Spectrum Documentation
# =============================================================================

def test_spectrum_documentation():
    """Document spectrum changes under R (always passes)."""
    T = 512
    t = np.arange(T)
    k = np.fft.fftfreq(T)
    
    test_cases = [
        ("Analytic + shift", np.exp(1j * 2 * np.pi * 10 * t / T) + 2.0),
        ("Pure oscillator", 2.0 * np.exp(1j * 2 * np.pi * 20 * t / T)),
        ("AM signal", (1 + 0.3*np.cos(2*np.pi*2*t/T)) * np.exp(1j * 2 * np.pi * 10 * t / T)),
    ]
    
    print("\n    Spectrum after R transform:")
    for name, x in test_cases:
        y = glinsky_R(x)
        y_hat = np.fft.fft(y)
        en_pos = np.sum(np.abs(y_hat[k >= 0])**2)
        en_neg = np.sum(np.abs(y_hat[k < 0])**2)
        total = en_pos + en_neg
        print(f"      {name:20s}: {en_pos/total*100:5.1f}% pos, {en_neg/total*100:5.1f}% neg")


# =============================================================================
# Performance
# =============================================================================

def test_performance():
    """Basic performance check."""
    import time
    
    filters, _ = two_channel_paul_filterbank(512, 4, 4)
    x = np.random.randn(512) + 1j * np.random.randn(512)
    
    start = time.time()
    for _ in range(100):
        coeffs = forward_transform(x, filters)
        x_rec = inverse_transform(coeffs, filters)
    elapsed = time.time() - start
    
    assert elapsed < 2.0, f"Transform too slow: {elapsed:.2f}s for 100 iterations"
    print(f"\n    Performance: 100 round-trips in {elapsed:.3f}s")


# =============================================================================
# Main runner
# =============================================================================

def main():
    print("=" * 70)
    print("HST WAVELETS TEST SUITE")
    print("Testing Glinsky's claims from glinsky_collective.pdf")
    print("=" * 70)
    
    result = TestResult()
    
    # CLAIM 1: Partition of Unity
    print("\n[CLAIM 1] Partition of Unity (Section VII, Page 15)")
    print("-" * 50)
    run_test(result, "pou_equals_one", test_pou_equals_one)
    run_test(result, "pou_various_sizes", test_pou_various_sizes)
    run_test(result, "pou_various_parameters", test_pou_various_parameters)
    
    # CLAIM 2: Invertibility
    print("\n[CLAIM 2] Invertibility (Section VII, Page 15)")
    print("-" * 50)
    run_test(result, "analytic_signal", test_analytic_signal)
    run_test(result, "real_signal", test_real_signal)
    run_test(result, "broadband_random", test_broadband_random)
    
    # CLAIM 3: R Mapping
    print("\n[CLAIM 3] R Mapping Invertibility (Section III)")
    print("-" * 50)
    run_test(result, "R_inverse_roundtrip", test_R_inverse_roundtrip)
    run_test(result, "R_inverse_various_inputs", test_R_inverse_various_inputs)
    
    # CLAIM 4: Reconstruction after R
    print("\n[CLAIM 4] Reconstruction After R Transform")
    print("-" * 50)
    run_test(result, "R_shifted_signal", test_R_shifted_signal)
    run_test(result, "R_oscillator_hardcase", test_R_oscillator_hardcase)
    run_test(result, "simple_R_comparison", test_simple_R_comparison)
    
    # CLAIM 5: Spectrum documentation
    print("\n[CLAIM 5] Spectrum Properties (Documentation)")
    print("-" * 50)
    run_test(result, "spectrum_documentation", test_spectrum_documentation)
    
    # Performance
    print("\n[PERFORMANCE]")
    print("-" * 50)
    run_test(result, "performance", test_performance)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    
    if result.failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED:")
        for name, error in result.errors:
            print(f"  - {name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
