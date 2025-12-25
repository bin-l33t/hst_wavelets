#!/usr/bin/env python3
"""
Physical Systems Tests for HST Wavelets

Based on:
- Lopatin (1996) "Symmetry in Nonlinear Mechanics"  
- Glinsky (2025) "Collective Fields and Heisenberg Scattering"

Tests physical systems where Lie group structure is explicit:
- SO(2): Harmonic oscillator, Van der Pol, Duffing
- SO(3): Motion on sphere

Verifies claims about:
1. Group averaging / Bogolyubov projection
2. Separation of fast/slow variables
3. R mapping linearizes phase dynamics
4. Two-channel filter bank handles all cases
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except:
    sys.path.insert(0, '.')

from hst.filter_bank import (
    two_channel_paul_filterbank, 
    forward_transform, 
    inverse_transform
)
from hst.conformal import simple_R, simple_R_unwrapped


# =============================================================================
# Physical System Simulators
# =============================================================================

def simulate_harmonic(T, dt=0.01, omega=1.0, rho0=1.0):
    """
    Simple harmonic oscillator - pure SO(2) motion.
    
    In Lopatin's coordinates (Eq. 25):
        x₁ = ρ sin φ
        x₂ = ρ cos φ
        
    Unperturbed: ρ̇ = 0, φ̇ = ω
    
    Returns progressive signal z = ρ exp(iωt) for H⁺ analysis.
    """
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    z = rho0 * np.exp(1j * omega * t)
    return z, t


def simulate_van_der_pol(T, dt=0.01, eps=0.1, x0=None):
    """
    Van der Pol oscillator (Lopatin Eq. 19):
        ẍ + x = ε(1 - x²)ẋ
        
    First order form:
        ẋ₁ = x₂
        ẋ₂ = -x₁ + ε(1 - x₁²)x₂
        
    Has limit cycle at ρ = 2 for any ε > 0.
    Averaged equation (Lopatin): ρ̇ = (ε/2)(1 - ρ²/4)ρ
    """
    n_steps = int(T / dt)
    if x0 is None:
        x0 = np.array([0.1, 0.0])
    
    x = np.zeros((n_steps, 2))
    x[0] = x0
    
    for i in range(1, n_steps):
        x1, x2 = x[i-1]
        # RK4
        def f(s):
            return np.array([s[1], -s[0] + eps * (1 - s[0]**2) * s[1]])
        k1 = f(x[i-1])
        k2 = f(x[i-1] + dt/2 * k1)
        k3 = f(x[i-1] + dt/2 * k2)
        k4 = f(x[i-1] + dt * k3)
        x[i] = x[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    z = x[:, 0] + 1j * x[:, 1]
    rho = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    phi = np.arctan2(x[:, 0], x[:, 1])  # Lopatin convention
    
    return z, x, rho, phi


def simulate_duffing(T, dt=0.01, eps=0.1, x0=None):
    """
    Duffing equation (Lopatin Eq. 27):
        ÿ + y + εy³ = 0
        
    First order form:
        ẏ₁ = y₂
        ẏ₂ = -y₁ - εy₁³
        
    Averaged equation: ρ̇ = 0, φ̇ = 1 + ε(3/8)ρ²
    Frequency increases with amplitude.
    """
    n_steps = int(T / dt)
    if x0 is None:
        x0 = np.array([1.0, 0.0])
    
    x = np.zeros((n_steps, 2))
    x[0] = x0
    
    for i in range(1, n_steps):
        def f(s):
            return np.array([s[1], -s[0] - eps * s[0]**3])
        k1 = f(x[i-1])
        k2 = f(x[i-1] + dt/2 * k1)
        k3 = f(x[i-1] + dt/2 * k2)
        k4 = f(x[i-1] + dt * k3)
        x[i] = x[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    z = x[:, 0] + 1j * x[:, 1]
    return z, x


def simulate_sphere(T, dt=0.01, eps=0.1, x0=None):
    """
    Motion on sphere (Lopatin Eq. 31):
        ẋ₁ = x₂ - ε(x₁² - x₃²)
        ẋ₂ = x₃ - x₁ - 2ε(x₁x₂ + x₂x₃)
        ẋ₃ = -x₂ + ε(x₂ + (x₁² - x₃²))
        
    SO(3) symmetry. Slow variable: y₁ = x₁ + x₃
    """
    n_steps = int(T / dt)
    if x0 is None:
        x0 = np.array([0.5, 0.5, np.sqrt(0.5)])
        x0 = x0 / np.linalg.norm(x0)
    
    x = np.zeros((n_steps, 3))
    x[0] = x0
    
    for i in range(1, n_steps):
        def f(s):
            x1, x2, x3 = s
            return np.array([
                x2 - eps * (x1**2 - x3**2),
                x3 - x1 - 2*eps * (x1*x2 + x2*x3),
                -x2 + eps * (x2 + (x1**2 - x3**2))
            ])
        k1 = f(x[i-1])
        k2 = f(x[i-1] + dt/2 * k1)
        k3 = f(x[i-1] + dt/2 * k2)
        k4 = f(x[i-1] + dt * k3)
        x[i] = x[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        x[i] = x[i] / np.linalg.norm(x[i])  # Project to sphere
    
    return x


# =============================================================================
# Test Runner
# =============================================================================

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


def resample(z, T_target):
    """Resample complex signal to target length."""
    t_old = np.arange(len(z))
    t_new = np.linspace(0, len(z)-1, T_target)
    return np.interp(t_new, t_old, z.real) + 1j * np.interp(t_new, t_old, z.imag)


# =============================================================================
# Tests
# =============================================================================

def test_harmonic_is_progressive():
    """Harmonic oscillator z = ρ exp(iωt) is in H⁺."""
    z, _ = simulate_harmonic(T=100)
    z_hat = np.fft.fft(z)
    k = np.fft.fftfreq(len(z))
    en_pos = np.sum(np.abs(z_hat[k > 0])**2)
    en_neg = np.sum(np.abs(z_hat[k < 0])**2)
    ratio = en_pos / (en_pos + en_neg)
    return ratio > 0.99, f"H⁺ content: {ratio*100:.1f}%"


def test_van_der_pol_limit_cycle():
    """Van der Pol approaches ρ = 2 limit cycle."""
    _, _, rho, _ = simulate_van_der_pol(T=300, eps=0.3)
    final_rho = np.mean(rho[-1000:])
    return np.abs(final_rho - 2.0) < 0.2, f"Final ρ = {final_rho:.3f}"


def test_van_der_pol_reconstruction():
    """Two-channel bank reconstructs Van der Pol perfectly."""
    z, _, _, _ = simulate_van_der_pol(T=100, eps=0.1)
    z_res = resample(z, 512)
    
    filters, _ = two_channel_paul_filterbank(512, J=4, Q=4)
    coeffs = forward_transform(z_res, filters)
    z_rec = inverse_transform(coeffs, filters)
    
    err = np.linalg.norm(z_res - z_rec) / np.linalg.norm(z_res)
    return err < 1e-10, f"Error: {err:.2e}"


def test_van_der_pol_after_R():
    """R(Van der Pol) is still reconstructible."""
    z, _, _, _ = simulate_van_der_pol(T=100, eps=0.1)
    z_res = resample(z, 512)
    
    # Shift away from origin for log
    y = simple_R(z_res + 5.0)
    
    filters, _ = two_channel_paul_filterbank(512, J=4, Q=4)
    coeffs = forward_transform(y, filters)
    y_rec = inverse_transform(coeffs, filters)
    
    err = np.linalg.norm(y - y_rec) / np.linalg.norm(y)
    return err < 1e-10, f"Error: {err:.2e}"


def test_duffing_frequency_increases():
    """Duffing: frequency increases with amplitude (Lopatin Eq. 28)."""
    z_small, _ = simulate_duffing(T=100, eps=0.3, x0=np.array([0.5, 0.0]))
    z_large, _ = simulate_duffing(T=100, eps=0.3, x0=np.array([1.5, 0.0]))
    
    def count_periods(z):
        crossings = np.where(np.diff(np.sign(z.real)))[0]
        if len(crossings) > 4:
            return len(crossings) // 2
        return 0
    
    periods_small = count_periods(z_small[1000:])
    periods_large = count_periods(z_large[1000:])
    
    return periods_large > periods_small, f"Periods: small={periods_small}, large={periods_large}"


def test_r_linearizes_phase():
    """R mapping makes phase evolution linear (Glinsky's geodesic claim)."""
    z, t = simulate_harmonic(T=100, omega=1.0, rho0=2.0)
    
    # Use unwrapped R for continuous dynamics
    y = simple_R_unwrapped(z)
    
    # Real part should be -ωt (linear)
    coeffs = np.polyfit(t, y.real, 1)
    residual = np.std(y.real - np.polyval(coeffs, t)) / np.std(y.real)
    
    # Imaginary part should be ln(ρ) (constant)
    imag_std = np.std(y.imag)
    
    linear_ok = residual < 0.001
    const_ok = imag_std < 0.001
    
    return linear_ok and const_ok, f"Re slope: {coeffs[0]:.4f}, residual: {residual:.4f}"


def test_sphere_constraint():
    """SO(3) motion stays on unit sphere."""
    x = simulate_sphere(T=100, eps=0.1)
    radii = np.linalg.norm(x, axis=1)
    max_dev = np.max(np.abs(radii - 1.0))
    return max_dev < 1e-10, f"Max deviation: {max_dev:.2e}"


def test_sphere_slow_variable():
    """Lopatin's slow variable y₁ = x₁ + x₃ is approximately conserved."""
    x = simulate_sphere(T=100, eps=0.1)
    y1 = x[:, 0] + x[:, 2]
    rel_var = np.std(y1) / np.abs(np.mean(y1))
    return rel_var < 0.05, f"Relative variation: {rel_var*100:.2f}%"


def test_so3_lie_algebra():
    """Verify [U₁, U₂] = U₃ for so(3) algebra."""
    def apply_U1(f, x, h=1e-6):
        x1, x2, x3 = x
        df1 = (f(x1+h, x2, x3) - f(x1-h, x2, x3)) / (2*h)
        df2 = (f(x1, x2+h, x3) - f(x1, x2-h, x3)) / (2*h)
        return x2 * df1 - x1 * df2
    
    def apply_U2(f, x, h=1e-6):
        x1, x2, x3 = x
        df2 = (f(x1, x2+h, x3) - f(x1, x2-h, x3)) / (2*h)
        df3 = (f(x1, x2, x3+h) - f(x1, x2, x3-h)) / (2*h)
        return x3 * df2 - x2 * df3
    
    def apply_U3(f, x, h=1e-6):
        x1, x2, x3 = x
        df1 = (f(x1+h, x2, x3) - f(x1-h, x2, x3)) / (2*h)
        df3 = (f(x1, x2, x3+h) - f(x1, x2, x3-h)) / (2*h)
        return x1 * df3 - x3 * df1
    
    test_f = lambda x1, x2, x3: x1**2 + x2*x3
    test_point = (0.5, 0.6, 0.7)
    
    def U2_f(x1, x2, x3):
        return apply_U2(test_f, (x1, x2, x3))
    def U1_f(x1, x2, x3):
        return apply_U1(test_f, (x1, x2, x3))
    
    comm_12 = apply_U1(U2_f, test_point) - apply_U2(U1_f, test_point)
    U3_f = apply_U3(test_f, test_point)
    
    return np.abs(comm_12 - U3_f) < 1e-6, f"[U₁,U₂] - U₃ = {comm_12 - U3_f:.2e}"


def test_bogolyubov_projection():
    """Group averaging on SO(2) annihilates oscillatory terms."""
    phi = np.linspace(0, 2*np.pi, 1000)
    
    # cos(nφ) and sin(nφ) average to 0 for n ≠ 0
    avg_cos = np.mean(np.cos(phi))
    avg_sin = np.mean(np.sin(phi))
    avg_cos2 = np.mean(np.cos(2*phi))
    
    # ρ-dependent terms survive
    rho = 2.0
    # <ρ² sin²φ> = ρ²/2
    avg_sin2 = np.mean(rho**2 * np.sin(phi)**2)
    expected = rho**2 / 2
    
    all_ok = (np.abs(avg_cos) < 0.01 and 
              np.abs(avg_sin) < 0.01 and 
              np.abs(avg_cos2) < 0.01 and
              np.abs(avg_sin2 - expected) < 0.01)
    
    return all_ok, f"<sin>={avg_sin:.3f}, <cos>={avg_cos:.3f}, <ρ²sin²>={avg_sin2:.3f}"


def test_two_channel_full_spectrum():
    """Two-channel bank has partition of unity on full L²."""
    filters, info = two_channel_paul_filterbank(512, J=4, Q=4)
    return info['pou_ok'], f"PoU: [{info['pou_min']:.6f}, {info['pou_max']:.6f}]"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PHYSICAL SYSTEMS TESTS")
    print("Based on Lopatin (1996) and Glinsky (2025)")
    print("=" * 70)
    
    result = TestResult()
    
    print("\n[SO(2) SYMMETRY - Oscillators]")
    print("-" * 50)
    
    p, m = test_harmonic_is_progressive()
    result.record("harmonic_is_progressive", p, m)
    
    p, m = test_van_der_pol_limit_cycle()
    result.record("van_der_pol_limit_cycle", p, m)
    
    p, m = test_van_der_pol_reconstruction()
    result.record("van_der_pol_reconstruction", p, m)
    
    p, m = test_van_der_pol_after_R()
    result.record("van_der_pol_after_R", p, m)
    
    p, m = test_duffing_frequency_increases()
    result.record("duffing_frequency_increases", p, m)
    
    print("\n[R MAPPING - Glinsky's Claims]")
    print("-" * 50)
    
    p, m = test_r_linearizes_phase()
    result.record("r_linearizes_phase", p, m)
    
    print("\n[SO(3) SYMMETRY - Sphere Motion]")
    print("-" * 50)
    
    p, m = test_sphere_constraint()
    result.record("sphere_constraint", p, m)
    
    p, m = test_sphere_slow_variable()
    result.record("sphere_slow_variable", p, m)
    
    p, m = test_so3_lie_algebra()
    result.record("so3_lie_algebra", p, m)
    
    print("\n[GROUP THEORY]")
    print("-" * 50)
    
    p, m = test_bogolyubov_projection()
    result.record("bogolyubov_projection", p, m)
    
    p, m = test_two_channel_full_spectrum()
    result.record("two_channel_full_spectrum", p, m)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {result.passed}/{result.passed + result.failed}")
    
    if result.failed == 0:
        print("\n✓ ALL PHYSICAL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        for name, passed, msg in result.results:
            if not passed:
                print(f"  - {name}: {msg}")
    
    return result.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
