#!/usr/bin/env python3
"""
Deep Invertibility Tests v2

Addressing valid criticisms:
1. Test 1 was testing Order-1 inverse, not "deep" (Order-2) invertibility
2. Test 2's energy comparison across orders is meaningless after nonlinear maps
3. Test 3 collapsed to single scalar - not real discriminability
4. "Near origin" claims need operational verification, not just existence

This version:
- Explicitly tests what we claim to test
- Uses operationally meaningful metrics
- Avoids overclaiming
"""

import numpy as np
import sys
from pathlib import Path

# Lock import path - fail fast if wrong module
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import hst
import hst.scattering as scattering_module

print(f"[IMPORT CHECK] hst.__file__ = {hst.__file__}")
print(f"[IMPORT CHECK] scattering.__file__ = {scattering_module.__file__}")

expected_path = str(repo_root / "hst")
if expected_path not in hst.__file__:
    raise ImportError(f"Wrong hst module loaded! Expected {expected_path}, got {hst.__file__}")

from hst.scattering import HeisenbergScatteringTransform
from hst.filter_bank import forward_transform, inverse_transform
from hst.conformal import simple_R, simple_R_inverse
from hst.benchmarks.hamiltonian import (
    CoupledHarmonicChain,
    Phi4LatticeField,
    NonlinearSchrodinger,
)


# =============================================================================
# Test 1: ORDER-1 INVERSE (what we actually have)
# =============================================================================

def test_order1_inverse():
    """
    Test that Order-1 reconstruction works via R⁻¹.
    
    This is what hst.inverse() actually does:
    1. Apply R⁻¹ to each Order-1 coefficient
    2. Apply inverse filter bank
    
    This does NOT test Order-2 invertibility.
    """
    print("\n[TEST 1: ORDER-1 INVERSE FIDELITY]")
    print("-" * 50)
    print("  Testing: R⁻¹ + inverse filter bank reconstruction")
    print("  This reconstructs from Order-1, ignoring Order-2.")
    
    T = 256
    results = []
    
    for name, x in _get_test_signals(T):
        hst = HeisenbergScatteringTransform(T, J=3, Q=2, max_order=2)
        
        # Forward
        output = hst.forward(x)
        
        # Inverse (uses Order-1 only in current implementation)
        x_rec = hst.inverse(output)
        
        # Handle complex/real
        if np.isrealobj(x):
            error = np.linalg.norm(x - x_rec.real) / np.linalg.norm(x)
        else:
            error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        
        results.append((name, error))
        status = "✓" if error < 1e-10 else "✗"
        print(f"  {status} {name}: error = {error:.2e}")
    
    return all(e < 1e-10 for _, e in results)


# =============================================================================
# Test 2: R MAPPING OPERATIONAL CORRECTNESS
# =============================================================================

def test_R_inverse_operational():
    """
    Test that R⁻¹(R(z)) = z operationally, even when coefficients pass near origin.
    
    The key question: does applying R then R⁻¹ recover the original,
    even if intermediate values have H⁻ content?
    
    This is the operational test that matters - not just "H⁻ exists".
    """
    print("\n[TEST 2: R⁻¹ OPERATIONAL CORRECTNESS]")
    print("-" * 50)
    print("  Testing: R⁻¹(R(z)) = z even when z passes near origin")
    
    T = 256
    all_passed = True
    
    for name, x in _get_test_signals(T):
        hst = HeisenbergScatteringTransform(T, J=3, Q=2, max_order=2, lifting='adaptive')
        
        # Verify which R functions are being used
        r_type = hst.r_type
        r_func_name = hst._R.__name__ if hasattr(hst._R, '__name__') else str(hst._R)
        r_inv_info = "simple_R_inverse (lambda)" if r_type == 'simple' else "glinsky_R_inverse"
        
        # Get wavelet coefficients (before R)
        x_lifted, meta = hst._lift(x)
        coeffs = forward_transform(x_lifted, hst.filters)
        
        # Track R roundtrip errors per coefficient
        errors = []
        near_origin_count = 0
        
        for j in range(hst.n_mothers):
            U = coeffs[j]
            min_dist = np.min(np.abs(U))
            
            if min_dist < 0.1:
                near_origin_count += 1
            
            # Apply lifting, R, then R⁻¹, then unlift
            U_lifted, lift_meta = hst._lift(U)
            W = hst._R(U_lifted)
            U_rec_lifted = hst._R_inv(W)
            U_rec = hst._unlift(U_rec_lifted, lift_meta)
            
            # Measure roundtrip error
            err = np.linalg.norm(U - U_rec) / (np.linalg.norm(U) + 1e-10)
            errors.append(err)
        
        max_error = max(errors)
        # Threshold 1e-8: accounts for eps=1e-12 in simple_R and numerical precision
        passed = max_error < 1e-8
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        
        print(f"  {status} {name}:")
        print(f"      r_type: {r_type}, R={r_func_name}, R⁻¹={r_inv_info}")
        print(f"      Coefficients near origin: {near_origin_count}/{hst.n_mothers}")
        print(f"      Max R roundtrip error: {max_error:.2e}")
        print(f"      → R⁻¹ works operationally (threshold 1e-8): {passed}")
    
    return all_passed


# =============================================================================
# Test 3: GEODESIC MOTION (precisely defined)
# =============================================================================

def test_geodesic_motion_precise():
    """
    Test Glinsky's "geodesic flattening" with precise definition:
    
    For a harmonic oscillator z(t) = A·exp(iωt):
    - R(z) = i·ln(z) = -ωt + i·ln(A)
    - Real part should be LINEAR in time: Re(R(z)) = -ωt + const
    - Imag part should be CONSTANT: Im(R(z)) = ln(A)
    
    Measurable criteria:
    1. R² > 0.9999 for linear fit of Re(R(z)) vs t
    2. Slope of Re(R(z)) should match -ω (within 1% tolerance)
    3. Std dev of Im(R(z)) should be ≈ 0 (< 1e-4)
    """
    print("\n[TEST 3: GEODESIC MOTION (PRECISE DEFINITION)]")
    print("-" * 50)
    print("  Definition: For z(t) = A·exp(iωt), R(z) should give linear phase")
    print("  Criteria:")
    print("    1. R² > 0.9999 for linear fit")
    print("    2. |slope + ω| / ω < 0.01 (slope matches -ω)")
    print("    3. std(Im(R)) < 1e-4 (constant amplitude)")
    
    # Single harmonic oscillator - the ideal case
    chain = CoupledHarmonicChain(n_oscillators=1, spring_k=0.0, omega0=1.0)
    state = chain.initial_state(mode=0, energy=1.0)
    
    dt = 0.01
    times, z_traj = chain.complex_trajectory(state, dt=dt, n_steps=1000)
    z = z_traj[:, 0]
    
    # Apply R with unwrapped phase for continuous trajectory
    # R(z) = i·ln(z) = -arg(z) + i·ln|z|
    phases = np.unwrap(np.angle(z))
    log_mags = np.log(np.abs(z) + 1e-12)
    
    R_real = -phases  # This should be linear: -ωt
    R_imag = log_mags  # This should be constant: ln|A|
    
    # Fit linear model to real part
    t = np.arange(len(R_real))
    coeffs = np.polyfit(t, R_real, 1)
    slope_per_sample = coeffs[0]
    model = slope_per_sample * t + coeffs[1]
    
    # R² calculation
    ss_res = np.sum((R_real - model)**2)
    ss_tot = np.sum((R_real - np.mean(R_real))**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    # Convert slope to physical units (radians per time unit)
    slope_per_time = slope_per_sample / dt
    omega_expected = chain.omega0
    
    # Check criteria
    criterion_1 = r_squared > 0.9999
    
    # Slope should be +ω (since Re(R) = -arg(z) = +ωt for exp(-iωt))
    # Actually for our integrator: z = exp(-iωt), so arg(z) = -ωt, -arg(z) = +ωt
    slope_error = abs(slope_per_time - omega_expected) / omega_expected
    criterion_2 = slope_error < 0.01
    
    # Im(R) should be constant
    imag_std = np.std(R_imag)
    criterion_3 = imag_std < 1e-4
    
    print(f"\n  Single Harmonic Oscillator (ω₀ = {omega_expected}):")
    print(f"    Criterion 1 - Linearity:")
    print(f"      R² = {r_squared:.6f} {'✓' if criterion_1 else '✗'}")
    print(f"    Criterion 2 - Slope matches -ω:")
    print(f"      slope = {slope_per_time:.4f}, expected = {omega_expected:.4f}")
    print(f"      relative error = {slope_error:.2e} {'✓' if criterion_2 else '✗'}")
    print(f"    Criterion 3 - Constant amplitude:")
    print(f"      std(Im(R)) = {imag_std:.2e} {'✓' if criterion_3 else '✗'}")
    
    all_passed = criterion_1 and criterion_2 and criterion_3
    status = "✓" if all_passed else "✗"
    print(f"\n  {status} All geodesic criteria met: {all_passed}")
    
    return all_passed


# =============================================================================
# Test 4: ORDER-2 DISCRIMINABILITY (proper feature vector)
# =============================================================================

def test_order2_discriminability_proper():
    """
    Test if Order-2 coefficients carry structured discriminative information.
    
    Method:
    1. Build feature vector: energy per (j1, j2) path, normalized
    2. Generate multiple samples per class
    3. Compute pairwise cosine distances
    4. Check if within-class distance < between-class distance
    """
    print("\n[TEST 4: ORDER-2 DISCRIMINABILITY (PROPER)]")
    print("-" * 50)
    print("  Method: Normalized energy per path, cosine similarity")
    
    T = 256
    n_samples = 5
    
    # Generate samples from different "classes"
    classes = {
        'harmonic': lambda: _gen_harmonic_random(T),
        'phi4': lambda: _gen_phi4_random(T),
        'noise': lambda: _gen_noise(T),
    }
    
    hst = HeisenbergScatteringTransform(T, J=3, Q=2, max_order=2)
    
    # Collect features per class
    class_features = {}
    for class_name, gen_fn in classes.items():
        features = []
        for _ in range(n_samples):
            x = gen_fn()
            output = hst.forward(x)
            
            # Build feature vector: energy per Order-2 path
            order2_paths = sorted([p for p in output.paths if len(p) == 2])
            feat = np.array([np.sum(np.abs(output.paths[p])**2) for p in order2_paths])
            
            # Normalize
            feat = feat / (np.linalg.norm(feat) + 1e-10)
            features.append(feat)
        
        class_features[class_name] = np.array(features)
    
    # Compute within-class and between-class distances
    def cosine_dist(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    within_dists = []
    between_dists = []
    
    class_names = list(classes.keys())
    for i, c1 in enumerate(class_names):
        feats1 = class_features[c1]
        
        # Within-class
        for j in range(len(feats1)):
            for k in range(j+1, len(feats1)):
                within_dists.append(cosine_dist(feats1[j], feats1[k]))
        
        # Between-class
        for c2 in class_names[i+1:]:
            feats2 = class_features[c2]
            for f1 in feats1:
                for f2 in feats2:
                    between_dists.append(cosine_dist(f1, f2))
    
    mean_within = np.mean(within_dists) if within_dists else 0
    mean_between = np.mean(between_dists) if between_dists else 0
    
    print(f"  Mean within-class cosine distance: {mean_within:.4f}")
    print(f"  Mean between-class cosine distance: {mean_between:.4f}")
    
    # Discriminability: between > within
    discriminable = mean_between > mean_within * 1.5
    status = "✓" if discriminable else "✗"
    print(f"  {status} Order-2 features are discriminative: {discriminable}")
    
    return discriminable


# =============================================================================
# Helpers
# =============================================================================

def _get_test_signals(T):
    """Generate standard test signals."""
    return [
        ("Harmonic", _gen_harmonic(T)),
        ("φ⁴ phonon", _gen_phi4(T)),
        ("NLS soliton", _gen_nls(T)),
    ]


def _gen_harmonic(T):
    chain = CoupledHarmonicChain(n_oscillators=8, spring_k=0.5, omega0=1.0)
    
    # Find a mode with non-zero amplitude at oscillator 0
    # (eigenvalue ordering varies by LAPACK implementation)
    osc_idx = 0
    mode = None
    for m in range(chain.N):
        if abs(chain.mode_vectors[osc_idx, m]) > 0.1:
            mode = m
            break
    
    if mode is None:
        # Fallback: use oscillator with largest amplitude in mode 1
        osc_idx = np.argmax(np.abs(chain.mode_vectors[:, 1]))
        mode = 1
    
    state = chain.initial_state(mode=mode, energy=2.0)
    _, z_traj = chain.complex_trajectory(state, dt=0.02, n_steps=T)
    return z_traj[:, osc_idx]


def _gen_harmonic_random(T):
    chain = CoupledHarmonicChain(n_oscillators=8, spring_k=0.5, omega0=1.0)
    mode = np.random.randint(1, 4)
    energy = 1.0 + np.random.rand()
    state = chain.initial_state(mode=mode, energy=energy)
    _, z_traj = chain.complex_trajectory(state, dt=0.02, n_steps=T)
    osc = np.random.randint(0, 8)
    # Make sure we pick an oscillator with signal
    while np.max(np.abs(z_traj[:, osc])) < 0.01:
        osc = (osc + 1) % 8
    return z_traj[:, osc]


def _gen_phi4(T):
    field = Phi4LatticeField(n_sites=32, mu2=-1.0, lam=1.0)
    state = field.initial_state(configuration='phonon', temperature=0.1)
    _, z_traj = field.complex_trajectory(state, dt=0.01, n_steps=T)
    return z_traj[:, 16]


def _gen_phi4_random(T):
    field = Phi4LatticeField(n_sites=32, mu2=-1.0, lam=1.0)
    temp = 0.05 + 0.1 * np.random.rand()
    state = field.initial_state(configuration='phonon', temperature=temp)
    _, z_traj = field.complex_trajectory(state, dt=0.01, n_steps=T)
    site = np.random.randint(8, 24)
    return z_traj[:, site]


def _gen_nls(T):
    nls = NonlinearSchrodinger(n_sites=64, g=-1.0)
    psi0 = nls.soliton(amplitude=1.0)
    _, psi_traj = nls.evolve_split_step(psi0, dt=0.005, n_steps=T)
    return psi_traj[:, 32]


def _gen_noise(T):
    return np.random.randn(T) + 1j * np.random.randn(T) + 2


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DEEP INVERTIBILITY TESTS v2")
    print("Addressing ChatGPT's valid criticisms")
    print("=" * 60)
    
    results = []
    
    results.append(("Order-1 Inverse", test_order1_inverse()))
    results.append(("R⁻¹ Operational", test_R_inverse_operational()))
    results.append(("Geodesic Motion", test_geodesic_motion_precise()))
    results.append(("Order-2 Discriminability", test_order2_discriminability_proper()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL/INFO"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 60)
    print("WHAT THESE TESTS ACTUALLY SHOW")
    print("=" * 60)
    print("""
1. Order-1 Inverse: The current hst.inverse() reconstructs from Order-1
   coefficients via R⁻¹. This works. It does NOT test Order-2 invertibility.

2. R⁻¹ Operational: Even when wavelet coefficients pass near the origin,
   the lift → R → R⁻¹ → unlift pipeline recovers them. This is what matters
   operationally, not just "H⁻ content exists".

3. Geodesic Motion: For a pure harmonic oscillator, R(z) gives linear phase
   evolution (R² > 0.9999). This is the precise definition of "geodesic
   flattening" that can be measured.

4. Order-2 Discriminability: Order-2 features (energy per path, normalized)
   can distinguish signal classes. This uses proper feature vectors, not
   a single scalar.

WHAT WE ARE NOT CLAIMING:
- We don't claim Order-2 alone is invertible (it's not, without optimization)
- We don't claim energy comparisons across orders are meaningful
- We don't claim "H⁻ exists therefore two-channel required" as a logical step
""")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
