"""
Minimal 2D Winding Test Suite v2

Surgical tests to isolate where the 2D HST pipeline breaks.

KEY FINDING: Cross-channel H+ * conj(H-) approach DOES NOT work for chirality!
The chirality signal is in the PHASE CURL, not in directional frequency content.

Tests:
- Test 0: Dataset sanity (paired vortices, winding labels)
- Test 1: Filter-bank chirality sensitivity (EXPECTED TO FAIL - wrong approach)
- Test 2: Rectifier phase behavior
- Test 3: Feature U(1)-invariance verification  
- Test 4: Chirality accessibility with PHASE CURL features (SHOULD WORK)

Run: python test_2d_winding_minimal.py
"""

import numpy as np
from typing import Tuple, Dict
import sys


# =============================================================================
# VORTEX GENERATION
# =============================================================================

def create_vortex_field(L: int, charge: int = 1, center: Tuple[int, int] = None, 
                        noise_std: float = 0.1, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if center is None:
        center = (L//2, L//2)
    
    y, x = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    dx = x - center[1]
    dy = y - center[0]
    
    z_raw = (dx + 1j * dy) ** charge
    z_raw /= (np.abs(z_raw).max() + 1e-10)
    
    noise = noise_std * (np.random.randn(L, L) + 1j * np.random.randn(L, L))
    return (z_raw + noise).astype(np.complex128)


def create_paired_vortices(L: int, center: Tuple[int, int], 
                           noise_std: float = 0.1, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    z_plus = create_vortex_field(L, charge=+1, center=center, noise_std=noise_std, seed=seed)
    z_minus = np.conj(z_plus)
    return z_plus, z_minus


def compute_local_winding(z: np.ndarray, center: Tuple[int, int], radius: int = 8) -> float:
    L = z.shape[0]
    cy, cx = center
    n_points = 128
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    winding = 0.0
    for i in range(n_points):
        a1, a2 = angles[i], angles[(i+1) % n_points]
        y1 = int(round(cy + radius * np.sin(a1))) % L
        x1 = int(round(cx + radius * np.cos(a1))) % L
        y2 = int(round(cy + radius * np.sin(a2))) % L
        x2 = int(round(cx + radius * np.cos(a2))) % L
        
        z1, z2 = z[y1, x1], z[y2, x2]
        if np.abs(z1) > 1e-10 and np.abs(z2) > 1e-10:
            winding += np.angle(z2 / z1)
    
    return winding / (2 * np.pi)


# =============================================================================
# PHASE CURL FEATURES (THE CORRECT APPROACH)
# =============================================================================

def phase_curl_features(z: np.ndarray, scales: list = [1, 2, 4]) -> np.ndarray:
    """
    Compute gauge-invariant phase curl features.
    
    KEY INSIGHT: Chirality is in the CURL of the phase gradient!
    - curl = dA_y/dx - dA_x/dy where A = Im(dz/z)
    - This is U(1)-invariant: z → z*e^{iφ} doesn't change curl
    - It flips sign under conjugation: conj(z) has opposite curl
    
    This is what Glinsky's gauge-invariant approach should capture.
    """
    L = z.shape[0]
    features = []
    
    kx = np.fft.fftfreq(L)
    ky = np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky)
    
    for scale in scales:
        # Smooth at this scale
        gauss = np.exp(-(KX**2 + KY**2) * (scale**2) / 2)
        z_smooth = np.fft.ifft2(np.fft.fft2(z) * gauss)
        
        # Compute derivatives
        z_padded = np.pad(z_smooth, 1, mode='wrap')
        dz_dx = (z_padded[1:-1, 2:] - z_padded[1:-1, :-2]) / 2
        dz_dy = (z_padded[2:, 1:-1] - z_padded[:-2, 1:-1]) / 2
        
        # Gauge-invariant connection: A = Im(dz/z)
        A_x = np.imag(dz_dx / (z_smooth + 1e-10))
        A_y = np.imag(dz_dy / (z_smooth + 1e-10))
        
        # Curl = dA_y/dx - dA_x/dy (the Berry curvature!)
        curl = np.gradient(A_y, axis=1) - np.gradient(A_x, axis=0)
        
        # Features: sum (total winding), mean, variance
        features.extend([curl.sum(), curl.mean(), (curl**2).mean()])
    
    return np.array(features)


# =============================================================================
# TEST 0: DATASET SANITY
# =============================================================================

def test_0_dataset_sanity(verbose: bool = True) -> Dict:
    """Verify paired vortices have correct properties."""
    L = 32
    center = (L//2, L//2)
    results = {'passed': True, 'checks': {}}
    
    for seed in range(5):
        z_plus, z_minus = create_paired_vortices(L, center, noise_std=0.1, seed=seed)
        
        mag_diff = np.abs(np.abs(z_plus) - np.abs(z_minus)).max()
        mag_ok = mag_diff < 1e-10
        
        w_plus = compute_local_winding(z_plus, center)
        w_minus = compute_local_winding(z_minus, center)
        
        w_plus_ok = abs(w_plus - 1.0) < 0.1
        w_minus_ok = abs(w_minus - (-1.0)) < 0.1
        
        results['checks'][f'pair_{seed}'] = {
            'mag_diff': mag_diff, 'w_plus': w_plus, 'w_minus': w_minus,
            'passed': mag_ok and w_plus_ok and w_minus_ok,
        }
        
        if not (mag_ok and w_plus_ok and w_minus_ok):
            results['passed'] = False
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 0: DATASET SANITY")
        print("="*60)
        for name, check in results['checks'].items():
            status = "✓" if check['passed'] else "✗"
            print(f"  {name}: w+={check['w_plus']:.3f}, w-={check['w_minus']:.3f} {status}")
        print(f"  OVERALL: {'PASS ✓' if results['passed'] else 'FAIL ✗'}")
    
    return results


# =============================================================================
# TEST 1: CROSS-CHANNEL APPROACH (EXPECTED TO FAIL)
# =============================================================================

def test_1_cross_channel_fails(verbose: bool = True) -> Dict:
    """
    Test 1: Show that H+ * conj(H-) cross-channel approach FAILS.
    
    This is expected! The chirality is in phase CURL, not directional frequency.
    """
    L = 32
    center = (L//2, L//2)
    
    results = {'passed': True, 'note': 'This test is expected to show ~0 signal'}
    
    # Build minimal filters
    filters = _build_minimal_filters(L, J=3, L_orient=4)
    filters_by_jt = {}
    for psi in filters['psi']:
        key = (psi['j'], psi['theta'])
        if key not in filters_by_jt:
            filters_by_jt[key] = {}
        filters_by_jt[key][psi['channel']] = psi['filter']
    
    A_values = []
    for seed in range(5):
        z_plus, z_minus = create_paired_vortices(L, center, noise_std=0.1, seed=seed)
        
        z_plus_hat = np.fft.fft2(z_plus)
        z_minus_hat = np.fft.fft2(z_minus)
        
        A_plus_list, A_minus_list = [], []
        for (j, theta), channels in filters_by_jt.items():
            if 'H+' in channels and 'H-' in channels:
                U_hp = np.fft.ifft2(z_plus_hat * channels['H+'])
                U_hm = np.fft.ifft2(z_plus_hat * channels['H-'])
                A_plus_list.append(np.imag(U_hp * np.conj(U_hm)).mean())
                
                U_hp = np.fft.ifft2(z_minus_hat * channels['H+'])
                U_hm = np.fft.ifft2(z_minus_hat * channels['H-'])
                A_minus_list.append(np.imag(U_hp * np.conj(U_hm)).mean())
        
        A_plus = np.mean(A_plus_list)
        A_minus = np.mean(A_minus_list)
        A_values.append((A_plus, A_minus))
    
    results['A_values'] = A_values
    results['mean_magnitude'] = np.mean([abs(a[0]) + abs(a[1]) for a in A_values]) / 2
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 1: CROSS-CHANNEL H+*conj(H-) (EXPECTED TO FAIL)")
        print("="*60)
        print("  This approach is WRONG for chirality detection!")
        print("  Chirality is in phase CURL, not directional frequency content.")
        print()
        for i, (A_plus, A_minus) in enumerate(A_values):
            print(f"  seed={i}: A+={A_plus:+.6f}, A-={A_minus:+.6f}")
        print(f"\n  Mean |A| = {results['mean_magnitude']:.6f} (expected ~0)")
        print("  → Cross-channel approach cannot see chirality (as expected)")
    
    return results


def _build_minimal_filters(L: int, J: int, L_orient: int) -> Dict:
    """Minimal filter bank."""
    kx = np.fft.fftfreq(L)
    ky = np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky)
    
    filters = {'psi': [], 'phi': None}
    
    for j in range(J):
        for theta_idx in range(L_orient):
            theta = 2 * np.pi * theta_idx / L_orient
            k0 = 0.4 * (2.0 ** (-j))
            sigma = k0 / 4
            
            for positive in [True, False]:
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                k_par = KX * cos_t + KY * sin_t
                if not positive:
                    k_par = -k_par
                k_perp = -KX * sin_t + KY * cos_t
                
                k_norm = np.sqrt(KX**2 + KY**2)
                cone_mask = 0.5 * (1 + np.tanh(k_par / (sigma * 0.5)))
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_k = np.where(k_norm > 1e-10, np.log(k_norm / k0), -100)
                    radial = np.exp(-0.5 * (log_k / 0.5)**2)
                    radial = np.where(k_norm > 1e-10, radial, 0)
                
                angular = np.exp(-0.5 * (k_perp / sigma)**2)
                psi_hat = cone_mask * radial * angular
                
                energy = np.sum(psi_hat**2)
                if energy > 1e-10:
                    psi_hat = psi_hat / np.sqrt(energy)
                
                filters['psi'].append({
                    'filter': psi_hat.astype(np.complex128),
                    'j': j, 'theta': theta_idx,
                    'channel': 'H+' if positive else 'H-',
                })
    
    return filters


# =============================================================================
# TEST 2: PHASE CURL FEATURES (THE CORRECT APPROACH)
# =============================================================================

def test_2_phase_curl_works(verbose: bool = True) -> Dict:
    """
    Test 2: Show that phase curl features CORRECTLY detect chirality.
    
    The gauge-invariant phase curl:
    - Flips sign under conjugation (z → conj(z))
    - Is U(1)-invariant (z → z*e^{iφ} doesn't change it)
    - Has magnitude proportional to winding number
    """
    L = 32
    center = (L//2, L//2)
    
    results = {'passed': True, 'pairs': []}
    
    for seed in range(5):
        z_plus, z_minus = create_paired_vortices(L, center, noise_std=0.1, seed=seed)
        
        feat_plus = phase_curl_features(z_plus)
        feat_minus = phase_curl_features(z_minus)
        
        # Check: curl_sum should flip sign
        curl_sum_plus = feat_plus[0]  # First feature is curl.sum() at scale 1
        curl_sum_minus = feat_minus[0]
        
        sign_flip = curl_sum_plus * curl_sum_minus < 0
        ratio = curl_sum_minus / curl_sum_plus if abs(curl_sum_plus) > 1e-10 else 0
        ratio_ok = abs(ratio - (-1.0)) < 0.1
        
        pair_ok = sign_flip and ratio_ok
        
        results['pairs'].append({
            'seed': seed,
            'curl_sum_plus': curl_sum_plus,
            'curl_sum_minus': curl_sum_minus,
            'ratio': ratio,
            'passed': pair_ok,
        })
        
        if not pair_ok:
            results['passed'] = False
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 2: PHASE CURL FEATURES (CORRECT APPROACH)")
        print("="*60)
        print("  Curl = dA_y/dx - dA_x/dy where A = Im(dz/z)")
        print("  Expected: curl(z+) ≈ -curl(z-), ratio ≈ -1.0")
        print()
        for p in results['pairs']:
            status = "✓" if p['passed'] else "✗"
            print(f"  seed={p['seed']}: curl+={p['curl_sum_plus']:+.4f}, "
                  f"curl-={p['curl_sum_minus']:+.4f}, ratio={p['ratio']:.4f} {status}")
        print(f"\n  OVERALL: {'PASS ✓' if results['passed'] else 'FAIL ✗'}")
        
        if results['passed']:
            print("  → Phase curl CORRECTLY captures chirality!")
    
    return results


# =============================================================================
# TEST 3: U(1) INVARIANCE OF PHASE CURL
# =============================================================================

def test_3_phase_curl_invariance(verbose: bool = True) -> Dict:
    """
    Test 3: Verify phase curl features are U(1)-invariant.
    
    z → z * e^{iφ} should NOT change the curl features.
    """
    L = 32
    center = (L//2, L//2)
    
    results = {'passed': True, 'tests': []}
    
    z_plus, _ = create_paired_vortices(L, center, noise_std=0.1, seed=0)
    feat_orig = phase_curl_features(z_plus)
    
    phases = [0.1, 0.5, 1.0, np.pi/2, np.pi]
    
    for phi in phases:
        z_rotated = z_plus * np.exp(1j * phi)
        feat_rot = phase_curl_features(z_rotated)
        
        diff = np.abs(feat_rot - feat_orig).max()
        invariant = diff < 1e-6
        
        results['tests'].append({
            'phi': phi,
            'diff': diff,
            'invariant': invariant,
        })
        
        if not invariant:
            results['passed'] = False
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 3: PHASE CURL U(1)-INVARIANCE")
        print("="*60)
        print("  Testing: f(z*e^{iφ}) == f(z)?")
        print()
        for t in results['tests']:
            status = "✓" if t['invariant'] else "✗"
            print(f"  φ={t['phi']:.2f}: max_diff={t['diff']:.2e} {status}")
        print(f"\n  OVERALL: {'PASS ✓' if results['passed'] else 'FAIL ✗'}")
        
        if results['passed']:
            print("  → Phase curl features ARE U(1)-invariant!")
    
    return results


# =============================================================================
# TEST 4: CLASSIFICATION SMOKE TEST
# =============================================================================

def test_4_classification_smoke(verbose: bool = True) -> Dict:
    """
    Test 4: Classification test comparing approaches.
    
    - Direct winding: ~100% (ground truth)
    - Cross-channel: ~50% (wrong approach)
    - Phase curl: ~100% (correct approach)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    L = 32
    center = (L//2, L//2)
    n_pairs = 30
    
    results = {'accuracies': {}, 'passed': True}
    
    # Build filters for cross-channel
    filters = _build_minimal_filters(L, J=3, L_orient=4)
    filters_by_jt = {}
    for psi in filters['psi']:
        key = (psi['j'], psi['theta'])
        if key not in filters_by_jt:
            filters_by_jt[key] = {}
        filters_by_jt[key][psi['channel']] = psi['filter']
    
    # Generate dataset
    data, labels = [], []
    for seed in range(n_pairs):
        z_plus, z_minus = create_paired_vortices(L, center, noise_std=0.1, seed=seed)
        data.extend([z_plus, z_minus])
        labels.extend([+1, -1])
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Feature extractors
    def extract_winding(samples):
        return np.array([[compute_local_winding(s, center)] for s in samples])
    
    def extract_cross_channel(samples):
        features = []
        for s in samples:
            s_hat = np.fft.fft2(s)
            feat = []
            for (j, theta), channels in filters_by_jt.items():
                if 'H+' in channels and 'H-' in channels:
                    U_hp = np.fft.ifft2(s_hat * channels['H+'])
                    U_hm = np.fft.ifft2(s_hat * channels['H-'])
                    cross = U_hp * np.conj(U_hm)
                    feat.extend([np.real(cross).mean(), np.imag(cross).mean()])
            features.append(feat)
        return np.array(features)
    
    def extract_phase_curl(samples):
        return np.array([phase_curl_features(s) for s in samples])
    
    # Train/test split
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=labels)
    
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    # Evaluate
    extractors = {
        'direct_winding': extract_winding,
        'cross_channel': extract_cross_channel,
        'phase_curl': extract_phase_curl,
    }
    
    for name, extractor in extractors.items():
        train_feat = extractor(train_data)
        test_feat = extractor(test_data)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_feat, train_labels)
        acc = clf.score(test_feat, test_labels)
        
        results['accuracies'][name] = acc
    
    # Check expectations
    results['winding_ok'] = results['accuracies']['direct_winding'] > 0.9
    results['cross_fails'] = results['accuracies']['cross_channel'] < 0.6
    results['curl_works'] = results['accuracies']['phase_curl'] > 0.9
    
    results['passed'] = results['winding_ok'] and results['curl_works']
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 4: CLASSIFICATION SMOKE TEST")
        print("="*60)
        print(f"  Dataset: {n_pairs} pairs, train={len(train_data)}, test={len(test_data)}")
        print()
        
        for name, acc in results['accuracies'].items():
            if name == 'direct_winding':
                expected, status = ">90%", "✓" if acc > 0.9 else "✗"
            elif name == 'cross_channel':
                expected, status = "~50%", "✓" if acc < 0.6 else "?"
            else:
                expected, status = ">90%", "✓" if acc > 0.9 else "✗"
            
            print(f"  {name:20s}: {acc:.1%} (expect {expected}) {status}")
        
        print(f"\n  OVERALL: {'PASS ✓' if results['passed'] else 'FAIL ✗'}")
        
        if results['curl_works'] and results['cross_fails']:
            print("\n  CONCLUSION: Phase curl is the CORRECT approach for chirality!")
            print("  Cross-channel H+*conj(H-) does NOT work.")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests(verbose: bool = True) -> Dict:
    """Run all tests and return summary."""
    print("\n" + "#"*60)
    print("# MINIMAL 2D WINDING TEST SUITE v2")
    print("# Key finding: Use PHASE CURL, not cross-channel wavelets")
    print("#"*60)
    
    results = {}
    
    results['test_0'] = test_0_dataset_sanity(verbose)
    results['test_1'] = test_1_cross_channel_fails(verbose)
    results['test_2'] = test_2_phase_curl_works(verbose)
    results['test_3'] = test_3_phase_curl_invariance(verbose)
    results['test_4'] = test_4_classification_smoke(verbose)
    
    # Summary
    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    
    print("  test_0 (dataset sanity):     ", "PASS ✓" if results['test_0']['passed'] else "FAIL ✗")
    print("  test_1 (cross-channel):       Expected to show ~0 signal")
    print("  test_2 (phase curl chirality):", "PASS ✓" if results['test_2']['passed'] else "FAIL ✗")
    print("  test_3 (phase curl U(1)):    ", "PASS ✓" if results['test_3']['passed'] else "FAIL ✗")
    print("  test_4 (classification):     ", "PASS ✓" if results['test_4']['passed'] else "FAIL ✗")
    
    print("\n  KEY INSIGHT:")
    print("  → Chirality is in the PHASE CURL (Berry curvature)")
    print("  → Cross-channel H+*conj(H-) approach is WRONG")
    print("  → Need to implement phase-curl-aware HST features")
    
    return results


if __name__ == '__main__':
    run_all_tests(verbose=True)
