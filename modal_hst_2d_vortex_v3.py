"""
2D Vortex Chirality Benchmark v3 - Proper 2D HST

This version uses a TRUE 2D wavelet cascade scattering transform
with Glinsky's R rectifier, not just FFT+binning.

Architecture comparison:
- v2 (FFT pooling): signal → 2D FFT → R → angular/radial bins
- v3 (Wavelet cascade): signal → [wavelet*R]^n → average

The wavelet cascade is what Glinsky and Mallat actually describe.
"""

import modal
import sys

app = modal.App("hst-2d-vortex-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10", 
        "scikit-learn>=1.3",
        "torch>=2.0",
        "kymatio>=0.3",
    )
)


@app.function(image=image, gpu="T4", timeout=1800)
def run_hst_2d_vortex_v3():
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ========================================
    # VORTEX FIELD GENERATION
    # ========================================
    
    def create_vortex_field(L, charge=1, center=None, noise_std=0.15, seed=None):
        """Create complex vortex field with given winding number."""
        if seed is not None:
            np.random.seed(seed)
        if center is None:
            center = (L//2, L//2)
        
        y, x = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        dx = x - center[1]
        dy = y - center[0]
        
        # Complex vortex: z = (dx + i*dy)^charge / normalization
        z_raw = (dx + 1j * dy) ** charge
        z_raw /= (np.abs(z_raw).max() + 1e-10)
        
        # Add complex noise
        noise = noise_std * (np.random.randn(L, L) + 1j * np.random.randn(L, L))
        z = z_raw + noise
        
        return z.astype(np.complex128)
    
    def create_paired_vortices(L, center, noise_std=0.15, seed=None):
        """Create paired vortices: z_minus = conj(z_plus)."""
        z_plus = create_vortex_field(L, charge=+1, center=center, 
                                      noise_std=noise_std, seed=seed)
        z_minus = np.conj(z_plus)  # Exact conjugate!
        return z_plus, z_minus
    
    def compute_local_winding(z, center, radius=5):
        """Compute winding number via contour integral around center."""
        L = z.shape[0]
        cy, cx = center
        
        n_points = 64
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        winding = 0.0
        for i in range(n_points):
            a1, a2 = angles[i], angles[(i+1) % n_points]
            
            y1 = int(cy + radius * np.sin(a1)) % L
            x1 = int(cx + radius * np.cos(a1)) % L
            y2 = int(cy + radius * np.sin(a2)) % L
            x2 = int(cx + radius * np.cos(a2)) % L
            
            z1, z2 = z[y1, x1], z[y2, x2]
            
            if np.abs(z1) > 1e-10 and np.abs(z2) > 1e-10:
                dtheta = np.angle(z2 / z1)
                winding += dtheta
        
        return winding / (2 * np.pi)
    
    # ========================================
    # 2D HST IMPLEMENTATION (PROPER CASCADE)
    # ========================================
    
    def glinsky_R(z, eps=1e-12):
        """Glinsky's R = i*ln(R0) rectifier."""
        # Simple version: R(z) = i * ln(z)
        # Full version would use Joukowsky transform
        z_safe = z + eps * (1 + 1j)
        return 1j * torch.log(z_safe)
    
    def lift_radial(z, eps=1e-8):
        """Lift signal away from origin."""
        r = torch.abs(z)
        r_floor = torch.sqrt(r**2 + eps**2)
        scale = r_floor / torch.clamp(r, min=1e-30)
        return z * scale
    
    class HST2DSimple:
        """
        Simplified 2D HST using direct 2D wavelet cascade.
        
        Structure:
            Order 0: x * phi (lowpass)
            Order 1: R(x * psi_j,θ) * phi
            Order 2: R(R(x * psi_j1,θ1) * psi_j2,θ2) * phi
        """
        
        def __init__(self, L, J=3, L_orient=4, rectifier='glinsky', device='cpu'):
            self.L = L
            self.J = J  # scales
            self.L_orient = L_orient  # orientations
            self.rectifier_type = rectifier
            self.device = device
            
            # Build filter bank in Fourier domain
            self.psi, self.phi = self._build_filters()
        
        def _build_filters(self):
            """Build 2D Morlet wavelets in Fourier domain."""
            L = self.L
            
            # Frequency grid
            kx = torch.fft.fftfreq(L, device=self.device).float()
            ky = torch.fft.fftfreq(L, device=self.device).float()
            KX, KY = torch.meshgrid(kx, ky, indexing='ij')
            K_norm = torch.sqrt(KX**2 + KY**2)
            K_angle = torch.atan2(KY, KX)
            
            psi_list = []
            
            for j in range(self.J):
                for theta_idx in range(self.L_orient):
                    # Scale: sigma ~ 2^j
                    sigma = 0.1 * (2 ** j)  # Bandwidth in frequency
                    
                    # Orientation
                    theta = theta_idx * np.pi / self.L_orient
                    
                    # Central frequency along theta direction
                    k0 = 0.4 / (2 ** j)  # Peak frequency
                    
                    # Gabor-like wavelet in Fourier domain
                    # Gaussian envelope in frequency centered at k0 along theta
                    k_parallel = KX * np.cos(theta) + KY * np.sin(theta)
                    k_perp = -KX * np.sin(theta) + KY * np.cos(theta)
                    
                    # Anisotropic Gaussian (elongated along orientation)
                    psi_hat = torch.exp(-((k_parallel - k0)**2 / (2 * sigma**2) + 
                                          k_perp**2 / (2 * (sigma/2)**2)))
                    
                    # Zero mean correction (Morlet)
                    psi_hat = psi_hat - psi_hat.mean()
                    
                    psi_list.append({
                        'filter': psi_hat.to(torch.complex128),
                        'j': j,
                        'theta': theta_idx,
                    })
            
            # Lowpass (father wavelet) - isotropic Gaussian
            sigma_phi = 0.1 * (2 ** (self.J - 1))
            phi_hat = torch.exp(-K_norm**2 / (2 * sigma_phi**2))
            phi_hat = phi_hat.to(torch.complex128)
            
            return psi_list, phi_hat
        
        def _rectify(self, z):
            """Apply rectifier."""
            z_lifted = lift_radial(z)
            
            if self.rectifier_type == 'glinsky':
                return glinsky_R(z_lifted)
            elif self.rectifier_type == 'modulus':
                return torch.abs(z_lifted).to(torch.complex128)
            else:
                raise ValueError(f"Unknown rectifier: {self.rectifier_type}")
        
        def forward(self, x):
            """
            Compute 2D HST scattering coefficients.
            
            Returns dict with S0, S1, S2 coefficients.
            """
            x = torch.as_tensor(x, dtype=torch.complex128, device=self.device)
            
            # FFT of input
            x_hat = torch.fft.fft2(x)
            
            S = {'S0': None, 'S1': [], 'S2': []}
            
            # Order 0: lowpass
            S0_hat = x_hat * self.phi
            S0 = torch.fft.ifft2(S0_hat)
            S['S0'] = S0
            
            # Order 1
            U1_dict = {}  # Store rectified for order 2
            
            for psi in self.psi:
                j1 = psi['j']
                theta1 = psi['theta']
                
                # Convolve
                U1_hat = x_hat * psi['filter']
                U1 = torch.fft.ifft2(U1_hat)
                
                # Rectify
                U1_rect = self._rectify(U1)
                U1_dict[(j1, theta1)] = U1_rect
                
                # Average
                U1_rect_hat = torch.fft.fft2(U1_rect)
                S1_hat = U1_rect_hat * self.phi
                S1 = torch.fft.ifft2(S1_hat)
                
                S['S1'].append({
                    'coef': S1,
                    'j': j1,
                    'theta': theta1,
                })
            
            # Order 2 (frequency ordering: j2 > j1)
            for (j1, theta1), U1_rect in U1_dict.items():
                U1_rect_hat = torch.fft.fft2(U1_rect)
                
                for psi2 in self.psi:
                    j2 = psi2['j']
                    theta2 = psi2['theta']
                    
                    if j2 <= j1:
                        continue
                    
                    # Convolve
                    U2_hat = U1_rect_hat * psi2['filter']
                    U2 = torch.fft.ifft2(U2_hat)
                    
                    # Rectify
                    U2_rect = self._rectify(U2)
                    
                    # Average
                    U2_rect_hat = torch.fft.fft2(U2_rect)
                    S2_hat = U2_rect_hat * self.phi
                    S2 = torch.fft.ifft2(S2_hat)
                    
                    S['S2'].append({
                        'coef': S2,
                        'j1': j1, 'j2': j2,
                        'theta1': theta1, 'theta2': theta2,
                    })
            
            return S
        
        def extract_features(self, x):
            """Extract flat feature vector."""
            S = self.forward(x)
            
            features = []
            
            # S0 features
            s0 = S['S0']
            features.extend([torch.abs(s0).mean().item(),
                            torch.abs(s0).std().item(),
                            torch.real(s0).mean().item(),
                            torch.imag(s0).mean().item()])
            
            # S1 features
            for s1 in S['S1']:
                c = s1['coef']
                features.extend([torch.abs(c).mean().item(),
                                torch.real(c).mean().item(),
                                torch.imag(c).mean().item()])
            
            # S2 features
            for s2 in S['S2']:
                c = s2['coef']
                features.extend([torch.abs(c).mean().item(),
                                torch.real(c).mean().item(),
                                torch.imag(c).mean().item()])
            
            return np.array(features)
    
    # ========================================
    # FEATURE EXTRACTORS
    # ========================================
    
    def make_extractors(L, device):
        """Create feature extractors to compare."""
        
        # Glinsky HST (wavelet cascade with R)
        hst_glinsky = HST2DSimple(L, J=3, L_orient=4, rectifier='glinsky', device=device)
        
        # Modulus scattering (wavelet cascade with |·|)
        hst_modulus = HST2DSimple(L, J=3, L_orient=4, rectifier='modulus', device=device)
        
        def extract_hst_glinsky(data):
            N = len(data)
            features = []
            for i in range(N):
                feat = hst_glinsky.extract_features(data[i])
                features.append(feat)
            return np.array(features)
        
        def extract_hst_modulus(data):
            N = len(data)
            features = []
            for i in range(N):
                feat = hst_modulus.extract_features(data[i])
                features.append(feat)
            return np.array(features)
        
        def extract_direct_winding(data):
            """Direct winding computation (ground truth)."""
            L = data.shape[1]
            center = (L//2, L//2)
            N = len(data)
            features = np.zeros((N, 1))
            for i in range(N):
                features[i, 0] = compute_local_winding(data[i], center, radius=5)
            return features
        
        return {
            "HST_cascade_R": extract_hst_glinsky,
            "HST_cascade_mod": extract_hst_modulus,
            "Direct_winding": extract_direct_winding,
        }
    
    # ========================================
    # CLASSIFICATION
    # ========================================
    
    def classify(train_feat, train_labels, test_feat, test_labels):
        """Train logistic regression and return accuracy."""
        # Handle complex features
        if np.iscomplexobj(train_feat):
            train_feat = np.hstack([train_feat.real, train_feat.imag])
            test_feat = np.hstack([test_feat.real, test_feat.imag])
        
        # Remove NaN/Inf
        train_feat = np.nan_to_num(train_feat, nan=0, posinf=1e10, neginf=-1e10)
        test_feat = np.nan_to_num(test_feat, nan=0, posinf=1e10, neginf=-1e10)
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(train_feat, train_labels)
        return clf.score(test_feat, test_labels)
    
    # ========================================
    # MAIN EXPERIMENT
    # ========================================
    
    results = {}
    L = 32  # Field size
    n_pairs = 100  # Number of vortex pairs
    
    print("\n" + "="*70)
    print("2D VORTEX CHIRALITY BENCHMARK v3 - PROPER WAVELET CASCADE HST")
    print("="*70)
    
    # Generate paired dataset
    print(f"\n[DATA] Generating {n_pairs} paired vortices at fixed center...")
    
    fixed_center = (L//2, L//2)
    data_plus, data_minus = [], []
    
    for i in range(n_pairs):
        z_plus, z_minus = create_paired_vortices(L, fixed_center, noise_std=0.15, seed=i)
        data_plus.append(z_plus)
        data_minus.append(z_minus)
    
    data_plus = np.array(data_plus)
    data_minus = np.array(data_minus)
    
    # Verify one pair
    w_plus = compute_local_winding(data_plus[0], fixed_center)
    w_minus = compute_local_winding(data_minus[0], fixed_center)
    print(f"[DATA] Verification: winding(z+)={w_plus:.2f}, winding(z-)={w_minus:.2f}")
    print(f"[DATA] Modulus identical: {np.allclose(np.abs(data_plus[0]), np.abs(data_minus[0]))}")
    
    # Group split by pair
    data = np.empty((2 * n_pairs, L, L), dtype=np.complex128)
    labels = np.empty(2 * n_pairs, dtype=int)
    pair_ids = np.empty(2 * n_pairs, dtype=int)
    
    for i in range(n_pairs):
        data[2*i] = data_plus[i]
        data[2*i + 1] = data_minus[i]
        labels[2*i] = +1
        labels[2*i + 1] = -1
        pair_ids[2*i] = i
        pair_ids[2*i + 1] = i
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(data, labels, groups=pair_ids))
    
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    print(f"[DATA] Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    print(f"[DATA] Pairs grouped: no pair split across train/test")
    
    # Create extractors
    extractors = make_extractors(L, str(device))
    
    # Extract features
    print("\n[FEATURES] Extracting...")
    train_feat = {}
    test_feat_orig = {}
    test_feat_rot = {}
    
    for name, extractor in extractors.items():
        t0 = time.time()
        train_feat[name] = extractor(train_data)
        test_feat_orig[name] = extractor(test_data)
        
        # Rotated test (global U(1) phase)
        np.random.seed(999)
        phases = np.random.uniform(0, 2*np.pi, len(test_data))
        test_data_rot = np.array([test_data[i] * np.exp(1j * phases[i]) 
                                  for i in range(len(test_data))])
        test_feat_rot[name] = extractor(test_data_rot)
        
        print(f"[FEATURES] {name}: {train_feat[name].shape[1]} dims, {time.time()-t0:.1f}s")
    
    # Classification
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    print("\nMethod              | Orig Acc   | Rot Acc    | Interpretation")
    print("-" * 70)
    
    task_results = {}
    
    for name in extractors:
        acc_orig = classify(train_feat[name], train_labels, 
                           test_feat_orig[name], test_labels)
        acc_rot = classify(train_feat[name], train_labels,
                          test_feat_rot[name], test_labels)
        
        task_results[name] = {'orig_acc': acc_orig, 'rot_acc': acc_rot}
        
        # Interpret
        if 'winding' in name.lower():
            interp = "GROUND TRUTH ✓" if acc_rot > 0.9 else "GROUND TRUTH FAILED ✗"
        elif 'mod' in name.lower():
            interp = "PHASE-BLIND ✓" if acc_rot < 0.55 else f"UNEXPECTED ({acc_rot:.0%})"
        elif acc_rot > 0.85:
            interp = "CHIRALITY-AWARE ✓"
        elif acc_rot < 0.55:
            interp = "NOT WORKING ✗"
        else:
            interp = "PARTIAL"
        
        print(f"{name:20s}| {acc_orig:.1%}      | {acc_rot:.1%}      | {interp}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    winding = task_results["Direct_winding"]["rot_acc"]
    modulus = task_results["HST_cascade_mod"]["rot_acc"]
    glinsky = task_results["HST_cascade_R"]["rot_acc"]
    
    print(f"\n1. Sanity checks:")
    print(f"   Direct_winding (ground truth): {winding:.1%} {'✓' if winding > 0.9 else '✗'}")
    print(f"   HST_cascade_mod (phase-blind): {modulus:.1%} {'✓' if modulus < 0.55 else '✗'}")
    
    print(f"\n2. Main result:")
    print(f"   HST_cascade_R (Glinsky):       {glinsky:.1%}")
    
    sanity_ok = winding > 0.9 and modulus < 0.55
    
    if sanity_ok:
        print(f"\n3. Conclusion:")
        if glinsky > 0.7:
            print(f"   ✓ GLINSKY HST CASCADE SEES CHIRALITY")
        else:
            print(f"   ✗ Glinsky HST cascade does not see chirality")
    else:
        print(f"\n3. ✗ SANITY CHECK FAILED - results not trustworthy")
    
    return results


@app.local_entrypoint()
def main():
    run_hst_2d_vortex_v3.remote()
