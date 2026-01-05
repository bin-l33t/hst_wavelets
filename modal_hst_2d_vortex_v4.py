"""
2D Vortex Chirality Benchmark v4 - Proper Library Import

This version properly imports the hst_2d library with:
- Two-channel (H+/H-) Paul wavelets covering full Fourier plane
- Glinsky's R = i·ln(R₀) rectifier
- Littlewood-Paley normalized filter bank

Run: modal run modal_hst_2d_vortex_v4.py
"""

import modal

app = modal.App("hst-2d-vortex-v4")

# Build image with dependencies and local source
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10", 
        "scikit-learn>=1.3",
        "torch>=2.0",
    )
    # Add the local hst_2d package to the image
    .add_local_dir("./hst_2d", remote_path="/root/hst_2d")
)


@app.function(image=image, gpu="T4", timeout=1800)
def run_hst_2d_vortex_v4():
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    import time
    
    # Import the library
    from hst_2d import HST2D, create_hst_2d, verify_littlewood_paley, create_filter_bank
    
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
    
    def compute_local_winding(z, center, radius=8):
        """Compute winding number via contour integral around center."""
        L = z.shape[0]
        cy, cx = center
        
        n_points = 128  # More points for accuracy
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
                dtheta = np.angle(z2 / z1)
                winding += dtheta
        
        return winding / (2 * np.pi)
    
    # ========================================
    # CLASSIFICATION HELPER
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
    print("2D VORTEX CHIRALITY BENCHMARK v4")
    print("Proper library import with two-channel Paul wavelets")
    print("="*70)
    
    # ========================================
    # STEP 1: VERIFY FILTER BANK
    # ========================================
    
    print("\n" + "-"*70)
    print("STEP 1: VERIFY FILTER BANK (LITTLEWOOD-PALEY)")
    print("-"*70)
    
    J, L_orient = 3, 4  # 3 scales, 4 orientations
    filters = create_filter_bank(L, L, J, L_orient, normalize=True)
    lp_check = verify_littlewood_paley(filters)
    
    print(f"Filter bank: {filters['info']['n_wavelets']} wavelets "
          f"(J={J} scales × L={L_orient} orientations × 2 channels)")
    print(f"Littlewood-Paley sum of squares:")
    print(f"  min={lp_check['min']:.4f}, max={lp_check['max']:.4f}, "
          f"mean={lp_check['mean']:.4f}, std={lp_check['std']:.4f}")
    print(f"  PASSED: {lp_check['passed']}")
    
    if not lp_check['passed']:
        print("  WARNING: Littlewood-Paley condition not satisfied!")
    
    # ========================================
    # STEP 2: GENERATE DATASET
    # ========================================
    
    print("\n" + "-"*70)
    print("STEP 2: GENERATE PAIRED VORTEX DATASET")
    print("-"*70)
    
    fixed_center = (L//2, L//2)
    data_plus, data_minus = [], []
    
    for i in range(n_pairs):
        z_plus, z_minus = create_paired_vortices(L, fixed_center, noise_std=0.1, seed=i)
        data_plus.append(z_plus)
        data_minus.append(z_minus)
    
    data_plus = np.array(data_plus)
    data_minus = np.array(data_minus)
    
    # Verify
    w_plus = compute_local_winding(data_plus[0], fixed_center)
    w_minus = compute_local_winding(data_minus[0], fixed_center)
    mod_check = np.allclose(np.abs(data_plus[0]), np.abs(data_minus[0]))
    
    print(f"Generated {n_pairs} paired vortices at center={fixed_center}")
    print(f"Verification sample 0:")
    print(f"  winding(z+) = {w_plus:.3f} (expect +1)")
    print(f"  winding(z-) = {w_minus:.3f} (expect -1)")
    print(f"  |z+| == |z-|: {mod_check}")
    
    # Group split
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
    
    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    print(f"Pairs grouped (no split across train/test)")
    
    # ========================================
    # STEP 3: CREATE HST INSTANCES
    # ========================================
    
    print("\n" + "-"*70)
    print("STEP 3: CREATE HST INSTANCES")
    print("-"*70)
    
    hst_glinsky = create_hst_2d(L, L, J=J, L=L_orient, max_order=2, 
                                 rectifier='glinsky', device=str(device))
    hst_modulus = create_hst_2d(L, L, J=J, L=L_orient, max_order=2,
                                 rectifier='modulus', device=str(device))
    
    print(f"HST Glinsky: {hst_glinsky}")
    print(f"HST Modulus: {hst_modulus}")
    
    # ========================================
    # STEP 4: EXTRACT FEATURES
    # ========================================
    
    print("\n" + "-"*70)
    print("STEP 4: EXTRACT FEATURES")
    print("-"*70)
    
    def extract_batch(hst, data_batch, method='default'):
        """Extract features for a batch."""
        features = []
        for i in range(len(data_batch)):
            x = torch.from_numpy(data_batch[i]).to(device)
            if method == 'magnitude_only':
                feat = hst.extract_features_magnitude_only(x)
            else:
                feat = hst.extract_features(x)
            features.append(feat)
        return np.array(features)
    
    def extract_direct_winding(data_batch):
        """Direct winding computation."""
        center = (data_batch.shape[1]//2, data_batch.shape[2]//2)
        features = np.zeros((len(data_batch), 1))
        for i in range(len(data_batch)):
            features[i, 0] = compute_local_winding(data_batch[i], center)
        return features
    
    # Create phase-rotated test set FIRST (before feature extraction)
    print("Creating phase-rotated test set...")
    np.random.seed(999)
    phases = np.random.uniform(0, 2*np.pi, len(test_data))
    test_data_rot = np.array([test_data[i] * np.exp(1j * phases[i]) 
                              for i in range(len(test_data))])
    
    # Extract features with BOTH methods
    feature_sets = {}
    
    # Method 1: Magnitude-only (baseline - should be U(1) invariant but may miss chirality)
    print("\nExtracting MAGNITUDE-ONLY features (U(1)-invariant baseline)...")
    t0 = time.time()
    feature_sets['Glinsky_mag'] = {
        'train': extract_batch(hst_glinsky, train_data, 'magnitude_only'),
        'test_orig': extract_batch(hst_glinsky, test_data, 'magnitude_only'),
        'test_rot': extract_batch(hst_glinsky, test_data_rot, 'magnitude_only'),
    }
    print(f"  Glinsky mag: {feature_sets['Glinsky_mag']['train'].shape[1]} dims, {time.time()-t0:.1f}s")
    
    t0 = time.time()
    feature_sets['Modulus_mag'] = {
        'train': extract_batch(hst_modulus, train_data, 'magnitude_only'),
        'test_orig': extract_batch(hst_modulus, test_data, 'magnitude_only'),
        'test_rot': extract_batch(hst_modulus, test_data_rot, 'magnitude_only'),
    }
    print(f"  Modulus mag: {feature_sets['Modulus_mag']['train'].shape[1]} dims, {time.time()-t0:.1f}s")
    
    # Method 2: Cross-channel features (should capture chirality via H+ * conj(H-))
    print("\nExtracting CROSS-CHANNEL features (U(1)-invariant + chirality)...")
    t0 = time.time()
    feature_sets['Glinsky_cross'] = {
        'train': extract_batch(hst_glinsky, train_data, 'default'),
        'test_orig': extract_batch(hst_glinsky, test_data, 'default'),
        'test_rot': extract_batch(hst_glinsky, test_data_rot, 'default'),
    }
    print(f"  Glinsky cross: {feature_sets['Glinsky_cross']['train'].shape[1]} dims, {time.time()-t0:.1f}s")
    
    t0 = time.time()
    feature_sets['Modulus_cross'] = {
        'train': extract_batch(hst_modulus, train_data, 'default'),
        'test_orig': extract_batch(hst_modulus, test_data, 'default'),
        'test_rot': extract_batch(hst_modulus, test_data_rot, 'default'),
    }
    print(f"  Modulus cross: {feature_sets['Modulus_cross']['train'].shape[1]} dims, {time.time()-t0:.1f}s")
    
    # Direct winding (ground truth)
    print("\nExtracting Direct Winding features...")
    t0 = time.time()
    feature_sets['Direct_Winding'] = {
        'train': extract_direct_winding(train_data),
        'test_orig': extract_direct_winding(test_data),
        'test_rot': extract_direct_winding(test_data_rot),
    }
    print(f"  Direct winding: {feature_sets['Direct_Winding']['train'].shape[1]} dims, {time.time()-t0:.1f}s")
    
    # SANITY CHECK: Verify U(1) invariance of features
    print("\n" + "-"*70)
    print("SANITY CHECK: U(1) INVARIANCE OF FEATURES")
    print("-"*70)
    
    for name, fs in feature_sets.items():
        # Features should be (nearly) identical for orig and rot if U(1)-invariant
        diff = np.abs(fs['test_orig'] - fs['test_rot']).mean()
        max_diff = np.abs(fs['test_orig'] - fs['test_rot']).max()
        print(f"  {name:20s}: mean_diff={diff:.6f}, max_diff={max_diff:.6f}")
        if diff > 0.01:
            print(f"    WARNING: Features are NOT U(1)-invariant!")
    
    # ========================================
    # STEP 5: CLASSIFICATION
    # ========================================
    
    print("\n" + "-"*70)
    print("STEP 5: CLASSIFICATION RESULTS")
    print("-"*70)
    
    print("\nMethod              | Orig Acc   | Rot Acc    | Interpretation")
    print("-" * 70)
    
    task_results = {}
    
    for name, fs in feature_sets.items():
        acc_orig = classify(fs['train'], train_labels, fs['test_orig'], test_labels)
        acc_rot = classify(fs['train'], train_labels, fs['test_rot'], test_labels)
        
        task_results[name] = {'orig_acc': acc_orig, 'rot_acc': acc_rot}
        
        # Interpret based on method type
        if 'winding' in name.lower():
            interp = "GROUND TRUTH ✓" if acc_rot > 0.9 else f"GROUND TRUTH ✗ ({acc_rot:.0%})"
        elif '_mag' in name.lower():
            # Magnitude-only should be ~50% (no chirality info)
            if acc_rot < 0.55 and acc_orig < 0.55:
                interp = "NO CHIRALITY (expected for mag-only)"
            elif abs(acc_rot - acc_orig) < 0.05:
                interp = f"U(1)-INVARIANT ✓ ({acc_rot:.0%})"
            else:
                interp = f"U(1) BROKEN? orig={acc_orig:.0%}, rot={acc_rot:.0%}"
        elif 'modulus' in name.lower():
            if acc_rot < 0.55:
                interp = "PHASE-BLIND ✓ (no chirality)"
            else:
                interp = f"UNEXPECTED chirality ({acc_rot:.0%})"
        elif 'glinsky' in name.lower():
            if acc_rot > 0.85:
                interp = "CHIRALITY-AWARE ✓"
            elif acc_rot > 0.6:
                interp = "PARTIAL chirality"
            else:
                interp = "NO CHIRALITY ✗"
        else:
            interp = ""
        
        print(f"{name:20s}| {acc_orig:.1%}      | {acc_rot:.1%}      | {interp}")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    winding = task_results["Direct_Winding"]["rot_acc"]
    glinsky_mag = task_results["Glinsky_mag"]["rot_acc"]
    modulus_mag = task_results["Modulus_mag"]["rot_acc"]
    glinsky_cross = task_results["Glinsky_cross"]["rot_acc"]
    modulus_cross = task_results["Modulus_cross"]["rot_acc"]
    
    print(f"\n1. Ground truth:")
    print(f"   Direct_Winding: {winding:.1%} {'✓' if winding > 0.9 else '✗'}")
    
    print(f"\n2. Magnitude-only features (U(1)-invariant, no chirality expected):")
    print(f"   Glinsky_mag:  {glinsky_mag:.1%} (expect ~50%)")
    print(f"   Modulus_mag:  {modulus_mag:.1%} (expect ~50%)")
    
    print(f"\n3. Cross-channel features (U(1)-invariant, chirality via H+*conj(H-)):")
    print(f"   Glinsky_cross: {glinsky_cross:.1%}")
    print(f"   Modulus_cross: {modulus_cross:.1%}")
    
    print(f"\n4. Conclusion:")
    
    # Check sanity
    mag_sanity = glinsky_mag < 0.55 and modulus_mag < 0.55
    winding_sanity = winding > 0.9
    
    if not winding_sanity:
        print(f"   ✗ GROUND TRUTH FAILED - dataset or winding calculation broken")
    elif not mag_sanity:
        print(f"   ✗ MAGNITUDE SANITY FAILED - features not properly U(1)-invariant")
    else:
        print(f"   ✓ Sanity checks passed")
        
        if glinsky_cross > 0.7 and modulus_cross < 0.55:
            print(f"   ✓ GLINSKY BEATS MODULUS: {glinsky_cross:.0%} vs {modulus_cross:.0%}")
            print(f"   → R rectifier preserves chirality that modulus discards!")
        elif glinsky_cross > modulus_cross + 0.1:
            print(f"   ~ Glinsky shows improvement: {glinsky_cross:.0%} vs {modulus_cross:.0%}")
        elif glinsky_cross < 0.55:
            print(f"   ✗ Glinsky cross-channel doesn't see chirality ({glinsky_cross:.0%})")
        else:
            print(f"   ? Inconclusive: Glinsky={glinsky_cross:.0%}, Modulus={modulus_cross:.0%}")
    
    results['task_results'] = task_results
    results['sanity'] = {
        'winding_ok': winding_sanity,
        'magnitude_ok': mag_sanity,
    }
    results['littlewood_paley'] = lp_check
    
    return results


@app.local_entrypoint()
def main():
    run_hst_2d_vortex_v4.remote()
