"""
2D Spatial RG Task v4: Proper RG benchmark

Key fixes from ChatGPT:
1. Cross-scale with explicit leakage prevention (no scales >= b)
2. Locked evaluation protocol: same pipeline for ALL methods
3. Apples-to-apples k-budget comparison
4. Cross-temperature generalization test

Protocol for ALL methods:
  features → standardize (train-only fit) → PCA(k) → ridge
  Same k grid, same splits, report mean±std across seeds
"""

import modal

app = modal.App("ising-2d-rg-v4")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=3600, cpu=4, memory=8192)
def run_rg_experiment_v4(
    L: int = 64,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 20,
    n_eq: int = 1000,
    n_gap: int = 50,
    coarse_b: int = 4,  # Block size for coarse-graining
    k_values: list = None,
    test_seed_frac: float = 0.3,
) -> dict:
    """
    Run v4 RG experiment with proper cross-scale constraints.
    
    Key: features computed at scales < b, targets at scale b.
    This prevents trivial leakage where DC directly encodes magnetization.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    if k_values is None:
        k_values = [4, 8, 16, 32, 64]
    
    print(f"Config: L={L}, T={len(T_values)} points, seeds={n_seeds}, snaps={n_snapshots}")
    print(f"Coarse block size b={coarse_b}")
    print(f"k values: {k_values}")
    
    # ===== ISING SIMULATION =====
    def generate_snapshots(L, T, seed, n_eq, n_snap, n_gap):
        rng = np.random.default_rng(seed)
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.float32)
        
        def wolff_step():
            p_add = 1 - np.exp(-2.0 / T)
            i0, j0 = rng.integers(0, L), rng.integers(0, L)
            seed_spin = spins[i0, j0]
            cluster = {(i0, j0)}
            stack = [(i0, j0)]
            while stack:
                i, j = stack.pop()
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = (i + di) % L, (j + dj) % L
                    if (ni, nj) not in cluster and spins[ni, nj] == seed_spin:
                        if rng.random() < p_add:
                            cluster.add((ni, nj))
                            stack.append((ni, nj))
            for i, j in cluster:
                spins[i, j] *= -1
        
        for _ in range(n_eq):
            wolff_step()
        
        snapshots = []
        for _ in range(n_snap):
            for _ in range(n_gap):
                wolff_step()
            snapshots.append(spins.copy())
        
        return np.array(snapshots)
    
    # ===== COARSE GRAINING =====
    def coarse_grain(s, b):
        """Block average with block size b."""
        L = s.shape[-1]
        L_coarse = L // b
        if len(s.shape) == 3:
            return s.reshape(s.shape[0], L_coarse, b, L_coarse, b).mean(axis=(2, 4))
        else:
            return s.reshape(L_coarse, b, L_coarse, b).mean(axis=(1, 3))
    
    def compute_targets(s, b):
        """Compute coarse-grained targets at scale b."""
        s_coarse = coarse_grain(s, b)
        s_sign = np.sign(s_coarse)
        
        # |m^(b)|: absolute coarse magnetization
        abs_m = np.abs(s_coarse.mean(axis=(1, 2)))
        
        # E^(b): coarse energy
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +
            s_sign * np.roll(s_sign, 1, axis=2)
        ).sum(axis=(1, 2))
        E = -nn_sum / (2 * L_c * L_c)
        
        return {"abs_m": abs_m, "E": E}
    
    # ===== FEATURE EXTRACTION WITH SCALE CONSTRAINTS =====
    
    def extract_fft_high_freq(snapshots, b, n_bins=16):
        """
        FFT features with LOW frequencies REMOVED (no scales >= b).
        Only keep frequencies above the coarse cutoff.
        """
        N, L, _ = snapshots.shape
        features = []
        
        # Cutoff: frequencies corresponding to wavelength >= b*2 are removed
        # In FFT terms, keep |k| > L/(b*2)
        k_cutoff = L // (b * 2)
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Only use frequencies ABOVE cutoff (high-freq only)
            # This removes DC and low-frequency components
            r_min = k_cutoff  # Minimum radius (exclude low freq)
            r_max = L // 2
            
            bins = np.linspace(r_min, r_max, n_bins + 1)
            radial_profile = np.zeros(n_bins)
            for j in range(n_bins):
                mask = (r >= bins[j]) & (r < bins[j+1])
                if mask.sum() > 0:
                    radial_profile[j] = mag[mask].mean()
            
            features.append(radial_profile)
        
        return np.array(features)
    
    def extract_raw_no_blockmean(snapshots, b, pool_size=4):
        """
        Raw features with block-means at scale b REMOVED.
        Project out the coarse structure to prevent leakage.
        """
        N, L, _ = snapshots.shape
        
        # Compute and subtract block means at scale b
        s_coarse = coarse_grain(snapshots, b)  # (N, L/b, L/b)
        
        # Upsample coarse back to fine resolution
        s_coarse_up = np.repeat(np.repeat(s_coarse, b, axis=1), b, axis=2)
        
        # Residual: fine - coarse (removes block-mean structure)
        s_residual = snapshots - s_coarse_up  # (N, L, L)
        
        # Pool residuals to manageable size
        L_pool = L // pool_size
        pooled = s_residual.reshape(N, L_pool, pool_size, L_pool, pool_size).mean(axis=(2, 4))
        
        features = pooled.reshape(N, -1)
        
        # Add small noise to prevent zero variance (for numerical stability)
        # This is a tiny perturbation that won't affect results but prevents NaN
        features = features + np.random.randn(*features.shape) * 1e-10
        
        return features
    
    def extract_scattering_small_scales(snapshots, b):
        """
        Scattering features using ONLY scales smaller than b.
        Exclude lowpass and large-scale wavelet channels.
        """
        from kymatio.numpy import Scattering2D
        
        N, L, _ = snapshots.shape
        
        # J controls max scale: 2^J pixels
        # We want scales < b, so J_max = floor(log2(b)) - 1
        # For b=4: max useful scale is 2^1 = 2, so J=1 or 2
        J_max = max(1, int(np.floor(np.log2(b))) - 1)
        
        print(f"  Scattering with J={J_max} (scales < {b})")
        
        scattering = Scattering2D(J=J_max, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        print(f"  Scattering output shape: {Sx.shape}")
        
        # Pool spatially to get (N, C)
        if len(Sx.shape) == 4:
            # (N, C, H, W) -> (N, C)
            # IMPORTANT: Exclude channel 0 which is the lowpass (contains DC/mean)
            Sx_no_lp = Sx[:, 1:, :, :]  # Remove lowpass channel
            features = Sx_no_lp.mean(axis=(-1, -2))
        elif len(Sx.shape) == 3:
            # (N, C, 1) or similar
            Sx_squeezed = Sx.squeeze()
            if len(Sx_squeezed.shape) == 2:
                features = Sx_squeezed[:, 1:]  # Remove lowpass
            else:
                features = Sx_squeezed.reshape(N, -1)
        else:
            features = Sx.reshape(N, -1)
        
        if len(features.shape) != 2:
            features = features.reshape(N, -1)
        
        print(f"  Scattering features shape (after removing lowpass): {features.shape}")
        return features
    
    # ===== GENERATE ALL DATA =====
    print("\nGenerating snapshots...")
    
    all_data = []
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
            targets = compute_targets(snaps, coarse_b)
            
            all_data.append({
                "seed": seed,
                "T": T,
                "snapshots": snaps,
                "targets": targets,
            })
    
    print(f"Generated {len(all_data)} (T, seed) combinations")
    print(f"Total snapshots: {len(all_data) * n_snapshots}")
    
    # ===== EXTRACT FEATURES (with scale constraints) =====
    print("\nExtracting features with scale constraints (scales < b)...")
    
    for item in all_data:
        snaps = item["snapshots"]
        item["feat_fft"] = extract_fft_high_freq(snaps, coarse_b)
        item["feat_raw"] = extract_raw_no_blockmean(snaps, coarse_b)
        item["feat_scatter"] = extract_scattering_small_scales(snaps, coarse_b)
    
    # Get feature dimensions
    feat_dims = {
        "fft": all_data[0]["feat_fft"].shape[1],
        "raw": all_data[0]["feat_raw"].shape[1],
        "scatter": all_data[0]["feat_scatter"].shape[1],
    }
    print(f"Feature dimensions: {feat_dims}")
    
    # ===== TRAIN/TEST SPLIT BY SEED =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    
    print(f"Train seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== LOCKED EVALUATION PROTOCOL =====
    # Same pipeline for ALL methods:
    # features → standardize (train fit) → PCA(k) → ridge
    
    def evaluate_method(X_train, X_test, y_train, y_test, k_values):
        """
        Locked evaluation protocol.
        Returns dict: {k: r2_score}
        """
        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Handle NaN/Inf
        X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
        
        # Check for zero-variance columns and remove them
        var = np.var(X_train_s, axis=0)
        valid_cols = var > 1e-10
        if valid_cols.sum() == 0:
            print(f"    Warning: All features have zero variance!")
            return {k: None for k in k_values}
        
        X_train_s = X_train_s[:, valid_cols]
        X_test_s = X_test_s[:, valid_cols]
        
        results = {}
        max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
        
        if max_k < min(k_values):
            print(f"    Warning: max_k={max_k} < min(k_values)={min(k_values)}")
            return {k: None for k in k_values}
        
        # PCA (fit on train only)
        pca = PCA(n_components=max_k)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        
        for k in k_values:
            if k > max_k:
                results[k] = None
                continue
            
            # Ridge with CV
            ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_train_pca[:, :k], y_train)
            y_pred = ridge.predict(X_test_pca[:, :k])
            
            results[k] = float(r2_score(y_test, y_pred))
        
        return results
    
    # ===== RESULTS STORAGE =====
    results = {
        "config": {
            "L": L,
            "T_values": T_values,
            "n_seeds": n_seeds,
            "n_snapshots": n_snapshots,
            "coarse_b": coarse_b,
            "k_values": k_values,
            "train_seeds": list(train_seeds),
            "test_seeds": list(test_seeds),
            "feat_dims": feat_dims,
        },
        "task1_same_T": {},  # Predict coarse obs at same T
        "task2_cross_T": {},  # Train on subset of T, test on held-out T
    }
    
    feature_types = ["fft", "raw", "scatter"]
    target_names = ["abs_m", "E"]
    
    # ===== TASK 1: SAME-T PREDICTION (with scale constraints) =====
    print("\n" + "="*60)
    print("TASK 1: Same-T prediction (features at scale<b, target at scale b)")
    print("="*60)
    
    for T in T_values:
        print(f"\nProcessing T={T:.3f}...")
        T_key = f"{T:.4f}"
        results["task1_same_T"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        for target_name in target_names:
            results["task1_same_T"][T_key][target_name] = {}
            
            y_train = np.concatenate([d["targets"][target_name] for d in train_data])
            y_test = np.concatenate([d["targets"][target_name] for d in test_data])
            
            # Standardize target (train-only fit)
            y_mean, y_std = y_train.mean(), y_train.std()
            if y_std < 1e-10:
                y_std = 1.0
            y_train_s = (y_train - y_mean) / y_std
            y_test_s = (y_test - y_mean) / y_std
            
            for feat_type in feature_types:
                feat_key = f"feat_{feat_type}"
                
                X_train = np.concatenate([d[feat_key] for d in train_data])
                X_test = np.concatenate([d[feat_key] for d in test_data])
                
                r2_by_k = evaluate_method(X_train, X_test, y_train_s, y_test_s, k_values)
                results["task1_same_T"][T_key][target_name][feat_type] = r2_by_k
    
    # ===== TASK 2: CROSS-T GENERALIZATION =====
    print("\n" + "="*60)
    print("TASK 2: Cross-T generalization")
    print("Train on T ∈ {1.5, 2.0, 3.0}, test on T near Tc")
    print("="*60)
    
    # Define train/test T splits
    train_T = [1.5, 2.0, 3.0]
    test_T = [t for t in T_values if abs(t - TC) < 0.2]  # Near Tc
    
    print(f"Train T: {train_T}")
    print(f"Test T: {test_T}")
    
    # Collect train data (all seeds from train_T)
    train_data_cross = [d for d in all_data if any(abs(d["T"] - t) < 0.01 for t in train_T)]
    
    for test_T_val in test_T:
        T_key = f"{test_T_val:.4f}"
        results["task2_cross_T"][T_key] = {}
        
        test_data_cross = [d for d in all_data if abs(d["T"] - test_T_val) < 0.01]
        
        print(f"\nTest T={test_T_val:.3f}, train samples={len(train_data_cross)*n_snapshots}, test samples={len(test_data_cross)*n_snapshots}")
        
        for target_name in target_names:
            results["task2_cross_T"][T_key][target_name] = {}
            
            y_train = np.concatenate([d["targets"][target_name] for d in train_data_cross])
            y_test = np.concatenate([d["targets"][target_name] for d in test_data_cross])
            
            y_mean, y_std = y_train.mean(), y_train.std()
            if y_std < 1e-10:
                y_std = 1.0
            y_train_s = (y_train - y_mean) / y_std
            y_test_s = (y_test - y_mean) / y_std
            
            for feat_type in feature_types:
                feat_key = f"feat_{feat_type}"
                
                X_train = np.concatenate([d[feat_key] for d in train_data_cross])
                X_test = np.concatenate([d[feat_key] for d in test_data_cross])
                
                r2_by_k = evaluate_method(X_train, X_test, y_train_s, y_test_s, k_values)
                results["task2_cross_T"][T_key][target_name][feat_type] = r2_by_k
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK v4: PROPER RG BENCHMARK")
    print("="*70)
    print("Key: Cross-scale with leakage prevention, locked evaluation protocol")
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    results = run_rg_experiment_v4.remote(
        L=64,
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=20,
        coarse_b=4,
        k_values=[4, 8, 16, 32, 64],
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v4_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    import numpy as np
    
    T_values = results["config"]["T_values"]
    k_values = results["config"]["k_values"]
    feat_dims = results["config"]["feat_dims"]
    
    print("\n" + "="*70)
    print(f"FEATURE DIMENSIONS (with scale < b={results['config']['coarse_b']} constraint)")
    print("="*70)
    for feat, dim in feat_dims.items():
        print(f"  {feat}: {dim}")
    
    print("\n" + "="*70)
    print("TASK 1: SAME-T PREDICTION (R² at each k)")
    print("="*70)
    
    for target in ["abs_m", "E"]:
        print(f"\n--- Target: {target} ---")
        print(f"{'T':<8} | ", end="")
        for k in k_values:
            print(f"k={k:<3} ", end="")
        print("| Best@k=16")
        print("-"*60)
        
        for T in T_values:
            T_key = f"{T:.4f}"
            T_data = results["task1_same_T"].get(T_key, {})
            target_data = T_data.get(target, {})
            
            marker = "*" if abs(T - TC) < 0.01 else " "
            print(f"{T:<7.3f}{marker} | ", end="")
            
            # Find best at k=16
            best_feat = None
            best_r2 = -999
            
            for k in k_values:
                # Average across feature types for display
                r2_scatter = target_data.get("scatter", {}).get(k)
                if r2_scatter is not None:
                    print(f"{r2_scatter:5.2f} ", end="")
                else:
                    print(f"{'N/A':>5} ", end="")
            
            # Find winner at k=16
            for feat in ["fft", "raw", "scatter"]:
                r2 = target_data.get(feat, {}).get(16)
                if r2 is not None and r2 > best_r2:
                    best_r2 = r2
                    best_feat = feat
            
            print(f"| {best_feat}={best_r2:.2f}" if best_feat else "| N/A")
    
    print("\n" + "="*70)
    print("TASK 1: COMPARISON AT k=16")
    print("="*70)
    
    print(f"\n{'T':<8} {'Target':<6} | {'FFT':<8} {'Raw':<8} {'Scatter':<8} | Winner")
    print("-"*60)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["task1_same_T"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m", "E"]:
            target_data = T_data.get(target, {})
            
            scores = {}
            for feat in ["fft", "raw", "scatter"]:
                r2 = target_data.get(feat, {}).get(16)
                scores[feat] = r2 if r2 is not None else -999
            
            winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
            
            print(f"{T:<7.3f}{marker} {target:<6} | ", end="")
            for feat in ["fft", "raw", "scatter"]:
                r2 = scores[feat]
                if r2 > -999:
                    print(f"{r2:<8.3f} ", end="")
                else:
                    print(f"{'N/A':<8} ", end="")
            print(f"| {winner}")
    
    print("\n" + "="*70)
    print("TASK 2: CROSS-T GENERALIZATION (Train: 1.5,2.0,3.0 → Test: near Tc)")
    print("="*70)
    
    print(f"\n{'Test T':<8} {'Target':<6} | {'FFT':<8} {'Raw':<8} {'Scatter':<8} | Winner")
    print("-"*60)
    
    for T_key, T_data in results["task2_cross_T"].items():
        T = float(T_key)
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m", "E"]:
            target_data = T_data.get(target, {})
            
            scores = {}
            for feat in ["fft", "raw", "scatter"]:
                r2 = target_data.get(feat, {}).get(16)
                scores[feat] = r2 if r2 is not None else -999
            
            winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
            
            print(f"{T:<7.3f}{marker} {target:<6} | ", end="")
            for feat in ["fft", "raw", "scatter"]:
                r2 = scores[feat]
                if r2 > -999:
                    print(f"{r2:<8.3f} ", end="")
                else:
                    print(f"{'N/A':<8} ", end="")
            print(f"| {winner}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
