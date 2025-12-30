"""
2D Spatial RG Task v2: Fixed version

Fixes from ChatGPT review:
1. Pooled scattering features (mean over spatial dims)
2. Nontrivial RG target: coarse field reconstruction
3. Trivial baseline check
4. Train-only preprocessing verification
5. |m2| instead of m2 for symmetry consistency
"""

import modal

app = modal.App("ising-2d-rg-v2")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=1800, cpu=4, memory=8192)
def run_rg_experiment_v2(
    L: int = 64,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 10,
    n_eq: int = 500,
    n_gap: int = 20,
    k_values: list = None,
    test_seed_frac: float = 0.3,
) -> dict:
    """
    Run the fixed 2D RG experiment.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score, mean_squared_error
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    if k_values is None:
        k_values = [4, 8, 16, 32]
    
    print(f"Config: L={L}, T={len(T_values)} points, seeds={n_seeds}, snaps={n_snapshots}")
    
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
    def coarse_grain(s, b=2):
        """Block average with block size b."""
        L = s.shape[-1]
        L_coarse = L // b
        if len(s.shape) == 3:
            return s.reshape(s.shape[0], L_coarse, b, L_coarse, b).mean(axis=(2, 4))
        else:
            return s.reshape(L_coarse, b, L_coarse, b).mean(axis=(1, 3))
    
    def compute_scalar_targets(s, b=2):
        """Compute scalar coarse-grained observables."""
        s_coarse = coarse_grain(s, b)
        s_sign = np.sign(s_coarse)
        
        # |m^(2)|: absolute coarse magnetization (symmetry-consistent)
        abs_m2 = np.abs(s_coarse.mean(axis=(1, 2)))
        
        # E^(2): coarse energy
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +
            s_sign * np.roll(s_sign, 1, axis=2)
        ).sum(axis=(1, 2))
        E2 = -nn_sum / (2 * L_c * L_c)
        
        return {"abs_m2": abs_m2, "E2": E2}
    
    def get_coarse_field(s, b=2):
        """Get flattened coarse field as target for reconstruction."""
        s_coarse = coarse_grain(s, b)
        # Flatten each coarse field
        return s_coarse.reshape(s_coarse.shape[0], -1)
    
    # ===== FEATURE EXTRACTION =====
    def extract_scattering_pooled(snapshots):
        """Extract POOLED Kymatio Scattering2D features."""
        from kymatio.numpy import Scattering2D
        
        L = snapshots.shape[1]
        J = 3
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]  # (N, 1, L, L)
        Sx = scattering(x)  # Output shape varies by Kymatio version
        
        # Debug shape
        print(f"  Scattering output shape: {Sx.shape}")
        
        # Handle different output shapes
        if len(Sx.shape) == 4:
            # (N, C, L', L') - pool over spatial
            features = Sx.mean(axis=(-1, -2))  # (N, C)
        elif len(Sx.shape) == 3:
            # (N, 1, C) or (N, C, 1) - squeeze
            features = Sx.squeeze()
            if len(features.shape) == 1:
                features = features[:, np.newaxis]
        else:
            # Fallback: flatten all but batch dim
            features = Sx.reshape(Sx.shape[0], -1)
        
        # Ensure 2D
        if len(features.shape) != 2:
            features = features.reshape(features.shape[0], -1)
        
        print(f"  Scattering features shape: {features.shape}")
        return features
    
    def extract_raw_pooled(snapshots):
        """Raw features with some pooling structure."""
        # Pool to 8x8 then flatten (comparable dim to scattering)
        N, L, _ = snapshots.shape
        pool_size = L // 8
        pooled = snapshots.reshape(N, 8, pool_size, 8, pool_size).mean(axis=(2, 4))
        return pooled.reshape(N, -1)  # (N, 64)
    
    def extract_fft_radial(snapshots, n_bins=16):
        """2D FFT magnitude in radial bins."""
        N, L, _ = snapshots.shape
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            r_max = L // 2
            bins = np.linspace(0, r_max, n_bins + 1)
            radial_profile = np.zeros(n_bins)
            for j in range(n_bins):
                mask = (r >= bins[j]) & (r < bins[j+1])
                if mask.sum() > 0:
                    radial_profile[j] = mag[mask].mean()
            
            features.append(radial_profile)
        
        return np.array(features)
    
    # ===== TRIVIAL BASELINES =====
    def trivial_abs_m2(snapshots):
        """Compute |m| directly - this is what we're predicting."""
        return np.abs(snapshots.mean(axis=(1, 2)))
    
    def trivial_E2(snapshots, b=2):
        """Compute E2 directly from fine field."""
        # This approximates E2 without going through features
        s_coarse = coarse_grain(snapshots, b)
        s_sign = np.sign(s_coarse)
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +
            s_sign * np.roll(s_sign, 1, axis=2)
        ).sum(axis=(1, 2))
        return -nn_sum / (2 * L_c * L_c)
    
    # ===== GENERATE ALL DATA =====
    print("Generating snapshots...")
    
    all_data = []
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
            scalar_targets = compute_scalar_targets(snaps, b=2)
            coarse_field = get_coarse_field(snaps, b=2)
            
            all_data.append({
                "seed": seed,
                "T": T,
                "snapshots": snaps,
                "scalar_targets": scalar_targets,
                "coarse_field": coarse_field,  # (N, (L/2)^2) for reconstruction
            })
    
    print(f"Generated {len(all_data)} (T, seed) combinations")
    
    # ===== EXTRACT FEATURES =====
    print("Extracting features...")
    
    for item in all_data:
        snaps = item["snapshots"]
        item["feat_scatter"] = extract_scattering_pooled(snaps)
        item["feat_raw"] = extract_raw_pooled(snaps)
        item["feat_fft"] = extract_fft_radial(snaps)
    
    print(f"Feature dims: scatter={all_data[0]['feat_scatter'].shape[1]}, "
          f"raw={all_data[0]['feat_raw'].shape[1]}, "
          f"fft={all_data[0]['feat_fft'].shape[1]}")
    
    # ===== TRAIN/TEST SPLIT BY SEED =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    
    print(f"Train seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== RESULTS STORAGE =====
    results = {
        "config": {
            "L": L,
            "T_values": T_values,
            "n_seeds": n_seeds,
            "n_snapshots": n_snapshots,
            "k_values": k_values,
            "train_seeds": list(train_seeds),
            "test_seeds": list(test_seeds),
        },
        "trivial_baselines": {},
        "scalar_targets": {},
        "coarse_reconstruction": {},
    }
    
    feature_types = ["scatter", "raw", "fft"]
    
    # ===== TASK 1: SCALAR TARGET PREDICTION =====
    print("\n=== TASK 1: Scalar target prediction ===")
    
    for T in T_values:
        print(f"Processing T={T:.3f}...")
        T_key = f"{T:.4f}"
        results["scalar_targets"][T_key] = {}
        results["trivial_baselines"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        # Trivial baseline: compute target directly
        test_snaps = np.concatenate([d["snapshots"] for d in test_data])
        trivial_abs_m2_pred = trivial_abs_m2(test_snaps)
        trivial_E2_pred = trivial_E2(test_snaps)
        
        y_abs_m2_test = np.concatenate([d["scalar_targets"]["abs_m2"] for d in test_data])
        y_E2_test = np.concatenate([d["scalar_targets"]["E2"] for d in test_data])
        
        # R² for trivial baseline (should be ~1.0)
        results["trivial_baselines"][T_key]["abs_m2"] = float(r2_score(y_abs_m2_test, trivial_abs_m2_pred))
        results["trivial_baselines"][T_key]["E2"] = float(r2_score(y_E2_test, trivial_E2_pred))
        
        for target_name in ["abs_m2", "E2"]:
            results["scalar_targets"][T_key][target_name] = {}
            
            y_train = np.concatenate([d["scalar_targets"][target_name] for d in train_data])
            y_test = np.concatenate([d["scalar_targets"][target_name] for d in test_data])
            
            for feat_type in feature_types:
                feat_key = f"feat_{feat_type}"
                
                # TRAIN-ONLY preprocessing
                X_train = np.concatenate([d[feat_key] for d in train_data])
                X_test = np.concatenate([d[feat_key] for d in test_data])
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)  # Fit on train only
                X_test_s = scaler.transform(X_test)  # Transform test
                
                X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
                X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
                
                r2_by_k = {}
                max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
                
                if max_k > 0:
                    pca = PCA(n_components=max_k)
                    X_train_pca = pca.fit_transform(X_train_s)  # Fit on train only
                    X_test_pca = pca.transform(X_test_s)
                    
                    for k in k_values:
                        if k > max_k:
                            continue
                        
                        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                        ridge.fit(X_train_pca[:, :k], y_train)
                        y_pred = ridge.predict(X_test_pca[:, :k])
                        
                        r2_by_k[k] = float(r2_score(y_test, y_pred))
                
                results["scalar_targets"][T_key][target_name][feat_type] = r2_by_k
    
    # ===== TASK 2: COARSE FIELD RECONSTRUCTION =====
    print("\n=== TASK 2: Coarse field reconstruction ===")
    
    for T in T_values:
        print(f"Processing T={T:.3f}...")
        T_key = f"{T:.4f}"
        results["coarse_reconstruction"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        # Target: flattened coarse field
        Y_train = np.concatenate([d["coarse_field"] for d in train_data])
        Y_test = np.concatenate([d["coarse_field"] for d in test_data])
        
        coarse_dim = Y_train.shape[1]  # (L/2)^2
        
        for feat_type in feature_types:
            feat_key = f"feat_{feat_type}"
            
            X_train = np.concatenate([d[feat_key] for d in train_data])
            X_test = np.concatenate([d[feat_key] for d in test_data])
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
            X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
            
            mse_by_k = {}
            r2_by_k = {}
            max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
            
            if max_k > 0:
                pca = PCA(n_components=max_k)
                X_train_pca = pca.fit_transform(X_train_s)
                X_test_pca = pca.transform(X_test_s)
                
                for k in k_values:
                    if k > max_k:
                        continue
                    
                    # Multi-output ridge regression
                    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                    ridge.fit(X_train_pca[:, :k], Y_train)
                    Y_pred = ridge.predict(X_test_pca[:, :k])
                    
                    mse = mean_squared_error(Y_test, Y_pred)
                    # Per-pixel R² (average over pixels)
                    r2_pixels = []
                    for p in range(coarse_dim):
                        if Y_test[:, p].std() > 1e-10:
                            r2_pixels.append(r2_score(Y_test[:, p], Y_pred[:, p]))
                    r2_avg = np.mean(r2_pixels) if r2_pixels else 0.0
                    
                    mse_by_k[k] = float(mse)
                    r2_by_k[k] = float(r2_avg)
            
            results["coarse_reconstruction"][T_key][feat_type] = {
                "mse": mse_by_k,
                "r2": r2_by_k,
            }
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK v2 (FIXED)")
    print("="*70)
    print("Fixes: pooled scattering, |m2| target, coarse field reconstruction")
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    results = run_rg_experiment_v2.remote(
        L=64,
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=10,
        k_values=[4, 8, 16, 32],
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v2_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    import numpy as np
    
    T_values = results["config"]["T_values"]
    k_values = results["config"]["k_values"]
    
    print("\n" + "="*70)
    print("TRIVIAL BASELINES (should be ~1.0 if target is directly computable)")
    print("="*70)
    
    print(f"\n{'T':<10} | {'|m2| trivial':<15} {'E2 trivial':<15}")
    print("-"*45)
    for T in T_values:
        T_key = f"{T:.4f}"
        trivial = results["trivial_baselines"].get(T_key, {})
        abs_m2 = trivial.get("abs_m2", "N/A")
        E2 = trivial.get("E2", "N/A")
        marker = "*" if abs(T - TC) < 0.01 else " "
        print(f"{T:<9.3f}{marker} | {abs_m2:<15.3f} {E2:<15.3f}")
    
    print("\n" + "="*70)
    print("TASK 1: SCALAR TARGET PREDICTION (R² at k=16)")
    print("="*70)
    
    print(f"\n{'T':<10} {'Target':<8} | {'Scatter':<10} {'Raw':<10} {'FFT':<10} | Winner")
    print("-"*65)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["scalar_targets"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m2", "E2"]:
            target_data = T_data.get(target, {})
            
            scores = {}
            for feat in ["scatter", "raw", "fft"]:
                feat_data = target_data.get(feat, {})
                r2 = feat_data.get(16, feat_data.get("16", None))
                scores[feat] = r2 if r2 is not None else -999
            
            winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
            
            print(f"{T:<9.3f}{marker} {target:<8} | ", end="")
            for feat in ["scatter", "raw", "fft"]:
                r2 = scores[feat]
                if r2 > -999:
                    print(f"{r2:<10.3f} ", end="")
                else:
                    print(f"{'N/A':<10} ", end="")
            print(f"| {winner}")
    
    print("\n" + "="*70)
    print("TASK 2: COARSE FIELD RECONSTRUCTION (avg R² at k=16)")
    print("="*70)
    
    print(f"\n{'T':<10} | {'Scatter':<10} {'Raw':<10} {'FFT':<10} | Winner")
    print("-"*55)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["coarse_reconstruction"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        scores = {}
        for feat in ["scatter", "raw", "fft"]:
            feat_data = T_data.get(feat, {})
            r2_dict = feat_data.get("r2", {})
            r2 = r2_dict.get(16, r2_dict.get("16", None))
            scores[feat] = r2 if r2 is not None else -999
        
        winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
        
        print(f"{T:<9.3f}{marker} | ", end="")
        for feat in ["scatter", "raw", "fft"]:
            r2 = scores[feat]
            if r2 > -999:
                print(f"{r2:<10.3f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
        print(f"| {winner}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
