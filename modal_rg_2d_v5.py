"""
2D Spatial RG Task v5: Fixed metrics and preprocessing

Fixes from ChatGPT:
A) Well-defined metrics: report std(y), skip R² when variance too low
B) Cross-T: single scaler/PCA on union of train temperatures
C) Log scale constraints explicitly
D) Add temperature prediction task (better RG test)
"""

import modal

app = modal.App("ising-2d-rg-v5")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=3600, cpu=4, memory=8192)
def run_rg_experiment_v5(
    L: int = 64,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 20,
    n_eq: int = 1000,
    n_gap: int = 50,
    coarse_b: int = 4,
    k_values: list = None,
    test_seed_frac: float = 0.3,
    min_std_for_r2: float = 1e-3,  # Skip R² if std(y) below this
) -> dict:
    """
    Run v5 RG experiment with fixed metrics and preprocessing.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    if k_values is None:
        k_values = [4, 8, 16, 32, 64]
    
    print(f"Config: L={L}, T={len(T_values)} points, seeds={n_seeds}, snaps={n_snapshots}")
    print(f"Coarse block size b={coarse_b}")
    print(f"k values: {k_values}")
    print(f"Min std for R²: {min_std_for_r2}")
    
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
        L = s.shape[-1]
        L_coarse = L // b
        if len(s.shape) == 3:
            return s.reshape(s.shape[0], L_coarse, b, L_coarse, b).mean(axis=(2, 4))
        else:
            return s.reshape(L_coarse, b, L_coarse, b).mean(axis=(1, 3))
    
    def compute_targets(s, b, T):
        """Compute targets including temperature label."""
        s_coarse = coarse_grain(s, b)
        s_sign = np.sign(s_coarse)
        
        abs_m = np.abs(s_coarse.mean(axis=(1, 2)))
        
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +
            s_sign * np.roll(s_sign, 1, axis=2)
        ).sum(axis=(1, 2))
        E = -nn_sum / (2 * L_c * L_c)
        
        # Temperature label (for T prediction task)
        T_label = np.full(s.shape[0], T)
        
        return {"abs_m": abs_m, "E": E, "T": T_label}
    
    # ===== FEATURE EXTRACTION WITH SCALE CONSTRAINTS =====
    
    def extract_fft_high_freq(snapshots, b, n_bins=16):
        """
        FFT features with LOW frequencies REMOVED.
        Explicitly log what's included.
        """
        N, L, _ = snapshots.shape
        
        # Cutoff: wavelength >= b*2 removed
        k_cutoff = L // (b * 2)
        print(f"  FFT: k_cutoff={k_cutoff} (removing wavelengths >= {b*2})")
        print(f"  FFT: keeping frequencies k > {k_cutoff} out of max {L//2}")
        
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            r_min = k_cutoff
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
        Raw features with block-means REMOVED.
        """
        N, L, _ = snapshots.shape
        
        s_coarse = coarse_grain(snapshots, b)
        s_coarse_up = np.repeat(np.repeat(s_coarse, b, axis=1), b, axis=2)
        s_residual = snapshots - s_coarse_up
        
        L_pool = L // pool_size
        pooled = s_residual.reshape(N, L_pool, pool_size, L_pool, pool_size).mean(axis=(2, 4))
        
        features = pooled.reshape(N, -1)
        
        # Small noise for numerical stability
        rng = np.random.default_rng(42)
        features = features + rng.standard_normal(features.shape) * 1e-10
        
        print(f"  Raw: removed block-means at scale b={b}, residual pooled to {L_pool}x{L_pool}")
        
        return features
    
    def extract_scattering_small_scales(snapshots, b):
        """
        Scattering with ONLY scales < b.
        """
        from kymatio.numpy import Scattering2D
        
        N, L, _ = snapshots.shape
        
        # J_max such that 2^J_max < b
        J_max = max(1, int(np.floor(np.log2(b))) - 1)
        
        print(f"  Scattering: J={J_max} (max scale 2^{J_max}={2**J_max} < b={b})")
        
        scattering = Scattering2D(J=J_max, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        print(f"  Scattering output shape: {Sx.shape}")
        
        # Pool spatially, EXCLUDE lowpass (channel 0)
        if len(Sx.shape) == 4:
            Sx_no_lp = Sx[:, 1:, :, :]
            features = Sx_no_lp.mean(axis=(-1, -2))
            print(f"  Scattering: excluded lowpass, keeping {features.shape[1]} channels")
        else:
            features = Sx.reshape(N, -1)
        
        if len(features.shape) != 2:
            features = features.reshape(N, -1)
        
        return features
    
    # ===== GENERATE ALL DATA =====
    print("\nGenerating snapshots...")
    
    all_data = []
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
            targets = compute_targets(snaps, coarse_b, T)
            
            all_data.append({
                "seed": seed,
                "T": T,
                "snapshots": snaps,
                "targets": targets,
            })
    
    print(f"Generated {len(all_data)} (T, seed) combinations")
    
    # ===== EXTRACT FEATURES =====
    print("\nExtracting features with scale constraints...")
    
    # Only print diagnostics once
    first = True
    for item in all_data:
        snaps = item["snapshots"]
        if first:
            item["feat_fft"] = extract_fft_high_freq(snaps, coarse_b)
            item["feat_raw"] = extract_raw_no_blockmean(snaps, coarse_b)
            item["feat_scatter"] = extract_scattering_small_scales(snaps, coarse_b)
            first = False
        else:
            # Suppress prints for subsequent extractions
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            item["feat_fft"] = extract_fft_high_freq(snaps, coarse_b)
            item["feat_raw"] = extract_raw_no_blockmean(snaps, coarse_b)
            item["feat_scatter"] = extract_scattering_small_scales(snaps, coarse_b)
            sys.stdout = old_stdout
    
    feat_dims = {
        "fft": all_data[0]["feat_fft"].shape[1],
        "raw": all_data[0]["feat_raw"].shape[1],
        "scatter": all_data[0]["feat_scatter"].shape[1],
    }
    print(f"\nFeature dimensions: {feat_dims}")
    
    # ===== TRAIN/TEST SPLIT BY SEED =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    
    print(f"Train seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== EVALUATION WITH PROPER METRICS =====
    
    def evaluate_method(X_train, X_test, y_train, y_test, k_values, min_std):
        """
        Locked evaluation with proper metrics.
        Returns dict with R², MSE, MAE, and diagnostics.
        """
        # Compute target statistics
        std_train = float(np.std(y_train))
        std_test = float(np.std(y_test))
        mean_train = float(np.mean(y_train))
        
        # Baseline: predict mean
        baseline_mse = float(np.mean((y_test - mean_train)**2))
        baseline_mae = float(np.mean(np.abs(y_test - mean_train)))
        
        # Standardize features (fit on train)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
        
        # Remove zero-variance columns
        var = np.var(X_train_s, axis=0)
        valid_cols = var > 1e-10
        if valid_cols.sum() == 0:
            return {
                "error": "all_zero_variance",
                "std_train": std_train,
                "std_test": std_test,
            }
        
        X_train_s = X_train_s[:, valid_cols]
        X_test_s = X_test_s[:, valid_cols]
        
        max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
        
        if max_k < min(k_values):
            return {
                "error": f"max_k={max_k}_too_small",
                "std_train": std_train,
                "std_test": std_test,
            }
        
        # PCA (fit on train)
        pca = PCA(n_components=max_k)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        
        results = {
            "std_train": std_train,
            "std_test": std_test,
            "baseline_mse": baseline_mse,
            "baseline_mae": baseline_mae,
            "by_k": {},
        }
        
        for k in k_values:
            if k > max_k:
                results["by_k"][k] = None
                continue
            
            ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_train_pca[:, :k], y_train)
            y_pred = ridge.predict(X_test_pca[:, :k])
            
            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            
            # Only compute R² if variance is sufficient
            if std_test > min_std:
                r2 = float(r2_score(y_test, y_pred))
            else:
                r2 = None  # Don't report meaningless R²
            
            results["by_k"][k] = {
                "r2": r2,
                "mse": mse,
                "mae": mae,
            }
        
        return results
    
    # ===== RESULTS STORAGE =====
    results = {
        "config": {
            "L": L,
            "T_values": T_values,
            "TC": TC,
            "n_seeds": n_seeds,
            "n_snapshots": n_snapshots,
            "coarse_b": coarse_b,
            "k_values": k_values,
            "train_seeds": list(train_seeds),
            "test_seeds": list(test_seeds),
            "feat_dims": feat_dims,
            "min_std_for_r2": min_std_for_r2,
        },
        "task1_same_T": {},
        "task2_cross_T": {},
        "task3_predict_T": {},
    }
    
    feature_types = ["fft", "raw", "scatter"]
    
    # ===== TASK 1: SAME-T PREDICTION =====
    print("\n" + "="*60)
    print("TASK 1: Same-T prediction")
    print("="*60)
    
    for T in T_values:
        print(f"\nT={T:.3f}")
        T_key = f"{T:.4f}"
        results["task1_same_T"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        for target_name in ["abs_m", "E"]:
            results["task1_same_T"][T_key][target_name] = {}
            
            y_train = np.concatenate([d["targets"][target_name] for d in train_data])
            y_test = np.concatenate([d["targets"][target_name] for d in test_data])
            
            print(f"  {target_name}: std_train={np.std(y_train):.4f}, std_test={np.std(y_test):.4f}")
            
            for feat_type in feature_types:
                feat_key = f"feat_{feat_type}"
                
                X_train = np.concatenate([d[feat_key] for d in train_data])
                X_test = np.concatenate([d[feat_key] for d in test_data])
                
                eval_results = evaluate_method(X_train, X_test, y_train, y_test, k_values, min_std_for_r2)
                results["task1_same_T"][T_key][target_name][feat_type] = eval_results
    
    # ===== TASK 2: CROSS-T GENERALIZATION =====
    print("\n" + "="*60)
    print("TASK 2: Cross-T generalization")
    print("Train on T ∈ {1.5, 2.0, 3.0}, test on T near Tc")
    print("="*60)
    
    train_T = [1.5, 2.0, 3.0]
    test_T = [t for t in T_values if abs(t - TC) < 0.2]
    
    print(f"Train T: {train_T}")
    print(f"Test T: {test_T}")
    
    # Collect ALL training data across temperatures
    train_data_cross = [d for d in all_data if any(abs(d["T"] - t) < 0.01 for t in train_T)]
    
    # FIT SINGLE SCALER/PCA ON UNION OF TRAIN TEMPERATURES
    print("\nFitting single scaler/PCA on union of train temperatures...")
    
    for feat_type in feature_types:
        feat_key = f"feat_{feat_type}"
        X_train_all = np.concatenate([d[feat_key] for d in train_data_cross])
        
        scaler = StandardScaler()
        X_train_all_s = scaler.fit_transform(X_train_all)
        X_train_all_s = np.nan_to_num(X_train_all_s, nan=0, posinf=0, neginf=0)
        
        # Remove zero-variance
        var = np.var(X_train_all_s, axis=0)
        valid_cols = var > 1e-10
        X_train_all_s = X_train_all_s[:, valid_cols]
        
        max_k = min(X_train_all_s.shape[1], X_train_all_s.shape[0] - 1, max(k_values))
        pca = PCA(n_components=max_k)
        pca.fit(X_train_all_s)
        
        # Store for later use
        for d in all_data:
            X = d[feat_key]
            X_s = scaler.transform(X)
            X_s = np.nan_to_num(X_s, nan=0, posinf=0, neginf=0)
            X_s = X_s[:, valid_cols]
            d[f"{feat_key}_pca_crossT"] = pca.transform(X_s)
        
        print(f"  {feat_type}: {valid_cols.sum()} valid features, max_k={max_k}")
    
    # Now evaluate cross-T
    for target_name in ["abs_m", "E"]:
        # Collect train targets
        y_train_all = np.concatenate([d["targets"][target_name] for d in train_data_cross])
        y_mean = y_train_all.mean()
        y_std = y_train_all.std()
        
        print(f"\n{target_name}: train std={y_std:.4f}")
        
        for test_T_val in test_T:
            T_key = f"{test_T_val:.4f}"
            if T_key not in results["task2_cross_T"]:
                results["task2_cross_T"][T_key] = {}
            results["task2_cross_T"][T_key][target_name] = {}
            
            test_data_cross = [d for d in all_data if abs(d["T"] - test_T_val) < 0.01]
            y_test = np.concatenate([d["targets"][target_name] for d in test_data_cross])
            
            std_test = float(np.std(y_test))
            baseline_mse = float(np.mean((y_test - y_mean)**2))
            
            print(f"  Test T={test_T_val:.3f}: std_test={std_test:.4f}")
            
            for feat_type in feature_types:
                X_train_pca = np.concatenate([d[f"feat_{feat_type}_pca_crossT"] for d in train_data_cross])
                X_test_pca = np.concatenate([d[f"feat_{feat_type}_pca_crossT"] for d in test_data_cross])
                
                max_k = X_train_pca.shape[1]
                
                feat_results = {
                    "std_test": std_test,
                    "baseline_mse": baseline_mse,
                    "by_k": {},
                }
                
                for k in k_values:
                    if k > max_k:
                        feat_results["by_k"][k] = None
                        continue
                    
                    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
                    ridge.fit(X_train_pca[:, :k], y_train_all)
                    y_pred = ridge.predict(X_test_pca[:, :k])
                    
                    mse = float(mean_squared_error(y_test, y_pred))
                    mae = float(mean_absolute_error(y_test, y_pred))
                    
                    if std_test > min_std_for_r2:
                        r2 = float(r2_score(y_test, y_pred))
                    else:
                        r2 = None
                    
                    feat_results["by_k"][k] = {"r2": r2, "mse": mse, "mae": mae}
                
                results["task2_cross_T"][T_key][target_name][feat_type] = feat_results
    
    # ===== TASK 3: PREDICT TEMPERATURE =====
    print("\n" + "="*60)
    print("TASK 3: Temperature prediction (regression)")
    print("="*60)
    
    # Use all data, split by seed
    train_data_T = [d for d in all_data if d["seed"] in train_seeds]
    test_data_T = [d for d in all_data if d["seed"] in test_seeds]
    
    y_train_T = np.concatenate([d["targets"]["T"] for d in train_data_T])
    y_test_T = np.concatenate([d["targets"]["T"] for d in test_data_T])
    
    print(f"Train samples: {len(y_train_T)}, Test samples: {len(y_test_T)}")
    print(f"T range: {y_train_T.min():.2f} - {y_train_T.max():.2f}")
    
    for feat_type in feature_types:
        feat_key = f"feat_{feat_type}"
        
        X_train = np.concatenate([d[feat_key] for d in train_data_T])
        X_test = np.concatenate([d[feat_key] for d in test_data_T])
        
        eval_results = evaluate_method(X_train, X_test, y_train_T, y_test_T, k_values, min_std_for_r2)
        results["task3_predict_T"][feat_type] = eval_results
        
        # Print summary
        if "by_k" in eval_results:
            r2_16 = eval_results["by_k"].get(16, {})
            if r2_16:
                print(f"  {feat_type}: R²@k=16 = {r2_16.get('r2', 'N/A')}")
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK v5: FIXED METRICS")
    print("="*70)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    results = run_rg_experiment_v5.remote(
        L=64,
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=20,
        coarse_b=4,
        k_values=[4, 8, 16, 32, 64],
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v5_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    import numpy as np
    
    T_values = results["config"]["T_values"]
    k_values = results["config"]["k_values"]
    
    print("\n" + "="*70)
    print("TASK 1: SAME-T PREDICTION (R² at k=16, or MSE if low variance)")
    print("="*70)
    
    print(f"\n{'T':<8} {'Target':<6} {'std':<8} | {'FFT':<10} {'Raw':<10} {'Scatter':<10}")
    print("-"*65)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["task1_same_T"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m", "E"]:
            target_data = T_data.get(target, {})
            
            # Get std from first feature type
            std_test = target_data.get("fft", {}).get("std_test", 0)
            
            print(f"{T:<7.3f}{marker} {target:<6} {std_test:<8.4f} | ", end="")
            
            for feat in ["fft", "raw", "scatter"]:
                feat_data = target_data.get(feat, {})
                by_k = feat_data.get("by_k", {})
                k16 = by_k.get(16, {}) if by_k else {}
                
                if k16:
                    r2 = k16.get("r2")
                    mse = k16.get("mse")
                    if r2 is not None:
                        print(f"{r2:<10.3f} ", end="")
                    else:
                        print(f"mse={mse:.2e} ", end="")
                else:
                    print(f"{'N/A':<10} ", end="")
            
            print()
    
    print("\n" + "="*70)
    print("TASK 2: CROSS-T GENERALIZATION (R² at k=16)")
    print("="*70)
    
    print(f"\n{'Test T':<8} {'Target':<6} | {'FFT':<10} {'Raw':<10} {'Scatter':<10}")
    print("-"*55)
    
    for T_key, T_data in results["task2_cross_T"].items():
        T = float(T_key)
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m", "E"]:
            target_data = T_data.get(target, {})
            
            print(f"{T:<7.3f}{marker} {target:<6} | ", end="")
            
            for feat in ["fft", "raw", "scatter"]:
                feat_data = target_data.get(feat, {})
                by_k = feat_data.get("by_k", {})
                k16 = by_k.get(16, {}) if by_k else {}
                
                if k16:
                    r2 = k16.get("r2")
                    if r2 is not None:
                        print(f"{r2:<10.3f} ", end="")
                    else:
                        print(f"{'low_var':<10} ", end="")
                else:
                    print(f"{'N/A':<10} ", end="")
            
            print()
    
    print("\n" + "="*70)
    print("TASK 3: TEMPERATURE PREDICTION (R² at each k)")
    print("="*70)
    
    print(f"\n{'Method':<10} | ", end="")
    for k in k_values:
        print(f"k={k:<4} ", end="")
    print()
    print("-"*50)
    
    for feat in ["fft", "raw", "scatter"]:
        feat_data = results["task3_predict_T"].get(feat, {})
        by_k = feat_data.get("by_k", {})
        
        print(f"{feat:<10} | ", end="")
        for k in k_values:
            k_data = by_k.get(k, {}) if by_k else {}
            r2 = k_data.get("r2") if k_data else None
            if r2 is not None:
                print(f"{r2:<6.3f} ", end="")
            else:
                print(f"{'N/A':<6} ", end="")
        print()
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
