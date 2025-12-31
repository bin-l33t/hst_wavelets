"""
2D Spatial RG Task v3: Proper benchmarks

Fixes from ChatGPT:
1. Scalar targets: No PCA bottleneck, or preserve DC explicitly
2. Field reconstruction: Keep spatial scattering, local readout
3. Fair comparison across feature types
"""

import modal

app = modal.App("ising-2d-rg-v3")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=2400, cpu=4, memory=8192)
def run_rg_experiment_v3(
    L: int = 64,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 10,
    n_eq: int = 500,
    n_gap: int = 20,
    test_seed_frac: float = 0.3,
) -> dict:
    """
    Run the v3 2D RG experiment with proper benchmarks.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    
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
        L = s.shape[-1]
        L_coarse = L // b
        if len(s.shape) == 3:
            return s.reshape(s.shape[0], L_coarse, b, L_coarse, b).mean(axis=(2, 4))
        else:
            return s.reshape(L_coarse, b, L_coarse, b).mean(axis=(1, 3))
    
    def compute_scalar_targets(s, b=2):
        s_coarse = coarse_grain(s, b)
        s_sign = np.sign(s_coarse)
        
        abs_m2 = np.abs(s_coarse.mean(axis=(1, 2)))
        
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +
            s_sign * np.roll(s_sign, 1, axis=2)
        ).sum(axis=(1, 2))
        E2 = -nn_sum / (2 * L_c * L_c)
        
        return {"abs_m2": abs_m2, "E2": E2}
    
    # ===== FEATURE EXTRACTION =====
    
    # 1. Scattering - POOLED (for scalar targets)
    def extract_scattering_pooled(snapshots):
        from kymatio.numpy import Scattering2D
        
        L = snapshots.shape[1]
        J = 3
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        # Pool over spatial dims
        if len(Sx.shape) == 4:
            features = Sx.mean(axis=(-1, -2))
        else:
            features = Sx.reshape(Sx.shape[0], -1)
        
        if len(features.shape) != 2:
            features = features.reshape(features.shape[0], -1)
        
        return features
    
    # 2. Scattering - SPATIAL (for field reconstruction)
    def extract_scattering_spatial(snapshots):
        from kymatio.numpy import Scattering2D
        
        L = snapshots.shape[1]
        J = 3
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)  # (N, C, L', L')
        
        # Keep spatial structure
        return Sx  # Will use for local prediction
    
    # 3. Raw features - with DC preserved
    def extract_raw_with_dc(snapshots):
        """Raw features that explicitly preserve DC (mean) and local structure."""
        N = snapshots.shape[0]
        
        # DC component (global mean)
        dc = snapshots.mean(axis=(1, 2), keepdims=False)  # (N,)
        
        # Pool to 8x8 (local structure)
        L = snapshots.shape[1]
        pool_size = L // 8
        pooled = snapshots.reshape(N, 8, pool_size, 8, pool_size).mean(axis=(2, 4))
        pooled_flat = pooled.reshape(N, -1)  # (N, 64)
        
        # Combine: [DC, pooled_flat]
        features = np.column_stack([dc, pooled_flat])  # (N, 65)
        
        return features
    
    # 4. FFT features - radial bins with DC explicit
    def extract_fft_with_dc(snapshots, n_bins=16):
        N, L, _ = snapshots.shape
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            # DC component explicitly
            dc = mag[L//2, L//2]
            
            # Radial bins (excluding DC)
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            r_max = L // 2
            bins = np.linspace(1, r_max, n_bins)  # Start from 1, not 0
            radial_profile = np.zeros(n_bins)
            for j in range(n_bins):
                r_lo = 0 if j == 0 else bins[j-1]
                r_hi = bins[j]
                mask = (r >= r_lo) & (r < r_hi) & (r > 0.5)  # Exclude DC
                if mask.sum() > 0:
                    radial_profile[j] = mag[mask].mean()
            
            features.append(np.concatenate([[dc], radial_profile]))
        
        return np.array(features)  # (N, 1 + n_bins)
    
    # ===== GENERATE ALL DATA =====
    print("Generating snapshots...")
    
    all_data = []
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
            scalar_targets = compute_scalar_targets(snaps, b=2)
            coarse_field = coarse_grain(snaps, b=2)  # (N, L/2, L/2)
            
            all_data.append({
                "seed": seed,
                "T": T,
                "snapshots": snaps,
                "scalar_targets": scalar_targets,
                "coarse_field": coarse_field,
            })
    
    print(f"Generated {len(all_data)} (T, seed) combinations")
    
    # ===== EXTRACT FEATURES =====
    print("Extracting features...")
    
    for item in all_data:
        snaps = item["snapshots"]
        item["feat_scatter_pooled"] = extract_scattering_pooled(snaps)
        item["feat_scatter_spatial"] = extract_scattering_spatial(snaps)
        item["feat_raw_dc"] = extract_raw_with_dc(snaps)
        item["feat_fft_dc"] = extract_fft_with_dc(snaps)
    
    print(f"Feature dims: scatter_pooled={all_data[0]['feat_scatter_pooled'].shape}, "
          f"raw_dc={all_data[0]['feat_raw_dc'].shape[1]}, "
          f"fft_dc={all_data[0]['feat_fft_dc'].shape[1]}")
    print(f"Scatter spatial shape: {all_data[0]['feat_scatter_spatial'].shape}")
    
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
            "train_seeds": list(train_seeds),
            "test_seeds": list(test_seeds),
        },
        "scalar_targets": {},
        "coarse_reconstruction": {},
    }
    
    # ===== TASK 1: SCALAR TARGET PREDICTION (NO PCA BOTTLENECK) =====
    print("\n=== TASK 1: Scalar target prediction (direct ridge, no PCA) ===")
    
    for T in T_values:
        print(f"Processing T={T:.3f}...")
        T_key = f"{T:.4f}"
        results["scalar_targets"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        for target_name in ["abs_m2", "E2"]:
            results["scalar_targets"][T_key][target_name] = {}
            
            y_train = np.concatenate([d["scalar_targets"][target_name] for d in train_data])
            y_test = np.concatenate([d["scalar_targets"][target_name] for d in test_data])
            
            for feat_name, feat_key in [
                ("scatter", "feat_scatter_pooled"),
                ("raw", "feat_raw_dc"),
                ("fft", "feat_fft_dc"),
            ]:
                X_train = np.concatenate([d[feat_key] for d in train_data])
                X_test = np.concatenate([d[feat_key] for d in test_data])
                
                # Ensure 2D
                if len(X_train.shape) != 2:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], -1)
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
                X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
                
                # Direct ridge - NO PCA
                ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_train_s, y_train)
                y_pred = ridge.predict(X_test_s)
                
                r2 = r2_score(y_test, y_pred)
                results["scalar_targets"][T_key][target_name][feat_name] = {
                    "r2": float(r2),
                    "n_features": X_train.shape[1],
                }
    
    # ===== TASK 2: COARSE FIELD RECONSTRUCTION (LOCAL READOUT) =====
    print("\n=== TASK 2: Coarse field reconstruction (local readout) ===")
    
    for T in T_values:
        print(f"Processing T={T:.3f}...")
        T_key = f"{T:.4f}"
        results["coarse_reconstruction"][T_key] = {}
        
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        # Target: coarse field
        Y_train = np.concatenate([d["coarse_field"] for d in train_data])  # (N, L/2, L/2)
        Y_test = np.concatenate([d["coarse_field"] for d in test_data])
        
        L_coarse = Y_train.shape[1]
        
        # Flatten targets for per-pixel ridge
        Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)
        Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)
        
        # Method 1: Raw pooled (8x8 -> upscale to 32x32)
        # This is a spatial baseline
        X_raw_train = np.concatenate([d["feat_raw_dc"][:, 1:].reshape(-1, 8, 8) for d in train_data])
        X_raw_test = np.concatenate([d["feat_raw_dc"][:, 1:].reshape(-1, 8, 8) for d in test_data])
        
        # Upscale 8x8 -> 32x32 via nearest neighbor
        from scipy.ndimage import zoom
        X_raw_train_up = np.array([zoom(x, L_coarse/8, order=0) for x in X_raw_train])
        X_raw_test_up = np.array([zoom(x, L_coarse/8, order=0) for x in X_raw_test])
        
        # Per-pixel R²
        r2_raw_pixels = []
        for i in range(L_coarse):
            for j in range(L_coarse):
                y_true = Y_test[:, i, j]
                y_pred = X_raw_test_up[:, i, j]
                if y_true.std() > 1e-10:
                    r2_raw_pixels.append(r2_score(y_true, y_pred))
        
        results["coarse_reconstruction"][T_key]["raw_upsample"] = {
            "r2_avg": float(np.mean(r2_raw_pixels)) if r2_raw_pixels else 0.0,
            "mse": float(mean_squared_error(Y_test_flat, X_raw_test_up.reshape(-1, L_coarse*L_coarse))),
        }
        
        # Method 2: Scattering spatial -> local linear
        # Scattering output shape varies by Kymatio version
        Sx_train = np.concatenate([d["feat_scatter_spatial"] for d in train_data])
        Sx_test = np.concatenate([d["feat_scatter_spatial"] for d in test_data])
        
        print(f"  Scattering spatial shape: {Sx_train.shape}")
        
        # Handle different possible shapes
        if len(Sx_train.shape) == 4:
            # (N, C, H, W) - standard
            Sx_h, Sx_w = Sx_train.shape[-2], Sx_train.shape[-1]
            n_channels = Sx_train.shape[1]
            Sx_train_mean = Sx_train.mean(axis=1)  # (N, H, W)
            Sx_test_mean = Sx_test.mean(axis=1)
        elif len(Sx_train.shape) == 3:
            # (N, C, 1) or similar - no spatial dims
            print("  Scattering has no spatial dims, skipping upsample method")
            Sx_train_mean = None
            Sx_test_mean = None
        else:
            print(f"  Unexpected scattering shape: {Sx_train.shape}")
            Sx_train_mean = None
            Sx_test_mean = None
        
        if Sx_train_mean is not None and len(Sx_train_mean.shape) == 3:
            Sx_h, Sx_w = Sx_train_mean.shape[-2], Sx_train_mean.shape[-1]
            
            # Only upscale if we have 2D spatial structure
            if Sx_h > 1 and Sx_w > 1:
                Sx_train_up = np.array([zoom(x, (L_coarse/Sx_h, L_coarse/Sx_w), order=1) for x in Sx_train_mean])
                Sx_test_up = np.array([zoom(x, (L_coarse/Sx_h, L_coarse/Sx_w), order=1) for x in Sx_test_mean])
                
                r2_scatter_pixels = []
                for i in range(L_coarse):
                    for j in range(L_coarse):
                        y_true = Y_test[:, i, j]
                        y_pred = Sx_test_up[:, i, j]
                        if y_true.std() > 1e-10:
                            r2_scatter_pixels.append(r2_score(y_true, y_pred))
                
                results["coarse_reconstruction"][T_key]["scatter_upsample"] = {
                    "r2_avg": float(np.mean(r2_scatter_pixels)) if r2_scatter_pixels else 0.0,
                    "mse": float(mean_squared_error(Y_test_flat, Sx_test_up.reshape(-1, L_coarse*L_coarse))),
                }
            else:
                results["coarse_reconstruction"][T_key]["scatter_upsample"] = {
                    "r2_avg": None,
                    "mse": None,
                    "note": "Scattering spatial dims too small",
                }
        else:
            results["coarse_reconstruction"][T_key]["scatter_upsample"] = {
                "r2_avg": None,
                "mse": None,
                "note": "No spatial structure in scattering",
            }
        
        # Method 3: Per-pixel ridge from all scattering channels
        # At each output location, use all scattering channels to predict
        # This is expensive, so we do it for a subset of pixels
        
        # Flatten scattering spatially and predict each output pixel
        Sx_train_flat = Sx_train.reshape(Sx_train.shape[0], -1)  # (N, C*H*W)
        Sx_test_flat = Sx_test.reshape(Sx_test.shape[0], -1)
        
        scaler = StandardScaler()
        Sx_train_s = scaler.fit_transform(Sx_train_flat)
        Sx_test_s = scaler.transform(Sx_test_flat)
        
        # Multi-output ridge
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        ridge.fit(Sx_train_s, Y_train_flat)
        Y_pred_scatter = ridge.predict(Sx_test_s)
        
        r2_scatter_ridge = r2_score(Y_test_flat, Y_pred_scatter)
        mse_scatter_ridge = mean_squared_error(Y_test_flat, Y_pred_scatter)
        
        results["coarse_reconstruction"][T_key]["scatter_ridge"] = {
            "r2_avg": float(r2_scatter_ridge),
            "mse": float(mse_scatter_ridge),
        }
        
        # Method 4: Raw flatten -> ridge (baseline)
        snaps_train = np.concatenate([d["snapshots"] for d in train_data])
        snaps_test = np.concatenate([d["snapshots"] for d in test_data])
        
        X_train_raw_full = snaps_train.reshape(snaps_train.shape[0], -1)
        X_test_raw_full = snaps_test.reshape(snaps_test.shape[0], -1)
        
        scaler = StandardScaler()
        X_train_raw_s = scaler.fit_transform(X_train_raw_full)
        X_test_raw_s = scaler.transform(X_test_raw_full)
        
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        ridge.fit(X_train_raw_s, Y_train_flat)
        Y_pred_raw = ridge.predict(X_test_raw_s)
        
        r2_raw_ridge = r2_score(Y_test_flat, Y_pred_raw)
        mse_raw_ridge = mean_squared_error(Y_test_flat, Y_pred_raw)
        
        results["coarse_reconstruction"][T_key]["raw_ridge"] = {
            "r2_avg": float(r2_raw_ridge),
            "mse": float(mse_raw_ridge),
        }
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK v3 (PROPER BENCHMARKS)")
    print("="*70)
    print("Fixes: No PCA bottleneck for scalars, spatial scattering for fields")
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    results = run_rg_experiment_v3.remote(
        L=64,
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=10,
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v3_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    import numpy as np
    
    T_values = results["config"]["T_values"]
    
    print("\n" + "="*70)
    print("TASK 1: SCALAR TARGET PREDICTION (Direct Ridge, No PCA)")
    print("="*70)
    
    print(f"\n{'T':<10} {'Target':<8} | {'Scatter':<12} {'Raw':<12} {'FFT':<12} | Winner")
    print("-"*70)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["scalar_targets"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["abs_m2", "E2"]:
            target_data = T_data.get(target, {})
            
            scores = {}
            for feat in ["scatter", "raw", "fft"]:
                feat_data = target_data.get(feat, {})
                r2 = feat_data.get("r2", None)
                scores[feat] = r2 if r2 is not None else -999
            
            winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
            
            print(f"{T:<9.3f}{marker} {target:<8} | ", end="")
            for feat in ["scatter", "raw", "fft"]:
                r2 = scores[feat]
                if r2 > -999:
                    print(f"{r2:<12.3f} ", end="")
                else:
                    print(f"{'N/A':<12} ", end="")
            print(f"| {winner}")
    
    print("\n" + "="*70)
    print("TASK 2: COARSE FIELD RECONSTRUCTION (R² avg)")
    print("="*70)
    
    print(f"\n{'T':<10} | {'Raw Ridge':<12} {'Scatter Ridge':<14} {'Raw Up':<10} {'Scatter Up':<12} | Winner")
    print("-"*75)
    
    for T in T_values:
        T_key = f"{T:.4f}"
        T_data = results["coarse_reconstruction"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        methods = ["raw_ridge", "scatter_ridge", "raw_upsample", "scatter_upsample"]
        scores = {}
        for method in methods:
            method_data = T_data.get(method, {})
            r2 = method_data.get("r2_avg", None)
            scores[method] = r2 if r2 is not None else -999
        
        winner = max(scores, key=lambda x: scores[x]) if any(v > -999 for v in scores.values()) else "N/A"
        
        print(f"{T:<9.3f}{marker} | ", end="")
        for method in methods:
            r2 = scores[method]
            if r2 > -999:
                print(f"{r2:<12.3f} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print(f"| {winner}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
