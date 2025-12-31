"""
2D Spatial RG Task v6: Better RG tests

Key changes from ChatGPT:
1. Cross-T: Train on T near Tc, test on T far from Tc (reverse direction)
2. Cross-L generalization: Train on L=32, test on L=64 (true RG test)
3. Better task: predict T with cross-L generalization
4. Simpler feature comparison at matched dimensions
"""

import modal

app = modal.App("ising-2d-rg-v6")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=3600, cpu=4, memory=8192)
def run_rg_experiment_v6(
    L_values: list = None,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 20,
    n_eq: int = 1000,
    n_gap: int = 50,
    coarse_b: int = 4,
    k_values: list = None,
    test_seed_frac: float = 0.3,
) -> dict:
    """
    Run v6 with cross-L generalization test.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score, mean_squared_error
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if L_values is None:
        L_values = [32, 64]
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    if k_values is None:
        k_values = [4, 8, 16, 32]
    
    print(f"Config: L={L_values}, T={len(T_values)} points, seeds={n_seeds}, snaps={n_snapshots}")
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
    
    # ===== FEATURE EXTRACTION =====
    # These must produce SAME dimension regardless of L
    
    def extract_fft_normalized(snapshots, n_bins=16):
        """
        FFT with normalized radial bins (L-independent dimensions).
        Uses fractional radius to get same n_bins regardless of L.
        """
        N, L, _ = snapshots.shape
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Normalize radius to [0, 1]
            r_norm = r / (L // 2)
            
            # Skip DC (r < 0.1), use rest
            bins = np.linspace(0.1, 1.0, n_bins + 1)
            radial_profile = np.zeros(n_bins)
            for j in range(n_bins):
                mask = (r_norm >= bins[j]) & (r_norm < bins[j+1])
                if mask.sum() > 0:
                    radial_profile[j] = mag[mask].mean()
            
            # Normalize by L^2 to make scale-independent
            radial_profile = radial_profile / (L * L)
            
            features.append(radial_profile)
        
        return np.array(features)
    
    def extract_scattering_pooled(snapshots, J=2):
        """
        Scattering with fixed J, pooled to L-independent dimensions.
        """
        from kymatio.numpy import Scattering2D
        
        N, L, _ = snapshots.shape
        
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        # Pool spatially -> (N, C)
        if len(Sx.shape) == 4:
            features = Sx.mean(axis=(-1, -2))
        elif len(Sx.shape) == 5:
            # (N, 1, C, H, W) format
            features = Sx.squeeze(1).mean(axis=(-1, -2))
        else:
            features = Sx.reshape(N, -1)
        
        if len(features.shape) != 2:
            features = features.reshape(N, -1)
        
        return features
    
    def extract_raw_pooled(snapshots, pool_to=8):
        """
        Raw features pooled to fixed size (L-independent).
        """
        N, L, _ = snapshots.shape
        
        pool_size = L // pool_to
        pooled = snapshots.reshape(N, pool_to, pool_size, pool_to, pool_size).mean(axis=(2, 4))
        
        return pooled.reshape(N, -1)
    
    # ===== GENERATE ALL DATA =====
    print("\nGenerating snapshots...")
    
    all_data = []
    
    for L in L_values:
        print(f"  L={L}...")
        for T in T_values:
            for seed in range(n_seeds):
                snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
                
                all_data.append({
                    "L": L,
                    "seed": seed,
                    "T": T,
                    "snapshots": snaps,
                })
    
    print(f"Generated {len(all_data)} (L, T, seed) combinations")
    
    # ===== EXTRACT FEATURES =====
    print("\nExtracting features...")
    
    for item in all_data:
        snaps = item["snapshots"]
        item["feat_fft"] = extract_fft_normalized(snaps)
        item["feat_scatter"] = extract_scattering_pooled(snaps)
        item["feat_raw"] = extract_raw_pooled(snaps)
    
    # Verify dimensions match across L
    for L in L_values:
        sample = [d for d in all_data if d["L"] == L][0]
        print(f"  L={L}: fft={sample['feat_fft'].shape[1]}, scatter={sample['feat_scatter'].shape[1]}, raw={sample['feat_raw'].shape[1]}")
    
    # ===== TRAIN/TEST SPLIT BY SEED =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    
    print(f"Train seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== EVALUATION HELPER =====
    def evaluate(X_train, X_test, y_train, y_test, k_values):
        """Standard evaluation pipeline."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
        
        # Remove zero-variance
        var = np.var(X_train_s, axis=0)
        valid = var > 1e-10
        if valid.sum() == 0:
            return {"error": "zero_variance"}
        X_train_s = X_train_s[:, valid]
        X_test_s = X_test_s[:, valid]
        
        max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
        if max_k < min(k_values):
            return {"error": "insufficient_dims"}
        
        pca = PCA(n_components=max_k)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        
        results = {"by_k": {}}
        for k in k_values:
            if k > max_k:
                results["by_k"][k] = None
                continue
            
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_train_pca[:, :k], y_train)
            y_pred = ridge.predict(X_test_pca[:, :k])
            
            r2 = float(r2_score(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            
            results["by_k"][k] = {"r2": r2, "mse": mse}
        
        return results
    
    # ===== RESULTS =====
    results = {
        "config": {
            "L_values": L_values,
            "T_values": T_values,
            "TC": TC,
            "n_seeds": n_seeds,
            "n_snapshots": n_snapshots,
            "k_values": k_values,
        },
        "task1_predict_T_same_L": {},
        "task2_predict_T_cross_L": {},
        "task3_cross_T_same_L": {},
    }
    
    feature_types = ["fft", "raw", "scatter"]
    
    # ===== TASK 1: PREDICT T (SAME L) =====
    print("\n" + "="*60)
    print("TASK 1: Temperature prediction (same L, split by seed)")
    print("="*60)
    
    for L in L_values:
        print(f"\nL={L}")
        L_data = [d for d in all_data if d["L"] == L]
        train_data = [d for d in L_data if d["seed"] in train_seeds]
        test_data = [d for d in L_data if d["seed"] in test_seeds]
        
        y_train = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in train_data])
        y_test = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in test_data])
        
        results["task1_predict_T_same_L"][L] = {}
        
        for feat_type in feature_types:
            feat_key = f"feat_{feat_type}"
            X_train = np.concatenate([d[feat_key] for d in train_data])
            X_test = np.concatenate([d[feat_key] for d in test_data])
            
            eval_result = evaluate(X_train, X_test, y_train, y_test, k_values)
            results["task1_predict_T_same_L"][L][feat_type] = eval_result
            
            r2_16 = eval_result.get("by_k", {}).get(16, {})
            if r2_16:
                print(f"  {feat_type}: R²@k=16 = {r2_16.get('r2', 'N/A'):.3f}")
    
    # ===== TASK 2: PREDICT T (CROSS-L) =====
    print("\n" + "="*60)
    print("TASK 2: Temperature prediction (train L=32, test L=64)")
    print("This is the key RG test: does representation generalize across scales?")
    print("="*60)
    
    if 32 in L_values and 64 in L_values:
        train_data = [d for d in all_data if d["L"] == 32]
        test_data = [d for d in all_data if d["L"] == 64]
        
        y_train = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in train_data])
        y_test = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in test_data])
        
        print(f"Train: L=32, {len(y_train)} samples")
        print(f"Test: L=64, {len(y_test)} samples")
        
        for feat_type in feature_types:
            feat_key = f"feat_{feat_type}"
            X_train = np.concatenate([d[feat_key] for d in train_data])
            X_test = np.concatenate([d[feat_key] for d in test_data])
            
            eval_result = evaluate(X_train, X_test, y_train, y_test, k_values)
            results["task2_predict_T_cross_L"][feat_type] = eval_result
            
            print(f"\n{feat_type}:")
            for k in k_values:
                k_result = eval_result.get("by_k", {}).get(k, {})
                if k_result:
                    print(f"  k={k}: R²={k_result.get('r2', 'N/A'):.3f}")
    else:
        print("Need both L=32 and L=64 for cross-L test")
    
    # ===== TASK 3: CROSS-T (BETTER DESIGN) =====
    print("\n" + "="*60)
    print("TASK 3: Cross-T prediction (train near Tc, test far from Tc)")
    print("Reversed direction: can near-critical features generalize?")
    print("="*60)
    
    # Train on T near Tc, test on far T
    train_T = [t for t in T_values if abs(t - TC) < 0.15]  # Near Tc
    test_T = [t for t in T_values if abs(t - TC) > 0.5]    # Far from Tc
    
    print(f"Train T (near Tc): {[f'{t:.2f}' for t in train_T]}")
    print(f"Test T (far from Tc): {[f'{t:.2f}' for t in test_T]}")
    
    for L in L_values:
        print(f"\nL={L}")
        results["task3_cross_T_same_L"][L] = {}
        
        L_data = [d for d in all_data if d["L"] == L]
        train_data = [d for d in L_data if any(abs(d["T"] - t) < 0.01 for t in train_T)]
        
        # Fit scaler/PCA on train
        for feat_type in feature_types:
            feat_key = f"feat_{feat_type}"
            X_train_all = np.concatenate([d[feat_key] for d in train_data])
            y_train_all = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in train_data])
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_all)
            X_train_s = np.nan_to_num(X_train_s)
            
            var = np.var(X_train_s, axis=0)
            valid = var > 1e-10
            X_train_s = X_train_s[:, valid]
            
            max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
            pca = PCA(n_components=max_k)
            X_train_pca = pca.fit_transform(X_train_s)
            
            results["task3_cross_T_same_L"][L][feat_type] = {"test_T": {}}
            
            for test_T_val in test_T:
                test_data = [d for d in L_data if abs(d["T"] - test_T_val) < 0.01]
                if not test_data:
                    continue
                
                X_test = np.concatenate([d[feat_key] for d in test_data])
                y_test = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in test_data])
                
                X_test_s = scaler.transform(X_test)
                X_test_s = np.nan_to_num(X_test_s)
                X_test_s = X_test_s[:, valid]
                X_test_pca = pca.transform(X_test_s)
                
                T_key = f"{test_T_val:.2f}"
                results["task3_cross_T_same_L"][L][feat_type]["test_T"][T_key] = {}
                
                for k in k_values:
                    if k > max_k:
                        continue
                    
                    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                    ridge.fit(X_train_pca[:, :k], y_train_all)
                    y_pred = ridge.predict(X_test_pca[:, :k])
                    
                    # For T prediction, compute MAE (more interpretable)
                    mae = float(np.mean(np.abs(y_test - y_pred)))
                    r2 = float(r2_score(y_test, y_pred)) if np.std(y_test) > 0.01 else None
                    
                    results["task3_cross_T_same_L"][L][feat_type]["test_T"][T_key][k] = {
                        "mae": mae,
                        "r2": r2,
                        "mean_pred": float(np.mean(y_pred)),
                        "true_T": float(test_T_val),
                    }
            
            # Print summary
            print(f"  {feat_type}:")
            for test_T_val in test_T:
                T_key = f"{test_T_val:.2f}"
                k16 = results["task3_cross_T_same_L"][L][feat_type]["test_T"].get(T_key, {}).get(16, {})
                if k16:
                    print(f"    T={test_T_val:.2f}: MAE={k16['mae']:.3f}, pred={k16['mean_pred']:.2f}")
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK v6: CROSS-L GENERALIZATION")
    print("="*70)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    results = run_rg_experiment_v6.remote(
        L_values=[32, 64],
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=20,
        k_values=[4, 8, 16, 32],
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v6_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "="*70)
    print("TASK 1: TEMPERATURE PREDICTION (SAME L)")
    print("="*70)
    
    k_values = results["config"]["k_values"]
    
    for L, L_data in results["task1_predict_T_same_L"].items():
        print(f"\nL={L}:")
        print(f"  {'Method':<10} | ", end="")
        for k in k_values:
            print(f"k={k:<3} ", end="")
        print()
        
        for feat in ["fft", "raw", "scatter"]:
            feat_data = L_data.get(feat, {})
            by_k = feat_data.get("by_k", {})
            
            print(f"  {feat:<10} | ", end="")
            for k in k_values:
                k_data = by_k.get(k, {})
                r2 = k_data.get("r2") if k_data else None
                if r2 is not None:
                    print(f"{r2:.3f} ", end="")
                else:
                    print(f"{'N/A':<5} ", end="")
            print()
    
    print("\n" + "="*70)
    print("TASK 2: CROSS-L GENERALIZATION (Train L=32 → Test L=64)")
    print("THIS IS THE KEY RG TEST")
    print("="*70)
    
    print(f"\n{'Method':<10} | ", end="")
    for k in k_values:
        print(f"k={k:<3} ", end="")
    print()
    print("-"*40)
    
    for feat in ["fft", "raw", "scatter"]:
        feat_data = results["task2_predict_T_cross_L"].get(feat, {})
        by_k = feat_data.get("by_k", {})
        
        print(f"{feat:<10} | ", end="")
        for k in k_values:
            k_data = by_k.get(k, {})
            r2 = k_data.get("r2") if k_data else None
            if r2 is not None:
                print(f"{r2:.3f} ", end="")
            else:
                print(f"{'N/A':<5} ", end="")
        print()
    
    print("\n" + "="*70)
    print("TASK 3: CROSS-T (Train near Tc → Test far from Tc)")
    print("="*70)
    
    for L, L_data in results["task3_cross_T_same_L"].items():
        print(f"\nL={L}:")
        for feat in ["fft", "raw", "scatter"]:
            feat_data = L_data.get(feat, {})
            test_T_data = feat_data.get("test_T", {})
            
            print(f"  {feat}:")
            for T_key, T_results in test_T_data.items():
                k16 = T_results.get(16, {})
                if k16:
                    print(f"    T={T_key}: MAE={k16['mae']:.3f}, pred={k16['mean_pred']:.2f} (true={k16['true_T']:.2f})")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
