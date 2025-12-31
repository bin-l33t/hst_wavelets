"""
2D Spatial RG Task v7: Fixed FFT + True RG Task

Fixes from ChatGPT:
1. FFT: Use power spectrum mean(|F|²), not magnitude mean(|F|)
2. Add true RG task: fine-scale features → coarse observables
3. Both cross-L temperature inference AND proper RG coarse-graining
"""

import modal

app = modal.App("ising-2d-rg-v7")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=3600, cpu=4, memory=8192)
def run_rg_experiment_v7(
    L_values: list = None,
    T_values: list = None,
    n_seeds: int = 10,
    n_snapshots: int = 20,
    n_eq: int = 1000,
    n_gap: int = 50,
    coarse_b: int = 4,  # Block size for RG coarse-graining
    k_values: list = None,
    test_seed_frac: float = 0.3,
) -> dict:
    """
    v7: Fixed FFT + True RG coarse-graining task.
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
    print(f"Coarse block b={coarse_b} for RG task")
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
        """Block average."""
        L = s.shape[-1]
        L_c = L // b
        if len(s.shape) == 3:
            return s.reshape(s.shape[0], L_c, b, L_c, b).mean(axis=(2, 4))
        return s.reshape(L_c, b, L_c, b).mean(axis=(1, 3))
    
    def compute_coarse_observables(s, b):
        """Compute observables on coarse-grained field."""
        s_c = coarse_grain(s, b)
        s_sign = np.sign(s_c)
        
        # |m^(b)|
        abs_m = np.abs(s_c.mean(axis=(1, 2)))
        
        # E^(b)
        L_c = s_sign.shape[1]
        nn = (s_sign * np.roll(s_sign, 1, axis=1) + 
              s_sign * np.roll(s_sign, 1, axis=2)).sum(axis=(1, 2))
        E = -nn / (2 * L_c * L_c)
        
        return {"abs_m": abs_m, "E": E}
    
    # ===== FEATURE EXTRACTION =====
    
    def extract_fft_power(snapshots, n_bins=16):
        """
        FFT POWER spectrum (FIXED): mean(|F|²), not mean(|F|).
        Log-power for better conditioning.
        """
        N, L, _ = snapshots.shape
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            
            # POWER, not magnitude
            power = np.abs(fft2_shift) ** 2
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Normalized radius
            r_norm = r / (L // 2)
            
            # Skip DC (r < 0.05), use rest
            bins = np.linspace(0.05, 1.0, n_bins + 1)
            radial_power = np.zeros(n_bins)
            for j in range(n_bins):
                mask = (r_norm >= bins[j]) & (r_norm < bins[j+1])
                if mask.sum() > 0:
                    # Mean power in bin, normalized by L^4
                    radial_power[j] = power[mask].mean() / (L ** 4)
            
            # Log-power (with floor for stability)
            log_power = np.log(radial_power + 1e-10)
            
            features.append(log_power)
        
        return np.array(features)
    
    def extract_fft_power_fine_only(snapshots, b, n_bins=12):
        """
        FFT power for scales < b only (for RG task).
        Excludes low frequencies that encode coarse structure.
        """
        N, L, _ = snapshots.shape
        features = []
        
        # Cutoff: keep |k| > L/(2b) (wavelength < 2b)
        k_cutoff_norm = 1.0 / (2 * b) * 2  # normalized radius cutoff
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            power = np.abs(fft2_shift) ** 2
            
            cy, cx = L // 2, L // 2
            y, x = np.ogrid[:L, :L]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            r_norm = r / (L // 2)
            
            # Only high frequencies (fine scales)
            bins = np.linspace(k_cutoff_norm, 1.0, n_bins + 1)
            radial_power = np.zeros(n_bins)
            for j in range(n_bins):
                mask = (r_norm >= bins[j]) & (r_norm < bins[j+1])
                if mask.sum() > 0:
                    radial_power[j] = power[mask].mean() / (L ** 4)
            
            log_power = np.log(radial_power + 1e-10)
            features.append(log_power)
        
        return np.array(features)
    
    def extract_scattering_pooled(snapshots, J=2):
        """Scattering with fixed J, pooled."""
        from kymatio.numpy import Scattering2D
        
        N, L, _ = snapshots.shape
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        if len(Sx.shape) == 4:
            features = Sx.mean(axis=(-1, -2))
        elif len(Sx.shape) == 5:
            features = Sx.squeeze(1).mean(axis=(-1, -2))
        else:
            features = Sx.reshape(N, -1)
        
        if len(features.shape) != 2:
            features = features.reshape(N, -1)
        
        return features
    
    def extract_scattering_fine_only(snapshots, b):
        """
        Scattering with J constrained to scales < b.
        For RG task: no coarse-scale info in features.
        """
        from kymatio.numpy import Scattering2D
        
        N, L, _ = snapshots.shape
        
        # J such that 2^J < b
        J = max(1, int(np.floor(np.log2(b))) - 1)
        
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        x = snapshots[:, np.newaxis, :, :]
        Sx = scattering(x)
        
        if len(Sx.shape) == 4:
            # Exclude channel 0 (lowpass/DC)
            features = Sx[:, 1:, :, :].mean(axis=(-1, -2))
        elif len(Sx.shape) == 5:
            features = Sx.squeeze(1)[:, 1:, :, :].mean(axis=(-1, -2))
        else:
            features = Sx.reshape(N, -1)
        
        if len(features.shape) != 2:
            features = features.reshape(N, -1)
        
        return features
    
    def extract_raw_pooled(snapshots, pool_to=8):
        """Raw pooled to fixed size."""
        N, L, _ = snapshots.shape
        pool_size = L // pool_to
        pooled = snapshots.reshape(N, pool_to, pool_size, pool_to, pool_size).mean(axis=(2, 4))
        return pooled.reshape(N, -1)
    
    def extract_raw_fine_only(snapshots, b, pool_to=8):
        """
        Raw with coarse structure (block means) removed.
        For RG task.
        """
        N, L, _ = snapshots.shape
        
        s_c = coarse_grain(snapshots, b)
        s_c_up = np.repeat(np.repeat(s_c, b, axis=1), b, axis=2)
        s_res = snapshots - s_c_up
        
        pool_size = L // pool_to
        pooled = s_res.reshape(N, pool_to, pool_size, pool_to, pool_size).mean(axis=(2, 4))
        
        # Add tiny noise for numerical stability
        pooled = pooled + np.random.randn(*pooled.shape) * 1e-10
        
        return pooled.reshape(N, -1)
    
    # ===== GENERATE DATA =====
    print("\nGenerating snapshots...")
    
    all_data = []
    for L in L_values:
        print(f"  L={L}...")
        for T in T_values:
            for seed in range(n_seeds):
                snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
                coarse_obs = compute_coarse_observables(snaps, coarse_b)
                
                all_data.append({
                    "L": L,
                    "seed": seed,
                    "T": T,
                    "snapshots": snaps,
                    "coarse_obs": coarse_obs,
                })
    
    print(f"Generated {len(all_data)} configurations")
    
    # ===== EXTRACT FEATURES =====
    print("\nExtracting features...")
    
    for item in all_data:
        snaps = item["snapshots"]
        L = item["L"]
        
        # Full features (for T prediction)
        item["feat_fft"] = extract_fft_power(snaps)
        item["feat_scatter"] = extract_scattering_pooled(snaps)
        item["feat_raw"] = extract_raw_pooled(snaps)
        
        # Fine-only features (for RG task)
        item["feat_fft_fine"] = extract_fft_power_fine_only(snaps, coarse_b)
        item["feat_scatter_fine"] = extract_scattering_fine_only(snaps, coarse_b)
        item["feat_raw_fine"] = extract_raw_fine_only(snaps, coarse_b)
    
    # Print dims
    for L in L_values:
        sample = [d for d in all_data if d["L"] == L][0]
        print(f"  L={L}:")
        print(f"    Full:      fft={sample['feat_fft'].shape[1]}, scatter={sample['feat_scatter'].shape[1]}, raw={sample['feat_raw'].shape[1]}")
        print(f"    Fine-only: fft={sample['feat_fft_fine'].shape[1]}, scatter={sample['feat_scatter_fine'].shape[1]}, raw={sample['feat_raw_fine'].shape[1]}")
    
    # ===== TRAIN/TEST SPLIT =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    print(f"\nTrain seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== EVALUATION =====
    def evaluate(X_train, X_test, y_train, y_test, k_values):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
        
        var = np.var(X_train_s, axis=0)
        valid = var > 1e-10
        if valid.sum() == 0:
            return {"error": "zero_var"}
        X_train_s, X_test_s = X_train_s[:, valid], X_test_s[:, valid]
        
        max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
        if max_k < min(k_values):
            return {"error": "low_dims"}
        
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
            
            std_test = np.std(y_test)
            results["by_k"][k] = {
                "r2": float(r2_score(y_test, y_pred)) if std_test > 1e-3 else None,
                "mse": float(mean_squared_error(y_test, y_pred)),
                "std_test": float(std_test),
            }
        return results
    
    # ===== RESULTS =====
    results = {
        "config": {
            "L_values": L_values,
            "T_values": T_values,
            "TC": TC,
            "coarse_b": coarse_b,
            "k_values": k_values,
        },
        "task1_T_same_L": {},
        "task2_T_cross_L": {},
        "task3_RG_coarse_obs": {},
    }
    
    feature_types = ["fft", "scatter", "raw"]
    
    # ===== TASK 1: T PREDICTION (SAME L) =====
    print("\n" + "="*60)
    print("TASK 1: Temperature prediction (same L)")
    print("="*60)
    
    for L in L_values:
        L_data = [d for d in all_data if d["L"] == L]
        train_data = [d for d in L_data if d["seed"] in train_seeds]
        test_data = [d for d in L_data if d["seed"] in test_seeds]
        
        y_train = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in train_data])
        y_test = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in test_data])
        
        results["task1_T_same_L"][L] = {}
        print(f"\nL={L}:")
        
        for feat in feature_types:
            X_train = np.concatenate([d[f"feat_{feat}"] for d in train_data])
            X_test = np.concatenate([d[f"feat_{feat}"] for d in test_data])
            
            res = evaluate(X_train, X_test, y_train, y_test, k_values)
            results["task1_T_same_L"][L][feat] = res
            
            r2_16 = res.get("by_k", {}).get(16, {})
            print(f"  {feat}: R²@k=16 = {r2_16.get('r2', 'N/A') if r2_16 else 'N/A'}")
    
    # ===== TASK 2: T PREDICTION (CROSS-L) =====
    print("\n" + "="*60)
    print("TASK 2: Temperature prediction (train L=32 → test L=64)")
    print("With FIXED FFT (power spectrum)")
    print("="*60)
    
    if 32 in L_values and 64 in L_values:
        train_data = [d for d in all_data if d["L"] == 32]
        test_data = [d for d in all_data if d["L"] == 64]
        
        y_train = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in train_data])
        y_test = np.concatenate([np.full(d["snapshots"].shape[0], d["T"]) for d in test_data])
        
        print(f"Train: L=32, n={len(y_train)}")
        print(f"Test: L=64, n={len(y_test)}")
        
        for feat in feature_types:
            X_train = np.concatenate([d[f"feat_{feat}"] for d in train_data])
            X_test = np.concatenate([d[f"feat_{feat}"] for d in test_data])
            
            res = evaluate(X_train, X_test, y_train, y_test, k_values)
            results["task2_T_cross_L"][feat] = res
            
            print(f"\n{feat}:")
            for k in k_values:
                k_res = res.get("by_k", {}).get(k, {})
                if k_res:
                    print(f"  k={k}: R²={k_res.get('r2', 'N/A')}")
    
    # ===== TASK 3: TRUE RG (FINE FEATURES → COARSE OBSERVABLES) =====
    print("\n" + "="*60)
    print(f"TASK 3: TRUE RG (fine-scale features → coarse observables at b={coarse_b})")
    print("Features constrained to scales < b")
    print("="*60)
    
    for L in L_values:
        L_data = [d for d in all_data if d["L"] == L]
        train_data = [d for d in L_data if d["seed"] in train_seeds]
        test_data = [d for d in L_data if d["seed"] in test_seeds]
        
        results["task3_RG_coarse_obs"][L] = {}
        print(f"\nL={L}:")
        
        for target_name in ["abs_m", "E"]:
            y_train = np.concatenate([d["coarse_obs"][target_name] for d in train_data])
            y_test = np.concatenate([d["coarse_obs"][target_name] for d in test_data])
            
            std_train, std_test = np.std(y_train), np.std(y_test)
            print(f"\n  Target: {target_name} (std_train={std_train:.4f}, std_test={std_test:.4f})")
            
            results["task3_RG_coarse_obs"][L][target_name] = {}
            
            for feat in feature_types:
                # Use FINE-ONLY features
                X_train = np.concatenate([d[f"feat_{feat}_fine"] for d in train_data])
                X_test = np.concatenate([d[f"feat_{feat}_fine"] for d in test_data])
                
                res = evaluate(X_train, X_test, y_train, y_test, k_values)
                results["task3_RG_coarse_obs"][L][target_name][feat] = res
                
                r2_16 = res.get("by_k", {}).get(16, {})
                r2_val = r2_16.get("r2") if r2_16 else None
                print(f"    {feat}: R²@k=16 = {r2_val:.3f}" if r2_val else f"    {feat}: R²@k=16 = N/A (low var)")
    
    return results


@app.local_entrypoint()
def main():
    import json
    from datetime import datetime
    
    print("="*70)
    print("2D RG v7: FIXED FFT + TRUE RG TASK")
    print("="*70)
    
    results = run_rg_experiment_v7.remote(
        L_values=[32, 64],
        T_values=[1.5, 2.0, 2.2, 2.269, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=20,
        coarse_b=4,
        k_values=[4, 8, 16, 32],
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_v7_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY =====
    k_values = results["config"]["k_values"]
    
    print("\n" + "="*70)
    print("TASK 2: CROSS-L TEMPERATURE PREDICTION (L=32 → L=64)")
    print("With FIXED FFT (power spectrum, log-normalized)")
    print("="*70)
    
    print(f"\n{'Method':<10} | ", end="")
    for k in k_values:
        print(f"k={k:<3} ", end="")
    print()
    print("-"*45)
    
    for feat in ["fft", "scatter", "raw"]:
        feat_data = results["task2_T_cross_L"].get(feat, {})
        by_k = feat_data.get("by_k", {})
        
        print(f"{feat:<10} | ", end="")
        for k in k_values:
            k_data = by_k.get(k, {})
            r2 = k_data.get("r2") if k_data else None
            print(f"{r2:.3f} " if r2 is not None else "N/A   ", end="")
        print()
    
    print("\n" + "="*70)
    print("TASK 3: TRUE RG (fine features → coarse observables)")
    print("="*70)
    
    for L in results["task3_RG_coarse_obs"]:
        print(f"\nL={L}:")
        for target in ["abs_m", "E"]:
            print(f"  {target}:")
            target_data = results["task3_RG_coarse_obs"][L].get(target, {})
            
            for feat in ["fft", "scatter", "raw"]:
                feat_data = target_data.get(feat, {})
                r2_16 = feat_data.get("by_k", {}).get(16, {})
                r2 = r2_16.get("r2") if r2_16 else None
                print(f"    {feat}: R²={r2:.3f}" if r2 is not None else f"    {feat}: N/A")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("- Task 2: If FFT now works cross-L, scatter's advantage was an artifact")
    print("- Task 3: True RG test - which features best predict coarse from fine?")
    print("="*70)
    
    return results
