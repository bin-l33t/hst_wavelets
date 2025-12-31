"""
2D Spatial RG Task: Predict coarse-grained observables from fine-scale features

Following ChatGPT's tight spec:
- Generate 2D Ising snapshots
- Compute coarse-grained targets (m^(2), E^(2), C1^(2))
- Extract features (Kymatio Scattering2D baseline, raw PCA)
- Ridge regression with seed-based train/test split
- Report R² vs k across T
"""

import modal

app = modal.App("ising-2d-rg")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, timeout=1800, cpu=4, memory=8192)
def run_rg_experiment(
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
    Run the full 2D RG experiment for all temperatures.
    
    Returns R² vs k for each target and feature type.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    if T_values is None:
        T_values = [1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0]
    if k_values is None:
        k_values = [8, 16, 32, 64, 128]
    
    print(f"Config: L={L}, T={len(T_values)} points, seeds={n_seeds}, snaps={n_snapshots}")
    
    # ===== ISING SIMULATION =====
    def generate_snapshots(L, T, seed, n_eq, n_snap, n_gap):
        """Generate equilibrated snapshots."""
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
        
        # Equilibrate
        for _ in range(n_eq):
            wolff_step()
        
        # Collect snapshots
        snapshots = []
        for _ in range(n_snap):
            for _ in range(n_gap):
                wolff_step()
            snapshots.append(spins.copy())
        
        return np.array(snapshots)  # (n_snap, L, L)
    
    # ===== COARSE GRAINING =====
    def coarse_grain(s, b=2):
        """Block average with block size b."""
        L = s.shape[-1]
        L_coarse = L // b
        s_coarse = s.reshape(s.shape[0], L_coarse, b, L_coarse, b).mean(axis=(2, 4))
        return s_coarse  # (N, L/b, L/b)
    
    def compute_targets(s, b=2):
        """
        Compute coarse-grained observables.
        s: (N, L, L) array of ±1 spins
        Returns dict of targets, each shape (N,)
        """
        s_coarse = coarse_grain(s, b)  # (N, L/b, L/b)
        s_sign = np.sign(s_coarse)  # Signed block spins (0 for ties)
        
        # m^(2): coarse magnetization
        m2 = s_coarse.mean(axis=(1, 2))  # (N,)
        
        # E^(2): coarse energy (using signed spins)
        # E = -sum_{<i,j>} s_i * s_j / (2 * N_sites)
        L_c = s_sign.shape[1]
        nn_sum = (
            s_sign * np.roll(s_sign, 1, axis=1) +  # vertical
            s_sign * np.roll(s_sign, 1, axis=2)    # horizontal
        ).sum(axis=(1, 2))
        E2 = -nn_sum / (2 * L_c * L_c)  # (N,)
        
        # C1^(2): nearest-neighbor correlation
        C1_x = (s_sign * np.roll(s_sign, 1, axis=2)).mean(axis=(1, 2))
        C1_y = (s_sign * np.roll(s_sign, 1, axis=1)).mean(axis=(1, 2))
        C1 = (C1_x + C1_y) / 2  # (N,)
        
        return {"m2": m2, "E2": E2, "C1": C1}
    
    # ===== FEATURE EXTRACTION =====
    def extract_scattering_features(snapshots):
        """Extract Kymatio Scattering2D features."""
        from kymatio.numpy import Scattering2D
        
        L = snapshots.shape[1]
        J = 3  # Scales
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2)
        
        # Kymatio expects (N, 1, L, L)
        x = snapshots[:, np.newaxis, :, :]
        
        # Compute scattering
        Sx = scattering(x)  # (N, C, L', L')
        
        # Flatten spatial dims, keep channels
        features = Sx.reshape(Sx.shape[0], -1)  # (N, C*L'*L')
        
        return features
    
    def extract_raw_features(snapshots):
        """Just flatten the raw field."""
        return snapshots.reshape(snapshots.shape[0], -1)
    
    def extract_fft_features(snapshots, n_bins=32):
        """2D FFT magnitude in radial bins."""
        N, L, _ = snapshots.shape
        features = []
        
        for i in range(N):
            fft2 = np.fft.fft2(snapshots[i])
            fft2_shift = np.fft.fftshift(fft2)
            mag = np.abs(fft2_shift)
            
            # Radial binning
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
    
    # ===== GENERATE ALL DATA =====
    print("Generating snapshots...")
    
    all_data = []  # List of (seed, T, snapshots, targets)
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = generate_snapshots(L, T, seed, n_eq, n_snapshots, n_gap)
            targets = compute_targets(snaps, b=2)
            all_data.append({
                "seed": seed,
                "T": T,
                "snapshots": snaps,
                "targets": targets,
            })
    
    print(f"Generated {len(all_data)} (T, seed) combinations")
    print(f"Total snapshots: {len(all_data) * n_snapshots}")
    
    # ===== EXTRACT FEATURES =====
    print("Extracting features...")
    
    for item in all_data:
        snaps = item["snapshots"]
        item["feat_scatter"] = extract_scattering_features(snaps)
        item["feat_raw"] = extract_raw_features(snaps)
        item["feat_fft"] = extract_fft_features(snaps)
    
    print(f"Feature dims: scatter={all_data[0]['feat_scatter'].shape[1]}, "
          f"raw={all_data[0]['feat_raw'].shape[1]}, "
          f"fft={all_data[0]['feat_fft'].shape[1]}")
    
    # ===== TRAIN/TEST SPLIT BY SEED =====
    n_test_seeds = max(1, int(n_seeds * test_seed_frac))
    test_seeds = set(range(n_seeds - n_test_seeds, n_seeds))
    train_seeds = set(range(n_seeds)) - test_seeds
    
    print(f"Train seeds: {sorted(train_seeds)}, Test seeds: {sorted(test_seeds)}")
    
    # ===== RUN REGRESSIONS =====
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
        "by_T": {},
    }
    
    feature_types = ["scatter", "raw", "fft"]
    target_names = ["m2", "E2", "C1"]
    
    for T in T_values:
        print(f"\nProcessing T={T:.3f}...")
        T_results = {}
        
        # Get data for this T
        T_data = [d for d in all_data if abs(d["T"] - T) < 0.001]
        
        # Split by seed
        train_data = [d for d in T_data if d["seed"] in train_seeds]
        test_data = [d for d in T_data if d["seed"] in test_seeds]
        
        for target_name in target_names:
            T_results[target_name] = {}
            
            # Collect targets
            y_train = np.concatenate([d["targets"][target_name] for d in train_data])
            y_test = np.concatenate([d["targets"][target_name] for d in test_data])
            
            for feat_type in feature_types:
                feat_key = f"feat_{feat_type}"
                
                # Collect features
                X_train = np.concatenate([d[feat_key] for d in train_data])
                X_test = np.concatenate([d[feat_key] for d in test_data])
                
                # Standardize
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                # Handle NaN/Inf
                X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
                X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)
                
                # PCA + Ridge for each k
                r2_by_k = {}
                max_k = min(X_train_s.shape[1], X_train_s.shape[0] - 1, max(k_values))
                
                if max_k > 0:
                    pca = PCA(n_components=max_k)
                    X_train_pca = pca.fit_transform(X_train_s)
                    X_test_pca = pca.transform(X_test_s)
                    
                    for k in k_values:
                        if k > max_k:
                            continue
                        
                        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                        ridge.fit(X_train_pca[:, :k], y_train)
                        y_pred = ridge.predict(X_test_pca[:, :k])
                        
                        r2 = r2_score(y_test, y_pred)
                        r2_by_k[k] = float(r2)
                
                T_results[target_name][feat_type] = r2_by_k
        
        results["by_T"][str(T)] = T_results
    
    return results


@app.local_entrypoint()
def main():
    import json
    import math
    from datetime import datetime
    
    print("="*70)
    print("2D SPATIAL RG TASK: COARSE OBSERVABLE PREDICTION")
    print("="*70)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Run experiment
    results = run_rg_experiment.remote(
        L=64,
        T_values=[1.5, 2.0, 2.2, TC, 2.35, 2.5, 3.0],
        n_seeds=10,
        n_snapshots=10,
        k_values=[8, 16, 32, 64, 128],
    )
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rg_2d_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== DISPLAY RESULTS =====
    import numpy as np
    
    T_values = results["config"]["T_values"]
    k_values = results["config"]["k_values"]
    
    print("\n" + "="*70)
    print("R² RESULTS (higher is better)")
    print("="*70)
    
    for target in ["m2", "E2", "C1"]:
        print(f"\n{'='*70}")
        print(f"TARGET: {target}")
        print(f"{'='*70}")
        
        print(f"\n{'T':<8} | ", end="")
        for k in k_values:
            print(f"k={k:<4} ", end="")
        print("| Best feat")
        print("-"*70)
        
        for T in T_values:
            T_key = str(T)
            T_data = results["by_T"].get(T_key, {})
            target_data = T_data.get(target, {})
            
            marker = "*" if abs(T - TC) < 0.01 else " "
            print(f"{T:<7.3f}{marker} | ", end="")
            
            # Find best feature type at k=32
            best_feat = None
            best_r2 = -999
            
            for feat in ["scatter", "raw", "fft"]:
                feat_data = target_data.get(feat, {})
                r2_32 = feat_data.get(32, feat_data.get("32", None))
                if r2_32 is not None and r2_32 > best_r2:
                    best_r2 = r2_32
                    best_feat = feat
            
            # Print scatter results
            scatter_data = target_data.get("scatter", {})
            for k in k_values:
                r2 = scatter_data.get(k, scatter_data.get(str(k), None))
                if r2 is not None:
                    print(f"{r2:6.3f} ", end="")
                else:
                    print(f"{'N/A':>6} ", end="")
            
            print(f"| {best_feat or 'N/A'}")
    
    # ===== COMPARISON TABLE =====
    print("\n" + "="*70)
    print("FEATURE COMPARISON at k=32")
    print("="*70)
    
    print(f"\n{'T':<8} {'Target':<6} | {'Scatter':<8} {'Raw':<8} {'FFT':<8} | Winner")
    print("-"*60)
    
    for T in T_values:
        T_key = str(T)
        T_data = results["by_T"].get(T_key, {})
        marker = "*" if abs(T - TC) < 0.01 else " "
        
        for target in ["m2", "E2", "C1"]:
            target_data = T_data.get(target, {})
            
            scores = {}
            for feat in ["scatter", "raw", "fft"]:
                feat_data = target_data.get(feat, {})
                r2 = feat_data.get(32, feat_data.get("32", None))
                scores[feat] = r2 if r2 is not None else -999
            
            winner = max(scores, key=lambda x: scores[x])
            
            print(f"{T:<7.3f}{marker} {target:<6} | ", end="")
            for feat in ["scatter", "raw", "fft"]:
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
