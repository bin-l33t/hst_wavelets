"""
XY Model / BKT Physics Benchmark

This is where HST should actually beat Mallat:
- Data: XY spins s(x) = e^(iθ(x)) near BKT transition
- Label: vortex density or T classification
- Nuisance: random global phase at test time (NOT in training)

Why Mallat should lose: Re/Im channels mix under global phase rotation.
Why HST could win: if it builds invariants from phase gradients/circulation.

Also adds holonomy/circulation features as suggested.
"""

import modal

app = modal.App("xy-bkt-benchmark")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "kymatio>=0.3",
)


@app.function(image=image, gpu="T4", timeout=2400, memory=8192)
def run_xy_benchmark() -> dict:
    import numpy as np
    import torch
    from sklearn.linear_model import RidgeClassifier, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    # ========================================
    # XY MODEL SIMULATION
    # ========================================
    
    def simulate_xy(L, T, n_eq=2000, n_snap=10, n_gap=100, seed=0):
        """
        Simulate 2D XY model using Metropolis.
        Returns complex field z(x,y) = e^(iθ(x,y))
        """
        rng = np.random.default_rng(seed)
        
        # Initialize random angles
        theta = rng.uniform(0, 2*np.pi, (L, L))
        
        def energy_diff(i, j, new_theta):
            """Energy change from flipping spin (i,j) to new_theta."""
            old = theta[i, j]
            neighbors = [
                theta[(i+1) % L, j],
                theta[(i-1) % L, j],
                theta[i, (j+1) % L],
                theta[i, (j-1) % L],
            ]
            old_E = -sum(np.cos(old - n) for n in neighbors)
            new_E = -sum(np.cos(new_theta - n) for n in neighbors)
            return new_E - old_E
        
        def metropolis_sweep():
            for _ in range(L * L):
                i, j = rng.integers(0, L), rng.integers(0, L)
                new_theta = rng.uniform(0, 2*np.pi)
                dE = energy_diff(i, j, new_theta)
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    theta[i, j] = new_theta
        
        # Equilibrate
        for _ in range(n_eq):
            metropolis_sweep()
        
        # Collect snapshots
        snapshots = []
        for _ in range(n_snap):
            for _ in range(n_gap):
                metropolis_sweep()
            # Convert to complex field
            z = np.exp(1j * theta)
            snapshots.append(z.copy())
        
        return np.array(snapshots)
    
    def compute_vortex_density(z):
        """
        Compute vortex density from complex field.
        Vortex = 2π winding around a plaquette.
        
        FIXED: Don't wrap the final circulation - that erases vortices!
        """
        L = z.shape[0]
        theta = np.angle(z)
        
        # Phase differences (wrapped to [-π, π])
        def wrap(x):
            return np.mod(x + np.pi, 2*np.pi) - np.pi
        
        # Edge phase differences (wrapped is correct here)
        d_right = wrap(np.roll(theta, -1, axis=1) - theta)
        d_down = wrap(np.roll(theta, -1, axis=0) - theta)
        
        # Plaquette circulation: Σ dθ around square
        # DO NOT WRAP THIS - vortices have circulation ±2π
        circulation = (
            d_right + 
            np.roll(d_down, -1, axis=1) - 
            np.roll(d_right, -1, axis=0) - 
            d_down
        )
        
        # Vortex charge = circulation / 2π (should be integers ±1, 0)
        vortex_charge = np.rint(circulation / (2 * np.pi))
        
        # Vortex density = mean |charge|
        return np.mean(np.abs(vortex_charge))
    
    # ========================================
    # FEATURE EXTRACTORS
    # ========================================
    
    def R_mod(z):
        return torch.abs(z)
    
    def R_glinsky(z):
        """Glinsky's conformal rectifier."""
        def joukowsky_inverse(w):
            sqrt_term = torch.sqrt(w**2 - 1.0 + 0j)
            z1 = w + sqrt_term
            z2 = w - sqrt_term
            return torch.where(torch.abs(z1) >= torch.abs(z2), z1, z2)
        
        w = 2.0 * z / np.pi
        h_inv = joukowsky_inverse(w)
        R0 = h_inv / 1j
        EPS = 1e-10
        R0_safe = torch.where(torch.abs(R0) < EPS, 
                              torch.ones_like(R0) * EPS, R0)
        return 1j * torch.log(R0_safe)
    
    def extract_hst(data, rectifier="mod", J=3, Q=2):
        """HST with selectable rectifier."""
        R_func = R_mod if rectifier == "mod" else R_glinsky
        
        N, L, _ = data.shape
        
        def make_filterbank(signal_len):
            k = torch.fft.fftfreq(signal_len, dtype=torch.float64, device=device)
            omega = k * 2 * np.pi
            w = torch.abs(omega)
            pos_mask = k > 0
            neg_mask = k < 0
            
            num_mothers = J * Q
            xi_max = np.pi
            m = 4
            lp_sum_sq = torch.zeros(signal_len, dtype=torch.float64, device=device)
            filters = []
            
            for channel_mask in [pos_mask, neg_mask]:
                for j in range(num_mothers):
                    xi_j = xi_max * (2 ** (-j / Q))
                    arg = (m / xi_j) * w
                    psi = torch.zeros(signal_len, dtype=torch.float64, device=device)
                    valid = channel_mask & (arg > 1e-10)
                    if valid.any():
                        psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
                    filters.append(psi)
                    lp_sum_sq += psi ** 2
            
            xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
            phi = torch.exp(-w**2 / (2 * xi_min**2))
            filters.append(phi)
            lp_sum_sq += phi ** 2
            
            lp_sum = torch.sqrt(torch.clamp(lp_sum_sq, min=1e-20))
            return [(f / lp_sum).to(torch.complex128) for f in filters], 2 * num_mothers
        
        def hst_1d(x, filters, n_mothers):
            EPS = 1e-8
            def lift(z):
                r = torch.abs(z)
                return z * torch.sqrt(r**2 + EPS**2) / torch.clamp(r, min=1e-300)
            def conv(x, f):
                return torch.fft.ifft(torch.fft.fft(x) * f)
            
            phi = filters[-1]
            mothers = filters[:-1]
            
            features = []
            x_lifted = lift(x)
            lp = conv(x_lifted, phi)
            features.append(torch.mean(torch.abs(lp)**2).item())
            
            for j1 in range(min(n_mothers, 8)):
                U1 = conv(x_lifted, mothers[j1])
                W1 = R_func(lift(U1))
                if torch.is_complex(W1):
                    features.extend([
                        torch.mean(W1.real).item(),
                        torch.mean(W1.imag).item(),
                        torch.std(W1.real).item(),
                        torch.std(W1.imag).item(),
                    ])
                else:
                    features.extend([torch.mean(W1).item(), torch.std(W1).item()])
            
            return np.array(features)
        
        filters, n_mothers = make_filterbank(L)
        
        all_features = []
        for i in range(N):
            z = torch.from_numpy(data[i]).to(torch.complex128).to(device)
            row_feats = [hst_1d(z[r, :], filters, n_mothers) for r in range(L)]
            col_feats = [hst_1d(z[:, c], filters, n_mothers) for c in range(L)]
            all_features.append(np.concatenate([
                np.mean(row_feats, axis=0),
                np.mean(col_feats, axis=0)
            ]))
        
        return np.array(all_features)
    
    def extract_holonomy(data):
        """
        Gauge-invariant holonomy/circulation features.
        These should be naturally invariant to global phase.
        
        FIXED: Use vortex charge (circulation/2π) not wrap(circulation)
        """
        N, L, _ = data.shape
        features = []
        
        for i in range(N):
            z = data[i]
            theta = np.angle(z)
            
            def wrap(x):
                return np.mod(x + np.pi, 2*np.pi) - np.pi
            
            # Phase gradients (gauge-covariant, wrapped on edges)
            dx = wrap(np.roll(theta, -1, axis=1) - theta)
            dy = wrap(np.roll(theta, -1, axis=0) - theta)
            
            # Plaquette circulation (DO NOT WRAP - preserves vortices)
            circulation = (
                dx + 
                np.roll(dy, -1, axis=1) - 
                np.roll(dx, -1, axis=0) - 
                dy
            )
            
            # Vortex charge = circulation / 2π (integers)
            vortex_charge = np.rint(circulation / (2 * np.pi))
            
            # Features from vortex charge (gauge-invariant)
            feat = [
                # Vortex statistics
                np.mean(np.abs(vortex_charge)),  # Vortex density
                np.mean(vortex_charge**2),       # Second moment
                np.sum(vortex_charge > 0.5) / L**2,   # Positive vortex density
                np.sum(vortex_charge < -0.5) / L**2,  # Negative vortex density
                np.mean(vortex_charge),          # Net vorticity (should be ~0)
                
                # Gradient magnitude features (gauge-invariant)
                np.mean(np.abs(dx)),
                np.mean(np.abs(dy)),
                np.std(dx),
                np.std(dy),
                np.mean(dx**2 + dy**2),  # "Energy" proxy
                
                # Vortex correlations
                np.mean(vortex_charge * np.roll(vortex_charge, 1, axis=0)),
                np.mean(vortex_charge * np.roll(vortex_charge, 1, axis=1)),
                np.mean(vortex_charge * np.roll(vortex_charge, 2, axis=0)),
                np.mean(vortex_charge * np.roll(vortex_charge, 2, axis=1)),
            ]
            
            features.append(feat)
        
        return np.array(features)
    
    def extract_mallat_reim(data, J=2):
        """Mallat with Re/Im channels."""
        from kymatio.torch import Scattering2D
        
        N, L, _ = data.shape
        re = torch.from_numpy(data.real).float()
        im = torch.from_numpy(data.imag).float()
        x = torch.stack([re, im], dim=1).to(device)
        
        scattering = Scattering2D(J=J, shape=(L, L), max_order=2).to(device)
        
        with torch.no_grad():
            Sx = scattering(x)
        
        if len(Sx.shape) == 5:
            features = Sx.mean(dim=(-1, -2)).reshape(N, -1)
        else:
            features = Sx.mean(dim=(-1, -2))
        
        return features.cpu().numpy()
    
    # ========================================
    # GENERATE XY DATA
    # ========================================
    
    print("="*70)
    print("XY MODEL / BKT BENCHMARK")
    print("="*70)
    
    L = 32
    # BKT transition around T ≈ 0.89 for 2D XY
    T_values = [0.5, 0.7, 0.89, 1.1, 1.5]
    n_seeds = 8
    n_snap = 5
    
    print(f"\nGenerating XY model data: L={L}, T={T_values}")
    
    all_data = []
    all_T = []
    all_vortex = []
    
    for T in T_values:
        for seed in range(n_seeds):
            snaps = simulate_xy(L, T, seed=seed + int(T*1000))
            for snap in snaps:
                all_data.append(snap)
                all_T.append(T)
                all_vortex.append(compute_vortex_density(snap))
    
    all_data = np.array(all_data)
    all_T = np.array(all_T)
    all_vortex = np.array(all_vortex)
    
    print(f"Generated {len(all_data)} samples")
    print(f"Vortex density range: {all_vortex.min():.3f} - {all_vortex.max():.3f}")
    
    # ========================================
    # TASK 1: TEMPERATURE CLASSIFICATION
    # ========================================
    
    print("\n" + "="*70)
    print("TASK 1: Temperature Classification (5 classes)")
    print("="*70)
    
    # Create labels
    T_labels = np.array([T_values.index(t) for t in all_T])
    
    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(all_data)), test_size=0.3, 
        random_state=42, stratify=T_labels
    )
    
    train_data = all_data[train_idx]
    test_data = all_data[test_idx]
    train_labels = T_labels[train_idx]
    test_labels = T_labels[test_idx]
    
    # Apply random global phase to test data (THE KEY TEST)
    rng = np.random.default_rng(456)
    random_phases = rng.uniform(0, 2*np.pi, len(test_data))
    test_data_rotated = np.array([
        test_data[i] * np.exp(1j * random_phases[i])
        for i in range(len(test_data))
    ])
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print("Test data has RANDOM GLOBAL PHASE applied")
    
    # Extract features
    print("\nExtracting features...")
    
    methods = {
        "HST_mod": lambda d: extract_hst(d, rectifier="mod"),
        "HST_glinsky": lambda d: extract_hst(d, rectifier="glinsky"),
        "Holonomy": extract_holonomy,
        "Mallat_ReIm": extract_mallat_reim,
    }
    
    train_features = {}
    test_features_orig = {}
    test_features_rot = {}
    
    for name, extractor in methods.items():
        print(f"  {name}...")
        train_features[name] = extractor(train_data)
        test_features_orig[name] = extractor(test_data)
        test_features_rot[name] = extractor(test_data_rotated)
    
    # Also test combined features
    print("  HST_glinsky + Holonomy...")
    train_features["Glinsky+Holonomy"] = np.hstack([
        train_features["HST_glinsky"], 
        train_features["Holonomy"]
    ])
    test_features_orig["Glinsky+Holonomy"] = np.hstack([
        test_features_orig["HST_glinsky"],
        test_features_orig["Holonomy"]
    ])
    test_features_rot["Glinsky+Holonomy"] = np.hstack([
        test_features_rot["HST_glinsky"],
        test_features_rot["Holonomy"]
    ])
    
    # Evaluate
    print("\nTemperature Classification Accuracy:")
    print("-"*70)
    print(f"{'Method':<20} | {'Test (orig)':<12} | {'Test (rotated)':<14} | {'Drop':<8}")
    print("-"*70)
    
    task1_results = {}
    
    for name in train_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_features[name])
        X_test_orig = scaler.transform(test_features_orig[name])
        X_test_rot = scaler.transform(test_features_rot[name])
        
        X_train = np.nan_to_num(X_train)
        X_test_orig = np.nan_to_num(X_test_orig)
        X_test_rot = np.nan_to_num(X_test_rot)
        
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_train, train_labels)
        
        acc_orig = accuracy_score(test_labels, clf.predict(X_test_orig))
        acc_rot = accuracy_score(test_labels, clf.predict(X_test_rot))
        drop = acc_orig - acc_rot
        
        task1_results[name] = {
            "acc_original": float(acc_orig),
            "acc_rotated": float(acc_rot),
            "drop": float(drop),
        }
        
        marker = "←UNSTABLE" if drop > 0.15 else ("STABLE→" if drop < 0.05 else "")
        print(f"{name:<20} | {acc_orig:<12.3f} | {acc_rot:<14.3f} | {drop:<+8.3f} {marker}")
    
    results["task1_T_classification"] = task1_results
    
    # ========================================
    # TASK 2: VORTEX DENSITY PREDICTION
    # ========================================
    
    print("\n" + "="*70)
    print("TASK 2: Vortex Density Prediction (regression)")
    print("This is a gauge-INVARIANT observable")
    print("="*70)
    
    train_vortex = all_vortex[train_idx]
    test_vortex = all_vortex[test_idx]
    
    print("\nVortex Density R²:")
    print("-"*70)
    print(f"{'Method':<20} | {'Test (orig)':<12} | {'Test (rotated)':<14} | {'Drop':<8}")
    print("-"*70)
    
    task2_results = {}
    
    for name in train_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_features[name])
        X_test_orig = scaler.transform(test_features_orig[name])
        X_test_rot = scaler.transform(test_features_rot[name])
        
        X_train = np.nan_to_num(X_train)
        X_test_orig = np.nan_to_num(X_test_orig)
        X_test_rot = np.nan_to_num(X_test_rot)
        
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
        ridge.fit(X_train, train_vortex)
        
        r2_orig = r2_score(test_vortex, ridge.predict(X_test_orig))
        r2_rot = r2_score(test_vortex, ridge.predict(X_test_rot))
        drop = r2_orig - r2_rot
        
        task2_results[name] = {
            "r2_original": float(r2_orig),
            "r2_rotated": float(r2_rot),
            "drop": float(drop),
        }
        
        marker = "←UNSTABLE" if drop > 0.15 else ("STABLE→" if drop < 0.05 else "")
        print(f"{name:<20} | {r2_orig:<12.3f} | {r2_rot:<14.3f} | {drop:<+8.3f} {marker}")
    
    results["task2_vortex_prediction"] = task2_results
    
    # ========================================
    # TASK 3: BKT PHASE CLASSIFICATION
    # ========================================
    
    print("\n" + "="*70)
    print("TASK 3: BKT Phase Classification (low T vs high T)")
    print("="*70)
    
    # Binary labels: T < 0.89 (ordered) vs T >= 0.89 (disordered)
    bkt_labels = (all_T >= 0.89).astype(int)
    train_bkt = bkt_labels[train_idx]
    test_bkt = bkt_labels[test_idx]
    
    print("\nBKT Phase Classification Accuracy:")
    print("-"*70)
    print(f"{'Method':<20} | {'Test (orig)':<12} | {'Test (rotated)':<14} | {'Drop':<8}")
    print("-"*70)
    
    task3_results = {}
    
    for name in train_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_features[name])
        X_test_orig = scaler.transform(test_features_orig[name])
        X_test_rot = scaler.transform(test_features_rot[name])
        
        X_train = np.nan_to_num(X_train)
        X_test_orig = np.nan_to_num(X_test_orig)
        X_test_rot = np.nan_to_num(X_test_rot)
        
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_train, train_bkt)
        
        acc_orig = accuracy_score(test_bkt, clf.predict(X_test_orig))
        acc_rot = accuracy_score(test_bkt, clf.predict(X_test_rot))
        drop = acc_orig - acc_rot
        
        task3_results[name] = {
            "acc_original": float(acc_orig),
            "acc_rotated": float(acc_rot),
            "drop": float(drop),
        }
        
        marker = "←UNSTABLE" if drop > 0.15 else ("STABLE→" if drop < 0.05 else "")
        print(f"{name:<20} | {acc_orig:<12.3f} | {acc_rot:<14.3f} | {drop:<+8.3f} {marker}")
    
    results["task3_bkt_phase"] = task3_results
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "="*70)
    print("SUMMARY: Gauge Stability Under Global Phase Rotation")
    print("="*70)
    
    print("\nMean drop across tasks:")
    for name in train_features:
        drops = [
            task1_results[name]["drop"],
            task2_results[name]["drop"],
            task3_results[name]["drop"],
        ]
        mean_drop = np.mean(drops)
        stability = "STABLE" if mean_drop < 0.05 else ("UNSTABLE" if mean_drop > 0.15 else "PARTIAL")
        print(f"  {name:<20}: mean drop = {mean_drop:+.3f} [{stability}]")
    
    # Determine winners
    print("\n" + "-"*70)
    
    # Best on rotated test
    best_t_rot = max(task1_results.items(), key=lambda x: x[1]["acc_rotated"])
    best_vortex_rot = max(task2_results.items(), key=lambda x: x[1]["r2_rotated"])
    best_bkt_rot = max(task3_results.items(), key=lambda x: x[1]["acc_rotated"])
    
    print(f"Best on rotated T classification: {best_t_rot[0]} ({best_t_rot[1]['acc_rotated']:.3f})")
    print(f"Best on rotated vortex prediction: {best_vortex_rot[0]} ({best_vortex_rot[1]['r2_rotated']:.3f})")
    print(f"Best on rotated BKT phase: {best_bkt_rot[0]} ({best_bkt_rot[1]['acc_rotated']:.3f})")
    
    # Does Holonomy help?
    holonomy_wins = []
    for task_name, task_results in [
        ("T", task1_results), 
        ("vortex", task2_results), 
        ("BKT", task3_results)
    ]:
        metric = "acc_rotated" if task_name != "vortex" else "r2_rotated"
        if task_results["Holonomy"][metric] > task_results["Mallat_ReIm"][metric]:
            holonomy_wins.append(task_name)
    
    print(f"\nHolonomy beats Mallat on: {holonomy_wins if holonomy_wins else 'nothing'}")
    
    # Does Glinsky help?
    glinsky_wins = []
    for task_name, task_results in [
        ("T", task1_results), 
        ("vortex", task2_results), 
        ("BKT", task3_results)
    ]:
        metric = "acc_rotated" if task_name != "vortex" else "r2_rotated"
        if task_results["HST_glinsky"][metric] > task_results["HST_mod"][metric]:
            glinsky_wins.append(task_name)
    
    print(f"Glinsky beats HST_mod on: {glinsky_wins if glinsky_wins else 'nothing'}")
    
    results["summary"] = {
        "holonomy_wins": holonomy_wins,
        "glinsky_wins": glinsky_wins,
        "best_T_rotated": best_t_rot[0],
        "best_vortex_rotated": best_vortex_rot[0],
        "best_BKT_rotated": best_bkt_rot[0],
    }
    
    return results


@app.local_entrypoint()
def main():
    import json
    from datetime import datetime
    
    print("="*70)
    print("XY MODEL / BKT PHYSICS BENCHMARK")
    print("Testing gauge stability under global phase rotation")
    print("="*70)
    
    results = run_xy_benchmark.remote()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"xy_bkt_benchmark_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    return results
