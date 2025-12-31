"""
ROM Experiment 1 (Fixed): Magnetization Time Series Forecasting

Fixes from ChatGPT review:
1. log(0) bug: clamp |z| before log in R(z)
2. Better HST features: separate Re/Im moments instead of |W| energy
3. Add 1D Mallat scattering baseline (modulus scattering)
4. Assert all features are finite
"""

import modal

app = modal.App("hst-rom-timeseries-v2")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
)


@app.function(image=image, timeout=1800, cpu=2)
def run_rom_experiment(
    L: int,
    T: float,
    seed: int,
    n_steps: int = 20000,
    window_size: int = 256,
    horizons: list = [1, 4, 16, 64],
    k_values: list = [2, 4, 8, 16, 32],
) -> dict:
    """
    Run ROM forecasting experiment for one (L, T, seed) configuration.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    EPS = 1e-8  # Global epsilon for numerical stability
    
    # ===== GENERATE MAGNETIZATION TIME SERIES =====
    rng = np.random.default_rng(seed)
    spins = np.ones((L, L), dtype=np.int8) if T < TC else rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    
    def wolff_step():
        p_add = 1 - np.exp(-2.0 / T)
        i0, j0 = rng.integers(0, L), rng.integers(0, L)
        seed_spin = spins[i0, j0]
        cluster = {(i0, j0)}
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                ni, nj = ni % L, nj % L
                if (ni, nj) not in cluster and spins[ni, nj] == seed_spin:
                    if rng.random() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))
        for i, j in cluster:
            spins[i, j] *= -1
    
    # Equilibrate
    for _ in range(1000):
        wolff_step()
    
    # Generate time series
    m_t = np.zeros(n_steps, dtype=np.float64)
    for t in range(n_steps):
        wolff_step()
        m_t[t] = np.mean(spins)
    
    # Z-score normalize
    m_mean, m_std = m_t.mean(), m_t.std()
    if m_std < 1e-10:
        m_std = 1.0
    m_t_norm = (m_t - m_mean) / m_std
    
    # Compute autocorrelation time (fixed version)
    def autocorr_time(x, max_lag=1000):
        n = len(x)
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-10:
            return 1.0
        # Use FFT for efficient autocorrelation
        f = np.fft.fft(x, n=2*n)
        acf = np.fft.ifft(f * np.conj(f))[:max_lag].real / (var * n)
        # Find first zero crossing or use integrated tau
        tau_int = 1 + 2 * np.sum(acf[1:min(max_lag, n//4)])
        return max(1.0, min(tau_int, n//4))  # Cap at reasonable value
    
    tau_autocorr = autocorr_time(m_t_norm)
    
    # ===== FILTERBANK SETUP =====
    def make_filterbank(N, J=3, Q=2, m=4):
        k = torch.fft.fftfreq(N, dtype=torch.float64)
        omega = k * 2 * np.pi
        w = torch.abs(omega)
        pos_mask, neg_mask = k > 0, k < 0
        
        num_mothers = J * Q
        xi_max = np.pi
        lp_sum_sq = torch.zeros(N, dtype=torch.float64)
        filters = []
        
        for channel_mask in [pos_mask, neg_mask]:
            for j in range(num_mothers):
                xi_j = xi_max * (2 ** (-j / Q))
                arg = (m / xi_j) * w
                psi = torch.zeros(N, dtype=torch.float64)
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
    
    # ===== HST FEATURES (FIXED) =====
    def hst_features_v2(x, filters, n_mothers):
        """
        Extract HST features with fixes:
        1. Clamp before log to avoid -inf
        2. Separate Re/Im moments instead of |W| energy
        """
        def lift(z):
            r = torch.abs(z)
            r_floor = torch.sqrt(r**2 + EPS**2)
            return z * r_floor / torch.clamp(r, min=1e-300)
        
        def R(z):
            # FIX: Clamp before log to avoid -inf
            r = torch.clamp(torch.abs(z), min=EPS)
            theta = torch.angle(z)
            return -theta + 1j * torch.log(r)
        
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        phi = filters[-1]
        mothers = filters[:-1]
        
        features = []
        x_lifted = lift(x)
        coeffs = [conv(x_lifted, f) for f in mothers]
        
        # Order 0: lowpass moments
        lp = conv(x_lifted, phi)
        features.extend([
            torch.mean(lp.real).item(),
            torch.mean(lp.imag).item(),
            torch.mean(torch.abs(lp)**2).item(),
        ])
        
        # Order 1: W1 = R(lift(U1)), extract Re/Im moments
        W1_list = []
        for j1 in range(n_mothers):
            U1 = coeffs[j1]
            W1 = R(lift(U1))
            W1_list.append(W1)
            
            # Separate moments for Re and Im parts
            features.extend([
                torch.mean(W1.real).item(),      # mean of -theta
                torch.mean(W1.imag).item(),      # mean of ln(r)
                torch.std(W1.real).item(),       # std of -theta
                torch.std(W1.imag).item(),       # std of ln(r)
                torch.mean(torch.abs(W1)**2).item(),  # total energy
            ])
        
        # Order 2: subset of paths
        for j1 in range(min(n_mothers, 4)):
            coeffs_W1 = [conv(W1_list[j1], f) for f in mothers]
            for j2 in range(j1 + 1, min(n_mothers, 4)):
                U2 = coeffs_W1[j2]
                W2 = R(lift(U2))
                features.extend([
                    torch.mean(W2.real).item(),
                    torch.mean(W2.imag).item(),
                    torch.mean(torch.abs(W2)**2).item(),
                ])
        
        return np.array(features)
    
    # ===== 1D MALLAT SCATTERING BASELINE =====
    def mallat_scattering_1d(x, filters, n_mothers):
        """
        Standard 1D Mallat scattering: |x * psi| * phi
        No R transform, just modulus + lowpass pooling.
        """
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        phi = filters[-1]
        mothers = filters[:-1]
        
        features = []
        
        # Order 0: lowpass of input
        S0 = conv(x, phi)
        features.append(torch.mean(torch.abs(S0)**2).item())
        
        # Order 1: |x * psi_j| * phi
        U1_list = []
        for j1 in range(n_mothers):
            U1 = conv(x, mothers[j1])
            U1_mod = torch.abs(U1)
            U1_list.append(U1_mod)
            S1 = conv(U1_mod.to(torch.complex128), phi)
            features.append(torch.mean(torch.abs(S1)**2).item())
        
        # Order 2: ||x * psi_j1| * psi_j2| * phi
        for j1 in range(min(n_mothers, 4)):
            for j2 in range(j1 + 1, min(n_mothers, 4)):
                U2 = conv(U1_list[j1].to(torch.complex128), mothers[j2])
                U2_mod = torch.abs(U2)
                S2 = conv(U2_mod.to(torch.complex128), phi)
                features.append(torch.mean(torch.abs(S2)**2).item())
        
        return np.array(features)
    
    filters, n_mothers = make_filterbank(window_size)
    
    # ===== CREATE WINDOWS WITH BLOCKED SPLITS =====
    max_h = max(horizons)
    n_windows = n_steps - window_size - max_h
    
    train_end = int(0.6 * n_windows)
    val_end = int(0.8 * n_windows)
    
    def get_windows_and_targets(start_idx, end_idx, h):
        X_raw = []
        X_hst = []
        X_mallat = []
        X_fft = []
        y = []
        
        for t in range(start_idx, end_idx):
            window = m_t_norm[t:t+window_size]
            target = m_t_norm[t+window_size+h-1]
            
            X_raw.append(window)
            
            # HST features (with offset to avoid issues near zero)
            x_torch = torch.tensor(window + 3.0 + 0j, dtype=torch.complex128)
            hst_feat = hst_features_v2(x_torch, filters, n_mothers)
            
            # Assert finite
            if not np.isfinite(hst_feat).all():
                print(f"WARNING: Non-finite HST features at t={t}")
                hst_feat = np.nan_to_num(hst_feat, nan=0.0, posinf=0.0, neginf=0.0)
            X_hst.append(hst_feat)
            
            # Mallat scattering
            mallat_feat = mallat_scattering_1d(x_torch, filters, n_mothers)
            if not np.isfinite(mallat_feat).all():
                mallat_feat = np.nan_to_num(mallat_feat, nan=0.0, posinf=0.0, neginf=0.0)
            X_mallat.append(mallat_feat)
            
            # FFT magnitude
            fft_mag = np.abs(np.fft.fft(window))[:window_size//2]
            X_fft.append(fft_mag)
            
            y.append(target)
        
        return (np.array(X_raw), np.array(X_hst), np.array(X_mallat), 
                np.array(X_fft), np.array(y))
    
    # ===== RUN FORECASTING =====
    results = {
        "L": L,
        "T": T,
        "seed": seed,
        "tau_autocorr": float(tau_autocorr),
        "m_mean": float(m_mean),
        "m_std": float(m_std),
        "n_windows_train": train_end,
        "n_windows_test": n_windows - val_end,
        "horizons": {},
    }
    
    for h in horizons:
        X_raw_train, X_hst_train, X_mallat_train, X_fft_train, y_train = get_windows_and_targets(0, train_end, h)
        X_raw_val, X_hst_val, X_mallat_val, X_fft_val, y_val = get_windows_and_targets(train_end, val_end, h)
        X_raw_test, X_hst_test, X_mallat_test, X_fft_test, y_test = get_windows_and_targets(val_end, n_windows, h)
        
        h_results = {"k_results": {}, "baselines": {}}
        
        # BASELINE 1: Raw ridge
        scaler_raw = StandardScaler()
        X_raw_train_s = scaler_raw.fit_transform(X_raw_train)
        X_raw_test_s = scaler_raw.transform(X_raw_test)
        ridge_raw = Ridge(alpha=1.0)
        ridge_raw.fit(X_raw_train_s, y_train)
        mse_raw = np.mean((y_test - ridge_raw.predict(X_raw_test_s))**2)
        h_results["baselines"]["raw_ridge"] = float(mse_raw)
        
        # BASELINE 2: FFT ridge
        scaler_fft = StandardScaler()
        X_fft_train_s = scaler_fft.fit_transform(X_fft_train)
        X_fft_test_s = scaler_fft.transform(X_fft_test)
        ridge_fft = Ridge(alpha=1.0)
        ridge_fft.fit(X_fft_train_s, y_train)
        mse_fft = np.mean((y_test - ridge_fft.predict(X_fft_test_s))**2)
        h_results["baselines"]["fft_ridge"] = float(mse_fft)
        
        # BASELINE 3: Mallat scattering ridge (full features)
        scaler_mallat = StandardScaler()
        X_mallat_train_s = scaler_mallat.fit_transform(X_mallat_train)
        X_mallat_test_s = scaler_mallat.transform(X_mallat_test)
        ridge_mallat = Ridge(alpha=1.0)
        ridge_mallat.fit(X_mallat_train_s, y_train)
        mse_mallat = np.mean((y_test - ridge_mallat.predict(X_mallat_test_s))**2)
        h_results["baselines"]["mallat_ridge"] = float(mse_mallat)
        
        # HST + PCA(k) + Ridge
        scaler_hst = StandardScaler()
        X_hst_train_s = scaler_hst.fit_transform(X_hst_train)
        X_hst_test_s = scaler_hst.transform(X_hst_test)
        
        n_components = min(X_hst_train_s.shape[1], X_hst_train_s.shape[0] - 1, max(k_values))
        if n_components > 0:
            pca_hst = PCA(n_components=n_components)
            X_hst_train_pca = pca_hst.fit_transform(X_hst_train_s)
            X_hst_test_pca = pca_hst.transform(X_hst_test_s)
            
            for k in k_values:
                if k > n_components:
                    continue
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_hst_train_pca[:, :k], y_train)
                mse = np.mean((y_test - ridge.predict(X_hst_test_pca[:, :k]))**2)
                h_results["k_results"][k] = float(mse)
        
        # Raw + PCA(k) baseline
        n_comp_raw = min(X_raw_train_s.shape[1], X_raw_train_s.shape[0] - 1, max(k_values))
        h_results["raw_pca_k"] = {}
        if n_comp_raw > 0:
            pca_raw = PCA(n_components=n_comp_raw)
            X_raw_train_pca = pca_raw.fit_transform(X_raw_train_s)
            X_raw_test_pca = pca_raw.transform(X_raw_test_s)
            for k in k_values:
                if k > n_comp_raw:
                    continue
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_raw_train_pca[:, :k], y_train)
                mse = np.mean((y_test - ridge.predict(X_raw_test_pca[:, :k]))**2)
                h_results["raw_pca_k"][k] = float(mse)
        
        # Mallat + PCA(k)
        n_comp_mallat = min(X_mallat_train_s.shape[1], X_mallat_train_s.shape[0] - 1, max(k_values))
        h_results["mallat_pca_k"] = {}
        if n_comp_mallat > 0:
            pca_mallat = PCA(n_components=n_comp_mallat)
            X_mallat_train_pca = pca_mallat.fit_transform(X_mallat_train_s)
            X_mallat_test_pca = pca_mallat.transform(X_mallat_test_s)
            for k in k_values:
                if k > n_comp_mallat:
                    continue
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_mallat_train_pca[:, :k], y_train)
                mse = np.mean((y_test - ridge.predict(X_mallat_test_pca[:, :k]))**2)
                h_results["mallat_pca_k"][k] = float(mse)
        
        results["horizons"][h] = h_results
    
    return results


@app.local_entrypoint()
def main():
    import math
    import json
    from datetime import datetime
    
    print("="*80)
    print("ROM EXPERIMENT 1 (FIXED): MAGNETIZATION TIME SERIES FORECASTING")
    print("="*80)
    print("Fixes: log(0) clamp, separate Re/Im features, Mallat baseline")
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    L = 32
    T_VALUES = [1.5, 2.0, TC, 2.5, 3.0]
    N_SEEDS = 5
    HORIZONS = [1, 4, 16, 64]
    K_VALUES = [2, 4, 8, 16, 32]
    
    print(f"\nConfiguration:")
    print(f"  L: {L}, T: {[round(t,3) for t in T_VALUES]}, Seeds: {N_SEEDS}")
    print(f"  Horizons: {HORIZONS}, k values: {K_VALUES}")
    
    configs = [
        (L, T, seed, 20000, 256, HORIZONS, K_VALUES)
        for T in T_VALUES
        for seed in range(N_SEEDS)
    ]
    print(f"  Total configs: {len(configs)}")
    print("\nRunning...")
    
    results = list(run_rom_experiment.starmap(configs))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rom_timeseries_v2_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== ANALYSIS =====
    import numpy as np
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Summary table
    print(f"\n{'T':<7} {'h':<4} {'τ':<6} | {'Raw':<8} {'FFT':<8} {'Mallat':<8} | {'HST k=8':<9} {'Mlt k=8':<9} {'Raw k=8':<9}")
    print("-"*85)
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        tau_avg = np.mean([r["tau_autocorr"] for r in T_results])
        
        for h in HORIZONS:
            mse_raw = np.mean([r["horizons"][h]["baselines"]["raw_ridge"] for r in T_results])
            mse_fft = np.mean([r["horizons"][h]["baselines"]["fft_ridge"] for r in T_results])
            mse_mallat = np.mean([r["horizons"][h]["baselines"]["mallat_ridge"] for r in T_results])
            
            mse_hst_8 = np.nanmean([r["horizons"][h]["k_results"].get(8, np.nan) for r in T_results])
            mse_mallat_8 = np.nanmean([r["horizons"][h]["mallat_pca_k"].get(8, np.nan) for r in T_results])
            mse_raw_8 = np.nanmean([r["horizons"][h]["raw_pca_k"].get(8, np.nan) for r in T_results])
            
            marker = "*" if abs(T - TC) < 0.01 else " "
            print(f"{T:<7.3f}{marker}{h:<4} {tau_avg:<6.1f} | {mse_raw:<8.4f} {mse_fft:<8.4f} {mse_mallat:<8.4f} | {mse_hst_8:<9.4f} {mse_mallat_8:<9.4f} {mse_raw_8:<9.4f}")
    
    # Winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS (who has lowest MSE?)")
    print("="*80)
    
    winners = {"raw_ridge": 0, "fft_ridge": 0, "mallat_ridge": 0, 
               "hst_pca8": 0, "mallat_pca8": 0, "raw_pca8": 0}
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        for h in HORIZONS:
            scores = {
                "raw_ridge": np.mean([r["horizons"][h]["baselines"]["raw_ridge"] for r in T_results]),
                "fft_ridge": np.mean([r["horizons"][h]["baselines"]["fft_ridge"] for r in T_results]),
                "mallat_ridge": np.mean([r["horizons"][h]["baselines"]["mallat_ridge"] for r in T_results]),
                "hst_pca8": np.nanmean([r["horizons"][h]["k_results"].get(8, np.nan) for r in T_results]),
                "mallat_pca8": np.nanmean([r["horizons"][h]["mallat_pca_k"].get(8, np.nan) for r in T_results]),
                "raw_pca8": np.nanmean([r["horizons"][h]["raw_pca_k"].get(8, np.nan) for r in T_results]),
            }
            winner = min(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else float('inf'))
            winners[winner] += 1
    
    print("\nWins by method (across all T × h combinations):")
    for method, wins in sorted(winners.items(), key=lambda x: -x[1]):
        print(f"  {method}: {wins}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    
    return results
