"""
ROM Experiment 1: Magnetization Time Series Forecasting

Tests whether HST features provide a low-dimensional predictive state (ROM)
for Ising magnetization dynamics.

Key design choices (per ChatGPT):
- Blocked time splits (no leakage from overlapping windows)
- Multiple horizons h={1,4,16,64} to detect long memory effects
- Strong baselines: AR/ridge on raw windows, FFT magnitude
- Primary metric: test MSE vs PCA(k) across T and h
"""

import modal

app = modal.App("hst-rom-timeseries")

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
    k_values: list = [2, 4, 8, 16, 32, 64],
) -> dict:
    """
    Run ROM forecasting experiment for one (L, T, seed) configuration.
    
    Returns MSE vs k for HST features and baselines across horizons.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
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
        m_std = 1.0  # Avoid division by zero for very ordered states
    m_t_norm = (m_t - m_mean) / m_std
    
    # Compute autocorrelation time (for physics check)
    def autocorr_time(x, max_lag=500):
        n = len(x)
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-10:
            return 1.0
        acf = np.correlate(x, x, mode='full')[n-1:n-1+max_lag] / (var * n)
        # Integrated autocorrelation time
        tau = 1 + 2 * np.sum(acf[1:])
        return max(1.0, tau)
    
    tau_autocorr = autocorr_time(m_t_norm)
    
    # ===== HST SETUP =====
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
    
    def hst_features(x, filters, n_mothers, eps=1e-8):
        """Extract pooled scattering features from window."""
        def lift(z):
            r = torch.abs(z)
            return z * torch.sqrt(r**2 + eps**2) / torch.clamp(r, min=1e-300)
        
        def R(z):
            return -torch.angle(z) + 1j * torch.log(torch.abs(z))
        
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        phi = filters[-1]
        mothers = filters[:-1]
        
        features = []
        x_lifted = lift(x)
        coeffs = [conv(x_lifted, f) for f in mothers]
        
        # Order 0: lowpass energy
        lp = conv(x_lifted, phi)
        features.append(torch.mean(torch.abs(lp)**2).item())
        
        # Order 1: pooled |W1| energies
        W1_list = []
        for j1 in range(n_mothers):
            U1 = coeffs[j1]
            W1 = R(lift(U1))
            W1_lp = conv(torch.abs(W1), phi)
            features.append(torch.mean(torch.abs(W1_lp)**2).item())
            W1_list.append(W1)
        
        # Order 2: pooled |W2| energies (subset to keep feature dim manageable)
        for j1 in range(min(n_mothers, 6)):  # Limit order-2 paths
            coeffs_W1 = [conv(W1_list[j1], f) for f in mothers]
            for j2 in range(j1 + 1, min(n_mothers, 6)):
                U2 = coeffs_W1[j2]
                W2 = R(lift(U2))
                W2_lp = conv(torch.abs(W2), phi)
                features.append(torch.mean(torch.abs(W2_lp)**2).item())
        
        return np.array(features)
    
    filters, n_mothers = make_filterbank(window_size)
    
    # ===== CREATE WINDOWS WITH BLOCKED SPLITS =====
    max_h = max(horizons)
    n_windows = n_steps - window_size - max_h
    
    # Block splits: train [0, 60%), val [60%, 80%), test [80%, 100%)
    train_end = int(0.6 * n_windows)
    val_end = int(0.8 * n_windows)
    
    def get_windows_and_targets(start_idx, end_idx, h):
        """Extract windows and targets for a contiguous block."""
        X_raw = []  # Raw window values
        X_hst = []  # HST features
        X_fft = []  # FFT magnitude features
        y = []
        
        for t in range(start_idx, end_idx):
            window = m_t_norm[t:t+window_size]
            target = m_t_norm[t+window_size+h-1]  # Predict h steps ahead
            
            # Raw window (for AR baseline)
            X_raw.append(window)
            
            # HST features
            x_torch = torch.tensor(window + 3.0 + 0j, dtype=torch.complex128)
            X_hst.append(hst_features(x_torch, filters, n_mothers))
            
            # FFT magnitude features
            fft_mag = np.abs(np.fft.fft(window))[:window_size//2]
            X_fft.append(fft_mag)
            
            y.append(target)
        
        return np.array(X_raw), np.array(X_hst), np.array(X_fft), np.array(y)
    
    # ===== RUN FORECASTING FOR EACH HORIZON =====
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
        # Get data splits
        X_raw_train, X_hst_train, X_fft_train, y_train = get_windows_and_targets(0, train_end, h)
        X_raw_val, X_hst_val, X_fft_val, y_val = get_windows_and_targets(train_end, val_end, h)
        X_raw_test, X_hst_test, X_fft_test, y_test = get_windows_and_targets(val_end, n_windows, h)
        
        h_results = {"k_results": {}, "baselines": {}}
        
        # ===== BASELINE 1: AR on raw window (ridge regression) =====
        scaler_raw = StandardScaler()
        X_raw_train_s = scaler_raw.fit_transform(X_raw_train)
        X_raw_test_s = scaler_raw.transform(X_raw_test)
        
        ridge_raw = Ridge(alpha=1.0)
        ridge_raw.fit(X_raw_train_s, y_train)
        y_pred_raw = ridge_raw.predict(X_raw_test_s)
        mse_raw = np.mean((y_test - y_pred_raw)**2)
        h_results["baselines"]["raw_ridge"] = float(mse_raw)
        
        # ===== BASELINE 2: FFT magnitude features =====
        scaler_fft = StandardScaler()
        X_fft_train_s = scaler_fft.fit_transform(X_fft_train)
        X_fft_test_s = scaler_fft.transform(X_fft_test)
        
        ridge_fft = Ridge(alpha=1.0)
        ridge_fft.fit(X_fft_train_s, y_train)
        y_pred_fft = ridge_fft.predict(X_fft_test_s)
        mse_fft = np.mean((y_test - y_pred_fft)**2)
        h_results["baselines"]["fft_ridge"] = float(mse_fft)
        
        # ===== HST + PCA(k) + Ridge =====
        scaler_hst = StandardScaler()
        X_hst_train_s = scaler_hst.fit_transform(X_hst_train)
        X_hst_test_s = scaler_hst.transform(X_hst_test)
        
        # Full PCA on training data
        n_components = min(X_hst_train_s.shape[1], X_hst_train_s.shape[0], max(k_values))
        pca = PCA(n_components=n_components)
        X_hst_train_pca = pca.fit_transform(X_hst_train_s)
        X_hst_test_pca = pca.transform(X_hst_test_s)
        
        for k in k_values:
            if k > n_components:
                continue
            
            # Use first k components
            X_train_k = X_hst_train_pca[:, :k]
            X_test_k = X_hst_test_pca[:, :k]
            
            ridge_hst = Ridge(alpha=1.0)
            ridge_hst.fit(X_train_k, y_train)
            y_pred_hst = ridge_hst.predict(X_test_k)
            mse_hst = np.mean((y_test - y_pred_hst)**2)
            
            h_results["k_results"][k] = float(mse_hst)
        
        # ===== RAW + PCA(k) baseline (to isolate HST contribution) =====
        n_comp_raw = min(X_raw_train_s.shape[1], X_raw_train_s.shape[0], max(k_values))
        pca_raw = PCA(n_components=n_comp_raw)
        X_raw_train_pca = pca_raw.fit_transform(X_raw_train_s)
        X_raw_test_pca = pca_raw.transform(X_raw_test_s)
        
        h_results["raw_pca_k"] = {}
        for k in k_values:
            if k > n_comp_raw:
                continue
            X_train_k = X_raw_train_pca[:, :k]
            X_test_k = X_raw_test_pca[:, :k]
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_k, y_train)
            y_pred = ridge.predict(X_test_k)
            h_results["raw_pca_k"][k] = float(np.mean((y_test - y_pred)**2))
        
        results["horizons"][h] = h_results
    
    return results


@app.local_entrypoint()
def main():
    import math
    import json
    from datetime import datetime
    
    print("="*70)
    print("ROM EXPERIMENT 1: MAGNETIZATION TIME SERIES FORECASTING")
    print("="*70)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Configuration
    L = 32
    T_VALUES = [1.5, 2.0, TC, 2.5, 3.0]  # Ordered, near-Tc, critical, disordered
    N_SEEDS = 5
    HORIZONS = [1, 4, 16, 64]
    K_VALUES = [2, 4, 8, 16, 32]
    
    print(f"\nConfiguration:")
    print(f"  L: {L}")
    print(f"  T: {T_VALUES}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  k values: {K_VALUES}")
    
    configs = [
        (L, T, seed, 20000, 256, HORIZONS, K_VALUES)
        for T in T_VALUES
        for seed in range(N_SEEDS)
    ]
    
    print(f"  Total configs: {len(configs)}")
    print("\nRunning...")
    
    results = list(run_rom_experiment.starmap(configs))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"rom_timeseries_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: rom_timeseries_{timestamp}.json")
    
    # ===== ANALYSIS =====
    import numpy as np
    
    print("\n" + "="*70)
    print("RESULTS: TEST MSE vs k (averaged over seeds)")
    print("="*70)
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        tau_avg = np.mean([r["tau_autocorr"] for r in T_results])
        
        marker = " <-- Tc" if abs(T - TC) < 0.01 else ""
        print(f"\n{'='*70}")
        print(f"T = {T:.3f}{marker}  |  τ_autocorr = {tau_avg:.1f}")
        print(f"{'='*70}")
        
        for h in HORIZONS:
            print(f"\n  Horizon h={h}:")
            
            # Baselines
            mse_raw = np.mean([r["horizons"][h]["baselines"]["raw_ridge"] for r in T_results])
            mse_fft = np.mean([r["horizons"][h]["baselines"]["fft_ridge"] for r in T_results])
            print(f"    Baselines: raw_ridge={mse_raw:.4f}, fft_ridge={mse_fft:.4f}")
            
            # HST+PCA(k)
            print(f"    HST+PCA(k):  ", end="")
            for k in K_VALUES:
                mses = [r["horizons"][h]["k_results"].get(k, np.nan) for r in T_results]
                mse_avg = np.nanmean(mses)
                print(f"k={k}:{mse_avg:.4f}  ", end="")
            print()
            
            # Raw+PCA(k) for comparison
            print(f"    Raw+PCA(k):  ", end="")
            for k in K_VALUES:
                mses = [r["horizons"][h]["raw_pca_k"].get(k, np.nan) for r in T_results]
                mse_avg = np.nanmean(mses)
                print(f"k={k}:{mse_avg:.4f}  ", end="")
            print()
    
    # ===== SUMMARY: Does HST beat baselines? =====
    print("\n" + "="*70)
    print("SUMMARY: HST vs BASELINES")
    print("="*70)
    
    print(f"\n{'T':<8} {'h':<4} {'Raw Ridge':<12} {'FFT Ridge':<12} {'HST k=8':<12} {'Raw k=8':<12} {'HST wins?':<10}")
    print("-"*70)
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        for h in HORIZONS:
            mse_raw = np.mean([r["horizons"][h]["baselines"]["raw_ridge"] for r in T_results])
            mse_fft = np.mean([r["horizons"][h]["baselines"]["fft_ridge"] for r in T_results])
            mse_hst_8 = np.nanmean([r["horizons"][h]["k_results"].get(8, np.nan) for r in T_results])
            mse_raw_8 = np.nanmean([r["horizons"][h]["raw_pca_k"].get(8, np.nan) for r in T_results])
            
            best_baseline = min(mse_raw, mse_fft, mse_raw_8)
            wins = "✓" if mse_hst_8 < best_baseline else ""
            
            marker = "*" if abs(T - TC) < 0.01 else " "
            print(f"{T:<8.3f}{marker}{h:<4} {mse_raw:<12.4f} {mse_fft:<12.4f} {mse_hst_8:<12.4f} {mse_raw_8:<12.4f} {wins:<10}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
