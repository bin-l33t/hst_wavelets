"""
ROM Experiment: Time Series Forecasting (reads from volume)

Reads pre-generated Ising datasets from Modal Volume.
Runs forecasting with multiple methods.
"""

import modal

app = modal.App("hst-rom-forecast")

volume = modal.Volume.from_name("ising-datasets", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
)


@app.function(image=image, timeout=1200, cpu=2, volumes={"/data": volume})
def run_forecasting(
    filename: str,
    window_size: int = 128,
    horizons: list = [1, 4, 16, 64],
    k_values: list = [2, 4, 8, 16],
    max_windows: int = 5000,  # Cap number of windows
) -> dict:
    """
    Run forecasting experiment on pre-generated dataset.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    EPS = 1e-8
    
    # Load data
    data = np.load(f"/data/{filename}")
    m_t = data["m_t"]
    L = int(data["L"])
    T = float(data["T"])
    seed = int(data["seed"])
    
    # Normalize
    m_mean, m_std = m_t.mean(), m_t.std()
    if m_std < 1e-10:
        m_std = 1.0
    m_t_norm = (m_t - m_mean) / m_std
    
    # ===== SIMPLE FILTERBANK =====
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
    
    # ===== FEATURE EXTRACTORS =====
    def hst_features(x, filters, n_mothers):
        """HST features with fixed R(z)."""
        def lift(z):
            r = torch.abs(z)
            return z * torch.sqrt(r**2 + EPS**2) / torch.clamp(r, min=1e-300)
        
        def R(z):
            r = torch.clamp(torch.abs(z), min=EPS)
            return -torch.angle(z) + 1j * torch.log(r)
        
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        phi = filters[-1]
        mothers = filters[:-1]
        
        features = []
        x_lifted = lift(x)
        coeffs = [conv(x_lifted, f) for f in mothers]
        
        # Order 0
        lp = conv(x_lifted, phi)
        features.append(torch.mean(torch.abs(lp)**2).item())
        
        # Order 1
        for j1 in range(n_mothers):
            U1 = coeffs[j1]
            W1 = R(lift(U1))
            features.extend([
                torch.mean(W1.real).item(),
                torch.mean(W1.imag).item(),
                torch.mean(torch.abs(W1)**2).item(),
            ])
        
        return np.array(features)
    
    def mallat_features(x, filters, n_mothers):
        """Standard Mallat scattering."""
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        phi = filters[-1]
        mothers = filters[:-1]
        
        features = []
        
        # Order 0
        S0 = conv(x, phi)
        features.append(torch.mean(torch.abs(S0)**2).item())
        
        # Order 1
        for j1 in range(n_mothers):
            U1 = conv(x, mothers[j1])
            S1 = conv(torch.abs(U1).to(torch.complex128), phi)
            features.append(torch.mean(torch.abs(S1)**2).item())
        
        return np.array(features)
    
    filters, n_mothers = make_filterbank(window_size)
    
    # ===== BUILD WINDOWS =====
    max_h = max(horizons)
    n_total = len(m_t_norm) - window_size - max_h
    
    # Subsample if too many
    step = max(1, n_total // max_windows)
    indices = list(range(0, n_total, step))[:max_windows]
    
    # Block split
    n_windows = len(indices)
    train_end = int(0.6 * n_windows)
    test_start = int(0.8 * n_windows)
    
    train_idx = indices[:train_end]
    test_idx = indices[test_start:]
    
    # Extract features once
    print(f"Extracting features for {len(indices)} windows...")
    
    X_raw_all = []
    X_hst_all = []
    X_mallat_all = []
    X_fft_all = []
    
    for i, t in enumerate(indices):
        window = m_t_norm[t:t+window_size]
        X_raw_all.append(window)
        
        x_torch = torch.tensor(window + 3.0 + 0j, dtype=torch.complex128)
        
        hst_feat = hst_features(x_torch, filters, n_mothers)
        hst_feat = np.nan_to_num(hst_feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_hst_all.append(hst_feat)
        
        mallat_feat = mallat_features(x_torch, filters, n_mothers)
        mallat_feat = np.nan_to_num(mallat_feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_mallat_all.append(mallat_feat)
        
        fft_mag = np.abs(np.fft.fft(window))[:window_size//2]
        X_fft_all.append(fft_mag)
        
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(indices)}")
    
    X_raw_all = np.array(X_raw_all)
    X_hst_all = np.array(X_hst_all)
    X_mallat_all = np.array(X_mallat_all)
    X_fft_all = np.array(X_fft_all)
    
    # ===== RUN FORECASTING =====
    results = {
        "filename": filename,
        "L": L,
        "T": T,
        "seed": seed,
        "n_windows": n_windows,
        "horizons": {},
    }
    
    for h in horizons:
        # Get targets
        y_all = np.array([m_t_norm[indices[i] + window_size + h - 1] for i in range(len(indices))])
        
        # Split
        X_raw_train = X_raw_all[:train_end]
        X_raw_test = X_raw_all[test_start:]
        X_hst_train = X_hst_all[:train_end]
        X_hst_test = X_hst_all[test_start:]
        X_mallat_train = X_mallat_all[:train_end]
        X_mallat_test = X_mallat_all[test_start:]
        X_fft_train = X_fft_all[:train_end]
        X_fft_test = X_fft_all[test_start:]
        y_train = y_all[:train_end]
        y_test = y_all[test_start:]
        
        h_results = {"baselines": {}, "hst_pca": {}, "mallat_pca": {}, "raw_pca": {}}
        
        # Baseline: Raw ridge
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw_train)
        X_te = scaler.transform(X_raw_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_train)
        h_results["baselines"]["raw_ridge"] = float(np.mean((y_test - ridge.predict(X_te))**2))
        
        # Baseline: FFT ridge
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_fft_train)
        X_te = scaler.transform(X_fft_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_train)
        h_results["baselines"]["fft_ridge"] = float(np.mean((y_test - ridge.predict(X_te))**2))
        
        # Baseline: Mallat ridge (full)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_mallat_train)
        X_te = scaler.transform(X_mallat_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_train)
        h_results["baselines"]["mallat_ridge"] = float(np.mean((y_test - ridge.predict(X_te))**2))
        
        # HST + PCA(k)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_hst_train)
        X_te = scaler.transform(X_hst_test)
        n_comp = min(X_tr.shape[1], X_tr.shape[0] - 1, max(k_values))
        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            for k in k_values:
                if k <= n_comp:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_tr_pca[:, :k], y_train)
                    h_results["hst_pca"][k] = float(np.mean((y_test - ridge.predict(X_te_pca[:, :k]))**2))
        
        # Mallat + PCA(k)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_mallat_train)
        X_te = scaler.transform(X_mallat_test)
        n_comp = min(X_tr.shape[1], X_tr.shape[0] - 1, max(k_values))
        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            for k in k_values:
                if k <= n_comp:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_tr_pca[:, :k], y_train)
                    h_results["mallat_pca"][k] = float(np.mean((y_test - ridge.predict(X_te_pca[:, :k]))**2))
        
        # Raw + PCA(k)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw_train)
        X_te = scaler.transform(X_raw_test)
        n_comp = min(X_tr.shape[1], X_tr.shape[0] - 1, max(k_values))
        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            for k in k_values:
                if k <= n_comp:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_tr_pca[:, :k], y_train)
                    h_results["raw_pca"][k] = float(np.mean((y_test - ridge.predict(X_te_pca[:, :k]))**2))
        
        results["horizons"][h] = h_results
    
    return results


@app.function(image=image, timeout=60, volumes={"/data": volume})
def list_datasets() -> list:
    """List available datasets."""
    import os
    return sorted([f for f in os.listdir("/data") if f.endswith(".npz")])


@app.local_entrypoint()
def main():
    import json
    from datetime import datetime
    import math
    
    print("="*70)
    print("ROM FORECASTING EXPERIMENT")
    print("="*70)
    
    # Get available datasets
    datasets = list_datasets.remote()
    print(f"\nFound {len(datasets)} datasets")
    
    if not datasets:
        print("No datasets found! Run modal_ising_datagen.py first.")
        return
    
    for d in datasets[:5]:
        print(f"  {d}")
    if len(datasets) > 5:
        print(f"  ... and {len(datasets) - 5} more")
    
    print("\nRunning forecasting experiments...")
    
    # Run on all datasets
    results = list(run_forecasting.map(datasets))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rom_forecast_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {fname}")
    
    # ===== ANALYSIS =====
    import numpy as np
    
    TC = 2 / math.log(1 + math.sqrt(2))
    T_VALUES = sorted(set(r["T"] for r in results))
    HORIZONS = [1, 4, 16, 64]
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'T':<7} {'h':<4} | {'Raw':<8} {'FFT':<8} {'Mallat':<8} | {'HST k=8':<9} {'Mlt k=8':<9} {'Best':<10}")
    print("-"*75)
    
    for T in T_VALUES:
        T_results = [r for r in results if abs(r["T"] - T) < 0.01]
        
        for h in HORIZONS:
            raw = np.mean([r["horizons"][str(h)]["baselines"]["raw_ridge"] for r in T_results])
            fft = np.mean([r["horizons"][str(h)]["baselines"]["fft_ridge"] for r in T_results])
            mallat = np.mean([r["horizons"][str(h)]["baselines"]["mallat_ridge"] for r in T_results])
            
            hst_8 = np.nanmean([r["horizons"][str(h)]["hst_pca"].get("8", np.nan) for r in T_results])
            mlt_8 = np.nanmean([r["horizons"][str(h)]["mallat_pca"].get("8", np.nan) for r in T_results])
            
            scores = {"raw": raw, "fft": fft, "mallat": mallat, "hst_k8": hst_8, "mlt_k8": mlt_8}
            best = min(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else float('inf'))
            
            marker = "*" if abs(T - TC) < 0.01 else " "
            print(f"{T:<7.3f}{marker}{h:<4} | {raw:<8.4f} {fft:<8.4f} {mallat:<8.4f} | {hst_8:<9.4f} {mlt_8:<9.4f} {best:<10}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
