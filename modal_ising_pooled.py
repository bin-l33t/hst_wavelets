"""
HST Ising Experiment - Field vs Pooled Compactness
Addresses ChatGPT's critique: measure both raw field energy and pooled scattering features

Two metrics:
- d_eff_field: energy over full spatial signal (current)  
- d_eff_pooled: energy over lowpass-pooled |W| (Mallat-style)

Two encodings:
- magnitude: x = row + 3.0
- phase: x = 3.0 * exp(i * pi * row / 2)  [constant magnitude, spin in phase]
"""

import modal

app = modal.App("hst-ising-pooled")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
)


@app.function(image=image, timeout=900)
def run_config(L: int, T: float, seed: int, encoding: str = "magnitude", 
               n_snapshots: int = 4, n_rows: int = 8) -> dict:
    """
    Run HST with both field and pooled compactness metrics.
    """
    import numpy as np
    import torch
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # ===== ISING MODEL =====
    rng = np.random.default_rng(seed)
    spins = np.ones((L, L), dtype=int) if T < TC else rng.choice([-1, 1], size=(L, L))
    
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
    for _ in range(500):
        wolff_step()
    
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
        
        # Father wavelet (lowpass)
        xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
        phi = torch.exp(-w**2 / (2 * xi_min**2))
        filters.append(phi)
        lp_sum_sq += phi ** 2
        
        lp_sum = torch.sqrt(torch.clamp(lp_sum_sq, min=1e-20))
        normalized = [(f / lp_sum).to(torch.complex128) for f in filters]
        
        # Return filters and the lowpass separately
        return normalized[:-1], normalized[-1], 2 * num_mothers  # mothers, phi, n_mothers
    
    def hst_forward(x, mothers, phi, n_mothers, eps=1e-8):
        """Returns both raw coefficients and pooled features."""
        
        def lift(z):
            r = torch.abs(z)
            return z * torch.sqrt(r**2 + eps**2) / torch.clamp(r, min=1e-300)
        
        def R(z):
            return -torch.angle(z) + 1j * torch.log(torch.abs(z))
        
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        def lowpass_pool(z):
            """Apply lowpass and return global average (scalar feature)."""
            z_lp = conv(z, phi)
            return torch.mean(torch.abs(z_lp)**2).item()  # Scalar energy
        
        paths_field = {}  # Full spatial coefficients
        paths_pooled = {}  # Lowpass-pooled scalar energies
        
        x_lifted = lift(x)
        coeffs = [conv(x_lifted, f) for f in mothers]
        
        # Order 0: just the lowpass of input
        lp_input = conv(x_lifted, phi)
        paths_field[()] = lp_input
        paths_pooled[()] = torch.mean(torch.abs(lp_input)**2).item()
        
        # Order 1
        W1_dict = {}
        for j1 in range(n_mothers):
            U1 = coeffs[j1]
            U1_lifted = lift(U1)
            W1 = R(U1_lifted)
            paths_field[(j1,)] = W1
            # Pooled: lowpass(|W1|) then average
            paths_pooled[(j1,)] = lowpass_pool(torch.abs(W1))
            W1_dict[j1] = W1
        
        # Order 2
        for j1 in range(n_mothers):
            W1 = W1_dict[j1]
            coeffs_W1 = [conv(W1, f) for f in mothers]
            for j2 in range(j1 + 1, n_mothers):
                U2 = coeffs_W1[j2]
                U2_lifted = lift(U2)
                W2 = R(U2_lifted)
                paths_field[(j1, j2)] = W2
                paths_pooled[(j1, j2)] = lowpass_pool(torch.abs(W2))
        
        return paths_field, paths_pooled
    
    mothers, phi, n_mothers = make_filterbank(L)
    n_paths = {0: 1, 1: n_mothers, 2: n_mothers * (n_mothers - 1) // 2}
    
    # Energy threshold for reliable d_eff (Step 2: energy floor guard)
    ENERGY_FLOOR = 1e-10
    
    # ===== COLLECT DATA =====
    metrics = {
        'field': {m: {'d_eff': [], 'energy': []} for m in [0, 1, 2]},
        'pooled': {m: {'d_eff': [], 'energy': []} for m in [0, 1, 2]},
    }
    mags = []
    skipped_low_energy = {0: 0, 1: 0, 2: 0}
    
    for snap in range(n_snapshots):
        if snap > 0:
            for _ in range(100):
                wolff_step()
        
        mags.append(abs(np.mean(spins)))
        row_indices = rng.choice(L, size=min(n_rows, L), replace=False)
        
        for row_idx in row_indices:
            row = spins[row_idx, :].astype(float)
            
            # Apply encoding
            if encoding == "magnitude":
                x = torch.tensor(row + 3.0 + 0j, dtype=torch.complex128)
            elif encoding == "phase":
                # Constant magnitude, spin encoded in phase
                # -1 -> phase=0, +1 -> phase=pi
                phase = np.pi * (row + 1) / 2  # Maps -1->0, +1->pi
                x = torch.tensor(3.0 * np.exp(1j * phase), dtype=torch.complex128)
            else:
                x = torch.tensor(row + 3.0 + 0j, dtype=torch.complex128)
            
            paths_field, paths_pooled = hst_forward(x, mothers, phi, n_mothers)
            
            # Compute d_eff for both field and pooled
            for mode, paths in [('field', paths_field), ('pooled', paths_pooled)]:
                for m in [0, 1, 2]:
                    paths_m = {p: c for p, c in paths.items() if len(p) == m}
                    if not paths_m:
                        continue
                    
                    if mode == 'field':
                        # Field: energy is sum |coef|^2 over spatial dimension
                        energies = np.array([
                            torch.sum(torch.abs(c)**2).item() for c in paths_m.values()
                        ])
                    else:
                        # Pooled: energies are already scalars
                        energies = np.array(list(paths_m.values()))
                    
                    total = np.sum(energies)
                    
                    # Energy floor guard
                    if total < ENERGY_FLOOR:
                        skipped_low_energy[m] += 1
                        continue
                    
                    metrics[mode][m]['energy'].append(total)
                    
                    if np.sum(energies**2) > 0:
                        d_eff = (total**2) / np.sum(energies**2)
                        metrics[mode][m]['d_eff'].append(d_eff)
    
    # ===== AGGREGATE =====
    result = {
        "L": L,
        "T": T,
        "seed": seed,
        "encoding": encoding,
        "n_samples": n_snapshots * min(n_rows, L),
        "mean_mag": float(np.mean(mags)),
    }
    
    for mode in ['field', 'pooled']:
        for m in [0, 1, 2]:
            prefix = f"{mode}_{m}"
            d_eff_list = metrics[mode][m]['d_eff']
            energy_list = metrics[mode][m]['energy']
            
            result[f"d_eff_{prefix}"] = float(np.mean(d_eff_list)) if d_eff_list else np.nan
            result[f"d_eff_{prefix}_std"] = float(np.std(d_eff_list)) if d_eff_list else np.nan
            result[f"d_eff_{prefix}_norm"] = float(np.mean(d_eff_list)) / n_paths[m] if d_eff_list else np.nan
            result[f"energy_{prefix}"] = float(np.mean(energy_list)) if energy_list else np.nan
            result[f"n_valid_{prefix}"] = len(d_eff_list)
    
    result["n_paths_1"] = n_paths[1]
    result["n_paths_2"] = n_paths[2]
    result["skipped_low_energy"] = dict(skipped_low_energy)
    
    return result


@app.local_entrypoint()
def main():
    import math
    import json
    from datetime import datetime
    
    print("="*70)
    print("HST ISING: FIELD vs POOLED COMPACTNESS")
    print("="*70)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Configuration
    L_VALUES = [32, 64]
    T_VALUES = [1.5, 1.8, 2.0, 2.1, 2.2, TC, 2.3, 2.4, 2.5, 3.0, 3.5]
    N_SEEDS = 10
    ENCODINGS = ["magnitude", "phase"]
    
    total = len(L_VALUES) * len(T_VALUES) * N_SEEDS * len(ENCODINGS)
    print(f"\nConfiguration:")
    print(f"  L: {L_VALUES}")
    print(f"  T: {len(T_VALUES)} points")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Encodings: {ENCODINGS}")
    print(f"  Total configs: {total}")
    
    print("\nRunning...")
    
    configs = [
        (L, T, seed, enc)
        for enc in ENCODINGS
        for L in L_VALUES
        for T in T_VALUES
        for seed in range(N_SEEDS)
    ]
    
    results = list(run_config.starmap(configs))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"ising_pooled_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: ising_pooled_{timestamp}.json")
    
    # ===== ANALYSIS =====
    import numpy as np
    
    for enc in ENCODINGS:
        print("\n" + "="*70)
        print(f"ENCODING: {enc.upper()}")
        print("="*70)
        
        enc_results = [r for r in results if r["encoding"] == enc]
        
        for L in L_VALUES:
            L_results = [r for r in enc_results if r["L"] == L]
            if not L_results:
                continue
                
            n1 = L_results[0]["n_paths_1"]
            n2 = L_results[0]["n_paths_2"]
            
            print(f"\n{'='*70}")
            print(f"L = {L}  (n_paths: order1={n1}, order2={n2})")
            print(f"{'='*70}")
            
            # Header
            print(f"{'T':<6} {'|m|':<6} | {'FIELD d1/n':<11} {'FIELD d2/n':<11} | {'POOLED d1/n':<11} {'POOLED d2/n':<11}")
            print("-"*70)
            
            for T in sorted(set(r["T"] for r in L_results)):
                T_res = [r for r in L_results if abs(r["T"] - T) < 0.001]
                
                mag = np.mean([r["mean_mag"] for r in T_res])
                
                # Field metrics
                f1 = np.nanmean([r["d_eff_field_1_norm"] for r in T_res])
                f2 = np.nanmean([r["d_eff_field_2_norm"] for r in T_res])
                
                # Pooled metrics
                p1 = np.nanmean([r["d_eff_pooled_1_norm"] for r in T_res])
                p2 = np.nanmean([r["d_eff_pooled_2_norm"] for r in T_res])
                
                marker = " <--Tc" if abs(T - TC) < 0.01 else ""
                print(f"{T:<6.3f} {mag:<6.3f} | {f1:<11.4f} {f2:<11.4f} | {p1:<11.4f} {p2:<11.4f}{marker}")
            
            # Find minima
            print(f"\nMinima analysis (L={L}, {enc}):")
            T_data = {}
            for T in sorted(set(r["T"] for r in L_results)):
                T_res = [r for r in L_results if abs(r["T"] - T) < 0.001]
                T_data[T] = {
                    "field_1": np.nanmean([r["d_eff_field_1_norm"] for r in T_res]),
                    "field_2": np.nanmean([r["d_eff_field_2_norm"] for r in T_res]),
                    "pooled_1": np.nanmean([r["d_eff_pooled_1_norm"] for r in T_res]),
                    "pooled_2": np.nanmean([r["d_eff_pooled_2_norm"] for r in T_res]),
                }
            
            for metric in ["field_1", "field_2", "pooled_1", "pooled_2"]:
                valid = {T: v[metric] for T, v in T_data.items() if not np.isnan(v[metric])}
                if valid:
                    min_T = min(valid.keys(), key=lambda T: valid[T])
                    near_Tc = "✓ near Tc" if abs(min_T - TC) < 0.3 else ""
                    print(f"  {metric}: min at T={min_T:.3f} {near_Tc}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results
