"""
HST Ising Experiment - Full Scale
Tests whether critical regime shows minimum d_eff/n (maximum compactness)
"""

import modal

app = modal.App("hst-ising-full")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "torch>=2.0",
)


@app.function(image=image, timeout=900)
def run_config(L: int, T: float, seed: int, n_snapshots: int = 4, n_rows: int = 8) -> dict:
    """
    Run HST on Ising config with proper averaging.
    
    Args:
        L: Lattice size
        T: Temperature  
        seed: Random seed
        n_snapshots: Number of independent snapshots after equilibration
        n_rows: Rows to sample per snapshot
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
        
        xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
        phi = torch.exp(-w**2 / (2 * xi_min**2))
        filters.append(phi)
        lp_sum_sq += phi ** 2
        
        lp_sum = torch.sqrt(torch.clamp(lp_sum_sq, min=1e-20))
        return [(f / lp_sum).to(torch.complex128) for f in filters], 2 * num_mothers
    
    def hst_forward(x, filters, n_mothers, eps=1e-8):
        def lift(z):
            r = torch.abs(z)
            return z * torch.sqrt(r**2 + eps**2) / torch.clamp(r, min=1e-300)
        
        def R(z):
            return -torch.angle(z) + 1j * torch.log(torch.abs(z))
        
        def conv(x, f):
            return torch.fft.ifft(torch.fft.fft(x) * f)
        
        paths = {}
        x_lifted = lift(x)
        coeffs = [conv(x_lifted, f) for f in filters]
        paths[()] = coeffs[-1]
        
        for j1 in range(n_mothers):
            W1 = R(lift(coeffs[j1]))
            paths[(j1,)] = W1
            coeffs_W1 = [conv(W1, f) for f in filters]
            for j2 in range(j1 + 1, n_mothers):
                paths[(j1, j2)] = R(lift(coeffs_W1[j2]))
        
        return paths
    
    filters, n_mothers = make_filterbank(L)
    
    # Count paths per order
    n_paths = {0: 1, 1: n_mothers, 2: n_mothers * (n_mothers - 1) // 2}
    
    # ===== COLLECT DATA =====
    all_d_eff = {0: [], 1: [], 2: []}
    all_d_eff_norm = {0: [], 1: [], 2: []}
    all_energy = {0: [], 1: [], 2: []}
    mags = []
    
    for snap in range(n_snapshots):
        if snap > 0:
            for _ in range(100):  # Decorrelate
                wolff_step()
        
        mags.append(abs(np.mean(spins)))
        
        # Sample rows
        row_indices = rng.choice(L, size=min(n_rows, L), replace=False)
        
        for row_idx in row_indices:
            row = spins[row_idx, :].astype(float)
            x = torch.tensor(row + 3.0 + 0j, dtype=torch.complex128)
            paths = hst_forward(x, filters, n_mothers)
            
            for m in [0, 1, 2]:
                paths_m = {p: c for p, c in paths.items() if len(p) == m}
                if paths_m:
                    energies = np.array([torch.sum(torch.abs(c)**2).item() for c in paths_m.values()])
                    total = np.sum(energies)
                    all_energy[m].append(total)
                    
                    if np.sum(energies**2) > 0:
                        d_eff = (total**2) / np.sum(energies**2)
                        all_d_eff[m].append(d_eff)
                        all_d_eff_norm[m].append(d_eff / n_paths[m])
    
    # ===== AGGREGATE =====
    result = {
        "L": L,
        "T": T,
        "seed": seed,
        "n_samples": n_snapshots * min(n_rows, L),
        "mean_mag": float(np.mean(mags)),
        "std_mag": float(np.std(mags)),
    }
    
    for m in [0, 1, 2]:
        result[f"d_eff_{m}"] = float(np.mean(all_d_eff[m])) if all_d_eff[m] else 0
        result[f"d_eff_{m}_std"] = float(np.std(all_d_eff[m])) if all_d_eff[m] else 0
        result[f"d_eff_norm_{m}"] = float(np.mean(all_d_eff_norm[m])) if all_d_eff_norm[m] else 0
        result[f"energy_{m}"] = float(np.mean(all_energy[m])) if all_energy[m] else 0
        result[f"n_paths_{m}"] = n_paths[m]
    
    return result


@app.local_entrypoint()
def main():
    import math
    import json
    from datetime import datetime
    
    print("="*60)
    print("HST ISING FINITE-SIZE SCALING")
    print("="*60)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Configuration
    L_VALUES = [32, 64]
    T_VALUES = [
        1.5, 1.8, 2.0, 2.1, 2.15, 2.2,
        TC,  # 2.269
        2.3, 2.35, 2.4, 2.5, 2.7, 3.0, 3.5
    ]
    N_SEEDS = 10
    
    total = len(L_VALUES) * len(T_VALUES) * N_SEEDS
    print(f"\nConfiguration:")
    print(f"  L: {L_VALUES}")
    print(f"  T: {len(T_VALUES)} points (fine grid around Tc={TC:.3f})")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Total configs: {total}")
    
    print("\nRunning (this may take a few minutes)...")
    
    # Build configs
    configs = [
        (L, T, seed)
        for L in L_VALUES
        for T in T_VALUES
        for seed in range(N_SEEDS)
    ]
    
    # Run in parallel
    results = list(run_config.starmap(configs))
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"ising_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: ising_results_{timestamp}.json")
    
    # ===== ANALYSIS =====
    import numpy as np
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for L in L_VALUES:
        L_results = [r for r in results if r["L"] == L]
        n_paths_1 = L_results[0]["n_paths_1"]
        n_paths_2 = L_results[0]["n_paths_2"]
        
        print(f"\n{'='*60}")
        print(f"L = {L}  (n_paths: order1={n_paths_1}, order2={n_paths_2})")
        print(f"{'='*60}")
        print(f"{'T':<7} {'|m|':<7} {'d_eff_1':<9} {'d_eff_1/n':<10} {'d_eff_2':<9} {'d_eff_2/n':<10}")
        print("-"*60)
        
        for T in sorted(set(r["T"] for r in L_results)):
            T_res = [r for r in L_results if abs(r["T"] - T) < 0.001]
            
            mag = np.mean([r["mean_mag"] for r in T_res])
            d1 = np.mean([r["d_eff_1"] for r in T_res])
            d1_norm = np.mean([r["d_eff_norm_1"] for r in T_res])
            d2 = np.mean([r["d_eff_2"] for r in T_res])
            d2_norm = np.mean([r["d_eff_norm_2"] for r in T_res])
            
            marker = " <-- Tc" if abs(T - TC) < 0.01 else ""
            print(f"{T:<7.3f} {mag:<7.3f} {d1:<9.2f} {d1_norm:<10.4f} {d2:<9.2f} {d2_norm:<10.4f}{marker}")
    
    # Find minima
    print("\n" + "="*60)
    print("COMPACTNESS ANALYSIS")
    print("="*60)
    
    for L in L_VALUES:
        L_results = [r for r in results if r["L"] == L]
        
        # Group by T and average
        T_data = {}
        for T in sorted(set(r["T"] for r in L_results)):
            T_res = [r for r in L_results if abs(r["T"] - T) < 0.001]
            T_data[T] = {
                "d_eff_norm_1": np.mean([r["d_eff_norm_1"] for r in T_res]),
                "d_eff_norm_2": np.mean([r["d_eff_norm_2"] for r in T_res]),
            }
        
        # Find minimum d_eff_norm (maximum compactness)
        min_T_1 = min(T_data.keys(), key=lambda T: T_data[T]["d_eff_norm_1"])
        min_T_2 = min(T_data.keys(), key=lambda T: T_data[T]["d_eff_norm_2"])
        
        print(f"\nL = {L}:")
        print(f"  Order 1: min d_eff/n at T = {min_T_1:.3f} (Tc = {TC:.3f})")
        print(f"  Order 2: min d_eff/n at T = {min_T_2:.3f} (Tc = {TC:.3f})")
        
        if abs(min_T_1 - TC) < 0.2 or abs(min_T_2 - TC) < 0.2:
            print(f"  ✓ Compactness minimum near Tc!")
        else:
            print(f"  ✗ Compactness minimum NOT near Tc")
    
    print("\nDone!")
    return results
