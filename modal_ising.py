"""
HST Ising Experiment - Modal App
Based on working hello world template
"""

import modal

app = modal.App("hst-ising-experiment")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
    "scipy>=1.10",
    "torch>=2.0",
)


@app.function(image=image, timeout=600)
def run_single_config(L: int, T: float, seed: int) -> dict:
    """Run HST on single Ising config."""
    import numpy as np
    import torch
    import math
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Ising model
    rng = np.random.default_rng(seed)
    spins = np.ones((L, L), dtype=int) if T < TC else rng.choice([-1, 1], size=(L, L))
    
    # Wolff equilibration
    for _ in range(200):
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
    
    # HST filterbank (inline)
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
                s, arg = m / xi_j, (m / xi_j) * w
                psi = torch.zeros(N, dtype=torch.float64)
                valid = channel_mask & (arg > 1e-10)
                if valid.any():
                    psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
                filters.append(psi)
                lp_sum_sq += psi ** 2
        
        # Father
        xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
        phi = torch.exp(-w**2 / (2 * xi_min**2))
        filters.append(phi)
        lp_sum_sq += phi ** 2
        
        lp_sum = torch.sqrt(torch.clamp(lp_sum_sq, min=1e-20))
        return [(f / lp_sum).to(torch.complex128) for f in filters], 2 * num_mothers
    
    # HST forward
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
    
    # Compute d_eff
    filters, n_mothers = make_filterbank(L)
    row = torch.tensor(spins[0, :].astype(float) + 3.0 + 0j, dtype=torch.complex128)
    paths = hst_forward(row, filters, n_mothers)
    
    d_effs = {}
    for m in [0, 1, 2]:
        paths_m = {p: c for p, c in paths.items() if len(p) == m}
        if paths_m:
            energies = np.array([torch.sum(torch.abs(c)**2).item() for c in paths_m.values()])
            total = np.sum(energies)
            d_effs[m] = (total**2) / np.sum(energies**2) if np.sum(energies**2) > 0 else 1.0
    
    return {"L": L, "T": T, "seed": seed, "mag": float(np.abs(np.mean(spins))), **{f"d_eff_{m}": d_effs.get(m, 0) for m in [0,1,2]}}


@app.local_entrypoint()
def main():
    import math
    
    print("="*50)
    print("HST ISING EXPERIMENT")
    print("="*50)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    L_VALUES = [32]
    T_VALUES = [1.5, TC, 3.0]
    
    print(f"\nRunning L={L_VALUES}, T={len(T_VALUES)} points...")
    
    configs = [(L, T, seed) for L in L_VALUES for T in T_VALUES for seed in range(3)]
    
    results = list(run_single_config.starmap(configs))
    
    print("\nResults:")
    print(f"{'T':<8} {'|m|':<8} {'d_eff_1':<10} {'d_eff_2':<10}")
    print("-"*36)
    
    for T in T_VALUES:
        T_res = [r for r in results if abs(r['T'] - T) < 0.01]
        if T_res:
            import numpy as np
            mag = np.mean([r['mag'] for r in T_res])
            d1 = np.mean([r['d_eff_1'] for r in T_res])
            d2 = np.mean([r['d_eff_2'] for r in T_res])
            marker = " *" if abs(T - TC) < 0.01 else ""
            print(f"{T:<8.3f} {mag:<8.3f} {d1:<10.2f} {d2:<10.2f}{marker}")
    
    print("\nDone!")
