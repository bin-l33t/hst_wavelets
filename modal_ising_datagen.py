"""
Ising Dataset Generator

Generates magnetization time series for multiple (L, T, seed) configurations
and saves to Modal Volume for reuse across experiments.
"""

import modal

app = modal.App("ising-dataset-gen")

# Create a volume to persist data
volume = modal.Volume.from_name("ising-datasets", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.24",
)


@app.function(image=image, timeout=600, volumes={"/data": volume})
def generate_timeseries(
    L: int,
    T: float,
    seed: int,
    n_steps: int = 50000,
) -> str:
    """
    Generate and save magnetization time series.
    Returns the path to saved file.
    """
    import numpy as np
    import math
    import os
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
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
    for _ in range(2000):
        wolff_step()
    
    # Generate time series
    m_t = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        wolff_step()
        m_t[t] = np.mean(spins)
    
    # Save to volume
    T_str = f"{T:.4f}".replace(".", "p")
    filename = f"/data/L{L}_T{T_str}_seed{seed}.npz"
    
    # Compute basic stats
    m_mean = float(m_t.mean())
    m_std = float(m_t.std())
    
    np.savez_compressed(
        filename,
        m_t=m_t,
        L=L,
        T=T,
        seed=seed,
        n_steps=n_steps,
        m_mean=m_mean,
        m_std=m_std,
    )
    
    # Commit the volume
    volume.commit()
    
    return filename


@app.function(image=image, timeout=60, volumes={"/data": volume})
def list_datasets() -> list:
    """List all generated datasets."""
    import os
    files = []
    for f in os.listdir("/data"):
        if f.endswith(".npz"):
            files.append(f)
    return sorted(files)


@app.local_entrypoint()
def main():
    import math
    
    print("="*60)
    print("ISING DATASET GENERATOR")
    print("="*60)
    
    TC = 2 / math.log(1 + math.sqrt(2))
    
    # Configuration
    L = 32
    T_VALUES = [1.5, 2.0, TC, 2.5, 3.0]
    N_SEEDS = 5
    N_STEPS = 50000  # Long time series for multiple experiments
    
    print(f"\nConfiguration:")
    print(f"  L: {L}")
    print(f"  T: {[round(t, 3) for t in T_VALUES]}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Steps per series: {N_STEPS}")
    
    configs = [
        (L, T, seed, N_STEPS)
        for T in T_VALUES
        for seed in range(N_SEEDS)
    ]
    
    print(f"  Total datasets: {len(configs)}")
    print("\nGenerating...")
    
    # Generate in parallel
    results = list(generate_timeseries.starmap(configs))
    
    print(f"\nGenerated {len(results)} datasets:")
    for path in results[:5]:
        print(f"  {path}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more")
    
    # List all datasets
    print("\nAll datasets in volume:")
    all_files = list_datasets.remote()
    for f in all_files:
        print(f"  {f}")
    
    print("\n" + "="*60)
    print("DONE - Datasets saved to 'ising-datasets' volume")
    print("="*60)
