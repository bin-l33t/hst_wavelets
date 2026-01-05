import modal
import numpy as np
import time

app = modal.App("fast-hst-benchmark")

# Definition of the container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_dir("hst", remote_path="/root/hst") # Upload your library
)

@app.function(image=image, gpu="T4")
def benchmark_energy():
    import torch
    from hst.fast_scattering import FastHST
    
    print("="*60)
    print("FAST HST (DECIMATED) ENERGY BENCHMARK")
    print("="*60)
    
    # 1. Setup
    T = 4096 # Long signal
    J = 8    # Deep scattering
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create a random signal with Unit Energy
    x = torch.randn(1, T, dtype=torch.complex128, device=device)
    x = x / torch.norm(x) # Energy = 1.0
    print(f"Input Energy: {torch.norm(x)**2:.4f}")
    
    # 2. Initialize Fast HST
    model = FastHST(J=J, kernel_size=16, rectifier='glinsky').to(device)
    
    # 3. Run Forward Pass
    start = time.time()
    coeffs = model(x)
    dt = time.time() - start
    
    # 4. Analyze Output Energy
    print(f"\nTransform Config: J={J}, Decimated (Fast)")
    print(f"Time: {dt*1000:.2f} ms")
    
    total_energy = 0.0
    print(f"\nScale | Size | Energy (Norm^2)")
    print("-" * 30)
    
    for j, s in enumerate(coeffs):
        # We sum the energy of the coefficients
        # Note: In a decimated transform, energy summation requires care with frame bounds,
        # but it should NOT explode geometrically like the CWT.
        e = torch.norm(s)**2
        total_energy += e
        print(f"  {j}   | {s.shape[-1]:4d} | {e:.4f}")
        
    print("-" * 30)
    print(f"Total Output Energy: {total_energy:.4f}")
    
    # 5. The Verdict
    # In the CWT version, you likely saw Total Energy > 10.0 or 100.0.
    # Here, for a tight frame, it should be close to 1.0 (or bounded, e.g., 2.0).
    # Ideally, we want conservation or mild attenuation, not explosion.
    
    if total_energy < 5.0:
        print("\n✅ SUCCESS: Energy is bounded. No explosion.")
    else:
        print("\n❌ FAIL: Energy is still exploding.")

    return float(total_energy)

@app.local_entrypoint()
def main():
    benchmark_energy.remote()
