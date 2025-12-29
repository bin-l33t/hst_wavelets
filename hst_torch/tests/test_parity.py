#!/usr/bin/env python3
"""
Parity Tests: PyTorch vs NumPy HST Implementation

These tests verify that the Torch implementation produces identical
results to the NumPy reference implementation.

Tests are designed to be fast (<10 seconds total).

Run from hst_wavelets directory:
    python -m hst_torch.tests.test_parity
  or:
    python hst_torch/tests/test_parity.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for direct execution
hst_wavelets_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(hst_wavelets_dir))

# Now import torch (real or mock)
try:
    import torch
    HAS_REAL_TORCH = not getattr(torch, '_is_mock', False)
    print(f"Using PyTorch {torch.__version__}")
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


def test_filterbank_parity():
    """Test that Torch filterbank matches NumPy filterbank."""
    from hst.filter_bank import two_channel_paul_filterbank
    from hst_torch.filter_bank import two_channel_paul_filterbank_torch
    
    print("Testing filterbank parity...")
    
    N = 256
    J, Q = 3, 2
    
    # NumPy version
    filters_np, info_np = two_channel_paul_filterbank(N, J, Q)
    
    # Torch version
    filters_torch, info_torch = two_channel_paul_filterbank_torch(N, J, Q)
    
    # Check counts match
    assert len(filters_np) == len(filters_torch), "Filter count mismatch"
    assert info_np['n_mothers'] == info_torch['n_mothers'], "Mother count mismatch"
    
    # Check each filter
    max_diff = 0
    for i, (f_np, f_torch) in enumerate(zip(filters_np, filters_torch)):
        f_torch_np = f_torch.cpu().numpy()
        diff = np.max(np.abs(f_np - f_torch_np))
        max_diff = max(max_diff, diff)
        
        if diff > 1e-10:
            print(f"  Filter {i}: max diff = {diff:.2e}")
    
    print(f"  ✓ Filterbank parity: max diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Filterbank mismatch: {max_diff}"
    return True


def test_forward_transform_parity():
    """Test that Torch forward transform matches NumPy."""
    from hst.filter_bank import two_channel_paul_filterbank, forward_transform
    from hst_torch.filter_bank import two_channel_paul_filterbank_torch, forward_transform_torch
    
    print("Testing forward transform parity...")
    
    N = 256
    J, Q = 3, 2
    
    # Create test signal
    np.random.seed(42)
    x_np = np.random.randn(N) + 1j * np.random.randn(N) + 3.0
    x_torch = torch.tensor(x_np, dtype=torch.complex128)
    
    # NumPy
    filters_np, _ = two_channel_paul_filterbank(N, J, Q)
    coeffs_np = forward_transform(x_np, filters_np)
    
    # Torch
    filters_torch, _ = two_channel_paul_filterbank_torch(N, J, Q)
    coeffs_torch = forward_transform_torch(x_torch, filters_torch)
    
    # Compare
    max_diff = 0
    for i, (c_np, c_torch) in enumerate(zip(coeffs_np, coeffs_torch)):
        c_torch_np = c_torch.cpu().numpy()
        diff = np.max(np.abs(c_np - c_torch_np))
        max_diff = max(max_diff, diff)
    
    print(f"  ✓ Forward transform parity: max diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Forward transform mismatch: {max_diff}"
    return True


def test_hst_output_shapes():
    """Test that Torch HST produces same output shapes as NumPy."""
    from hst.scattering import HeisenbergScatteringTransform
    from hst_torch.scattering import HeisenbergScatteringTransformTorch
    
    print("Testing HST output shapes...")
    
    T = 256
    J, Q = 3, 2
    max_order = 2
    
    # Create test signal
    np.random.seed(42)
    x_np = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    x_torch = torch.tensor(x_np, dtype=torch.complex128)
    
    # NumPy HST
    hst_np = HeisenbergScatteringTransform(T, J, Q, max_order=max_order)
    out_np = hst_np.forward(x_np)
    
    # Torch HST
    hst_torch = HeisenbergScatteringTransformTorch(T, J, Q, max_order=max_order)
    out_torch = hst_torch.forward(x_torch)
    
    # Check path counts
    for m in range(max_order + 1):
        paths_np = out_np.order(m)
        paths_torch = out_torch.order(m)
        
        assert len(paths_np) == len(paths_torch), f"Order {m} path count mismatch"
        
        # Check same paths exist
        assert set(paths_np.keys()) == set(paths_torch.keys()), f"Order {m} path keys mismatch"
        
        # Check shapes
        for p in paths_np.keys():
            assert paths_np[p].shape == paths_torch[p].shape, f"Path {p} shape mismatch"
    
    print(f"  ✓ Output shapes match for all orders")
    return True


def test_hst_values_parity():
    """Test that Torch HST produces same values as NumPy."""
    from hst.scattering import HeisenbergScatteringTransform
    from hst_torch.scattering import HeisenbergScatteringTransformTorch
    
    print("Testing HST values parity...")
    
    T = 256
    J, Q = 3, 2
    max_order = 2
    
    # Create test signal (shifted from origin)
    np.random.seed(42)
    x_np = np.random.randn(T) + 1j * np.random.randn(T) + 5.0
    x_torch = torch.tensor(x_np, dtype=torch.complex128)
    
    # NumPy HST
    hst_np = HeisenbergScatteringTransform(T, J, Q, max_order=max_order,
                                            lifting='radial_floor', epsilon=1e-8)
    out_np = hst_np.forward(x_np)
    
    # Torch HST
    hst_torch = HeisenbergScatteringTransformTorch(T, J, Q, max_order=max_order,
                                                    lifting='radial_floor', epsilon=1e-8)
    out_torch = hst_torch.forward(x_torch)
    
    # Compare values
    max_diff = 0
    max_rel_diff = 0
    
    for m in range(max_order + 1):
        paths_np = out_np.order(m)
        paths_torch = out_torch.order(m)
        
        for p in paths_np.keys():
            v_np = paths_np[p]
            v_torch = paths_torch[p].cpu().numpy()
            
            diff = np.max(np.abs(v_np - v_torch))
            max_diff = max(max_diff, diff)
            
            # Relative difference
            scale = np.max(np.abs(v_np)) + 1e-10
            rel_diff = diff / scale
            max_rel_diff = max(max_rel_diff, rel_diff)
    
    print(f"  ✓ HST values parity: max abs diff = {max_diff:.2e}, max rel diff = {max_rel_diff:.2e}")
    
    # Allow small numerical differences
    assert max_rel_diff < 1e-8, f"HST values mismatch: rel diff = {max_rel_diff}"
    return True


def test_per_path_energy_parity():
    """Test that per-path energies match between Torch and NumPy."""
    from hst.scattering import HeisenbergScatteringTransform
    from hst_torch.scattering import HeisenbergScatteringTransformTorch
    
    print("Testing per-path energy parity...")
    
    T = 256
    J, Q = 3, 2
    max_order = 2
    
    # Create test signal
    np.random.seed(123)
    x_np = np.cos(2 * np.pi * 5 * np.arange(T) / T) + 3.0 + 0j
    x_torch = torch.tensor(x_np, dtype=torch.complex128)
    
    # NumPy HST
    hst_np = HeisenbergScatteringTransform(T, J, Q, max_order=max_order)
    out_np = hst_np.forward(x_np)
    
    # Torch HST
    hst_torch = HeisenbergScatteringTransformTorch(T, J, Q, max_order=max_order)
    out_torch = hst_torch.forward(x_torch)
    
    # Compare energies
    max_rel_diff = 0
    
    for m in range(max_order + 1):
        for p in out_np.order(m).keys():
            e_np = np.sum(np.abs(out_np.paths[p]) ** 2)
            v_torch = out_torch.paths[p].cpu().numpy()
            e_torch = np.sum(np.abs(v_torch) ** 2)
            
            rel_diff = abs(e_np - e_torch) / (e_np + 1e-10)
            max_rel_diff = max(max_rel_diff, rel_diff)
    
    print(f"  ✓ Per-path energy parity: max rel diff = {max_rel_diff:.2e}")
    assert max_rel_diff < 1e-8, f"Energy mismatch: {max_rel_diff}"
    return True


def test_ising_signal_parity():
    """Test parity on Ising-like signals (the actual use case)."""
    from hst.scattering import HeisenbergScatteringTransform
    from hst_torch.scattering import HeisenbergScatteringTransformTorch
    
    print("Testing Ising signal parity...")
    
    T = 64  # Row length for Ising
    J, Q = 2, 2
    max_order = 2
    
    # Create Ising-like signal (binary + offset)
    np.random.seed(99)
    spins = np.random.choice([-1, 1], size=T)
    x_np = spins + 3.0 + 0j  # Magnitude encoding
    x_torch = torch.tensor(x_np, dtype=torch.complex128)
    
    # NumPy HST
    hst_np = HeisenbergScatteringTransform(T, J, Q, max_order=max_order)
    out_np = hst_np.forward(x_np)
    
    # Torch HST
    hst_torch = HeisenbergScatteringTransformTorch(T, J, Q, max_order=max_order)
    out_torch = hst_torch.forward(x_torch)
    
    # Compute compactness metrics and compare
    def compute_d_eff_np(paths_dict, m):
        paths_m = {p: c for p, c in paths_dict.items() if len(p) == m}
        energies = np.array([np.sum(np.abs(c)**2) for c in paths_m.values()])
        total = np.sum(energies)
        if np.sum(energies**2) == 0:
            return 0
        return (total**2) / np.sum(energies**2)
    
    def compute_d_eff_torch(paths_dict, m):
        paths_m = {p: c for p, c in paths_dict.items() if len(p) == m}
        energies = np.array([np.sum(np.abs(c.cpu().numpy())**2) for c in paths_m.values()])
        total = np.sum(energies)
        if np.sum(energies**2) == 0:
            return 0
        return (total**2) / np.sum(energies**2)
    
    for m in [1, 2]:
        d_eff_np = compute_d_eff_np(out_np.paths, m)
        d_eff_torch = compute_d_eff_torch(out_torch.paths, m)
        
        rel_diff = abs(d_eff_np - d_eff_torch) / (d_eff_np + 1e-10)
        print(f"  Order {m}: d_eff_np={d_eff_np:.4f}, d_eff_torch={d_eff_torch:.4f}, rel_diff={rel_diff:.2e}")
        
        assert rel_diff < 1e-8, f"d_eff mismatch at order {m}"
    
    print(f"  ✓ Ising signal parity verified")
    return True


def run_all_tests():
    """Run all parity tests."""
    print("="*60)
    print("PARITY TESTS: PyTorch vs NumPy HST")
    print("="*60)
    
    tests = [
        test_filterbank_parity,
        test_forward_transform_parity,
        test_hst_output_shapes,
        test_hst_values_parity,
        test_per_path_energy_parity,
        test_ising_signal_parity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
