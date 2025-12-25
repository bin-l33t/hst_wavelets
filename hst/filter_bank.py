"""
Two-Channel Paul Wavelet Filter Bank: H⁺ ⊕ H⁻

Implements a tight frame filter bank based on Cauchy-Paul wavelets
(Ali et al., 2014, Eq. 12.20) with full spectrum coverage.

Theory:
    L² = H⁺ ⊕ H⁻ where:
    - H⁺ = {f : f̂(ω) = 0 for ω < 0} (analytic/progressive)
    - H⁻ = {f : f̂(ω) = 0 for ω > 0} (anti-analytic/regressive)
    
    Using wavelets on both channels ensures perfect reconstruction
    for arbitrary signals, including after nonlinear transforms like
    Glinsky's R mapping.

References:
    - Ali, Antoine, Gazeau (2014), Ch. 12
    - Glinsky (2025), Section VII
"""

import numpy as np
from typing import Tuple, Dict, Any

# Try torch import for GPU support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def two_channel_paul_filterbank(
    T: int,
    J: int,
    Q: int,
    m: int = 4,
    backend: str = "numpy"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a two-channel tight frame Paul wavelet filter bank.
    
    Parameters
    ----------
    T : int
        Signal length (number of samples)
    J : int
        Number of octaves
    Q : int
        Number of filters per octave
    m : int, optional
        Paul wavelet order (default 4). Higher = better frequency
        localization, worse time localization.
    backend : str, optional
        "numpy" or "torch" (for GPU)
        
    Returns
    -------
    filters : ndarray, shape (N_filters, T)
        Complex filter bank in frequency domain
    info : dict
        Metadata including partition of unity verification
        
    Examples
    --------
    >>> filters, info = two_channel_paul_filterbank(512, 4, 4)
    >>> assert info['pou_ok'], "Partition of unity failed"
    >>> coeffs = forward_transform(signal, filters)
    >>> reconstructed = inverse_transform(coeffs, filters)
    """
    if backend == "torch" and HAS_TORCH:
        return _two_channel_torch(T, J, Q, m)
    else:
        return _two_channel_numpy(T, J, Q, m)


def _two_channel_numpy(T: int, J: int, Q: int, m: int) -> Tuple[np.ndarray, Dict]:
    """NumPy implementation."""
    k = np.fft.fftfreq(T)
    omega = k * 2 * np.pi
    w = np.abs(omega)
    
    pos_mask = k > 0
    neg_mask = k < 0
    
    num_mothers = J * Q
    xi_max = np.pi
    
    lp_sum_sq = np.zeros(T, dtype=np.float64)
    all_filters = []
    
    # H⁺ mothers (positive frequencies)
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        # Paul wavelet: (s·|ω|)^m · exp(-s·|ω|)
        psi = np.zeros(T, dtype=np.float64)
        valid = pos_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = np.exp(m * np.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq += psi ** 2
    
    # H⁻ mothers (negative frequencies, mirror of H⁺)
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        psi = np.zeros(T, dtype=np.float64)
        valid = neg_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = np.exp(m * np.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq += psi ** 2
    
    # Father wavelet (symmetric Gaussian, covers DC)
    xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
    sigma_phi = xi_min
    phi = np.exp(-w**2 / (2 * sigma_phi**2))
    all_filters.append(phi)
    lp_sum_sq += phi ** 2
    
    # Normalize for partition of unity
    lp_sum = np.sqrt(lp_sum_sq)
    lp_sum = np.maximum(lp_sum, 1e-10)
    
    filters = np.stack([f / lp_sum for f in all_filters]).astype(np.complex128)
    
    # Verify partition of unity
    pou = np.sum(np.abs(filters)**2, axis=0)
    
    info = {
        'T': T,
        'J': J,
        'Q': Q,
        'm': m,
        'n_pos_mothers': num_mothers,
        'n_neg_mothers': num_mothers,
        'n_father': 1,
        'n_total': filters.shape[0],
        'pou_min': float(pou.min()),
        'pou_max': float(pou.max()),
        'pou_ok': np.allclose(pou, 1.0, atol=1e-6),
        'backend': 'numpy',
    }
    
    return filters, info


def _two_channel_torch(T: int, J: int, Q: int, m: int, device: str = 'cpu'):
    """PyTorch implementation for GPU support."""
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")
    
    k = torch.fft.fftfreq(T, device=device)
    omega = k * 2 * np.pi
    w = torch.abs(omega)
    
    pos_mask = k > 0
    neg_mask = k < 0
    
    num_mothers = J * Q
    xi_max = np.pi
    
    lp_sum_sq = torch.zeros(T, device=device, dtype=torch.float32)
    all_filters = []
    
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        psi = torch.zeros(T, device=device, dtype=torch.float32)
        valid = pos_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq += psi ** 2
    
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        psi = torch.zeros(T, device=device, dtype=torch.float32)
        valid = neg_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq += psi ** 2
    
    xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
    sigma_phi = xi_min
    phi = torch.exp(-w**2 / (2 * sigma_phi**2))
    all_filters.append(phi)
    lp_sum_sq += phi ** 2
    
    lp_sum = torch.sqrt(lp_sum_sq)
    lp_sum = torch.clamp(lp_sum, min=1e-10)
    
    filters = torch.stack([f / lp_sum for f in all_filters]).to(torch.complex64)
    
    pou = torch.sum(torch.abs(filters)**2, dim=0)
    
    info = {
        'T': T, 'J': J, 'Q': Q, 'm': m,
        'n_pos_mothers': num_mothers,
        'n_neg_mothers': num_mothers,
        'n_father': 1,
        'n_total': filters.shape[0],
        'pou_min': float(pou.min()),
        'pou_max': float(pou.max()),
        'pou_ok': torch.allclose(pou, torch.ones_like(pou), atol=1e-6),
        'backend': 'torch',
        'device': str(device),
    }
    
    return filters, info


def forward_transform(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """
    Apply wavelet transform (analysis).
    
    Parameters
    ----------
    x : ndarray, shape (T,)
        Input signal (real or complex)
    filters : ndarray, shape (N, T)
        Filter bank from two_channel_paul_filterbank
        
    Returns
    -------
    coeffs : ndarray, shape (N, T)
        Wavelet coefficients in time domain
    """
    x_hat = np.fft.fft(x)
    coeffs_hat = x_hat[np.newaxis, :] * filters
    return np.fft.ifft(coeffs_hat, axis=-1)


def inverse_transform(coeffs: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """
    Reconstruct signal from wavelet coefficients (synthesis).
    
    For a tight frame, the inverse is the adjoint.
    
    Parameters
    ----------
    coeffs : ndarray, shape (N, T)
        Wavelet coefficients from forward_transform
    filters : ndarray, shape (N, T)
        Same filter bank used in forward_transform
        
    Returns
    -------
    x : ndarray, shape (T,)
        Reconstructed signal
    """
    coeffs_hat = np.fft.fft(coeffs, axis=-1)
    x_hat = np.sum(coeffs_hat * np.conj(filters), axis=0)
    return np.fft.ifft(x_hat)


# Torch versions
def forward_transform_torch(x, filters):
    """PyTorch version of forward_transform."""
    x_hat = torch.fft.fft(x)
    coeffs_hat = x_hat.unsqueeze(0) * filters
    return torch.fft.ifft(coeffs_hat)


def inverse_transform_torch(coeffs, filters):
    """PyTorch version of inverse_transform."""
    coeffs_hat = torch.fft.fft(coeffs)
    x_hat = torch.sum(coeffs_hat * torch.conj(filters), dim=0)
    return torch.fft.ifft(x_hat)
