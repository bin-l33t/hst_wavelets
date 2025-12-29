"""
PyTorch Filterbank for HST

Mirrors the NumPy implementation in hst/filter_bank.py but uses torch operations.
"""

import math
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


def paul_wavelet_torch(
    N: int,
    scale: float,
    order: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Generate Paul wavelet in frequency domain (PyTorch).
    
    Parameters
    ----------
    N : int
        Signal length
    scale : float
        Wavelet scale (center frequency ~ 1/scale)
    order : int
        Paul wavelet order (default 4)
    device : torch.device
        Target device
    dtype : torch.dtype
        Complex dtype
        
    Returns
    -------
    psi_hat : torch.Tensor
        Wavelet in frequency domain, shape (N,)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Frequency axis (same convention as numpy fft)
    k = torch.arange(N, device=device, dtype=torch.float64)
    k[N//2 + 1:] = k[N//2 + 1:] - N
    
    # Normalized frequency
    omega = 2 * np.pi * k / N
    
    # Paul wavelet in frequency domain (analytic, positive frequencies only)
    # H(ω) * ω^m * exp(-ω) for ω > 0
    m = order
    
    # Normalization constant
    norm = (2 ** m) / np.sqrt(N * math.factorial(2 * m - 1))
    
    # Scaled frequency
    w = scale * omega
    
    # Paul wavelet (only positive frequencies)
    # Compute real values first, then convert to complex
    psi_real = torch.zeros(N, device=device, dtype=torch.float64)
    pos_mask = omega > 0
    w_pos = w[pos_mask]
    psi_real[pos_mask] = norm * (w_pos ** m) * torch.exp(-w_pos)
    
    # Convert to complex
    psi_hat = psi_real.to(dtype)
    
    return psi_hat


def two_channel_paul_filterbank_torch(
    N: int,
    J: int = 4,
    Q: int = 2,
    order: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex128,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Create Paul wavelet filterbank (PyTorch version).
    
    Parameters
    ----------
    N : int
        Signal length
    J : int
        Number of octaves
    Q : int
        Wavelets per octave
    order : int
        Paul wavelet order
    device : torch.device
        Target device
    dtype : torch.dtype
        Complex dtype
        
    Returns
    -------
    filters : list of torch.Tensor
        List of filters in frequency domain. Last one is lowpass (father).
    info : dict
        Filter metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    filters = []
    scales = []
    
    # Mother wavelets at dyadic scales with Q subdivisions
    for j in range(J):
        for q in range(Q):
            scale = 2 ** (j + q / Q)
            psi_hat = paul_wavelet_torch(N, scale, order, device, dtype)
            filters.append(psi_hat)
            scales.append(scale)
    
    # Father wavelet (lowpass) - Gaussian
    k = torch.arange(N, device=device, dtype=torch.float64)
    k[N//2 + 1:] = k[N//2 + 1:] - N
    omega = 2 * np.pi * k / N
    
    # Lowpass: Gaussian centered at 0
    max_scale = 2 ** J
    sigma = max_scale / (2 * np.pi)
    phi_hat = torch.exp(-0.5 * (omega * sigma) ** 2).to(dtype)
    filters.append(phi_hat)
    
    info = {
        'N': N,
        'J': J,
        'Q': Q,
        'order': order,
        'n_mothers': J * Q,
        'scales': scales,
    }
    
    return filters, info


def forward_transform_torch(
    x: torch.Tensor,
    filters: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Apply filterbank to signal (convolution in frequency domain).
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal, shape (N,) complex
    filters : list of torch.Tensor
        Filters in frequency domain
        
    Returns
    -------
    coeffs : list of torch.Tensor
        Filtered signals (same length as input)
    """
    # FFT of input
    x_hat = torch.fft.fft(x)
    
    # Convolve with each filter
    coeffs = []
    for filt in filters:
        y_hat = x_hat * filt
        y = torch.fft.ifft(y_hat)
        coeffs.append(y)
    
    return coeffs


def lowpass_torch(
    x: torch.Tensor,
    phi_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Apply lowpass filter.
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal
    phi_hat : torch.Tensor
        Lowpass filter in frequency domain
        
    Returns
    -------
    y : torch.Tensor
        Lowpass filtered signal
    """
    x_hat = torch.fft.fft(x)
    y_hat = x_hat * phi_hat
    return torch.fft.ifft(y_hat)
