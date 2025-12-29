"""
PyTorch Filterbank for HST

Mirrors the NumPy implementation in hst/filter_bank.py exactly.
"""

import math
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


def two_channel_paul_filterbank_torch(
    T: int,
    J: int = 4,
    Q: int = 2,
    m: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex128,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Create a two-channel tight frame Paul wavelet filter bank (PyTorch version).
    
    This exactly mirrors the NumPy implementation in hst/filter_bank.py.
    
    Parameters
    ----------
    T : int
        Signal length
    J : int
        Number of octaves
    Q : int
        Wavelets per octave
    m : int
        Paul wavelet order
    device : torch.device
        Target device
    dtype : torch.dtype
        Complex dtype for output
        
    Returns
    -------
    filters : list of torch.Tensor
        List of filters in frequency domain
    info : dict
        Filter metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    # Frequency axis (same as np.fft.fftfreq)
    k = torch.fft.fftfreq(T, device=device, dtype=torch.float64)
    omega = k * 2 * np.pi
    w = torch.abs(omega)
    
    pos_mask = k > 0
    neg_mask = k < 0
    
    num_mothers = J * Q
    xi_max = np.pi
    
    lp_sum_sq = torch.zeros(T, device=device, dtype=torch.float64)
    all_filters = []
    
    # H⁺ mothers (positive frequencies)
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        # Paul wavelet: (s·|ω|)^m · exp(-s·|ω|) = exp(m·log(arg) - arg)
        psi = torch.zeros(T, device=device, dtype=torch.float64)
        valid = pos_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq = lp_sum_sq + psi ** 2
    
    # H⁻ mothers (negative frequencies, mirror of H⁺)
    for j in range(num_mothers):
        xi_j = xi_max * (2 ** (-j / Q))
        s = m / xi_j
        arg = s * w
        
        psi = torch.zeros(T, device=device, dtype=torch.float64)
        valid = neg_mask & (arg > 1e-10)
        if valid.any():
            psi[valid] = torch.exp(m * torch.log(arg[valid]) - arg[valid])
        
        all_filters.append(psi)
        lp_sum_sq = lp_sum_sq + psi ** 2
    
    # Father wavelet (symmetric Gaussian, covers DC)
    xi_min = xi_max * (2 ** (-(num_mothers - 1) / Q))
    sigma_phi = xi_min
    phi = torch.exp(-w**2 / (2 * sigma_phi**2))
    all_filters.append(phi)
    lp_sum_sq = lp_sum_sq + phi ** 2
    
    # Normalize for partition of unity
    lp_sum = torch.sqrt(lp_sum_sq)
    lp_sum = torch.maximum(lp_sum, torch.tensor(1e-10, device=device))
    
    # Normalize and convert to complex
    filters = [(f / lp_sum).to(dtype) for f in all_filters]
    
    # Verify partition of unity
    pou = sum(torch.abs(f)**2 for f in filters)
    
    info = {
        'T': T,
        'J': J,
        'Q': Q,
        'm': m,
        'n_pos_mothers': num_mothers,
        'n_neg_mothers': num_mothers,
        'n_father': 1,
        'n_total': len(filters),
        'n_mothers': 2 * num_mothers,  # For compatibility
        'pou_min': float(pou.min().item()),
        'pou_max': float(pou.max().item()),
        'pou_ok': torch.allclose(pou, torch.ones_like(pou), atol=1e-6),
        'backend': 'torch',
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
        Input signal, shape (T,) complex
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
