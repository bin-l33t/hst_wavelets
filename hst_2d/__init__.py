"""
2D Heisenberg Scattering Transform

A proper 2D implementation of Glinsky's HST using wavelet cascade structure.

Features:
- Two-channel (H+/H-) Paul wavelets for full Fourier plane coverage
- Glinsky's R = i·ln(R₀) rectifier
- Littlewood-Paley normalized filter bank
- Complex data flow throughout
"""

from .scattering import (
    HST2D, 
    HST2DOutput,
    create_hst_2d, 
    glinsky_R_torch,
    glinsky_R0_torch, 
    simple_R_torch,
    lift_radial_torch,
)
from .filter_bank import (
    create_filter_bank,
    filter_bank_2d,
    paul_wavelet_2d_fourier,
    gaussian_lowpass_2d_fourier,
    verify_littlewood_paley,
    normalize_filterbank_littlewood_paley,
)

__all__ = [
    # Main class
    'HST2D',
    'HST2DOutput',
    'create_hst_2d',
    # Rectifiers
    'glinsky_R_torch',
    'glinsky_R0_torch', 
    'simple_R_torch',
    'lift_radial_torch',
    # Filter bank
    'create_filter_bank',
    'filter_bank_2d',
    'paul_wavelet_2d_fourier',
    'gaussian_lowpass_2d_fourier',
    'verify_littlewood_paley',
    'normalize_filterbank_littlewood_paley',
]
