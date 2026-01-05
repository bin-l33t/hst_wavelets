"""
2D Filter Bank for Heisenberg Scattering Transform

Based on Kymatio's filter_bank.py but adapted for HST.
Uses Morlet wavelets initially (can be swapped to Cauchy/Paul).
"""

import numpy as np
from scipy.fft import fft2


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
    Computes a 2D Gabor filter in spatial domain.
    
    psi(u) = g_{sigma}(u) * exp(i * xi^T * u)
    
    Parameters
    ----------
    M, N : int
        Spatial sizes
    sigma : float
        Bandwidth parameter (scale)
    theta : float
        Orientation angle in [0, pi]
    xi : float
        Central frequency
    slant : float
        Ellipticity parameter
    """
    gab = np.zeros((M, N), np.complex128)
    
    # Rotation matrices
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]], np.float64)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], 
                      [-np.sin(theta), np.cos(theta)]], np.float64)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / (2 * sigma * sigma)
    
    # Sum over periodic copies for proper tiling
    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, 
                                offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * xx * xx + 
                   (curv[0, 1] + curv[1, 0]) * xx * yy + 
                   curv[1, 1] * yy * yy) + \
                  1j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)
    
    norm_factor = 2 * np.pi * sigma * sigma / slant
    gab /= norm_factor
    
    return gab


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
    Computes a 2D Morlet filter (Gabor with zero-mean correction).
    
    psi(u) = g_{sigma}(u) * (exp(i*xi^T*u) - beta)
    
    The beta term ensures zero mean (admissibility condition).
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)
    
    mor = wv - K * wv_modulus
    return mor


def periodize_filter_fft(x, res):
    """
    Periodize filter in Fourier domain for subsampling.
    
    Cropping in Fourier = periodization in space.
    """
    M, N = x.shape
    crop = np.zeros((M // 2**res, N // 2**res), x.dtype)
    
    # Mask to avoid aliasing
    mask = np.ones(x.shape, np.float64)
    len_x = int(M * (1 - 2**(-res)))
    start_x = int(M * 2**(-res - 1))
    len_y = int(N * (1 - 2**(-res)))
    start_y = int(N * 2**(-res - 1))
    mask[start_x:start_x + len_x, :] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = x * mask
    
    # Periodize by summing shifted copies
    M_crop = M // 2**res
    N_crop = N // 2**res
    for k in range(M_crop):
        for l in range(N_crop):
            for i in range(2**res):
                for j in range(2**res):
                    crop[k, l] += x[k + i * M_crop, l + j * N_crop]
    
    return crop


def filter_bank(M, N, J, L=8):
    """
    Build 2D Morlet filter bank for scattering transform.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    J : int
        Number of scales (octaves)
    L : int
        Number of orientations (angles)
        
    Returns
    -------
    filters : dict
        'psi': list of wavelet dicts with 'levels', 'j', 'theta'
        'phi': lowpass filter dict with 'levels', 'j'
    """
    filters = {'psi': []}
    
    # Mother wavelets at each scale and orientation
    for j in range(J):
        for theta_idx in range(L):
            psi = {'levels': [], 'j': j, 'theta': theta_idx}
            
            # Kymatio's parameter choices
            sigma = 0.8 * 2**j
            theta = (int(L - L/2 - 1) - theta_idx) * np.pi / L
            xi = 3.0 / 4.0 * np.pi / 2**j
            slant = 4.0 / L
            
            # Generate in spatial domain, then FFT
            psi_signal = morlet_2d(M, N, sigma, theta, xi, slant)
            psi_signal_fourier = fft2(psi_signal)
            
            # Generate periodized versions for each resolution
            psi_levels = []
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_levels.append(periodize_filter_fft(psi_signal_fourier, res))
            psi['levels'] = psi_levels
            filters['psi'].append(psi)
    
    # Father wavelet (lowpass) - just a Gaussian
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = fft2(phi_signal)
    
    filters['phi'] = {'levels': [], 'j': J}
    for res in range(J):
        filters['phi']['levels'].append(
            periodize_filter_fft(phi_signal_fourier, res))
    
    return filters


def compute_padding(M, N, J):
    """Compute padded size to avoid boundary effects."""
    M_padded = ((M + 2**J) // 2**J + 1) * 2**J
    N_padded = ((N + 2**J) // 2**J + 1) * 2**J
    return M_padded, N_padded
