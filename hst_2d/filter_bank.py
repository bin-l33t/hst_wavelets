"""
2D Filter Bank for Heisenberg Scattering Transform

Two-channel design (H+ and H-) for complex signals, analogous to 1D HST.

Key differences from standard (Kymatio-style) scattering filter banks:
1. Uses Paul/Cauchy wavelets instead of Morlet (analytic, one-sided in freq)
2. Two channels: H+ (positive freq half-plane) and H- (negative freq half-plane)
3. Covers full [0, 2π) orientations, not just [0, π)
4. Designed for complex input signals where Hermitian symmetry doesn't hold

The two-channel design ensures we capture all information from complex
intermediate signals after applying Glinsky's R rectifier.
"""

import numpy as np
from typing import Dict, List, Tuple


def paul_wavelet_2d_fourier(
    M: int, 
    N: int, 
    j: int, 
    theta: float, 
    m: int = 4,
    positive: bool = True,
) -> np.ndarray:
    """
    Create 2D Paul wavelet in Fourier domain.
    
    Paul wavelets are analytic (one-sided in frequency). In 2D, we create
    a directional wavelet supported on a cone in the frequency half-plane
    defined by the orientation theta.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    j : int
        Scale index (larger j = coarser scale = lower frequency)
    theta : float
        Orientation angle in [0, 2π)
    m : int
        Paul wavelet order (higher = more oscillations)
    positive : bool
        If True, H+ channel (positive freq cone along theta)
        If False, H- channel (negative freq cone, opposite direction)
        
    Returns
    -------
    psi_hat : ndarray, shape (M, N), complex
        Wavelet in Fourier domain
    """
    # Frequency grid (centered at DC)
    kx = np.fft.fftfreq(N)  # Normalized frequencies in [-0.5, 0.5)
    ky = np.fft.fftfreq(M)
    KX, KY = np.meshgrid(kx, ky)
    
    # Rotate frequency coordinates to align with theta
    # k_parallel: component along theta direction
    # k_perp: component perpendicular to theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    k_parallel = KX * cos_t + KY * sin_t
    k_perp = -KX * sin_t + KY * cos_t
    
    # For H- channel, flip the direction
    if not positive:
        k_parallel = -k_parallel
    
    # Scale parameter: peak frequency decreases with j
    # At j=0, peak near Nyquist; at j=J-1, peak near DC
    k0 = 0.4 * (2.0 ** (-j))  # Central frequency for this scale
    sigma = k0 / m  # Bandwidth (narrower for higher m)
    
    # Paul wavelet in frequency domain:
    # Supported only on k_parallel > 0 (one-sided / analytic)
    # Shape: (k_parallel)^m * exp(-k_parallel / sigma) for k_parallel > 0
    
    # Radial distance from origin
    k_norm = np.sqrt(KX**2 + KY**2)
    
    # Cone mask: only positive k_parallel (with soft edge)
    # Using sigmoid for smooth transition
    cone_width = 0.5  # Angular width parameter
    cone_mask = 0.5 * (1 + np.tanh(k_parallel / (sigma * cone_width)))
    
    # Paul envelope along the radial direction
    # Peaked at k0, decaying for larger/smaller k
    # Using log-normal-like shape centered at k0
    with np.errstate(divide='ignore', invalid='ignore'):
        # Avoid log(0)
        log_k = np.where(k_norm > 1e-10, np.log(k_norm / k0), -100)
        radial_envelope = np.exp(-0.5 * (log_k / (0.5))**2)
        radial_envelope = np.where(k_norm > 1e-10, radial_envelope, 0)
    
    # Angular selectivity (Gaussian in perpendicular direction)
    angular_envelope = np.exp(-0.5 * (k_perp / sigma)**2)
    
    # Combine: cone * radial * angular
    psi_hat = cone_mask * radial_envelope * angular_envelope
    
    # Normalize to unit energy
    energy = np.sum(np.abs(psi_hat)**2)
    if energy > 1e-10:
        psi_hat = psi_hat / np.sqrt(energy)
    
    return psi_hat.astype(np.complex128)


def gaussian_lowpass_2d_fourier(M: int, N: int, sigma: float) -> np.ndarray:
    """
    Create isotropic Gaussian lowpass filter in Fourier domain.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    sigma : float
        Cutoff frequency (larger = more lowpass)
        
    Returns
    -------
    phi_hat : ndarray, shape (M, N), real
        Lowpass filter in Fourier domain
    """
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(M)
    KX, KY = np.meshgrid(kx, ky)
    k_norm_sq = KX**2 + KY**2
    
    phi_hat = np.exp(-k_norm_sq / (2 * sigma**2))
    
    # Normalize
    phi_hat = phi_hat / np.sqrt(np.sum(phi_hat**2))
    
    return phi_hat.astype(np.complex128)


def filter_bank_2d(
    M: int, 
    N: int, 
    J: int = 4, 
    L: int = 8,
    m: int = 4,
) -> Dict:
    """
    Build 2D two-channel Paul wavelet filter bank for HST.
    
    Creates wavelets at J scales and L orientations, with both H+ and H-
    channels at each (j, theta) pair. This ensures full coverage of the
    Fourier plane for complex signals.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    J : int
        Number of scales (octaves)
    L : int
        Number of orientations per half-plane
        Total orientations = L (covering [0, 2π) with both H+ and H-)
    m : int
        Paul wavelet order
        
    Returns
    -------
    filters : dict
        'psi': list of wavelet dicts, each with:
            'filter': ndarray (M, N) - wavelet in Fourier domain
            'j': int - scale index
            'theta': int - orientation index
            'channel': str - 'H+' or 'H-'
        'phi': dict with:
            'filter': ndarray (M, N) - lowpass in Fourier domain
            'j': int - J (coarsest scale)
        'info': dict with metadata
    """
    filters = {
        'psi': [],
        'phi': None,
        'info': {
            'M': M, 'N': N, 'J': J, 'L': L, 'm': m,
            'n_wavelets': 0,
            'design': 'two_channel_paul',
        }
    }
    
    # Mother wavelets: J scales × L orientations × 2 channels
    for j in range(J):
        for theta_idx in range(L):
            # Orientation angle covering [0, 2π)
            theta = 2 * np.pi * theta_idx / L
            
            # H+ channel (positive frequency cone)
            psi_plus = paul_wavelet_2d_fourier(M, N, j, theta, m, positive=True)
            filters['psi'].append({
                'filter': psi_plus,
                'j': j,
                'theta': theta_idx,
                'theta_rad': theta,
                'channel': 'H+',
            })
            
            # H- channel (negative frequency cone)
            psi_minus = paul_wavelet_2d_fourier(M, N, j, theta, m, positive=False)
            filters['psi'].append({
                'filter': psi_minus,
                'j': j,
                'theta': theta_idx,
                'theta_rad': theta,
                'channel': 'H-',
            })
    
    filters['info']['n_wavelets'] = len(filters['psi'])
    
    # Father wavelet (lowpass) - covers DC
    sigma_phi = 0.05 * (2 ** J)  # Cutoff decreases with J
    phi = gaussian_lowpass_2d_fourier(M, N, sigma_phi)
    filters['phi'] = {
        'filter': phi,
        'j': J,
    }
    
    return filters


def verify_littlewood_paley(filters: Dict, tol: float = 0.1) -> Dict:
    """
    Verify Littlewood-Paley condition (partition of unity).
    
    For a proper tight frame: Σ|ψ_j,θ(k)|² + |φ(k)|² ≈ 1 for all k
    
    Parameters
    ----------
    filters : dict
        Filter bank from filter_bank_2d()
    tol : float
        Tolerance for deviation from 1
        
    Returns
    -------
    result : dict
        'sum_sq': ndarray - sum of squared magnitudes at each frequency
        'min': float - minimum value
        'max': float - maximum value  
        'mean': float - mean value
        'passed': bool - whether within tolerance
    """
    M = filters['info']['M']
    N = filters['info']['N']
    
    # Sum |ψ|² over all wavelets
    sum_sq = np.zeros((M, N), dtype=np.float64)
    
    for psi in filters['psi']:
        sum_sq += np.abs(psi['filter'])**2
    
    # Add lowpass
    sum_sq += np.abs(filters['phi']['filter'])**2
    
    result = {
        'sum_sq': sum_sq,
        'min': float(np.min(sum_sq)),
        'max': float(np.max(sum_sq)),
        'mean': float(np.mean(sum_sq)),
        'std': float(np.std(sum_sq)),
        'passed': bool(np.abs(sum_sq - 1.0).max() < tol),
    }
    
    return result


def normalize_filterbank_littlewood_paley(filters: Dict) -> Dict:
    """
    Normalize filter bank to satisfy Littlewood-Paley condition.
    
    Divides each filter by sqrt(sum_sq) so that Σ|ψ|² + |φ|² = 1.
    
    Parameters
    ----------
    filters : dict
        Filter bank from filter_bank_2d()
        
    Returns
    -------
    filters_normalized : dict
        Normalized filter bank
    """
    M = filters['info']['M']
    N = filters['info']['N']
    
    # Compute current sum of squares
    sum_sq = np.zeros((M, N), dtype=np.float64)
    for psi in filters['psi']:
        sum_sq += np.abs(psi['filter'])**2
    sum_sq += np.abs(filters['phi']['filter'])**2
    
    # Normalization factor
    norm_factor = np.sqrt(np.maximum(sum_sq, 1e-10))
    
    # Create normalized copy
    filters_norm = {
        'psi': [],
        'phi': None,
        'info': filters['info'].copy(),
    }
    filters_norm['info']['normalized'] = True
    
    for psi in filters['psi']:
        filters_norm['psi'].append({
            'filter': psi['filter'] / norm_factor,
            'j': psi['j'],
            'theta': psi['theta'],
            'theta_rad': psi['theta_rad'],
            'channel': psi['channel'],
        })
    
    filters_norm['phi'] = {
        'filter': filters['phi']['filter'] / norm_factor,
        'j': filters['phi']['j'],
    }
    
    return filters_norm


# =============================================================================
# Convenience functions
# =============================================================================

def create_filter_bank(
    M: int, 
    N: int, 
    J: int = 4, 
    L: int = 8,
    normalize: bool = True,
) -> Dict:
    """
    Create 2D HST filter bank with optional Littlewood-Paley normalization.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    J : int
        Number of scales
    L : int
        Number of orientations
    normalize : bool
        If True, normalize to satisfy Littlewood-Paley
        
    Returns
    -------
    filters : dict
        Filter bank ready for use
    """
    filters = filter_bank_2d(M, N, J, L)
    
    if normalize:
        filters = normalize_filterbank_littlewood_paley(filters)
    
    return filters


def compute_padding(M: int, N: int, J: int) -> Tuple[int, int]:
    """Compute padded size to avoid boundary effects."""
    M_padded = ((M + 2**J) // 2**J + 1) * 2**J
    N_padded = ((N + 2**J) // 2**J + 1) * 2**J
    return M_padded, N_padded
