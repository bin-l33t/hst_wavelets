"""
Conformal Mappings for Heisenberg Scattering Transform

Implements Glinsky's R mapping using the Joukowsky transform.

Theory:
    R₀(z) = -i · h⁻¹(2z/π)
    R(z) = i · ln(R₀(z))
    
    where h(z) = (z + 1/z) / 2 is the Joukowsky transform.
    
    R "flattens" the complex plane so that group action becomes linear.

References:
    - Glinsky (2025), Section III, Eq. 12
"""

import numpy as np
from typing import Union

# Try torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def joukowsky(z: np.ndarray) -> np.ndarray:
    """
    Joukowsky transform: h(z) = (z + 1/z) / 2
    
    Maps circles to ellipses. Maps unit circle to [-1, 1].
    """
    return (z + 1.0 / z) / 2.0


def joukowsky_inverse(w: np.ndarray) -> np.ndarray:
    """
    Inverse Joukowsky transform.
    
    Given w = (z + 1/z) / 2, solve for z:
    z² - 2wz + 1 = 0
    z = w ± √(w² - 1)
    
    We choose the root closer to the unit circle.
    """
    disc = np.sqrt(w**2 - 1.0 + 0j)
    z1 = w + disc
    z2 = w - disc
    # Choose root closer to unit circle
    return np.where(np.abs(np.abs(z1) - 1) < np.abs(np.abs(z2) - 1), z1, z2)


def glinsky_R0(z: np.ndarray) -> np.ndarray:
    """
    Glinsky's R₀ mapping.
    
    R₀(z) = -i · h⁻¹(2z/π)
    
    Parameters
    ----------
    z : ndarray
        Complex input
        
    Returns
    -------
    R0_z : ndarray
        Transformed values
    """
    return -1j * joukowsky_inverse(2.0 * z / np.pi)


def glinsky_R(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Glinsky's full R mapping.
    
    R(z) = i · ln(R₀(z))
    
    This is the nonlinearity used in the Heisenberg Scattering Transform.
    It combines modulus and phase information:
        R(z) ≈ -arg(R₀(z)) + i·ln|R₀(z)|
    
    Parameters
    ----------
    z : ndarray
        Complex input signal
    eps : float, optional
        Small value added for numerical stability of log
        
    Returns
    -------
    R_z : ndarray
        Transformed values
        
    Notes
    -----
    Branch cut warning: For signals that wrap around the origin,
    discontinuities will appear. Use two-channel filter bank
    for robust reconstruction.
    """
    R0_z = glinsky_R0(z)
    return 1j * np.log(R0_z + eps * 1j)


def glinsky_R_inverse(w: np.ndarray) -> np.ndarray:
    """
    Inverse of Glinsky's R mapping.
    
    Given w = i·ln(R₀(z)), solve for z:
        R₀(z) = exp(-i·w)
        -i·h⁻¹(2z/π) = exp(-i·w)
        h⁻¹(2z/π) = i·exp(-i·w)
        2z/π = h(i·exp(-i·w))
        z = (π/2)·h(i·exp(-i·w))
    
    Parameters
    ----------
    w : ndarray
        Values in R-space
        
    Returns
    -------
    z : ndarray
        Original values
    """
    R0 = np.exp(-1j * w)
    inner = 1j * R0
    return (np.pi / 2.0) * joukowsky(inner)


def simple_R(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Simplified R mapping (without Joukowsky).
    
    R(z) = i · ln(z)
    
    Useful for testing and comparison.
    """
    return 1j * np.log(z + eps)


def simple_R_inverse(w: np.ndarray) -> np.ndarray:
    """
    Inverse of simple R mapping.
    
    Given w = i·ln(z), we have z = exp(-i·w).
    """
    return np.exp(-1j * w)


def simple_R_unwrapped(z: np.ndarray) -> np.ndarray:
    """
    R mapping with unwrapped phase (for continuous dynamics).
    
    R(z) = -unwrap(arg(z)) + i·ln|z|
    
    This avoids branch cut discontinuities and reveals the
    linear "geodesic" dynamics that Glinsky describes.
    
    Use this for:
    - Physical interpretation of dynamics
    - Visualizing phase flow
    - Verifying "linearization" claims
    
    Note: simple_R (with branch cuts) is fine for reconstruction
    since the two-channel filter bank handles discontinuities.
    """
    phase_unwrapped = np.unwrap(np.angle(z))
    log_magnitude = np.log(np.abs(z) + 1e-12)
    return -phase_unwrapped + 1j * log_magnitude


# =============================================================================
# Verification functions
# =============================================================================

def verify_R_inverse(n_samples: int = 1000, eps: float = 1e-6) -> dict:
    """
    Verify that R⁻¹(R(z)) = z.
    
    Returns
    -------
    results : dict
        Contains max_error, mean_error, passed (bool)
    """
    np.random.seed(42)
    
    # Generate test points (avoid origin and branch cuts)
    z = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    z = z + 2.0  # Shift away from origin
    
    # Round trip
    w = glinsky_R(z)
    z_rec = glinsky_R_inverse(w)
    
    errors = np.abs(z - z_rec)
    
    return {
        'max_error': float(errors.max()),
        'mean_error': float(errors.mean()),
        'passed': errors.max() < eps,
    }


def verify_simple_R_inverse(n_samples: int = 1000, eps: float = 1e-6) -> dict:
    """Verify simple R inverse."""
    np.random.seed(42)
    z = np.random.randn(n_samples) + 1j * np.random.randn(n_samples) + 2.0
    
    w = simple_R(z)
    z_rec = simple_R_inverse(w)
    
    errors = np.abs(z - z_rec)
    return {
        'max_error': float(errors.max()),
        'mean_error': float(errors.mean()),
        'passed': errors.max() < eps,
    }


# =============================================================================
# PyTorch versions
# =============================================================================

if HAS_TORCH:
    def joukowsky_torch(z: torch.Tensor) -> torch.Tensor:
        return (z + 1.0 / z) / 2.0
    
    def joukowsky_inverse_torch(w: torch.Tensor) -> torch.Tensor:
        disc = torch.sqrt(w**2 - 1.0 + 0j)
        z1 = w + disc
        z2 = w - disc
        return torch.where(
            torch.abs(torch.abs(z1) - 1) < torch.abs(torch.abs(z2) - 1),
            z1, z2
        )
    
    def glinsky_R_torch(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        R0_z = -1j * joukowsky_inverse_torch(2.0 * z / np.pi)
        return 1j * torch.log(R0_z + eps * 1j)
    
    def glinsky_R_inverse_torch(w: torch.Tensor) -> torch.Tensor:
        R0 = torch.exp(-1j * w)
        inner = 1j * R0
        return (np.pi / 2.0) * joukowsky_torch(inner)
