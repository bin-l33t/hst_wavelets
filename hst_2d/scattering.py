"""
2D Heisenberg Scattering Transform - Core Implementation

This implements Glinsky's HST as a proper wavelet scattering cascade:
    signal → [wavelet convolution → R rectifier]^n → lowpass average

Key features:
1. Two-channel (H+/H-) Paul wavelets for full Fourier plane coverage
2. Glinsky's R = i·ln(R₀) rectifier preserving complex/phase information
3. Complex data flow throughout (no Hermitian symmetry assumptions)
4. Littlewood-Paley normalized filter bank

Architecture:
    Order 0: S₀ = x * φ  (lowpass average)
    Order 1: S₁ = R(x * ψ_{j,θ,±}) * φ
    Order 2: S₂ = R(R(x * ψ₁) * ψ₂) * φ
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .filter_bank import create_filter_bank, verify_littlewood_paley


# =============================================================================
# Glinsky Rectifier Functions
# =============================================================================

def joukowsky_inverse_torch(w: torch.Tensor) -> torch.Tensor:
    """
    Inverse Joukowsky transform: solve z² - 2wz + 1 = 0 for z.
    
    h(z) = (z + 1/z) / 2  →  h⁻¹(w) = w ± √(w² - 1)
    Choose root closer to unit circle.
    """
    disc = torch.sqrt(w**2 - 1.0 + 0j)
    z1 = w + disc
    z2 = w - disc
    return torch.where(
        torch.abs(torch.abs(z1) - 1) < torch.abs(torch.abs(z2) - 1),
        z1, z2
    )


def glinsky_R0_torch(z: torch.Tensor) -> torch.Tensor:
    """
    Glinsky's R₀ mapping.
    
    R₀(z) = -i · h⁻¹(2z/π)
    """
    return -1j * joukowsky_inverse_torch(2.0 * z / np.pi)


def glinsky_R_torch(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Glinsky's full R rectifier.
    
    R(z) = i · ln(R₀(z))
    
    This maps complex → complex while preserving topological information.
    Unlike modulus which discards phase, R encodes it in the real part.
    
    R(z) ≈ -arg(R₀(z)) + i·ln|R₀(z)|
    """
    R0_z = glinsky_R0_torch(z)
    return 1j * torch.log(R0_z + eps * 1j)


def simple_R_torch(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Simplified R mapping (without Joukowsky transform).
    
    R(z) = i · ln(z)
    
    Useful for testing. This is what Glinsky's R approaches for small z.
    """
    return 1j * torch.log(z + eps * (1 + 1j))


def lift_radial_torch(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Lift signal away from origin to avoid log singularity.
    
    r̃ = √(|z|² + ε²), angle unchanged.
    """
    r = torch.abs(z)
    r_floor = torch.sqrt(r**2 + eps**2)
    scale = r_floor / torch.clamp(r, min=1e-30)
    return z * scale


# =============================================================================
# HST2D Class
# =============================================================================

@dataclass
class HST2DOutput:
    """Container for HST2D output coefficients."""
    S0: torch.Tensor  # Order 0 (lowpass)
    S1: List[Dict]    # Order 1 coefficients with metadata
    S2: List[Dict]    # Order 2 coefficients with metadata
    info: Dict        # Transform metadata


class HST2D:
    """
    2D Heisenberg Scattering Transform.
    
    Implements Glinsky's HST using proper wavelet cascade with:
    - Two-channel Paul wavelets (H+ and H-)
    - Glinsky's R = i·ln(R₀) rectifier
    - Full complex data flow
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of input
    J : int
        Number of scales (octaves)
    L : int  
        Number of orientations
    max_order : int
        Maximum scattering order (1 or 2)
    rectifier : str
        'glinsky': Full R = i·ln(R₀) 
        'simple': Simplified R = i·ln(z)
        'modulus': Standard |z| (for comparison)
    device : str or torch.device
        Compute device ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        M: int,
        N: int,
        J: int = 4,
        L: int = 8,
        max_order: int = 2,
        rectifier: str = 'glinsky',
        device: Union[str, torch.device] = 'cpu',
    ):
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.max_order = max_order
        self.rectifier = rectifier
        self.device = torch.device(device)
        
        # Build and normalize filter bank
        self._build_filters()
        
    def _build_filters(self):
        """Build two-channel Paul wavelet filter bank."""
        # Create NumPy filter bank
        filters_np = create_filter_bank(self.M, self.N, self.J, self.L, normalize=True)
        
        # Verify Littlewood-Paley
        lp_check = verify_littlewood_paley(filters_np)
        self.littlewood_paley_info = lp_check
        
        # Convert to PyTorch tensors
        self.psi = []
        for psi_dict in filters_np['psi']:
            self.psi.append({
                'filter': torch.from_numpy(psi_dict['filter']).to(self.device),
                'j': psi_dict['j'],
                'theta': psi_dict['theta'],
                'theta_rad': psi_dict['theta_rad'],
                'channel': psi_dict['channel'],
            })
        
        self.phi = {
            'filter': torch.from_numpy(filters_np['phi']['filter']).to(self.device),
            'j': filters_np['phi']['j'],
        }
        
        self.filter_info = filters_np['info']
        
    def _rectify(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the selected rectifier."""
        # First lift away from origin
        z_lifted = lift_radial_torch(z)
        
        if self.rectifier == 'glinsky':
            return glinsky_R_torch(z_lifted)
        elif self.rectifier == 'simple':
            return simple_R_torch(z_lifted)
        elif self.rectifier == 'modulus':
            # Standard scattering - returns magnitude (but as complex for consistency)
            return torch.abs(z_lifted).to(z.dtype)
        else:
            raise ValueError(f"Unknown rectifier: {self.rectifier}")
    
    def forward(self, x: torch.Tensor) -> HST2DOutput:
        """
        Compute 2D HST scattering coefficients.
        
        Parameters
        ----------
        x : torch.Tensor
            Input complex field, shape (..., M, N)
            
        Returns
        -------
        output : HST2DOutput
            S0: Order 0 coefficients
            S1: List of order 1 coefficient dicts
            S2: List of order 2 coefficient dicts
        """
        # Ensure complex tensor on correct device
        if not x.is_complex():
            x = x.to(torch.complex128)
        x = x.to(self.device)
        
        batch_shape = x.shape[:-2]
        x = x.reshape(-1, self.M, self.N)
        batch_size = x.shape[0]
        
        S1_list = []
        S2_list = []
        
        # FFT of input
        x_hat = torch.fft.fft2(x)
        
        # === Order 0: Lowpass average ===
        S0_hat = x_hat * self.phi['filter'].unsqueeze(0)
        S0 = torch.fft.ifft2(S0_hat)
        
        if self.max_order < 1:
            return HST2DOutput(
                S0=S0.reshape(batch_shape + (self.M, self.N)),
                S1=S1_list,
                S2=S2_list,
                info={'J': self.J, 'L': self.L, 'rectifier': self.rectifier}
            )
        
        # === Order 1: Wavelet → Rectify → Average ===
        U1_dict = {}  # Store rectified signals for order 2
        
        for idx, psi in enumerate(self.psi):
            j1 = psi['j']
            
            # Convolve with wavelet (in Fourier domain)
            U1_hat = x_hat * psi['filter'].unsqueeze(0)
            U1 = torch.fft.ifft2(U1_hat)
            
            # Apply rectifier
            U1_rect = self._rectify(U1)
            
            # Store for order 2
            U1_dict[idx] = U1_rect
            
            # Average with lowpass
            U1_rect_hat = torch.fft.fft2(U1_rect)
            S1_hat = U1_rect_hat * self.phi['filter'].unsqueeze(0)
            S1 = torch.fft.ifft2(S1_hat)
            
            S1_list.append({
                'coef': S1,
                'j': j1,
                'theta': psi['theta'],
                'channel': psi['channel'],
                'idx': idx,
            })
        
        if self.max_order < 2:
            # Reshape outputs
            for item in S1_list:
                item['coef'] = item['coef'].reshape(batch_shape + (self.M, self.N))
            
            return HST2DOutput(
                S0=S0.reshape(batch_shape + (self.M, self.N)),
                S1=S1_list,
                S2=S2_list,
                info={'J': self.J, 'L': self.L, 'rectifier': self.rectifier}
            )
        
        # === Order 2: Second cascade ===
        for idx1, psi1 in enumerate(self.psi):
            j1 = psi1['j']
            U1_rect = U1_dict[idx1]
            U1_rect_hat = torch.fft.fft2(U1_rect)
            
            for idx2, psi2 in enumerate(self.psi):
                j2 = psi2['j']
                
                # Frequency ordering: j2 > j1 (second wavelet at coarser scale)
                if j2 <= j1:
                    continue
                
                # Convolve with second wavelet
                U2_hat = U1_rect_hat * psi2['filter'].unsqueeze(0)
                U2 = torch.fft.ifft2(U2_hat)
                
                # Apply rectifier
                U2_rect = self._rectify(U2)
                
                # Average with lowpass
                U2_rect_hat = torch.fft.fft2(U2_rect)
                S2_hat = U2_rect_hat * self.phi['filter'].unsqueeze(0)
                S2 = torch.fft.ifft2(S2_hat)
                
                S2_list.append({
                    'coef': S2,
                    'j1': j1, 'j2': j2,
                    'theta1': psi1['theta'], 'theta2': psi2['theta'],
                    'channel1': psi1['channel'], 'channel2': psi2['channel'],
                    'idx1': idx1, 'idx2': idx2,
                })
        
        # Reshape outputs
        for item in S1_list:
            item['coef'] = item['coef'].reshape(batch_shape + (self.M, self.N))
        for item in S2_list:
            item['coef'] = item['coef'].reshape(batch_shape + (self.M, self.N))
        
        return HST2DOutput(
            S0=S0.reshape(batch_shape + (self.M, self.N)),
            S1=S1_list,
            S2=S2_list,
            info={'J': self.J, 'L': self.L, 'rectifier': self.rectifier,
                  'n_S1': len(S1_list), 'n_S2': len(S2_list)}
        )
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract U(1)-INVARIANT feature vector from scattering coefficients.
        
        Key insight: Global phase rotation x → x·e^{iφ} should not change features.
        
        U(1)-invariant quantities:
        - |c| (magnitude)
        - c_a * conj(c_b) for different channels (phase cancels)
        
        For chirality detection, cross-channel terms H+ * conj(H-) are crucial:
        these flip sign under conjugation (chirality flip) but are phase-invariant.
        
        Parameters
        ----------
        x : torch.Tensor
            Input field, shape (M, N) or (batch, M, N)
            
        Returns
        -------
        features : ndarray
            U(1)-invariant feature vector
        """
        output = self.forward(x)
        
        features = []
        
        # S0 features - ONLY magnitude (S0 = lowpass of raw x, contains phase)
        s0 = output.S0
        features.append(torch.abs(s0).mean().item())
        features.append(torch.abs(s0).std().item())
        
        # S1 features - magnitude only for individual coefficients
        for s1 in output.S1:
            c = s1['coef']
            features.append(torch.abs(c).mean().item())
        
        # S1 cross-channel features (H+ * conj(H-)) - these detect chirality!
        # Group S1 by (j, theta), then compute H+ * conj(H-)
        s1_by_jt = {}
        for s1 in output.S1:
            key = (s1['j'], s1['theta'])
            if key not in s1_by_jt:
                s1_by_jt[key] = {}
            s1_by_jt[key][s1['channel']] = s1['coef']
        
        for (j, theta), channels in s1_by_jt.items():
            if 'H+' in channels and 'H-' in channels:
                # Cross-term: H+ * conj(H-) is U(1)-invariant but chirality-sensitive
                cross = channels['H+'] * torch.conj(channels['H-'])
                # The REAL part is symmetric, IMAG part is antisymmetric under conjugation
                features.append(torch.real(cross).mean().item())  # Symmetric
                features.append(torch.imag(cross).mean().item())  # ANTISYMMETRIC - chirality!
        
        # S2 features - magnitude only
        for s2 in output.S2:
            c = s2['coef']
            features.append(torch.abs(c).mean().item())
        
        # S2 cross-channel features (more complex but same principle)
        # Group by (j1, j2, theta1, theta2) and look for channel pairs
        s2_by_path = {}
        for s2 in output.S2:
            key = (s2['j1'], s2['j2'], s2['theta1'], s2['theta2'])
            if key not in s2_by_path:
                s2_by_path[key] = {}
            ch_key = (s2['channel1'], s2['channel2'])
            s2_by_path[key][ch_key] = s2['coef']
        
        for path_key, channels in s2_by_path.items():
            # Look for complementary channel pairs
            for (ch1a, ch2a), coef_a in channels.items():
                for (ch1b, ch2b), coef_b in channels.items():
                    if (ch1a, ch2a) < (ch1b, ch2b):  # Avoid duplicates
                        cross = coef_a * torch.conj(coef_b)
                        features.append(torch.real(cross).mean().item())
                        features.append(torch.imag(cross).mean().item())
        
        return np.array(features)
    
    def extract_features_magnitude_only(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract ONLY magnitude features (simplest U(1)-invariant, but may lose chirality).
        
        Use this as a baseline to verify U(1) invariance is working.
        """
        output = self.forward(x)
        
        features = []
        
        # S0
        features.append(torch.abs(output.S0).mean().item())
        
        # S1
        for s1 in output.S1:
            features.append(torch.abs(s1['coef']).mean().item())
        
        # S2
        for s2 in output.S2:
            features.append(torch.abs(s2['coef']).mean().item())
        
        return np.array(features)
    
    def __repr__(self) -> str:
        return (f"HST2D(M={self.M}, N={self.N}, J={self.J}, L={self.L}, "
                f"max_order={self.max_order}, rectifier='{self.rectifier}', "
                f"n_wavelets={len(self.psi)})")


# =============================================================================
# Convenience factory function
# =============================================================================

def create_hst_2d(
    M: int,
    N: int,
    J: int = 4,
    L: int = 8,
    max_order: int = 2,
    rectifier: str = 'glinsky',
    device: str = 'cpu',
) -> HST2D:
    """
    Create HST2D instance.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions
    J : int
        Number of scales
    L : int
        Number of orientations  
    max_order : int
        Maximum scattering order
    rectifier : str
        'glinsky', 'simple', or 'modulus'
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    hst : HST2D
        Configured HST instance
    """
    return HST2D(M, N, J=J, L=L, max_order=max_order, 
                 rectifier=rectifier, device=device)
