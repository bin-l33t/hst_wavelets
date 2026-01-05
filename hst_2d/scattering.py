"""
2D Heisenberg Scattering Transform - Core Implementation

This is the key file that implements Glinsky's HST in 2D:
- Uses wavelet cascade like Mallat/Kymatio
- Replaces modulus with Glinsky's R = i*ln(R0) rectifier  
- Maintains complex data flow throughout (not real like standard scattering)

Key difference from Kymatio:
    Kymatio: signal → wavelet → |·| (modulus) → wavelet → |·| → average
    HST:     signal → wavelet → R(·) (complex) → wavelet → R(·) → average
    
The R rectifier preserves phase information that modulus discards.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def glinsky_R0(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Glinsky's R0 mapping using inverse Joukowsky transform.
    
    R0(z) = -i * h^{-1}(2z/π)
    
    where h(z) = (z + 1/z) / 2 is the Joukowsky transform.
    """
    w = 2.0 * z / np.pi
    # h^{-1}(w): solve z^2 - 2wz + 1 = 0 → z = w ± sqrt(w^2 - 1)
    disc = torch.sqrt(w**2 - 1.0 + 0j)
    z1 = w + disc
    z2 = w - disc
    # Choose root closer to unit circle
    h_inv = torch.where(torch.abs(torch.abs(z1) - 1) < torch.abs(torch.abs(z2) - 1), z1, z2)
    return -1j * h_inv


def glinsky_R(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Glinsky's full R rectifier.
    
    R(z) = i * ln(R0(z))
    
    This is the nonlinearity that replaces modulus in the scattering cascade.
    Unlike modulus which discards phase, R preserves it in a gauge-invariant way.
    
    Output: R(z) ≈ -arg(R0(z)) + i*ln|R0(z)|
    """
    R0_z = glinsky_R0(z, eps)
    # Add small imaginary part for numerical stability of log
    return 1j * torch.log(R0_z + eps * 1j)


def simple_R(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Simplified R mapping (without Joukowsky).
    
    R(z) = i * ln(z)
    
    Useful for testing. This is what Glinsky uses in the limit.
    """
    return 1j * torch.log(z + eps)


def lift_radial(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Lift signal away from origin to avoid log singularity.
    
    r̃ = sqrt(|z|² + eps²), angle unchanged
    """
    r = torch.abs(z)
    r_floor = torch.sqrt(r**2 + eps**2)
    scale = r_floor / torch.clamp(r, min=1e-30)
    return z * scale


class HST2D:
    """
    2D Heisenberg Scattering Transform.
    
    This implements Glinsky's HST using the Kymatio cascade structure:
    
    Order 0: S0 = x * phi  (lowpass average)
    Order 1: U1 = x * psi_j,θ  → R(U1) → S1 = R(U1) * phi
    Order 2: U2 = R(U1) * psi_j',θ' → R(U2) → S2 = R(U2) * phi
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of input
    J : int
        Number of scales
    L : int
        Number of orientations
    max_order : int
        Maximum scattering order (1 or 2)
    rectifier : str
        'glinsky' for full R, 'simple' for i*ln(z), 'modulus' for |z|
    """
    
    def __init__(
        self,
        M: int,
        N: int,
        J: int = 4,
        L: int = 8,
        max_order: int = 2,
        rectifier: str = 'glinsky',
        device: torch.device = None,
    ):
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.max_order = max_order
        self.rectifier = rectifier
        self.device = device or torch.device('cpu')
        
        # Build filter bank
        from hst_2d.filter_bank import filter_bank
        self._filters_np = filter_bank(M, N, J, L)
        
        # Convert to torch tensors
        self.phi = self._convert_filter(self._filters_np['phi'])
        self.psi = [self._convert_filter(p) for p in self._filters_np['psi']]
        
    def _convert_filter(self, filt_dict: dict) -> dict:
        """Convert numpy filter dict to torch."""
        result = {'j': filt_dict['j']}
        if 'theta' in filt_dict:
            result['theta'] = filt_dict['theta']
        result['levels'] = [
            torch.from_numpy(lvl.astype(np.complex128)).to(self.device)
            for lvl in filt_dict['levels']
        ]
        return result
    
    def _rectify(self, z: torch.Tensor) -> torch.Tensor:
        """Apply rectifier based on setting."""
        # Lift away from origin first
        z_lifted = lift_radial(z)
        
        if self.rectifier == 'glinsky':
            return glinsky_R(z_lifted)
        elif self.rectifier == 'simple':
            return simple_R(z_lifted)
        elif self.rectifier == 'modulus':
            # Standard scattering - returns REAL
            return torch.abs(z_lifted)
        else:
            raise ValueError(f"Unknown rectifier: {self.rectifier}")
    
    def _cdgmm(self, x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """Complex-domain element-wise multiplication."""
        return x * filt
    
    def _subsample_fourier(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Subsample in Fourier domain by factor k.
        
        Subsampling in space = periodization in Fourier.
        """
        if k == 1:
            return x
        
        M, N = x.shape[-2:]
        M_new, N_new = M // k, N // k
        
        # Reshape and average over k×k blocks
        y = x.view(*x.shape[:-2], k, M_new, k, N_new)
        y = y.mean(dim=(-4, -2))  # Average over the k dimensions
        
        return y
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute 2D HST coefficients.
        
        Parameters
        ----------
        x : torch.Tensor
            Input complex field, shape (..., M, N)
            
        Returns
        -------
        S : dict
            'S0': Order 0 coefficients (lowpass)
            'S1': Order 1 coefficients, list of dicts with 'coef', 'j', 'theta'
            'S2': Order 2 coefficients, list of dicts with 'coef', 'j1', 'j2', etc.
        """
        # Ensure complex
        if not x.is_complex():
            x = x.to(torch.complex128)
        x = x.to(self.device)
        
        results = {'S0': [], 'S1': [], 'S2': []}
        
        # FFT of input
        U_0_hat = torch.fft.fft2(x)
        
        # === Order 0: Lowpass average ===
        S_0_hat = self._cdgmm(U_0_hat, self.phi['levels'][0])
        S_0_hat = self._subsample_fourier(S_0_hat, 2**self.J)
        S_0 = torch.fft.ifft2(S_0_hat)
        results['S0'] = S_0
        
        if self.max_order < 1:
            return results
        
        # === Order 1: Wavelet → Rectify → Average ===
        U1_rectified = {}  # Store for order 2
        
        for n1, psi1 in enumerate(self.psi):
            j1 = psi1['j']
            theta1 = psi1['theta']
            
            # Convolve with wavelet
            U_1_hat = self._cdgmm(U_0_hat, psi1['levels'][0])
            
            # Subsample in Fourier
            if j1 > 0:
                U_1_hat = self._subsample_fourier(U_1_hat, 2**j1)
            
            # Back to spatial domain
            U_1 = torch.fft.ifft2(U_1_hat)
            
            # === KEY DIFFERENCE: Apply R instead of modulus ===
            U_1_rect = self._rectify(U_1)
            
            # Store for order 2
            U1_rectified[n1] = U_1_rect
            
            # Average with lowpass (use appropriate resolution level)
            # For complex R output, we stay in complex domain
            if self.rectifier == 'modulus':
                # Standard scattering: modulus output is real, use rfft logic
                U_1_rect_hat = torch.fft.fft2(U_1_rect.real.to(torch.complex128))
            else:
                # HST: R output is complex, use full fft
                U_1_rect_hat = torch.fft.fft2(U_1_rect)
            
            S_1_hat = self._cdgmm(U_1_rect_hat, self.phi['levels'][min(j1, len(self.phi['levels'])-1)])
            S_1_hat = self._subsample_fourier(S_1_hat, 2**(self.J - j1))
            S_1 = torch.fft.ifft2(S_1_hat)
            
            results['S1'].append({
                'coef': S_1,
                'j': j1,
                'theta': theta1,
                'n': n1,
            })
        
        if self.max_order < 2:
            return results
        
        # === Order 2: Second wavelet → Rectify → Average ===
        for n1, psi1 in enumerate(self.psi):
            j1 = psi1['j']
            theta1 = psi1['theta']
            
            U_1_rect = U1_rectified[n1]
            
            # FFT of rectified signal
            if self.rectifier == 'modulus':
                U_1_rect_hat = torch.fft.fft2(U_1_rect.real.to(torch.complex128))
            else:
                U_1_rect_hat = torch.fft.fft2(U_1_rect)
            
            for n2, psi2 in enumerate(self.psi):
                j2 = psi2['j']
                theta2 = psi2['theta']
                
                # Frequency ordering constraint: j2 > j1
                if j2 <= j1:
                    continue
                
                # Convolve with second wavelet
                # Use appropriate resolution level
                level_idx = min(j1, len(psi2['levels']) - 1)
                U_2_hat = self._cdgmm(U_1_rect_hat, psi2['levels'][level_idx])
                U_2_hat = self._subsample_fourier(U_2_hat, 2**(j2 - j1))
                
                # Back to spatial
                U_2 = torch.fft.ifft2(U_2_hat)
                
                # Rectify
                U_2_rect = self._rectify(U_2)
                
                # Average
                if self.rectifier == 'modulus':
                    U_2_rect_hat = torch.fft.fft2(U_2_rect.real.to(torch.complex128))
                else:
                    U_2_rect_hat = torch.fft.fft2(U_2_rect)
                
                level_idx_phi = min(j2, len(self.phi['levels']) - 1)
                S_2_hat = self._cdgmm(U_2_rect_hat, self.phi['levels'][level_idx_phi])
                S_2_hat = self._subsample_fourier(S_2_hat, 2**(self.J - j2))
                S_2 = torch.fft.ifft2(S_2_hat)
                
                results['S2'].append({
                    'coef': S_2,
                    'j1': j1,
                    'j2': j2,
                    'theta1': theta1,
                    'theta2': theta2,
                    'n1': n1,
                    'n2': n2,
                })
        
        return results
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract flat feature vector from scattering coefficients.
        
        Concatenates |S0|, |S1|, |S2| into a single vector.
        """
        S = self.forward(x)
        
        features = []
        
        # S0
        features.append(torch.abs(S['S0']).flatten())
        
        # S1
        for s1 in S['S1']:
            features.append(torch.abs(s1['coef']).flatten())
        
        # S2
        for s2 in S['S2']:
            features.append(torch.abs(s2['coef']).flatten())
        
        return torch.cat(features)


def create_hst_2d(M, N, J=4, L=8, max_order=2, rectifier='glinsky', device=None):
    """Factory function to create HST2D instance."""
    return HST2D(M, N, J=J, L=L, max_order=max_order, rectifier=rectifier, device=device)
