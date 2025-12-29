"""
PyTorch Heisenberg Scattering Transform

Mirrors the NumPy implementation in hst/scattering.py but uses torch operations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from hst_torch.filter_bank import (
    two_channel_paul_filterbank_torch,
    forward_transform_torch,
    lowpass_torch,
)


@dataclass
class HSTOutputTorch:
    """Container for HST output (PyTorch version)."""
    paths: Dict[Tuple[int, ...], torch.Tensor]
    T: int
    max_order: int
    
    def order(self, m: int) -> Dict[Tuple[int, ...], torch.Tensor]:
        """Get all paths of a specific order."""
        return {p: c for p, c in self.paths.items() if len(p) == m}
    
    def to_numpy(self) -> 'HSTOutputNumpy':
        """Convert to numpy arrays."""
        from hst.scattering import HSTOutput
        np_paths = {p: c.cpu().numpy() for p, c in self.paths.items()}
        return HSTOutput(paths=np_paths, T=self.T, max_order=self.max_order)


class HeisenbergScatteringTransformTorch:
    """
    PyTorch implementation of Heisenberg Scattering Transform.
    
    This is a forward-only implementation for inference.
    Matches the NumPy version's output format for parity testing.
    """
    
    def __init__(
        self,
        T: int,
        J: int = 4,
        Q: int = 2,
        max_order: int = 2,
        lifting: str = 'radial_floor',
        epsilon: float = 1e-8,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex128,
    ):
        """
        Initialize HST.
        
        Parameters
        ----------
        T : int
            Signal length
        J : int
            Number of octaves
        Q : int
            Wavelets per octave
        max_order : int
            Maximum scattering order
        lifting : str
            Lifting strategy ('radial_floor' or 'joukowsky')
        epsilon : float
            Minimum distance from origin for lifting
        device : torch.device
            Target device (cpu or cuda)
        dtype : torch.dtype
            Complex dtype
        """
        self.T = T
        self.J = J
        self.Q = Q
        self.max_order = max_order
        self.lifting = lifting
        self.epsilon = epsilon
        
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.dtype = dtype
        
        # Create filterbank
        self.filters, self.filter_info = two_channel_paul_filterbank_torch(
            T, J, Q, device=device, dtype=dtype
        )
        self.n_mothers = self.filter_info['n_mothers']
        self.phi_hat = self.filters[-1]  # Lowpass (father)
        self.psi_hats = self.filters[:-1]  # Bandpass (mothers)
    
    def _lift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Lift signal away from origin.
        
        Parameters
        ----------
        z : torch.Tensor
            Complex signal
            
        Returns
        -------
        z_lifted : torch.Tensor
            Signal with |z| >= epsilon
        """
        if self.lifting == 'radial_floor':
            # Simple radial projection
            r = torch.abs(z)
            # Where r < epsilon, scale up to epsilon
            scale = torch.where(
                r < self.epsilon,
                self.epsilon / (r + 1e-16),
                torch.ones_like(r)
            )
            return z * scale
        elif self.lifting == 'joukowsky':
            # Joukowsky-style: z + epsilon²/z (pushes away from origin)
            r = torch.abs(z)
            safe_z = torch.where(
                r < self.epsilon,
                z + self.epsilon * torch.exp(1j * torch.angle(z)),
                z
            )
            return safe_z
        else:
            return z
    
    def _R(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply R = i * log activation.
        
        R(z) = i * ln(z) = i * (ln|z| + i*arg(z)) = -arg(z) + i*ln|z|
        
        Parameters
        ----------
        z : torch.Tensor
            Complex signal (should be lifted away from origin)
            
        Returns
        -------
        w : torch.Tensor
            Transformed signal
        """
        # Lift first
        z_lifted = self._lift(z)
        
        # R = i * log(z)
        # log(z) = ln|z| + i*arg(z)
        # i * log(z) = i*ln|z| - arg(z) = -arg(z) + i*ln|z|
        ln_r = torch.log(torch.abs(z_lifted))
        theta = torch.angle(z_lifted)
        
        return -theta + 1j * ln_r
    
    def forward(
        self,
        x: torch.Tensor,
        max_order: Optional[int] = None,
    ) -> HSTOutputTorch:
        """
        Compute HST coefficients.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (T,)
        max_order : int, optional
            Override max_order for this call
            
        Returns
        -------
        output : HSTOutputTorch
            Container with path coefficients
        """
        if max_order is None:
            max_order = self.max_order
        
        # Ensure input is on correct device and dtype
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
        
        paths = {}
        
        # Order 0: lowpass of input
        S0 = lowpass_torch(x, self.phi_hat)
        paths[()] = S0
        
        if max_order == 0:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 1: wavelet transform + R + lowpass
        U1 = {}  # Store for higher orders
        
        for j, psi_hat in enumerate(self.psi_hats):
            # Wavelet modulation
            x_hat = torch.fft.fft(x)
            W_x = torch.fft.ifft(x_hat * psi_hat)
            
            # Apply R activation
            R_W = self._R(W_x)
            
            # Store for recursion
            U1[j] = R_W
            
            # Lowpass for output
            S1 = lowpass_torch(R_W, self.phi_hat)
            paths[(j,)] = S1
        
        if max_order == 1:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 2: cascade
        U2 = {}
        
        for j1, R1 in U1.items():
            for j2, psi_hat in enumerate(self.psi_hats):
                if j2 <= j1:
                    continue  # Frequency ordering constraint
                
                # Wavelet of R1
                R1_hat = torch.fft.fft(R1)
                W_R1 = torch.fft.ifft(R1_hat * psi_hat)
                
                # Apply R activation
                R_W_R1 = self._R(W_R1)
                
                # Store for order 3
                U2[(j1, j2)] = R_W_R1
                
                # Lowpass for output
                S2 = lowpass_torch(R_W_R1, self.phi_hat)
                paths[(j1, j2)] = S2
        
        if max_order == 2:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 3
        for (j1, j2), R2 in U2.items():
            for j3, psi_hat in enumerate(self.psi_hats):
                if j3 <= j2:
                    continue
                
                R2_hat = torch.fft.fft(R2)
                W_R2 = torch.fft.ifft(R2_hat * psi_hat)
                R_W_R2 = self._R(W_R2)
                S3 = lowpass_torch(R_W_R2, self.phi_hat)
                paths[(j1, j2, j3)] = S3
        
        return HSTOutputTorch(paths=paths, T=self.T, max_order=min(max_order, 3))
    
    def to(self, device: torch.device) -> 'HeisenbergScatteringTransformTorch':
        """Move filterbank to device."""
        self.device = device
        self.filters = [f.to(device) for f in self.filters]
        self.phi_hat = self.filters[-1]
        self.psi_hats = self.filters[:-1]
        return self
