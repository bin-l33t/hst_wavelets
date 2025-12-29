"""
PyTorch Heisenberg Scattering Transform

Mirrors the NumPy implementation in hst/scattering.py exactly.
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
    
    def to_numpy(self):
        """Convert to numpy arrays."""
        np_paths = {p: c.cpu().numpy() for p, c in self.paths.items()}
        return np_paths


class HeisenbergScatteringTransformTorch:
    """
    PyTorch implementation of Heisenberg Scattering Transform.
    
    This mirrors the NumPy version's output format for parity testing.
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
            Lifting strategy ('radial_floor' supported)
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
        
        # Create filterbank (two-channel: H+ and H- mothers + father)
        self.filters, self.filter_info = two_channel_paul_filterbank_torch(
            T, J, Q, device=device, dtype=dtype
        )
        
        # Number of mother wavelets (H+ and H- combined)
        self.n_mothers = self.filter_info['n_mothers']  # = 2 * J * Q
        
    def _lift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Lift signal away from origin using radial floor.
        
        r̃ = sqrt(|z|² + eps²), angle unchanged
        """
        if self.lifting == 'radial_floor':
            r = torch.abs(z)
            r_floor = torch.sqrt(r**2 + self.epsilon**2)
            tiny = 1e-300
            scale = r_floor / torch.clamp(r, min=tiny)
            return z * scale
        else:
            # Simple shift fallback
            return z + self.epsilon
    
    def _R(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply R = i * log activation.
        
        R(z) = i * ln(z) = i * (ln|z| + i*arg(z)) = -arg(z) + i*ln|z|
        """
        ln_r = torch.log(torch.abs(z))
        theta = torch.angle(z)
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
        
        # Lift input signal
        x_lifted = self._lift(x)
        
        # Order 0: All filter outputs
        coeffs_all = forward_transform_torch(x_lifted, self.filters)
        paths[()] = coeffs_all[-1]  # Father coefficient (lowpass)
        
        if max_order == 0:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 1: Apply R to each mother wavelet output
        W1_dict = {}  # Store for order 2
        
        for j1 in range(self.n_mothers):
            U1 = coeffs_all[j1]
            U1_lifted = self._lift(U1)
            W1 = self._R(U1_lifted)
            paths[(j1,)] = W1
            W1_dict[j1] = W1
        
        if max_order == 1:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 2: Cascade from each W1
        W2_dict = {}
        
        for j1 in range(self.n_mothers):
            W1 = W1_dict[j1]
            coeffs_W1 = forward_transform_torch(W1, self.filters)
            
            for j2 in range(self.n_mothers):
                if j2 <= j1:
                    continue  # Frequency ordering constraint
                
                U2 = coeffs_W1[j2]
                U2_lifted = self._lift(U2)
                W2 = self._R(U2_lifted)
                paths[(j1, j2)] = W2
                W2_dict[(j1, j2)] = W2
        
        if max_order == 2:
            return HSTOutputTorch(paths=paths, T=self.T, max_order=max_order)
        
        # Order 3+: Recursive
        for order in range(3, max_order + 1):
            prev_paths = {k: v for k, v in paths.items() if len(k) == order - 1}
            
            for prev_path, W_prev in prev_paths.items():
                coeffs_prev = forward_transform_torch(W_prev, self.filters)
                j_prev = prev_path[-1]
                
                for j_new in range(self.n_mothers):
                    if j_new <= j_prev:
                        continue
                    
                    U_new = coeffs_prev[j_new]
                    U_new_lifted = self._lift(U_new)
                    W_new = self._R(U_new_lifted)
                    new_path = prev_path + (j_new,)
                    paths[new_path] = W_new
        
        return HSTOutputTorch(paths=paths, T=self.T, max_order=min(max_order, 3))
    
    def to(self, device: torch.device) -> 'HeisenbergScatteringTransformTorch':
        """Move filterbank to device."""
        self.device = device
        self.filters = [f.to(device) for f in self.filters]
        return self
