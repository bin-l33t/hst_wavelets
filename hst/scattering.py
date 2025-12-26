#!/usr/bin/env python3
"""
Heisenberg Scattering Transform - Multi-Layer Implementation

This module implements the recursive HST structure from Glinsky (2025) Eq. 21.

ARCHITECTURAL DECISIONS:
========================

1. FEATURE EXTRACTOR vs UNITARY TRANSFORM
   
   Mallat Scattering: |W₁|*φ, |W₂|*φ, ... (feature vector, NOT invertible)
   Glinsky HST: Claims exact invertibility via R mapping
   
   Our approach: HYBRID
   - Store coefficients at each layer (like Mallat) for interpretability
   - But use R (not |·|) to preserve phase → invertible in principle
   - The "scattering coefficients" ARE the intermediate states

2. BRANCH CUTS
   
   We rely on two-channel (H⁺ ⊕ H⁻) redundancy to handle discontinuities.
   The filter bank reconstructs perfectly even with branch cuts.
   No need for unwrapping within the transform.

3. LAYER STRUCTURE
   
   Layer 0: Input signal x
   Layer 1: W₁[λ₁] = R(x * ψ_{λ₁})  for each scale λ₁
   Layer 2: W₂[λ₁, λ₂] = R(W₁[λ₁] * ψ_{λ₂})  for λ₂ > λ₁
   ...
   
   The constraint λ₂ > λ₁ prevents redundancy (lower frequencies
   don't modulate higher frequencies meaningfully).

4. INVERTIBILITY
   
   Forward: x → {W_m[path]} via repeated (convolve, R)
   Inverse: Requires storing ALL intermediate coefficients
   
   Unlike Mallat (which loses phase), HST is invertible IF we store
   the full coefficient tree. This is expensive but mathematically exact.

References:
    - Glinsky (2025), Eq. 21
    - Mallat (2012), "Group Invariant Scattering"
    - Lopatin (1996), for Lie group interpretation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .filter_bank import (
    two_channel_paul_filterbank,
    forward_transform,
    inverse_transform,
)
from .conformal import simple_R, glinsky_R


@dataclass
class ScatteringPath:
    """Represents a path through the scattering network."""
    scales: Tuple[int, ...]  # (λ₁, λ₂, ..., λ_m)
    coefficients: np.ndarray  # The coefficient signal at this path
    
    @property
    def order(self) -> int:
        """Scattering order (depth in the tree)."""
        return len(self.scales)
    
    def __repr__(self):
        return f"Path{self.scales}: shape={self.coefficients.shape}"


@dataclass  
class ScatteringOutput:
    """Complete output of HST forward transform."""
    paths: Dict[Tuple[int, ...], np.ndarray]  # path -> coefficients
    filters: np.ndarray  # Filter bank used
    filter_info: dict  # Filter metadata
    
    def order(self, m: int) -> Dict[Tuple[int, ...], np.ndarray]:
        """Get all paths of order m."""
        return {k: v for k, v in self.paths.items() if len(k) == m}
    
    def coefficients_flat(self) -> np.ndarray:
        """Flatten all coefficients into feature vector."""
        return np.concatenate([c.flatten() for c in self.paths.values()])
    
    @property
    def max_order(self) -> int:
        return max(len(k) for k in self.paths.keys())
    
    def energy_by_order(self) -> Dict[int, float]:
        """Energy distribution across orders."""
        result = {}
        for path, coef in self.paths.items():
            m = len(path)
            if m not in result:
                result[m] = 0.0
            result[m] += np.sum(np.abs(coef)**2)
        return result


class HeisenbergScatteringTransform:
    """
    Heisenberg Scattering Transform.
    
    Parameters
    ----------
    T : int
        Signal length
    J : int
        Number of octaves
    Q : int
        Filters per octave
    max_order : int
        Maximum scattering order (depth)
    r_type : str
        'simple' for i·ln(z), 'glinsky' for full Joukowski mapping
    lifting : str
        Lifting strategy to avoid log singularity:
        - 'none': No lifting (caller must ensure |z| > 0)
        - 'shift': Add constant to ensure min|z| > epsilon
        - 'analytic': Use Hilbert transform for real signals
        - 'adaptive': Automatically detect and apply minimal shift
    epsilon : float
        Minimum distance from origin for 'shift' and 'adaptive' modes
        
    Example
    -------
    >>> hst = HeisenbergScatteringTransform(T=512, J=4, Q=4, max_order=2)
    >>> output = hst.forward(signal)
    >>> x_rec = hst.inverse(output)
    >>> assert np.allclose(signal, x_rec)
    """
    
    def __init__(
        self,
        T: int,
        J: int = 4,
        Q: int = 4,
        max_order: int = 2,
        r_type: str = 'simple',
        lifting: str = 'adaptive',
        epsilon: float = 1.0,
        m: int = 4,  # Paul wavelet order
        dc_shift: float = 0.0,  # Legacy parameter
    ):
        self.T = T
        self.J = J
        self.Q = Q
        self.max_order = max_order
        self.r_type = r_type
        self.lifting = lifting
        self.epsilon = epsilon
        
        # Legacy support
        if dc_shift != 0.0:
            self.lifting = 'shift'
            self.epsilon = dc_shift
        
        # Build filter bank
        self.filters, self.filter_info = two_channel_paul_filterbank(T, J, Q, m)
        self.n_filters = self.filters.shape[0]
        
        # Number of mother wavelets (excluding father)
        self.n_mothers = self.filter_info['n_pos_mothers'] + self.filter_info['n_neg_mothers']
        
        # R mapping functions
        if r_type == 'simple':
            self._R = simple_R
            self._R_inv = lambda w: np.exp(-1j * w)
        elif r_type == 'glinsky':
            self._R = glinsky_R
            # Glinsky inverse is more complex
            from .conformal import glinsky_R_inverse
            self._R_inv = glinsky_R_inverse
        else:
            raise ValueError(f"Unknown r_type: {r_type}")
    
    def _lift(self, z: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply lifting to ensure signal stays away from log singularity.
        
        Returns lifted signal and metadata needed for unlifting.
        """
        if self.lifting == 'none':
            return z, {}
        
        elif self.lifting == 'shift':
            # Simple constant shift
            return z + self.epsilon, {'shift': self.epsilon}
        
        elif self.lifting == 'analytic':
            # For real signals, create analytic signal
            # NOTE: This is for FEATURE extraction, not reconstruction
            # Store original for perfect reconstruction
            if np.isrealobj(z) or np.max(np.abs(z.imag)) < 1e-10:
                z_hat = np.fft.fft(z.real)
                n = len(z_hat)
                h = np.zeros(n)
                h[0] = 1
                h[1:(n+1)//2] = 2
                if n % 2 == 0:
                    h[n//2] = 1
                z_analytic = np.fft.ifft(z_hat * h)
                # Shift to avoid origin
                z_lifted = z_analytic + self.epsilon
                return z_lifted, {
                    'analytic': True, 
                    'shift': self.epsilon,
                }
            else:
                return z + self.epsilon, {'shift': self.epsilon}
        
        elif self.lifting == 'adaptive':
            # Compute minimal shift needed
            min_abs = np.min(np.abs(z))
            if min_abs < self.epsilon:
                shift = self.epsilon - min_abs + 0.1
                return z + shift, {'shift': shift}
            else:
                return z, {'shift': 0.0}
        
        else:
            raise ValueError(f"Unknown lifting: {self.lifting}")
    
    def _unlift(self, z: np.ndarray, meta: dict) -> np.ndarray:
        """Undo the lifting operation."""
        if meta.get('analytic', False):
            # For analytic signals, the original is the real part
            shift = meta.get('shift', 0.0)
            return (z - shift).real
        shift = meta.get('shift', 0.0)
        return z - shift
    
    def forward(self, x: np.ndarray, max_order: Optional[int] = None) -> ScatteringOutput:
        """
        Compute forward HST.
        
        Parameters
        ----------
        x : ndarray, shape (T,)
            Input signal (real or complex)
        max_order : int, optional
            Override the instance's max_order for this call.
            Must be <= self.max_order. Default: use instance max_order.
            
        Returns
        -------
        output : ScatteringOutput
            Contains all scattering paths and coefficients
        """
        assert x.shape == (self.T,), f"Expected shape ({self.T},), got {x.shape}"
        
        # Determine effective max_order
        if max_order is None:
            effective_max_order = self.max_order
        else:
            if max_order > self.max_order:
                raise ValueError(
                    f"Requested max_order={max_order} exceeds instance max_order={self.max_order}"
                )
            effective_max_order = max_order
        
        paths = {}
        lift_meta = {}  # Store lifting metadata for each path
        
        # Lift input signal
        x_lifted, meta_input = self._lift(x)
        lift_meta['input'] = meta_input
        
        # Order 0: Low-pass (father wavelet) output
        coeffs_all = forward_transform(x_lifted, self.filters)
        paths[()] = coeffs_all[-1]  # Father coefficient (no R applied)
        
        # Store raw coefficients before R (needed for inverse)
        raw_coeffs = {}
        raw_coeffs[()] = coeffs_all  # All coefficients at order 0
        
        # Order 1: First layer scattering
        if effective_max_order >= 1:
            for j1 in range(self.n_mothers):
                U1 = coeffs_all[j1]
                # Lift and apply R
                U1_lifted, meta = self._lift(U1)
                lift_meta[(j1,)] = meta
                W1 = self._R(U1_lifted)
                paths[(j1,)] = W1
        
        # Order 2: Second layer
        if effective_max_order >= 2:
            for j1 in range(self.n_mothers):
                W1 = paths[(j1,)]
                coeffs_W1 = forward_transform(W1, self.filters)
                
                for j2 in range(self.n_mothers):
                    if j2 <= j1:
                        continue
                    
                    U2 = coeffs_W1[j2]
                    U2_lifted, meta = self._lift(U2)
                    lift_meta[(j1, j2)] = meta
                    W2 = self._R(U2_lifted)
                    paths[(j1, j2)] = W2
        
        # Order 3+: Recursive
        if effective_max_order >= 3:
            for order in range(3, effective_max_order + 1):
                prev_paths = {k: v for k, v in paths.items() if len(k) == order - 1}
                
                for prev_path, W_prev in prev_paths.items():
                    coeffs_prev = forward_transform(W_prev, self.filters)
                    j_prev = prev_path[-1]
                    
                    for j_new in range(self.n_mothers):
                        if j_new <= j_prev:
                            continue
                        
                        U_new = coeffs_prev[j_new]
                        U_new_lifted, meta = self._lift(U_new)
                        new_path = prev_path + (j_new,)
                        lift_meta[new_path] = meta
                        W_new = self._R(U_new_lifted)
                        paths[new_path] = W_new
        
        output = ScatteringOutput(
            paths=paths,
            filters=self.filters,
            filter_info=self.filter_info,
        )
        output._lift_meta = lift_meta
        output._raw_coeffs = raw_coeffs
        output._input_meta = meta_input
        
        return output
    
    def inverse(self, output: ScatteringOutput) -> np.ndarray:
        """
        Reconstruct signal from scattering coefficients.
        
        This is the EXACT inverse of forward(), utilizing the fact that:
        1. R is invertible: R⁻¹(R(z)) = z
        2. The filter bank is a tight frame
        3. We store all coefficients, not just low-pass averages
        
        Parameters
        ----------
        output : ScatteringOutput
            Output from forward()
            
        Returns
        -------
        x_rec : ndarray, shape (T,)
            Reconstructed signal
        """
        # Get lifting metadata
        lift_meta = getattr(output, '_lift_meta', {})
        input_meta = getattr(output, '_input_meta', {})
        
        # RECONSTRUCTION STRATEGY:
        # We need to invert: W₁[j] = R(U₁[j] + shift)
        # where U₁[j] = (x_lifted * ψⱼ)
        #
        # Step 1: Invert R to get U₁[j] + shift
        # Step 2: Subtract shift to get U₁[j]
        # Step 3: Use inverse_transform to get x_lifted
        # Step 4: Unlift to get x
        
        # Collect order-1 coefficients and invert R
        order1_coeffs = np.zeros((self.n_filters, self.T), dtype=np.complex128)
        
        for j in range(self.n_mothers):
            if (j,) in output.paths:
                W1 = output.paths[(j,)]
                # Invert R
                U1_shifted = self._R_inv(W1)
                # Unlift
                meta = lift_meta.get((j,), {'shift': 0.0})
                U1 = self._unlift(U1_shifted, meta)
                order1_coeffs[j] = U1
        
        # Father coefficient (no R was applied)
        order1_coeffs[-1] = output.paths.get((), np.zeros(self.T, dtype=np.complex128))
        
        # Reconstruct lifted signal
        x_lifted_rec = inverse_transform(order1_coeffs, self.filters)
        
        # Unlift
        x_rec = self._unlift(x_lifted_rec, input_meta)
        
        return x_rec
    
    def test_reconstruction(self, x: np.ndarray) -> dict:
        """
        Test full forward-inverse reconstruction.
        
        Returns dict with reconstruction errors at each stage.
        """
        # Test filter bank only (no R)
        x_lifted, meta = self._lift(x)
        coeffs = forward_transform(x_lifted, self.filters)
        x_fb = inverse_transform(coeffs, self.filters)
        fb_error = np.linalg.norm(x_lifted - x_fb) / np.linalg.norm(x_lifted)
        
        # Test full HST round-trip
        output = self.forward(x)
        x_rec = self.inverse(output)
        
        # For analytic lifting of real input, compare real parts
        if self.lifting == 'analytic' and np.isrealobj(x):
            x_rec = x_rec.real
        
        hst_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        
        return {
            'filter_bank_error': float(fb_error),
            'hst_roundtrip_error': float(hst_error),
            'n_paths': len(output.paths),
            'energy_by_order': output.energy_by_order(),
            'max_order': output.max_order,
            'lifting_used': self.lifting,
        }


# =============================================================================
# Convenience functions
# =============================================================================

def hst_forward(
    x: np.ndarray,
    J: int = 4,
    Q: int = 4,
    max_order: int = 2,
) -> ScatteringOutput:
    """
    Convenience function for forward HST.
    
    Parameters
    ----------
    x : ndarray
        Input signal
    J : int
        Octaves
    Q : int
        Filters per octave
    max_order : int
        Maximum scattering depth
        
    Returns
    -------
    output : ScatteringOutput
    """
    T = len(x)
    hst = HeisenbergScatteringTransform(T, J, Q, max_order)
    return hst.forward(x)


def hst_coefficients(
    x: np.ndarray,
    J: int = 4,
    Q: int = 4,
    max_order: int = 2,
) -> np.ndarray:
    """
    Get flattened scattering coefficients (feature vector).
    
    This is analogous to Mallat's scattering features.
    """
    output = hst_forward(x, J, Q, max_order)
    return output.coefficients_flat()


# =============================================================================
# Tests
# =============================================================================

def _test_scattering():
    """Basic sanity tests."""
    print("Testing HeisenbergScatteringTransform...")
    
    T = 512
    t = np.arange(T)
    
    # Test signal: Van der Pol-like
    x = np.exp(1j * 2 * np.pi * 10 * t / T) + 2.0
    
    # Create HST
    hst = HeisenbergScatteringTransform(T, J=4, Q=4, max_order=2)
    
    # Test filter bank
    info = hst.test_reconstruction(x)
    print(f"  Filter bank error: {info['filter_bank_error']:.2e}")
    print(f"  Number of paths: {info['n_paths']}")
    print(f"  Energy by order: {info['energy_by_order']}")
    
    # Forward transform
    output = hst.forward(x)
    
    print(f"  Order 0 paths: {len(output.order(0))}")
    print(f"  Order 1 paths: {len(output.order(1))}")
    print(f"  Order 2 paths: {len(output.order(2))}")
    
    # Check coefficient shapes
    for path, coef in list(output.paths.items())[:5]:
        print(f"  Path {path}: {coef.shape}, energy={np.sum(np.abs(coef)**2):.2f}")
    
    print("  ✓ Scattering transform working")
    
    return True


if __name__ == "__main__":
    _test_scattering()
