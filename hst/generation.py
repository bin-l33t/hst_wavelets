#!/usr/bin/env python3
"""
HST Generation Module

Implements gradient-based reconstruction from deep scattering coefficients.
This enables "dreaming" signals that match specific topological features.

The key insight: Since Layer 2+ loses information through H⁻ projection
(now fixed with two-channel), we can't analytically invert the deep tree.
Instead, we optimize: find x such that HST(x) ≈ target coefficients.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of gradient-based generation."""
    signal: np.ndarray
    loss_history: List[float]
    final_loss: float
    iterations: int
    converged: bool


class HSTGenerator:
    """
    Generate signals from deep scattering coefficients.
    
    Uses gradient descent to find x such that HST(x) matches target.
    
    Parameters
    ----------
    hst : HeisenbergScatteringTransform
        The HST instance to use for forward computation
    """
    
    def __init__(self, hst):
        self.hst = hst
        self.T = hst.T
    
    def _compute_loss(
        self,
        x: np.ndarray,
        target_coeffs: Dict[tuple, np.ndarray],
        order_weights: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, Dict[tuple, np.ndarray]]:
        """
        Compute loss between HST(x) and target coefficients.
        
        Returns loss value and current coefficients.
        """
        output = self.hst.forward(x)
        
        if order_weights is None:
            order_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        
        total_loss = 0.0
        
        for path, target in target_coeffs.items():
            if path in output.paths:
                current = output.paths[path]
                order = len(path) if path else 0
                weight = order_weights.get(order, 1.0)
                
                # L2 loss on complex coefficients
                diff = current - target
                path_loss = np.sum(np.abs(diff)**2)
                total_loss += weight * path_loss
        
        return total_loss, output.paths
    
    def _numerical_gradient(
        self,
        x: np.ndarray,
        target_coeffs: Dict[tuple, np.ndarray],
        order_weights: Optional[Dict[int, float]] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. x using finite differences.
        
        For complex x, we compute gradients w.r.t. real and imag parts.
        """
        grad = np.zeros_like(x, dtype=np.complex128)
        
        # Gradient w.r.t. real part
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            loss_plus, _ = self._compute_loss(x_plus, target_coeffs, order_weights)
            loss_minus, _ = self._compute_loss(x_minus, target_coeffs, order_weights)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Gradient w.r.t. imaginary part
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += 1j * eps
            x_minus[i] -= 1j * eps
            
            loss_plus, _ = self._compute_loss(x_plus, target_coeffs, order_weights)
            loss_minus, _ = self._compute_loss(x_minus, target_coeffs, order_weights)
            
            grad[i] += 1j * (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    def _stochastic_gradient(
        self,
        x: np.ndarray,
        target_coeffs: Dict[tuple, np.ndarray],
        order_weights: Optional[Dict[int, float]] = None,
        n_samples: int = 50,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        Estimate gradient using random perturbations (faster than full FD).
        """
        grad = np.zeros_like(x, dtype=np.complex128)
        base_loss, _ = self._compute_loss(x, target_coeffs, order_weights)
        
        for _ in range(n_samples):
            # Random direction (complex)
            direction = np.random.randn(len(x)) + 1j * np.random.randn(len(x))
            direction = direction / np.linalg.norm(direction)
            
            # Perturb
            x_pert = x + eps * direction
            loss_pert, _ = self._compute_loss(x_pert, target_coeffs, order_weights)
            
            # Directional derivative
            deriv = (loss_pert - base_loss) / eps
            
            # Accumulate
            grad += deriv * np.conj(direction)
        
        return grad / n_samples
    
    def reconstruct_from_deep(
        self,
        target_coeffs: Dict[tuple, np.ndarray],
        init: Optional[np.ndarray] = None,
        order_weights: Optional[Dict[int, float]] = None,
        lr: float = 0.01,
        n_iter: int = 500,
        tol: float = 1e-8,
        gradient_method: str = 'stochastic',
        verbose: bool = True,
        momentum: float = 0.9,
    ) -> GenerationResult:
        """
        Reconstruct signal from deep scattering coefficients.
        
        Parameters
        ----------
        target_coeffs : dict
            Target coefficients {path: values}
            Can include any subset of paths (Order 0, 1, 2, ...)
        init : ndarray, optional
            Initial signal guess. If None, uses noise + DC from target.
        order_weights : dict, optional
            Weights for different orders {order: weight}
            Default: equal weights
        lr : float
            Learning rate
        n_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance (relative loss change)
        gradient_method : str
            'stochastic' (faster) or 'numerical' (more accurate)
        verbose : bool
            Print progress
        momentum : float
            Momentum coefficient for gradient descent
            
        Returns
        -------
        result : GenerationResult
            Contains reconstructed signal and diagnostics
        """
        # Initialize
        if init is None:
            # Start with noise + DC if available
            if () in target_coeffs:
                dc = target_coeffs[()].mean()
                init = dc + 0.1 * (np.random.randn(self.T) + 1j * np.random.randn(self.T))
            else:
                init = np.random.randn(self.T) + 1j * np.random.randn(self.T)
        
        x = init.copy().astype(np.complex128)
        
        # Gradient function
        if gradient_method == 'stochastic':
            grad_fn = lambda x: self._stochastic_gradient(x, target_coeffs, order_weights)
        else:
            grad_fn = lambda x: self._numerical_gradient(x, target_coeffs, order_weights)
        
        loss_history = []
        velocity = np.zeros_like(x)
        
        prev_loss = float('inf')
        
        for iteration in range(n_iter):
            # Compute loss
            loss, _ = self._compute_loss(x, target_coeffs, order_weights)
            loss_history.append(loss)
            
            if verbose and iteration % 50 == 0:
                print(f"  Iter {iteration:4d}: loss = {loss:.6e}")
            
            # Check convergence (handle NaN)
            if np.isfinite(loss) and np.isfinite(prev_loss):
                rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-10)
                if rel_change < tol:
                    if verbose:
                        print(f"  Converged at iteration {iteration}")
                    break
            
            # Compute gradient
            grad = grad_fn(x)
            
            # Update with momentum
            velocity = momentum * velocity - lr * grad
            x = x + velocity
            
            # Apply lifting constraint (stay away from origin)
            if hasattr(self.hst, 'epsilon'):
                min_abs = np.min(np.abs(x))
                if min_abs < self.hst.epsilon:
                    x = x + (self.hst.epsilon - min_abs + 0.1)
            
            prev_loss = loss
        
        final_loss, _ = self._compute_loss(x, target_coeffs, order_weights)
        
        return GenerationResult(
            signal=x,
            loss_history=loss_history,
            final_loss=final_loss,
            iterations=len(loss_history),
            converged=len(loss_history) < n_iter,
        )
    
    def reconstruct_order2_only(
        self,
        target_output,
        include_dc: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Reconstruct from Order-2 coefficients only (plus optionally DC).
        
        This is the key test: can we recover x from just the deep features?
        
        Parameters
        ----------
        target_output : ScatteringOutput
            Output from HST forward pass
        include_dc : bool
            Whether to include Order-0 (DC) in target
        **kwargs
            Passed to reconstruct_from_deep
        """
        target_coeffs = {}
        
        # Order 0 (DC)
        if include_dc and () in target_output.paths:
            target_coeffs[()] = target_output.paths[()]
        
        # Order 2 only
        for path, coeff in target_output.paths.items():
            if len(path) == 2:
                target_coeffs[path] = coeff
        
        print(f"  Reconstructing from {len(target_coeffs)} paths (Order 2 + DC)")
        
        return self.reconstruct_from_deep(target_coeffs, **kwargs)


# =============================================================================
# PyTorch-based generator (if available)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    TORCH_AVAILABLE = True
    
    class HSTGeneratorTorch(nn.Module):
        """
        PyTorch-based HST generator for GPU acceleration and autograd.
        """
        
        def __init__(self, hst):
            super().__init__()
            self.hst = hst
            self.T = hst.T
            
            # Convert filters to torch tensors
            self.register_buffer(
                'filters_real',
                torch.tensor(hst.filters.real, dtype=torch.float64)
            )
            self.register_buffer(
                'filters_imag',
                torch.tensor(hst.filters.imag, dtype=torch.float64)
            )
        
        def forward_transform_torch(self, x_real, x_imag):
            """
            Forward wavelet transform in PyTorch.
            
            Parameters
            ----------
            x_real, x_imag : torch.Tensor
                Real and imaginary parts of signal
            
            Returns
            -------
            coeffs_real, coeffs_imag : torch.Tensor
                Real and imaginary parts of coefficients
            """
            # FFT
            x = torch.complex(x_real, x_imag)
            x_hat = torch.fft.fft(x)
            
            filters = torch.complex(self.filters_real, self.filters_imag)
            
            # Convolution in frequency domain
            coeffs_hat = x_hat.unsqueeze(0) * filters
            coeffs = torch.fft.ifft(coeffs_hat, dim=-1)
            
            return coeffs.real, coeffs.imag
        
        def simple_R_torch(self, z_real, z_imag, eps=1e-10):
            """R mapping in PyTorch: R(z) = i * ln(z)"""
            # ln(z) = ln|z| + i*arg(z)
            abs_z = torch.sqrt(z_real**2 + z_imag**2 + eps)
            arg_z = torch.atan2(z_imag, z_real)
            
            # i * ln(z) = i * (ln|z| + i*arg(z)) = -arg(z) + i*ln|z|
            w_real = -arg_z
            w_imag = torch.log(abs_z)
            
            return w_real, w_imag
        
        def compute_hst_order1(self, x_real, x_imag, shift=1.0):
            """Compute Order-1 HST coefficients."""
            # Forward transform
            U_real, U_imag = self.forward_transform_torch(x_real, x_imag)
            
            # Apply R with shift
            W_real, W_imag = self.simple_R_torch(U_real + shift, U_imag)
            
            return W_real, W_imag
        
        def reconstruct_from_deep_torch(
            self,
            target_coeffs: Dict[tuple, np.ndarray],
            init: Optional[np.ndarray] = None,
            lr: float = 0.01,
            n_iter: int = 1000,
            verbose: bool = True,
        ) -> GenerationResult:
            """
            Reconstruct using PyTorch autograd.
            """
            device = self.filters_real.device
            
            # Initialize
            if init is None:
                if () in target_coeffs:
                    dc = target_coeffs[()].mean()
                    init = dc + 0.1 * (np.random.randn(self.T) + 1j * np.random.randn(self.T))
                else:
                    init = np.random.randn(self.T) + 1j * np.random.randn(self.T)
            
            # Parameters to optimize
            x_real = nn.Parameter(torch.tensor(init.real, dtype=torch.float64, device=device))
            x_imag = nn.Parameter(torch.tensor(init.imag, dtype=torch.float64, device=device))
            
            # Convert targets to torch
            targets = {}
            for path, coeff in target_coeffs.items():
                targets[path] = (
                    torch.tensor(coeff.real, dtype=torch.float64, device=device),
                    torch.tensor(coeff.imag, dtype=torch.float64, device=device),
                )
            
            optimizer = optim.Adam([x_real, x_imag], lr=lr)
            
            loss_history = []
            
            for iteration in range(n_iter):
                optimizer.zero_grad()
                
                # Forward HST (Order 1)
                W_real, W_imag = self.compute_hst_order1(x_real, x_imag)
                
                # Compute loss
                loss = torch.tensor(0.0, dtype=torch.float64, device=device)
                
                for path, (target_real, target_imag) in targets.items():
                    if len(path) == 1:
                        j = path[0]
                        if j < W_real.shape[0]:
                            diff_real = W_real[j] - target_real
                            diff_imag = W_imag[j] - target_imag
                            loss = loss + torch.sum(diff_real**2 + diff_imag**2)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                loss_history.append(loss_val)
                
                if verbose and iteration % 100 == 0:
                    print(f"  Iter {iteration:4d}: loss = {loss_val:.6e}")
            
            # Extract result
            x_result = x_real.detach().cpu().numpy() + 1j * x_imag.detach().cpu().numpy()
            
            return GenerationResult(
                signal=x_result,
                loss_history=loss_history,
                final_loss=loss_history[-1],
                iterations=len(loss_history),
                converged=True,
            )

except ImportError:
    TORCH_AVAILABLE = False
    HSTGeneratorTorch = None


# =============================================================================
# Convenience functions
# =============================================================================

def reconstruct_from_coefficients(
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    method: str = 'numpy',
    **kwargs
) -> GenerationResult:
    """
    Reconstruct signal from target scattering coefficients.
    
    Parameters
    ----------
    hst : HeisenbergScatteringTransform
        HST instance
    target_coeffs : dict
        Target coefficients {path: values}
    method : str
        'numpy' (CPU, numerical gradients) or 'torch' (GPU, autograd)
    **kwargs
        Passed to generator
        
    Returns
    -------
    result : GenerationResult
    """
    if method == 'torch' and TORCH_AVAILABLE:
        generator = HSTGeneratorTorch(hst)
        return generator.reconstruct_from_deep_torch(target_coeffs, **kwargs)
    else:
        generator = HSTGenerator(hst)
        return generator.reconstruct_from_deep(target_coeffs, **kwargs)


def dream_signal(
    hst,
    modifications: Dict[tuple, Callable],
    base_signal: np.ndarray,
    **kwargs
) -> GenerationResult:
    """
    "Dream" a new signal by modifying specific coefficients.
    
    Parameters
    ----------
    hst : HeisenbergScatteringTransform
        HST instance
    modifications : dict
        {path: modifier_fn} where modifier_fn(coeff) -> new_coeff
    base_signal : ndarray
        Signal to start from
    **kwargs
        Passed to reconstruct_from_coefficients
        
    Returns
    -------
    result : GenerationResult
        
    Example
    -------
    >>> # Amplify high-frequency Order-2 coefficients
    >>> mods = {
    ...     (15, 20): lambda c: 2.0 * c,
    ...     (16, 21): lambda c: 2.0 * c,
    ... }
    >>> result = dream_signal(hst, mods, original_signal)
    """
    # Get base coefficients
    output = hst.forward(base_signal)
    
    # Apply modifications
    target_coeffs = {}
    for path, coeff in output.paths.items():
        if path in modifications:
            target_coeffs[path] = modifications[path](coeff)
        else:
            target_coeffs[path] = coeff
    
    return reconstruct_from_coefficients(hst, target_coeffs, **kwargs)
