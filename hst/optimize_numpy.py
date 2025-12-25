#!/usr/bin/env python3
"""
HST Optimization Module (Pure NumPy)

Implements gradient-based optimization for HST coefficients with analytic gradients.

Mathematical Foundation:
========================

For Order 1, the forward pass is:
    U₁[j] = x * ψⱼ     (convolution with wavelet)
    W₁[j] = R(U₁[j])   (R mapping: R(z) = i·ln(z))

Loss function:
    L = Σⱼ wⱼ |W₁[j] - W_target[j]|²

Backward pass (chain rule):
    ∂L/∂W₁[j] = 2·wⱼ·(W₁[j] - W_target[j])
    
    For R(U) = i·ln(U):
        dW/dU = i/U
        
    Using Wirtinger calculus for real-valued loss L(W, W̄):
        ∂L/∂U = (∂L/∂W)·(∂W/∂U) + (∂L/∂W̄)·(∂W̄/∂U)
        
    Since L = |W - W_t|² = (W - W_t)·conj(W - W_t):
        ∂L/∂W̄ = (W - W_t)
        ∂L/∂W = conj(W - W_t)  [but this doesn't contribute for holomorphic W(U)]
        
    For holomorphic R: ∂L/∂Ū = (∂L/∂W̄)·conj(dW/dU) = (W - W_t)·conj(i/U)
    
    The gradient for optimization is: ∇_U L = 2·∂L/∂Ū = 2·(W - W_t)·conj(i/U)
    
    Filter backward (adjoint of convolution):
        If U = x * ψ, then ∂L/∂x = Σⱼ (∂L/∂Uⱼ) ★ ψⱼ
        where ★ denotes correlation (convolution with time-reversed conjugate)

References:
    - Wirtinger calculus: Kreutz-Delgado (2009)
    - Mallat scattering gradients: Angles & Mallat (2018)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result of gradient-based optimization."""
    signal: np.ndarray
    loss_history: List[float]
    grad_norm_history: List[float]
    final_loss: float
    converged: bool


def compute_loss_and_grad_order1(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute Order-1 loss and analytic gradient.
    
    Parameters
    ----------
    x : ndarray, shape (T,), complex
        Input signal
    hst : HeisenbergScatteringTransform
        HST instance (provides filters, R mapping, lifting)
    target_coeffs : dict
        Target coefficients {(j,): W_target[j]} for Order-1 paths
    weights : dict, optional
        Per-order weights {order: weight}. Default: {1: 1.0}
        
    Returns
    -------
    loss : float
        L2 loss on Order-1 coefficients
    grad_x : ndarray, shape (T,), complex
        Gradient of loss w.r.t. x
    """
    # Delegate to Order 2 with no Order 2 targets
    return compute_loss_and_grad_order2(x, hst, target_coeffs, weights)


def compute_loss_and_grad_order2(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute Order-1 and Order-2 loss with analytic gradient.
    
    Backpropagation through the full scattering tree:
    
        x → U1[j] → W1[j] → U2[j,k] → W2[j,k]
        
    The gradient computation handles "fan-out": one W1[j] feeds multiple W2[j,k].
    
    Parameters
    ----------
    x : ndarray, shape (T,), complex
        Input signal
    hst : HeisenbergScatteringTransform
        HST instance
    target_coeffs : dict
        Target coefficients for any combination of paths:
        {(): S0, (j,): W1[j], (j,k): W2[j,k]}
    weights : dict, optional
        Per-order weights {order: weight}. Default: all 1.0
        
    Returns
    -------
    loss : float
        Total L2 loss
    grad_x : ndarray, shape (T,), complex
        Gradient of loss w.r.t. x
    """
    T = len(x)
    
    if weights is None:
        weights = {0: 1.0, 1: 1.0, 2: 1.0}
    
    # =========================================================================
    # FORWARD PASS - Cache all intermediate values
    # =========================================================================
    
    # Lift input signal
    x_lifted, input_meta = hst._lift(x)
    x_hat = np.fft.fft(x_lifted)
    
    # --- Order 0: Father wavelet (no R) ---
    father_idx = -1
    S0_hat = x_hat * hst.filters[father_idx]
    S0 = np.fft.ifft(S0_hat)
    
    # --- Order 1: U1[j] = x * ψj, W1[j] = R(U1[j]) ---
    U1 = {}           # Before R
    U1_lifted = {}    # After lifting
    W1 = {}           # After R
    lift_meta_1 = {}  # Lifting metadata
    
    for j in range(hst.n_mothers):
        U1_hat = x_hat * hst.filters[j]
        U1[j] = np.fft.ifft(U1_hat)
        U1_lifted[j], lift_meta_1[j] = hst._lift(U1[j])
        W1[j] = hst._R(U1_lifted[j])
    
    # --- Order 2: U2[j,k] = W1[j] * ψk, W2[j,k] = R(U2[j,k]) ---
    U2 = {}           # Before R
    U2_lifted = {}    # After lifting
    W2 = {}           # After R
    lift_meta_2 = {}  # Lifting metadata
    W1_hat = {}       # Cache FFT of W1 for efficiency
    
    for j in range(hst.n_mothers):
        W1_hat[j] = np.fft.fft(W1[j])
        
        for k in range(hst.n_mothers):
            if k <= j:  # Constraint: k > j to avoid redundancy
                continue
            
            path = (j, k)
            U2_hat = W1_hat[j] * hst.filters[k]
            U2[path] = np.fft.ifft(U2_hat)
            U2_lifted[path], lift_meta_2[path] = hst._lift(U2[path])
            W2[path] = hst._R(U2_lifted[path])
    
    # =========================================================================
    # COMPUTE LOSS
    # =========================================================================
    
    loss = 0.0
    
    # Order 0 loss
    if () in target_coeffs:
        w0 = weights.get(0, 1.0)
        diff0 = S0 - target_coeffs[()]
        loss += w0 * np.sum(np.abs(diff0)**2).real
    
    # Order 1 loss
    w1 = weights.get(1, 1.0)
    for j in range(hst.n_mothers):
        path = (j,)
        if path in target_coeffs:
            diff = W1[j] - target_coeffs[path]
            loss += w1 * np.sum(np.abs(diff)**2).real
    
    # Order 2 loss
    w2 = weights.get(2, 1.0)
    for path in W2:
        if path in target_coeffs:
            diff = W2[path] - target_coeffs[path]
            loss += w2 * np.sum(np.abs(diff)**2).real
    
    # =========================================================================
    # BACKWARD PASS
    # =========================================================================
    
    # Initialize gradient accumulators
    grad_x_hat = np.zeros(T, dtype=np.complex128)
    
    # Accumulator for gradients flowing back to W1 from Order 2
    # One W1[j] feeds multiple W2[j,k], so we accumulate
    grad_W1_accum = {j: np.zeros(T, dtype=np.complex128) for j in range(hst.n_mothers)}
    
    # --- Layer 2 Backward ---
    w2 = weights.get(2, 1.0)
    
    for path in W2:
        if path not in target_coeffs:
            continue
        
        j, k = path
        
        # Gradient w.r.t W2[j,k]
        diff = W2[path] - target_coeffs[path]
        grad_W2 = 2 * w2 * diff
        
        # Backward through R2: W2 = i·ln(U2)
        # dW/dU = i/U, grad_U = grad_W * conj(i/U)
        dW2_dU2 = 1j / U2_lifted[path]
        grad_U2_lifted = grad_W2 * np.conj(dW2_dU2)
        
        # Backward through lifting (additive, gradient passes through)
        grad_U2 = grad_U2_lifted
        
        # Backward through Filter 2: U2 = W1[j] * ψk
        # grad_W1_contrib = grad_U2 ★ ψk (correlation = conv with conj)
        grad_U2_hat = np.fft.fft(grad_U2)
        grad_W1_contrib_hat = grad_U2_hat * np.conj(hst.filters[k])
        grad_W1_contrib = np.fft.ifft(grad_W1_contrib_hat)
        
        # Accumulate into W1[j]'s gradient buffer
        grad_W1_accum[j] += grad_W1_contrib
    
    # --- Layer 1 Backward ---
    w1 = weights.get(1, 1.0)
    
    for j in range(hst.n_mothers):
        path = (j,)
        
        # Direct gradient from Order 1 loss
        grad_W1_direct = np.zeros(T, dtype=np.complex128)
        if path in target_coeffs:
            diff = W1[j] - target_coeffs[path]
            grad_W1_direct = 2 * w1 * diff
        
        # Total gradient at W1[j] = direct + accumulated from Order 2
        grad_W1_total = grad_W1_direct + grad_W1_accum[j]
        
        # Backward through R1: W1 = i·ln(U1)
        dW1_dU1 = 1j / U1_lifted[j]
        grad_U1_lifted = grad_W1_total * np.conj(dW1_dU1)
        
        # Backward through lifting
        grad_U1 = grad_U1_lifted
        
        # Backward through Filter 1: U1 = x * ψj
        grad_U1_hat = np.fft.fft(grad_U1)
        grad_x_hat += grad_U1_hat * np.conj(hst.filters[j])
    
    # --- Order 0 Backward (no R) ---
    if () in target_coeffs:
        w0 = weights.get(0, 1.0)
        diff0 = S0 - target_coeffs[()]
        grad_S0 = 2 * w0 * diff0
        grad_S0_hat = np.fft.fft(grad_S0)
        grad_x_hat += grad_S0_hat * np.conj(hst.filters[father_idx])
    
    # Convert to time domain
    grad_x_lifted = np.fft.ifft(grad_x_hat)
    
    # Backward through input lifting (additive)
    grad_x = grad_x_lifted
    
    return float(loss), grad_x


def compute_loss_only(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
) -> float:
    """Compute loss without gradient (for finite difference testing)."""
    loss, _ = compute_loss_and_grad_order1(x, hst, target_coeffs, weights)
    return loss


def finite_difference_gradient(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute gradient using finite differences (for testing).
    
    Uses central differences on real and imaginary parts separately.
    """
    T = len(x)
    grad = np.zeros(T, dtype=np.complex128)
    
    # Gradient w.r.t real part
    for i in range(T):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        loss_plus = compute_loss_only(x_plus, hst, target_coeffs, weights)
        loss_minus = compute_loss_only(x_minus, hst, target_coeffs, weights)
        
        grad[i] += (loss_plus - loss_minus) / (2 * eps)
    
    # Gradient w.r.t imaginary part
    for i in range(T):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += 1j * eps
        x_minus[i] -= 1j * eps
        
        loss_plus = compute_loss_only(x_plus, hst, target_coeffs, weights)
        loss_minus = compute_loss_only(x_minus, hst, target_coeffs, weights)
        
        grad[i] += 1j * (loss_plus - loss_minus) / (2 * eps)
    
    return grad


def optimize_signal(
    target_coeffs: Dict[tuple, np.ndarray],
    hst,
    x0: np.ndarray,
    n_steps: int = 100,
    lr: float = 0.1,
    weights: Optional[Dict[int, float]] = None,
    momentum: float = 0.9,
    verbose: bool = True,
    callback: Optional[callable] = None,
) -> OptimizationResult:
    """
    Optimize signal to match target HST coefficients.
    
    Uses gradient descent with momentum.
    
    Parameters
    ----------
    target_coeffs : dict
        Target coefficients {path: values}
    hst : HeisenbergScatteringTransform
        HST instance
    x0 : ndarray
        Initial signal guess
    n_steps : int
        Number of optimization steps
    lr : float
        Learning rate
    weights : dict, optional
        Per-order weights
    momentum : float
        Momentum coefficient (0 = no momentum)
    verbose : bool
        Print progress
    callback : callable, optional
        Called each step with (step, x, loss, grad)
        
    Returns
    -------
    result : OptimizationResult
    """
    x = x0.copy().astype(np.complex128)
    velocity = np.zeros_like(x)
    
    loss_history = []
    grad_norm_history = []
    
    for step in range(n_steps):
        loss, grad = compute_loss_and_grad_order1(x, hst, target_coeffs, weights)
        
        grad_norm = np.linalg.norm(grad)
        loss_history.append(loss)
        grad_norm_history.append(grad_norm)
        
        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"  Step {step:4d}: loss = {loss:.6e}, |grad| = {grad_norm:.6e}")
        
        if callback is not None:
            callback(step, x, loss, grad)
        
        # Gradient descent with momentum
        velocity = momentum * velocity - lr * grad
        x = x + velocity
        
        # Ensure signal stays away from singularities (optional constraint)
        # This is handled by the lifting mechanism in the forward pass
    
    # Final loss
    final_loss, _ = compute_loss_and_grad_order1(x, hst, target_coeffs, weights)
    loss_history.append(final_loss)
    
    if verbose:
        print(f"  Final: loss = {final_loss:.6e}")
    
    converged = len(loss_history) > 1 and loss_history[-1] < loss_history[0] * 0.01
    
    return OptimizationResult(
        signal=x,
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        final_loss=final_loss,
        converged=converged,
    )


# =============================================================================
# Utility functions
# =============================================================================

def extract_order1_targets(hst, x: np.ndarray) -> Dict[tuple, np.ndarray]:
    """Extract Order-1 coefficients from a signal as targets."""
    output = hst.forward(x)
    return {path: coeff for path, coeff in output.paths.items() if len(path) == 1}


def extract_order2_targets(hst, x: np.ndarray) -> Dict[tuple, np.ndarray]:
    """Extract Order-2 coefficients from a signal as targets."""
    output = hst.forward(x)
    return {path: coeff for path, coeff in output.paths.items() if len(path) == 2}


def extract_all_targets(hst, x: np.ndarray) -> Dict[tuple, np.ndarray]:
    """Extract all coefficients from a signal as targets."""
    output = hst.forward(x)
    return output.paths.copy()


# =============================================================================
# Self-test
# =============================================================================

def _test_gradient():
    """Quick gradient check."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from hst.scattering import HeisenbergScatteringTransform
    
    print("Testing gradient computation...")
    
    T = 32
    np.random.seed(42)
    
    # Create HST
    hst = HeisenbergScatteringTransform(T, J=2, Q=2, max_order=1, lifting='shift', epsilon=2.0)
    
    # Create signal shifted away from origin
    x = np.random.randn(T) + 1j * np.random.randn(T) + 3.0
    
    # Get target coefficients
    target_coeffs = extract_order1_targets(hst, x)
    
    # Perturb signal
    x_perturbed = x + 0.1 * (np.random.randn(T) + 1j * np.random.randn(T))
    
    # Compute analytic gradient
    loss, grad_analytic = compute_loss_and_grad_order1(x_perturbed, hst, target_coeffs)
    
    # Compute numeric gradient
    grad_numeric = finite_difference_gradient(x_perturbed, hst, target_coeffs, eps=1e-7)
    
    # Compare
    diff = grad_analytic - grad_numeric
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(grad_numeric) + 1e-10)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  |grad_analytic|: {np.linalg.norm(grad_analytic):.6e}")
    print(f"  |grad_numeric|: {np.linalg.norm(grad_numeric):.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    print(f"  Test {'PASSED' if rel_error < 1e-4 else 'FAILED'}")
    
    return rel_error < 1e-4


if __name__ == "__main__":
    _test_gradient()
