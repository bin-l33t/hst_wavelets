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


# =============================================================================
# Energy Normalization Utilities
# =============================================================================

def normalize_energy(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize signal to unit energy.
    
    Parameters
    ----------
    x : ndarray, complex
        Input signal
        
    Returns
    -------
    x_unit : ndarray, complex
        Normalized signal with ||x_unit||_2 = 1
    scale : float
        Original energy scale (||x||_2)
        
    Usage
    -----
    For topology-preserving optimization (radial_floor), normalizing
    to unit energy ensures consistent gradient magnitudes regardless
    of input scale. This prevents the ∂W/∂U = i/U gradient from
    exploding or vanishing due to scaling artifacts.
    """
    energy = np.linalg.norm(x)
    if energy < 1e-15:
        return x.copy(), 1.0
    x_unit = x / energy
    return x_unit, float(energy)


def denormalize_energy(x_unit: np.ndarray, scale: float) -> np.ndarray:
    """
    Restore signal to original energy scale.
    
    Parameters
    ----------
    x_unit : ndarray, complex
        Unit-energy signal
    scale : float
        Energy scale to restore
        
    Returns
    -------
    x : ndarray, complex
        Signal with ||x||_2 = scale
    """
    return x_unit * scale


def demean(x: np.ndarray) -> Tuple[np.ndarray, complex]:
    """
    Remove DC component (mean) from signal.
    
    This is essential for proper energy analysis because:
    - Order 0 captures DC, which dominates energy ratios
    - "Super-convergence" claims should only apply to fluctuations
    
    Parameters
    ----------
    x : ndarray, complex
        Input signal
        
    Returns
    -------
    x_zero_mean : ndarray, complex
        Signal with mean = 0
    dc : complex
        Removed DC component
    """
    dc = np.mean(x)
    return x - dc, dc


def compute_loss_and_grad_order1(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute Order-1 loss and analytic gradient.
    
    DEPRECATED: Use compute_loss_and_grad() for new code.
    This function is kept for backward compatibility.
    """
    return compute_loss_and_grad(x, hst, target_coeffs, weights, loss_type, phase_lambda)


def _compute_phase_robust_loss_and_grad(
    W: np.ndarray, 
    W_target: np.ndarray,
    weight: float = 1.0,
    phase_lambda: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute phase-robust loss and gradient for a single coefficient array.
    
    Loss: L = w * (|ρ - ρ_t|² + λ * |exp(iθ) - exp(iθ_t)|²)
    
    where W = θ + iρ (θ = Re(W), ρ = Im(W))
    
    The circular distance |exp(iθ) - exp(iθ_t)|² = 2(1 - cos(θ - θ_t))
    avoids branch cut discontinuities.
    
    Parameters
    ----------
    W : ndarray, complex
        Coefficient array (W = R(U) = i·ln(U) = -arg(U) + i·ln|U|)
    W_target : ndarray, complex
        Target coefficient array
    weight : float
        Overall weight on this loss term
    phase_lambda : float
        Relative weight on phase vs magnitude
        
    Returns
    -------
    loss : float
        Loss value
    grad_W : ndarray, complex
        Gradient ∂L/∂W̄ for backpropagation
    """
    # Decompose W = θ + iρ
    theta = W.real       # Phase: -arg(z)
    rho = W.imag         # Log-magnitude: ln|z|
    theta_t = W_target.real
    rho_t = W_target.imag
    
    # Magnitude loss: |ρ - ρ_t|²
    rho_diff = rho - rho_t
    loss_mag = np.sum(rho_diff**2)
    
    # Phase loss: |exp(iθ) - exp(iθ_t)|² = 2(1 - cos(θ - θ_t))
    theta_diff = theta - theta_t
    loss_phase = 2 * np.sum(1 - np.cos(theta_diff))
    
    # Total loss
    loss = weight * (loss_mag + phase_lambda * loss_phase)
    
    # Gradients
    # ∂L/∂ρ = 2(ρ - ρ_t)
    grad_rho = 2 * weight * rho_diff
    
    # ∂L/∂θ = 2λ·sin(θ - θ_t)
    grad_theta = 2 * weight * phase_lambda * np.sin(theta_diff)
    
    # For complex gradient: W = θ + iρ, so ∂L/∂W̄ = (1/2)(∂L/∂θ + i·∂L/∂ρ)
    # But for our backprop we need grad that when multiplied gives the right update
    # Since W = θ + iρ, we have: grad_W_bar = grad_theta + 1j * grad_rho
    grad_W = grad_theta + 1j * grad_rho
    
    return float(loss), grad_W


def _compute_l2_loss_and_grad(
    W: np.ndarray,
    W_target: np.ndarray, 
    weight: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute standard L2 loss and gradient.
    
    Loss: L = w * |W - W_t|²
    Gradient: ∂L/∂W̄ = 2w(W - W_t)
    """
    diff = W - W_target
    loss = weight * np.sum(np.abs(diff)**2).real
    grad_W = 2 * weight * diff
    return float(loss), grad_W


# =============================================================================
# Generic Arbitrary-Order Loss and Gradient Computation
# =============================================================================

def compute_loss_and_grad(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute loss and analytic gradient for arbitrary scattering order.
    
    This function dynamically builds the scattering tree to the maximum
    order required by target_coeffs, then backpropagates gradients through
    the entire tree structure.
    
    Parameters
    ----------
    x : ndarray, shape (T,), complex
        Input signal
    hst : HeisenbergScatteringTransform
        HST instance
    target_coeffs : dict
        Target coefficients for any paths: {path: W_target}
        where path is a tuple like (), (j1,), (j1, j2), (j1, j2, j3), etc.
    weights : dict, optional
        Per-order weights {order: weight}. Default: all 1.0
    loss_type : str
        'l2': Standard |W - W_target|² loss
        'phase_robust': Circular phase distance + log-magnitude L2
    phase_lambda : float
        Weight on phase term for phase_robust loss (default 1.0)
        
    Returns
    -------
    loss : float
        Total loss
    grad_x : ndarray, shape (T,), complex
        Gradient of loss w.r.t. x
        
    Notes
    -----
    **Topology Preservation (Important for High Winding Signals)**
    
    For signals with high winding numbers (trajectories that encircle the
    origin many times), use `loss_type='phase_robust'` instead of 'l2'.
    
    The L2 loss treats phase as a linear quantity, which can cause issues
    at branch cuts (θ jumping from +π to -π). The phase_robust loss uses
    circular distance |exp(iθ) - exp(iθ_t)|², which is 2π-periodic and
    smooth across branch cuts.
    
    Additionally, use `lifting='radial_floor'` in the HST to preserve
    the winding number (topology) of the signal. The 'shift' lifting mode
    can destroy topology when epsilon > signal radius.
    
    Example for high-winding signals:
    
        hst = HeisenbergScatteringTransform(
            T, J=4, Q=4, lifting='radial_floor', epsilon=1e-8
        )
        loss, grad = compute_loss_and_grad(
            x, hst, targets, loss_type='phase_robust'
        )
    """
    T = len(x)
    
    # Determine maximum order needed from target_coeffs
    max_order_needed = max((len(path) for path in target_coeffs), default=0)
    
    if weights is None:
        weights = {i: 1.0 for i in range(max_order_needed + 1)}
    
    # Select loss function
    if loss_type == 'l2':
        loss_fn = _compute_l2_loss_and_grad
    elif loss_type == 'phase_robust':
        loss_fn = lambda W, Wt, w: _compute_phase_robust_loss_and_grad(W, Wt, w, phase_lambda)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    # =========================================================================
    # FORWARD PASS - Build complete scattering tree
    # =========================================================================
    
    # Data structures to cache intermediate values
    # W[path] = coefficient after R mapping
    # U_lifted[path] = coefficient after lifting (before R)
    # parent_W_hat[path] = FFT of parent's W (for convolution)
    
    W = {}           # After R mapping
    U_lifted = {}    # Before R (after lifting)
    
    # Lift input signal
    x_lifted, _ = hst._lift(x)
    x_hat = np.fft.fft(x_lifted)
    
    # --- Order 0: Father wavelet (no R) ---
    father_idx = -1
    S0_hat = x_hat * hst.filters[father_idx]
    S0 = np.fft.ifft(S0_hat)
    W[()] = S0  # Store for consistency (though Order 0 has no R)
    
    # --- Order 1 ---
    if max_order_needed >= 1:
        for j1 in range(hst.n_mothers):
            path = (j1,)
            U1_hat = x_hat * hst.filters[j1]
            U1 = np.fft.ifft(U1_hat)
            U1_lifted_val, _ = hst._lift(U1)
            U_lifted[path] = U1_lifted_val
            W[path] = hst._R(U1_lifted_val)
    
    # --- Order 2+ (recursive) ---
    for order in range(2, max_order_needed + 1):
        # Get all paths of previous order
        prev_paths = [p for p in W.keys() if len(p) == order - 1 and p != ()]
        
        for prev_path in prev_paths:
            W_prev = W[prev_path]
            W_prev_hat = np.fft.fft(W_prev)
            j_prev = prev_path[-1]
            
            for j_new in range(hst.n_mothers):
                if j_new <= j_prev:  # Monotone constraint: j_new > j_prev
                    continue
                
                new_path = prev_path + (j_new,)
                U_new_hat = W_prev_hat * hst.filters[j_new]
                U_new = np.fft.ifft(U_new_hat)
                U_new_lifted, _ = hst._lift(U_new)
                U_lifted[new_path] = U_new_lifted
                W[new_path] = hst._R(U_new_lifted)
    
    # =========================================================================
    # COMPUTE LOSS (and cache gradients w.r.t. W)
    # =========================================================================
    
    loss = 0.0
    grad_W_from_loss = {}  # Gradients from loss for each path
    
    # Order 0 loss (always L2, no R mapping)
    if () in target_coeffs:
        w0 = weights.get(0, 1.0)
        diff0 = S0 - target_coeffs[()]
        loss += w0 * np.sum(np.abs(diff0)**2).real
    
    # Order 1+ loss
    for path in target_coeffs:
        if len(path) == 0:
            continue  # Already handled Order 0
        
        order = len(path)
        w = weights.get(order, 1.0)
        
        if path in W:
            l, g = loss_fn(W[path], target_coeffs[path], w)
            loss += l
            grad_W_from_loss[path] = g
    
    # =========================================================================
    # BACKWARD PASS - Traverse tree in reverse order
    # =========================================================================
    
    # Gradient accumulators
    # grad_W_accum[path] = accumulated gradient flowing back to W[path]
    # from all its children in higher orders
    grad_W_accum = {path: np.zeros(T, dtype=np.complex128) for path in W if path != ()}
    
    # Process orders from highest to lowest
    for order in range(max_order_needed, 0, -1):
        paths_at_order = [p for p in W.keys() if len(p) == order]
        
        for path in paths_at_order:
            # Total gradient at W[path] = direct from loss + accumulated from children
            grad_W_direct = grad_W_from_loss.get(path, np.zeros(T, dtype=np.complex128))
            grad_W_total = grad_W_direct + grad_W_accum.get(path, np.zeros(T, dtype=np.complex128))
            
            # Backward through R: W = i·ln(U)
            # dW/dU = i/U
            dW_dU = 1j / U_lifted[path]
            grad_U_lifted = grad_W_total * np.conj(dW_dU)
            
            # Backward through lifting (additive, gradient passes through)
            grad_U = grad_U_lifted
            
            # Backward through convolution: U = W_parent * ψ_j
            # grad_W_parent = grad_U ★ ψ_j (correlation)
            j = path[-1]
            grad_U_hat = np.fft.fft(grad_U)
            grad_parent_hat = grad_U_hat * np.conj(hst.filters[j])
            grad_parent = np.fft.ifft(grad_parent_hat)
            
            # Accumulate into parent's gradient
            if order == 1:
                # Parent is x, accumulate to grad_x
                # We'll handle this after the loop
                pass
            else:
                parent_path = path[:-1]
                grad_W_accum[parent_path] += grad_parent
    
    # --- Collect Order 1 gradients into grad_x ---
    grad_x_hat = np.zeros(T, dtype=np.complex128)
    
    for j1 in range(hst.n_mothers):
        path = (j1,)
        if path not in W:
            continue
        
        grad_W_direct = grad_W_from_loss.get(path, np.zeros(T, dtype=np.complex128))
        grad_W_total = grad_W_direct + grad_W_accum.get(path, np.zeros(T, dtype=np.complex128))
        
        # Backward through R
        dW_dU = 1j / U_lifted[path]
        grad_U_lifted = grad_W_total * np.conj(dW_dU)
        grad_U = grad_U_lifted
        
        # Backward through filter
        grad_U_hat = np.fft.fft(grad_U)
        grad_x_hat += grad_U_hat * np.conj(hst.filters[j1])
    
    # --- Order 0 backward (no R) ---
    if () in target_coeffs:
        w0 = weights.get(0, 1.0)
        diff0 = S0 - target_coeffs[()]
        grad_S0 = 2 * w0 * diff0
        grad_S0_hat = np.fft.fft(grad_S0)
        grad_x_hat += grad_S0_hat * np.conj(hst.filters[father_idx])
    
    # Convert to time domain
    grad_x = np.fft.ifft(grad_x_hat)
    
    return float(loss), grad_x


# =============================================================================
# Order-Specific Wrappers (Backward Compatibility)
# =============================================================================


def compute_loss_and_grad_order2(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute Order-1 and Order-2 loss with analytic gradient.
    
    DEPRECATED: Use compute_loss_and_grad() for new code.
    This function is kept for backward compatibility.
    """
    return compute_loss_and_grad(x, hst, target_coeffs, weights, loss_type, phase_lambda)


def compute_loss_only(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
) -> float:
    """Compute loss without gradient (for finite difference testing)."""
    loss, _ = compute_loss_and_grad(x, hst, target_coeffs, weights, loss_type, phase_lambda)
    return loss


def finite_difference_gradient(
    x: np.ndarray,
    hst,
    target_coeffs: Dict[tuple, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    eps: float = 1e-7,
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
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
        
        loss_plus = compute_loss_only(x_plus, hst, target_coeffs, weights, loss_type, phase_lambda)
        loss_minus = compute_loss_only(x_minus, hst, target_coeffs, weights, loss_type, phase_lambda)
        
        grad[i] += (loss_plus - loss_minus) / (2 * eps)
    
    # Gradient w.r.t imaginary part
    for i in range(T):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += 1j * eps
        x_minus[i] -= 1j * eps
        
        loss_plus = compute_loss_only(x_plus, hst, target_coeffs, weights, loss_type, phase_lambda)
        loss_minus = compute_loss_only(x_minus, hst, target_coeffs, weights, loss_type, phase_lambda)
        
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
    loss_type: str = 'l2',
    phase_lambda: float = 1.0,
    grad_clip: Optional[float] = None,
    normalize: bool = False,
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
    loss_type : str
        'l2' or 'phase_robust'
    phase_lambda : float
        Weight on phase term for phase_robust loss
    grad_clip : float, optional
        Maximum gradient norm. If provided, gradients are clipped.
        Useful for signals with poor conditioning (e.g., radial_floor on
        zero-mean signals).
    normalize : bool
        If True, optimize on unit-energy signal internally and restore
        scale at the end. Improves conditioning for radial_floor lifting.
        
    Returns
    -------
    result : OptimizationResult
    """
    x = x0.copy().astype(np.complex128)
    
    # Optional energy normalization for better conditioning
    if normalize:
        x, original_scale = normalize_energy(x)
    else:
        original_scale = 1.0
    
    velocity = np.zeros_like(x)
    
    loss_history = []
    grad_norm_history = []
    
    # Compute initial signal energy for normalized logging
    initial_energy = np.linalg.norm(x0)
    
    for step in range(n_steps):
        loss, grad = compute_loss_and_grad(x, hst, target_coeffs, weights, loss_type, phase_lambda)
        
        grad_norm = np.linalg.norm(grad)
        signal_norm = np.linalg.norm(x)
        
        # Gradient clipping
        if grad_clip is not None and grad_norm > grad_clip:
            grad = grad * (grad_clip / grad_norm)
            grad_norm = grad_clip
        
        loss_history.append(loss)
        grad_norm_history.append(grad_norm)
        
        if verbose and step % max(1, n_steps // 10) == 0:
            # Show normalized loss (per unit energy) for scale-invariant tracking
            normalized_loss = loss / (signal_norm**2 + 1e-15)
            print(f"  Step {step:4d}: loss = {loss:.6e}, "
                  f"loss/||x||² = {normalized_loss:.6e}, |grad| = {grad_norm:.6e}")
        
        if callback is not None:
            callback(step, x, loss, grad)
        
        # Gradient descent with momentum
        velocity = momentum * velocity - lr * grad
        x = x + velocity
        
        # Ensure signal stays away from singularities (optional constraint)
        # This is handled by the lifting mechanism in the forward pass
    
    # Final loss
    final_loss, _ = compute_loss_and_grad(x, hst, target_coeffs, weights, loss_type, phase_lambda)
    loss_history.append(final_loss)
    
    # Restore original scale if normalized
    if normalize:
        x = denormalize_energy(x, original_scale)
    
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
