#!/usr/bin/env python3
"""
Hamiltonian Physics Benchmarks

Implements proper Hamiltonian systems with explicit conjugate variables (q, p)
for testing Glinsky's "geodesic motion" claims.

Systems:
1. Coupled Harmonic Oscillator Chain (Linear - the "Hello World")
2. φ⁴ Lattice Field (Nonlinear - matches Glinsky's Fig 3)

All integrators are symplectic (preserve phase space volume).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional


@dataclass
class HamiltonianState:
    """State of a Hamiltonian system."""
    q: np.ndarray  # Generalized coordinates
    p: np.ndarray  # Conjugate momenta
    t: float       # Time
    
    def to_complex(self) -> np.ndarray:
        """Construct complex signal z = q + ip (Glinsky's prescription)."""
        return self.q + 1j * self.p
    
    def energy(self, hamiltonian: Callable) -> float:
        """Compute total energy H(q, p)."""
        return hamiltonian(self.q, self.p)


# =============================================================================
# Symplectic Integrators
# =============================================================================

def symplectic_euler(
    q: np.ndarray,
    p: np.ndarray,
    dHdq: Callable,
    dHdp: Callable,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symplectic Euler integrator.
    
    q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_{n+1})
    p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_n)
    """
    p_new = p - dt * dHdq(q, p)
    q_new = q + dt * dHdp(q, p_new)
    return q_new, p_new


def velocity_verlet(
    q: np.ndarray,
    p: np.ndarray,
    dHdq: Callable,
    dHdp: Callable,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity Verlet (Störmer-Verlet) integrator.
    
    Second-order symplectic method.
    """
    # Half step in p
    p_half = p - 0.5 * dt * dHdq(q, p)
    
    # Full step in q using p_half
    q_new = q + dt * dHdp(q, p_half)
    
    # Another half step in p using new q
    p_new = p_half - 0.5 * dt * dHdq(q_new, p_half)
    
    return q_new, p_new


def leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    dHdq: Callable,
    dHdp: Callable,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leapfrog integrator (equivalent to Verlet for separable Hamiltonians).
    """
    # Kick
    p_half = p - 0.5 * dt * dHdq(q, p)
    
    # Drift
    q_new = q + dt * dHdp(q, p_half)
    
    # Kick
    p_new = p_half - 0.5 * dt * dHdq(q_new, p_half)
    
    return q_new, p_new


# =============================================================================
# System 1: Coupled Harmonic Oscillator Chain
# =============================================================================

class CoupledHarmonicChain:
    """
    1D chain of coupled harmonic oscillators.
    
    Hamiltonian:
        H = Σ_i [p_i²/2m + k(q_i - q_{i+1})²/2 + ω₀²q_i²/2]
    
    This is the "Hello World" of Hamiltonian systems.
    The dynamics are LINEAR → geodesics should be perfect.
    
    Parameters
    ----------
    n_oscillators : int
        Number of oscillators in the chain
    mass : float
        Mass of each oscillator
    spring_k : float
        Coupling spring constant between neighbors
    omega0 : float
        On-site frequency (potential well)
    boundary : str
        'periodic' or 'fixed'
    """
    
    def __init__(
        self,
        n_oscillators: int = 32,
        mass: float = 1.0,
        spring_k: float = 1.0,
        omega0: float = 0.5,
        boundary: str = 'periodic',
    ):
        self.N = n_oscillators
        self.m = mass
        self.k = spring_k
        self.omega0 = omega0
        self.boundary = boundary
        
        # Build coupling matrix for convenience
        self._build_coupling_matrix()
    
    def _build_coupling_matrix(self):
        """Build the coupling matrix K such that F = -K @ q."""
        K = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            # On-site term
            K[i, i] = self.omega0**2
            
            # Neighbor coupling (must use += to avoid overwriting diagonal for N=1)
            if self.boundary == 'periodic':
                # Add coupling terms: each site couples to its neighbors
                # For site i: potential = k/2 * [(q_i - q_{i+1})^2 + (q_i - q_{i-1})^2]
                # This gives diagonal +2k and off-diagonal -k
                if self.N > 1:
                    K[i, i] += 2 * self.k / self.m
                    K[i, (i+1) % self.N] += -self.k / self.m
                    K[i, (i-1) % self.N] += -self.k / self.m
                # For N=1 with periodic BC, no coupling (particle to itself)
            else:  # fixed boundaries
                if i > 0:
                    K[i, i] += self.k / self.m
                    K[i, i-1] += -self.k / self.m
                if i < self.N - 1:
                    K[i, i] += self.k / self.m
                    K[i, i+1] += -self.k / self.m
        
        self.K = K
        
        # Compute normal modes (eigenvalues are squared frequencies)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        self.mode_frequencies = np.sqrt(np.maximum(eigenvalues, 0))
        self.mode_vectors = eigenvectors
    
    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute total energy."""
        kinetic = np.sum(p**2) / (2 * self.m)
        
        # Potential: on-site + coupling
        potential = 0.5 * self.omega0**2 * np.sum(q**2)
        
        # Coupling (sum of (q_i - q_{i+1})²)
        if self.boundary == 'periodic':
            dq = q - np.roll(q, -1)
        else:
            dq = np.diff(q)
        potential += 0.5 * self.k * np.sum(dq**2)
        
        return kinetic + potential
    
    def dHdq(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of H w.r.t. q (force = -dH/dq)."""
        return self.K @ q
    
    def dHdp(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of H w.r.t. p (velocity)."""
        return p / self.m
    
    def initial_state(
        self,
        mode: Optional[int] = None,
        energy: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> HamiltonianState:
        """
        Create initial state.
        
        Parameters
        ----------
        mode : int, optional
            If specified, excite only this normal mode
        energy : float
            Total energy
        random_seed : int, optional
            For reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if mode is not None:
            # Excite a single normal mode
            omega = self.mode_frequencies[mode]
            q = self.mode_vectors[:, mode].copy()
            
            # Scale to desired energy
            # E = (1/2)ω²A² for harmonic oscillator
            amplitude = np.sqrt(2 * energy / (self.m * omega**2 + 1e-10))
            q = amplitude * q / np.linalg.norm(q)
            p = np.zeros(self.N)
        else:
            # Random initial conditions
            q = np.random.randn(self.N)
            p = np.random.randn(self.N)
            
            # Scale to desired energy
            current_E = self.hamiltonian(q, p)
            scale = np.sqrt(energy / (current_E + 1e-10))
            q *= scale
            p *= scale
        
        return HamiltonianState(q=q, p=p, t=0.0)
    
    def evolve(
        self,
        state: HamiltonianState,
        dt: float = 0.01,
        n_steps: int = 1000,
        integrator: str = 'verlet',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evolve the system and return trajectory.
        
        Returns
        -------
        times : ndarray of shape (n_steps,)
        q_traj : ndarray of shape (n_steps, N)
        p_traj : ndarray of shape (n_steps, N)
        """
        if integrator == 'verlet':
            step_fn = velocity_verlet
        elif integrator == 'leapfrog':
            step_fn = leapfrog
        else:
            step_fn = symplectic_euler
        
        q = state.q.copy()
        p = state.p.copy()
        
        times = np.zeros(n_steps)
        q_traj = np.zeros((n_steps, self.N))
        p_traj = np.zeros((n_steps, self.N))
        
        for i in range(n_steps):
            times[i] = state.t + i * dt
            q_traj[i] = q
            p_traj[i] = p
            
            q, p = step_fn(q, p, self.dHdq, self.dHdp, dt)
        
        return times, q_traj, p_traj
    
    def complex_trajectory(
        self,
        state: HamiltonianState,
        dt: float = 0.01,
        n_steps: int = 1000,
        integrator: str = 'verlet',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve and return complex signal z(t) = q(t) + i*p(t).
        
        Returns
        -------
        times : ndarray of shape (n_steps,)
        z_traj : ndarray of shape (n_steps, N) complex
        """
        times, q_traj, p_traj = self.evolve(state, dt, n_steps, integrator)
        z_traj = q_traj + 1j * p_traj
        return times, z_traj


# =============================================================================
# System 2: φ⁴ Lattice Field Theory
# =============================================================================

class Phi4LatticeField:
    """
    φ⁴ lattice field theory in 1D.
    
    Hamiltonian:
        H = Σ_i [π_i²/2 + (φ_i - φ_{i+1})²/2 - μ²φ_i²/2 + λφ_i⁴/4]
    
    This is the nonlinear benchmark matching Glinsky's Fig 3.
    
    The parameters control the phase:
    - μ² > 0: Symmetric phase (single minimum at φ=0)
    - μ² < 0: Broken symmetry (double-well potential)
    
    Parameters
    ----------
    n_sites : int
        Number of lattice sites
    mu2 : float
        Mass parameter (μ²)
    lam : float
        Coupling constant (λ)
    boundary : str
        'periodic' or 'fixed'
    """
    
    def __init__(
        self,
        n_sites: int = 64,
        mu2: float = -1.0,  # Broken symmetry by default
        lam: float = 1.0,
        boundary: str = 'periodic',
    ):
        self.N = n_sites
        self.mu2 = mu2
        self.lam = lam
        self.boundary = boundary
    
    def hamiltonian(self, phi: np.ndarray, pi: np.ndarray) -> float:
        """Compute total energy."""
        # Kinetic
        H = 0.5 * np.sum(pi**2)
        
        # Gradient term (nearest neighbor coupling)
        if self.boundary == 'periodic':
            dphi = phi - np.roll(phi, -1)
        else:
            dphi = np.diff(phi)
            dphi = np.concatenate([dphi, [0]])
        H += 0.5 * np.sum(dphi**2)
        
        # Mass term
        H -= 0.5 * self.mu2 * np.sum(phi**2)
        
        # φ⁴ interaction
        H += 0.25 * self.lam * np.sum(phi**4)
        
        return H
    
    def dHdphi(self, phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Force on φ (= -∂H/∂φ is the equation of motion)."""
        # Gradient contribution (Laplacian)
        if self.boundary == 'periodic':
            laplacian = (np.roll(phi, 1) + np.roll(phi, -1) - 2*phi)
        else:
            laplacian = np.zeros_like(phi)
            laplacian[1:-1] = phi[:-2] + phi[2:] - 2*phi[1:-1]
            laplacian[0] = phi[1] - phi[0]
            laplacian[-1] = phi[-2] - phi[-1]
        
        # Total: -Laplacian + mass + nonlinear
        return -laplacian - self.mu2 * phi + self.lam * phi**3
    
    def dHdpi(self, phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Velocity (= ∂H/∂π)."""
        return pi
    
    def vacuum_expectation(self) -> float:
        """
        Classical vacuum expectation value ⟨φ⟩.
        
        For μ² < 0: ⟨φ⟩ = ±√(-μ²/λ)
        For μ² > 0: ⟨φ⟩ = 0
        """
        if self.mu2 < 0:
            return np.sqrt(-self.mu2 / self.lam)
        return 0.0
    
    def initial_state(
        self,
        configuration: str = 'kink',
        temperature: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> HamiltonianState:
        """
        Create initial state.
        
        Parameters
        ----------
        configuration : str
            'kink' - topological soliton connecting two vacua
            'phonon' - small oscillations around vacuum
            'random' - thermal-like random configuration
            'vacuum' - at the vacuum expectation value
        temperature : float
            For thermal fluctuations
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        v = self.vacuum_expectation()
        
        if configuration == 'kink':
            # Kink soliton: φ(x) = v * tanh(m(x - x₀))
            x = np.arange(self.N)
            x0 = self.N / 2
            m = np.sqrt(-self.mu2) if self.mu2 < 0 else 1.0
            phi = v * np.tanh(m * (x - x0) / 10)
            pi = np.zeros(self.N)
        
        elif configuration == 'phonon':
            # Small oscillation (normal mode)
            phi = v * np.ones(self.N)
            k = 2 * np.pi / self.N  # Lowest mode
            phi += 0.1 * v * np.sin(k * np.arange(self.N))
            pi = np.zeros(self.N)
        
        elif configuration == 'vacuum':
            phi = v * np.ones(self.N)
            pi = np.zeros(self.N)
        
        else:  # random
            phi = v + 0.5 * np.random.randn(self.N)
            pi = np.zeros(self.N)
        
        # Add thermal fluctuations
        if temperature > 0:
            phi += np.sqrt(temperature) * np.random.randn(self.N)
            pi = np.sqrt(temperature) * np.random.randn(self.N)
        
        return HamiltonianState(q=phi, p=pi, t=0.0)
    
    def evolve(
        self,
        state: HamiltonianState,
        dt: float = 0.01,
        n_steps: int = 1000,
        integrator: str = 'verlet',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evolve the system.
        
        Returns times, φ trajectory, π trajectory.
        """
        if integrator == 'verlet':
            step_fn = velocity_verlet
        elif integrator == 'leapfrog':
            step_fn = leapfrog
        else:
            step_fn = symplectic_euler
        
        phi = state.q.copy()
        pi = state.p.copy()
        
        times = np.zeros(n_steps)
        phi_traj = np.zeros((n_steps, self.N))
        pi_traj = np.zeros((n_steps, self.N))
        
        for i in range(n_steps):
            times[i] = state.t + i * dt
            phi_traj[i] = phi
            pi_traj[i] = pi
            
            phi, pi = step_fn(phi, pi, self.dHdphi, self.dHdpi, dt)
        
        return times, phi_traj, pi_traj
    
    def complex_trajectory(
        self,
        state: HamiltonianState,
        dt: float = 0.01,
        n_steps: int = 1000,
        integrator: str = 'verlet',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve and return z(t) = φ(t) + i*π(t).
        """
        times, phi_traj, pi_traj = self.evolve(state, dt, n_steps, integrator)
        z_traj = phi_traj + 1j * pi_traj
        return times, z_traj


# =============================================================================
# System 3: Nonlinear Schrödinger (bonus - natively complex)
# =============================================================================

class NonlinearSchrodinger:
    """
    1D Nonlinear Schrödinger Equation (NLS).
    
    i ∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ
    
    This is special: the field ψ is NATIVELY COMPLEX.
    No need to construct z = q + ip artificially.
    
    Supports soliton solutions for g < 0 (focusing NLS).
    
    Parameters
    ----------
    n_sites : int
        Number of spatial grid points
    dx : float
        Grid spacing
    g : float
        Nonlinearity strength (g < 0 for focusing, solitons exist)
    """
    
    def __init__(
        self,
        n_sites: int = 128,
        dx: float = 0.1,
        g: float = -1.0,  # Focusing by default
    ):
        self.N = n_sites
        self.dx = dx
        self.g = g
        
        # Build Laplacian matrix
        self._build_laplacian()
    
    def _build_laplacian(self):
        """Build discrete Laplacian with periodic BC."""
        self.laplacian = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.laplacian[i, i] = -2.0
            self.laplacian[i, (i+1) % self.N] = 1.0
            self.laplacian[i, (i-1) % self.N] = 1.0
        self.laplacian /= self.dx**2
    
    def dpsi_dt(self, psi: np.ndarray) -> np.ndarray:
        """
        Time derivative: dψ/dt = i(∂²ψ/∂x² - g|ψ|²ψ)
        """
        kinetic = self.laplacian @ psi
        nonlinear = self.g * np.abs(psi)**2 * psi
        return 1j * (kinetic - nonlinear)
    
    def energy(self, psi: np.ndarray) -> float:
        """
        Total energy (conserved).
        E = ∫ [|∂ψ/∂x|² + g|ψ|⁴/2] dx
        """
        # Gradient energy
        dpsi = np.diff(psi) / self.dx
        E_grad = np.sum(np.abs(dpsi)**2) * self.dx
        
        # Interaction energy
        E_int = 0.5 * self.g * np.sum(np.abs(psi)**4) * self.dx
        
        return E_grad + E_int
    
    def norm(self, psi: np.ndarray) -> float:
        """Total norm (conserved): N = ∫|ψ|² dx."""
        return np.sum(np.abs(psi)**2) * self.dx
    
    def soliton(self, amplitude: float = 1.0, x0: Optional[float] = None) -> np.ndarray:
        """
        Create a soliton solution (for focusing NLS, g < 0).
        
        ψ(x) = A sech(A(x-x₀)/√2) for g = -1
        """
        if x0 is None:
            x0 = self.N * self.dx / 2
        
        x = np.arange(self.N) * self.dx
        
        # Soliton width depends on amplitude and g
        width = np.sqrt(2 / abs(self.g)) / amplitude
        
        psi = amplitude / np.cosh((x - x0) / width)
        
        return psi.astype(np.complex128)
    
    def evolve_split_step(
        self,
        psi: np.ndarray,
        dt: float = 0.001,
        n_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve using split-step Fourier method (symplectic for NLS).
        
        Returns times and ψ trajectory.
        """
        psi = psi.copy().astype(np.complex128)
        
        # Frequency grid
        k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        
        # Propagators
        kinetic_prop = np.exp(-1j * k**2 * dt / 2)
        
        times = np.zeros(n_steps)
        psi_traj = np.zeros((n_steps, self.N), dtype=np.complex128)
        
        for i in range(n_steps):
            times[i] = i * dt
            psi_traj[i] = psi
            
            # Half step kinetic in Fourier space
            psi_hat = np.fft.fft(psi)
            psi_hat *= kinetic_prop
            psi = np.fft.ifft(psi_hat)
            
            # Full step nonlinear
            psi *= np.exp(-1j * self.g * np.abs(psi)**2 * dt)
            
            # Half step kinetic
            psi_hat = np.fft.fft(psi)
            psi_hat *= kinetic_prop
            psi = np.fft.ifft(psi_hat)
        
        return times, psi_traj


# =============================================================================
# Convenience functions
# =============================================================================

def create_hamiltonian_test_signals(
    system: str = 'harmonic',
    T: int = 512,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Create a complex test signal z(t) from a Hamiltonian system.
    
    Parameters
    ----------
    system : str
        'harmonic', 'phi4', or 'nls'
    T : int
        Number of time points
    **kwargs
        Passed to the system
    
    Returns
    -------
    z : ndarray of shape (T,) complex
        Complex signal z = q + ip
    info : dict
        System information
    """
    if system == 'harmonic':
        chain = CoupledHarmonicChain(**kwargs)
        state = chain.initial_state(mode=1, energy=1.0)
        times, z_traj = chain.complex_trajectory(state, n_steps=T)
        
        # Use the first oscillator as our signal
        z = z_traj[:, 0]
        info = {
            'system': 'Coupled Harmonic Chain',
            'n_oscillators': chain.N,
            'mode_frequencies': chain.mode_frequencies,
            'initial_energy': chain.hamiltonian(state.q, state.p),
        }
    
    elif system == 'phi4':
        field = Phi4LatticeField(**kwargs)
        state = field.initial_state(configuration='kink')
        times, z_traj = field.complex_trajectory(state, n_steps=T)
        
        # Use the center site
        z = z_traj[:, field.N // 2]
        info = {
            'system': 'φ⁴ Lattice Field',
            'n_sites': field.N,
            'mu2': field.mu2,
            'lambda': field.lam,
            'vacuum_expectation': field.vacuum_expectation(),
        }
    
    elif system == 'nls':
        nls = NonlinearSchrodinger(**kwargs)
        psi0 = nls.soliton(amplitude=1.0)
        times, psi_traj = nls.evolve_split_step(psi0, n_steps=T)
        
        # ψ is already complex!
        z = psi_traj[:, nls.N // 2]
        info = {
            'system': 'Nonlinear Schrödinger',
            'n_sites': nls.N,
            'g': nls.g,
            'initial_norm': nls.norm(psi0),
        }
    
    else:
        raise ValueError(f"Unknown system: {system}")
    
    return z, info
