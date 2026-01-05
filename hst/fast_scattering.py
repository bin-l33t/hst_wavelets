import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Use the existing R implementation if available, or define minimal version
try:
    from .conformal import glinsky_R_torch
except ImportError:
    def glinsky_R_torch(z, eps=1.0): # CHANGED default eps to 1.0 for stability
        # Fallback: R(z) = i * ln(z)
        # We use eps=1.0 ("lifting") to ensure log(r) ~ 0 when r ~ 0
        # This prevents energy explosion for small wavelet coefficients
        r = torch.abs(z) + eps
        theta = torch.angle(z)
        return -theta + 1j * torch.log(r)

class FastHST(nn.Module):
    """
    Fast Heisenberg Scattering Transform (Decimated/Orthogonal).
    
    Implements the algorithm from Glinsky (2025) Section VII:
    1. "Mother Wavelet of constant length" (Fixed kernel size)
    2. "The signal is then binated" (Decimation by 2)
    3. "N log N scaling" (Due to shrinking signal size)
    
    This architecture prevents the "energy from nowhere" problem by ensuring
    the representation is not redundant (critically sampled).
    """
    def __init__(self, J=4, kernel_size=16, m=4, rectifier='glinsky'):
        super().__init__()
        # DEBUG PRINT TO VERIFY VERSION
        print(f"DEBUG: FastHST initialized with J={J}, rectifier={rectifier} (Normalization Fixed)")
        self.J = J
        self.rectifier = rectifier
        self.kernel_size = kernel_size
        
        # 1. Build Constant-Length Kernels (Time Domain)
        # Glinsky: "Mother Wavelet of constant length in samples"
        # We use a discretized Cauchy-Paul wavelet: psi(t) ~ (1 - it)^-(m+1)
        t = torch.linspace(-2, 2, kernel_size)
        
        # Analytic Mother Wavelet (H+)
        z = 1 - 1j * t
        psi = (1/z)**(m+1)
        psi = psi - psi.mean()      # Admissibility (zero mean)
        
        # Father Wavelet (Smoothing / Anti-aliasing)
        # Gaussian approximation for the scaling function
        phi = torch.exp(-2 * t**2)
        phi = phi / phi.sum()       # L1=1 to preserve DC mean
        
        # Normalization for Decimated Transform (Parseval Frame)
        # In a decimated transform (downsample by 2), the filters must have norm 1/sqrt(2) 
        # (approx) to conserve energy, because we are splitting the signal bandwidth.
        # Classic orthogonal wavelets satisfy |H(w)|^2 + |G(w)|^2 = 2.
        # Here we approximate by normalizing L2 norm to 1/sqrt(2).
        
        # Normalize energy of psi
        psi = psi / torch.norm(psi) / np.sqrt(2) 
        
        # Normalize energy of phi
        # We divide by sqrt(2) because decimation removes half the samples/energy.
        phi = phi / torch.norm(phi) / np.sqrt(2)

        # Register buffers (complex weights split into real/imag for conv1d)
        # Shape: (Out, In, Len) = (1, 1, K)
        # We start with float32 but will cast during forward if needed
        self.register_buffer('psi_real', psi.real.view(1, 1, -1).to(torch.float32))
        self.register_buffer('psi_imag', psi.imag.view(1, 1, -1).to(torch.float32))
        self.register_buffer('phi', phi.view(1, 1, -1).to(torch.float32))
        
        # Padding for "same" convolution
        self.pad = kernel_size // 2

    def _conv_complex(self, x, w_real, w_imag):
        """Complex convolution using real ops: (a+bi)*(c+di) = (ac-bd) + i(ad+bc)"""
        # Ensure weights match input dtype
        w_real = w_real.to(x.real.dtype)
        w_imag = w_imag.to(x.real.dtype)
        
        xr, xi = x.real.unsqueeze(1), x.imag.unsqueeze(1)
        
        # Convolution logic
        rr = F.conv1d(xr, w_real, padding=self.pad)
        ri = F.conv1d(xr, w_imag, padding=self.pad)
        ir = F.conv1d(xi, w_real, padding=self.pad)
        ii = F.conv1d(xi, w_imag, padding=self.pad)
        
        out_r = rr - ii
        out_i = ri + ir
        
        # Remove extra padding if kernel size is even/odd mismatch
        if out_r.shape[-1] > x.shape[-1]:
            out_r = out_r[..., :x.shape[-1]]
            out_i = out_i[..., :x.shape[-1]]
            
        return torch.complex(out_r.squeeze(1), out_i.squeeze(1))

    def _conv_real(self, x, w):
        """Complex signal * Real filter"""
        # Ensure weights match input dtype
        w = w.to(x.real.dtype)
        
        xr, xi = x.real.unsqueeze(1), x.imag.unsqueeze(1)
        out_r = F.conv1d(xr, w, padding=self.pad)
        out_i = F.conv1d(xi, w, padding=self.pad)
        
        if out_r.shape[-1] > x.shape[-1]:
            out_r = out_r[..., :x.shape[-1]]
            out_i = out_i[..., :x.shape[-1]]

        return torch.complex(out_r.squeeze(1), out_i.squeeze(1))

    def forward(self, x):
        """
        Input: x (Batch, Time) complex
        Output: List of coefficients [S0, S1, S2...] at decreasing resolutions
        """
        # Ensure input is complex
        if not x.is_complex():
            x = x.to(torch.complex128) # Default to double for scientific calc
        
        coefficients = []
        current_x = x
        
        for j in range(self.J):
            # --- Branch 1: Invariant Coefficients (The "Output") ---
            # Convolve with Father (Lowpass) -> Decimate
            # This captures the "mean" behavior at this scale
            s_out = self._conv_real(current_x, self.phi)
            s_out = s_out[..., ::2] # Bination (Decimation)
            coefficients.append(s_out)
            
            # --- Branch 2: Scattering Propagator (The "Next Layer") ---
            # 1. Convolve with Mother (Highpass)
            u = self._conv_complex(current_x, self.psi_real, self.psi_imag)
            
            # 2. Rectify (Non-linearity)
            if self.rectifier == 'glinsky':
                # Use default eps=1.0 defined in this file's glinsky_R_torch
                w = glinsky_R_torch(u)
            else:
                w = torch.abs(u).to(u.dtype) # Mallat mode
                
            # 3. Binate (Decimate)
            current_x = w[..., ::2]
            
            # Sanity check: Stop if signal vanishes
            if current_x.shape[-1] < self.kernel_size:
                break
                
        return coefficients
