"""
HST Wavelets - Heisenberg Scattering Transform

A Python library implementing the Heisenberg Scattering Transform
based on Glinsky (2025), with two-channel (H⁺ ⊕ H⁻) extension.

Basic usage:
    >>> from hst import HeisenbergScatteringTransform
    >>> hst = HeisenbergScatteringTransform(T=512, J=4, Q=4, max_order=2)
    >>> output = hst.forward(signal)
    >>> print(output.energy_by_order())
    
Generation:
    >>> from hst import HSTGenerator
    >>> generator = HSTGenerator(hst)
    >>> result = generator.reconstruct_order2_only(output)
    
Low-level usage:
    >>> from hst import two_channel_paul_filterbank, forward_transform, inverse_transform
    >>> filters, info = two_channel_paul_filterbank(512, J=4, Q=4)
    >>> coeffs = forward_transform(signal, filters)
    >>> reconstructed = inverse_transform(coeffs, filters)
"""

__version__ = "0.2.0"

from .filter_bank import (
    two_channel_paul_filterbank,
    forward_transform,
    inverse_transform,
)

from .conformal import (
    glinsky_R,
    glinsky_R_inverse,
    simple_R,
    simple_R_inverse,
    simple_R_unwrapped,
    joukowsky,
    joukowsky_inverse,
)

from .scattering import (
    HeisenbergScatteringTransform,
    ScatteringOutput,
    ScatteringPath,
    hst_forward,
    hst_coefficients,
)

from .generation import (
    HSTGenerator,
    GenerationResult,
    reconstruct_from_coefficients,
    dream_signal,
    TORCH_AVAILABLE,
)

__all__ = [
    # High-level API
    "HeisenbergScatteringTransform",
    "ScatteringOutput",
    "ScatteringPath",
    "hst_forward",
    "hst_coefficients",
    # Filter bank
    "two_channel_paul_filterbank",
    "forward_transform",
    "inverse_transform",
    # Conformal maps
    "glinsky_R",
    "glinsky_R_inverse",
    "simple_R",
    "simple_R_inverse",
    "simple_R_unwrapped",
    "joukowsky",
    "joukowsky_inverse",
    # Generation
    "HSTGenerator",
    "GenerationResult",
    "reconstruct_from_coefficients",
    "dream_signal",
    "TORCH_AVAILABLE",
]
