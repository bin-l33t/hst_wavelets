"""
PyTorch implementation of Heisenberg Scattering Transform.

This module provides GPU-accelerated HST for large-scale experiments.
"""

from hst_torch.scattering import HeisenbergScatteringTransformTorch, HSTOutputTorch
from hst_torch.filter_bank import (
    two_channel_paul_filterbank_torch,
    forward_transform_torch,
    lowpass_torch,
)

__all__ = [
    'HeisenbergScatteringTransformTorch',
    'HSTOutputTorch',
    'two_channel_paul_filterbank_torch',
    'forward_transform_torch',
    'lowpass_torch',
]
