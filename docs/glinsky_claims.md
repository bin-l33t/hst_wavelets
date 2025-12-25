# HST Wavelets: Next Steps Discussion

## Current Status (Verified)

| Component | Status | Test |
|-----------|--------|------|
| Two-channel filter bank | ✅ | Σ\|ψ̂\|² = 1.000000 |
| Forward/inverse transform | ✅ | Error < 1e-15 |
| Glinsky R mapping | ✅ | R⁻¹(R(z)) error < 1e-10 |
| Reconstruction after R | ✅ | Error < 1e-15 even for 42% neg freq |

## Glinsky's Claims Checklist

### Verified ✅
1. **Partition of Unity** (Page 15) - Father + mothers sum to 1
2. **Invertibility** (Page 15) - N log N forward/inverse
3. **R mapping structure** (Eq. 12) - Joukowsky-based conformal map

### Partially Verified ⚠️
4. **Analyticity preservation** - Only works for constrained inputs. General signals need two-channel bank.

### Not Yet Tested ⏳
5. **Phase-space input** (f = π + iφ) - Need Hamiltonian test case
6. **Multi-layer recursion** (Eq. 21) - Only single layer implemented
7. **RG flow linearity** (Page 6) - Physics claim, need dynamics test
8. **Covariance properties** - Group action tests needed

## Proposed Next Steps

### Priority 1: Complete Single Layer
- [ ] Implement full HST layer: `convolve → R → convolve`
- [ ] Test reconstruction through full layer
- [ ] Benchmark performance

### Priority 2: Multi-Layer Recursion
- [ ] Implement Eq. 21 recursive structure
- [ ] Determine scattering path indexing
- [ ] Test depth-wise reconstruction

### Priority 3: Physics Validation
- [ ] Create harmonic oscillator test case (p + iq)
- [ ] Test anharmonic oscillator
- [ ] Verify RG interpretation

### Priority 4: Applications
- [ ] Asset price representation (log price + i·returns?)
- [ ] Correlation structure extraction
- [ ] Compare with Mallat scattering on same data

## Open Design Questions

### Q1: How to handle multi-layer reconstruction?

Standard scattering: `|W₁|` → `|W₂|` → ... (lossy, no reconstruction)

HST: `R(W₁)` → `R(W₂)` → ... (claimed invertible)

Do we need to store intermediate coefficients? Or does R⁻¹ composition work?

### Q2: What's the right input representation for finance?

Options:
- `log(price) + i·returns`
- `price + i·d(price)/dt`
- Hilbert analytic signal of prices

Which satisfies Glinsky's "analytic trajectory" condition?

### Q3: Should we implement single-channel (H⁺ only) option?

Pros:
- Matches Glinsky's paper more closely
- Simpler, fewer filters
- Group-theoretic purity

Cons:
- Only works for constrained inputs
- 50% energy loss for general signals

### Q4: CUDA optimization priorities?

Current bottleneck is likely:
- FFT (already optimized via cuFFT)
- Filter application (embarrassingly parallel)
- R mapping (element-wise, trivial to parallelize)

Probably already fast enough with PyTorch defaults?

## Code Quality TODOs

- [ ] Add proper docstrings (NumPy style)
- [ ] Add type hints
- [ ] Set up pytest with fixtures
- [ ] Add CI/CD
- [ ] Add benchmarks
- [ ] Create tutorial notebook

## Files Delivered

```
hst/
├── __init__.py           # Public API
├── filter_bank.py        # Two-channel Paul wavelet bank
├── conformal.py          # Glinsky R mapping
└── tests/
    └── test_glinsky_claims.py  # Comprehensive tests
```

All tests pass with:
- Partition of unity: exact (1.000000)
- Reconstruction: ~1e-16 error
- R inverse: ~1e-11 error
