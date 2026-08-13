# GPU Numerical Reproducibility

## Purpose

The CPU KFParticle finder and the HIP implementation must follow the same
ordered single-precision calculation closely enough that small intermediate
rounding differences do not grow into different covariance, topology, or
selection decisions. Clang may otherwise contract an expression such as
`a * b + c` into one fused multiply-add (FMA). FMA is normally faster and more
accurate in isolation, but it rounds once instead of reproducing the two
rounding points used by the current CPU SIMD reference.

The port therefore disables only floating-point contraction, and only inside
the CPU-parity functions listed below. It does not disable GPU parallelism,
normal compiler optimization, inlining, or hardware arithmetic elsewhere.

## Current contraction-off sites

`KFParticleGpuMath.h`:

- `CpuCompatibleAtan2`: ordered polynomial used by the CPU-compatible DCA
  angle calculation.
- `CpuCompatibleSinCos`: ordered range reduction and sine/cosine polynomials.
- `MultQSQt`: covariance transport matrix products.
- `TransportFullField`: full-field state, Jacobian, and covariance transport.
- `GetDStoParticleBzCpuCompatibleDerivatives`: two-particle DCA roots and
  derivatives used by the CPU-compatible path.
- `GetDStoParticleBzCpuCompatible`: corresponding value-only DCA calculation.
- `BuildFullFieldDcaMeasurementSeedAnalytic`: coupled full-field daughter
  measurement and cross-covariance construction.
- `AddDaughterWithEnergyFit`: Kalman energy-fit update where small covariance
  differences can be amplified.

`KFParticleGpuField.h`:

- `KFParticleGpuFieldRegion::Get`: quadratic field interpolation. This nested
  function needs its own pragma because a pragma in its caller does not govern
  a separately compiled function body.

Every site uses `#pragma clang fp contract(off)` under `__clang__`. These are
the first locations to revisit during performance hardening. Any attempted
removal must be checked on HIP with the complete lifecycle suite, especially
the real-scale full-field regressions, default V0 references, fused routing,
and selection-boundary tests.

## Debugging

The lifecycle test keeps its detailed DCA, transport, covariance, energy-fit,
and selection traces, but normal successful runs print only PASS records.
Set `KFPARTICLE_GPU_TEST_VERBOSE_TRACE=1` when the full intermediate CPU/GPU
comparison is required. A true tolerance failure still prints its compact
`DETAIL` diagnosis automatically; the verbose flag adds the complete trace.

## Diagnostic comparison bounds

Parity tolerances support both an absolute floor and a relative term. Existing
callers retain purely absolute comparison because every relative term defaults
to zero. The CBMRoot default-V0 policy additionally permits a `5e-4` relative
parameter residual and a `5e-2` relative fit-scalar residual while retaining
its absolute limits.

This was introduced from an offline detector event whose CPU SIMD and GPU
scalar paths produced the same K0S lineage and mass within `1.2e-5`, while a
position differed by `1.9e-3` at a scale of about `6.3` and chi2 differed by
about three percent. Replaying the GPU builder on the host reproduced the
device result, excluding transfer corruption and a device race. A fixed
`1e-3` absolute limit therefore classified expected scale-dependent SIMD versus
scalar rounding as a physics mismatch. The combined bound still rejects
differences larger than either the validated absolute floor or the small
relative fraction and does not alter reconstruction or selection decisions.

## Expected performance effect

There is no reliable end-to-end percentage without profiling the target GPU.
The pragma can turn some single FMA instructions into a multiply plus an add,
so the affected arithmetic fragments can theoretically approach twice as many
instructions in the FMA-heavy limit. In this implementation the restriction
is local, and routing, memory transfers, synchronization, selection, and most
control flow are unchanged. A reasonable planning estimate is a low
single-digit to roughly 10% cost for the full-field fit stage, and usually less
for the whole reconstruction chain. This is a hypothesis for Step 20
measurement, not a performance claim.
