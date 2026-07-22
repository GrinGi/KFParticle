# KFParticle GPU CBMRoot Validation Gate

Step 9 remains diagnostics-only. GPU output must not be routed into the
CBMRoot selection result until this gate has recorded CPU and HIP runs.

## Build And Unit Checks

```bash
cmake -S /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot \
  -B /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
  -DCBM_KFPARTICLE_USE_XPU=ON \
  -DCBM_KFPARTICLE_GPU_DIAGNOSTICS=ON

cmake --build /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
  --target _GTestKfpGpuDiagnosticPacker AlgoOffline -j

ctest --test-dir /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
  -R _GTestKfpGpuDiagnosticPacker --output-on-failure
```

Check embedded runtime ownership independently:

```bash
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_xpu_smoke.sh
```

Exercise the built CBMRoot diagnostic adapter on a minimal field-aware V0
input. This is external to the regular CBMRoot CTest suite.

```bash
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_gpu_diagnostic_smoke.sh
```

The macro verifies CBMRoot-owned XPU initialization, the KFParticle adapter,
default V0 execution, lineage, phase telemetry, reporter aggregation, and an
invalid-input rejection. It must not be added to the production CBMRoot test
set because it requires a configured accelerator.

## Hermetic CPU/GPU Equivalence Gate

Before a real reconstruction campaign, run the external synthetic executable.
It constructs the same controlled V0 input for the existing CPU
`KFParticleTopoReconstructor` and `GpuDiagnosticRunner`, then compare their
channel/lineage/count results exactly and their numerical fit observables with
explicit tolerances. The initial cases are K0S, Lambda, anti-Lambda, and
invalid input. Bounded-pool overflow remains covered by the standalone
lifecycle test, where allocation limits are directly controllable.

This is the Step 9 closure gate. It deliberately avoids detector files and
production ROOT macros while validating the old CPU finder against the new GPU
path. It does not demand bitwise equality: current CPU transport and the GPU
constant-By approximation have different floating-point execution paths.

```bash
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_cpu_gpu_v0_equivalence.sh
```

The executable uses a controlled constant-By field and the ordinary default GPU
V0 plan. It compares raw construction and lineage because the CPU candidate output has no equivalent
of the current GPU selection record. It checks K0S, Lambda, anti-Lambda, and
invalid input; pool-overflow behaviour remains covered by the standalone XPU
lifecycle test, where allocation limits are directly controllable.

After a successful run, the launcher writes an `*.evidence.txt` file beside the
log. It records the build/run commands, selected device, ROCm root and version
when available, host, log hash, and final status. Set
`KFPARTICLE_CBMROOT_EQUIVALENCE_EVIDENCE_FILE` to store that record elsewhere.

## Deferred Reconstruction Campaign

Enable diagnostics explicitly for a small stable sample:

```yaml
kfp:
  selector:
    gpuDiagnostics:
      samplePeriod: 1
```

Retain the reconstruction command, input identity, XPU device and architecture,
ROCm version, and the `KFParticle GPU diagnostics` line for each timeslice.
For longer runs, increase `samplePeriod`; this never changes CPU selection.

For a stable real input, the external campaign wrapper records independent
baseline and diagnostic logs without changing the reconstruction macro. This
is intentionally deferred until a later production-routing decision. Supply
two commands that differ only by the `gpuDiagnostics` YAML block; optional
output paths make it verify byte-identical CPU output.

```bash
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_BASELINE_COMMAND='your reconstruction command with diagnostics disabled' \
KFPARTICLE_CBMROOT_DIAGNOSTIC_COMMAND='your reconstruction command with diagnostics enabled' \
KFPARTICLE_CBMROOT_BASELINE_OUTPUT=/path/to/baseline-output.root \
KFPARTICLE_CBMROOT_DIAGNOSTIC_OUTPUT=/path/to/diagnostic-output.root \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_gpu_diagnostic_campaign.sh
```

The wrapper writes `validation-evidence.txt` beside its two logs. It records
the exact commands, host and kernel, selected XPU device, ROCm root/version,
log hashes, the aggregate diagnostic line, per-channel lines, and (when
requested) the byte-identical CPU-output check. Set
`KFPARTICLE_CBMROOT_EVIDENCE_FILE` to retain it outside the build directory.

The report contains host wall-time totals for CPU reference, input preparation,
queue/capacity, host packing, H2D input, construction pipeline, selection
pipeline, D2H output, result extraction, comparison, and the complete
transaction. The construction and selection timings include required queue
waits and scalar status copies; they are not device-event or kernel-only
measurements.

## Promotion Decision

Retain diagnostics-only output when a GPU status is unavailable, failed, or
overflowed; lineage matching has unexplained unresolved, CPU-only, or GPU-only
candidates; residuals are not understood relative to the present constant-By
approximation and CPU `TransportCBM`; or serialized per-event transactions do
not provide a useful throughput case.

A production-routing proposal requires stable lineage agreement, understood
residuals, validated overflow handling, and a batching design. Higher-
generation decays, full nonhomogeneous transport, and replacement of the CPU
finder are outside Step 9.
