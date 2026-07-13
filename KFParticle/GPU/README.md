# KFParticle GPU infrastructure

This directory contains the experiment-independent implementation of
KFParticle reconstruction on XPU devices.

The GPU implementation requires C++17. It intentionally avoids C++23-only
features until C++23 becomes the minimum supported KFParticle standard.

The intended first CBMRoot adapter boundary, including field-coefficient
semantics and CPU/GPU comparison order, is documented in
[`KFParticleGpuCbmRootIntegration.md`](KFParticleGpuCbmRootIntegration.md).

## Ownership and lifetime

Ownership is process-wide:

```text
KFParticleGpuRuntime
  `- KFParticleGpuSteering
       |- KFParticleGpuBufferManager
       |- KFParticleGpuKernels (constant-memory state)
       `- KFParticleGpuDecayPlan
```

`KFParticleGpuRuntime` is initialized once per process. It attaches to an
already initialized XPU runtime or initializes XPU in standalone mode, preloads
the KFParticle device image, and owns one persistent queue. Finalization
destroys KFParticle resources but leaves the shared XPU runtime active.

`KFParticleGpuSteering` owns the reconstruction workflow. It reuses allocations
between events and launches the stages described by `KFParticleGpuDecayPlan`.
`KFParticleGpuBufferManager` owns XPU allocations. `KFParticleGpuKernels` is a
trivially-copyable collection of non-owning device views and device methods,
published through `TheKFParticleFinder` constant memory before kernel launch.
Views never own memory and must not outlive their buffers.

## Data layout

Persistent track and candidate data use component-major structure-of-arrays
storage. Adjacent GPU threads therefore access adjacent particle values when
reading the same parameter or covariance component.

Thread-local numerical states are separate from persistent storage:

```text
KFParticleGpuTrackState  input track parameters and covariance
KFParticleGpuFitState    mutable KF fit state
KFParticleGpuFieldRegion magnetic-field approximation
```

Fit states contain only values used by KF mathematics. PDG codes, particle
identifiers, primary-vertex associations, and daughter references belong to
separate metadata storage so they do not increase per-thread register pressure.

## Batching and candidate pools

Event descriptors will provide offsets into shared track, vertex, and candidate
buffers. Kernels must not infer event boundaries from global indices.

The input descriptor preserves the eight `KFPTrackVector` sets used by the CPU
Finder: primary or secondary, positive or negative, and first or last hit
position. Species ranges are stored as absolute ranges in the packed track SoA
and exclude CPU SIMD padding.

Input `SourceId` is the original track identifier used for daughter lineage.
It is intentionally distinct from the candidate identifier assigned during
reconstruction; the CPU implementation stores both meanings in `Id` at
different times.

Candidate pools have an explicit size, capacity, and overflow flag. Kernels
never write past capacity. A stage that cannot store all accepted candidates
reports overflow to the steering layer instead of silently truncating memory.

Variable-length daughter lists use offset/count metadata with a flat CSR buffer
of final input `SourceId` values. Flattened lineage makes duplicate-track checks
independent of the intermediate decay path.

Candidate and daughter counters are non-owning pointers in the pool view.
Atomic reservation and overflow-bit updates belong to the kernel layer; the
data view only checks capacities and writes to already reserved ranges.

Early task kernels use fixed candidate slots to keep the first mathematical
ports deterministic and easy to compare with host references. A failed task
therefore leaves a reserved candidate slot marked with `KFGpuCandidateBuildFailed`
and `DaughterCount = 0`; later compacting stages can replace this with atomic
or prefix-sum allocation once the physics content is stable.

## Execution rules

- XPU initialization and queue construction happen outside the event loop.
- Device allocations are owned by `KFParticleGpuBufferManager` and reused when their
  capacity is sufficient.
- XPU action structs delegate to `KFParticleGpuKernels` through
  `TheKFParticleFinder`; host orchestration remains in `KFParticleGpuSteering`.
- Cheap metadata cuts run before loading a full covariance matrix.
- Kernel fusion is considered only after individual stages are validated and
  profiled.
- GPU results are checked against scalar KFParticle calculations before a new
  mathematical operation is used by a reconstruction stage.

GPU sources are enabled with `KFPARTICLE_USE_XPU`. Standalone builds can supply
`KFPARTICLE_XPU_SOURCE_DIR`; parent builds may provide an existing `xpu` target.

## Round-trip kernel

`KFParticleGpuRoundTrip` is the first ABI and lifecycle smoke test. It
maps each input track to the same candidate index, initializes a fit state with
a supplied mass hypothesis, and writes metadata plus one source-track daughter.
The deterministic mapping deliberately avoids atomics; it is not a particle
finding stage.

The host launch sequence is owned by `KFParticleGpuSteering`:

```text
upload input -> reset output -> launch -> wait -> download output
```

Candidate and daughter capacities limit the output independently. The kernel
sets the corresponding overflow bits and never writes outside either buffer.

## XPU lifecycle test

The standalone test builds without ROOT or FairRoot. The simplest entry point
is the runner script:

```sh
KFParticle/GPU/test/run_xpu_lifecycle_test.sh
KFPARTICLE_GPU_TEST_DEVICE=hip0 KFParticle/GPU/test/run_xpu_lifecycle_test.sh
```

The first command is a CPU backend smoke test. The second one builds the HIP
device image and runs the same lifecycle checks on `hip0`. Backend-specific XPU
cache variables can be passed through the environment, for example
`XPU_HIP_ARCH=gfx906` or `XPU_ROCM_ROOT=/opt/rocm`.

The runner derives the CBMRoot checkout from its own location and builds by
default in the sibling build tree
`<CBMRoot-parent>/build/kfparticle-gpu-xpu-test`. This matches the usual
`cbmroot/` and `build/` workspace layout. Use
`KFPARTICLE_GPU_TEST_BUILD_DIR=/path/to/build` to override that location for an
isolated build or debugging session.

The lifecycle test currently launches the round-trip kernel with explicit
kernel arguments. This isolates buffer upload, HIP kernel launch, execution,
and download while the HIP `xpu::set` constant-memory path is investigated
separately.

The same test can also be driven manually with CMake:

```sh
cmake -S KFParticle/GPU/test/xpu -B build-kfparticle-xpu-test
cmake --build build-kfparticle-xpu-test
ctest --test-dir build-kfparticle-xpu-test --output-on-failure
```

It verifies XPU initialization and attachment, image preload, persistent queue
identity, reusable `buf_io` allocations, input upload, candidate reset and
download, empty inputs, and runtime reattachment after KFParticle finalization.
