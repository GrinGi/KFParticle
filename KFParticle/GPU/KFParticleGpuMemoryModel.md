# KFParticle GPU Memory Model

This document describes the current standalone KFParticle XPU ownership and
transfer model. It records implemented behavior, not a claim that every later
physics stage is already device-resident.

## Owners And Views

`KFParticleGpuRuntime` owns one process queue. `KFParticleGpuSteering` owns
the reconstruction orchestration. The visible buffer owner is
`KFParticleGpuDeviceStorage`:

```text
runtime queue
  -> steering
       -> device storage
            input: tracks, vertices, events, field coefficients
            work: two-daughter tasks and task counters
            raw output: candidates, daughters, raw counters, selection records
            selected output: candidate indices, channel IDs, selected counters
```

Every `xpu::buffer` grows monotonically through the capacity policy and is
released only when the runtime is finalized. `KFParticleGpuBufferManager` is a
temporary API adapter around this owner; `Storage()` exposes the allocation
layout for inspection.

`KFParticleGpuKernelState` owns no memory. It is a trivially-copyable set of
device pointers, SoA views, and capacities. It may be published to XPU
constant memory with `xpu::set<TheKFParticleFinder>`; the state-probe kernel
validates that ABI separately from reconstruction mathematics.

## Layout

Track, vertex, and fit values use component-major SoA arrays. A view stores
component pointers, logical size, and capacity/stride. Passing a view to a
kernel transfers this descriptor, never the arrays it references.

Candidate fit, covariance, metadata, daughter source IDs, and one
`KFParticleGpuV0SelectionResult` per raw candidate are separate raw-output
arrays. The selected V0 pool stores candidate indices plus their channel IDs;
it does not duplicate fit state, covariance, daughter lineage, or observables.

## Copies And Synchronization

| Boundary | Current operation | Frequency |
|---|---|---|
| Packed input -> device | `UploadInput()` H2D | once per event/batch |
| Diagnostic raw candidates -> device | `UploadCandidates()` H2D | diagnostics only |
| Counter reset -> device | small H2D copies | before a stage/channel |
| Task generation -> construction | queue ordering, no D2H | every channel |
| Channel bookkeeping | task/candidate scalar D2H status | once per channel |
| Final raw diagnostics | `DownloadCandidates()` D2H | once after `RunDecayPlan()` |
| Isolated selection diagnostics | `DownloadV0SelectionResults()` D2H | after `RunV0Selection()` |
| Final selected result | `DownloadSelectedCandidates()` D2H | after decay-plan selection |

The compact task path launches construction over task capacity and guards with
the device-side accepted-task counter. It therefore performs no host readback
between task generation and construction. Queue order provides the dependency.

## Present Boundaries

`RunDecayPlan()` uploads input once, builds channels into one persistent raw
candidate pool, runs configured default-V0 selection over that resident pool,
then performs final selected-index/channel-ID and raw-pool downloads. Generic channels
without an expected V0 mass skip the physics-specific selection continuation.
It currently retains scalar per-channel status readbacks to preserve the
host-visible channel-range API. This is deliberately distinct from copying the
raw SoA pool and is a later optimization target.

`RunV0Selection()` remains an explicit diagnostic API. It consumes the same
resident raw pool and packed primary vertices, but resets and downloads the
selected-index pool itself for isolated testing.
