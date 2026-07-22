# KFParticle GPU CBMRoot Input Contract

This note defines the first CBMRoot adapter boundary for the standalone
KFParticle GPU implementation. It deliberately describes data ownership and
validation only; it does not add a CBMRoot dependency to this directory.

## Scope Of The First Adapter

The first integration target is the default secondary two-daughter V0 stage:

- `K0S -> pi+ pi-`,
- `Lambda -> p pi-`,
- `anti-Lambda -> anti-p pi+`.

The adapter packs one or more events into `KFParticleGpuBufferManager`, runs
the existing `KFParticleGpuDecayPlan`, and transfers its candidate pool back to
the CBMRoot-side comparison or hand-off code. Higher-generation decays,
primary-V0 extrapolation, and replacement of the CPU finder are outside this
boundary.

## Stage 9.1 Execution-Path Decision

The first field-aware adapter belongs beside the modern
`cbm::algo::kfp::Selector`, not beside the older FairTask finder and not in the
specialized mCBM V0 trigger.

`algo::Reco` creates one `KfpSelectorChain` per OpenMP worker and calls
`ProcessEvent()` after `EventReconstruction` has produced a `RecoEvent`. The
chain's `Selector` already performs PID, preselection, and optional track
refitting, then creates the two CPU `KFPTrackVector` inputs immediately before
the CPU topology reconstructor is invoked. `TrackFitter` obtains field values
from `kf::Setup`, turns them into ten-coefficient `FieldRegion` objects, and
`Selector::MakeKfpTrackVector()` copies those coefficients to every track.
This is the narrowest current boundary that provides the GPU diagnostic with
the complete, field-aware CPU input without reimplementing track fitting.

The legacy `reco/KF/CbmKFParticleFinder` is a valid offline reference: it also
builds first/last track vectors, field regions, and a primary vertex before
calling its CPU topology reconstructor. It remains useful for later comparison
coverage, but is not the first implementation target because it is a separate
FairTask workflow.

`algo::kfp::V0Finder` is not a suitable field-aware target yet. It currently
accepts only Lambda reconstruction and explicitly fills all ten KFP field
coefficients with zero for mCBM. A GPU diagnostic may be added there later as
a separate zero-field mode, once its physics scope is made explicit; it must
not be used to validate the default field-aware V0 route.

The selector runs in an OpenMP event loop, while `KFParticleGpuRuntime` is
intentionally a process-wide singleton with one algorithm-local XPU queue and
one persistent buffer manager. The adapter therefore has a light handle at
each `KfpSelectorChain`, but all handles use one shared GPU diagnostic service.
The complete `pack -> launch -> download` transaction must be serialized while
the service owns its shared buffers. This preserves one KFParticle queue for
the process, never borrows a CBMRoot queue, and never creates a queue per
event. CPU selector work remains parallel; only the opt-in diagnostic GPU
transaction is serialized until a later batched design makes a different
ownership model worthwhile.

## Current Diagnostic Execution

With `CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON`, the CBMRoot-side
`GpuDiagnosticRunner` is called only after `Selector` has completed its normal
CPU `ReconstructParticles()` call. It receives exactly the first/last
`KFPTrackVector` objects that were supplied to the CPU finder, their
chi-to-primary-vertex values, and the same converted primary vertex. The CPU
selection result is returned unchanged regardless of the diagnostic outcome.

The runner uses `DigiEvent::fNumber` as its source-event identifier (the
32-bit value in the current GPU event descriptor is only an internal ABI
field). It records a structured result with the input counts, requested pair
capacity, candidate and selected-candidate counts, per-channel summaries, and
one of: completed, empty input, no pairs, input rejected, runtime unavailable,
execution failure, or pool overflow. Exceptions are converted to that status;
they never propagate into the CPU selector.

For the current default V0 plan, capacity is the exact sum of the physical
pair counts for K0S, Lambda, and anti-Lambda. Candidate, daughter-ID,
selected-index, and compact-task pools are then grown before fresh host views
are acquired and packed. This is deliberately conservative and lossless for
the diagnostic run. A later batching stage may replace this per-event capacity
policy without changing the data ABI.

## CPU/GPU Candidate Comparison

The diagnostic comparison never uses a candidate-pool index as identity. A
two-daughter V0 is keyed by `(source event ID, channel ID, canonical daughter
source IDs)`. The CPU extractor resolves the two topology-reconstructor
daughter indices through their single-daughter track particles; the GPU
snapshot reads the same source IDs from its flat CSR daughter pool. A CPU
candidate that cannot be resolved to two direct source tracks is counted as
`unresolvedCpuCandidates`, never silently matched.

The current comparison is order-independent and preserves duplicate keys as
separate records. For every default V0 channel it reports CPU/GPU candidate
counts, exact matches, CPU-only and GPU-only records, mass-validity, NDF, and
selection mismatches, plus maximum absolute mass, mass-error, and chi2
residuals. It also retains lineage-keyed discrepancy records. GPU snapshots
retain raw candidate flags, primary-vertex metadata, best-PV result, selection
class, rejection mask, and selected-pool membership. The current CPU
`KFParticle` output does not retain an equivalent selection record, so PV and
rejection values are intentionally GPU-only observations rather than
fabricated CPU/GPU mismatches. These values are diagnostics only; no tolerance
is used as a production reconstruction cut.

## Running The Diagnostic In CBMRoot

Building with `CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON` only makes the adapter
available. It remains inactive until the KFP selector YAML contains:

```yaml
kfp:
  selector:
    gpuDiagnostics:
      samplePeriod: 1
```

`samplePeriod: 1` processes every eligible selector event; a larger positive
value processes one event per period. If `DigiEvent::fNumber` is unavailable,
each thread-local selector uses its own monotonically increasing fallback
ordinal. After each timeslice, `Reco` emits one compact aggregate line with
attempted/completed/unavailable/failed event counts and CPU/GPU comparison
counts. No per-event log is emitted by default.

When this YAML block is present, `Reco` creates the KFP selector even if no
KFP trigger bit is configured. In that diagnostic-only mode the selector still
builds the CPU reference and GPU observation, but its CPU bitmask is not added
to `DigiEvent::fSelectionMask`. The diagnostic tail is isolated after CPU
particle reconstruction: unexpected packing, comparison, or reporting errors
are counted as failed diagnostics and cannot alter the CPU selection result.

## Required Track Input

For every physical track, without the SIMD padding used by `KFPTrackVector`,
the adapter must pack the following component-major arrays:

| GPU destination | CPU source / meaning |
| --- | --- |
| six numerical parameters | `KFPTrackVector::Parameter(0..5)` (`x, y, z, px, py, pz`) |
| 21 covariance elements | `KFPTrackVector::Covariance(0..20)` in existing KFParticle packed order |
| source ID | `KFPTrackVector::Id()` before CPU finder rewrites candidate IDs |
| PDG, charge, PV index, pixel hits | `PDG()`, `Q()`, `PVIndex()`, `NPixelHits()` |
| chi-to-primary-vertex | corresponding `ChiToPrimVtx` value for that track set |
| field region | ten `FieldCoefficient(0..9)` values when nonhomogeneous field is enabled |

`KFParticleGpuEventDesc` must retain the CPU set meaning exactly:

| GPU set index | CPU `vRTracks` index |
| --- | --- |
| `SecondaryPositiveFirst` | 0 |
| `SecondaryNegativeFirst` | 1 |
| `PrimaryPositiveFirst` | 2 |
| `PrimaryNegativeFirst` | 3 |
| `SecondaryPositiveLast` | 4 |
| `SecondaryNegativeLast` | 5 |
| `PrimaryPositiveLast` | 6 |
| `PrimaryNegativeLast` | 7 |

Each event descriptor stores absolute packed ranges for every populated set
and species. A range contains physical tracks only: CPU SIMD lanes introduced
for vector width alignment must never be packed or counted as candidates.

Primary vertices are packed independently into the GPU vertex SoA: three
parameters, six covariance elements, chi2, NDF, and number of contributors.
The first default V0 stage currently keeps `primaryVertexIndex = -1`, but
preserving vertices now avoids changing the input ABI when PV-dependent cuts
are enabled later.

## Field Contract

`KFParticleGpuFieldRegion` stores the same ten-coefficient parabolic region
already present in a nonhomogeneous `KFPTrackVector`:

```text
dz = z - coefficient[9]
B{xyz}(z) = coefficient[{0,3,6}]
         + coefficient[{1,4,7}] * dz
         + coefficient[{2,5,8}] * dz * dz
```

The CBMRoot adapter must request `nonhomogeneousField = true` before first
allocation and fill all ten coefficients for every packed track. A null GPU
field array is interpreted as zero field; it is not a valid substitute for a
CBM nonhomogeneous field. If the CPU input was built without
`NonhomogeneousField`, the field-aware default V0 route must be disabled or
reported as unavailable instead of silently being compared as physics-equivalent.

Current limitation: the validated first field-aware two-daughter algorithm
evaluates the first daughter's field region at its current z and uses its `By`
component in a constant-By DCA and energy-fit approximation. It transports the
full ten coefficients through the input contract so the later nonhomogeneous
transport can use them, but it is not yet equivalent to CPU `TransportCBM`.

## Ownership And Event Sequence

One `KFParticleGpuRuntime` and its XPU queue are process-persistent. The
CBMRoot adapter must not create a queue per event. For each batch it performs:

```text
EnsureCapacity -> SetInputSizes -> fill host SoA/event descriptors
-> UploadInput -> RunDecayPlan -> read host candidate pool/results
```

`EnsureCapacity` may grow allocations but never shrinks them. Packed views are
non-owning and become invalid after a growth operation, so the adapter obtains
fresh views only after `EnsureCapacity`. `RunDecayPlan` resets candidate pools
itself and is event-indexed; it must never receive pairs spanning two event
descriptors.

Candidate capacity and daughter-ID capacity are independent. CBMRoot must
inspect `KFParticleGpuTwoDaughterChannelResult::Truncated()` and overflow flags
before treating a GPU result as complete.

## XPU Build And Runtime Contract

KFParticle keeps its own persistent `xpu::queue` in both supported modes. A
queue is algorithm-local state: it owns the ordering of KFParticle uploads,
kernels, and downloads, while the XPU runtime and selected device remain
process-wide.

| Mode | XPU build ownership | Runtime initialization | Queue ownership |
| --- | --- | --- | --- |
| standalone KFParticle | KFParticle adds its selected XPU source tree to its own CMake build | `KFParticleGpuRuntime` initializes XPU when needed | `KFParticleGpuRuntime` |
| embedded CBMRoot | CBMRoot's top-level `xpu` target is shared directly | `cbm::Xpu` initializes XPU first; KFParticle detects and reuses it | `KFParticleGpuRuntime` |

The CBMRoot path must not configure KFParticle through `ExternalProject_Add`.
That would create a second XPU target graph and may install another
`libxpu.so`, backend driver, or device-image library into the same `build/lib`
directory. `CBM_KFPARTICLE_USE_XPU=ON` instead embeds the KFParticle CMake
project below CBMRoot and passes the existing `xpu` target to `xpu_attach()`.
Consequently all XPU images use the same headers, ABI, backend drivers, and
runtime library.

`KFParticleGpuRuntime::Finalize()` always releases only KFParticle buffers and
its queue. It never finalizes the global XPU runtime, whether that runtime was
initialized by CBMRoot or by standalone KFParticle.

## Selected V0 Hand-Off

After `RunDecayPlan()`, the compact output is available through
`LastDecayPlanSelectedCandidates()`, `LastDecayPlanSelectedChannels()`, and
the selected-index pool. A downstream GPU stage should construct
`KFParticleGpuSelectedV0View` from that pool, the raw candidate pool, and the
raw selection-result view. It resolves, without copying fit data:

| Handoff field | Source |
| --- | --- |
| raw candidate index | selected-index pool entry |
| channel ID | compact parallel channel-ID array and raw metadata |
| mother fit/covariance | raw candidate SoA at that index |
| mother PDG and event index | raw candidate metadata |
| canonical daughter source IDs | raw daughter pool through metadata offset/count |
| selection observables, class, rejection bits, best PV | raw selection-result record at that index |

Selected-index order inside a channel is atomic-compaction order, not physics
order. Queue-ordered channel launches make each channel's compact entries
contiguous; `LastDecayPlanSelectedChannels()` publishes those ranges. CBMRoot
comparison and later GPU stages must match by `(channel ID, event index, mother
PDG, canonical daughter source IDs)`. A selected-output overflow means the
selected list is incomplete even when raw construction did not overflow and
must stop a diagnostics-to-production promotion.

The standalone regression covers the three default channels, full host/device
selection-record equality, deterministic equal-PV ties, an exact
candidate-to-PV distance boundary, and selected-output truncation. These are
the minimum comparison counters an initial CBMRoot adapter must preserve.

The following remain CPU-only or approximate at this boundary: full
`SetProductionVertex` compatibility, CPU `TransportCBM` equivalence, primary
V0 extrapolation, higher-generation neutral-daughter/track-V0 reconstruction,
and final CPU finder replacement.

## First CPU/GPU Comparison

Initial validation should run the CPU finder and GPU default-V0 plan over the
same packed secondary tracks, then compare channel by channel:

1. total enumerated, accepted, and stored pair counts plus overflow status;
2. mother PDG, event index, candidate flags, and canonical daughter source IDs;
3. fitted `x/y/z`, momentum, energy, charge, chi2/NDF, mass, and mass error;
4. selected-index membership, selected-output overflow, and selection decision
separately from candidate construction.

The aggregate GPU transaction time includes queue acquisition, buffer growth,
host packing, transfer, kernels, download, and host snapshot creation. It is
not a kernel-only timing. Reproducible commands and the diagnostics-only
promotion gate are documented in `KFParticleGpuCbmRootValidation.md`.

Match candidates by `(channel ID, event ID, canonical daughter source IDs)`,
not by candidate array index. Channel identity preserves the physical daughter
assignment for the default channels; canonicalization only makes the comparison
robust to the CPU and GPU emitting the same pair in opposite storage order.

The standalone Step 7 tolerances are a device-consistency contract, not yet
CBM physics tolerances. The first CBMRoot comparison should record residual
distributions and acceptance/count differences for representative events
before any fixed production threshold is adopted.

## Integration Stop Conditions

Do not promote the GPU output past the comparison layer when any of these is
true:

- field coefficients are unavailable for a requested field-aware run;
- a packed event range includes SIMD padding or an invalid source ID;
- a channel reports candidate or daughter overflow;
- CPU/GPU matching cannot preserve daughter lineage;
- a discrepancy is attributable to the current constant-By approximation
  rather than a known, documented validation tolerance.

The next implementation step can therefore add a small CBMRoot adapter without
changing GPU ownership or data structures. It should first populate this
contract and return comparison diagnostics; replacing CPU V0 output is a later
decision.
