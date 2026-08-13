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
      mode: diagnostic
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

## Step 20 Qualified Routing

The same optional block now selects one of three explicit modes:

| `mode` | Behaviour |
| --- | --- |
| `cpu-only` | Run and publish the established CPU finder only |
| `diagnostic` | Publish CPU output and compare an isolated GPU observation |
| `qualified-gpu` | Prepared no-reference GPU route; locked to an event-atomic CPU fallback until Step 20.3 qualifies the exact mode/backend manifest |

Omitting `gpuDiagnostics` remains the normal CPU-only default. Existing
configurations that provide the block without `mode` retain `diagnostic`
semantics.

The first production capability manifest is deliberately narrower than the
complete external GPU graph. It admits only the three default field-aware
track-track V0 channels (K0S, Lambda, anti-Lambda), the
construct/transport/select operation set, and the default CPU finder cuts.
This prevents later graph work from becoming production-visible merely
because its device kernels exist.

`validated-gpu` remains a compatibility spelling for `qualified-gpu`.
Stage 20.2 checks capability and the qualification lock before attaching to
XPU. While the lock is closed, the ordinary CPU finder runs and no GPU output
can be published. The prepared unlocked branch runs GPU before CPU, requires a
complete materialized event, and publishes through one `ReplaceParticles()`
swap. Any unsupported scope, invalid input or field, unavailable runtime,
overflow, execution error, or materialization failure runs the complete CPU
event instead. There is no partial CPU/GPU candidate merge. Diagnostic mode
continues to run and compare both implementations.

`GpuRoutingMonitor` records one deterministic reason per decision: GPU
accepted, explicit CPU request, diagnostic-only, not sampled, qualification
locked, unsupported
capability, unavailable runtime, invalid input, invalid field, overflow,
failed validation, failed materialization, or execution failure. CBMRoot
prints these counters once per timeslice beside the existing diagnostic
monitoring.

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

Current implementation: default V0 diagnostics use the full packed field
region for each daughter during the bounded DCA/transport seed. The explicit
`KFGpuTransportFieldAware` compatibility mode remains a constant-By baseline
for diagnostics. The full-field energy-fit path still uses the validated
line-DCA cross-daughter correlation approximation, so it is not yet equivalent
to the complete CPU `TransportCBM` covariance contract.

## Planned Full-Field Upgrade (Step 11)

Step 11 keeps this input ABI and the process-local KFParticle queue unchanged.
Its remaining work validates the full-field default-V0 diagnostic path through
the CBMRoot boundary and records its cost. Missing, non-finite, or
non-convergent field input is reported as `GpuDiagnosticStatus::FieldRejected`;
it must never be silently treated as a physics-equivalent zero-field result.
CPU reconstruction remains authoritative throughout this upgrade.

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

Selected-index order inside a channel is atomic-scatter order, not physics
order. The device prefix phase reserves one bounded segment per event and
descriptor before scatter; `LastDecayPlanSelectedChannels()` publishes those
ranges. CBMRoot
comparison and later GPU stages must match by `(channel ID, event index, mother
PDG, canonical daughter source IDs)`. A selected-output overflow means the
selected list is incomplete even when raw construction did not overflow and
must stop a diagnostics-to-production promotion.

The standalone regression covers the three default channels, full host/device
selection-record equality, deterministic equal-PV ties, an exact
candidate-to-PV distance boundary, and selected-output truncation. These are
the minimum comparison counters an initial CBMRoot adapter must preserve.

## Mask-Driven Continuation Contract (Steps 15+)

The qualified default-V0 and Xi/Omega generations both use device-resident
descriptor tables, role masks, compatibility lookup, and block-scan task
compaction. Default-V0 selection continues directly from its shared raw
generation through a descriptor-index sidecar.

The integration boundary remains unchanged:

- CBMRoot packs event/species ranges and physical metadata; it does not build
  channel masks or launch individual decay channels.
- KFParticle keeps descriptor tables, compatibility lookup data, task queues,
  and intermediate candidate pools on the selected XPU device.
- Range filtering happens before mask lookup. A routing kernel visits a pair
  once per compatible source-range group and emits one task for every active
  channel bit.
- A channel bit is never persisted as physics identity. Tasks and candidates
  retain the stable channel ID, event ID, and canonical source lineage used by
  the current comparison gate.
- Masks perform only discrete hypothesis routing. Field transport, DCA,
  fitting, topology, and continuous cuts retain their validated numerical
  paths and run only for compacted tasks.
- Reconstruction remains an ordered pipeline per dependency generation, not
  one monolithic kernel. The KFParticle-owned persistent queue provides the
  generation barriers in standalone and embedded CBMRoot modes.

The old explicit cascade route and the simple atomic fused route remain test
oracles. Production cascade steering uses block-scan routing; this change does
not itself enable replacement of the complete CPU finder.

Step 15 is executed in three externally visible checkpoints:

1. Routing masks, descriptors, execution groups, and the scalar oracle are
   added without changing the active reconstruction path.
2. A fused atomic-reference route is added and compared with the explicit
   route on CPU and HIP. CBMRoot still observes the same stable channel IDs,
   lineage, aggregate statuses, and diagnostic-only ownership.
3. XPU block-scan compaction replaces per-task global reservations when its
   measured result is beneficial, steering switches to generation/group
   launches, and the CBMRoot cascade smoke closes the step.

Descriptor and compatibility-table uploads are plan-revision operations, not
event operations. CBMRoot is not responsible for their memory, bit numbering,
or construction. Step 15 must not add a host wait between routing and candidate
construction, and it must not require a CBMRoot source change outside the
external KFParticle diagnostic harness.

Step 15 implements this boundary without changing CBMRoot source outside the
external diagnostic harness. `KFParticleGpuV0TrackRoutingPlan` compiles the host decay
plan into flat descriptors, role compatibility entries, and two source-range
execution groups for the current Xi/Omega channels. The resulting tables,
enabled-channel mask, and counters are owned by
`KFParticleGpuDeviceStorage`, uploaded only for a new decay-plan revision, and
published as a non-owning kernel-state view. The lifecycle device probe checks
that the same table is visible through XPU constant memory.

Production consumes those tables through block-scan compaction and constructs
one unordered event-level cascade generation in the KFParticle-owned queue,
without a host synchronization between routing and construction. Stable
channel IDs and canonical lineage remain the comparison keys. The atomic
route and explicit scalar route are available only to qualification tests;
neither queue ownership nor CBMRoot's XPU initialization contract changes.

`LastV0TrackRoutingMonitorData()` exposes group launches, descriptor count,
visited pairs, active bits, accepted/stored tasks, global block reservations,
candidate/daughter counts, and overflow. `LastV0TrackCascadeResults()` exposes
per-channel counters plus the shared generation range. Consumers must not
infer channel membership from contiguous atomic output order.

The following remain CPU-only or incomplete at this boundary: full
`SetProductionVertex` compatibility, the remaining neutral-daughter and
composite-composite channel families, complete final selection, and production
CPU finder replacement.

Step 15 status: complete (100%). Standalone CPU/HIP lifecycle validation, the
controlled HIP routing benchmark, and the external CBMRoot cascade smoke gate
all pass. This qualifies the mask-driven Xi/Omega generation without enabling
production replacement of the complete CPU finder.

Step 16 applies the same boundary to K0S, Lambda, and anti-Lambda in three
checkpoints: a revision-owned two-daughter routing/selection ABI, an isolated
fused raw-generation path, and a generation-wide selection continuation plus
production steering switch. CBMRoot continues to provide only packed tracks,
vertices, events, and field data. KFParticle compiles all masks and
descriptors, owns the persistent task/candidate/selection buffers, and
executes the ordered queue pipeline.

The raw candidate retains its stable channel ID and uses only a transient
device descriptor-index sidecar for O(1) selection-config lookup. This sidecar
is not part of the public physics identity or CBMRoot comparison key. The
explicit V0 path remains a qualification oracle. No CBMRoot source outside the
external diagnostic/build integration was required for Step 16.

Stage 16.1 implements that ABI entirely inside external KFParticle. The
revision compiler validates and builds three default-V0 descriptors, role
compatibility masks, and two charge/source execution groups. KFParticle owns
the persistent XPU tables, routed-task/status storage, per-channel counters,
and candidate descriptor-index sidecar; CBMRoot neither allocates nor uploads
them. The lifecycle kernel-state probe confirms device visibility.

Stage 16.2 added an external-KFParticle-only qualification transaction:
`RunTwoDaughterFusedStage()` launches descriptor-mask routing followed by raw
candidate construction on KFParticle's process-lifetime queue. Its routed
pool, counters, candidate SoA, daughter lineage, and descriptor sidecar remain
KFParticle-owned. The transaction has no route-to-construction host
synchronization and compares its unordered default-V0 output against the
explicit path.

Stage 16.3 switches KFParticle production steering to the same generation-wide
route. Selection resolves its immutable configuration through the
KFParticle-owned descriptor sidecar and writes diagnostic records plus compact
selected indices before the existing cascade continuation. CBMRoot still
provides only packed tracks, vertices, events, and field coefficients; it does
not own routing tables, task pools, selection workspace, or queue control.
The public comparison key remains stable channel/event/source lineage, not raw
candidate position.

Step 16 status: complete (100%). Standalone CPU/HIP lifecycle validation, the
controlled HIP batch benchmark matrix, CBMRoot XPU and diagnostic smoke tests,
the hermetic CPU/GPU V0 equivalence gate, and cascade continuation tests all
pass with the generation-wide production route.

## Step 17 Integration Boundary

Step 17 completes the supported decay graph inside external KFParticle in four
checkpoints:

1. compile a complete CPU-channel manifest into a flat, revision-owned graph
   contract without changing active execution;
2. add generic charged track-track and track-composite generations;
3. add composite-composite, neutral or missing-mass, projection, matching, and
   final-selection operations;
4. execute and audit the complete supported graph on the persistent
   KFParticle queue.

CBMRoot remains an input and diagnostic adapter throughout this step. It
provides packed tracks, vertices, events, field coefficients, and optional
detector inputs already present in its reconstruction boundary. It does not
allocate graph descriptors, routing masks, task pools, candidate pools,
selection workspaces, or XPU queues. Those objects remain visible in
KFParticle's device-storage owner and are uploaded only when the graph revision
or storage capacity changes.

The scheduler may order dependent generations on the queue, but it must not
download counters or wait on the host between graph nodes. Missing optional
inputs and unimplemented operations produce an explicit unsupported-channel
status before execution. They must not trigger an implicit CPU call from
inside the GPU graph.

Each checkpoint keeps the current standalone CPU/HIP lifecycle gates. The
fourth checkpoint also runs the batch benchmark and the external CBMRoot XPU
and diagnostic smoke gates. Step 17 does not switch normal CBMRoot
reconstruction to GPU output; materialization, exhaustive physics parity, and
controlled fallback belong to Step 18.

Stage 17.1 is implemented entirely inside external KFParticle. The flat graph
contract inventories nine CPU finder families and maps the currently
validated default-V0 and Xi/Omega channels into seven generation-ordered
nodes. Three persistent XPU buffers store graph nodes, execution groups, and
family coverage. CBMRoot neither allocates nor uploads them.

The graph compiler rejects omitted families, duplicate channel or family IDs,
invalid source topology, forward candidate dependencies, unsupported
operation contracts, invalid support reasons, and insufficient configured
capacity before any queue operation. The content-derived revision makes an
unchanged upload a no-op. The graph view is visible through the KFParticle
constant-memory state.

Stage 17.2 keeps the CBMRoot boundary unchanged while activating the charged
part of this contract inside `RunDecayPlan()`. KFParticle compiles and uploads
the graph revision, resolves graph payload kinds to resident two-track and
composite-track descriptor tables, and runs both generations on its own
queue. CBMRoot still supplies only packed tracks, vertices, event ranges, and
field coefficients.

Composite inputs are qualified by stable parent channel ID as well as parent
PDG. A CBMRoot event may enable several equal-PDG hypotheses, but the next GPU
generation consumes only the output declared by its graph edge. Candidate
indices and counters do not return through CBMRoot between charged
generations.

Stage 17.3 added the flat execution ABI and device action for
composite-composite, legacy missing-mass/kaon matching, and unary
finalization operations. These operations consume resident candidate indices,
merge bounded physical lineage, and append results to the resident candidate
pool. The filtered `ReconstructMissingMass()` mode and detector-neutral inputs
remain explicitly unsupported.

Stage 17.4 transfers descriptor, bounded task/result, and monitoring-counter
ownership to persistent KFParticle storage. `RunDecayPlan()` queues routing
and execution for each supported graph generation without a host counter read
or an intermediate wait. CBMRoot still owns only its normal XPU
initialization boundary; KFParticle retains its own process-persistent queue
and every graph allocation.

Step 17 implementation status: complete (100%). The 86-check standalone
lifecycle covers deterministic repeated execution and graph task overflow.
Target HIP lifecycle, batch benchmark, and external CBMRoot runtime/diagnostic
smokes remain the qualification gates; they require no CBMRoot ownership or
API change.

## Step 18 Integration Boundary

Step 18 is the first step allowed to prepare an alternative public particle
result, but it does not make GPU reconstruction the default. Its three
checkpoints preserve the ownership boundary established above:

1. external KFParticle and the existing CBMRoot diagnostic adapter generalize
   order-independent CPU/GPU comparison to every implemented topology and
   produce an explicit promotion verdict;
2. external KFParticle owns bounded conversion from resident candidate pools
   to complete `KFParticle` objects, including direct composite ancestry;
   CBMRoot adds only an isolated destination adapter;
3. CBMRoot adds explicit CPU-only, diagnostic, and validated-GPU modes plus
   event-atomic fallback and monitoring.

Physical source lineage and direct candidate ancestry are separate contracts.
Canonical source IDs remain the comparison key. Materialization additionally
needs immediate parent/daughter references so a cascade or
composite-composite candidate can be inserted into the public particle vector
with valid IDs. If current metadata cannot express both, KFParticle extends
its device sidecar rather than asking CBMRoot to infer ancestry from PDG or
output order.

The GPU eligibility decision is made before public output mutation. A complete
event falls back to the existing CPU finder when the requested graph exceeds
the declared capability manifest, XPU is unavailable, input or field data is
invalid, any bounded pool overflows, parity validation fails, or
materialization cannot resolve all references. Partial CPU/GPU result merging
is outside Step 18.

CPU-only remains the configuration default. Diagnostic dual execution remains
available after validated GPU routing is added. Synthetic CBMRoot fixtures are
the required correctness gate for Step 18; representative detector-data
throughput and promotion to a production default belong to Step 20.

Stage 18.1 keeps this boundary diagnostic-only. KFParticle owns the flat
parity snapshot and persistent topology/output/operation metadata. The
CBMRoot adapter downloads event ranges once, builds order-independent
variable-lineage snapshots, applies channel/output-specific tolerances, and
records a promotion verdict. It does not materialize or publish GPU particles.
The runner includes first-generation, cascade, and later graph ranges without
asking CBMRoot to infer graph identity from PDG or output order.

Stage 18.2 adds `KFParticleGpuMaterializer` on the external side and
`GpuMaterializationAdapter` as a thin CBMRoot wrapper. The external component
owns dependency validation, leaf de-duplication, full fit/covariance transfer,
direct daughter IDs, metadata, and atomic commit. The adapter has no XPU
ownership and does not alter `GpuDiagnosticRunner` or the normal CPU finder.
It exists only so isolated CBMRoot tests can exercise the real scalar
`KFParticle` destination before routing is introduced in Stage 18.3.

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

## Step 21 Complete Channel Boundary

The complete-input diagnostic route now uses the same dependency-closed plan
as the Step 21 manifest gate. It contains 252 active channels: 50 two-track,
124 composite-track, and 78 graph-operation channels. Eight same-sign CPU
contracts disabled in the source configuration remain explicit in coverage
but are not emitted as executable nodes.

`AddCompleteCpuFinderChannels()` is the only builder used to claim complete
CPU-finder coverage. Exhaustive standalone and complete-manifest tests use
that plan. An ordinary CBMRoot diagnostic batch records the mother PDGs
requested by its CPU `KfpSelector` and uses
`AddRequestedCpuFinderChannels()` to build the dependency-closed subset of the
same catalogue. Capacity accounting and GPU execution therefore cover exactly
the same physics request; an explicitly supplied narrow plan remains narrow
and must not be reported as complete. In the complete manifest every active
channel is uploaded to its routing or graph descriptor table, has a
corresponding per-channel monitoring slot, uses the common parity snapshot
identity, and can cross the bounded materialization boundary without
channel-specific CBMRoot code.

This closes physics scope only. CPU-only remains the default, diagnostic mode
remains the validated integration route, and the qualification lock from Step
20 is unchanged. Production promotion requires a later performance and
real-data requalification of this enlarged complete plan.

Target standalone HIP, serial/batch, CBMRoot unit and smoke, CPU/GPU V0, and
bounded online/offline diagnostic gates pass for the completed contract.
Step 21 is accepted at 100%.

## Step 22 Performance Report

Performance monitoring remains opt-in through
`KFPARTICLE_GPU_PERFORMANCE_MONITORING=1`. Online and offline adapters use the
same formatter and print the transaction in execution order. Every row names
its domain (`CPU`, `COPY H2D`, `GPU/QUEUE`, or `COPY D2H`) and wall time. The
report separates CPU reference, input preparation, capacity planning, queue
and buffer management, host packing, upload, first-generation construction,
dependent graph generations, selection, download, extraction, materialization,
and comparison.

The report also records events, batches, tracks, primary vertices, raw-task
capacity, monitored-channel work, raw/selected candidates, comparison result,
kernel launches, queue waits, capacity growth, transfer bytes, memory
high-water mark, work density, and overflow. `GPU/QUEUE` rows are host wall
intervals around queue-ordered work; they are not device-event timings for one
isolated kernel. Existing compact `KFParticle GPU qualification metrics:` and
`KFParticle GPU performance:` records remain the machine-readable campaign
contract.
