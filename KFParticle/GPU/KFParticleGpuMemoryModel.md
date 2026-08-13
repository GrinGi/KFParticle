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
            plan: two-daughter and V0-track routing descriptors, compatibility
                  entries, execution groups, and enabled-channel masks;
                  decay-graph nodes, graph execution groups, and CPU-family
                  coverage entries; later-generation operation descriptors
            work: explicit and routed two-daughter tasks, routed V0-track
                  tasks, graph-operation tasks/results, aggregate status, and
                  per-channel counters
            raw output: candidates, daughters, descriptor-index sidecar,
                        raw counters, selection records
            selected output: candidate indices, channel IDs, selected counters
```

Every `xpu::buffer` grows monotonically through the capacity policy and is
released only when the runtime is finalized. `KFParticleGpuBufferManager` is a
temporary API adapter around this owner; `Storage()` exposes the allocation
layout for inspection.

`KFParticleGpuKernelState` owns no memory. It is a trivially-copyable set of
device pointers, SoA views, and capacities. It may be published to XPU
constant memory with `xpu::set<TheKFParticleFinder>`; the state-probe kernel
validates that ABI separately from reconstruction mathematics. Its
`KFParticleGpuV0TrackRoutingView` and
`KFParticleGpuTwoDaughterRoutingView` reference persistent plan buffers and
contain no `xpu::buffer` ownership. `KFParticleGpuDecayGraphView` similarly
references revision-owned graph nodes, execution groups, and family coverage.
`KFParticleGpuGraphOperationStorageView` publishes immutable descriptors,
reusable tasks/results, physical capacities, revision identity, and aggregate
and per-channel counters. It does not own any of those allocations.
`KFParticleGpuTwoDaughterGenerationStorageView` publishes the routed
two-track task pool, aggregate routing counters, and descriptor-sized
selection workspace. `KFParticleGpuV0TrackGenerationStorageView` publishes
the corresponding composite-track task pool and aggregate counters. These
generation views are also non-owning and are rebuilt after any capacity
growth.
The candidate descriptor-index view is also non-owning and capacity-matched
to the raw candidate pool.

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
| Compiled routing plans -> device | `UploadTwoDaughterRoutingPlan()` / `UploadV0TrackRoutingPlan()` H2D | only after decay-plan revision |
| Compiled decay graph -> device | `UploadDecayGraphPlan()` H2D | only after graph revision |
| Coherent kernel state -> constant memory | `PublishKernelState()` / `xpu::set<TheKFParticleFinder>` | once after transaction capacities and revision tables are stable |
| Diagnostic raw candidates -> device | `UploadCandidates()` H2D | diagnostics only |
| Counter reset -> device | small H2D copies | before a stage/generation |
| Explicit task generation -> construction | queue ordering, no D2H | qualification APIs only |
| Fused two-daughter route -> construction -> selection | queue ordering over routed tasks, raw candidates, descriptor sidecar, and selection workspace, no D2H | once per production event generation |
| Fused cascade routing -> construction | queue ordering over one routed pool, no D2H | once per event generation |
| Graph-operation route -> construction | queue ordering over persistent descriptors, bounded routed tasks, and resident candidate indices | once per supported graph execution group; no host-consumed counter read |
| Production generation bookkeeping | queued aggregate/per-channel/selection D2H snapshots, resolved after the transaction wait | once per event generation |
| Fused cascade bookkeeping | one queued aggregate/per-channel D2H snapshot | once per event generation |
| Final raw diagnostics | `DownloadCandidates()` D2H | once after `RunDecayPlan()` |
| Isolated selection diagnostics | `DownloadV0SelectionResults()` D2H | after `RunV0Selection()` |
| Final selected result | `DownloadSelectedCandidates()` D2H | after decay-plan selection |

The compact task path launches construction over task capacity and guards with
the device-side accepted-task counter. It therefore performs no host readback
between task generation and construction. Queue order provides the dependency.

Each routing-plan upload copies four immutable persistent allocations:
descriptors, compatibility entries, execution groups, and the enabled-channel
mask. It also resizes four channel-counter arrays when needed. The two-daughter
and V0-track families own separate allocations and revision records. Visited,
accepted, stored, and constructed counters are reset per generation rather
than re-uploaded with unchanged descriptors. Capacities grow monotonically. A
compiled plan retains the source decay-plan revision; uploading the same
revision is a no-op, while uploading an uncompiled plan is rejected. These
table transfers are configuration work and are not part of the event loop.

Stage 17.1 adds three immutable persistent allocations:
`fDecayGraphNodes`, `fDecayGraphGroups`, and
`fDecayGraphFamilyCoverage`. The host compiler validates the complete
CPU-family inventory, stable channel IDs, source topology, generation
dependencies, operation masks, support reasons, and configured capacities
before upload. It sorts supported nodes into generation order and builds
contiguous launch-compatible groups. A field-by-field content hash is the
graph revision, so rebuilding identical content is an upload no-op while any
descriptor or coverage change produces a new revision.

The graph view is published through `KFParticleGpuKernelState`. Stage 17.2
adds `payloadKind` and `payloadIndex` to each node and makes
`RunDecayPlan()` validate its active two-track and composite-track tables
against those resident graph payloads before queueing event work.

`KFParticleGpuGraphOperationDescriptor` is immutable revision data;
`KFParticleGpuGraphOperationTask` stores only descriptor and
earlier-generation candidate indices; `KFParticleGpuGraphOperationResult`
stores bounded status and output identity. The execution action reads source
fit, metadata, and lineage directly from the resident raw candidate SoA and
appends to that same bounded pool. Composite lineage is merged in registers
and is limited to 16 sorted source IDs.

Stage 18.2 keeps immediate ancestry in five additional unsigned components of
the existing candidate metadata SoA: direct daughter count and two bounded
`kind/index` references. `kind` distinguishes a packed input-track index from
a candidate-pool index. This is not another allocation: the owning unsigned
metadata buffer grows with `NumberOfUnsignedComponents` and follows the same
candidate H2D/D2H operations as channel, topology, and operation status.
Flattened physical source IDs remain in the CSR daughter pool for matching;
the direct references are used only to rebuild the ordinary particle graph.

The host materializer consumes downloaded views after the transaction wait.
It recursively validates candidate dependencies, emits each referenced input
track once, then emits composite candidates in topological order. All output
is first built in a temporary vector. The public destination is swapped only
after ranges, overflow state, event identity, numerical state, direct
references, output class, and canonical leaf lineage have all passed.
Materialization therefore introduces no event-loop upload and cannot expose a
partial GPU event.

Stage 17.4 places descriptor, task, and result buffers in
`KFParticleGpuDeviceStorage` together with reusable aggregate and per-channel
visited, accepted, stored, constructed, rejected, and overflow counters.
Descriptors are uploaded only when the decay-plan revision changes. For each
event and execution group the queue resets counters, routes candidate
combinations, executes the bounded task pool, and then proceeds directly to
the dependent generation. `storedTasks` is consumed on device; host snapshots
are queued only for final transaction monitoring and resolved by the existing
single wait.

The compact graph scheduler ABI reads those persistent views from
`TheKFParticleFinder`. Reset has no ordinary arguments; routing retains only
`eventIndex`, `groupIndex`, and the effective task limit; execution retains
only that limit. The limit is distinct from the monotonically grown physical
task-buffer capacity and preserves intentional truncation tests. Steering
prepares all two-daughter, cascade, graph, and operation capacities before one
state publication, so no graph group performs its own `xpu::set`.
`KFParticleGpuExecuteGraphOperationsExplicit` remains a test-only explicit-view
oracle and is not a production fallback.

The same compact ABI now covers the complete production transaction:

| Production action | Ordinary launch controls |
|---|---|
| round-trip smoke | mass, event index |
| reset two-daughter generation | none |
| route two-daughter group | event index, group index, effective task limit |
| construct routed two-daughter tasks | effective task limit |
| evaluate/scatter first-generation selection | event index |
| prepare selected segments | none |
| reset V0-track generation | none |
| route V0-track group | event index, group index, selected range, effective task limit |
| construct routed V0-track tasks | effective task limit |
| reset graph generation | none |
| route graph group | event index, group index, effective task limit |
| execute routed graph tasks | effective task limit |

No action in this table receives a persistent input, descriptor, counter,
task, candidate, or selection view as an ordinary kernel argument. The
selected range is launch-local work selection rather than storage ownership.
Explicit generation, atomic routing, isolated selection, numerical probes,
and `KFParticleGpuExecuteGraphOperationsExplicit` retain explicit views only
as qualification APIs or independent host/device oracles.

Steering grows every transaction-owned pool and uploads changed plan tables
before one `PublishKernelState()` call. The published state then remains
unchanged until all reconstruction actions and their queued monitoring
snapshots finish. Diagnostic fused entry points publish once after their own
capacity preparation. Any later allocation growth requires a new publication
before another state-backed launch.

Aggregate and per-channel monitoring snapshots are queued D2H after each group,
but are not consumed until the single transaction wait. They therefore do not
control later launches or create a host synchronization dependency. Moving
these optional diagnostics into a final bulk snapshot remains a performance
refinement, not a correctness dependency.

Task-pool truncation uses the graph-specific
`KFGpuGraphTaskCapacityExceeded` flag. Candidate and daughter pool exhaustion
remain independent candidate-pool flags. Capacities grow monotonically, so
normal event processing does not allocate or upload graph configuration.

Compatibility tables now use exact charged-PDG keys. Composite-track entries
also contain `parentChannelId`; this metadata is immutable configuration and
is uploaded only with a new plan revision. Event-local selected indices,
candidate fits, daughter source IDs, and routing counters remain in their
existing device pools. The first generation scatters selected candidate
indices in device memory, and the second generation reads those indices plus
raw candidate metadata directly. There is no new per-event graph upload and
no intermediate device-to-host candidate transfer.

## Present Boundaries

`RunDecayPlan()` uploads input once and queues one fused two-daughter
generation per event. It enumerates each execution-group pair once, constructs
an unordered raw generation, evaluates selection through the descriptor
sidecar, and scatters selected indices into descriptor-contiguous segments.
The optional cascade generation consumes that selected pool on the same queue.
Supported composite-composite, legacy missing-mass, and unary graph groups
then consume resident earlier-generation candidates in graph order on that
same queue. One transaction wait resolves all queued snapshots; only then are
host result records assembled and final selected/raw outputs downloaded.

`RunV0Selection()` remains an explicit diagnostic API. It consumes the same
resident raw pool and packed primary vertices, but resets and downloads the
selected-index pool itself for isolated testing.

Stage 15 adds `fV0TrackRoutedTasks` plus visited-pair, active-bit,
accepted-task, stored-task, block-reservation, overflow, and four per-channel
counters to the visible storage owner. The retained
`KFParticleGpuRouteV0TrackTasksAtomic` is a device reference that performs one
global reservation per emitted task. Production uses
`KFParticleGpuRouteV0TrackTasksBlockScan`: each thread contributes its number
of accepted descriptor bits, one thread reserves the block range, and all
threads write their exclusive-scan slots. The following
`KFParticleGpuV0TrackRoutedCandidatePoolKernel` reads both the pool and its
stored counter directly on the same queue. Scalar status, per-channel counters,
routed-task diagnostics, and candidate SoA are downloaded only after both
kernels have completed.

`RunDecayPlan()` now compiles/uploads the routing plan on revision, launches
the two charge/source execution groups per event, constructs one unordered
cascade generation, queues its aggregate and per-channel snapshots, and waits
once after all events. It never downloads the routed-task count between route
and construction. Per-channel result records carry visited, accepted, stored,
and constructed counts; `generationCandidates` identifies the shared event
range, while stable candidate `channelId` metadata provides membership.

`RunV0TrackFusedStage()` remains an isolated qualification API. It can select
the atomic or block-scan route at call granularity, enabling device-to-device
correctness and performance comparison without changing production control
flow. Its explicit host oracle is test-only.

Stage 16.1 adds `fTwoDaughterRoutingDescriptors`,
`fTwoDaughterRoutingCompatibility`, `fTwoDaughterRoutingGroups`, the enabled
mask, four per-channel counters, `fTwoDaughterRoutedTasks`, and its aggregate
status buffers. `fCandidateRoutingDescriptorIndices` is a transient
candidate-indexed sidecar used for O(1) selection-config lookup; stable
channel ID remains in candidate metadata and remains the public identity.
The selection continuation additionally owns four descriptor-sized arrays:
accepted counts, bounded stored counts, output offsets, and scatter cursors.
They form `KFParticleGpuTwoDaughterSelectionWorkspaceView`; kernels receive
only that non-owning flat view.

The constructed raw candidate retains stable `channelId` metadata while
`fCandidateRoutingDescriptorIndices[candidateIndex]` records the transient
descriptor bit. The sidecar is downloaded only by explicit diagnostics;
normal `DownloadCandidates()` does not add that transfer. Atomic and 64-thread
block-scan routing share one pair resolver. Production
`RunDecayPlanBatch()` uses the block-scan route and descriptor-driven
construction, then evaluates, prefixes, and scatters selection without a host
counter read. Raw channel membership is recovered by stable metadata plus
per-descriptor counters; only the selected pool is physically segmented by
event and descriptor. The explicit per-channel APIs remain test oracles.
