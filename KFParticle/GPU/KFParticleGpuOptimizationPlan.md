# KFParticle GPU Optimization Plan

## Purpose And Authority

This document is the detailed optimization plan for the complete KFParticle
GPU event-batch route. `KFParticleGpuPortingPlan.md` remains the global project
plan and defines the physics, integration, qualification, and completion
boundaries. During Step 22, this document is authoritative for performance
work: optimization hypotheses, implementation order, measurements, rejected
experiments, accepted changes, and remaining risks must be recorded here.

Optimization must not change the Step 21 physics contract. The complete
252-channel manifest, requested-channel dependency closure, CPU/GPU numerical
tolerances, lineage, bounded-output behavior, and event-atomic fallback remain
fixed. A lower launch count is not a success if it changes results, hides
overflow, increases memory without a bound, or makes failure recovery partial.

The active implementation scope is deliberately narrower than the final
execution model. Step 22 implements optimized batches of independent events
and prepares a common transaction/domain ABI that will also be usable by a
future whole-time-slice adapter. It does not implement time-window finding,
whole-time-slice KFParticle reconstruction, CA-to-KFParticle device hand-off,
window overlap ownership, or time-slice duplicate suppression. Those features
remain blocked until the CA GPU architecture can publish the required data and
lifetime contract described below.

## Baseline Observation

The first Step 22.1 real-data report exposed excessive launch granularity.
The requested-channel online diagnostic run processed 31 events in four
batches, with a largest batch of eight, but still submitted 449 GPU kernels.
The event-based offline diagnostic run processed 32 events as 32 batches of
one and submitted 315 kernels. Both modes produced exact CPU/GPU membership
agreement for all compared candidates and no overflow.

Representative measurements on HIP `hip1` were:

| Mode | Events | Batches | Largest batch | Launches | Queue waits | Stored tasks | GPU transaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online, first/cold repetition | 31 | 4 | 8 | 449 | 32 | 15544 | 269.274 ms |
| Online, second/warm repetition | 31 | 4 | 8 | 449 | 32 | 15544 | 155.946 ms |
| Offline, repetition 1 | 32 | 32 | 1 | 315 | 151 | 5849 | 185.732 ms |
| Offline, repetition 2 | 32 | 32 | 1 | 315 | 151 | 5849 | 184.303 ms |

The online cold repetition is not a steady-state performance baseline. Its
91.542 ms H2D interval versus 0.649 ms in the second repetition demonstrates
runtime, allocation, or cache warm-up. The offline repetitions are stable to
approximately one percent. Final baseline evidence still requires explicit
warm-up, more repetitions, medians or robust percentiles, no-reference GPU
measurements, and the complete synthetic manifest in addition to the real
requested-channel plan.

The central architectural finding is already valid: `RunDecayPlanBatch()`
owns a batch, but its production submission remains nested by event and then
by execution group. Batching therefore amortizes some transfer and queue
boundaries without making most kernel launches batch-wide.

## Current Launch Decomposition

The performance snapshot currently estimates production launches as:

```text
two-daughter route-group launches
+ V0-track route-group launches
+ 3 * later graph-operation groups
+ 5 * events with two-daughter descriptors
+ 2 * events with V0-track descriptors
```

The two-daughter event path queues:

```text
reset generation state
route each compatible track-pair execution group
construct routed candidates
evaluate selection
prepare selection segments
scatter selected candidates
```

Later graph operations queue, for every event and execution group:

```text
reset generation state
route operation tasks
execute routed operations
```

Composite-plus-track generations similarly route each group and launch a
candidate constructor at generation boundaries. Host-side asynchronous
counter snapshots add many small D2H operations even though the final wait is
already shared.

The CPU finder supplies the semantic reference, not the GPU execution shape.
Its main order is `Find2DaughterDecay`, PV extrapolation,
`NeutralDaughterDecay`, `FindTrackV0Decay`, and subsequent dependent channels.
Within one CPU pass, pair checks, candidate construction, and acceptance are
ordinary function calls and loops. The GPU must preserve this dependency
order between generations, but independent events and independent channels in
one generation are legitimate parallel work.

## Unified Transaction And Domain Model

The optimization must not make an event batch the permanent physics ABI. The
common execution object is a transaction containing one or more independent
or logically partitioned domains plus flat work ranges:

```text
KFParticle GPU transaction
  immutable input and decay-plan views
  one or more domain descriptors
  flattened work descriptors grouped by graph generation/operation family
  bounded per-domain output and status
  transaction-level queue and memory lifetime
```

A domain identifies the ownership and legal source ranges of one unit of
physics work. It is an event in the active implementation and will be a time
window or another explicitly owned time partition in the future. Kernels must
consume a `domainIndex` or a flat work descriptor, not infer isolation from a
host launch or assume that every domain is semantically an event.

The same execution core must represent:

```text
single event       transaction with one event domain
event batch        transaction with several mutually isolated event domains
whole time slice   transaction with several overlapping time-window domains
```

A whole time slice is therefore one possible transaction container, but it is
not semantically one event and is not merely an event batch. Event domains
forbid all cross-domain combinations. Future time-window domains may inspect
explicit halo tracks from neighbours, then apply an ownership rule so each
physical candidate is published once.

The first implementation must prove that an event batch of size one and a
batch of many events use the same kernels, numerical helpers, output contract,
and fallback path. No single-event-only kernel fork is allowed.

### Input Required For Event Domains Now

The current `KFParticleGpuEventDesc` and packed SoA provide the starting event
contract. Before an event enters a GPU transaction, the CBMRoot adapter must
provide:

- a stable event identity unique within the transaction;
- fitted track parameters and complete covariance in the common component-
  major SoA, containing physical tracks only and no CPU SIMD padding;
- the packed ten-coefficient magnetic-field region for every track when the
  configured transport requires it;
- `chiToPrimaryVertex`, charge, PDG/PID hypothesis, associated primary-vertex
  index, pixel-hit count, and a stable input `SourceId` for every track;
- eight absolute Finder-compatible track-set ranges: secondary/primary,
  positive/negative, and first/last-hit representations;
- absolute species subranges for every populated track set, or an explicit
  generic-track range where the channel contract allows it;
- matching first/last representations with the same `SourceId`, so lineage
  and duplicate-track rejection remain independent of packed array position;
- a bounded absolute range of primary vertices with their fit parameters,
  covariance, chi2, NDF, and contributor information;
- the event's secondary-track chi-to-PV threshold and every other
  configuration value not already frozen in the shared decay plan;
- validated aggregate task, candidate, daughter, selected-output, and
  monitoring capacities, with no cross-event write outside the domain's
  declared ownership;
- one immutable decay/routing-plan revision shared by every event in the GPU
  transaction.

Event input must preserve strict isolation. A work item belonging to one event
may only read that event's track/PV ranges and candidates. Overflow, malformed
input, unsupported scope, execution failure, or materialization failure is
recorded per event, and fallback replaces that event's complete GPU result
with its complete CPU result. Partial CPU/GPU particle-list merging remains
forbidden.

### Input Required From CA For Future Time-Slice Domains

The future CA-to-KFParticle contract must provide enough information for
KFParticle to form legal time-window work without reconstructing CA ownership
on the host. At minimum it must publish:

- device-resident fitted track parameters and complete covariance in a
  documented layout, units, alignment, and version that can be viewed or
  transformed device-to-device into the KFParticle SoA;
- stable time-slice-global track identifiers, retained across first/last
  states and every overlapping window, for lineage and duplicate suppression;
- track time and its uncertainty, charge, hit/PID information, and the
  first/last fitted states required by the Finder's track-source semantics;
- the magnetic-field representation needed by KFParticle transport for each
  track or a documented device-resident field service with equivalent
  numerical meaning;
- explicit window descriptors: window identity, central/core time interval,
  readable halo/overlap interval, track ranges, and neighbouring-window or
  boundary information;
- one deterministic ownership rule for a candidate visible in overlapping
  windows, based on a declared quantity such as fitted decay time, production
  time, or canonical source-track ownership;
- primary vertices or time-compatible vertex hypotheses, including their
  covariance, time/range association, and enough information to classify
  primary/secondary tracks and compute chi-to-PV consistently;
- stable source ranges or classification metadata from which the eight
  Finder-compatible track sets and species ranges can be viewed or generated
  without a host round trip;
- a transaction memory bound and, for very large slices, legal chunk
  boundaries that retain halo data and ownership semantics across chunks;
- device allocation ownership, pointer lifetime, producing XPU queue/device,
  and an explicit dependency primitive so KFParticle starts only after CA has
  finished writing the input and CA does not reuse it too early;
- status for missing modules, empty windows, invalid tracks/vertices, and
  partially unavailable detector/PID information, with deterministic
  KFParticle fallback policy.

KFParticle must not guess window overlap, synthesize global track identities
from local array positions, or infer input readiness from an implicit host
synchronization. If CA cannot provide the required ownership, time, identity,
field, PV, or lifetime data, whole-time-slice KFParticle execution remains
disabled.

### Future Time-Slice Output Rules

The future output must retain time-slice-global leaf `SourceId` lineage,
originating domain/window identity, event/PV association when available, and a
canonical candidate key formed from channel identity plus canonical source
lineage and any required PV identity. A candidate may read halo tracks but is
published only by its owning core domain. A final compact duplicate pass is
allowed where ownership cannot be decided during construction.

The initial safe fallback unit for a future time-slice transaction is the
whole submitted transaction or explicitly self-contained chunk. Per-window
fallback is deferred until overlap ownership and cross-window duplicate
suppression are proven; otherwise replacing one window can silently lose or
duplicate boundary candidates.

### Active Preparation Versus Deferred Implementation

Implemented now:

- a domain-neutral transaction/work vocabulary and flat descriptors whose
  indexing is not hard-coded to one host launch per event;
- event domains backed by the current packed event input;
- one-event transactions and bounded multi-event batches through the same
  execution path;
- per-event source ranges, outputs, status, overflow, monitoring, and atomic
  CPU fallback;
- graph-generation boundaries and operation-family metadata suitable for
  future time-window domains;
- tests proving that no candidate crosses event-domain boundaries.

Prepared but not executed now:

- domain kind and optional time/ownership extension points that do not impose
  time-slice overhead on the event path;
- a documented CA device input/lifetime ABI and a future adapter boundary;
- stable global source-lineage and output ownership requirements;
- capacity/chunking interfaces that do not assume the transaction is a small
  list of events.

Explicitly deferred:

- construction of time-window descriptors from real CA output;
- whole-time-slice and overlapping-window kernel execution;
- time-window candidate ownership and duplicate suppression;
- device-to-device zero-copy CA hand-off;
- per-window fallback and final time-slice production qualification.

## Optimization Opportunities

### 1. Flatten Every Launch Across Transaction Domains

Replace per-event launches with flat work tables containing event identity,
execution-group identity, and pair/source ranges. One launch processes all
compatible event domains in the active transaction. Per-domain counters,
output ranges, overflow, and failure status remain separate. The descriptor
shape must also accept future time-window domains without changing numerical
kernels, but this stage instantiates event domains only.

Conditions:

- all events use the same immutable decay/routing plan revision;
- every work item carries or can derive its event index;
- candidate, daughter, task, selected-output, and monitoring reservations are
  bounded per event or can be mapped back to an event without ambiguity;
- one failing or overflowing event remains independently replaceable by its
  complete CPU result.

Expected effect: make launches scale primarily with execution groups and graph
generations instead of events. For the measured four online batches, the first
target is approximately 50-80 launches instead of 449, an 80-90% reduction.
Offline receives little benefit while `largest_batch=1`; its adapter must form
real multi-event transactions before launch amortization can occur.

### 2. Fuse Two-Daughter Routing And Candidate Construction

The routing kernel already resolves a pair and produces compatible and
accepted channel masks. A fused path can build the accepted channel
hypotheses directly instead of writing routed tasks and reading them in a
second kernel.

Conditions:

- capacity reservation remains bounded and reports per-event overflow;
- the accepted-channel loop does not produce unacceptable wavefront
  divergence for dense masks;
- full-field and simplified transport variants are grouped or specialized if
  mixing them causes excessive register pressure or divergence;
- the current routed-task path remains available as a correctness reference
  and fallback until qualification.

Expected effect: remove one global write/read round trip through the task pool
and one launch per route domain. The benefit should be strongest for small and
medium pair sets where launch overhead and task traffic dominate arithmetic.

### 3. Fuse Candidate Construction And Selection Where Legal

Evaluate cuts immediately after a candidate fit and reserve compact selected
output directly. The existing correctness contract treats compact selected
index order as unordered, so an atomic append is acceptable for candidate
membership.

Conditions:

- all selection observables are available from the just-built fit state or
  can be computed without reloading the full candidate covariance;
- channel-segment offsets are not required by a downstream consumer, or are
  produced by one small final compaction pass;
- diagnostic mode can still retain raw rejected candidates and rejection
  reasons when requested;
- the no-reference production path does not pay for diagnostic snapshots.

Expected effect: replace the current evaluate, single-thread segment prepare,
and scatter sequence with direct selection or, where stable channel grouping
is required, direct evaluation plus one batch-wide compaction kernel. This
removes up to three launches per event in the present implementation.

### 4. Execute Later Graph Work Once Per Generation

Do not launch `reset -> route -> execute` for every event/group. Flatten all
independent work in a graph generation across events and compatible groups.
Use one generation-level route/execute path, or a single fused kernel when its
source cardinality and operation family make that profitable.

Conditions:

- all inputs of generation `N` are complete before generation `N+1` begins;
- no group in the same generation consumes another group's newly produced
  candidate unless the graph compiler moves it to a later generation;
- operation families with materially different register or control-flow cost
  may use separate specialized kernels;
- no portable cross-block global synchronization is assumed inside a normal
  kernel.

Expected effect: make launch count depend on graph depth and a small number of
operation families, not on event count multiplied by group count. A single
kernel for the entire decay graph is not the target; generation boundaries are
the safe global synchronization points.

### 5. Batch Composite-Plus-Track Cascades By Generation

Flatten V0/particle-plus-track pair domains across all events and channels of
the same generation. Route and construct them in one generation-level launch
where transport and source forms are compatible.

Conditions:

- bachelor-track ranges and selected composite ranges are explicit per event;
- duplicate-lineage checks remain event-local;
- incompatible transport modes are split into a small number of specializations;
- a generation cannot observe partially produced candidates from itself.

Expected effect: replace per-event route groups and per-generation candidate
constructors with a small number of generation/family launches.

### 6. Remove Standalone Reset Kernels

Prefer generation/epoch tags, offsets initialized as part of the producer, or
one batch-wide reset over every active event. Do not launch a reset separately
for each event and group.

Conditions:

- stale counters cannot be mistaken for current transaction data;
- counter wraparound has an explicit policy if epoch tags are used;
- diagnostic counters preserve their defined aggregation semantics.

Expected effect: remove many very small, low-occupancy launches. This is a
lower-risk change after batch-local storage has been made explicit.

### 7. Consolidate Monitoring And Snapshot Readback

Write event/group monitoring into persistent device arrays and download one
compact report after the transaction. Deep traces and detailed rejected-state
snapshots remain conditional. Ordinary no-reference execution must not submit
diagnostic copies or waits.

Conditions:

- correctness tests still have an explicit detailed-download mode;
- production status retains enough information for overflow and event-atomic
  fallback decisions;
- performance counters distinguish requested physics work from diagnostic work.

Expected effect: fewer small D2H commands and synchronization dependencies,
especially in offline mode, which reported 151 queue waits for 32 single-event
batches.

### 8. Form Useful Event Batches And Route Small Work To CPU

The event-based offline adapter currently reports 32 batches of one. Add a
bounded collector using track/pair/task estimates and the measured crossover.
Very small, irregular, unsupported, or memory-risk events remain CPU work;
suitable events form deterministic GPU transactions.

Conditions:

- latency and framework ownership permit delayed submission;
- batches have explicit event ranges and bounded aggregate capacities;
- fallback replaces a complete event, never merges partial CPU and GPU lists;
- online and offline policies may choose different thresholds.

Expected effect: expose enough parallel work to benefit from all launch-fusion
changes. Without this, offline optimization is fundamentally capped.

## Deliberate Non-Goals And Unsafe Fusion

- Do not implement one monolithic kernel for the entire KFParticle graph.
  Generation dependencies require global ordering, and the resulting register
  pressure and branch divergence are likely to erase launch savings.
- Do not combine unrelated numerical operation families solely to reduce a
  counter. Specialization is preferred when transport, topology, or missing-
  mass mathematics differ materially.
- Do not mix full-field and simplified transport blindly in one wavefront.
- Do not remove bounded pools or overflow checks.
- Do not discard the current staged implementation until the fused path has
  complete CPU/HIP equivalence and fallback coverage.
- Do not fabricate a temporary host-built time-slice route merely to exercise
  the future descriptors. CA device input, ownership, time-window, and lifetime
  contracts must exist first.
- Device-to-device CA handoff, time-window execution, overlap duplicate
  suppression, and whole-time-slice qualification remain deferred final
  milestones, not shortcuts for Step 22 event-batch optimization.

## Ordered Implementation Plan

### O.1 Freeze A Kernel-Level Launch Map

Extend performance monitoring with launch counts and submitted work by kernel
family: reset, two-daughter route, two-daughter construct, selection evaluate,
selection prepare/scatter, cascade route/construct, graph route/execute, and
diagnostic probes/copies. Record batch size, active threads, useful tasks,
empty work, and approximate occupancy input for each family.

Run warm-up plus repeated requested-plan and complete-manifest baselines. This
is the measurement gate for all later work; estimates in this document must be
replaced by measured deltas as implementation proceeds.

### O.2 Establish The Common Transaction ABI And Optimize Event Batches

O.2 is the controlled transition from a host loop that submits work per event
to one domain-neutral GPU transaction that currently contains event domains.
It changes execution shape before O.3 changes producer/consumer kernel
boundaries. No real or synthetic whole-time-slice execution is required.

O.2 is divided into four substantial stages. Percentages below describe O.2
itself, not the global Step 22 percentage.

#### O.2A: Freeze The Transaction/Domain ABI Without Changing Results (0% -> 25%)

Define small, trivially-copyable, non-owning device contracts for:

- transaction identity, plan revision, domain range, and aggregate capacities;
- one domain's stable identity, active input descriptor, output/status slots,
  and legal source/PV ownership;
- flat work ranges identifying domain, graph generation, operation family,
  execution group, source offsets/counts, and bounded output partition;
- per-domain counters for tasks, candidates, daughters, selected output,
  overflow, unsupported input, execution failure, and fallback reason;
- optional extension views for future time/halo/ownership data, kept outside
  the hot event descriptor so event kernels do not load unused time-slice
  fields.

Keep `KFParticleGpuEventDesc` as the active CBMRoot input adapter. Map each
event to one domain and preserve all current track-set, species, PV, cut, and
`SourceId` semantics. Internally, new production work must carry a
`domainIndex`; compatibility APIs may continue exposing event IDs and event
results at the host boundary.

The buffer owner gains persistent domain, work-range, and per-domain-status
arrays with monotonic capacity reuse. Kernel state publishes non-owning views
only. Allocation, packing, and publication must remain outside numerical
helpers. No launch count or physics behavior is intentionally changed in this
stage.

Primary time-slice preparation in O.2A is structural only:

- domain and work descriptors do not encode `event` into their type names or
  require a host launch to define isolation;
- optional future time-domain metadata has a separate versioned view;
- source identity, input lifetime, output ownership, and chunk boundaries are
  explicit extension points;
- no time-window builder, overlap logic, duplicate suppression, CA pointer, or
  time-slice branch is implemented.

Acceptance:

- compile-time flat-ABI and ownership checks pass;
- one event routed through one domain reproduces the current host/device
  candidates, selected output, lineage, status, and overflow exactly;
- multi-event descriptors preserve distinct track/PV/output ranges;
- invalid/overlapping ranges and mismatched plan revisions are rejected before
  launch;
- event mode has no mandatory time-domain allocation or per-work-item load;
- existing CPU, HIP, CBMRoot smoke, and V0 equivalence gates remain unchanged.

#### O.2B: Flatten The Complete Two-Daughter Event-Batch Path (25% -> 50%)

Build one flat two-daughter work table for every sealed event batch. It covers
all non-empty event-domain/execution-group pair ranges and records enough
offset information for a kernel to map a global work index back to its domain
and local pair. Empty events produce status but no GPU work item.

Change submission from `for event -> launch group` to batch-wide stages:

```text
one batch/domain-state initialization
batch-wide two-daughter routing over all event/group work ranges
one routed-candidate construction stage over the batch task pool
batch-wide selection and per-domain compact output finalization
```

This stage must reuse the existing routing, fit, transport, topology, and
selection mathematics. Routing and construction remain separate kernels here;
their fusion belongs to O.3. Work scheduling may use a prefix table, tiled work
descriptors, or another measured mapping, but it must not scan every domain for
every pair or introduce a serial single-thread dispatcher.

Replace transaction-global task/output counters where necessary with
per-domain bounded ranges or deterministic batch reservations. One overflowing
domain must not consume another domain's reserved output or corrupt its status.
Candidate order may remain nondeterministic where the existing contract treats
membership as unordered, but event result ordering remains source-event order.

Acceptance:

- batch size one follows exactly the same submission path as batch size many;
- serial execution of `N` events and one `N`-event batch have identical
  event-local membership, lineage, selection classes, and overflow behavior;
- adversarial adjacent domains cannot read one another's tracks, PVs,
  candidates, daughter lineage, or selected indices;
- empty, no-PV, no-pair, invalid, and individually oversized events remain
  isolated;
- two-daughter launch count is independent of event count for a fixed set of
  active execution groups, apart from capacity chunking explicitly reported by
  monitoring;
- before/after reports record launch count, active threads, useful pairs/tasks,
  mapping overhead, memory high-water, and transaction time.

#### O.2C: Flatten Cascades And Every Later Graph Generation (50% -> 75%)

Extend the same transaction work model to composite-plus-track cascades and
all later graph-operation families. Compile host-side execution metadata into
generation-ordered work tables covering every event domain. Within one graph
generation, independent domains and compatible groups are submitted together;
the queue boundary between generations remains the global dependency barrier.

Target submission shape before kernel fusion is:

```text
for each graph generation:
  initialize active per-domain generation state
  route all compatible domains/groups of each operation family
  execute the resulting batch task ranges
```

Do not create one monolithic graph kernel. Operation families may retain
separate kernels where source layout, transport, missing-mass, topology, or
register cost differs. O.2C removes the event multiplier from launches; O.3
later decides which route/execute boundaries are profitable to fuse.

Monitoring snapshots move from one host object and many scalar copies per
event/group to persistent per-domain/per-generation device records with one
compact transaction readback. Deep trace may still request detailed records,
but normal monitoring must not restore the old per-event submission model.

Acceptance:

- every complete-manifest generation and all nine CPU finder families execute
  through the flat domain work model;
- generation dependencies, source ranges, canonical lineage, PV association,
  terminal selection, and missing-mass behavior match the staged reference;
- serial/batch equivalence includes long dependency chains and conjugate
  channels, not only default V0;
- overflow and failure remain domain-local at every generation;
- launches scale with graph generations and compatible operation families,
  not with events multiplied by graph groups;
- standalone CPU/HIP complete-plan lifecycle, batch, graph-operation, and
  bounded-memory tests pass before CBMRoot integration is changed.

#### O.2D: Integrate Real Event Batches And Freeze The Time-Slice Seam (75% -> 100%)

Connect the new transaction path to the existing CBMRoot diagnostic batch
builder and both official launch modes. Online and offline adapters prepare the
same event-domain transaction ABI. A batch of one remains legal; bounded
multi-event collection is enabled where framework ownership permits it.
Adaptive CPU/GPU crossover policy remains O.5, but O.2D must prove that real
multi-event transactions are possible and do not depend on synthetic input.

Preserve source-event order at materialization and reporting boundaries.
Fallback remains event-atomic: a failed event contributes only its complete
CPU result while successful neighbouring domains retain their complete GPU
results. Transaction failure before trustworthy per-domain status may fall
back the whole batch.

Freeze the future time-slice seam as documentation and ABI tests only:

- record the exact CA-provided track, covariance, time, identity, first/last
  state, field, PID, PV, window core/halo, ownership, chunk, memory-lifetime,
  device, queue, and dependency requirements from this document;
- verify that numerical kernels consume domain work and source ranges without
  requiring an event-only host launch;
- verify that optional time-domain views can be absent with zero event-path
  behavior change;
- do not construct time windows, run a fake time slice, copy CA data, or claim
  time-slice support.

Acceptance:

- real bounded online and event-based offline event-batch diagnostics pass on
  the target HIP device, including at least batch sizes 1, 2, 4, and 8 where
  the input supplies enough valid events;
- framework summaries, monitoring, evidence files, and materialized results
  retain event identity and deterministic source-event ordering;
- existing fallback, unavailable-device, invalid-input, overflow, and retained-
  evidence gates pass;
- requested-channel and complete-manifest routes both use the transaction ABI;
- the frozen CA/time-slice seam has no implementation dependency on current
  host packing and introduces no measured event-batch regression;
- the O.1 baseline is repeated and shows a reproducible launch and/or time
  improvement without increased unbounded memory or changed physics.

O.2 completion gate: the old per-event launch topology is absent from the
production event-batch path; one and many events share one domain-neutral
transaction executor; all current physics and event-atomic fallback contracts
pass; and future time-slice activation requires an adapter plus ownership
logic, not a redesign of numerical kernels, graph scheduling, or output
storage.

Status: O.2 planned, 0% implemented. O.2A is the next implementation stage
after the O.1 kernel-family baseline is frozen.

### O.3 Fuse The Proven Producer/Consumer Pairs

In this order:

1. fuse two-daughter routing with construction;
2. fuse construction with direct selection, retaining an optional final
   compaction pass;
3. batch or fuse composite-plus-track route/construct by generation;
4. reduce later graph `route + execute` to generation/family kernels where
   measurements justify it.

Each fusion is a separately switchable strategy with the staged route retained
until the fused path passes standalone CPU/HIP, complete-manifest, overflow,
and real CBMRoot equivalence checks.

### O.4 Remove Administrative Launches And Diagnostic Cost

Replace repeated resets with epochs or batch-wide initialization. Consolidate
monitoring arrays and D2H snapshots. Ensure no-reference execution excludes CPU
comparison, deep traces, rejected-candidate snapshots, and diagnostic copies.
Tighten capacities from measured per-event/batch bounds without weakening
overflow handling.

### O.5 Establish Adaptive Online And Offline Event-Batch Policies

Use measured work estimates and memory bounds to select CPU or GPU and choose a
bounded batch. Make offline event-based processing collect real batches.
Preserve deterministic event identity and event-atomic fallback. Record the
crossover separately for online and offline modes. This stage does not route
time slices; it only verifies that the transaction API and capacity policy do
not assume a permanent maximum number or size of event domains.

### O.6 Reprofile, Requalify, And Decide

Repeat low, typical, and high matrices after warm-up for requested and complete
plans. Publish median and spread for KFP transaction and end-to-end framework
time, launch reduction, transfer bytes, memory high-water, overflow margin,
fallback counts, and physics agreement. Step 22.3 makes the final per-mode and
per-workload qualification decision using a policy frozen before measurement.

## Expected Improvement Envelope

These are planning estimates, not acceptance claims:

- batch-wide launches alone: approximately 10-25% transaction improvement;
- batch-wide work plus removal of task-pool traffic and repeated selection
  passes: approximately 25-45% transaction improvement;
- online launch count: first target 50-80, later target approximately 15-35
  for the measured requested-plan transaction;
- offline launch count after real batching: first target 30-60;
- complete-manifest launch count should primarily follow graph generation
  depth and a small number of operation families, not 252 channels times the
  number of events.

Small workloads may still be faster on CPU after optimization because launch
and transfer costs never become zero. The routing policy must use the measured
crossover rather than forcing every event onto the GPU.

## Validation Matrix For Every Accepted Change

- standalone CPU lifecycle and operation fixtures;
- standalone HIP lifecycle, complete-manifest graph, and serial/batch equivalence;
- deterministic bounded overflow and undersized-pool cases;
- empty, invalid, unsupported, and unavailable-device fallback;
- CBMRoot unit, smoke, V0 equivalence, online, and offline diagnostic gates;
- no-reference GPU result/materialization checks;
- retained baseline evidence and machine-readable before/after reports;
- memory high-water and capacity-growth regression;
- numerical residual, lineage, candidate membership, and event identity parity.

## Current Status

Planning baseline created during Step 22.1. No fusion optimization described
here is yet accepted. The measured requested-plan runs establish correctness
and expose launch granularity, but the complete Step 22.1 baseline still needs
warm-up-controlled repeated evidence, no-reference timing, complete-manifest
coverage, and kernel-family launch accounting. The execution-model decision is
now frozen: current optimization implements event batches on a domain-neutral
transaction ABI and only prepares the documented extension boundary for future
time-slice reconstruction. Time-slice execution remains blocked on the CA
device-data, window-ownership, identity, PV/time, and lifetime contract.
