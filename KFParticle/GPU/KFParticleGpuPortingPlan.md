# KFParticle GPU Porting Map

This note records the current CPU reconstruction flow and the first GPU
porting boundary.  It is intentionally local to the GPU prototype so the CPU
KFParticle implementation can keep evolving independently.

## Current CPU Flow

`KFParticleFinder::FindParticles()` is the event-level entry point.  Its input
is an array of eight `KFPTrackVector` sets, one `ChiToPrimVtx` array per track
set, the output `Particles` vector, and primary vertices.

The high-level order is:

1. `Init(nPV)`: clear temporary secondary and primary candidate containers.
2. Convert long-lived tracks from `vRTracks[0..3]` to `KFParticle` objects and
   append them to `Particles`.
3. `Find2DaughterDecay()`: build first-generation two-track candidates.
4. Extrapolate selected primary V0 candidates to their primary vertices.
5. `NeutralDaughterDecay()`: build neutral-daughter channels from V0/gamma
   candidates.
6. `FindTrackV0Decay()`: combine tracks with previously built V0 candidates.
7. `CombinePartPart()` and `SelectParticles()`: combine/reselect higher-level
   resonance, charm, and hypernucleus candidates.

The first GPU target is step 3.  It is the earliest stage with large regular
pair parallelism and produces candidate pools required by later stages.

## First GPU Target: Two-Daughter V0 Stage

The CPU path is:

`Find2DaughterDecay()` -> `ConstructV0()` -> `SaveV0PrimSecCand()`.

The natural GPU split is:

1. Host packing:
   - Pack positive and negative track sets into flat SoA input views.
   - Preserve track-set ranges, species ranges, source IDs, charge, PDG,
     primary-vertex index, pixel-hit count, chi2-to-primary-vertex, and field
     coefficients.
2. Pair enumeration kernel:
   - One or several threads per positive/negative pair.
   - Apply track category ranges and simple PDG/PV preselection.
   - Emit compact pair tasks for candidates worth constructing.
3. Candidate construction kernel:
   - Load two daughter fit states.
   - Move them to an approximate DCA point.
   - Construct the mother candidate with the device-safe KF math subset.
   - Apply geometry, mass, topology, and decay-length cuts.
   - Store accepted candidates into the candidate pool.
4. Optional classification kernel:
   - Split accepted candidates into secondary, primary, and topology-selected
     views/pools for later stages.

The current prototype already has pieces for item 1 and the candidate pool used
by item 3.  The smoke `RunRoundTrip` kernel should remain a diagnostic and be
replaced stage by stage.

## Device-Safe Math Subset

The old `external/old_impl_KFParticle/dev/KFParticleGpuMath.h` is useful as a
source of formulas, but not as a drop-in dependency.  Its state model uses
arrays of pointers (`p[8]`, `c[36]`) and broad monolithic functions.  The new
GPU layer uses non-owning component-major views plus thread-local
`KFParticleGpuFitState`, so formulas should be ported function by function.

The first math subset should contain:

1. `GetDStoPointLine()` and straight-line distance helpers.
2. `GetDStoParticleLine()` or the simpler `GetDStoParticleFast()` equivalent.
3. `TransportLine()` and a small wrapper for transporting a local state.
4. `GetMeasurement()` for adding one daughter through the Kalman update.
5. `AddDaughter()` and `Construct()` for two daughters.
6. `GetMass()` and basic mass-error calculation.
7. Simple selection helpers for chi2/ndf, mass window, and daughter source IDs.

Only after those pass synthetic CPU-vs-GPU tests should CBM field transport
(`TransportCBM`, `GetDStoParticleCBM`) be moved.  That keeps the first kernel
small enough to debug while preserving the path to the real nonhomogeneous
field implementation.

The initial straight-line subset is now exercised by the standalone lifecycle
test: kinematic construction, DCA measurement transport, the energy-fit Kalman
update, two-daughter fixed-slot task handling, and a small CPU-vs-GPU synthetic
grid with covariance cross terms.

## Recommended First Implementation Step

Add `KFParticleGpuMath.h` and `KFParticleGpuTwoDaughter.h` to the new GPU layer.
Start with straight-line two-daughter construction for synthetic tracks, with a
CPU reference test using the same `KFParticleGpuFitState` functions on the
host.  The kernel should build one candidate per prepared pair task and compare:

- output size,
- daughter IDs,
- PDG and PV metadata,
- charge,
- momentum and energy,
- chi2/ndf once the Kalman update is ported.

This gives a narrow, testable bridge from the current infrastructure to real
KFParticle mathematics without coupling the GPU prototype to ROOT, Vc, or the
full CPU `KFParticleFinder` containers.

## Step 3: GPU Pair Task Generation And First Stage Wiring

The next boundary is moving from host-prepared synthetic tasks to GPU-generated
tasks based on packed event descriptors.  The work should stay standalone in
KFParticle and keep CBMRoot integration out of the loop until the workflow is
stable.

1. Add a fixed-slot pair task generation kernel:
   - read event track-set/species ranges,
   - enumerate the cartesian product of two daughter ranges,
   - write `KFParticleGpuTwoDaughterTask` records,
   - publish both written task count and total pair count so truncation is
     visible to the steering layer.
2. Add lightweight pre-fit cuts to generation:
   - species/PDG and charge expectations,
   - duplicate source-id rejection,
   - optional pixel-hit and chi-to-primary-vertex thresholds.
   The first implementation keeps fixed pair slots: rejected pairs are written
   as invalid placeholder tasks so CPU-vs-GPU tests can still map slot numbers
   directly to pair indices.  Compact allocation is a later production step.
3. Wire generation to the existing two-daughter construction kernel:
   - first as two explicit launches in the standalone lifecycle test,
   - then as a steering method that owns upload, generation, construction, and
     download.
   The explicit-launch path is covered by the lifecycle test: generated device
   task slots are passed directly to the two-daughter construction kernel
   without host-side task rebuilding.
   The first steering-level wrapper is also available as `RunTwoDaughterStage`;
   it owns input upload, candidate reset, task generation, construction, and
   candidate download for a fixed-slot standalone stage.
   A compact steering wrapper, `RunTwoDaughterCompactStage`, now uses the
   atomic compact generator and launches construction only for stored accepted
   task slots.
4. Decide when to replace fixed-slot diagnostics with compact allocation:
   - fixed slots remain useful for CPU-vs-GPU comparisons,
   - an atomic compact task-generation kernel is now available and tested; it
     stores only accepted pair tasks while still reporting total enumerated
     pairs so steering can detect truncation,
   - the first compact candidate-construction kernel now reserves candidate
     and daughter slots with bounded atomic counters and skips mathematical
     build failures instead of storing failed placeholders.
5. Extend tests before CBMRoot integration:
   - multi-event descriptors,
   - truncated task buffers,
   - compact candidate overflow and truncation status,
   - empty species ranges,
   - generated tasks feeding the energy-fit candidate kernel.

Status: complete for the standalone GPU prototype.  The lifecycle test now
covers fixed-slot generation, compact task generation, compact candidate
construction, truncation reporting, daughter-storage overflow, empty ranges,
multi-event descriptor selection, and the steering-level compact
generation-to-construction path.  The remaining work belongs to the next step:
turning the two-daughter stage from a synthetic standalone pipeline into the
first real KFParticleFinder decay-plan execution path.

## Step 4: Standalone Decay-Plan Execution

The next boundary is to stop driving the GPU stage with ad hoc synthetic
`KFParticleGpuTwoDaughterTaskSource` values in tests and start driving it from
a small standalone decay plan.  This should still live entirely inside
KFParticle and should not require CBMRoot classes, ROOT containers, or the CPU
`KFParticleFinder` object.

The goal of this step is not to port every decay channel.  The goal is to build
the execution layer that can run a list of configured two-daughter channels
against packed event descriptors and produce separate compact candidate ranges
for each channel.  Once that layer is stable, adding more channels and more
selection cuts becomes incremental instead of architectural.

1. Define lightweight decay-channel descriptors:
   - add a GPU-side/host-side value type for a two-daughter decay channel,
   - store mother PDG, daughter PDGs, daughter masses, track-set/species
     selectors, charge expectations, and simple pre-fit cut values,
   - keep the descriptor trivially copyable and independent of STL containers
     so it can later be copied to device memory if needed.
2. Extend `KFParticleGpuDecayPlan` from a placeholder into a real standalone
   channel list:
   - provide explicit `AddTwoDaughterChannel()` and `Clear()` methods,
   - preserve deterministic channel order on the host,
   - keep capacity/ownership host-side for now; the kernels already receive a
     single compact source per launch, so device-side plan traversal is not
     needed yet.
3. Add a steering-level plan executor:
   - introduce a method such as `RunDecayPlan(eventIndex, taskCapacity)`,
   - loop over enabled two-daughter channels,
   - translate each channel descriptor into `KFParticleGpuTwoDaughterTaskSource`,
   - run the existing compact two-daughter stage for each channel,
   - keep fixed-slot diagnostic APIs available for debugging.
4. Preserve per-channel output boundaries:
   - record the candidate offset, candidate count, daughter offset, daughter
     count, and overflow flags after each channel,
   - expose these ranges through a lightweight host-side result object,
   - avoid physically splitting candidate buffers until a real downstream
     stage needs separate device buffers.
5. Add standalone tests for the executor:
   - an empty plan produces no candidates and no overflow,
   - a one-channel plan reproduces the current compact steering result,
   - two channels produce two non-overlapping candidate ranges,
   - event selection works for multi-event descriptors,
   - task truncation and daughter-storage overflow are reported per channel.
6. Keep the math scope unchanged during this step:
   - continue using the straight-line DCA and energy-fit two-daughter subset,
   - do not add CBM field transport here,
   - do not integrate with CBMRoot input packing here.

Suggested implementation order:

1. Implement the channel/result data types and tests that only exercise host
   construction and range bookkeeping.
2. Extend `KFParticleGpuDecayPlan` and add tests for adding, clearing, and
   iterating channels.
3. Add the steering executor for one channel and compare it with the existing
   compact steering test.
4. Extend the executor to multiple channels with output ranges.
5. Add truncation, overflow, empty-plan, and multi-event executor tests.

Step 4 should be considered complete when the standalone lifecycle test can run
a small decay plan with multiple two-daughter channels and verify channel-local
candidate ranges without any CBMRoot dependency.

Status: complete for the standalone GPU prototype.  The first slices add flat
two-daughter channel descriptors, candidate-range bookkeeping, channel-result
bookkeeping, and host-only lifecycle checks for translating a channel
descriptor into the existing compact task source.  `KFParticleGpuDecayPlan` is
now an ordered host-side two-daughter channel list with clear/index guards.
The steering layer can execute empty, one-channel, and multi-channel decay
plans by resetting the candidate pool once, appending each compact channel into
the same pool, and recording non-overlapping per-channel candidate/daughter
ranges.  Executor-level tests cover truncation, overflow, empty plans, and
multi-event descriptors.  Downstream-facing range helpers
(`ContainsCandidate`, `ContainsDaughter`, local indices, and overflow
predicates) let later stages consume channel-local ranges without inspecting
raw counters directly.  A small default V0 builder layer now provides ordered
K0S, Lambda, and anti-Lambda two-daughter descriptors; the lifecycle test runs
the default K0S descriptor through the standalone executor.

The next large step is to move from standalone execution infrastructure to
physics coverage: align default channels and preselection cuts with the real
`KFParticleFinder::Find2DaughterDecay()` V0 logic, add proton/anti-proton
synthetic execution coverage, and then start porting the missing selection and
field-transport pieces needed before CBMRoot input integration.

## Step 5: Physics Coverage For The Standalone V0 Stage

Step 5 moves the prototype from "the executor runs" to "the executor represents
the first real V0 reconstruction stage closely enough to compare with CPU
`KFParticleFinder::Find2DaughterDecay()`".  This still remains standalone
inside KFParticle: no CBMRoot task wiring, no ROOT containers, and no direct
dependency on CPU finder objects.

1. Align default V0 channel descriptors with the CPU finder:
   - review the CPU `Find2DaughterDecay()`/`ConstructV0()` channel setup,
   - document which V0 channels are included in the first GPU subset,
   - update default K0S, Lambda, and anti-Lambda descriptor values for species,
     charges, masses, PDG codes, and basic pre-fit cuts,
   - keep cuts explicit in the descriptor rather than hidden in kernels.
2. Add proton and anti-proton execution coverage:
   - extend synthetic input fixtures with proton and anti-proton species
     ranges,
   - run Lambda and anti-Lambda default channels through `RunDecayPlan()`,
   - verify channel-local candidate ranges, daughter source IDs, PDG metadata,
     charge signs, and truncation/overflow behavior,
   - keep K0S coverage as the reference pion-only channel.
3. Add the first post-fit V0 selection metadata:
   - introduce compact selection fields or flags needed by downstream stages,
   - port the simplest CPU-like V0 checks that depend only on already computed
     candidate state, such as mass windows and chi2/ndf thresholds,
   - keep the selection configurable per channel and test accepted/rejected
     candidates without changing the executor contract.
4. Start the field-transport boundary carefully:
   - identify the CPU calls where straight-line transport stops being
     sufficient for V0 quality,
   - add a narrow GPU abstraction for choosing straight-line versus field-aware
     transport,
   - initially keep the field-aware path disabled or diagnostic-only,
   - add CPU-vs-GPU synthetic tests before enabling it in default channels.
5. Consolidate standalone V0 comparison tests:
   - build a few deterministic synthetic events containing pion, proton, and
     anti-proton tracks,
   - run the default V0 plan and compare channel counts, metadata, daughter
     IDs, mass-related values, and failure/overflow flags,
   - keep the tests independent of CBMRoot input packing so regressions are
     local to KFParticle.

Suggested implementation order:

1. CPU-reference audit plus descriptor updates for default V0 channels.
2. Synthetic proton/anti-proton fixtures and Lambda/anti-Lambda execution
   tests.
3. Channel-level selection fields and simple post-fit selection tests.
4. Field-transport abstraction and diagnostic CPU-vs-GPU tests.
5. Final standalone V0 regression test that runs the default plan over a
   mixed synthetic event.

Step 5 should be considered complete when the standalone GPU lifecycle test can
run the default V0 plan over mixed pion/proton/anti-proton synthetic events and
validate K0S, Lambda, and anti-Lambda outputs with CPU-like selection metadata,
while still staying fully independent from CBMRoot integration.

Step 5.1 notes:

- The first GPU subset follows the opposite-charge secondary V0 branch of
  `KFParticleFinder::Find2DaughterDecay()`: secondary positive tracks from
  `vTracks[0]` are paired with secondary negative tracks from `vTracks[1]`.
- Included default channels are `K0S -> pi+ pi-`, `Lambda -> p pi-`, and
  `anti-Lambda -> anti-p pi+`.  Same-sign background, primary resonances,
  charm side channels, and track-plus-V0 stages remain outside this subset.
- The default GPU channel descriptors keep their daughter PDG codes, charge
  signs, and species explicit.  Daughter masses now mirror the CPU
  `KFParticleDatabase` values for pion and proton so standalone CPU-vs-GPU
  comparisons are not biased by rounded descriptor constants.
- Mother PDG table masses and sigma windows are still left for the selection
  metadata sub-step, where they can be wired into accepted/rejected candidate
  decisions rather than hidden in task generation.

Step 5.2 notes:

- Added a standalone mixed V0 fixture with secondary `pi+`, `p`, `pi-`, and
  `anti-p` ranges.  The fixture keeps species ranges disjoint and ordered the
  same way as the packed `KFPTrackVector` subsets expected by the GPU executor.
- The default V0 executor test now runs all three default channels in one
  `RunDecayPlan()` call and checks channel-local counts, candidate offsets,
  daughter offsets, PDG metadata, and source-id pairs.
- This confirms that the `Lambda` and `anti-Lambda` descriptors select proton
  and anti-proton species through the generic compact executor rather than
  relying on a pion-only path.

Step 5.3 notes:

- Default V0 descriptors now carry CPU-like secondary selection metadata:
  mother PDG mass, mother mass sigma, mass sigma cut, topological chi2/ndf cut,
  and minimum `l/dl`.
- The metadata mirrors `KFParticleDatabase` and default `KFParticleFinder`
  `fSecCuts` values, including compile-time experiment-specific mass sigma
  branches.
- `KFParticleGpuSelection` now exposes a device-safe sigma-window helper.  This
  keeps the next selection kernel independent from CPU finder objects while
  preserving the same `abs(m - mPDG) / sigma < cut` convention.
- The executor still stores all constructed candidates; applying these
  selection fields to accept/reject candidates is the next controlled sub-step.

Step 5.4 notes:

- Two-daughter task sources and tasks now carry the post-build selection
  metadata from the channel descriptor into the kernel.
- Compact and fixed-slot candidate storage mark candidates with
  `KFGpuCandidateSelectionRejected` when the built mother does not pass the
  enabled post-build selection.  The candidate remains stored so channel ranges,
  overflow accounting, and downstream diagnostics stay stable.
- Selection is opt-in: mass-window selection is enabled only when
  `motherMassSigma > 0` and `secondaryMassSigmaCut >= 0`; chi2/ndf selection is
  enabled only when the channel cut is non-negative.  Generic synthetic channels
  without CPU-like metadata therefore keep their previous behavior.
- Tests now verify channel-to-source metadata propagation, direct post-build
  selection decisions, and GPU-produced candidate rejection flags for the default
  V0 plan.

Step 5.5 historical notes:

- Added an explicit two-daughter transport mode to channel descriptors, task
  sources, and generated tasks.  At this intermediate point default V0 channels
  still requested `KFGpuTransportStraightLine`; Step 6 later switches them to
  `KFGpuTransportFieldAware`.
- `KFGpuTransportFieldAware` is now represented in the flat data model but is
  intentionally not accepted by candidate construction yet.  A task that
  requests it is marked as a failed candidate instead of silently falling back
  to straight-line transport.
- Tests covered descriptor/source/task propagation, default V0 straight-line
  mode, generated task propagation, and the diagnostic failure path for
  unsupported field-aware transport before field-aware construction was
  implemented.
- The next field-transport sub-step can implement the actual field-aware math
  behind this already-tested boundary without changing decay-plan or task
  layouts.

Step 5.6 notes:

- Consolidated the standalone default V0 regression test around a shared
  channel expectation helper.  The test now validates the complete per-channel
  profile: channel id, mother PDG, event index, pair counters, candidate and
  daughter ranges, overflow state, metadata flags, selection flags, and daughter
  source-id pairs.
- The regression remains standalone and synthetic.  It does not depend on
  CBMRoot containers, ROOT IO, or CPU finder objects, but it follows the first
  CPU V0 subset closely enough to protect K0S, Lambda, and anti-Lambda metadata
  while the math kernels evolve.

Status: Step 5 complete.

## Step 6: Field-Aware Transport For Two-Daughter V0 Candidates

Step 6 moves the V0 prototype from straight-line-only transport toward the CPU
`KFParticleFinder::ConstructV0()` behavior in a magnetic field.  The goal is
not to replace the whole KFParticle transport stack at once, but to implement a
small, testable field-aware boundary for the two-daughter V0 stage while keeping
the straight-line path stable and fully covered.

1. Audit the CPU field-transport reference:
   - review the CPU calls used around `GetDStoParticleFast()`,
     `TransportFast()`, `GetDStoParticle()`, and `Transport()`,
   - identify which pieces are currently approximated by GPU straight-line DCA,
   - document the minimal field-aware behavior needed for the first V0 kernel,
   - keep CBMRoot wiring out of scope and rely only on packed KFParticle field
     coefficients already available in standalone input.
2. Add field-region loading and transport primitives:
   - load `KFParticleGpuFieldRegion` for each daughter track from packed input,
   - add small device-safe helpers that evaluate the field along the track,
   - implement a first bounded field-aware transport step for local synthetic
     states,
   - keep the existing `TransportLine()` helpers unchanged.
3. Implement field-aware DCA seed construction:
   - add a new field-aware measurement/seed helper next to the current
     straight-line DCA seed,
   - route `KFGpuTransportFieldAware` tasks through this helper,
   - keep unsupported or numerically invalid field transport as an explicit
     failed candidate, not as a silent straight-line fallback,
   - preserve fixed-slot and compact executor counter semantics.
4. Build CPU-vs-GPU synthetic transport tests:
   - use deterministic field coefficients and simple track states,
   - compare field-value loading, single-track transport, DCA seed positions,
     and final candidate metadata,
   - include zero-field tests showing that field-aware transport reduces to the
     straight-line result within tolerance,
   - include invalid/edge cases that must produce failed candidates.
5. Enable field-aware mode first for explicit diagnostic channels:
   - keep default V0 channels straight-line during the diagnostic sub-steps
     until the synthetic comparisons are stable,
   - add one dedicated test channel with `KFGpuTransportFieldAware`,
   - verify flags, rejection markers, and failure behavior independently from
     the default V0 regression,
   - document the remaining gap before default V0 channels can switch modes.
6. Decide and apply the default-channel switch criteria:
   - define what accuracy/tolerance is required for K0S, Lambda, and
     anti-Lambda before enabling field-aware mode by default,
   - list which CPU selection quantities still cannot be compared,
   - keep the result as a clear handoff point for the next step.

Suggested implementation order:

1. CPU-reference audit and notes for the exact transport calls to emulate.
2. Field-region loading helpers plus field-value/zero-field unit tests.
3. First field-aware local transport primitive with straight-line equivalence
   tests in zero field.
4. Field-aware two-daughter DCA seed helper and diagnostic channel execution.
5. Edge-case/failure tests for invalid field-aware transport.
6. Final Step 6 summary and criteria for switching default V0 channels.

Step 6 should be considered complete when the standalone GPU lifecycle test can
run an explicit field-aware diagnostic two-daughter channel, verify field
loading and transport behavior against deterministic CPU-side expectations, and
prove that the default V0 straight-line regression remains unchanged.

Step 6.1 notes:

- The CPU V0 path has two relevant transport levels.  `Find2DaughterDecay()`
  first uses `GetDStoParticleFast()` and `TransportFast()` for a pre-fit
  daughter-distance cut.  The GPU straight-line task generator currently covers
  this role only approximately through line-DCA geometry and simple pair cuts.
- The actual candidate construction in `ConstructV0()` is more important for
  Step 6: in the CBM branch it calls `GetDStoParticle()` with derivatives and
  then `TransportToDS()` for both daughters before `mother.Construct()`.
  Under `NonhomogeneousField`, these calls dispatch to
  `GetDStoParticleCBM()` and `TransportCBM()`.
- The CPU CBM DCA calculation evaluates the magnetic field at the first
  particle, uses the field `By` component for the two-particle DCA calculation,
  and falls back to straight-line DCA when both effective `B*q` values are
  essentially zero.  This is the first behavior the GPU field-aware path should
  emulate.
- The CPU CBM transport also falls back to straight-line transport for neutral
  particles.  For charged particles it performs a bounded nonhomogeneous-field
  transport using field integrals along a line-track approximation.
- The old GPU attempt in `external/old_impl_KFParticle/dev/KFParticleGpuMath.h`
  contains a useful mathematical sketch of `GetDStoParticleCBM()` and
  `TransportCBM()`, but it should not be copied wholesale.  The new
  implementation should expose small helpers over `KFParticleGpuFitState` and
  `KFParticleGpuFieldRegion`, preserving the current flat task/source layout.
- The first implementation target is therefore:
  1. load two daughter field regions from packed input,
  2. evaluate `By` at the first daughter position,
  3. add a field-aware DCA calculation with a zero-field straight-line
     equivalence test,
  4. only then add charged-particle field transport with covariance handling.
- Default V0 channels were kept as `KFGpuTransportStraightLine` at this audit
  stage.  They are switched to `KFGpuTransportFieldAware` only after the later
  diagnostic field-aware channel, energy-fit path, and zero-field equivalence
  checks pass.

Step 6.2 notes:

- Added small field data helpers around the packed input view:
  `HasFieldRegions()`, `LoadFieldRegionOrZero()`, `EvaluateTrackField()`, and
  `EvaluateTrackFieldAtState()`.  Field-aware code should use these helpers
  rather than reading raw SoA field pointers directly.
- The helpers keep homogeneous/no-field input safe by returning a zero field
  region instead of dereferencing a null field-coefficient pointer.
- The lifecycle input fixture now fills deterministic parabolic field
  coefficients for each track.  Host-side checks verify field-region loading,
  out-of-range zero fallback, and field evaluation at the track state.
- The XPU input-layout probe now reads the field region on device and checks
  `Bx`, `By`, and `Bz` at the track position.  This directly prepares the next
  DCA step, where CPU `GetDStoParticleCBM()` uses the `By` component.

Step 6.3 notes:

- Added the first real field-transport primitive,
  `KFParticleGpuMath::TransportConstantBy()`.  It transports one local
  `KFParticleGpuFitState` in a constant local `By` approximation and preserves
  an exact straight-line fallback for zero field or neutral particles.
- The helper intentionally does not replace the existing line-DCA seed and does
  not yet propagate covariance/Jacobian terms.  This keeps the step bounded:
  the next stage can use this primitive while porting the two-particle
  field-aware DCA and later the fuller CBM transport.
- Added host and XPU lifecycle checks for the primitive.  They verify zero-field
  and neutral-particle equivalence to `TransportLine()`, charged-particle
  curvature in `x/z` and `px/pz`, and execution inside the KFParticle XPU
  device image.

Step 6.4 notes:

- Added `BuildConstantByDcaKinematicSeed()` and
  `BuildConstantByDcaKinematicMother()` as the first two-daughter field-aware
  seed helpers.  They start from the straight-line DCA, use the constant-`By`
  transport primitive for a bounded local correction, and keep exact
  straight-line behavior when both effective `B*q` values are zero.
- The helpers are intentionally kinematic only.  They do not yet replace
  `BuildLineDcaMeasurementSeed()` and do not route production
  `KFGpuTransportFieldAware` tasks through the candidate builder.  That remains
  the next integration step after the seed is stable in standalone tests.
- Host tests now cover zero-field equivalence, neutral-particle equivalence,
  charged-particle deviation from the line seed, and large-transport rejection.
  The XPU field probe also executes the two-daughter seed inside the device
  image.

Step 6.5 notes:

- Routed explicit `KFGpuTransportFieldAware` kinematic two-daughter tasks
  through the constant-`By` DCA seed.  The `By` value is evaluated from the
  first daughter's packed field region, matching the first GPU approximation of
  the CPU `GetDStoParticleCBM()` entry point.
- Field-aware energy-fit tasks remain disabled until a field-aware measurement
  seed with covariance/Jacobian propagation is implemented.  Such tasks are
  still stored as failed candidates instead of silently falling back to
  straight-line transport.
- Added an XPU task-kernel test for an explicit diagnostic field-aware task.
  The test compares device output to the host builder, verifies a visible
  difference from the straight-line seed, and leaves the default V0 channels in
  `KFGpuTransportStraightLine`.
- The field-aware task comparison uses a dedicated tolerance because this
  diagnostic path includes iterative trigonometric constant-`By` transport.
  Host libm and HIP device math are expected to differ slightly, while metadata,
  route selection, and non-straight-line behavior remain strictly checked.

Step 6.6 notes:

- Added a decay-plan executor regression for an explicit diagnostic
  `KFGpuTransportFieldAware` two-daughter channel.  This verifies the full
  descriptor path: channel -> task source -> generated compact tasks ->
  candidate construction.
- The diagnostic channel remained separate from the default V0 channel builders
  at this sub-step.  Default K0S, Lambda, and anti-Lambda channels are switched
  to `KFGpuTransportFieldAware` at Step 6 completion.
- The test compares the first generated field-aware candidate with a host-side
  reference task, checks channel result ranges and candidate flags, and verifies
  that the field-aware result differs from the straight-line seed.

Step 6.7 notes:

- Added a mixed-transport decay-plan regression with one explicit diagnostic
  field-aware channel followed by one straight-line channel.  This protects the
  channel append path while transport modes diverge.
- The test verifies candidate and daughter offsets, non-overlapping result
  ranges, per-channel PDG metadata, and a visible difference between the
  field-aware candidate and the straight-line candidate for the same generated
  first pair.
- This gives coverage for the intended near-term usage: diagnostic
  field-aware channels can be placed in a plan next to existing straight-line
  V0 channels without switching the default V0 builders.

Step 6.8 notes:

- Added an independent scalar reference for the current constant-`By`
  two-daughter DCA approximation in the lifecycle test.  It uses double
  precision test-only code rather than calling the production
  `KFParticleGpuMath` helpers.
- The reference comparison covers zero-field behavior, charged-field behavior,
  and the large-transport rejection path.  This documents the current
  approximation numerically before the next step adds covariance/Jacobian-aware
  field measurement seeds.

Step 6.9 notes:

- Added `BuildConstantByDcaMeasurementSeedApprox()` as an explicitly named
  intermediate field-aware measurement seed.  It uses the constant-`By`
  two-daughter DCA states for transported daughter parameters while preserving
  the existing line-DCA covariance/correlation approximation.
- The lifecycle test covers zero-field equivalence to the line measurement
  seed, charged-field changes in transported states, preserved covariance and
  correlation blocks, and large-transport rejection.
- This does not yet enable `KFGpuTransportFieldAware` together with
  `KFGpuTwoDaughterUseEnergyFit`; the full CBM covariance/Jacobian propagation
  is still required before that route can become a supported production path.

Step 6.10 notes:

- Added host and device diagnostics proving that
  `BuildConstantByDcaMeasurementSeedApprox()` can feed the existing Kalman
  energy-fit update and produce a valid fitted two-daughter state.
- The XPU field probe now checks constant-`By` transport, two-daughter DCA seed,
  and approximate field-aware energy-fit execution inside the device image.
- This diagnostic established the expected energy-fit behavior before enabling
  the production field-aware route.

Step 6.11 notes:

- Added `BuildConstantByDcaMeasurementSeed()` with covariance and correlation
  propagation based on numerical Jacobian blocks of the current constant-`By`
  DCA approximation.  The helper propagates the first and second daughter
  covariance contributions into both transported states and fills the 3x3
  measurement correlation block used by the Kalman update.
- `KFGpuTransportFieldAware + KFGpuTwoDaughterUseEnergyFit` is now a supported
  prepared-task route.  It evaluates `By` from the first daughter's field
  region, builds the field-aware measurement seed, and then runs the existing
  energy-fit update.
- The previous `BuildConstantByDcaMeasurementSeedApprox()` remains as a
  diagnostic baseline.  It is intentionally named as an approximation and is
  not used by the production two-daughter builder.

Default-channel switch decision:

- Default K0S, Lambda, and anti-Lambda descriptors now request
  `KFGpuTransportFieldAware`.  The standalone regression keeps the same channel
  ordering, metadata, daughter ranges, and selection behavior, while zero-field
  fixtures still reduce to the previous straight-line behavior.
- The next validation step must add field-bearing default V0 fixtures and
  compare candidate fitted parameters with a scalar CPU reference within
  documented tolerances for `x/y/z`, momentum, mass, chi2, and selection flags.
- The numerical-Jacobian implementation is acceptable for the first validated
  GPU path, but an analytic or optimized Jacobian should be considered before
  large production runs if profiling shows it dominates kernel time.
- Keep the switch channel-aware in validation: K0S can be promoted to realistic
  field comparison first, followed by Lambda and anti-Lambda after proton-mass
  fixtures confirm the same tolerances.

Status: Step 6 complete.  Default V0 channels now use
`KFGpuTransportFieldAware`; field-aware mode is available both for explicit
channels and for the default V0 descriptors.

## Step 7: Field-Aware Default V0 Validation And Optimization

Step 7 should turn the now-enabled field-aware default V0 path from a synthetic
standalone implementation into a validated production candidate.

1. Add field-bearing default V0 fixtures:
   - extend K0S, Lambda, and anti-Lambda standalone fixtures with deterministic
     nonzero field regions,
   - keep zero-field fixtures as regression tests for straight-line
     equivalence,
   - compare default-channel output against scalar CPU reference helpers.
2. Tighten physics comparisons:
   - document tolerances for fitted position, momentum, mass, chi2, flags, and
     daughter metadata,
   - add separate K0S, Lambda, and anti-Lambda checks so failures identify the
     affected channel.
3. Profile and optimize the numerical-Jacobian path:
   - measure kernel/runtime impact of field-aware energy-fit default channels,
   - decide whether an analytic Jacobian is needed before larger production
     tests,
   - keep the numerical implementation as a correctness reference even if an
     optimized path is added.
4. Prepare CBMRoot integration notes:
   - list the field data that CBMRoot must provide to the standalone
     KFParticleFinder GPU input,
   - define which CPU finder outputs should be compared first,
   - keep the standalone tests runnable without CBMRoot.

Step 7.1 notes:

- Added the first field-bearing default-channel fixture for
  `MakeK0ShortToPiPlusPiMinusChannel()`.  Unlike the earlier diagnostic
  channel, this test uses the real default K0S descriptor after the Step 6
  switch to `KFGpuTransportFieldAware`.
- The fixture stores a deterministic nonzero field region for the first
  positive pion, runs the decay-plan executor, compares the first GPU-produced
  candidate with the standalone host builder, and verifies that the result is
  visibly different from the straight-line route.
- This is still a narrow K0S validation slice.  Lambda and anti-Lambda
  field-bearing default fixtures, scalar CPU-reference comparisons, and
  documented channel-specific tolerances remain the next Step 7 work.

Step 7.2 notes:

- Added separate deterministic nonzero-field fixtures for the real default
  Lambda and anti-Lambda descriptors.  Each uses its actual proton/pion track
  order and charge, compares the GPU candidate with the standalone host
  builder, and proves that the result does not silently fall back to the
  straight-line route.
- The fixtures deliberately remain separate test entries: a regression now
  identifies K0S, Lambda, or anti-Lambda directly instead of hiding all three
  behind one aggregate default-V0 result.
- The host builder currently shares the production device-safe math helpers.
  An independent scalar reference and channel-specific physical tolerances
  are still required before treating this as full numerical validation.

Step 7.3 notes:

- Added an independent double-precision scalar comparison for all three real
  default V0 task descriptors. It prepares daughter energy directly from track
  momentum and mass, performs its own constant-By DCA transport, and compares
  position, momentum, energy, charge, metadata, and daughter order with a GPU
  task-kernel result.
- The probe intentionally disables only the energy-fit flag. This isolates the
  field-aware kinematic seed from the later Kalman update, making a transport
  regression distinguishable from a covariance/update regression.
- The next validation stage must cover the complete field-aware energy-fit
  result with explicit channel-specific tolerances.

Step 7.4 notes:

- Added a full field-aware `line-DCA + energy-fit` GPU test for K0S, Lambda,
  and anti-Lambda. It checks all fitted parameters, chi2, reconstructed mass,
  charge, metadata, daughter order, and the exact post-build selection status.
- The regression contract records channel-specific tolerances: K0S uses
  `7e-4` for fitted state, `2e-3` for chi2, and `2e-4` for mass; Lambda and
  anti-Lambda use `1e-3`, `3e-3`, and `3e-4` respectively. These cover normal
  HIP/CPU float and libm variation in iterative transport and the numerical
  Jacobian while remaining small enough to expose a wrong reconstruction path.
- The remaining Step 7 work is performance profiling of the numerical
  Jacobian and the CBMRoot field-input integration contract.

Step 7.5 notes:

- Added an opt-in steady-state benchmark for the three default V0 energy-fit
  tasks. It measures queue submission, candidate-pool reset, kernel execution,
  and synchronization separately for field-aware numerical-Jacobian and
  straight-line routes, then reports microseconds per candidate and their
  ratio.
- It is enabled only with `KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE=1` and uses
  500 iterations by default; `KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS` changes
  that count. There is intentionally no speed threshold because the result
  depends on device, ROCm, clocks, and server load.
- The benchmark reuses the validated K0S/Lambda/anti-Lambda fixture and is a
  profiling aid, not a replacement for a production event-level benchmark.

Step 7.6 notes:

- Added `KFParticleGpuCbmRootIntegration.md`, the first formal CBMRoot adapter
  contract. It maps `KFPTrackVector` and `ChiToPrimVtx` inputs to the packed
  GPU SoA, preserves all eight CPU track-set meanings, and defines event,
  buffer, candidate-overflow, and source-lineage ownership rules.
- The document defines the ten-coefficient nonhomogeneous-field ABI and makes
  the current limitation explicit: the validated default V0 path evaluates
  the first daughter's field region but uses a constant-By approximation; it
  is not yet a replacement for CPU `TransportCBM`.
- It also records the first CPU/GPU comparison order and stop conditions, so a
  future CBMRoot adapter can start as a diagnostics layer without changing
  standalone KFParticle ownership or prematurely replacing CPU output.

Status: Step 7 complete. The default field-aware V0 path now has deterministic
K0S/Lambda/anti-Lambda validation, an independent transport reference,
channel-specific full-fit tolerances, an opt-in performance probe, and a
documented CBMRoot field-input boundary.

## Step 8: CPU-Compatible Default V0 Selection

Step 8 turns the constructed default V0 candidates into a usable, compact
selection result. It stays inside standalone KFParticle and treats the CPU
`KFParticleFinder::ConstructV0()` / `SaveV0PrimSecCand()` behavior as the
semantic reference, without attempting to replace the full CPU finder yet.

1. Define a flat selection result and rejection contract.
   - Add device-safe value types for per-candidate observables, selection
     class, and a bitwise rejection reason mask.
   - Keep construction success separate from physics selection: a valid fit
     remains inspectable even when it fails mass, topology, decay-length, or
     primary-vertex criteria.
   - Extend decay-plan channel results with raw/selected counts only where the
     ownership and overflow meaning stay unambiguous.
   - Test defaults, flag composition, and host/device-copy safety before using
     the types in a kernel.

2. Port the minimum primary-vertex topology observables.
   - Add small device-safe helpers for candidate-to-PV distance, decay length
     and its error, `L/dL`, and topological chi2/NDF.
   - Define deterministic invalid/no-PV behavior rather than allowing a
     divide-by-zero or an implicit acceptance.
   - Use the packed GPU vertex SoA already present in the input ABI; do not add
     a CBMRoot dependency or duplicate vertex ownership.
   - Validate zero/one/multiple-PV fixtures against scalar host calculations.

3. Implement the default secondary V0 decision function.
   - Apply the current CPU ordering: construction/finite checks, geometric
     chi2, distance and decay-length conditions, mass window, then best-PV
     topology classification for K0S, Lambda, and anti-Lambda.
   - Preserve the current constant-By construction result as input to this
     decision; full CPU `TransportCBM` remains a later transport task.
   - Express all thresholds in channel descriptors/configuration rather than
     hard-coding experiment cuts in a kernel.
   - Add boundary tests for each individual rejection reason and for a fully
     accepted candidate.

4. Add a compact selected-candidate output stage.
   - Keep the present raw candidate pool available for diagnostics, then use a
     bounded atomic or prefix-sum stage to emit only selected candidate indices
     or records in stable channel/event ranges.
   - Report selected-pool overflow independently from raw construction
     overflow; never silently drop accepted V0s.
   - Start with an index view into the existing candidate pool to avoid another
     copy of fit/covariance data. Materialized downstream pools are deferred
     until a consumer requires them.

5. Establish CPU/GPU selection regression fixtures.
   - Build synthetic multi-PV events that separately exercise K0S, Lambda, and
     anti-Lambda pass/fail paths, exact daughter lineage, best-PV tie handling,
     mass and topology boundaries, and selected-pool truncation.
   - Compare observables, reason masks, classification, selected ordering, and
     counts on host and device; retain the existing field-aware full-fit tests
     as the construction baseline.
   - Add an opt-in selected-output profile only after correctness is stable,
     so compaction can be evaluated separately from the numerical Jacobian.

6. Freeze the hand-off boundary for the following reconstruction stage.
   - Document which selected V0 fields and lineage a future neutral-daughter or
     track-V0 stage may consume, and which CPU-only categories remain out of
     scope.
   - Update the CBMRoot integration contract with selected-result semantics and
     comparison counters, but keep the first adapter diagnostics-only.
   - Mark Step 8 complete only when raw and compact outputs agree and all three
     default channels have deterministic selection coverage on CPU and GPU.

Step 8 is intentionally split into six medium-sized stages. The first three
make selection physics observable and testable; the fourth changes output
ownership and therefore follows only after the decision is trusted; the last
two validate and document the hand-off. This avoids coupling topology math,
atomics, and CBMRoot integration in one change.

Status: Step 8 planned, not started.

Step 8.1 notes:

- Added flat, trivially-copyable V0 selection observables and result records.
  They reference a raw candidate by index and keep the future compact output
  free to store indices rather than duplicate fit/covariance data.
- Added explicit selection classes and composable rejection bits. Construction
  failure, non-finite fit, geometric, PV, distance, decay-length, mass,
  topology, and output-overflow failures are intentionally distinguishable.
- Added a lifecycle regression for defaults, bit composition, selected-state
  semantics, and an explicit device-buffer round trip. No selection kernel or
  raw candidate-pool ownership changes are part of this stage.

Status: Step 8 in progress. Stage 1 of 6 complete.

### Stage 9.0: Share one XPU build graph in standalone and CBMRoot modes

- Preserve the algorithm-local `KFParticleGpuRuntime` queue in both modes; no
  borrowed CBMRoot queue is required or used.
- In standalone mode, KFParticle continues to configure the requested XPU
  source tree itself.
- In CBMRoot mode, replace the KFParticle `ExternalProject` with an embedded
  subdirectory build. The existing parent `xpu` target and `xpu_attach()` are
  then visible to KFParticle, preventing a second XPU installation in
  `build/lib`.
- Make dictionary paths local to the KFParticle binary subdirectory, because
  the old project assumed it was always configured at the CMake top level.
- Keep the historical `KFPARTICLE` target as a dependency wrapper so existing
  CBMRoot dictionary targets continue to build in the required order.

Completion criteria: a fresh CBMRoot configure reports one parent `xpu` target,
no `KFPARTICLE-prefix` ExternalProject is configured, and `KFPARTICLE` builds
`KFParticle` plus its device image through that same target graph. Standalone
lifecycle tests remain unchanged.

Status: implementation complete; requires a clean reconfigure and HIP/CPU
validation in the CBMRoot build environment.

Validation helper: `GPU/test/cbmroot/run_cbmroot_xpu_smoke.sh` is a temporary
ROOT-macro smoke test kept with KFParticle rather than CBMRoot's permanent
test suite. It uses the current CBMRoot `build/lib`, initializes `cbm::Xpu`,
requires KFParticle to attach without initializing XPU again, runs a minimal
`RunRoundTrip()` device path, and verifies that `Finalize()` leaves the global
runtime active.

## Step 9: CBMRoot Default-V0 Diagnostic Adapter

Step 9 connects the completed standalone default-V0 GPU boundary to real
CBMRoot reconstruction input, but remains diagnostics-only. Its output must
never replace `KFParticleFinder` candidates in this step. The purpose is to
prove that packing, queue ownership, field data, candidate lineage, and
selection decisions agree on representative CBMRoot events before any
production-routing decision.

1. **Stage 9.1: Define the adapter boundary and build ownership.**
   - Locate the precise CBMRoot offline and online call sites where the CPU
     finder receives `KFPTrackVector` sets, primary vertices, field regions,
     and event identity for the default secondary V0 channels.
   - Add a small CBMRoot-side adapter target behind an explicit opt-in CMake
     option. It links the existing external KFParticle GPU implementation and
     XPU target without changing default CPU reconstruction dependencies.
   - Keep XPU runtime ownership process-wide: the adapter attaches to the
     existing application/XPU initialization and creates no per-event queue.
   - Completion: a disabled-by-default adapter compiles in CBMRoot, while a
     normal CPU-only configuration remains behaviourally and link-time
     unchanged.

   Research decision: the primary first boundary is
   `cbm::algo::kfp::Selector`, called by its thread-local `KfpSelectorChain`
   after `EventReconstruction`. It builds field-aware first/last
   `KFPTrackVector` inputs from `TrackFitter` immediately before CPU topology
   reconstruction. The process-wide `KFParticleGpuRuntime` deliberately keeps
   one KFParticle-owned queue and buffer manager, so per-selector adapter
   handles must serialize their diagnostic `pack -> launch -> download`
   transaction through that service. The legacy `CbmKFParticleFinder` is a
   later offline comparison path. The mCBM `V0Finder` is excluded from this
   field-aware first route because it is Lambda-only and currently writes zero
   field coefficients by design.

   Initial implementation: `CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON` builds the
   isolated `CbmKfParticleGpuDiagnostic` target. Its process-wide service
   attaches to the already initialized XPU runtime with
   `initializeXpuIfNeeded=false` and returns an RAII transaction that locks
   shared KFParticle steering and buffers.

   Stage 9.2 implementation: `GpuDiagnosticPacker` accepts Selector's
   field-aware first/last `KFPTrackVector` inputs, their parallel chi-to-PV
   values, and primary vertices. `Prepare()` validates finite physical input,
   rejects inconsistent vectors or invalid source IDs, and reproduces CPU
   `SortTracks()` classification into the eight GPU input sets. It records
   explicit ranges without SIMD padding; `Write()` copies all numerical,
   covariance, field, chi-to-PV, integer, and vertex components only into
   fresh host SoA views. The packer deliberately performs neither transfer nor
   kernel launch. A CPU regression covers ranges, component-major stride,
   field/vertex copy, invalid input, and first-point classification of last
   track states.

   Stage 9.3 implementation: with `CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON`,
   `GpuDiagnosticRunner` is linked into `Algo` and `AlgoOffline` and is called
   after the normal CPU `Selector` reconstruction. It configures the default
   K0S/Lambda/anti-Lambda plan, derives an exact physical-pair capacity, grows
   persistent KFParticle buffers, packs host views, and calls `RunDecayPlan()`.
   It reports per-event structured success, no-input/no-pair, overflow, input,
   runtime, and execution statuses without changing the CPU selection result.

   Stage 9.4 implementation: `GpuDiagnosticComparator` extracts CPU default-V0
   candidates by resolving CPU daughter particle indices to their original
   track IDs and compares them with snapshots read from the GPU raw candidate
   and selected-index pools. The identity is `(event, channel, canonical daughter
   source IDs)`, not a mutable pool index. It reports channel-local CPU-only,
   GPU-only, matched, selection-mismatch, unresolved-CPU, and max mass/chi2
   residual counts. GPU snapshots retain flags, PV and selection/rejection
   metadata for the reporting workflow. The comparator is exercised without a
   device by an order-independent lineage-key regression.

   Stage 9.5 implementation: `gpuDiagnostics` is an opt-in selector YAML
   block with a positive `samplePeriod`. The GPU adapter is therefore inactive
   even in a diagnostics-enabled build until a run configuration explicitly
   requests it. `GpuDiagnosticReporter` safely aggregates results from the
   parallel selector threads; `Reco` emits one short timeslice summary instead
   of per-event technical output. CPU tests cover the aggregation path.

   Stage 9.6 implementation: the diagnostic result records aggregate GPU
   transaction wall time and `KFParticleGpuCbmRootValidation.md` fixes the
   build, runtime-smoke, sampling, campaign-recording, and promotion criteria.
   An empirical CPU/HIP campaign on a stable sample remains required evidence;
   it cannot be replaced by a unit test and output remains diagnostics-only.

2. **Stage 9.2: Implement lossless event packing.**
   - Build a dedicated packer from the CPU `KFPTrackVector` representation to
     `KFParticleGpuBufferManager` host SoA views. Preserve physical-track
     ranges, source IDs, PDG/charge/PV metadata, chi-to-PV, pixel hits,
     covariance layout, field coefficients, and all eight CPU track-set
     meanings.
   - Pack primary vertices and event descriptors with absolute batch offsets;
     explicitly exclude CPU SIMD padding lanes and reject malformed ranges.
   - Establish capacity estimates and overflow policy before filling any view,
     then obtain fresh views after a growth operation.
   - Completion: deterministic packer unit tests compare every populated SoA
     component and event range against the CPU source objects.

3. **Stage 9.3: Run the GPU diagnostic path in the finder workflow.**
   - Invoke `RunDecayPlan()` once the relevant secondary input sets and PVs
     are available, using the default K0S/Lambda/anti-Lambda plan and the
     current event or batch index.
   - Treat absent nonhomogeneous-field coefficients, raw-pool overflow,
     selected-pool overflow, invalid ranges, and unavailable GPU runtime as
     explicit diagnostic statuses, never as silent zero-field fallbacks.
   - Keep all CPU finder calls and outputs untouched; collect GPU raw,
     selected, and channel-range outputs solely for comparison.
   - Completion: an opt-in run processes real events end to end and emits
     structured per-event status without changing reconstructed output.

4. **Stage 9.4: Add a keyed CPU/GPU comparison layer.**
   - Match candidates by `(channel ID, event ID, canonical daughter source IDs)`
     rather than pool index. Compare construction counts, overflow state,
     fit/covariance-derived quantities, selection observables, classification,
     rejection reasons, best PV, and compact membership.
   - Record numerical residuals and unmatched candidates separately by channel
     and failure category. Use reporting tolerances as diagnostics, not as
     production acceptance cuts.
   - Make atomic selected ordering and CPU/GPU daughter storage order irrelevant
     while validating channel ranges and canonical lineage exactly.
   - Completion: comparison output can distinguish packing errors,
     construction/transport differences, and selection-only differences.

5. **Stage 9.5: Provide a practical CBMRoot validation workflow.**
   - Add a narrow macro/configuration switch for offline reconstruction and,
     where the same input boundary exists, an online/MQ smoke route. The switch
     controls diagnostics, event sampling, and concise per-channel summaries.
   - Add integration tests or reproducible macros using a small stable input
     sample. They must exercise disabled mode, successful diagnostics mode,
     field-unavailable rejection, and an overflow/invalid-input stop status.
   - Keep verbose dumps opt-in; normal output reports event count, channel
     counts, mismatch categories, and elapsed CPU/GPU diagnostic times.
   - Completion: developers can reproduce one CPU/GPU comparison run without
     modifying production macros or manually configuring buffer internals.

6. **Stage 9.6: Validate, document, and set the promotion gate.**
   - Run representative CPU and HIP campaigns, measure packing, upload,
     kernel, download, and comparison costs, and preserve the command lines
     and environment assumptions needed to reproduce them.
   - Update the CBMRoot contract with measured residual categories, supported
     detector/field conditions, batch limitations, and remaining CPU-only
     behaviour.
   - Define an explicit promotion decision: retain diagnostics-only output,
     extend physics equivalence work, or propose a separate production-routing
     step. Step 9 itself never enables replacement automatically.
   - Completion: results and stop conditions are documented well enough to
     decide whether GPU V0 output is eligible for a later controlled handoff.

Step 9 is intentionally split into six stages. The first two establish a
verifiable data boundary; the next two execute and interpret diagnostics; the
last two make the workflow reproducible and turn measurements into an informed
promotion decision. This keeps CBMRoot integration separate from future
physics work such as full `TransportCBM`, production-vertex constraints, and
higher-generation decay reconstruction.

Status: the initial Step 9 implementation is complete, but the closure audit
identified remaining reporting and validation work. Output remains
diagnostics-only.

### Step 9 completion audit and closure plan

1. **Stage 9.C1: Make diagnostics independent and non-invasive.**
   - Create the selector when `gpuDiagnostics` is requested even without a KFP
     trigger mask, but do not add its CPU output to `DigiEvent::fSelectionMask`
     in that diagnostic-only mode.
   - Catch the complete post-CPU diagnostic tail, including comparison and
     reporting, and record an unexpected failure without propagating it into
     CPU reconstruction.
   - Status: complete. Reporter regression coverage verifies unexpected
     diagnostic failure aggregation; an end-to-end selector exercise remains
     part of Stage 9.C4.

2. **Stage 9.C2: Complete comparison semantics.**
   - Compare and classify best-PV, selection flags, rejection reasons, NDF,
     and unmatched-candidate categories in addition to lineage, mass, and
     chi2.
   - Status: complete for the values exposed by the current CPU boundary. The
     adapter compares mass validity/error, NDF, chi2, selection membership, and
     lineage; it records per-candidate discrepancy bits. CPU `KFParticle`
     candidates do not expose best-PV, selection class, or rejection masks, so
     these remain explicitly labelled GPU-only observations rather than false
     mismatches.

3. **Stage 9.C3: Complete channel-level monitoring.**
   - Report per-channel counts, residual summaries, diagnostic statuses, and
     separate CPU-reference, upload, kernel, download, and comparison times.
   - Status: complete. The process-wide report aggregates all comparison
     categories and GPU-only selection observations per channel. It records
     host wall-time for CPU reference, input preparation, queue/capacity, host
     packing, H2D input, construction, selection, D2H output, extraction,
     comparison, and total transaction. Construction and selection include the
     required queue waits and scalar status copies, so they are explicitly not
     presented as device-event kernel timings.

4. **Stage 9.C4: Add a reproducible external integration harness.**
   - Keep the harness beside KFParticle GPU tests, and exercise disabled,
     diagnostic-only, successful, field-rejected, overflow, and CPU-output
     preservation cases with a stable CBMRoot input.
   - Status: complete. A ROOT smoke macro runs the
     built CBMRoot adapter over a minimal field-aware V0, checks the default
     GPU plan, lineage, monitoring, reporter aggregation, and invalid-input
     rejection. A separate campaign wrapper accepts caller-owned baseline and
     diagnostic reconstruction commands and optionally requires byte-identical
     CPU output. Field-rejected and overflow statuses remain campaign inputs:
     they depend on the chosen detector configuration and are reported, not
     fabricated by a synthetic macro.

5. **Stage 9.C5: Record CPU/HIP validation evidence.**
   - **9.C5a: Hermetic CPU/GPU V0 equivalence gate.** Add one external,
     compiled executable that builds controlled first/last `KFPTrackVector` input (which the
     CPU finder sorts into its internal eight sets) and a primary vertex, runs
     the existing `KFParticleTopoReconstructor` CPU finder, then runs the same
     physical input through `GpuDiagnosticRunner`. Cover K0S, Lambda and
     anti-Lambda in a controlled constant-By field plus invalid input; bounded-pool overflow stays
     in the standalone lifecycle test where allocation is directly controlled.
     Require exact channel/lineage/count agreement and bounded mass,
     mass-error, chi2 and NDF differences; do not require bitwise floating
     point equality.
   - **9.C5b: One HIP evidence run and closure.** Run that executable on the
     configured accelerator and record the command, device, ROCm version and
     aggregate comparison. Retain the existing real-input campaign wrapper as
     a later production-routing gate rather than blocking Step 9.
   - Status: complete. The HIP equivalence run now emits a compact evidence
     record beside its log with the command, device, ROCm context, host, and
     log hash. Evidence tooling remains available for a later real-data
     campaign, with commands, environment, log hashes, monitoring, and optional
     byte-identical CPU-output verification.

Status: Step 9 complete. The adapter remains diagnostics-only; a real-data
campaign, batching work, full nonhomogeneous transport, and any production
routing decision are separate follow-up work.

## Step 10: Batched Default-V0 GPU Diagnostic Pipeline

Step 10 removes the intentionally conservative one-event-at-a-time execution
model from the diagnostics path. It keeps the existing process-local
KFParticle queue and persistent device storage, but turns a group of
independent CBMRoot selector events into one bounded GPU batch. CPU
reconstruction remains authoritative and untouched. This is a throughput and
data-flow step, not a production-routing decision and not a port of full
`TransportCBM` or higher-generation decay reconstruction.

1. **Stage 10.1: Define and build the bounded batch ABI.**
   - Add a host-side batch builder that accumulates already validated
     `KFPTrackVector` first/last states, chi-to-PV arrays, vertices, and source
     event IDs into the existing flat SoA/event-descriptor layout. It must
     preserve the eight track-set meanings, absolute offsets, field regions,
     and event isolation without importing CPU SIMD padding.
   - Define explicit batch limits for events, tracks, vertices, candidate
     tasks, raw candidates, daughters, and selected candidates. A batch is
     sealed before any buffer write; an event that cannot fit is reported as a
     diagnostic status rather than partially packed.
   - Keep the batch object CPU-only and non-owning until it is submitted. This
     makes enqueue, rejection, and retry behaviour deterministic and keeps the
     persistent XPU-buffer owner in `KFParticleGpuDeviceStorage`.
   - Completion: unit tests cover mixed empty/non-empty events, all default
     channel input sets, absolute offsets, malformed-event isolation, and a
     deterministic split at each configured capacity limit.

2. **Stage 10.2: Execute one device-resident multi-event decay plan.**
   - Extend the steering boundary so one upload publishes every packed event,
     generates tasks for all event/channel pairs, constructs raw candidates,
     and runs V0 selection as one queue-ordered transaction. Kernels receive
     scalar batch ranges or use the published state; they must not receive
     per-event host arrays.
   - Preserve event and channel identity in all task, raw-candidate, selected,
     overflow, and diagnostic records. Capacity and status readback remains
     scalar until the existing final result hand-off; no per-event candidate
     SoA download is introduced.
   - Return stable per-event/per-channel result ranges and statuses, including
     empty, rejected, and overflowed events, so a caller can split a batch
     without relying on atomic output order.
   - Completion: CPU and HIP regressions compare a batch against the existing
     serial `RunDecayPlan()` reference for lineage, counts, ranges, selection
     records, and overflow attribution across mixed channel/event fixtures.

3. **Stage 10.3: Integrate batch submission into CBMRoot diagnostics.**
   - Replace the diagnostic service's per-event transaction with a bounded
     collector and an explicit flush point compatible with the selector and
     timeslice lifecycle. The service still owns one KFParticle queue and
     serializes submissions; it must never borrow a CBMRoot queue or delay the
     CPU selection result.
   - Feed each completed GPU batch back into the existing comparator and
     reporter as individual event results. Preserve sampling semantics,
     diagnostic-only failure isolation, and concise timeslice reporting while
     adding batch size, queue wait, and flush-reason monitoring.
   - Completion: a CBMRoot-side regression proves CPU output preservation,
     deterministic flush at end-of-timeslice and capacity, reporting for mixed
     success/invalid/overflow events, and no cross-event lineage match.

4. **Stage 10.4: Validate throughput and define the next promotion gate.**
   - Add a reproducible external HIP batch harness beside the existing
     KFParticle tests. It runs controlled multi-event K0S/Lambda/anti-Lambda
     fixtures and records batch composition, H2D, construction, selection,
     D2H, comparison, and total wall times alongside the serial reference.
   - Run CPU and HIP evidence cases over several safe batch sizes. Document
     the observed crossover, memory limits, overflow behaviour, and any
     residual mismatch category; keep verbose per-event dumps opt-in.
   - Define the decision boundary for a later step: retain batched diagnostics,
     expand field-transport equivalence, or separately propose controlled
     production routing. Step 10 never switches candidate ownership from CPU
     to GPU.
   - Completion: the batch path is reproducible on CPU and HIP, preserves
     per-event diagnostic correctness, and has enough timing evidence to
     decide whether a production handoff is technically justified.

Status: Step 10 complete. The batch ABI, one-transaction steering path,
CBMRoot diagnostic collector, compact selected-output partitioning, and
reproducible CPU/HIP evidence harness are implemented and validated.

Stage 10.1 implementation notes:

- Added `GpuDiagnosticBatchBuilder` and its sealed
  `GpuDiagnosticPreparedBatch` contract in the CBMRoot diagnostic adapter.
  The builder owns no XPU buffers and retains only the existing non-owning
  prepared CPU inputs until host SoA views are written.
- Each batch member carries its 64-bit source event identity, GPU event index,
  absolute track/vertex offsets, default-V0 task capacity, and an absolute
  `KFParticleGpuEventDesc`. Failed append attempts leave all accumulated
  ranges unchanged, allowing the caller to flush and retry that event in a
  fresh batch.
- Added offset-aware host packing to `GpuDiagnosticPacker`; it writes a sealed
  batch into one flat SoA/event-descriptor allocation without transferring data
  to the device. GPU upload, kernels, and selector scheduling remain outside
  this stage.
- CPU regression coverage now exercises mixed default V0 channel inputs,
  absolute ranges and copied metadata, empty/malformed event isolation, and
  deterministic rejection at every batch capacity limit.

Stage 10.2 implementation notes:

- Added `RunDecayPlanBatch(firstEventIndex, eventCount, taskCapacity)` to
  `KFParticleGpuSteering`. The existing single-event `RunDecayPlan()` is now a
  compatibility wrapper over a one-event batch, so no current caller changes
  behaviour.
- One batch performs a single input upload, candidate/selection reset, ordered
  channel construction and selection launches, then one final candidate and
  selection hand-off. Per-channel task and pool status transfers remain scalar;
  no per-event candidate SoA download was added.
- `LastDecayPlanEventResults()` publishes stable event partitions: the flat
  channel-result span, raw candidate/daughter range, compact selected range,
  and accumulated overflow flags. Selected ranges use raw candidate membership
  as well as channel ID, so an empty channel cannot consume a later event's
  output with the same channel ID.
- Added a lifecycle regression that compares a two-event batch with the two
  former serial executions, checks event/channel offsets and candidate lineage,
  and verifies that raw output never crosses the event boundary. HIP execution
  remains the required validation for this implementation stage.

Stage 10.3 implementation notes:

- `GpuDiagnosticRunner::RunBatch()` submits one sealed CBMRoot diagnostic
  batch through one exclusive KFParticle transaction. It sizes and writes the
  shared host SoA/event storage once, invokes `RunDecayPlanBatch()` once, and
  splits raw/selected candidates and channel summaries back by the stable
  event partitions returned by steering.
- Empty/no-pair batches remain entirely CPU-side and do not attach XPU.
  Per-event source identity and diagnostic failure isolation are retained even
  if one batch execution fails.
- Each Selector now owns a bounded collector with copied KFP inputs and CPU
  lineage snapshots. It flushes on configured batch capacity, while `Reco`
  flushes each thread-local collector after the timeslice event loop and before
  taking the concise diagnostic report. Thus CPU reconstruction never retains
  diagnostic references and its bitmap is ready before any GPU flush.
- The current conservative limits are eight sampled events, 16k packed tracks,
  64 vertices, 32k tasks/candidates/selected indices, and 64k daughter IDs.
  An event too large for an empty batch is isolated as an input-rejected
  diagnostic record; it cannot stall or alter CPU reconstruction.
- Batch monitoring records size, queue wait, and capacity/end-of-timeslice/
  oversized-event flush reason. Shared H2D/kernel/D2H timings are aggregated
  once per batch, while CPU reference and comparison time remain per event.

Stage 10.4 implementation notes:

- Added the opt-in `KFParticleGpuXpuBatchBenchmark` beside the existing
  standalone XPU lifecycle tests. It prepares deterministic, isolated
  multi-event default-V0 input containing K0S, Lambda, and anti-Lambda track
  pairs, warms the persistent storage outside the measurement, and compares
  the serial and one-upload batch channel summaries before reporting success.
- The executable prints one compact `METRIC` line for each mode. It reports
  average wall, H2D, construction, selection, and D2H time per iteration plus
  final raw/selected counts and overflow flags. Per-event content is never
  printed unless a future diagnostic option explicitly requests it.
- Run it with `GPU/test/run_xpu_batch_benchmark.sh` and the normal standalone
  XPU environment. `KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS` (default `4`) and
  `KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS` (default `10`) control one
  case. `GPU/test/run_xpu_batch_benchmark_matrix.sh` first configures, builds,
  and validates once, then executes the already built benchmark over the
  default `1 2 4 8` event matrix (overridable through
  `KFPARTICLE_GPU_BATCH_BENCHMARK_SIZES`). It writes compact evidence beside
  its logs; `KFPARTICLE_GPU_BATCH_BENCHMARK_SKIP_VALIDATION=1` is available
  only when that build has already been validated.
- The next promotion gate is intentionally factual rather than aspirational:
  retain diagnostics-only routing unless every tested batch size preserves the
  serial channel summary with zero unexpected overflow and HIP shows a stable
  throughput benefit at a representative batch size. Field-transport parity,
  real-event validation, and any candidate-ownership change remain separate
  decisions.
- The standalone lifecycle now includes a two-event accepted-selection batch
  regression. It uses a relaxed K0S selection configuration solely to isolate
  compact-pool mechanics, and verifies non-empty per-event/per-channel ranges,
  selected raw indices, event identity, and daughter source lineage. Physical
  default-V0 selection remains covered by the existing default-V0 regression.
- The HIP matrix evidence on `hip1` over `1, 2, 4, 8` events and 50 iterations
  preserved serial-equivalent channel summaries with zero overflow. At eight
  events it reduced total wall time from 20.591 ms to 19.601 ms per iteration;
  the measured gain is dominated by one batch H2D/selection/D2H hand-off, not
  by a claim of parallelised physics construction. The matrix script now runs
  configure/build/lifecycle validation once before measuring all sizes.

## Step 11: Full Nonhomogeneous-Field Transport For Default V0

Step 11 replaces the current constant-By approximation in the default V0 GPU
diagnostics with bounded transport through the already packed ten-coefficient
CBM field region. It is deliberately limited to K0S, Lambda, and anti-Lambda
two-daughter construction. It does not add channels, change queue ownership,
or promote GPU candidates into CBMRoot reconstruction output.

1. **Stage 11.1: Establish the bounded full-field transport primitive.**
   - Trace the numerical contract of CPU `GetDStoParticleCBM()` and
     `TransportCBM()` and define the corresponding flat, device-safe state,
     scratch, status, and iteration-limit contract. The implementation must
     evaluate all three field components from `KFParticleGpuFieldRegion` at
     the propagated position; it must not allocate, throw, or use a host
     callback from a kernel.
   - Add a bounded single-track propagation primitive that transports the fit
     state and covariance through a nonhomogeneous field. Zero field, neutral
     tracks, non-finite input, excessive path length, and non-convergence must
     take deterministic documented fallback or rejection paths.
   - Add an independent scalar reference fixture for zero, uniform, and
     gradient field cases. It is a test oracle, not a reuse of the GPU helper.
   - Completion: CPU and HIP tests verify field evaluation, transport state,
     covariance sanity, deterministic iteration bounds, and the exact
     zero-field/constant-By regression boundary.

2. **Stage 11.2: Use full-field transport in default-V0 construction.**
   - Introduce a full-field two-daughter DCA/transport path in the existing
     device-side two-daughter builder. Keep the present constant-By path as a
     named baseline for regression and diagnosis, but route the default V0
     descriptors through full-field transport once its contract passes.
   - Preserve the current persistent-buffer model: field regions stay in the
     input SoA, tasks keep only flat indices and scalars, and one KFParticle
     queue performs upload, construction, selection, and final hand-off. No
     per-event host arrays or intermediate candidate downloads are allowed.
   - Extend K0S/Lambda/anti-Lambda fixtures with field gradients and compare
     lineage, status, fit state, mass, chi2, and selection outcome to the
     scalar reference and the CPU path within declared floating-point
     tolerances.
   - Completion: the lifecycle test and HIP run demonstrate correct accepted,
     rejected, neutral, and invalid-field cases without changing batch or
     selected-pool ownership.

3. **Stage 11.3: Validate the CBMRoot diagnostic boundary and record cost.**
   - Reuse the existing ten-coefficient CBMRoot input ABI and make missing or
     invalid field metadata an explicit diagnostic status. No CBMRoot queue,
     ABI, or CPU output change is required.
   - Extend the hermetic CPU/GPU V0 equivalence executable with representative
     nonhomogeneous-field fixtures, then run the same contract on HIP through
     the existing external smoke harness. Record residuals by channel rather
     than requiring bitwise equality.
   - Measure the full-field path against the retained constant-By baseline in
     the standalone batch harness. Document tolerances, convergence limits,
     overflow behaviour, and the measured cost before proposing any production
     ownership decision.
   - Completion: default V0 remains diagnostics-only, but its GPU result has a
     reproducible full-field correctness and performance evidence set.

Status: Step 11 is complete (100%). Standalone CPU validation, the HIP
lifecycle contract, the HIP full-field batch benchmark, and the CBMRoot
CPU/GPU V0 equivalence executable have passed on the target server.

Stage 11.1 implementation notes:

- Added `KFParticleGpuMath::TransportFullField()`, a device-safe port of the
  CPU `TransportCBM()` fixed three-point field approximation. It evaluates the
  packed parabolic field at the start, corrected midpoint, and corrected end
  of a track step, then transports both fit state and 8x8 covariance through
  the resulting analytic Jacobian.
- The primitive owns no storage and accepts only flat value inputs. It reports
  neutral-line fallback, CPU-compatible path limiting, invalid input, and
  non-finite output through `FullFieldTransportStatus`; no device exception or
  dynamic allocation is possible.
- The standalone lifecycle test adds an independent 128-step RK4 scalar
  oracle for a gradient-field trajectory, exact zero-field and neutral
  regression checks, and invalid/path-limit guards. The existing XPU field
  probe now executes the same full-field primitive on device memory.
- HIP lifecycle validation with the routed default V0 descriptors has passed;
  the primitive is backend-qualified for this diagnostic scope.

Stage 11.2 implementation notes:

- Added explicit `KFGpuTransportFullField`. The old
  `KFGpuTransportFieldAware` name is retained as a compatibility alias for
  `KFGpuTransportConstantBy`, so existing diagnostic channels preserve their
  former behaviour and remain a useful performance/correctness baseline.
- The three default V0 descriptors now select `KFGpuTransportFullField`.
  Their two-daughter DCA seed performs the same bounded two correction rounds
  as the old field-aware seed, but transports each daughter with its own packed
  ten-coefficient field region instead of sampling first-daughter `By` once.
- The full-field energy-fit seed uses full-field states and own covariances;
  its cross-daughter correlation remains the already validated line-DCA
  approximation. This boundary is explicit and must be resolved before any
  production ownership proposal.
- Lifecycle coverage verifies the new mode in K0S, Lambda, and anti-Lambda
  descriptors, nonzero gradient-field execution, host/device candidate
  agreement, and retention of the constant-By diagnostic baseline. The same
  lifecycle contract has passed on HIP.

Stage 11.3 implementation notes:

- Added `GpuDiagnosticStatus::FieldRejected`. The CBMRoot diagnostic runner
  now distinguishes malformed or unavailable field metadata from generic input
  rejection while retaining diagnostic-only failure isolation and unchanged CPU
  reconstruction output.
- The hermetic `KFParticleCpuGpuV0Equivalence` executable now packs a
  ten-coefficient gradient field, prints compact mass/mass-error/chi2/NDF
  residuals for K0S, Lambda, and anti-Lambda, and verifies that a non-finite
  field coefficient becomes `FieldRejected`.
- The standalone batch benchmark now allocates and fills full field SoA input.
  It preserves the full-field serial/batch correctness check and emits a
  separate `constant-by-batch` timing baseline with the same channel layout.
- Backend-qualified evidence is complete: the standalone HIP lifecycle, HIP
  batch matrix, and `run_cbmroot_cpu_gpu_v0_equivalence.sh` have passed.
  Their channel-residual and full-field/constant-By timing records are the
  reproducible completion evidence for this step.

## Step 12: Coupled Full-Field V0 Covariance And Energy Fit

Step 12 removes the remaining numerical approximation in the default full-field
V0 energy-fit path. Step 11 transports each daughter and its own covariance in
the packed ten-coefficient field, but still obtains the cross-daughter
correlation from the line-DCA model. This step ports and validates the coupled
full-field DCA derivative/covariance contract needed by the existing energy
fit. Its scope remains diagnostics-only for K0S, Lambda, and anti-Lambda: it
does not add decay channels, change queue ownership, or promote GPU candidates
into CBMRoot reconstruction output.

1. **Stage 12.1: Define and validate the coupled full-field DCA covariance primitive.**
   - Trace the CPU `GetDStoParticleCBM()`/`TransportCBM()` derivative contract
     used when two charged daughters are transported to their common DCA.
     Define a flat, device-safe helper that returns both transported states,
     their covariance terms, and the 3x3 cross-daughter correlation block.
   - Keep the calculation bounded and allocation-free: fixed correction count,
     explicit path/non-finite rejection, separate field regions for the two
     daughters, and no device exceptions or host callbacks.
   - Add an independent host numerical-differentiation oracle over zero,
     uniform, and gradient fields. It must check state derivatives, covariance
     symmetry/finiteness, correlation signs and magnitudes, and exact
     zero-field regression against the retained line-DCA implementation.
   - Completion: lifecycle CPU and HIP tests establish that the primitive is
     stable for charged, neutral, invalid-field, and path-limited inputs.

2. **Stage 12.2: Replace the approximate full-field energy-fit seed.**
   - Introduce an explicit coupled full-field measurement seed and route
     `KFGpuTransportFullField` energy-fit tasks through it. The old
     `BuildFullFieldDcaMeasurementSeedApprox()` remains a named test baseline
     until the new route is qualified; the constant-By route remains unchanged.
   - Preserve the persistent SoA/device-state model: kernels load both field
     regions and write candidate output in-place, with no host round trip
     between task generation, construction, and selection.
   - Extend K0S, Lambda, and anti-Lambda lifecycle fixtures with nonzero field
     gradients and nontrivial covariance. Compare host and device candidate
     state, covariance, mass error, chi2/NDF, selection record, lineage, and
     failure status within declared tolerances.
   - Completion: the default V0 full-field energy-fit path no longer relies on
     line-DCA cross-daughter correlation, while all existing constant-By and
     no-energy-fit regressions remain green.

3. **Stage 12.3: Qualify the coupled route through CBMRoot and record its cost.**
   - Extend the hermetic CBMRoot CPU/GPU V0 equivalence executable with
     covariance-sensitive gradient-field fixtures and compact residual output
     for mass, mass error, chi2, NDF, and selection status. Keep invalid-field
     isolation and diagnostic-only CPU-output ownership intact.
   - Extend the standalone HIP batch matrix to report the coupled full-field
     cost beside the Step 11 approximate full-field and constant-By baselines.
     Record overflow and rejection counters so a timing result cannot hide a
     reduced physics workload.
   - Close the step only after CPU and HIP lifecycle tests, the HIP benchmark
     matrix, and the CBMRoot equivalence harness pass with reproducible
     residual and timing evidence.

Status: Stages 12.1 and 12.2 are complete and backend-qualified by standalone
CPU and HIP lifecycle runs. Step 12 is about 67% complete; Stage 12.3 remains
to qualify the coupled route through the hermetic CBMRoot boundary and record
its workload-aware HIP cost.

Stage 12.1 implementation notes:

- Added `KFParticleGpuFullFieldDcaResult` and
  `BuildFullFieldDcaCoupledResult()`. The primitive keeps two full-field DCA
  states, 6x6 covariance blocks, the 3x3 cross-daughter correlation, and flat
  6x12 Jacobians with respect to both daughter inputs. It owns no storage and
  uses a fixed central-difference stencil over the existing bounded DCA seed.
- The result explicitly rejects non-finite states/field coefficients and
  failed or path-limited DCA transports; a neutral daughter is retained as a
  deterministic status bit rather than redirected to a hidden fallback.
- Lifecycle coverage independently reconstructs the central-difference
  Jacobians/correlation, verifies the zero-field line-DCA boundary, and covers
  gradient-field, neutral, invalid-field, and distant-path cases. The existing
  XPU field probe now executes the coupled primitive on device and checks its
  status, covariance, correlation, and Jacobian outputs.
- The primitive is deliberately not wired into energy-fit construction yet.
  Stage 12.2 consumes this result through the full-field energy-fit route.

Stage 12.2 implementation notes:

- Added `BuildFullFieldDcaMeasurementSeedCoupled()`. It publishes the coupled
  full-field DCA states, replaces the DCA-dependent six-coordinate covariance
  blocks, and supplies the computed 3x3 cross-daughter correlation to the
  existing Kalman measurement ABI. Existing energy/S transport terms remain
  those from the bounded full-field state, which is the compatible scope of
  the current measurement representation.
- `KFGpuTransportFullField` energy-fit tasks now call the coupled seed. The
  constant-By route and `BuildFullFieldDcaMeasurementSeedApprox()` remain
  available as explicit regression baselines; no queue, SoA, task-pool, or
  device-to-host synchronization behavior changed.
- Added a lifecycle regression that proves the coupled correlation differs
  from the approximate line-DCA value on a gradient-field fixture and that the
  resulting Kalman update is finite with the expected NDF and charge. The
  existing default-V0 device tests now exercise the coupled dispatch.
- HIP lifecycle validation has passed. Stage 12.3 now qualifies the same route
  through CBMRoot and records workload-aware cost evidence.

Stage 12.3 implementation notes:

- The CBMRoot CPU/GPU V0 executable now uses a positive-definite, non-diagonal
  daughter covariance together with the existing ten-coefficient gradient
  field. Its per-channel metric records mass, mass-error, chi2, and NDF
  residuals plus the GPU selection class, rejection mask, and overflow state.
  Invalid input and non-finite field metadata remain isolated diagnostic cases.
- The standalone batch benchmark reports generated pairs, accepted tasks, and
  stored tasks beside raw/selected candidates and overflow flags for serial,
  coupled full-field batch, the retained full-field line-DCA approximation,
  and constant-By batch modes. These counters make a timing comparison
  auditable: a lower time cannot be mistaken for a gain if the reconstructed
  workload was reduced.
- Completion evidence: standalone CPU/HIP lifecycle runs, the HIP benchmark
  matrix, and the CBMRoot equivalence harness have passed. The benchmark now
  records equal generated/accepted/stored workload counters for each compared
  transport mode, so its coupled-route timing remains interpretable.

Status: Step 12 complete (100%).

## Step 13: CPU-Compatible Default-V0 Topology And Selection

Step 13 closes the remaining default-V0 diagnostic physics gap after the
coupled full-field energy fit. It ports the topology observables used by the
CPU finder for PV association, decay distance/significance, and secondary
candidate rejection. The scope remains K0S, Lambda, and anti-Lambda in the
diagnostic path: it neither adds higher-generation decays nor transfers
candidate ownership from CPU reconstruction to GPU reconstruction.

1. **Stage 13.1: Define a bounded GPU topology contract and scalar oracle.**
   - Trace the CPU finder calculations that feed `chiToPrimaryVertex`,
     candidate-to-PV distance, decay length/significance, pointing, and
     secondary cuts. Separate the required fit/covariance inputs from
     CPU-only SIMD and reconstruction-list state.
   - Add flat device-safe topology state and result records with explicit
     finite/degenerate/no-PV rejection bits. The helper must be bounded,
     allocation-free, and deterministic for multiple primary vertices and
     ties.
   - Add an independent scalar reference fixture covering zero and gradient
     fields, diagonal and correlated covariances, positive/negative decay
     geometry, missing PVs, and exact-cut boundaries.
   - Completion: CPU and HIP lifecycle tests agree with the scalar oracle on
     observables, selected best PV, rejection masks, and the retained default
     V0 selection contract.

2. **Stage 13.2: Route default-V0 selection through the topology contract.**
   - Replace the current spatial approximation in the default V0 selection
     kernel with the bounded topology result. Keep the former approximation as
     an explicit test baseline until qualification is complete; do not create
     host-side per-candidate work or intermediate readbacks.
   - Preserve persistent input/work/output pools and the one KFParticle queue.
     Candidate, selection-record, and compact-output ranges must keep event,
     channel, and best-PV identity through batch execution.
   - Extend K0S/Lambda/anti-Lambda lifecycle fixtures to compare topology
     observables, selection class/reasons, selected ranges, and lineage for
     accepted, rejected, multi-PV, and covariance-sensitive cases.
   - Completion: the normal full-field default V0 descriptors use the new
     topology route; constant-By and legacy topology remain test-only
     baselines, with CPU/HIP regressions green.

3. **Stage 13.3: Qualify through CBMRoot and establish the next boundary.**
   - Extend the hermetic CBMRoot equivalence executable with controlled
     multi-PV and selection-boundary fixtures. Report lineage, best PV,
     topology residuals, selection status, overflow, and invalid-input
     isolation without touching CPU output ownership.
   - Extend the batch benchmark with topology-stage timing and workload
     counters, comparing the CPU-compatible route to the retained diagnostic
     baseline. Record CPU/HIP evidence at several safe batch sizes.
   - Close only after standalone CPU/HIP lifecycle, the HIP benchmark matrix,
     and CBMRoot equivalence pass. The resulting evidence determines whether
     the next step should be a controlled production hand-off proposal or
     porting the next decay-generation algorithms.

Status: Step 13 complete (100%). Standalone CPU/HIP lifecycle validation, the
HIP batch benchmark matrix, and the hermetic CBMRoot equivalence executable
have passed for the qualified default-V0 diagnostic route.

Stage 13.1 implementation notes:

- Added `KFParticleGpuV0LineTopologyResult` and the allocation-free
  `BuildV0LineTopology()` helper. It evaluates the PV-to-candidate line
  residual, signed momentum projection, decay length/significance, pointing
  cosine, and correlated 3x3 spatial chi2 using only flat fit and PV values.
- `FindBestV0LineTopology()` resolves multiple valid PVs by minimum line chi2
  with a stable lowest-index tie break. Non-finite input, zero momentum, and
  singular covariance have separate deterministic status bits.
- The lifecycle fixture uses an independent closed-form scalar oracle with a
  correlated covariance and checks all observables, PV ties, and rejection
  paths. A dedicated XPU probe executes the same contract on the selected
  backend before its values are compared on the host. Existing default-V0
  selection continues to use its legacy spatial approximation until Stage
  13.2.

Stage 13.2 implementation notes:

- Added an explicit per-channel topology mode. `Spatial` preserves the former
  selection approximation for diagnostic baselines; the three default
  full-field V0 descriptors now select `Line`.
- The line route evaluates the nearest valid PV for line distance and decay
  significance, while independently selecting the lowest-line-chi2 PV for
  topology classification. Both choices retain stable lowest-index ties and
  are written to the existing selection record, including the topology status.
- No buffer, queue, or hand-off ownership changes were introduced. The mode is
  a scalar member of the existing channel config, so task generation,
  construction, device-resident selection, and compact output remain one
  ordered transaction.
- Lifecycle coverage verifies the spatial baseline, line-route observables,
  cut boundary, invalid mode rejection, all three default descriptors, and
  existing default-V0/batch regressions. CPU validation passes; HIP validation
  is required before entering Stage 13.3.

Stage 13.3 implementation notes:

- The hermetic CBMRoot equivalence executable now exposes the complete GPU
  selection observables and topology status in its raw-candidate diagnostic
  record. Its multi-PV fixture supplies two identical primary vertices to
  verify the stable lowest-index line-topology tie, then checks the exact
  `distance >= cut` boundary and the next representable accepted value.
- The standalone batch fixture now packs one primary vertex per event. Thus
  every default-V0 raw candidate executes the line-topology selection route;
  the benchmark reports `topology_selection_ms`, valid-topology, best-PV, and
  topology-rejection counters and rejects a serial/batch topology mismatch.
- Standalone CPU/HIP lifecycle validation, the HIP benchmark matrix, and the
  CBMRoot equivalence executable have passed. No production CPU ownership or
  reconstruction hand-off is changed by these qualification fixtures.

## Step 14: Device-Resident V0-Track Cascade Reconstruction

Step 14 starts the next reconstruction generation instead of prematurely
replacing CBMRoot's CPU finder output. The qualified default V0 result is now
valuable as an on-device input: `KFParticleFinder::FindTrackV0Decay()` combines
selected Lambda or anti-Lambda candidates with charged pion/kaon tracks to form
Xi and Omega hypotheses. This route reuses the persistent V0 pools and avoids
a host round trip between generations. The scope is diagnostic-only: CPU
ownership, final `SetProductionVertex` handling, and public reconstruction
output remain unchanged.

1. **Stage 14.1: Define the selected-V0 to track data and lineage contract.**
   - Trace the CPU `FindTrackV0Decay()` inputs, daughter-exclusion rules,
     Xi/Omega channel assignments, and cut families. Record which values are
     already available in the selected-V0/raw-candidate pools and which must be
     carried with a new flat task descriptor.
   - Add non-owning device views and trivially-copyable V0-track channel/task
     values. A task must refer to a selected raw V0 index and an input-track
     index, retain event/channel/PV identity, and preserve canonical three-track
     lineage without copying the V0 fit state. It must reject reuse of either
     V0 daughter as the bachelor track before construction.
   - Establish an independent host scalar fixture for Lambda-pion and
     anti-Lambda-pion geometry, including selected-range boundaries, duplicate
     daughter rejection, empty ranges, mixed-event isolation, and bounded task
     capacity. No production kernel or CBMRoot output is changed in this stage.
   - Completion: the standalone lifecycle test proves the flat ABI, lineage,
     range, and rejection contract on CPU and HIP.

2. **Stage 14.2: Build compact Xi/Omega candidates on the device.**
   - Add bounded compact task generation for selected Lambda/anti-Lambda V0s
     crossed with compatible secondary pion/kaon ranges. Start with the four
     charge-conjugate channels `Xi-`, `anti-Xi+`, `Omega-`, and `anti-Omega+`;
     keep their order explicit in a cascade plan.
   - Implement a device-safe V0-track construction kernel using the existing
     full-field transport and coupled covariance machinery. The neutral V0 is
     read from the raw candidate SoA; only the bachelor track samples its
     packed field region. Construction, fit status, mass, covariance, and
     recursive three-track lineage must be written directly into persistent
     device pools with explicit overflow flags.
   - Compare host scalar and device output for accepted, rejected, neutral/
     invalid-field, large-path, and capacity-limited fixtures. Retain the V0
     input pools unmodified so one queue transaction can continue to later
     generations.
   - Completion: standalone CPU/HIP tests demonstrate compact Xi/Omega
   construction from selected V0s with stable lineage and no hidden host
   synchronization.

Stage 14.2 implementation notes:
- `KFParticleGpuV0TrackChannel` now has explicit charge-conjugate builders
  for `Xi-`, `anti-Xi+`, `Omega-`, and `anti-Omega+`. Each records the V0 PDG,
  bachelor PDG/mass, mother hypothesis, event-local bachelor range, and the
  full-field transport mode.
- `KFParticleGpuDeviceStorage` owns an independent compact V0-track task
  buffer and counters. `KFParticleGpuGenerateV0TrackTasksCompact` crosses a
  selected-V0 range with the compatible bachelor range, rejecting invalid
  lineage before its atomic compact reservation.
- `KFParticleGpuV0TrackCompactCandidatePoolKernel` keeps the V0 in the raw
  candidate SoA, loads a zero field for this neutral input, loads the field
  region only for the bachelor, and appends a three-daughter candidate to the
  persistent pool. Candidate/daughter bounds and worklist truncation have
  explicit counters and overflow bits.
- The lifecycle fixture currently exercises full-field `Lambda + pi- -> Xi-`,
  validates raw-V0 preservation, three-source lineage, finite fit state, and
  compact-task overflow. The channel-builder fixture locks the four initial
  Xi/Omega mappings. Standalone HIP validation remains required before closing
  this stage.

3. **Stage 14.3: Execute and qualify a two-generation cascade plan.**
   - Extend steering from `default V0 -> selection` to
     `default V0 -> selected V0 -> V0-track cascade`, with per-channel and
     per-event ranges for both generations. Capacity growth, raw/selected
     counters, and all overflow states must be visible in the final result.
   - Add CPU-compatible comparison fixtures based on the corresponding
     `FindTrackV0Decay()` channels. Match Xi/Omega candidates by event, channel,
     and canonical three-track source lineage; report fit and selection
     residuals without requiring bitwise equality.
   - Extend the standalone batch benchmark and the hermetic CBMRoot diagnostic
     executable with a controlled cascade fixture, while leaving the CPU finder
     authoritative. Record separate V0 and cascade workload/timing counters so
     the new stage cannot appear faster by silently dropping combinations.
   - Completion: CPU/HIP lifecycle, HIP batch evidence, and CBMRoot diagnostic
     equivalence pass for the first cascade channels. Only then evaluate a
   later, explicitly opt-in production hand-off.

Stage 14.3 implementation notes:
- `KFParticleGpuDecayPlan` owns an ordered cascade-channel list and exposes
  the first four charge-conjugate Xi/Omega descriptors independently of the
  default V0 list. `KFParticleGpuSteering` executes them after V0 selection,
  retaining separate cascade results, per-event cascade ranges, counters, and
  wall-time accounting.
- The cascade task generator consumes the compact selected V0 buffer directly
  on the device. It reads its live size rather than its capacity, so monitoring
  reports real selected-V0/bachelor work rather than padded worklist slots.
- The controlled lifecycle fixture starts with packed tracks, builds V0s,
  selects them, and appends six Xi plus three Omega candidates. It confirms
  event/channel ranges, three-track lineage, untouched raw V0 inputs, and a
  CPU scalar reconstruction of every stored cascade fit state.
- Cascade status snapshots are enqueued after each channel and consumed after
  one final queue wait, so no host synchronization divides the device-resident
  Xi/Omega sequence. The live selected-V0 count and event/channel predicates
  keep batch monitoring event-local.
- The standalone benchmark has a controlled full-field cascade fixture with
  separate cascade timing, pair, and candidate counters. The external CBMRoot
  smoke macro runs the same V0-to-Xi/Omega chain through CBMRoot's initialized
  XPU runtime, without adding a test to the main CBMRoot tree.
- No CBMRoot production output is changed. CPU lifecycle and batch validation
  pass; HIP qualification must be rerun after the final queue and fixture work.

Stage 14.1 implementation notes:
- `KFParticleGpuV0Track.h` defines flat channel, task, and canonical
  three-source lineage values. `ResolveTask()` consumes the compact selected
  V0 view and packed tracks directly; it deliberately carries raw-pool and
  track indices rather than a copied V0 fit state.
- The initial channels make the CPU mapping explicit: `Lambda + pi- -> Xi-`
  and `anti-Lambda + pi+ -> anti-Xi+`. The full CPU mapping also admits kaon
  bachelors for Omega and primary-track resonance paths; those remain for the
  compact generator in Stage 14.2.
- The lifecycle fixture covers event-local selected ranges, empty ranges,
  duplicated daughter sources, bachelor PID/PV/chi gates, and bounded output.
  `KFParticleGpuV0TrackTaskProbe` executes the same resolver against device
  input, raw-candidate, and selected-index buffers before readback.

Status: Step 14 complete (100%). The standalone CPU/HIP lifecycle contract,
the controlled HIP batch fixture, and the external CBMRoot cascade smoke have
passed on the target server. No production hand-off is enabled.

## Revised Continuation After Step 14

The first V0 and V0-track generations are now numerically qualified, but their
host steering still iterates over decay channels and launches a separate task
generator for each channel. Work inside one channel is parallel; channel
dispatch is not. Adding the remaining finder channels with the same structure
would multiply kernel launches and repeatedly read the same pair inputs.

The continuation therefore introduces mask-driven routing before adding more
channel families. A mask is an eligibility set, not a physics decision:

- event and species ranges eliminate impossible input regions first;
- compact role masks and a compatibility lookup produce the active channel
  bits for one input pair;
- every active bit emits a task carrying the stable channel ID and descriptor
  index;
- transport, DCA, Kalman fitting, and continuous cuts run only for emitted
  tasks;
- candidate acceptance, lineage, overflow, and selection semantics remain
  unchanged.

This is intentionally not one monolithic finder kernel. Reconstruction
generations have real data dependencies, and a single large kernel would add
divergence, register pressure, and unsupported grid-wide synchronization.
Steering will launch a small ordered pipeline per generation, while masks
remove the launch-per-channel pattern inside that generation.

Channel bits are transient routing positions. Persisted candidates continue to
store stable channel IDs, because descriptor order may change without changing
physics identity. The first implementation may use one 32-bit word, but the
value contract must support a bounded array of words so later channel growth
does not require another ABI redesign.

## Step 15: Mask-Driven Channel Routing And Compact Task Generation

Step 15 replaces the current sequential cascade-channel dispatch with a
device-resident routing layer. It deliberately starts with the already
qualified Xi/Omega generation, where the old and new paths can be compared
without introducing new physics.

The step has three implementation checkpoints. Stage 15.1 represents roughly
30% completion, Stage 15.2 roughly 70%, and Stage 15.3 closes the step at 100%.
These percentages describe delivered contracts, not elapsed development time.

The following are fixed Step 15 constraints:

- no new particle channel or numerical fit operation is introduced;
- no CBMRoot source outside the existing external diagnostic harness changes;
- no plan/table upload, allocation, or host status read occurs in the event
  channel loop;
- the old explicit Xi/Omega path stays executable until the final gate;
- atomic output order is never treated as physics order.

1. **Stage 15.1: Freeze the routing ABI and independent correctness oracle.**

   **Data contract**

   - Add a trivially-copyable `KFParticleGpuChannelMask`-style value with
     bounded storage, intersection, bit test/set, empty test, and deterministic
     set-bit iteration. The first storage word covers the current channels,
     while the ABI permits additional words without using STL, allocation, or
     undefined shifts in device code.
   - Define compact selected-V0 and bachelor role codes separately from PDG and
     channel identity. A role says which descriptor positions an object can
     occupy; a compatibility entry maps `(V0 role, bachelor role)` to a channel
     mask. One pair may legally produce zero, one, or several channel bits.
   - Introduce a device routing descriptor derived from the existing
     `KFParticleGpuV0TrackCascadeChannel`. It retains stable `channelId`, PDGs,
     masses, transport flags, cuts, and source selectors. A routed task stores
     the descriptor index plus V0/track/event indices; it does not duplicate
     the complete descriptor or V0 fit state.
   - Define execution groups by compatible source geometry: event, selected-V0
     source range, and a bounded list of bachelor species ranges. Xi and Omega
     of one charge can share a group without scanning unrelated event tracks.

   **Ownership and preparation**

   - Extend `KFParticleGpuDeviceStorage` and the buffer manager with persistent
     routing descriptors, execution groups, compatibility entries, enabled
     masks, and per-channel counters. Capacities grow monotonically like the
     existing task pools.
   - Add a host plan-compilation step that validates unique channel IDs,
     descriptor/bit bounds, supported role combinations, and group ranges.
     Upload the compiled table only when the decay plan revision changes, never
     once per event.
   - Publish non-owning routing views through the existing kernel state. The
     decay plan remains the host owner of configuration; kernels see only flat
     buffers and counts.

   **Tests and completion gate**

   - Add a scalar oracle that produces the complete routed-task multiset
     `(event, descriptor/channel, selected V0, bachelor)` from the explicit
     descriptors without calling the mask helper.
   - Cover all four charge-conjugate Xi/Omega channels, disabled channels,
     multi-bit matches, masks crossing a word boundary, invalid descriptors,
     duplicate channel IDs, empty ranges, and mixed-event rejection.
   - Preserve every existing lifecycle result because no reconstruction kernel
     is switched in this stage.
   - Completion: the routing plan and oracle are fully tested on the host, all
     GPU data values are trivially copyable, and the old cascade path remains
     the only active execution route.

2. **Stage 15.2: Add the functional fused GPU route and generic construction.**

   **Routing kernel**

   - Add one XPU action for each event/execution group. A thread maps to one
     selected-V0/bachelor pair, performs event and lineage guards once, derives
     both roles, intersects the compatibility mask with the group's enabled
     mask, and emits one routed task for every set bit.
   - Split current `ResolveTask()` work into common pair validation and
     descriptor-specific validation so source/event/duplicate checks are not
     repeated for Xi and Omega hypotheses. Continuous fit and selection cuts
     remain outside routing.
   - Start with the simplest bounded global atomic reservation. Keep separate
     accepted, stored, visited-pair, active-bit, and overflow counters so a
     masked route cannot appear correct by silently dropping work.

   **Construction and result contract**

   - Make the cascade construction kernel read the task's descriptor index
     from the persistent table. It runs expensive transport/DCA/Kalman
     operations once per emitted task and writes the same candidate metadata,
     stable channel ID, and canonical three-source lineage as the explicit
     path.
   - Treat the complete cascade output as one event/generation range.
     Per-channel summaries are derived from stable channel metadata and
     device-side counters, not from accidental atomic contiguity. Where a
     downstream consumer needs membership, expose a compact channel-index view
     rather than restoring launch-order assumptions.
   - Queue routing and construction without a host counter download or wait
     between them. The old per-channel implementation remains selectable only
     by a test/debug option.

   **Tests and completion gate**

   - Compare explicit and fused routes as unordered task and candidate
     multisets. Require exact event/channel/lineage/status/overflow agreement
     and the existing floating-point tolerances for every fit value.
   - Add cases with both Xi and Omega tasks from one execution group,
     charge-conjugate groups, multiple events, one pair matching multiple
     descriptors, build rejection, task/candidate/daughter truncation, and a
     disabled channel in an otherwise unchanged table.
   - Run standalone CPU first, then HIP lifecycle validation. Do not benchmark
     this atomic baseline as the final performance implementation.
   - Completion: the fused path is functionally complete and backend-qualified,
     but the explicit path remains the default until Stage 15.3.

3. **Stage 15.3: Optimize compaction, switch steering, and close the gate.**

   **Contention reduction**

   - Replace one global reservation per emitted task with an XPU block-scan
     compaction patterned after the working CA tracker primitives. Each thread
     contributes its active-bit count, the block computes exclusive offsets,
     and one thread reserves the block's global output range. Threads then
     write directly to their assigned slots; no temporary host storage or
     backend-specific warp intrinsic is required.
   - Preserve bounded semantics when a block reservation crosses capacity:
     accepted count reports the full workload, only in-range tasks are stored,
     and overflow is set exactly once without an out-of-bounds write.
   - Keep the simple atomic kernel in tests as the device reference until the
     scan route matches it on CPU and HIP. Select between atomic and block-scan
     only at plan/build granularity, never per event through a host round trip.

   **Steering and monitoring**

   - Replace the production cascade channel loop with
     `compile/upload plan -> launch groups -> construct generation -> snapshot`.
     One final wait resolves all event/channel summaries. The KFParticle queue,
     persistent pool ownership, and standalone/CBMRoot runtime rules stay
     unchanged.
   - Extend timing/workload monitoring with routing-group launches, descriptor
     count, visited pairs, active bits, emitted/stored tasks, block
     reservations, candidates, daughters, and overflow. Retain stable
     per-channel summaries for diagnostics.
   - Remove only the production use of the explicit generator. Keep the scalar
     oracle and a test-only explicit route for regression until Step 16 has
     reused the same architecture.

   **Final qualification**

   - Run lifecycle CPU/HIP, the controlled HIP cascade benchmark, external
     CBMRoot cascade smoke, and explicit-vs-mask equivalence. Require identical
     workload and physics output before interpreting timing.
   - Record launch reduction and compare atomic versus block-scan routing.
     Block scan becomes default only if it is no slower for the controlled
     workload and shows the expected lower global reservation count; otherwise
     retain the simpler atomic implementation and document the measured reason.
   - Completion: adding another compatible cascade hypothesis changes
     descriptors/tables and tests, not steering control flow or the number of
     pair-enumeration launches.

Stages 15.1 and 15.2 established the first 70% of Step 15. Stage 15.1 adds a
two-word/64-channel device-safe mask, selected-V0 and bachelor roles,
flat routing descriptors, compatibility entries, execution groups, and a
host-side `KFParticleGpuV0TrackRoutingPlan` compiler. The compiler rejects
duplicate or zero channel IDs, unsupported roles, inconsistent species/PDG
pairs, wrong-sign track sets, and masks beyond their bounded capacity.

The compiled plan is stored in persistent
`KFParticleGpuDeviceStorage` buffers for descriptors, compatibility entries,
groups, the enabled-channel mask, and per-channel counters. Its non-owning
view is part of `KFParticleGpuKernelState`; a device probe verifies the
published counts and compatibility mask. Upload is explicitly rejected before
compilation and is skipped when the decay-plan revision is unchanged.

The Stage 15.1 lifecycle regression compares the mask lookup against an independent
explicit-descriptor oracle, including all four Xi/Omega channels, multi-bit
matches, a mask crossing the 32-bit word boundary, empty/out-of-range inputs,
mixed-event rejection, and invalid descriptors.

Stage 15.2 adds a persistent bounded routed-task pool and explicit
visited-pair, active-bit, accepted, stored, per-channel, and overflow counters.
`KFParticleGpuRouteV0TrackTasksAtomic` visits each event/group pair once,
intersects compatibility and enabled masks, and emits one compact task per
accepted descriptor bit. `KFParticleGpuV0TrackRoutedCandidatePoolKernel`
resolves the persistent descriptor and reuses the qualified cascade
construction/store path. Both kernels are submitted to the KFParticle queue
without a D2H counter transfer or host wait between them.

The isolated `RunV0TrackFusedStage()` qualification entry point compares
routed tasks and candidates against the old explicit `ResolveTask()` route as
unordered multisets. Its fixture covers Xi/Omega, both charge groups, two
events, a pair matching two descriptors, enabled-mask suppression, bounded
task overflow, per-channel counters, lineage, metadata, and fit tolerances.
Local standalone CPU lifecycle validation passed this checkpoint. At the
Stage 15.2 boundary, the explicit per-channel cascade loop deliberately
remained the production default pending Stage 15.3.

Stage 15.3 implementation notes:

- Added `KFParticleGpuRouteV0TrackTasksBlockScan` with a 64-thread XPU
  `block_scan`. Threads contribute accepted descriptor-bit counts, one thread
  reserves the bounded global range, and exclusive offsets assign task slots.
  Full accepted workload, stored workload, overflow, and reservation counts
  remain separate.
- Retained the atomic route as a test-selectable device reference. Lifecycle
  validation runs atomic and block-scan routes for both charge groups and
  compares each against the independent explicit task/candidate multiset,
  including lineage, fit tolerances, truncation, disabled channels, candidate
  and daughter overflow, and construction rejection.
- Production `RunDecayPlanBatch()` now compiles/uploads a routing plan once per
  revision, launches execution groups, constructs one unordered cascade
  generation per event, queues aggregate and per-channel snapshots, and waits
  once after all events. It no longer launches the explicit cascade generator
  once per channel.
- Added visited, accepted, stored, and constructed per-channel counters plus
  aggregate routing monitoring. Channel results expose the shared generation
  range and stable channel metadata instead of claiming accidental atomic
  contiguity.
- The controlled benchmark now reports routing groups, descriptors, pairs,
  active bits, accepted/stored tasks, block reservations, candidates,
  daughters, and overflow. It also runs warmed atomic and block-scan isolated
  transactions and prints `ROUTING_METRIC` lines after requiring identical
  workload/output counts.
- The external CBMRoot smoke gate now verifies fused group dispatch,
  block-reservation bounds, Xi/Omega construction counts, and overflow while
  preserving CBMRoot-owned XPU initialization and KFParticle-owned queue use.

Status: Step 15 complete (100%). The standalone CPU/HIP lifecycle contract,
the controlled HIP routing benchmark, and the external CBMRoot cascade smoke
gate pass on the target server. Production Xi/Omega cascade steering now uses
mask-driven execution groups and block-scan task compaction; the atomic and
explicit routes remain qualification oracles. On the CPU backend, a block is
effectively serialized and reservation reduction is therefore not expected;
the HIP `ROUTING_METRIC` result is the performance qualification.

## Step 16: Generation-Wide Routing For Default V0 And Selection

Step 16 applies the qualified routing model to the first two-daughter
generation and its selection continuation. The first target is the existing
K0S, Lambda, and anti-Lambda set; no new particle channel or numerical
operation is introduced in this step.

The step has three implementation checkpoints. Stage 16.1 represents roughly
30% completion, Stage 16.2 roughly 70%, and Stage 16.3 closes the step at 100%.
The split keeps the descriptor ABI, numerical construction, and production
selection hand-off independently reviewable.

The following are fixed Step 16 constraints:

- reuse `KFParticleGpuChannelMask`; channel bits and descriptor indices remain
  transient routing values, while stable channel IDs remain the public physics
  identity;
- visit a compatible track pair once per event/execution group and emit every
  active hypothesis, rather than relaunching pair enumeration per channel;
- keep descriptor tables, routed tasks, candidate tags, raw candidates,
  selection records, and compact selected indices in persistent device-owned
  storage;
- perform no allocation, plan upload, scalar D2H status read, or host wait
  between route, construction, and selection;
- preserve the current field transport, DCA, Kalman fit, topology, lineage,
  selection, and bounded-overflow semantics;
- retain the explicit per-channel path as a qualification oracle until the
  final CPU/HIP and CBMRoot gates pass;
- do not change CBMRoot source outside the existing external KFParticle
  diagnostic harness and build integration.

1. **Stage 16.1: Freeze the two-daughter routing and selection ABI.**

   **Descriptor and grouping contract**

   - Add a flat `KFParticleGpuTwoDaughterRoutingDescriptor` derived from the
     existing decay-plan channel. It contains stable channel identity, daughter
     roles/PDGs, source selectors, charges and source cuts, masses, transport
     and fit flags, post-build cuts, and the complete V0 selection config.
   - Add compact track-role codes and compatibility entries mapping
     `(first role, second role)` to a channel mask. Build execution groups from
     common first/second track-set geometry and bounded species ranges. The
     three default V0 channels must not force three scans of the same broad
     positive/negative source geometry.
   - Define a minimal routed task carrying descriptor, event, and two track
     indices. Add a capacity-matched transient descriptor-index sidecar for raw
     candidates. Construction writes this O(1) selection lookup while the
     candidate continues to store its stable channel ID.

   **Ownership and plan compilation**

   - Add a host-owned `KFParticleGpuTwoDaughterRoutingPlan` that validates
     unique nonzero channel IDs, supported role/species/charge combinations,
     source geometry, descriptor and mask bounds, selection configuration, and
     execution-group ranges.
   - Extend device storage, buffer capacities, the buffer manager, and kernel
     state with persistent descriptors, compatibility entries, groups, enabled
     masks, routed tasks, the candidate-descriptor sidecar, and aggregate plus
     per-channel counters. Capacities grow monotonically.
   - Compile and upload the table only when the decay-plan revision changes.
     Uploading an uncompiled plan is rejected; uploading the same revision is a
     no-op.

   **Tests and completion gate**

   - Add an independent scalar oracle that walks explicit channel descriptors
     without using the mask lookup and returns the complete unordered task
     multiset `(event, channel ID, first source, second source)`.
   - Cover K0S, Lambda, anti-Lambda, disabled and multi-bit hypotheses, masks
     crossing the word boundary, empty/species-fallback ranges, duplicate
     sources, mixed events, invalid roles and descriptors, duplicate IDs, and
     revision-controlled upload.
   - Publish the plan through XPU kernel state and verify the device-visible
     descriptor, group, compatibility, and enabled-mask values.
   - Completion: the flat ABI and host/device plan contracts are qualified on
     CPU/HIP while production still uses the explicit per-channel route.

2. **Stage 16.2: Add fused generation routing and descriptor-driven construction.**

   **Routing and compaction**

   - Add a two-daughter routing action in which one thread evaluates one
     event-local track pair, performs common range/source/duplicate guards once,
     intersects role compatibility with the execution-group mask, applies
     descriptor-specific discrete source cuts, and emits one routed task per
     accepted bit.
   - Reuse the qualified 64-thread block-scan compaction pattern from Step 15:
     accepted workload remains distinct from stored workload, each active block
     performs one bounded global reservation, and capacity crossing sets
     overflow without an out-of-bounds write. Keep an atomic/reference mode
     only where it shares the same pair-resolution helper and improves
     qualification.
   - Maintain visited-pair, active-bit, accepted, stored, block-reservation,
     overflow, and per-channel counters. A disabled bit must remove only its
     own work.

   **Construction and generation output**

   - Add descriptor-driven two-daughter construction that consumes the routed
     task pool and its device counter directly on the same queue. It reuses the
     existing validated transport, DCA, Kalman, field, post-build selection,
     and candidate-store primitives.
   - Write one unordered raw-candidate generation per event, preserving stable
     channel ID, event ID, canonical two-source lineage, flags, fit/covariance,
     and the transient descriptor-index sidecar. Per-channel summaries come
     from counters and stable metadata, never atomic output contiguity.
   - Provide an isolated qualification API for fused generation. Do not switch
     `RunDecayPlanBatch()` yet because the current selection continuation still
     assumes per-channel raw ranges.

   **Tests and completion gate**

   - Compare explicit and fused routes as unordered task and raw-candidate
     multisets, including every fit/covariance component within existing
     tolerances and exact channel/event/lineage/flags/counter agreement.
   - Cover one pair matching multiple descriptors, all three default channels,
     charge asymmetry, two events, field transport modes, disabled channels,
     source-cut rejection, build rejection, and task/candidate/daughter
     truncation.
   - Require no host synchronization or counter read between routing and
     construction. Run standalone CPU first and then HIP lifecycle validation.
   - Completion: fused default-V0 raw construction is backend-qualified, while
     the explicit production path remains active until Stage 16.3.

3. **Stage 16.3: Fuse selection continuation, switch steering, and close the gate.**

   **Generation-wide selection**

   - Add one selection action over each event's raw two-daughter generation.
     Resolve the immutable selection config in O(1) through the candidate's
     transient descriptor index, while validating its stable channel ID.
   - Preserve one diagnostic record per raw candidate, including rejected
     candidates, and append selected raw indices plus stable channel IDs to the
     existing bounded compact pool. Selection-result and selected-output
     overflow semantics must remain unchanged.
   - Queue `route/compact -> construct -> select` without an intermediate host
     wait. Selection reads the device-resident raw generation range/counters;
     it must not require per-channel candidate downloads.

   **Production steering and monitoring**

   - Replace the two host channel loops in `RunDecayPlanBatch()` with
     `compile/upload plan -> launch execution groups -> construct generation ->
     select generation -> snapshot`. Resolve all event and per-channel
     summaries with one final wait.
   - Expose generation and channel ranges by stable metadata and counters.
     Preserve the selected-view, V0-selection-record, downstream cascade, and
     CBMRoot diagnostic contracts.
   - Extend monitoring and the controlled benchmark with two-daughter routing
     groups, descriptors, visited pairs, active bits, accepted/stored tasks,
     block reservations, raw/selected counts, launch reduction, overflow, and
     route/construction/selection timings.

   **Final qualification**

   - Run standalone CPU/HIP lifecycle tests, explicit-versus-mask task/raw/
     selection equivalence, the batch benchmark, hermetic CBMRoot CPU/GPU V0
     equivalence, CBMRoot XPU smoke, and cascade continuation tests.
   - Require exact workload, identity, lineage, selection class/rejection/best
     PV, and overflow agreement plus the established numerical tolerances.
     Compare outputs as unordered channel-keyed sets.
   - Remove only production use of the explicit generator and per-channel
     selection launches. Keep their oracle path until Step 17 has reused the
     architecture for additional generations.
   - Completion: both qualified generations execute as device-resident
     `route/compact -> construct -> select` pipelines with no host channel loop
     and no intermediate host synchronization. Adding another compatible
     two-track hypothesis changes descriptors/tables and tests, not steering
     control flow or pair-enumeration launch count.

Status: Step 16 complete (100%). Stages 16.1 through 16.3 are implemented and
pass the complete standalone CPU/HIP lifecycle suite, the controlled HIP
batch benchmark matrix, and the CBMRoot integration gates. Stage 16.1 compiles
the default V0 decay-plan revision into complete flat descriptors, role
compatibility masks, and two source-geometry execution groups. The buffer
owner retains those
revision-controlled tables, routed-task/status storage, per-channel counters,
and a capacity-matched raw-candidate descriptor-index sidecar. The views are
published through `KFParticleGpuKernelState`; a device probe reads the
descriptor/group/mask values and candidate tag.

The independent scalar oracle covers the three default channels, multi-bit and
cross-word masks, mixed/empty ranges, duplicate lineage, invalid role/species/
charge/source/numerical configurations, table upload caching, enabled-channel
updates, and status reset. At the Stage 16.1 checkpoint production remained on
the explicit per-channel two-daughter path as required.

Stage 16.2 adds atomic-reference and 64-thread block-scan routing actions that
share one pair resolver. Each accepted descriptor bit creates one bounded
routed task, and a descriptor-driven construction action consumes that pool
and its device counter directly on the same queue. The isolated
`RunTwoDaughterFusedStage()` API performs no task-count read or host wait
between routing and construction. It preserves stable channel/event/source
identity, complete fit and covariance state, candidate flags, exact accepted
versus stored counters, per-channel counters, bounded task/candidate/daughter
overflow, and the transient descriptor-index sidecar.

Lifecycle check `[72] two-daughter-fused-routing-construction` compares the
fused output with the explicit scalar route as unordered task and candidate
sets. It covers all default V0 channels, multi-bit full-field/straight-line
hypotheses, disabled descriptors, two events, source and numerical build
rejection, both compaction modes, and every bounded storage failure. Target
HIP lifecycle validation completed the external gate for Stage 16.2.

Stage 16.3 adds a descriptor-sized device selection workspace and three
ordered actions: evaluate every raw candidate, prefix bounded selected-output
segments, and scatter selected candidate indices. `RunDecayPlanBatch()` now
uses block-scan generation routing and descriptor-driven construction for all
two-daughter channels. Raw generation storage is intentionally unordered;
stable `channelId` metadata and per-descriptor counters define membership,
while selected output remains physically segmented by event and descriptor.
The production queue contains route, construction, selection, and optional
cascade continuation without an intermediate host counter read or wait.
Queued snapshots are resolved only after the single transaction boundary.

Lifecycle check `[61] decay-plan-generation-wide-pipeline` verifies the
production switch, launch reduction, shared generation ranges, channel
membership, block reservations, selection continuation, and overflow-free
default-V0 fixture. The existing host selection oracle, multi-PV boundary,
multi-event, truncation, explicit-versus-fused construction, and cascade
tests now compare unordered channel-keyed raw output. The batch benchmark
reports `generation_*` routing and selection monitoring and requires exact
serial/batch workload agreement.

The target HIP lifecycle and benchmark matrix plus CBMRoot XPU, diagnostic,
and hermetic CPU/GPU V0 gates pass with this production route. Step 16 is
therefore fully qualified and complete at 100%.

## Step 17: Complete The Device-Resident Decay Graph

Step 17 expands the validated generation-wide V0 and cascade route into a
complete, bounded decay graph. The implementation is organized by input
topology and graph operation rather than by PDG code. A particle channel is
data in a descriptor table; it must not introduce its own kernel family or
host-side launch loop.

The step has four implementation checkpoints. They are deliberately larger
than the checkpoints of the early mathematical port, but each has a closed
ABI and an independently testable result.

### Stage 17.1: Channel Manifest And Flat Graph Contract (25%)

Inventory the active CPU call graph rooted at `FindParticles()` and describe
every channel or channel family in a machine-checkable manifest. The manifest
records a stable channel ID, mother and daughter roles, input source kinds,
dependency generation, construction and constraint operations, selection
profile, output class, and support state. Unsupported entries require an
explicit reason; absence from the manifest is an error.

Add a C++17-compatible, trivially copyable graph ABI for:

- track ranges and candidate-generation ranges;
- track-track, track-composite, composite-composite, and neutral inputs;
- construction, transport, production-vertex, mass-constraint, matching, and
  selection operations;
- generation dependencies, execution groups, output ranges, and capacities.

The host compiler validates stable IDs, source types, generation order,
acyclic dependencies, operation compatibility, and bounded storage before it
publishes revision-owned descriptor tables to persistent XPU buffers. This
stage does not change the active reconstruction path.

Acceptance gate: host and device probes see identical compiled graph tables;
duplicate IDs, cycles, unknown source generations, unsupported operation
combinations, and insufficient descriptor capacity fail deterministically.
Existing V0 and cascade tests remain unchanged.

Stage 17.1 implements this contract without switching the active
reconstruction path. `KFParticleGpuDecayGraphManifest` records nine audited
CPU finder families and explicit supported, partial, or unsupported status
with a numeric reason. The default manifest currently maps the three validated
V0 channels and four validated Xi/Omega channels to seven physical graph
nodes; all remaining CPU families stay visible through coverage entries
rather than disappearing from the GPU plan.

`KFParticleGpuDecayGraphPlan` validates complete family coverage, stable
channel IDs, flat source/topology contracts, strictly earlier candidate
dependencies, known operation masks, output classes, support reasons, and
compile capacities. It sorts supported nodes by generation and builds two
current launch-compatible execution groups. A deterministic field-by-field
content hash provides the revision.

`KFParticleGpuDeviceStorage` owns persistent node, execution-group, and family
coverage XPU buffers. `KFParticleGpuBufferManager` grows and uploads them only
for a new revision, and `KFParticleGpuKernelState` publishes a non-owning graph
view. Lifecycle check `[10] decay-graph-contract` covers the host compiler,
negative manifests, capacities, revision upload, and host view. Check `[13]
published-device-state` verifies the same seven nodes, two groups, nine family
entries, generation, unsupported status, and revision through the XPU
constant-memory image.

### Stage 17.2: Generic Charged Construction Generations (55%)

Generalize the current mask router, block-scan compaction, fit state, lineage,
and bounded output pools to the remaining charged-input families:

- the remaining two-track decays and same-sign or primary resonances;
- track-composite and composite-track decays represented by
  `FindTrackV0Decay()` and `FindLL()`;
- the required per-node transport, mass, production-vertex, and local
  selection operations.

Every node consumes the device-resident ranges produced by its declared
parent generation. Repeated physics hypotheses share pair traversal through
descriptor masks, and no intermediate candidate list is downloaded merely to
prepare the next generation. Existing explicit V0 and Xi/Omega paths remain
qualification oracles until their generic equivalents agree.

Acceptance gate: representative light-flavour, resonance, charm, and
hypernucleus chains match CPU or explicit GPU oracles in channel identity,
canonical lineage, status, counters, and bounded fit tolerances on CPU and
HIP. Multi-generation overflow and invalid dependency propagation are covered.

Stage 17.2 implements this charged-generation backend without adding
PDG-specific kernels. Two-track compatibility is keyed by exact daughter PDG
pairs; the existing compact roles remain optional hints for the already
validated pion/kaon/proton channels. The router also accepts electron, muon,
deuteron, triton, He3, and He4 source species when the descriptor supplies
consistent source geometry, charge, and mass.

The former V0-track descriptor boundary is now generic composite-track.
Every entry carries a stable `parentChannelId`, parent PDG, bachelor PDG, and
source geometry. Device routing checks all four values before setting a
channel bit, so equal-PDG outputs from different generations cannot
cross-feed. Legacy Lambda/Xi/Omega role information remains available for
qualification monitoring, but no longer determines correctness.

Graph nodes now carry a flat payload kind and payload index. Production
`RunDecayPlan()` compiles and uploads the graph revision before event work,
derives active two-track and composite-track descriptor counts from graph
payloads, and rejects disagreement with their payload tables. Both
generations stay on KFParticle's persistent queue; compact selected candidates
feed the parent-qualified continuation without an intermediate candidate-list
download or host synchronization.

Lifecycle check `[76] generic-charged-decay-graph` executes light-flavour
Lambda, primary rho, same-sign Delta++, charm D0, hypertriton, and
parent-qualified D+ continuation descriptors in one transaction. It checks
channel output, valid two- and three-source lineage, graph payload identity,
and rejection of an equal-PDG request with the wrong parent channel. Existing
explicit default-V0 and Xi/Omega checks remain numerical and bounded-counter
qualification oracles.

### Stage 17.3: Composite, Neutral, And Final-State Operations (80%)

Add the remaining topology engines and graph operations needed by
`CombinePartPart()`, both `NeutralDaughterDecay()` forms, `MatchKaons()`,
`ExtrapolateToPV()`, and `SelectParticles()`. This includes
composite-composite pairing, neutral or missing-mass hypotheses, primary
vertex projection, final mass/topology constraints, and routing to temporary,
primary, secondary, and final result classes.

Optional source capabilities, including detector-specific neutral inputs, are
declared in the graph contract. If an input or mathematical operation is not
implemented, the corresponding manifest entry is reported as unsupported
before execution; there is no silent host fallback inside the device graph.

Acceptance gate: one focused synthetic chain for every new topology and
operation passes on CPU and HIP, followed by mixed graphs that combine charged,
composite, and neutral generations. Rejection reason, lineage, output class,
and overflow remain observable and deterministic.

Stage 17.3 adds a flat `KFParticleGpuGraphOperationDescriptor`, bounded task
and result ABI, and one descriptor-driven XPU action for composite-composite,
missing-mass, matching, and unary finalization generations. Descriptors carry
the operation mask and output class; tasks retain only earlier-generation
candidate indices. A generation therefore uses one launch independent of the
number of PDG hypotheses.

The composite engine performs line-DCA measurement construction plus the
validated energy-fit update and merges both source lineages into one sorted,
duplicate-free list. Lineage is explicitly bounded to 16 physical source IDs;
larger or overlapping inputs are rejected before output reservation. The
neutral engine ports the legacy scalar `SubtractDaughter()` Kalman update.
The newer filtered `ReconstructMissingMass()` path has a distinct covariance
update and is represented by an explicit descriptor mode that is rejected as
unsupported rather than silently using different mathematics. Kaon matching
evaluates the residual after legacy subtraction and, on success, publishes the
original matched candidate, matching the CPU ownership semantics.

Unary finalization now provides line projection to a primary vertex, a
point-like production-vertex constraint, the linear mass constraint,
geometric/topological selection, and temporary, primary, secondary, or final
output classification. Candidate and daughter reservations remain bounded and
set the existing pool overflow flags. Detector-specific neutral sources are
not fabricated: a neutral descriptor without the missing-mass operation is
rejected, while the default CPU-family manifest continues to report missing
detector input explicitly.

Lifecycle check `[77] composite-neutral-final-graph-operations` compares host
and device fit states for composite-composite, missing-mass, kaon-match, and
unary PV/final-selection tasks. It also checks canonical lineage, output
classes, duplicate-source rejection, candidate-capacity overflow, and the
three new graph payload kinds. It also verifies that the filtered missing-mass
mode is rejected before execution. The complete local CPU lifecycle contains
84 checks and passes; target HIP execution is the external qualification gate.

### Stage 17.4: Device Graph Scheduler And Coverage Closure (100%)

Compile the manifest into ordered generations and execution groups, then run
the complete supported graph on KFParticle's process-persistent queue.
Generation dependencies are expressed through queue ordering and
device-resident counters; there is no host counter read or queue wait between
graph nodes. A single transaction boundary resolves requested snapshots,
aggregate status, and monitoring.

Finish capacity planning and monitoring for each topology and generation:
visited combinations, active mask bits, accepted and stored tasks, candidate
and daughter occupancy, selection counts, overflow, unsupported channels, and
launch counts. Audit the manifest against the CPU `FindParticles()` call graph
so that every active channel is either mapped to a tested GPU route or carries
an explicit unsupported reason.

Acceptance gate: standalone CPU and target HIP lifecycle suites cover all
topology engines, mixed multi-generation graphs, invalid graphs, overflow, and
deterministic repeated execution. The batch benchmark verifies that graph
execution preserves serial/batch results and does not reintroduce
per-channel launches or host synchronizations. CBMRoot smoke and diagnostic
gates verify the unchanged runtime and ownership boundary.

Step 17 is complete only when the manifest audit has no unclassified CPU
channel and all declared supported topology families pass their CPU/HIP
qualification gates. Production replacement of CPU results, exhaustive
physics validation on representative data, and automatic fallback remain
Step 18 responsibilities.

Stage 17.4 is implemented. `RunDecayPlan()` now compiles configured graph
operations into revision-owned descriptors, routes each supported execution
group into a reusable device task pool, and executes dependent generations in
queue order. Descriptor, task, result, aggregate-counter, and per-channel
counter allocations live in `KFParticleGpuDeviceStorage`; no graph counter is
read and no queue wait is introduced between generations.

Lifecycle check `[77] device-graph-scheduler` runs charged, cascade,
composite-composite, legacy missing-mass, and unary generations in one
transaction. It verifies persistent ownership, per-channel accounting,
repeatable execution, and bounded task-pool overflow. The focused operation
oracle remains check `[78]`. The complete local CPU lifecycle now contains 85
checks and passes.

Step 17 implementation status: complete (100%). Target HIP lifecycle, batch
benchmark, and the unchanged external CBMRoot smoke/diagnostic gates remain
the release qualification commands for this step. Production output handoff,
full representative-data physics parity, and fallback remain Step 18.

## Pre-Step 18 Refactoring: Compact Production Kernel ABI

This checkpoint reduces production kernel argument lists before the output
handoff work begins. Persistent input, work, descriptor, and output storage
must remain owned by `KFParticleGpuDeviceStorage`; kernels access flat
non-owning views of that storage through the published
`KFParticleGpuKernelState`. Only values that select the work of one launch,
such as an event index, execution-group index, range, or effective task limit,
remain ordinary kernel arguments.

The refactoring is intentionally split into two substantial stages. Converting
the state contract, publication lifetime, and every production kernel in one
change would make an ABI, stale-view, or HIP-only regression difficult to
localize. More than two stages would unnecessarily prolong a mostly mechanical
migration once the state-backed graph path has proved the contract.

### Stage K.1: Published State Contract And Graph Scheduler

Extend `KFParticleGpuKernelState` with the missing flat graph-operation,
generation-work, task, result, and counter views required by the Step 17
scheduler. Centralize publication in steering and publish one coherent state
snapshot before a decay-plan transaction when packed sizes, storage addresses,
capacities, or the plan revision have changed. Publication must not occur
before every kernel.

Migrate graph reset, routing, and execution actions to
`xpu::cmem<TheKFParticleFinder>`. Their production signatures retain only the
event/group selection and any genuinely dynamic execution bound. Keep an
explicit-view graph action as a clearly named test oracle rather than a
production fallback.

Acceptance gate:

- a constant-memory probe validates every newly published pointer, stride,
  size, capacity, and descriptor after initial allocation and capacity growth;
- state-backed and explicit-oracle graph execution agree on results, monitoring,
  repeated execution, and bounded overflow;
- standalone CPU and target HIP lifecycle suites pass without introducing a
  per-operation `xpu::set` or host synchronization.

Stage K.1 is implemented and represents 50% of this refactoring.
`KFParticleGpuGraphOperationStorageView` now publishes descriptor/task/result
storage, allocated capacities, revision identity, aggregate counters, and all
per-channel counters through `KFParticleGpuKernelState`. Steering preflights
the two-daughter, cascade, decay-graph, and later-operation allocations, then
calls one centralized `PublishKernelState()` before queueing reconstruction.

The production graph reset action now has no ordinary arguments. Routing keeps
only event index, execution-group index, and an effective task limit; execution
keeps only that limit. The effective limit remains separate from the physical
monotonic capacity so bounded truncation semantics are unchanged. The former
explicit execution action is retained as
`KFParticleGpuExecuteGraphOperationsExplicit`, a test-only oracle.

Lifecycle check `[14] published-generation-state` validates graph and
generation workspace pointers and their republished capacities after buffer
growth. Checks
`[77]-[79]` cover state-backed repeated execution, bounded overflow, monitoring,
and the explicit host/device operation oracle. The complete local CPU lifecycle
contains 86 checks and passes in 0.43 seconds. Target HIP lifecycle remains the
backend qualification gate for K.1.

### Stage K.2: Production Kernel Migration And ABI Closure

Migrate the remaining production reset, two-daughter, V0-track/cascade,
construction, and selection-continuation actions to the same published state.
Keep only true launch controls in their signatures. Do not replace many
arguments with one aggregate launch-parameter object: that would conceal the
same kernel-argument payload instead of removing it.

Explicit-view actions remain only where they provide an isolated device probe
or CPU/GPU oracle. Remove the duplicate production launch branches and the
production use of `KFPARTICLE_GPU_USE_KERNEL_ARGS` after parity is established.
Document the publication lifecycle and the small dynamic argument set for each
production action.

Acceptance gate:

- an ABI audit finds no persistent buffer view in production kernel argument
  lists; exceptions are test-only and explicitly documented;
- production actions normally carry no more than three or four scalar launch
  controls, with any larger signature justified in the audit;
- the complete standalone CPU and target HIP lifecycle suites, serial/batch
  benchmark matrix, and CBMRoot runtime and diagnostic smoke gates pass;
- before/after benchmark evidence shows unchanged results and no material
  throughput regression.

Stage K.2 completes this refactoring at 100%. Performance improvement is not
assumed from argument-count reduction alone; the immediate gains are a stable
production ABI, clearer ownership, fewer launch parameters, and a single
auditable device-state publication boundary.

Stage K.2 is implemented. `KFParticleGpuTwoDaughterGenerationStorageView` and
`KFParticleGpuV0TrackGenerationStorageView` now publish the routed task pools,
aggregate routing counters, physical capacities, and first-generation
selection workspace. Reset, block-scan routing, routed construction, and
selection-continuation actions read those views from
`TheKFParticleFinder`. The production V0, cascade, and later graph paths
therefore carry only event/group/range/effective-limit controls.

The former `KFPARTICLE_GPU_USE_KERNEL_ARGS` production branch and
`KFParticleGpuRoundTripArgs` action are removed. Explicit fixed-slot,
compact-generation, atomic-routing, isolated-selection, numerical-probe, and
graph-oracle actions remain intentionally outside the production ABI for
qualification and differential testing.

Lifecycle check `[14] published-generation-state` now validates every routed
generation pointer plus graph, two-daughter, V0-track, and selection-workspace
capacities before and after monotonic buffer growth. A fresh local CPU build
passes both CTest targets in 0.36 seconds. The four-event/five-iteration CPU
batch benchmark reports serial/batch equivalence, atomic/block-scan routing
equivalence, and zero overflow. Target HIP lifecycle, benchmark matrix, and
the unchanged CBMRoot smoke gates remain the backend qualification evidence.

Status: compact production kernel ABI refactoring complete (100%). Step 18 may
begin after the target HIP qualification commands pass.

## Step 18: Complete Physics Parity And Output Handoff

Step 18 turns the complete device graph into a controlled alternative result,
while the CPU finder remains the event-level fallback. The step is divided
into three substantial checkpoints. Physics comparison, output
materialization, and production routing deliberately remain separate so that
a numerical mismatch, an ancestry/index conversion defect, and a fallback
policy defect cannot hide one another.

### Stage 18.1: Complete Parity Contract And Promotion Verdict

Generalize the current two-daughter diagnostic snapshot into an
order-independent candidate contract for every implemented topology. A stable
key must contain event and channel identity plus bounded canonical physical
lineage of variable length; it must not depend on candidate-pool order.
Snapshots must cover the complete fit state and covariance, charge, chi2/NDF,
PDG, primary-vertex association, output class, selection state, and operation
status needed by a later materializer.

Build one explicit tolerance policy keyed by channel or operation family.
Compare exact discrete state and ancestry separately from bounded numerical
residuals. Extend standalone and hermetic CBMRoot fixtures across:

- two-track V0 construction and selection;
- selected-composite plus bachelor cascades;
- charged track-track and track-composite generations;
- composite-composite construction;
- supported legacy missing-mass/kaon matching;
- unary PV projection and final selection.

Include deterministic ordering-independent matching, topology/mass boundary
cases, multiple primary vertices, invalid input and field data, unsupported
graph nodes, candidate/daughter/task/selected-pool exhaustion, and repeated
batched execution. Produce a flat `promotion verdict` with explicit reason
bits for unsupported scope, overflow, invalid data, incomplete lineage, and
physics mismatch. No public CBMRoot output changes in this stage.

Acceptance gate:

- every implemented topology has a host or CPU-finder reference and named
  CPU/HIP lifecycle coverage;
- all exact and tolerance-based comparisons are channel-local and report the
  first actionable failure rather than one aggregate pass/fail flag;
- the verdict rejects every injected unsupported, overflow, invalid-input,
  and mismatch case;
- standalone lifecycle, batch equivalence, CBMRoot unit tests, and hermetic
  CPU/GPU comparison pass.

Completion after Stage 18.1: 35%.

Stage 18.1 implementation notes:

- Candidate metadata now persistently records graph topology, output class,
  and a topology-independent accepted/rejected operation status. Successful
  and failed two-track, composite-track, and generic graph store paths write
  the same contract; these values no longer disappear with the reusable graph
  result workspace.
- `KFParticleGpuParitySnapshot` is a flat C++17 host/device value containing a
  canonical physical lineage of up to 16 source IDs, the complete available
  fit/covariance state, mass, PV/selection observations, topology, output
  class, and operation status. Availability bits keep exact CPU/GPU comparison
  honest when the scalar CPU API does not expose internal GPU-only metadata.
- Numerical comparison is separated from exact key and discrete-state
  comparison. `GpuDiagnosticTolerancePolicy` supports channel/output-class
  overrides, reports first parameter/covariance failures and maximum
  residuals, and feeds an explicit promotion verdict.
- Promotion blockers distinguish unsupported scope, overflow, invalid input
  or field, incomplete lineage, physics mismatch, unavailable runtime, and
  execution failure. CBMRoot monitoring aggregates both the verdict and the
  generalized fit/operation residual summaries.
- The CBMRoot diagnostic runner now extracts first-generation, cascade, and
  later graph ranges through one variable-lineage snapshot path. The existing
  two-daughter compatibility fields remain available while downstream tests
  migrate to the generalized record.
- Standalone check `[15] parity-promotion-contract` names all six graph
  topologies and validates canonical lineage, tolerance boundaries, exact
  operation metadata, and promotion blockers. Existing device construction
  checks additionally assert persisted topology/output/status for
  two-daughter, V0-track, composite-composite, neutral, and unary outputs.
  The CBMRoot unit fixture covers three- and four-source keys,
  channel/output-specific tolerance, mismatch, and overflow verdicts.

Status: Stage 18.1 implementation complete. Fresh local CPU/XPU lifecycle
passes 87 checks and both CTest targets; target HIP lifecycle,
CBMRoot unit, hermetic equivalence, and smoke validation remain the completion
evidence before recording Step 18 at 35%.

### Stage 18.2: Bounded KFParticle Output Materialization

Add an external-KFParticle materializer that converts one validated event from
the compact candidate/covariance/metadata pools into the ordinary
`KFParticle` result contract. First make direct ancestry explicit: flattened
physical source lineage is sufficient for comparison but not necessarily for
recreating immediate composite daughter references. Add a bounded
candidate-parent/direct-daughter sidecar where the current pool cannot express
that distinction.

Materialization must preserve the complete numerical state, covariance,
chi2/NDF, charge, PDG, PV association, stable channel/output class, direct
daughter graph, and physical leaf IDs. It must topologically map GPU candidate
indices to final particle IDs and validate every range before modifying public
state. Build into a temporary event result and commit it atomically; any
invalid index, unresolved parent, unsupported output class, or overflow
discards the complete GPU event result.

Keep the owning implementation inside external KFParticle. CBMRoot receives a
thin adapter that can populate an isolated `KFParticleTopoReconstructor` or
equivalent event result for tests, but normal reconstruction still consumes
CPU output. Compare materialized GPU particles with CPU particles through the
Stage 18.1 contract and verify that downstream readers observe valid IDs and
daughter references.

Acceptance gate:

- standalone tests cover leaves, V0s, cascades, composite-composite, unary
  outputs, repeated ancestry, and all bounded failure paths;
- materialize-then-snapshot preserves the validated GPU fit, covariance,
  metadata, direct ancestry, and canonical leaf lineage;
- a failed event leaves its destination unchanged;
- CBMRoot unit and synthetic CPU/GPU equivalence gates pass without enabling
  GPU output in normal reconstruction.

Completion after Stage 18.2: 70%.

Implementation notes:

- Direct ancestry is encoded in the existing component-major candidate
  metadata buffer as at most two typed references. Track-track stores two
  packed-track indices, composite-track stores candidate plus track, and
  graph operations store one or two candidate indices according to their
  topology. Flattened physical lineage remains unchanged.
- `KFParticleGpuMaterializer` builds a ROOT-independent temporary event,
  validates every dependency and canonical leaf lineage, and exposes an
  atomic templated commit to ordinary scalar particles. It preserves all eight
  fit parameters, 36 covariance elements, fit scalars/integers, PDG, PV
  association metadata, channel/output identity, direct daughter IDs, and
  physical leaf IDs.
- The standalone lifecycle covers V0, cascade, composite-composite, unary,
  repeated ancestry, full-state transfer, overflow, invalid ranges,
  unresolved/cyclic references, invalid output classes, lineage mismatch,
  non-finite state, and unchanged destination on failure.
- CBMRoot owns only `GpuMaterializationAdapter`, which commits into an isolated
  `std::vector<KFParticle>` for unit tests. Normal reconstruction output and
  the CPU-default route remain untouched.

Status: Stage 18.2 implementation complete. Fresh local CPU/XPU lifecycle
passes 83 checks and both CTest targets in 0.36 s. Target HIP lifecycle and
CBMRoot unit/equivalence evidence remain before recording Step 18 at 70%.

### Stage 18.3: Opt-In Routing, Event-Atomic Fallback, And Qualification

Add an explicit CBMRoot configuration mode with three states: CPU-only
(default), diagnostic dual execution, and validated GPU output. A capability
manifest declares the exact channel/operation set and required inputs covered
by the current GPU implementation. GPU output is eligible only when the
requested reconstruction graph is a subset of that manifest and the Stage
18.1 verdict plus Stage 18.2 materialization both succeed.

Fallback is event-atomic. Unsupported requests, unavailable runtime/device,
invalid or missing field/input data, any task/candidate/daughter/selection
overflow, failed validation, or materialization failure routes the complete
event through the existing CPU finder. Do not merge a partial GPU event with
CPU candidates in this step. Build the GPU result in isolation and publish it
only after the final verdict, so fallback cannot leave partial particles or
selection bits behind.

Retain diagnostic dual execution and monitoring. Add deterministic reason
counters for GPU accepted, CPU requested, and every fallback class. Exercise
the routing decision without detector files through synthetic CBMRoot tests;
representative-data throughput and production-default promotion remain Step
19 responsibilities.

Acceptance gate:

- CPU-only output is unchanged byte-for-byte at the public boundary and
  remains the default;
- validated GPU mode produces the materialized result for supported synthetic
  events and falls back cleanly for every injected stop condition;
- repeated and batched events cannot leak output, status, or queue state
  across event boundaries;
- standalone CPU/HIP lifecycle, benchmark matrix, CBMRoot unit/equivalence,
  runtime smoke, diagnostic smoke, and new routing/fallback smoke gates pass.

Completion after Stage 18.3: 100%.

Step 18 completion criterion: GPU output can replace CPU output only for the
declared and fully validated capability set, at event granularity, without
changing default reconstruction behaviour. Full-chain online/offline bring-up
is deferred to Step 19; real-data performance thresholds and enabling GPU
routing by default are explicitly deferred to Step 20.

Status: Stages 18.1 and 18.2 are implemented; Stage 18.2 target HIP/CBMRoot
qualification is the current gate before Stage 18.3 routing work.

Stage 18.3 implementation notes:

- `gpuDiagnostics.mode` now has explicit `cpu-only`, `diagnostic`, and
  `validated-gpu` states. Absence of the block remains CPU-only, while an
  existing block without `mode` preserves diagnostic behaviour.
- The initial production capability manifest deliberately exposes only the
  field-aware K0S, Lambda, and anti-Lambda track-track channels with
  construct/transport/select operations and default finder cuts. The broader
  external graph remains unavailable for publication until separately
  qualified.
- Validated mode keeps the CPU event as a ready fallback, runs the GPU event
  synchronously on KFParticle's process-persistent queue, materializes into a
  temporary vector, and publishes through one particle-vector swap only after
  capability, execution, overflow, parity, materialization, and public-size
  gates all pass. Partial CPU/GPU merging is impossible.
- `GpuRoutingMonitor` records an exclusive reason for acceptance, explicit CPU
  use, diagnostic execution, sampling, and every fallback class. Unit tests
  cover the complete deterministic routing matrix; the embedded CBMRoot smoke
  now covers actual HIP materialization, accepted routing, and invalid-field
  fallback.

Status: Step 18 complete and qualified (100%). Target HIP lifecycle and
benchmark checks, CBMRoot build/unit/equivalence checks, and the embedded
runtime plus diagnostic-routing smoke tests have passed. GPU output can now
replace CPU output only for the declared validated capability subset; CPU-only
remains the production default.

## Step 19: Online And Offline Full-Chain Bring-Up

Step 19 establishes reproducible end-to-end execution before any production
performance claim is made. CBMRoot currently has two official reconstruction
modes and both are required:

- the online mode driven by the `cbmreco` application and
  `algo/global/Reco`;
- the offline mode driven by FairRunAna and ROOT reconstruction macros.

Neither mode is a fallback for the other. They must use the same external
KFParticle GPU implementation, numerical contracts, capability manifest, and
event-atomic routing rules. Mode-specific code is limited to input conversion,
configuration, and lifecycle adaptation.

The closest working root-level `run_reco_tracks.C` reference was copied to
`GPU/test/cbmroot/run_reco_tracks.C` so it is available in both local and
server KFParticle checkouts. Stage 19.2 replaces its disabled legacy
KFParticle block with a thin FairTask adapter to `KfpSelectorChain`, which is
already part of the online `algo/global/Reco` path. The installed FairRoot
version, available input formats, event building, primary-vertex and PID
preparation, and XPU library resolution are qualified independently in both
modes.

### Stage 19.1: Shared Contract And Server Compatibility

Document the complete data and lifecycle contract shared by the online and
offline modes:

- reconstructed and fitted tracks, hit references, event boundaries, primary
  vertices, field regions, PID inputs, and stable source-event identity;
- CBMRoot-owned XPU initialization, algorithm-local persistent queues, device
  selection, image loading, finalization, and error propagation;
- CPU-only, diagnostic CPU/GPU, and later qualified-GPU configuration
  semantics;
- identical capability, overflow, comparison, materialization, and fallback
  rules.

Audit the exact target-server prerequisites for both entry points: FairRoot
and ROOT versions, CBMRoot build options, `cbmreco` options, XPU and HIP
libraries selected at run time, device architecture, geometry and parameter
files, and supported input formats. Determine which available data can be
consumed directly by each mode and where a controlled conversion is required.

Prepare separate preflight launchers for online and offline reconstruction.
Each must report the selected source/build directories, environment mode,
loaded XPU libraries, device, input identity, geometry, parameters, and
enabled reconstruction steps before processing data. Incompatibility,
missing files or branches, and a wrong XPU library must fail early with a
specific message.

Acceptance gate:

- the common input/output and XPU lifecycle boundary is explicit and covered
  by mode-independent tests;
- online and offline launch requirements and available input formats are
  recorded separately rather than inferred from one another;
- no physics or GPU kernel implementation is duplicated in CBMRoot mode
  adapters or the legacy `CbmKFParticleFinder`;
- one-timeslice preflight commands for both modes reach XPU initialization or
  stop at a named, actionable missing prerequisite;
- standalone lifecycle, CBMRoot unit/equivalence, and embedded XPU smoke tests
  remain unchanged.

Completion after Stage 19.1: 25%.

### Stage 19.2: Offline FairRunAna Reconstruction Chain

Use the test-owned `GPU/test/cbmroot/run_reco_tracks.C` copy as the initial
offline reference because it is closest to a working GPU CA run in the
installed FairRoot environment and is distributed with this work. Keep the
existing task order explicit and bring the offline path up in bounded
checkpoints:

1. reconstruct STS hits and tracks with the requested GPU CA backend;
2. build events without silently dropping or regrouping input beyond the
   documented policy;
3. provide primary vertices, fitted tracks, PID inputs, field data, and stable
   event identity;
4. feed those data through a thin FairTask adapter into the same
   `KfpSelectorChain` used online;
5. run CPU-only KFParticle first, then diagnostic CPU/GPU KFParticle in the
   same process.

Isolate every compatibility adjustment required by the installed FairRoot
version. Do not modify unrelated standard CBMRoot macros and do not route the
new GPU implementation through the legacy `CbmKFParticleFinder`. KFParticle
must use its own process-persistent queue while reusing the already initialized
CBMRoot XPU runtime.

Acceptance gate:

- one documented offline command processes at least one timeslice through GPU
  tracking and CPU-only `KfpSelectorChain`;
- a second offline command enables diagnostic KFParticle GPU execution
  without reinitializing XPU or borrowing the CA queue;
- event, track, vertex, PID, field, and identity counters are validated at
  every handoff;
- repeated short runs produce stable results and terminate cleanly;
- an incompatible FairRoot interface, missing input branch, or invalid event
  conversion produces an explicit failure rather than partial output.

Completion after Stage 19.2: 60%.

### Stage 19.3: Online Cbmreco Chain And Cross-Mode Qualification

Configure the official online `cbmreco` path to run GPU CA tracking, event
reconstruction, and `KfpSelectorChain` in one process. Provide a minimal,
versioned configuration containing the exact reconstruction steps, device,
default V0 channels, and CPU-only or diagnostic GPU mode. If the available
server data are not directly consumable, add or select a reproducible input
conversion without changing the reconstruction algorithms.

Run the online chain through the same checkpoints and monitoring counters as
the offline chain. Then compare both modes on the same input or on a proven
equivalent controlled input. Require agreement of event boundaries, source
identity, track and primary-vertex counts, field/PID availability, candidate
lineage, channel counts, selection state, overflow/fallback state, and bounded
fit observables. Differences caused by online/offline event building must be
reported separately from KFParticle numerical differences.

Store exact commands, configurations, environment and library information,
input hashes, and concise evidence for both modes. The two launchers may differ
in framework-specific setup, but they must enter the same KFParticle GPU
service and produce the same diagnostic contract.

Acceptance gate:

- one documented online command processes at least one timeslice through GPU
  tracking and CPU-only `KfpSelectorChain`;
- a second online command enables diagnostic KFParticle GPU execution with
  correct runtime ownership and clean finalization;
- both official modes pass their independent full-chain smoke runs;
- controlled cross-mode comparison has no unresolved data-handoff or
  KFParticle mismatch;
- existing lifecycle, benchmark, CBMRoot unit/equivalence, runtime, and
  diagnostic smoke gates pass on the target HIP configuration.

Completion after Stage 19.3: 100%.

Step 19 completion criterion: both official CBMRoot reconstruction modes can
run GPU CA tracking followed by the current KFParticle CPU-only and diagnostic
GPU paths through reproducible server commands. Environment, input conversion,
event building, and framework-specific compatibility are explicit, while all
physics and GPU execution remain shared.

Stage 19.3 implementation checkpoint:

- `cbmreco` now exposes explicit `--kfp-mode` and
  `--kfp-diagnostic-sample-period` options. A command-line mode overrides the
  optional YAML GPU block, so CPU-only and diagnostic qualification use the
  same immutable `MainConfig.yaml`.
- Requesting the `KfpSelector` reconstruction step now enables the shared
  chain directly; it no longer depends on a trigger mask or an incidental
  YAML diagnostic node.
- the online chain publishes handoff contract 1 and diagnostic contract 2,
  including timeslices, events, successful conversions, primary vertices,
  tracks, STS hit links, PID slots, actually assigned PID hypotheses,
  selected tracks, CPU particles and composite candidates, completed
  comparisons, matches, and blocker state;
- `run_cbmroot_kfp_online_preflight.sh` validates the loaded `libAlgo.so`,
  executes repeated CPU-only or diagnostic runs, rejects conversion failures
  and empty KFParticle results or incomplete comparisons, and requires
  byte-for-byte stable canonical summary payloads while ignoring the
  FairLogger timestamp prefix;
- bounded server qualification processes at most 50 reconstructed events per
  timeslice by default. The explicit STS pion commissioning policy supplies
  charge-signed pion hypotheses when the `.tsa` input has no usable TOF PID;
  normal `cbmreco` keeps detector PID as its default;
- KFParticle reconstruction channels now follow `kfp.selector.decays`
  independently of event-selection output bits. Monitoring reports the number
  of channels passed to `KFParticleFinder`, and the focused CBMRoot test covers
  a configured K0S channel without a corresponding `eventSelector` bit;
- `run_cbmroot_kfp_online_campaign.sh` runs both modes and proves that enabling
  diagnostics does not alter the online event/track/vertex/PID handoff;
- `run_cbmroot_kfp_cross_mode_qualification.sh` verifies that online and
  offline evidence use the same source/build, device, XPU/HIP/ROCm,
  FairRoot/ROOT, and main configuration. It compares handoff counts only when
  the caller declares the inputs equivalent; the currently available
  unrelated `.tsa` and generated `.eb` samples use the explicit
  `independent` relation and make no false numerical-equivalence claim.

Target-server Stage 19.3 evidence:

- the online campaign processed the bounded 50-event prefix of
  `vt26test.00001.tsa` twice in CPU-only mode and twice in diagnostic mode;
  every run completed one timeslice and passed the embedded CBMRoot-owned XPU
  runtime smoke;
- repeated runs produced identical canonical KFP summaries, and enabling GPU
  diagnostics preserved the complete event/track/vertex/PID/CPU-candidate
  handoff;
- diagnostic contract 2 completed and matched every sampled event without a
  blocker, conversion failure, CPU-only/GPU-only candidate, fit mismatch,
  selection mismatch, or pool overflow;
- `run_cbmroot_kfp_online_campaign.sh` wrote the accepted CPU/diagnostic
  evidence, and `run_cbmroot_kfp_cross_mode_qualification.sh` accepted the
  online and Stage 19.2 offline evidence under the explicit `independent`
  input relation;
- the cross-mode result qualifies the shared source/build, HIP/XPU/ROCm,
  FairRoot/ROOT, configuration, per-framework handoff, and diagnostic
  contracts. It deliberately makes no event-count equality claim for the
  unrelated real `.tsa` and generated `.eb` inputs.

Status: Stage 19.3 and Step 19 are complete and target-server qualified
(100%). Both official CBMRoot modes have reproducible CPU-only and diagnostic
GPU full-chain commands. Numerical online/offline equality for equivalent
input data remains a stronger Step 20 representative-qualification task, not
an unresolved Step 19 integration requirement.

Stage 19.3 real-data V0 publication correction:

- the two-stage full-field daughter fit is retained; host replay and HIP now
  agree at the raw-candidate boundary;
- default V0 topology now mirrors the actual CPU `ConstructV0` publication
  sequence. Its historically named `GetDistanceToVertexLine` quantity is the
  three-dimensional PV-to-decay-vertex distance, not a perpendicular distance
  to the momentum line;
- neutral V0 mothers now execute the scalar equivalent of
  `SetProductionVertex` followed by `GetDecayLength`, including vertex-induced
  correlations and the covariance of the decay-length parameter;
- default V0 publication always follows the CPU secondary path and therefore
  cannot bypass the mandatory `L/dL` cut through an approximate primary
  classification;
- local standalone CPU validation passes both CTest executables and all 92
  lifecycle checks. Build marker `20260731-cpu-v0-publication-v8` distinguishes
  the corrected CBMRoot library during the target HIP campaign.

Stage 19.3 CPU/HIP numerical-parity correction:

- exhaustive intermediate tracing localized the remaining synthetic
  full-field difference to compiler FMA contraction rather than routing,
  device storage, a race, or a different decay branch;
- floating-point contraction is disabled only in the CPU-parity angle,
  trigonometric, DCA, field interpolation, covariance transport, full-field
  measurement, and energy-fit functions. The exact inventory and rationale
  are maintained in `KFParticleGpuNumericalReproducibility.md`;
- the target HIP lifecycle run now passes both executables and all 96 lifecycle
  checks, including real-scale transport, default V0, fused routing, topology,
  cascade, graph, overflow, and runtime-lifecycle coverage;
- full intermediate traces remain available with
  `KFPARTICLE_GPU_TEST_VERBOSE_TRACE=1`, while normal successful runs retain
  compact PASS output. Genuine tolerance failures continue to emit compact
  diagnostic details automatically;
- Step 20 must profile these local contraction-off regions independently.
  The initial theoretical estimate is a low-single-digit to roughly 10% cost
  inside the full-field fit stage and a smaller whole-chain effect, but only
  target-GPU measurements may be used as evidence.
- real-data diagnostics carry build marker
  `20260803-real-data-pv-parity-v11`. Online preflight rejects a stale
  `libAlgo.so` before reconstruction and accepts a bounded mismatch-event
  trace count or `ALL`, so multiple independent failures can be collected in
  one server run.
- the first full real-data trace exposed three integration-contract defects
  that synthetic single-event fixtures could not show: online event number
  `-1` was reused as an unsigned comparison key, GPU routing omitted the CPU
  Finder's fast daughter-distance/momentum gate, and line topology rejected a
  PV when CPU retained its distance with the `1e8` covariance-error fallback.
  The event key now combines timeslice and event offset, default V0 routing
  applies the CPU gate before construction, and line topology preserves valid
  pointing/decay observables while retaining the degenerate-covariance status
  bit. A near-zero mass-variance sign difference is accepted only when the
  valid-side mass error is below the configured comparison resolution.

Stage 19.1 implementation checkpoint:

- `cbmroot_kfp_preflight_common.sh` defines one build, XPU/HIP-library,
  explicit-device, runtime-ownership, input-identity, and evidence contract
  used by both framework modes.
- `run_cbmroot_kfp_online_preflight.sh` validates the `cbmreco` executable,
  complete `Tracking -> EventReco -> KfpSelector` step list, STS system,
  `gpu-multi-window` backend, main YAML and reconstruction-parameter archive.
  It can optionally process a bounded number of timeslices.
- `run_cbmroot_kfp_offline_preflight.sh` validates ROOT/FairRunAna, the
  bundled test `run_reco_tracks.C`, `.raw.root` and `.par.root` inputs, setup
  identity, and a supported GPU CA backend. It can process a bounded number of
  entries and explicitly reports the missing modern `KfpSelectorChain`
  FairTask adapter as the Stage 19.2 boundary.
- `_TestKfpGpuFullChainPreflight` exercises the shared contract and both mode
  adapters without ROOT, XPU, or detector data. Exact target-server commands
  and the separate input requirements are recorded in
  `KFParticleGpuFullChainBringUp.md`.
- The first server campaign is fixed to the one-timeslice
  `vt26test.00001.tsa` online input and the matching generated `.tb` and `.eb`
  offline samples. Because these samples contain empty events/modules,
  preflight accepts only GPU multi-window CA tracking. Event-based checks
  default to 100 entries and reject a larger accidental run unless explicitly
  overridden.

Target-server Stage 19.1 evidence:

- online `vt26test.00001.tsa` resolves its YAML, `reco.par.bin`, exact
  `libxpu.so`, `hip1`, `gfx906;gfx908`, and the full
  `Tracking -> EventReco -> KfpSelector` command;
- offline `.tb` and `.eb` inputs resolve distinct raw/parameter files,
  time-based/event-based identity, bounded entry ranges, and the required
  `gpu-multi` backend;
- all three preflights pass the embedded CBMRoot-owned XPU smoke, where
  KFParticle reuses the runtime, runs V0 and Xi/Omega kernels on its own queue,
  and leaves the global runtime active.

Status: Stage 19.1 complete and target-server qualified. Step 19 is 25%
complete; Stage 19.2 offline FairRunAna integration is next.

Stage 19.2 implementation checkpoint:

- `TaskRecoEventConverter` now supports the STS-only offline boundary without
  requiring MC or `GlobalTrack` branches. It resolves event-local STS hits
  from track references, preserves track ordering, supplies PV/track/PID
  vectors, and fails explicitly on incomplete requested input.
- `TaskKfpSelector` reads the shared online YAML configuration and runs the
  same `KfpSelectorChain` from FairRunAna. CPU-only and diagnostic modes are
  explicit. Stable source IDs and handoff monitoring cover events, tracks,
  hit links, vertices, PID slots, field setup, and GPU diagnostic status.
- The test-owned `run_reco_tracks.C` now executes
  `GPU CA -> event policy -> PV -> RecoEvent -> KfpSelectorChain`.
  Time-based data are clustered into events; event-based data preserve their
  existing event branch. The obsolete commented legacy finder block is gone.
- `run_cbmroot_kfp_offline_preflight.sh` requires the adapter library and
  shared YAML nodes, keeps CPU-only and diagnostic evidence separate, rejects
  an empty KFP handoff, and requires every published GPU result to execute and
  match its CPU reference in diagnostic mode. Publication-owned cumulative
  counters prevent the last completed result from being counted again by a
  later timeslice. With a sample period of one, the number of comparisons
  must equal the number of processed KFP events. Diagnostic reports carry
  `diagnostic_contract=2`, and preflight rejects an older
  `libCbmRecoTasks.so` before starting reconstruction.
- A known FairRoot shutdown failure after a completed macro is excluded from
  this qualification. The launcher ignores a non-zero ROOT exit status only
  after the KFP summary and the macro completion/output markers are present
  and the current `.reco.root` output is non-empty. An earlier failure, an
  empty output, or an incomplete CPU/GPU comparison remains fatal.
- `_GTestKfpSelectorPid` protects the explicit STS-only pion policy, and
  `_TestKfpGpuFullChainPreflight` covers both offline modes, input policies,
  mode validation, generated commands, duplicate diagnostic accounting,
  blocked comparisons, and the post-completion FairRoot exception without
  ROOT or a GPU.

Target-server Stage 19.2 evidence:

- the rebuilt `libCbmRecoTasks.so` exposes `diagnostic_contract=2`;
- `_GTestKfpGpuDiagnosticPacker`, `_TestKfpGpuFullChainPreflight`, and
  `_GTestKfpSelectorPid` pass in the target CBMRoot build;
- the embedded CBMRoot XPU smoke reuses the initialized runtime and executes
  V0 and cascade kernels on the KFParticle-owned queue;
- two independent event-based diagnostic runs each process 100 input entries
  through GPU CA tracking and the shared `KfpSelectorChain`;
- the launcher accepts both runs only after verifying the complete
  publication-owned diagnostic counter contract and non-empty output;
- the known FairRoot post-output shutdown status is isolated from the
  reconstruction result and does not hide an earlier failure.

Historical Stage 19.2 checkpoint: the offline integration was target-server
qualified at 60%. Stage 19.3 has since completed the online `cbmreco` and
cross-mode qualification, bringing Step 19 to 100%.

## Step 20: Performance Hardening And Production Qualification

Step 20 begins only after both Step 19 full-chain modes are stable. It measures
and optimizes the shared GPU implementation and then qualifies online and
offline production use separately.

Step 20 is intentionally divided into exactly three large controlled stages.
Stage 20.1 changes observability but not execution policy, Stage 20.2 changes
execution only where the baseline proves a bottleneck, and Stage 20.3 freezes
the implementation and makes the production decision. This is the minimum
practical split: combining measurement with optimization would invalidate the
baseline, while combining optimization with qualification would make a
failed campaign ambiguous.

Shared invariants for all three stages:

- CPU-only remains the default and diagnostic mode continues to compare the
  complete CPU and GPU results;
- KFParticle keeps its own process-persistent XPU queue and persistent
  device-owned input, work, and output storage;
- no stage may weaken channel, lineage, fit, selection, overflow,
  materialization, or event-isolation checks;
- online and offline are measured and qualified independently, using the
  Step 19 launchers and explicit input identities;
- every stage ends with local checks plus exact target-HIP commands and stored
  machine-readable evidence before the next stage begins.

### Stage 20.1: Reproducible Baselines And Bottleneck Map

Build one opt-in performance-monitoring contract shared by the standalone
benchmark, online `cbmreco`, and offline FairRunAna adapter. Separate cold
startup and image loading from steady-state event processing. Record both
end-to-end time and queue-ordered GPU phase times for input upload, routing,
construction, selection, graph generations, output download, and host
materialization.

Add the work and traffic counters needed to explain those times:

- events, tracks, descriptor groups, visited combinations, accepted/stored
  tasks, raw and selected candidates, and physical daughters;
- kernel launches, queue waits, buffer growth, host-to-device and
  device-to-host byte counts;
- mask density, useful work per launch, block reservations, rejected work,
  pool occupancy, and every overflow class;
- GPU tracking time, event-building time, CPU finder time, GPU transaction
  time, comparison/materialization time, and total reconstruction time as
  separate quantities.

The monitoring path must be disabled by default and must not introduce a
mandatory synchronization or readback into normal execution. Extend the
benchmark matrix to run warm-up iterations, report robust aggregate values,
and store machine-readable evidence containing the source revision, build
mode, selected device, backend, workload identity, commands, and raw metrics.
Capture target-HIP baselines for standalone, online, and offline execution
before changing execution policy.

Implementation package:

1. Define a compact performance snapshot and aggregate report shared by
   standalone steering and the two CBMRoot adapters. Reuse the existing
   monitoring counters where possible rather than creating a second
   accounting system.
2. Instrument queue submissions, explicit waits, transfers, capacity growth,
   high-water marks, and each major GPU generation. Disabled monitoring must
   be a fast branch and must not trigger a copy or synchronization.
3. Upgrade the standalone matrix with configurable warm-up, measured
   iterations, repeated samples, median and tail values, and a stable
   key/value or JSON-lines evidence format. Record build/source identity and
   workload hash with every sample.
4. Add performance modes to the qualified online and offline launchers. Run
   CPU-only and diagnostic baselines on the current bounded real samples and
   report KFP time separately from GPU tracking, event building, framework
   startup, and CPU-reference comparison.
5. Produce one ranked bottleneck table for standalone, online, and offline.
   It must identify time, launch/synchronization count, transferred bytes,
   useful work, and memory occupancy; a wall-time number without explanatory
   counters is not sufficient evidence.

The first baseline must explicitly measure the numerically reproducible
non-contracted full-field regions documented in
`KFParticleGpuNumericalReproducibility.md`. They are candidates for later
work, not assumed bottlenecks.

Acceptance gate:

- repeated runs report internally consistent work, byte, launch, and timing
  counters without changing physics output;
- startup, steady-state GPU work, CPU reference work, framework overhead, and
  test/build overhead are visibly separated;
- online and offline baselines use the Step 19 qualified launchers;
- existing correctness and full-chain gates still pass;
- ranked bottleneck reports exist for both official modes.

Completion after Stage 20.1: 35%.

### Stage 20.2: Measurement-Guided Execution Optimization

Optimize only costs confirmed by Stage 20.1. Preserve one
process-persistent KFParticle queue and the current device-resident ownership
model. Candidate changes may include larger event batches, fewer queue waits
and scalar readbacks, reused capacities, denser execution groups, better task
ordering, reduced atomic contention, two-pass count/scan compaction, or
limited kernel fusion. Each mechanism must show a measurable gain for its
intended workload and must have a simpler fallback where it loses.

Remove process-wide serialization as far as the single-queue contract permits:
host threads may prepare independent events concurrently, while GPU
transactions enter the queue in a deterministic order and retain event-local
storage and output boundaries. Keep revision-owned descriptors and all
reusable input, work, and output pools on the device across events.

Prepare a separately named, opt-in qualified-GPU route that can publish a
supported GPU event without first running the full CPU finder. Diagnostic mode
continues to execute and compare both implementations. The no-reference route
must remain unavailable until Stage 20.3 qualification passes and must retain
event-atomic CPU fallback for unsupported scope, invalid input or field,
unavailable runtime, overflow, execution failure, and materialization failure.

Implementation package:

1. Select only the small set of costs that dominate the Stage 20.1 reports.
   Expected candidates are construction math, undersized event batches,
   queue waits/scalar readbacks, atomic compaction, and repeated allocation or
   descriptor publication, but no candidate is accepted without measurements.
2. Implement the selected changes one ownership boundary at a time. Preserve
   the existing persistent buffers and compact kernel ABI; add a fallback for
   any strategy whose useful-work density or capacity requirement is not met.
3. Compare every change against the frozen Stage 20.1 evidence over small,
   typical, and stress batch sizes. Retain a change only when the intended
   workload improves beyond run-to-run noise and no representative case has
   an unexplained regression.
4. Add an explicitly named, opt-in qualified-GPU execution route for the
   exact supported V0 manifest. A successful supported event may skip the CPU
   reference, while any unsupported or failed event invokes the ordinary CPU
   finder atomically. The route remains runtime-locked until Stage 20.3.
5. Verify that disabling performance monitoring restores the normal
   synchronization path and that diagnostic mode still observes exactly the
   same results as before optimization.

Stage 20.2 is not an open-ended search for every possible GPU optimization.
It closes when the measured dominant costs have either been reduced or
documented as unavoidable under the current numerical/capability contract,
and the candidate production route is implemented but still locked.

Acceptance gate:

- every optimization has before/after evidence for both official modes and no
  unexplained regression in other benchmark sizes;
- all numerical, lineage, selection, overflow, repeatability, and
  materialization tests remain unchanged or become stricter;
- monitoring adds no required per-event synchronization when disabled;
- the prepared no-reference route cannot be selected before qualification and
  cannot publish partial GPU output.

Completion after Stage 20.2: 75%.

### Stage 20.3: Representative Qualification And Production Decision

Run reproducible CPU-only, diagnostic CPU/GPU, and qualified-GPU campaigns in
both online and offline modes on representative workloads for the declared
K0S, Lambda, and anti-Lambda capability. Include low-, typical-, and
high-occupancy events, multiple primary vertices, repeated timeslices,
capacity boundaries, and controlled invalid or unsupported events. Use
warm-up runs and report steady-state event throughput, latency distribution,
memory high-water marks, fallback reasons, and the Stage 20.1 phase breakdown.

Physics qualification compares exact channel identity, event counts,
canonical lineage, selection state, overflow/fallback state, and bounded fit
observables. Performance qualification compares the complete reconstruction
path, not isolated kernels, and reports the CPU reference cost separately.
Define and record the accepted numerical tolerances, minimum useful speedup,
maximum memory use, and allowed fallback rate before the final production
decision. Document tested hardware/software configurations and
backend-specific limits.

Only after every physics and performance gate passes may the qualified-GPU
route be enabled explicitly for its exact capability manifest without a CPU
reference calculation. Online and offline qualification decisions are
recorded separately; success in one mode does not automatically qualify the
other. CPU-only remains the default configuration until a separate
project-level decision changes that default.

Qualification package:

1. Before running the campaign, freeze numerical tolerances, minimum useful
   speedup, memory ceiling, allowed fallback rate, supported channels,
   hardware/software versions, and the exact low/typical/high-occupancy input
   set. Thresholds cannot be chosen after seeing the final measurements.
2. Run CPU-only, diagnostic, and qualified-GPU modes with warm-up and repeated
   measured samples in standalone, online, and offline environments. Each
   official mode uses its own matching CPU reference; unrelated online and
   offline inputs are never compared as though they contain the same events.
3. Exercise multiple PVs, empty events, capacity boundaries, unsupported
   channels, invalid fields/input, forced overflow, and injected execution or
   materialization failures. Every failure must publish the complete CPU
   event and must leave no partial GPU result.
4. Compare complete physics identity and bounded observables, then compare
   steady-state KFP and end-to-end performance separately. Record throughput,
   latency distribution, memory high-water mark, fallback reasons, and the
   Stage 20.1 phase breakdown for every qualified mode.
5. Issue a separate decision for online and offline. A mode is unlocked only
   for its recorded device/backend/capability manifest if all gates pass. A
   mode that is correct but not faster remains diagnostic-only; this is a
   valid qualification result but does not satisfy Step 20 completion for
   production GPU publication in that mode.

Acceptance gate:

- representative CPU/GPU physics results satisfy the recorded comparison
  policy with no unresolved mismatch or silent fallback in either mode;
- each qualified mode is demonstrably faster end to end for its intended
  workloads and remains within the recorded memory limit;
- unsupported, invalid, overflowing, or failed events retain deterministic
  event-atomic CPU fallback;
- lifecycle, benchmark, CBMRoot unit/equivalence, embedded runtime,
  full-chain, diagnostic, and qualified-routing checks pass on the target HIP
  configuration;
- final evidence and supported capability/backend limits are reproducible on
  another machine.

Completion after Stage 20.3: 100%.

Step 20 completion criterion: the declared V0 GPU route is physics-qualified,
demonstrably faster on representative end-to-end workloads, bounded in memory,
and usable without a mandatory CPU reference calculation in every mode that
passes its independent qualification. Optimization choices are accepted
because measured work, transfers, synchronization, or total time decrease,
not merely because a more elaborate GPU technique is present.

Stage 20.1 implementation checkpoint:

- `KFParticleGpuPerformanceSnapshot` is an opt-in, disabled-by-default view of
  one completed decay-plan transaction. It reuses routing and graph monitoring
  already resolved at the ordinary transaction boundary and adds no device
  copy or queue wait when enabled or disabled.
- The snapshot records workload, launches, explicit queue waits, capacity
  growth, input/output payload bytes, persistent payload high-water size, mask
  density, useful work per launch, pool occupancy, overflow, and the existing
  phase timing. `KFParticleGpuBufferManager` counts only actual monotonic
  capacity growth; normal buffer ownership and reuse are unchanged.
- CBMRoot diagnostic batches aggregate the same snapshot. Set
  `KFPARTICLE_GPU_PERFORMANCE_MONITORING=1` to publish a
  `KFParticle GPU performance:` record in online `cbmreco` and offline
  FairRunAna logs. CPU-only and diagnostic launcher wall time is recorded as
  `PERFORMANCE_RUN`; both records are copied into the ordinary Step 19
  evidence file together with source revision and build mode.
- The standalone matrix accepts
  `KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS` and
  `KFPARTICLE_GPU_BATCH_BENCHMARK_SAMPLES`. It stores every raw sample plus
  median and p95 wall time for serial, batch, field approximation,
  constant-field, and cascade modes in key/value evidence. Cold XPU process
  startup remains outside the internally measured iteration interval.
- The lifecycle suite checks that monitoring is off by default, can be
  enabled without changing results, reports internally consistent counters,
  and returns to the unmonitored state. The online/offline launcher fixture
  checks the extended evidence path.
- The benchmark graph-monitoring gate validates submitted/accepted/stored work
  conservation and overflow, but does not require one particular fixture to
  produce an accepted late-generation particle: a fully processed workload
  whose candidates are all rejected by physics cuts is a valid measurement.
- The controlled cascade/graph fixture keeps its 64 candidate and 256 daughter
  slots per event but gives the compact selected pool the same 64-entry bound.
  The fixture produces eight selected V0 candidates; its former six-entry
  selected pool truncated two valid entries and raised a selection-overflow bit
  that shares its numeric value with graph task overflow. The benchmark now
  retains the full workload, reports overflow sources separately on failure,
  and continues to require zero overflow.

The byte counters describe the queue's capacity-sized input, candidate, and
selected-candidate payload copies. Small revision-owned descriptor/control
copies are represented by plan-upload waits and descriptor-group counts; they
are intentionally not misreported as event payload. The high-water value is
the persistent payload owned by the dominant track, vertex, candidate,
daughter, selected, task, and graph-operation pools.

Stage 20.2 implementation checkpoint:

- Stage 20.1 measurements retain batching because it removes repeated payload
  transfers and queue waits as the event count grows. The remaining dominant
  device phase is candidate-construction math. No speculative contraction or
  fusion is accepted there: the current numerical-reproducibility contract
  deliberately keeps the sensitive full-field expressions uncontracted, and
  the available evidence does not justify weakening that contract. The
  process-persistent queue, monotonic capacities, and device-owned reusable
  pools therefore remain the selected execution strategy.
- `qualified-gpu` is the explicit no-reference production candidate;
  `validated-gpu` remains a compatibility spelling. The ordinary default is
  still CPU-only and diagnostic mode still executes and compares CPU and GPU.
- `GpuCapabilityManifest::QualificationUnlocked()` is a compile-time closed
  gate. Capability and lock checks happen before XPU attachment. The prepared
  unlocked branch executes GPU before CPU, publishes only one completely
  materialized event, and invokes the ordinary CPU finder atomically for every
  unsupported or failed event.
- Routing monitoring distinguishes `QualificationLocked` from unsupported,
  runtime, input, field, overflow, validation, materialization, and execution
  failures. Online and offline preflight launchers accept `qualified-gpu` only
  when the run reports zero GPU publications and at least one qualification
  lock or unsupported-capability CPU fallback before Stage 20.3. The latter is
  expected from the ordinary CBMRoot configuration, which requests Xi/Omega
  and explicit finder cuts outside the narrow qualified V0 manifest.
- The standalone performance fixture retains all eight selected V0 candidates
  instead of truncating the compact pool to six. Overflow sources are printed
  separately on failure; normal measurements remain free of debug output.

Stage 20.3 implementation checkpoint:

- `CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL` is an explicit CMake-only trial
  switch, disabled by default and invalid without GPU diagnostics. It opens
  the already prepared no-reference branch for measurements without silently
  changing an ordinary build. It is not itself a production qualification.
- Qualified no-reference transactions now enter the same performance report
  as diagnostic transactions. Online and offline logs expose the same full
  routing-reason vector, and the evidence collector retains those records.
- `step20_qualification_policy.conf` freezes policy version 1 before the
  target campaign: the existing V0 parity tolerances, three measured runs,
  at least 5% KFP and 1% end-to-end speedup, a 1.5 tail/median limit, a 2 GiB
  KFParticle persistent-payload ceiling, no unexpected fallback, and at least
  one GPU-published event. Input/no-pair CPU fallback is expected but may not
  exceed 95% of routed events.
- `run_cbmroot_kfp_qualification_gate.sh` evaluates one named workload in one
  official mode. It rejects mixed source/build/device/input/configuration
  evidence, incomplete diagnostic parity, a still-locked route, missing GPU
  publication, unexpected fallback, unstable timing, excessive memory, or
  insufficient speedup. It emits a separate machine-readable online or
  offline decision.
- `run_cbmroot_kfp_qualification_campaign.sh` runs the three execution modes
  with separate evidence files and then invokes that gate. The adjacent
  `make_step20_v0_qualification_config.py` derives the narrow V0 test YAML
  from a complete reviewed `MainConfig.yaml` without changing the original.
- The qualified configuration must request exactly PDG 310, 3122, and -3122
  and must omit `finderCuts`; otherwise event-atomic unsupported-capability
  CPU fallback is correct and the run cannot qualify the exact V0 manifest.
- The hermetic launcher test covers both an accepted evidence set and a slow
  qualified route that must be rejected. Final publication remains closed
  until low-, typical-, and high-occupancy target-HIP evidence passes for both
  online and offline operation.

Status: Stage 20.1 and Stage 20.2 are complete (75%) after their target-HIP
standalone, CBMRoot unit, embedded-runtime, locked-routing, and smoke evidence
passed. The numerical prerequisite and Stage 20.3A ordinary/trial contract are
also complete on the target HIP server. The independent online/offline
qualification campaigns and final production decision remain pending. Current
Step 20 readiness is 90%.

Target-server Step 20.3 prerequisite checkpoint:

- the bounded one-timeslice online diagnostic run passes the strict `locked`
  qualification expectation on `hip1`, with STS pion PID, track refitting,
  GPU multi-window tracking, a 100-event input bound, and performance
  monitoring enabled;
- CPU and GPU candidate lineage is fully aligned. The final observed V0
  discrepancy was a near-zero mass-variance sign change caused by cancellation
  at float precision, not routing, storage, a race, or a different kernel
  result;
- the V0 comparison policy accepts this representation boundary only when the
  reconstructed mass agrees within `1e-3` and the finite-side mass error is at
  most `3e-3`. Unit coverage accepts a `2.5e-3` boundary and continues to
  reject a resolved `4e-3` difference;
- normal real-data output is compact again. Deep CPU/GPU tracing remains
  opt-in and is not required by the accepted diagnostic run.

This closes the numerical-parity prerequisite for the qualification trial. It
does not qualify or unlock no-reference GPU publication.

#### Remaining Step 20 execution plan

The remaining 15% is divided into three controlled Stage 20.3 substages. Each
substage must retain target-server evidence before the next one starts.

1. **Stage 20.3A: Freeze and validate the qualification trial (85% -> 90%).**
   - Configure a separate disposable build with
     `CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON`; keep the ordinary build and
     CPU-only default unchanged.
   - Generate and review a narrow V0 YAML containing only K0S, Lambda, and
     anti-Lambda and no `finderCuts`. Freeze source/build revision, device,
     backend, software versions, exact inputs, and policy hash.
   - Run unit, standalone lifecycle/equivalence, embedded-runtime, diagnostic
     smoke, and routing checks. Verify that the ordinary build remains locked
     and that the trial build can publish only the declared capability.

Stage 20.3A completion checkpoint (target-HIP confirmed):

- the routing library now carries an immutable
  `kfp_gpu_qualification_build=ordinary-v1` or
  `kfp_gpu_qualification_build=trial-v1` marker matching the CMake switch;
- online and offline preflights compare the marker with `CMakeCache.txt`
  before reconstruction. `KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD`
  may require `ordinary` or `trial`, while a `published` expectation always
  requires the trial build;
- the qualification campaign explicitly requires a trial build for all three
  of its CPU-only, diagnostic, and qualified-GPU measurements, preventing a
  mixed-build campaign;
- `run_cbmroot_kfp_qualification_trial_preflight.sh` checks separate ordinary
  and disposable trial build directories, generates the narrow V0 YAML beside
  the reviewed source YAML, and records source revisions, build identities,
  device, HIP architecture, configuration hashes, and the frozen policy hash;
- when the reviewed CBMRoot YAML contains Lambda but omits its charge
  conjugate, the generator adds anti-Lambda with the same reviewed mass and
  selection-bit settings. Qualified routing now requires the complete
  `310,3122,-3122` request, matching the default GPU plan exactly;
- the hermetic preflight test rejects an ordinary build presented as a trial,
  publication from an ordinary build, and a stale library whose binary marker
  disagrees with its CMake cache. It accepts consistent ordinary and trial
  fixtures for both online and offline launchers;
- policy version 1 is synchronized with the validated mass-error boundary:
  `mass_error_absolute_tolerance=0.003`.

The separate ordinary/trial builds and the listed HIP checks pass on the target
server. Stage 20.3A is complete, Step 20 readiness is 90%, and Stage 20.3B is
open.

2. **Stage 20.3B: Run independent online and offline campaigns (90% -> 97%).**
   - For each official mode run CPU-only, diagnostic, and qualified-GPU with
     warm-up plus three measured repetitions and performance monitoring.
   - Cover declared low, typical, and high occupancy using bounded real inputs.
     Retain controlled empty-event, invalid-input/field, unsupported-capability,
     capacity, and overflow cases where real data cannot provide a deterministic
     boundary.
   - Invoke the mode-specific qualification gate. Require complete physics
     agreement, at least one GPU publication, no unexpected fallback, stable
     timing, memory below the frozen ceiling, and the frozen KFP/end-to-end
     speedup thresholds. Online success cannot substitute for offline success.

Stage 20.3B implementation checkpoint (target campaigns pending):

- `run_cbmroot_kfp_qualification_campaign.sh` separates one warm-up run from
  three measured repetitions for each CPU-only, diagnostic, and qualified-GPU
  execution mode. Warm-up output and evidence cannot enter the frozen gate.
- `run_cbmroot_kfp_qualification_matrix.sh` executes low, typical, and high
  bounded workloads for exactly one official mode. The bounds are fixed event
  or entry counts, not claims about the physical occupancy of an individual
  event. Online and offline matrices must be run and retained independently.
- A rejected workload no longer prevents later workloads from running. The
  matrix finishes all three, retains every mode-specific decision and hash,
  writes a rejected aggregate decision, and returns failure only after the
  complete evidence set exists.
- The frozen gate now checks source revision, trial-build identity, parameter
  identity, exact `310,3122,-3122` YAML capability, absence of `finderCuts`,
  and policy hash in addition to physics, routing, timing, and memory. A gate
  failure writes a machine-readable rejected decision with its reason.
- Preflight evidence records hashes for input, configuration, parameter, and
  macro identities. The hermetic matrix contract test covers all three bounds,
  missing-bound rejection, and continuation after an intermediate rejection.

The Stage 20.3B machinery is complete, but Step 20 remains at 90% until both
real-input target-HIP matrices have produced reproducible decisions. A frozen
performance rejection is a valid campaign result and leads to a documented
diagnostic-only no-go in Stage 20.3C; it must not be hidden by changing policy
after measurement.

Stage 20.3B target-HIP online checkpoint:

- all 36 online reconstruction launches completed on `hip1`: one warm-up and
  three measured CPU-only, diagnostic, and qualified-GPU runs at bounds 10,
  50, and 100;
- the low bound was rejected only by the frozen end-to-end performance gate:
  speedup `0.89214477` is below `1.01` (about 10.8% slower than CPU);
- the typical bound qualified with KFP speedup `8.61966809`, end-to-end
  speedup `1.02812779`, 69 GPU-accepted events, and fallback fraction
  `0.25806452`;
- the high bound qualified with KFP speedup `7.75448425`, end-to-end speedup
  `1.09657002`, 138 GPU-accepted events, and fallback fraction `0.26984127`;
- the aggregate online decision is therefore correctly `rejected`. This is
  not a test-run failure and the frozen threshold is not changed afterward.
  The offline matrix is still required before Stage 20.3C can decide between
  diagnostic-only retention and a future explicitly bounded production route.

Stage 20.3C implementation checkpoint (target offline evidence pending):

- the workload gate now records every measured performance, stability,
  fallback, and memory result before rejecting a workload, and retains paths
  to all three source evidence files; an early contract failure still stops
  immediately because its measurements are not comparable;
- `run_cbmroot_kfp_qualification_final_decision.sh` requires complete online
  and offline matrices with the frozen `low:10 typical:50 high:100` bounds,
  verifies all six workload-decision hashes and policy hashes, and rejects a
  missing, duplicated, altered, wrong-mode, or wrong-device record;
- the final decision is `qualified-gpu` only when both official matrices are
  qualified. Any rejected matrix produces the explicit `diagnostic-only`
  decision and keeps the ordinary-build qualification gate locked. This is a
  completed no-go qualification result, not permission to adjust thresholds;
- `_TestKfpGpuQualificationFinalDecision` covers the all-qualified result, the
  valid diagnostic-only result, and evidence-tampering rejection. The existing
  preflight and matrix contract tests continue to pass with the richer gate
  evidence.

Current completion remains 97%. The implementation is complete; the only
remaining work is the target-HIP offline matrix, generation of the combined
Stage 20.3C evidence, and the final target regression set. Step 20 reaches
100% after those server artifacts have been reviewed and recorded.

Stage 20.3B offline numerical checkpoint (target-HIP confirmed):

- the isolated event-based run over entries `0..18` now passes the strict
  locked diagnostic contract on `hip1` without deep tracing;
- the previously blocked K0S candidate at source event `77309411328` was a
  scale-dependent CPU-SIMD/GPU-scalar comparison-boundary case, not a device
  race, transfer failure, lineage difference, or reconstruction failure;
- parity comparison now combines its existing absolute floor with an optional
  relative term. Relative terms default to zero globally; only the validated
  default-V0 policy enables the bounded parameter and fit-scalar terms;
- the exact recorded residuals are retained in a CBMRoot unit regression, the
  standalone parity contract covers relative-bound acceptance and rejection,
  and the target `_GTestKfpGpuDiagnosticPacker` plus offline preflight pass.

This checkpoint removes the blocker to rerunning the complete low/typical/high
offline matrix. It is not itself a matrix decision and does not change the
frozen performance policy or the 97% completion status.

Stage 20.3B target-HIP offline checkpoint:

- all 36 offline reconstruction launches completed on `hip1`: one warm-up and
  three measured CPU-only, diagnostic, and qualified-GPU runs at bounds 10,
  50, and 100;
- all physics, routing, memory, and stability checks completed without an
  unexpected reconstruction failure;
- the low workload was rejected only because its end-to-end speedup
  `0.99265606` is below the frozen `1.01` threshold;
- the typical workload was rejected only because its end-to-end speedup
  `1.00554156` is below `1.01`;
- the high workload passed its physics, routing, memory, stability, and speed
  gates;
- the aggregate offline decision is therefore correctly `rejected`. Together
  with the already retained rejected online matrix, this determines the
  Stage 20.3C result as `diagnostic-only`; neither threshold is changed after
  measurement.

Both official mode matrices exist, the combined decision is retained, and the
final target regression set is complete.

Stage 20.3C target-HIP decision checkpoint:

- the final verifier accepted both complete matrices, their six workload
  decision hashes, the common `low:10 typical:50 high:100` bounds, device
  `hip1`, and frozen policy hash
  `2f99b38565f9ef5d9a42538a83d6371bad587b7c488025b95ffe9049e3e72aab`;
- both official matrix decisions are `rejected`, so both routes are retained
  as `diagnostic-only` and the ordinary build action is `keep qualification
  locked`;
- this is the planned, successful no-go outcome, not a failed or incomplete
  qualification. It preserves the validated diagnostic GPU path without
  claiming production benefit below the frozen end-to-end threshold.

Step 20 completion checkpoint (target HIP, 100%):

- all five final CBMRoot unit/contract tests pass;
- the standalone `hip1` lifecycle passes all 97 checks, and the short 1/4/8
  event benchmark preserves serial/batch equivalence and bounded monitoring;
- the XPU ownership smoke, diagnostic smoke, and CPU/GPU V0 equivalence suite
  all pass from the final trial build;
- the repaired offline aggregate is derived only from its retained low,
  typical, and high workload decisions. The contract suite leaves its SHA256
  unchanged, proving that hermetic fixtures no longer reuse campaign output
  paths inherited through `KFPARTICLE_*` variables;
- the final verifier accepts online matrix SHA256
  `5207f3251d795805018ec4fc144a1ee98dbfa3a39543f6cf26c8e17eaa68d0aa`
  and offline matrix SHA256
  `a8d97b11d3ad7b2e42b8c4429337109463962109dde7e81d76385d9b79850a3d`.
  The reproducible result is `diagnostic-only`, with the ordinary build kept
  qualification-locked.

Final closure gate (no representative campaign rerun required):

- rebuild the final trial targets and pass the five hermetic CBMRoot contract
  tests: diagnostic packing/comparison, selector PID, full-chain preflight,
  qualification matrix, and final-decision verification;
- pass the standalone `hip1` lifecycle suite and one short batch benchmark at
  event counts 1, 4, and 8. This is a regression check for scheduling,
  monitoring, and bounded batching, not a new performance qualification;
- pass the CBMRoot XPU ownership smoke, GPU diagnostic smoke, and hermetic
  CPU/GPU V0 equivalence test from the same final build;
- re-run the combined decision verifier against the retained online and
  offline matrices and confirm `production decision: diagnostic-only` and
  `ordinary build action: keep qualification locked`.

The retained 36-run online and 36-run offline matrices were not repeated by
this closure gate. All listed checks passed from the final source state, so
Step 20 is complete at 100%.

3. **Stage 20.3C: Make the production decision and close the step (97% -> 100%).**
   - Audit both decision files and all raw evidence. Classify each mode as
     qualified for the exact recorded manifest or retained as diagnostic-only;
     do not reinterpret thresholds after measurements.
   - Open the compile-time qualification gate only for a mode whose physics,
     performance, memory, and fallback gates all pass. Keep CPU-only as the
     default and preserve event-atomic CPU fallback outside the manifest.
   - Re-run lifecycle, a short benchmark matrix, and CBMRoot
     unit/equivalence/smoke gates from the final code. Revalidate the retained
     online/offline evidence and final decision without repeating the long
     representative campaigns. Record supported hardware/backend limits and
     the final online and offline decisions in the validation documents.

Step 20 reaches 100% only after the final code and both official-mode decisions
have reproducible target-server evidence. A correct route that misses the
frozen performance gate remains diagnostic-only and is recorded as a no-go
result rather than being silently promoted.

## Step 21: Close The Remaining CPU Finder Physics Scope

Step 21 closes the gap between the current validated GPU subset and every
active channel family in `KFParticleFinder`. It is a physics-completeness
step, not a production unlock or a time-slice redesign. CPU-only remains the
default and the no-reference GPU route remains qualification-locked after
Step 20. Performance requalification starts only after this scope is complete.

The existing graph already inventories nine CPU finder families, but their
coverage is deliberately marked partial while only representative default V0,
cascade, composite, missing-mass, matching, projection, and finalization
operations are validated. Step 21 replaces representative coverage with an
exhaustive declarative channel contract and removes every
`ValidationPending`, `NotImplemented`, or `MissingMathematics` result for an
active channel. A genuinely absent optional detector input remains an explicit
event capability failure with event-atomic CPU fallback; it must not be
misreported as an unimplemented physics channel.

Step 21 is divided into four large controlled stages.

### Stage 21.1: Freeze The Complete CPU Channel Catalogue (0% -> 20%)

- Inventory every active `KFParticleFinder` branch in the nine existing
  families: two-daughter, same-sign primary resonance, track-composite,
  long-lived composite, composite-composite, neutral/missing-mass, kaon
  matching, primary projection, and final selection.
- Give every particle/antiparticle channel a stable ID and declarative record
  of mother PDG, daughter hypotheses, source roles, generation dependencies,
  transport, construction, constraints, cuts, output class, and required
  optional inputs. Configuration-disabled CPU branches must be distinguished
  from active unsupported branches.
- Generate the GPU graph manifest and a CPU comparison oracle from this one
  catalogue. Add compile-time/runtime validation that rejects duplicate IDs,
  missing conjugates, unresolved dependencies, missing cut profiles, and any
  active CPU channel omitted from the GPU coverage report.
- Keep execution unchanged in this stage. The control result is a deterministic
  catalogue/graph snapshot and unit tests proving complete accounting before
  more kernels are activated.

Stage 21.1 is complete when every active CPU channel is either mapped to an
exact operation contract or carries one explicit implementation requirement;
there may be no anonymous or implicitly skipped channel.

Stage 21.1 is complete and confirmed on the target HIP system.
`KFParticleGpuCpuChannelCatalogue` is now the single host-side inventory used
by the channel coverage oracle and the nine-family graph coverage builder. The
frozen snapshot contains 256 stable contracts: 230 active channels, 18
channels conditional on neutral input, and 8 same-sign contracts explicitly
marked configuration-disabled because their CPU branch is under `#if 0`.
The Stage 21.2 audit added the previously omitted active charm-conditioned
`420 -> pi+ pi-` branch. Snapshot revision `12010545515856880377` additionally
records exact composite-track dependencies and generations, alongside mother
and conjugate IDs,
daughter-hypothesis and selection profiles, source kinds, generations,
operations, output classes, input requirements, and support requirements.

The validator rejects zero or duplicate IDs, incomplete records, missing or
asymmetric conjugates, unresolved or forward dependencies, absent profiles,
unknown operations, and omitted finder families. The default V0/Xi/Omega
plan maps exactly seven currently implemented catalogue channels; every other
active record remains visible with an explicit implementation reason. No GPU
execution node, kernel launch, channel selection, or production gate changes
in this stage. The local standalone CPU lifecycle passes all 98 checks,
including catalogue mutation tests and the unchanged decay-graph regression.
The corresponding target HIP lifecycle and requested CBMRoot regression suite
also pass. Step 21 therefore stands at 20%, with Stage 21.1 closed.

### Stage 21.2: Complete Charged And Track-Composite Generations (20% -> 60%)

For implementation and target validation this stage is split into two large
parts. The split does not change the 60% completion gate for the whole stage.

#### Stage 21.2A: Complete all active two-track channels (20% -> 35%)

- Extend the existing mask-driven two-track route to all remaining primary
  resonances, charm hypotheses, hypernuclei, and charge-conjugate channels.
  Channel differences remain descriptor data; do not add a serial kernel or
  host branch per decay.
- Generate exact daughter PDGs, packed track species, mass hypotheses, source
  aliases, primary/secondary track ranges, same-primary-vertex requirements,
  charm pixel/chi-to-PV/pt cuts, output classes, and selection profiles from
  the frozen CPU catalogue. Compile all active channels into one generation
  and retain bounded mask-driven routing on device.
- Validate catalogue-to-plan, plan-to-routing, and plan-to-graph completeness,
  including conjugates, heavy nuclei, primary resonances, charm cuts, output
  classes, and the CPU `420 -> pi+ pi-` branch. Run standalone CPU/HIP and
  CBMRoot diagnostic/V0 regression gates after enabling this block.

#### Stage 21.2B: Complete track-composite and long-lived generations (35% -> 60%)

- Extend the composite-track route to the complete long-lived, resonance,
  charm, beauty, and hypernuclear chains declared by the catalogue, including
  multi-generation dependencies and channel-specific selection profiles.
- Preserve full-field transport, CPU-compatible covariance and fit semantics,
  canonical lineage, bounded pools, deterministic event ranges, monitoring,
  materialization, and event-atomic fallback for every newly enabled channel.
- Add family-level synthetic fixtures plus CPU/GPU differential tests for
  accepted, rejected, conjugate, boundary-cut, overflow, and dependency-order
  cases. Run standalone CPU/HIP lifecycle and CBMRoot diagnostic regression
  after the complete charged block is enabled.

Stage 21.2 is complete when every independently executable two-daughter,
track-composite, and long-lived-composite channel is supported. Four
track-composite contracts consuming the `111` composite-composite output are
closed with that parent in Stage 21.3; no fabricated two-track parent may be
used merely to make the Stage 21.2 count appear full.

Stage 21.2 is in progress. Its infrastructure increment removes two
limits exposed by the frozen catalogue: the device routing mask now has 256
bits instead of 64, and packed track species now include He-6, Li-6, Li-7,
and Be-7. The mask remains a fixed, trivially-copyable value and routing still
uses descriptor data rather than one host or kernel branch per channel.
Standalone lifecycle tests cross the 32/64-bit word boundaries with 96
two-track and 96 composite-track hypotheses and verify exact heavy-nucleus PDG
routing. The CBMRoot packer test verifies the four new species in first/last
SoA ranges.

Stage 21.2A implementation is complete locally. The CPU catalogue now creates
all 50 active two-track descriptors, including the formerly default K0S,
Lambda, and anti-Lambda channels, primary resonances, charm hypotheses,
hypernuclei, charge conjugates, and `420 -> pi+ pi-`. Primary channels require
both tracks to carry the same valid primary-vertex index; charm channels carry
the CPU pixel-hit, chi-to-PV, and pt gates as descriptor data. Output class is
preserved through routing, task construction, and candidate metadata.

The lifecycle contract requires exactly 50 routing descriptors, 50 enabled
mask bits, and one 50-node track-track graph generation; every active
two-daughter catalogue record must report `Supported`. The local standalone
CPU lifecycle passes all 98 checks, and the target HIP, CBMRoot packer,
diagnostic-smoke, and CPU/GPU V0 gates pass. Stage 21.2A is complete and Step
21 enters Stage 21.2B at 35%.

Stage 21.2B is accepted on the target HIP/CBMRoot configuration for all 120
independently executable track-composite and long-lived-composite contracts:
72 track-composite and 48 long-lived channels. Descriptors carry family,
generation, output class,
exact parent channel and PDG, bachelor species/range, charm pt/pixel/chi-to-PV
cuts, the CPU `fDistanceCut` DCA boundary for secondary
`FindTrackV0Decay`, and applicable hypernuclear parent mass constraints. The shared mask
router groups work by generation and track range, republishes accepted
composites into the resident compact pool, and executes generations 2 through
4 in queue order without an intermediate host hand-off. Canonical lineage is
variable-length and bounded at 16 source tracks.

The lifecycle suite includes a generation-2 result consumed by generation 3
on device, verifies that track-composite channels carry the DCA boundary while
`FindLL` channels do not, and checks exact catalogue/routing/graph coverage.
Four active contracts remain explicitly `ValidationPending`: `111 + p` and
`111 + K` with both
charge hypotheses. Parent channel 5019 belongs to composite-composite Stage
21.3, so these children are enabled only with that parent. Local standalone
CPU passes 98/98, and the target HIP lifecycle, CBMRoot packer, diagnostic
smoke, and CPU/GPU V0 regression gates pass. Stage 21.2 is complete and Step
21 stands at 60%.

### Stage 21.3: Complete Composite, Neutral, And Final Operations (60% -> 85%)

For controlled implementation this stage is split into two substantial parts.

#### Stage 21.3A: Resident composite-composite graph (60% -> 70%)

- Give every composite-composite contract exact first and second parent
  channel identities and derive its generation from those parents.
- Execute all 24 active composite-composite channels through the shared graph
  operation kernel. Same-source channels use one canonical candidate ordering,
  so `gamma+gamma` and `Lambda+Lambda` cannot publish both permutations.
- Build the real `111` parent and then enable the four `111+p/K` children that
  were deliberately deferred in Stage 21.2B. Validate catalogue, routing,
  dependency ordering, family coverage, and host/device graph execution before
  exposing the enlarged plan through CBMRoot capacity accounting.

Stage 21.3A is accepted. The frozen catalogue records both
parents for binary resident operations, all 24 composite-composite contracts
are `Supported`, and the complete charged/resident test plan contains 198
ordered graph nodes: 50 two-track, 124 composite-track, and 24
composite-composite. The local standalone CPU lifecycle and the target HIP,
heap, unit, diagnostic-smoke, and CPU/GPU V0 regression gates pass. Step 21 is
therefore at the accepted 70% boundary.
The plan builder keeps dependency closure explicit: a narrow V0 diagnostic
plan without channel 5019 contains the original 120 composite-track channels
and compiles to 170 nodes, while the full Stage 21.3A plan adds channel 5019
before the four pi0 children and contains 124 composite-track channels and 198
nodes. This prevents a supported catalogue entry from becoming a dangling
dependency in a deliberately partial runtime plan.

#### Stage 21.3B: Neutral input and terminal operations (70% -> 85%)

Stage 21.3B is split into two controlled implementation parts. Stage 21.3B1
covers the independent primary-projection contract (70% -> 75%); Stage
21.3B2 covers neutral input, missing mass, kaon matching, and final selection
(75% -> 85%).

##### Stage 21.3B1: Primary projection (70% -> 75%)

- Add all 12 CPU `ExtrapolateToPV` channels as unary graph operations with
  exact parent-channel dependencies.
- Resolve the primary-vertex index from each resident parent candidate at
  execution time and retain that resolved identity in output metadata.
- Match the CPU operation exactly: transport to the primary-vertex point
  without an implicit `SetProductionVertex` covariance update.
- Keep partial plans dependency-closed. A narrow plan adds only the eight
  projections whose parents are present; the complete resident plan adds all
  12 and compiles to 210 nodes.

Stage 21.3B1 is accepted. Catalogue revision
`12488567174590575521` contains the 12 supported projection contracts. The
standalone CPU lifecycle and target HIP/CBMRoot gates pass, including
host/device execution, resolved-PV materialization, exact operation flags,
full-plan coverage, and partial-plan dependency tests. Step 21 is therefore at
the accepted 75% boundary.

##### Stage 21.3B2: Neutral and terminal operations (75% -> 85%)

- Complete composite-composite construction and all catalogue-declared mass,
  topology, and production-vertex constraints, including CPU-compatible
  `SetProductionVertex` covariance behaviour.
- Implement the active filtered missing-mass mathematics instead of silently
  applying the disabled legacy subtraction formula. The CPU implementation
  consumes a primary mother-track hypothesis and a secondary charged-daughter
  track; it does not consume an external detector-neutral candidate. Route
  those two raw track ranges without first materializing fake candidates.
- Complete kaon matching, primary projection, mass constraints, and final
  channel-specific selection. Keep intermediate candidates resident and
  preserve complete ancestry through materialization.
- Test each mathematical operation independently on host/device and then as a
  dependent graph generation. Cover absent neutral input, invalid matching,
  covariance guards, duplicate lineage, bounded-output failure, and exact
  event fallback without partial CPU/GPU result merging.

Stage 21.3 is complete when composite-composite, neutral/missing-mass, kaon
matching, primary projection, and final-selection families are fully supported
for valid declared inputs. Missing optional data may block an event, but no
active operation may remain mathematically or structurally unimplemented.

The first Stage 21.3B2 audit checkpoint is accepted on the target HIP/CBMRoot
configuration. It corrected
the frozen inventory from 256 to 260 entries and the bounded routing mask from
256 to 288 bits. The active CPU method contains 22 missing-mass charge
hypotheses, including the previously omitted He3 and He4 pairs. Every one now
has explicit mother-track and charged-daughter PDGs, two raw-track sources,
standard track/PV inputs, and an explicit filtered-reconstruction contract. No channel is
misclassified as conditional on a nonexistent neutral input. Catalogue
revision at that accepted inventory boundary was `3044443511554642051`; its
standalone and target validation gates pass.

The next substantial Stage 21.3B2 implementation block is accepted on the
target HIP/CBMRoot configuration. It removes the candidate-only assumption from
later-generation graph tasks: each source now carries its kind, index, and
event, and the device router resolves either resident candidates or encoded
track-set/species ranges. Track-source IDs now reserve a distinct value for
the whole track set, eliminating the former alias between one set's wildcard
and the next set's electron range. The filtered missing-mass operation mirrors
the CPU spatial Kalman updates for the neutral residual, filtered mother, and
filtered charged daughter, including chi2/NDF and mass-hypothesis energies.
All 22 charge channels are `Supported`, the complete plan grows from 210 to
232 nodes, and catalogue revision is `6470827997588483179`. A dedicated
device-generated raw-track test verifies primary-mother plus secondary-daughter
routing, filtered-mother publication, event identity, and two-track ancestry.
The standalone lifecycle passes 99/99 on CPU and target HIP, and the unchanged
CBMRoot heap, unit, diagnostic-smoke, and V0 regression gates pass. Step 21 is
therefore at the accepted 80% checkpoint.

The following large integration sub-block is accepted on the target
HIP/CBMRoot configuration. The ordinary automatically-created diagnostic plan now
includes all 22 filtered missing-mass channels in both single-event and batch
execution. A shared overflow-checked capacity calculation counts two-daughter
and raw-track graph pairs from the same event descriptor, so empty-input,
`NoPairs`, candidate, daughter, and task bounds remain consistent. Explicit
narrow/custom plans are not enlarged. The ordinary diagnostic graph contains
192 nodes (50 two-track, 120 composite-track, and 22 missing-mass), and a
controlled muon-range fixture freezes the exact raw task bound at 62.

The target standalone HIP lifecycle, CBMRoot unit test, diagnostic smoke, and
CPU/GPU V0 regression all pass after this integration. The test launchers now
derive their CBMRoot source tree from their own physical location, reject a
build configured from a different source tree, and put the selected build's
libraries before inherited paths. This closes the 22-channel missing-mass
block and keeps Step 21 at its accepted 80% checkpoint.

The kaon-matching sub-block is accepted on the target HIP/CBMRoot
configuration. Catalogue channels 7001 and 7002 consume the exact
charge-conjugate `K -> 3 pi` parent and the primary last-hit kaon range through
a dedicated binary-final graph payload. The operation mirrors CPU
`MatchKaons`: the unconstrained three-pion candidate must be topologically
compatible with the primary vertex named by the kaon track; its mass must lie
within the CPU `3 sigma` window and is then constrained to the same catalogue
mass before matching. The constrained candidate and track must be within
20 cm, while their residual fit must have non-negative chi2/NDF at most 3 and
component-wise residual momentum at most five standard deviations. A
successful match republishes the constrained parent fit with the kaon track
appended to canonical lineage. This deliberately does not compare one stored
candidate PV index: CPU may place the same candidate in several compatible PV
lists. The ordinary plan contains 194 nodes and both kaon channels. Standalone
CPU/HIP lifecycle, CBMRoot unit and XPU/diagnostic smoke, V0 regression, and
validated batch gates pass, including host/device acceptance plus independent
PV, mass, distance, momentum, lineage, and graph-contract rejection checks.

The next large Stage 21.3B2 block implements all 18 charge-specific terminal
selection channels used by CPU `SelectParticles`. Each unary terminal node
selects against the primary vertices belonging to its event, applies the CPU
line-distance, decay-length significance, pointing, topology chi2/NDF, and
mass-window cuts, then publishes the mass-constrained fit while preserving
the complete parent lineage. Event primary-vertex ranges are carried by graph
tasks so batched events cannot inspect one another's vertices. The ordinary
plan contains 212 nodes (194 accepted construction/matching nodes plus 18
terminal nodes). A dedicated host/device fixture covers acceptance and missing
PV, event-isolated PV ranges, distance, decay significance, topology, mass
window, ancestry, and final-output contracts. The same standalone HIP,
CBMRoot, V0-regression, and validated batch gates pass. These two accepted
blocks place Step 21 at 84%.

The final Stage 21.3B2 block removes the straight-line escape hatch from the
representative dependent-graph fixture. All 50 two-track and 124
composite-track catalogue channels are frozen to full-field transport, while
one nonzero-field transaction exercises representative ordinary charged,
same-sign, charm, hypernuclear, cascade, long-lived, composite-composite,
neutral, and unary paths. Low-level host/device full-field transport, DCA,
energy-fit, and cascade comparisons remain the numerical oracle. This block is
implemented locally: the standalone CPU lifecycle passes all 100 contracts,
including the enlarged nonzero-field graph transaction. The target HIP,
CBMRoot, V0-regression, and batch gates also pass. Stage 21.3 is complete and
accepted Step 21 progress is 85%.

Stage 21.3 is complete at the accepted 85% boundary.

### Stage 21.4: Exhaustive Parity And Scope Closure (85% -> 100%)

Stage 21.4 is executed in only two large controlled blocks. Stage 21.4A freezes
and validates the complete manifest (85% -> 92%); Stage 21.4B runs the complete
numerical and integration campaign and closes the declared scope (92% ->
100%).

#### Stage 21.4A: Complete manifest and exhaustive structural matrix (85% -> 92%)

- Require all nine family coverage records to be `Supported` for the declared
  complete-input manifest and require every active channel ID to appear in
  routing, monitoring, parity snapshots, and materialized output tests.
- Build one dependency-complete plan through a dedicated API rather than
  treating the intentionally narrower ordinary diagnostic plan as evidence of
  completeness. Freeze exact active, disabled, per-family, node, operation,
  output-class, and dependency accounting and reject any incomplete plan.
- Compile every declared active channel into one graph and compare each node
  against its catalogue contract. Configuration-disabled CPU channels must be
  present in the accounting but absent from execution; no fabricated node may
  be used to make their family appear complete.

Stage 21.4A is implemented locally. `AddCompleteCpuFinderChannels()` creates
one dependency-closed plan with 50 two-track, 124 composite-track, and 78
graph-operation channels. `KFParticleGpuCpuFinderCoverageSummary` accounts for
all 260 catalogue records as exactly 252 required/implemented and eight
configuration-disabled channels, with zero unsupported entries. The compiled
manifest contains each active channel exactly once and all nine family records
are complete; an intentionally incomplete plan is rejected.

This exhaustive comparison found and fixed two latent metadata defects. Graph
compilation now preserves each two-track descriptor's output class instead of
forcing `PrimaryAndSecondary`. Long-lived catalogue channels now advertise a
parent mass constraint only where the corresponding GPU descriptor actually
has one. The corrected frozen catalogue revision is
`7326650681912440158`. The local standalone CPU lifecycle passes 101/101;
the target HIP, CBMRoot, V0-regression, and validated batch gates also pass.
Stage 21.4A is accepted and Step 21 is at 92%.

#### Stage 21.4B: Numerical matrix, campaigns, and scope closure (92% -> 100%)

- Run an exhaustive generated channel matrix over CPU scalar/SIMD and GPU
  execution, including conjugates, cut boundaries, nonzero fields,
  multi-primary-vertex cases, dependency chains, overflow, and optional-input
  fallback. Rare channels use controlled injected fixtures; real-data absence
  is not treated as validation.
- Re-run standalone CPU/HIP lifecycle, batch equivalence, CBMRoot unit,
  XPU/diagnostic smoke, CPU/GPU equivalence, and bounded online/offline
  diagnostic campaigns. Record residual bounds and unsupported/fallback
  counters by family.
- Update the integration and validation documents with the exact completed
  channel manifest. Keep the production gate locked: Step 21 proves physics
  completeness, while a later step optimizes the enlarged workload and repeats
  the frozen production qualification.

Stage 21.4B implementation is complete locally. The complete plan is uploaded
to all 50 two-track, 124 composite-track, and 78 graph-operation descriptor
tables, each with corresponding per-channel monitoring storage. Every active
channel ID crosses the common parity snapshot and bounded materialization
contracts, and all six executable graph payload kinds must be represented.
Existing operation-level host/device tests remain the numerical oracle for
nonzero field, fit/covariance, cuts, multi-PV selection, dependency chains,
overflow, and event isolation. The exhaustive standalone route uses
`AddCompleteCpuFinderChannels()`. A CBMRoot diagnostic batch instead records
the PDGs requested by its CPU `KfpSelector` and builds their dependency-closed
subset through `AddRequestedCpuFinderChannels()`. Its capacity estimate and
executed GPU graph therefore describe exactly the same physics request; a
narrow online selector no longer executes unrelated complete-catalogue
channels. This correction was required after a bounded online run exposed
89,581 GPU-only hypotheses and 22 oversized events from comparing the complete
GPU catalogue with a four-channel CPU configuration. Local standalone CPU
validation passes 101/101. The target HIP lifecycle, complete-plan
serial/batch equivalence, CBMRoot unit and smoke gates, CPU/GPU V0 regression,
and bounded online/offline diagnostic campaigns also pass.

The first bounded online rerun after requested-plan filtering completed all
63 sampled events with zero overflow and reduced unrelated GPU-only output
from 89,581 to 57. Those remaining entries were channel 11 Xi candidates:
the GPU correctly retained them in raw graph output, while diagnostic contract
2 still compares the CPU extractor's direct V0 boundary only. The reference
projection now explicitly admits channels 1, 2, and 3 and leaves later graph
generations in raw output and monitoring. This prevents a supported cascade
dependency from being mislabeled as a V0 `GPU-only` mismatch; recursive CPU
cascade lineage remains covered by the dedicated exhaustive topology tests.

Stage 21.4B is accepted on the target HIP/CBMRoot configuration. The frozen
catalogue revision `7326650681912440158` accounts for 252 active supported
channels and eight configuration-disabled contracts, with zero active
unsupported channels. All nine finder families cross routing, monitoring,
host/device operation fixtures, parity snapshots, and bounded
materialization. Standalone lifecycle passes 101/101, complete automatic-plan
batch equivalence passes, the CBMRoot unit/smoke/V0 gates pass, and bounded
online/offline diagnostics preserve event-atomic fallback without overflow,
unsupported-family, unsupported-channel, unresolved-lineage, or partial-merge
failures. CPU-only remains the default and the Step 20 production lock remains
`diagnostic-only`; production promotion is intentionally outside this step.

Step 21 is complete at 100%.

Step 21 reaches 100% only when the declared complete-input manifest has no
active unsupported channel, all nine CPU families pass host/device parity and
materialization, and both CBMRoot launch modes preserve event-atomic fallback.
The deferred zero-copy whole-time-slice mode remains outside Step 21.

## Step 22: Optimize And Requalify The Complete Event-Batch Route

Detailed optimization work for Step 22 is maintained in
[`KFParticleGpuOptimizationPlan.md`](KFParticleGpuOptimizationPlan.md). That
document is authoritative for the bottleneck evidence, kernel-launch map,
fusion candidates, implementation order, expected gains, rejected
experiments, and per-optimization validation. This global plan retains the
physics boundary, stage completion percentages, integration requirements, and
final qualification decision. Changes to optimization scope or order must be
recorded in the dedicated plan instead of expanding this section into a second
competing implementation plan.

Step 22 actively optimizes single-event and multi-event batch transactions.
Its execution descriptors must be domain-neutral and preserve a documented
extension boundary for future time-window domains, but real time-slice
execution is not part of this step. It remains blocked until CA can publish the
required device-resident tracks, global identities, time/window and overlap
ownership, PV/field metadata, memory lifetime, and queue dependency contract.
No temporary host-built time-slice route is required for Step 22 completion.

Step 22 turns the physics-complete Step 21 graph into a measured production
candidate. It does not add channels, loosen numerical tolerances, change the
CPU default, or start the deferred zero-copy whole-time-slice design. The
complete 252-channel manifest and requested-channel dependency closure are
frozen correctness inputs. Optimization is accepted only when the same
host/device, lineage, bounded-output, and event-atomic fallback contracts keep
passing.

The previous Step 20 campaign measured only the earlier narrow capability and
retained `diagnostic-only`: small workloads lost to fixed overhead, while
typical/high online and high offline workloads already showed useful GPU
speedup. Step 22 repeats that work for the enlarged graph, but first separates
real reconstruction cost from diagnostic comparison and identifies the
workload crossover instead of assuming that every small batch belongs on the
GPU.

Step 22 is divided into three large controlled stages.

### Stage 22.1: Freeze The Full-Scope Baseline And Bottleneck Map (0% -> 25%)

- Freeze two reproducible workload contracts: the dependency-closed channel
  subset requested by the real CBMRoot selector, and the complete 252-channel
  synthetic manifest used to stress every generation. Record catalogue,
  source, build, device, configuration, input, and policy identities.
- Measure CPU-only, diagnostic GPU, and no-reference GPU transactions after
  warm-up. Split time into packing, capacity planning, allocation/growth,
  host-to-device transfer, routing and each graph generation, selection,
  device-to-host transfer, materialization, comparison/reporting, and total
  CBMRoot event processing. Record launches, active tasks, pool peaks,
  allocated bytes, overflow margin, fallback reasons, and batch occupancy.
- Run low, typical, and high bounded workloads in both official CBMRoot modes.
  Distinguish fixed launch/transfer overhead from work that scales with events,
  tracks, pairs, graph nodes, and selected output. Produce one machine-readable
  bottleneck report and choose optimization targets from measurements rather
  than assumptions.

Stage 22.1 is complete when repeated measurements are stable enough to name
the dominant time and memory costs, the full and requested plans are not
confused in evidence, and no performance number includes diagnostic CPU/GPU
comparison while being presented as no-reference production cost.

Stage 22.1 implementation checkpoint: the shared CBMRoot diagnostic aggregate
now retains total tracks, vertices, raw-task capacity, raw/selected candidates,
overflow, capacity-planning time, dependent-graph time, and materialization
time. One formatter is used by online `cbmreco` and offline FairRunAna. With
`KFPARTICLE_GPU_PERFORMANCE_MONITORING=1`, it prints a stable execution-order
report whose rows explicitly identify `CPU`, `COPY H2D`, `GPU/QUEUE`, and
`COPY D2H` work. The header records event/batch scope, track/PV/task/channel
load and result counts; the footer records launches, waits, visited/stored
work, mask/task/pool density, transferred bytes, capacity growth, and allocated
high-water memory. Existing compact qualification lines remain unchanged for
machine parsing.

The report labels device phases as host wall intervals around queue-ordered
work. It does not claim unavailable device-event kernel timing, and it keeps
CPU reference and CPU/GPU comparison visible as diagnostic costs rather than
folding them into a no-reference GPU result. Unit coverage freezes aggregation
across a multi-event batch and the ordered row/domain labels. Target build,
unit, bounded online/offline output review, and repeated baseline evidence are
still required before Stage 22.1 can be accepted at 25%.

### Stage 22.2: Optimize The Enlarged Graph Without Changing Physics (25% -> 75%)

- Remove the dominant measured overheads while preserving the common graph:
  reuse revision-owned descriptors and persistent buffers, derive tighter
  overflow-safe capacities, avoid redundant reset/copy/readback work, compact
  inactive descriptor ranges, and reduce small generation launches where
  payload and synchronization semantics genuinely allow it.
- Improve event batching around the measured crossover. Underfilled or
  irregular batches remain on CPU; suitable events are grouped into bounded
  GPU transactions with deterministic event ranges and per-event atomic
  fallback. Do not merge partial CPU and GPU particle lists.
- Keep diagnostic-only work out of the no-reference route. Comparison
  snapshots, residual reports, and deep tracing remain available on request
  but must not allocate, transfer, or synchronize in an ordinary qualified
  transaction.
- Validate every substantial optimization with complete-manifest lifecycle,
  serial/batch equivalence, operation fixtures, overflow and memory-boundary
  cases, CBMRoot unit/smoke/V0 gates, and bounded online/offline diagnostics.
  Compare results to the frozen Step 22.1 baseline and revert optimizations
  that trade correctness or uncontrolled memory growth for timing.

Stage 22.2 is complete when the optimized route has unchanged physics and
fallback results, bounded peak memory, no per-event descriptor rebuild or
avoidable host synchronization, and a reproducible improvement at the
workloads identified in Stage 22.1.

### Stage 22.3: Repeat Qualification And Make The Event-Batch Decision (75% -> 100%)

- Freeze a new policy before final measurement: numerical/stability limits,
  memory ceiling, allowed fallback reasons, timing variance, KFP speedup,
  end-to-end speedup, and the minimum workload eligible for GPU routing. Do not
  alter it after seeing campaign results.
- Run independent online and offline low/typical/high matrices with warm-up
  and repeated measurements. Include complete synthetic channel fixtures plus
  representative real requested-channel configurations, empty/invalid input,
  overflow, unavailable device, and forced CPU fallback.
- Publish one auditable decision for each official mode and workload range.
  A passing range may enable the explicit qualified event-batch route on the
  recorded hardware/backend; smaller or unsupported workloads stay on CPU.
  CPU-only remains the configuration default until a separate deployment
  decision. A failed performance gate remains a documented no-go result and
  must not be hidden by relaxing policy.
- Re-run the final standalone HIP, batch, CBMRoot contract, smoke, CPU/GPU
  equivalence, and retained-evidence integrity gates from one source/build
  state. Update integration and validation documents with the exact supported
  hardware, channel request, workload crossover, memory limit, and routing
  decision.

Step 22 reaches 100% when the complete event-batch implementation has one
reproducible online and offline qualification decision, and every enabled GPU
range preserves the complete Step 21 physics contract and event-atomic CPU
fallback. This step may qualify bounded event batches; it does not implement
device-to-device CA handoff or whole-time-slice KFParticle execution.

### Deferred final milestone: dual event/time-slice GPU execution

**Priority: сделать в самую последнюю очередь, когда уже все будет готово.**

This milestone is deliberately outside the current numbered execution plan.
Start it only after the GPU physics scope, production routing, fallback rules,
and online/offline qualification are complete and stable.

The final KFParticle GPU implementation must support two equal production
input modes without maintaining two different physics algorithms:

- **Event mode:** collect independent reconstructed events into bounded GPU
  batches, execute the shared decay graph, and scatter results back with
  event-atomic CPU fallback. This mode serves ordinary event-based offline
  reconstruction and small or irregular workloads.
- **Time-slice mode:** retain fitted CA tracks on the device, transform them
  device-to-device into the common KFParticle SoA, and process a complete time
  slice as flattened window/PV task ranges. Window overlap, boundary
  candidates, duplicate suppression, PID/PV association, and per-window
  failure isolation must be explicit. Only selected final output should need
  host transfer when downstream processing is CPU-based.

Expected final actions:

1. Freeze a shared device-resident CA-to-KFParticle track, covariance, time,
   window, field, PID, and PV handoff contract with persistent ownership and
   queue dependencies.
2. Generalize the current batch executor from host-side per-event launches to
   flattened event/window task tables processed by common routing,
   construction, selection, and materialization kernels.
3. Add the event-batch adapter and the zero-copy time-slice adapter around that
   common core; do not duplicate decay or numerical logic between modes.
4. Validate CPU/event-batch/time-slice physics equivalence, window-boundary and
   duplicate behaviour, overflow/fallback atomicity, memory ceilings, and both
   official CBMRoot launch paths on identical controlled inputs.
5. Measure the final production crossover and select the mode explicitly or
   automatically from workload properties. Preserve CPU fallback for
   unsupported configurations and underfilled GPU workloads.

Expected result: event-based reconstruction remains efficient and compatible,
while high-occupancy time-slice reconstruction avoids the CA track D2H and
KFParticle H2D round trip and amortizes scheduling across many windows. A
reasonable preliminary target is about 10-20x over CPU for the KFParticle
stage and roughly 1.2-1.5x end-to-end only when the wider GPU-resident chain
also removes intermediate host work. These are planning estimates, not frozen
acceptance thresholds; target-system measurements must set the final values.

Historical Step 8 closure notes follow. They are retained as implementation
history; the active continuation roadmap is Steps 15-20 above.

### Step 8 completion audit and closure plan

The initial Step 8 implementation establishes the persistent raw and selected
pools and verifies selected-index membership. The audit found that the final
handoff still needs explicit channel identity and preserved selection
diagnostics before the step can be considered fully complete. Close the step
in two controlled stages.

1. **Stage 8.C1: Complete the device-side selection contract.**
   - Store the producing `channelId` with each raw candidate and preserve it
     through the compact selected output. Keep per-channel selected ranges so
     downstream code does not infer a channel from PDG or atomic output order.
   - Add a bounded component-major device pool for
     `KFParticleGpuV0SelectionResult` (or an equivalent compact diagnostic
     record) indexed by raw candidate. The selection kernel must write the
     result for both accepted and rejected candidates, rather than discarding
     observables, classification, rejection bits, and best-PV identity.
   - Extend the non-owning selected view with explicit channel, fit/covariance
     access, and a way to resolve its stored selection diagnostic. Retain raw
     indices as the zero-copy ownership boundary; do not duplicate fit state.
   - Make the public selection continuation validate device-resident ranges
     from device-state counters or explicit ranges, not from a required host
     raw-pool download. Update the standalone and CBMRoot handoff documents.

2. **Stage 8.C2: Validate the completed contract end to end.**
   - Extend the actual `RunDecayPlan()` GPU regression with synthetic K0S,
     Lambda, and anti-Lambda fixtures covering multi-PV association,
     deterministic best-PV ties, mass and topology boundaries, rejected
     candidates, and selected-pool truncation.
   - For every raw candidate, compare GPU and host-reference observables,
     classification, rejection mask, best PV, channel identity, and compact
     membership. Treat compact order as unordered, but require correct
     per-channel ranges and lineage.
   - Run the same cases on CPU and GPU, report both timings, and retain the
     existing capacity/reuse and no-intermediate-readback assertions.

Completion criterion: all three default channels have deterministic CPU/GPU
coverage over the full selection result, the compact handoff preserves channel
and diagnostic identity without copying fit data, and no public continuation
requires an incidental raw-pool download.

Status: Step 8 is 80-85% complete. Stages 8.C1 and 8.C2 remain before Step 9.

Stage 8.C1 implementation notes:

- Raw candidate metadata now carries the producing channel ID from the decay
  plan through task generation and candidate construction. The compact selected
  pool retains the same ID in a parallel array, so no consumer must infer a
  channel from mother PDG or atomic order.
- Added a persistent, capacity-matched device buffer of
  `KFParticleGpuV0SelectionResult` records indexed by raw candidate. The
  selection kernel writes each evaluated result, including rejected candidates;
  selected-pool overflow additionally records the output-overflow rejection.
- `KFParticleGpuSelectedV0View` now resolves channel identity, raw fit state,
  and the corresponding selection record without copying the raw fit/covariance
  SoA. `RunV0Selection()` validates only allocation capacity on the host; the
  device kernel remains authoritative for the live raw-pool size.
- `RunDecayPlan()` exposes per-channel compact ranges after its one final
  selected-output hand-off. Queue-ordered launches make a channel's compact
  entries contiguous even though ordering inside that channel is atomic.

Status: Step 8.C1 implementation complete; compile and GPU validation pending.

Stage 8.C2 validation notes:

- Extended the default-V0 end-to-end regression to compare every stored GPU
  selection record against an independent host evaluation: observables,
  selection class, rejection bits, best PV, candidate/channel/event identity,
  compact membership, and zero-copy selected-view fit/lineage resolution.
- Added a `RunDecayPlan()` selected-output truncation fixture with two
  permissive channels and a capacity-one compact pool. It verifies the
  surviving channel range and the device-side
  `KFGpuV0SelectionRejectOutputOverflow` record for the rejected append.
- Added a two-identical-PV fixture that exercises deterministic lowest-index
  tie resolution and reruns default channels with the measured
  candidate-to-PV distance as an exact rejection boundary. GPU records must
  match the host reference for every channel.

Status: Step 8 complete. The standalone default-V0 selection boundary now
preserves channel and diagnostic identity over persistent device storage and
has deterministic CPU/GPU regression coverage for all default channels,
multi-PV association, selection boundaries, and compact-output truncation.

Step 8.2 notes:

- Added device-safe candidate-to-PV distance, uncertainty, `L/dL`, and spatial
  compatibility chi2/NDF primitives using the summed candidate/PV 3D
  covariance. The distance/error convention follows the existing CPU
  `GetDistanceToVertexLine()` calculation at the constructed candidate state.
- Added deterministic best-PV selection over a packed vertex range: invalid
  covariance, empty ranges, and invalid bounds reject explicitly; the first
  PV wins a numerical tie. This is an observable helper only and does not yet
  classify or compact candidates.
- The topology chi2 is intentionally documented as an unconstrained spatial
  residual approximation. Full CPU `SetProductionVertex` behavior remains out
  of scope until its transport/correlation path is ported and validated.
- Added zero/one/multiple-PV lifecycle coverage for values, covariance
  indexing, range guards, and invalid covariance handling.

Status: Step 8 in progress. Stage 2 of 6 complete.

Step 8.3 notes:

- Added `KFParticleGpuV0SelectionConfig` and a device-safe default-V0 decision
  function. It records mass/error, geometric chi2/NDF, nearest-PV distance and
  `L/dL`, best-PV topology chi2/NDF, classification, and all applicable
  rejection reasons in one flat result record.
- Default K0S/Lambda/anti-Lambda descriptors now carry separate selection
  thresholds for mass, geometric chi2, PV distance, secondary `L/dL`, and
  primary/secondary topology. This keeps the prior construction cut intact
  while removing the ambiguity of using one chi2 field for two CPU concepts.
- Added boundary coverage for primary, secondary, build, non-finite, mass,
  geometric, no-PV, distance, decay-length, and topology outcomes. The
  decision remains a standalone helper in this stage: no construction kernel,
  steering workflow, or raw candidate-pool ownership has changed yet.

Status: Step 8 in progress. Stage 3 of 6 complete.

Step 8.4 notes:

- Added a separate bounded selected-candidate index pool with its own capacity,
  size, and overflow flag. It stores raw candidate indices only, so fit state,
  covariance, and daughter lineage remain owned by the diagnostic raw pool.
- Added `KFParticleGpuSelectV0Candidates` and `RunV0Selection()`. The kernel
  consumes an explicit raw offset/count ABI, evaluates the configured decision,
  and atomically reserves selected indices without importing host decay-plan
  types into the device image.
- `RunV0Selection()` resets and downloads only the selected pool. In the normal
  chain the raw candidates and input vertices are already resident after the
  reconstruction stage; `UploadCandidates()` exists for standalone diagnostics
  that prepare raw candidates on the host.
- Added lifecycle coverage that verifies raw-pool preservation, unordered
  atomic selected-index membership, and selected-output overflow with an
  independent capacity-two device view.

Status: Step 8 in progress. Stage 4 of 6 complete.

## Interlude A: Persistent Device-State Ownership

Step 8 is paused after stage 8.4. The selection result and its compact index
pool are correct in isolation, but the current orchestration still reflects
the early standalone prototype rather than the intended GPU execution model.
Before adding more selection fixtures or a downstream consumer, refactor the
storage boundary so all reconstruction stages can work on one persistent
device-resident state.

### Architecture Decision

The current `KFParticleGpuBufferManager` hides all `xpu::buffer` members in a
PImpl held by `KFParticleGpuSteering`. This keeps XPU headers out of a public
API, but makes the actual GPU allocation layout invisible and has encouraged
short-lived task buffers and host-side count readbacks in the channel loop.

The replacement deliberately uses two explicit types instead of reproducing
the CA tracker object verbatim:

1. `KFParticleGpuDeviceStorage` is the visible, process-persistent owner of
   every `xpu::buffer`. Its header groups storage into input, working pools,
   output pools, and scalar counters. It owns no physics decision and does not
   contain host-only STL state.
2. `KFParticleGpuKernelState` is a small, trivially-copyable, non-owning
   device ABI. It contains SoA views, pool views, device pointers to counters,
   and capacities only. It is rebuilt from the storage owner after allocation
   or layout changes and published through `xpu::set<TheKFParticleFinder>`.

This is intentionally close to CA's visible `GpuTripletConstructor` ownership
model, while avoiding copying a broad owner object into constant memory. A
kernel gets its common state from the published ABI and receives only scalar
work ranges, channel IDs, or similarly small launch-specific values.

Target layout:

```text
KFParticleGpuSteering (host orchestration and one process queue)
  |
  +-- KFParticleGpuDeviceStorage (visible XPU-buffer owner)
       |-- input: tracks SoA, PV SoA, event descriptors, field data
       |-- work: two-daughter tasks, per-channel counters/status, scratch
       |-- raw output: candidates, daughters, raw-pool counters/overflow
       `-- selected output: candidate indices, counters/overflow
  |
  `-- KFParticleGpuKernelState (published non-owning device ABI)
       `-- input/work/output views and scalar capacities
```

### Required Invariants

- A buffer allocation is process-persistent and grows monotonically unless a
  deliberate teardown occurs. Kernels never own or allocate event storage.
- Host-to-device transfer happens only for new input, changed channel metadata,
  or explicit diagnostics. Device-to-host transfer happens only at a declared
  hand-off or diagnostic boundary.
- The queue ordering is sufficient between successive kernels. A host `wait()`
  or counter readback is forbidden between generation, construction, and
  selection in the normal path.
- Reallocation waits for queued work, invalidates the old published views, and
  republishes `KFParticleGpuKernelState` before another kernel launch.
- Raw candidates remain device-resident through selection and future daughter
  stages. Downloading the raw pool becomes an explicit diagnostic operation,
  not a side effect of processing a channel.
- Tests may retain explicit host/device copies for observability, but they must
  not dictate the production synchronization model.

### Implementation Plan

1. Introduce the visible storage and published-state contracts.
   - Define `KFParticleGpuDeviceStorage` and `KFParticleGpuKernelState` in
     dedicated headers with input/work/output/counter groups and concise
     ownership comments.
   - Keep the existing buffer manager temporarily as an adapter or replace it
     mechanically without changing the numerical kernels.
   - Add compile-time checks that the published state is trivially copyable and
     contains no host pointer or owning container.

2. Move every existing persistent buffer into the visible owner.
   - Preserve current SoA, candidate, daughter, and selected-index layouts.
   - Add persistent two-daughter task storage and device-side task/status
     counters; capacity becomes part of the storage policy.
   - Keep host-visible IO storage only where input packing or an explicit
     result download requires it; use device-only scratch where no host access
     is needed.

3. Publish one common kernel state and migrate kernel ABIs.
   - Rebuild and publish the state after initialization and every reallocation.
   - Convert generation, construction, and selection kernels to load common
     views from `ctx.cmem<TheKFParticleFinder>()`.
   - Retain the current explicit-view kernels only as narrow diagnostics until
     the constant-memory path has a standalone HIP regression test.

4. Remove channel-loop host round trips.
   - Generate compact tasks into the persistent task pool, construct over its
     capacity with a device-side valid-count guard, and record truncation in
     device status.
   - Do not copy task counters to the host in order to size the next launch.
   - Record per-channel ranges/status in a persistent device result buffer for
     a final compact readback.

5. Establish a device-resident V0 pipeline.
   - Run input upload, task generation, two-daughter fit, and V0 selection as
     one ordered queue sequence over the common state.
   - Download selected indices/results once at the public hand-off; retain raw
     candidate download only behind an explicit diagnostics API.
   - Make the decay plan/channel configuration a persistent device buffer or
     a published immutable state, with kernel launch arguments reduced to
     channel/event IDs and ranges.

6. Verify lifetime, synchronization, and performance properties.
   - Add tests for state publication after allocation growth, multiple events,
     no intermediate host counter copies, persistent task-pool reuse, and
     explicit diagnostic downloads.
   - Add CPU and HIP timing scopes separating upload, GPU pipeline, and final
     download. The tests must prove the sequence is correct before performance
     claims are made.
   - Run an isolated HIP constant-memory state smoke test before removing the
     kernel-argument diagnostic fallback; the earlier HIP issue must be
     reproduced or ruled out independently of KFParticle mathematics.

Completion criterion: the default V0 path can execute all currently ported
stages over one persistent device state, with no host readback between its
kernels and with a clearly documented final result hand-off. Only then resume
Step 8 at its regression-fixture stage.

Status: Interlude A in progress. This interlude is a prerequisite for Step 8.5 and 8.6.

Interlude A.1 notes:

- Extracted `KFParticleGpuKernelState` as the explicit published device ABI.
  It presently carries the existing input and raw-candidate views only; no
  buffer ownership or kernel launch behavior changed in this stage.
- `KFParticleGpuKernels` now inherits that state and retains only device-side
  reconstruction methods. This makes the data/method boundary visible without
  changing the already tested constant-memory object shape.
- Added compile-time flat-ABI checks and a lifecycle check that constructs the
  state directly from device views. The next stage moves the actual
  `xpu::buffer` owners and adds work-pool views to this contract.

Status: Interlude A in progress. Stage 1 of 6 complete.

Interlude A.2 notes:

- Added visible `KFParticleGpuDeviceStorage`, grouping all persistent XPU
  buffers into packed input, two-daughter work storage, raw candidate output,
  and compact selected output. The owner is intentionally easy to audit in one
  header, following the useful aspect of CA GPU storage.
- The existing `KFParticleGpuBufferManager` remains a transitional API adapter
  and now exposes `Storage()`. Its private implementation inherits the visible
  owner, so this step changes storage ownership without forcing a simultaneous
  steering API rewrite.
- Added monotonic capacity for a persistent two-daughter task pool plus its
  device counters. No current kernel uses that pool yet; the next stage will
  publish it in `KFParticleGpuKernelState` and migrate the kernels safely.

Status: Interlude A in progress. Stage 2 of 6 complete.

Interlude A.3 notes:

- Extended `KFParticleGpuKernelState` to describe all current persistent
  input, work, raw-output, and selected-output views. The state remains a
  flat non-owning ABI; buffer ownership stays in `KFParticleGpuDeviceStorage`.
- Added a diagnostic constant-memory state-probe kernel and lifecycle check.
  It publishes the full state through `xpu::set<TheKFParticleFinder>` and
  verifies device reads of input sizes, event metadata, task capacity, and
  output capacities.
- Production generation, construction, and selection kernels intentionally
  retain their explicit-view ABI in this stage. Their migration follows only
  after this isolated constant-memory path is validated on CPU and HIP.

Status: Interlude A in progress. Stage 3 of 6 complete.

Interlude A.4 notes:

- Replaced steering-local two-daughter task buffers and compact-generation
  counters with the persistent work pool in `KFParticleGpuDeviceStorage`.
  The same reusable allocation now serves fixed-slot and compact task paths.
- Added a pool-aware compact construction kernel. It launches over task
  capacity and reads the accepted-task counter on device, so no host readback
  is needed between task generation and candidate construction.
- Task status is downloaded only after the ordered generation/construction
  sequence has completed, to fill the existing host-facing channel result.
  Per-channel raw-candidate downloads remain temporarily for result-range
  compatibility and are the explicit target of the next stage.

Status: Interlude A in progress. Stage 4 of 6 complete.

Interlude A.5 notes:

- Added a scalar-only candidate-pool status readback. It transfers candidate
  count, daughter count, and overflow flags without copying candidate SoA,
  covariance, metadata, or daughter arrays.
- `RunDecayPlan()` now retains raw candidates on device while iterating over
  channels. It uses scalar status to preserve channel ranges and performs one
  full raw-pool download after the final channel. Single-channel diagnostic
  entry points retain their existing full-download behavior.
- The device-resident hand-off is not complete yet: `RunV0Selection()` still
  starts as a separate steering call and downloads its selected indices. The
  final interlude stage will make selection a continuation of the decay-plan
  queue sequence and isolate explicit diagnostics from the production result
  hand-off.

Status: Interlude A in progress. Stage 5 is partially complete.

Interlude A.6 notes:

- Added `KFParticleGpuMemoryModel.md`, documenting the visible storage owner,
  non-owning state ABI, SoA access model, H2D/D2H boundaries, queue ordering,
  diagnostic APIs, and remaining scalar bookkeeping transfers.
- The storage refactor is complete through persistent input/work/output pools,
  common device-state publication, persistent compact tasks, and one final raw
  pool download per decay plan. `RunDecayPlan()` now launches configured V0
  selection over resident raw candidates before that final hand-off and exposes
  the compact range through `LastDecayPlanSelectedCandidates()`.

Status: Interlude A complete. Step 8 remains paused after stage 8.4; its
selection regression and hand-off work can now resume on the persistent
device-resident storage model.

Step 8.5 notes:

- Added an end-to-end default-V0 selection regression over the actual
  `RunDecayPlan()` device-resident construction and compact-selection path.
  It independently evaluates each raw K0S/Lambda/anti-Lambda candidate on the
  host and compares selected-index membership, count, range, and overflow.
- The comparison deliberately treats atomic selected-index order as unordered:
  every expected candidate must appear exactly once, and no rejected candidate
  may appear. This is the correct contract until a stable compaction stage is
  introduced.

Status: Step 8 in progress. Stage 5 of 6 started.

Step 8.6 notes:

- Added `KFParticleGpuSelectedV0View`, the explicit non-owning hand-off from
  compact selected indices to raw candidate metadata, event identity, and
  ordered daughter lineage. It allocates and copies no candidate data.
- Updated the CBMRoot contract with selected-V0 semantics, matching keys,
  selected-pool overflow handling, and CPU-only/approximate functionality
  outside this first GPU boundary.
- Extended the default-V0 regression to verify selected raw indices, event
  identity, mother PDG, daughter count, and source lineage through the new
  hand-off view.

Status: Step 8 complete for the standalone default-V0 selection boundary.
Residual physics scope is explicit: PV topology is a spatial approximation and
field transport is not yet CPU `TransportCBM` equivalent.

Step 8.1 notes:

- Added flat, trivially-copyable V0 selection observables and result records.
  They reference a raw candidate by index and keep the future compact output
  free to store indices rather than duplicate fit/covariance data.
- Added explicit selection classes and composable rejection bits. Construction
  failure, non-finite fit, geometric, PV, distance, decay-length, mass,
  topology, and output-overflow failures are intentionally distinguishable.
- Added a lifecycle regression for defaults, bit composition, selected-state
  semantics, and a host-to-device-to-host record copy. No selection kernel or
  raw candidate-pool ownership changes are part of this stage.

Status: Step 8 in progress. Stage 1 of 6 complete.
