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

3. **Stage 9.C3: Complete channel-level telemetry.**
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
     GPU plan, lineage, telemetry, reporter aggregation, and invalid-input
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
     campaign, with commands, environment, log hashes, telemetry, and optional
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
     adding batch size, queue wait, and flush-reason telemetry.
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
- Batch telemetry records size, queue wait, and capacity/end-of-timeslice/
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
Step 8 is the standalone prerequisite.

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
