# KFParticle GPU CBMRoot Validation Gate

Normal CBMRoot reconstruction remains CPU-only by default. GPU output must not
be routed into the public particle or selection result until the staged Step
18 gates below have recorded CPU and HIP runs.

## Step 18 Planned Gates

Stage 18.1 extends the hermetic comparison from two-daughter V0s to every
implemented topology. It requires exact channel/event/output-class/status and
variable-length lineage agreement, channel-specific numerical tolerances,
boundary fixtures, and explicit rejection of unsupported, invalid, overflow,
and mismatched events.

Implementation checkpoint: the flat parity contract, persistent
topology/output/operation metadata, channel/output tolerance policy, CBMRoot
full-range snapshot adapter, and explicit promotion blockers are present.
Standalone lifecycle check `[15] parity-promotion-contract` and the CBMRoot
`AppliesFullTopologyParityAndPromotionPolicy` unit fixture are the named
Stage 18.1 additions. Target HIP and CBMRoot runs below must pass before the
35% completion evidence is recorded.

Stage 18.2 adds bounded materialization tests. A validated GPU event is
converted into an isolated public particle result and snapshotted again. Fit
state, covariance, PV association, IDs, immediate daughter references, and
canonical physical lineage must survive the round trip. Every injected
conversion failure must leave the destination empty and unchanged.

Implementation checkpoint: standalone lifecycle check
`[83] bounded-output-materialization` covers the ROOT-independent graph and
failure contract. The CBMRoot
`MaterializesGpuEventAtomicallyIntoKFParticles` unit fixture exercises the
thin adapter and actual scalar `KFParticle` destination. The unit target must
be rebuilt after reconfiguration because the adapter is a new source in
`CbmKfParticleGpuDiagnostic`.

Stage 18.3 adds a routing matrix covering CPU-only, diagnostic dual execution,
qualified GPU policy acceptance, unavailable XPU, unsupported graph scope, invalid
field/input, all pool-overflow classes, failed parity, and failed
materialization. Fallback is required to be event-atomic and to reproduce the
ordinary CPU result. The final gate comprises:

- standalone CPU and target-HIP lifecycle suites;
- serial/batch and atomic/block-scan benchmark equivalence;
- CBMRoot GPU adapter unit tests and hermetic full-topology equivalence;
- embedded XPU runtime and diagnostic smoke tests;
- a dedicated validated-routing/fallback smoke test.

Implementation checkpoint: `_GTestKfpGpuDiagnosticPacker` now contains
`RoutesQualifiedGpuEventsAtomically`, which verifies mode parsing, the exact
default-V0 capability manifest, GPU acceptance, every routing fallback class,
and resettable monitoring counters. The existing embedded diagnostic smoke
also verifies successful materialization plus an accepted route and an
invalid-field event-atomic fallback on the configured accelerator.

Full-chain online/offline bring-up is complete under Step 19. Representative
equivalent-input detector-data campaigns and throughput thresholds remain
Step 20 work; neither was a prerequisite for implementing the opt-in Step 18
route.

Step 20.2 subsequently separates policy acceptance from production
qualification. The `qualified-gpu` route is implemented without a mandatory
CPU reference, but `GpuCapabilityManifest::QualificationUnlocked()` remains
false until the representative online and offline decisions in Step 20.3.
Real launchers must therefore report zero GPU publications and a nonzero
`QualificationLocked` or `UnsupportedCapability` fallback count at this
checkpoint. The ordinary CBMRoot configuration requests Xi/Omega and explicit
finder cuts, so it is expected to exercise the capability fallback; the exact
V0 lock is covered independently by the unit and embedded smoke tests.

The Stage 20.1 batch evidence supports retaining the persistent queue,
monotonic buffer capacities, and multi-event payload transactions. Candidate
construction remains the dominant device phase. Its sensitive non-contracted
full-field expressions are not speculatively fused in Stage 20.2 because that
would weaken the established numerical-reproducibility contract without a
measured end-to-end justification.

Qualification status: Step 18 is complete (100%). The target HIP lifecycle
and benchmark checks, CBMRoot build and unit/equivalence checks, and embedded
runtime plus diagnostic-routing smoke tests all pass. Representative-data
performance qualification remains deliberately deferred to Step 20.

Step 19 qualification status: complete (100%). The target server passed
repeated online CPU-only and diagnostic `cbmreco` runs, stable handoff
comparison, the embedded XPU smoke, the previously qualified offline
FairRunAna runs, and the final independent-input cross-mode contract. Because
the available online and offline samples are unrelated, this qualification
does not claim equal event or candidate counts across frameworks; that
stronger equivalent-input check remains part of Step 20.

## Step 20 Planned Gates

Step 20 has three qualification gates. Stage 20.1 adds disabled-by-default
performance monitoring, warm-up/repeat support, machine-readable evidence,
and target-HIP standalone/online/offline baselines without changing execution
policy. Stage 20.2 optimizes only measured bottlenecks and prepares an opt-in
GPU-without-reference route that remains locked behind qualification. Stage
20.3 freezes tolerances and performance/memory/fallback thresholds, runs
representative campaigns, and records independent online and offline
production decisions.

At every gate, CPU-only remains the default, diagnostic comparison remains
available, and all existing lifecycle, numerical, lineage, selection,
overflow, materialization, runtime-ownership, and full-chain checks must pass.
Stage 20.1 and Stage 20.2 are complete on the target HIP system. Enable
performance monitoring only for a measurement campaign with
`KFPARTICLE_GPU_PERFORMANCE_MONITORING=1`. Normal CPU-only and diagnostic runs
retain the previous default with no additional device readback or wait. The
online and offline evidence files now include source/build identity, complete
run wall time, the existing phase timing, and explanatory launch, wait,
traffic, work-density, capacity-growth, and persistent-payload counters.
The bounded target-HIP online diagnostic prerequisite also passes the strict
locked contract after the real-data numerical-parity audit. The separate
ordinary/trial build contract and Stage 20.3A HIP checks pass as well. The
online and offline qualification matrices, combined production decision, and
final target-HIP regression set are complete. The result is a validated
`diagnostic-only` route with the ordinary build qualification-locked. Step 20
readiness is 100%.

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
default V0 execution, lineage, materialization, validated routing,
invalid-field fallback, phase monitoring, reporter aggregation, and an
invalid-input rejection. It must not be added to the production CBMRoot test
set because it requires a configured accelerator.

For an opt-in selector campaign, the validated mode is:

```yaml
kfp:
  selector:
    gpuDiagnostics:
      mode: qualified-gpu
      samplePeriod: 1
```

The mode is currently qualified only when requested decays are a non-empty
subset of K0S, Lambda, and anti-Lambda and no custom `finderCuts` block is
present. Every other request deterministically retains the complete CPU event.

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

## Additional Reconstruction Campaign Wrapper

Step 19 now provides directly runnable online `cbmreco` and offline
FairRunAna launchers and has qualified both official modes. The older generic
wrapper below remains available for an additional pair of caller-supplied
baseline and diagnostic commands, but it is no longer the full-chain bring-up
gate. It is useful when an equivalent input and byte-comparable output become
available for the stronger Step 20 production campaign.

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

## Step 20.3 Qualification Trial

An ordinary build keeps `qualified-gpu` locked. Configure
`CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON` only in a disposable qualification
build to measure the prepared no-reference route. The tested YAML must request
exactly K0S, Lambda, and anti-Lambda and must omit `finderCuts`; otherwise the
exact capability contract deliberately falls back to CPU.

Run CPU-only, diagnostic, and qualification-trial modes with one independent
warm-up and three measured repetitions, with
`KFPARTICLE_GPU_PERFORMANCE_MONITORING=1` and distinct evidence files. Then
invoke `run_cbmroot_kfp_qualification_gate.sh` separately for `online` and
`offline`. The gate reads the frozen
`step20_qualification_policy.conf`, checks input/build identity, physics
comparison, routing, memory, timing stability, KFP speedup, and end-to-end
speedup, and writes a mode-specific decision. A passing trial is evidence for
a later production change; it does not mutate or unlock an ordinary build.
`run_cbmroot_kfp_qualification_campaign.sh` performs the three runs and gate
in one command. Use `make_step20_v0_qualification_config.py` to create a
separate narrow YAML beside the original configuration so relative parameter
paths remain valid. If the reviewed configuration lists Lambda but not its
charge conjugate, the generator adds anti-Lambda with the same reviewed cuts;
the qualification route requires the complete `310,3122,-3122` set so it
cannot publish a channel the CPU configuration did not request.

For Stage 20.3B, `run_cbmroot_kfp_qualification_matrix.sh` runs one official
mode across the required `low`, `typical`, and `high` bounds. Online bounds
limit reconstructed events in the fixed timeslice; offline bounds limit input
entries in the event-based sample. These names describe reproducible workload
volume, not measured occupancy of each event. Run online and offline matrices
separately: success in one mode cannot replace evidence from the other.

The matrix completes all three workloads even if an earlier frozen gate
rejects qualification. Every workload retains its decision and hash, followed
by one aggregate mode decision. Rejection is therefore useful evidence for a
Stage 20.3C diagnostic-only no-go rather than an incomplete run. Do not adjust
the frozen policy between workloads or reinterpret its thresholds afterward.
The gate also requires trial-build identity, matching source and parameter
identities, exact `310,3122,-3122` configuration without `finderCuts`, and the
recorded policy hash.

The first target-HIP online matrix completed all 36 requested reconstruction
launches. Bounds 50 and 100 passed their physics, routing, memory, stability,
and speed gates. Their measured KFP/end-to-end speedups were respectively
`8.61966809/1.02812779` and `7.75448425/1.09657002`. Bound 10 failed only the
frozen end-to-end threshold with speedup `0.89214477`; consequently the online
aggregate decision is `rejected`. This is retained as valid qualification
evidence rather than being reclassified or used to modify policy. The offline
matrix remains pending.

`run_cbmroot_kfp_qualification_final_decision.sh` closes the qualification
only from complete online and offline matrix files. It verifies each matrix's
mode, device, frozen workload bounds, all workload decision hashes, and the
frozen policy hash. It emits `qualified-gpu` only if both official modes pass;
otherwise it emits a successful, reproducible `diagnostic-only` no-go and
requires the ordinary build to remain locked. The companion hermetic test is
registered as `_TestKfpGpuQualificationFinalDecision`.

The target offline matrix and final combined decision have not yet been run.
Until they are retained and the final regression set passes on `hip1`, Step 20
is 97% complete rather than 100%.

The isolated offline numerical blocker is resolved. On the target HIP system,
`_GTestKfpGpuDiagnosticPacker` passes and the event-based diagnostic preflight
over entries `0..18` completes under the strict locked contract. The repaired
comparison uses absolute-plus-relative bounds for the validated default-V0
policy while all relative terms remain disabled by default elsewhere. The
complete offline workload matrix subsequently ran all 36 requested launches.
Bounds 10 and 50 were rejected only by end-to-end speedups `0.99265606` and
`1.00554156` below the frozen `1.01` threshold; bound 100 passed every gate.
The aggregate offline decision is correctly `rejected`. Both official mode
matrices are now complete, so the next required artifact is the combined
`diagnostic-only` decision, followed by the final target regression set.

The target final-decision verifier accepted both matrix files, all
workload-decision hashes, device `hip1`, the frozen workload bounds, and the
frozen policy hash. Its completed production decision is `diagnostic-only`;
online and offline remain diagnostic-only and the ordinary build remains
qualification-locked.

The final closure regression is intentionally short and must use one final
source state. It consists of the five hermetic CBMRoot contract tests, the
standalone `hip1` lifecycle suite, a non-qualifying 1/4/8-event batch
benchmark, the XPU ownership and diagnostic smoke tests, the hermetic CPU/GPU
V0 equivalence test, and one revalidation of the retained combined decision.
The long online/offline matrices are immutable qualification evidence and
were not rerun. The closure set passed: five CBMRoot unit/contract checks, all
97 standalone `hip1` lifecycle checks, the 1/4/8-event batch benchmark, XPU
ownership and diagnostic smoke, and CPU/GPU V0 equivalence. The hermetic
contract tests were additionally verified not to change the retained offline
matrix after their inherited `KFPARTICLE_*` environment was isolated.

The final evidence records online matrix SHA256
`5207f3251d795805018ec4fc144a1ee98dbfa3a39543f6cf26c8e17eaa68d0aa`
and offline matrix SHA256
`a8d97b11d3ad7b2e42b8c4429337109463962109dde7e81d76385d9b79850a3d`.
Both are rejected only according to their frozen workload decisions, so the
combined route remains `diagnostic-only`. Step 20 is complete at 100%.

## Step 21 Planned Gates

Step 21 closes active CPU finder channel coverage in four gates without
unlocking production. Stage 21.1 freezes one exhaustive declarative channel
catalogue and proves that every active CPU branch is accounted for. Stage 21.2
enables all charged two-track and track-composite generations through the
existing mask-driven execution. Stage 21.3 completes composite-composite,
neutral/missing-mass, matching, projection, production-vertex, and final
selection operations, including the optional neutral-input ABI. Stage 21.4
requires all nine family records to be supported and runs exhaustive
host/device, standalone HIP, and bounded CBMRoot online/offline parity gates.

Every stage retains CPU-only as the default and preserves event-atomic CPU
fallback. The final Step 21 evidence must distinguish a valid event missing an
optional detector input from an unimplemented channel: the former is a runtime
capability fallback, while the latter prevents step completion.

Stage 21.1 is complete on the target system. The frozen CPU catalogue has
256 contracts and revision `12010545515856880377`: 230 active, 18 conditional
on neutral input, and 8 configuration-disabled same-sign records. Its lifecycle
gate checks all nine families, exact seven-channel default-plan mapping,
explicit requirements for the remaining active channels, stable conjugates,
dependencies, operation/cut profiles, and deterministic snapshot identity.
The revised count includes the active charm-conditioned CPU branch
`420 -> pi+ pi-`, found while activating the complete two-track family.
The local CPU standalone lifecycle passes 98/98 checks, and the corresponding
target HIP lifecycle plus requested CBMRoot regressions pass. Stage 21.1 is
therefore recorded as 20% complete; production remains `diagnostic-only`.

Stage 21.2 is split into two target-verifiable parts. Stage 21.2A activates all
50 active two-track channels; Stage 21.2B will activate the track-composite and
long-lived-composite generations. The channel mask
capacity is now 256 bits, with lifecycle coverage across the 32- and 64-bit
boundaries and 96-channel routing plans. He-6, Li-6, Li-7, and Be-7 are now
first-class packed GPU track species, with exact-PDG routing and CBMRoot packer
tests. Target acceptance for this increment requires a full rebuild because
both the event descriptor species array and device routing state ABI changed.

Stage 21.2A is complete on the target HIP system. The catalogue generates the
exact daughter species, masses, source aliases, primary/secondary ranges,
same-PV requirements, charm cuts, selection profiles, and output classes for
all 50 active two-track channels. The complete plan compiles to 50 routing
descriptors and one 50-node track-track graph generation, with full
two-daughter family coverage. The local standalone CPU lifecycle passes
98/98 checks. Target HIP lifecycle, CBMRoot packer, diagnostic smoke, and
CPU/GPU V0 equivalence also pass. Step 21 enters Stage 21.2B at 35%.

Stage 21.2B is accepted on the target HIP/CBMRoot configuration for 120
independently executable composite-track contracts. One descriptor-driven route executes
track-composite and long-lived channels through generations 2–4, republishes
accepted intermediates without a host copy, preserves bounded canonical
lineage, and carries charm bachelor cuts, the secondary composite-track DCA
boundary used by CPU `FindTrackV0Decay`, and hypernuclear parent mass
constraints. `FindLL` remains free of that DCA cut, matching its CPU branch.
The local standalone lifecycle passes 98/98, including a generation-2 result
consumed by a generation-3 kernel.

The four `111 + p/K` charge contracts were explicitly held at this boundary
because stable parent channel 5019 belongs to Stage 21.3. The target HIP
lifecycle and CBMRoot packer/diagnostic/V0 regressions passed for the 120
independently executable channels; production remains `diagnostic-only`.
Step 21 is at the accepted 60% boundary.

Stage 21.3A is accepted. The catalogue adds an explicit second parent identity for binary
resident operations. All 24 composite-composite channels compile into the
generation graph, identical-source pairs use canonical index ordering, and
the real channel 5019 `gamma+gamma -> pi0` unlocks the four deferred
`pi0+p/K` channels. The resulting complete charged/resident manifest has 198
nodes (50 two-track, 124 composite-track, 24 composite-composite). Standalone
CPU and target HIP, heap, CBMRoot unit, diagnostic-smoke, and CPU/GPU V0
regression checks pass, so Step 21 is at the accepted 70% boundary.
The lifecycle also freezes the partial-plan regression: without graph channel
5019 the diagnostic plan contains 120 composite-track channels and 170 valid
nodes; with the composite-composite block present it contains all 124 and 198
nodes. Therefore CBMRoot V0-only diagnostics cannot acquire a dangling pi0
dependency merely because the complete catalogue supports it.

Stage 21.3B is split into primary projection (21.3B1, 70% -> 75%) and the
remaining neutral/terminal operations (21.3B2, 75% -> 85%). Stage 21.3B1 is
accepted at catalogue revision `12488567174590575521`: all 12 CPU
`ExtrapolateToPV` channels are exact unary graph operations, take the primary
vertex from resident candidate metadata, preserve the resolved vertex in the
published candidate, and do not add the `SetProductionVertex` operation that
the CPU path does not perform. The complete manifest has 210 nodes. A partial
diagnostic plan adds only eight parent-resolvable projections and has 178
nodes, preventing dangling graph dependencies. Standalone CPU and target
HIP/CBMRoot acceptance pass, so Step 21 is at 75%.

The first 21.3B2 audit checkpoint corrected a stale input assumption and is
accepted on the target system. Active CPU `NeutralDaughterDecay` consumes two
track ranges and performs filtered missing-mass reconstruction internally; it
does not require an external neutral-candidate range. The accepted inventory
contains 260 entries at revision `3044443511554642051`, including all 22
missing-mass charge hypotheses and the formerly absent He3/He4 pairs. The
routing mask is 288 bits and target acceptance keeps Step 21 at 75%.

The following implementation checkpoint is accepted on the target system. It
adds unambiguous track-set/species source IDs, routes candidate and raw-track
sources through the same device graph task ABI, ports the CPU filtered
missing-mass covariance/energy update, and marks all 22 contracts supported.
The complete plan has 232 nodes at catalogue revision
`6470827997588483179`. CPU and target HIP lifecycle pass 99/99, including a
device-generated primary-mother/secondary-daughter task and verification of
the filtered mother, event identity, and two-track ancestry. The unchanged
CBMRoot heap/unit/smoke/V0 gates pass, accepting the 80% checkpoint.

The ordinary CBMRoot single-event and batch diagnostic builders now append all
22 missing-mass channels when they create the default plan. Their common
overflow-checked capacity function counts both initial two-track hypotheses and
raw-track graph pairs; custom narrow plans remain unchanged. The resulting
ordinary graph has 192 nodes, and a controlled range fixture fixes the expected
raw-task capacity at 62. This integration is locally complete and awaits the
target unit, diagnostic-smoke, and online/offline bounded diagnostic gates.
Kaon matching and terminal selection remain for the 85% boundary.

Before a qualification campaign, run
`run_cbmroot_kfp_qualification_trial_preflight.sh` with distinct ordinary and
trial build directories. The ordinary cache must explicitly contain
`CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=OFF`, the trial cache must contain
`ON`, and both `libAlgo.so` and `libAlgoOffline.so` must carry the matching
binary contract marker. The script creates the narrow YAML and records one
evidence file with source/build/configuration/policy identities. Online and
offline launchers repeat this check before reconstruction; set
`KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=ordinary` for a locked control
or `trial` for qualification work. A `published` expectation implies `trial`
even when the variable is omitted.

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

## Step 21 Final Scope Gate

The frozen complete catalogue has 260 records at revision
`7326650681912440158`: 252 active and supported channels plus eight
configuration-disabled same-sign records. The complete graph must contain
exactly 252 unique nodes and all nine family records must be supported. The
zero-active same-sign family is complete only because every member is
explicitly configuration-disabled; no placeholder GPU node is permitted.

Final acceptance requires all of the following on the target HIP system:

1. standalone lifecycle with the `complete-cpu-finder-manifest` contract and
   all 101 checks passing;
2. validated serial/batch equivalence over the complete automatic plan;
3. CBMRoot packer/unit, XPU smoke, diagnostic smoke, and CPU/GPU V0 gates;
4. bounded online and offline diagnostic campaigns with no unsupported-family,
   unsupported-channel, overflow, unresolved-lineage, or partial-event merge;
5. reproducible evidence naming the catalogue revision, source/build identity,
   device, workloads, and retained diagnostic-only production decision.

The lifecycle matrix binds every active channel ID to one graph node, one
tested operation payload, routing/monitoring storage, parity snapshot identity,
and bounded materialization. Operation-level host/device fixtures remain the
numerical oracle for full-field transport, DCA, energy fit, composite
construction, filtered missing mass, matching, projection, constraints, and
final selection. Real detector data need not contain every rare channel;
controlled fixtures provide that exhaustive coverage.

The final Step 21 campaign is accepted on the target HIP/CBMRoot system.
Standalone lifecycle passes all 101 contracts; complete-plan serial/batch
equivalence, CBMRoot packer/unit, XPU and diagnostic smoke, CPU/GPU V0, and
bounded online/offline diagnostic gates pass. The bounded runtime plan is
derived from the CPU selector's requested mother PDGs and includes dependency
closure, so unrelated catalogue channels do not inflate capacities or create
false GPU-only differences. Direct CPU V0 comparison is restricted to
channels 1, 2, and 3; supported later-generation candidates remain visible in
raw graph output and per-channel monitoring and are validated by the exhaustive
topology fixtures.

No active catalogue channel remains unsupported, all nine families satisfy
the parity and materialization contracts, and event-atomic fallback remains
intact in both CBMRoot launch modes. Step 21 is complete at 100%. CPU-only and
the Step 20 `diagnostic-only` production decision remain unchanged.

## Step 22.1 Baseline Output Gate

Set `KFPARTICLE_GPU_PERFORMANCE_MONITORING=1` for a measured diagnostic or
qualification run. Both official CBMRoot modes must print the same
`KFParticle GPU performance report` layout. Review it in this order:

1. scope and input: events, batches, tracks, vertices, and raw-task capacity;
2. channel/output load: visited pairs, stored tasks, constructed, raw,
   selected, matched, CPU-only/GPU-only, and overflow;
3. ordered stages 01-13 with explicit CPU, transfer, or GPU/queue domain;
4. total transaction and queue wait;
5. launches, work density, transferred bytes, capacity growth, and memory
   high-water mark.

The report is explanatory host-wall evidence. `GPU/QUEUE` rows include the
queue-ordered interval and required synchronization; they are not isolated
device-event timings. The compact qualification/performance records remain
the machine-readable source for campaign scripts. Acceptance requires the
unit aggregation test plus bounded online and offline runs whose input/output
counts agree with their existing KFP summaries and whose repeated workloads
produce stable stage and memory values.
