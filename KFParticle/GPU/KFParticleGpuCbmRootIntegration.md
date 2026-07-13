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
| ordered daughter source IDs | raw daughter pool through metadata offset/count |
| selection observables, class, rejection bits, best PV | raw selection-result record at that index |

Selected-index order inside a channel is atomic-compaction order, not physics
order. Queue-ordered channel launches make each channel's compact entries
contiguous; `LastDecayPlanSelectedChannels()` publishes those ranges. CBMRoot
comparison and later GPU stages must match by `(channel ID, event index, mother
PDG, ordered daughter source IDs)`. A selected-output overflow means the
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
2. mother PDG, event index, candidate flags, and ordered daughter source IDs;
3. fitted `x/y/z`, momentum, energy, charge, chi2/NDF, mass, and mass error;
4. selected-index membership, selected-output overflow, and selection decision
   separately from candidate construction.

Match candidates by `(channel ID, event ID, ordered daughter source IDs)`, not
by candidate array index. The daughter order is physics-significant for the
default channels: positive pion then negative pion for K0S, proton then
negative pion for Lambda, anti-proton then positive pion for anti-Lambda.

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
