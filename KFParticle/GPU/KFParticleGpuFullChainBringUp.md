# KFParticle GPU Full-Chain Bring-Up

This runbook covers Step 19 execution in the two official CBMRoot modes. Online
and offline reconstruction share the KFParticle data, capability, routing,
fallback, and XPU lifecycle contracts, but they have separate framework
adapters and launch requirements.

## Shared Contract

Both modes must provide the same event-level KFParticle boundary:

- fitted reconstructed tracks with hit references, covariance, stable source
  IDs, charge and PID hypotheses;
- explicit event boundaries and a stable source-event identity;
- reconstructed primary vertices and the field region used for track fitting;
- the same requested decay manifest and CPU-only, diagnostic, or
  `qualified-gpu` routing mode (`validated-gpu` is a compatibility spelling);
- event-atomic overflow, comparison, materialization, and CPU fallback rules.

CBMRoot initializes XPU once. KFParticle attaches to that runtime, owns one
process-persistent queue and its persistent buffers, and does not borrow a CA
queue. A mode adapter may convert framework data but must not duplicate
KFParticle mathematics or GPU kernels.

The shared preflight requires `CBM_KFPARTICLE_USE_XPU=ON` and
`CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON`, selects all XPU/KFParticle libraries from
the same CBMRoot `build/lib`, rejects an implicit non-HIP device, and runs the
existing embedded XPU ownership smoke unless explicitly disabled.

## Online: `cbmreco`

The online entry point is `build/bin/cbmreco`. Its minimal Step 19 chain is:

```text
Unpack -> DigiTrigger -> LocalReco -> Tracking -> EventReco -> KfpSelector
```

GPU CA tracking uses `--ca-timeslice-tracking-backend gpu-multi-window`.
`cbmreco` receives the XPU device through `--device`; the default Step 19
preflight device is `hip1`. The main YAML and its `parFiles.recoParameters`
archive are separate required inputs. A local timeslice file/directory or a
supported `TimesliceAutoSource` URI may be supplied as the input locator.

The initial target-server input is one timeslice:

```text
/home/kozlov/test_cbmroot/001/run/vt26_1/vt26test.00001.tsa
```

Its initial configuration is:

```text
/home/kozlov/test_cbmroot/001/run/vt26_1/pars_vt26/MainConfig.yaml
```

The preflight resolves `parFiles.recoParameters` relative to that YAML and
fails before launching `cbmreco` when the referenced archive is absent.

Run a prerequisite and embedded-runtime check without processing a timeslice:

```bash
KFPARTICLE_CBMROOT_SOURCE_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot \
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
KFPARTICLE_CBMROOT_ONLINE_CONFIG=/path/to/MainConfig.yaml \
KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR=/path/or/source/locator \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_kfp_online_preflight.sh
```

Add `KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1` to process one timeslice. The
default mode is `cpu-only`. Stage 19 accepts `cpu-only` and `diagnostic`; the
launcher passes the selected mode and sample period directly to `cbmreco`, so
the same YAML file is used for both runs.

The qualification launcher defaults to at most 50 reconstructed events per
timeslice (`KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=50`). Event building still
sees the complete timeslice, while the expensive event-reconstruction and
KFParticle loop receives only the first 50 stable event records. Set another
positive value explicitly when a larger sample is required.

The current `.tsa` commissioning sample does not provide usable detector PID
for the reconstructed STS tracks. The online qualification therefore defaults
to `KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID=1`, which passes
`--kfp-use-sts-pion-pid` and assigns charge-signed pion hypotheses before
detector PID. This is an explicit diagnostic policy and does not change the
normal `cbmreco` default. Set the variable to `0` to test detector PID instead.

Run the complete online CPU/diagnostic campaign:

```bash
KFPARTICLE_CBMROOT_SOURCE_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot \
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
KFPARTICLE_CBMROOT_ONLINE_CONFIG=/home/kozlov/test_cbmroot/001/run/vt26_1/pars_vt26/MainConfig.yaml \
KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR=/home/kozlov/test_cbmroot/001/run/vt26_1/vt26test.00001.tsa \
KFPARTICLE_CBMROOT_ONLINE_TRACKING_BACKEND=gpu-multi-window \
KFPARTICLE_CBMROOT_ONLINE_NUM_TS=1 \
KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=50 \
KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID=1 \
KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD=1 \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_kfp_online_campaign.sh
```

The campaign requires handoff contract 1, diagnostic contract 2, two stable
runs per mode, no failed event conversion, complete CPU/GPU comparison, and
identical event/track/vertex/PID handoff in CPU-only and diagnostic modes. It
also rejects a formally successful run unless assigned PID hypotheses,
selected tracks, CPU particles, and at least one composite CPU decay candidate
are all non-zero.

Real-data mismatch tracing is controlled by `KFPARTICLE_GPU_DEEP_TRACE`.
Use `0` to suppress it, `1` for the first mismatched event, a positive number
for that many mismatched events, or `ALL` for the bounded internal maximum.
The trace is emitted only when comparison differences exist. The online
preflight also requires build marker `20260803-real-data-pv-parity-v11` when
tracing is enabled, preventing an expensive run with a stale `libAlgo.so`.

`kfp.selector.decays` is the reconstruction-channel list. `eventSelector`
only assigns optional event-selection output bits to those channels; omitting
a decay from `eventSelector.selectors` does not disable its reconstruction.
The online summary reports `decay_channels` so an empty reconstruction plan is
detected before GPU comparison.

## Offline: FairRunAna

The offline entry point is the test-owned
`GPU/test/cbmroot/run_reco_tracks.C`. It accepts an input prefix and derives
`.raw.root`, `.par.root`, and `.reco.root` names. Geometry is selected by the
setup tag. GPU CA tracking is selected with `CBM_CA_TRACKING_BACKEND` and
`CBM_CA_XPU_DEVICE`.

The Stage 19.2 task order is:

```text
CbmRecoSts
  -> CbmL1 / CbmStsFindTracks
  -> CbmBuildEventsFromTracksReal (time-based input only)
  -> CbmFindPrimaryVertex
  -> TaskRecoEventConverter
  -> TaskKfpSelector
  -> shared KfpSelectorChain
```

`TaskRecoEventConverter` accepts STS-only tracks without requiring a
`GlobalTrack` or MC branch. It derives the event hit set from STS-track
references when the event builder supplies track indices only. The first
offline qualification uses fixed charge-dependent pion hypotheses because
the macro intentionally does not reconstruct TOF PID. This policy is applied
before detector PID and is covered by `_GTestKfpSelectorPid`.

`TaskKfpSelector` owns framework adaptation and monitoring only. It reads the
same `kfp` and `eventSelector` YAML nodes as online reconstruction, builds the
shared `KfpSelectorChain`, encodes stable source IDs as
`(ROOT entry << 32) | event index`, and reports event, track, STS-hit-link,
vertex, PID, field, and diagnostic counters. Diagnostic counters are advanced
when a completed GPU result is published, so a result cannot be counted again
by a later timeslice. Reports identify this schema as
`diagnostic_contract=2`; preflight checks that the loaded
`libCbmRecoTasks.so` contains this contract before spending time on
reconstruction. In diagnostic mode KFParticle attaches to CBMRoot's already
initialized XPU runtime but keeps its own queue.

The initial target-server inputs are two representations of the same generated
sample:

```text
/home/kozlov/test_cbmroot/001/run/sg_data/hdr.p12.mbias.tb
/home/kozlov/test_cbmroot/001/run/sg_data/hdr.p12.mbias.eb
```

The `.tb` prefix denotes the unsplit time-based input. The `.eb` prefix denotes
the event-based representation. The launcher infers the representation from
the suffix, defaults to one entry for `.tb` and 100 entries for `.eb`, and
records it in the evidence file. Larger `.eb` runs require the explicit
`KFPARTICLE_CBMROOT_OFFLINE_ALLOW_LARGE_EVENT_RUN=1` override.

These data contain empty events or detector modules that the current
single-window GPU tracking route does not handle correctly. Step 19 therefore
accepts only `gpu-multi` or its `gpu-multi-window` alias. This is a declared
input/backend limitation, not a KFParticle numerical requirement.

Build the new offline adapter and its focused tests:

```bash
cmake --build /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
  --target CbmRecoTasks _GTestKfpSelectorPid _GTestKfpGpuDiagnosticPacker \
  --parallel 8

ctest --test-dir /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
  -R '(_GTestKfpSelectorPid|_GTestKfpGpuDiagnosticPacker|_TestKfpGpuFullChainPreflight)' \
  --output-on-failure
```

Run a CPU-only time-based qualification:

```bash
KFPARTICLE_CBMROOT_SOURCE_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot \
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
KFPARTICLE_CBMROOT_OFFLINE_CONFIG=/home/kozlov/test_cbmroot/001/run/vt26_1/pars_vt26/MainConfig.yaml \
KFPARTICLE_CBMROOT_KFP_MODE=cpu-only \
KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX=/home/kozlov/test_cbmroot/001/run/sg_data/hdr.p12.mbias.tb \
KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
KFPARTICLE_CBMROOT_OFFLINE_SETUP=sis100_hadron \
KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=1 \
KFPARTICLE_CBMROOT_OFFLINE_TRACKING_BACKEND=gpu-multi \
KFPARTICLE_CBMROOT_OFFLINE_REPEAT=2 \
KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_kfp_offline_preflight.sh
```

Repeat the same command with
`KFPARTICLE_CBMROOT_KFP_MODE=diagnostic`. The launcher accepts the run only
when the summary contains at least one processed event and all published GPU
results executed and matched their CPU references. With the qualification
sample period of one, the required invariant is
`diagnostic_seen == diagnostic_ok == diagnostic_compared ==
diagnostic_matched == processed`, while `diagnostic_blocked` and
`diagnostic_blockers` must both be zero.

For the bounded event-based check replace the input prefix with
`hdr.p12.mbias.eb`, set
`KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based`, and set
`KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=100`. Event-based input preserves its
existing `CbmEvent` boundaries; it is not passed through the time-clustering
event builder again.

`KFPARTICLE_CBMROOT_OFFLINE_REPEAT=2` starts two independent ROOT processes
and validates each log before accepting the server qualification. Separate
evidence and per-run log files are written for the input kind and KFP mode. A
successful process that does not produce a non-empty KFP handoff is treated
as a failure. The known FairRoot shutdown fault is outside this qualification:
a non-zero ROOT status is ignored only when the same run already emitted the
KFP summary and macro completion/output markers and produced a non-empty
`.reco.root` file. Any earlier ROOT failure still fails the preflight.

## Contract Test

The mode-independent script test builds a temporary fake source/build tree and
checks the shared build/XPU contract plus both sets of framework-specific
requirements. It does not need ROOT, FairRoot, XPU, or a GPU:

```bash
bash external/KFParticle/KFParticle/GPU/test/cbmroot/test_cbmroot_kfp_preflight.sh
```

Every successful real preflight writes a compact evidence file under
`build/kfparticle-cbmroot-preflight`. Online uses
`online-<device>.evidence.txt`; offline uses separate
`offline-<input-kind>-<kfp-mode>-<device>.evidence.txt` files. They record the
selected source/build trees, device, build-time HIP architecture,
FairRoot/ROOT identity, input, configuration, geometry/parameters, steps, and
exact reconstruction command.

The Stage 19.2 target qualification completed with diagnostic contract 2,
all three focused CBMRoot tests passing, the embedded XPU smoke passing, and
two accepted event-based diagnostic runs of 100 entries. Offline FairRunAna
bring-up is therefore complete. Stage 19.3 subsequently completed the online
`cbmreco` chain and the controlled comparison of the two official modes.

After the online campaign, combine its evidence with the existing event-based
offline evidence:

```bash
KFPARTICLE_CBMROOT_BUILD_DIR=/home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/build \
KFPARTICLE_CBMROOT_DEVICE=hip1 \
KFPARTICLE_CBMROOT_CROSS_MODE_INPUT_RELATION=independent \
bash /home/kozlov/test_cbmroot/dev26/kfp-cbmroot-dev/cbmroot/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_kfp_cross_mode_qualification.sh
```

`independent` is required for the currently available real `.tsa` and
generated `.eb` samples. It qualifies the shared build, environment,
configuration, per-mode handoff, and CPU/GPU diagnostic contracts without
claiming equality of unrelated event counts. Use `equivalent` only for two
framework inputs derived from the same data; in that mode the script also
requires equal handoff counts.

## Step 19 Qualification Result

Step 19 is complete and target-server qualified (100%). On the target HIP
server, the bounded online campaign processed one real timeslice twice in
CPU-only mode and twice in diagnostic mode. All four runs completed, reused
the CBMRoot-owned XPU runtime with a KFParticle-owned queue, and produced
stable canonical summaries. CPU-only and diagnostic modes preserved the same
online event, track, vertex, PID, selected-track, particle, and candidate
handoff.

The final cross-mode launcher accepted the online evidence together with the
previously qualified event-based offline evidence. The current inputs are
explicitly `independent`, so this result qualifies both official framework
paths and their common build/runtime/diagnostic contracts without claiming
equal event counts for unrelated data. Equivalent-input numerical and
performance campaigns belong to Step 20.

## Step 20 Baseline Boundary

Stage 20.1 will extend these same launchers with opt-in performance evidence;
it will not introduce a third framework path. Cold startup, GPU tracking,
event building, CPU KFParticle reference, GPU KFParticle transaction,
comparison/materialization, and total wall time must be reported separately.
The current 50-event online `.tsa` campaign and bounded event-based offline
campaign are the initial integration baselines, while standalone synthetic
batch sizes provide controlled scaling and capacity coverage.

These inputs are sufficient to locate framework and KFParticle bottlenecks,
but they are not by themselves the final representative production set.
Stage 20.3 must add declared low-, typical-, and high-occupancy workloads and
freeze speed, memory, fallback, and numerical thresholds before making an
online or offline production decision.

Stage 20.1 instrumentation is now available through the same launchers. With
`KFPARTICLE_GPU_PERFORMANCE_MONITORING=1`, diagnostic runs emit one shared
performance record per resolved batch and evidence retains those records plus
complete-process `PERFORMANCE_RUN` wall time. CPU-only runs provide the
matching framework baseline without entering KFParticle GPU code. Monitoring
does not change the selected execution mode, event limit, CPU reference, or
fallback policy. Target-server standalone, online, and offline baselines have
closed Stages 20.1 and 20.2. A bounded online diagnostic run also passes the
strict locked contract after the real-data numerical-parity audit. The
remaining work is the Stage 20.3 trial campaign and separate online/offline
production decisions; accepted baselines alone do not unlock no-reference GPU
publication.
