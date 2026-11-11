#!/usr/bin/env bash
set -euo pipefail

# Contract fixtures must not inherit paths or policy controls from a real run.
while IFS= read -r variable; do
  unset "${variable}"
done < <(compgen -A variable KFPARTICLE_ || true)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/kfparticle-cbmroot-preflight.XXXXXX")"
cleanup()
{
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "FAIL cbmroot-kfp-preflight-contract - internal command failed; collected logs follow" >&2
    local log_file
    for log_file in "${tmp_dir}"/*.log; do
      [[ -f "${log_file}" ]] || continue
      echo "---- ${log_file}" >&2
      tail -120 "${log_file}" >&2
    done
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

source_dir="${tmp_dir}/cbmroot"
build_dir="${tmp_dir}/build"
mkdir -p "${source_dir}/macro/algo" "${build_dir}/bin" "${build_dir}/lib"
touch "${source_dir}/CMakeLists.txt"
touch "${source_dir}/online-input.tsa"
touch "${source_dir}/offline-input.raw.root"
touch "${source_dir}/offline-input.par.root"
touch "${source_dir}/macro/algo/reco.par.bin"
cat >"${source_dir}/macro/algo/MainConfig.yaml" <<'EOF'
parFiles:
  recoParameters: reco.par.bin
caTimeslice:
  core: {}
caEvent:
  core: {}
trigger:
  detector: TOF
kfp:
  pid:
    maxMassSqDeviation: 5
    tofTimeMin: 0
    tofTimeMax: 100
    trackLengthMin: 0
    trackLengthMax: 1000
    useStsELoss: false
  selector:
    doRefitTracks: true
    decays: []
eventSelector:
  selectors: [HadronLambda]
  selectionType: MinimumBias
EOF
cat >"${source_dir}/run_reco_tracks.C" <<'EOF'
// FairRunAna preflight fixture
// CbmL1 GPU tracking preflight fixture
EOF
cat >"${build_dir}/CMakeCache.txt" <<'EOF'
CBM_KFPARTICLE_USE_XPU:BOOL=ON
CBM_KFPARTICLE_GPU_DIAGNOSTICS:BOOL=ON
CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL:BOOL=OFF
XPU_HIP_ARCH:STRING=gfx906;gfx908
FairRoot_DIR:PATH=/fixture/fairroot
ROOT_VERSION:STRING=fixture
EOF
for library in libxpu.so libKFParticle.so libxpu_Hip.so libKFParticle_Hip.so libAlgo.so libAlgoOffline.so libCbmRecoBase.so libCbmRecoTasks.so; do
  touch "${build_dir}/lib/${library}"
done
printf 'KFP online summary:\npid_assigned=\nsecondary_pos=\ncpu_candidates=\ndecay_channels=\nrefit_policy=\nbatch_limit=\nqualification locked\nkfp_gpu_qualification_build=ordinary-v1\n20260803-real-data-pv-parity-v11\n20260820-kfp-gpu-single-host-owner-v1\n' >>"${build_dir}/lib/libAlgo.so"
printf 'kfp_gpu_qualification_build=ordinary-v1\n' >>"${build_dir}/lib/libAlgoOffline.so"
printf 'diagnostic_contract=\nbatch_limit=\nqualification locked\nadaptive work=\n' >>"${build_dir}/lib/libCbmRecoTasks.so"
cat >"${build_dir}/bin/cbmreco" <<'EOF'
#!/usr/bin/env bash
mode="config"
if [[ "${KFPARTICLE_GPU_REPORT:-0}" != 1 ]]; then
  echo "fixture did not receive the explicit test-report request" >&2
  exit 1
fi
if [[ "${KFPARTICLE_GPU_PERFORMANCE_MONITORING:-0}" != "${KFPARTICLE_TEST_EXPECT_PERFORMANCE_MONITORING:-0}" ]]; then
  echo "requesting test reports changed performance monitoring" >&2
  exit 1
fi
if [[ "${KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES:-unset}" != "${KFPARTICLE_TEST_EXPECT_RESOURCE_RETRY:-0}" ]]; then
  echo "fixture received an unexpected resource retry mode" >&2
  exit 1
fi
max_events=0
pid_policy="detector"
refit_policy="disabled"
while (($#)); do
  case "$1" in
    --kfp-mode)
      mode="$2"
      shift 2
      ;;
    --event-reco-max-events)
      max_events="$2"
      shift 2
      ;;
    --kfp-use-sts-pion-pid)
      pid_policy="sts-pion"
      shift
      ;;
    --kfp-refit-tracks)
      refit_policy="enabled"
      shift
      ;;
    *)
      shift
      ;;
  esac
done
diagnostic_seen=0
if [[ "${mode}" == "diagnostic" ]]; then
  diagnostic_seen=1
fi
pid_assigned="${KFPARTICLE_TEST_PID_ASSIGNED:-2}"
selected_tracks="${KFPARTICLE_TEST_SELECTED_TRACKS:-2}"
cpu_particles="${KFPARTICLE_TEST_CPU_PARTICLES:-3}"
cpu_candidates="${KFPARTICLE_TEST_CPU_CANDIDATES:-1}"
observed_events="${KFPARTICLE_TEST_EVENTS:-${max_events}}"
if [[ "${max_events}" == 0 ]]; then observed_events="${KFPARTICLE_TEST_EVENTS:-5}"; fi
echo "[fixture $$] INFO: KFP online handoff: contract=1 handoff_phases=2 TS=0 events_available=5 events=${observed_events} processed=1 no_pv=1 conversion_failed=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 pid_assigned=${pid_assigned} selected_tracks=${selected_tracks} secondary_pos=1 secondary_neg=1 primary_pos=0 primary_neg=0 cpu_particles=${cpu_particles} cpu_candidates=${cpu_candidates} decay_channels=3 pid_policy=${pid_policy} refit_policy=${refit_policy} mode=${mode} diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_seen} diagnostic_blocked=0 diagnostic_blockers=0"
echo "[fixture $$] INFO: KFP online summary: contract=1 handoff_phases=2 timeslices=1 events_available=5 events=${observed_events} processed=1 no_pv=1 conversion_failed=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 pid_assigned=${pid_assigned} selected_tracks=${selected_tracks} secondary_pos=1 secondary_neg=1 primary_pos=0 primary_neg=0 cpu_particles=${cpu_particles} cpu_candidates=${cpu_candidates} decay_channels=3 pid_policy=${pid_policy} refit_policy=${refit_policy} mode=${mode} diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_seen} diagnostic_blocked=0 diagnostic_blockers=0"
exit 0
EOF
cat >"${build_dir}/bin/root" <<'EOF'
#!/usr/bin/env bash
if [[ "${KFPARTICLE_GPU_REPORT:-0}" != 1 ]]; then
  echo "fixture did not receive the explicit test-report request" >&2
  exit 1
fi
if [[ "${KFPARTICLE_GPU_PERFORMANCE_MONITORING:-0}" != "${KFPARTICLE_TEST_EXPECT_PERFORMANCE_MONITORING:-0}" ]]; then
  echo "requesting test reports changed performance monitoring" >&2
  exit 1
fi
if [[ "${KFPARTICLE_TEST_ROOT_EARLY_FAILURE:-0}" == "1" ]]; then
  echo "fixture reconstruction failed before completion" >&2
  exit 2
fi
diagnostic_seen="${KFPARTICLE_TEST_DIAGNOSTIC_SEEN:-1}"
diagnostic_matched="${diagnostic_seen}"
diagnostic_blocked=0
diagnostic_blockers=0
if [[ "${KFPARTICLE_TEST_DIAGNOSTIC_BLOCKED:-0}" == "1" ]]; then
  diagnostic_matched=0
  diagnostic_blocked=1
  diagnostic_blockers=32
fi
echo "[fixture $$] INFO: KFP selector handoff: TS=0 events=1 processed=1 no_pv=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 source_base=0 field=ca-kf-setup masks=1 mode=diagnostic diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_matched} diagnostic_blocked=${diagnostic_blocked} diagnostic_blockers=${diagnostic_blockers}"
echo "[fixture $$] INFO: KFP selector summary: timeslices=1 events=1 processed=1 no_pv=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 field=ca-kf-setup mode=diagnostic diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_matched} diagnostic_blocked=${diagnostic_blocked} diagnostic_blockers=${diagnostic_blockers}"
echo "-I- run_reco_tracks: finished"
echo "-I- run_reco_tracks: output fixture.reco.root"
if [[ "${KFPARTICLE_TEST_ROOT_SHUTDOWN_FAILURE:-0}" == "1" ]]; then
  echo "double free or corruption (!prev)" >&2
  exit 134
fi
exit 0
EOF
chmod +x "${build_dir}/bin/cbmreco" "${build_dir}/bin/root"

common_env=(
  KFPARTICLE_CBMROOT_SOURCE_DIR="${source_dir}"
  KFPARTICLE_CBMROOT_BUILD_DIR="${build_dir}"
  KFPARTICLE_CBMROOT_DEVICE=hip1
  KFPARTICLE_CBMROOT_KFP_MODE=cpu-only
  KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD=1
  KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION=locked
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=any
  KFPARTICLE_GPU_REPORT=0
  KFPARTICLE_GPU_PERFORMANCE_MONITORING=0
  KFPARTICLE_GPU_DEEP_TRACE=0
  KFPARTICLE_GPU_DEEP_TRACE_COMPACT=0
  KFPARTICLE_CPU_FINDER_TRACE_PAIRS=
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=0
  KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE=
  KFPARTICLE_CBMROOT_PREFLIGHT_SOURCE_CONFIG=0
  KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE=0
  KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_LINK_CHECK=1
  KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_HASH=1
  KFPARTICLE_CBMROOT_ONLINE_EXECUTABLE="${build_dir}/bin/cbmreco"
  KFPARTICLE_CBMROOT_ONLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml"
  KFPARTICLE_CBMROOT_ONLINE_STEPS="Unpack DigiTrigger LocalReco Tracking EventReco KfpSelector"
  KFPARTICLE_CBMROOT_ONLINE_SYSTEMS=STS
  KFPARTICLE_CBMROOT_ONLINE_TRACKING_BACKEND=gpu-multi-window
  KFPARTICLE_CBMROOT_ONLINE_NUM_TS=1
  KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=2
  KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID=1
  KFPARTICLE_CBMROOT_ONLINE_REFIT_TRACKS=1
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=1
  KFPARTICLE_CBMROOT_OFFLINE_MACRO="${script_dir}/run_reco_tracks.C"
  KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX="${source_dir}/offline-output"
  KFPARTICLE_CBMROOT_OFFLINE_PARAMETER_PREFIX="${source_dir}/offline-input"
  KFPARTICLE_CBMROOT_OFFLINE_SETUP=sis100_hadron
  KFPARTICLE_CBMROOT_OFFLINE_FIRST_ENTRY=0
  KFPARTICLE_CBMROOT_OFFLINE_ENABLE_MC_QA=0
  KFPARTICLE_CBMROOT_OFFLINE_TRACKING_BACKEND=gpu-multi
  KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=1
  KFPARTICLE_CBMROOT_OFFLINE_REPEAT=1
  KFPARTICLE_CBMROOT_OFFLINE_ALLOW_LARGE_EVENT_RUN=0
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa"
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root"
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml"
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input"
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based
)

run_preflight()
{
  local mode="$1"
  shift
  env "${common_env[@]}" "$@" bash "${script_dir}/run_cbmroot_kfp_${mode}_preflight.sh"
}

expect_failure()
{
  local message="$1" log="$2"
  shift 2
  if "$@" >"$log" 2>&1; then
    echo "FAIL preflight accepted a case requiring: $message" >&2
    exit 1
  fi
  grep -q -- "$message" "$log"
}

online_log="${tmp_dir}/online.log"
run_preflight online >"${online_log}"
grep -q '^PASS cbmroot-kfp-online-preflight' "${online_log}"
grep -q -- '--kfp-mode cpu-only' "${online_log}"
grep -q -- '--systems STS TOF' "${online_log}"
grep -q -- '--event-reco-max-events 2' "${online_log}"
grep -q '^  STS memory : exact per timeslice$' "${online_log}"
grep -q '^  KFP materialization schedule: guided$' "${online_log}"
grep -q '^  KFP resource retry: 0 ' "${online_log}"
grep -q -- '--kfp-use-sts-pion-pid' "${online_log}"
grep -q -- '--kfp-refit-tracks' "${online_log}"

full_timeslice_log="${tmp_dir}/full-timeslice.log"
run_preflight online KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=0 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 >"${full_timeslice_log}"
grep -q -- '--event-reco-max-events 0' "${full_timeslice_log}"
grep -q '^PASS cbmroot-kfp-online-preflight' "${full_timeslice_log}"
for retry in 0 1; do
  retry_log="${tmp_dir}/resource-retry-${retry}.log"
  run_preflight online KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
    KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES="${retry}" \
    KFPARTICLE_TEST_EXPECT_RESOURCE_RETRY="${retry}" >"${retry_log}"
  grep -q "^  KFP resource retry: ${retry} " "${retry_log}"
  grep -q "resource-retry=${retry}" \
    "${build_dir}/kfparticle-cbmroot-preflight/online-cpu-only-hip1.evidence.txt"
done
for invalid in -1 2 bad; do
  expect_failure 'RETRY_RESOURCE_BATCHES must be 0 or 1' "${tmp_dir}/invalid-resource-retry.log" \
    run_preflight online KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES="${invalid}"
done
expect_failure 'did not satisfy non-empty online KFP handoff contract 1' \
  "${tmp_dir}/truncated-timeslice.log" run_preflight online \
  KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=0 KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_EVENTS=4
for invalid in -1 bad; do
  expect_failure 'MAX_EVENTS must be non-negative' "${tmp_dir}/invalid-limit.log" \
    run_preflight online KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS="$invalid"
done

benchmark_schedule_log="${tmp_dir}/benchmark-materialization-schedule.log"
if run_preflight online \
  KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE=weighted-contiguous \
    >"${benchmark_schedule_log}" 2>&1; then
  echo "FAIL preflight accepted benchmark-only materialization schedule without monitoring" >&2
  exit 1
fi
grep -q 'non-default materialization schedules require performance monitoring' \
  "${benchmark_schedule_log}"

for profile in 0 1; do
  timer_log="${tmp_dir}/construction-timers-${profile}.log"
  if run_preflight online \
      KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
    KFPARTICLE_GPU_CONSTRUCTION_TIMERS=1 XPU_PROFILE="${profile}" >"${timer_log}" 2>&1; then
    echo "FAIL preflight accepted missing construction device timings" >&2
    exit 1
  fi
  if [[ "${profile}" == 0 ]]; then
    grep -q 'construction timers require XPU_PROFILE=1' "${timer_log}"
  else
    grep -q 'missing detailed XPU construction timers' "${timer_log}"
  fi
done

online_batch_alias_log="${tmp_dir}/online-batch-alias.log"
run_preflight online \
  KFPARTICLE_CBMROOT_EVENT_BATCH_SIZE=64 \
  KFPARTICLE_CBMROOT_KFP_MATERIALIZATION_THREADS=3 \
  KFPARTICLE_CBMROOT_KFP_PACKING_THREADS=2 >"${online_batch_alias_log}"
grep -q '^  KFP batch  : 64 event(s)$' "${online_batch_alias_log}"
grep -q '^  KFP materialization threads: 3 ' "${online_batch_alias_log}"
grep -q '^  KFP packing threads: 2 ' "${online_batch_alias_log}"
grep -q -- '--kfp-diagnostic-batch-events 64' "${online_batch_alias_log}"
grep -q -- '--kfp-materialization-threads 3' "${online_batch_alias_log}"
grep -q -- '--kfp-packing-threads 2' "${online_batch_alias_log}"

online_cpu_backend_log="${tmp_dir}/online-cpu-backend.log"
run_preflight online \
  KFPARTICLE_CBMROOT_ONLINE_TRACKING_BACKEND=cpu >"${online_cpu_backend_log}"
grep -q '^PASS cbmroot-kfp-online-preflight' "${online_cpu_backend_log}"
grep -q -- '--ca-timeslice-tracking-backend cpu' "${online_cpu_backend_log}"

online_cpu_execute_log="${tmp_dir}/online-cpu-execute.log"
run_preflight online \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 >"${online_cpu_execute_log}"
grep -q '^PASS cbmroot-kfp-online-preflight.*completed 2 run(s)' "${online_cpu_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/online-cpu-only-hip1.evidence.txt"

online_cpu_adaptive_log="${tmp_dir}/online-cpu-adaptive.log"
run_preflight online \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_CBMROOT_ADAPTIVE_MINIMUM_WORK=1 \
  KFPARTICLE_CBMROOT_ADAPTIVE_TARGET_WORK=1 \
  KFPARTICLE_CBMROOT_ADAPTIVE_MAXIMUM_BYTES=1024 >"${online_cpu_adaptive_log}"
grep -q '^PASS cbmroot-kfp-online-preflight' "${online_cpu_adaptive_log}"

online_diagnostic_execute_log="${tmp_dir}/online-diagnostic-execute.log"
run_preflight online \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 >"${online_diagnostic_execute_log}"
grep -q '^PASS cbmroot-kfp-online-preflight.*completed 2 run(s)' "${online_diagnostic_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/online-diagnostic-hip1.evidence.txt"

empty_online_log="${tmp_dir}/empty-online.log"
expect_failure 'did not satisfy non-empty online KFP handoff contract 1' "${empty_online_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_CPU_CANDIDATES=0

offline_log="${tmp_dir}/offline.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MATERIALIZATION_THREADS=3 \
  KFPARTICLE_CBMROOT_KFP_PACKING_THREADS=2 >"${offline_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_log}"
grep -q 'KfpSelectorChain prerequisites' "${offline_log}"
grep -q '^  KFP materialization threads: 3 ' "${offline_log}"
grep -q '^  KFP packing threads: 2 ' "${offline_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-cpu-only-hip1.evidence.txt"

offline_event_log="${tmp_dir}/offline-event.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based >"${offline_event_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_event_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-event-based-cpu-only-hip1.evidence.txt"

offline_diagnostic_log="${tmp_dir}/offline-diagnostic.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic >"${offline_diagnostic_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_diagnostic_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.evidence.txt"

stale_library_log="${tmp_dir}/stale-library.log"
: >"${build_dir}/lib/libCbmRecoTasks.so"
expect_failure 'libCbmRecoTasks.so predates diagnostic contract 2' "${stale_library_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic
printf 'diagnostic_contract=\nbatch_limit=\nqualification locked\nadaptive work=\n' >"${build_dir}/lib/libCbmRecoTasks.so"

offline_execute_log="${tmp_dir}/offline-execute.log"
printf 'completed fixture output\n' >"${source_dir}/offline-output.reco.root"
run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_OFFLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_ROOT_SHUTDOWN_FAILURE=1 >"${offline_execute_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight.*completed 2 run(s)' "${offline_execute_log}"
grep -q 'ignored FairRoot shutdown status 134 after complete output' "${offline_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.run-1.log"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.run-2.log"

offline_event_cpu_execute_log="${tmp_dir}/offline-event-cpu-execute.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_KFP_MODE=cpu-only \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 >"${offline_event_cpu_execute_log}"

offline_event_diagnostic_execute_log="${tmp_dir}/offline-event-diagnostic-execute.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 >"${offline_event_diagnostic_execute_log}"

blocked_diagnostic_log="${tmp_dir}/blocked-diagnostic.log"
expect_failure 'did not satisfy diagnostic physics contract 2' "${blocked_diagnostic_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_DIAGNOSTIC_BLOCKED=1

retained_diagnostic_log="${tmp_dir}/retained-blocked-diagnostic.log"
retained_diagnostic_evidence="${tmp_dir}/retained-blocked-diagnostic.evidence.txt"
run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_CBMROOT_PREFLIGHT_ALLOW_DIAGNOSTIC_REJECTION=1 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE="${retained_diagnostic_evidence}" \
  KFPARTICLE_TEST_DIAGNOSTIC_BLOCKED=1 >"${retained_diagnostic_log}" 2>&1
grep -q 'REJECTED run 1/1: diagnostic physics contract blocked' "${retained_diagnostic_log}"
grep -Eq '^result: .*diagnostic_blocked=1 .*diagnostic_blockers=32' "${retained_diagnostic_evidence}"

stale_diagnostic_log="${tmp_dir}/stale-diagnostic.log"
expect_failure 'did not complete diagnostic contract 2' "${stale_diagnostic_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_DIAGNOSTIC_SEEN=2

early_failure_log="${tmp_dir}/early-root-failure.log"
expect_failure 'fixture reconstruction failed before completion' "${early_failure_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=cpu-only \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_ROOT_EARLY_FAILURE=1

cat >"${source_dir}/macro/algo/MissingEventSelector.yaml" <<'EOF'
kfp:
  pid:
    maxMassSqDeviation: 5
    tofTimeMin: 0
    tofTimeMax: 100
    trackLengthMin: 0
    trackLengthMax: 1000
    useStsELoss: false
  selector:
    doRefitTracks: true
    decays: []
EOF
missing_event_selector_log="${tmp_dir}/missing-event-selector.log"
expect_failure 'missing the top-level eventSelector node' "${missing_event_selector_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MissingEventSelector.yaml"

sed '/doRefitTracks:/d' "${source_dir}/macro/algo/MainConfig.yaml" \
  >"${source_dir}/macro/algo/MissingDoRefitTracks.yaml"
missing_refit_log="${tmp_dir}/missing-refit.log"
expect_failure 'kfp.selector must define doRefitTracks' "${missing_refit_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MissingDoRefitTracks.yaml"

wrong_device_log="${tmp_dir}/wrong-device.log"
expect_failure 'requires an explicit HIP device' "${wrong_device_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_DEVICE=cpu

sed 's/recoParameters: reco.par.bin/recoParameters: missing-reco.par.bin/' \
  "${source_dir}/macro/algo/MainConfig.yaml" \
  >"${source_dir}/macro/algo/MissingParameters.yaml"
missing_parameters_log="${tmp_dir}/missing-parameters.log"
expect_failure 'missing online reconstruction parameter archive' "${missing_parameters_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_ONLINE_CONFIG="${source_dir}/macro/algo/MissingParameters.yaml"

single_backend_log="${tmp_dir}/single-backend.log"
expect_failure 'require gpu-multi or gpu-multi-window' "${single_backend_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_TRACKING_BACKEND=gpu-single

large_event_log="${tmp_dir}/large-event.log"
expect_failure 'event-based preflight is limited to 100 entries' "${large_event_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=101

qualified_mode_log="${tmp_dir}/qualified-mode.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu >"${qualified_mode_log}" 2>&1
grep -q 'KFP mode   : qualified-gpu' "${qualified_mode_log}"

invalid_mode_log="${tmp_dir}/invalid-mode.log"
expect_failure 'offline KFP mode must be cpu-only, diagnostic, or qualified-gpu' "${invalid_mode_log}" \
  run_preflight offline \
  KFPARTICLE_CBMROOT_KFP_MODE=automatic

ordinary_as_trial_log="${tmp_dir}/ordinary-as-trial.log"
expect_failure 'qualification build mismatch: expected trial, found ordinary' "${ordinary_as_trial_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial

ordinary_publish_log="${tmp_dir}/ordinary-publish.log"
expect_failure 'qualification build mismatch: expected trial, found ordinary' "${ordinary_publish_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu \
  KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION=published \
  KFPARTICLE_GPU_PERFORMANCE_MONITORING=1

ordinary_build_dir="${tmp_dir}/ordinary-build"
cp -R "${build_dir}" "${ordinary_build_dir}"
sed 's/CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL:BOOL=OFF/CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL:BOOL=ON/' \
  "${build_dir}/CMakeCache.txt" >"${build_dir}/CMakeCache.txt.new"
mv "${build_dir}/CMakeCache.txt.new" "${build_dir}/CMakeCache.txt"
stale_trial_log="${tmp_dir}/stale-trial.log"
expect_failure 'libAlgo.so does not match CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON' "${stale_trial_log}" \
  run_preflight online \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial

for library in libAlgo.so libAlgoOffline.so; do
  sed 's/kfp_gpu_qualification_build=ordinary-v1/kfp_gpu_qualification_build=trial-v1/' \
    "${build_dir}/lib/${library}" >"${build_dir}/lib/${library}.new"
  mv "${build_dir}/lib/${library}.new" "${build_dir}/lib/${library}"
done
trial_online_log="${tmp_dir}/trial-online.log"
run_preflight online \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial >"${trial_online_log}"
grep -q 'qualification build: trial' "${trial_online_log}"

missing_published_policy_log="${tmp_dir}/missing-published-policy.log"
if run_preflight online \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu \
  KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION=published \
  KFPARTICLE_GPU_PERFORMANCE_MONITORING=1 \
    >"${missing_published_policy_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - published GPU run accepted a disabled adaptive policy" >&2
  exit 1
fi
grep -q 'published qualified-gpu execution requires an enabled adaptive policy' \
  "${missing_published_policy_log}"

trial_offline_log="${tmp_dir}/trial-offline.log"
run_preflight offline \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial >"${trial_offline_log}"
grep -q 'qualification build: trial' "${trial_offline_log}"

performance_report_log="${tmp_dir}/performance-report.log"
cat >"${performance_report_log}" <<'EOF'
[fixture] INFO: unrelated prefix
[fixture] INFO: KFParticle GPU performance report (host wall time; monitoring enabled)
[fixture] INFO:   Scope      events=2 completed=2 failed=0 samples=1 batches=1 largest_batch=2
[fixture] INFO:   Pipeline (execution order)
[fixture] INFO:   01  CPU       reference finder                    1.000 ms
[fixture] INFO:   06  COPY H2D  input upload                        2.000 ms  1024 B (0.001 MiB)
[fixture] INFO:   Note       GPU/QUEUE values are host wall intervals, not device-event kernel timings
[fixture] INFO: unrelated suffix
EOF
performance_report="$({
  source "${script_dir}/cbmroot_kfp_preflight_common.sh"
  kfp_preflight_print_performance_report "${performance_report_log}"
})"
grep -q 'KFParticle GPU performance report' <<<"${performance_report}"
grep -q '06  COPY H2D' <<<"${performance_report}"
grep -q 'GPU/QUEUE values are host wall intervals' <<<"${performance_report}"
if grep -q 'unrelated' <<<"${performance_report}"; then
  echo "FAIL cbmroot-kfp-preflight-contract - performance report extraction escaped its block" >&2
  exit 1
fi

# Exercise all regression modes without a ROOT installation or GPU.
cat >"${build_dir}/bin/root" <<'EOF'
#!/usr/bin/env bash
[[ ! -f "$(dirname "$0")/reject-smoke" ]] || exit 1
[[ ! -f "$(dirname "$0")/no-smoke-marker" ]] || exit 0
case "$*" in
  *cbmroot_xpu_smoke.C*) echo 'PASS cbmroot-xpu-runtime' ;;
  *) exit 2 ;;
esac
EOF
cat >"${build_dir}/bin/KFParticleCpuGpuV0Equivalence" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == hip1 && "$XPU_DEVICE" == hip1 ]] || exit 2
[[ ! -f "$(dirname "$0")/reject" ]] || exit 1
[[ ! -f "$(dirname "$0")/no-marker" ]] || exit 0
echo 'PASS cbmroot-cpu-gpu-v0-equivalence'
EOF
chmod +x "${build_dir}/bin/KFParticleCpuGpuV0Equivalence"
regression_env=(
  KFPARTICLE_CBMROOT_BUILD_DIR="$build_dir"
  KFPARTICLE_CBMROOT_DEVICE=hip1
  KFPARTICLE_CBMROOT_EQUIVALENCE_SKIP_BUILD=1
  KFPARTICLE_CBMROOT_SMOKE_LOG_DIR="$tmp_dir/regression"
  ROOTSYS="$build_dir"
)
for mode in runtime equivalence all; do
  env "${regression_env[@]}" bash "$script_dir/run_cbmroot_regression.sh" "$mode" \
    >"$tmp_dir/regression-$mode.log" 2>&1
done
[[ "$(grep -c '^PASS cbmroot-' "$tmp_dir/regression-all.log")" == 3 ]]
evidence="$tmp_dir/regression/cbmroot-cpu-gpu-v0-equivalence-hip1.evidence.txt"
grep -q '^build_skipped=1$' "$evidence"
for failure in reject-smoke no-smoke-marker; do
  touch "$build_dir/bin/$failure"
  if env "${regression_env[@]}" bash "$script_dir/run_cbmroot_regression.sh" all \
      >"$tmp_dir/regression-$failure.log" 2>&1; then
    echo "FAIL regression accepted $failure" >&2; exit 1
  fi
  [[ ! -e "$evidence" ]]
  ! grep -q '^PASS cbmroot-cpu-gpu-v0-equivalence' "$tmp_dir/regression-$failure.log"
  rm "$build_dir/bin/$failure"
done
for failure in reject no-marker; do
  touch "$build_dir/bin/$failure"
  if env "${regression_env[@]}" bash "$script_dir/run_cbmroot_regression.sh" equivalence \
      >"$tmp_dir/regression-$failure.log" 2>&1; then
    echo "FAIL regression accepted $failure" >&2; exit 1
  fi
  [[ ! -e "$evidence" ]]
  rm "$build_dir/bin/$failure"
done
if env "${regression_env[@]}" bash "$script_dir/run_cbmroot_regression.sh" unknown \
    >"$tmp_dir/regression-unknown.log" 2>&1; then
  echo "FAIL regression accepted unknown mode" >&2; exit 1
fi
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "$source_dir" >>"$build_dir/CMakeCache.txt"
if env "${regression_env[@]}" bash "$script_dir/run_cbmroot_regression.sh" runtime \
    >"$tmp_dir/regression-mismatch.log" 2>&1; then
  echo "FAIL regression accepted wrong source tree" >&2; exit 1
fi
grep -q 'build/source mismatch' "$tmp_dir/regression-mismatch.log"

echo "PASS cbmroot-kfp-preflight-contract - online/offline policies and regression modes, failure propagation, evidence and source identity"
