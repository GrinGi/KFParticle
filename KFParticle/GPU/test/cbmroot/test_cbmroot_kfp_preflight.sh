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
printf 'KFP online summary:\npid_assigned=\nsecondary_pos=\ncpu_candidates=\ndecay_channels=\nrefit_policy=\nqualification locked\nkfp_gpu_qualification_build=ordinary-v1\n20260803-real-data-pv-parity-v11\n' >>"${build_dir}/lib/libAlgo.so"
printf 'kfp_gpu_qualification_build=ordinary-v1\n' >>"${build_dir}/lib/libAlgoOffline.so"
printf 'diagnostic_contract=\nqualification locked\n' >>"${build_dir}/lib/libCbmRecoTasks.so"
cat >"${build_dir}/bin/cbmreco" <<'EOF'
#!/usr/bin/env bash
mode="config"
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
echo "[fixture $$] INFO: KFP online handoff: contract=1 TS=0 events_available=5 events=${max_events} processed=1 no_pv=1 conversion_failed=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 pid_assigned=${pid_assigned} selected_tracks=${selected_tracks} secondary_pos=1 secondary_neg=1 primary_pos=0 primary_neg=0 cpu_particles=${cpu_particles} cpu_candidates=${cpu_candidates} decay_channels=3 pid_policy=${pid_policy} refit_policy=${refit_policy} mode=${mode} diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_seen} diagnostic_blocked=0 diagnostic_blockers=0"
echo "[fixture $$] INFO: KFP online summary: contract=1 timeslices=1 events_available=5 events=${max_events} processed=1 no_pv=1 conversion_failed=0 tracks=2 sts_hit_links=16 vertices=1 pid_slots=2 pid_assigned=${pid_assigned} selected_tracks=${selected_tracks} secondary_pos=1 secondary_neg=1 primary_pos=0 primary_neg=0 cpu_particles=${cpu_particles} cpu_candidates=${cpu_candidates} decay_channels=3 pid_policy=${pid_policy} refit_policy=${refit_policy} mode=${mode} diagnostic_contract=2 diagnostic_seen=${diagnostic_seen} diagnostic_ok=${diagnostic_seen} diagnostic_compared=${diagnostic_seen} diagnostic_matched=${diagnostic_seen} diagnostic_blocked=0 diagnostic_blockers=0"
exit 0
EOF
cat >"${build_dir}/bin/root" <<'EOF'
#!/usr/bin/env bash
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
)

online_log="${tmp_dir}/online.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${online_log}"
grep -q '^PASS cbmroot-kfp-online-preflight' "${online_log}"
grep -q -- '--kfp-mode cpu-only' "${online_log}"
grep -q -- '--systems STS TOF' "${online_log}"
grep -q -- '--event-reco-max-events 2' "${online_log}"
grep -q -- '--kfp-use-sts-pion-pid' "${online_log}"
grep -q -- '--kfp-refit-tracks' "${online_log}"

online_cpu_execute_log="${tmp_dir}/online-cpu-execute.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${online_cpu_execute_log}"
grep -q '^PASS cbmroot-kfp-online-preflight.*completed 2 run(s)' "${online_cpu_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/online-cpu-only-hip1.evidence.txt"

online_diagnostic_execute_log="${tmp_dir}/online-diagnostic-execute.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${online_diagnostic_execute_log}"
grep -q '^PASS cbmroot-kfp-online-preflight.*completed 2 run(s)' "${online_diagnostic_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/online-diagnostic-hip1.evidence.txt"

empty_online_log="${tmp_dir}/empty-online.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_CPU_CANDIDATES=0 \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${empty_online_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - empty online KFP result was accepted" >&2
  exit 1
fi
grep -q 'did not satisfy non-empty online KFP handoff contract 1' "${empty_online_log}"

offline_log="${tmp_dir}/offline.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_log}"
grep -q 'KfpSelectorChain prerequisites' "${offline_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-cpu-only-hip1.evidence.txt"

offline_event_log="${tmp_dir}/offline-event.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_event_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_event_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-event-based-cpu-only-hip1.evidence.txt"

offline_diagnostic_log="${tmp_dir}/offline-diagnostic.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_diagnostic_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight' "${offline_diagnostic_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.evidence.txt"

stale_library_log="${tmp_dir}/stale-library.log"
: >"${build_dir}/lib/libCbmRecoTasks.so"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${stale_library_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - stale CbmRecoTasks library was accepted" >&2
  exit 1
fi
grep -q 'libCbmRecoTasks.so predates diagnostic contract 2' "${stale_library_log}"
printf 'diagnostic_contract=\nqualification locked\n' >"${build_dir}/lib/libCbmRecoTasks.so"

offline_execute_log="${tmp_dir}/offline-execute.log"
printf 'completed fixture output\n' >"${source_dir}/offline-output.reco.root"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_OFFLINE_REPEAT=2 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_ROOT_SHUTDOWN_FAILURE=1 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_execute_log}"
grep -q '^PASS cbmroot-kfp-offline-preflight.*completed 2 run(s)' "${offline_execute_log}"
grep -q 'ignored FairRoot shutdown status 134 after complete output' "${offline_execute_log}"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.run-1.log"
test -f "${build_dir}/kfparticle-cbmroot-preflight/offline-time-based-diagnostic-hip1.run-2.log"

offline_event_cpu_execute_log="${tmp_dir}/offline-event-cpu-execute.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_KFP_MODE=cpu-only \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_event_cpu_execute_log}"

offline_event_diagnostic_execute_log="${tmp_dir}/offline-event-diagnostic-execute.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${offline_event_diagnostic_execute_log}"

online_campaign_log="${tmp_dir}/online-campaign.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=2 \
  bash "${script_dir}/run_cbmroot_kfp_online_campaign.sh" >"${online_campaign_log}"
grep -q '^PASS cbmroot-kfp-online-campaign' "${online_campaign_log}"

cross_mode_log="${tmp_dir}/cross-mode.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_CROSS_MODE_INPUT_RELATION=independent \
  bash "${script_dir}/run_cbmroot_kfp_cross_mode_qualification.sh" >"${cross_mode_log}"
grep -q '^PASS cbmroot-kfp-cross-mode' "${cross_mode_log}"

write_qualification_evidence()
{
  local path="$1"
  local mode="$2"
  local wall1="$3"
  local wall2="$4"
  local wall3="$5"
  {
    echo "source: ${source_dir}"
    echo "source revision: fixture-revision"
    echo "build: ${build_dir}"
    echo "qualification build: trial"
    echo "device: hip1"
    echo "XPU HIP architecture: gfx906;gfx908"
    echo "ROCm: fixture"
    echo "FairRoot: /fixture/fairroot"
    echo "ROOT: fixture"
    echo "input: fixture-input"
    echo "parameters: fixture-parameters"
    echo "configuration: ${tmp_dir}/qualification-gate.yaml"
    echo "result: KFP selector summary: mode=${mode} diagnostic_blocked=0 diagnostic_blockers=0"
    echo "PERFORMANCE_RUN mode=${mode} run=1 wall_ms=${wall1}"
    echo "PERFORMANCE_RUN mode=${mode} run=2 wall_ms=${wall2}"
    echo "PERFORMANCE_RUN mode=${mode} run=3 wall_ms=${wall3}"
  } >"${path}"
}

cat >"${tmp_dir}/qualification-gate.yaml" <<'EOF'
kfp:
  selector:
    decays:
      - pdg: 310
      - pdg: 3122
      - pdg: -3122
EOF
qualification_cpu="${tmp_dir}/qualification-cpu.evidence.txt"
qualification_diagnostic="${tmp_dir}/qualification-diagnostic.evidence.txt"
qualification_gpu="${tmp_dir}/qualification-gpu.evidence.txt"
write_qualification_evidence "${qualification_cpu}" cpu-only 1000 1010 990
write_qualification_evidence "${qualification_diagnostic}" diagnostic 1250 1260 1240
write_qualification_evidence "${qualification_gpu}" qualified-gpu 790 800 810
for run in 1 2 3; do
  echo "KFParticle GPU qualification metrics: events=10 completed=10 failed=0 cpu_reference_ms=100 transaction_ms=180" \
    >>"${qualification_diagnostic}"
  echo "KFParticle GPU diagnostics: events 10, timings [CPU reference 100, transaction 180] ms" \
    >>"${qualification_diagnostic}"
  echo "KFParticle GPU qualification metrics: events=10 completed=10 failed=0 cpu_reference_ms=0 transaction_ms=80" \
    >>"${qualification_gpu}"
  echo "KFParticle GPU diagnostics: events 10, timings [CPU reference 0, transaction 80] ms" \
    >>"${qualification_gpu}"
  echo "KFParticle GPU performance: samples=10 allocated_high_water_bytes=1048576" \
    >>"${qualification_gpu}"
  echo "KFParticle GPU routing monitoring: GPU accepted 9, CPU requested 0, diagnostic 0, not sampled 0, qualification locked 0, fallback unsupported/runtime/input/field/overflow/validation/materialization/execution 0/0/1/0/0/0/0/0" \
    >>"${qualification_gpu}"
done

qualification_log="${tmp_dir}/qualification.log"
env \
  KFPARTICLE_CBMROOT_QUALIFICATION_MODE=online \
  KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD=fixture \
  KFPARTICLE_CBMROOT_QUALIFICATION_CPU_EVIDENCE="${qualification_cpu}" \
  KFPARTICLE_CBMROOT_QUALIFICATION_DIAGNOSTIC_EVIDENCE="${qualification_diagnostic}" \
  KFPARTICLE_CBMROOT_QUALIFICATION_GPU_EVIDENCE="${qualification_gpu}" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_gate.sh" >"${qualification_log}"
grep -q '^PASS cbmroot-kfp-qualification-online' "${qualification_log}"
grep -q '^decision: qualified' "${tmp_dir}/qualification-online-fixture.evidence.txt"

qualification_slow_gpu="${tmp_dir}/qualification-slow-gpu.evidence.txt"
sed 's/wall_ms=790/wall_ms=1190/; s/wall_ms=800/wall_ms=1200/; s/wall_ms=810/wall_ms=1210/' \
  "${qualification_gpu}" >"${qualification_slow_gpu}"
qualification_slow_log="${tmp_dir}/qualification-slow.log"
if env \
  KFPARTICLE_CBMROOT_QUALIFICATION_MODE=offline \
  KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD=fixture \
  KFPARTICLE_CBMROOT_QUALIFICATION_CPU_EVIDENCE="${qualification_cpu}" \
  KFPARTICLE_CBMROOT_QUALIFICATION_DIAGNOSTIC_EVIDENCE="${qualification_diagnostic}" \
  KFPARTICLE_CBMROOT_QUALIFICATION_GPU_EVIDENCE="${qualification_slow_gpu}" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_gate.sh" >"${qualification_slow_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - slow qualified route was accepted" >&2
  exit 1
fi
grep -q 'end-to-end speedup .* is below' "${qualification_slow_log}"

cat >"${tmp_dir}/qualification-source.yaml" <<'EOF'
kfp:
  selector:
    decays:
      - pdg: 310
        minInvMass: 0.45
      - pdg: 3122
        minInvMass: 1.10
      - pdg: 3312
        minInvMass: 1.30
      - pdg: 3334
        minInvMass: 1.64
    preSelection:
      minTrackerHits: 3
    finderCuts:
      Chi2Cut2D: 3.
  pid:
    useStsELoss: false
EOF
python3 "${script_dir}/make_step20_v0_qualification_config.py" \
  "${tmp_dir}/qualification-source.yaml" "${tmp_dir}/qualification-v0.yaml" \
  >"${tmp_dir}/qualification-config.log"
grep -q 'pdg: 310' "${tmp_dir}/qualification-v0.yaml"
grep -q 'pdg: 3122' "${tmp_dir}/qualification-v0.yaml"
grep -q 'pdg: -3122' "${tmp_dir}/qualification-v0.yaml"
if grep -qE 'pdg: (3312|3334)|finderCuts:' "${tmp_dir}/qualification-v0.yaml"; then
  echo "FAIL cbmroot-kfp-preflight-contract - qualification config retained unsupported scope" >&2
  exit 1
fi

blocked_diagnostic_log="${tmp_dir}/blocked-diagnostic.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_DIAGNOSTIC_BLOCKED=1 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${blocked_diagnostic_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - blocked CPU/GPU comparison was accepted" >&2
  exit 1
fi
grep -q 'has missing, failed, or blocked CPU/GPU comparisons' "${blocked_diagnostic_log}"

stale_diagnostic_log="${tmp_dir}/stale-diagnostic.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=diagnostic \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_DIAGNOSTIC_SEEN=2 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${stale_diagnostic_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - duplicate diagnostic accounting was accepted" >&2
  exit 1
fi
grep -q 'has missing, failed, or blocked CPU/GPU comparisons' "${stale_diagnostic_log}"

early_failure_log="${tmp_dir}/early-root-failure.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=cpu-only \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_TEST_ROOT_EARLY_FAILURE=1 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${early_failure_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - early ROOT failure was accepted" >&2
  exit 1
fi
grep -q 'fixture reconstruction failed before completion' "${early_failure_log}"

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
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MissingEventSelector.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${missing_event_selector_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - missing eventSelector was accepted" >&2
  exit 1
fi
grep -q 'missing the top-level eventSelector node' "${missing_event_selector_log}"

sed '/doRefitTracks:/d' "${source_dir}/macro/algo/MainConfig.yaml" \
  >"${source_dir}/macro/algo/MissingDoRefitTracks.yaml"
missing_refit_log="${tmp_dir}/missing-refit.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MissingDoRefitTracks.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${missing_refit_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - missing doRefitTracks was accepted" >&2
  exit 1
fi
grep -q 'kfp.selector must define doRefitTracks' "${missing_refit_log}"

wrong_device_log="${tmp_dir}/wrong-device.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_DEVICE=cpu \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${wrong_device_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - non-HIP device was accepted" >&2
  exit 1
fi
grep -q 'requires an explicit HIP device' "${wrong_device_log}"

sed 's/recoParameters: reco.par.bin/recoParameters: missing-reco.par.bin/' \
  "${source_dir}/macro/algo/MainConfig.yaml" \
  >"${source_dir}/macro/algo/MissingParameters.yaml"
missing_parameters_log="${tmp_dir}/missing-parameters.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_CONFIG="${source_dir}/macro/algo/MissingParameters.yaml" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${missing_parameters_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - missing online parameters were accepted" >&2
  exit 1
fi
grep -q 'missing online reconstruction parameter archive' "${missing_parameters_log}"

single_backend_log="${tmp_dir}/single-backend.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_OFFLINE_TRACKING_BACKEND=gpu-single \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${single_backend_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - gpu-single was accepted for the Step 19 data" >&2
  exit 1
fi
grep -q 'require gpu-multi or gpu-multi-window' "${single_backend_log}"

large_event_log="${tmp_dir}/large-event.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=event-based \
  KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=101 \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${large_event_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - oversized event-based preflight was accepted" >&2
  exit 1
fi
grep -q 'event-based preflight is limited to 100 entries' "${large_event_log}"

qualified_mode_log="${tmp_dir}/qualified-mode.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${qualified_mode_log}" 2>&1
grep -q 'KFP mode   : qualified-gpu' "${qualified_mode_log}"

invalid_mode_log="${tmp_dir}/invalid-mode.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_KFP_MODE=automatic \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${invalid_mode_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - unknown offline mode was accepted" >&2
  exit 1
fi
grep -q 'offline KFP mode must be cpu-only, diagnostic, or qualified-gpu' "${invalid_mode_log}"

ordinary_as_trial_log="${tmp_dir}/ordinary-as-trial.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${ordinary_as_trial_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - ordinary build was accepted as a qualification trial" >&2
  exit 1
fi
grep -q 'qualification build mismatch: expected trial, found ordinary' "${ordinary_as_trial_log}"

ordinary_publish_log="${tmp_dir}/ordinary-publish.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu \
  KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION=published \
  KFPARTICLE_GPU_PERFORMANCE_MONITORING=1 \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${ordinary_publish_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - ordinary build was allowed to publish GPU results" >&2
  exit 1
fi
grep -q 'qualification build mismatch: expected trial, found ordinary' "${ordinary_publish_log}"

ordinary_build_dir="${tmp_dir}/ordinary-build"
cp -R "${build_dir}" "${ordinary_build_dir}"
sed 's/CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL:BOOL=OFF/CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL:BOOL=ON/' \
  "${build_dir}/CMakeCache.txt" >"${build_dir}/CMakeCache.txt.new"
mv "${build_dir}/CMakeCache.txt.new" "${build_dir}/CMakeCache.txt"
stale_trial_log="${tmp_dir}/stale-trial.log"
if env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${stale_trial_log}" 2>&1; then
  echo "FAIL cbmroot-kfp-preflight-contract - stale ordinary library was accepted by a trial cache" >&2
  exit 1
fi
grep -q 'libAlgo.so does not match CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON' "${stale_trial_log}"

for library in libAlgo.so libAlgoOffline.so; do
  sed 's/kfp_gpu_qualification_build=ordinary-v1/kfp_gpu_qualification_build=trial-v1/' \
    "${build_dir}/lib/${library}" >"${build_dir}/lib/${library}.new"
  mv "${build_dir}/lib/${library}.new" "${build_dir}/lib/${library}"
done
trial_online_log="${tmp_dir}/trial-online.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR="${source_dir}/online-input.tsa" \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial \
  bash "${script_dir}/run_cbmroot_kfp_online_preflight.sh" >"${trial_online_log}"
grep -q 'qualification build: trial' "${trial_online_log}"

trial_offline_log="${tmp_dir}/trial-offline.log"
env "${common_env[@]}" \
  KFPARTICLE_CBMROOT_OFFLINE_ROOT="${build_dir}/bin/root" \
  KFPARTICLE_CBMROOT_OFFLINE_CONFIG="${source_dir}/macro/algo/MainConfig.yaml" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX="${source_dir}/offline-input" \
  KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based \
  KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial \
  bash "${script_dir}/run_cbmroot_kfp_offline_preflight.sh" >"${trial_offline_log}"
grep -q 'qualification build: trial' "${trial_offline_log}"

trial_preflight_log="${tmp_dir}/qualification-trial-preflight.log"
env \
  KFPARTICLE_CBMROOT_SOURCE_DIR="${source_dir}" \
  KFPARTICLE_CBMROOT_ORDINARY_BUILD_DIR="${ordinary_build_dir}" \
  KFPARTICLE_CBMROOT_TRIAL_BUILD_DIR="${build_dir}" \
  KFPARTICLE_CBMROOT_DEVICE=hip1 \
  KFPARTICLE_CBMROOT_QUALIFICATION_SOURCE_CONFIG="${tmp_dir}/qualification-source.yaml" \
  KFPARTICLE_CBMROOT_QUALIFICATION_CONFIG="${tmp_dir}/qualification-trial.yaml" \
  KFPARTICLE_CBMROOT_QUALIFICATION_TRIAL_EVIDENCE="${tmp_dir}/qualification-trial.evidence.txt" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_trial_preflight.sh" >"${trial_preflight_log}"
grep -q '^PASS cbmroot-kfp-qualification-trial-preflight' "${trial_preflight_log}"
grep -q '^ordinary qualification trial: OFF$' "${tmp_dir}/qualification-trial.evidence.txt"
grep -q '^trial qualification trial: ON$' "${tmp_dir}/qualification-trial.evidence.txt"

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

echo "PASS cbmroot-kfp-preflight-contract - shared XPU/build checks, qualification-build identity, and independent online/offline contracts work"
