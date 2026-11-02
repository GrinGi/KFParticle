#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/cbmroot_kfp_preflight_common.sh"
kfp_preflight_initialize "online"
kfp_preflight_options

cbmreco="${KFPARTICLE_CBMROOT_ONLINE_EXECUTABLE:-${KFP_PREFLIGHT_BUILD_DIR}/bin/cbmreco}"
config="${KFPARTICLE_CBMROOT_ONLINE_CONFIG:-${KFP_PREFLIGHT_SOURCE_DIR}/macro/algo/MainConfig.yaml}"
input="${KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR:-}"
steps="${KFPARTICLE_CBMROOT_ONLINE_STEPS:-Unpack DigiTrigger LocalReco Tracking EventReco KfpSelector}"
systems="${KFPARTICLE_CBMROOT_ONLINE_SYSTEMS:-STS}"
backend="${KFPARTICLE_CBMROOT_ONLINE_TRACKING_BACKEND:-gpu-multi-window}"
num_ts="${KFPARTICLE_CBMROOT_ONLINE_NUM_TS:-1}"
max_events="${KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS:-50}"
omp_threads="${KFPARTICLE_CBMROOT_ONLINE_OMP_THREADS:-0}"
use_sts_pion_pid="${KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID:-1}"
refit_tracks="${KFPARTICLE_CBMROOT_ONLINE_REFIT_TRACKS:-1}"
materialization_schedule="${KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE:-guided}"
sts_exact_memory="${KFPARTICLE_CBMROOT_STS_EXACT_MEMORY:-1}"
deep_trace="${KFPARTICLE_GPU_DEEP_TRACE:-0}"
deep_trace_compact="${KFPARTICLE_GPU_DEEP_TRACE_COMPACT:-0}"
cpu_finder_trace_pairs="${KFPARTICLE_CPU_FINDER_TRACE_PAIRS:-}"
cpu_finder_trace_build_id='20260806-cpu-finder-routing-v16'
transaction_reset_build_id='20260820-transaction-journal-v18'
gpu_owner_build_id='20260820-kfp-gpu-single-host-owner-v1'
repeat="${KFPARTICLE_CBMROOT_ONLINE_REPEAT:-1}"
transaction_diagnostics="${KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS:-0}"
retry_resource_batches="${KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES:-0}"
construction_timers="${KFPARTICLE_GPU_CONSTRUCTION_TIMERS:-0}"
if [[ "${construction_timers}" == "1" && "${XPU_PROFILE:-0}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "construction timers require XPU_PROFILE=1 before XPU initialization (intrusive diagnostic run)"
fi

kfp_preflight_require_executable "${cbmreco}" "cbmreco executable"
kfp_preflight_require_file "${config}" "online main configuration"
kfp_preflight_require_file "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" "online reconstruction library"
if [[ -n "${cpu_finder_trace_pairs}" ]] \
   && { [[ ! -f "${KFP_PREFLIGHT_LIBRARY_DIR}/libKFParticle.so" ]] \
        || ! grep -a -q "${cpu_finder_trace_build_id}" "${KFP_PREFLIGHT_LIBRARY_DIR}/libKFParticle.so"; }; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libKFParticle.so does not contain CPU Finder routing trace ${cpu_finder_trace_build_id}; rebuild KFParticle, Algo, and cbmreco"
fi
if [[ -z "${input}" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "set KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR to a local timeslice file/directory or source URI"
fi
kfp_preflight_input_identity "${input}"

kfp_preflight_validate_options "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"

for required_step in Tracking EventReco KfpSelector; do
  if ! kfp_preflight_contains_word "${steps}" "${required_step}"; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "online steps must include ${required_step}: ${steps}"
  fi
done
if ! kfp_preflight_contains_word "${systems}" "STS"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "online systems must include STS: ${systems}"
fi
case "${backend}" in
  cpu|gpu-multi-window) ;;
  *) kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
       "online CA tracking backend must be cpu or gpu-multi-window" ;;
esac
if [[ ! "${num_ts}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_ONLINE_NUM_TS must be positive"
fi
if [[ ! "${max_events}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS must be non-negative (0 = all)"
fi
if [[ ! "${omp_threads}" =~ ^[0-9]+$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_ONLINE_OMP_THREADS must be a non-negative integer"
fi
if [[ "${use_sts_pion_pid}" != "0" && "${use_sts_pion_pid}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID must be 0 or 1"
fi
if [[ "${refit_tracks}" != "0" && "${refit_tracks}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_ONLINE_REFIT_TRACKS must be 0 or 1"
fi
if [[ "${sts_exact_memory}" != "0" && "${sts_exact_memory}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_STS_EXACT_MEMORY must be 0 or 1"
fi
if [[ "${deep_trace}" != "0" && "${deep_trace}" != "ALL" \
      && ! "${deep_trace}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_GPU_DEEP_TRACE must be 0, ALL, or a positive mismatch-event limit"
fi
if [[ "${deep_trace_compact}" != "0" && "${deep_trace_compact}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_GPU_DEEP_TRACE_COMPACT must be 0 or 1"
fi
if [[ ! "${repeat}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_ONLINE_REPEAT must be positive"
fi
case "${materialization_schedule}" in
  guided) ;;
  weighted-contiguous|static|dynamic)
    if [[ "${performance_monitoring}" != "1" ]]; then
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "non-default materialization schedules require performance monitoring"
    fi
    ;;
  *)
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE must be weighted-contiguous, static, dynamic, or guided"
    ;;
esac
if [[ "${transaction_diagnostics}" != "0" && "${transaction_diagnostics}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS must be 0 or 1"
fi
if [[ "${retry_resource_batches}" != "0" && "${retry_resource_batches}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES must be 0 or 1"
fi
if ! grep -q '^kfp:' "${config}" || ! grep -q '^eventSelector:' "${config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "online MainConfig.yaml must contain top-level kfp and eventSelector nodes: ${config}"
fi

trigger_detector="$(
  awk '
    /^trigger:[[:space:]]*($|#)/ {
      in_trigger = 1
      next
    }
    in_trigger && /^[^[:space:]#]/ {
      exit
    }
    in_trigger && /^[[:space:]]+detector:[[:space:]]*/ {
      value = $0
      sub(/^[[:space:]]+detector:[[:space:]]*/, "", value)
      sub(/[[:space:]]*#.*/, "", value)
      gsub(/^[[:space:]"]+|[[:space:]"]+$/, "", value)
      print toupper(value)
      exit
    }
  ' "${config}"
)"
if kfp_preflight_contains_word "${steps}" "DigiTrigger" \
   && [[ -n "${trigger_detector}" ]] \
   && ! kfp_preflight_contains_word "${systems}" "${trigger_detector}"; then
  systems="${systems} ${trigger_detector}"
fi

if [[ -z "${KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE:-}" ]]; then
  KFP_PREFLIGHT_EVIDENCE_FILE="${KFP_PREFLIGHT_LOG_DIR}/online-${kfp_mode}-${KFP_PREFLIGHT_DEVICE}.evidence.txt"
fi
if ! grep -a -q 'KFP online summary:' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'pid_assigned=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'secondary_pos=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'cpu_candidates=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'decay_channels=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'refit_policy=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" \
   || ! grep -a -q 'batch_limit=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so predates the current Stage 19.3 KFP classification contract; rebuild cbmreco from the current source tree"
fi
if [[ "${kfp_mode}" == "qualified-gpu" ]] \
   && ! grep -a -q 'qualification locked' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so predates the Stage 20.2 qualification lock; rebuild cbmreco from the current source tree"
fi
if [[ "${kfp_mode}" == "diagnostic" || "${kfp_mode}" == "qualified-gpu" ]] \
   && ! grep -a -q "${gpu_owner_build_id}" "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so does not contain the single-host GPU owner ${gpu_owner_build_id}; rebuild Algo and cbmreco"
fi
if [[ "${kfp_mode}" == "qualified-gpu" ]] \
   && { [[ ! -f "${KFP_PREFLIGHT_LIBRARY_DIR}/libKFParticle.so" ]] \
        || ! grep -a -q "${transaction_reset_build_id}" \
             "${KFP_PREFLIGHT_LIBRARY_DIR}/libKFParticle.so"; }; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libKFParticle.so does not contain transaction reset ${transaction_reset_build_id}; rebuild the selected qualification build, Algo, and cbmreco"
fi
deep_trace_build_id='20260806-real-data-fit-routing-v17'
if [[ "${deep_trace}" != "0" ]] \
   && ! grep -a -q "${deep_trace_build_id}" "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so does not contain deep diagnostic trace ${deep_trace_build_id}; rebuild CbmKfParticleGpuAdapter, Algo, and cbmreco"
fi

reco_parameters="$(sed -n 's/^[[:space:]]*recoParameters:[[:space:]]*\([^#[:space:]]*\).*/\1/p' "${config}" | head -1)"
reco_parameters="${reco_parameters%\"}"
reco_parameters="${reco_parameters#\"}"
if [[ -z "${reco_parameters}" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "parFiles.recoParameters is missing in ${config}"
fi
if [[ "${reco_parameters}" = /* ]]; then
  reco_parameters_path="${reco_parameters}"
else
  reco_parameters_path="$(cd "$(dirname "${config}")" && pwd)/${reco_parameters}"
fi
kfp_preflight_require_file "${reco_parameters_path}" "online reconstruction parameter archive"
kfp_preflight_check_linked_xpu "${cbmreco}"

read -r -a step_args <<<"${steps}"
read -r -a system_args <<<"${systems}"
command=(
  "${cbmreco}"
  --config "${config}"
  --input-locator "${input}"
  --device "${KFP_PREFLIGHT_DEVICE}"
  --num-ts "${num_ts}"
  --steps "${step_args[@]}"
  --systems "${system_args[@]}"
  --ca-timeslice-tracking-backend "${backend}"
  --event-reco-max-events "${max_events}"
  --kfp-mode "${kfp_mode}"
  --kfp-diagnostic-sample-period "${diagnostic_sample_period}"
  --kfp-diagnostic-batch-events "${diagnostic_batch_events}"
)
if [[ "${omp_threads}" != "0" ]]; then
  command+=(--omp "${omp_threads}")
fi
if [[ "${materialization_threads}" != "0" ]]; then
  command+=(--kfp-materialization-threads "${materialization_threads}")
fi
if [[ "${packing_threads}" != "0" ]]; then
  command+=(--kfp-packing-threads "${packing_threads}")
fi
if [[ "${adaptive_override_count}" == "3" ]]; then
  command+=(
    --kfp-adaptive-minimum-work "${adaptive_minimum_work}"
    --kfp-adaptive-target-work "${adaptive_target_work}"
    --kfp-adaptive-maximum-bytes "${adaptive_maximum_bytes}"
  )
fi
if [[ "${use_sts_pion_pid}" == "1" ]]; then
  command+=(--kfp-use-sts-pion-pid)
fi
if [[ "${refit_tracks}" == "1" ]]; then
  command+=(--kfp-refit-tracks)
fi
kfp_preflight_quote_command "${command[@]}"

KFP_PREFLIGHT_INPUT_IDENTITY="$(kfp_preflight_path_identity "${input}")"
KFP_PREFLIGHT_CONFIGURATION_IDENTITY="$(kfp_preflight_path_identity "${config}")"
KFP_PREFLIGHT_GEOMETRY_IDENTITY="embedded in ${reco_parameters_path}"
KFP_PREFLIGHT_PARAMETER_IDENTITY="$(kfp_preflight_path_identity "${reco_parameters_path}")"
pid_policy="detector"
if [[ "${use_sts_pion_pid}" == "1" ]]; then
  pid_policy="sts-pion"
fi
configured_refit_tracks="$(sed -nE 's/^[[:space:]]*doRefitTracks:[[:space:]]*(true|false).*/\1/p' "${config}" | head -1)"
refit_policy="disabled"
if [[ "${refit_tracks}" == "1" || "${configured_refit_tracks}" == "true" ]]; then
  refit_policy="enabled"
fi
KFP_PREFLIGHT_STEPS_IDENTITY="${steps}; systems=${systems}; CA=${backend}; KFP=${kfp_mode}; event-limit=${max_events} (0=all); diagnostic-batch<=${diagnostic_batch_events}; materialization-threads=${materialization_threads}; packing-threads=${packing_threads}; STS-exact-memory=${sts_exact_memory}; PID=${pid_policy}; refit=${refit_policy}"
KFP_PREFLIGHT_STEPS_IDENTITY+="; resource-retry=${retry_resource_batches}"
if [[ "${adaptive_override_count}" == "3" ]]; then
  KFP_PREFLIGHT_STEPS_IDENTITY+="; adaptive-work=${adaptive_minimum_work}:${adaptive_target_work}; adaptive-bytes<=${adaptive_maximum_bytes}"
fi

echo "KFParticle CBMRoot online preflight"
echo "  source dir : ${KFP_PREFLIGHT_SOURCE_DIR}"
echo "  build dir  : ${KFP_PREFLIGHT_BUILD_DIR}"
echo "  mode       : online (cbmreco)"
echo "  device     : ${KFP_PREFLIGHT_DEVICE}"
echo "  XPU libs   : ${KFP_PREFLIGHT_LIBRARY_DIR}"
echo "  HIP arch   : ${KFP_PREFLIGHT_HIP_ARCH:-unset}"
echo "  ROCm       : ${KFP_PREFLIGHT_ROCM_VERSION:-${KFP_PREFLIGHT_ROCM_ROOT}}"
echo "  FairRoot   : ${KFP_PREFLIGHT_FAIRROOT:-unset}"
echo "  ROOT       : ${KFP_PREFLIGHT_ROOT_VERSION:-unset}"
echo "  config     : ${config}"
echo "  parameters : ${reco_parameters_path}"
echo "  geometry   : reconstruction parameter archive"
echo "  steps      : ${steps}"
echo "  systems    : ${systems}"
echo "  trigger    : ${trigger_detector:-not configured}"
echo "  CA backend : ${backend}"
echo "  KFP mode   : ${kfp_mode}"
echo "  KFP sample : ${diagnostic_sample_period}"
echo "  adaptive policy: minimum=${adaptive_minimum_work} target=${adaptive_target_work} max-bytes=${adaptive_maximum_bytes}"
echo "  KFP batch  : ${diagnostic_batch_events} event(s)"
echo "  KFP resource retry: ${retry_resource_batches} (0 = isolate rejected groups into single events; 1 = retry smaller batches until they fit)"
echo "  KFP packing threads: ${packing_threads} (0 = YAML/OMP/default)"
echo "  KFP materialization threads: ${materialization_threads} (0 = inherit explicit OMP/YAML, else one)"
echo "  KFP materialization schedule: ${materialization_schedule}"
echo "  STS memory : $([[ "${sts_exact_memory}" == "1" ]] && echo 'exact per timeslice' || echo 'configuration default')"
echo "  deep trace : ${deep_trace}"
echo "  compact trace: ${deep_trace_compact}"
echo "  KFP PID    : ${pid_policy}"
echo "  KFP refit  : ${refit_policy}"
echo "  timeslices : ${num_ts}"
echo "  event limit: ${max_events} per timeslice (0 = all)"
echo "  OMP threads: ${omp_threads} (0 = runtime default)"
echo "  repetitions: ${repeat}"
echo "  performance: ${performance_monitoring}"
echo "  transaction diagnostics: ${transaction_diagnostics}"
echo "  construction timers: ${construction_timers} (XPU_PROFILE=${XPU_PROFILE:-0}; diagnostic timings are intrusive)"
echo "  qualification expectation: ${qualification_expectation}"
echo "  qualification build: ${KFP_PREFLIGHT_QUALIFICATION_BUILD}"
echo "  command    : ${KFP_PREFLIGHT_COMMAND_TEXT}"

kfp_preflight_runtime_smoke

if [[ "${execute}" != "1" ]]; then
  kfp_preflight_write_evidence "prerequisites validated; reconstruction not requested"
  echo "PASS ${KFP_PREFLIGHT_LABEL} - online prerequisites and CBMRoot-owned XPU runtime are valid"
  echo "  evidence   : ${KFP_PREFLIGHT_EVIDENCE_FILE}"
  exit 0
fi

reference_summary=""
reference_adaptive_shadow=""
performance_logs=()
for ((run_index = 1; run_index <= repeat; ++run_index)); do
  log_file="${KFP_PREFLIGHT_LOG_DIR}/online-${kfp_mode}-${KFP_PREFLIGHT_DEVICE}.run-${run_index}.log"
  run_started_ms="$(kfp_preflight_now_milliseconds)"
  if ! LD_LIBRARY_PATH="${KFP_PREFLIGHT_LD_LIBRARY_PATH}" \
    XPU_DEVICE="${KFP_PREFLIGHT_DEVICE}" \
    KFPARTICLE_GPU_DEEP_TRACE="${deep_trace}" \
    KFPARTICLE_GPU_DEEP_TRACE_COMPACT="${deep_trace_compact}" \
    KFPARTICLE_CPU_FINDER_TRACE_PAIRS="${cpu_finder_trace_pairs}" \
    KFPARTICLE_GPU_PERFORMANCE_MONITORING="${performance_monitoring}" \
    KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE="${materialization_schedule}" \
    KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS="${transaction_diagnostics}" \
    KFPARTICLE_GPU_RETRY_RESOURCE_BATCHES="${retry_resource_batches}" \
    CBM_STS_HITFINDER_EXACT_MEMORY="${sts_exact_memory}" \
    "${command[@]}" 2>&1 | tee "${log_file}"; then
    kfp_preflight_write_evidence "reconstruction run ${run_index}/${repeat} failed; see ${log_file}"
    exit 1
  fi
  run_finished_ms="$(kfp_preflight_now_milliseconds)"
  echo "PERFORMANCE_RUN mode=${kfp_mode} run=${run_index} wall_ms=$((run_finished_ms - run_started_ms))" >>"${log_file}"
  performance_logs+=("${log_file}")
  if [[ "${construction_timers}" == "1" ]] \
     && ! grep -q 'KFP_STEP_TRACE BEGIN contract=1 section=two-track-construction details=1' "${log_file}"; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "missing detailed XPU construction timers; check XPU_PROFILE and rebuild KFParticle/CBMRoot"
  fi
  if [[ "${performance_monitoring}" == "1" ]]; then
    kfp_preflight_print_performance_report "${log_file}"
  fi

  adaptive_shadow="$(grep 'KFParticle GPU adaptive online shadow:' "${log_file}" | tail -1 || true)"
  if [[ -n "${adaptive_shadow}" ]]; then
    adaptive_shadow="KFParticle GPU adaptive online shadow: ${adaptive_shadow#*KFParticle GPU adaptive online shadow: }"
    echo "  ${adaptive_shadow}"
  fi
  if [[ "${adaptive_override_count}" == "3" && "${kfp_mode}" != "cpu-only" ]]; then
    adaptive_observations="$(sed -nE 's/.* observations=([0-9]+).*/\1/p' <<<"${adaptive_shadow}")"
    adaptive_gpu_ready="$(sed -nE 's@.* decisions collecting/gpu/disabled/no-work/below/resource=[0-9]+/([0-9]+)/.*@\1@p' <<<"${adaptive_shadow}")"
    if [[ -z "${adaptive_observations}" || "${adaptive_observations}" == "0" \
          || -z "${adaptive_gpu_ready}" || "${adaptive_gpu_ready}" == "0" ]]; then
      cat "${log_file}" >&2
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "calibrated adaptive policy produced no observable GPU-ready online decision"
    fi
    if [[ -n "${reference_adaptive_shadow}" \
          && "${adaptive_shadow}" != "${reference_adaptive_shadow}" ]]; then
      printf 'adaptive reference: %s\nadaptive current:   %s\n' \
        "${reference_adaptive_shadow}" "${adaptive_shadow}" >&2
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "adaptive online shadow changed between repeated runs"
    fi
    reference_adaptive_shadow="${adaptive_shadow}"
  fi

  summary="$(grep 'KFP online summary:' "${log_file}" | tail -1)"
  # FairLogger prefixes each record with a wall-clock timestamp. Compare and
  # publish only the stable summary payload across repeated reconstructions.
  summary="KFP online summary: ${summary#*KFP online summary: }"
  contract="$(sed -nE 's/.* contract=([0-9]+).*/\1/p' <<<"${summary}")"
  handoff_phases="$(sed -nE 's/.* handoff_phases=([0-9]+).*/\1/p' <<<"${summary}")"
  events_available="$(sed -nE 's/.* events_available=([0-9]+).*/\1/p' <<<"${summary}")"
  events="$(sed -nE 's/.* events=([0-9]+).*/\1/p' <<<"${summary}")"
  processed="$(sed -nE 's/.* processed=([0-9]+).*/\1/p' <<<"${summary}")"
  conversion_failed="$(sed -nE 's/.* conversion_failed=([0-9]+).*/\1/p' <<<"${summary}")"
  pid_assigned="$(sed -nE 's/.* pid_assigned=([0-9]+).*/\1/p' <<<"${summary}")"
  selected_tracks="$(sed -nE 's/.* selected_tracks=([0-9]+).*/\1/p' <<<"${summary}")"
  cpu_particles="$(sed -nE 's/.* cpu_particles=([0-9]+).*/\1/p' <<<"${summary}")"
  cpu_candidates="$(sed -nE 's/.* cpu_candidates=([0-9]+).*/\1/p' <<<"${summary}")"
  decay_channels="$(sed -nE 's/.* decay_channels=([0-9]+).*/\1/p' <<<"${summary}")"
  reported_pid_policy="$(sed -nE 's/.* pid_policy=([^ ]+).*/\1/p' <<<"${summary}")"
  reported_refit_policy="$(sed -nE 's/.* refit_policy=([^ ]+).*/\1/p' <<<"${summary}")"
  reported_mode="$(sed -nE 's/.* mode=([^ ]+).*/\1/p' <<<"${summary}")"
  diagnostic_contract="$(sed -nE 's/.* diagnostic_contract=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_seen="$(sed -nE 's/.* diagnostic_seen=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_ok="$(sed -nE 's/.* diagnostic_ok=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_compared="$(sed -nE 's/.* diagnostic_compared=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_matched="$(sed -nE 's/.* diagnostic_matched=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_blocked="$(sed -nE 's/.* diagnostic_blocked=([0-9]+).*/\1/p' <<<"${summary}")"
  diagnostic_blockers="$(sed -nE 's/.* diagnostic_blockers=([0-9]+).*/\1/p' <<<"${summary}")"

  if [[ "${contract}" != "1" || "${diagnostic_contract}" != "2" \
        || "${reported_mode}" != "${kfp_mode}" || -z "${processed}" \
        || -z "${events_available}" || -z "${events}" \
        || -z "${conversion_failed}" || "${processed}" == "0" \
        || "${conversion_failed}" != "0" || "${reported_pid_policy}" != "${pid_policy}" \
        || "${reported_refit_policy}" != "${refit_policy}" \
        || "${events}" == "0" \
        || ("${max_events}" != "0" && "${events}" -gt "${max_events}") \
        || ("${max_events}" == "0" && "${events}" != "${events_available}") \
        || -z "${pid_assigned}" || "${pid_assigned}" == "0" \
        || -z "${selected_tracks}" || "${selected_tracks}" == "0" \
        || -z "${cpu_particles}" || "${cpu_particles}" == "0" \
        || -z "${cpu_candidates}" || "${cpu_candidates}" == "0" \
        || -z "${decay_channels}" || "${decay_channels}" == "0" ]]; then
    cat "${log_file}" >&2
    kfp_preflight_write_evidence \
      "run ${run_index}/${repeat} has an empty or incomplete online KFP handoff; see ${log_file}"
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "cbmreco run ${run_index}/${repeat} did not satisfy non-empty online KFP handoff contract 1"
  fi
  if [[ "${adaptive_override_count}" == "3" && "${handoff_phases}" != "2" ]]; then
    cat "${log_file}" >&2
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "adaptive online run did not use the two-phase reconstructed-event handoff"
  fi

  if [[ "${kfp_mode}" == "cpu-only" ]]; then
    if [[ "${diagnostic_seen}" != "0" || "${diagnostic_ok}" != "0" \
          || "${diagnostic_compared}" != "0" || "${diagnostic_matched}" != "0" \
          || "${diagnostic_blocked}" != "0" || "${diagnostic_blockers}" != "0" ]]; then
      cat "${log_file}" >&2
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "cpu-only run ${run_index}/${repeat} unexpectedly executed GPU diagnostics"
    fi
  elif [[ "${kfp_mode}" == "diagnostic" ]]; then
    if [[ -z "${diagnostic_seen}" || "${diagnostic_seen}" == "0" \
          || -z "${diagnostic_ok}" || -z "${diagnostic_compared}" \
          || -z "${diagnostic_matched}" || -z "${diagnostic_blocked}" \
          || -z "${diagnostic_blockers}" \
          || "${diagnostic_ok}" != "${diagnostic_seen}" \
          || "${diagnostic_compared}" != "${diagnostic_seen}" \
          || ("${diagnostic_sample_period}" == "1" && "${diagnostic_seen}" != "${processed}") ]]; then
      cat "${log_file}" >&2
      KFP_PREFLIGHT_RESULT_IDENTITY="${summary}"
      kfp_preflight_write_evidence \
        "diagnostic run ${run_index}/${repeat} has incomplete CPU/GPU comparisons; see ${log_file}"
      kfp_preflight_append_performance_evidence "${performance_logs[@]}"
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "diagnostic run ${run_index}/${repeat} did not complete diagnostic contract 2"
    elif [[ "${diagnostic_matched}" != "${diagnostic_compared}" \
            || "${diagnostic_blocked}" != "0" || "${diagnostic_blockers}" != "0" ]]; then
      if [[ "${allow_diagnostic_rejection}" == "1" ]]; then
        echo "  REJECTED run ${run_index}/${repeat}: diagnostic physics contract blocked; evidence retained for qualification"
      else
        cat "${log_file}" >&2
        KFP_PREFLIGHT_RESULT_IDENTITY="${summary}"
        kfp_preflight_write_evidence \
          "diagnostic run ${run_index}/${repeat} has blocked CPU/GPU comparisons; see ${log_file}"
        kfp_preflight_append_performance_evidence "${performance_logs[@]}"
        kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
          "diagnostic run ${run_index}/${repeat} did not satisfy diagnostic physics contract 2"
      fi
    fi
  elif [[ "${kfp_mode}" == "qualified-gpu" ]]; then
    routing_line="$(grep 'KFParticle GPU routing monitoring:' "${log_file}" | tail -1 || true)"
    gpu_accepted="$(sed -nE 's/.*GPU accepted ([0-9]+).*/\1/p' <<<"${routing_line}")"
    qualification_locked="$(sed -nE 's/.*qualification locked ([0-9]+).*/\1/p' <<<"${routing_line}")"
    unsupported="$(sed -nE 's/.*fallback unsupported[^0-9]*([0-9]+).*/\1/p' <<<"${routing_line}")"
    gpu_accepted="${gpu_accepted:-0}"
    qualification_locked="${qualification_locked:-0}"
    unsupported="${unsupported:-0}"
    if [[ "${qualification_expectation}" == "locked" ]]; then
      if ((gpu_accepted != 0 || qualification_locked + unsupported == 0)); then
        cat "${log_file}" >&2
        kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
          "qualified-gpu run did not report an atomic qualification/capability fallback"
      fi
    elif ((gpu_accepted == 0 || qualification_locked != 0)); then
      cat "${log_file}" >&2
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "qualification trial did not publish a supported GPU event"
    fi
  fi

  if [[ -n "${reference_summary}" && "${summary}" != "${reference_summary}" ]]; then
    printf 'reference: %s\ncurrent:   %s\n' "${reference_summary}" "${summary}" >&2
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "online KFP summary changed between repeated runs"
  fi
  reference_summary="${summary}"
  echo "  PASS run ${run_index}/${repeat}: ${log_file}"
done

KFP_PREFLIGHT_RESULT_IDENTITY="${reference_summary}"
kfp_preflight_write_evidence \
  "${repeat} online GPU CA plus KfpSelectorChain ${kfp_mode} run(s) completed"
kfp_preflight_append_performance_evidence "${performance_logs[@]}"
echo "PASS ${KFP_PREFLIGHT_LABEL} - cbmreco completed ${repeat} run(s) of ${num_ts} timeslice(s) with KfpSelectorChain (${kfp_mode})"
echo "  evidence   : ${KFP_PREFLIGHT_EVIDENCE_FILE}"
