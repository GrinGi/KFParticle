#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/cbmroot_kfp_preflight_common.sh"
kfp_preflight_initialize "online"

cbmreco="${KFPARTICLE_CBMROOT_ONLINE_EXECUTABLE:-${KFP_PREFLIGHT_BUILD_DIR}/bin/cbmreco}"
config="${KFPARTICLE_CBMROOT_ONLINE_CONFIG:-${KFP_PREFLIGHT_SOURCE_DIR}/macro/algo/MainConfig.yaml}"
input="${KFPARTICLE_CBMROOT_ONLINE_INPUT_LOCATOR:-}"
steps="${KFPARTICLE_CBMROOT_ONLINE_STEPS:-Unpack DigiTrigger LocalReco Tracking EventReco KfpSelector}"
systems="${KFPARTICLE_CBMROOT_ONLINE_SYSTEMS:-STS}"
backend="${KFPARTICLE_CBMROOT_ONLINE_TRACKING_BACKEND:-gpu-multi-window}"
num_ts="${KFPARTICLE_CBMROOT_ONLINE_NUM_TS:-1}"
max_events="${KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS:-50}"
use_sts_pion_pid="${KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID:-1}"
refit_tracks="${KFPARTICLE_CBMROOT_ONLINE_REFIT_TRACKS:-1}"
kfp_mode="${KFPARTICLE_CBMROOT_KFP_MODE:-cpu-only}"
diagnostic_sample_period="${KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD:-1}"
deep_trace="${KFPARTICLE_GPU_DEEP_TRACE:-0}"
deep_trace_compact="${KFPARTICLE_GPU_DEEP_TRACE_COMPACT:-0}"
cpu_finder_trace_pairs="${KFPARTICLE_CPU_FINDER_TRACE_PAIRS:-}"
cpu_finder_trace_build_id='20260806-cpu-finder-routing-v16'
repeat="${KFPARTICLE_CBMROOT_ONLINE_REPEAT:-1}"
execute="${KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE:-0}"
performance_monitoring="${KFPARTICLE_GPU_PERFORMANCE_MONITORING:-0}"
qualification_expectation="${KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION:-locked}"

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

for required_step in Tracking EventReco KfpSelector; do
  if ! kfp_preflight_contains_word "${steps}" "${required_step}"; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "online steps must include ${required_step}: ${steps}"
  fi
done
if ! kfp_preflight_contains_word "${systems}" "STS"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "online systems must include STS: ${systems}"
fi
if [[ "${backend}" != "gpu-multi-window" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "cbmreco GPU timeslice tracking requires ca-timeslice backend gpu-multi-window"
fi
if [[ ! "${num_ts}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_ONLINE_NUM_TS must be positive"
fi
if [[ ! "${max_events}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS must be positive"
fi
if [[ "${use_sts_pion_pid}" != "0" && "${use_sts_pion_pid}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_ONLINE_USE_STS_PION_PID must be 0 or 1"
fi
if [[ "${refit_tracks}" != "0" && "${refit_tracks}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_ONLINE_REFIT_TRACKS must be 0 or 1"
fi
case "${kfp_mode}" in
  cpu|cpu-only) kfp_mode="cpu-only" ;;
  diagnostic) ;;
  qualified|qualified-gpu|validated|validated-gpu) kfp_mode="qualified-gpu" ;;
  *) kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
       "online KFP mode must be cpu-only, diagnostic, or qualified-gpu" ;;
esac
if [[ ! "${diagnostic_sample_period}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD must be positive"
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
if [[ "${performance_monitoring}" != "0" && "${performance_monitoring}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_GPU_PERFORMANCE_MONITORING must be 0 or 1"
fi
case "${qualification_expectation}" in locked|published) ;; *)
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION must be locked or published"
esac
if [[ "${qualification_expectation}" == "published" \
      && ("${kfp_mode}" != "qualified-gpu" || "${performance_monitoring}" != "1") ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "published qualification expectation requires qualified-gpu mode and performance monitoring"
fi
kfp_preflight_check_qualification_build \
  "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so" "${qualification_expectation}"
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
   || ! grep -a -q 'refit_policy=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so predates the current Stage 19.3 KFP classification contract; rebuild cbmreco from the current source tree"
fi
if [[ "${kfp_mode}" == "qualified-gpu" ]] \
   && ! grep -a -q 'qualification locked' "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so predates the Stage 20.2 qualification lock; rebuild cbmreco from the current source tree"
fi
deep_trace_build_id='20260806-real-data-fit-routing-v17'
if [[ "${deep_trace}" != "0" ]] \
   && ! grep -a -q "${deep_trace_build_id}" "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgo.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libAlgo.so does not contain deep diagnostic trace ${deep_trace_build_id}; rebuild CbmKfParticleGpuDiagnostic, Algo, and cbmreco"
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
)
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
KFP_PREFLIGHT_STEPS_IDENTITY="${steps}; systems=${systems}; CA=${backend}; KFP=${kfp_mode}; events<=${max_events}; PID=${pid_policy}; refit=${refit_policy}"

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
echo "  deep trace : ${deep_trace}"
echo "  compact trace: ${deep_trace_compact}"
echo "  KFP PID    : ${pid_policy}"
echo "  KFP refit  : ${refit_policy}"
echo "  timeslices : ${num_ts}"
echo "  event limit: ${max_events} per timeslice"
echo "  repetitions: ${repeat}"
echo "  performance: ${performance_monitoring}"
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
    "${command[@]}" >"${log_file}" 2>&1; then
    cat "${log_file}" >&2
    kfp_preflight_write_evidence "reconstruction run ${run_index}/${repeat} failed; see ${log_file}"
    exit 1
  fi
  run_finished_ms="$(kfp_preflight_now_milliseconds)"
  echo "PERFORMANCE_RUN mode=${kfp_mode} run=${run_index} wall_ms=$((run_finished_ms - run_started_ms))" >>"${log_file}"
  performance_logs+=("${log_file}")
  if [[ "${performance_monitoring}" == "1" ]]; then
    kfp_preflight_print_performance_report "${log_file}"
  fi

  summary="$(grep 'KFP online summary:' "${log_file}" | tail -1)"
  # FairLogger prefixes each record with a wall-clock timestamp. Compare and
  # publish only the stable summary payload across repeated reconstructions.
  summary="KFP online summary: ${summary#*KFP online summary: }"
  contract="$(sed -nE 's/.* contract=([0-9]+).*/\1/p' <<<"${summary}")"
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
        || "${events}" == "0" || "${events}" -gt "${max_events}" \
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

  if [[ "${kfp_mode}" == "cpu-only" ]]; then
    if [[ "${diagnostic_seen}" != "0" || "${diagnostic_ok}" != "0" \
          || "${diagnostic_compared}" != "0" || "${diagnostic_matched}" != "0" \
          || "${diagnostic_blocked}" != "0" || "${diagnostic_blockers}" != "0" ]]; then
      cat "${log_file}" >&2
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "cpu-only run ${run_index}/${repeat} unexpectedly executed GPU diagnostics"
    fi
  elif [[ "${kfp_mode}" == "diagnostic" ]] \
       && [[ -z "${diagnostic_seen}" || "${diagnostic_seen}" == "0" \
          || "${diagnostic_ok}" != "${diagnostic_seen}" \
          || "${diagnostic_compared}" != "${diagnostic_seen}" \
          || "${diagnostic_matched}" != "${diagnostic_compared}" \
          || "${diagnostic_blocked}" != "0" || "${diagnostic_blockers}" != "0" \
          || ("${diagnostic_sample_period}" == "1" && "${diagnostic_seen}" != "${processed}") ]]; then
    cat "${log_file}" >&2
    kfp_preflight_write_evidence \
      "diagnostic run ${run_index}/${repeat} has failed or blocked CPU/GPU comparisons; see ${log_file}"
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "diagnostic run ${run_index}/${repeat} did not satisfy diagnostic contract 2"
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
