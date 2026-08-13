#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/cbmroot_kfp_preflight_common.sh"
kfp_preflight_initialize "offline"

root_bin="${KFPARTICLE_CBMROOT_OFFLINE_ROOT:-${ROOTSYS:+${ROOTSYS}/bin/root}}"
root_bin="${root_bin:-$(command -v root || true)}"
macro="${KFPARTICLE_CBMROOT_OFFLINE_MACRO:-${script_dir}/run_reco_tracks.C}"
input_prefix="${KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX:-}"
output_prefix="${KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX:-${input_prefix}}"
parameter_prefix="${KFPARTICLE_CBMROOT_OFFLINE_PARAMETER_PREFIX:-${input_prefix}}"
setup="${KFPARTICLE_CBMROOT_OFFLINE_SETUP:-sis100_hadron}"
first_entry="${KFPARTICLE_CBMROOT_OFFLINE_FIRST_ENTRY:-0}"
enable_mc_qa="${KFPARTICLE_CBMROOT_OFFLINE_ENABLE_MC_QA:-0}"
backend="${KFPARTICLE_CBMROOT_OFFLINE_TRACKING_BACKEND:-gpu-multi}"
input_kind="${KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND:-auto}"
kfp_config="${KFPARTICLE_CBMROOT_OFFLINE_CONFIG:-${KFPARTICLE_CBMROOT_ONLINE_CONFIG:-}}"
kfp_mode="${KFPARTICLE_CBMROOT_KFP_MODE:-cpu-only}"
diagnostic_sample_period="${KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD:-1}"
repeat="${KFPARTICLE_CBMROOT_OFFLINE_REPEAT:-1}"
execute="${KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE:-0}"
performance_monitoring="${KFPARTICLE_GPU_PERFORMANCE_MONITORING:-0}"
qualification_expectation="${KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION:-locked}"

if [[ -z "${root_bin}" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "ROOT executable was not found after loading the CBMRoot environment"
fi
kfp_preflight_require_executable "${root_bin}" "ROOT executable"
kfp_preflight_require_file "${macro}" "offline FairRunAna macro"
kfp_preflight_require_file "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgoOffline.so" "offline reconstruction library"
kfp_preflight_require_file "${KFP_PREFLIGHT_LIBRARY_DIR}/libCbmRecoBase.so" "offline CBMRoot base library"
kfp_preflight_require_file "${KFP_PREFLIGHT_LIBRARY_DIR}/libCbmRecoTasks.so" "offline KFP FairTask library"
if [[ -z "${input_prefix}" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "set KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX without the .raw.root suffix"
fi
if [[ -z "${kfp_config}" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "set KFPARTICLE_CBMROOT_OFFLINE_CONFIG to MainConfig.yaml"
fi
kfp_preflight_require_file "${kfp_config}" "offline KFParticle configuration"
if ! grep -q '^kfp:' "${kfp_config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "offline MainConfig.yaml is missing the top-level kfp node: ${kfp_config}"
fi
if ! grep -q '^eventSelector:' "${kfp_config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "offline MainConfig.yaml is missing the top-level eventSelector node: ${kfp_config}"
fi
if ! grep -q '^[[:space:]]*selectors:' "${kfp_config}" \
   || ! grep -q '^[[:space:]]*selectionType:' "${kfp_config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "eventSelector must define selectors and selectionType: ${kfp_config}"
fi
if ! grep -q '^[[:space:]]*doRefitTracks:' "${kfp_config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "kfp.selector must define doRefitTracks: ${kfp_config}"
fi
case "${kfp_mode}" in
  cpu|cpu-only) kfp_mode="cpu-only" ;;
  diagnostic) ;;
  qualified|qualified-gpu|validated|validated-gpu) kfp_mode="qualified-gpu" ;;
  *) kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
       "offline KFP mode must be cpu-only, diagnostic, or qualified-gpu" ;;
esac
if [[ "${kfp_mode}" == "diagnostic" ]] \
   && ! grep -a -q 'diagnostic_contract=' "${KFP_PREFLIGHT_LIBRARY_DIR}/libCbmRecoTasks.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libCbmRecoTasks.so predates diagnostic contract 2; rebuild CbmRecoTasks from the current source tree"
fi
if [[ "${kfp_mode}" == "qualified-gpu" ]] \
   && ! grep -a -q 'qualification locked' "${KFP_PREFLIGHT_LIBRARY_DIR}/libCbmRecoTasks.so"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "libCbmRecoTasks.so predates the Stage 20.2 qualification lock; rebuild CbmRecoTasks from the current source tree"
fi
if [[ ! "${diagnostic_sample_period}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_DIAGNOSTIC_SAMPLE_PERIOD must be positive"
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
  "${KFP_PREFLIGHT_LIBRARY_DIR}/libAlgoOffline.so" "${qualification_expectation}"
if [[ ! "${repeat}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_OFFLINE_REPEAT must be positive"
fi

if [[ "${input_kind}" == "auto" ]]; then
  case "${input_prefix}" in
    *.tb) input_kind="time-based" ;;
    *.eb) input_kind="event-based" ;;
    *)
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "cannot infer offline input kind; set KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND=time-based or event-based"
      ;;
  esac
fi
case "${input_kind}" in time-based|event-based) ;; *)
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND must be time-based or event-based"
esac
ca_config_node="$([[ "${input_kind}" == "event-based" ]] && echo caEvent || echo caTimeslice)"
if ! grep -q "^${ca_config_node}:" "${kfp_config}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "offline MainConfig.yaml is missing the top-level ${ca_config_node} node required by ${input_kind} CA parameters: ${kfp_config}"
fi
if [[ -z "${KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE:-}" ]]; then
  KFP_PREFLIGHT_EVIDENCE_FILE="${KFP_PREFLIGHT_LOG_DIR}/offline-${input_kind}-${kfp_mode}-${KFP_PREFLIGHT_DEVICE}.evidence.txt"
fi

default_entries=1
if [[ "${input_kind}" == "event-based" ]]; then
  default_entries=100
fi
num_entries="${KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES:-${default_entries}}"
if [[ ! "${num_entries}" =~ ^[1-9][0-9]*$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES must be positive"
fi
if [[ ! "${first_entry}" =~ ^[0-9]+$ ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_OFFLINE_FIRST_ENTRY must be non-negative"
fi
case "${enable_mc_qa}" in 0|1) ;; *)
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "KFPARTICLE_CBMROOT_OFFLINE_ENABLE_MC_QA must be 0 or 1"
esac
case "${backend}" in gpu-multi|gpu-multi-window) ;; *)
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "the Step 19 data contain empty events/modules and require gpu-multi or gpu-multi-window, not ${backend}"
esac
if [[ "${input_kind}" == "event-based" && "${num_entries}" -gt 100 \
      && "${KFPARTICLE_CBMROOT_OFFLINE_ALLOW_LARGE_EVENT_RUN:-0}" != "1" ]]; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "event-based preflight is limited to 100 entries; set KFPARTICLE_CBMROOT_OFFLINE_ALLOW_LARGE_EVENT_RUN=1 to override"
fi

raw_file="${input_prefix}.raw.root"
parameter_file="${parameter_prefix}.par.root"
output_file="${output_prefix}.reco.root"
kfp_preflight_require_file "${raw_file}" "offline raw input"
kfp_preflight_require_file "${parameter_file}" "offline FairRoot parameter file"
if [[ "${enable_mc_qa}" == "1" ]]; then
  kfp_preflight_require_file "${input_prefix}.tra.root" "offline MC friend input"
fi
if ! grep -q 'FairRunAna' "${macro}" || ! grep -q 'CbmL1' "${macro}" \
   || ! grep -q 'ParametersHandler::Instance' "${macro}" \
   || ! grep -q 'run->AddTask(&caParameters)' "${macro}" \
   || ! grep -q 'TaskRecoEventConverter' "${macro}" \
   || ! grep -q 'TaskKfpSelector' "${macro}"; then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "offline macro does not expose the expected FairRunAna, shared CA parameters, GPU CA, and KfpSelectorChain adapter: ${macro}"
fi
ca_parameters_line="$(grep -n 'run->AddTask(&caParameters)' "${macro}" | head -1 | cut -d: -f1 || true)"
sts_line="$(grep -n 'new CbmStsFindTracks' "${macro}" | head -1 | cut -d: -f1 || true)"
event_line="$(grep -n 'new CbmBuildEventsFromTracksReal' "${macro}" | head -1 | cut -d: -f1 || true)"
pv_line="$(grep -n 'new CbmFindPrimaryVertex' "${macro}" | head -1 | cut -d: -f1 || true)"
converter_line="$(grep -n 'new cbm::TaskRecoEventConverter' "${macro}" | head -1 | cut -d: -f1 || true)"
selector_line="$(grep -n 'new cbm::TaskKfpSelector' "${macro}" | head -1 | cut -d: -f1 || true)"
if [[ -z "${ca_parameters_line}" || -z "${sts_line}" || -z "${event_line}" || -z "${pv_line}" \
      || -z "${converter_line}" || -z "${selector_line}" ]] \
   || ! ((ca_parameters_line < converter_line
          && sts_line < event_line && event_line < pv_line
          && pv_line < converter_line && converter_line < selector_line)); then
  kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
    "offline macro task order must initialize shared CA parameters before RecoEvent and preserve STS tracking -> event policy -> PV -> RecoEvent -> KfpSelectorChain"
fi
kfp_preflight_check_linked_xpu "${root_bin}"

mc_qa_arg="$([[ "${enable_mc_qa}" == "1" ]] && echo kTRUE || echo kFALSE)"
event_based_arg="$([[ "${input_kind}" == "event-based" ]] && echo kTRUE || echo kFALSE)"
macro_call="${macro}(\"${input_prefix}\",${num_entries},${first_entry},\"${output_prefix}\",\"${setup}\",\"${parameter_prefix}\",${mc_qa_arg},\"${kfp_config}\",\"${kfp_mode}\",${event_based_arg},${diagnostic_sample_period})"
command=("${root_bin}" -l -b -q "${macro_call}")
kfp_preflight_quote_command "${command[@]}"

KFP_PREFLIGHT_INPUT_IDENTITY="$(kfp_preflight_path_identity "${raw_file}")"
KFP_PREFLIGHT_INPUT_KIND="${input_kind}"
KFP_PREFLIGHT_CONFIGURATION_IDENTITY="$(kfp_preflight_path_identity "${kfp_config}"); ca=${ca_config_node}; macro=$(kfp_preflight_path_identity "${macro}"); mode=${kfp_mode}"
KFP_PREFLIGHT_GEOMETRY_IDENTITY="${setup}"
KFP_PREFLIGHT_PARAMETER_IDENTITY="$(kfp_preflight_path_identity "${parameter_file}")"
KFP_PREFLIGHT_STEPS_IDENTITY="FairRunAna STS reco; CA=${backend}; events=${input_kind}; PV; RecoEvent; KfpSelectorChain=${kfp_mode}"

echo "KFParticle CBMRoot offline preflight"
echo "  source dir : ${KFP_PREFLIGHT_SOURCE_DIR}"
echo "  build dir  : ${KFP_PREFLIGHT_BUILD_DIR}"
echo "  mode       : offline (FairRunAna)"
echo "  device     : ${KFP_PREFLIGHT_DEVICE}"
echo "  XPU libs   : ${KFP_PREFLIGHT_LIBRARY_DIR}"
echo "  HIP arch   : ${KFP_PREFLIGHT_HIP_ARCH:-unset}"
echo "  ROCm       : ${KFP_PREFLIGHT_ROCM_VERSION:-${KFP_PREFLIGHT_ROCM_ROOT}}"
echo "  FairRoot   : ${KFP_PREFLIGHT_FAIRROOT:-unset}"
echo "  ROOT       : ${KFP_PREFLIGHT_ROOT_VERSION:-unset}"
kfp_preflight_file_identity "${raw_file}" "input"
echo "  input kind : ${input_kind}"
echo "  macro      : ${macro}"
echo "  KFP config : ${kfp_config}"
echo "  CA config  : ${ca_config_node}"
echo "  KFP mode   : ${kfp_mode}"
echo "  KFP sample : ${diagnostic_sample_period}"
echo "  geometry   : ${setup}"
echo "  parameters : ${parameter_file}"
echo "  output     : ${output_file}"
echo "  CA backend : ${backend}"
echo "  entries    : ${first_entry} .. $((first_entry + num_entries - 1))"
echo "  repetitions: ${repeat}"
echo "  performance: ${performance_monitoring}"
echo "  qualification expectation: ${qualification_expectation}"
echo "  qualification build: ${KFP_PREFLIGHT_QUALIFICATION_BUILD}"
echo "  KFP path   : event/PV conversion -> shared KfpSelectorChain"
echo "  command    : ${KFP_PREFLIGHT_COMMAND_TEXT}"

kfp_preflight_runtime_smoke

if [[ "${execute}" != "1" ]]; then
  kfp_preflight_write_evidence "offline KfpSelectorChain prerequisites validated; reconstruction not requested"
  echo "PASS ${KFP_PREFLIGHT_LABEL} - FairRunAna/GPU-tracking/KfpSelectorChain prerequisites and XPU runtime are valid"
  echo "  evidence   : ${KFP_PREFLIGHT_EVIDENCE_FILE}"
  exit 0
fi

reference_summary=""
performance_logs=()
for ((run_index = 1; run_index <= repeat; ++run_index)); do
  log_file="${KFP_PREFLIGHT_LOG_DIR}/offline-${input_kind}-${kfp_mode}-${KFP_PREFLIGHT_DEVICE}.run-${run_index}.log"
  run_status=0
  run_started_ms="$(kfp_preflight_now_milliseconds)"
  if LD_LIBRARY_PATH="${KFP_PREFLIGHT_LD_LIBRARY_PATH}" \
    XPU_DEVICE="${KFP_PREFLIGHT_DEVICE}" \
    CBM_CA_XPU_DEVICE="${KFP_PREFLIGHT_DEVICE}" \
    CBM_CA_TRACKING_BACKEND="${backend}" \
    KFPARTICLE_GPU_PERFORMANCE_MONITORING="${performance_monitoring}" \
    "${command[@]}" >"${log_file}" 2>&1; then
    :
  else
    run_status=$?
    if grep -q 'KFP selector summary:' "${log_file}" \
       && grep -q -- '-I- run_reco_tracks: finished' "${log_file}" \
       && grep -q -- '-I- run_reco_tracks: output' "${log_file}" \
       && [[ -s "${output_file}" ]]; then
      echo "  WARN run ${run_index}/${repeat}: ignored FairRoot shutdown status ${run_status} after complete output"
    else
      cat "${log_file}" >&2
      kfp_preflight_write_evidence "reconstruction run ${run_index}/${repeat} failed; see ${log_file}"
      exit 1
    fi
  fi
  run_finished_ms="$(kfp_preflight_now_milliseconds)"
  launcher_ms=$((run_finished_ms - run_started_ms))
  reconstruction_seconds="$(
    sed -nE 's/^-I- run_reco_tracks: real time ([0-9.]+) s, CPU time .*/\1/p' "${log_file}" | tail -1
  )"
  if [[ "${reconstruction_seconds}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    reconstruction_ms="$(LC_NUMERIC=C awk -v seconds="${reconstruction_seconds}" \
      'BEGIN {printf "%.0f", 1000 * seconds}')"
  else
    reconstruction_ms="${launcher_ms}"
  fi
  echo "PERFORMANCE_RUN mode=${kfp_mode} run=${run_index} wall_ms=${reconstruction_ms} launcher_ms=${launcher_ms}" \
    >>"${log_file}"
  performance_logs+=("${log_file}")
  if [[ "${performance_monitoring}" == "1" ]]; then
    kfp_preflight_print_performance_report "${log_file}"
  fi

  if ! grep -q 'KFP selector handoff:' "${log_file}" \
     || ! grep -Eq 'KFP selector summary:.*processed=[1-9][0-9]*' "${log_file}"; then
    cat "${log_file}" >&2
    kfp_preflight_write_evidence \
      "run ${run_index}/${repeat} finished without a non-empty KfpSelectorChain handoff; see ${log_file}"
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "FairRunAna run ${run_index}/${repeat} did not report a processed KFP event"
  fi
  summary="$(grep 'KFP selector summary:' "${log_file}" | tail -1)"
  # ROOT/FairLogger decorations are execution metadata, not reconstruction
  # output. Keep the repeatability contract limited to the stable payload.
  summary="KFP selector summary: ${summary#*KFP selector summary: }"
  if [[ "${kfp_mode}" == "diagnostic" ]]; then
    diagnostic_contract="$(sed -nE 's/.* diagnostic_contract=([0-9]+).*/\1/p' <<<"${summary}")"
    processed_events="$(sed -nE 's/.* processed=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_seen="$(sed -nE 's/.* diagnostic_seen=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_ok="$(sed -nE 's/.* diagnostic_ok=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_compared="$(sed -nE 's/.* diagnostic_compared=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_matched="$(sed -nE 's/.* diagnostic_matched=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_blocked="$(sed -nE 's/.* diagnostic_blocked=([0-9]+).*/\1/p' <<<"${summary}")"
    diagnostic_blockers="$(sed -nE 's/.* diagnostic_blockers=([0-9]+).*/\1/p' <<<"${summary}")"

    if [[ "${diagnostic_contract}" != "2" || -z "${processed_events}" \
          || -z "${diagnostic_seen}" || -z "${diagnostic_ok}" \
          || -z "${diagnostic_compared}" \
          || -z "${diagnostic_matched}" || -z "${diagnostic_blocked}" || -z "${diagnostic_blockers}" ]] \
       || ((diagnostic_seen == 0 || diagnostic_ok != diagnostic_seen
            || diagnostic_compared != diagnostic_seen || diagnostic_matched != diagnostic_compared
            || diagnostic_blocked != 0 || diagnostic_blockers != 0)) \
       || [[ "${diagnostic_sample_period}" == "1" && "${diagnostic_seen}" != "${processed_events}" ]]; then
      cat "${log_file}" >&2
      kfp_preflight_write_evidence \
        "diagnostic run ${run_index}/${repeat} did not report complete matching CPU/GPU results; see ${log_file}"
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "diagnostic run ${run_index}/${repeat} has missing, failed, or blocked CPU/GPU comparisons"
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
      "offline KFP summary changed between repeated runs"
  fi
  reference_summary="${summary}"
  echo "  PASS run ${run_index}/${repeat}: ${log_file}"
done

KFP_PREFLIGHT_RESULT_IDENTITY="${reference_summary}"
kfp_preflight_write_evidence \
  "${repeat} GPU CA plus offline KfpSelectorChain ${kfp_mode} run(s) completed"
kfp_preflight_append_performance_evidence "${performance_logs[@]}"
echo "PASS ${KFP_PREFLIGHT_LABEL} - FairRunAna completed ${repeat} run(s) of ${num_entries} entry/entries through GPU CA and KfpSelectorChain (${kfp_mode})"
echo "  evidence   : ${KFP_PREFLIGHT_EVIDENCE_FILE}"
