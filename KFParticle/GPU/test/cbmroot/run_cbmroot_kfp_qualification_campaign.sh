#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${KFPARTICLE_CBMROOT_QUALIFICATION_MODE:-}"
workload="${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD:-representative}"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
warmup_repetitions="${KFPARTICLE_CBMROOT_QUALIFICATION_WARMUP_REPETITIONS:-1}"
measured_repetitions="${KFPARTICLE_CBMROOT_QUALIFICATION_MEASURED_REPETITIONS:-3}"

case "${mode}" in online|offline) ;; *)
  echo "FAIL cbmroot-kfp-qualification-campaign - set KFPARTICLE_CBMROOT_QUALIFICATION_MODE to online or offline" >&2
  exit 2
esac
[[ -d "${build_dir}" ]] || {
  echo "FAIL cbmroot-kfp-qualification-campaign - set KFPARTICLE_CBMROOT_BUILD_DIR" >&2
  exit 2
}
[[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "FAIL cbmroot-kfp-qualification-campaign - workload name must be filesystem-safe" >&2
  exit 2
}
[[ "${warmup_repetitions}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FAIL cbmroot-kfp-qualification-campaign - warm-up repetitions must be positive" >&2
  exit 2
}
[[ "${measured_repetitions}" =~ ^[1-9][0-9]*$ ]] && ((measured_repetitions >= 3)) || {
  echo "FAIL cbmroot-kfp-qualification-campaign - at least three measured repetitions are required" >&2
  exit 2
}

log_dir="${KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR:-${build_dir}/kfparticle-step20-3/${mode}-${workload}}"
mkdir -p "${log_dir}"
if [[ "${mode}" == "online" ]]; then
  preflight="${script_dir}/run_cbmroot_kfp_online_preflight.sh"
  cpu_evidence="${log_dir}/online-cpu-only-${device}.evidence.txt"
  diagnostic_evidence="${log_dir}/online-diagnostic-${device}.evidence.txt"
  qualified_evidence="${log_dir}/online-qualified-gpu-${device}.evidence.txt"
else
  preflight="${script_dir}/run_cbmroot_kfp_offline_preflight.sh"
  input_kind="${KFPARTICLE_CBMROOT_OFFLINE_INPUT_KIND:-auto}"
  if [[ "${input_kind}" == "auto" ]]; then
    case "${KFPARTICLE_CBMROOT_OFFLINE_INPUT_PREFIX:-}" in
      *.tb) input_kind="time-based" ;;
      *.eb) input_kind="event-based" ;;
      *) echo "FAIL cbmroot-kfp-qualification-campaign - set offline input kind" >&2; exit 2 ;;
    esac
  fi
  cpu_evidence="${log_dir}/offline-${input_kind}-cpu-only-${device}.evidence.txt"
  diagnostic_evidence="${log_dir}/offline-${input_kind}-diagnostic-${device}.evidence.txt"
  qualified_evidence="${log_dir}/offline-${input_kind}-qualified-gpu-${device}.evidence.txt"
fi

run_mode()
{
  local execution_mode="$1"
  local expectation="$2"
  local evidence="$3"
  local phase="$4"
  local repetitions="$5"
  local output_prefix="${KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX:-}"
  if [[ "${mode}" == "offline" && -n "${output_prefix}" ]]; then
    output_prefix="${output_prefix}-${workload}-${execution_mode}-${phase}"
  fi
  local phase_log_dir="${log_dir}/${phase}/${execution_mode}"
  local -a environment=(
    KFPARTICLE_CBMROOT_KFP_MODE="${execution_mode}"
    KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION="${expectation}"
    KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial
    KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1
    KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE="${KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE:-0}"
    KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_HASH=0
    KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR="${phase_log_dir}"
    KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE="${evidence}"
    KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX="${output_prefix}"
    KFPARTICLE_GPU_PERFORMANCE_MONITORING=1
  )
  if [[ "${mode}" == "online" ]]; then
    environment+=(KFPARTICLE_CBMROOT_ONLINE_REPEAT="${repetitions}")
  else
    environment+=(KFPARTICLE_CBMROOT_OFFLINE_REPEAT="${repetitions}")
  fi
  env "${environment[@]}" bash "${preflight}"
}

echo "KFParticle CBMRoot Step 20.3 qualification campaign"
echo "  mode       : ${mode}"
echo "  workload   : ${workload}"
echo "  build dir  : ${build_dir}"
echo "  device     : ${device}"
echo "  log dir    : ${log_dir}"
echo "  warm-up    : ${warmup_repetitions} per execution mode"
echo "  measured   : ${measured_repetitions} per execution mode"

run_mode cpu-only locked "${log_dir}/warmup-cpu-only.evidence.txt" warmup "${warmup_repetitions}"
run_mode cpu-only locked "${cpu_evidence}" measured "${measured_repetitions}"
run_mode diagnostic locked "${log_dir}/warmup-diagnostic.evidence.txt" warmup "${warmup_repetitions}"
run_mode diagnostic locked "${diagnostic_evidence}" measured "${measured_repetitions}"
run_mode qualified-gpu published "${log_dir}/warmup-qualified-gpu.evidence.txt" warmup "${warmup_repetitions}"
run_mode qualified-gpu published "${qualified_evidence}" measured "${measured_repetitions}"

KFPARTICLE_CBMROOT_QUALIFICATION_MODE="${mode}" \
KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD="${workload}" \
KFPARTICLE_CBMROOT_QUALIFICATION_POLICY="${KFPARTICLE_CBMROOT_QUALIFICATION_POLICY:-${script_dir}/step20_qualification_policy.conf}" \
KFPARTICLE_CBMROOT_QUALIFICATION_CPU_EVIDENCE="${cpu_evidence}" \
KFPARTICLE_CBMROOT_QUALIFICATION_DIAGNOSTIC_EVIDENCE="${diagnostic_evidence}" \
KFPARTICLE_CBMROOT_QUALIFICATION_GPU_EVIDENCE="${qualified_evidence}" \
KFPARTICLE_CBMROOT_QUALIFICATION_OUTPUT="${log_dir}/qualification-${mode}-${workload}.evidence.txt" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_gate.sh"

echo "PASS cbmroot-kfp-qualification-campaign - ${mode}/${workload} completed all three execution modes and the frozen gate"
