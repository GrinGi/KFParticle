#!/usr/bin/env bash
set -euo pipefail

# This wrapper deliberately executes caller-owned reconstruction commands. It
# does not modify a production macro or guess an experiment-specific input.
baseline_command="${KFPARTICLE_CBMROOT_BASELINE_COMMAND:-}"
diagnostic_command="${KFPARTICLE_CBMROOT_DIAGNOSTIC_COMMAND:-}"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-${VMCWORKDIR:+${VMCWORKDIR}/../build}}"
log_dir="${KFPARTICLE_CBMROOT_CAMPAIGN_LOG_DIR:-${build_dir:-.}/kfparticle-cbmroot-gpu-campaign}"
baseline_output="${KFPARTICLE_CBMROOT_BASELINE_OUTPUT:-}"
diagnostic_output="${KFPARTICLE_CBMROOT_DIAGNOSTIC_OUTPUT:-}"
evidence_file="${KFPARTICLE_CBMROOT_EVIDENCE_FILE:-}"

if [[ -z "${baseline_command}" || -z "${diagnostic_command}" ]]; then
  cat >&2 <<'EOF'
FAIL cbmroot-kfp-gpu-campaign - set both KFPARTICLE_CBMROOT_BASELINE_COMMAND and
KFPARTICLE_CBMROOT_DIAGNOSTIC_COMMAND. Each command must run the same stable
input; only the latter enables kfp.selector.gpuDiagnostics.
EOF
  exit 2
fi

mkdir -p "${log_dir}"
baseline_log="${log_dir}/baseline.log"
diagnostic_log="${log_dir}/diagnostic.log"
evidence_file="${evidence_file:-${log_dir}/validation-evidence.txt}"

run_command() {
  local label="$1"
  local command="$2"
  local log_file="$3"
  echo "  ${label} log : ${log_file}"
  if ! bash -c "${command}" >"${log_file}" 2>&1; then
    cat "${log_file}" >&2
    echo "FAIL cbmroot-kfp-gpu-campaign - ${label} command failed" >&2
    exit 1
  fi
}

echo "KFParticle CBMRoot GPU diagnostic campaign"
echo "  logs         : ${log_dir}"
run_command "baseline" "${baseline_command}" "${baseline_log}"
run_command "diagnostic" "${diagnostic_command}" "${diagnostic_log}"

if grep -q 'KFParticle GPU diagnostics:' "${baseline_log}"; then
  echo "FAIL cbmroot-kfp-gpu-campaign - baseline unexpectedly enabled diagnostics" >&2
  exit 1
fi
if ! grep -q 'KFParticle GPU diagnostics:' "${diagnostic_log}"; then
  cat "${diagnostic_log}" >&2
  echo "FAIL cbmroot-kfp-gpu-campaign - diagnostic run emitted no summary" >&2
  exit 1
fi

if [[ -n "${baseline_output}" || -n "${diagnostic_output}" ]]; then
  if [[ -z "${baseline_output}" || -z "${diagnostic_output}" ]]; then
    echo "FAIL cbmroot-kfp-gpu-campaign - provide both CPU output paths or neither" >&2
    exit 2
  fi
  if ! cmp -s "${baseline_output}" "${diagnostic_output}"; then
    echo "FAIL cbmroot-kfp-gpu-campaign - CPU outputs differ between baseline and diagnostic runs" >&2
    exit 1
  fi
fi

write_evidence() {
  local rocm_root="${XPU_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
  local output_comparison="not requested"
  if [[ -n "${baseline_output}" ]]; then
    output_comparison="byte-identical"
  fi

  {
    echo "KFParticle CBMRoot GPU diagnostic validation evidence"
    echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host: $(hostname)"
    echo "kernel: $(uname -srmo)"
    echo "device: ${KFPARTICLE_CBMROOT_DEVICE:-${XPU_DEVICE:-unset}}"
    echo "ROCm root: ${rocm_root}"
    if [[ -r "${rocm_root}/.info/version" ]]; then
      echo "ROCm version: $(tr '\n' ' ' < "${rocm_root}/.info/version")"
    fi
    echo
    echo "baseline command:"
    printf '%s\n' "${baseline_command}"
    echo
    echo "diagnostic command:"
    printf '%s\n' "${diagnostic_command}"
    echo
    echo "baseline log: ${baseline_log}"
    echo "diagnostic log: ${diagnostic_log}"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${baseline_log}" "${diagnostic_log}"
    fi
    echo "CPU output comparison: ${output_comparison}"
    if [[ -n "${baseline_output}" ]]; then
      echo "baseline output: ${baseline_output}"
      echo "diagnostic output: ${diagnostic_output}"
      if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${baseline_output}" "${diagnostic_output}"
      fi
    fi
    echo
    echo "diagnostic aggregate:"
    grep -F 'KFParticle GPU diagnostics:' "${diagnostic_log}"
    echo
    echo "diagnostic channels:"
    grep -F 'KFParticle GPU diagnostics channel ' "${diagnostic_log}" || true
  } >"${evidence_file}"
}

write_evidence

echo "PASS cbmroot-kfp-gpu-campaign - baseline remains diagnostic-free, diagnostic run reports GPU telemetry"
if [[ -n "${baseline_output}" ]]; then
  echo "PASS cbmroot-kfp-gpu-campaign-output - CPU output files are byte-identical"
fi
echo "PASS cbmroot-kfp-gpu-campaign-evidence - ${evidence_file}"
