#!/usr/bin/env bash
set -euo pipefail

build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
relation="${KFPARTICLE_CBMROOT_CROSS_MODE_INPUT_RELATION:-independent}"

if [[ -z "${build_dir}" || ! -d "${build_dir}" ]]; then
  echo "FAIL cbmroot-kfp-cross-mode - set KFPARTICLE_CBMROOT_BUILD_DIR" >&2
  exit 2
fi
case "${relation}" in independent|equivalent) ;; *)
  echo "FAIL cbmroot-kfp-cross-mode - input relation must be independent or equivalent" >&2
  exit 2
esac

log_dir="${KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR:-${build_dir}/kfparticle-cbmroot-preflight}"
online_cpu="${KFPARTICLE_CBMROOT_ONLINE_CPU_EVIDENCE:-${log_dir}/online-cpu-only-${device}.evidence.txt}"
online_diagnostic="${KFPARTICLE_CBMROOT_ONLINE_DIAGNOSTIC_EVIDENCE:-${log_dir}/online-diagnostic-${device}.evidence.txt}"
offline_cpu="${KFPARTICLE_CBMROOT_OFFLINE_CPU_EVIDENCE:-${log_dir}/offline-event-based-cpu-only-${device}.evidence.txt}"
offline_diagnostic="${KFPARTICLE_CBMROOT_OFFLINE_DIAGNOSTIC_EVIDENCE:-${log_dir}/offline-event-based-diagnostic-${device}.evidence.txt}"
output="${KFPARTICLE_CBMROOT_CROSS_MODE_EVIDENCE:-${log_dir}/cross-mode-${relation}-${device}.evidence.txt}"

for evidence in "${online_cpu}" "${online_diagnostic}" "${offline_cpu}" "${offline_diagnostic}"; do
  if [[ ! -s "${evidence}" ]]; then
    echo "FAIL cbmroot-kfp-cross-mode - missing evidence: ${evidence}" >&2
    exit 2
  fi
done

field()
{
  local evidence="$1"
  local key="$2"
  sed -n "s/^${key}: //p" "${evidence}" | tail -1
}

result_handoff()
{
  local evidence="$1"
  local result
  result="$(field "${evidence}" result)"
  if [[ -z "${result}" || "${result}" == "unset" ]]; then
    local basename fallback_log
    basename="$(basename "${evidence}")"
    case "${basename}" in
      offline-event-based-cpu-only-*)
        fallback_log="${log_dir}/offline-event-based-cpu-only-${device}.run-1.log"
        ;;
      offline-event-based-diagnostic-*)
        fallback_log="${log_dir}/offline-event-based-diagnostic-${device}.run-1.log"
        ;;
      online-cpu-only-*)
        fallback_log="${log_dir}/online-cpu-only-${device}.run-1.log"
        ;;
      online-diagnostic-*)
        fallback_log="${log_dir}/online-diagnostic-${device}.run-1.log"
        ;;
    esac
    if [[ -n "${fallback_log:-}" && -f "${fallback_log}" ]]; then
      result="$(grep -E 'KFP (selector|online) summary:' "${fallback_log}" | tail -1)"
    fi
  fi
  sed -nE \
    's/.* events=([0-9]+) processed=([0-9]+) no_pv=([0-9]+).* tracks=([0-9]+) sts_hit_links=([0-9]+) vertices=([0-9]+) pid_slots=([0-9]+).*/events=\1 processed=\2 no_pv=\3 tracks=\4 sts_hit_links=\5 vertices=\6 pid_slots=\7/p' \
    <<<"${result}"
}

reference="${online_cpu}"
for evidence in "${online_diagnostic}" "${offline_cpu}" "${offline_diagnostic}"; do
  for key in source build device "XPU HIP architecture" ROCm FairRoot ROOT; do
    if [[ "$(field "${evidence}" "${key}")" != "$(field "${reference}" "${key}")" ]]; then
      echo "FAIL cbmroot-kfp-cross-mode - ${key} differs between ${reference} and ${evidence}" >&2
      exit 2
    fi
  done
done
reference_config="$(field "${reference}" configuration)"
reference_config="${reference_config%%;*}"
for evidence in "${online_diagnostic}" "${offline_cpu}" "${offline_diagnostic}"; do
  compared_config="$(field "${evidence}" configuration)"
  compared_config="${compared_config%%;*}"
  if [[ "${compared_config}" != "${reference_config}" ]]; then
    echo "FAIL cbmroot-kfp-cross-mode - MainConfig.yaml differs between ${reference} and ${evidence}" >&2
    exit 2
  fi
done

online_cpu_handoff="$(result_handoff "${online_cpu}")"
online_diagnostic_handoff="$(result_handoff "${online_diagnostic}")"
offline_cpu_handoff="$(result_handoff "${offline_cpu}")"
offline_diagnostic_handoff="$(result_handoff "${offline_diagnostic}")"

if [[ -z "${online_cpu_handoff}" || -z "${offline_cpu_handoff}" \
      || "${online_cpu_handoff}" != "${online_diagnostic_handoff}" \
      || "${offline_cpu_handoff}" != "${offline_diagnostic_handoff}" ]]; then
  echo "FAIL cbmroot-kfp-cross-mode - CPU and diagnostic handoff differ within an official mode" >&2
  exit 2
fi
if [[ "${relation}" == "equivalent" && "${online_cpu_handoff}" != "${offline_cpu_handoff}" ]]; then
  echo "online handoff : ${online_cpu_handoff}" >&2
  echo "offline handoff: ${offline_cpu_handoff}" >&2
  echo "FAIL cbmroot-kfp-cross-mode - equivalent inputs produced different framework handoffs" >&2
  exit 2
fi

{
  echo "KFParticle CBMRoot cross-mode qualification evidence"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "device: ${device}"
  echo "input relation: ${relation}"
  echo "online handoff: ${online_cpu_handoff}"
  echo "offline handoff: ${offline_cpu_handoff}"
  echo "online CPU evidence: ${online_cpu}"
  echo "online diagnostic evidence: ${online_diagnostic}"
  echo "offline CPU evidence: ${offline_cpu}"
  echo "offline diagnostic evidence: ${offline_diagnostic}"
} >"${output}"

if [[ "${relation}" == "equivalent" ]]; then
  echo "PASS cbmroot-kfp-cross-mode - online and offline equivalent inputs satisfy the same handoff and diagnostic contracts"
else
  echo "PASS cbmroot-kfp-cross-mode - both official modes satisfy the shared environment and diagnostic contracts; data-count comparison is not claimed for independent inputs"
fi
echo "  evidence   : ${output}"
