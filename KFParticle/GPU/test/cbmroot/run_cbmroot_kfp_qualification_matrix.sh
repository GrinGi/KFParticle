#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${KFPARTICLE_CBMROOT_QUALIFICATION_MODE:-}"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
workloads="${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS:-low:10 typical:50 high:100}"
root_log_dir="${KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR:-${build_dir}/kfparticle-step20-3}"
campaign_launcher="${KFPARTICLE_CBMROOT_QUALIFICATION_CAMPAIGN_LAUNCHER:-${script_dir}/run_cbmroot_kfp_qualification_campaign.sh}"
policy="${KFPARTICLE_CBMROOT_QUALIFICATION_POLICY:-${script_dir}/step20_qualification_policy.conf}"

fail()
{
  echo "FAIL cbmroot-kfp-qualification-matrix - $*" >&2
  exit 2
}

file_hash()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

case "${mode}" in online|offline) ;; *) fail "set KFPARTICLE_CBMROOT_QUALIFICATION_MODE to online or offline" ;; esac
[[ -d "${build_dir}" ]] || fail "set KFPARTICLE_CBMROOT_BUILD_DIR to the trial build"
[[ -f "${campaign_launcher}" ]] || fail "missing qualification campaign launcher: ${campaign_launcher}"
[[ -s "${policy}" ]] || fail "missing frozen qualification policy: ${policy}"

seen_low=0
seen_typical=0
seen_high=0
decisions=()
all_qualified=1
for specification in ${workloads}; do
  [[ "${specification}" =~ ^(low|typical|high):([1-9][0-9]*)$ ]] \
    || fail "workloads must use low:N typical:N high:N"
  workload="${BASH_REMATCH[1]}"
  amount="${BASH_REMATCH[2]}"
  seen_variable="seen_${workload}"
  [[ "${!seen_variable}" == "0" ]] || fail "duplicate workload ${workload}"
  printf -v "${seen_variable}" '%s' 1

  workload_log_dir="${root_log_dir}/${mode}-${workload}"
  environment=(
    KFPARTICLE_CBMROOT_QUALIFICATION_MODE="${mode}"
    KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD="${workload}"
    KFPARTICLE_CBMROOT_QUALIFICATION_LOG_DIR="${workload_log_dir}"
  )
  if [[ "${mode}" == "online" ]]; then
    environment+=(KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS="${amount}")
  else
    environment+=(KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES="${amount}")
  fi

  echo "KFParticle qualification matrix: mode=${mode} workload=${workload} bound=${amount}"
  decision="${workload_log_dir}/qualification-${mode}-${workload}.evidence.txt"
  rm -f "${decision}"
  campaign_status=0
  env "${environment[@]}" bash "${campaign_launcher}" || campaign_status=$?
  if [[ ! -s "${decision}" ]]; then
    mkdir -p "$(dirname "${decision}")"
    {
      echo "KFParticle CBMRoot Step 20.3 qualification decision"
      echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "decision: rejected"
      echo "mode: ${mode}"
      echo "workload: ${workload}"
      echo "reason: campaign stopped before the qualification gate (status ${campaign_status})"
      echo "policy: ${policy}"
      echo "policy sha256: $(file_hash "${policy}")"
    } >"${decision}"
  fi
  if ((campaign_status != 0)) || ! grep -q '^decision: qualified$' "${decision}"; then
    all_qualified=0
    echo "  REJECTED ${mode}/${workload}: $(sed -n 's/^reason: //p' "${decision}" | tail -1)"
  fi
  decisions+=("${decision}")
done

for required in low typical high; do
  seen_variable="seen_${required}"
  [[ "${!seen_variable}" == "1" ]] || fail "missing required ${required} workload"
done

output="${KFPARTICLE_CBMROOT_QUALIFICATION_MATRIX_EVIDENCE:-${root_log_dir}/qualification-${mode}-matrix.evidence.txt}"
mkdir -p "$(dirname "${output}")"
{
  echo "KFParticle CBMRoot Step 20.3B qualification matrix"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if ((all_qualified)); then
    echo "decision: qualified"
  else
    echo "decision: rejected"
  fi
  echo "mode: ${mode}"
  echo "device: ${device}"
  echo "workloads: ${workloads}"
  for decision in "${decisions[@]}"; do
    echo "workload decision: ${decision}"
    echo "workload decision sha256: $(file_hash "${decision}")"
  done
} >"${output}"

if ((!all_qualified)); then
  echo "FAIL cbmroot-kfp-qualification-matrix - ${mode} completed all workloads but at least one frozen gate rejected qualification" >&2
  echo "  evidence   : ${output}" >&2
  exit 1
fi
echo "PASS cbmroot-kfp-qualification-matrix - ${mode} low/typical/high campaigns passed independently"
echo "  evidence   : ${output}"
