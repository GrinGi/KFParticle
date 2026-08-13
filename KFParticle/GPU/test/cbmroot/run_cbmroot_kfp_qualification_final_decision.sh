#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
online_matrix="${KFPARTICLE_CBMROOT_QUALIFICATION_ONLINE_MATRIX:-}"
offline_matrix="${KFPARTICLE_CBMROOT_QUALIFICATION_OFFLINE_MATRIX:-}"
policy="${KFPARTICLE_CBMROOT_QUALIFICATION_POLICY:-${script_dir}/step20_qualification_policy.conf}"
output="${KFPARTICLE_CBMROOT_QUALIFICATION_FINAL_OUTPUT:-}"
expected_workloads="${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOADS:-low:10 typical:50 high:100}"

fail()
{
  echo "FAIL cbmroot-kfp-qualification-final-decision - $*" >&2
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

field()
{
  sed -n "s/^$2: //p" "$1" | tail -1
}

validate_matrix()
{
  local matrix="$1"
  local expected_mode="$2"
  local matrix_decision matrix_device decision_path expected_hash actual_hash
  local decision_mode decision_workload decision_policy_hash
  local -a paths hashes
  local seen_low=0 seen_typical=0 seen_high=0
  local rejected_count=0 workload_decision

  [[ -s "${matrix}" ]] || fail "missing ${expected_mode} matrix: ${matrix:-unset}"
  [[ "$(field "${matrix}" mode)" == "${expected_mode}" ]] \
    || fail "${matrix} is not an ${expected_mode} matrix"
  matrix_decision="$(field "${matrix}" decision)"
  case "${matrix_decision}" in qualified|rejected) ;; *) fail "invalid matrix decision in ${matrix}" ;; esac
  [[ "$(field "${matrix}" workloads)" == "${expected_workloads}" ]] \
    || fail "${expected_mode} workloads differ from frozen bounds ${expected_workloads}"
  matrix_device="$(field "${matrix}" device)"
  [[ -n "${matrix_device}" ]] || fail "${expected_mode} matrix is missing device"

  while IFS= read -r decision_path; do
    paths+=("${decision_path}")
  done < <(sed -n 's/^workload decision: //p' "${matrix}")
  while IFS= read -r expected_hash; do
    hashes+=("${expected_hash}")
  done < <(sed -n 's/^workload decision sha256: //p' "${matrix}")
  ((${#paths[@]} == 3 && ${#hashes[@]} == 3)) \
    || fail "${expected_mode} matrix must contain exactly three decisions and hashes"

  for index in 0 1 2; do
    decision_path="${paths[index]}"
    expected_hash="${hashes[index]}"
    [[ -s "${decision_path}" ]] || fail "missing workload decision: ${decision_path}"
    actual_hash="$(file_hash "${decision_path}")"
    [[ "${actual_hash}" == "${expected_hash}" ]] \
      || fail "workload decision hash differs: ${decision_path}"
    decision_mode="$(field "${decision_path}" mode)"
    decision_workload="$(field "${decision_path}" workload)"
    [[ "${decision_mode}" == "${expected_mode}" ]] \
      || fail "workload decision mode differs: ${decision_path}"
    case "${decision_workload}" in
      low) ((seen_low += 1)) ;;
      typical) ((seen_typical += 1)) ;;
      high) ((seen_high += 1)) ;;
      *) fail "unknown workload ${decision_workload:-unset} in ${decision_path}" ;;
    esac
    workload_decision="$(field "${decision_path}" decision)"
    case "${workload_decision}" in
      qualified|rejected) ;;
      *) fail "invalid workload decision in ${decision_path}" ;;
    esac
    [[ "${workload_decision}" == rejected ]] && ((rejected_count += 1))
    decision_policy_hash="$(field "${decision_path}" "policy sha256")"
    [[ "${decision_policy_hash}" == "${policy_hash}" ]] \
      || fail "workload decision uses a different frozen policy: ${decision_path}"
  done
  [[ "${seen_low}" == 1 ]] || fail "${expected_mode} matrix does not contain exactly one low decision"
  [[ "${seen_typical}" == 1 ]] || fail "${expected_mode} matrix does not contain exactly one typical decision"
  [[ "${seen_high}" == 1 ]] || fail "${expected_mode} matrix does not contain exactly one high decision"
  if [[ "${matrix_decision}" == qualified && "${rejected_count}" != 0 ]]; then
    fail "${expected_mode} matrix is qualified but contains a rejected workload"
  fi
  if [[ "${matrix_decision}" == rejected && "${rejected_count}" == 0 ]]; then
    fail "${expected_mode} matrix is rejected but contains no rejected workload"
  fi

  printf '%s\t%s\n' "${matrix_decision}" "${matrix_device}"
}

[[ -s "${policy}" ]] || fail "missing frozen policy: ${policy}"
policy_hash="$(file_hash "${policy}")"
online_result="$(validate_matrix "${online_matrix}" online)"
offline_result="$(validate_matrix "${offline_matrix}" offline)"
IFS=$'\t' read -r online_decision online_device <<<"${online_result}"
IFS=$'\t' read -r offline_decision offline_device <<<"${offline_result}"
[[ "${online_device}" == "${offline_device}" ]] \
  || fail "online device ${online_device} differs from offline device ${offline_device}"

online_route=diagnostic-only
offline_route=diagnostic-only
[[ "${online_decision}" == qualified ]] && online_route=qualified-gpu
[[ "${offline_decision}" == qualified ]] && offline_route=qualified-gpu
production_decision=diagnostic-only
ordinary_build_action="keep qualification locked"
if [[ "${online_route}" == qualified-gpu && "${offline_route}" == qualified-gpu ]]; then
  production_decision=qualified-gpu
  ordinary_build_action="eligible for a separately reviewed exact-manifest unlock"
fi

output="${output:-$(dirname "${online_matrix}")/qualification-final-decision.evidence.txt}"
mkdir -p "$(dirname "${output}")"
{
  echo "KFParticle CBMRoot Step 20.3C final qualification decision"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "qualification evidence: complete"
  echo "production decision: ${production_decision}"
  echo "ordinary build action: ${ordinary_build_action}"
  echo "policy: ${policy}"
  echo "policy sha256: ${policy_hash}"
  echo "workloads: ${expected_workloads}"
  echo "device: ${online_device}"
  echo "online matrix: ${online_matrix}"
  echo "online matrix sha256: $(file_hash "${online_matrix}")"
  echo "online matrix decision: ${online_decision}"
  echo "online route: ${online_route}"
  echo "offline matrix: ${offline_matrix}"
  echo "offline matrix sha256: $(file_hash "${offline_matrix}")"
  echo "offline matrix decision: ${offline_decision}"
  echo "offline route: ${offline_route}"
} >"${output}"

echo "PASS cbmroot-kfp-qualification-final-decision - qualification evidence is complete; production decision=${production_decision}"
echo "  evidence   : ${output}"
