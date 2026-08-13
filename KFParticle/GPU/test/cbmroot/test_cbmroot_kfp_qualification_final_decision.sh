#!/usr/bin/env bash
set -euo pipefail

# The verifier fixtures must never read or overwrite retained campaign evidence.
while IFS= read -r variable; do
  unset "${variable}"
done < <(compgen -A variable KFPARTICLE_ || true)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/kfparticle-qualification-final.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT
policy="${script_dir}/step20_qualification_policy.conf"

file_hash()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

write_matrix()
{
  local mode="$1"
  local matrix_decision="$2"
  local rejected_workload="${3:-}"
  local matrix="${tmp_dir}/qualification-${mode}-matrix.evidence.txt"
  local policy_hash
  policy_hash="$(file_hash "${policy}")"
  {
    echo "decision: ${matrix_decision}"
    echo "mode: ${mode}"
    echo "device: hip1"
    echo "workloads: low:10 typical:50 high:100"
    for workload in low typical high; do
      decision=qualified
      [[ "${workload}" == "${rejected_workload}" ]] && decision=rejected
      decision_path="${tmp_dir}/qualification-${mode}-${workload}.evidence.txt"
      {
        echo "decision: ${decision}"
        echo "mode: ${mode}"
        echo "workload: ${workload}"
        echo "policy sha256: ${policy_hash}"
      } >"${decision_path}"
      echo "workload decision: ${decision_path}"
      echo "workload decision sha256: $(file_hash "${decision_path}")"
    done
  } >"${matrix}"
}

write_matrix online qualified
write_matrix offline qualified
qualified_output="${tmp_dir}/qualified.evidence.txt"
KFPARTICLE_CBMROOT_QUALIFICATION_ONLINE_MATRIX="${tmp_dir}/qualification-online-matrix.evidence.txt" \
KFPARTICLE_CBMROOT_QUALIFICATION_OFFLINE_MATRIX="${tmp_dir}/qualification-offline-matrix.evidence.txt" \
KFPARTICLE_CBMROOT_QUALIFICATION_FINAL_OUTPUT="${qualified_output}" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_final_decision.sh" >"${tmp_dir}/qualified.log"
grep -q '^production decision: qualified-gpu$' "${qualified_output}"
grep -q '^qualification evidence: complete$' "${qualified_output}"

write_matrix online rejected low
diagnostic_output="${tmp_dir}/diagnostic-only.evidence.txt"
KFPARTICLE_CBMROOT_QUALIFICATION_ONLINE_MATRIX="${tmp_dir}/qualification-online-matrix.evidence.txt" \
KFPARTICLE_CBMROOT_QUALIFICATION_OFFLINE_MATRIX="${tmp_dir}/qualification-offline-matrix.evidence.txt" \
KFPARTICLE_CBMROOT_QUALIFICATION_FINAL_OUTPUT="${diagnostic_output}" \
  bash "${script_dir}/run_cbmroot_kfp_qualification_final_decision.sh" >"${tmp_dir}/diagnostic.log"
grep -q '^production decision: diagnostic-only$' "${diagnostic_output}"
grep -q '^ordinary build action: keep qualification locked$' "${diagnostic_output}"

printf '\n' >>"${tmp_dir}/qualification-online-low.evidence.txt"
if KFPARTICLE_CBMROOT_QUALIFICATION_ONLINE_MATRIX="${tmp_dir}/qualification-online-matrix.evidence.txt" \
   KFPARTICLE_CBMROOT_QUALIFICATION_OFFLINE_MATRIX="${tmp_dir}/qualification-offline-matrix.evidence.txt" \
     bash "${script_dir}/run_cbmroot_kfp_qualification_final_decision.sh" >"${tmp_dir}/tampered.log" 2>&1; then
  echo "FAIL cbmroot-kfp-qualification-final-decision-contract - changed evidence hash was accepted" >&2
  exit 1
fi
grep -q 'workload decision hash differs' "${tmp_dir}/tampered.log"

echo "PASS cbmroot-kfp-qualification-final-decision-contract - complete qualified and diagnostic-only decisions are integrity checked"
