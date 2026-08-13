#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${KFPARTICLE_CBMROOT_QUALIFICATION_MODE:-}"
workload="${KFPARTICLE_CBMROOT_QUALIFICATION_WORKLOAD:-representative}"
policy="${KFPARTICLE_CBMROOT_QUALIFICATION_POLICY:-${script_dir}/step20_qualification_policy.conf}"
cpu_evidence="${KFPARTICLE_CBMROOT_QUALIFICATION_CPU_EVIDENCE:-}"
diagnostic_evidence="${KFPARTICLE_CBMROOT_QUALIFICATION_DIAGNOSTIC_EVIDENCE:-}"
qualified_evidence="${KFPARTICLE_CBMROOT_QUALIFICATION_GPU_EVIDENCE:-}"
output="${KFPARTICLE_CBMROOT_QUALIFICATION_OUTPUT:-}"

file_hash()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fail()
{
  local reason="$*"
  if [[ -n "${output:-}" ]]; then
    mkdir -p "$(dirname "${output}")"
    {
      echo "KFParticle CBMRoot Step 20.3 qualification decision"
      echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "decision: rejected"
      echo "mode: ${mode:-unset}"
      echo "workload: ${workload}"
      echo "reason: ${reason}"
      echo "policy: ${policy}"
      if [[ -s "${policy}" ]]; then
        echo "policy sha256: $(file_hash "${policy}")"
      fi
      [[ -n "${cpu_evidence:-}" ]] && echo "CPU evidence: ${cpu_evidence}"
      [[ -n "${diagnostic_evidence:-}" ]] && echo "diagnostic evidence: ${diagnostic_evidence}"
      [[ -n "${qualified_evidence:-}" ]] && echo "qualified GPU evidence: ${qualified_evidence}"
    } >"${output}"
  fi
  echo "FAIL cbmroot-kfp-qualification-${mode:-unset} - ${reason}" >&2
  exit 2
}

case "${mode}" in online|offline) ;; *) fail "set KFPARTICLE_CBMROOT_QUALIFICATION_MODE to online or offline" ;; esac
[[ -s "${policy}" ]] || fail "missing frozen policy: ${policy}"
for evidence in "${cpu_evidence}" "${diagnostic_evidence}" "${qualified_evidence}"; do
  [[ -s "${evidence}" ]] || fail "missing evidence: ${evidence:-unset}"
done

# shellcheck disable=SC1090
source "${policy}"
for name in policy_version supported_pdgs supported_backend minimum_repetitions \
            minimum_gpu_accepted_events minimum_kfp_speedup minimum_end_to_end_speedup \
            maximum_tail_to_median_ratio maximum_allocated_high_water_bytes \
            maximum_total_fallback_fraction maximum_unexpected_fallbacks \
            parameter_absolute_tolerance covariance_absolute_tolerance \
            fit_scalar_absolute_tolerance mass_absolute_tolerance mass_error_absolute_tolerance; do
  [[ -n "${!name:-}" ]] || fail "policy is missing ${name}"
done

field()
{
  sed -n "s/^$2: //p" "$1" | tail -1
}

for key in source "source revision" build "qualification build" device \
           "XPU HIP architecture" ROCm FairRoot ROOT input parameters; do
  reference="$(field "${cpu_evidence}" "${key}")"
  [[ -n "${reference}" ]] || fail "CPU evidence is missing ${key}"
  [[ "$(field "${diagnostic_evidence}" "${key}")" == "${reference}" ]] \
    || fail "${key} differs between CPU and diagnostic evidence"
  [[ "$(field "${qualified_evidence}" "${key}")" == "${reference}" ]] \
    || fail "${key} differs between CPU and qualified-GPU evidence"
done
[[ "$(field "${qualified_evidence}" "qualification build")" == "trial" ]] \
  || fail "qualified evidence was not produced by a trial build"
qualified_device="$(field "${qualified_evidence}" device)"
if [[ "${supported_backend}" != "HIP" || ! "${qualified_device}" =~ ^hip[0-9]+$ ]]; then
  fail "device ${qualified_device:-unset} does not satisfy frozen backend ${supported_backend}"
fi
cpu_configuration="$(field "${cpu_evidence}" configuration)"
diagnostic_configuration="$(field "${diagnostic_evidence}" configuration)"
qualified_configuration="$(field "${qualified_evidence}" configuration)"
cpu_configuration="${cpu_configuration%; mode=*}"
diagnostic_configuration="${diagnostic_configuration%; mode=*}"
qualified_configuration="${qualified_configuration%; mode=*}"
[[ -n "${cpu_configuration}" && "${diagnostic_configuration}" == "${cpu_configuration}" \
   && "${qualified_configuration}" == "${cpu_configuration}" ]] \
  || fail "configuration differs between CPU, diagnostic, and qualified-GPU evidence"
configuration_path="${cpu_configuration%%;*}"
[[ -f "${configuration_path}" ]] || fail "qualification configuration is not readable: ${configuration_path}"
if grep -q '^[[:space:]]*finderCuts:' "${configuration_path}"; then
  fail "qualification configuration must not contain finderCuts"
fi
requested_pdgs="$(sed -nE 's/^[[:space:]]*-[[:space:]]*pdg:[[:space:]]*(-?[0-9]+).*/\1/p' \
  "${configuration_path}" | sort -n | paste -sd, -)"
expected_pdgs="$(tr ',' '\n' <<<"${supported_pdgs}" | sort -n | paste -sd, -)"
[[ "${requested_pdgs}" == "${expected_pdgs}" ]] \
  || fail "qualification configuration PDGs ${requested_pdgs:-none} do not match ${supported_pdgs}"

grep -Eq '^result: .*mode=cpu-only([[:space:]]|$)' "${cpu_evidence}" \
  || fail "CPU evidence does not describe cpu-only output"
grep -Eq '^result: .*mode=diagnostic([[:space:]]|$)' "${diagnostic_evidence}" \
  || fail "diagnostic evidence does not describe diagnostic output"
grep -Eq '^result: .*mode=qualified-gpu([[:space:]]|$)' "${qualified_evidence}" \
  || fail "qualified evidence does not describe qualified-gpu output"
grep -Eq '^result: .*diagnostic_blocked=0([[:space:]]|$).*diagnostic_blockers=0([[:space:]]|$)' \
  "${diagnostic_evidence}" || fail "diagnostic physics comparison is blocked"

wall_values()
{
  sed -nE 's/^PERFORMANCE_RUN .*wall_ms=([0-9]+).*/\1/p' "$1"
}

median()
{
  sort -n | awk '{v[NR]=$1} END {if (NR == 0) exit 1; if (NR % 2) print v[(NR+1)/2]; else print (v[NR/2]+v[NR/2+1])/2}'
}

tail95()
{
  sort -n | awk '{v[NR]=$1} END {if (NR == 0) exit 1; i=int((95*NR+99)/100); if (i<1) i=1; print v[i]}'
}

cpu_runs="$(wall_values "${cpu_evidence}" | wc -l | tr -d ' ')"
diagnostic_runs="$(wall_values "${diagnostic_evidence}" | wc -l | tr -d ' ')"
qualified_runs="$(wall_values "${qualified_evidence}" | wc -l | tr -d ' ')"
((cpu_runs >= minimum_repetitions && diagnostic_runs >= minimum_repetitions \
  && qualified_runs >= minimum_repetitions)) \
  || fail "at least ${minimum_repetitions} measured repetitions are required for every mode"

cpu_wall_median="$(wall_values "${cpu_evidence}" | median)"
qualified_wall_median="$(wall_values "${qualified_evidence}" | median)"
qualified_wall_p95="$(wall_values "${qualified_evidence}" | tail95)"

diagnostic_cpu_ms="$(
  sed -nE 's/.*KFParticle GPU qualification metrics:.*cpu_reference_ms=([0-9.eE+-]+).*/\1/p' \
    "${diagnostic_evidence}" | median
)" || fail "diagnostic evidence is missing CPU-reference timing"
qualified_gpu_ms="$(
  sed -nE 's/.*KFParticle GPU qualification metrics:.*transaction_ms=([0-9.eE+-]+).*/\1/p' \
    "${qualified_evidence}" | median
)" || fail "qualified evidence is missing GPU transaction timing"

allocated_high_water="$(
  sed -nE 's/.*allocated_high_water_bytes=([0-9]+).*/\1/p' "${qualified_evidence}" \
    | sort -n | tail -1
)"
[[ -n "${allocated_high_water}" ]] || fail "qualified evidence is missing memory high-water monitoring"

gpu_accepted=0
qualification_locked=0
fallback_total=0
fallback_unexpected=0
routing_lines=0
while IFS= read -r line; do
  [[ -n "${line}" ]] || continue
  ((++routing_lines))
  accepted="$(sed -nE 's/.*GPU accepted ([0-9]+).*/\1/p' <<<"${line}")"
  locked="$(sed -nE 's/.*qualification locked ([0-9]+).*/\1/p' <<<"${line}")"
  fallbacks="$(sed -nE 's@.*fallback unsupported/runtime/input/field/overflow/validation/materialization/execution ([0-9]+/[0-9]+/[0-9]+/[0-9]+/[0-9]+/[0-9]+/[0-9]+/[0-9]+).*@\1@p' <<<"${line}")"
  [[ -n "${accepted}" && -n "${locked}" && -n "${fallbacks}" ]] \
    || fail "incomplete qualified-GPU routing monitoring"
  IFS=/ read -r unsupported runtime input field overflow validation materialization execution <<<"${fallbacks}"
  gpu_accepted=$((gpu_accepted + accepted))
  qualification_locked=$((qualification_locked + locked))
  fallback_total=$((fallback_total + unsupported + runtime + input + field + overflow + validation + materialization + execution))
  fallback_unexpected=$((fallback_unexpected + unsupported + runtime + field + overflow + validation + materialization + execution))
done < <(grep 'KFParticle GPU routing monitoring:' "${qualified_evidence}" || true)
((routing_lines >= minimum_repetitions)) || fail "qualified evidence is missing routing samples"
((gpu_accepted >= minimum_gpu_accepted_events)) || fail "no representative GPU event was published"
((qualification_locked == 0)) || fail "qualification build remained locked"
((fallback_unexpected <= maximum_unexpected_fallbacks)) \
  || fail "unexpected fallback count ${fallback_unexpected} exceeds ${maximum_unexpected_fallbacks}"

route_total=$((gpu_accepted + fallback_total))
((route_total > 0)) || fail "qualified routing did not observe any event"
fallback_fraction="$(awk -v f="${fallback_total}" -v n="${route_total}" 'BEGIN {printf "%.8f", f/n}')"
kfp_speedup="$(awk -v c="${diagnostic_cpu_ms}" -v g="${qualified_gpu_ms}" 'BEGIN {if (g<=0) exit 1; printf "%.8f", c/g}')" \
  || fail "qualified GPU transaction time must be positive"
end_to_end_speedup="$(awk -v c="${cpu_wall_median}" -v g="${qualified_wall_median}" 'BEGIN {if (g<=0) exit 1; printf "%.8f", c/g}')" \
  || fail "qualified wall time must be positive"
tail_to_median="$(awk -v p="${qualified_wall_p95}" -v m="${qualified_wall_median}" 'BEGIN {if (m<=0) exit 1; printf "%.8f", p/m}')" \
  || fail "qualified median wall time must be positive"

output="${output:-$(dirname "${qualified_evidence}")/qualification-${mode}-${workload}.evidence.txt}"
decision=qualified
failures=("")
failure_count=0
awk -v actual="${kfp_speedup}" -v required="${minimum_kfp_speedup}" 'BEGIN {exit !(actual >= required)}' \
  || { ((failure_count += 1)); failures+=("KFP speedup ${kfp_speedup} is below ${minimum_kfp_speedup}"); }
awk -v actual="${end_to_end_speedup}" -v required="${minimum_end_to_end_speedup}" 'BEGIN {exit !(actual >= required)}' \
  || { ((failure_count += 1)); failures+=("end-to-end speedup ${end_to_end_speedup} is below ${minimum_end_to_end_speedup}"); }
awk -v actual="${tail_to_median}" -v maximum="${maximum_tail_to_median_ratio}" 'BEGIN {exit !(actual <= maximum)}' \
  || { ((failure_count += 1)); failures+=("tail/median ratio ${tail_to_median} exceeds ${maximum_tail_to_median_ratio}"); }
awk -v actual="${fallback_fraction}" -v maximum="${maximum_total_fallback_fraction}" 'BEGIN {exit !(actual <= maximum)}' \
  || { ((failure_count += 1)); failures+=("fallback fraction ${fallback_fraction} exceeds ${maximum_total_fallback_fraction}"); }
((allocated_high_water <= maximum_allocated_high_water_bytes)) \
  || { ((failure_count += 1)); failures+=("memory high-water ${allocated_high_water} exceeds ${maximum_allocated_high_water_bytes}"); }
if ((failure_count)); then
  decision=rejected
fi
{
  echo "KFParticle CBMRoot Step 20.3 qualification decision"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "decision: ${decision}"
  echo "mode: ${mode}"
  echo "workload: ${workload}"
  echo "policy: ${policy}"
  echo "policy sha256: $(file_hash "${policy}")"
  echo "policy version: ${policy_version}"
  echo "supported PDGs: ${supported_pdgs}"
  echo "supported backend: ${supported_backend}"
  echo "CPU evidence: ${cpu_evidence}"
  echo "diagnostic evidence: ${diagnostic_evidence}"
  echo "qualified GPU evidence: ${qualified_evidence}"
  echo "CPU wall median ms: ${cpu_wall_median}"
  echo "qualified wall median ms: ${qualified_wall_median}"
  echo "qualified wall p95 ms: ${qualified_wall_p95}"
  echo "KFP CPU reference median ms: ${diagnostic_cpu_ms}"
  echo "KFP GPU transaction median ms: ${qualified_gpu_ms}"
  echo "KFP speedup: ${kfp_speedup}"
  echo "end-to-end speedup: ${end_to_end_speedup}"
  echo "tail to median ratio: ${tail_to_median}"
  echo "allocated high water bytes: ${allocated_high_water}"
  echo "GPU accepted events: ${gpu_accepted}"
  echo "fallback events: ${fallback_total}"
  echo "unexpected fallback events: ${fallback_unexpected}"
  echo "fallback fraction: ${fallback_fraction}"
  for ((index = 1; index <= failure_count; ++index)); do
    echo "reason: ${failures[index]}"
  done
} >"${output}"

if [[ "${decision}" == "rejected" ]]; then
  reason_summary="${failures[1]}"
  for ((index = 2; index <= failure_count; ++index)); do
    reason_summary="${reason_summary}; ${failures[index]}"
  done
  printf 'FAIL cbmroot-kfp-qualification-%s - %s\n' "${mode}" "${reason_summary}" >&2
  echo "  evidence   : ${output}" >&2
  exit 1
fi
echo "PASS cbmroot-kfp-qualification-${mode} - ${workload} physics, routing, memory, stability, and speed gates passed"
echo "  evidence   : ${output}"
