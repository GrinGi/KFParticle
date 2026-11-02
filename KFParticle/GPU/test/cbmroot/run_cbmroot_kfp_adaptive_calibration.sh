#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE:-}"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
profiles="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES:-batch8:1:1000000000:8:1073741824 batch16:1:1000000000:16:1073741824 batch32:1:1000000000:32:1073741824}"
screen_warmup="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_SCREEN_WARMUP:-}"
warmup="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_WARMUP:-}"
measured="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_REPETITIONS:-}"
minimum_attained="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MINIMUM_ATTAINED_BATCH:-8}"
minimum_wall_speedup="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MINIMUM_WALL_SPEEDUP:-1.0}"
allow_rejected_screening="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_ALLOW_REJECTED_SCREENING:-0}"
output_dir="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR:-${build_dir}/kfparticle-adaptive-calibration/${mode}}"

fail() { echo "FAIL cbmroot-kfp-adaptive-calibration - $*" >&2; exit 2; }
case "${mode}" in
  online)
    preflight="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER:-${script_dir}/run_cbmroot_kfp_online_preflight.sh}"
    screen_warmup="${screen_warmup:-1}"
    warmup="${warmup:-1}"
    measured="${measured:-3}"
    ;;
  offline)
    preflight="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER:-${script_dir}/run_cbmroot_kfp_offline_preflight.sh}"
    screen_warmup="${screen_warmup:-0}"
    warmup="${warmup:-0}"
    measured="${measured:-1}"
    ;;
  *) fail "set KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE to online or offline" ;;
esac
[[ -d "${build_dir}" ]] || fail "set KFPARTICLE_CBMROOT_BUILD_DIR"
[[ -x "${preflight}" || -f "${preflight}" ]] || fail "missing preflight launcher: ${preflight}"
[[ "${screen_warmup}" =~ ^[0-9]+$ ]] || fail "screening warm-up count must be non-negative"
[[ "${warmup}" =~ ^[0-9]+$ ]] || fail "final warm-up count must be non-negative"
[[ "${measured}" =~ ^[1-9][0-9]*$ ]] || fail "at least one final repetition is required"
[[ "${minimum_attained}" =~ ^[1-9][0-9]*$ ]] || fail "minimum attained batch must be positive"
[[ "${minimum_wall_speedup}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "minimum wall speedup is invalid"
[[ "${allow_rejected_screening}" == "0" || "${allow_rejected_screening}" == "1" ]] \
  || fail "KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_ALLOW_REJECTED_SCREENING must be 0 or 1"
mkdir -p "${output_dir}"

median() { sort -n | awk '{v[NR]=$1} END {if (!NR) exit 1; print NR%2?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
wall_median() { sed -nE 's/^PERFORMANCE_RUN .*wall_ms=([0-9]+).*/\1/p' "$1" | median; }
launcher_median() { sed -nE 's/^PERFORMANCE_RUN .*launcher_ms=([0-9]+).*/\1/p' "$1" | median; }
sum_numbers() { awk -v a="$1" -v b="$2" -v c="${3:-0}" 'BEGIN {printf "%.6f", a+b+c}'; }
subtract_numbers() { awk -v a="$1" -v b="$2" 'BEGIN {printf "%.6f", a-b}'; }
divide_numbers() { awk -v a="$1" -v b="$2" 'BEGIN {if (b<=0) exit 1; printf "%.6f", a/b}'; }
metric_values()
{
  awk -v key="$1" '/KFParticle GPU qualification metrics:/ {for(i=1;i<=NF;++i){split($i,a,"=");if(a[1]==key)print a[2]}}' "$2"
}
metric_median() { metric_values "$1" "$2" | median; }
metric_unique()
{
  local values count
  values="$(metric_values "$1" "$2" | sort -u)"
  count="$(printf '%s\n' "${values}" | sed '/^$/d' | wc -l | tr -d ' ')"
  [[ "${count}" == 1 ]] || return 1
  printf '%s\n' "${values}"
}
summary_values()
{
  local marker='KFP online summary:'
  [[ "${mode}" != offline ]] || marker='KFP selector summary:'
  awk -v marker="${marker}" -v key="$1" \
    'index($0, marker) {for(i=1;i<=NF;++i){split($i,a,"=");if(a[1]==key)print a[2]}}' "$2"
}
summary_unique()
{
  local values count
  values="$(summary_values "$1" "$2" | sort -u)"
  count="$(printf '%s\n' "${values}" | sed '/^$/d' | wc -l | tr -d ' ')"
  [[ "${count}" == 1 ]] || return 1
  printf '%s\n' "${values}"
}

run_preflight()
{
  local execution="$1" phase="$2" repetitions="$3" name="$4"
  local minimum="$5" target="$6" events="$7" bytes="$8" evidence="$9"
  local run_dir="${output_dir}/${name}/${phase}-${execution}"
  local output_prefix="${KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX:-}"
  local expectation=locked
  [[ "${execution}" != qualified-gpu ]] || expectation=published
  [[ "${mode}" != offline || -z "${output_prefix}" ]] || output_prefix="${output_prefix}-${name}-${phase}-${execution}"
  if [[ "${execution}" == cpu-only ]]; then minimum=0; target=0; bytes=0; fi
  local -a environment=(
    KFPARTICLE_CBMROOT_KFP_MODE="${execution}"
    KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION="${expectation}"
    KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD=trial
    KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1
    KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE=0
    KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR="${run_dir}"
    KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE="${evidence}"
    KFPARTICLE_CBMROOT_OFFLINE_OUTPUT_PREFIX="${output_prefix}"
    KFPARTICLE_CBMROOT_DIAGNOSTIC_BATCH_EVENTS="${events}"
    KFPARTICLE_CBMROOT_ADAPTIVE_MINIMUM_WORK="${minimum}"
    KFPARTICLE_CBMROOT_ADAPTIVE_TARGET_WORK="${target}"
    KFPARTICLE_CBMROOT_ADAPTIVE_MAXIMUM_BYTES="${bytes}"
    KFPARTICLE_CBMROOT_PREFLIGHT_ALLOW_DIAGNOSTIC_REJECTION="${allow_rejected_screening}"
    KFPARTICLE_GPU_PERFORMANCE_MONITORING=1
  )
  if [[ "${mode}" == online ]]; then
    environment+=(KFPARTICLE_CBMROOT_ONLINE_REPEAT="${repetitions}")
  else
    environment+=(KFPARTICLE_CBMROOT_OFFLINE_REPEAT="${repetitions}")
  fi
  rm -f "${evidence}"
  env "${environment[@]}" bash "${preflight}"
}

validate_profile_metadata()
{
  local evidence="$1" minimum="$2" target="$3" events="$4" bytes="$5" line
  line="$(grep "KFParticle GPU adaptive ${mode} calibration:" "${evidence}" | tail -1 || true)"
  [[ "${line}" == *"profile=enabled"* \
     && "${line}" == *"minimum_work=${minimum} target_work=${target} maximum_bytes=${bytes} maximum_events=${events}"* \
     && "${line}" == *"profile_mismatches=0"* ]] || return 1
  printf '%s\n' "${line#*KFParticle GPU adaptive ${mode} calibration: }"
}

diagnostic_physics_status()
{
  local evidence="$1" blocked blockers
  blocked="$(sed -nE 's/^result: .*diagnostic_blocked=([0-9]+).*/\1/p' "${evidence}" | tail -1)"
  blockers="$(sed -nE 's/^result: .*diagnostic_blockers=([0-9]+).*/\1/p' "${evidence}" | tail -1)"
  [[ "${blocked}" =~ ^[0-9]+$ && "${blockers}" =~ ^[0-9]+$ ]] || return 1
  if [[ "${blocked}" == "0" && "${blockers}" == "0" ]]; then
    echo passed
  else
    echo rejected
  fi
}

validate_profile()
{
  local attempted
  validate_profile_metadata "$@" >/dev/null || return 1
  attempted="$(metric_unique events "$1")" || return 1
  [[ "${attempted}" =~ ^[1-9][0-9]*$ \
     && "$(metric_unique completed "$1")" == "${attempted}" \
     && "$(metric_unique failed "$1")" == 0 \
     && "$(metric_unique source_order_violations "$1")" == 0 ]] || return 1
  [[ "$(diagnostic_physics_status "$1")" == passed ]]
}

results="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT:-${output_dir}/adaptive-${mode}-calibration.evidence.txt}"
tmp_results="${results}.tmp"
ranking="${output_dir}/screening-ranking.tsv"
mkdir -p "$(dirname "${results}")"
# A failed fresh measurement must not leave a previous selection available.
rm -f "${results}"
trap 'rm -f "${tmp_results}"' EXIT
: >"${ranking}"
{
  echo "KFParticle adaptive ${mode} same-cohort calibration"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "mode: ${mode}"
  echo "device: ${device}"
  echo "profiles: ${profiles}"
  echo "screening repetitions per profile: warm-up ${screen_warmup}, measured 1"
  echo "final repetitions: warm-up ${warmup}, measured ${measured}"
  echo "minimum attained batch: ${minimum_attained}"
  echo "minimum end-to-end speedup: ${minimum_wall_speedup}"
  echo "allow rejected diagnostic screening: ${allow_rejected_screening}"
} >"${tmp_results}"

seen=" "
reference_cohort=""
screened=0
for specification in ${profiles}; do
  IFS=: read -r name minimum target events bytes extra <<<"${specification}"
  [[ -z "${extra:-}" && "${name}" =~ ^[A-Za-z0-9._-]+$ && "${minimum}" =~ ^[1-9][0-9]*$ \
     && "${target}" =~ ^[1-9][0-9]*$ && "${events}" =~ ^[1-9][0-9]*$ && "${bytes}" =~ ^[1-9][0-9]*$ ]] \
    || fail "profiles must use name:minimum:target:events:maximum-bytes"
  ((target >= minimum)) || fail "profile ${name} target is below minimum"
  [[ "${seen}" != *" ${name} "* ]] || fail "duplicate profile ${name}"
  seen+="${name} "
  dir="${output_dir}/${name}"; mkdir -p "${dir}"
  status=0
  if ((screen_warmup > 0)); then
    run_preflight diagnostic screen-warmup "${screen_warmup}" "${name}" "${minimum}" "${target}" "${events}" "${bytes}" "${dir}/screen-warmup.evidence.txt" || status=$?
  fi
  evidence="${dir}/screen-diagnostic.evidence.txt"
  ((status != 0)) || run_preflight diagnostic screen 1 "${name}" "${minimum}" "${target}" "${events}" "${bytes}" "${evidence}" || status=$?
  reason=""; calibration=""; physics_contract=""
  if ((status != 0)); then
    reason="preflight status ${status}"
  elif ! calibration="$(validate_profile_metadata "${evidence}" "${minimum}" "${target}" "${events}" "${bytes}")"; then
    reason="adaptive profile contract failed"
  elif ! physics_contract="$(diagnostic_physics_status "${evidence}")"; then
    reason="diagnostic physics status is missing"
  elif [[ "${physics_contract}" == rejected && "${allow_rejected_screening}" != "1" ]]; then
    reason="diagnostic physics contract rejected"
  else
    attempted="$(metric_unique events "${evidence}" || true)"
    completed="$(metric_unique completed "${evidence}" || true)"
    failed="$(metric_unique failed "${evidence}" || true)"
    largest="$(metric_unique largest_batch "${evidence}" || true)"
    order="$(metric_unique source_order_violations "${evidence}" || true)"
    cohort_xor="$(metric_unique cohort_xor "${evidence}" || true)"
    cohort_sum="$(metric_unique cohort_sum "${evidence}" || true)"
    if [[ -z "${attempted}" || "${attempted}" != "${completed}" || "${failed}" != 0 || "${order}" != 0 \
          || -z "${cohort_xor}" || -z "${cohort_sum}" ]]; then
      reason="incomplete, failed, unordered, or unidentified cohort"
    else
      cohort="${attempted}:${cohort_xor}:${cohort_sum}"
      [[ -n "${reference_cohort}" ]] || reference_cohort="${cohort}"
      [[ "${cohort}" == "${reference_cohort}" ]] || reason="cohort differs from ${reference_cohort}"
      required=$(((events + 1) / 2)); ((required >= minimum_attained)) || required="${minimum_attained}"
      ((required <= events)) || required="${events}"
      [[ -n "${reason}" ]] || ((largest >= required)) || reason="attained batch ${largest} is below ${required}"
    fi
  fi
  if [[ -n "${reason}" ]]; then
    printf 'candidate: %s\ncandidate decision: rejected during screening\ncandidate reason: %s\n' "${name}" "${reason}" >>"${tmp_results}"
    continue
  fi
  cpu="$(metric_median cpu_reference_ms "${evidence}" || true)"
  gpu="$(metric_median transaction_ms "${evidence}" || true)"
  high_water="$(sed -nE 's/.*allocated_high_water_bytes=([0-9]+).*/\1/p' "${evidence}" | sort -n | tail -1)"
  [[ -n "${cpu}" && -n "${gpu}" && -n "${high_water}" ]] || fail "profile ${name} lacks timing or memory evidence"
  if ((high_water > bytes)); then
    printf 'candidate: %s\ncandidate decision: rejected during screening\ncandidate reason: memory high-water %s exceeds profile limit %s\n' \
      "${name}" "${high_water}" "${bytes}" >>"${tmp_results}"
    continue
  fi
  speedup="$(awk -v c="${cpu}" -v g="${gpu}" 'BEGIN{if(g<=0)exit 1;printf "%.8f",c/g}')" || fail "invalid timing for ${name}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${speedup}" "${minimum}" "${target}" "${events}" "${bytes}" \
    "${physics_contract}" "${evidence}" >>"${ranking}"
  {
    echo "candidate: ${name}"
    echo "candidate profile: minimum=${minimum} target=${target} events=${events} maximum-bytes=${bytes}"
    echo "candidate decision: passed screening"
    echo "candidate physics contract: ${physics_contract}"
    echo "candidate cohort: count=${attempted} xor=${cohort_xor} sum=${cohort_sum}"
    echo "candidate attained batch: ${largest}"
    echo "candidate same-cohort timing: cpu-reference-ms=${cpu} gpu-transaction-ms=${gpu} kfp-speedup=${speedup}"
    echo "candidate memory high-water bytes: ${high_water}"
    echo "candidate calibration: ${calibration}"
  } >>"${tmp_results}"
  ((++screened))
done

((screened > 0)) || { mv "${tmp_results}" "${results}"; fail "no profile passed screening; evidence: ${results}"; }
IFS=$'\t' read -r finalist finalist_speedup minimum target events bytes finalist_physics diagnostic_evidence < \
  <(sort -t $'\t' -k2,2gr "${ranking}" | head -1)
cpu_dir="${output_dir}/cpu-baseline"; mkdir -p "${cpu_dir}"
if ((warmup > 0)); then
  run_preflight cpu-only final-warmup "${warmup}" cpu-baseline "${minimum}" "${target}" "${events}" "${bytes}" "${cpu_dir}/warmup.evidence.txt"
fi
cpu_evidence="${cpu_dir}/measured.evidence.txt"
run_preflight cpu-only final "${measured}" cpu-baseline "${minimum}" "${target}" "${events}" "${bytes}" "${cpu_evidence}"
cpu_wall="$(wall_median "${cpu_evidence}")" || fail "CPU evidence lacks wall times"
final_dir="${output_dir}/${finalist}"
if ((warmup > 0)); then
  run_preflight qualified-gpu final-warmup "${warmup}" "${finalist}" "${minimum}" "${target}" "${events}" "${bytes}" "${final_dir}/warmup-qualified.evidence.txt"
fi
qualified_evidence="${final_dir}/measured-qualified.evidence.txt"
run_preflight qualified-gpu final "${measured}" "${finalist}" "${minimum}" "${target}" "${events}" "${bytes}" "${qualified_evidence}"
validate_profile "${qualified_evidence}" "${minimum}" "${target}" "${events}" "${bytes}" >/dev/null || fail "finalist ${finalist} failed qualified contract"
qualified_wall="$(wall_median "${qualified_evidence}")" || fail "qualified evidence lacks wall times"
wall_speedup="$(awk -v c="${cpu_wall}" -v g="${qualified_wall}" 'BEGIN{if(g<=0)exit 1;printf "%.8f",c/g}')" || fail "invalid finalist wall timing"

kfp_events="$(metric_unique events "${diagnostic_evidence}")" || fail "diagnostic finalist lacks one event count"
cpu_kfp="$(metric_median cpu_reference_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks CPU time"
input_preparation="$(metric_median input_preparation_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks input-preparation time"
capacity_planning="$(metric_median capacity_planning_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks capacity-planning time"
gpu_pipeline="$(metric_median gpu_pipeline_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks GPU-pipeline time"
kernel_work="$(metric_median kernel_work_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks kernel-work time"
gpu_transaction="$(metric_median transaction_ms "${diagnostic_evidence}")" || fail "diagnostic finalist lacks transaction time"
gpu_complete="$(sum_numbers "${input_preparation}" "${capacity_planning}" "${gpu_transaction}")"
gpu_overhead="$(subtract_numbers "${gpu_complete}" "${kernel_work}")"

qualified_events="$(metric_unique events "${qualified_evidence}")" || fail "qualified finalist lacks one event count"
qualified_largest="$(metric_unique largest_batch "${qualified_evidence}")" || fail "qualified finalist lacks attained batch size"
qualified_preparation="$(metric_median input_preparation_ms "${qualified_evidence}")" || fail "qualified finalist lacks input-preparation time"
qualified_planning="$(metric_median capacity_planning_ms "${qualified_evidence}")" || fail "qualified finalist lacks capacity-planning time"
qualified_transaction="$(metric_median transaction_ms "${qualified_evidence}")" || fail "qualified finalist lacks transaction time"
qualified_complete="$(sum_numbers "${qualified_preparation}" "${qualified_planning}" "${qualified_transaction}")"

if [[ "${mode}" == online ]]; then
  requested_full_events="${KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS:-${kfp_events}}"
else
  requested_full_events="${KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES:-${kfp_events}}"
fi
[[ "${requested_full_events}" =~ ^[1-9][0-9]*$ ]] || fail "requested complete-run event count is invalid"
cpu_full_events="$(summary_unique processed "${cpu_evidence}" || true)"
qualified_full_events="$(summary_unique processed "${qualified_evidence}" || true)"
if [[ -n "${cpu_full_events}" || -n "${qualified_full_events}" ]]; then
  [[ -n "${cpu_full_events}" && "${cpu_full_events}" == "${qualified_full_events}" ]] \
    || fail "CPU and qualified complete runs processed different event counts"
  full_events="${cpu_full_events}"
else
  # Synthetic launchers predating the processed-event summary retain their
  # configured count. Real CBMRoot preflights always publish processed.
  full_events="${requested_full_events}"
fi
[[ "${full_events}" =~ ^[1-9][0-9]*$ ]] || fail "processed complete-run event count is invalid"
cpu_launcher="$(launcher_median "${cpu_evidence}" || printf '%s\n' "${cpu_wall}")"
qualified_launcher="$(launcher_median "${qualified_evidence}" || printf '%s\n' "${qualified_wall}")"

required=$(((events + 1) / 2)); ((required >= minimum_attained)) || required="${minimum_attained}"
((required <= events)) || required="${events}"
decision=selected
decision_reason=none
if [[ "${finalist_physics}" != passed ]]; then
  decision=rejected
  decision_reason="diagnostic physics contract rejected"
fi
if ((qualified_largest < required)); then
  decision=rejected
  if [[ "${decision_reason}" == none ]]; then
    decision_reason="qualified attained batch ${qualified_largest} is below ${required}"
  else
    decision_reason+="; qualified attained batch ${qualified_largest} is below ${required}"
  fi
fi
if ! awk -v a="${wall_speedup}" -v r="${minimum_wall_speedup}" 'BEGIN{exit !(a>=r)}'; then
  decision=rejected
  if [[ "${decision_reason}" == none ]]; then
    decision_reason="complete-run speedup ${wall_speedup} is below ${minimum_wall_speedup}"
  else
    decision_reason+="; complete-run speedup ${wall_speedup} is below ${minimum_wall_speedup}"
  fi
fi
{
  echo "screened candidates: ${screened}"
  echo "source cohort: ${reference_cohort}"
  echo "evaluated finalist: ${finalist}"
  echo "finalist physics contract: ${finalist_physics}"
  [[ "${decision}" != selected ]] || echo "selected profile: ${finalist}"
  [[ "${decision}" == selected ]] || echo "selected profile: none"
  echo "finalist profile passport: minimum=${minimum} target=${target} events=${events} maximum-bytes=${bytes}"
  echo "selected same-cohort diagnostic evidence: ${diagnostic_evidence}"
  echo "selected CPU evidence: ${cpu_evidence}"
  echo "selected qualified evidence: ${qualified_evidence}"
  echo "KFP same-cohort comparison: events=${kfp_events}"
  echo "KFP CPU: total-ms=${cpu_kfp} per-event-ms=$(divide_numbers "${cpu_kfp}" "${kfp_events}")"
  echo "KFP GPU kernels: total-ms=${kernel_work} per-event-ms=$(divide_numbers "${kernel_work}" "${kfp_events}") speedup=$(divide_numbers "${cpu_kfp}" "${kernel_work}")"
  echo "KFP GPU pipeline: total-ms=${gpu_pipeline} per-event-ms=$(divide_numbers "${gpu_pipeline}" "${kfp_events}") speedup=$(divide_numbers "${cpu_kfp}" "${gpu_pipeline}")"
  echo "KFP GPU transaction: total-ms=${gpu_transaction} per-event-ms=$(divide_numbers "${gpu_transaction}" "${kfp_events}") speedup=$(divide_numbers "${cpu_kfp}" "${gpu_transaction}")"
  echo "KFP GPU complete: total-ms=${gpu_complete} per-event-ms=$(divide_numbers "${gpu_complete}" "${kfp_events}") speedup=$(divide_numbers "${cpu_kfp}" "${gpu_complete}")"
  echo "KFP GPU overhead outside kernels: total-ms=${gpu_overhead} per-event-ms=$(divide_numbers "${gpu_overhead}" "${kfp_events}")"
  echo "qualified GPU route: events=${qualified_events} largest-batch=${qualified_largest} complete-kfp-ms=${qualified_complete} per-routed-event-ms=$(divide_numbers "${qualified_complete}" "${qualified_events}")"
  echo "complete reconstruction comparison: requested-events=${requested_full_events} processed-events=${full_events}"
  echo "complete CPU: wall-total-ms=${cpu_wall} wall-per-event-ms=$(divide_numbers "${cpu_wall}" "${full_events}") launcher-total-ms=${cpu_launcher} launcher-per-event-ms=$(divide_numbers "${cpu_launcher}" "${full_events}")"
  echo "complete qualified GPU: wall-total-ms=${qualified_wall} wall-per-event-ms=$(divide_numbers "${qualified_wall}" "${full_events}") launcher-total-ms=${qualified_launcher} launcher-per-event-ms=$(divide_numbers "${qualified_launcher}" "${full_events}")"
  echo "complete reconstruction speedup: ${wall_speedup}"
  echo "decision: ${decision}"
  echo "decision reason: ${decision_reason}"
} >>"${tmp_results}"
mv "${tmp_results}" "${results}"
echo "PASS cbmroot-kfp-adaptive-calibration - ${mode} campaign completed; decision=${decision} finalist=${finalist}"
echo "  evidence   : ${results}"
