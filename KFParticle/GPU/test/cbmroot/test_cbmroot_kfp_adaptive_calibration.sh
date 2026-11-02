#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/build"

# CTest inherits the caller's real campaign environment. Keep this synthetic
# contract deterministic even when it is launched from a configured terminal.
while IFS= read -r variable; do
  unset "$variable"
done < <(compgen -A variable KFPARTICLE_ || true)
export KFPARTICLE_CBMROOT_ONLINE_MAX_EVENTS=64
export KFPARTICLE_CBMROOT_OFFLINE_NUM_ENTRIES=64

cat >"${tmp_dir}/fake-preflight.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
evidence="${KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE}"
mode="${KFPARTICLE_CBMROOT_KFP_MODE}"
environment="${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE}"
repeat="${KFPARTICLE_CBMROOT_ONLINE_REPEAT:-${KFPARTICLE_CBMROOT_OFFLINE_REPEAT:-1}}"
[[ -z "${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_CALL_LOG:-}" ]] \
  || echo "${mode}" >>"${KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_CALL_LOG}"
if [[ "${mode}" == "cpu-only" \
      && ("${KFPARTICLE_CBMROOT_ADAPTIVE_MINIMUM_WORK}" != "0" \
          || "${KFPARTICLE_CBMROOT_ADAPTIVE_TARGET_WORK}" != "0" \
          || "${KFPARTICLE_CBMROOT_ADAPTIVE_MAXIMUM_BYTES}" != "0") ]]; then
  echo "CPU baseline received an adaptive GPU profile" >&2
  exit 9
fi
events="${KFPARTICLE_CBMROOT_DIAGNOSTIC_BATCH_EVENTS}"
largest="${events}"
[[ "${KFPARTICLE_CBMROOT_ADAPTIVE_MINIMUM_WORK}" != "999" ]] || largest=1
wall=100
[[ "${mode}" != "cpu-only" ]] || wall=200
transaction=30
[[ "${events}" != 8 ]] || transaction=20
diagnostic_blocked=0
diagnostic_blockers=0
completed=64
if [[ "${mode}" == qualified-gpu ]]; then
  case "${KFPARTICLE_TEST_CALIBRATION_FAULT:-}" in
    abort) exit 7 ;;
    missing) exit 0 ;;
    incomplete) completed=63 ;;
  esac
fi
if [[ "${mode}" == diagnostic && "${KFPARTICLE_TEST_DIAGNOSTIC_REJECTED:-0}" == 1 ]]; then
  diagnostic_blocked=3
  diagnostic_blockers=32
fi
mkdir -p "$(dirname "${evidence}")"
{
  for ((index=1; index<=repeat; ++index)); do
    echo "PERFORMANCE_RUN mode=${mode} run=${index} wall_ms=$((wall + index)) launcher_ms=$((wall + index + 10))"
    if [[ "${environment}" == online ]]; then
      echo "[INFO] KFP online summary: events=64 processed=60 mode=${mode}"
    else
      echo "[INFO] KFP selector summary: events=64 processed=60 mode=${mode}"
    fi
    if [[ "${mode}" != "cpu-only" ]]; then
      echo "[INFO] KFParticle GPU qualification metrics: events=64 completed=${completed} failed=0 batch_limit=${events} largest_batch=${largest} source_order_violations=0 cohort_xor=123456 cohort_sum=654321 cpu_reference_ms=80 input_preparation_ms=4 capacity_planning_ms=6 queue_wait_ms=1 queue_capacity_ms=2 host_packing_ms=3 h2d_ms=4 kernel_work_ms=8 d2h_ms=5 gpu_pipeline_ms=17 result_extraction_ms=2 materialization_ms=1 transaction_ms=${transaction}"
      echo "[INFO] KFParticle GPU performance: samples=1 allocated_high_water_bytes=4096"
      echo "[INFO] KFParticle GPU adaptive ${environment} calibration: profile=enabled minimum_work=${KFPARTICLE_CBMROOT_ADAPTIVE_MINIMUM_WORK} target_work=${KFPARTICLE_CBMROOT_ADAPTIVE_TARGET_WORK} maximum_bytes=${KFPARTICLE_CBMROOT_ADAPTIVE_MAXIMUM_BYTES} maximum_events=${KFPARTICLE_CBMROOT_DIAGNOSTIC_BATCH_EVENTS} profile_mismatches=0 reason_events collecting/gpu/disabled/no-work/below/resource=0/4/0/0/0/0 reason_work=0/80/0/0/0/0 reason_bytes=0/4096/0/0/0/0"
    fi
  done
  echo "result: mode=${mode} diagnostic_blocked=${diagnostic_blocked} diagnostic_blockers=${diagnostic_blockers}"
} >"${evidence}"
EOF
chmod +x "${tmp_dir}/fake-preflight.sh"

run_calibration()
{
  env KFPARTICLE_CBMROOT_BUILD_DIR="${tmp_dir}/build" \
    KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER="${tmp_dir}/fake-preflight.sh" \
    "$@" bash "${script_dir}/run_cbmroot_kfp_adaptive_calibration.sh"
}

for mode in online offline; do
  output="${tmp_dir}/${mode}.evidence.txt"
  run_calibration \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE="${mode}" \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES='batch8:10:20:8:4096 batch16:20:40:16:8192' \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR="${tmp_dir}/${mode}" \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT="${output}" \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_WARMUP=1 \
  KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_REPETITIONS=3 >/dev/null
  grep -q "^mode: ${mode}$" "${output}"
  grep -q '^screened candidates: 2$' "${output}"
  grep -q '^candidate: batch8$' "${output}"
  grep -q '^candidate: batch16$' "${output}"
  grep -q '^source cohort: 64:123456:654321$' "${output}"
  grep -q '^selected profile: batch8$' "${output}"
  grep -q '^finalist physics contract: passed$' "${output}"
  grep -q '^KFP CPU: total-ms=80 per-event-ms=1.250000$' "${output}"
  grep -q '^KFP GPU complete: total-ms=30.000000 per-event-ms=0.468750 speedup=2.666667$' "${output}"
  grep -q '^complete reconstruction comparison: requested-events=64 processed-events=60$' "${output}"
  grep -q '^decision reason: none$' "${output}"
  grep -q '^decision: selected$' "${output}"
  grep -q 'profile_mismatches=0' "${output}"
done

rejected_output="${tmp_dir}/rejected.evidence.txt"
run_calibration \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE=online \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER="${tmp_dir}/fake-preflight.sh" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES='rejected:999:999:16:4096 accepted:10:20:8:4096' \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR="${tmp_dir}/mixed" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT="${rejected_output}" >/dev/null
grep -A3 '^candidate: rejected$' "${rejected_output}" | grep -q '^candidate decision: rejected during screening$'
grep -q '^screened candidates: 1$' "${rejected_output}"
grep -q '^selected profile: accepted$' "${rejected_output}"

# A complete diagnostic run may be retained as negative evidence, but must not
# authorize publication or turn malformed input into a successful candidate.
physics_rejected_output="${tmp_dir}/physics-rejected.evidence.txt"
run_calibration \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE=online \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER="${tmp_dir}/fake-preflight.sh" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES='blocked:10:20:8:4096' \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR="${tmp_dir}/physics-rejected" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT="${physics_rejected_output}" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_ALLOW_REJECTED_SCREENING=1 \
KFPARTICLE_TEST_DIAGNOSTIC_REJECTED=1 >/dev/null
grep -q '^candidate physics contract: rejected$' "${physics_rejected_output}"
grep -q '^finalist physics contract: rejected$' "${physics_rejected_output}"
grep -q '^selected profile: none$' "${physics_rejected_output}"
grep -q '^decision: rejected$' "${physics_rejected_output}"
grep -q '^decision reason: diagnostic physics contract rejected$' "${physics_rejected_output}"

lean_output="${tmp_dir}/offline-lean.evidence.txt"
call_log="${tmp_dir}/offline-lean.calls.txt"
run_calibration \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE=offline \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_LAUNCHER="${tmp_dir}/fake-preflight.sh" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES='batch8:10:20:8:4096 batch16:20:40:16:8192 batch32:30:60:32:16384' \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR="${tmp_dir}/offline-lean" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT="${lean_output}" \
KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_CALL_LOG="${call_log}" >/dev/null
[[ "$(wc -l <"${call_log}" | tr -d ' ')" == 5 ]]
grep -q '^screening repetitions per profile: warm-up 0, measured 1$' "${lean_output}"
grep -q '^final repetitions: warm-up 0, measured 1$' "${lean_output}"

for fault in abort missing incomplete; do
  output="${tmp_dir}/failure.evidence.txt"
  # Seed old success: neither a failed launch nor missing output may reuse it.
  cp "${lean_output}" "${output}"
  if run_calibration \
    KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_MODE=offline \
      KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_PROFILES='batch8:10:20:8:4096' \
    KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_DIR="${tmp_dir}/offline-lean" \
    KFPARTICLE_CBMROOT_ADAPTIVE_CALIBRATION_OUTPUT="${output}" \
    KFPARTICLE_TEST_CALIBRATION_FAULT="${fault}" >"${tmp_dir}/${fault}.log" 2>&1; then
    echo "FAIL calibration accepted ${fault}" >&2; exit 1
  fi
  [[ ! -e "${output}" && ! -e "${output}.tmp" ]]
done

echo "PASS cbmroot-kfp-adaptive-calibration-contract - same-cohort selection works and default offline calibration is bounded to five launches"
