#!/usr/bin/env bash
set -euo pipefail

source_dir="${KFPARTICLE_CBMROOT_SOURCE_DIR:?set KFPARTICLE_CBMROOT_SOURCE_DIR}"
build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:?set KFPARTICLE_CBMROOT_BUILD_DIR}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
samples="${KFPARTICLE_GPU_IN_PROCESS_REPEATS:-5}"
materialization_schedule="${KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE:-guided}"
profile_mode="${KFPARTICLE_GPU_OPTIMIZED_REPLAY_PROFILE:-0}"
preflight="${source_dir}/external/KFParticle/KFParticle/GPU/test/cbmroot/run_cbmroot_kfp_online_preflight.sh"
output_dir="${KFPARTICLE_GPU_OPTIMIZED_REPLAY_OUTPUT_DIR:-${build_dir}/kfp-optimized-replay}"
aggregate_program="${source_dir}/external/KFParticle/KFParticle/GPU/test/cbmroot/optimized_replay_aggregate.awk"

[[ "${samples}" =~ ^[1-9][0-9]*$ ]] && (( samples <= 20 )) || {
  echo "FAIL optimized-replay - samples must be an integer from 1 to 20" >&2
  exit 1
}
case "${materialization_schedule}" in
  weighted-contiguous|static|dynamic|guided) ;;
  *)
    echo "FAIL optimized-replay - materialization schedule must be weighted-contiguous, static, dynamic, or guided" >&2
    exit 1
    ;;
esac
[[ "${profile_mode}" == "0" || "${profile_mode}" == "1" ]] || {
  echo "FAIL optimized-replay - KFPARTICLE_GPU_OPTIMIZED_REPLAY_PROFILE must be 0 or 1" >&2
  exit 1
}
if [[ "${profile_mode}" == "1" && "${XPU_PROFILE:-0}" != "1" ]]; then
  echo "FAIL optimized-replay - profile mode requires XPU_PROFILE=1 before XPU initialization" >&2
  exit 1
fi
(( samples >= 2 )) || {
  echo "FAIL optimized-replay - at least two samples are required" >&2
  exit 1
}
[[ -x "${build_dir}/bin/cbmreco" ]] || {
  echo "FAIL optimized-replay - missing cbmreco: ${build_dir}/bin/cbmreco" >&2
  exit 1
}
[[ -f "${aggregate_program}" ]] || {
  echo "FAIL optimized-replay - missing replay aggregator: ${aggregate_program}" >&2
  exit 1
}
mkdir -p "${output_dir}"

field()
{
  local key="$1"
  awk -v key="${key}" '{
    for (i = 1; i <= NF; ++i) {
      split($i, value, "=")
      if (value[1] == key) { print value[2]; exit }
    }
  }'
}

if ! KFPARTICLE_GPU_TRANSACTION_VALIDATION_REFERENCE=0 \
  KFPARTICLE_GPU_IN_PROCESS_REPLAY=1 \
  KFPARTICLE_GPU_IN_PROCESS_REPEATS="${samples}" \
  KFPARTICLE_GPU_DEEP_TRACE=0 \
  KFPARTICLE_GPU_DEEP_TRACE_COMPACT=0 \
  KFPARTICLE_CPU_FINDER_TRACE_PAIRS= \
  KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS=0 \
  KFPARTICLE_GPU_CONSTRUCTION_TIMERS="${profile_mode}" \
  KFPARTICLE_GPU_DEVICE_PROFILE="${profile_mode}" \
  KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE="${materialization_schedule}" \
  KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR="${output_dir}" \
  KFPARTICLE_CBMROOT_KFP_MODE=qualified-gpu \
  KFPARTICLE_CBMROOT_ONLINE_REPEAT=1 \
  KFPARTICLE_CBMROOT_PREFLIGHT_EXECUTE=1 \
  KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE=0 \
  KFPARTICLE_CBMROOT_QUALIFICATION_EXPECTATION=published \
  KFPARTICLE_GPU_PERFORMANCE_MONITORING=1 \
  bash "${preflight}" 2>&1 \
    | tee "${output_dir}/console.log" \
    | awk '
        /KFParticle CBMRoot online preflight/ ||
        /^  (source dir|build dir|mode|device|event limit|event batch|performance|command) *:/ ||
        />>> Processing TS / ||
        /KFP online summary:/ ||
        /^PERFORMANCE_RUN / ||
        /^  PASS run / ||
        /^PASS cbmroot-kfp-online-preflight / ||
        /^FAIL / { print; fflush() }
      '; then
  echo "FAIL optimized-replay - online preflight failed; last output follows" >&2
  tail -80 "${output_dir}/console.log" >&2
  exit 1
fi

run_log="${output_dir}/online-qualified-gpu-${device}.run-1.log"
processing_count="$(grep -c '>>> Processing TS ' "${run_log}" || true)"
[[ "${processing_count}" -eq 1 ]] || {
  echo "FAIL optimized-replay - expected one CBMRoot timeslice reconstruction, found ${processing_count}" >&2
  exit 1
}

metrics_line="$(grep 'KFParticle GPU qualification metrics:' "${run_log}" | tail -1 || true)"
[[ -n "${metrics_line}" ]] || {
  echo "FAIL optimized-replay - missing qualification metrics" >&2
  exit 1
}
timing_line="$(grep 'KFP_TIMING_V2 ' "${run_log}" | tail -1 || true)"
[[ -n "${timing_line}" ]] || {
  echo "FAIL optimized-replay - missing KFP_TIMING_V2 summary" >&2
  exit 1
}
plan_builds="$(printf '%s\n' "${metrics_line}" | field capacity_plan_builds)"
plan_setup_ms="$(printf '%s\n' "${metrics_line}" | field capacity_plan_setup_ms)"
physical_batches="$(printf '%s\n' "${timing_line}" | field batches)"
[[ "${plan_builds}" =~ ^[0-9]+$ && "${physical_batches}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FAIL optimized-replay - invalid capacity-plan/batch counters; rebuild Algo and cbmreco" >&2
  exit 1
}
[[ -n "${plan_setup_ms}" ]] || {
  echo "FAIL optimized-replay - missing one-time capacity-plan setup timing" >&2
  exit 1
}
[[ "${plan_builds}" -eq "${physical_batches}" ]] || {
  echo "FAIL optimized-replay - capacity plans=${plan_builds}, physical batches=${physical_batches}; expected one plan per batch" >&2
  exit 1
}

aggregate_file="${output_dir}/optimized-replay.aggregate.txt"
if ! awk -v samples="${samples}" -f "${aggregate_program}" "${run_log}" >"${aggregate_file}"; then
  exit 1
fi
aggregate_meta="$(grep '^KFP_REPLAY_AGGREGATE_META ' "${aggregate_file}")"
aggregate_batches="$(printf '%s\n' "${aggregate_meta}" | field physical_batches)"
aggregate_records="$(printf '%s\n' "${aggregate_meta}" | field records)"
[[ "${aggregate_batches}" -eq "${physical_batches}" ]] || {
  echo "FAIL optimized-replay - replay records cover ${aggregate_batches} batches, performance report covers ${physical_batches}" >&2
  exit 1
}
[[ "${aggregate_records}" -eq $((physical_batches * samples)) ]] || {
  echo "FAIL optimized-replay - expected $((physical_batches * samples)) batch samples, found ${aggregate_records}" >&2
  exit 1
}
mapfile -t lines < <(grep '^KFP_IN_PROCESS_AGGREGATE ' "${aggregate_file}")
[[ "${#lines[@]}" -eq "${samples}" ]] || {
  echo "FAIL optimized-replay - expected ${samples} aggregate samples, found ${#lines[@]}" >&2
  exit 1
}

declare -a warmed_transaction=()
declare -a warmed_pipeline=()
declare -a warmed_construction=()
declare -a warmed_materialization=()
declare -a warmed_materialization_cpu_work=()
declare -a warmed_packing=()
declare -a warmed_extraction=()
for line in "${lines[@]}"; do
  sample_id="$(printf '%s\n' "${line}" | field sample)"
  if [[ "${sample_id%%/*}" -gt 1 ]]; then
    warmed_transaction+=("$(printf '%s\n' "${line}" | field transaction_ms)")
    warmed_pipeline+=("$(printf '%s\n' "${line}" | field pipeline_ms)")
    warmed_construction+=("$(printf '%s\n' "${line}" | field construction_ms)")
    warmed_materialization+=("$(printf '%s\n' "${line}" | field materialization_ms)")
    warmed_materialization_cpu_work+=("$(printf '%s\n' "${line}" | field materialization_cpu_work_ms)")
    warmed_packing+=("$(printf '%s\n' "${line}" | field packing_ms)")
    warmed_extraction+=("$(printf '%s\n' "${line}" | field extraction_ms)")
  fi
done

echo
echo "Upstream preparation performed once (not part of each replay)"
processed_events="$(printf '%s\n' "${metrics_line}" | field events)"
largest_batch="$(printf '%s\n' "${metrics_line}" | field largest_batch)"
printf 'input_preparation_ms=%s capacity_planning_ms=%s plan_builds=%s one_time_plan_setup_ms=%s events=%s largest_batch=%s\n' \
  "$(printf '%s\n' "${metrics_line}" | field input_preparation_ms)" \
  "$(printf '%s\n' "${metrics_line}" | field capacity_planning_ms)" \
  "${plan_builds}" \
  "${plan_setup_ms}" \
  "${processed_events}" \
  "${largest_batch}"
if [[ "${processed_events}" =~ ^[1-9][0-9]*$ ]] \
   && (( physical_batches * 2 > processed_events )); then
  echo "WARN optimized-replay - severe batch fragmentation: ${physical_batches} physical batches for ${processed_events} GPU events (largest ${largest_batch})" >&2
  echo "WARN optimized-replay - correctness remains testable, but use a larger event/resource limit for comparison with a large-batch baseline" >&2
fi

echo
echo "CPU/host transaction phases (ms)"
printf '%-8s %12s %10s %10s %11s %9s %11s %12s %12s %9s %9s\n' \
  sample transaction queue_wait queue_cap packing pack_wrk extraction mat_wall cpu_work mat_wrk allocations
for line in "${lines[@]}"; do
  printf '%-8s %12s %10s %10s %11s %9s %11s %12s %12s %9s %9s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field transaction_ms)" \
    "$(printf '%s\n' "${line}" | field queue_wait_ms)" \
    "$(printf '%s\n' "${line}" | field queue_capacity_ms)" \
    "$(printf '%s\n' "${line}" | field packing_ms)" \
    "$(printf '%s\n' "${line}" | awk '{
      for(i=1;i<=NF;++i){split($i,v,"=");if(v[1]=="packing_threads_used")u=v[2];if(v[1]=="packing_threads_requested")r=v[2]} print u "/" r}')" \
    "$(printf '%s\n' "${line}" | field extraction_ms)" \
    "$(printf '%s\n' "${line}" | field materialization_ms)" \
    "$(printf '%s\n' "${line}" | field materialization_cpu_work_ms)" \
    "$(printf '%s\n' "${line}" | awk '{
      for(i=1;i<=NF;++i){split($i,v,"=");if(v[1]=="materialization_threads_used")u=v[2];if(v[1]=="materialization_threads_requested")r=v[2]} print u "/" r}')" \
    "$(printf '%s\n' "${line}" | awk '{
      for(i=1;i<=NF;++i){split($i,v,"=");if(v[1]=="initial_allocations")a=v[2];if(v[1]=="replacement_allocations")b=v[2]} print a "/" b}')"
done

echo
echo "GPU pipeline and transfers (host-wall ms)"
printf '%-8s %10s %9s %12s %10s %10s %9s %9s %10s %12s %12s\n' \
  sample pipeline h2d construction graph selection d2h pair_plan host_asm status_B snapshots_B
for line in "${lines[@]}"; do
  printf '%-8s %10s %9s %12s %10s %10s %9s %9s %10s %12s %12s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field pipeline_ms)" \
    "$(printf '%s\n' "${line}" | field h2d_ms)" \
    "$(printf '%s\n' "${line}" | field construction_ms)" \
    "$(printf '%s\n' "${line}" | field graph_ms)" \
    "$(printf '%s\n' "${line}" | field selection_ms)" \
    "$(printf '%s\n' "${line}" | field d2h_ms)" \
    "$(printf '%s\n' "${line}" | field pair_plan_ms)" \
    "$(printf '%s\n' "${line}" | field host_assembly_ms)" \
    "$(printf '%s\n' "${line}" | field status_bytes)" \
    "$(printf '%s\n' "${line}" | field retained_snapshot_bytes)"
done

echo
echo "Nested construction/allocation phases (ms)"
printf '%-8s %11s %12s %11s %11s %11s\n' \
  sample count_read count_bytes dense_plan initial_alloc replace_alloc
for line in "${lines[@]}"; do
  printf '%-8s %11s %12s %11s %11s %11s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field counting_readback_ms)" \
    "$(printf '%s\n' "${line}" | field counting_readback_bytes)" \
    "$(printf '%s\n' "${line}" | field dense_plan_ms)" \
    "$(printf '%s\n' "${line}" | field initial_allocation_ms)" \
    "$(printf '%s\n' "${line}" | field replacement_allocation_ms)"
done

echo
echo "Optimized transaction validation (ms)"
printf '%-8s %12s %12s %12s\n' sample packing initial compact
for line in "${lines[@]}"; do
  printf '%-8s %12s %12s %12s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field validation_packing_ms)" \
    "$(printf '%s\n' "${line}" | field validation_initial_ms)" \
    "$(printf '%s\n' "${line}" | field validation_compact_ms)"
done

host_aggregate="${output_dir}/optimized-replay.host-result.aggregate.txt"
if ! awk -v samples="${samples}" -v batches="${physical_batches}" \
  -v expected_schedule="${materialization_schedule}" \
  -v kind=host -f "${aggregate_program}" "${run_log}" >"${host_aggregate}"; then
  echo "FAIL optimized-replay - invalid host materialization evidence" >&2
  exit 1
fi
mapfile -t host_results <"${host_aggregate}"
echo
echo "Publication input contract: validated once during packing; repeated track sort/validation disabled"
echo "Materialization schedule: ${materialization_schedule}"
echo
echo "CPU result materialization breakdown, summed over physical batches (ms)"
printf '%-8s %11s %11s %11s %11s\n' sample graph leaves candidates commit
for line in "${host_results[@]}"; do
  printf '%-8s %11s %11s %11s %11s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field graph_ms)" \
    "$(printf '%s\n' "${line}" | field leaves_ms)" \
    "$(printf '%s\n' "${line}" | field candidates_ms)" \
    "$(printf '%s\n' "${line}" | field commit_ms)"
done
echo
echo "CPU materialization workspace, summed over physical batches"
printf '%-8s %12s %12s %14s %14s %14s\n' \
  sample initialized reset capacity_B candidate_map track_map
for line in "${host_results[@]}"; do
  printf '%-8s %12s %12s %14s %14s %14s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field scratch_initialized)" \
    "$(printf '%s\n' "${line}" | field scratch_reset)" \
    "$(printf '%s\n' "${line}" | field scratch_capacity_bytes)" \
    "$(printf '%s\n' "${line}" | field candidate_map_capacity_entries)" \
    "$(printf '%s\n' "${line}" | field track_map_capacity_entries)"
done

echo
echo "CPU materialization worker balance, summed over physical batches"
printf '%-8s %10s %10s %11s %11s %11s %11s %11s %11s %12s %12s\n' \
  sample event_min event_max track_min track_max cand_min cand_max est_min est_max cpu_min_ms cpu_max_ms
for line in "${host_results[@]}"; do
  printf '%-8s %10s %10s %11s %11s %11s %11s %11s %11s %12s %12s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field worker_events_min)" \
    "$(printf '%s\n' "${line}" | field worker_events_max)" \
    "$(printf '%s\n' "${line}" | field worker_tracks_min)" \
    "$(printf '%s\n' "${line}" | field worker_tracks_max)" \
    "$(printf '%s\n' "${line}" | field worker_candidates_min)" \
    "$(printf '%s\n' "${line}" | field worker_candidates_max)" \
    "$(printf '%s\n' "${line}" | field worker_estimated_weight_min)" \
    "$(printf '%s\n' "${line}" | field worker_estimated_weight_max)" \
    "$(printf '%s\n' "${line}" | field worker_cpu_min_ms)" \
    "$(printf '%s\n' "${line}" | field worker_cpu_max_ms)"
done

echo
echo "GPU work and transfer volumes"
printf '%-8s %9s %9s %12s %12s %12s\n' sample launches waits real_pairs h2d_bytes d2h_bytes
for line in "${lines[@]}"; do
  printf '%-8s %9s %9s %12s %12s %12s\n' \
    "$(printf '%s\n' "${line}" | field sample)" \
    "$(printf '%s\n' "${line}" | field launches)" \
    "$(printf '%s\n' "${line}" | field queue_waits)" \
    "$(printf '%s\n' "${line}" | field real_pairs)" \
    "$(printf '%s\n' "${line}" | field h2d_bytes)" \
    "$(printf '%s\n' "${line}" | field d2h_bytes)"
done

summarize()
{
  local metric="$1"
  shift
  printf '%s\n' "$@" | sort -n | awk -v metric="${metric}" '
    { value[NR]=$1; sum+=$1 }
    END {
      if (NR % 2) median=value[(NR+1)/2]
      else median=(value[NR/2]+value[NR/2+1])/2
      printf "KFP_IN_PROCESS_WARMED metric=%s samples=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f mean_ms=%.6f\n",
             metric, NR, median, value[1], value[NR], sum/NR
    }'
}

summarize transaction "${warmed_transaction[@]}"
summarize pipeline "${warmed_pipeline[@]}"
summarize construction "${warmed_construction[@]}"
summarize materialization "${warmed_materialization[@]}"
summarize materialization_cpu_work "${warmed_materialization_cpu_work[@]}"
summarize packing "${warmed_packing[@]}"
summarize extraction "${warmed_extraction[@]}"

if [[ "${profile_mode}" == "1" ]]; then
  expected_profiles=$((physical_batches * samples))
  observed_profiles="$(grep -c '^KFP_DEVICE_PROFILE ' "${run_log}" || true)"
  [[ "${observed_profiles}" -eq "${expected_profiles}" ]] || {
    echo "FAIL optimized-replay - expected ${expected_profiles} intrusive device profiles, found ${observed_profiles}" >&2
    exit 1
  }
  echo
  echo "Intrusive PC4.2 XPU profile (synchronized operations; not throughput evidence)"
  grep -E '^KFP_DEVICE_(PROFILE|KERNEL) |^KFP_STEP .*\| +((CPU/)?D2H retained-task counting status|D2H DownloadTransactionStatus \+ wait|KERNEL RouteTwoDaughterBatch|CPU dense output sizes and task tiles|ALLOC dense output capacities)' \
    "${run_log}"
fi

echo "PASS optimized-replay - schedule=${materialization_schedule}; one reconstruction supplied ${samples} aggregate optimized samples across ${physical_batches} physical batches"
echo "identity: stable across every replay of all ${physical_batches} physical batches"
echo "log: ${run_log}"
