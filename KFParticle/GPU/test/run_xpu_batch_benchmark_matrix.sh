#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${script_dir}/../../.." && pwd)"
cbmroot_dir="$(cd "${repository_dir}/../.." && pwd)"
workspace_dir="$(cd "${cbmroot_dir}/.." && pwd)"
build_dir="${KFPARTICLE_GPU_TEST_BUILD_DIR:-${workspace_dir}/build/kfparticle-gpu-xpu-test}"
device="${KFPARTICLE_GPU_TEST_DEVICE:-cpu}"
sizes="${KFPARTICLE_GPU_BATCH_BENCHMARK_SIZES:-1 2 4 8}"
iterations="${KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS:-50}"
warmup_iterations="${KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS:-3}"
samples="${KFPARTICLE_GPU_BATCH_BENCHMARK_SAMPLES:-5}"
log_dir="${build_dir}/logs"
evidence="${log_dir}/kfparticle-gpu-batch-benchmark-${device}.txt"
validation_log="${log_dir}/KFParticleGpuXpuBatchBenchmark.validation.out"
test_env_script="${script_dir}/xpu/run_kfparticle_xpu_test_env.sh"
benchmark="${build_dir}/bin/KFParticleGpuXpuBatchBenchmark"

for value_name in iterations warmup_iterations samples; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
done

mkdir -p "${log_dir}"
source_revision="$(git -C "${repository_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
build_mode="$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "${build_dir}/CMakeCache.txt" 2>/dev/null | head -1)"
build_mode="${build_mode:-unknown}"
backend="${device%%[0-9]*}"
workload_id="$(printf '%s' "sizes=${sizes};iterations=${iterations};warmup=${warmup_iterations};samples=${samples}" | cksum | awk '{print $1}')"
{
  echo "KFParticle GPU XPU batch benchmark evidence"
  echo "format_version=2"
  echo "source_revision=${source_revision}"
  echo "build_mode=${build_mode}"
  echo "  device     : ${device}"
  echo "backend=${backend}"
  echo "  sizes      : ${sizes}"
  echo "  iterations : ${iterations}"
  echo "warmup_iterations=${warmup_iterations}"
  echo "samples=${samples}"
  echo "workload_id=${workload_id}"
  echo "command=${0}"
} >"${evidence}"

# Configure/build and run the complete fast regression suite once.  Individual
# matrix entries below execute only the already built benchmark binary.
if [[ "${KFPARTICLE_GPU_BATCH_BENCHMARK_SKIP_VALIDATION:-0}" == "0" ]]; then
  if ! bash "${script_dir}/run_xpu_lifecycle_test.sh" >"${validation_log}" 2>&1; then
    echo "FAIL batch-benchmark validation" >&2
    echo "  log: ${validation_log}" >&2
    tail -n 80 "${validation_log}" >&2 || true
    exit 1
  fi
fi
if [[ ! -x "${benchmark}" ]]; then
  echo "KFParticle GPU batch benchmark is not runnable: ${benchmark}" >&2
  exit 2
fi

for events in ${sizes}; do
  case "${events}" in
    ''|*[!0-9]*|0)
      echo "Invalid positive batch size '${events}'." >&2
      exit 2
      ;;
  esac

  metrics_file="${log_dir}/KFParticleGpuXpuBatchBenchmark.events-${events}.metrics"
  : >"${metrics_file}"
  echo "KFParticle GPU XPU batch benchmark: events=${events}"
  for sample in $(seq 1 "${samples}"); do
    run_log="${log_dir}/KFParticleGpuXpuBatchBenchmark.events-${events}.sample-${sample}.out"
    if KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS="${events}" \
        KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS="${iterations}" \
        KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS="${warmup_iterations}" \
        KFPARTICLE_GPU_TEST_LIBRARY_DIR="${build_dir}/lib" \
        KFPARTICLE_GPU_XPU_LIBRARY_DIR="${build_dir}/lib" \
        KFPARTICLE_GPU_TEST_DEVICE="${device}" \
        KFPARTICLE_GPU_TEST_ENV_MODE="${KFPARTICLE_GPU_TEST_ENV_MODE:-standalone}" \
        bash "${test_env_script}" "${benchmark}" >"${run_log}" 2>&1; then
      metrics="$(grep -E '^(PASS kfparticle-gpu-batch-benchmark|METRIC |ROUTING_METRIC |PERFORMANCE_METRIC )' "${run_log}")"
      if [[ -z "${metrics}" ]]; then
        echo "FAIL batch-benchmark events=${events} sample=${sample} - metrics were not reported" >&2
        echo "  log: ${run_log}" >&2
        exit 1
      fi
      while IFS= read -r metric; do
        printf '%s events=%s sample=%s workload_id=%s\n' \
          "${metric}" "${events}" "${sample}" "${workload_id}" | tee -a "${evidence}" "${metrics_file}"
      done <<<"${metrics}"
    else
      echo "FAIL batch-benchmark events=${events} sample=${sample}" >&2
      echo "  log: ${run_log}" >&2
      tail -n 80 "${run_log}" >&2 || true
      exit 1
    fi
  done

  for mode in serial batch full-field-approx-batch constant-by-batch cascade-serial cascade-batch; do
    aggregate="$(awk -v wanted="${mode}" '
      $1 == "METRIC" {
        mode=""; wall="";
        for (i=2; i<=NF; ++i) {
          split($i, pair, "=");
          if (pair[1] == "mode") mode=pair[2];
          if (pair[1] == "wall_ms") wall=pair[2];
        }
        if (mode == wanted && wall != "") print wall;
      }' "${metrics_file}" | sort -n | awk '
      { value[NR]=$1 }
      END {
        if (NR == 0) exit 1;
        if (NR % 2) median=value[(NR+1)/2];
        else median=(value[NR/2]+value[NR/2+1])/2;
        tailIndex=int((95*NR+99)/100);
        if (tailIndex < 1) tailIndex=1;
        printf "samples=%d median_wall_ms=%.6f p95_wall_ms=%.6f", NR, median, value[tailIndex];
      }')"
    printf 'AGGREGATE events=%s mode=%s %s workload_id=%s\n' \
      "${events}" "${mode}" "${aggregate}" "${workload_id}" | tee -a "${evidence}"
  done
done

echo "PASS kfparticle-gpu-batch-benchmark-matrix - evidence: ${evidence}"
