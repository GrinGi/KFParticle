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
log_dir="${build_dir}/logs"
evidence="${log_dir}/kfparticle-gpu-batch-benchmark-${device}.txt"
validation_log="${log_dir}/KFParticleGpuXpuBatchBenchmark.validation.out"
test_env_script="${script_dir}/xpu/run_kfparticle_xpu_test_env.sh"
benchmark="${build_dir}/bin/KFParticleGpuXpuBatchBenchmark"

mkdir -p "${log_dir}"
{
  echo "KFParticle GPU XPU batch benchmark evidence"
  echo "  device     : ${device}"
  echo "  sizes      : ${sizes}"
  echo "  iterations : ${iterations}"
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

  run_log="${log_dir}/KFParticleGpuXpuBatchBenchmark.events-${events}.out"
  echo "KFParticle GPU XPU batch benchmark: events=${events}"
  if KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS="${events}" \
      KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS="${iterations}" \
      KFPARTICLE_GPU_TEST_LIBRARY_DIR="${build_dir}/lib" \
      KFPARTICLE_GPU_XPU_LIBRARY_DIR="${build_dir}/lib" \
      KFPARTICLE_GPU_TEST_DEVICE="${device}" \
      KFPARTICLE_GPU_TEST_ENV_MODE="${KFPARTICLE_GPU_TEST_ENV_MODE:-standalone}" \
      bash "${test_env_script}" "${benchmark}" >"${run_log}" 2>&1; then
    metrics="$(grep -E '^(PASS kfparticle-gpu-batch-benchmark|METRIC )' "${run_log}")"
    if [[ -z "${metrics}" ]]; then
      echo "FAIL batch-benchmark events=${events} - benchmark metrics were not reported" >&2
      echo "  log: ${run_log}" >&2
      exit 1
    fi
    printf '%s\n' "${metrics}" | tee -a "${evidence}"
  else
    echo "FAIL batch-benchmark events=${events}" >&2
    echo "  log: ${run_log}" >&2
    tail -n 80 "${run_log}" >&2 || true
    exit 1
  fi
done

echo "PASS kfparticle-gpu-batch-benchmark-matrix - evidence: ${evidence}"
