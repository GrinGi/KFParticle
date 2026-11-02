#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-lifecycle}"
run_only=0
case "$mode" in build|lifecycle|benchmark|matrix|ctest) ;; *)
  echo "Usage: $0 [build|lifecycle|benchmark|matrix|ctest] [--run-only]" >&2; exit 2 ;;
esac
if [[ "${2:-}" == --run-only ]]; then run_only=1
elif [[ $# -gt 1 ]]; then echo "Unknown option: $2" >&2; exit 2; fi
[[ $# -le 2 && ! ( "$mode" == build && "$run_only" == 1 ) ]] || exit 2
test_source="${script_dir}/KFParticleGpuXpuLifecycleTest.cxx"
source_dir="${script_dir}/xpu"
repository_dir="$(cd "${script_dir}/../../.." && pwd)"
cbmroot_dir="$(cd "${repository_dir}/../.." && pwd)"
workspace_dir="$(cd "${cbmroot_dir}/.." && pwd)"
build_dir="${KFPARTICLE_GPU_TEST_BUILD_DIR:-${workspace_dir}/build/kfparticle-gpu-xpu-test}"
xpu_source_dir="${KFPARTICLE_GPU_XPU_SOURCE_DIR:-${script_dir}/../../../../xpu}"
xpu_source_dir="$(cd "${xpu_source_dir}" && pwd)"
device="${KFPARTICLE_GPU_TEST_DEVICE:-cpu}"
env_mode="${KFPARTICLE_GPU_TEST_ENV_MODE:-standalone}"
diagnostics="${KFPARTICLE_GPU_TEST_DIAGNOSTICS:-0}"
quiet_build="${KFPARTICLE_GPU_TEST_QUIET_BUILD:-1}"
log_dir="${build_dir}/logs"

source_id()
{
  if command -v sha256sum >/dev/null 2>&1; then
    cat "$@" | sha256sum | awk '{print substr($1, 1, 12)}'
  elif command -v shasum >/dev/null 2>&1; then
    cat "$@" | shasum -a 256 | awk '{print substr($1, 1, 12)}'
  else
    echo unavailable
  fi
}
test_source_id="$(source_id "$test_source" "${script_dir}"/lifecycle/*.inc)"
# CMake's timestamp dependency checks are insufficient when source files are
# copied from another filesystem with preserved or older mtimes. Carry the
# content identity into the compiler command so changed sources necessarily
# rebuild both the host support library and its test executables.
source_fingerprint="${test_source_id}-$(source_id "${script_dir}/KFParticleGpuXpuBatchBenchmark.cxx")-$(
  shopt -s nullglob
  sources=("${script_dir}/../KFParticleGpuPlatform.h"
           "${script_dir}/../KFParticleGpuBuildOptions.h.in")
  for directory in data planning execution algorithms diagnostics; do
    sources+=("${script_dir}/../${directory}/"*.h
              "${script_dir}/../${directory}/"*.cxx)
  done
  source_id "${sources[@]}"
)"

if [[ "$mode" == matrix ]]; then
  sizes="${KFPARTICLE_GPU_BATCH_BENCHMARK_SIZES-1 2 4 8}"
  samples="${KFPARTICLE_GPU_BATCH_BENCHMARK_SAMPLES-5}"
  iterations="${KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS-50}"
  warmup="${KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS-3}"
  for value in $sizes "$samples" "$iterations" "$warmup"; do
    [[ "$value" =~ ^[1-9][0-9]{0,3}$ ]] && ((10#$value <= 1024)) || {
      echo "FAIL matrix counts must be integers in 1..1024: $value" >&2; exit 2;
    }
  done
  [[ -n "${sizes//[[:space:]]/}" ]] || { echo "FAIL empty matrix sizes" >&2; exit 2; }
  evidence="${log_dir}/kfparticle-gpu-batch-benchmark-${device}.txt"
  mkdir -p "$log_dir"
  rm -f "$evidence"
  trap 'rm -f "${evidence}.tmp"' EXIT
fi

mkdir -p "${log_dir}"

run_logged()
{
  local label="$1"
  local log_file="$2"
  shift 2

  echo "  START ${label}"
  echo "        log: ${log_file}"
  if [[ "${quiet_build}" != "0" && "${diagnostics}" == "0" ]]; then
    if ! "$@" >"${log_file}" 2>&1; then
      echo "FAIL ${label}"
      echo "  log: ${log_file}"
      tail -n 80 "${log_file}" || true
      exit 1
    fi
  else
    "$@"
  fi
  echo "  PASS  ${label}"
}

echo "KFParticle GPU XPU tests: ${mode}"
echo "  build dir  : ${build_dir}"
echo "  xpu dir    : ${xpu_source_dir}"
echo "  device     : ${device}"
echo "  env mode   : ${env_mode}"
echo "  logs       : ${log_dir}"
echo "  lifecycle source id : ${test_source_id}"
echo "  source fingerprint: ${source_fingerprint}"

backend_args=()
cmake_args=(
  "-S" "${source_dir}"
  "-B" "${build_dir}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}"
  "-DKFPARTICLE_GPU_TEST_DEVICE=${device}"
  "-DKFPARTICLE_GPU_TEST_ENV_MODE=${env_mode}"
  "-DKFPARTICLE_GPU_SOURCE_FINGERPRINT=${source_fingerprint}"
  "-DXPU_SOURCE_DIR=${xpu_source_dir}"
)

case "${device}" in
  cuda* )
    backend_args+=("-DXPU_ENABLE_CUDA=ON")
    ;;
  hip* )
    backend_args+=("-DXPU_ENABLE_HIP=ON")
    ;;
  sycl* )
    backend_args+=("-DXPU_ENABLE_SYCL=ON")
    ;;
  cpu )
    ;;
  * )
    echo "Unknown XPU device '${device}'. Use cpu, cudaN, hipN, or syclN." >&2
    exit 2
    ;;
esac
if [[ ${#backend_args[@]} -gt 0 ]]; then
  cmake_args+=("${backend_args[@]}")
fi

for option in \
  XPU_ENABLE_OPENMP \
  XPU_CUDA_ARCH \
  XPU_HIP_ARCH \
  XPU_ROCM_ROOT \
  XPU_SYCL_CXX \
  XPU_SYCL_TARGETS \
  XPU_DEBUG
do
  if [[ -n "${!option:-}" ]]; then
    cmake_args+=("-D${option}=${!option}")
  fi
done

if [[ "$run_only" == 0 ]]; then
  run_logged "configure KFParticle GPU XPU tests" "${log_dir}/configure.log" \
    cmake "${cmake_args[@]}"
  run_logged "build KFParticle GPU XPU tests" "${log_dir}/build.log" \
    cmake --build "${build_dir}" --parallel "${KFPARTICLE_GPU_TEST_JOBS:-4}"
else
  echo "  build      : skipped (explicit run-only)"
  cache="${build_dir}/CMakeCache.txt"
  [[ -r "$cache" ]] || { echo "FAIL missing CMake cache: $cache" >&2; exit 2; }
  for setting in KFPARTICLE_GPU_TEST_DEVICE KFPARTICLE_GPU_TEST_ENV_MODE; do
    configured="$(sed -n "s/^${setting}:[^=]*=//p" "$cache")"
    expected="$device"
    [[ "$setting" != KFPARTICLE_GPU_TEST_ENV_MODE ]] || expected="$env_mode"
    [[ "$configured" == "$expected" ]] || {
      echo "FAIL $setting requested=$expected configured=$configured; configure/build this directory first" >&2
      exit 2
    }
  done
fi

if [[ "$mode" == build ]]; then
  echo "PASS build-only KFParticle GPU XPU tests"
  exit 0
fi

ctest_args=(--test-dir "${build_dir}" --output-on-failure --no-tests=error --parallel 1)
if [[ "$mode" == lifecycle || "$mode" == benchmark ]]; then
  ctest_args+=(-R '^KFParticleGpuXpuLifecycleTest$')
fi
if [[ "${diagnostics}" != 0 ]]; then ctest_args+=(--verbose); fi
ctest "${ctest_args[@]}"

common_test_env=(
  "KFPARTICLE_GPU_TEST_LIBRARY_DIR=${build_dir}/lib"
  "KFPARTICLE_GPU_XPU_LIBRARY_DIR=${build_dir}/lib"
  "KFPARTICLE_GPU_TEST_DEVICE=${device}"
  "KFPARTICLE_GPU_TEST_ENV_MODE=${env_mode}"
  "KFPARTICLE_GPU_TEST_DIAGNOSTICS=${diagnostics}"
)
if [[ "$mode" == benchmark ]]; then
  env "${common_test_env[@]}" bash "${source_dir}/run_kfparticle_xpu_test_env.sh" \
    "${build_dir}/bin/KFParticleGpuXpuBatchBenchmark" \
    2>&1 | tee "${log_dir}/KFParticleGpuXpuBatchBenchmark.out"
fi

if [[ "$mode" == matrix ]]; then
  echo "Matrix: sizes=$sizes samples=$samples iterations=$iterations warmup=$warmup"
  printf 'format_version=3\nsource_fingerprint=%s\ndevice=%s\niterations=%s\nwarmup=%s\n' \
    "$source_fingerprint" "$device" "$iterations" "$warmup" >"${evidence}.tmp"
  for events in $sizes; do
    for ((sample=1; sample<=samples; ++sample)); do
      output="${log_dir}/KFParticleGpuXpuBatchBenchmark.events-${events}.sample-${sample}.out"
      echo "  events=$events sample=$sample/$samples"
      if ! env KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS="$events" \
          KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS="$iterations" \
          KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS="$warmup" \
          "${common_test_env[@]}" \
          bash "${source_dir}/run_kfparticle_xpu_test_env.sh" \
          "${build_dir}/bin/KFParticleGpuXpuBatchBenchmark" >"$output" 2>&1; then
        cat "$output" >&2; exit 1
      fi
      grep -q "^PASS kfparticle-gpu-batch-benchmark events=$events " "$output" || {
        cat "$output" >&2; echo "FAIL missing benchmark success marker" >&2; exit 1;
      }
      # Validate both modes before publishing this sample; never reuse old metrics.
      awk -v events="$events" -v sample="$sample" '
        $1 == "METRIC" {
          mode=wall=n=""; for(i=2;i<=NF;i++) {
            split($i,p,"="); if(p[1]=="mode") mode=p[2];
            if(p[1]=="wall_ms") wall=p[2]; if(p[1]=="events") n=p[2];
          }
          if ((mode!="serial" && mode!="batch") || seen[mode]++ || n!=events ||
              wall !~ /^[0-9]+([.][0-9]+)?$/ || wall+0<=0) exit 1;
          print $0, "sample=" sample;
        }
        END {if(seen["serial"]!=1 || seen["batch"]!=1) exit 1}
      ' "$output" | tee -a "${evidence}.tmp"
    done
  done
  mv "${evidence}.tmp" "$evidence"
  echo "PASS kfparticle-gpu-batch-benchmark-matrix - evidence: $evidence"
fi
