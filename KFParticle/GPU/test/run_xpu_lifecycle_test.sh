#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/xpu"
repository_dir="$(cd "${script_dir}/../../.." && pwd)"
cbmroot_dir="$(cd "${repository_dir}/../.." && pwd)"
workspace_dir="$(cd "${cbmroot_dir}/.." && pwd)"
build_dir="${KFPARTICLE_GPU_TEST_BUILD_DIR:-${workspace_dir}/build/kfparticle-gpu-xpu-test}"
default_xpu_source_dir="$(cd "${script_dir}/../../../../xpu" && pwd)"
xpu_source_dir="${KFPARTICLE_GPU_XPU_SOURCE_DIR:-${default_xpu_source_dir}}"
xpu_source_dir="$(cd "${xpu_source_dir}" && pwd)"
xpu_selftest_build_dir="${KFPARTICLE_GPU_XPU_SELFTEST_BUILD_DIR:-${build_dir}/xpu-selftest}"
device="${KFPARTICLE_GPU_TEST_DEVICE:-cpu}"
env_mode="${KFPARTICLE_GPU_TEST_ENV_MODE:-standalone}"
diagnostics="${KFPARTICLE_GPU_TEST_DIAGNOSTICS:-0}"
quiet_build="${KFPARTICLE_GPU_TEST_QUIET_BUILD:-1}"
log_dir="${build_dir}/logs"

mkdir -p "${log_dir}"

run_logged()
{
  local label="$1"
  local log_file="$2"
  shift 2

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
}

timestamp_ms()
{
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import time; print(time.time_ns() // 1000000)'
  elif command -v perl >/dev/null 2>&1; then
    perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000'
  else
    echo "$(($(date +%s) * 1000))"
  fi
}

format_elapsed()
{
  local elapsed_ms="$1"
  printf '%d.%03d' "$((elapsed_ms / 1000))" "$((elapsed_ms % 1000))"
}

run_standard_test()
{
  local index="$1"
  local total="$2"
  local name="$3"
  local overview="$4"
  local executable="$5"
  local output_log="${log_dir}/${name}.out"
  local started
  local finished
  local elapsed

  echo "    Start ${index}: ${name}"
  echo "           | ${overview}"
  started="$(timestamp_ms)"
  if env "${common_test_env[@]}" bash "${test_env_script}" "${executable}" >"${output_log}" 2>&1; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "           | ${line}"
    done <"${output_log}"
    finished="$(timestamp_ms)"
    elapsed="$(format_elapsed "$((finished - started))")"
    echo "${index}/${total} Test #${index}: ${name} ..... Passed ${elapsed} sec"
  else
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "           | ${line}"
    done <"${output_log}"
    finished="$(timestamp_ms)"
    elapsed="$(format_elapsed "$((finished - started))")"
    echo "${index}/${total} Test #${index}: ${name} .....***Failed ${elapsed} sec"
    echo "           | log: ${output_log}"
    exit 1
  fi
}

echo "KFParticle GPU XPU lifecycle test"
echo "  build dir  : ${build_dir}"
echo "  xpu dir    : ${xpu_source_dir}"
echo "  device     : ${device}"
echo "  env mode   : ${env_mode}"
echo "  logs       : ${log_dir}"

backend_args=()
cmake_args=(
  "-S" "${source_dir}"
  "-B" "${build_dir}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}"
  "-DKFPARTICLE_GPU_TEST_DEVICE=${device}"
  "-DKFPARTICLE_GPU_TEST_ENV_MODE=${env_mode}"
  "-DKFPARTICLE_GPU_TEST_TRACE=${KFPARTICLE_GPU_TEST_TRACE:-OFF}"
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

if [[ "${KFPARTICLE_GPU_RUN_XPU_SELFTEST:-0}" != "0" ]]; then
  echo "KFParticle GPU XPU official self-test"
  echo "  source dir : ${xpu_source_dir}"
  echo "  build dir  : ${xpu_selftest_build_dir}"
  echo "  device     : ${device}"

  xpu_selftest_args=(
    "-S" "${xpu_source_dir}"
    "-B" "${xpu_selftest_build_dir}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}"
    "-DXPU_BUILD_EXAMPLES=ON"
    "-DXPU_BUILD_TESTS=OFF"
    "-DXPU_BUILD_DOCS=OFF"
  )
  if [[ ${#backend_args[@]} -gt 0 ]]; then
    xpu_selftest_args+=("${backend_args[@]}")
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
      xpu_selftest_args+=("-D${option}=${!option}")
    fi
  done

  run_logged "configure XPU self-test" "${log_dir}/xpu-selftest-configure.log" \
    cmake "${xpu_selftest_args[@]}"
  run_logged "build XPU self-test" "${log_dir}/xpu-selftest-build.log" \
    cmake --build "${xpu_selftest_build_dir}" --target vector_add --parallel "${KFPARTICLE_GPU_TEST_JOBS:-4}"
  export LD_LIBRARY_PATH="${xpu_selftest_build_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  if [[ "${diagnostics}" != "0" ]]; then
    echo "KFParticle GPU XPU official self-test diagnostics"
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo "  relevant environment:"
    env | sort | grep -E '^(CUDA|CMAKE_PREFIX_PATH|FAIR|HIP|HSA|KFPARTICLE|LD_LIBRARY_PATH|PATH|ROCM|ROCR|ROOTSYS|SIMPATH|XPU)=' || true
    echo "  shared libraries:"
    find "${xpu_selftest_build_dir}" -type f \( -name 'lib*.so' -o -name 'lib*.dylib' \) -print | sort
    echo "  generated unity sources:"
    find "${xpu_selftest_build_dir}" -type f -name '*Unity.cpp' -print | sort
    echo "  HIP subproject files:"
    find "${xpu_selftest_build_dir}" -path '*_Hip*' \
      \( -name 'CMakeLists.txt' -o -name 'compile_commands.json' -o -name '*build*.log' -o -name '*configure*.log' \) \
      -print | sort
    while IFS= read -r hip_subproject; do
      if [[ -f "${hip_subproject}/CMakeLists.txt" ]]; then
        echo "  ${hip_subproject}/CMakeLists.txt:"
        sed -n '1,220p' "${hip_subproject}/CMakeLists.txt"
      fi
      while IFS= read -r compile_commands; do
        echo "  ${compile_commands} entries:"
        grep -n "Unity.cpp\\|VectorOps.cpp\\|cuhip_driver.cpp\\|command" "${compile_commands}" || true
      done < <(find "${hip_subproject}" -type f -name compile_commands.json -print | sort)
    done < <(find "${xpu_selftest_build_dir}" -maxdepth 3 -type d -name '*_Hip' -print | sort)
    if command -v ldd >/dev/null 2>&1; then
      for library in \
        "${xpu_selftest_build_dir}/lib/libxpu_Hip.so" \
        "${xpu_selftest_build_dir}/lib/libVectorOps_Hip.so" \
        "${xpu_selftest_build_dir}/lib/libVectorOps.so" \
        "${xpu_selftest_build_dir}/bin/vector_add"
      do
        if [[ -e "${library}" ]]; then
          echo "  ldd ${library}:"
          ldd "${library}" || true
        fi
      done
    fi
  fi

  echo "KFParticle GPU XPU official vector_add run"
  if [[ "${KFPARTICLE_GPU_XPU_SELFTEST_CLEAN_ENV:-0}" != "0" ]]; then
    echo "  mode      : clean environment"
    env -i \
      HOME="${HOME:-}" \
      PATH="${PATH}" \
      LD_LIBRARY_PATH="${xpu_selftest_build_dir}/lib${KFPARTICLE_GPU_XPU_SELFTEST_EXTRA_LD_LIBRARY_PATH:+:${KFPARTICLE_GPU_XPU_SELFTEST_EXTRA_LD_LIBRARY_PATH}}" \
      XPU_DEVICE="${device}" \
      XPU_VERBOSE="${KFPARTICLE_GPU_XPU_SELFTEST_VERBOSE:-1}" \
      "${xpu_selftest_build_dir}/bin/vector_add"
  else
    echo "  mode      : inherited environment"
    XPU_DEVICE="${device}" \
      XPU_VERBOSE="${KFPARTICLE_GPU_XPU_SELFTEST_VERBOSE:-1}" \
      "${xpu_selftest_build_dir}/bin/vector_add"
  fi

  if [[ "${KFPARTICLE_GPU_XPU_SELFTEST_ONLY:-0}" != "0" ]]; then
    exit 0
  fi
fi

run_logged "configure KFParticle GPU XPU tests" "${log_dir}/configure.log" \
  cmake "${cmake_args[@]}"
run_logged "build KFParticle GPU XPU tests" "${log_dir}/build.log" \
  cmake --build "${build_dir}" --parallel "${KFPARTICLE_GPU_TEST_JOBS:-4}"

if [[ "${diagnostics}" != "0" ]]; then
  echo "KFParticle GPU XPU diagnostics"
  echo "  inherited LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "  controlled test library dir=${build_dir}/lib"
  echo "  shared libraries:"
  find "${build_dir}" -type f \( -name 'lib*.so' -o -name 'lib*.dylib' \) -print | sort
  echo "  generated unity sources:"
  find "${build_dir}" -type f -name '*Unity.cpp' -print | sort
  echo "  HIP subproject files:"
  find "${build_dir}" -path '*_Hip*' \
    \( -name 'CMakeLists.txt' -o -name 'compile_commands.json' -o -name '*build*.log' -o -name '*configure*.log' \) \
    -print | sort
  while IFS= read -r hip_subproject; do
    if [[ -f "${hip_subproject}/CMakeLists.txt" ]]; then
      echo "  ${hip_subproject}/CMakeLists.txt:"
      sed -n '1,220p' "${hip_subproject}/CMakeLists.txt"
    fi
    while IFS= read -r compile_commands; do
      echo "  ${compile_commands} entries:"
      grep -n "Unity.cpp\\|command" "${compile_commands}" || true
    done < <(find "${hip_subproject}" -type f -name compile_commands.json -print | sort)
  done < <(find "${build_dir}" -maxdepth 3 -type d -name '*_Hip' -print | sort)
  if command -v ldd >/dev/null 2>&1; then
    for library in \
      "${build_dir}/lib/libxpu_Hip.so" \
      "${build_dir}/lib/libKFParticleGpuXpuTestSupport_Hip.so" \
      "${build_dir}/lib/libKFParticleGpuXpuTestSupport.so" \
      "${build_dir}/bin/KFParticleGpuXpuLifecycleTest"
    do
      if [[ -e "${library}" ]]; then
        echo "  ldd ${library}:"
        ldd "${library}" || true
      fi
    done
  fi
fi

if [[ "${KFPARTICLE_GPU_TEST_USE_CTEST:-0}" != "0" ]]; then
  ctest --test-dir "${build_dir}" --output-on-failure
else
  test_env_script="${source_dir}/run_kfparticle_xpu_test_env.sh"
  common_test_env=(
    "KFPARTICLE_GPU_TEST_LIBRARY_DIR=${build_dir}/lib"
    "KFPARTICLE_GPU_XPU_LIBRARY_DIR=${build_dir}/lib"
    "KFPARTICLE_GPU_TEST_DEVICE=${device}"
    "KFPARTICLE_GPU_TEST_ENV_MODE=${env_mode}"
    "KFPARTICLE_GPU_TEST_DIAGNOSTICS=${diagnostics}"
    "KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE=${KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE:-0}"
    "KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS=${KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS:-}"
  )

  echo "Test project ${build_dir}"
  run_standard_test \
    1 2 \
    "KFParticleGpuXpuBaselineTest" \
    "minimal XPU image preload, queue launch, and marker readback" \
    "${build_dir}/bin/KFParticleGpuXpuBaselineTest"
  run_standard_test \
    2 2 \
    "KFParticleGpuXpuLifecycleTest" \
    "runtime, KFParticle image kernels, SoA buffers, upload/download, candidate pools" \
    "${build_dir}/bin/KFParticleGpuXpuLifecycleTest"
fi
