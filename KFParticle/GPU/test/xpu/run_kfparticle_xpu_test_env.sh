#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test executable> [args...]" >&2
  exit 2
fi

test_executable="$1"
shift

if [[ ! -x "${test_executable}" ]]; then
  echo "KFParticle GPU test executable is not runnable: ${test_executable}" >&2
  exit 2
fi

executable_dir="$(cd "$(dirname "${test_executable}")" && pwd)"
default_library_dir="$(cd "${executable_dir}/../lib" && pwd)"
library_dir="${KFPARTICLE_GPU_TEST_LIBRARY_DIR:-${default_library_dir}}"
xpu_library_dir="${KFPARTICLE_GPU_XPU_LIBRARY_DIR:-${library_dir}}"
mode="${KFPARTICLE_GPU_TEST_ENV_MODE:-standalone}"
device="${KFPARTICLE_GPU_TEST_DEVICE:-${XPU_DEVICE:-cpu}}"
verbose="${KFPARTICLE_GPU_TEST_XPU_VERBOSE:-${XPU_VERBOSE:-0}}"
path_value="${KFPARTICLE_GPU_TEST_PATH:-${PATH:-/usr/bin:/bin}}"

rocm_root="${XPU_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
rocm_library_dir=""
if [[ -d "${rocm_root}/lib" ]]; then
  rocm_library_dir="${rocm_root}/lib"
fi

append_path()
{
  local current="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    printf '%s' "${current}"
  elif [[ -z "${current}" ]]; then
    printf '%s' "${value}"
  else
    case ":${current}:" in
      *":${value}:"*) printf '%s' "${current}" ;;
      *) printf '%s:%s' "${current}" "${value}" ;;
    esac
  fi
}

controlled_ld_path=""
controlled_ld_path="$(append_path "${controlled_ld_path}" "${library_dir}")"
controlled_ld_path="$(append_path "${controlled_ld_path}" "${xpu_library_dir}")"
controlled_ld_path="$(append_path "${controlled_ld_path}" "${rocm_library_dir}")"
controlled_ld_path="$(append_path "${controlled_ld_path}" "${KFPARTICLE_GPU_TEST_EXTRA_LD_LIBRARY_PATH:-}")"

case "${mode}" in
  standalone|clean)
    if [[ "${KFPARTICLE_GPU_TEST_DIAGNOSTICS:-0}" != "0" ]]; then
      echo "KFParticle GPU XPU test environment"
      echo "  mode       : standalone"
      echo "  executable : ${test_executable}"
      echo "  lib dir    : ${library_dir}"
      echo "  xpu lib dir: ${xpu_library_dir}"
      echo "  device     : ${device}"
      echo "  LD_LIBRARY_PATH=${controlled_ld_path}"
    fi

    # Keep the standalone test hermetic enough to avoid FairSoft/CBMRoot library
    # precedence bugs, but preserve basic process state required by compilers,
    # debuggers, temporary files, and ROCm runtime discovery.
    env -i \
      HOME="${HOME:-}" \
      TMPDIR="${TMPDIR:-/tmp}" \
      PATH="${path_value}" \
      LD_LIBRARY_PATH="${controlled_ld_path}" \
      XPU_DEVICE="${device}" \
      XPU_VERBOSE="${verbose}" \
      XPU_ROCM_ROOT="${XPU_ROCM_ROOT:-${rocm_root}}" \
      ROCM_PATH="${ROCM_PATH:-${rocm_root}}" \
      KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE="${KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE:-0}" \
      KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS="${KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS:-}" \
      KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS="${KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS:-}" \
      KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS="${KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS:-}" \
      "${test_executable}" "$@"
    ;;

  cbmroot|inherited)
    inherited_ld_path="${LD_LIBRARY_PATH:-}"
    run_ld_path="${controlled_ld_path}"
    run_ld_path="$(append_path "${run_ld_path}" "${inherited_ld_path}")"

    if [[ "${KFPARTICLE_GPU_TEST_DIAGNOSTICS:-0}" != "0" ]]; then
      echo "KFParticle GPU XPU test environment"
      echo "  mode       : cbmroot"
      echo "  executable : ${test_executable}"
      echo "  lib dir    : ${library_dir}"
      echo "  xpu lib dir: ${xpu_library_dir}"
      echo "  device     : ${device}"
      echo "  LD_LIBRARY_PATH=${run_ld_path}"
    fi

    LD_LIBRARY_PATH="${run_ld_path}" \
      XPU_DEVICE="${device}" \
      XPU_VERBOSE="${verbose}" \
      "${test_executable}" "$@"
    ;;

  *)
    echo "Unknown KFPARTICLE_GPU_TEST_ENV_MODE='${mode}'. Use standalone or cbmroot." >&2
    exit 2
    ;;
esac
