#!/usr/bin/env bash

# Shared environment contract for the online cbmreco and offline FairRunAna
# launchers. The mode wrappers own their framework-specific arguments.

kfp_preflight_fail()
{
  local label="$1"
  shift
  echo "FAIL ${label} - $*" >&2
  exit 2
}

kfp_preflight_cache_value()
{
  local key="$1"
  local cache="${KFP_PREFLIGHT_BUILD_DIR}/CMakeCache.txt"
  sed -n "s/^${key}:[^=]*=//p" "${cache}" | head -1
}

kfp_preflight_require_cache_on()
{
  local key="$1"
  local value
  value="$(kfp_preflight_cache_value "${key}")"
  if [[ "${value}" != "ON" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "${key}=ON is required in ${KFP_PREFLIGHT_BUILD_DIR}/CMakeCache.txt (found: ${value:-unset})"
  fi
}

kfp_preflight_check_qualification_build()
{
  local library="$1"
  local qualification_expectation="$2"
  local cache_value expected_build expected_marker
  cache_value="$(kfp_preflight_cache_value CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL)"
  case "${cache_value}" in
    ON)
      KFP_PREFLIGHT_QUALIFICATION_BUILD="trial"
      expected_marker="kfp_gpu_qualification_build=trial-v1"
      ;;
    OFF)
      KFP_PREFLIGHT_QUALIFICATION_BUILD="ordinary"
      expected_marker="kfp_gpu_qualification_build=ordinary-v1"
      ;;
    *)
      kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
        "CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL must be explicitly ON or OFF in ${KFP_PREFLIGHT_BUILD_DIR}/CMakeCache.txt (found: ${cache_value:-unset})"
      ;;
  esac

  if ! grep -a -q "${expected_marker}" "${library}"; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "$(basename "${library}") does not match CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=${cache_value}; rebuild the selected build directory"
  fi

  expected_build="${KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD:-any}"
  case "${expected_build}" in any|ordinary|trial) ;; *)
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "KFPARTICLE_CBMROOT_EXPECT_QUALIFICATION_BUILD must be any, ordinary, or trial"
  esac
  if [[ "${qualification_expectation}" == "published" ]]; then
    expected_build="trial"
  fi
  if [[ "${expected_build}" != "any" && "${KFP_PREFLIGHT_QUALIFICATION_BUILD}" != "${expected_build}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "qualification build mismatch: expected ${expected_build}, found ${KFP_PREFLIGHT_QUALIFICATION_BUILD}"
  fi
}

kfp_preflight_require_file()
{
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "missing ${description}: ${path}"
  fi
}

kfp_preflight_require_executable()
{
  local path="$1"
  local description="$2"
  if [[ ! -x "${path}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "missing ${description}: ${path}"
  fi
}

kfp_preflight_contains_word()
{
  local words="$1"
  local expected="$2"
  local word
  for word in ${words}; do
    if [[ "${word}" == "${expected}" ]]; then
      return 0
    fi
  done
  return 1
}

kfp_preflight_file_identity()
{
  local path="$1"
  local label="$2"
  local size
  size="$(wc -c <"${path}" | tr -d '[:space:]')"
  echo "  ${label} : ${path}"
  echo "  ${label} size : ${size} bytes"
  if [[ "${KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_HASH:-0}" == "1" ]]; then
    return
  elif command -v sha256sum >/dev/null 2>&1; then
    echo "  ${label} sha256 : $(sha256sum "${path}" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    echo "  ${label} sha256 : $(shasum -a 256 "${path}" | awk '{print $1}')"
  fi
}

kfp_preflight_path_identity()
{
  local path="$1"
  local identity="${path}"
  if [[ -f "${path}" && "${KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_HASH:-0}" != "1" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      identity+="; sha256=$(sha256sum "${path}" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      identity+="; sha256=$(shasum -a 256 "${path}" | awk '{print $1}')"
    fi
  fi
  printf '%s\n' "${identity}"
}

kfp_preflight_input_identity()
{
  local locator="$1"
  if [[ -f "${locator}" ]]; then
    kfp_preflight_file_identity "${locator}" "input"
  elif [[ -d "${locator}" ]]; then
    echo "  input : ${locator}"
    echo "  input kind : directory"
  elif [[ "${locator}" == *"://"* || "${locator}" == *":"* ]]; then
    echo "  input : ${locator}"
    echo "  input kind : non-local locator; existence is checked by the reconstruction source"
  else
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "input does not exist and is not a URI: ${locator}"
  fi
}

kfp_preflight_check_linked_xpu()
{
  local executable="$1"
  if [[ "${KFPARTICLE_CBMROOT_PREFLIGHT_SKIP_LINK_CHECK:-0}" == "1" ]] || ! command -v ldd >/dev/null 2>&1; then
    return
  fi

  local resolved
  resolved="$(LD_LIBRARY_PATH="${KFP_PREFLIGHT_LD_LIBRARY_PATH}" ldd "${executable}" 2>/dev/null \
    | awk '$1 == "libxpu.so" && $2 == "=>" {print $3; exit}')"
  if [[ -z "${resolved}" ]]; then
    echo "  linked xpu : loaded dynamically; build/lib is first in LD_LIBRARY_PATH"
    return
  fi

  local expected="${KFP_PREFLIGHT_LIBRARY_DIR}/libxpu.so"
  local resolved_real expected_real
  resolved_real="$(readlink -f "${resolved}")"
  expected_real="$(readlink -f "${expected}")"
  if [[ "${resolved_real}" != "${expected_real}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "wrong libxpu.so selected: ${resolved_real}; expected ${expected_real}"
  fi
  echo "  linked xpu : ${resolved_real}"
}

kfp_preflight_quote_command()
{
  printf -v KFP_PREFLIGHT_COMMAND_TEXT '%q ' "$@"
  KFP_PREFLIGHT_COMMAND_TEXT="${KFP_PREFLIGHT_COMMAND_TEXT% }"
}

kfp_preflight_now_milliseconds()
{
  local nanoseconds
  nanoseconds="$(date +%s%N)"
  if [[ "${nanoseconds}" =~ ^[0-9]+$ ]]; then
    echo $((nanoseconds / 1000000))
  else
    echo "$(($(date +%s) * 1000))"
  fi
}

kfp_preflight_write_evidence()
{
  local status="$1"
  {
    echo "KFParticle CBMRoot ${KFP_PREFLIGHT_MODE} preflight evidence"
    echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host: $(hostname)"
    echo "status: ${status}"
    echo "source: ${KFP_PREFLIGHT_SOURCE_DIR}"
    echo "source revision: $(git -C "${KFP_PREFLIGHT_SOURCE_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "build: ${KFP_PREFLIGHT_BUILD_DIR}"
    echo "build mode: $(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "${KFP_PREFLIGHT_BUILD_DIR}/CMakeCache.txt" 2>/dev/null | head -1)"
    echo "qualification build: ${KFP_PREFLIGHT_QUALIFICATION_BUILD:-unchecked}"
    echo "device: ${KFP_PREFLIGHT_DEVICE}"
    echo "XPU HIP architecture: ${KFP_PREFLIGHT_HIP_ARCH:-unset}"
    echo "ROCm: ${KFP_PREFLIGHT_ROCM_VERSION:-${KFP_PREFLIGHT_ROCM_ROOT:-unset}}"
    echo "FairRoot: ${KFP_PREFLIGHT_FAIRROOT:-unset}"
    echo "ROOT: ${KFP_PREFLIGHT_ROOT_VERSION:-unset}"
    echo "input: ${KFP_PREFLIGHT_INPUT_IDENTITY:-unset}"
    echo "input kind: ${KFP_PREFLIGHT_INPUT_KIND:-unset}"
    echo "configuration: ${KFP_PREFLIGHT_CONFIGURATION_IDENTITY:-unset}"
    echo "geometry: ${KFP_PREFLIGHT_GEOMETRY_IDENTITY:-unset}"
    echo "parameters: ${KFP_PREFLIGHT_PARAMETER_IDENTITY:-unset}"
    echo "steps: ${KFP_PREFLIGHT_STEPS_IDENTITY:-unset}"
    echo "command: ${KFP_PREFLIGHT_COMMAND_TEXT:-not executed}"
    echo "result: ${KFP_PREFLIGHT_RESULT_IDENTITY:-unset}"
  } >"${KFP_PREFLIGHT_EVIDENCE_FILE}"
}

kfp_preflight_print_performance_report()
{
  local log_file="$1"
  awk '
    /KFParticle GPU performance report \(host wall time; monitoring enabled\)/ {
      printing = 1
    }
    printing { print }
    printing && /Note[[:space:]]+GPU\/QUEUE values are host wall intervals/ {
      printing = 0
    }
  ' "${log_file}"
}

kfp_preflight_append_performance_evidence()
{
  local log_file
  {
    echo "performance monitoring: ${KFPARTICLE_GPU_PERFORMANCE_MONITORING:-0}"
    for log_file in "$@"; do
      [[ -f "${log_file}" ]] || continue
      grep -E '^(PERFORMANCE_RUN|.*KFParticle GPU qualification metrics:|.*KFParticle GPU performance:|.*KFParticle GPU diagnostics:|.*KFParticle GPU routing monitoring:|.*KFP selector summary:|.*KFP online summary:)' \
        "${log_file}" || true
      kfp_preflight_print_performance_report "${log_file}"
    done
  } >>"${KFP_PREFLIGHT_EVIDENCE_FILE}"
}

kfp_preflight_runtime_smoke()
{
  if [[ "${KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE:-1}" != "1" ]]; then
    echo "  runtime smoke : skipped by KFPARTICLE_CBMROOT_PREFLIGHT_RUNTIME_SMOKE"
    return
  fi

  local smoke="${KFP_PREFLIGHT_SCRIPT_DIR}/run_cbmroot_xpu_smoke.sh"
  local smoke_scope="${KFP_PREFLIGHT_MODE}"
  if [[ -n "${KFP_PREFLIGHT_INPUT_KIND:-}" ]]; then
    smoke_scope+="-${KFP_PREFLIGHT_INPUT_KIND}"
  fi
  kfp_preflight_require_file "${smoke}" "embedded XPU smoke launcher"
  echo "  runtime smoke : ${smoke}"
  KFPARTICLE_CBMROOT_SOURCE_DIR="${KFP_PREFLIGHT_SOURCE_DIR}" \
  KFPARTICLE_CBMROOT_BUILD_DIR="${KFP_PREFLIGHT_BUILD_DIR}" \
  KFPARTICLE_CBMROOT_DEVICE="${KFP_PREFLIGHT_DEVICE}" \
  KFPARTICLE_CBMROOT_SMOKE_LOG_DIR="${KFP_PREFLIGHT_LOG_DIR}/${smoke_scope}" \
    bash "${smoke}"
}

kfp_preflight_initialize()
{
  KFP_PREFLIGHT_MODE="$1"
  KFP_PREFLIGHT_LABEL="cbmroot-kfp-${KFP_PREFLIGHT_MODE}-preflight"
  KFP_PREFLIGHT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  KFP_PREFLIGHT_SOURCE_DIR="${KFPARTICLE_CBMROOT_SOURCE_DIR:-${VMCWORKDIR:-}}"
  if [[ -z "${KFP_PREFLIGHT_SOURCE_DIR}" ]]; then
    KFP_PREFLIGHT_SOURCE_DIR="$(cd "${KFP_PREFLIGHT_SCRIPT_DIR}/../../../../../../" && pwd)"
  fi
  KFP_PREFLIGHT_BUILD_DIR="${KFPARTICLE_CBMROOT_BUILD_DIR:-$(cd "${KFP_PREFLIGHT_SOURCE_DIR}/.." && pwd)/build}"
  KFP_PREFLIGHT_LIBRARY_DIR="${KFP_PREFLIGHT_BUILD_DIR}/lib"
  KFP_PREFLIGHT_DEVICE="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
  KFP_PREFLIGHT_LOG_DIR="${KFPARTICLE_CBMROOT_PREFLIGHT_LOG_DIR:-${KFP_PREFLIGHT_BUILD_DIR}/kfparticle-cbmroot-preflight}"
  KFP_PREFLIGHT_EVIDENCE_FILE="${KFPARTICLE_CBMROOT_PREFLIGHT_EVIDENCE_FILE:-${KFP_PREFLIGHT_LOG_DIR}/${KFP_PREFLIGHT_MODE}-${KFP_PREFLIGHT_DEVICE}.evidence.txt}"

  if [[ ! -d "${KFP_PREFLIGHT_SOURCE_DIR}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "missing CBMRoot source directory: ${KFP_PREFLIGHT_SOURCE_DIR}"
  fi
  if [[ ! -d "${KFP_PREFLIGHT_BUILD_DIR}" ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" "missing CBMRoot build directory: ${KFP_PREFLIGHT_BUILD_DIR}"
  fi
  kfp_preflight_require_file "${KFP_PREFLIGHT_SOURCE_DIR}/CMakeLists.txt" "CBMRoot CMakeLists.txt"
  kfp_preflight_require_file "${KFP_PREFLIGHT_BUILD_DIR}/CMakeCache.txt" "configured CBMRoot CMake cache"

  if [[ "${KFPARTICLE_CBMROOT_PREFLIGHT_SOURCE_CONFIG:-1}" == "1" \
        && -f "${KFP_PREFLIGHT_BUILD_DIR}/config.sh" ]]; then
    set +u
    source "${KFP_PREFLIGHT_BUILD_DIR}/config.sh"
    set -u
  fi

  local library
  for library in libxpu.so libKFParticle.so libxpu_Hip.so libKFParticle_Hip.so; do
    kfp_preflight_require_file "${KFP_PREFLIGHT_LIBRARY_DIR}/${library}" "CBMRoot-owned ${library}"
  done
  kfp_preflight_require_cache_on "CBM_KFPARTICLE_USE_XPU"
  kfp_preflight_require_cache_on "CBM_KFPARTICLE_GPU_DIAGNOSTICS"

  if [[ "${KFP_PREFLIGHT_DEVICE}" != hip[0-9]* ]]; then
    kfp_preflight_fail "${KFP_PREFLIGHT_LABEL}" \
      "Stage 19 target-server preflight requires an explicit HIP device such as hip1"
  fi

  KFP_PREFLIGHT_HIP_ARCH="$(kfp_preflight_cache_value XPU_HIP_ARCH)"
  KFP_PREFLIGHT_FAIRROOT="$(kfp_preflight_cache_value FairRoot_DIR)"
  if [[ -z "${KFP_PREFLIGHT_FAIRROOT}" ]]; then
    KFP_PREFLIGHT_FAIRROOT="$(kfp_preflight_cache_value FAIRROOTPATH)"
  fi
  if [[ -z "${KFP_PREFLIGHT_FAIRROOT}" ]]; then
    KFP_PREFLIGHT_FAIRROOT="${FAIRROOTPATH:-${FAIRROOT_ROOT:-}}"
  fi
  KFP_PREFLIGHT_ROOT_VERSION="$(kfp_preflight_cache_value ROOT_VERSION)"
  local root_config="${ROOTSYS:+${ROOTSYS}/bin/root-config}"
  root_config="${root_config:-$(command -v root-config || true)}"
  if [[ -n "${root_config}" && -x "${root_config}" ]]; then
    KFP_PREFLIGHT_ROOT_VERSION="$("${root_config}" --version 2>/dev/null || true)"
  elif [[ -z "${KFP_PREFLIGHT_ROOT_VERSION}" && -n "${ROOTSYS:-}" ]]; then
    KFP_PREFLIGHT_ROOT_VERSION="${ROOTSYS}"
  fi
  KFP_PREFLIGHT_ROCM_ROOT="${XPU_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
  KFP_PREFLIGHT_ROCM_VERSION=""
  if [[ -r "${KFP_PREFLIGHT_ROCM_ROOT}/.info/version" ]]; then
    KFP_PREFLIGHT_ROCM_VERSION="$(tr '\n' ' ' <"${KFP_PREFLIGHT_ROCM_ROOT}/.info/version")"
  fi

  KFP_PREFLIGHT_LD_LIBRARY_PATH="${KFP_PREFLIGHT_LIBRARY_DIR}"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    KFP_PREFLIGHT_LD_LIBRARY_PATH+=":${LD_LIBRARY_PATH}"
  fi
  mkdir -p "${KFP_PREFLIGHT_LOG_DIR}"
}
