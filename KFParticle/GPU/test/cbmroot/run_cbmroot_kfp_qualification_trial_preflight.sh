#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${KFPARTICLE_CBMROOT_SOURCE_DIR:-$(cd "${script_dir}/../../../../../../" && pwd)}"
ordinary_build="${KFPARTICLE_CBMROOT_ORDINARY_BUILD_DIR:-}"
trial_build="${KFPARTICLE_CBMROOT_TRIAL_BUILD_DIR:-${KFPARTICLE_CBMROOT_BUILD_DIR:-}}"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
source_config="${KFPARTICLE_CBMROOT_QUALIFICATION_SOURCE_CONFIG:-}"
trial_config="${KFPARTICLE_CBMROOT_QUALIFICATION_CONFIG:-}"
policy="${KFPARTICLE_CBMROOT_QUALIFICATION_POLICY:-${script_dir}/step20_qualification_policy.conf}"

fail()
{
  echo "FAIL cbmroot-kfp-qualification-trial-preflight - $*" >&2
  exit 2
}

cache_value()
{
  local build="$1"
  local key="$2"
  sed -n "s/^${key}:[^=]*=//p" "${build}/CMakeCache.txt" | head -1
}

file_hash()
{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check_build()
{
  local build="$1"
  local expected="$2"
  local cache_flag marker library
  [[ -f "${build}/CMakeCache.txt" ]] || fail "missing ${expected} CMakeCache.txt: ${build}"
  [[ "$(cache_value "${build}" CBM_KFPARTICLE_USE_XPU)" == "ON" ]] \
    || fail "${expected} build does not have CBM_KFPARTICLE_USE_XPU=ON"
  [[ "$(cache_value "${build}" CBM_KFPARTICLE_GPU_DIAGNOSTICS)" == "ON" ]] \
    || fail "${expected} build does not have CBM_KFPARTICLE_GPU_DIAGNOSTICS=ON"
  cache_flag="$(cache_value "${build}" CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL)"
  if [[ "${expected}" == "ordinary" ]]; then
    [[ "${cache_flag}" == "OFF" ]] || fail "ordinary build must explicitly set CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=OFF"
    marker="kfp_gpu_qualification_build=ordinary-v1"
  else
    [[ "${cache_flag}" == "ON" ]] || fail "trial build must set CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=ON"
    marker="kfp_gpu_qualification_build=trial-v1"
  fi
  for library in libAlgo.so libAlgoOffline.so; do
    [[ -f "${build}/lib/${library}" ]] || fail "missing ${expected} ${library}: ${build}/lib/${library}"
    grep -a -q "${marker}" "${build}/lib/${library}" \
      || fail "${expected} ${library} does not contain ${marker}; rebuild it"
  done
}

[[ -d "${source_dir}" ]] || fail "missing source directory: ${source_dir}"
[[ -n "${ordinary_build}" ]] || fail "set KFPARTICLE_CBMROOT_ORDINARY_BUILD_DIR"
[[ -n "${trial_build}" ]] || fail "set KFPARTICLE_CBMROOT_TRIAL_BUILD_DIR"
[[ "${ordinary_build}" != "${trial_build}" ]] || fail "ordinary and trial build directories must be different"
[[ "${device}" == hip[0-9]* ]] || fail "set an explicit HIP device such as hip1"
[[ -f "${source_config}" ]] || fail "set KFPARTICLE_CBMROOT_QUALIFICATION_SOURCE_CONFIG to the reviewed MainConfig.yaml"
[[ -f "${policy}" ]] || fail "missing frozen qualification policy: ${policy}"

if [[ -z "${trial_config}" ]]; then
  trial_config="${source_config%.yaml}.step20-v0.yaml"
fi
[[ "$(cd "$(dirname "${source_config}")" && pwd)" == "$(cd "$(dirname "${trial_config}")" && pwd)" ]] \
  || fail "qualification YAML must be beside the source YAML so relative parameter paths remain valid"

check_build "${ordinary_build}" ordinary
check_build "${trial_build}" trial
python3 "${script_dir}/make_step20_v0_qualification_config.py" \
  "${source_config}" "${trial_config}"

evidence="${KFPARTICLE_CBMROOT_QUALIFICATION_TRIAL_EVIDENCE:-${trial_build}/kfparticle-step20-3/trial-preflight.evidence.txt}"
mkdir -p "$(dirname "${evidence}")"
{
  echo "KFParticle CBMRoot Step 20.3A qualification-trial preflight evidence"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host: $(hostname)"
  echo "status: ordinary build locked; separate trial build identified; narrow V0 configuration frozen"
  echo "source: ${source_dir}"
  echo "source revision: $(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "KFParticle revision: $(git -C "${source_dir}/external/KFParticle" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ordinary build: ${ordinary_build}"
  echo "ordinary build type: $(cache_value "${ordinary_build}" CMAKE_BUILD_TYPE)"
  echo "ordinary qualification trial: $(cache_value "${ordinary_build}" CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL)"
  echo "trial build: ${trial_build}"
  echo "trial build type: $(cache_value "${trial_build}" CMAKE_BUILD_TYPE)"
  echo "trial qualification trial: $(cache_value "${trial_build}" CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL)"
  echo "device: ${device}"
  echo "HIP architecture: $(cache_value "${trial_build}" XPU_HIP_ARCH)"
  echo "source configuration: ${source_config}"
  echo "source configuration sha256: $(file_hash "${source_config}")"
  echo "qualification configuration: ${trial_config}"
  echo "qualification configuration sha256: $(file_hash "${trial_config}")"
  echo "qualification policy: ${policy}"
  echo "qualification policy sha256: $(file_hash "${policy}")"
  echo "supported PDGs: 310,3122,-3122"
  echo "finderCuts: removed"
} >"${evidence}"

echo "PASS cbmroot-kfp-qualification-trial-preflight - ordinary/trial builds and narrow V0 configuration are unambiguous"
echo "  config     : ${trial_config}"
echo "  evidence   : ${evidence}"
