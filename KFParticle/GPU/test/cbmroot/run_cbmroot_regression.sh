#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../../../../../../" && pwd)"
mode="${1:-equivalence}"
case "$mode" in runtime|equivalence|all) ;; *)
  echo "Usage: $0 [runtime|equivalence|all]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }
fail() { echo "FAIL cbmroot-regression - $*" >&2; exit 2; }

build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-$(cd "${source_dir}/.." && pwd)/build}"
library_dir="${build_dir}/lib"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
log_dir="${KFPARTICLE_CBMROOT_SMOKE_LOG_DIR:-${build_dir}/kfparticle-cbmroot-xpu-smoke}"
evidence_file="${KFPARTICLE_CBMROOT_EQUIVALENCE_EVIDENCE_FILE:-${log_dir}/cbmroot-cpu-gpu-v0-equivalence-${device}.evidence.txt}"
# Invalidate old success even if the combined run fails before equivalence.
if [[ "$mode" == equivalence || "$mode" == all ]]; then rm -f "$evidence_file"; fi
run_ld_path="${library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
include_path="${source_dir}:${source_dir}/reco/base:${source_dir}/algo/kfp/gpu"
include_path+=":${source_dir}/external/KFParticle/KFParticle/GPU"
include_path+=":${source_dir}/external/KFParticle/KFParticle:${source_dir}/external/xpu/src"
include_path+="${ROOT_INCLUDE_PATH:+:${ROOT_INCLUDE_PATH}}"

if [[ -r "${build_dir}/CMakeCache.txt" ]]; then
  configured_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${build_dir}/CMakeCache.txt" | head -n 1)"
  if [[ -n "$configured_source" && "$(cd "$configured_source" && pwd)" != "$source_dir" ]]; then
    fail "build/source mismatch: ${build_dir} was configured from ${configured_source}, script belongs to ${source_dir}"
  fi
fi
libraries=(libxpu.so libCbmRecoBase.so libKFParticle.so libxpu_Hip.so libKFParticle_Hip.so libAlgoOffline.so)
for library in "${libraries[@]}"; do
  [[ -f "${library_dir}/${library}" ]] || fail "missing ${library_dir}/${library}"
done

root_bin="${ROOTSYS:+${ROOTSYS}/bin/root}"
root_bin="${root_bin:-$(command -v root || true)}"
if [[ "$mode" != equivalence ]]; then
  [[ -n "$root_bin" && -x "$root_bin" ]] || fail "ROOT executable was not found"
fi
mkdir -p "$log_dir"
echo "KFParticle CBMRoot regression: ${mode}"
echo "  source dir : ${source_dir}"
echo "  build dir  : ${build_dir}"
echo "  device     : ${device}"

run_check()
{
  local label="$1"
  shift
  local log_file="${log_dir}/${label}-${device}.log"
  echo "  log        : ${log_file}"
  if ! env LD_LIBRARY_PATH="$run_ld_path" ROOT_INCLUDE_PATH="$include_path" \
      XPU_DEVICE="$device" "$@" >"$log_file" 2>&1; then
    cat "$log_file" >&2
    return 1
  fi
  cat "$log_file"
  grep -q "^PASS ${label}" "$log_file" || fail "${label} did not report success"
}

run_equivalence()
{
  local build_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
  local skip_build="${KFPARTICLE_CBMROOT_EQUIVALENCE_SKIP_BUILD:-0}"
  local executable="${build_dir}/bin/KFParticleCpuGpuV0Equivalence"
  local log_file="${log_dir}/cbmroot-cpu-gpu-v0-equivalence-${device}.log"
  [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || fail "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
  if [[ "$skip_build" != 0 && "${KFPARTICLE_CBMROOT_FORCE_KFPARTICLE_REBUILD:-0}" != 0 ]]; then
    fail "skip-build and forced rebuild are mutually exclusive"
  fi
  if [[ "${KFPARTICLE_CBMROOT_FORCE_KFPARTICLE_REBUILD:-0}" != 0 ]]; then
    local target_dir="${build_dir}/external/KFParticle/CMakeFiles/KFParticle.dir"
    if [[ -d "$target_dir" ]]; then
      find "$target_dir" -type f \( -name '*.o' -o -name '*.o.d' \) -delete
    fi
    rm -f "${library_dir}/libKFParticle.so" "${library_dir}/libKFParticle.rootmap" \
      "${library_dir}/libKFParticle_rdict.pcm"
  fi
  if [[ "$skip_build" != 0 ]]; then
    echo "  build      : skipped (explicit run-only)"
  else
    cmake --build "$build_dir" --target KFParticleCpuGpuV0Equivalence --parallel "$build_jobs"
  fi
  [[ -x "$executable" ]] || fail "missing ${executable}"
  run_check cbmroot-cpu-gpu-v0-equivalence "$executable" "$device"

  local rocm_root="${XPU_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
  local rocm_version="unknown" log_hash="unavailable"
  if [[ -r "${rocm_root}/.info/version" ]]; then
    rocm_version="$(<"${rocm_root}/.info/version")"
  elif [[ -x "${rocm_root}/bin/hipconfig" ]]; then
    rocm_version="$("${rocm_root}/bin/hipconfig" --version 2>/dev/null | head -n 1 || true)"
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    log_hash="$(sha256sum "$log_file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    log_hash="$(shasum -a 256 "$log_file" | awk '{print $1}')"
  fi
  {
    echo "KFParticle CBMRoot CPU/GPU V0 equivalence evidence"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "kernel=$(uname -sr)"
    echo "source_dir=${source_dir}"
    echo "build_dir=${build_dir}"
    echo "device=${device}"
    echo "rocm_root=${rocm_root}"
    echo "rocm_version=${rocm_version}"
    echo "build_skipped=${skip_build}"
    echo "build_command=cmake --build ${build_dir} --target KFParticleCpuGpuV0Equivalence --parallel ${build_jobs}"
    echo "run_command=LD_LIBRARY_PATH=${library_dir}:<inherited> XPU_DEVICE=${device} ${executable} ${device}"
    echo "log_file=${log_file}"
    echo "log_sha256=${log_hash}"
    echo "result=PASS cbmroot-cpu-gpu-v0-equivalence"
  } >"$evidence_file"
  echo "PASS cbmroot-cpu-gpu-v0-evidence - wrote ${evidence_file}"
}

checks=("$mode")
if [[ "$mode" == all ]]; then checks=(runtime equivalence); fi
for check in "${checks[@]}"; do
  case "$check" in
    runtime)
      run_check cbmroot-xpu-runtime "$root_bin" -l -b -q \
        "${script_dir}/cbmroot_xpu_smoke.C(\"${device}\")" ;;
    equivalence) run_equivalence ;;
  esac
done
