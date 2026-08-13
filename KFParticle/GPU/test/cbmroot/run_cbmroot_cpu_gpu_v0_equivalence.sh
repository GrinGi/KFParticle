#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../../../../../../" && pwd)"

build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-$(cd "${source_dir}/.." && pwd)/build}"
library_dir="${build_dir}/lib"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
log_dir="${KFPARTICLE_CBMROOT_SMOKE_LOG_DIR:-${build_dir}/kfparticle-cbmroot-xpu-smoke}"
if [[ -r "${build_dir}/CMakeCache.txt" ]]; then
  configured_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${build_dir}/CMakeCache.txt" | head -n 1)"
  if [[ -n "${configured_source}" && "$(cd "${configured_source}" && pwd)" != "${source_dir}" ]]; then
    echo "FAIL cbmroot-cpu-gpu-v0-equivalence - build/source mismatch: ${build_dir} was configured from ${configured_source}, script belongs to ${source_dir}" >&2
    exit 2
  fi
fi
for library in libxpu.so libCbmRecoBase.so libKFParticle.so libAlgoOffline.so libxpu_Hip.so libKFParticle_Hip.so; do
  if [[ ! -f "${library_dir}/${library}" ]]; then
    echo "FAIL cbmroot-cpu-gpu-v0-equivalence - missing ${library_dir}/${library}" >&2
    exit 2
  fi
done

mkdir -p "${log_dir}"
log_file="${log_dir}/cbmroot-cpu-gpu-v0-equivalence-${device}.log"
evidence_file="${KFPARTICLE_CBMROOT_EQUIVALENCE_EVIDENCE_FILE:-${log_dir}/cbmroot-cpu-gpu-v0-equivalence-${device}.evidence.txt}"
executable="${build_dir}/bin/KFParticleCpuGpuV0Equivalence"

echo "KFParticle CBMRoot CPU/GPU V0 equivalence test"
echo "  source dir : ${source_dir}"
echo "  build dir  : ${build_dir}"
echo "  device     : ${device}"
echo "  log        : ${log_file}"
echo "  evidence   : ${evidence_file}"

if [[ "${KFPARTICLE_CBMROOT_FORCE_KFPARTICLE_REBUILD:-0}" != "0" ]]; then
  kfparticle_target_dir="${build_dir}/external/KFParticle/CMakeFiles/KFParticle.dir"
  if [[ -d "${kfparticle_target_dir}" ]]; then
    find "${kfparticle_target_dir}" -type f \( -name '*.o' -o -name '*.o.d' \) -delete
  fi
  rm -f "${library_dir}/libKFParticle.so" \
        "${library_dir}/libKFParticle.rootmap" \
        "${library_dir}/libKFParticle_rdict.pcm"
fi

if ! cmake --build "${build_dir}" --target KFParticle -j; then
  echo "FAIL cbmroot-cpu-gpu-v0-equivalence - KFParticle build target failed" >&2
  exit 1
fi
if ! cmake --build "${build_dir}" --target KFParticleCpuGpuV0Equivalence -j; then
  echo "FAIL cbmroot-cpu-gpu-v0-equivalence - build target failed" >&2
  exit 1
fi
if [[ ! -x "${executable}" ]]; then
  echo "FAIL cbmroot-cpu-gpu-v0-equivalence - missing ${executable}" >&2
  exit 2
fi

run_ld_path="${library_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  run_ld_path+=":${LD_LIBRARY_PATH}"
fi

if ! LD_LIBRARY_PATH="${run_ld_path}" \
  XPU_DEVICE="${device}" \
  "${executable}" "${device}" \
  >"${log_file}" 2>&1; then
  cat "${log_file}" >&2
  exit 1
fi

cat "${log_file}"
if ! grep -q '^PASS cbmroot-cpu-gpu-v0-equivalence' "${log_file}"; then
  echo "FAIL cbmroot-cpu-gpu-v0-equivalence - macro did not report success" >&2
  exit 1
fi

rocm_root="${XPU_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
rocm_version="unknown"
if [[ -r "${rocm_root}/.info/version" ]]; then
  rocm_version="$(<"${rocm_root}/.info/version")"
elif [[ -x "${rocm_root}/bin/hipconfig" ]]; then
  rocm_version="$("${rocm_root}/bin/hipconfig" --version 2>/dev/null | head -n 1 || true)"
fi
log_hash="unavailable"
if command -v sha256sum >/dev/null 2>&1; then
  log_hash="$(sha256sum "${log_file}" | awk '{print $1}')"
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
  echo "build_command=cmake --build ${build_dir} --target KFParticleCpuGpuV0Equivalence -j"
  echo "run_command=LD_LIBRARY_PATH=${library_dir}:<inherited> XPU_DEVICE=${device} ${executable} ${device}"
  echo "log_file=${log_file}"
  echo "log_sha256=${log_hash}"
  echo "result=PASS cbmroot-cpu-gpu-v0-equivalence"
} >"${evidence_file}"

echo "PASS cbmroot-cpu-gpu-v0-evidence - wrote ${evidence_file}"
