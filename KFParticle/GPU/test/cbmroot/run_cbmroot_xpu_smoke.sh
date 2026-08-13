#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../../../../../../" && pwd)"

build_dir="${KFPARTICLE_CBMROOT_BUILD_DIR:-$(cd "${source_dir}/.." && pwd)/build}"
library_dir="${build_dir}/lib"
device="${KFPARTICLE_CBMROOT_DEVICE:-hip1}"
log_dir="${KFPARTICLE_CBMROOT_SMOKE_LOG_DIR:-${build_dir}/kfparticle-cbmroot-xpu-smoke}"
root_bin="${ROOTSYS:+${ROOTSYS}/bin/root}"
root_bin="${root_bin:-$(command -v root || true)}"

if [[ -r "${build_dir}/CMakeCache.txt" ]]; then
  configured_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${build_dir}/CMakeCache.txt" | head -n 1)"
  if [[ -n "${configured_source}" && "$(cd "${configured_source}" && pwd)" != "${source_dir}" ]]; then
    echo "FAIL cbmroot-xpu-runtime - build/source mismatch: ${build_dir} was configured from ${configured_source}, script belongs to ${source_dir}" >&2
    exit 2
  fi
fi

if [[ -z "${root_bin}" || ! -x "${root_bin}" ]]; then
  echo "FAIL cbmroot-xpu-runtime - ROOT executable was not found" >&2
  exit 2
fi

for library in libxpu.so libCbmRecoBase.so libKFParticle.so libxpu_Hip.so libKFParticle_Hip.so; do
  if [[ ! -f "${library_dir}/${library}" ]]; then
    echo "FAIL cbmroot-xpu-runtime - missing ${library_dir}/${library}" >&2
    exit 2
  fi
done

mkdir -p "${log_dir}"
log_file="${log_dir}/cbmroot-xpu-runtime-${device}.log"
macro="${script_dir}/cbmroot_xpu_smoke.C"
include_path="${source_dir}/reco/base"
include_path+=":${source_dir}/external/KFParticle/KFParticle/GPU"
include_path+=":${source_dir}/external/KFParticle/KFParticle"
include_path+=":${source_dir}/external/xpu/src"
if [[ -n "${ROOT_INCLUDE_PATH:-}" ]]; then
  include_path+=":${ROOT_INCLUDE_PATH}"
fi

echo "KFParticle CBMRoot XPU smoke test"
echo "  source dir : ${source_dir}"
echo "  build dir  : ${build_dir}"
echo "  device     : ${device}"
echo "  log        : ${log_file}"

run_ld_path="${library_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  run_ld_path+=":${LD_LIBRARY_PATH}"
fi

if ! LD_LIBRARY_PATH="${run_ld_path}" \
  ROOT_INCLUDE_PATH="${include_path}" \
  XPU_DEVICE="${device}" \
  "${root_bin}" -l -b -q "${macro}(\"${device}\")" \
  >"${log_file}" 2>&1; then
  cat "${log_file}" >&2
  exit 1
fi

cat "${log_file}"
if ! grep -q '^PASS cbmroot-xpu-runtime' "${log_file}"; then
  echo "FAIL cbmroot-xpu-runtime - macro did not report success" >&2
  exit 1
fi
