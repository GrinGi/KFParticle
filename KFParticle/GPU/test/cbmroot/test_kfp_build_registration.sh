#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

contract_mode="${1:-cbmroot}"
case "$contract_mode" in
  cbmroot|sdk) ;;
  -h|--help)
    echo "Usage: $(basename "$0") [cbmroot|sdk]"
    echo "  cbmroot: embedded targets, ABI settings and shared XPU (default; no SDK required)"
    echo "  sdk: standalone package export/import and installed-XPU checks (no CBMRoot required)"
    exit 0 ;;
  *) echo "FAIL unknown build contract: $contract_mode (expected cbmroot or sdk)" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "FAIL unexpected build-contract arguments" >&2; exit 2; }
echo "KFParticle build contract: $contract_mode"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
trap 'echo "FAIL kfp-build-registration at line $LINENO: $BASH_COMMAND" >&2
  if [[ -n "${failure_log:-}" && -f "$failure_log" ]]; then tail -n 80 "$failure_log" >&2; fi' ERR

gpu_dir="$(cd "$here/../.." && pwd)"
kfp_source="$(cd "$gpu_dir/../.." && pwd)"

if [[ "$contract_mode" == cbmroot ]]; then
source_dir="$(cd "$here/../../../../../.." && pwd)"
adapter_dir="$source_dir/algo/kfp/gpu"
if [[ ! -r "$adapter_dir/CMakeLists.txt" || ! -r "$source_dir/external/InstallKFParticle.cmake" ]]; then
  echo "FAIL cbmroot mode requires a CBMRoot checkout; use sdk for standalone package checks" >&2
  exit 2
fi
for retired in KfpGpuMaterializationAdapter.h KfpGpuMaterializationAdapter.cxx \
    KfpGpuDiagnosticStatus.h KfpGpuBatchPolicy.h KfpGpuBatchPolicy.cxx \
    KfpGpuPerformanceReport.h KfpGpuPerformanceReport.cxx \
    KfpGpuInputPacker.h KfpGpuInputPacker.cxx KfpGpuBatch.h KfpGpuBatch.cxx \
    KfpGpuExecutor.h KfpGpuExecutor.cxx KfpGpuRuntimeService.h KfpGpuRuntimeService.cxx \
    KfpGpuMaterializationSchedule.h KfpGpuTrace.h; do
  if [[ -e "$adapter_dir/$retired" ]]; then
    echo "FAIL stale adapter source: $adapter_dir/$retired" >&2
    exit 1
  fi
done
for stale in "$adapter_dir"/KfpGpuDiagnostic*.h "$adapter_dir"/KfpGpuDiagnostic*.cxx \
    "$adapter_dir/test/_GTestKfpGpuDiagnosticPacker.cxx" \
    "$adapter_dir/test/_GTestKfpGpuInputPacker.cxx" \
    "$adapter_dir/test/_GTestKfpGpuBatch.cxx" \
    "$adapter_dir/test/_GTestKfpGpuRunner.cxx" \
    "$adapter_dir/test/_GTestKfpGpuExecutor.cxx"; do
  if [[ -e "$stale" ]]; then
    echo "FAIL stale adapter source: $stale" >&2
    exit 1
  fi
done
for patch in KFParticle_applem1.patch KFParticle_clang_tidy.patch; do
  if [[ -e "$source_dir/external/$patch" ]]; then
    echo "FAIL retired KFParticle patch: $source_dir/external/$patch" >&2; exit 1
  fi
done
for directory in data planning execution algorithms diagnostics; do
  [[ -d "$gpu_dir/$directory" ]]
done
# A stale flat copy could silently shadow a moved header after server deployment.
for file in "$gpu_dir"/*.h "$gpu_dir"/*.cxx; do
  [[ -f "$file" ]] || continue
  [[ "$(basename "$file")" == KFParticleGpuPlatform.h ]] || {
    echo "FAIL stale flat GPU source: $file" >&2; exit 1;
  }
done
echo "PASS kfp-build-registration source layout has no stale flat headers or sources"

# Common execution is owned by KFParticle, not built a second time in CBMRoot.
for component in data/InputPacker planning/Batch execution/Executor execution/RuntimeService execution/Finder; do
  directory="${component%/*}"
  name="KFParticleGpu${component#*/}"
  for extension in h cxx; do
    file="$gpu_dir/$directory/$name.$extension"
    [[ -r "$file" ]]
    if grep -Eq '#include "(algo/|reco/|Cbm)|cbm::|CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL' "$file"; then
      echo "FAIL common KFParticle source depends on CBMRoot: $file" >&2; exit 1
    fi
  done
  grep -Fq "KFParticle/GPU/$directory/$name.cxx" "$gpu_dir/../../CMakeLists.txt"
done
grep -Fq 'execution/KFParticleGpuRuntimeService.cxx' "$gpu_dir/test/xpu/CMakeLists.txt"
if grep -Eq 'GpuBatch(Action|Decision|PolicyProfile)|GpuRouter' \
    "$gpu_dir/planning/KFParticleGpuBatch.h"; then
  echo "FAIL framework routing policy leaked into the common batch" >&2; exit 1
fi
echo "PASS kfp-build-registration common execution has no CBMRoot dependency"

for source in "$adapter_dir/KfpGpuQualifiedBatch.cxx" \
    "$adapter_dir/KfpGpuReferenceCollector.cxx" \
    "$source_dir/algo/kfp/selector/KfpSelector.cxx" \
    "$source_dir/algo/global/Reco.cxx" "$source_dir/reco/tasks/CbmTaskKfpSelector.cxx"; do
  if grep -Eq 'GpuExecutor[[:space:]]*\(|GpuExecutor[[:space:]]+[[:alnum:]_]+|GpuRuntimeService::Instance' "$source"; then
    echo "FAIL CBMRoot bypasses the public Finder: $source" >&2; exit 1
  fi
done
if grep -Eq '#include "algo/|using cbm::' "$gpu_dir/test/standalone/KFParticleGpuTestFixture.h"; then
  echo "FAIL neutral test fixture depends on CBMRoot" >&2; exit 1
fi

# Configure the real adapter CMake with dependency targets, not fake headers.
# Compile the ROOT-free comparison/trace declarations without initializing XPU.
cat >"$temporary/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.14)
project(KfpBuildRegistration LANGUAGES CXX)
enable_testing()
foreach(dependency ROOT::Core xpu KFParticle CbmKFParticleOnlineInterface CbmRecoBase)
  add_library(${dependency} INTERFACE IMPORTED)
endforeach()
if(WITH_GTEST)
  add_library(GTest::gtest_main INTERFACE IMPORTED)
endif()
set(KFPARTICLE_USE_XPU ON)
add_subdirectory("${ADAPTER_DIR}" adapter)

get_property(contracts DIRECTORY "${ADAPTER_DIR}" PROPERTY TESTS)
get_target_property(adapter_sources CbmKfParticleGpuAdapter SOURCES)
get_target_property(adapter_definitions CbmKfParticleGpuAdapter COMPILE_DEFINITIONS)
if(NOT "CBM_KFPARTICLE_GPU_DIAGNOSTICS" IN_LIST adapter_definitions)
  message(FATAL_ERROR "legacy GPU adapter build interface is missing")
endif()
if(CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL)
  if(NOT "CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL" IN_LIST adapter_definitions)
    message(FATAL_ERROR "trial qualification marker is missing")
  endif()
elseif("CBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL" IN_LIST adapter_definitions)
  message(FATAL_ERROR "ordinary build unexpectedly unlocks qualification")
endif()
foreach(component ReferenceCollector Comparison QualifiedBatch Report Trace Routing)
  if(NOT "KfpGpu${component}.cxx" IN_LIST adapter_sources)
    message(FATAL_ERROR "adapter implementation is missing: ${component}")
  endif()
endforeach()
foreach(component Batch InputPacker Executor RuntimeService)
  if("KfpGpu${component}.cxx" IN_LIST adapter_sources)
    message(FATAL_ERROR "common implementation is still compiled by CBMRoot: ${component}")
  endif()
endforeach()
if(NOT BUILD_TESTING)
  if(contracts OR TARGET KFParticleCpuGpuV0Equivalence
      OR TARGET _GTestKfpGpuAdapter OR TARGET KFParticleGpuDataTest
      OR TARGET KFParticleGpuHostInterfaceTest)
    message(FATAL_ERROR "BUILD_TESTING=OFF retained test targets")
  endif()
  # Include the consolidated declarations first, without ROOT or a result definition.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/trace-header.cxx"
    "#include \"${ADAPTER_DIR}/KfpGpuComparison.h\"\n"
    "#include \"${ADAPTER_DIR}/KfpGpuComparison.h\"\n"
    "#include <type_traits>\n"
    "static_assert(std::is_pointer_v<decltype(&cbm::algo::kfp::TraceFirstGpuMismatch)>);\n"
    "int main() { return 0; }\n")
  try_compile(trace_header_compiles "${CMAKE_CURRENT_BINARY_DIR}/trace-header-check"
    "${CMAKE_CURRENT_BINARY_DIR}/trace-header.cxx"
    CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=17"
      "-DINCLUDE_DIRECTORIES:STRING=${ADAPTER_DIR}/../../../external/KFParticle/KFParticle/GPU"
    OUTPUT_VARIABLE trace_header_output)
  if(NOT trace_header_compiles)
    message(FATAL_ERROR "comparison/trace declarations are not self-contained:\n${trace_header_output}")
  endif()
  return()
endif()

if(NOT TARGET KFParticleCpuGpuV0Equivalence)
  message(FATAL_ERROR "explicit device regression target is missing")
endif()
if(NOT TARGET KFParticleGpuDataTest)
  message(FATAL_ERROR "shared host data test target is missing")
endif()
get_property(data_tests DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/adapter/host-tests" PROPERTY TESTS)
foreach(component State SoAView InputData CandidatePool)
  if(NOT "KFParticleGpu${component}Test" IN_LIST data_tests)
    message(FATAL_ERROR "host data contract is missing: ${component}")
  endif()
endforeach()
foreach(contract TimingReport FullChainPreflight AdaptiveCalibration BuildRegistration)
  if(NOT "_TestKfpGpu${contract}" IN_LIST contracts)
    message(FATAL_ERROR "Bash contract is missing: ${contract}")
  endif()
endforeach()
foreach(retired QualificationMatrix QualificationFinalDecision Step22Comparison Step22Finalization)
  if("_TestKfpGpu${retired}" IN_LIST contracts)
    message(FATAL_ERROR "retired contract is still registered: ${retired}")
  endif()
endforeach()
if(WITH_GTEST)
  if(NOT TARGET _GTestKfpGpuAdapter)
    message(FATAL_ERROR "GTest adapter target is missing")
  endif()
  get_target_property(sources _GTestKfpGpuAdapter SOURCES)
  foreach(component Comparison Materialization Policy Report)
    if(NOT "test/_GTestKfpGpu${component}.cxx" IN_LIST sources)
      message(FATAL_ERROR "adapter coverage source is missing: ${component}")
    endif()
  endforeach()
  foreach(component InputPacker Batch Executor)
    if("test/_GTestKfpGpu${component}.cxx" IN_LIST sources)
      message(FATAL_ERROR "common host checks are still compiled by CBMRoot: ${component}")
    endif()
  endforeach()
  if(NOT TARGET KFParticleGpuHostInterfaceTest)
    message(FATAL_ERROR "shared full-library host checks are missing")
  endif()
  get_target_property(core_sources KFParticleGpuHostInterfaceTest SOURCES)
  foreach(component InputPacker Batch)
    if(NOT "standalone/KFParticleGpu${component}Test.cxx" IN_LIST core_sources)
      message(FATAL_ERROR "shared host coverage source is missing: ${component}")
    endif()
  endforeach()
elseif(TARGET _GTestKfpGpuAdapter)
  message(FATAL_ERROR "GTest target exists without GTest")
endif()
CMAKE

for variant in disabled no-gtest gtest gtest-trial; do
  testing=ON
  gtest=OFF
  trial=OFF
  case "$variant" in
    disabled) testing=OFF ;;
    gtest) gtest=ON ;;
    gtest-trial) gtest=ON; trial=ON ;;
  esac
  if ! cmake -S "$temporary" -B "$temporary/$variant" \
      "-DADAPTER_DIR=$source_dir/algo/kfp/gpu" \
      "-DBUILD_TESTING=$testing" "-DWITH_GTEST=$gtest" \
      "-DCBM_KFPARTICLE_GPU_QUALIFICATION_TRIAL=$trial" \
      >"$temporary/$variant.log" 2>&1; then
    cat "$temporary/$variant.log" >&2
    exit 1
  fi
  echo "PASS kfp-build-registration $variant"
done

# Dispatch tests are hermetic: fake build tools must never run in --run-only.
mkdir -p "$temporary/dispatch/bin" "$temporary/dispatch/lib" "$temporary/tools" "$temporary/xpu"
cat >"$temporary/tools/cmake" <<'BASH'
#!/usr/bin/env bash
echo "$*" >>"$(dirname "$0")/cmake.calls"
BASH
cat >"$temporary/dispatch/bin/fixture" <<'BASH'
#!/usr/bin/env bash
[[ ! -e "$(dirname "$0")/reject" ]] || exit 1
echo "PASS $(basename "$0")"
if [[ "$(basename "$0")" == KFParticleGpuXpuBatchBenchmark ]]; then
  [[ ! -e "$(dirname "$0")/no-marker" ]] || exit 0
  echo "PASS kfparticle-gpu-batch-benchmark events=$KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS iterations=1"
  echo "METRIC mode=serial events=$KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS wall_ms=2.0"
  [[ ! -e "$(dirname "$0")/missing-mode" ]] || exit 0
  echo "METRIC mode=batch events=$KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS wall_ms=1.0"
fi
BASH
cat >"$temporary/tools/ctest" <<'BASH'
#!/usr/bin/env bash
echo "$*" >>"$(dirname "$0")/ctest.calls"
[[ ! -e "$(dirname "$0")/reject-ctest" ]] || exit 1
while (($#)); do
  if [[ "$1" == --test-dir ]]; then build="$2"; break; fi
  shift
done
"$build/bin/KFParticleGpuXpuLifecycleTest"
BASH
chmod +x "$temporary/tools/cmake" "$temporary/tools/ctest" "$temporary/dispatch/bin/fixture"
for name in KFParticleGpuXpuLifecycleTest KFParticleGpuXpuBatchBenchmark; do
  cp "$temporary/dispatch/bin/fixture" "$temporary/dispatch/bin/$name"
done
printf 'KFPARTICLE_GPU_TEST_DEVICE:STRING=cpu\nKFPARTICLE_GPU_TEST_ENV_MODE:STRING=standalone\n' \
  >"$temporary/dispatch/CMakeCache.txt"
launcher="$here/../run_xpu_tests.sh"
dispatch_env=(
  "PATH=$temporary/tools:$PATH"
  "KFPARTICLE_GPU_TEST_BUILD_DIR=$temporary/dispatch"
  "KFPARTICLE_GPU_XPU_SOURCE_DIR=$temporary/xpu"
  KFPARTICLE_GPU_TEST_DEVICE=cpu KFPARTICLE_GPU_TEST_ENV_MODE=standalone
  KFPARTICLE_GPU_TEST_DIAGNOSTICS=0
  KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS=2
  KFPARTICLE_GPU_BATCH_BENCHMARK_SIZES='1 3'
  KFPARTICLE_GPU_BATCH_BENCHMARK_SAMPLES=2
  KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS=1
  KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS=1
)
for mode in lifecycle benchmark; do
  env "${dispatch_env[@]}" bash "$launcher" "$mode" --run-only >"$temporary/$mode.log"
  grep -q 'build .* skipped' "$temporary/$mode.log"
  grep -q 'PASS KFParticleGpuXpuLifecycleTest' "$temporary/$mode.log"
done
[[ "$(wc -l <"$temporary/tools/ctest.calls" | tr -d ' ')" == 2 ]]
grep -q -- '-R \^KFParticleGpuXpuLifecycleTest\$' "$temporary/tools/ctest.calls"
! grep -q 'PASS KFParticleGpuXpuBatchBenchmark' "$temporary/lifecycle.log"
grep -q 'PASS KFParticleGpuXpuBatchBenchmark' "$temporary/benchmark.log"
[[ ! -e "$temporary/tools/cmake.calls" ]]
env "${dispatch_env[@]}" bash "$launcher" build >"$temporary/build.log"
[[ "$(wc -l <"$temporary/tools/cmake.calls" | tr -d ' ')" == 2 ]]
! grep -q 'PASS KFParticleGpuXpuLifecycleTest' "$temporary/build.log"
env "${dispatch_env[@]}" bash "$launcher" matrix --run-only >"$temporary/matrix.log"
matrix_evidence="$temporary/dispatch/logs/kfparticle-gpu-batch-benchmark-cpu.txt"
[[ "$(grep -c '^METRIC ' "$matrix_evidence")" == 8 ]]
[[ "$(wc -l <"$temporary/tools/cmake.calls" | tr -d ' ')" == 2 ]]
[[ "$(wc -l <"$temporary/tools/ctest.calls" | tr -d ' ')" == 3 ]]
for override in KFPARTICLE_GPU_TEST_DEVICE=hip1 KFPARTICLE_GPU_TEST_ENV_MODE=inherited; do
  if env "${dispatch_env[@]}" "$override" bash "$launcher" lifecycle --run-only \
      >"$temporary/mismatched-cache.log" 2>&1; then
    echo "FAIL run-only accepted a mismatched CMake cache" >&2; exit 1
  fi
  grep -q 'configure/build this directory first' "$temporary/mismatched-cache.log"
done
for failure in no-marker missing-mode; do
  touch "$temporary/dispatch/bin/$failure"
  if env "${dispatch_env[@]}" bash "$launcher" matrix --run-only >"$temporary/$failure.log" 2>&1; then
    echo "FAIL matrix accepted $failure" >&2; exit 1
  fi
  [[ ! -e "$matrix_evidence" && ! -e "$matrix_evidence.tmp" ]]
  rm "$temporary/dispatch/bin/$failure"
done
for invalid in '' 0 -1 1025 '1 nope'; do
  if env "${dispatch_env[@]}" KFPARTICLE_GPU_BATCH_BENCHMARK_SIZES="$invalid" \
      bash "$launcher" matrix --run-only >"$temporary/invalid-matrix.log" 2>&1; then
    echo "FAIL matrix accepted invalid sizes: $invalid" >&2; exit 1
  fi
done
touch "$temporary/tools/reject-ctest"
if env "${dispatch_env[@]}" bash "$launcher" matrix --run-only >"$temporary/ctest-failure.log" 2>&1; then
  echo "FAIL matrix ignored failed validation" >&2; exit 1
fi
[[ ! -e "$matrix_evidence" ]]
rm "$temporary/tools/reject-ctest"
touch "$temporary/dispatch/bin/reject"
if env "${dispatch_env[@]}" bash "$launcher" lifecycle --run-only >"$temporary/reject.log" 2>&1; then
  echo "FAIL standalone launcher hid a test failure" >&2; exit 1
fi
if env "${dispatch_env[@]}" bash "$launcher" unknown >"$temporary/unknown.log" 2>&1; then
  echo "FAIL standalone launcher accepted an unknown mode" >&2; exit 1
fi
echo "PASS kfp-build-registration standalone dispatch and explicit run-only contract"

# Copy only the inputs required by the launcher. Mutate the fixture, never real
# sources, and ensure each functional directory participates in build identity.
fixture_gpu="$temporary/layout/external/KFParticle/KFParticle/GPU"
mkdir -p "$fixture_gpu/test"
for directory in data planning execution algorithms diagnostics; do
  cp -R "$gpu_dir/$directory" "$fixture_gpu/$directory"
done
cp "$gpu_dir/KFParticleGpuPlatform.h" "$gpu_dir/KFParticleGpuBuildOptions.h.in" "$fixture_gpu/"
cp "$gpu_dir/test/run_xpu_tests.sh" "$gpu_dir/test/KFParticleGpuXpuLifecycleTest.cxx" \
  "$gpu_dir/test/KFParticleGpuXpuBatchBenchmark.cxx" "$fixture_gpu/test/"
cp -R "$gpu_dir/test/lifecycle" "$fixture_gpu/test/lifecycle"
mkdir -p "$fixture_gpu/test/xpu" "$fixture_gpu/.local"

fixture_fingerprint()
{
  env "${dispatch_env[@]}" bash "$fixture_gpu/test/run_xpu_tests.sh" build \
    >"$temporary/fingerprint.log"
  sed -n 's/.*-DKFPARTICLE_GPU_SOURCE_FINGERPRINT=\([^ ]*\).*/\1/p' \
    "$temporary/tools/cmake.calls" | tail -n 1
}

before="$(fixture_fingerprint)"
[[ -n "$before" && "$before" != *unavailable* ]]
for relative in data/KFParticleGpuTransaction.h planning/KFParticleGpuDecayPlan.cxx \
    execution/KFParticleGpuRuntime.cxx algorithms/KFParticleGpuSelection.h \
    diagnostics/KFParticleGpuHostTiming.h test/KFParticleGpuXpuBatchBenchmark.cxx; do
  printf '\n// Fingerprint coverage fixture.\n' >>"$fixture_gpu/$relative"
  after="$(fixture_fingerprint)"
  [[ -n "$after" && "$after" != "$before" ]]
  before="$after"
done
printf '// Not compiled.\n' >"$fixture_gpu/.local/ignored.h"
[[ "$(fixture_fingerprint)" == "$before" ]]
echo "PASS kfp-build-registration nested sources affect identity; local notes do not"
fi

# Dependency fixtures for configuration checks; these are not numerical tests.
sdk="$temporary/sdk"
mkdir -p "$sdk/deps" "$sdk/xpu/src/xpu" "$sdk/project" "$sdk/consumer" \
  "$sdk/installed-xpu/include/xpu" "$sdk/installed-xpu/lib64"
touch "$sdk/installed-xpu/include/xpu/host.h" \
  "$sdk/installed-xpu/lib64/libxpu.so" "$sdk/installed-xpu/lib64/libxpu.dylib"
cat >"$sdk/deps/ROOTConfig.cmake" <<'CMAKE'
set(ROOT_FOUND TRUE)
set(ROOT_VERSION_MAJOR 6)
set(ROOT_USE_FILE "${CMAKE_CURRENT_LIST_DIR}/RootUse.cmake")
set(ROOT_LIBRARIES "")
foreach(component Core Hist Matrix Physics EG Gpad Graf Graf3d RIO MathCore)
  if(NOT TARGET ROOT::${component})
    add_library(ROOT::${component} INTERFACE IMPORTED GLOBAL)
  endif()
  list(APPEND ROOT_LIBRARIES ROOT::${component})
endforeach()
# Old CMake dependency configs may overwrite this caller variable.
set(PACKAGE_PREFIX_DIR /unrelated/dependency)
CMAKE
cat >"$sdk/deps/RootUse.cmake" <<'CMAKE'
function(ROOT_GENERATE_DICTIONARY name)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${name}.cxx" "// Configuration fixture.\n")
endfunction()
CMAKE
cat >"$sdk/deps/VcConfig.cmake" <<'CMAKE'
set(Vc_FOUND TRUE)
set(Vc_LIBRARIES Vc::Vc)
if(NOT TARGET Vc::Vc)
  add_library(Vc::Vc INTERFACE IMPORTED GLOBAL)
endif()
CMAKE
cat >"$sdk/xpu/CMakeLists.txt" <<'CMAKE'
add_library(xpu SHARED dummy.cpp)
target_include_directories(xpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
function(xpu_attach target)
  # Match upstream's host-runtime dependency, without compiling device images.
  set_property(TARGET ${target} APPEND PROPERTY LINK_LIBRARIES xpu)
endfunction()
CMAKE
touch "$sdk/xpu/dummy.cpp" "$sdk/xpu/src/xpu/host.h" "$sdk/xpu/LICENSE"

# Both CBMRoot entry points must reuse the current KFParticle target, without
# applying patches for retired source layouts or compiling a second copy.
if [[ "$contract_mode" == cbmroot ]]; then
mkdir -p "$sdk/cbm/external"
ln -s "$gpu_dir/../.." "$sdk/cbm/external/KFParticle"
cat >"$sdk/cbm/external/CMakeLists.txt" <<'CMAKE'
include("${CBM_SOURCE}/external/InstallKFParticle.cmake")
CMAKE
cat >"$sdk/cbm/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpCbmGraph LANGUAGES CXX)
function(download_project_if_needed)
  if("PATCH_COMMAND" IN_LIST ARGN)
    message(FATAL_ERROR "obsolete KFParticle patch requested")
  endif()
  set_property(GLOBAL PROPERTY download_requested TRUE)
endfunction()
if(CBM_KFPARTICLE_USE_XPU AND NOT OMIT_PARENT_XPU)
  add_subdirectory("${XPU_FIXTURE}" xpu)
  get_target_property(parent_xpu_source xpu SOURCE_DIR)
  # Check the fixture independently: some KFParticle versions link xpu
  # explicitly as well, which would conceal a no-op attachment stub.
  add_library(XpuAttachContract STATIC EXCLUDE_FROM_ALL "${XPU_FIXTURE}/dummy.cpp")
  xpu_attach(XpuAttachContract)
  get_target_property(attach_links XpuAttachContract LINK_LIBRARIES)
  if(NOT "xpu" IN_LIST attach_links)
    message(FATAL_ERROR "XPU fixture xpu_attach did not link its runtime")
  endif()
endif()
file(WRITE "${CMAKE_BINARY_DIR}/external/KFPARTICLE-prefix/keep.txt" "Unrelated legacy build artifacts.\n")
set(parent_sdk_setting "${KFPARTICLE_INSTALL_SDK}")
add_subdirectory(external)
if(NOT EXISTS "${CMAKE_BINARY_DIR}/external/KFPARTICLE-prefix/keep.txt")
  message(FATAL_ERROR "KFParticle configuration deleted an existing build directory")
endif()
if(NOT "${KFPARTICLE_INSTALL_SDK}" STREQUAL "${parent_sdk_setting}")
  message(FATAL_ERROR "Embedded SDK override changed the parent configuration")
endif()
add_subdirectory("${CBM_SOURCE}/algo/kfp/interface" interface)
get_target_property(interface_type CbmKFParticleOnlineInterface TYPE)
get_target_property(interface_links CbmKFParticleOnlineInterface INTERFACE_LINK_LIBRARIES)
if(NOT interface_type STREQUAL "INTERFACE_LIBRARY" OR NOT interface_links STREQUAL "KFParticle")
  message(FATAL_ERROR "CBMRoot does not reuse the actual KFParticle target")
endif()
get_target_property(interface_definitions CbmKFParticleOnlineInterface INTERFACE_COMPILE_DEFINITIONS)
foreach(definition DO_TPCCATRACKER_EFF_PERFORMANCE NonhomogeneousField CBM USE_TIMERS)
  if(NOT "${definition}" IN_LIST interface_definitions)
    message(FATAL_ERROR "CBMRoot KFParticle ABI definition is missing: ${definition}")
  endif()
endforeach()
if(DROP_PARENT_XPU_LINK)
  set_property(TARGET KFParticle PROPERTY LINK_LIBRARIES "")
endif()
get_target_property(kfp_links KFParticle LINK_LIBRARIES)
if(CBM_KFPARTICLE_USE_XPU AND NOT "xpu" IN_LIST kfp_links)
  message(FATAL_ERROR "parent XPU was not reused; KFParticle links: ${kfp_links}")
endif()
if(CBM_KFPARTICLE_USE_XPU)
  get_target_property(xpu_source xpu SOURCE_DIR)
  get_target_property(xpu_imported xpu IMPORTED)
  if(xpu_imported OR NOT xpu_source STREQUAL parent_xpu_source)
    message(FATAL_ERROR "KFParticle replaced the parent XPU target")
  endif()
endif()
if(EXISTS "${CMAKE_BINARY_DIR}/external/KFParticle/KFParticleConfig.cmake")
  message(FATAL_ERROR "Embedded KFParticle unexpectedly enabled standalone SDK installation")
endif()
get_property(download_requested GLOBAL PROPERTY download_requested)
if(KFPARTICLE_UPDATE_SOURCE AND NOT download_requested)
  message(FATAL_ERROR "explicit source update was ignored")
elseif(NOT KFPARTICLE_UPDATE_SOURCE AND download_requested)
  message(FATAL_ERROR "configure modified an existing checkout")
endif()
CMAKE
for standalone in OFF ON; do
  for gpu in OFF ON; do
    log="$sdk/cbm-$standalone-$gpu.log"
    if ! cmake -S "$sdk/cbm" -B "$sdk/cbm-$standalone-$gpu" \
        -DCMAKE_PREFIX_PATH="$sdk/deps" -DCBM_SOURCE="$source_dir" \
        -DXPU_FIXTURE="$sdk/xpu" -DKFPARTICLE_INSTALL_SDK=ON \
        -DCBM_ONLINE_STANDALONE="$standalone" -DCBM_KFPARTICLE_USE_XPU="$gpu" \
        -DKFPARTICLE_UPDATE_SOURCE="$standalone" >"$log" 2>&1; then
      cat "$log" >&2; exit 1
    fi
  done
done
echo "PASS kfp-build-registration CBMRoot/online CPU/GPU target reuse and patch-free source update"

for failure in OMIT_PARENT_XPU DROP_PARENT_XPU_LINK; do
  failure_log="$sdk/cbm-$failure.log"
  if cmake --log-level=WARNING -S "$sdk/cbm" -B "$sdk/cbm-$failure" \
      -DCMAKE_PREFIX_PATH="$sdk/deps" -DCBM_SOURCE="$source_dir" \
      -DXPU_FIXTURE="$sdk/xpu" -DCBM_KFPARTICLE_USE_XPU=ON "-D$failure=ON" \
      >"$failure_log" 2>&1; then
    echo "FAIL embedded build accepted $failure" >&2; exit 1
  fi
  case "$failure" in
    OMIT_PARENT_XPU) grep -q 'requires the parent xpu target' "$failure_log" ;;
    DROP_PARENT_XPU_LINK) grep -q 'parent XPU was not reused' "$failure_log" ;;
  esac
done
echo "PASS kfp-build-registration missing parent runtime/dependency rejected"

# Exercise the pre-SDK embedded interface, including private-only ABI flags.
mkdir -p "$sdk/embedded-only"
cat >"$sdk/embedded-only/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpEmbeddedOnly LANGUAGES CXX)
add_library(KFParticle SHARED "${XPU_FIXTURE}/dummy.cpp")
set_target_properties(KFParticle PROPERTIES COMPILE_FLAGS
  "-DDO_TPCCATRACKER_EFF_PERFORMANCE -DNonhomogeneousField -DCBM -DUSE_TIMERS")
if(KFPARTICLE_USE_XPU)
  xpu_attach(KFParticle)
endif()
CMAKE
rm "$sdk/cbm/external/KFParticle"
ln -s "$sdk/embedded-only" "$sdk/cbm/external/KFParticle"
for gpu in OFF ON; do
  failure_log="$sdk/embedded-only-$gpu.log"
  if ! cmake -S "$sdk/cbm" -B "$sdk/embedded-only-$gpu" \
      -DCMAKE_PREFIX_PATH="$sdk/deps" -DCBM_SOURCE="$source_dir" \
      -DXPU_FIXTURE="$sdk/xpu" -DCBM_KFPARTICLE_USE_XPU="$gpu" \
      >"$failure_log" 2>&1; then
    cat "$failure_log" >&2; exit 1
  fi
done
echo "PASS kfp-build-registration embedded ABI and XPU checks without SDK support"
exit 0
fi

# Only the explicitly selected standalone-SDK contract needs these files.
for file in CMakeLists.txt cmake/KFParticleSdk.cmake cmake/KFParticleXpu.cmake cmake/KFParticleConfig.cmake.in; do
  if [[ ! -r "$kfp_source/$file" ]]; then
    echo "FAIL incomplete KFParticle SDK build files: $kfp_source/$file" >&2
    exit 1
  fi
done

cat >"$sdk/project/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpSdkGraph LANGUAGES CXX)
add_subdirectory("${KFP_SOURCE}" kfp)
get_property(sdk_option_type CACHE KFPARTICLE_INSTALL_SDK PROPERTY TYPE)
if(NOT sdk_option_type STREQUAL "BOOL")
  message(FATAL_ERROR
    "KFParticle does not provide the requested standalone SDK. SDK support is not required for the cbmroot contract.")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/fixture-prefix.cmake" CONTENT "
file(MAKE_DIRECTORY \"${CMAKE_INSTALL_PREFIX}/lib64/cmake/KFParticle\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/GPU\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/KFPSimd/Utils\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/KFPSimd/SSE\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticlePerformance\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticleTest\")
file(WRITE \"${CMAKE_INSTALL_PREFIX}/lib64/$<TARGET_FILE_NAME:KFParticle>\" \"\")
file(WRITE \"${CMAKE_INSTALL_PREFIX}/lib64/$<TARGET_FILE_NAME:xpu>\" \"\")
file(GLOB_RECURSE exports \"${CMAKE_BINARY_DIR}/kfp/CMakeFiles/Export/KFParticleTargets*.cmake\")
if(NOT exports)
  message(FATAL_ERROR \"KFParticle SDK export files were not generated; check the KFParticle build-file version\")
endif()
foreach(metadata KFParticleConfig.cmake KFParticleConfigVersion.cmake)
  if(NOT EXISTS \"${CMAKE_BINARY_DIR}/kfp/\${metadata}\")
    message(FATAL_ERROR \"KFParticle SDK metadata is missing: ${CMAKE_BINARY_DIR}/kfp/\${metadata}\")
  endif()
endforeach()
file(COPY \${exports} \"${CMAKE_BINARY_DIR}/kfp/KFParticleConfig.cmake\"
  \"${CMAKE_BINARY_DIR}/kfp/KFParticleConfigVersion.cmake\"
  \"${KFP_SOURCE}/cmake/KFParticleXpu.cmake\"
  DESTINATION \"${CMAKE_INSTALL_PREFIX}/lib64/cmake/KFParticle\")
")
CMAKE
sdk_args=(-S "$sdk/project" -DKFP_SOURCE="$kfp_source"
  -DKFPARTICLE_USE_XPU=ON -DFIXTARGET=ON -DKFPARTICLE_INSTALL_SDK=ON
  -DKFPARTICLE_XPU_SOURCE_DIR="$sdk/xpu" -DCMAKE_PREFIX_PATH="$sdk/deps"
  -DXPU_ENABLE_HIP=OFF -DXPU_ENABLE_OPENMP=OFF
  -DCMAKE_INSTALL_LIBDIR=lib64 -DCMAKE_INSTALL_PREFIX="$sdk/prefix")
# Reject an entry point that ignores the requested standalone SDK installation.
mkdir -p "$sdk/legacy"
cat >"$sdk/legacy/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpWithoutSdk LANGUAGES CXX)
add_library(KFParticle SHARED "${CMAKE_CURRENT_SOURCE_DIR}/../xpu/dummy.cpp")
CMAKE
failure_log="$sdk/legacy.log"
if cmake --log-level=WARNING "${sdk_args[@]}" -B "$sdk/legacy-build" \
    -DKFP_SOURCE="$sdk/legacy" >"$failure_log" 2>&1; then
  echo "FAIL SDK check accepted an entry point that ignores SDK options" >&2; exit 1
fi
grep -q 'KFParticle does not provide the requested standalone SDK' "$failure_log"
echo "PASS kfp-build-registration SDK mode rejects missing SDK support"
failure_log="$sdk/build.log"
if ! cmake "${sdk_args[@]}" -B "$sdk/build" \
    >"$sdk/build.log" 2>&1; then
  cat "$sdk/build.log" >&2; exit 1
fi
cmake -P "$sdk/build/fixture-prefix.cmake"
mv "$sdk/prefix" "$sdk/relocated"
cat >"$sdk/consumer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpInstalledContract LANGUAGES CXX)
find_package(KFParticle 0.1 CONFIG REQUIRED COMPONENTS GPU)
if(EXPECT_EXTERNAL)
  get_target_property(xpu_library XPU::xpu IMPORTED_LOCATION)
  string(FIND "${xpu_library}" "${KFPARTICLE_XPU_ROOT}/lib64/" prefix)
  if(NOT prefix EQUAL 0)
    message(FATAL_ERROR "SDK did not reuse the selected external XPU")
  endif()
endif()
get_target_property(includes KFParticle::KFParticle INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(definitions KFParticle::KFParticle INTERFACE_COMPILE_DEFINITIONS)
if(NOT "NonhomogeneousField" IN_LIST definitions OR NOT "KFPARTICLE_USE_XPU" IN_LIST definitions)
  message(FATAL_ERROR "Installed scalar/GPU ABI definitions are missing")
endif()
foreach(directory IN LISTS includes)
  string(FIND "${directory}" "${SDK_PREFIX}/include" prefix)
  if(NOT prefix EQUAL 0)
    message(FATAL_ERROR "Nonrelocatable installed include: ${directory}")
  endif()
endforeach()
if(NOT KFParticle_LIBRARY_DIR STREQUAL "${SDK_PREFIX}/lib64")
  message(FATAL_ERROR "Dependency discovery overwrote the SDK prefix")
endif()
if(REQUIRE_HIP)
  find_package(KFParticle CONFIG REQUIRED COMPONENTS HIP)
endif()
CMAKE
consumer_args=(-S "$sdk/consumer" -DCMAKE_PREFIX_PATH="$sdk/relocated;$sdk/deps"
  -DSDK_PREFIX="$sdk/relocated")
failure_log="$sdk/import.log"
if ! cmake "${consumer_args[@]}" -B "$sdk/import" >"$sdk/import.log" 2>&1; then
  cat "$sdk/import.log" >&2; exit 1
fi
failure_log="$sdk/hip.log"
if cmake "${consumer_args[@]}" -B "$sdk/hip" -DREQUIRE_HIP=ON >"$sdk/hip.log" 2>&1; then
  echo "FAIL CPU-only SDK advertised HIP" >&2; exit 1
fi
failure_log="$sdk/external.log"
if ! cmake "${sdk_args[@]}" -B "$sdk/external" \
    -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" -DCMAKE_INSTALL_PREFIX="$sdk/external-prefix" \
    >"$sdk/external.log" 2>&1; then
  cat "$sdk/external.log" >&2; exit 1
fi
if grep -q 'TYPE SHARED_LIBRARY.*libxpu' "$sdk/external/kfp/cmake_install.cmake"; then
  echo "FAIL SDK installs another copy of external XPU" >&2; exit 1
fi
cmake -P "$sdk/external/fixture-prefix.cmake"
failure_log="$sdk/external-import.log"
if ! cmake -S "$sdk/consumer" -B "$sdk/external-import" \
    -DCMAKE_PREFIX_PATH="$sdk/external-prefix;$sdk/deps" \
    -DSDK_PREFIX="$sdk/external-prefix" -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" \
    -DEXPECT_EXTERNAL=ON >"$sdk/external-import.log" 2>&1; then
  cat "$sdk/external-import.log" >&2; exit 1
fi
failure_log="$sdk/missing-xpu.log"
if cmake "${sdk_args[@]}" -B "$sdk/missing-xpu" -DKFPARTICLE_XPU_ROOT="$sdk/absent" \
    >"$sdk/missing-xpu.log" 2>&1; then
  echo "FAIL SDK accepted a missing XPU installation" >&2; exit 1
fi
grep -q 'XPU headers/library not found' "$sdk/missing-xpu.log"
failure_log="$sdk/missing-backend.log"
if cmake "${sdk_args[@]}" -B "$sdk/missing-backend" \
    -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" -DXPU_ENABLE_HIP=ON \
    >"$sdk/missing-backend.log" 2>&1; then
  echo "FAIL HIP build accepted CPU-only XPU" >&2; exit 1
fi
grep -q 'XPU installation has no HIP backend' "$sdk/missing-backend.log"
mkdir -p "$sdk/rocm/llvm/bin"
touch "$sdk/rocm/llvm/bin/clang++" \
  "$sdk/installed-xpu/lib64/libxpu_Hip.so" "$sdk/installed-xpu/lib64/libxpu_Hip.dylib"
failure_log="$sdk/hip-image.log"
if ! cmake "${sdk_args[@]}" -B "$sdk/hip-image" \
    -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" -DXPU_ENABLE_HIP=ON \
    -DXPU_ROCM_ROOT="$sdk/rocm" '-DXPU_HIP_ARCH=gfx906;gfx908' \
    >"$sdk/hip-image.log" 2>&1; then
  cat "$sdk/hip-image.log" >&2; exit 1
fi
image_source="$sdk/hip-image/kfp/KFParticle_Hip-source"
grep -Fq 'set(GPU_TARGETS "gfx906;gfx908")' "$image_source/CMakeLists.txt"
grep -Fq -- '-std=c++17' "$image_source/CMakeLists.txt"
grep -Fq "$sdk/installed-xpu/include" "$image_source/includes.cmake"
grep -Fq 'execution/KFParticleGpuKernels.cxx' "$image_source/kernels.cxx"
echo "PASS kfp-build-registration standalone SDK ABI, relocation, installed XPU and backend guards"
