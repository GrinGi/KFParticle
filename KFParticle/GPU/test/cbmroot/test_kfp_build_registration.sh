#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "$here/../../../../../.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
trap 'echo "FAIL kfp-build-registration at line $LINENO" >&2' ERR

gpu_dir="$(cd "$here/../.." && pwd)"
adapter_dir="$source_dir/algo/kfp/gpu"
for retired in KfpGpuMaterializationAdapter.h KfpGpuMaterializationAdapter.cxx \
    KfpGpuDiagnosticStatus.h KfpGpuBatchPolicy.h KfpGpuBatchPolicy.cxx \
    KfpGpuPerformanceReport.h KfpGpuPerformanceReport.cxx \
    KfpGpuInputPacker.h KfpGpuInputPacker.cxx KfpGpuBatch.h KfpGpuBatch.cxx \
    KfpGpuExecutor.h KfpGpuExecutor.cxx KfpGpuRuntimeService.h KfpGpuRuntimeService.cxx \
    KfpGpuMaterializationSchedule.h; do
  if [[ -e "$adapter_dir/$retired" ]]; then
    echo "FAIL stale adapter source: $adapter_dir/$retired" >&2
    exit 1
  fi
done
for stale in "$adapter_dir"/KfpGpuDiagnostic*.h "$adapter_dir"/KfpGpuDiagnostic*.cxx \
    "$adapter_dir/test/_GTestKfpGpuDiagnosticPacker.cxx" \
    "$adapter_dir/test/_GTestKfpGpuInputPacker.cxx" \
    "$adapter_dir/test/_GTestKfpGpuBatch.cxx" \
    "$adapter_dir/test/_GTestKfpGpuRunner.cxx"; do
  if [[ -e "$stale" ]]; then
    echo "FAIL stale adapter source: $stale" >&2
    exit 1
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
# Compile its ROOT-free trace declaration in isolation; do not initialize XPU.
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
  # Include Trace first, without a transitive declaration of the moved result.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/trace-header.cxx"
    "#include \"${ADAPTER_DIR}/KfpGpuTrace.h\"\n"
    "#include \"${ADAPTER_DIR}/KfpGpuTrace.h\"\n"
    "#include <type_traits>\n"
    "static_assert(std::is_pointer_v<decltype(&cbm::algo::kfp::TraceFirstGpuMismatch)>);\n"
    "int main() { return 0; }\n")
  try_compile(trace_header_compiles "${CMAKE_CURRENT_BINARY_DIR}/trace-header-check"
    "${CMAKE_CURRENT_BINARY_DIR}/trace-header.cxx"
    CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=17" OUTPUT_VARIABLE trace_header_output)
  if(NOT trace_header_compiles)
    message(FATAL_ERROR "trace header is not self-contained:\n${trace_header_output}")
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
  foreach(component Comparison Materialization Policy Report Executor)
    if(NOT "test/_GTestKfpGpu${component}.cxx" IN_LIST sources)
      message(FATAL_ERROR "adapter coverage source is missing: ${component}")
    endif()
  endforeach()
  foreach(component InputPacker Batch)
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

# Export/import graph only: dependency fixtures below are not numerical tests.
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
endfunction()
CMAKE
touch "$sdk/xpu/dummy.cpp" "$sdk/xpu/src/xpu/host.h" "$sdk/xpu/LICENSE"
cat >"$sdk/project/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(KfpSdkGraph LANGUAGES CXX)
if(PARENT_XPU)
  add_subdirectory("${KFPARTICLE_XPU_SOURCE_DIR}" parent-xpu)
  set(KFPARTICLE_CBMROOT_EMBEDDED ON)
endif()
add_subdirectory("${KFP_SOURCE}" kfp)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/fixture-prefix.cmake" CONTENT "
file(MAKE_DIRECTORY \"${CMAKE_INSTALL_PREFIX}/lib64/cmake/KFParticle\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/GPU\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/KFPSimd/Utils\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticle/KFPSimd/SSE\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticlePerformance\"
  \"${CMAKE_INSTALL_PREFIX}/include/KFParticleTest\")
file(WRITE \"${CMAKE_INSTALL_PREFIX}/lib64/$<TARGET_FILE_NAME:KFParticle>\" \"\")
file(WRITE \"${CMAKE_INSTALL_PREFIX}/lib64/$<TARGET_FILE_NAME:xpu>\" \"\")
file(GLOB exports \"${CMAKE_BINARY_DIR}/kfp/CMakeFiles/Export/*/KFParticleTargets*.cmake\")
file(COPY \${exports} \"${CMAKE_BINARY_DIR}/kfp/KFParticleConfig.cmake\"
  \"${CMAKE_BINARY_DIR}/kfp/KFParticleConfigVersion.cmake\"
  \"${KFP_SOURCE}/cmake/KFParticleXpu.cmake\"
  DESTINATION \"${CMAKE_INSTALL_PREFIX}/lib64/cmake/KFParticle\")
")
CMAKE
kfp_source="$(cd "$gpu_dir/../.." && pwd)"
sdk_args=(-S "$sdk/project" -DKFP_SOURCE="$kfp_source"
  -DKFPARTICLE_USE_XPU=ON -DFIXTARGET=ON -DKFPARTICLE_INSTALL_SDK=ON
  -DKFPARTICLE_XPU_SOURCE_DIR="$sdk/xpu" -DCMAKE_PREFIX_PATH="$sdk/deps"
  -DXPU_ENABLE_HIP=OFF -DXPU_ENABLE_OPENMP=OFF
  -DCMAKE_INSTALL_LIBDIR=lib64 -DCMAKE_INSTALL_PREFIX="$sdk/prefix")
if ! cmake "${sdk_args[@]}" -B "$sdk/embedded" -DPARENT_XPU=ON \
    -DKFPARTICLE_INSTALL_SDK=OFF >"$sdk/embedded.log" 2>&1; then
  cat "$sdk/embedded.log" >&2; exit 1
fi
grep -q 'reusing application-owned XPU' "$sdk/embedded.log"
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
if ! cmake "${consumer_args[@]}" -B "$sdk/import" >"$sdk/import.log" 2>&1; then
  cat "$sdk/import.log" >&2; exit 1
fi
if cmake "${consumer_args[@]}" -B "$sdk/hip" -DREQUIRE_HIP=ON >"$sdk/hip.log" 2>&1; then
  echo "FAIL CPU-only SDK advertised HIP" >&2; exit 1
fi
if ! cmake "${sdk_args[@]}" -B "$sdk/external" \
    -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" -DCMAKE_INSTALL_PREFIX="$sdk/external-prefix" \
    >"$sdk/external.log" 2>&1; then
  cat "$sdk/external.log" >&2; exit 1
fi
if grep -q 'TYPE SHARED_LIBRARY.*libxpu' "$sdk/external/kfp/cmake_install.cmake"; then
  echo "FAIL SDK installs another copy of external XPU" >&2; exit 1
fi
cmake -P "$sdk/external/fixture-prefix.cmake"
if ! cmake -S "$sdk/consumer" -B "$sdk/external-import" \
    -DCMAKE_PREFIX_PATH="$sdk/external-prefix;$sdk/deps" \
    -DSDK_PREFIX="$sdk/external-prefix" -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" \
    -DEXPECT_EXTERNAL=ON >"$sdk/external-import.log" 2>&1; then
  cat "$sdk/external-import.log" >&2; exit 1
fi
if cmake "${sdk_args[@]}" -B "$sdk/missing-xpu" -DKFPARTICLE_XPU_ROOT="$sdk/absent" \
    >"$sdk/missing-xpu.log" 2>&1; then
  echo "FAIL SDK accepted a missing XPU installation" >&2; exit 1
fi
grep -q 'XPU headers/library not found' "$sdk/missing-xpu.log"
if cmake "${sdk_args[@]}" -B "$sdk/missing-backend" \
    -DKFPARTICLE_XPU_ROOT="$sdk/installed-xpu" -DXPU_ENABLE_HIP=ON \
    >"$sdk/missing-backend.log" 2>&1; then
  echo "FAIL HIP build accepted CPU-only XPU" >&2; exit 1
fi
grep -q 'XPU installation has no HIP backend' "$sdk/missing-backend.log"
mkdir -p "$sdk/rocm/llvm/bin"
touch "$sdk/rocm/llvm/bin/clang++" \
  "$sdk/installed-xpu/lib64/libxpu_Hip.so" "$sdk/installed-xpu/lib64/libxpu_Hip.dylib"
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
echo "PASS kfp-build-registration parent/installed XPU reuse, SDK ABI, lib64 import and dependency guards"
