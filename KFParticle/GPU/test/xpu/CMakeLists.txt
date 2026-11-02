cmake_minimum_required(VERSION 3.14)
project(KFParticleGpuXpuTest LANGUAGES CXX)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(KFPARTICLE_GPU_TEST_DEVICE "cpu" CACHE STRING "XPU device used by the KFParticle GPU lifecycle test")
set(KFPARTICLE_GPU_TEST_ENV_MODE "standalone" CACHE STRING "Environment mode used by the KFParticle GPU XPU tests")
set(KFPARTICLE_GPU_SOURCE_FINGERPRINT "unversioned" CACHE STRING
  "Content identity used to invalidate copied KFParticle GPU sources")

if(NOT DEFINED XPU_ENABLE_OPENMP)
  set(XPU_ENABLE_OPENMP OFF CACHE BOOL "")
endif()
if(NOT DEFINED XPU_ENABLE_CUDA)
  set(XPU_ENABLE_CUDA OFF CACHE BOOL "")
endif()
if(NOT DEFINED XPU_ENABLE_HIP)
  set(XPU_ENABLE_HIP OFF CACHE BOOL "")
endif()
if(NOT DEFINED XPU_ENABLE_SYCL)
  set(XPU_ENABLE_SYCL OFF CACHE BOOL "")
endif()
set(XPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

get_filename_component(KFPARTICLE_GPU_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT DEFINED XPU_SOURCE_DIR)
  get_filename_component(XPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../xpu" ABSOLUTE)
endif()
if(KFPARTICLE_XPU_ROOT)
  include(GNUInstallDirs)
  include("${KFPARTICLE_GPU_DIR}/../../cmake/KFParticleXpu.cmake")
  kfparticle_configure_xpu()
  set(xpu_test_library_dir "${KFPARTICLE_XPU_LIBRARY_DIR}")
else()
  get_filename_component(XPU_SOURCE_DIR "${XPU_SOURCE_DIR}" ABSOLUTE)
  add_subdirectory("${XPU_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/xpu")
  set(xpu_test_library_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

option(KFPARTICLE_GPU_FIT_UNROLL "Enable explicit HIP fitting-loop unroll hints" ON)

add_library(KFParticleGpuXpuTestSupport SHARED
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuBufferManager.cxx"
  "${KFPARTICLE_GPU_DIR}/planning/KFParticleGpuCpuChannelCatalogue.cxx"
  "${KFPARTICLE_GPU_DIR}/planning/KFParticleGpuDecayGraphPlan.cxx"
  "${KFPARTICLE_GPU_DIR}/planning/KFParticleGpuDecayPlan.cxx"
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuKernels.cxx"
  "${KFPARTICLE_GPU_DIR}/planning/KFParticleGpuRoutingPlan.cxx"
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuRuntime.cxx"
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuRuntimeService.cxx"
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuSteering.cxx"
)
set_target_properties(KFParticleGpuXpuTestSupport PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_definitions(KFParticleGpuXpuTestSupport
  PUBLIC KFPARTICLE_USE_XPU
  PRIVATE KFPARTICLE_GPU_SOURCE_FINGERPRINT="${KFPARTICLE_GPU_SOURCE_FINGERPRINT}")
target_include_directories(KFParticleGpuXpuTestSupport PUBLIC "${KFPARTICLE_GPU_DIR}")
target_compile_options(KFParticleGpuXpuTestSupport PRIVATE -Wall -Wextra -Werror)
target_link_libraries(KFParticleGpuXpuTestSupport PUBLIC xpu)
if(KFPARTICLE_GPU_FIT_UNROLL)
  set(KFPARTICLE_GPU_FIT_UNROLL_VALUE 1)
else()
  set(KFPARTICLE_GPU_FIT_UNROLL_VALUE 0)
endif()
set(KFPARTICLE_GPU_TIMING_BUILD_ID "${KFPARTICLE_GPU_SOURCE_FINGERPRINT}")
configure_file(${KFPARTICLE_GPU_DIR}/KFParticleGpuBuildOptions.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/gpu-config/KFParticleGpuBuildOptions.h @ONLY)
target_include_directories(KFParticleGpuXpuTestSupport BEFORE PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/gpu-config)
xpu_attach(KFParticleGpuXpuTestSupport
  "${KFPARTICLE_GPU_DIR}/execution/KFParticleGpuKernels.cxx"
)

add_executable(KFParticleGpuXpuLifecycleTest
  "${KFPARTICLE_GPU_DIR}/test/KFParticleGpuXpuLifecycleTest.cxx"
)
set_target_properties(KFParticleGpuXpuLifecycleTest PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_options(KFParticleGpuXpuLifecycleTest PRIVATE -Wall -Wextra -Werror -UNDEBUG)
target_compile_definitions(KFParticleGpuXpuLifecycleTest
  PRIVATE
    KFPARTICLE_GPU_TEST_DEVICE="${KFPARTICLE_GPU_TEST_DEVICE}"
    KFPARTICLE_GPU_SOURCE_FINGERPRINT="${KFPARTICLE_GPU_SOURCE_FINGERPRINT}")
target_link_libraries(KFParticleGpuXpuLifecycleTest
  PRIVATE xpu KFParticleGpuXpuTestSupport Threads::Threads)

# This is an opt-in evidence tool rather than a fast CTest regression.  It
# shares the lifecycle test's XPU image and hermetic runner, but reports
# repeatable serial-versus-batch timings for controlled default-V0 input.
add_executable(KFParticleGpuXpuBatchBenchmark
  "${KFPARTICLE_GPU_DIR}/test/KFParticleGpuXpuBatchBenchmark.cxx"
)
set_target_properties(KFParticleGpuXpuBatchBenchmark PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_options(KFParticleGpuXpuBatchBenchmark PRIVATE -Wall -Wextra -Werror -UNDEBUG)
target_compile_definitions(KFParticleGpuXpuBatchBenchmark
  PRIVATE
    KFPARTICLE_GPU_TEST_DEVICE="${KFPARTICLE_GPU_TEST_DEVICE}"
    KFPARTICLE_GPU_SOURCE_FINGERPRINT="${KFPARTICLE_GPU_SOURCE_FINGERPRINT}")
target_link_libraries(KFParticleGpuXpuBatchBenchmark
  PRIVATE xpu KFParticleGpuXpuTestSupport)

enable_testing()
add_subdirectory("${KFPARTICLE_GPU_DIR}/test" "${CMAKE_CURRENT_BINARY_DIR}/host-tests")
add_executable(KFParticleGpuStepTraceTest "${KFPARTICLE_GPU_DIR}/test/KFParticleGpuStepTraceTest.cxx")
target_include_directories(KFParticleGpuStepTraceTest PRIVATE "${KFPARTICLE_GPU_DIR}")
target_compile_options(KFParticleGpuStepTraceTest PRIVATE -Wall -Wextra -Werror -UNDEBUG)
target_link_libraries(KFParticleGpuStepTraceTest PRIVATE xpu)
add_test(NAME KFParticleGpuStepTraceTest COMMAND KFParticleGpuStepTraceTest)
set_tests_properties(KFParticleGpuStepTraceTest PROPERTIES LABELS "kfp;host" TIMEOUT 30)
set(KFPARTICLE_GPU_XPU_TEST_ENV_SCRIPT
  "${CMAKE_CURRENT_LIST_DIR}/run_kfparticle_xpu_test_env.sh")

add_test(NAME KFParticleGpuXpuLifecycleTest
  COMMAND bash "${KFPARTICLE_GPU_XPU_TEST_ENV_SCRIPT}"
    "$<TARGET_FILE:KFParticleGpuXpuLifecycleTest>")
set_tests_properties(KFParticleGpuXpuLifecycleTest
  PROPERTIES LABELS "kfp;xpu" TIMEOUT 180 ENVIRONMENT
    "KFPARTICLE_GPU_TEST_LIBRARY_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};KFPARTICLE_GPU_XPU_LIBRARY_DIR=${xpu_test_library_dir};KFPARTICLE_GPU_TEST_DEVICE=${KFPARTICLE_GPU_TEST_DEVICE};KFPARTICLE_GPU_TEST_ENV_MODE=${KFPARTICLE_GPU_TEST_ENV_MODE}")
