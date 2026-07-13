cmake_minimum_required(VERSION 3.14)
project(KFParticleGpuXpuTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(KFPARTICLE_GPU_TEST_DEVICE "cpu" CACHE STRING "XPU device used by the KFParticle GPU lifecycle test")
set(KFPARTICLE_GPU_TEST_ENV_MODE "standalone" CACHE STRING "Environment mode used by the KFParticle GPU XPU tests")
set(KFPARTICLE_GPU_TEST_TRACE OFF CACHE BOOL "Print verbose KFParticle GPU steering trace in XPU tests")

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
get_filename_component(XPU_SOURCE_DIR "${XPU_SOURCE_DIR}" ABSOLUTE)
add_subdirectory("${XPU_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/xpu")

add_library(KFParticleGpuXpuBaseline SHARED
  "${CMAKE_CURRENT_LIST_DIR}/KFParticleGpuXpuBaseline.cxx"
)
set_target_properties(KFParticleGpuXpuBaseline PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_include_directories(KFParticleGpuXpuBaseline PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
xpu_attach(KFParticleGpuXpuBaseline
  "${CMAKE_CURRENT_LIST_DIR}/KFParticleGpuXpuBaseline.cxx"
)

add_executable(KFParticleGpuXpuBaselineTest
  "${CMAKE_CURRENT_LIST_DIR}/KFParticleGpuXpuBaselineTest.cxx"
)
set_target_properties(KFParticleGpuXpuBaselineTest PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_options(KFParticleGpuXpuBaselineTest PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(KFParticleGpuXpuBaselineTest
  PRIVATE KFPARTICLE_GPU_TEST_DEVICE="${KFPARTICLE_GPU_TEST_DEVICE}")
target_link_libraries(KFParticleGpuXpuBaselineTest PRIVATE xpu KFParticleGpuXpuBaseline)

add_library(KFParticleGpuXpuTestSupport SHARED
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuBufferManager.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuDecayPlan.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuDeviceImage.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuKernels.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuRuntime.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuSteering.cxx"
)
set_target_properties(KFParticleGpuXpuTestSupport PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_definitions(KFParticleGpuXpuTestSupport
  PUBLIC KFPARTICLE_USE_XPU
  PRIVATE KFPARTICLE_GPU_USE_KERNEL_ARGS)
if(KFPARTICLE_GPU_TEST_TRACE)
  target_compile_definitions(KFParticleGpuXpuTestSupport PRIVATE KFPARTICLE_GPU_TRACE)
endif()
target_include_directories(KFParticleGpuXpuTestSupport PUBLIC "${KFPARTICLE_GPU_DIR}")
target_compile_options(KFParticleGpuXpuTestSupport PRIVATE -Wall -Wextra -Werror)
target_link_libraries(KFParticleGpuXpuTestSupport PUBLIC xpu)
xpu_attach(KFParticleGpuXpuTestSupport
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuDeviceImage.cxx"
  "${KFPARTICLE_GPU_DIR}/KFParticleGpuKernels.cxx"
)

add_executable(KFParticleGpuXpuLifecycleTest
  "${KFPARTICLE_GPU_DIR}/test/KFParticleGpuXpuLifecycleTest.cxx"
)
set_target_properties(KFParticleGpuXpuLifecycleTest PROPERTIES
  BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
target_compile_options(KFParticleGpuXpuLifecycleTest PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(KFParticleGpuXpuLifecycleTest
  PRIVATE KFPARTICLE_GPU_TEST_DEVICE="${KFPARTICLE_GPU_TEST_DEVICE}")
target_link_libraries(KFParticleGpuXpuLifecycleTest
  PRIVATE xpu KFParticleGpuXpuBaseline KFParticleGpuXpuTestSupport)

enable_testing()
set(KFPARTICLE_GPU_XPU_TEST_ENV_SCRIPT
  "${CMAKE_CURRENT_LIST_DIR}/run_kfparticle_xpu_test_env.sh")

add_test(NAME KFParticleGpuXpuBaselineTest
  COMMAND bash "${KFPARTICLE_GPU_XPU_TEST_ENV_SCRIPT}"
    "$<TARGET_FILE:KFParticleGpuXpuBaselineTest>")
set_tests_properties(KFParticleGpuXpuBaselineTest
  PROPERTIES ENVIRONMENT
    "KFPARTICLE_GPU_TEST_LIBRARY_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};KFPARTICLE_GPU_XPU_LIBRARY_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};KFPARTICLE_GPU_TEST_DEVICE=${KFPARTICLE_GPU_TEST_DEVICE};KFPARTICLE_GPU_TEST_ENV_MODE=${KFPARTICLE_GPU_TEST_ENV_MODE}")

add_test(NAME KFParticleGpuXpuLifecycleTest
  COMMAND bash "${KFPARTICLE_GPU_XPU_TEST_ENV_SCRIPT}"
    "$<TARGET_FILE:KFParticleGpuXpuLifecycleTest>")
set_tests_properties(KFParticleGpuXpuLifecycleTest
  PROPERTIES ENVIRONMENT
    "KFPARTICLE_GPU_TEST_LIBRARY_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};KFPARTICLE_GPU_XPU_LIBRARY_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};KFPARTICLE_GPU_TEST_DEVICE=${KFPARTICLE_GPU_TEST_DEVICE};KFPARTICLE_GPU_TEST_ENV_MODE=${KFPARTICLE_GPU_TEST_ENV_MODE}")
