cmake_minimum_required(VERSION 3.14)
project(KFParticleGpuFinderConsumer LANGUAGES CXX)

# Both modes are independent of CBMRoot. Installed mode must not access sources.
option(KFPARTICLE_GPU_TEST_INSTALLED "Consume only the installed find_package SDK" OFF)
if(KFPARTICLE_GPU_TEST_INSTALLED)
  find_package(KFParticle CONFIG REQUIRED COMPONENTS GPU)
else()
  get_filename_component(kfparticle_source "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
  set(KFPARTICLE_USE_XPU ON CACHE BOOL "Build the GPU Finder" FORCE)
  set(FIXTARGET ON CACHE BOOL "The current GPU host-input field model" FORCE)
  set(XPU_BUILD_TESTS OFF CACHE BOOL "Do not build XPU's separate test suite" FORCE)
  set(XPU_BUILD_EXAMPLES OFF CACHE BOOL "Do not build XPU's separate examples" FORCE)
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(KFPARTICLE_GPU_TEST_DEVICE "cpu" CACHE STRING "Additional device to test, e.g. hip1")
if(KFPARTICLE_GPU_TEST_INSTALLED AND KFPARTICLE_GPU_TEST_DEVICE MATCHES "^hip"
    AND NOT KFParticle_HIP_FOUND)
  message(FATAL_ERROR "The installed SDK was not built with HIP")
endif()

if(NOT KFPARTICLE_GPU_TEST_INSTALLED)
  add_subdirectory("${kfparticle_source}" kfparticle)
endif()
add_executable(KFParticleGpuFinderTest KFParticleGpuFinderTest.cxx)
find_package(Threads REQUIRED)
target_link_libraries(KFParticleGpuFinderTest PRIVATE KFParticle::KFParticle Threads::Threads)

enable_testing()
if(NOT KFPARTICLE_GPU_TEST_INSTALLED)
  find_package(GTest QUIET)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." host-tests)
endif()
if(KFPARTICLE_GPU_TEST_INSTALLED)
  set(finder_library_dir "${KFParticle_LIBRARY_DIR}")
  set(finder_xpu_dir "${KFParticle_XPU_LIBRARY_DIR}")
else()
  set(finder_library_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  get_target_property(xpu_imported xpu IMPORTED)
  if(xpu_imported)
    get_target_property(xpu_library xpu IMPORTED_LOCATION)
    get_filename_component(finder_xpu_dir "${xpu_library}" DIRECTORY)
  else()
    set(finder_xpu_dir "${finder_library_dir}")
  endif()
endif()
set(finder_test_environment
  "LD_LIBRARY_PATH=${finder_library_dir}:${finder_xpu_dir}:$ENV{LD_LIBRARY_PATH}"
  "OMP_NUM_THREADS=2" "OMP_DYNAMIC=FALSE" "XPU_VERBOSE=0" "XPU_PROFILE=0"
  "KFPARTICLE_GPU_CONSTRUCTION_TIMERS=0" "KFPARTICLE_GPU_DEVICE_PROFILE=0"
  "KFPARTICLE_GPU_TRANSACTION_STAGE_FENCE=0" "KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS=0")
add_test(NAME KFParticleGpuFinderCpu COMMAND KFParticleGpuFinderTest cpu)
set_tests_properties(KFParticleGpuFinderCpu PROPERTIES LABELS "kfp;standalone" TIMEOUT 180
  ENVIRONMENT "${finder_test_environment};XPU_DEVICE=cpu")
if(NOT KFPARTICLE_GPU_TEST_DEVICE STREQUAL "cpu")
  add_test(NAME KFParticleGpuFinderDevice
    COMMAND KFParticleGpuFinderTest "${KFPARTICLE_GPU_TEST_DEVICE}")
  set_tests_properties(KFParticleGpuFinderDevice PROPERTIES LABELS "kfp;standalone" TIMEOUT 180
    ENVIRONMENT "${finder_test_environment};XPU_DEVICE=${KFPARTICLE_GPU_TEST_DEVICE}")
endif()
