cmake_minimum_required(VERSION 3.20)
project(KFParticleGpuExample LANGUAGES CXX)

if(NOT TARGET KFParticle::KFParticle)
  find_package(KFParticle CONFIG REQUIRED COMPONENTS GPU)
endif()
add_executable(KFParticleGpuExample KFParticleGpuExample.cxx ExampleData.cxx ExampleQa.cxx)
target_link_libraries(KFParticleGpuExample PRIVATE KFParticle::KFParticle)
target_compile_features(KFParticleGpuExample PRIVATE cxx_std_17)

include(CTest)
if(BUILD_TESTING)
  set(KFPARTICLE_EXAMPLE_TEST_DEVICE "cpu" CACHE STRING "Additional example smoke-test device")
  set(devices cpu)
  if(NOT KFPARTICLE_EXAMPLE_TEST_DEVICE STREQUAL "cpu")
    list(APPEND devices "${KFPARTICLE_EXAMPLE_TEST_DEVICE}")
  endif()
  foreach(device IN LISTS devices)
    set(work "${CMAKE_CURRENT_BINARY_DIR}/${device}")
    file(MAKE_DIRECTORY "${work}")
    set(test "KfpExample-${device}")
    add_test(NAME "${test}" COMMAND KFParticleGpuExample "${device}")
    set_tests_properties("${test}" PROPERTIES
      WORKING_DIRECTORY "${work}" LABELS "kfp;example" RUN_SERIAL TRUE TIMEOUT 180
      PASS_REGULAR_EXPRESSION "Wrote kfparticle-qa.root"
      ENVIRONMENT "XPU_DEVICE=${device};OMP_NUM_THREADS=2;OMP_DYNAMIC=FALSE;XPU_VERBOSE=0;XPU_PROFILE=0;KFPARTICLE_GPU_CONSTRUCTION_TIMERS=0;KFPARTICLE_GPU_TRANSACTION_STAGE_FENCE=0")
  endforeach()
endif()
