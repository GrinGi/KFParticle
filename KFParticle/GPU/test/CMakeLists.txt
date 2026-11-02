cmake_minimum_required(VERSION 3.14)
project(KFParticleGpuHostTests LANGUAGES CXX)

include(CTest)
if(NOT BUILD_TESTING)
  return()
endif()

# Compile shared layout/transfer fixtures once; keep each contract visible in CTest.
add_executable(KFParticleGpuDataTest KFParticleGpuDataTest.cxx)
target_compile_features(KFParticleGpuDataTest PRIVATE cxx_std_17)
target_include_directories(KFParticleGpuDataTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
# Compile core headers independently with the GPU root as the only project
# include path. Do not let sibling include directories hide a broken moved include.
set(gpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
file(GLOB headers CONFIGURE_DEPENDS RELATIVE "${gpu_dir}"
  "${gpu_dir}/*.h" "${gpu_dir}/data/*.h" "${gpu_dir}/planning/*.h"
  "${gpu_dir}/execution/*.h" "${gpu_dir}/algorithms/*.h" "${gpu_dir}/diagnostics/*.h")
foreach(header IN LISTS headers)
  # The high-level executor returns KFParticle objects and needs the full library.
  # Its header is exercised by the adapter/CPU-GPU regression, not ROOT-free tests.
  if(header STREQUAL "execution/KFParticleGpuExecutor.h"
      OR header STREQUAL "execution/KFParticleGpuFinder.h")
    continue()
  endif()
  get_filename_component(component "${header}" NAME_WE)
  string(REGEX REPLACE "^KFParticleGpu" "" component "${component}")
  set(check "${CMAKE_CURRENT_BINARY_DIR}/header-checks/${component}.cxx")
  # Refresh old flat includes, but preserve mtimes for unchanged checks. Create
  # during configure: the XPU target consumes these sources from a parent scope.
  set(content "#include \"${header}\"\n")
  set(previous "")
  if(EXISTS "${check}")
    file(READ "${check}" previous)
  endif()
  if(NOT previous STREQUAL content)
    file(WRITE "${check}" "${content}")
  endif()
  if(NOT component STREQUAL "StepTrace")
    target_sources(KFParticleGpuDataTest PRIVATE "${check}")
  endif()
  if(TARGET KFParticleGpuXpuTestSupport)
    target_sources(KFParticleGpuXpuTestSupport PRIVATE "${check}")
  endif()
endforeach()

# Shared full-library host checks: usable by CBMRoot and an independent consumer.
if(TARGET KFParticle AND TARGET GTest::gtest_main)
  add_executable(KFParticleGpuHostInterfaceTest
    standalone/KFParticleGpuInputPackerTest.cxx
    standalone/KFParticleGpuBatchTest.cxx)
  target_link_libraries(KFParticleGpuHostInterfaceTest PRIVATE KFParticle GTest::gtest_main)
  add_test(NAME KFParticleGpuHostInterfaceTest
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
      $<TARGET_FILE:KFParticleGpuHostInterfaceTest>)
  set_tests_properties(KFParticleGpuHostInterfaceTest PROPERTIES LABELS "kfp;host" TIMEOUT 120)
endif()
if(MSVC)
  target_compile_options(KFParticleGpuDataTest PRIVATE /UNDEBUG)
else()
  target_compile_options(KFParticleGpuDataTest PRIVATE -Wall -Wextra -Werror -UNDEBUG)
endif()
foreach(component State SoAView InputData CandidatePool)
  set(test KFParticleGpu${component}Test)
  add_test(NAME ${test} COMMAND KFParticleGpuDataTest ${component})
  set_tests_properties(${test} PROPERTIES LABELS "kfp;host" TIMEOUT 30)
endforeach()
