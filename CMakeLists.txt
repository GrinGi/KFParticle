cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)

project(KFParticleLibrary)

add_compile_options(-O3 -Wall -Wextra -Wsign-promo -Wctor-dtor-privacy -Wreorder -Wno-deprecated -Wno-parentheses)
# add_compile_options(-Weffc++ -Wnon-virtual-dtor -Woverloaded-virtual -Wold-style-cast) # for the future
If(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-pmf-conversions -Wstrict-null-sentinel -Wno-non-template-friend -Wmissing-field-initializers)
EndIf()


if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
message(STATUS "===== KFParticle project is built standalone ======")
  add_compile_options(-msse4.2)
else()
  message(STATUS "===== KFParticle project is embedded, use SIMD flags from the host ======")
endif()


macro(add_target_property _target _prop _value)
  get_target_property(_oldprop "${_target}" ${_prop})
  if(NOT _oldprop)
    set_target_properties("${_target}" PROPERTIES ${_prop} "${_value}")
  else(NOT _oldprop)
    set_target_properties("${_target}" PROPERTIES ${_prop} "${_oldprop} ${_value}")
  endif(NOT _oldprop)
endmacro(add_target_property)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory (externals)

option(KFPARTICLE_USE_XPU "Build the KFParticle XPU infrastructure." OFF)
set(KFPARTICLE_XPU_SOURCE_DIR "" CACHE PATH "Optional XPU source tree for standalone KFParticle builds")
set(XPU_SOURCE_DIR "" CACHE PATH "Legacy alias for KFPARTICLE_XPU_SOURCE_DIR")
set(XPU_INCLUDE_DIR "" CACHE PATH "Optional legacy XPU include directory")

if(KFPARTICLE_USE_XPU)
  if(NOT KFPARTICLE_XPU_SOURCE_DIR AND XPU_SOURCE_DIR)
    set(KFPARTICLE_XPU_SOURCE_DIR "${XPU_SOURCE_DIR}")
  endif()

  if(NOT CMAKE_VERSION VERSION_LESS "3.1")
    if(NOT DEFINED CMAKE_CXX_STANDARD)
      set(CMAKE_CXX_STANDARD 17)
    elseif(CMAKE_CXX_STANDARD LESS 17)
      set(CMAKE_CXX_STANDARD 17)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()

  if(NOT TARGET xpu)
    if(EXISTS "${KFPARTICLE_XPU_SOURCE_DIR}/CMakeLists.txt")
      # KFParticle is often built as an ExternalProject, where the parent xpu
      # target is not visible. Attach the same XPU source tree locally instead.
      add_subdirectory("${KFPARTICLE_XPU_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/xpu")
    else()
      message(FATAL_ERROR
        "KFPARTICLE_USE_XPU requires an existing xpu target or "
        "KFPARTICLE_XPU_SOURCE_DIR/XPU_SOURCE_DIR")
    endif()
  endif()

  if(XPU_INCLUDE_DIR)
    list(APPEND KFPARTICLE_XPU_INCLUDE_DIRS "${XPU_INCLUDE_DIR}")
  elseif(KFPARTICLE_XPU_SOURCE_DIR)
    list(APPEND KFPARTICLE_XPU_INCLUDE_DIRS "${KFPARTICLE_XPU_SOURCE_DIR}/src")
  endif()

  if(KFPARTICLE_XPU_INCLUDE_DIRS)
    foreach(KFPARTICLE_XPU_INCLUDE_DIR ${KFPARTICLE_XPU_INCLUDE_DIRS})
      if(NOT EXISTS "${KFPARTICLE_XPU_INCLUDE_DIR}/xpu/host.h")
        message(FATAL_ERROR
          "KFParticle XPU include directory is invalid: ${KFPARTICLE_XPU_INCLUDE_DIR}")
      endif()
    endforeach()
  else()
    message(FATAL_ERROR
      "KFPARTICLE_USE_XPU could not determine the XPU include directory")
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS} ${ROOTSYS})

set(FIXTARGET FALSE CACHE BOOL "Compile for fix target geometry.")

find_package(ROOT REQUIRED COMPONENTS Core Hist Matrix Physics EG Gpad Graf Graf3d RIO MathCore)

Set(INCLUDE_DIRECTORIES
  ${PROJECT_SOURCE_DIR}/KFParticle
  ${PROJECT_SOURCE_DIR}/KFParticlePerformance
  ${PROJECT_SOURCE_DIR}/KFParticleTest
  ${PROJECT_SOURCE_DIR}/externals/vir-simd
)

include(${ROOT_USE_FILE})
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
Include_Directories( ${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} )
if(KFPARTICLE_USE_XPU)
  Include_Directories(${KFPARTICLE_XPU_INCLUDE_DIRS})
endif()
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE OFF)

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set (SOURCES
  KFParticle/KFParticleTopoReconstructor.cxx
  KFParticle/KFVertex.cxx
  KFParticle/KFPTrack.cxx
  KFParticle/KFPTrackVector.cxx
  KFParticle/KFPVertex.cxx
  KFParticle/KFParticlePVReconstructor.cxx
  KFParticle/KFParticleDatabase.cxx
#  KFParticle/KFParticleBase.cxx
#  KFParticle/KFParticleBaseSIMD.cxx
  KFParticle/KFParticle.cxx
  KFParticle/KFParticleSIMD.cxx
  KFParticle/KFParticleFinder.cxx
  KFParticle/KFPEmcCluster.cxx
  KFParticlePerformance/KFMCVertex.cxx
  KFParticlePerformance/KFParticlePerformanceBase.cxx
  KFParticlePerformance/KFTopoPerformance.cxx
  KFParticlePerformance/KFMCParticle.cxx
  KFParticleTest/KFParticleTest.cxx
)

if(KFPARTICLE_USE_XPU)
  list(APPEND SOURCES
    KFParticle/GPU/KFParticleGpuBufferManager.cxx
    KFParticle/GPU/KFParticleGpuDecayPlan.cxx
    KFParticle/GPU/KFParticleGpuDeviceImage.cxx
    KFParticle/GPU/KFParticleGpuKernels.cxx
    KFParticle/GPU/KFParticleGpuRuntime.cxx
    KFParticle/GPU/KFParticleGpuSteering.cxx
  )
endif()

set (HEADERS
#  KFParticle/KFParticleBase.h
  KFParticle/KFParticle.h
  KFParticle/KFVertex.h
  KFParticlePerformance/KFPartEfficiencies.h
  KFParticlePerformance/KFMCParticle.h
  KFParticleTest/KFParticleTest.h
)

Message("CMAKE_BUILD_DIRECTORY: ${CMAKE_BUILD_DIRECTORY}")

if(FIXTARGET)

  find_program(BIN_SED sed)
  if(BIN_SED)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/G__KFParticle_mod.cxx
                       COMMAND sed -e "s#${CMAKE_SOURCE_DIR}/##g" ${CMAKE_BINARY_DIR}/G__KFParticle.cxx > ${CMAKE_BINARY_DIR}/G__KFParticle_mod.cxx
                       DEPENDS ${CMAKE_BINARY_DIR}/G__KFParticle.cxx
                      )
  else()
    Message("Can't correct KFParticle dictionary. Running KFParticle from an installed CbmRoot will not be possible")
  endif()

#  set(SourceFile "${CMAKE_BINARY_DIR}/G__KFParticle.cxx")
#  set(SourceFile_new "${CMAKE_BINARY_DIR}/G__KFParticle_mod.cxx")
#  file(READ ${SourceFile} Contents)
#  STRING(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" Contents_Modified "${Contents}" )
#  file(WRITE ${SourceFile_new} ${Contents_Modified})

  add_library(KFParticle SHARED ${SOURCES} G__KFParticle_mod.cxx)
  target_include_directories(KFParticle
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/KFParticle
        ${PROJECT_SOURCE_DIR}/KFParticlePerformance
        ${PROJECT_SOURCE_DIR}/KFParticleTest
        ${PROJECT_SOURCE_DIR}/externals/vir-simd
)
  target_link_libraries(KFParticle ${ROOT_LIBRARIES} )
  add_target_property(KFParticle COMPILE_FLAGS "-DDO_TPCCATRACKER_EFF_PERFORMANCE -DNonhomogeneousField -DCBM -DUSE_TIMERS")
#  add_target_property(KFParticle COMPILE_FLAGS "-DDO_TPCCATRACKER_EFF_PERFORMANCE -DHomogeneousField -DCBM -DUSE_TIMERS")


  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DNonhomogeneousField" "-DCBM" "-DUSE_TIMERS")
#  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DHomogeneousField" "-DCBM" "-DUSE_TIMERS")

else(FIXTARGET)
  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DHomogeneousField" "-DUSE_TIMERS")
  add_library(KFParticle SHARED ${SOURCES} G__KFParticle.cxx)
  target_link_libraries(KFParticle ${ROOT_LIBRARIES} )
  add_target_property(KFParticle COMPILE_FLAGS "-DDO_TPCCATRACKER_EFF_PERFORMANCE -DHomogeneousField -DUSE_TIMERS")
endif(FIXTARGET)

if(KFPARTICLE_USE_XPU)
  target_compile_definitions(KFParticle PUBLIC KFPARTICLE_USE_XPU)
  target_include_directories(KFParticle PUBLIC
    ${PROJECT_SOURCE_DIR}/KFParticle/GPU
    ${KFPARTICLE_XPU_INCLUDE_DIRS})
  if(TARGET xpu)
    target_link_libraries(KFParticle xpu)
  endif()
  get_target_property(KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY KFParticle LIBRARY_OUTPUT_DIRECTORY)
  if(NOT KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(KFParticle PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if(COMMAND xpu_attach)
    xpu_attach(KFParticle
      KFParticle/GPU/KFParticleGpuDeviceImage.cxx
      KFParticle/GPU/KFParticleGpuKernels.cxx
    )
  else()
    message(FATAL_ERROR "KFPARTICLE_USE_XPU requires the xpu_attach CMake command")
  endif()
endif()

if (ROOT_VERSION_MAJOR LESS 6)
    add_custom_target(libKFParticle.rootmap ALL DEPENDS KFParticle COMMAND rlibmap -o libKFParticle.rootmap -l libKFParticle.so -c ${PROJECT_SOURCE_DIR}/KFLinkDef.h)
endif (ROOT_VERSION_MAJOR LESS 6)


Set(NODICT_HEADERS
  KFParticle/KFParticleTopoReconstructor.h
  KFParticle/KFParticlePVReconstructor.h
  KFParticle/KFPVertex.h
  KFParticle/KFPTrack.h
  KFParticle/KFParticleFinder.h
  KFParticle/KFParticleSIMD.h
  KFParticle/KFParticleDef.h
  KFParticle/KFPSimd.h
  KFParticle/KFPSimdPaddedVector.h
  KFParticle/KFPTrackVector.h
  KFParticle/KFPEmcCluster.h
  KFParticle/KFParticleField.h
  KFParticle/KFPTrackVector.h
  KFParticle/KFParticleDatabase.h
  KFParticle/KFParticleMath.h
  KFParticlePerformance/KFTopoPerformance.h
  KFParticlePerformance/KFParticlePerformanceBase.h
  KFParticlePerformance/KFMCCounter.h
  KFParticlePerformance/KFPVEfficiencies.h
  KFParticlePerformance/KFMCVertex.h
  KFParticlePerformance/KFMCTrack.h
  KFParticlePerformance/KFPartMatch.h
)

install(TARGETS KFParticle DESTINATION lib)

install(DIRECTORY KFParticle
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")

install(DIRECTORY KFParticlePerformance
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")

# CBMRoot still includes these public performance headers through the legacy
# KFParticle/ prefix, including while ROOT generates its dictionaries.
install(FILES
  KFParticlePerformance/KFTopoPerformance.h
  KFParticlePerformance/KFParticlePerformanceBase.h
  KFParticlePerformance/KFPVEfficiencies.h
  KFParticlePerformance/KFMCCounter.h
  KFParticlePerformance/KFMCVertex.h
  KFParticlePerformance/KFPartMatch.h
  KFParticlePerformance/KFMCTrack.h
  KFParticlePerformance/KFMCParticle.h
  KFParticlePerformance/KFPartEfficiencies.h
  DESTINATION include/KFParticle)

install(DIRECTORY KFParticleTest
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")


#if(EXISTS ${CMAKE_BINARY_DIR}/lib/libKFParticle.rootmap)     
  install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle.rootmap DESTINATION lib)
  install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle_rdict.pcm DESTINATION lib)
#endif()

#install(FILES ${HEADERS} ${NODICT_HEADERS} DESTINATION include/KFParticle)

#install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle_rdict.pcm DESTINATION lib OPTIONAL)
#install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle.rootmap DESTINATION lib)
#install(TARGETS KFParticle DESTINATION lib)
