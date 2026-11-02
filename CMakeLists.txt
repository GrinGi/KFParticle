cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(KFParticleLibrary LANGUAGES CXX)

# add at the top of your KFParticle CMakeLists.txt
if(NOT CMAKE_CXX_FLAGS MATCHES "-msse4.1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

# The GPU implementation and its XPU dependency require the KFParticle
# baseline to be C++17 in both standalone and embedded CBMRoot builds.
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KFPARTICLE_GPU_FIT_UNROLL "Enable explicit HIP fitting-loop unroll hints" ON)
option(KFPARTICLE_BUILD_GPU_EXAMPLE "Build the file-driven GPU example in this target graph" OFF)
set(KFPARTICLE_GPU_TIMING_BUILD_ID "unversioned" CACHE STRING
    "Content identity for reproducible KFParticle timing builds")

option(KFPARTICLE_USE_XPU "Build the KFParticle XPU infrastructure." OFF)
option(KFPARTICLE_CBMROOT_EMBEDDED
       "Build KFParticle as part of the CBMRoot CMake target graph."
       OFF)
set(KFPARTICLE_XPU_SOURCE_DIR "" CACHE PATH "Optional XPU source tree for standalone KFParticle builds")
set(KFPARTICLE_XPU_ROOT "" CACHE PATH "Installed XPU prefix")
set(XPU_SOURCE_DIR "" CACHE PATH "Legacy alias for KFPARTICLE_XPU_SOURCE_DIR")
set(XPU_INCLUDE_DIR "" CACHE PATH "Optional legacy XPU include directory")

include(GNUInstallDirs)
if(KFPARTICLE_CBMROOT_EMBEDDED)
  set(sdk_default OFF)
else()
  set(sdk_default ON)
  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endif()
endif()
option(KFPARTICLE_INSTALL_SDK "Install the standalone CMake SDK" ${sdk_default})
if(KFPARTICLE_CBMROOT_EMBEDDED AND KFPARTICLE_INSTALL_SDK)
  message(FATAL_ERROR "Build the standalone SDK separately from CBMRoot")
endif()

if(KFPARTICLE_USE_XPU)
  include(cmake/KFParticleXpu.cmake)
  kfparticle_configure_xpu()
endif()
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS} ${ROOTSYS})

set(FIXTARGET FALSE CACHE BOOL "Compile for fix target geometry.")
if(KFPARTICLE_USE_XPU AND NOT FIXTARGET)
  message(FATAL_ERROR "The current GPU host-input interface requires FIXTARGET=ON")
endif()

find_package(ROOT REQUIRED COMPONENTS Core Hist Matrix Physics EG Gpad Graf Graf3d RIO MathCore)
If(DEFINED Vc_INCLUDE_DIR AND Vc_LIBRARIES)
  Message("Vc found")
Else()
  find_package(Vc REQUIRED)
EndIf()

Set(INCLUDE_DIRECTORIES
  ${PROJECT_SOURCE_DIR}/KFParticle
  ${PROJECT_SOURCE_DIR}/KFParticlePerformance
  ${PROJECT_SOURCE_DIR}/KFParticleTest
  ${PROJECT_SOURCE_DIR}/KFParticle/KFPSimd
  ${PROJECT_SOURCE_DIR}/KFParticle/KFPSimd/Utils
  ${PROJECT_SOURCE_DIR}/KFParticle/KFPSimd/SSE
)

include(${ROOT_USE_FILE})
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
Include_Directories( ${CMAKE_CURRENT_SOURCE_DIR} ${Vc_INCLUDE_DIR} ${INCLUDE_DIRECTORIES} )
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

set (SIMD_HEADERS
  KFParticle/KFPSimd/simd.h
)

set (SIMD_UTILS_HEADERS
  KFParticle/KFPSimd/Utils/macros.h
  KFParticle/KFPSimd/Utils/memory.h
  KFParticle/KFPSimd/Utils/simd_detect.h
  KFParticle/KFPSimd/Utils/tag.h
)

set (SIMD_SSE_HEADERS
  KFParticle/KFPSimd/SSE/constants.h
  KFParticle/KFPSimd/SSE/float32.h
  KFParticle/KFPSimd/SSE/int32.h
  KFParticle/KFPSimd/SSE/mask32.h
  KFParticle/KFPSimd/SSE/types.h
)

set (HEADERS
#  KFParticle/KFParticleBase.h
  KFParticle/KFParticle.h
  KFParticle/KFVertex.h
  KFParticlePerformance/KFPartEfficiencies.h
  KFParticlePerformance/KFMCParticle.h
  KFParticleTest/KFParticleTest.h
)

if(FIXTARGET)

  find_program(BIN_SED sed)
  if(BIN_SED)
    # ROOT generates the dictionary in this directory. This matters when
    # KFParticle is a CBMRoot subdirectory rather than a top-level project.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/G__KFParticle_mod.cxx
                       COMMAND sed -e "s#${CMAKE_CURRENT_SOURCE_DIR}/##g" ${CMAKE_CURRENT_BINARY_DIR}/G__KFParticle.cxx > ${CMAKE_CURRENT_BINARY_DIR}/G__KFParticle_mod.cxx
                       DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/G__KFParticle.cxx
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
  target_link_libraries(KFParticle PUBLIC ${ROOT_LIBRARIES} ${Vc_LIBRARIES})


  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DNonhomogeneousField" "-DCBM" "-DUSE_TIMERS")
#  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DHomogeneousField" "-DCBM" "-DUSE_TIMERS")

else(FIXTARGET)
  ROOT_GENERATE_DICTIONARY(G__KFParticle ${HEADERS} LINKDEF KFLinkDef.h OPTIONS "-DDO_TPCCATRACKER_EFF_PERFORMANCE" "-DHomogeneousField" "-DUSE_TIMERS")
  add_library(KFParticle SHARED ${SOURCES} G__KFParticle.cxx)
  target_link_libraries(KFParticle PUBLIC ${ROOT_LIBRARIES} ${Vc_LIBRARIES})
endif(FIXTARGET)

# Propagate the scalar/track layout and SIMD baseline to external consumers.
add_library(KFParticle::KFParticle ALIAS KFParticle)
target_compile_features(KFParticle PUBLIC cxx_std_17)
foreach(directory IN ITEMS "" KFParticle KFParticlePerformance KFParticleTest
    KFParticle/KFPSimd KFParticle/KFPSimd/Utils KFParticle/KFPSimd/SSE)
  target_include_directories(KFParticle PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${directory}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${directory}>")
endforeach()
target_include_directories(KFParticle PUBLIC
  "$<BUILD_INTERFACE:${ROOT_INCLUDE_DIRS}>"
  "$<BUILD_INTERFACE:${Vc_INCLUDE_DIR}>")
target_compile_definitions(KFParticle PUBLIC DO_TPCCATRACKER_EFF_PERFORMANCE USE_TIMERS)
if(FIXTARGET)
  target_compile_definitions(KFParticle PUBLIC NonhomogeneousField CBM)
else()
  target_compile_definitions(KFParticle PUBLIC HomogeneousField)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(KFParticle PUBLIC -msse4.1)
endif()

if(KFPARTICLE_USE_XPU)
  # Register host-side GPU sources on the final target. This keeps embedded
  # CBMRoot and standalone builds on one source list and prevents a stale
  # partial GPU library with unresolved graph-plan symbols.
  target_sources(KFParticle PRIVATE
    KFParticle/GPU/data/KFParticleGpuInputPacker.cxx
    KFParticle/GPU/planning/KFParticleGpuBatch.cxx
    KFParticle/GPU/execution/KFParticleGpuExecutor.cxx
    KFParticle/GPU/execution/KFParticleGpuFinder.cxx
    KFParticle/GPU/execution/KFParticleGpuRuntimeService.cxx
    KFParticle/GPU/execution/KFParticleGpuBufferManager.cxx
    KFParticle/GPU/planning/KFParticleGpuCpuChannelCatalogue.cxx
    KFParticle/GPU/planning/KFParticleGpuDecayGraphPlan.cxx
    KFParticle/GPU/planning/KFParticleGpuDecayPlan.cxx
    KFParticle/GPU/execution/KFParticleGpuKernels.cxx
    KFParticle/GPU/planning/KFParticleGpuRoutingPlan.cxx
    KFParticle/GPU/execution/KFParticleGpuRuntime.cxx
    KFParticle/GPU/execution/KFParticleGpuSteering.cxx
  )
  target_compile_definitions(KFParticle PUBLIC KFPARTICLE_USE_XPU)
  target_include_directories(KFParticle PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/KFParticle/GPU>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/KFParticle/GPU>")
  foreach(directory IN LISTS KFPARTICLE_XPU_INCLUDE_DIRS)
    target_include_directories(KFParticle PUBLIC "$<BUILD_INTERFACE:${directory}>")
  endforeach()
  # Host packing and materialization retain their explicitly bounded workers.
  # This dependency belongs to KFParticle now, not to its CBMRoot caller.
  find_package(Threads REQUIRED)
  target_link_libraries(KFParticle PUBLIC xpu PRIVATE Threads::Threads)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(KFParticle PRIVATE OpenMP::OpenMP_CXX)
  endif()
  get_target_property(KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY KFParticle LIBRARY_OUTPUT_DIRECTORY)
  if(NOT KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY)
    if(KFPARTICLE_CBMROOT_EMBEDDED)
      set(KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    else()
      set(KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set_target_properties(KFParticle PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${KFPARTICLE_LIBRARY_OUTPUT_DIRECTORY}")
  endif()
  if(COMMAND xpu_attach)
    if(KFPARTICLE_GPU_FIT_UNROLL)
      set(KFPARTICLE_GPU_FIT_UNROLL_VALUE 1)
    else()
      set(KFPARTICLE_GPU_FIT_UNROLL_VALUE 0)
    endif()
    configure_file(KFParticle/GPU/KFParticleGpuBuildOptions.h.in
      ${CMAKE_CURRENT_BINARY_DIR}/gpu-config/KFParticleGpuBuildOptions.h @ONLY)
    target_include_directories(KFParticle BEFORE PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gpu-config>")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gpu-config/KFParticleGpuBuildOptions.h
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/KFParticle/GPU")
    xpu_attach(KFParticle
      KFParticle/GPU/execution/KFParticleGpuKernels.cxx
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
#  KFParticle/KFParticleBaseSIMD.h
  KFParticle/KFParticleDef.h
  KFParticle/KFPSimdAllocator.h
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

if(KFPARTICLE_INSTALL_SDK)
  if(APPLE)
    set(sdk_origin "@loader_path")
  else()
    set(sdk_origin "$ORIGIN")
  endif()
  set_property(TARGET KFParticle PROPERTY INSTALL_RPATH "${sdk_origin}")
  if(KFPARTICLE_XPU_LIBRARY_DIR)
    set_property(TARGET KFParticle APPEND PROPERTY INSTALL_RPATH "${KFPARTICLE_XPU_LIBRARY_DIR}")
  endif()
  if(KFPARTICLE_OWNS_XPU)
    set_property(TARGET xpu PROPERTY INSTALL_RPATH "${sdk_origin}")
  endif()
  include(cmake/KFParticleSdk.cmake)
else()
  install(TARGETS KFParticle DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

install(DIRECTORY KFParticle
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "test" EXCLUDE
        PATTERN ".local" EXCLUDE)

install(DIRECTORY KFParticlePerformance
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
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
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/KFParticle")

install(DIRECTORY KFParticleTest
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.h")


# ROOT writes dictionary side files next to this subproject's generated source.
# Do not use CMAKE_BINARY_DIR here: embedded CBMRoot builds have a different
# top-level binary directory.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle.rootmap DESTINATION "${CMAKE_INSTALL_LIBDIR}" OPTIONAL)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle_rdict.pcm DESTINATION "${CMAKE_INSTALL_LIBDIR}" OPTIONAL)
#endif()

#install(FILES ${HEADERS} ${NODICT_HEADERS} DESTINATION include/KFParticle)

#install(FILES ${SIMD_HEADERS} DESTINATION include/KFParticle/KFPSimd)
#install(FILES ${SIMD_UTILS_HEADERS} DESTINATION include/KFParticle/Utils) 
#install(FILES ${SIMD_SSE_HEADERS} DESTINATION include/KFParticle/SSE) 

#install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle_rdict.pcm DESTINATION lib OPTIONAL)
#install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle.rootmap DESTINATION lib)
#install(TARGETS KFParticle DESTINATION lib)

if(KFPARTICLE_BUILD_GPU_EXAMPLE)
  if(NOT KFPARTICLE_USE_XPU)
    message(FATAL_ERROR "The GPU example requires KFPARTICLE_USE_XPU=ON")
  endif()
  add_subdirectory(KFParticle/GPU/example gpu-example)
endif()
