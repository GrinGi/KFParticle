include(CMakePackageConfigHelpers)
foreach(directory CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_DATADIR)
  if(IS_ABSOLUTE "${${directory}}")
    message(FATAL_ERROR "Relocatable SDK requires a relative ${directory}")
  endif()
endforeach()
set(KFPARTICLE_SDK_VERSION "0.1.0")
set(KFPARTICLE_SDK_HIP OFF)
set(KFPARTICLE_SDK_EXTERNAL_XPU OFF)
set(KFPARTICLE_SDK_XPU_PLUGINS "")
set(KFPARTICLE_SDK_NEEDS_OPENMP OFF)
set(KFPARTICLE_SDK_PLUGINS "")
set(KFPARTICLE_SDK_ROCM_VERSION "not-built")
file(GLOB_RECURSE sdk_inputs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/*.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/KFParticlePerformance/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/KFParticlePerformance/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/KFParticleTest/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/KFParticleTest/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.in")
list(APPEND sdk_inputs CMakeLists.txt KFLinkDef.h)
list(SORT sdk_inputs)
set(sdk_source_identity "")
foreach(path IN LISTS sdk_inputs)
  if(path MATCHES "(^|/)(\\.local|test)/")
    continue()
  endif()
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" hash)
  string(APPEND sdk_source_identity "${path}:${hash}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
endforeach()
string(SHA256 KFPARTICLE_SDK_SOURCE_SHA256 "${sdk_source_identity}")
set(sdk_revision "source-archive")
set(sdk_dirty "unknown")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse HEAD
      OUTPUT_VARIABLE sdk_revision OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" status --porcelain
      OUTPUT_VARIABLE sdk_changes OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(sdk_changes)
      set(sdk_dirty ON)
    else()
      set(sdk_dirty OFF)
    endif()
  endif()
endif()
if(NOT Vc_VERSION)
  set(Vc_VERSION "${Vc_VERSION_STRING}")
endif()
if(KFPARTICLE_USE_XPU)
  get_target_property(KFPARTICLE_SDK_EXTERNAL_XPU xpu IMPORTED)
  if(NOT KFPARTICLE_SDK_EXTERNAL_XPU)
    # Source-tree integration exports its one runtime; installed XPU stays external.
    set_property(TARGET xpu PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${KFPARTICLE_XPU_SOURCE_DIR}/src>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
    install(TARGETS xpu EXPORT KFParticleTargets
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(DIRECTORY "${KFPARTICLE_XPU_SOURCE_DIR}/src/xpu"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
      FILES_MATCHING PATTERN "*.h" PATTERN "*.tpp")
    install(FILES "${KFPARTICLE_XPU_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/KFParticle" RENAME XPU-LICENSE)
  endif()
  foreach(backend HIP CUDA SYCL)
    if(XPU_ENABLE_${backend})
      if(backend STREQUAL "HIP")
        set(suffix Hip)
        set(KFPARTICLE_SDK_HIP ON)
        set(KFPARTICLE_SDK_ROCM_VERSION "unknown")
        if(EXISTS "${XPU_ROCM_ROOT}/.info/version")
          file(STRINGS "${XPU_ROCM_ROOT}/.info/version" KFPARTICLE_SDK_ROCM_VERSION LIMIT_COUNT 1)
        endif()
      elseif(backend STREQUAL "CUDA")
        set(suffix Cuda)
      else()
        set(suffix Sycl)
      endif()
      list(APPEND KFPARTICLE_SDK_PLUGINS "${CMAKE_SHARED_LIBRARY_PREFIX}KFParticle_${suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      set(plugin "${CMAKE_SHARED_LIBRARY_PREFIX}xpu_${suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      if(KFPARTICLE_SDK_EXTERNAL_XPU)
        list(APPEND KFPARTICLE_SDK_XPU_PLUGINS "${plugin}")
      else()
        list(APPEND KFPARTICLE_SDK_PLUGINS "${plugin}")
        install(FILES "$<TARGET_FILE_DIR:xpu>/${plugin}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
      endif()
    endif()
  endforeach()
  if(XPU_ENABLE_OPENMP)
    set(KFPARTICLE_SDK_NEEDS_OPENMP ON)
    target_link_libraries(KFParticle PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# Dependencies are rediscovered at the consuming prefix, never encoded as build paths.
set(sdk_dependencies "")
foreach(component Core Hist Matrix Physics EG Gpad Graf Graf3d RIO MathCore)
  if(NOT TARGET ROOT::${component})
    message(FATAL_ERROR "Installed SDK requires ROOT CONFIG targets (missing ROOT::${component})")
  endif()
  list(APPEND sdk_dependencies ROOT::${component})
endforeach()
if(TARGET Vc::Vc)
  list(APPEND sdk_dependencies Vc::Vc)
elseif(TARGET Vc)
  list(APPEND sdk_dependencies Vc)
else()
  message(FATAL_ERROR "Installed SDK requires the Vc CONFIG target Vc::Vc or Vc")
endif()
set_property(TARGET KFParticle PROPERTY INTERFACE_LINK_LIBRARIES "${sdk_dependencies}")
if(KFPARTICLE_USE_XPU)
  if(KFPARTICLE_SDK_EXTERNAL_XPU)
    set_property(TARGET KFParticle APPEND PROPERTY INTERFACE_LINK_LIBRARIES XPU::xpu)
  else()
    set_property(TARGET KFParticle APPEND PROPERTY INTERFACE_LINK_LIBRARIES xpu)
  endif()
endif()
if(KFPARTICLE_SDK_NEEDS_OPENMP)
  set_property(TARGET KFParticle APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
endif()

install(TARGETS KFParticle EXPORT KFParticleTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/KFParticle")
configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/KFParticleConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/KFParticleConfig.cmake"
  INSTALL_DESTINATION "${package_dir}" PATH_VARS CMAKE_INSTALL_LIBDIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/KFParticleConfigVersion.cmake"
  VERSION "${KFPARTICLE_SDK_VERSION}" COMPATIBILITY SameMinorVersion)
install(EXPORT KFParticleTargets NAMESPACE KFParticle:: DESTINATION "${package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/KFParticleConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/KFParticleConfigVersion.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/KFParticleXpu.cmake" DESTINATION "${package_dir}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/KFParticleBuildInfo.txt"
  "sdk_version=${KFPARTICLE_SDK_VERSION}\nkfparticle_revision=${sdk_revision}\nkfparticle_dirty=${sdk_dirty}\nkfparticle_content_sha256=${KFPARTICLE_SDK_SOURCE_SHA256}\nxpu_external=${KFPARTICLE_SDK_EXTERNAL_XPU}\nxpu_library=${KFPARTICLE_XPU_LIBRARY}\ngpu=${KFPARTICLE_USE_XPU}\nhip=${KFPARTICLE_SDK_HIP}\nhip_arch=${XPU_HIP_ARCH}\nrocm=${KFPARTICLE_SDK_ROCM_VERSION}\nfixtarget=${FIXTARGET}\nunroll=${KFPARTICLE_GPU_FIT_UNROLL}\ncxx_standard=${CMAKE_CXX_STANDARD}\ncompiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\ncmake=${CMAKE_VERSION}\nroot=${ROOT_VERSION}\nvc=${Vc_VERSION}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/KFParticleBuildInfo.txt"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/KFParticle")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/GPU/README.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/KFParticle")
if(KFPARTICLE_USE_XPU)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/GPU/example/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/KFParticle/example"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cxx" PATTERN "CMakeLists.txt")
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/GPU/test/standalone/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/KFParticle/GPU/test/standalone/KFParticleGpuFinderTest.cxx"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/KFParticle/standalone")
endif()
