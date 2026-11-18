include(FetchContent)

if(TARGET vir-simd)
  return()
endif()

message(CHECK_START "Download project vir::simd")

FetchContent_Declare(
  vir-simd
  GIT_REPOSITORY "https://github.com/mattkretz/vir-simd"
  GIT_TAG        "848c298f879c9264d500c1d4ef5ad996333f9b64"
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/vir-simd
)

FetchContent_MakeAvailable(vir-simd)
