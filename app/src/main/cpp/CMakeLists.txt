cmake_minimum_required(VERSION 3.22.1)
project(speech_translation_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID AND NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "This project is configured for arm64-v8a only. Found ABI: ${ANDROID_ABI}")
endif()

# Prebuilt ONNX Runtime mobile package root (headers + static libs).
# Expected default layout:
# third_party/onnxruntime-mobile/
#   include/
#   lib/<abi>/libonnxruntime*.a  (or lib/libonnxruntime*.a)
set(
    ONNXRUNTIME_MOBILE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/third_party/onnxruntime-mobile"
    CACHE PATH "Path to prebuilt ONNX Runtime mobile package"
)

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_MOBILE_ROOT}/include")
if(
    NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h"
    AND NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h"
)
    message(FATAL_ERROR "Missing ONNX Runtime C++ headers under: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

set(_ORT_LIB_CANDIDATE_DIRS
    "${ONNXRUNTIME_MOBILE_ROOT}/lib/${ANDROID_ABI}"
    "${ONNXRUNTIME_MOBILE_ROOT}/lib"
    "${ONNXRUNTIME_MOBILE_ROOT}/${ANDROID_ABI}/lib"
)

set(ONNXRUNTIME_LIBS "")
foreach(_dir IN LISTS _ORT_LIB_CANDIDATE_DIRS)
    file(GLOB _static_libs "${_dir}/libonnxruntime*.a")
    if(_static_libs)
        list(APPEND ONNXRUNTIME_LIBS ${_static_libs})
        break()
    endif()

    file(GLOB _shared_libs "${_dir}/libonnxruntime*.so")
    if(_shared_libs)
        list(APPEND ONNXRUNTIME_LIBS ${_shared_libs})
        break()
    endif()
endforeach()

if(NOT ONNXRUNTIME_LIBS)
    message(FATAL_ERROR "Missing ONNX Runtime libs for ABI ${ANDROID_ABI} under ${ONNXRUNTIME_MOBILE_ROOT}")
endif()

# ================= ARM CPU FEATURE CONFIG (MUST BE BEFORE ADD_SUBDIRECTORY) =================

# Prevent ggml from trying runtime CPU detection paths
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_HBM OFF CACHE BOOL "" FORCE)

# Force ARMv8.2-A features for Snapdragon 732G
add_compile_options(
    -O3
    -ffast-math
    -funsafe-math-optimizations
    -fno-finite-math-only
    -fomit-frame-pointer
    -march=armv8.2-a+fp16+dotprod
)

# These macros are REQUIRED so ggml enables vector kernels
add_compile_definitions(
    __ARM_NEON
    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    __ARM_FEATURE_DOTPROD
)

# Disable desktop backends
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/third_party/whisper.cpp"
    "${CMAKE_BINARY_DIR}/whisper.cpp-build"
    EXCLUDE_FROM_ALL
)

add_library(
    speech_translation_native
    SHARED
    native-lib.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(dl-lib dl)
find_library(m-lib m)

target_include_directories(
    speech_translation_native
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    speech_translation_native
    whisper
    ${ONNXRUNTIME_LIBS}
    ${android-lib}
    ${log-lib}
    ${dl-lib}
    ${m-lib}
)
