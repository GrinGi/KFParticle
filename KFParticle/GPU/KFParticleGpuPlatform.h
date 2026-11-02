/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUPLATFORM_H
#define KFPARTICLEGPUPLATFORM_H

/**
 * @file
 * @brief Compile-time portability layer for shared KFParticle CPU/GPU numerical code.
 *
 * Selects XPU annotations and math functions when XPU is requested or a supported device
 * compiler is detected. Otherwise, uses ordinary C++ functions from <cmath>, allowing
 * numerical types to be tested without XPU. This header does not select a device at runtime,
 * launch kernels, or manage memory. Math accuracy and special-value behavior follow the
 * selected backend; bitwise-identical CPU/GPU results are not guaranteed.
 *
 * If available through __has_include, the CMake-generated KFParticleGpuBuildOptions.h
 * supplies fitting-loop unroll configuration before the fallback default is applied.
 */

/**
 * @def KFPARTICLE_GPU_XPU_ENABLED
 * @brief Presence flag indicating that annotations and math wrappers use XPU.
 * Defined for KFPARTICLE_USE_XPU, NVCC, HIPCC, or SYCL compilation; this does not
 * imply execution on a GPU, since XPU also supports a CPU backend.
 */
/**
 * @def KFPARTICLE_GPU_HOST_DEVICE
 * @brief Apply XPU host/device function annotations, or nothing in standalone C++ mode.
 * The meaning of XPU_H and XPU_D is determined by the selected XPU backend.
 */
#if defined(KFPARTICLE_USE_XPU) || defined(__NVCC__) || defined(__HIPCC__) || defined(SYCL_LANGUAGE_VERSION)
#define KFPARTICLE_GPU_XPU_ENABLED
#include <xpu/device.h>
#define KFPARTICLE_GPU_HOST_DEVICE XPU_H XPU_D
#else
#include <cmath>
#define KFPARTICLE_GPU_HOST_DEVICE
#endif

#if defined(__has_include)
#if __has_include("KFParticleGpuBuildOptions.h")
#include "KFParticleGpuBuildOptions.h"
#endif
#endif

/**
 * @def KFPARTICLE_GPU_FIT_UNROLL
 * @brief Nonzero enables explicit HIP device unroll hints for selected fitting loops.
 * Defaults to 1 only if no definition was supplied by build configuration.
 * A value of zero disables these hints, not the compiler's own unrolling optimizations.
 */
#ifndef KFPARTICLE_GPU_FIT_UNROLL
#define KFPARTICLE_GPU_FIT_UNROLL 1
#endif

/**
 * @def KFPARTICLE_GPU_UNROLL
 * @brief Request compiler unrolling of the immediately following loop in the HIP device pass.
 * Expands to the unroll pragma only when __HIP_DEVICE_COMPILE__ is defined and
 * KFPARTICLE_GPU_FIT_UNROLL is nonzero; otherwise expands to nothing. This is a
 * compiler directive, not a change to loop semantics or a guarantee of faster execution.
 */
#if defined(__HIP_DEVICE_COMPILE__) && KFPARTICLE_GPU_FIT_UNROLL
#define KFPARTICLE_GPU_UNROLL _Pragma("unroll")
#else
#define KFPARTICLE_GPU_UNROLL
#endif

/**
 * @brief Return the single-precision square root through XPU or the standard library.
 * The real-valued domain is value >= 0. No argument validation or clamping is performed;
 * negative inputs and special values follow the backend's square-root behavior.
 */
KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuSqrt(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::sqrt(value);
#else
  return std::sqrt(value);
#endif
}

/**
 * @brief Return the single-precision sine of an angle in radians.
 * Delegates to XPU or the standard library, including argument reduction and special-value handling.
 */
KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuSin(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::sin(value);
#else
  return std::sin(value);
#endif
}

/**
 * @brief Return the single-precision cosine of an angle in radians.
 * Delegates to XPU or the standard library, including argument reduction and special-value handling.
 */
KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuCos(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::cos(value);
#else
  return std::cos(value);
#endif
}

/**
 * @brief Round to an integral value represented as float, without converting to an integer type.
 * Delegates to the backend's rint: standard C++ follows the current rounding mode,
 * while device behavior follows the selected backend. With round-to-nearest-even,
 * halfway cases go to the even integer, unlike round()'s halfway-away-from-zero rule.
 */
KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuRint(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::rint(value);
#else
  return std::rint(value);
#endif
}

#endif
