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

#if defined(KFPARTICLE_USE_XPU) || defined(__NVCC__) || defined(__HIPCC__) || defined(SYCL_LANGUAGE_VERSION)
#define KFPARTICLE_GPU_XPU_ENABLED
#include <xpu/device.h>
#define KFPARTICLE_GPU_HOST_DEVICE XPU_H XPU_D
#else
#include <cmath>
#define KFPARTICLE_GPU_HOST_DEVICE
#endif

// Keep numerical types testable without XPU while sharing their device API.
KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuSqrt(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::sqrt(value);
#else
  return std::sqrt(value);
#endif
}

KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuSin(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::sin(value);
#else
  return std::sin(value);
#endif
}

KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuCos(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::cos(value);
#else
  return std::cos(value);
#endif
}

KFPARTICLE_GPU_HOST_DEVICE inline float KFParticleGpuRint(float value)
{
#ifdef KFPARTICLE_GPU_XPU_ENABLED
  return xpu::rint(value);
#else
  return std::rint(value);
#endif
}

#endif
