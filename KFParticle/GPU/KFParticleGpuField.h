/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUFIELD_H
#define KFPARTICLEGPUFIELD_H

#include "KFParticleGpuPlatform.h"

struct KFParticleGpuFieldValue
{
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue() : x(0.f), y(0.f), z(0.f) {}
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue(float bx, float by, float bz) : x(bx), y(by), z(bz) {}

  KFPARTICLE_GPU_HOST_DEVICE void Combine(const KFParticleGpuFieldValue& other, float weight)
  {
    x += weight * (other.x - x);
    y += weight * (other.y - y);
    z += weight * (other.z - z);
  }

  float x;
  float y;
  float z;
};

/**
 * Scalar parabolic field approximation kept outside the fit state.
 *
 * Separating it avoids carrying ten coefficients through kernels that do not
 * transport particles in a nonhomogeneous field.
 */
class KFParticleGpuFieldRegion
{
 public:
  static const int NumberOfCoefficients = 10;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldRegion() { Reset(); }

  KFPARTICLE_GPU_HOST_DEVICE explicit KFParticleGpuFieldRegion(
    const float coefficients[NumberOfCoefficients])
  {
    SetCoefficients(coefficients);
  }

  KFPARTICLE_GPU_HOST_DEVICE void Reset()
  {
    for (int i = 0; i < NumberOfCoefficients; ++i) {
      fCoefficients[i] = 0.f;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE void SetCoefficients(const float coefficients[NumberOfCoefficients])
  {
    for (int i = 0; i < NumberOfCoefficients; ++i) {
      fCoefficients[i] = coefficients[i];
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE float& Coefficient(int i) { return fCoefficients[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Coefficient(int i) const { return fCoefficients[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float* Coefficients() { return fCoefficients; }
  KFPARTICLE_GPU_HOST_DEVICE const float* Coefficients() const { return fCoefficients; }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFieldValue Get(float z) const
  {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float dz = z - fCoefficients[9];
    const float dz2 = dz * dz;
    return KFParticleGpuFieldValue(fCoefficients[0] + fCoefficients[1] * dz + fCoefficients[2] * dz2,
                                   fCoefficients[3] + fCoefficients[4] * dz + fCoefficients[5] * dz2,
                                   fCoefficients[6] + fCoefficients[7] * dz + fCoefficients[8] * dz2);
  }

 private:
  float fCoefficients[NumberOfCoefficients];
};

#endif
