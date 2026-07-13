/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUTRACKSTATE_H
#define KFPARTICLEGPUTRACKSTATE_H

#include "KFParticleGpuPlatform.h"

/**
 * Compact numerical input state before a mass hypothesis is applied.
 *
 * Selection metadata is deliberately stored elsewhere so lightweight kernels
 * do not have to load track parameters and covariance.
 */
class KFParticleGpuTrackState
{
 public:
  static const int NumberOfParameters = 6;
  static const int NumberOfCovarianceElements = 21;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackState() { Reset(); }

  KFPARTICLE_GPU_HOST_DEVICE void Reset()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const float parameters[NumberOfParameters],
                                             const float covariance[NumberOfCovarianceElements])
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = parameters[i];
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = covariance[i];
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE static int CovarianceIndex(int i, int j)
  {
    return (j <= i) ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float* Parameters() { return fParameters; }
  KFPARTICLE_GPU_HOST_DEVICE const float* Parameters() const { return fParameters; }

  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float* Covariances() { return fCovariance; }
  KFPARTICLE_GPU_HOST_DEVICE const float* Covariances() const { return fCovariance; }

  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i, int j) { return fCovariance[CovarianceIndex(i, j)]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i, int j) const
  {
    return fCovariance[CovarianceIndex(i, j)];
  }

  KFPARTICLE_GPU_HOST_DEVICE float& X() { return fParameters[0]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Y() { return fParameters[1]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Z() { return fParameters[2]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Px() { return fParameters[3]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Py() { return fParameters[4]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Pz() { return fParameters[5]; }

  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Px() const { return fParameters[3]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Py() const { return fParameters[4]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Pz() const { return fParameters[5]; }

 private:
  float fParameters[NumberOfParameters];
  float fCovariance[NumberOfCovarianceElements];
};

#endif
