/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUMEASUREMENT_H
#define KFPARTICLEGPUMEASUREMENT_H

#include "KFParticleGpuFitState.h"

/**
 * Daughter measurement layout used by the Kalman update.
 *
 * The arrays intentionally mirror the scalar KFParticle `GetMeasurement`
 * contract: transported daughter parameters, transported covariance, and the
 * 3x3 correlation block between the current particle and the daughter.
 */
class KFParticleGpuMeasurement
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuMeasurement() { Initialize(); }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    for (int i = 0; i < 9; ++i) {
      fCorrelation[i] = 0.f;
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE void StoreState(const KFParticleGpuFitState& state)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      fParameters[i] = state.Parameter(i);
    }
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = state.Covariance(i);
    }
  }

  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i, int j)
  {
    return fCovariance[KFParticleGpuFitState::CovarianceIndex(i, j)];
  }
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i, int j) const
  {
    return fCovariance[KFParticleGpuFitState::CovarianceIndex(i, j)];
  }
  KFPARTICLE_GPU_HOST_DEVICE float& Correlation(int i, int j) { return fCorrelation[i * 3 + j]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Correlation(int i, int j) const
  {
    return fCorrelation[i * 3 + j];
  }

 private:
  float fParameters[KFParticleGpuFitState::NumberOfParameters];
  float fCovariance[KFParticleGpuFitState::NumberOfCovarianceElements];
  float fCorrelation[9];
};

#endif
