/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUVERTEXSTATE_H
#define KFPARTICLEGPUVERTEXSTATE_H

#include "KFParticleGpuPlatform.h"

/** Numerical primary-vertex state without event or reconstruction metadata. */
class KFParticleGpuVertexState
{
 public:
  static const int NumberOfParameters = 3;
  static const int NumberOfCovarianceElements = 6;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexState() { Initialize(); }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    fChi2 = 0.f;
    fNDF = 0;
    fNContributors = 0;
  }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const float parameters[NumberOfParameters],
                                             const float covariance[NumberOfCovarianceElements],
                                             float chi2,
                                             int ndf,
                                             int nContributors)
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = parameters[i];
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = covariance[i];
    }
    fChi2 = chi2;
    fNDF = ndf;
    fNContributors = nContributors;
  }

  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Covariance(int i) { return fCovariance[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Covariance(int i) const { return fCovariance[i]; }

  KFPARTICLE_GPU_HOST_DEVICE float& X() { return fParameters[0]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Y() { return fParameters[1]; }
  KFPARTICLE_GPU_HOST_DEVICE float& Z() { return fParameters[2]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }

  KFPARTICLE_GPU_HOST_DEVICE float& Chi2() { return fChi2; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Chi2() const { return fChi2; }
  KFPARTICLE_GPU_HOST_DEVICE int& NDF() { return fNDF; }
  KFPARTICLE_GPU_HOST_DEVICE const int& NDF() const { return fNDF; }
  KFPARTICLE_GPU_HOST_DEVICE int& NContributors() { return fNContributors; }
  KFPARTICLE_GPU_HOST_DEVICE const int& NContributors() const { return fNContributors; }

 private:
  float fParameters[NumberOfParameters];
  float fCovariance[NumberOfCovarianceElements];
  float fChi2;
  int fNDF;
  int fNContributors;
};

#endif
