/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUFITSTATE_H
#define KFPARTICLEGPUFITSTATE_H

#include "KFParticleGpuTrackState.h"

/**
 * Thread-local numerical state used by KFParticle fit operations.
 *
 * Field coefficients and reconstruction metadata remain separate because they
 * are not needed by every mathematical operation.
 */
class KFParticleGpuFitState
{
 public:
  static const int NumberOfParameters = 8;
  static const int NumberOfCovarianceElements = 36;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitState() { Initialize(); }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize()
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }

    fCovariance[0] = 100.f;
    fCovariance[2] = 100.f;
    fCovariance[5] = 100.f;
    fCovariance[35] = 1.f;

    fChi2 = 0.f;
    fSFromDecay = 0.f;
    fSumDaughterMass = 0.f;
    fMassHypothesis = -1.f;
    fNdf = -3;
    fCharge = 0;
    fAtProductionVertex = 0;
    fConstructMethod = 0;
  }

  KFPARTICLE_GPU_HOST_DEVICE void Initialize(const KFParticleGpuTrackState& track, int charge, float mass)
  {
    for (int i = 0; i < NumberOfParameters; ++i) {
      fParameters[i] = 0.f;
    }
    for (int i = 0; i < NumberOfCovarianceElements; ++i) {
      fCovariance[i] = 0.f;
    }
    for (int i = 0; i < KFParticleGpuTrackState::NumberOfParameters; ++i) {
      fParameters[i] = track.Parameter(i);
    }
    for (int i = 0; i < KFParticleGpuTrackState::NumberOfCovarianceElements; ++i) {
      fCovariance[i] = track.Covariance(i);
    }

    const float energy = KFParticleGpuSqrt(mass * mass + Px() * Px() + Py() * Py() + Pz() * Pz());
    E() = energy;
    S() = 0.f;

    // Invalid zero-energy tracks stay finite and can be rejected by selection.
    if (energy > 0.f) {
      const float energyInv = 1.f / energy;
      const float h0 = Px() * energyInv;
      const float h1 = Py() * energyInv;
      const float h2 = Pz() * energyInv;

      fCovariance[21] = h0 * fCovariance[6] + h1 * fCovariance[10] + h2 * fCovariance[15];
      fCovariance[22] = h0 * fCovariance[7] + h1 * fCovariance[11] + h2 * fCovariance[16];
      fCovariance[23] = h0 * fCovariance[8] + h1 * fCovariance[12] + h2 * fCovariance[17];
      fCovariance[24] = h0 * fCovariance[9] + h1 * fCovariance[13] + h2 * fCovariance[18];
      fCovariance[25] = h0 * fCovariance[13] + h1 * fCovariance[14] + h2 * fCovariance[19];
      fCovariance[26] = h0 * fCovariance[18] + h1 * fCovariance[19] + h2 * fCovariance[20];
      fCovariance[27] = h0 * h0 * fCovariance[9] + h1 * h1 * fCovariance[14]
                        + h2 * h2 * fCovariance[20]
                        + 2.f * (h0 * h1 * fCovariance[13] + h0 * h2 * fCovariance[18]
                                   + h1 * h2 * fCovariance[19]);
    }
    fCovariance[35] = 1.f;

    fChi2 = 0.f;
    fSFromDecay = 0.f;
    fSumDaughterMass = mass;
    fMassHypothesis = mass;
    fNdf = 0;
    fCharge = charge;
    fAtProductionVertex = 0;
    fConstructMethod = 0;
  }

  KFPARTICLE_GPU_HOST_DEVICE static int CovarianceIndex(int i, int j)
  {
    return (j <= i) ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

  KFPARTICLE_GPU_HOST_DEVICE float& Parameter(int i) { return fParameters[i]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Parameter(int i) const { return fParameters[i]; }
  // Matrix routines need contiguous arrays, but storage remains owned by this state.
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
  KFPARTICLE_GPU_HOST_DEVICE float& E() { return fParameters[6]; }
  KFPARTICLE_GPU_HOST_DEVICE float& S() { return fParameters[7]; }

  KFPARTICLE_GPU_HOST_DEVICE const float& X() const { return fParameters[0]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Y() const { return fParameters[1]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Z() const { return fParameters[2]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Px() const { return fParameters[3]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Py() const { return fParameters[4]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Pz() const { return fParameters[5]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& E() const { return fParameters[6]; }
  KFPARTICLE_GPU_HOST_DEVICE const float& S() const { return fParameters[7]; }

  KFPARTICLE_GPU_HOST_DEVICE float& Chi2() { return fChi2; }
  KFPARTICLE_GPU_HOST_DEVICE const float& Chi2() const { return fChi2; }
  KFPARTICLE_GPU_HOST_DEVICE float& SFromDecay() { return fSFromDecay; }
  KFPARTICLE_GPU_HOST_DEVICE const float& SFromDecay() const { return fSFromDecay; }
  KFPARTICLE_GPU_HOST_DEVICE float& SumDaughterMass() { return fSumDaughterMass; }
  KFPARTICLE_GPU_HOST_DEVICE const float& SumDaughterMass() const { return fSumDaughterMass; }
  KFPARTICLE_GPU_HOST_DEVICE float& MassHypo() { return fMassHypothesis; }
  KFPARTICLE_GPU_HOST_DEVICE const float& MassHypo() const { return fMassHypothesis; }
  KFPARTICLE_GPU_HOST_DEVICE int& NDF() { return fNdf; }
  KFPARTICLE_GPU_HOST_DEVICE const int& NDF() const { return fNdf; }
  KFPARTICLE_GPU_HOST_DEVICE int& Q() { return fCharge; }
  KFPARTICLE_GPU_HOST_DEVICE const int& Q() const { return fCharge; }
  KFPARTICLE_GPU_HOST_DEVICE int& AtProductionVertex() { return fAtProductionVertex; }
  KFPARTICLE_GPU_HOST_DEVICE const int& AtProductionVertex() const { return fAtProductionVertex; }
  KFPARTICLE_GPU_HOST_DEVICE int& ConstructMethod() { return fConstructMethod; }
  KFPARTICLE_GPU_HOST_DEVICE const int& ConstructMethod() const { return fConstructMethod; }

 private:
  float fParameters[NumberOfParameters];
  float fCovariance[NumberOfCovarianceElements];
  float fChi2;
  float fSFromDecay;
  float fSumDaughterMass;
  float fMassHypothesis;
  int fNdf;
  int fCharge;
  int fAtProductionVertex;
  int fConstructMethod;
};

#endif
