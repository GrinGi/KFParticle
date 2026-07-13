/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUSTATETRANSFER_H
#define KFPARTICLEGPUSTATETRANSFER_H

#include "KFParticleGpuSoAView.h"

KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuConstTrackSoAView& source,
                                                      unsigned int particle,
                                                      KFParticleGpuTrackState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, particle);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuTrackState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, particle);
  }
}

KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuTrackSoAView& source,
                                                      unsigned int particle,
                                                      KFParticleGpuTrackState& destination)
{
  LoadTrackState(MakeConstView(source), particle, destination);
}

KFPARTICLE_GPU_HOST_DEVICE inline void StoreTrackState(const KFParticleGpuTrackState& source,
                                                       KFParticleGpuTrackSoAView& destination,
                                                       unsigned int particle)
{
  for (unsigned int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
    destination.Parameter(component, particle) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuTrackState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, particle) = source.Covariance(component);
  }
}

KFPARTICLE_GPU_HOST_DEVICE inline void LoadFitState(const KFParticleGpuConstFitSoAView& source,
                                                    unsigned int particle,
                                                    KFParticleGpuFitState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, particle);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuFitState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, particle);
  }

  destination.Chi2() =
    source.FitScalar(KFParticleGpuFitSoALayout::Chi2, particle);
  destination.SFromDecay() =
    source.FitScalar(KFParticleGpuFitSoALayout::SFromDecay, particle);
  destination.SumDaughterMass() =
    source.FitScalar(KFParticleGpuFitSoALayout::SumDaughterMass, particle);
  destination.MassHypo() =
    source.FitScalar(KFParticleGpuFitSoALayout::MassHypothesis, particle);

  destination.NDF() =
    source.FitInteger(KFParticleGpuFitSoALayout::Ndf, particle);
  destination.Q() =
    source.FitInteger(KFParticleGpuFitSoALayout::Charge, particle);
  destination.AtProductionVertex() =
    source.FitInteger(KFParticleGpuFitSoALayout::AtProductionVertex, particle);
  destination.ConstructMethod() =
    source.FitInteger(KFParticleGpuFitSoALayout::ConstructMethod, particle);
}

KFPARTICLE_GPU_HOST_DEVICE inline void LoadFitState(const KFParticleGpuFitSoAView& source,
                                                    unsigned int particle,
                                                    KFParticleGpuFitState& destination)
{
  LoadFitState(MakeConstView(source), particle, destination);
}

KFPARTICLE_GPU_HOST_DEVICE inline void StoreFitState(const KFParticleGpuFitState& source,
                                                     const KFParticleGpuFitSoAView& destination,
                                                     unsigned int particle)
{
  for (unsigned int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
    destination.Parameter(component, particle) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuFitState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, particle) = source.Covariance(component);
  }

  destination.FitScalar(KFParticleGpuFitSoALayout::Chi2, particle) = source.Chi2();
  destination.FitScalar(KFParticleGpuFitSoALayout::SFromDecay, particle) = source.SFromDecay();
  destination.FitScalar(KFParticleGpuFitSoALayout::SumDaughterMass, particle) = source.SumDaughterMass();
  destination.FitScalar(KFParticleGpuFitSoALayout::MassHypothesis, particle) = source.MassHypo();

  destination.FitInteger(KFParticleGpuFitSoALayout::Ndf, particle) = source.NDF();
  destination.FitInteger(KFParticleGpuFitSoALayout::Charge, particle) = source.Q();
  destination.FitInteger(KFParticleGpuFitSoALayout::AtProductionVertex, particle) =
    source.AtProductionVertex();
  destination.FitInteger(KFParticleGpuFitSoALayout::ConstructMethod, particle) =
    source.ConstructMethod();
}

#endif
