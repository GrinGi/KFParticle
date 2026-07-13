/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUINPUTDATATRANSFER_H
#define KFPARTICLEGPUINPUTDATATRANSFER_H

#include "KFParticleGpuInputData.h"
#include "KFParticleGpuStateTransfer.h"

KFPARTICLE_GPU_HOST_DEVICE inline void LoadTrackState(const KFParticleGpuConstInputTrackSoAView& source,
                                                      unsigned int track,
                                                      KFParticleGpuTrackState& destination)
{
  LoadTrackState(source.Numerical(), track, destination);
}

KFPARTICLE_GPU_HOST_DEVICE inline void StoreTrackState(const KFParticleGpuTrackState& source,
                                                       KFParticleGpuInputTrackSoAView& destination,
                                                       unsigned int track)
{
  StoreTrackState(source, destination.Numerical(), track);
}

KFPARTICLE_GPU_HOST_DEVICE inline bool HasFieldRegions(
  const KFParticleGpuConstInputTrackSoAView& source)
{
  return source.FieldCoefficientsData() != nullptr;
}

KFPARTICLE_GPU_HOST_DEVICE inline void LoadFieldRegion(const KFParticleGpuConstInputTrackSoAView& source,
                                                       unsigned int track,
                                                       KFParticleGpuFieldRegion& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
    destination.Coefficient(component) = source.FieldCoefficient(component, track);
  }
}

KFPARTICLE_GPU_HOST_DEVICE inline bool LoadFieldRegionOrZero(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  KFParticleGpuFieldRegion& destination)
{
  destination.Reset();
  if (!HasFieldRegions(source) || track >= source.Size()) {
    return false;
  }
  LoadFieldRegion(source, track, destination);
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuFieldValue EvaluateTrackField(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  float z)
{
  KFParticleGpuFieldRegion field;
  LoadFieldRegionOrZero(source, track, field);
  return field.Get(z);
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuFieldValue EvaluateTrackFieldAtState(
  const KFParticleGpuConstInputTrackSoAView& source,
  unsigned int track,
  const KFParticleGpuFitState& state)
{
  return EvaluateTrackField(source, track, state.Z());
}

KFPARTICLE_GPU_HOST_DEVICE inline void StoreFieldRegion(const KFParticleGpuFieldRegion& source,
                                                        KFParticleGpuInputTrackSoAView& destination,
                                                        unsigned int track)
{
  for (unsigned int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
    destination.FieldCoefficient(component, track) = source.Coefficient(component);
  }
}

KFPARTICLE_GPU_HOST_DEVICE inline void LoadVertexState(const KFParticleGpuConstVertexSoAView& source,
                                                       unsigned int vertex,
                                                       KFParticleGpuVertexState& destination)
{
  for (unsigned int component = 0; component < KFParticleGpuVertexState::NumberOfParameters; ++component) {
    destination.Parameter(component) = source.Parameter(component, vertex);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuVertexState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component) = source.Covariance(component, vertex);
  }
  destination.Chi2() = source.Chi2(vertex);
  destination.NDF() = source.NDF(vertex);
  destination.NContributors() = source.NContributors(vertex);
}

KFPARTICLE_GPU_HOST_DEVICE inline void StoreVertexState(const KFParticleGpuVertexState& source,
                                                        KFParticleGpuVertexSoAView& destination,
                                                        unsigned int vertex)
{
  for (unsigned int component = 0; component < KFParticleGpuVertexState::NumberOfParameters; ++component) {
    destination.Parameter(component, vertex) = source.Parameter(component);
  }
  for (unsigned int component = 0;
       component < KFParticleGpuVertexState::NumberOfCovarianceElements;
       ++component) {
    destination.Covariance(component, vertex) = source.Covariance(component);
  }
  destination.Chi2(vertex) = source.Chi2();
  destination.NDF(vertex) = source.NDF();
  destination.NContributors(vertex) = source.NContributors();
}

#endif
