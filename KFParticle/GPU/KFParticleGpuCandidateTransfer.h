/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUCANDIDATETRANSFER_H
#define KFPARTICLEGPUCANDIDATETRANSFER_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuStateTransfer.h"

KFPARTICLE_GPU_HOST_DEVICE inline void LoadCandidateFit(const KFParticleGpuConstCandidatePoolView& source,
                                                        unsigned int candidate,
                                                        KFParticleGpuFitState& destination)
{
  LoadFitState(source.Fit(), candidate, destination);
}

/**
 * Stores fit data at an index reserved by the caller.
 *
 * Capacity failure is returned to the future kernel layer, which owns the
 * atomic overflow update.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool StoreCandidateFit(const KFParticleGpuFitState& source,
                                                         const KFParticleGpuCandidatePoolView& destination,
                                                         unsigned int candidate)
{
  if (!destination.CanStoreCandidate(candidate)) {
    return false;
  }
  StoreFitState(source, destination.Fit(), candidate);
  return true;
}

#endif
