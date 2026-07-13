/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUKERNELSTATE_H
#define KFPARTICLEGPUKERNELSTATE_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuSelection.h"
#include "KFParticleGpuTwoDaughter.h"

#include <type_traits>

/**
 * Non-owning state published to all KFParticle GPU kernels.
 *
 * The future device-storage owner rebuilds this small ABI after a buffer
 * reallocation. Keeping only views here makes the device layout explicit
 * without copying ownership or host-only bookkeeping into constant memory.
 */
class KFParticleGpuKernelState
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernelState() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernelState(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuCandidatePoolView& candidates)
    : fInputTracks(inputTracks), fCandidates(candidates)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernelState(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuEventDesc* events,
    const KFParticleGpuTwoDaughterTask* twoDaughterTasks,
    unsigned int twoDaughterTaskCapacity,
    const KFParticleGpuCandidatePoolView& candidates,
    const KFParticleGpuV0SelectionResultView& selectionResults,
    const KFParticleGpuSelectedCandidateIndexView& selectedCandidates)
    : fInputTracks(inputTracks)
    , fPrimaryVertices(primaryVertices)
    , fEvents(events)
    , fTwoDaughterTasks(twoDaughterTasks)
    , fTwoDaughterTaskCapacity(twoDaughterTaskCapacity)
    , fCandidates(candidates)
    , fSelectionResults(selectionResults)
    , fSelectedCandidates(selectedCandidates)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuConstInputTrackSoAView& InputTracks() const
  {
    return fInputTracks;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuCandidatePoolView& Candidates() const
  {
    return fCandidates;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuConstVertexSoAView& PrimaryVertices() const
  {
    return fPrimaryVertices;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuEventDesc* Events() const { return fEvents; }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterTask* TwoDaughterTasks() const
  {
    return fTwoDaughterTasks;
  }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int TwoDaughterTaskCapacity() const
  {
    return fTwoDaughterTaskCapacity;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuSelectedCandidateIndexView& SelectedCandidates() const
  {
    return fSelectedCandidates;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0SelectionResultView& SelectionResults() const
  {
    return fSelectionResults;
  }

 protected:
  KFParticleGpuConstInputTrackSoAView fInputTracks;
  KFParticleGpuConstVertexSoAView fPrimaryVertices;
  const KFParticleGpuEventDesc* fEvents;
  const KFParticleGpuTwoDaughterTask* fTwoDaughterTasks;
  unsigned int fTwoDaughterTaskCapacity;
  KFParticleGpuCandidatePoolView fCandidates;
  KFParticleGpuV0SelectionResultView fSelectionResults;
  KFParticleGpuSelectedCandidateIndexView fSelectedCandidates;
};

static_assert(std::is_trivially_copyable<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must remain a flat device ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must not require dynamic initialization");

#endif
