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
#include "KFParticleGpuChannelRouting.h"
#include "KFParticleGpuDecayGraph.h"
#include "KFParticleGpuGraphOperations.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuSelection.h"
#include "KFParticleGpuTwoDaughter.h"
#include "KFParticleGpuTwoDaughterRouting.h"

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
    : fInputTracks(inputTracks)
    , fCandidates(candidates)
    , fV0TrackRouting()
    , fTwoDaughterRouting()
    , fCandidateDescriptorIndices()
    , fDecayGraph()
    , fGraphOperations()
    , fTwoDaughterGeneration()
    , fV0TrackGeneration()
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
    const KFParticleGpuSelectedCandidateIndexView& selectedCandidates,
    const KFParticleGpuV0TrackRoutingView& v0TrackRouting =
      KFParticleGpuV0TrackRoutingView(),
    const KFParticleGpuTwoDaughterRoutingView& twoDaughterRouting =
      KFParticleGpuTwoDaughterRoutingView(),
    const KFParticleGpuCandidateDescriptorIndexView& candidateDescriptorIndices =
      KFParticleGpuCandidateDescriptorIndexView(),
    const KFParticleGpuDecayGraphView& decayGraph =
      KFParticleGpuDecayGraphView(),
    const KFParticleGpuGraphOperationStorageView& graphOperations =
      KFParticleGpuGraphOperationStorageView(),
    const KFParticleGpuTwoDaughterGenerationStorageView& twoDaughterGeneration =
      KFParticleGpuTwoDaughterGenerationStorageView(),
    const KFParticleGpuV0TrackGenerationStorageView& v0TrackGeneration =
      KFParticleGpuV0TrackGenerationStorageView())
    : fInputTracks(inputTracks)
    , fPrimaryVertices(primaryVertices)
    , fEvents(events)
    , fTwoDaughterTasks(twoDaughterTasks)
    , fTwoDaughterTaskCapacity(twoDaughterTaskCapacity)
    , fCandidates(candidates)
    , fSelectionResults(selectionResults)
    , fSelectedCandidates(selectedCandidates)
    , fV0TrackRouting(v0TrackRouting)
    , fTwoDaughterRouting(twoDaughterRouting)
    , fCandidateDescriptorIndices(candidateDescriptorIndices)
    , fDecayGraph(decayGraph)
    , fGraphOperations(graphOperations)
    , fTwoDaughterGeneration(twoDaughterGeneration)
    , fV0TrackGeneration(v0TrackGeneration)
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

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackRoutingView& V0TrackRouting() const
  {
    return fV0TrackRouting;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterRoutingView&
  TwoDaughterRouting() const
  {
    return fTwoDaughterRouting;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuCandidateDescriptorIndexView&
  CandidateDescriptorIndices() const
  {
    return fCandidateDescriptorIndices;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuDecayGraphView& DecayGraph() const
  {
    return fDecayGraph;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphOperationStorageView&
  GraphOperations() const
  {
    return fGraphOperations;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterGenerationStorageView&
  TwoDaughterGeneration() const
  {
    return fTwoDaughterGeneration;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackGenerationStorageView&
  V0TrackGeneration() const
  {
    return fV0TrackGeneration;
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
  KFParticleGpuV0TrackRoutingView fV0TrackRouting;
  KFParticleGpuTwoDaughterRoutingView fTwoDaughterRouting;
  KFParticleGpuCandidateDescriptorIndexView fCandidateDescriptorIndices;
  KFParticleGpuDecayGraphView fDecayGraph;
  KFParticleGpuGraphOperationStorageView fGraphOperations;
  KFParticleGpuTwoDaughterGenerationStorageView fTwoDaughterGeneration;
  KFParticleGpuV0TrackGenerationStorageView fV0TrackGeneration;
};

static_assert(std::is_trivially_copyable<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must remain a flat device ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must not require dynamic initialization");

#endif
