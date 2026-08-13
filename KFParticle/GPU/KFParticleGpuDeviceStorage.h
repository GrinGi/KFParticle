/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDEVICESTORAGE_H
#define KFPARTICLEGPUDEVICESTORAGE_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuChannelRouting.h"
#include "KFParticleGpuDecayGraph.h"
#include "KFParticleGpuGraphOperations.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuTwoDaughter.h"
#include "KFParticleGpuTwoDaughterRouting.h"
#include "KFParticleGpuV0Track.h"

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>

/**
 * Visible owner layout for process-persistent KFParticle GPU buffers.
 *
 * Storage is grouped by its role in the reconstruction pipeline. The current
 * buffer-manager API is a transitional adapter; kernels must ultimately use a
 * KFParticleGpuKernelState rebuilt from this owner, never own event memory.
 */
class KFParticleGpuDeviceStorage
{
 public:
  KFParticleGpuDeviceStorage() = default;
  KFParticleGpuDeviceStorage(const KFParticleGpuDeviceStorage&) = delete;
  KFParticleGpuDeviceStorage& operator=(const KFParticleGpuDeviceStorage&) = delete;

  // Packed event input, uploaded once for a device-resident reconstruction chain.
  xpu::buffer<float> fTrackParameters;
  xpu::buffer<float> fTrackCovariances;
  xpu::buffer<float> fTrackField;
  xpu::buffer<float> fTrackChiToPrimaryVertex;
  xpu::buffer<int> fTrackIntegers;
  xpu::buffer<float> fVertexParameters;
  xpu::buffer<float> fVertexCovariances;
  xpu::buffer<float> fVertexChi2;
  xpu::buffer<int> fVertexIntegers;
  xpu::buffer<KFParticleGpuEventDesc> fEvents;

  // Reusable intermediate storage. Kernels will consume these pools directly.
  xpu::buffer<KFParticleGpuTwoDaughterTask> fTwoDaughterTasks;
  xpu::buffer<unsigned int> fTwoDaughterTaskCount;
  xpu::buffer<unsigned int> fTwoDaughterTotalPairCount;
  xpu::buffer<unsigned int> fTwoDaughterTaskOverflowFlags;

  // Step 16 mask-routed first-generation worklist and bounded status.
  xpu::buffer<KFParticleGpuTwoDaughterRoutedTask> fTwoDaughterRoutedTasks;
  xpu::buffer<unsigned int> fTwoDaughterRoutingVisitedPairCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutingActiveBitCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutedAcceptedTaskCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutedStoredTaskCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutingBlockReservationCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutedTaskOverflowFlags;

  // Revision-owned two-daughter descriptors and channel monitoring.
  xpu::buffer<KFParticleGpuTwoDaughterRoutingDescriptor> fTwoDaughterRoutingDescriptors;
  xpu::buffer<KFParticleGpuTwoDaughterCompatibilityEntry> fTwoDaughterRoutingCompatibility;
  xpu::buffer<KFParticleGpuTwoDaughterExecutionGroup> fTwoDaughterRoutingGroups;
  xpu::buffer<KFParticleGpuChannelMask> fTwoDaughterRoutingEnabledChannels;
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelVisitedCounters;
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelAcceptedCounters;
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelStoredCounters;
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelConstructedCounters;
  xpu::buffer<unsigned int> fTwoDaughterSelectionAcceptedCounters;
  xpu::buffer<unsigned int> fTwoDaughterSelectionStoredCounters;
  xpu::buffer<unsigned int> fTwoDaughterSelectionOffsets;
  xpu::buffer<unsigned int> fTwoDaughterSelectionCursors;

  // Compact second-generation worklist. Entries reference raw V0 candidates
  // and packed bachelor tracks; no V0 fit state is copied into this buffer.
  xpu::buffer<KFParticleGpuV0TrackTask> fV0TrackTasks;
  xpu::buffer<unsigned int> fV0TrackTaskCount;
  xpu::buffer<unsigned int> fV0TrackTotalPairCount;
  xpu::buffer<unsigned int> fV0TrackTaskOverflowFlags;

  // Mask-routed work items retain only descriptor and source indices. The
  // route and construction kernels consume this pool in one queue sequence.
  xpu::buffer<KFParticleGpuV0TrackRoutedTask> fV0TrackRoutedTasks;
  xpu::buffer<unsigned int> fV0TrackRoutingVisitedPairCount;
  xpu::buffer<unsigned int> fV0TrackRoutingActiveBitCount;
  xpu::buffer<unsigned int> fV0TrackRoutedAcceptedTaskCount;
  xpu::buffer<unsigned int> fV0TrackRoutedStoredTaskCount;
  xpu::buffer<unsigned int> fV0TrackRoutingBlockReservationCount;
  xpu::buffer<unsigned int> fV0TrackRoutedTaskOverflowFlags;

  // Plan-revision data for mask-driven cascade routing. These buffers are
  // uploaded once per decay-plan change, not once per event.
  xpu::buffer<KFParticleGpuV0TrackRoutingDescriptor> fV0TrackRoutingDescriptors;
  xpu::buffer<KFParticleGpuV0TrackCompatibilityEntry> fV0TrackRoutingCompatibility;
  xpu::buffer<KFParticleGpuV0TrackExecutionGroup> fV0TrackRoutingGroups;
  xpu::buffer<KFParticleGpuChannelMask> fV0TrackRoutingEnabledChannels;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelVisitedCounters;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelAcceptedCounters;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelStoredCounters;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelConstructedCounters;

  // Revision-owned graph contract consumed by the ordered generation scheduler.
  xpu::buffer<KFParticleGpuGraphNode> fDecayGraphNodes;
  xpu::buffer<KFParticleGpuGraphExecutionGroup> fDecayGraphGroups;
  xpu::buffer<KFParticleGpuGraphFamilyCoverage> fDecayGraphFamilyCoverage;

  // Stage 17 later-generation scheduler. Descriptors are revision-owned;
  // tasks, results, and counters are reused for every ordered graph group.
  xpu::buffer<KFParticleGpuGraphOperationDescriptor> fGraphOperationDescriptors;
  xpu::buffer<KFParticleGpuGraphOperationTask> fGraphOperationTasks;
  xpu::buffer<KFParticleGpuGraphOperationResult> fGraphOperationResults;
  xpu::buffer<unsigned int> fGraphOperationVisitedCombinations;
  xpu::buffer<unsigned int> fGraphOperationAcceptedTasks;
  xpu::buffer<unsigned int> fGraphOperationStoredTasks;
  xpu::buffer<unsigned int> fGraphOperationConstructedCandidates;
  xpu::buffer<unsigned int> fGraphOperationRejectedTasks;
  xpu::buffer<unsigned int> fGraphOperationOverflowFlags;
  xpu::buffer<unsigned int> fGraphOperationChannelVisitedCounters;
  xpu::buffer<unsigned int> fGraphOperationChannelAcceptedCounters;
  xpu::buffer<unsigned int> fGraphOperationChannelStoredCounters;
  xpu::buffer<unsigned int> fGraphOperationChannelConstructedCounters;
  xpu::buffer<unsigned int> fGraphOperationChannelRejectedCounters;

  // Diagnostic raw output retained for downstream device stages and debugging.
  xpu::buffer<float> fCandidateParameters;
  xpu::buffer<float> fCandidateCovariances;
  xpu::buffer<float> fCandidateFitScalars;
  xpu::buffer<int> fCandidateFitIntegers;
  xpu::buffer<int> fCandidateMetadataIntegers;
  xpu::buffer<unsigned int> fCandidateMetadataUnsigned;
  // Transient descriptor index for O(1) device-side selection lookup.
  xpu::buffer<unsigned int> fCandidateRoutingDescriptorIndices;
  xpu::buffer<int> fDaughterSourceIds;
  xpu::buffer<unsigned int> fCandidateSize;
  xpu::buffer<unsigned int> fDaughterSize;
  xpu::buffer<unsigned int> fOverflowFlags;

  // One selection diagnostic per raw candidate. It preserves the decision
  // without duplicating the raw fit/covariance SoA.
  xpu::buffer<KFParticleGpuV0SelectionResult> fV0SelectionResults;

  // Compact V0 output. It references the raw pool instead of duplicating fit data.
  xpu::buffer<unsigned int> fSelectedCandidateIndices;
  xpu::buffer<unsigned int> fSelectedCandidateChannelIds;
  xpu::buffer<unsigned int> fSelectedCandidateSize;
  xpu::buffer<unsigned int> fSelectedCandidateOverflowFlags;
};

#else

class KFParticleGpuDeviceStorage
{
};

#endif

#endif
