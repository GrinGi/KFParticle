/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @file
 * @brief Owning XPU buffer handles grouped by reconstruction stage.
 *
 * Allocation policy, capacities, transfers, and published views are managed by
 * KFParticleGpuBufferManager. This header describes storage, not kernel execution.
 */

#ifndef KFPARTICLEGPUDEVICESTORAGE_H
#define KFPARTICLEGPUDEVICESTORAGE_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuChannelRouting.h"
#include "planning/KFParticleGpuDecayGraph.h"
#include "algorithms/KFParticleGpuGraphOperations.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "data/KFParticleGpuTransaction.h"
#include "algorithms/KFParticleGpuTwoDaughter.h"
#include "planning/KFParticleGpuTwoDaughterRouting.h"
#include "algorithms/KFParticleGpuV0Track.h"

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>

/**
 * @brief Owns reusable XPU buffer handles for input, worklists, plans, and reconstruction output.
 *
 * Storage is grouped by its role in the reconstruction pipeline. The buffer
 * manager publishes non-owning kernel views and updates them after allocation;
 * kernels never own event memory. Its implementation inherits this storage layout.
 *
 * The default constructor creates empty handles, not event-sized allocations.
 * Despite the class name, a handle may represent device-only or host/device
 * storage according to the manager's allocation policy. Members do not imply
 * automatic transfers, and this class stores neither a queue nor logical sizes.
 * Callers must finish outstanding accesses before replacing or releasing buffers
 * and refresh affected views after allocation. Copying this owner is disabled.
 */
class KFParticleGpuDeviceStorage
{
 public:
  /// @brief Creates empty buffer handles; allocation is deferred to the buffer manager.
  KFParticleGpuDeviceStorage() = default;

  /// @brief Disallows copying the collection of owning handles.
  KFParticleGpuDeviceStorage(const KFParticleGpuDeviceStorage&) = delete;

  /// @brief Disallows replacing storage through copy assignment.
  KFParticleGpuDeviceStorage& operator=(const KFParticleGpuDeviceStorage&) = delete;

  /// @name Packed event input
  /// Component arrays use the corresponding input/SoA layout and allocation stride.
  /// @{
  xpu::buffer<float> fTrackParameters;  ///< Packed track position/momentum components.
  xpu::buffer<float> fTrackCovariances;  ///< Packed symmetric track covariance components.
  xpu::buffer<float> fTrackField;  ///< Per-track local magnetic-field coefficients.
  xpu::buffer<float> fTrackChiToPrimaryVertex;  ///< Track compatibility values relative to primary vertices.
  xpu::buffer<int> fTrackIntegers;  ///< Packed track identity, charge, PID, and other integer input fields.
  xpu::buffer<float> fVertexParameters;  ///< Packed primary-vertex positions.
  xpu::buffer<float> fVertexCovariances;  ///< Packed primary-vertex covariance components.
  xpu::buffer<float> fVertexChi2;  ///< Primary-vertex fit chi2 values.
  xpu::buffer<int> fVertexIntegers;  ///< Packed vertex integer metadata and fit counters.
  xpu::buffer<KFParticleGpuEventDesc> fEvents;  ///< Packed-event descriptors referring to input ranges.

  /// @}

  /// @name Transaction and event-domain work
  /// Descriptors partition packed events; these buffers do not define continuous-time event membership.
  /// @{
  /// Transaction descriptors referring to domain/work ranges.
  xpu::buffer<KFParticleGpuTransactionDesc> fTransactions;
  xpu::buffer<KFParticleGpuDomainDesc> fDomains;  ///< Event-domain descriptors and storage reservations.
  xpu::buffer<KFParticleGpuWorkRange> fWorkRanges;  ///< Ranges of scheduled work within transactions.
  xpu::buffer<KFParticleGpuWorkStatus> fWorkStatuses;  ///< Per-work-range execution status.
  xpu::buffer<KFParticleGpuPairTile> fTwoDaughterPairTiles;  ///< Event-local tiles of input track pairs.
  xpu::buffer<KFParticleGpuTaskTile> fTwoDaughterTaskTiles;  ///< Tiles of stored routed two-track construction tasks.
  xpu::buffer<KFParticleGpuDomainStatus> fDomainStatuses;  ///< Per-domain progress, counts, and error state.
  /// Per-domain/per-channel reconstruction accounting.
  xpu::buffer<KFParticleGpuDomainChannelStatus> fDomainChannelStatuses;

  /// @}

  /// @name Explicit two-track tasks
  /// Reusable task storage and generation counters.
  /// @{
  xpu::buffer<KFParticleGpuTwoDaughterTask> fTwoDaughterTasks;  ///< Explicit two-track construction tasks.
  xpu::buffer<unsigned int> fTwoDaughterTaskCount;  ///< Explicit two-track task-generation count.
  xpu::buffer<unsigned int> fTwoDaughterTotalPairCount;  ///< Explicit two-track input-pair count.
  xpu::buffer<unsigned int> fTwoDaughterTaskOverflowFlags;  ///< Explicit two-track task-pool overflow flags.

  /// @}

  /// @name Mask-routed two-track tasks
  /// Worklist and counters distinguish accepted work from work retained within capacity.
  /// @{
  xpu::buffer<KFParticleGpuTwoDaughterRoutedTask> fTwoDaughterRoutedTasks;  ///< Compact mask-routed two-track worklist.
  xpu::buffer<unsigned int> fTwoDaughterRoutingVisitedPairCount;  ///< Total pair visits during two-track routing.
  /// Active channel-hypothesis count reported by two-track routing.
  xpu::buffer<unsigned int> fTwoDaughterRoutingActiveBitCount;
  /// Accepted routed tasks, including tasks beyond storage capacity.
  xpu::buffer<unsigned int> fTwoDaughterRoutedAcceptedTaskCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutedStoredTaskCount;  ///< Routed tasks retained within storage capacity.
  /// Block-level output reservations made by two-track routing.
  xpu::buffer<unsigned int> fTwoDaughterRoutingBlockReservationCount;
  xpu::buffer<unsigned int> fTwoDaughterRoutedTaskOverflowFlags;  ///< Routed two-track worklist overflow flags.

  /// @}

  /// @name Two-track routing and selection
  /// Plan descriptors are reusable across events; counters and selection scratch describe current work.
  /// @{
  /// Compiled two-track hypotheses and fit/selection settings.
  xpu::buffer<KFParticleGpuTwoDaughterRoutingDescriptor> fTwoDaughterRoutingDescriptors;
  /// Input-role/PID compatibility entries and channel masks.
  xpu::buffer<KFParticleGpuTwoDaughterCompatibilityEntry> fTwoDaughterRoutingCompatibility;
  /// Groups of two-track hypotheses sharing input roles.
  xpu::buffer<KFParticleGpuTwoDaughterExecutionGroup> fTwoDaughterRoutingGroups;
  /// Mask of enabled compiled two-track hypotheses.
  xpu::buffer<KFParticleGpuChannelMask> fTwoDaughterRoutingEnabledChannels;
  /// Pair visits attributed to each two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelVisitedCounters;
  /// Accepted tasks attributed to each two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelAcceptedCounters;
  /// Stored tasks attributed to each two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelStoredCounters;
  /// Constructed candidates attributed to each two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterRoutingChannelConstructedCounters;
  /// Accepted selection counts per two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterSelectionAcceptedCounters;
  xpu::buffer<unsigned int> fTwoDaughterSelectionStoredCounters;  ///< Stored selection counts per two-track channel.
  xpu::buffer<unsigned int> fTwoDaughterSelectionOffsets;  ///< Per-channel output offsets for selection compaction.
  xpu::buffer<unsigned int> fTwoDaughterSelectionCursors;  ///< Per-channel write cursors for selection compaction.

  /// @}

  /// @name Explicit composite-plus-track tasks
  /// Entries reference candidates and bachelor tracks without embedding parent fit states.
  /// @{
  /// Explicit composite/bachelor references, without embedded parent fit states.
  xpu::buffer<KFParticleGpuV0TrackTask> fV0TrackTasks;
  xpu::buffer<unsigned int> fV0TrackTaskCount;  ///< Explicit composite-plus-track task-generation count.
  xpu::buffer<unsigned int> fV0TrackTotalPairCount;  ///< Explicit composite/bachelor pair count.
  xpu::buffer<unsigned int> fV0TrackTaskOverflowFlags;  ///< Explicit composite-plus-track task-pool overflow flags.

  /// @}

  /// @name Mask-routed composite-plus-track tasks
  /// Compact source/descriptor references consumed by ordered routing and construction stages.
  /// @{
  /// Compact routed composite/bachelor references and channel masks.
  xpu::buffer<KFParticleGpuV0TrackRoutedTask> fV0TrackRoutedTasks;
  xpu::buffer<unsigned int> fV0TrackRoutingVisitedPairCount;  ///< Composite/bachelor pair visits during routing.
  /// Active channel-hypothesis count reported by cascade routing.
  xpu::buffer<unsigned int> fV0TrackRoutingActiveBitCount;
  /// Accepted routed cascade tasks, including capacity losses.
  xpu::buffer<unsigned int> fV0TrackRoutedAcceptedTaskCount;
  xpu::buffer<unsigned int> fV0TrackRoutedStoredTaskCount;  ///< Routed cascade tasks retained within storage capacity.
  /// Block-level reservations made by cascade routing.
  xpu::buffer<unsigned int> fV0TrackRoutingBlockReservationCount;
  xpu::buffer<unsigned int> fV0TrackRoutedTaskOverflowFlags;  ///< Routed cascade task-pool overflow flags.

  /// @}

  /// @name Composite-plus-track routing
  /// Plan tables are refreshed on plan changes; monitoring counters describe current work.
  /// @{
  /// Compiled composite-plus-track hypothesis settings.
  xpu::buffer<KFParticleGpuV0TrackRoutingDescriptor> fV0TrackRoutingDescriptors;
  /// Parent/bachelor compatibility entries and channel masks.
  xpu::buffer<KFParticleGpuV0TrackCompatibilityEntry> fV0TrackRoutingCompatibility;
  /// Cascade hypotheses grouped by input role and generation.
  xpu::buffer<KFParticleGpuV0TrackExecutionGroup> fV0TrackRoutingGroups;
  /// Mask of enabled compiled cascade hypotheses.
  xpu::buffer<KFParticleGpuChannelMask> fV0TrackRoutingEnabledChannels;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelVisitedCounters;  ///< Pair visits attributed to each cascade channel.
  /// Accepted tasks attributed to each cascade channel.
  xpu::buffer<unsigned int> fV0TrackRoutingChannelAcceptedCounters;
  xpu::buffer<unsigned int> fV0TrackRoutingChannelStoredCounters;  ///< Stored tasks attributed to each cascade channel.
  /// Constructed candidates attributed to each cascade channel.
  xpu::buffer<unsigned int> fV0TrackRoutingChannelConstructedCounters;

  /// @}

  /// @name Compiled decay graph
  /// Revision-associated structural tables for ordered generation scheduling.
  /// @{
  /// Compiled channel nodes with dependency and operation metadata.
  xpu::buffer<KFParticleGpuGraphNode> fDecayGraphNodes;
  /// Ordered execution groups referencing compiled nodes.
  xpu::buffer<KFParticleGpuGraphExecutionGroup> fDecayGraphGroups;
  /// Implementation-coverage records for CPU finder families.
  xpu::buffer<KFParticleGpuGraphFamilyCoverage> fDecayGraphFamilyCoverage;

  /// @}

  /// @name Generic graph operations
  /// Descriptors belong to the plan; task/result storage and counters are reused across execution groups.
  /// @{
  /// Numerical and selection settings for generic graph operations.
  xpu::buffer<KFParticleGpuGraphOperationDescriptor> fGraphOperationDescriptors;
  /// Source references for scheduled graph-operation tasks.
  xpu::buffer<KFParticleGpuGraphOperationTask> fGraphOperationTasks;
  /// Per-task results/status for graph operations.
  xpu::buffer<KFParticleGpuGraphOperationResult> fGraphOperationResults;
  /// Input combinations visited by graph task generation.
  xpu::buffer<unsigned int> fGraphOperationVisitedCombinations;
  xpu::buffer<unsigned int> fGraphOperationAcceptedTasks;  ///< Accepted graph tasks, including tasks beyond capacity.
  xpu::buffer<unsigned int> fGraphOperationStoredTasks;  ///< Graph tasks retained within storage capacity.
  xpu::buffer<unsigned int> fGraphOperationConstructedCandidates;  ///< Candidates constructed by graph operations.
  xpu::buffer<unsigned int> fGraphOperationRejectedTasks;  ///< Graph tasks rejected during execution.
  xpu::buffer<unsigned int> fGraphOperationOverflowFlags;  ///< Graph-operation capacity overflow flags.
  /// Visited combinations attributed to each operation channel.
  xpu::buffer<unsigned int> fGraphOperationChannelVisitedCounters;
  /// Accepted tasks attributed to each operation channel.
  xpu::buffer<unsigned int> fGraphOperationChannelAcceptedCounters;
  /// Stored tasks attributed to each operation channel.
  xpu::buffer<unsigned int> fGraphOperationChannelStoredCounters;
  /// Constructed candidates attributed to each operation channel.
  xpu::buffer<unsigned int> fGraphOperationChannelConstructedCounters;
  /// Rejected tasks attributed to each operation channel.
  xpu::buffer<unsigned int> fGraphOperationChannelRejectedCounters;

  /// @}

  /// @name Raw candidate pool
  /// Fit SoA and lineage storage serve downstream device stages as well as optional diagnostic readback.
  /// @{
  xpu::buffer<float> fCandidateParameters;  ///< Raw particle parameter components in SoA layout.
  xpu::buffer<float> fCandidateCovariances;  ///< Raw packed covariance components in SoA layout.
  xpu::buffer<float> fCandidateFitScalars;  ///< Raw fit-quality, mass-hypothesis, and transport scalars.
  xpu::buffer<int> fCandidateFitIntegers;  ///< Raw charge, NDF, and fit-state integer fields.
  xpu::buffer<int> fCandidateMetadataIntegers;  ///< Signed reconstruction metadata in SoA layout.
  xpu::buffer<unsigned int> fCandidateMetadataUnsigned;  ///< Unsigned reconstruction metadata in SoA layout.
  /// Transient descriptor indices for constant-time selection lookup.
  xpu::buffer<unsigned int> fCandidateRoutingDescriptorIndices;
  xpu::buffer<int> fDaughterSourceIds;  ///< Flattened source-track lineage of raw candidates.
  xpu::buffer<unsigned int> fCandidateSize;  ///< Logical raw-candidate pool size, separate from allocation capacity.
  xpu::buffer<unsigned int> fDaughterSize;  ///< Logical flattened daughter-ID pool size.
  xpu::buffer<unsigned int> fOverflowFlags;  ///< Raw candidate/lineage pool overflow flags.

  /// @}

  /// @name Selection decisions and compact output
  /// Decisions and selected indices reference raw candidates without duplicating fit/covariance data.
  /// @{
  /// Selection decision and diagnostic values for each raw candidate.
  xpu::buffer<KFParticleGpuV0SelectionResult> fV0SelectionResults;

  xpu::buffer<unsigned int> fSelectedCandidateIndices;  ///< Selected entries referencing the raw candidate pool.
  xpu::buffer<unsigned int> fSelectedCandidateChannelIds;  ///< Channel IDs corresponding to selected entries.
  xpu::buffer<unsigned int> fSelectedCandidateSize;  ///< Logical number of stored selected entries.
  xpu::buffer<unsigned int> fSelectedCandidateOverflowFlags;  ///< Selected-output capacity overflow flags.
  /// @}
};

#else

/// @brief Empty storage placeholder for builds without XPU; contains no allocations or device API.
class KFParticleGpuDeviceStorage
{
};

#endif

#endif
