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
 * @brief Shared non-owning kernel state, XPU image identity, and transaction diagnostic records.
 *
 * This header defines data passed to kernels, not launch policy or reconstruction
 * algorithms. Owners allocate the referenced buffers; steering publishes refreshed
 * views and coordinates their lifetime with device execution.
 */

#ifndef KFPARTICLEGPUKERNELSTATE_H
#define KFPARTICLEGPUKERNELSTATE_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuChannelRouting.h"
#include "planning/KFParticleGpuDecayGraph.h"
#include "algorithms/KFParticleGpuGraphOperations.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuPlatform.h"
#include "algorithms/KFParticleGpuSelection.h"
#include "algorithms/KFParticleGpuTwoDaughter.h"
#include "planning/KFParticleGpuTwoDaughterRouting.h"
#include "data/KFParticleGpuTransaction.h"

#include <type_traits>

#ifdef KFPARTICLE_GPU_XPU_ENABLED
/// @brief XPU device-image tag shared by KFParticle kernels and constant-memory declarations.
struct KFParticleGpuDeviceImage : xpu::device_image {};
#else
/// @brief Image-identity placeholder for builds without the XPU device interface.
struct KFParticleGpuDeviceImage {};
#endif

/**
 * @brief Requested checkpoints in deterministic two-track transaction diagnostics.
 * Values identify progressively deeper probes, not production scheduling stages or a bit mask.
 */
enum KFParticleGpuTransactionDiagnosticPhase : unsigned int
{
  KFGpuTransactionDiagnosticJournal = 1u,       ///< Initialize/write the device journal without accessing event input.
  KFGpuTransactionDiagnosticTransaction,      ///< Read the transaction descriptor and its identity.
  KFGpuTransactionDiagnosticWork,             ///< Check the selected work range and read its descriptor.
  KFGpuTransactionDiagnosticDomain,           ///< Resolve the work item's event domain.
  KFGpuTransactionDiagnosticEventRouting,     ///< Read event track ranges and the routing execution group.
  KFGpuTransactionDiagnosticInputScalars,     ///< Resolve a pair and load source IDs, PID, and PV associations.
  KFGpuTransactionDiagnosticCompatibility,    ///< Query known and enabled channel compatibility for the pair.
  KFGpuTransactionDiagnosticTrackCuts,        ///< Evaluate per-descriptor track preselection.
  KFGpuTransactionDiagnosticDcaStateField,    ///< Load pair-gate numerical states and local field information.
  KFGpuTransactionDiagnosticDcaRoots,         ///< Evaluate the pair-gate closest-approach transport parameters.
  KFGpuTransactionDiagnosticSecondTransport,  ///< Probe transport of the second track.
  KFGpuTransactionDiagnosticFirstTransport,   ///< Probe transport of the first track.
  KFGpuTransactionDiagnosticPairDecision      ///< Evaluate the final pair-gate decision and observables.
};

/// @brief Outcome of a diagnostic probe, distinct from candidate acceptance and production task status.
enum KFParticleGpuTransactionDiagnosticStatus : unsigned int
{
  KFGpuTransactionDiagnosticNotRun = 0u,               ///< Default record with no reported outcome.
  KFGpuTransactionDiagnosticCompleted,               ///< Requested probe completed successfully.
  KFGpuTransactionDiagnosticRejected,                ///< Pair lookup, compatibility, or pair gate rejected the probe.
  KFGpuTransactionDiagnosticInvalidContract,         ///< A checked pointer/range/role contract failed.
  KFGpuTransactionDiagnosticDescriptorNotApplicable  ///< Requested descriptor is absent or did not pass the pre-gate.
};

/**
 * @brief Checkpoint bits set after the probe passes the corresponding structural checks.
 * These report progress through implemented guards, not exhaustive validation of all referenced storage.
 */
enum KFParticleGpuTransactionDiagnosticGuard : unsigned int
{
  KFGpuTransactionDiagnosticTransactionGuard = 1u << 0u,  ///< Transaction descriptor was available and read.
  KFGpuTransactionDiagnosticWorkGuard = 1u << 1u,         ///< Selected work index passed range checks.
  KFGpuTransactionDiagnosticDomainGuard = 1u << 2u,       ///< Domain reference and required work family passed checks.
  KFGpuTransactionDiagnosticEventGuard = 1u << 3u,        ///< Event input pointer passed the availability check.
  KFGpuTransactionDiagnosticTrackGuard = 1u << 4u,        ///< Both resolved track indices fit the logical input size.
  KFGpuTransactionDiagnosticRoutingGuard = 1u << 5u       ///< Routing execution-group index passed its bounds check.
};

/**
 * @brief Flat journal slot for one deterministic transaction/work/pair probe.
 *
 * Diagnostic kernels reset and populate this value; the host reads it after each
 * phase. Fields beyond the reached checkpoint retain defaults and must not be
 * interpreted as measured zeros. phase names the requested checkpoint, while
 * completedPhase and guardFlags describe reported progress; rejection may still
 * occur after a checkpoint completes. This is not an accumulating production log.
 *
 * The record itself performs no synchronization or readback. A separately emitted
 * host BEGIN record can identify the attempted phase when a device fault prevents
 * retrieving this slot. Concurrent probes require separate records or serialized access.
 */
struct KFParticleGpuTransactionDiagnosticRecord
{
  /// Host-assigned probe sequence used to correlate host and device records.
  unsigned long long diagnosticSequence = 0u;
  unsigned long long transactionId = 0u;  ///< Transaction identity read from the supplied transaction descriptor.
  unsigned long long planRevision = 0u;  ///< Decay-plan revision recorded in the transaction descriptor.
  unsigned int phase = 0u;  ///< Requested KFParticleGpuTransactionDiagnosticPhase; zero before a probe sets it.
  /// Probe outcome from KFParticleGpuTransactionDiagnosticStatus.
  unsigned int status = KFGpuTransactionDiagnosticNotRun;
  unsigned int completedPhase = 0u;  ///< Checkpoint reported as completed; zero when none was recorded.
  unsigned int guardFlags = 0u;  ///< Progress bits from KFParticleGpuTransactionDiagnosticGuard.
  unsigned int workIndex = 0u;  ///< Absolute selected index in transaction work-range storage.
  unsigned int domainIndex = 0u;  ///< Domain index referenced by the selected work item.
  unsigned int eventIndex = 0u;  ///< Packed input-event descriptor index, not the producer's event ID.
  unsigned int groupIndex = 0u;  ///< Two-track routing execution-group index.
  /// Diagnostic Cartesian pair index within the two event-local track ranges, not a tile index.
  unsigned int pairIndex = 0u;
  /// Requested routing-descriptor index; InvalidBit means none was selected.
  unsigned int descriptorIndex = KFParticleGpuChannelMask::InvalidBit;
  unsigned int firstTrackIndex = 0u;  ///< Resolved first packed-track index.
  unsigned int secondTrackIndex = 0u;  ///< Resolved second packed-track index.
  int firstSourceId = 0;  ///< First track's source-lineage ID.
  int secondSourceId = 0;  ///< Second track's source-lineage ID.
  int firstPdg = 0;  ///< First track's input PID hypothesis.
  int secondPdg = 0;  ///< Second track's input PID hypothesis.
  int firstPrimaryVertex = -1;  ///< First track's packed PV association, or a negative unassigned value.
  int secondPrimaryVertex = -1;  ///< Second track's packed PV association, or a negative unassigned value.
  /// Enabled PID-compatible hypotheses intersected with the execution-group mask.
  KFParticleGpuChannelMask compatibleChannels;
  KFParticleGpuChannelMask preGateChannels;  ///< Compatible hypotheses that pass per-descriptor track preselection.
  unsigned int secondTransportStatus = 0u;  ///< Second-track transport status copied from the pair-gate trace.
  unsigned int firstTransportStatus = 0u;  ///< First-track transport status copied from the pair-gate trace.
  float fieldBy = 0.f;  ///< Local By value reported by the pair-gate probe.
  float secondDs = 0.f;  ///< Second-track transport parameter reported by the pair gate.
  float firstDs = 0.f;  ///< First-track transport parameter reported by the pair gate.
  float distance2 = 0.f;  ///< Squared daughter separation reported by the pair gate.
  float maxDistance2 = 0.f;  ///< Squared separation threshold used by the pair gate.
  float momentumDot = 0.f;  ///< Dot product of the two momenta reported by the pair gate.
  float firstMomentum2 = 0.f;  ///< Squared first-track momentum reported by the pair gate.
  float secondMomentum2 = 0.f;  ///< Squared second-track momentum reported by the pair gate.
  unsigned int accepted = 0u;  ///< Pair-gate trace acceptance flag, not a published-candidate count.
};

static_assert(std::is_trivially_copyable<KFParticleGpuTransactionDiagnosticRecord>::value,
              "KFParticle GPU transaction diagnostic ABI must remain trivially copyable");

/**
 * @brief Collects borrowed input, output, routing, graph, and transaction views for device kernels.
 *
 * The device-storage owner rebuilds this small ABI after a buffer
 * reallocation. Keeping only views here makes the device layout explicit
 * without copying ownership or host-only bookkeeping into constant memory.
 * KFParticleGpuKernels inherits this data layout and supplies device-side behavior.
 *
 * Constructors copy view values and pointers, not their backing arrays. They do
 * not validate dimensions, initialize output storage, transfer data, or launch work.
 * Owners must outlive all kernel accesses, and steering must republish changed views
 * before their next use. Const access to this state does not make output buffers
 * read-only: each view's element types determine whether pointees are writable.
 */
class KFParticleGpuKernelState
{
 public:
  /**
   * @brief Preserves trivial construction for the device ABI without binding buffers.
   * @note Plain default-initialization leaves scalar/pointer members indeterminate.
   * Value-initialization with {} zeroes the trivial state but does not make views usable.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernelState() = default;

  /**
   * @brief Binds only the input-track and candidate-pool views for minimal numerical/probe paths.
   * @param inputTracks Borrowed read-only packed tracks.
   * @param candidates Borrowed writable raw-candidate storage.
   * @note Routing, descriptor-index, graph, and generation views are value-initialized.
   * Omitted PV/event/task, selection, and transaction members are not established as
   * usable views by this constructor. In particular, raw pointers and task capacity
   * remain uninitialized. Do not use this overload for the full reconstruction chain.
   */
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

  /**
   * @brief Binds the complete published-state layout without taking ownership of any referenced storage.
   * @param inputTracks Packed read-only track parameters and metadata.
   * @param primaryVertices Packed read-only PV data.
   * @param events Borrowed event descriptors; this state stores no separate event-array capacity.
   * @param twoDaughterTasks Explicit two-track task array for paths that consume it.
   * @param twoDaughterTaskCapacity Allocation bound of that explicit task array, not a task count.
   * @param candidates Writable raw-candidate and flattened-lineage pool.
   * @param selectionResults Per-candidate selection diagnostics.
   * @param selectedCandidates Compact selected-output indices and associated counters.
   * @param v0TrackRouting Composite-plus-track routing tables and channel monitoring.
   * @param twoDaughterRouting Two-track routing tables and channel monitoring.
   * @param candidateDescriptorIndices Candidate-to-routing-descriptor lookup storage.
   * @param decayGraph Compiled graph nodes, groups, and family coverage.
   * @param graphOperations Generic graph-operation descriptors, tasks, results, and counters.
   * @param twoDaughterGeneration Reusable routed two-track generation workspace.
   * @param v0TrackGeneration Reusable routed composite-plus-track generation workspace.
   * @param transaction Event-domain transaction descriptors, reservations, and progress state.
   * @note Default optional views are empty, not automatically populated. Callers must
   * supply every view needed by their selected kernel path and keep all layouts consistent.
   */
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
      KFParticleGpuV0TrackGenerationStorageView(),
    const KFParticleGpuTransactionView& transaction = KFParticleGpuTransactionView())
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
    , fTransaction(transaction)
  {
  }

  /// @brief Borrows the read-only packed track view; does not copy input data.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuConstInputTrackSoAView& InputTracks() const
  {
    return fInputTracks;
  }

  /// @brief Borrows the writable raw-candidate pool view, even through a const state.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuCandidatePoolView& Candidates() const
  {
    return fCandidates;
  }

  /// @brief Borrows the read-only PV view; requires a constructor/path that binds it.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuConstVertexSoAView& PrimaryVertices() const
  {
    return fPrimaryVertices;
  }

  /// @brief Returns the borrowed event array without checking its pointer or index bounds.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuEventDesc* Events() const { return fEvents; }

  /// @brief Returns the borrowed explicit-task array; not the routed generation worklist.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterTask* TwoDaughterTasks() const
  {
    return fTwoDaughterTasks;
  }

  /// @brief Returns the explicit-task allocation bound, not the number of active tasks.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TwoDaughterTaskCapacity() const
  {
    return fTwoDaughterTaskCapacity;
  }

  /// @brief Borrows the compact selected-index output view and its counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuSelectedCandidateIndexView& SelectedCandidates() const
  {
    return fSelectedCandidates;
  }

  /// @brief Borrows the per-candidate selection-result view.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0SelectionResultView& SelectionResults() const
  {
    return fSelectionResults;
  }

  /// @brief Borrows composite-plus-track routing tables and monitoring counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackRoutingView& V0TrackRouting() const
  {
    return fV0TrackRouting;
  }

  /// @brief Borrows two-track routing tables and monitoring counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterRoutingView&
  TwoDaughterRouting() const
  {
    return fTwoDaughterRouting;
  }

  /// @brief Borrows the transient candidate-to-routing-descriptor lookup view.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuCandidateDescriptorIndexView&
  CandidateDescriptorIndices() const
  {
    return fCandidateDescriptorIndices;
  }

  /// @brief Borrows compiled graph metadata without validating dependencies or revision freshness.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuDecayGraphView& DecayGraph() const
  {
    return fDecayGraph;
  }

  /// @brief Borrows generic graph-operation descriptors, tasks, results, and counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphOperationStorageView&
  GraphOperations() const
  {
    return fGraphOperations;
  }

  /// @brief Borrows the routed two-track generation workspace and counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTwoDaughterGenerationStorageView&
  TwoDaughterGeneration() const
  {
    return fTwoDaughterGeneration;
  }

  /// @brief Borrows the routed composite-plus-track generation workspace and counters.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuV0TrackGenerationStorageView&
  V0TrackGeneration() const
  {
    return fV0TrackGeneration;
  }

  /// @brief Borrows transaction/domain/work descriptors and progress storage without validating them.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTransactionView& Transaction() const
  {
    return fTransaction;
  }

 protected:
  KFParticleGpuConstInputTrackSoAView fInputTracks;  ///< Borrowed read-only packed tracks and input metadata.
  KFParticleGpuConstVertexSoAView fPrimaryVertices;  ///< Borrowed read-only packed primary vertices.
  const KFParticleGpuEventDesc* fEvents;  ///< Borrowed event-descriptor array; extent is managed externally.
  const KFParticleGpuTwoDaughterTask* fTwoDaughterTasks;  ///< Borrowed explicit two-track task array.
  unsigned int fTwoDaughterTaskCapacity;  ///< Allocation bound of the explicit two-track task array.
  KFParticleGpuCandidatePoolView fCandidates;  ///< Borrowed writable raw-candidate and lineage pool.
  KFParticleGpuV0SelectionResultView fSelectionResults;  ///< Borrowed per-candidate selection decisions/diagnostics.
  /// Borrowed compact selected-index output and counters.
  KFParticleGpuSelectedCandidateIndexView fSelectedCandidates;
  KFParticleGpuV0TrackRoutingView fV0TrackRouting;  ///< Borrowed composite-plus-track routing tables and monitoring.
  KFParticleGpuTwoDaughterRoutingView fTwoDaughterRouting;  ///< Borrowed two-track routing tables and monitoring.
  /// Borrowed candidate-to-routing-descriptor mapping.
  KFParticleGpuCandidateDescriptorIndexView fCandidateDescriptorIndices;
  KFParticleGpuDecayGraphView fDecayGraph;  ///< Borrowed compiled graph tables.
  KFParticleGpuGraphOperationStorageView fGraphOperations;  ///< Borrowed generic graph-operation workspace.
  /// Borrowed routed two-track task-generation workspace.
  KFParticleGpuTwoDaughterGenerationStorageView fTwoDaughterGeneration;
  /// Borrowed routed composite-plus-track task-generation workspace.
  KFParticleGpuV0TrackGenerationStorageView fV0TrackGeneration;
  KFParticleGpuTransactionView fTransaction;  ///< Borrowed event-domain transaction layout and mutable progress state.
};

static_assert(std::is_trivially_copyable<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must remain a flat device ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must not require dynamic initialization");

#endif
