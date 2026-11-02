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
 * @brief Flat graph-operation descriptors, workspace views, and host/device execution helpers.
 *
 * These helpers evaluate individual composite, missing-mass, matching, and unary
 * tasks. Scheduling, buffer allocation, atomic output reservations, and transfers
 * belong to the caller. Local fit/lineage construction is separate from pool publication.
 */

#ifndef KFPARTICLEGPUGRAPHOPERATIONS_H
#define KFPARTICLEGPUGRAPHOPERATIONS_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuDecayGraph.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "algorithms/KFParticleGpuMath.h"
#include "algorithms/KFParticleGpuSelection.h"

#include <type_traits>

/**
 * @brief Task outcome bits shared by numerical execution and output-publication kernels.
 * Accepted denotes successful evaluation; output reservation/publication may still fail afterward.
 */
enum KFParticleGpuGraphTaskStatus : unsigned int
{
  KFGpuGraphTaskPending = 0u,                     ///< No outcome recorded yet.
  KFGpuGraphTaskAccepted = 1u << 0u,               ///< Numerical operation and requested selection succeeded.
  KFGpuGraphTaskRejectDescriptor = 1u << 1u,       ///< Invalid descriptor or descriptor index.
  KFGpuGraphTaskRejectSource = 1u << 2u,           ///< Source, its lineage, or required PV could not be loaded.
  KFGpuGraphTaskRejectLineage = 1u << 3u,          ///< Invalid source ordering, repeated lineage, or lineage overflow.
  KFGpuGraphTaskRejectMathematics = 1u << 4u,      ///< Fit, transport, constraint, or observable calculation failed.
  KFGpuGraphTaskRejectSelection = 1u << 5u,        ///< A requested matching or selection requirement failed.
  KFGpuGraphTaskRejectCandidateCapacity = 1u << 6u,  ///< No output candidate slot could be reserved.
  KFGpuGraphTaskRejectDaughterCapacity = 1u << 7u  ///< No output lineage range could be reserved.
};

/// @brief Overflow bits for the graph-operation worklist, separate from candidate-pool overflow.
enum KFParticleGpuGraphOperationOverflow : unsigned int
{
  KFGpuGraphTaskCapacityExceeded = 1u << 0u  ///< Generated work exceeded task storage capacity.
};

/**
 * @brief Additional source, matching, constraint, and selection policies.
 * @note Same-event and distinct-source bits express the descriptor contract.
 * ExecuteTask() checks candidate event identity and merged lineage unconditionally;
 * clearing these bits does not permit cross-event candidates or repeated lineage.
 * Raw-track event membership is established by the caller's event-local task generation.
 */
enum KFParticleGpuGraphOperationFlags : unsigned int
{
  KFGpuGraphRequireSameEvent = 1u << 0u,        ///< Declares an event-local source contract.
  KFGpuGraphRequireDistinctSources = 1u << 1u,  ///< Declares disjoint flattened source-track lineage.
  KFGpuGraphApplyMassConstraint = 1u << 2u,     ///< Requests the descriptor's mass constraint.
  KFGpuGraphRequirePositiveMissingEnergy = 1u << 3u,  ///< Requires first.E() > second.E() before subtraction.
  /// Validate the matching residual but return the first source, including any applied pre-match mass constraint.
  KFGpuGraphStoreFirstOnMatch = 1u << 4u,
  /// For equal source kinds, require the second source index to be greater than the first.
  KFGpuGraphRequireOrderedCandidatePair = 1u << 5u,
  /// Match a first-source candidate against the primary vertex assigned to the second-source track.
  KFGpuGraphMatchTrackPrimaryVertex = 1u << 6u,
  /// Select using the task's event-local PV range, choosing the best valid constrained topology.
  KFGpuGraphSelectEventPrimaryVertices = 1u << 7u
};

/// @brief Numerical interpretation of a missing-mass operation after preparing the daughter measurement.
enum KFParticleGpuMissingMassMode : unsigned int
{
  KFGpuGraphMissingMassLegacySubtract = 0u,  ///< Subtract the daughter and return the residual fit state.
  /// Reconstruct the neutral and filtered charged states; ExecuteTask() returns the filtered mother.
  KFGpuGraphMissingMassFilteredReconstruction = 1u
};

/// @brief Resolve the output PV from the first source's metadata, whether that source is a candidate or a track.
constexpr int KFGpuGraphPrimaryVertexFromCandidate = -2;

/**
 * @brief Event-independent configuration of one generic graph-operation channel.
 *
 * A flat value uploaded with a graph revision: topology selects the numerical
 * branch, operationMask selects processing steps, and flags refine their policy.
 * Fields are interpreted only by the relevant branch; negative cut defaults
 * generally omit optional cuts. Default construction is not a valid channel.
 * ValidateDescriptor() checks supported combinations, not all numerical inputs.
 */
struct KFParticleGpuGraphOperationDescriptor
{
  unsigned int channelId = 0u;  ///< Stable channel identity; zero is unconfigured.
  unsigned int topology = KFGpuGraphTopologyInvalid;  ///< Supported source/operation topology.
  unsigned int operationMask = 0u;  ///< Requested KFParticleGpuGraphOperation bits.
  unsigned int outputClass = KFGpuGraphOutputInvalid;  ///< Reconstruction output category.
  /// Additional source, constraint, and matching policies.
  unsigned int flags = KFGpuGraphRequireSameEvent | KFGpuGraphRequireDistinctSources;
  /// Subtraction or filtered-reconstruction branch.
  unsigned int missingMassMode = KFGpuGraphMissingMassLegacySubtract;
  int motherPdg = 0;  ///< PDG hypothesis assigned to the published output.
  int firstPdg = 0;  ///< First-source matching hypothesis; zero is unrestricted.
  int secondPdg = 0;  ///< Second-source matching hypothesis; ignored for unary operations.
  int primaryVertexIndex = -1;  ///< Packed PV index, -1 if unassigned, or KFGpuGraphPrimaryVertexFromCandidate.
  float firstMass = 0.f;  ///< First raw-track mass or filtered mother mass hypothesis in GeV/c^2.
  float secondMass = 0.f;  ///< Second raw-track mass or filtered daughter mass hypothesis in GeV/c^2.
  float neutralMass = 0.f;  ///< Neutral mass hypothesis for filtered missing-mass reconstruction in GeV/c^2.
  float massConstraint = -1.f;  ///< Constraint target and matching mass-window center; negative means unspecified.
  float massConstraintSigma = 0.f;  ///< Mass-constraint uncertainty; zero requests an exact constraint.
  /// Geometric fit chi2/NDF upper cut when Select is requested; negative omits it.
  float maxGeometricChi2PerNdf = -1.f;
  float maxTopologyChi2PerNdf = -1.f;  ///< PV topology chi2/NDF upper cut when Select is requested; negative omits it.
  float maxMatchMomentum = -1.f;  ///< Matching output momentum-magnitude upper cut; negative omits it.
  float maxMatchDistance = -1.f;  ///< Pre-fit source-position separation upper cut; negative omits it.
  /// Signed momentum-component upper cut in error units, not an absolute-value cut.
  float maxMatchMomentumSigma = -1.f;
  /// First-candidate topology upper cut at the matched track's PV; negative omits it.
  float maxMatchTopologyChi2PerNdf = -1.f;
  float matchMassWindowSigma = -1.f;  ///< Reference mass-window resolution; negative disables the window.
  /// Maximum absolute mass deviation in reference-resolution units; required for an enabled window.
  float matchMassWindowCut = -1.f;
  /// Exclusive upper limit on nearest-line distance in event-PV selection; negative omits it.
  float maxSelectionVertexDistance = -1.f;
  /// Exclusive lower decay-length-significance cut in event-PV selection; negative omits it.
  float minSelectionDecayLengthOverError = -1.f;
};

/**
 * @brief One scheduled operation with references to input sources and event-local context.
 *
 * Despite their names, candidate-index fields refer either to candidate pools or
 * input tracks, according to the corresponding source kind. The scheduler must
 * provide available predecessor candidates and event-local tracks/PV ranges.
 * This value owns no source data and does not validate generation dependencies.
 */
struct KFParticleGpuGraphOperationTask
{
  unsigned int descriptorIndex = 0u;  ///< Index in the compiled operation-descriptor table, not a stable channel ID.
  unsigned int firstCandidateIndex = 0u;  ///< First input index interpreted using firstSourceKind.
  /// Second input index interpreted using secondSourceKind; ignored for unary tasks.
  unsigned int secondCandidateIndex = 0u;
  /// First source category: candidate generation or raw-track range.
  unsigned int firstSourceKind = KFGpuGraphSourceCandidateGeneration;
  /// Second source category: candidate generation or raw-track range.
  unsigned int secondSourceKind = KFGpuGraphSourceCandidateGeneration;
  unsigned int eventIndex = 0u;  ///< Packed event index used to check candidate event identity.
  unsigned int primaryVertexOffset = 0u;  ///< First packed PV in the task's event-local selection range.
  unsigned int primaryVertexCount = 0u;  ///< Number of event-local PVs available to event-PV selection.
  unsigned int domainIndex = 0u;  ///< Transaction domain index used by scheduling/publication kernels.
  unsigned int workIndex = 0u;  ///< Transaction work-range index used for progress and capacity accounting.
};

/**
 * @brief Per-task outcome and observables, without a copy of the fitted particle.
 *
 * ExecuteTask() resets and fills this record; publication kernels assign the
 * output slot or replace acceptance with a capacity rejection. An assigned slot
 * alone does not imply valid output, since a later daughter reservation may fail.
 */
struct KFParticleGpuGraphOperationResult
{
  unsigned int status = KFGpuGraphTaskPending;  ///< Evaluation/publication outcome from KFParticleGpuGraphTaskStatus.
  unsigned int channelId = 0u;  ///< Stable channel ID copied from the descriptor.
  unsigned int outputClass = KFGpuGraphOutputInvalid;  ///< Output category copied from the descriptor.
  /// Reserved output slot, or 0xffffffffu before publication assigns one.
  unsigned int outputCandidateIndex = 0xffffffffu;
  /// Flattened source-track lineage size on successful evaluation, not direct daughter count.
  unsigned int daughterCount = 0u;
  int primaryVertexIndex = -1;  ///< Resolved or selected packed PV index; -1 if unassigned.
  float mass = 0.f;  ///< Output mass observable in GeV/c^2, including documented constraint fallbacks.
  float massError = 0.f;  ///< Mass uncertainty; filtered missing-mass fallback may use -1 for unavailable uncertainty.
  float topologyChi2PerNdf = 1.e8f;  ///< Evaluated topology chi2/NDF, or the initial 1.e8 sentinel when not calculated.
};

/**
 * @brief Flat non-owning view of reusable graph-operation tables, tasks, results, and counters.
 *
 * The device-storage owner retains every allocation. Publishing this view once
 * per reconstruction transaction keeps stable pointers out of kernel argument
 * lists while preserving explicit capacities and revision identity. Global counters
 * point to scalars; channel counters are indexed by descriptor index, not channel ID.
 * Task and result arrays share task capacity; channel arrays use descriptor capacity.
 *
 * Accessors do not validate pointers or bounds. A const view still exposes writable
 * task/result/counter storage. The owner must outlive all accesses and republish
 * views after reallocation; this class performs no synchronization or transfers.
 */
class KFParticleGpuGraphOperationStorageView
{
 public:
  /**
   * @brief Keeps construction trivial for device ABI use; does not establish a usable workspace.
   * @note Plain default-initialization leaves members indeterminate. Value-initialization
   * with {} zeroes them, but callers must still bind valid storage before dereferencing.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationStorageView() = default;

  /**
   * @brief Binds borrowed storage, counts, capacities, and a caller-supplied plan revision.
   * @param descriptors Read-only descriptor table with at least descriptorCapacity entries.
   * @param descriptorCount Number of active descriptors, no greater than descriptorCapacity.
   * @param descriptorCapacity Allocation bound shared by descriptor and per-channel arrays.
   * @param tasks Writable worklist with taskCapacity entries.
   * @param taskCapacity Allocation bound for both tasks and results.
   * @param results Writable per-task result array.
   * @param visited Global visited-combination counter.
   * @param accepted Global accepted-task counter, including capacity losses.
   * @param stored Global stored-task counter.
   * @param constructed Global constructed-output counter.
   * @param rejected Global rejected-task counter.
   * @param overflow Global task-overflow bit mask.
   * @param channelVisited Per-descriptor visited-combination counters.
   * @param channelAccepted Per-descriptor accepted-task counters.
   * @param channelStored Per-descriptor stored-task counters.
   * @param channelConstructed Per-descriptor constructed-output counters.
   * @param channelRejected Per-descriptor rejected-task counters.
   * @param revision Revision identity supplied by the owning plan/buffer manager.
   * @note Only stores the supplied values; does not allocate, clear, or validate arrays.
   * Null pointers are usable only where the consuming path explicitly allows them.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationStorageView(
    const KFParticleGpuGraphOperationDescriptor* descriptors,
    unsigned int descriptorCount,
    unsigned int descriptorCapacity,
    KFParticleGpuGraphOperationTask* tasks,
    unsigned int taskCapacity,
    KFParticleGpuGraphOperationResult* results,
    unsigned int* visited,
    unsigned int* accepted,
    unsigned int* stored,
    unsigned int* constructed,
    unsigned int* rejected,
    unsigned int* overflow,
    unsigned int* channelVisited,
    unsigned int* channelAccepted,
    unsigned int* channelStored,
    unsigned int* channelConstructed,
    unsigned int* channelRejected,
    unsigned long long revision)
    : fDescriptors(descriptors)
    , fDescriptorCount(descriptorCount)
    , fDescriptorCapacity(descriptorCapacity)
    , fTasks(tasks)
    , fTaskCapacity(taskCapacity)
    , fResults(results)
    , fVisited(visited)
    , fAccepted(accepted)
    , fStored(stored)
    , fConstructed(constructed)
    , fRejected(rejected)
    , fOverflow(overflow)
    , fChannelVisited(channelVisited)
    , fChannelAccepted(channelAccepted)
    , fChannelStored(channelStored)
    , fChannelConstructed(channelConstructed)
    , fChannelRejected(channelRejected)
    , fRevision(revision)
  {
  }

  /// @brief Borrows the read-only descriptor table; does not validate its pointer or extent.
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuGraphOperationDescriptor*
  Descriptors() const
  {
    return fDescriptors;
  }

  /// @brief Returns the active descriptor count, not the allocated capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCount() const
  {
    return fDescriptorCount;
  }

  /// @brief Returns the declared descriptor/per-channel-array capacity.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DescriptorCapacity() const
  {
    return fDescriptorCapacity;
  }

  /// @brief Borrows writable task storage, even through a const view.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationTask* Tasks() const
  {
    return fTasks;
  }

  /// @brief Returns the declared capacity shared by tasks and results.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TaskCapacity() const
  {
    return fTaskCapacity;
  }

  /// @brief Borrows writable per-task results; no entries are initialized here.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuGraphOperationResult* Results() const
  {
    return fResults;
  }

  /// @brief Borrows the global visited-combination counter.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Visited() const { return fVisited; }

  /// @brief Borrows the global accepted-task counter, including capacity losses.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Accepted() const { return fAccepted; }

  /// @brief Borrows the global retained-task counter.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Stored() const { return fStored; }

  /// @brief Borrows the global constructed-output counter.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Constructed() const
  {
    return fConstructed;
  }

  /// @brief Borrows the global rejected-task counter.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Rejected() const { return fRejected; }

  /// @brief Borrows the task-overflow bit mask; does not inspect or reset it.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* Overflow() const { return fOverflow; }

  /// @brief Borrows visited-combination counters indexed by descriptor index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelVisited() const
  {
    return fChannelVisited;
  }

  /// @brief Borrows accepted-task counters indexed by descriptor index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelAccepted() const
  {
    return fChannelAccepted;
  }

  /// @brief Borrows retained-task counters indexed by descriptor index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelStored() const
  {
    return fChannelStored;
  }

  /// @brief Borrows constructed-output counters indexed by descriptor index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelConstructed() const
  {
    return fChannelConstructed;
  }

  /// @brief Borrows rejected-task counters indexed by descriptor index.
  KFPARTICLE_GPU_HOST_DEVICE unsigned int* ChannelRejected() const
  {
    return fChannelRejected;
  }

  /// @brief Returns the supplied revision identity; does not check freshness or compute a hash.
  KFPARTICLE_GPU_HOST_DEVICE unsigned long long Revision() const
  {
    return fRevision;
  }

 private:
  const KFParticleGpuGraphOperationDescriptor* fDescriptors;  ///< Borrowed read-only descriptor table.
  unsigned int fDescriptorCount;  ///< Number of active descriptors.
  unsigned int fDescriptorCapacity;  ///< Descriptor/per-channel-array allocation bound.
  KFParticleGpuGraphOperationTask* fTasks;  ///< Borrowed writable task array.
  unsigned int fTaskCapacity;  ///< Shared task/result allocation bound.
  KFParticleGpuGraphOperationResult* fResults;  ///< Borrowed writable per-task results.
  unsigned int* fVisited;  ///< Borrowed global visited-combination counter.
  unsigned int* fAccepted;  ///< Borrowed global accepted-task counter.
  unsigned int* fStored;  ///< Borrowed global stored-task counter.
  unsigned int* fConstructed;  ///< Borrowed global constructed-output counter.
  unsigned int* fRejected;  ///< Borrowed global rejected-task counter.
  unsigned int* fOverflow;  ///< Borrowed task-overflow flags.
  unsigned int* fChannelVisited;  ///< Borrowed visited counters indexed by descriptor.
  unsigned int* fChannelAccepted;  ///< Borrowed accepted counters indexed by descriptor.
  unsigned int* fChannelStored;  ///< Borrowed stored counters indexed by descriptor.
  unsigned int* fChannelConstructed;  ///< Borrowed constructed counters indexed by descriptor.
  unsigned int* fChannelRejected;  ///< Borrowed rejected counters indexed by descriptor.
  unsigned long long fRevision;  ///< Caller-supplied plan revision identity.
};

/**
 * @brief Allocation-free host/device helpers for loading, evaluating, and storing one graph task.
 * @note Views must reference memory accessible in the calling execution context.
 * Output reservations, task scheduling, and concurrent-access safety are external responsibilities.
 */
namespace KFParticleGpuGraphOperations
{
  static const unsigned int MaximumLineageSize = 16u;  ///< Local flattened source-ID capacity per operation.

  /**
   * @brief Tests the numerical working range -1.e30 < value < 1.e30, rejecting NaN and infinities.
   * @note Stricter than a standard finite-value test: very large finite values also fail.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsFinite(float value)
  {
    return value == value && value > -1.e30f && value < 1.e30f;
  }

  /// @brief Recognizes composite-pair, neutral-daughter, composite-track, and unary graph-operation topologies.
  KFPARTICLE_GPU_HOST_DEVICE inline bool IsSupportedTopology(unsigned int topology)
  {
    return topology == KFGpuGraphTopologyCompositeComposite
           || topology == KFGpuGraphTopologyNeutralDaughter
           || topology == KFGpuGraphTopologyCompositeTrack
           || topology == KFGpuGraphTopologyUnaryComposite;
  }

  /**
   * @brief Checks channel identity, output category, known operation bits, and topology-specific requirements.
   *
   * Composite pairs require Construct; composite-track matching requires both
   * Match and Construct. Neutral-daughter operations require MissingMass and a
   * recognized mode, with nonnegative mass hypotheses for filtered reconstruction.
   * Unary operations require at least one supported final-processing operation.
   * @return True when these structural checks pass, not a guarantee that execution will succeed.
   * @note Does not validate source indices, graph dependencies, unknown flag bits,
   * PV ranges, or all floating-point settings and combinations of final operations.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool ValidateDescriptor(
    const KFParticleGpuGraphOperationDescriptor& descriptor)
  {
    if (descriptor.channelId == 0u || !IsSupportedTopology(descriptor.topology)
        || descriptor.outputClass == KFGpuGraphOutputInvalid
        || descriptor.outputClass >= KFGpuGraphOutputClassCount
        || (descriptor.operationMask & ~KFGpuGraphKnownOperations) != 0u) {
      return false;
    }
    if (descriptor.topology == KFGpuGraphTopologyCompositeComposite) {
      return (descriptor.operationMask & KFGpuGraphConstruct) != 0u;
    }
    if (descriptor.topology == KFGpuGraphTopologyCompositeTrack) {
      return (descriptor.operationMask & KFGpuGraphMatch) != 0u
             && (descriptor.operationMask & KFGpuGraphConstruct) != 0u;
    }
    if (descriptor.topology == KFGpuGraphTopologyNeutralDaughter) {
      return (descriptor.operationMask & KFGpuGraphMissingMass) != 0u
             && (descriptor.missingMassMode == KFGpuGraphMissingMassLegacySubtract
                 || (descriptor.missingMassMode
                       == KFGpuGraphMissingMassFilteredReconstruction
                     && descriptor.firstMass >= 0.f
                     && descriptor.secondMass >= 0.f
                     && descriptor.neutralMass >= 0.f));
    }
    return (descriptor.operationMask
            & (KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex
               | KFGpuGraphMassConstraint | KFGpuGraphMatch | KFGpuGraphSelect))
           != 0u;
  }

  /**
   * @brief Loads and insertion-sorts a candidate's bounded flattened source-track IDs, rejecting duplicates.
   * @param candidates Readable candidate metadata and daughter storage.
   * @param candidateIndex Index below the pool's logical size.
   * @param[out] lineage Caller-owned output array with MaximumLineageSize elements.
   * @param[out] lineageSize Number of sorted IDs on success.
   * @return False for an invalid candidate index, empty/oversized lineage, unavailable daughter range, or duplicate ID.
   * @note Early range failures leave outputs unchanged; duplicate detection may leave partial output.
   * Does not inspect candidate validity flags or establish event/generation membership.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool LoadLineage(
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int candidateIndex,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (candidateIndex >= candidates.Size()) {
      return false;
    }
    const unsigned int count = candidates.Metadata().DaughterCount(candidateIndex);
    const unsigned int offset = candidates.Metadata().DaughterOffset(candidateIndex);
    if (count == 0u || count > MaximumLineageSize
        || !candidates.Daughters().CanStore(offset, count)) {
      return false;
    }
    lineageSize = 0u;
    for (unsigned int daughter = 0u; daughter < count; ++daughter) {
      const int source = candidates.Daughters().SourceId(offset + daughter);
      unsigned int insertion = lineageSize;
      while (insertion > 0u && lineage[insertion - 1u] > source) {
        lineage[insertion] = lineage[insertion - 1u];
        --insertion;
      }
      if ((insertion > 0u && lineage[insertion - 1u] == source)
          || (insertion < lineageSize && lineage[insertion] == source)) {
        return false;
      }
      lineage[insertion] = source;
      ++lineageSize;
    }
    return true;
  }

  /**
   * @brief Loads a candidate fit/lineage or initializes a raw track with a mass hypothesis.
   * @param tracks Readable packed track input; used only for track-range sources.
   * @param candidates Readable candidate input; used only for candidate-generation sources.
   * @param sourceKind Candidate-generation or track-range source category.
   * @param sourceIndex Index in the corresponding input pool.
   * @param pdg Source matching hypothesis; zero disables the PID requirement.
   * @param mass Raw-track initialization mass; an existing candidate keeps its stored fit state.
   * @param eventIndex Required candidate event index; raw-track event membership is not checked here.
   * @param[out] fit Loaded or initialized numerical state.
   * @param[out] lineage Caller-owned array with MaximumLineageSize elements.
   * @param[out] lineageSize Loaded candidate lineage size, or one for a raw track.
   * @return True if source lookup, PID/event checks, and lineage loading succeed.
   * @note Candidate PID matching is signed and exact. Track matching uses absolute
   * PDGs below 1000, input code 2000003112 for hypotheses in [1000, 10000), and
   * nuclear input codes with absolute value at least 1000020030 for larger hypotheses.
   * Track charge is copied separately. The caller must enforce track event membership
   * and any additional charge/generation policy. Outputs can be partially changed on failure.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool LoadSource(
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuConstCandidatePoolView& candidates,
    unsigned int sourceKind,
    unsigned int sourceIndex,
    int pdg,
    float mass,
    unsigned int eventIndex,
    KFParticleGpuFitState& fit,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (sourceKind == KFGpuGraphSourceCandidateGeneration) {
      if (sourceIndex >= candidates.Size()
          || candidates.Metadata().EventIndex(sourceIndex) != eventIndex
          || (pdg != 0 && candidates.Metadata().Pdg(sourceIndex) != pdg)) {
        return false;
      }
      LoadCandidateFit(candidates, sourceIndex, fit);
      return LoadLineage(candidates, sourceIndex, lineage, lineageSize);
    }
    if (sourceKind == KFGpuGraphSourceTrackRange) {
      if (sourceIndex >= tracks.Size()) {
        return false;
      }
      const int sourcePdg = tracks.Pdg(sourceIndex);
      const int absolutePdg = pdg < 0 ? -pdg : pdg;
      const int absoluteSourcePdg = sourcePdg < 0 ? -sourcePdg : sourcePdg;
      const bool matches = pdg == 0
        || (absolutePdg < 1000 && absoluteSourcePdg == absolutePdg)
        || (absolutePdg >= 1000 && absolutePdg < 10000
            && absoluteSourcePdg == 2000003112)
        || (absolutePdg >= 10000 && absoluteSourcePdg >= 1000020030);
      if (!matches) return false;
      KFParticleGpuTrackState track;
      LoadTrackState(tracks, sourceIndex, track);
      fit.Initialize(track, tracks.Charge(sourceIndex), mass);
      lineage[0] = tracks.SourceId(sourceIndex);
      lineageSize = 1u;
      return true;
    }
    return false;
  }

  /**
   * @brief Merges two sorted source-ID lists, rejecting repeated IDs and excessive total length.
   * @param first Sorted first lineage with firstSize readable entries.
   * @param firstSize First lineage length, at most MaximumLineageSize.
   * @param second Sorted second lineage with secondSize readable entries.
   * @param secondSize Second lineage length, at most MaximumLineageSize.
   * @param[out] lineage Separate writable output array with MaximumLineageSize elements.
   * @param[out] lineageSize Merged length on success; may be partial if a duplicate is encountered.
   * @return False if the combined length exceeds local capacity or a duplicate ID is encountered.
   * @note Does not sort the inputs or verify their ordering. A capacity failure leaves outputs unchanged.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool MergeSourceLineage(
    const int first[MaximumLineageSize],
    unsigned int firstSize,
    const int second[MaximumLineageSize],
    unsigned int secondSize,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize)
  {
    if (firstSize + secondSize > MaximumLineageSize) return false;
    unsigned int firstIndex = 0u;
    unsigned int secondIndex = 0u;
    lineageSize = 0u;
    while (firstIndex < firstSize || secondIndex < secondSize) {
      const bool useFirst = secondIndex >= secondSize
        || (firstIndex < firstSize && first[firstIndex] < second[secondIndex]);
      const int source = useFirst ? first[firstIndex++] : second[secondIndex++];
      if (lineageSize > 0u && lineage[lineageSize - 1u] == source) return false;
      lineage[lineageSize++] = source;
    }
    return true;
  }

  /**
   * @brief Applies requested final transport, constraints, matching cuts, and selection to a local fit.
   *
   * Processing order is line extrapolation, production-vertex constraint, optional
   * mass constraint, momentum matching cuts, then selection. Event-PV selection
   * evaluates the supplied event range and defers its mass constraint until after
   * topology and mass-window checks. Trial PV constraints operate on copies; the
   * selected PV index/topology are recorded without replacing the fit by that trial.
   * @param descriptor Effective descriptor with any source-derived PV sentinel already resolved.
   * @param task Event-local primary-vertex range for multi-PV selection.
   * @param primaryVertices Readable packed primary vertices.
   * @param[in,out] fit Numerical state modified by requested transport/constraint operations.
   * @param[in,out] result Receives selected PV/topology and explicit rejection reasons where available.
   * @return True when all requested final operations pass.
   * @note Does not reset result or mark acceptance. Some mathematical/PV lookup failures
   * leave status unchanged; ExecuteTask() maps a still-pending failure to RejectMathematics.
   * No rollback is performed on failure. The caller must provide a valid event-local PV range.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool ApplyFinalOperations(
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    KFParticleGpuFitState& fit,
    KFParticleGpuGraphOperationResult& result)
  {
    const KFParticleGpuRange eventVertices(task.primaryVertexOffset, task.primaryVertexCount);
    if ((descriptor.operationMask & KFGpuGraphExtrapolate) != 0u) {
      if (!KFParticleGpuIsEventPrimaryVertex(
            descriptor.primaryVertexIndex, eventVertices, primaryVertices.Size())) {
        return false;
      }
      KFParticleGpuVertexState vertex;
      LoadVertexState(
        primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
      if (!KFParticleGpuMath::TransportLineToPoint(fit, vertex, fit)) {
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphSetProductionVertex) != 0u) {
      if (!KFParticleGpuIsEventPrimaryVertex(
            descriptor.primaryVertexIndex, eventVertices, primaryVertices.Size())) {
        return false;
      }
      KFParticleGpuVertexState vertex;
      LoadVertexState(
        primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
      if (!KFParticleGpuMath::ConstrainToProductionVertex(fit, vertex)) {
        return false;
      }
    }
    if (((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
         || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u)
        && (descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) == 0u) {
      if (descriptor.massConstraint < 0.f
          || !KFParticleGpuMath::ApplyLinearMassConstraint(
            fit, descriptor.massConstraint, descriptor.massConstraintSigma)) {
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && descriptor.maxMatchMomentum >= 0.f) {
      const float momentum2 = KFParticleGpuMath::Momentum2(fit);
      if (!IsFinite(momentum2)
          || momentum2 > descriptor.maxMatchMomentum * descriptor.maxMatchMomentum) {
        result.status = KFGpuGraphTaskRejectSelection;
        return false;
      }
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && descriptor.maxMatchMomentumSigma >= 0.f) {
      for (int parameter = 3; parameter <= 5; ++parameter) {
        const float variance = fit.Covariance(parameter, parameter);
        const float error = variance > 0.f ? KFParticleGpuSqrt(variance) : -1.f;
        if (!IsFinite(error) || error < 0.f
            || fit.Parameter(parameter)
                 > descriptor.maxMatchMomentumSigma * error) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
      }
    }
    if ((descriptor.operationMask & KFGpuGraphSelect) != 0u) {
      const float geometricChi2 =
        fit.NDF() > 0 ? fit.Chi2() / static_cast<float>(fit.NDF()) : 1.e8f;
      if (fit.NDF() < 0 || fit.Chi2() < 0.f || !IsFinite(geometricChi2)
          || (descriptor.maxGeometricChi2PerNdf >= 0.f
              && geometricChi2 > descriptor.maxGeometricChi2PerNdf)) {
        result.status = KFGpuGraphTaskRejectSelection;
        return false;
      }
      if (descriptor.maxTopologyChi2PerNdf >= 0.f
          && (descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) == 0u) {
        if (!KFParticleGpuIsEventPrimaryVertex(
              descriptor.primaryVertexIndex, eventVertices, primaryVertices.Size())) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        KFParticleGpuVertexState vertex;
        LoadVertexState(
          primaryVertices, static_cast<unsigned int>(descriptor.primaryVertexIndex), vertex);
        KFParticleGpuPrimaryVertexTopologyObservables topology;
        if (!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
              fit, vertex, topology)
            || topology.chi2PerNdf > descriptor.maxTopologyChi2PerNdf) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        result.topologyChi2PerNdf = topology.chi2PerNdf;
      }
      if ((descriptor.flags & KFGpuGraphSelectEventPrimaryVertices) != 0u) {
        const KFParticleGpuRange vertexRange(
          task.primaryVertexOffset, task.primaryVertexCount);
        KFParticleGpuV0LineTopologyResult nearestLine;
        if (!KFParticleGpuSelection::FindNearestV0LineTopology(
              fit, primaryVertices, vertexRange, nearestLine)
            || !KFParticleGpuSelection::PointsFromAnyPrimaryVertex(
                 fit, primaryVertices, vertexRange)
            || (descriptor.maxSelectionVertexDistance >= 0.f
                && nearestLine.lineDistance
                     >= descriptor.maxSelectionVertexDistance)
            || (descriptor.minSelectionDecayLengthOverError >= 0.f
                && nearestLine.decayLdL
                     <= descriptor.minSelectionDecayLengthOverError)) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        int bestVertex = -1;
        float bestChi2PerNdf = 1.e8f;
        for (unsigned int local = 0u; local < vertexRange.size; ++local) {
          const unsigned int vertexIndex = vertexRange.offset + local;
          if (vertexIndex >= primaryVertices.Size()) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
          KFParticleGpuVertexState vertex;
          LoadVertexState(primaryVertices, vertexIndex, vertex);
          KFParticleGpuFitState constrained = fit;
          if (!KFParticleGpuMath::ConstrainToProductionVertex(
                constrained, vertex)
              || constrained.NDF() <= 0 || constrained.Chi2() <= 0.f) {
            continue;
          }
          const float chi2PerNdf =
            constrained.Chi2() / static_cast<float>(constrained.NDF());
          if (IsFinite(chi2PerNdf) && chi2PerNdf < bestChi2PerNdf) {
            bestChi2PerNdf = chi2PerNdf;
            bestVertex = static_cast<int>(vertexIndex);
          }
        }
        if (bestVertex < 0
            || (descriptor.maxTopologyChi2PerNdf >= 0.f
                && bestChi2PerNdf > descriptor.maxTopologyChi2PerNdf)) {
          result.status = KFGpuGraphTaskRejectSelection;
          return false;
        }
        result.primaryVertexIndex = bestVertex;
        result.topologyChi2PerNdf = bestChi2PerNdf;
        if (descriptor.matchMassWindowSigma >= 0.f) {
          float mass = 0.f;
          float massError = 0.f;
          if (descriptor.matchMassWindowCut < 0.f
              || !KFParticleGpuMath::GetMass(fit, mass, massError)
              || !IsFinite(mass)
              || (mass > descriptor.massConstraint
                    ? mass - descriptor.massConstraint
                    : descriptor.massConstraint - mass)
                   > descriptor.matchMassWindowCut
                       * descriptor.matchMassWindowSigma) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
        if ((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
            || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u) {
          if (descriptor.massConstraint < 0.f
              || !KFParticleGpuMath::ApplyLinearMassConstraint(
                   fit, descriptor.massConstraint,
                   descriptor.massConstraintSigma)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * @brief Evaluates one graph task into caller-owned fit, lineage, and result values without publishing them.
   *
   * Loads sources, resolves a source-derived PV, checks/merges lineage, performs
   * unary copying, composite construction, or missing-mass/matching reconstruction,
   * then applies final operations. Filtered missing-mass reconstruction returns
   * the filtered mother, not the temporary neutral state. StoreFirstOnMatch returns
   * the matched first state after testing the residual, retaining the merged lineage.
   * @param tracks Packed track input for track-range sources.
   * @param candidates Readable predecessor candidates with valid metadata and lineage storage.
   * @param primaryVertices Readable packed PV input.
   * @param descriptor Configuration already selected by the caller; task.descriptorIndex is not resolved here.
   * @param task Source references and event/PV context established by the scheduler.
   * @param[out] output Local reconstructed fit state; must not alias input pool storage.
   * @param[out] lineage Writable array with MaximumLineageSize entries for sorted source IDs.
   * @param[out] lineageSize Flattened source-ID count on success.
   * @param[out] result Reset on entry, then filled with acceptance or a rejection reason and observables.
   * @return True on accepted numerical output; no candidate/daughter capacity is reserved or checked here.
   * @note Fit and lineage outputs may be partial on failure. Exact-constraint mass
   * roundoff can use the target mass with zero error; filtered missing-mass fallback
   * can use firstMass with error -1 when a finite fit has no usable mass uncertainty.
   * Generation dependencies, raw-track event membership, and publication remain caller responsibilities.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool ExecuteTask(
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    KFParticleGpuFitState& output,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize,
    KFParticleGpuGraphOperationResult& result)
  {
    result = KFParticleGpuGraphOperationResult();
    result.channelId = descriptor.channelId;
    result.outputClass = descriptor.outputClass;
    if (!ValidateDescriptor(descriptor)) {
      result.status = KFGpuGraphTaskRejectDescriptor;
      return false;
    }
    KFParticleGpuFitState first;
    int firstLineage[MaximumLineageSize];
    unsigned int firstLineageSize = 0u;
    if (!LoadSource(tracks, candidates, task.firstSourceKind,
                    task.firstCandidateIndex, descriptor.firstPdg,
                    descriptor.firstMass, task.eventIndex, first,
                    firstLineage, firstLineageSize)) {
      result.status = KFGpuGraphTaskRejectSource;
      return false;
    }
    KFParticleGpuGraphOperationDescriptor effectiveDescriptor = descriptor;
    if (effectiveDescriptor.primaryVertexIndex
          == KFGpuGraphPrimaryVertexFromCandidate) {
      effectiveDescriptor.primaryVertexIndex =
        task.firstSourceKind == KFGpuGraphSourceCandidateGeneration
          ? candidates.Metadata().PrimaryVertexIndex(task.firstCandidateIndex)
          : tracks.PrimaryVertexIndex(task.firstCandidateIndex);
      if (effectiveDescriptor.primaryVertexIndex < 0) {
        result.status = KFGpuGraphTaskRejectSource;
        return false;
      }
    }
    result.primaryVertexIndex = effectiveDescriptor.primaryVertexIndex;
    if (descriptor.topology == KFGpuGraphTopologyUnaryComposite) {
      for (unsigned int i = 0u; i < firstLineageSize; ++i) lineage[i] = firstLineage[i];
      lineageSize = firstLineageSize;
      output = first;
    }
    else {
      if ((descriptor.flags & KFGpuGraphRequireOrderedCandidatePair) != 0u
          && task.firstSourceKind == task.secondSourceKind
          && task.secondCandidateIndex <= task.firstCandidateIndex) {
        result.status = KFGpuGraphTaskRejectLineage;
        return false;
      }

      KFParticleGpuFitState second;
      int secondLineage[MaximumLineageSize];
      unsigned int secondLineageSize = 0u;
      if (!LoadSource(tracks, candidates, task.secondSourceKind,
                      task.secondCandidateIndex, descriptor.secondPdg,
                      descriptor.secondMass, task.eventIndex, second,
                      secondLineage, secondLineageSize)) {
        result.status = KFGpuGraphTaskRejectSource;
        return false;
      }
      if (!MergeSourceLineage(firstLineage, firstLineageSize,
                              secondLineage, secondLineageSize,
                              lineage, lineageSize)) {
        result.status = KFGpuGraphTaskRejectLineage;
        return false;
      }
      if ((descriptor.operationMask & KFGpuGraphMatch) != 0u) {
        if ((descriptor.flags & KFGpuGraphMatchTrackPrimaryVertex) != 0u) {
          if (task.firstSourceKind != KFGpuGraphSourceCandidateGeneration
              || task.firstCandidateIndex >= candidates.Size()
              || task.secondSourceKind != KFGpuGraphSourceTrackRange
              || task.secondCandidateIndex >= tracks.Size()) {
            result.status = KFGpuGraphTaskRejectSource;
            return false;
          }
          const int primaryVertexIndex =
            tracks.PrimaryVertexIndex(task.secondCandidateIndex);
          if (!KFParticleGpuIsEventPrimaryVertex(
                primaryVertexIndex,
                KFParticleGpuRange(task.primaryVertexOffset, task.primaryVertexCount),
                primaryVertices.Size())) {
            result.status = KFGpuGraphTaskRejectSource;
            return false;
          }
          if (descriptor.maxMatchTopologyChi2PerNdf >= 0.f) {
            KFParticleGpuVertexState vertex;
            LoadVertexState(primaryVertices,
                            static_cast<unsigned int>(primaryVertexIndex),
                            vertex);
            KFParticleGpuPrimaryVertexTopologyObservables topology;
            if (!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
                  first, vertex, topology)
                || topology.chi2PerNdf
                     > descriptor.maxMatchTopologyChi2PerNdf) {
              result.status = KFGpuGraphTaskRejectSelection;
              return false;
            }
            result.topologyChi2PerNdf = topology.chi2PerNdf;
          }
          result.primaryVertexIndex = primaryVertexIndex;
        }
        if (descriptor.matchMassWindowSigma >= 0.f) {
          float mass = 0.f;
          float massError = 0.f;
          if (descriptor.matchMassWindowCut < 0.f
              || !KFParticleGpuMath::GetMass(first, mass, massError)
              || !IsFinite(mass)
              || (mass > descriptor.massConstraint
                    ? mass - descriptor.massConstraint
                    : descriptor.massConstraint - mass)
                   > descriptor.matchMassWindowCut
                       * descriptor.matchMassWindowSigma) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
        if ((effectiveDescriptor.operationMask & KFGpuGraphMassConstraint) != 0u
            || (effectiveDescriptor.flags & KFGpuGraphApplyMassConstraint) != 0u) {
          if (effectiveDescriptor.massConstraint < 0.f
              || !KFParticleGpuMath::ApplyLinearMassConstraint(
                   first, effectiveDescriptor.massConstraint,
                   effectiveDescriptor.massConstraintSigma)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
          effectiveDescriptor.operationMask &= ~KFGpuGraphMassConstraint;
          effectiveDescriptor.flags &= ~KFGpuGraphApplyMassConstraint;
        }
        if (descriptor.maxMatchDistance >= 0.f) {
          const float dx = first.X() - second.X();
          const float dy = first.Y() - second.Y();
          const float dz = first.Z() - second.Z();
          const float distance2 = dx * dx + dy * dy + dz * dz;
          if (!IsFinite(distance2)
              || distance2 > descriptor.maxMatchDistance
                               * descriptor.maxMatchDistance) {
            result.status = KFGpuGraphTaskRejectSelection;
            return false;
          }
        }
      }
      if (descriptor.topology == KFGpuGraphTopologyCompositeComposite) {
        KFParticleGpuMeasurement measurement;
        if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
              first, second, output, measurement)
            || !KFParticleGpuMath::AddDaughterWithEnergyFit(
              output, measurement, second.Q())) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
      }
      else {
        if ((descriptor.flags & KFGpuGraphRequirePositiveMissingEnergy) != 0u
            && first.E() <= second.E()) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
        KFParticleGpuMeasurement measurement;
        KFParticleGpuFitState firstAtDca;
        if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
              first, second, firstAtDca, measurement)) {
          result.status = KFGpuGraphTaskRejectMathematics;
          return false;
        }
        if (descriptor.missingMassMode
              == KFGpuGraphMissingMassFilteredReconstruction) {
          KFParticleGpuFitState neutral;
          KFParticleGpuFitState daughterFiltered;
          if (!KFParticleGpuMath::ReconstructMissingMassFiltered(
                firstAtDca, measurement, second.Q(), descriptor.firstMass,
                descriptor.secondMass, descriptor.neutralMass, neutral,
                output, daughterFiltered)) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
        else {
          output = firstAtDca;
          output.NDF() = -1;
          output.Chi2() = 0.f;
          if (!KFParticleGpuMath::SubtractDaughterWithEnergyFit(
                output, measurement, second.Q())) {
            result.status = KFGpuGraphTaskRejectMathematics;
            return false;
          }
        }
      }
    }

    if (!ApplyFinalOperations(
          effectiveDescriptor, task, primaryVertices, output, result)) {
      if (result.status == KFGpuGraphTaskPending) {
        result.status = KFGpuGraphTaskRejectMathematics;
      }
      return false;
    }
    if ((descriptor.operationMask & KFGpuGraphMatch) != 0u
        && (descriptor.flags & KFGpuGraphStoreFirstOnMatch) != 0u) {
      output = first;
    }
    if (!KFParticleGpuMath::GetMass(output, result.mass, result.massError)
        || !IsFinite(result.mass)) {
      if (((descriptor.operationMask & KFGpuGraphMassConstraint) != 0u
           || (descriptor.flags & KFGpuGraphApplyMassConstraint) != 0u)
          && descriptor.massConstraint >= 0.f
          && KFParticleGpuMath::IsFiniteState(output)) {
        // Exact constraints can leave a tiny negative projected mass variance
        // after float roundoff. The constrained mass remains valid and exact.
        result.mass = descriptor.massConstraint;
        result.massError = 0.f;
      }
      else if (descriptor.topology == KFGpuGraphTopologyNeutralDaughter
          && descriptor.missingMassMode
               == KFGpuGraphMissingMassFilteredReconstruction
          && descriptor.firstMass >= 0.f
          && KFParticleGpuMath::IsFiniteState(output)) {
        result.mass = descriptor.firstMass;
        result.massError = -1.f;
      }
      else {
        result.status = KFGpuGraphTaskRejectMathematics;
        return false;
      }
    }
    result.status = KFGpuGraphTaskAccepted;
    result.daughterCount = lineageSize;
    return true;
  }

  /**
   * @brief Evaluates a candidate-only task by forwarding with an empty raw-track input view.
   * @note Uses the same output/failure contract as the track-aware overload.
   * A task that requires a raw track fails source lookup; no implicit track data is supplied.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool ExecuteTask(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    KFParticleGpuFitState& output,
    int lineage[MaximumLineageSize],
    unsigned int& lineageSize,
    KFParticleGpuGraphOperationResult& result)
  {
    return ExecuteTask(KFParticleGpuConstInputTrackSoAView(), candidates,
                       primaryVertices, descriptor, task, output, lineage,
                       lineageSize, result);
  }

  /**
   * @brief Writes an evaluated fit, metadata, direct source references, and flattened lineage to reserved storage.
   * @param candidates Writable output pool with valid component views.
   * @param descriptor Supplies output PDG, channel, topology, and category.
   * @param task Supplies direct daughter kinds/indices; unary operations store only the first reference.
   * @param eventIndex Packed event index to publish, supplied explicitly by the caller.
   * @param primaryVertexIndex Resolved/selected PV index to publish, rather than an unresolved descriptor sentinel.
   * @param candidateIndex Previously reserved output candidate slot.
   * @param daughterOffset Start of a previously reserved flattened daughter-ID range.
   * @param fit Accepted local numerical state to copy.
   * @param lineage Readable flattened source IDs, separate from direct daughter references.
   * @param lineageSize Number of IDs to copy, at most MaximumLineageSize by the caller's contract.
   * @return False without writing if the candidate slot or daughter range does not fit; true after storing output.
   * @note Does not reserve storage, advance pool sizes, update task results/counters,
   * set overflow flags, or revalidate the descriptor, fit, lineage uniqueness, or event identity.
   * Concurrent writers require disjoint reservations; consumers must wait for writes to complete.
   */
  KFPARTICLE_GPU_HOST_DEVICE inline bool StoreOutput(
    const KFParticleGpuCandidatePoolView& candidates,
    const KFParticleGpuGraphOperationDescriptor& descriptor,
    const KFParticleGpuGraphOperationTask& task,
    unsigned int eventIndex,
    int primaryVertexIndex,
    unsigned int candidateIndex,
    unsigned int daughterOffset,
    const KFParticleGpuFitState& fit,
    const int lineage[MaximumLineageSize],
    unsigned int lineageSize)
  {
    if (!candidates.CanStoreCandidate(candidateIndex)
        || !candidates.Daughters().CanStore(daughterOffset, lineageSize)) {
      return false;
    }
    StoreCandidateFit(fit, candidates, candidateIndex);
    candidates.Metadata().Pdg(candidateIndex) = descriptor.motherPdg;
    candidates.Metadata().PrimaryVertexIndex(candidateIndex) =
      primaryVertexIndex;
    candidates.Metadata().EventIndex(candidateIndex) = eventIndex;
    candidates.Metadata().DaughterOffset(candidateIndex) = daughterOffset;
    candidates.Metadata().DaughterCount(candidateIndex) = lineageSize;
    candidates.Metadata().Flags(candidateIndex) = KFGpuCandidateValid;
    candidates.Metadata().ChannelId(candidateIndex) = descriptor.channelId;
    candidates.Metadata().Topology(candidateIndex) = descriptor.topology;
    candidates.Metadata().OutputClass(candidateIndex) = descriptor.outputClass;
    candidates.Metadata().OperationStatus(candidateIndex) = KFGpuCandidateOperationAccepted;
    const bool unary = descriptor.topology == KFGpuGraphTopologyUnaryComposite;
    candidates.Metadata().DirectDaughterCount(candidateIndex) = unary ? 1u : 2u;
    SetCandidateDirectDaughter(
      candidates.Metadata(), candidateIndex, 0u,
      task.firstSourceKind == KFGpuGraphSourceTrackRange
        ? KFGpuDirectDaughterInputTrack : KFGpuDirectDaughterCandidate,
      task.firstCandidateIndex);
    if (unary) {
      candidates.Metadata().DirectSecondKind(candidateIndex) = KFGpuDirectDaughterNone;
      candidates.Metadata().DirectSecondIndex(candidateIndex) = 0u;
    }
    else {
      SetCandidateDirectDaughter(
        candidates.Metadata(), candidateIndex, 1u,
        task.secondSourceKind == KFGpuGraphSourceTrackRange
          ? KFGpuDirectDaughterInputTrack : KFGpuDirectDaughterCandidate,
        task.secondCandidateIndex);
    }
    for (unsigned int daughter = 0u; daughter < lineageSize; ++daughter) {
      candidates.Daughters().SourceId(daughterOffset + daughter) = lineage[daughter];
    }
    return true;
  }
}

static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationDescriptor>::value,
              "KFParticle GPU graph operation descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationTask>::value,
              "KFParticle GPU graph operation tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationResult>::value,
              "KFParticle GPU graph operation results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationStorageView>::value,
              "KFParticle GPU graph operation storage must remain a flat device ABI");
static_assert(
  std::is_trivially_default_constructible<KFParticleGpuGraphOperationStorageView>::value,
  "KFParticle GPU graph operation storage must not require dynamic initialization");

#endif
