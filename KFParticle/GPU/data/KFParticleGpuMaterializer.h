/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 */

/**
 * @file
 * @brief Host-side validation and conversion of GPU candidate graphs into scalar particles.
 * Input views must reference host-readable snapshots whose device producers have completed.
 * This header performs no GPU launches, device readback, or ROOT I/O.
 */
#ifndef KFPARTICLEGPUMATERIALIZER_H
#define KFPARTICLEGPUMATERIALIZER_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuDecayPlan.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Result of the materializer's validation and conversion checks.
 * These values report detected data errors, not allocation failures or C++ exceptions.
 */
enum KFParticleGpuMaterializationStatus
{
  /// Materialization completed successfully.
  KFGpuMaterializationSucceeded = 0u,
  /// Candidate pool or requested range reports truncated output.
  KFGpuMaterializationOverflow,
  /// Missing range/event data, out-of-bounds range, or unrepresentable output size.
  KFGpuMaterializationInvalidRange,
  /// Candidate identity, acceptance, metadata, or publication order is invalid.
  KFGpuMaterializationInvalidCandidate,
  /// A direct daughter is absent, unsupported, or cannot be mapped to output.
  KFGpuMaterializationInvalidDirectDaughter,
  /// An ancestry cycle or forbidden forward dependency was encountered.
  KFGpuMaterializationCyclicAncestry,
  /// Stored physical-track ancestry is invalid or differs from direct daughters.
  KFGpuMaterializationLineageMismatch,
  /// A fit has invalid numeric values/charge, or an input source ID is negative.
  KFGpuMaterializationInvalidNumericalState,
  /// Track pairing, source uniqueness, or trusted-input prerequisites are invalid.
  KFGpuMaterializationInvalidTrackLayout
};

/**
 * @brief Reusable host scratch storage owned by one materialization worker.
 *
 * Retains vector capacity across events and resets only previously touched map slots.
 * Map indices are absolute input indices in Build(), but may be rebased to event-local
 * intervals in the direct publication path. Candidate lists and traversal frames retain
 * absolute candidate indices. No GPU memory or output particles are owned here.
 * Do not share a workspace between concurrent materialization calls.
 */
struct KFParticleGpuMaterializationWorkspace
{
  /**
   * @brief Explicit depth-first traversal frame, avoiding recursive ancestry walks.
   */
  struct CandidateFrame
  {
    /**
     * @brief Resume traversal of candidateIndex at its daughterIndex-th direct daughter.
     */
    CandidateFrame(unsigned int candidateIndex = 0u,
                   unsigned int daughterIndex = 0u)
      : candidate(candidateIndex), nextDaughter(daughterIndex)
    {
    }

    unsigned int candidate; ///< Absolute index in the input candidate pool.
    unsigned int nextDaughter; ///< Next direct-daughter ordinal to visit.
  };

  /**
   * @brief Candidate traversal states (0=unseen, 1=active, 2=complete) and track-visit flags.
   */
  std::vector<unsigned char> visit, trackSeen;
  /**
   * @brief Maps from input slots to event-output particle indices; -1 means unmapped.
   */
  std::vector<int> trackParticleIds, candidateParticleIds;
  /**
   * @brief Maps to absolute input-track indices supplying inherited field coefficients.
   * An unsigned maximum value marks a field source not yet assigned.
   */
  std::vector<unsigned int> trackFieldSourceIndices, candidateFieldSourceIndices;
  /**
   * @brief Map slots requiring reset before reuse, including visits from an aborted traversal.
   */
  std::vector<unsigned int> touchedCandidates, touchedTracks;
  /**
   * @brief Requested candidate indices and dependency-first order produced by graph traversal.
   */
  std::vector<unsigned int> requested, orderedCandidates;
  /**
   * @brief Referenced absolute track indices and their deduplicated input-slot list.
   * Distinct slots with the same physical source ID are merged later when leaves are built.
   */
  std::vector<unsigned int> referencedTracks, uniqueTracks;
  /**
   * @brief Active candidate stack for the general depth-first graph walk.
   */
  std::vector<CandidateFrame> traversal;
  /**
   * @brief Scratch lists for stored candidate ancestry and physical-source uniqueness checks.
   */
  std::vector<int> storedLineage, physicalSourceIds;
  /**
   * @brief Cumulative newly initialized and touched-reset map-slot counts across Begin() calls.
   * These count logical candidate/track slots, not bytes, allocations, or individual map arrays.
   */
  std::size_t initializedEntries = 0u, resetEntries = 0u;

  /**
   * @brief Reset the previous event's touched slots and prepare maps for the next event.
   * Clears transient lists without discarding capacity. Reset precedes resizing so an earlier
   * failed traversal can be cleaned safely even when the next event needs smaller maps.
   * @param candidates Number of candidate map slots required by the selected materialization path.
   * @param tracks Number of track map slots required by that path.
   * @pre Workspace vectors and touched-slot lists are internally consistent.
   */
  void Begin(unsigned int candidates, unsigned int tracks)
  {
    // Clear touched slots before resize, including a DFS aborted by an error.
    for (unsigned int index : touchedCandidates) {
      visit[index] = 0u;
      candidateParticleIds[index] = -1;
      candidateFieldSourceIndices[index] = std::numeric_limits<unsigned int>::max();
    }
    for (unsigned int index : touchedTracks) {
      trackSeen[index] = 0u;
      trackParticleIds[index] = -1;
      trackFieldSourceIndices[index] = std::numeric_limits<unsigned int>::max();
    }
    resetEntries += touchedCandidates.size() + touchedTracks.size();
    touchedCandidates.clear();
    touchedTracks.clear();
    requested.clear();
    orderedCandidates.clear();
    referencedTracks.clear();
    uniqueTracks.clear();
    traversal.clear();
    storedLineage.clear();
    physicalSourceIds.clear();
    initializedEntries += candidates > visit.size() ? candidates - visit.size() : 0u;
    initializedEntries += tracks > trackSeen.size() ? tracks - trackSeen.size() : 0u;
    visit.resize(candidates, 0u);
    candidateParticleIds.resize(candidates, -1);
    candidateFieldSourceIndices.resize(candidates,
                                       std::numeric_limits<unsigned int>::max());
    trackSeen.resize(tracks, 0u);
    trackParticleIds.resize(tracks, -1);
    trackFieldSourceIndices.resize(tracks,
                                   std::numeric_limits<unsigned int>::max());
  }

  /**
   * @brief Sum reserved vector-element storage in bytes, excluding allocator/container overhead.
   */
  std::size_t CapacityBytes() const
  {
    return visit.capacity() * sizeof(visit[0])
           + trackSeen.capacity() * sizeof(trackSeen[0])
           + trackParticleIds.capacity() * sizeof(trackParticleIds[0])
           + candidateParticleIds.capacity() * sizeof(candidateParticleIds[0])
           + trackFieldSourceIndices.capacity() * sizeof(trackFieldSourceIndices[0])
           + candidateFieldSourceIndices.capacity() * sizeof(candidateFieldSourceIndices[0])
           + touchedCandidates.capacity() * sizeof(touchedCandidates[0])
           + touchedTracks.capacity() * sizeof(touchedTracks[0])
           + requested.capacity() * sizeof(requested[0])
           + orderedCandidates.capacity() * sizeof(orderedCandidates[0])
           + referencedTracks.capacity() * sizeof(referencedTracks[0])
           + uniqueTracks.capacity() * sizeof(uniqueTracks[0])
           + traversal.capacity() * sizeof(traversal[0])
           + storedLineage.capacity() * sizeof(storedLineage[0])
           + physicalSourceIds.capacity() * sizeof(physicalSourceIds[0]);
  }

  /**
   * @brief Largest reserved entry count among the candidate visit, particle-ID, and field-source maps.
   */
  std::size_t CandidateMapCapacityEntries() const
  {
    return std::max(visit.capacity(),
                    std::max(candidateParticleIds.capacity(),
                             candidateFieldSourceIndices.capacity()));
  }

  /**
   * @brief Largest reserved entry count among the track visit, particle-ID, and field-source maps.
   */
  std::size_t TrackMapCapacityEntries() const
  {
    return std::max(trackSeen.capacity(),
                    std::max(trackParticleIds.capacity(),
                             trackFieldSourceIndices.capacity()));
  }
};

/**
 * @brief Accumulated host wall times for materialization phases, in milliseconds.
 * Calls add to existing values, including time spent before a validation failure.
 * Phase boundaries differ between the intermediate and direct paths: conversion is measured
 * in commit for the former, but in leaves/candidates for the latter. This is not GPU timing.
 * Use a separate instance per concurrent worker or call.
 */
struct KFParticleGpuMaterializationTiming
{
  /// @brief Graph/range validation time and input-leaf preparation time.
  double graphMilliseconds = 0., leavesMilliseconds = 0.;
  /// @brief Composite construction time and final conversion/publication time, respectively.
  double candidatesMilliseconds = 0., commitMilliseconds = 0.;
};

/**
 * @brief Borrowed host input and optional scratch/measurement objects for one event.
 * Views and descriptors must remain valid and unchanged during a call; sizes must fit
 * their backing storage. The caller completes readback and establishes these storage contracts.
 * Fast-path flags express caller-proven contracts, not requests to repair or validate arbitrary memory.
 */
struct KFParticleGpuMaterializationRequest
{
  /**
   * @brief Host-readable track states, source IDs, PID, PV associations, and optional field data.
   */
  KFParticleGpuConstInputTrackSoAView tracks;
  /**
   * @brief Host-readable candidate fits, metadata, physical lineage, sizes, and overflow status.
   */
  KFParticleGpuConstCandidatePoolView candidates;
  /**
   * @brief Candidate/output-lineage ranges to materialize; required when numberOfRanges is nonzero.
   */
  const KFParticleGpuCandidateRange* ranges = nullptr;
  /**
   * @brief Number of entries addressable through ranges.
   */
  unsigned int numberOfRanges = 0u;
  /**
   * @brief Expected event label in candidate metadata, not an output particle ID.
   */
  unsigned int eventIndex = 0u;
  /**
   * @brief Matching event descriptor; required when includeAllInputTracks is enabled.
   */
  const KFParticleGpuEventDesc* event = nullptr;
  /**
   * @brief Include all physical input tracks of the event, not only tracks referenced by candidates.
   */
  bool includeAllInputTracks = false;
  /**
   * @brief Declare a complete event publication plan with increasing candidate indices.
   * Ranges collectively enumerate all produced candidates. Rejected slots are filtered;
   * each accepted candidate dependency must occur earlier in the plan and at a lower index.
   * With a reusable workspace, this avoids the general ancestry traversal.
   */
  bool completePublicationPlan = false;
  /**
   * @brief Declare track-layout invariants already proved by a sealed host packing plan.
   * Requires completePublicationPlan. Only the direct commit path skips repeated first/last
   * pairing and source-ID uniqueness checks; Build() retains its independent validation.
   * Set only after proving event-owned ranges, paired first/last IDs, and unique physical sources.
   */
  bool validatedPublicationInput = false;
  /**
   * @brief Optional exclusive scratch workspace; null uses temporary scratch and the general commit path.
   */
  KFParticleGpuMaterializationWorkspace* workspace = nullptr;
  /**
   * @brief Optional exclusive timing accumulator; null disables phase measurement.
   */
  KFParticleGpuMaterializationTiming* timing = nullptr;
};

/**
 * @brief Owned intermediate particle record for diagnostics and general scalar conversion.
 * Input leaves and composites share this record, but use different identity fields.
 * Direct daughter IDs refer to positions in the materialized event, whereas leafSourceIds
 * identify physical input tracks. Composite field coefficients follow the first direct daughter.
 */
struct KFParticleGpuMaterializedParticle
{
  /**
   * @brief Particle state, covariance, fit quality, and construction attributes.
   */
  KFParticleGpuFitState fit;
  /**
   * @brief Immediate daughter positions in the event output; empty for input-track leaves.
   */
  std::vector<int> daughterParticleIds;
  /**
   * @brief Physical-track ancestry; composite entries are sorted and unique.
   */
  std::vector<int> leafSourceIds;
  /**
   * @brief Copied/inherited field coefficients; zero when no input field data are supplied.
   */
  float fieldCoefficients[10] = {};
  /**
   * @brief Raw GPU candidate index; unsigned maximum for an input leaf.
   */
  unsigned int gpuCandidateIndex = std::numeric_limits<unsigned int>::max();
  /**
   * @brief Reconstruction channel ID for a composite; zero for an input leaf.
   */
  unsigned int channelId = 0u;
  /**
   * @brief Graph topology copied from candidate metadata.
   */
  unsigned int topology = KFGpuGraphTopologyInvalid;
  /**
   * @brief Candidate output class; invalid for an input leaf.
   */
  unsigned int outputClass = KFGpuGraphOutputInvalid;
  /**
   * @brief Candidate operation status; retains its default for an input leaf.
   */
  unsigned int operationStatus = KFGpuCandidateOperationPending;
  /**
   * @brief Particle hypothesis PDG code.
   */
  int pdg = 0;
  /**
   * @brief Input/candidate PV association, preserved as metadata.
   */
  int primaryVertexIndex = -1;
  /**
   * @brief Physical source ID for an input leaf; unused for a composite.
   */
  int sourceId = -1;
  /**
   * @brief Whether this record represents a physical input track rather than a composite.
   */
  bool inputTrack = false;
};

/**
 * @brief Owned intermediate event graph and its validation outcome.
 * Only a successful result may be published; failed results need not contain an empty particle vector.
 */
struct KFParticleGpuMaterializedEvent
{
  /**
   * @brief Input leaves followed by composites in dependency-before-parent order on success.
   */
  std::vector<KFParticleGpuMaterializedParticle> particles;
  /**
   * @brief Detected validation/conversion outcome.
   */
  KFParticleGpuMaterializationStatus status = KFGpuMaterializationSucceeded;

  /**
   * @brief Whether the event passed materialization validation.
   */
  bool Succeeded() const { return status == KFGpuMaterializationSucceeded; }
};

/**
 * @brief Validate a host-readable GPU candidate graph and create scalar event particles.
 *
 * Build() creates a ROOT-independent intermediate graph for diagnostics and standalone tests.
 * CommitToParticles() converts that graph, or uses a direct path when reusable scratch and
 * complete event-track input are available and intermediate metadata are not requested.
 * Both preserve direct decay links while checking physical-track ancestry and numerical states.
 *
 * Publication replaces the destination only after successful validation/construction.
 * This all-or-nothing replacement is not a synchronization primitive: concurrent calls need
 * separate output vectors, workspaces, timing objects, and status/metadata destinations.
 * The class is stateless and performs no device transfers or GPU scheduling.
 */
class KFParticleGpuMaterializer
{
 public:
  /**
   * @brief Construct a validated intermediate event graph from the requested candidate ranges.
   * General input uses an iterative ancestry walk. A complete publication plan with a reusable
   * workspace uses its supplied order instead. Physical source IDs unify track representations;
   * input leaves precede composites, whose field data follow their first direct daughter.
   * @param request Borrowed host data, publication contracts, and optional exclusive scratch/timing.
   * @return Owned records and status; check Succeeded() before consuming any records.
   * @note Allocation exceptions are not converted into materialization status codes.
   */
  static KFParticleGpuMaterializedEvent Build(const KFParticleGpuMaterializationRequest& request)
  {
    PhaseTimer timer(request.timing ? &request.timing->graphMilliseconds : nullptr);
    KFParticleGpuMaterializedEvent result;
    if (request.candidates.OverflowFlags() != 0u) {
      result.status = KFGpuMaterializationOverflow;
      return result;
    }

    const unsigned int candidateSize = request.candidates.Size();
    if (request.numberOfRanges != 0u && request.ranges == nullptr) {
      result.status = KFGpuMaterializationInvalidRange;
      return result;
    }
    if (request.validatedPublicationInput && !request.completePublicationPlan) {
      result.status = KFGpuMaterializationInvalidTrackLayout;
      return result;
    }
    KFParticleGpuMaterializationWorkspace localWorkspace;
    auto& workspace = request.workspace ? *request.workspace : localWorkspace;
    workspace.Begin(candidateSize, request.tracks.Size());
    auto& requested = workspace.requested;
    for (unsigned int rangeIndex = 0u; rangeIndex < request.numberOfRanges; ++rangeIndex) {
      const KFParticleGpuCandidateRange& range = request.ranges[rangeIndex];
      if (range.overflowFlags != 0u) {
        result.status = KFGpuMaterializationOverflow;
        return result;
      }
      if (range.offset > candidateSize || range.size > candidateSize - range.offset) {
        result.status = KFGpuMaterializationInvalidRange;
        return result;
      }
      const unsigned int daughterSize = request.candidates.Daughters().Size();
      if (range.daughterOffset > daughterSize
          || range.daughterSize > daughterSize - range.daughterOffset) {
        result.status = KFGpuMaterializationInvalidRange;
        return result;
      }
      for (unsigned int candidate = range.offset; candidate < range.offset + range.size; ++candidate) {
        if (request.completePublicationPlan) {
          const auto& metadata = request.candidates.Metadata();
          if (metadata.EventIndex(candidate) != request.eventIndex) {
            result.status = KFGpuMaterializationInvalidCandidate;
            return result;
          }
          if (metadata.OperationStatus(candidate) != KFGpuCandidateOperationAccepted
              || (metadata.Flags(candidate) & KFGpuCandidateValid) == 0u) {
            continue;
          }
        }
        requested.push_back(candidate);
      }
    }
    auto& visit = workspace.visit;
    auto& orderedCandidates = workspace.orderedCandidates;
    auto& referencedTracks = workspace.referencedTracks;
    const bool directPublicationOrder =
      request.workspace != nullptr && request.completePublicationPlan;
    const bool directTrackMapping = request.workspace != nullptr
                                    && request.includeAllInputTracks;
    if (request.includeAllInputTracks) {
      if (request.event == nullptr) {
        result.status = KFGpuMaterializationInvalidRange;
        return result;
      }
      if (!directTrackMapping) {
        for (unsigned int set = 0u; set < 4u; ++set) {
          const KFParticleGpuRange& range = request.event->trackSets[set].tracks;
          if (range.offset > request.tracks.Size()
              || range.size > request.tracks.Size() - range.offset) {
            result.status = KFGpuMaterializationInvalidRange;
            return result;
          }
          for (unsigned int track = range.offset; track < range.End(); ++track) {
            referencedTracks.push_back(track);
          }
        }
      }
    }
    const auto candidateIsValid = [&](unsigned int candidate) {
      if (candidate >= candidateSize) { return false; }
      const auto& metadata = request.candidates.Metadata();
      return metadata.EventIndex(candidate) == request.eventIndex
             && metadata.OperationStatus(candidate) == KFGpuCandidateOperationAccepted
             && (metadata.Flags(candidate) & KFGpuCandidateValid) != 0u
             && metadata.OutputClass(candidate) != KFGpuGraphOutputInvalid
             && metadata.OutputClass(candidate) < KFGpuGraphOutputClassCount
             && metadata.DirectDaughterCount(candidate) != 0u
             && metadata.DirectDaughterCount(candidate) <= 2u;
    };

    if (directPublicationOrder) {
      unsigned int previous = 0u;
      bool havePrevious = false;
      for (unsigned int candidate : requested) {
        if ((havePrevious && candidate <= previous) || !candidateIsValid(candidate)) {
          result.status = KFGpuMaterializationInvalidCandidate;
          return result;
        }
        const auto& metadata = request.candidates.Metadata();
        for (unsigned int daughter = 0u;
             daughter < metadata.DirectDaughterCount(candidate); ++daughter) {
          const unsigned int kind = daughter == 0u
                                      ? metadata.DirectFirstKind(candidate)
                                      : metadata.DirectSecondKind(candidate);
          const unsigned int index = daughter == 0u
                                       ? metadata.DirectFirstIndex(candidate)
                                       : metadata.DirectSecondIndex(candidate);
          if (kind == KFGpuDirectDaughterInputTrack) {
            if (index >= request.tracks.Size()) {
              result.status = KFGpuMaterializationInvalidDirectDaughter;
              return result;
            }
            if (!directTrackMapping) { referencedTracks.push_back(index); }
          }
          else if (kind == KFGpuDirectDaughterCandidate) {
            if (index >= candidate || visit[index] != 2u) {
              result.status = index >= candidate
                                ? KFGpuMaterializationCyclicAncestry
                                : KFGpuMaterializationInvalidDirectDaughter;
              return result;
            }
          }
          else {
            result.status = KFGpuMaterializationInvalidDirectDaughter;
            return result;
          }
        }
        workspace.touchedCandidates.push_back(candidate);
        visit[candidate] = 2u;
        previous = candidate;
        havePrevious = true;
      }
    }
    else {
      auto& traversal = workspace.traversal;
      traversal.reserve(requested.size());
      for (unsigned int root : requested) {
        if (root >= candidateSize) {
          result.status = KFGpuMaterializationInvalidCandidate;
          return result;
        }
        if (visit[root] == 2u) { continue; }
        if (visit[root] == 1u) {
          result.status = KFGpuMaterializationCyclicAncestry;
          return result;
        }
        traversal.push_back({root, 0u});
        while (!traversal.empty()) {
          KFParticleGpuMaterializationWorkspace::CandidateFrame& frame =
            traversal.back();
          const unsigned int candidate = frame.candidate;
          if (visit[candidate] == 0u) {
            if (!candidateIsValid(candidate)) {
              result.status = KFGpuMaterializationInvalidCandidate;
              return result;
            }
            workspace.touchedCandidates.push_back(candidate);
            visit[candidate] = 1u;
          }

          const auto& metadata = request.candidates.Metadata();
          if (frame.nextDaughter < metadata.DirectDaughterCount(candidate)) {
            const unsigned int daughter = frame.nextDaughter++;
            const unsigned int kind = daughter == 0u
                                        ? metadata.DirectFirstKind(candidate)
                                        : metadata.DirectSecondKind(candidate);
            const unsigned int index = daughter == 0u
                                         ? metadata.DirectFirstIndex(candidate)
                                         : metadata.DirectSecondIndex(candidate);
            if (kind == KFGpuDirectDaughterInputTrack) {
              if (index >= request.tracks.Size()) {
                result.status = KFGpuMaterializationInvalidDirectDaughter;
                return result;
              }
              if (!directTrackMapping) { referencedTracks.push_back(index); }
            }
            else if (kind == KFGpuDirectDaughterCandidate) {
              if (index >= candidateSize) {
                result.status = KFGpuMaterializationInvalidCandidate;
                return result;
              }
              if (index == candidate || visit[index] == 1u) {
                result.status = KFGpuMaterializationCyclicAncestry;
                return result;
              }
              if (visit[index] == 0u) { traversal.push_back({index, 0u}); }
            }
            else {
              result.status = KFGpuMaterializationInvalidDirectDaughter;
              return result;
            }
          }
          else {
            visit[candidate] = 2u;
            orderedCandidates.push_back(candidate);
            traversal.pop_back();
          }
        }
      }
    }
    const auto& candidateOrder = directPublicationOrder
                                   ? requested
                                   : orderedCandidates;

    timer.Next(request.timing ? &request.timing->leavesMilliseconds : nullptr);
    auto& uniqueTracks = workspace.uniqueTracks;
    auto& trackSeen = workspace.trackSeen;
    auto& trackParticleIds = workspace.trackParticleIds;
    auto& candidateParticleIds = workspace.candidateParticleIds;
    if (!directTrackMapping) {
      uniqueTracks.reserve(referencedTracks.size());
      for (unsigned int trackIndex : referencedTracks) {
        if (trackIndex >= request.tracks.Size() || trackSeen[trackIndex] != 0u) { continue; }
        workspace.touchedTracks.push_back(trackIndex);
        trackSeen[trackIndex] = 1u;
        uniqueTracks.push_back(trackIndex);
      }
    }

    const auto appendLeaf = [&](unsigned int trackIndex) {
      const int sourceId = request.tracks.SourceId(trackIndex);
      KFParticleGpuMaterializedParticle particle;
      particle.inputTrack = true;
      particle.sourceId = sourceId;
      particle.pdg = request.tracks.Pdg(trackIndex);
      particle.primaryVertexIndex = request.tracks.PrimaryVertexIndex(trackIndex);
      particle.leafSourceIds.push_back(particle.sourceId);
      KFParticleGpuTrackState track;
      LoadTrackState(request.tracks.Numerical(), trackIndex, track);
      particle.fit.Initialize(track, request.tracks.Charge(trackIndex), TrackMass(particle.pdg));
      if (particle.sourceId < 0 || !Finite(particle.fit)) {
        result.status = KFGpuMaterializationInvalidNumericalState;
        result.particles.clear();
        return false;
      }
      if (request.tracks.FieldCoefficientsData()) {
        for (unsigned int component = 0u; component < 10u; ++component) {
          particle.fieldCoefficients[component] =
            request.tracks.FieldCoefficient(component, trackIndex);
        }
      }
      trackParticleIds[trackIndex] = static_cast<int>(result.particles.size());
      result.particles.push_back(std::move(particle));
      return true;
    };

    if (directTrackMapping) {
      unsigned int physicalTracks = 0u;
      for (unsigned int set = 0u; set < 4u; ++set) {
        const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
        const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
        if (first.offset > request.tracks.Size()
            || first.size > request.tracks.Size() - first.offset
            || last.offset > request.tracks.Size()
            || last.size > request.tracks.Size() - last.offset
            || first.size != last.size
            || physicalTracks > std::numeric_limits<unsigned int>::max() - first.size) {
          result.status = KFGpuMaterializationInvalidTrackLayout;
          return result;
        }
        physicalTracks += first.size;
      }
      result.particles.reserve(static_cast<std::size_t>(physicalTracks)
                               + candidateOrder.size());
      auto& physicalSourceIds = workspace.physicalSourceIds;
      physicalSourceIds.reserve(physicalTracks);
      for (unsigned int set = 0u; set < 4u; ++set) {
        const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
        const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
        for (unsigned int ordinal = 0u; ordinal < first.size; ++ordinal) {
          const unsigned int firstIndex = first.offset + ordinal;
          const unsigned int lastIndex = last.offset + ordinal;
          if (firstIndex == lastIndex || trackSeen[firstIndex] != 0u
              || trackSeen[lastIndex] != 0u
              || request.tracks.SourceId(firstIndex) != request.tracks.SourceId(lastIndex)) {
            result.status = KFGpuMaterializationInvalidTrackLayout;
            result.particles.clear();
            return result;
          }
          workspace.touchedTracks.push_back(firstIndex);
          workspace.touchedTracks.push_back(lastIndex);
          trackSeen[firstIndex] = 1u;
          trackSeen[lastIndex] = 1u;
          physicalSourceIds.push_back(request.tracks.SourceId(firstIndex));
        }
      }
      std::sort(physicalSourceIds.begin(), physicalSourceIds.end());
      if (std::adjacent_find(physicalSourceIds.begin(), physicalSourceIds.end())
          != physicalSourceIds.end()) {
        result.status = KFGpuMaterializationInvalidTrackLayout;
        return result;
      }
      for (unsigned int set = 0u; set < 4u; ++set) {
        const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
        const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
        for (unsigned int ordinal = 0u; ordinal < first.size; ++ordinal) {
          const unsigned int firstIndex = first.offset + ordinal;
          const unsigned int lastIndex = last.offset + ordinal;
          if (!appendLeaf(firstIndex)) { return result; }
          trackParticleIds[lastIndex] = trackParticleIds[firstIndex];
        }
      }
    }
    else {
      std::unordered_map<int, int> sourceParticleIds;
      sourceParticleIds.reserve(uniqueTracks.size());
      result.particles.reserve(uniqueTracks.size() + candidateOrder.size());
      for (unsigned int trackIndex : uniqueTracks) {
        const int sourceId = request.tracks.SourceId(trackIndex);
        const auto existing = sourceParticleIds.find(sourceId);
        if (existing != sourceParticleIds.end()) {
          // First/last representations share one public leaf per source.
          trackParticleIds[trackIndex] = existing->second;
          continue;
        }
        if (!appendLeaf(trackIndex)) { return result; }
        sourceParticleIds.emplace(sourceId, trackParticleIds[trackIndex]);
      }
    }

    timer.Next(request.timing ? &request.timing->candidatesMilliseconds : nullptr);
    for (unsigned int candidate : candidateOrder) {
      KFParticleGpuMaterializedParticle particle;
      particle.gpuCandidateIndex = candidate;
      particle.channelId = request.candidates.Metadata().ChannelId(candidate);
      particle.topology = request.candidates.Metadata().Topology(candidate);
      particle.outputClass = request.candidates.Metadata().OutputClass(candidate);
      particle.operationStatus = request.candidates.Metadata().OperationStatus(candidate);
      particle.pdg = request.candidates.Metadata().Pdg(candidate);
      particle.primaryVertexIndex =
        request.candidates.Metadata().PrimaryVertexIndex(candidate);
      LoadCandidateFit(request.candidates, candidate, particle.fit);
      if (!Finite(particle.fit)) {
        result.status = KFGpuMaterializationInvalidNumericalState;
        result.particles.clear();
        return result;
      }

      const unsigned int directCount =
        request.candidates.Metadata().DirectDaughterCount(candidate);
      for (unsigned int daughter = 0u; daughter < directCount; ++daughter) {
        const unsigned int kind = daughter == 0u
                                    ? request.candidates.Metadata().DirectFirstKind(candidate)
                                    : request.candidates.Metadata().DirectSecondKind(candidate);
        const unsigned int index = daughter == 0u
                                     ? request.candidates.Metadata().DirectFirstIndex(candidate)
                                     : request.candidates.Metadata().DirectSecondIndex(candidate);
        const int particleId = kind == KFGpuDirectDaughterInputTrack
                                 ? trackParticleIds[index]
                                 : candidateParticleIds[index];
        if (particleId < 0) {
          result.status = KFGpuMaterializationInvalidDirectDaughter;
          result.particles.clear();
          return result;
        }
        particle.daughterParticleIds.push_back(particleId);
        const auto& directParticle =
          result.particles[static_cast<std::size_t>(particleId)];
        const auto& leaves = directParticle.leafSourceIds;
        particle.leafSourceIds.insert(
          particle.leafSourceIds.end(), leaves.begin(), leaves.end());
        if (daughter == 0u) {
          for (unsigned int component = 0u; component < 10u; ++component) {
            particle.fieldCoefficients[component] =
              directParticle.fieldCoefficients[component];
          }
        }
      }
      Canonicalize(particle.leafSourceIds);

      auto& storedLineage = workspace.storedLineage;
      storedLineage.clear();
      const unsigned int offset = request.candidates.Metadata().DaughterOffset(candidate);
      const unsigned int count = request.candidates.Metadata().DaughterCount(candidate);
      if (!request.candidates.Daughters().CanStore(offset, count)
          || offset > request.candidates.Daughters().Size()
          || count > request.candidates.Daughters().Size() - offset) {
        result.status = KFGpuMaterializationInvalidRange;
        result.particles.clear();
        return result;
      }
      storedLineage.reserve(count);
      for (unsigned int daughter = 0u; daughter < count; ++daughter) {
        storedLineage.push_back(request.candidates.Daughters().SourceId(offset + daughter));
      }
      Canonicalize(storedLineage);
      if (storedLineage != particle.leafSourceIds) {
        result.status = KFGpuMaterializationLineageMismatch;
        result.particles.clear();
        return result;
      }

      candidateParticleIds[candidate] = static_cast<int>(result.particles.size());
      result.particles.push_back(std::move(particle));
    }
    return result;
  }

  /**
   * @brief Replace scalar particle output only after successful event materialization.
   * Uses direct conversion when workspace, includeAllInputTracks, and event are present
   * and metadata is null; otherwise builds the independent intermediate graph first.
   * Each output particle receives its vector index as ID. Input-leaf DaughterIds contain the
   * physical source ID; composite DaughterIds reference immediate daughter particle IDs.
   * @tparam Particle Default-constructible scalar type exposing the KFParticle-style setters
   * and state accessors used here, including ReserveDaughterIds for the direct path.
   * @param request Host input and optional worker-local scratch/timing.
   * @param[in,out] destination Replaced on success; unchanged on validation failure.
   * @param[in,out] metadata Optional intermediate records, also replaced only on success.
   * @param[out] status Optional validation status; not an exception-reporting channel.
   * @return True on successful publication; false for a detected validation error.
   */
  template<typename Particle>
  static bool CommitToParticles(const KFParticleGpuMaterializationRequest& request,
                                std::vector<Particle>& destination,
                                std::vector<KFParticleGpuMaterializedParticle>* metadata = nullptr,
                                KFParticleGpuMaterializationStatus* status = nullptr)
  {
    if (request.workspace != nullptr && request.includeAllInputTracks
        && request.event != nullptr && metadata == nullptr) {
      return CommitDirect(request, destination, status);
    }

    KFParticleGpuMaterializedEvent event = Build(request);
    if (status) {
      *status = event.status;
    }
    if (!event.Succeeded()) {
      return false;
    }

    PhaseTimer timer(request.timing ? &request.timing->commitMilliseconds : nullptr);
    std::vector<Particle> temporary;
    temporary.reserve(event.particles.size());
    for (const auto& source : event.particles) {
      const int particleId = static_cast<int>(temporary.size());
      temporary.emplace_back();
      auto& particle = temporary.back();
      for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
        particle.Parameter(component) = source.fit.Parameter(component);
      }
      for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
        particle.Covariance(component) = source.fit.Covariance(component);
      }
      particle.Chi2() = source.fit.Chi2();
      particle.NDF() = source.fit.NDF();
      particle.Q() = static_cast<char>(source.fit.Q());
      particle.SetSFromDecay(source.fit.SFromDecay());
      particle.SetSumDaughterMass(source.fit.SumDaughterMass());
      particle.SetMassHypo(source.fit.MassHypo());
      particle.SetConstructMethod(source.fit.ConstructMethod());
      particle.SetPDG(source.pdg);
      particle.SetId(particleId);
#ifdef NonhomogeneousField
      for (int component = 0; component < 10; ++component) {
        particle.SetFieldCoeff(source.fieldCoefficients[component], component);
      }
#endif
      if (source.inputTrack) {
        particle.AddDaughterId(source.sourceId);
      }
      else {
        for (int daughterId : source.daughterParticleIds) {
          particle.AddDaughterId(daughterId);
        }
      }
    }

    destination.swap(temporary);
    if (metadata) {
      metadata->swap(event.particles);
    }
    return true;
  }

 private:
  /**
   * @brief Materialize directly into a temporary scalar vector without intermediate particle records.
   * Uses event-local map intervals for complete plans and the general ancestry walk otherwise.
   * Pairs first/last track representations, checks bounded canonical lineages, and inherits each
   * composite's field source from its first direct daughter. The destination is swapped only on success.
   * @pre request.workspace and request.event are non-null and includeAllInputTracks is true.
   * @note Only the caller-proven track-layout checks are skipped for validatedPublicationInput;
   * candidate ancestry, numeric validity, and lineage consistency are still checked.
   */
  template<typename Particle>
  static bool CommitDirect(const KFParticleGpuMaterializationRequest& request,
                           std::vector<Particle>& destination,
                           KFParticleGpuMaterializationStatus* status)
  {
    const auto fail = [&](KFParticleGpuMaterializationStatus failure) {
      if (status) { *status = failure; }
      return false;
    };
    if (status) { *status = KFGpuMaterializationSucceeded; }

    PhaseTimer timer(request.timing ? &request.timing->graphMilliseconds : nullptr);
    if (request.candidates.OverflowFlags() != 0u) {
      return fail(KFGpuMaterializationOverflow);
    }
    const unsigned int candidateSize = request.candidates.Size();
    if (request.numberOfRanges != 0u && request.ranges == nullptr) {
      return fail(KFGpuMaterializationInvalidRange);
    }
    if (request.validatedPublicationInput && !request.completePublicationPlan) {
      return fail(KFGpuMaterializationInvalidTrackLayout);
    }
    const bool trustPublicationInput = request.validatedPublicationInput;

    unsigned int candidateMapOffset = 0u;
    unsigned int candidateMapEnd = candidateSize;
    unsigned int trackMapOffset = 0u;
    unsigned int trackMapEnd = request.tracks.Size();
    if (request.completePublicationPlan) {
      candidateMapOffset = candidateSize;
      candidateMapEnd = 0u;
      for (unsigned int rangeIndex = 0u; rangeIndex < request.numberOfRanges;
           ++rangeIndex) {
        const KFParticleGpuCandidateRange& range = request.ranges[rangeIndex];
        if (range.overflowFlags != 0u) {
          return fail(KFGpuMaterializationOverflow);
        }
        if (range.offset > candidateSize
            || range.size > candidateSize - range.offset) {
          return fail(KFGpuMaterializationInvalidRange);
        }
        const unsigned int daughterSize = request.candidates.Daughters().Size();
        if (range.daughterOffset > daughterSize
            || range.daughterSize > daughterSize - range.daughterOffset) {
          return fail(KFGpuMaterializationInvalidRange);
        }
        if (range.size != 0u) {
          candidateMapOffset = std::min(candidateMapOffset, range.offset);
          candidateMapEnd = std::max(candidateMapEnd, range.offset + range.size);
        }
      }
      if (candidateMapEnd == 0u) { candidateMapOffset = 0u; }

      trackMapOffset = request.tracks.Size();
      trackMapEnd = 0u;
      for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
        const KFParticleGpuRange& range = request.event->trackSets[set].tracks;
        if (range.offset > request.tracks.Size()
            || range.size > request.tracks.Size() - range.offset) {
          return fail(KFGpuMaterializationInvalidTrackLayout);
        }
        if (range.size != 0u) {
          trackMapOffset = std::min(trackMapOffset, range.offset);
          trackMapEnd = std::max(trackMapEnd, range.offset + range.size);
        }
      }
      if (trackMapEnd == 0u) { trackMapOffset = 0u; }
    }

    const auto candidateInMap = [&](unsigned int candidate) {
      return candidate >= candidateMapOffset && candidate < candidateMapEnd;
    };
    const auto candidateMapIndex = [&](unsigned int candidate) {
      return candidate - candidateMapOffset;
    };
    const auto trackInMap = [&](unsigned int track) {
      return track >= trackMapOffset && track < trackMapEnd;
    };
    const auto trackMapIndex = [&](unsigned int track) {
      return track - trackMapOffset;
    };

    auto& workspace = *request.workspace;
    workspace.Begin(candidateMapEnd - candidateMapOffset,
                    trackMapEnd - trackMapOffset);
    auto& requested = workspace.requested;
    for (unsigned int rangeIndex = 0u; rangeIndex < request.numberOfRanges; ++rangeIndex) {
      const KFParticleGpuCandidateRange& range = request.ranges[rangeIndex];
      if (range.overflowFlags != 0u) { return fail(KFGpuMaterializationOverflow); }
      if (range.offset > candidateSize || range.size > candidateSize - range.offset) {
        return fail(KFGpuMaterializationInvalidRange);
      }
      const unsigned int daughterSize = request.candidates.Daughters().Size();
      if (range.daughterOffset > daughterSize
          || range.daughterSize > daughterSize - range.daughterOffset) {
        return fail(KFGpuMaterializationInvalidRange);
      }
      for (unsigned int candidate = range.offset;
           candidate < range.offset + range.size; ++candidate) {
        if (request.completePublicationPlan) {
          const auto& metadata = request.candidates.Metadata();
          if (metadata.EventIndex(candidate) != request.eventIndex) {
            return fail(KFGpuMaterializationInvalidCandidate);
          }
          if (metadata.OperationStatus(candidate) != KFGpuCandidateOperationAccepted
              || (metadata.Flags(candidate) & KFGpuCandidateValid) == 0u) {
            continue;
          }
        }
        requested.push_back(candidate);
      }
    }

    auto& visit = workspace.visit;
    auto& orderedCandidates = workspace.orderedCandidates;
    const bool directPublicationOrder = request.completePublicationPlan;
    const auto candidateIsValid = [&](unsigned int candidate) {
      if (candidate >= candidateSize) { return false; }
      const auto& metadataView = request.candidates.Metadata();
      return metadataView.EventIndex(candidate) == request.eventIndex
             && metadataView.OperationStatus(candidate) == KFGpuCandidateOperationAccepted
             && (metadataView.Flags(candidate) & KFGpuCandidateValid) != 0u
             && metadataView.OutputClass(candidate) != KFGpuGraphOutputInvalid
             && metadataView.OutputClass(candidate) < KFGpuGraphOutputClassCount
             && metadataView.DirectDaughterCount(candidate) != 0u
             && metadataView.DirectDaughterCount(candidate) <= 2u;
    };

    if (request.completePublicationPlan) {
      unsigned int previous = 0u;
      bool havePrevious = false;
      for (unsigned int candidate : requested) {
        if ((havePrevious && candidate <= previous) || !candidateIsValid(candidate)) {
          return fail(KFGpuMaterializationInvalidCandidate);
        }
        const auto& metadataView = request.candidates.Metadata();
        for (unsigned int daughter = 0u;
             daughter < metadataView.DirectDaughterCount(candidate); ++daughter) {
          const unsigned int kind = daughter == 0u
                                      ? metadataView.DirectFirstKind(candidate)
                                      : metadataView.DirectSecondKind(candidate);
          const unsigned int index = daughter == 0u
                                       ? metadataView.DirectFirstIndex(candidate)
                                       : metadataView.DirectSecondIndex(candidate);
          if (kind == KFGpuDirectDaughterInputTrack) {
            if (index >= request.tracks.Size() || !trackInMap(index)) {
              return fail(KFGpuMaterializationInvalidDirectDaughter);
            }
          }
          else if (kind == KFGpuDirectDaughterCandidate) {
            if (index >= candidate || !candidateInMap(index)
                || visit[candidateMapIndex(index)] != 2u) {
              return fail(index >= candidate ? KFGpuMaterializationCyclicAncestry
                                             : KFGpuMaterializationInvalidDirectDaughter);
            }
          }
          else {
            return fail(KFGpuMaterializationInvalidDirectDaughter);
          }
        }
        const unsigned int localCandidate = candidateMapIndex(candidate);
        workspace.touchedCandidates.push_back(localCandidate);
        visit[localCandidate] = 2u;
        previous = candidate;
        havePrevious = true;
      }
    }
    else {
      auto& traversal = workspace.traversal;
      traversal.reserve(requested.size());
      for (unsigned int root : requested) {
        if (root >= candidateSize) { return fail(KFGpuMaterializationInvalidCandidate); }
        if (visit[candidateMapIndex(root)] == 2u) { continue; }
        if (visit[candidateMapIndex(root)] == 1u) {
          return fail(KFGpuMaterializationCyclicAncestry);
        }
        traversal.push_back({root, 0u});
        while (!traversal.empty()) {
          KFParticleGpuMaterializationWorkspace::CandidateFrame& frame = traversal.back();
          const unsigned int candidate = frame.candidate;
          const unsigned int localCandidate = candidateMapIndex(candidate);
          if (visit[localCandidate] == 0u) {
            if (!candidateIsValid(candidate)) {
              return fail(KFGpuMaterializationInvalidCandidate);
            }
            workspace.touchedCandidates.push_back(localCandidate);
            visit[localCandidate] = 1u;
          }

          const auto& metadataView = request.candidates.Metadata();
          if (frame.nextDaughter < metadataView.DirectDaughterCount(candidate)) {
            const unsigned int daughter = frame.nextDaughter++;
            const unsigned int kind = daughter == 0u
                                        ? metadataView.DirectFirstKind(candidate)
                                        : metadataView.DirectSecondKind(candidate);
            const unsigned int index = daughter == 0u
                                         ? metadataView.DirectFirstIndex(candidate)
                                         : metadataView.DirectSecondIndex(candidate);
            if (kind == KFGpuDirectDaughterInputTrack) {
              if (index >= request.tracks.Size()) {
                return fail(KFGpuMaterializationInvalidDirectDaughter);
              }
            }
            else if (kind == KFGpuDirectDaughterCandidate) {
              if (index >= candidateSize) {
                return fail(KFGpuMaterializationInvalidCandidate);
              }
              if (index == candidate || visit[candidateMapIndex(index)] == 1u) {
                return fail(KFGpuMaterializationCyclicAncestry);
              }
              if (visit[candidateMapIndex(index)] == 0u) {
                traversal.push_back({index, 0u});
              }
            }
            else {
              return fail(KFGpuMaterializationInvalidDirectDaughter);
            }
          }
          else {
            visit[localCandidate] = 2u;
            orderedCandidates.push_back(candidate);
            traversal.pop_back();
          }
        }
      }
    }
    const auto& candidateOrder = directPublicationOrder
                                   ? requested
                                   : orderedCandidates;

    timer.Next(request.timing ? &request.timing->leavesMilliseconds : nullptr);
    unsigned int physicalTracks = 0u;
    for (unsigned int set = 0u; set < 4u; ++set) {
      const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
      const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
      if (!trustPublicationInput
          && (first.offset > request.tracks.Size()
              || first.size > request.tracks.Size() - first.offset
              || last.offset > request.tracks.Size()
              || last.size > request.tracks.Size() - last.offset
              || first.size != last.size
              || physicalTracks > std::numeric_limits<unsigned int>::max() - first.size)) {
        return fail(KFGpuMaterializationInvalidTrackLayout);
      }
      physicalTracks += first.size;
    }
    if (static_cast<std::size_t>(physicalTracks) + candidateOrder.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return fail(KFGpuMaterializationInvalidRange);
    }

    auto& trackSeen = workspace.trackSeen;
    auto& trackParticleIds = workspace.trackParticleIds;
    auto& trackFieldSourceIndices = workspace.trackFieldSourceIndices;
    auto& candidateParticleIds = workspace.candidateParticleIds;
    auto& candidateFieldSourceIndices = workspace.candidateFieldSourceIndices;
    if (!trustPublicationInput) {
      auto& physicalSourceIds = workspace.physicalSourceIds;
      physicalSourceIds.reserve(physicalTracks);
      for (unsigned int set = 0u; set < 4u; ++set) {
        const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
        const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
        for (unsigned int ordinal = 0u; ordinal < first.size; ++ordinal) {
          const unsigned int firstIndex = first.offset + ordinal;
          const unsigned int lastIndex = last.offset + ordinal;
          if (!trackInMap(firstIndex) || !trackInMap(lastIndex)) {
            return fail(KFGpuMaterializationInvalidTrackLayout);
          }
          const unsigned int localFirst = trackMapIndex(firstIndex);
          const unsigned int localLast = trackMapIndex(lastIndex);
          if (firstIndex == lastIndex || trackSeen[localFirst] != 0u
              || trackSeen[localLast] != 0u
              || request.tracks.SourceId(firstIndex) != request.tracks.SourceId(lastIndex)) {
            return fail(KFGpuMaterializationInvalidTrackLayout);
          }
          workspace.touchedTracks.push_back(localFirst);
          workspace.touchedTracks.push_back(localLast);
          trackSeen[localFirst] = 1u;
          trackSeen[localLast] = 1u;
          physicalSourceIds.push_back(request.tracks.SourceId(firstIndex));
        }
      }
      std::sort(physicalSourceIds.begin(), physicalSourceIds.end());
      if (std::adjacent_find(physicalSourceIds.begin(), physicalSourceIds.end())
          != physicalSourceIds.end()) {
        return fail(KFGpuMaterializationInvalidTrackLayout);
      }
    }

    std::vector<Particle> temporary;
    temporary.reserve(static_cast<std::size_t>(physicalTracks)
                      + candidateOrder.size());
    for (unsigned int set = 0u; set < 4u; ++set) {
      const KFParticleGpuRange& first = request.event->trackSets[set].tracks;
      const KFParticleGpuRange& last = request.event->trackSets[set + 4u].tracks;
      for (unsigned int ordinal = 0u; ordinal < first.size; ++ordinal) {
        const unsigned int firstIndex = first.offset + ordinal;
        const unsigned int lastIndex = last.offset + ordinal;
        const int sourceId = request.tracks.SourceId(firstIndex);
        const int pdg = request.tracks.Pdg(firstIndex);
        KFParticleGpuTrackState track;
        LoadTrackState(request.tracks.Numerical(), firstIndex, track);
        KFParticleGpuFitState fit;
        fit.Initialize(track, request.tracks.Charge(firstIndex), TrackMass(pdg));
        if (sourceId < 0 || !Finite(fit)) {
          return fail(KFGpuMaterializationInvalidNumericalState);
        }
        const int particleId = static_cast<int>(temporary.size());
        temporary.emplace_back();
        FillParticle(temporary.back(), fit, pdg, particleId, request,
                     firstIndex, &sourceId, 1u);
        const unsigned int localFirst = trackMapIndex(firstIndex);
        const unsigned int localLast = trackMapIndex(lastIndex);
        if (trustPublicationInput) {
          workspace.touchedTracks.push_back(localFirst);
          workspace.touchedTracks.push_back(localLast);
        }
        trackParticleIds[localFirst] = particleId;
        trackParticleIds[localLast] = particleId;
        trackFieldSourceIndices[localFirst] = firstIndex;
        trackFieldSourceIndices[localLast] = firstIndex;
      }
    }

    timer.Next(request.timing ? &request.timing->candidatesMilliseconds : nullptr);
    const unsigned int invalidTrack = std::numeric_limits<unsigned int>::max();
    for (unsigned int candidate : candidateOrder) {
      KFParticleGpuFitState fit;
      LoadCandidateFit(request.candidates, candidate, fit);
      if (!Finite(fit)) { return fail(KFGpuMaterializationInvalidNumericalState); }

      int daughterParticleIds[2] = {-1, -1};
      int directLineages[2][MaximumPublicationLineage] = {};
      unsigned int directLineageSizes[2] = {};
      unsigned int fieldSourceIndex = invalidTrack;
      const auto& metadataView = request.candidates.Metadata();
      const unsigned int directCount = metadataView.DirectDaughterCount(candidate);
      for (unsigned int daughter = 0u; daughter < directCount; ++daughter) {
        const unsigned int kind = daughter == 0u
                                    ? metadataView.DirectFirstKind(candidate)
                                    : metadataView.DirectSecondKind(candidate);
        const unsigned int index = daughter == 0u
                                     ? metadataView.DirectFirstIndex(candidate)
                                     : metadataView.DirectSecondIndex(candidate);
        if (kind == KFGpuDirectDaughterInputTrack) {
          if (index >= request.tracks.Size() || !trackInMap(index)) {
            return fail(KFGpuMaterializationInvalidDirectDaughter);
          }
          const unsigned int localTrack = trackMapIndex(index);
          if (trackParticleIds[localTrack] < 0
              || trackFieldSourceIndices[localTrack] == invalidTrack) {
            return fail(KFGpuMaterializationInvalidDirectDaughter);
          }
          daughterParticleIds[daughter] = trackParticleIds[localTrack];
          directLineages[daughter][0] = request.tracks.SourceId(index);
          directLineageSizes[daughter] = 1u;
          if (daughter == 0u) {
            fieldSourceIndex = trackFieldSourceIndices[localTrack];
          }
        }
        else if (kind == KFGpuDirectDaughterCandidate) {
          if (index >= candidateSize || !candidateInMap(index)) {
            return fail(KFGpuMaterializationInvalidDirectDaughter);
          }
          const unsigned int localCandidate = candidateMapIndex(index);
          if (candidateParticleIds[localCandidate] < 0
              || candidateFieldSourceIndices[localCandidate] == invalidTrack) {
            return fail(KFGpuMaterializationInvalidDirectDaughter);
          }
          daughterParticleIds[daughter] = candidateParticleIds[localCandidate];
          if (!LoadCanonicalLineage(request.candidates, index,
                                    directLineages[daughter],
                                    directLineageSizes[daughter])) {
            return fail(KFGpuMaterializationLineageMismatch);
          }
          if (daughter == 0u) {
            fieldSourceIndex = candidateFieldSourceIndices[localCandidate];
          }
        }
        else {
          return fail(KFGpuMaterializationInvalidDirectDaughter);
        }
      }
      int computedLineage[MaximumPublicationLineage] = {};
      unsigned int computedCount = 0u;
      if (!MergeCanonicalLineages(directLineages[0], directLineageSizes[0],
                                  directLineages[1], directLineageSizes[1],
                                  computedLineage, computedCount)) {
        return fail(KFGpuMaterializationLineageMismatch);
      }
      int storedLineage[MaximumPublicationLineage] = {};
      unsigned int storedCount = 0u;
      if (!LoadCanonicalLineage(request.candidates, candidate,
                                storedLineage, storedCount)
          || storedCount != computedCount
          || !std::equal(storedLineage, storedLineage + storedCount,
                         computedLineage)) {
        return fail(KFGpuMaterializationLineageMismatch);
      }
      if (fieldSourceIndex == invalidTrack) {
        return fail(KFGpuMaterializationInvalidDirectDaughter);
      }

      const int particleId = static_cast<int>(temporary.size());
      temporary.emplace_back();
      FillParticle(temporary.back(), fit, metadataView.Pdg(candidate), particleId,
                   request, fieldSourceIndex, daughterParticleIds, directCount);
      const unsigned int localCandidate = candidateMapIndex(candidate);
      candidateParticleIds[localCandidate] = particleId;
      candidateFieldSourceIndices[localCandidate] = fieldSourceIndex;
    }

    timer.Next(request.timing ? &request.timing->commitMilliseconds : nullptr);
    destination.swap(temporary);
    return true;
  }

  /**
   * @brief Scope-bound host wall timer that adds elapsed time to the current phase accumulator.
   * Switching phases closes the previous interval; destruction closes the final interval.
   * A null destination disables measurement until a non-null accumulator is supplied.
   */
  class PhaseTimer
  {
   public:
    /**
     * @brief Start measuring into destination, or stay disabled when it is null.
     */
    explicit PhaseTimer(double* destination) { Next(destination); }

    /**
     * @brief Add the final active interval to its accumulator.
     */
    ~PhaseTimer() { Next(nullptr); }

    /**
     * @brief Accumulate the current interval and begin timing into the next destination.
     */
    void Next(double* destination)
    {
      if (fDestination) {
        *fDestination += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - fStarted).count();
      }
      fDestination = destination;
      if (destination) { fStarted = std::chrono::steady_clock::now(); }
    }

   private:
    double* fDestination = nullptr; ///< Borrowed phase accumulator, or null when disabled.
    std::chrono::steady_clock::time_point fStarted; ///< Start of the currently measured interval.
  };

  /**
   * @brief Maximum physical-track lineage supported by the direct path's fixed-size scratch arrays.
   */
  static constexpr unsigned int MaximumPublicationLineage = 16u;

  /**
   * @brief Load a candidate's physical source IDs into a sorted, duplicate-free bounded array.
   * Uses insertion ordering because the direct path supports only short lineages.
   * @param candidates Host-readable candidate pool and daughter-lineage storage.
   * @param candidate Absolute candidate index.
   * @param[out] lineage Buffer with MaximumPublicationLineage entries.
   * @param[out] size Number of loaded entries on success.
   * @return False for invalid ranges, empty/oversized lineage, or duplicates.
   * Output contents are not usable as a complete lineage after failure.
   */
  static bool LoadCanonicalLineage(const KFParticleGpuConstCandidatePoolView& candidates,
                                   unsigned int candidate,
                                   int lineage[MaximumPublicationLineage],
                                   unsigned int& size)
  {
    if (candidate >= candidates.Size()) { return false; }
    const unsigned int offset = candidates.Metadata().DaughterOffset(candidate);
    const unsigned int count = candidates.Metadata().DaughterCount(candidate);
    if (count == 0u || count > MaximumPublicationLineage
        || !candidates.Daughters().CanStore(offset, count)
        || offset > candidates.Daughters().Size()
        || count > candidates.Daughters().Size() - offset) {
      return false;
    }
    size = 0u;
    for (unsigned int daughter = 0u; daughter < count; ++daughter) {
      const int source = candidates.Daughters().SourceId(offset + daughter);
      unsigned int insertion = size;
      while (insertion > 0u && lineage[insertion - 1u] > source) {
        lineage[insertion] = lineage[insertion - 1u];
        --insertion;
      }
      if ((insertion > 0u && lineage[insertion - 1u] == source)
          || (insertion < size && lineage[insertion] == source)) {
        return false;
      }
      lineage[insertion] = source;
      ++size;
    }
    return true;
  }

  /**
   * @brief Merge two sorted unique source-ID lists into their canonical set union.
   * Shared sources appear once, matching the intermediate path's union semantics.
   * The combined input length must fit MaximumPublicationLineage even when duplicates would shrink the union.
   * @pre Input lists are sorted and unique; output holds MaximumPublicationLineage entries.
   * @param[out] outputSize Number of union entries on success; unchanged on capacity failure.
   * @return Whether both input lengths fit the fixed scratch limit.
   */
  static bool MergeCanonicalLineages(const int* first,
                                     unsigned int firstSize,
                                     const int* second,
                                     unsigned int secondSize,
                                     int output[MaximumPublicationLineage],
                                     unsigned int& outputSize)
  {
    if (firstSize + secondSize > MaximumPublicationLineage) { return false; }
    unsigned int firstIndex = 0u;
    unsigned int secondIndex = 0u;
    outputSize = 0u;
    while (firstIndex < firstSize || secondIndex < secondSize) {
      const bool takeFirst = secondIndex >= secondSize
                             || (firstIndex < firstSize
                                 && first[firstIndex] < second[secondIndex]);
      const int source = takeFirst ? first[firstIndex++] : second[secondIndex++];
      // The independent reference canonicalizes the union. Production GPU
      // graph construction rejects physical source reuse before publication,
      // but retaining set-union semantics here keeps diagnostic fixtures and
      // externally supplied candidate pools behaviorally compatible.
      if (outputSize != 0u && output[outputSize - 1u] == source) { continue; }
      output[outputSize++] = source;
    }
    return true;
  }

  /**
   * @brief Look up the leaf mass hypothesis by absolute PDG code.
   * Handles the explicitly listed leptons, kaon, proton, and light nuclei; all other codes
   * fall back to the charged-pion mass. This is a local initialization table, not a full PDG database.
   */
  static float TrackMass(int pdg)
  {
    switch (pdg < 0 ? -pdg : pdg) {
      case 11: return 0.00051099895f;
      case 13: return 0.1056583755f;
      case 321: return 0.493677f;
      case 2212: return 0.9382720813f;
      case 1000010020: return 1.87561294257f;
      case 1000010030: return 2.80892113298f;
      case 1000020030: return 2.80839160743f;
      case 1000020040: return 3.7273794066f;
      case 1000020060: return 5.6055375f;
      case 1000030060: return 5.6015181f;
      case 1000030070: return 6.5338336f;
      case 1000040070: return 6.5341844f;
      default: return 0.13957039f;
    }
  }

  /**
   * @brief Copy a fit and identity into a newly constructed scalar particle.
   * Copies field coefficients from fieldSourceIndex when NonhomogeneousField is enabled,
   * or writes zeros when the input has no field data. Appends the supplied daughter IDs.
   * @pre particle has no existing daughters; fit and referenced input data have been validated.
   * @param particleId Event-output index used as the scalar particle ID.
   * @param daughterIds Physical source ID for a leaf, immediate particle IDs for a composite.
   */
  template<typename Particle>
  static void FillParticle(Particle& particle,
                           const KFParticleGpuFitState& fit,
                           int pdg,
                           int particleId,
                           const KFParticleGpuMaterializationRequest& request,
                           unsigned int fieldSourceIndex,
                           const int* daughterIds,
                           unsigned int numberOfDaughters)
  {
    for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
      particle.Parameter(component) = fit.Parameter(component);
    }
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      particle.Covariance(component) = fit.Covariance(component);
    }
    particle.Chi2() = fit.Chi2();
    particle.NDF() = fit.NDF();
    particle.Q() = static_cast<char>(fit.Q());
    particle.SetSFromDecay(fit.SFromDecay());
    particle.SetSumDaughterMass(fit.SumDaughterMass());
    particle.SetMassHypo(fit.MassHypo());
    particle.SetConstructMethod(fit.ConstructMethod());
    particle.SetPDG(pdg);
    particle.SetId(particleId);
#ifdef NonhomogeneousField
    for (int component = 0; component < 10; ++component) {
      const float coefficient = request.tracks.FieldCoefficientsData()
                                  ? request.tracks.FieldCoefficient(
                                      static_cast<unsigned int>(component),
                                      fieldSourceIndex)
                                  : 0.f;
      particle.SetFieldCoeff(coefficient, component);
    }
#else
    (void) request;
    (void) fieldSourceIndex;
#endif
    particle.ReserveDaughterIds(numberOfDaughters);
    for (unsigned int daughter = 0u; daughter < numberOfDaughters; ++daughter) {
      particle.AddDaughterId(daughterIds[daughter]);
    }
  }

  /**
   * @brief Check finite state/covariance/fit scalars and charge representability as char.
   * Does not test covariance positivity, physical kinematics, or selection quality.
   */
  static bool Finite(const KFParticleGpuFitState& fit)
  {
    for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
      if (!std::isfinite(fit.Parameter(component))) {
        return false;
      }
    }
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      if (!std::isfinite(fit.Covariance(component))) {
        return false;
      }
    }
    return std::isfinite(fit.Chi2()) && std::isfinite(fit.SFromDecay())
           && std::isfinite(fit.SumDaughterMass()) && std::isfinite(fit.MassHypo())
           && fit.Q() >= static_cast<int>(std::numeric_limits<char>::min())
           && fit.Q() <= static_cast<int>(std::numeric_limits<char>::max());
  }

  /**
   * @brief Sort physical source IDs and remove duplicates in place for reference-path comparisons.
   */
  static void Canonicalize(std::vector<int>& lineage)
  {
    std::sort(lineage.begin(), lineage.end());
    lineage.erase(std::unique(lineage.begin(), lineage.end()), lineage.end());
  }
};

#endif
