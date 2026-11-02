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
 * @brief Flat batch descriptors, device work/status views, and host launch validation.
 *
 * Domains partition input and reserve disjoint output storage. Work ranges describe
 * execution within a domain; mutable status records track actual device output.
 * Time-window metadata is a future producer contract, not an enabled execution mode.
 */
#ifndef KFPARTICLEGPUTRANSACTION_H
#define KFPARTICLEGPUTRANSACTION_H

#include "data/KFParticleGpuInputDataTransfer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>

/** @brief Logical input partition; the current host validator accepts event domains only. */
enum KFParticleGpuDomainKind : unsigned int
{
  KFParticleGpuEventDomain = 0u, ///< One packed event, supported by the current launch validator.
  KFParticleGpuTimeWindowDomain = 1u ///< Reserved time-window partition, not yet accepted by validation.
};

/** @brief Pipeline operation represented by a domain-local work range. */
enum KFParticleGpuWorkFamily : unsigned int
{
  KFParticleGpuDomainInputWork = 0u, ///< Domain input declaration.
  KFParticleGpuTwoDaughterWork = 1u, ///< Input-track pair construction.
  KFParticleGpuCompositeTrackWork = 2u, ///< Candidate plus bachelor-track construction.
  KFParticleGpuGraphOperationWork = 3u ///< Generic dependent graph operation.
};

/** @brief Domain planning options, independent of mutable execution status. */
enum KFParticleGpuDomainFlags : unsigned int
{
  /** Output ranges are assigned after the device routing pass. */
  KFParticleGpuDomainCompactOutput = 1u << 0u
};

/** @brief First launch-validation failure, or Valid; values are not a bit mask. */
enum KFParticleGpuTransactionValidation : unsigned int
{
  KFParticleGpuTransactionValid = 0u,
  KFParticleGpuTransactionMissingStorage,
  KFParticleGpuTransactionPlanRevisionMismatch,
  KFParticleGpuTransactionRangeOverflow,
  KFParticleGpuTransactionInvalidDomain,
  KFParticleGpuTransactionOverlappingOutput ///< Includes overlapping input track or vertex ranges.
};

/** @brief Optional cumulative overlap-check counters; validation does not reset them. */
struct KFParticleGpuTransactionValidationStats
{
  unsigned long long domainPairComparisons = 0u; ///< Pair comparisons made by the reference checker.
  unsigned long long adjacentRangeComparisons = 0u; ///< Neighbor comparisons in the optimized checker.
  unsigned int rangeFamilies = 0u; ///< Families visited before success or early rejection.
  unsigned int sortedFallbackFamilies = 0u; ///< Unordered families requiring a temporary sorted copy.
};

/** @brief Batch identity, compiled-plan revision, and absolute domain/work table ranges. */
struct KFParticleGpuTransactionDesc
{
  unsigned long long transactionId = 0u; ///< Host-assigned batch identity.
  unsigned long long decayPlanRevision = 0u; ///< Compiled plan revision required by this batch.
  KFParticleGpuRange domains; ///< Active interval in the domain table.
  KFParticleGpuRange workRanges; ///< Active interval in the work table.
  unsigned int channelCount = 0u; ///< Per-domain channel-status count; zero disables that table requirement.
};

/**
 * @brief One independently addressable event with input ranges and output reservations.
 *
 * Ranges use absolute indices into transaction storage. Reservations are capacity
 * bounds, not produced counts; statusIndex selects the mutable domain status.
 * Compact planning may replace output reservations after counting retained tasks.
 */
struct KFParticleGpuDomainDesc
{
  unsigned long long sourceId = 0u; ///< Producer identity, distinct from the packed descriptor index.
  unsigned int kind = KFParticleGpuEventDomain; ///< Domain interpretation.
  unsigned int inputDescriptorIndex = 0u; ///< Packed event descriptor index for event domains.
  KFParticleGpuRange sourceTracks; ///< Domain-owned packed input tracks.
  KFParticleGpuRange primaryVertices; ///< Domain-owned packed primary vertices.
  KFParticleGpuRange workRanges; ///< Domain's interval in the shared work table.
  KFParticleGpuRange taskReservation; ///< Bounded task output interval.
  KFParticleGpuRange candidateReservation; ///< Bounded raw-candidate output interval.
  KFParticleGpuRange daughterReservation; ///< Bounded flattened source-ID output interval.
  KFParticleGpuRange selectedReservation; ///< Bounded selected-index output interval.
  unsigned int statusIndex = 0u; ///< Mutable domain-status slot.
  unsigned int flags = 0u; ///< KFParticleGpuDomainFlags bits.
  KFParticleGpuRange channelStatuses; ///< Per-channel status interval for this domain.
};

/**
 * @brief Flat work partition for an input, two-track, composite-track, or graph stage.
 *
 * domainIndex is absolute. generation and executionGroup select plan operations;
 * source ranges identify inputs, while outputReservation bounds task storage.
 */
struct KFParticleGpuWorkRange
{
  unsigned int domainIndex = 0u; ///< Absolute owner-domain index.
  unsigned int family = KFParticleGpuDomainInputWork; ///< KFParticleGpuWorkFamily selector.
  unsigned int generation = 0u; ///< Dependency generation within the compiled plan.
  unsigned int executionGroup = 0u; ///< Family-specific routing/execution group index.
  KFParticleGpuRange sourceTracks; ///< First input-track interval.
  KFParticleGpuRange secondarySourceTracks; ///< Second input-track interval for pair work.
  KFParticleGpuRange sourceCandidates; ///< Candidate-source interval available to this work.
  KFParticleGpuRange primaryVertices; ///< Primary vertices used by this work item.
  KFParticleGpuRange outputReservation; ///< Task output reservation, not a produced count.
};

/** @brief One 64-lane pair-routing tile owned by a work range; firstPair is local to that work. */
struct KFParticleGpuPairTile
{
  unsigned int workIndex = 0u;
  unsigned int firstPair = 0u;
};

/** @brief Number of consecutive pair indices represented by a routing tile. */
constexpr unsigned int KFParticleGpuPairTileSize = 64u;

/** @brief Construction tile identified by absolute domain and first-task indices. */
struct KFParticleGpuTaskTile
{
  unsigned int domainIndex = 0u;
  unsigned int firstTask = 0u;
};

/** @brief Returns ceil(pairCount / tile size) without overflowing an added rounding term. */
KFPARTICLE_GPU_HOST_DEVICE inline unsigned int KFParticleGpuPairTileCount(
  unsigned int pairCount)
{
  return pairCount / KFParticleGpuPairTileSize
         + (pairCount % KFParticleGpuPairTileSize != 0u ? 1u : 0u);
}

/**
 * @brief Appends ordered pair tiles for one work range to caller-owned host storage.
 * @param workIndex Absolute work-table index stored in every tile.
 * @param pairCount Number of local pairs to cover, including a possible partial last tile.
 * @param tiles Destination array; may be null only when no tiles are needed.
 * @param capacity Number of available tile slots.
 * @param[in,out] cursor First free slot, advanced by the number of appended tiles.
 * @return False on insufficient storage, with no writes or cursor change.
 * @pre The final firstPair increment must be representable as unsigned int.
 */
inline bool KFParticleGpuAppendPairTiles(unsigned int workIndex,
                                         unsigned int pairCount,
                                         KFParticleGpuPairTile* tiles,
                                         unsigned int capacity,
                                         unsigned int& cursor)
{
  const unsigned int tileCount = KFParticleGpuPairTileCount(pairCount);
  if (cursor > capacity || tileCount > capacity - cursor
      || (tileCount != 0u && tiles == nullptr)) {
    return false;
  }
  for (unsigned int firstPair = 0u; firstPair < pairCount;
       firstPair += KFParticleGpuPairTileSize) {
    tiles[cursor++] = KFParticleGpuPairTile{workIndex, firstPair};
  }
  return true;
}

/** @brief Marks whether dependent sources have been resolved and whether they are empty. */
enum KFParticleGpuWorkStatusFlags : unsigned int
{
  KFParticleGpuWorkSourceResolved = 1u << 0u,
  KFParticleGpuWorkSourceEmpty = 1u << 1u
};

/**
 * @brief Device-resolved generation input and routing/construction counters.
 *
 * Resolved counts do not enlarge reservations. acceptedTasks can exceed storedTasks;
 * overflowFlags records capacity failures rather than authorizing out-of-range writes.
 */
struct KFParticleGpuWorkStatus
{
  KFParticleGpuRange sourceCandidates;
  unsigned int sourceTracks = 0u;
  unsigned int visitedCombinations = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int constructedCandidates = 0u;
  unsigned int blockReservations = 0u;
  unsigned int overflowFlags = 0u;
  unsigned int flags = 0u;
};

/** @brief Per-domain execution counts and overflow, failure, and fallback flags. */
struct KFParticleGpuDomainStatus
{
  unsigned int visitedPairs = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int tasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int candidates = 0u;
  unsigned int daughters = 0u;
  unsigned int selected = 0u;
  unsigned int cascadeTasks = 0u;
  unsigned int graphTasks = 0u;
  unsigned int overflowFlags = 0u;
  unsigned int failureFlags = 0u;
  unsigned int fallbackFlags = 0u;
  unsigned int blockReservations = 0u;
};

/** @brief Per-channel routing counts and selected-output segment allocation/scatter state. */
struct KFParticleGpuDomainChannelStatus
{
  unsigned int visited = 0u;
  unsigned int accepted = 0u;
  unsigned int stored = 0u;
  unsigned int constructed = 0u;
  unsigned int selectionAccepted = 0u;
  unsigned int selectionStored = 0u;
  unsigned int selectionOffset = 0u;
  unsigned int selectionCursor = 0u;
};

/** @brief Producer-declared data availability and residency for the future time-window contract. */
enum KFParticleGpuTimeDomainFlags : unsigned int
{
  KFParticleGpuTimeDomainHasTrackParameters = 1u << 0u,
  KFParticleGpuTimeDomainHasCovariance = 1u << 1u,
  KFParticleGpuTimeDomainHasTime = 1u << 2u,
  KFParticleGpuTimeDomainHasSourceIdentity = 1u << 3u,
  KFParticleGpuTimeDomainHasFirstState = 1u << 4u,
  KFParticleGpuTimeDomainHasLastState = 1u << 5u,
  KFParticleGpuTimeDomainHasField = 1u << 6u,
  KFParticleGpuTimeDomainHasPid = 1u << 7u,
  KFParticleGpuTimeDomainHasPrimaryVertices = 1u << 8u,
  KFParticleGpuTimeDomainHasLeftHalo = 1u << 9u,
  KFParticleGpuTimeDomainHasRightHalo = 1u << 10u,
  KFParticleGpuTimeDomainDeviceResident = 1u << 11u
};

/** @brief Version tag for the flat time-window metadata layout. */
constexpr unsigned int KFParticleGpuTimeDomainSchemaVersion = 1u;

/**
 * @brief Reserved CA-to-KFParticle time-window metadata contract.
 *
 * Event batches carry no instances of this type. Ranges use the same absolute
 * track storage as the owning domain; core tracks are owned by this window,
 * while halo tracks are read-only context owned by an adjacent window. The
 * producer tokens describe storage lifetime and queue dependencies without
 * transferring ownership to KFParticle.
 */
struct KFParticleGpuTimeDomainDesc
{
  unsigned int schemaVersion = KFParticleGpuTimeDomainSchemaVersion;
  unsigned int flags = 0u;
  unsigned int domainIndex = 0u;
  unsigned int windowIndex = 0u;
  unsigned int chunkIndex = 0u;
  unsigned int chunkCount = 0u;
  unsigned int producerDeviceIndex = 0u;
  unsigned int producerQueueIndex = 0u;
  unsigned long long timesliceId = 0u;
  unsigned long long storageLifetimeToken = 0u;
  unsigned long long dependencyToken = 0u;
  KFParticleGpuRange coreTracks;
  KFParticleGpuRange leftHaloTracks;
  KFParticleGpuRange rightHaloTracks;
  float coreStartTime = 0.f;
  float coreEndTime = 0.f;
  float haloStartTime = 0.f;
  float haloEndTime = 0.f;
};

static_assert(std::is_trivially_copyable<KFParticleGpuTimeDomainDesc>::value,
              "KFParticle GPU time-domain ABI must remain trivially copyable");

/**
 * @brief Non-owning host/device view of transaction tables and mutable status storage.
 *
 * Accessors neither validate pointers nor synchronize. Storage must outlive the view
 * and be accessible in the calling address space. Const view objects still expose
 * mutable tables. A default-initialized view must be bound before use; the default
 * constructor deliberately does not initialize its pointer/capacity members.
 */
class KFParticleGpuTransactionView
{
 public:
  /** @brief Preserves trivial default construction without binding storage. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTransactionView() = default;

  /**
   * @brief Binds caller-owned descriptor/status arrays and their element capacities.
   * @note workRangeCapacity also bounds workStatuses. Channel and time-domain tables
   * are optional; no allocation, copying, or consistency checks occur here.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTransactionView(
    KFParticleGpuTransactionDesc* transaction,
    KFParticleGpuDomainDesc* domains,
    KFParticleGpuWorkRange* workRanges,
    KFParticleGpuWorkStatus* workStatuses,
    KFParticleGpuDomainStatus* domainStatuses,
    unsigned int domainCapacity,
    unsigned int workRangeCapacity,
    unsigned int statusCapacity,
    KFParticleGpuDomainChannelStatus* domainChannelStatuses = nullptr,
    unsigned int domainChannelStatusCapacity = 0u,
    const KFParticleGpuTimeDomainDesc* timeDomains = nullptr,
    unsigned int timeDomainCapacity = 0u)
    : fTransaction(transaction)
    , fDomains(domains)
    , fWorkRanges(workRanges)
    , fWorkStatuses(workStatuses)
    , fDomainStatuses(domainStatuses)
    , fDomainChannelStatuses(domainChannelStatuses)
    , fTimeDomains(timeDomains)
    , fDomainCapacity(domainCapacity)
    , fWorkRangeCapacity(workRangeCapacity)
    , fStatusCapacity(statusCapacity)
    , fDomainChannelStatusCapacity(domainChannelStatusCapacity)
    , fTimeDomainCapacity(timeDomainCapacity)
  {
  }

  /** @brief Returns the single batch descriptor pointer. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTransactionDesc* Transaction() const
  {
    return fTransaction;
  }
  /** @brief Returns the absolute-indexed domain table. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDomainDesc* Domains() const { return fDomains; }

  /** @brief Returns the absolute-indexed work table. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuWorkRange* WorkRanges() const { return fWorkRanges; }

  /** @brief Returns mutable status records indexed like the work table. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuWorkStatus* WorkStatuses() const { return fWorkStatuses; }

  /** @brief Returns domain status storage addressed by each domain's statusIndex. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDomainStatus* DomainStatuses() const { return fDomainStatuses; }

  /** @brief Returns optional per-domain channel counters and selection segments. */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDomainChannelStatus* DomainChannelStatuses() const
  {
    return fDomainChannelStatuses;
  }
  /** @brief Returns optional future time-window metadata without enabling that execution mode. */
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTimeDomainDesc* TimeDomains() const { return fTimeDomains; }

  /** @brief Returns the number of allocated domain slots. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DomainCapacity() const { return fDomainCapacity; }

  /** @brief Returns the common work-table and work-status capacity. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int WorkRangeCapacity() const { return fWorkRangeCapacity; }

  /** @brief Returns the domain-status capacity. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int StatusCapacity() const { return fStatusCapacity; }

  /** @brief Returns the optional channel-status capacity in records. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DomainChannelStatusCapacity() const
  {
    return fDomainChannelStatusCapacity;
  }
  /** @brief Returns the optional time-window table capacity. */
  KFPARTICLE_GPU_HOST_DEVICE unsigned int TimeDomainCapacity() const { return fTimeDomainCapacity; }

 private:
  KFParticleGpuTransactionDesc* fTransaction; ///< Borrowed batch descriptor.
  KFParticleGpuDomainDesc* fDomains; ///< Borrowed domain table.
  KFParticleGpuWorkRange* fWorkRanges; ///< Borrowed work table.
  KFParticleGpuWorkStatus* fWorkStatuses; ///< Borrowed mutable work statuses.
  KFParticleGpuDomainStatus* fDomainStatuses; ///< Borrowed mutable domain statuses.
  KFParticleGpuDomainChannelStatus* fDomainChannelStatuses; ///< Optional mutable channel statuses.
  const KFParticleGpuTimeDomainDesc* fTimeDomains; ///< Optional future time-window metadata.
  unsigned int fDomainCapacity; ///< Domain-table element capacity.
  unsigned int fWorkRangeCapacity; ///< Work-table and work-status element capacity.
  unsigned int fStatusCapacity; ///< Domain-status element capacity.
  unsigned int fDomainChannelStatusCapacity; ///< Channel-status element capacity.
  unsigned int fTimeDomainCapacity; ///< Time-window metadata capacity.
};

/** @brief Checks a half-open range against capacity using overflow-safe subtraction. */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuRangeFits(
  const KFParticleGpuRange& range, unsigned int capacity)
{
  return range.offset <= capacity && range.size <= capacity - range.offset;
}

/**
 * @brief Tests intersection of two nonempty half-open ranges; touching ends do not overlap.
 * @pre Both offset-plus-size sums must be representable.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuRangesOverlap(
  const KFParticleGpuRange& first, const KFParticleGpuRange& second)
{
  return first.size != 0u && second.size != 0u
         && first.offset < second.offset + second.size
         && second.offset < first.offset + first.size;
}

/**
 * @brief Tests full containment, allowing an empty inner range at the outer end.
 * @pre The outer offset-plus-size sum must be representable.
 */
KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuRangeContains(
  const KFParticleGpuRange& outer, const KFParticleGpuRange& inner)
{
  const unsigned int outerEnd = outer.offset + outer.size;
  return inner.offset >= outer.offset && inner.offset <= outerEnd
         && inner.size <= outerEnd - inner.offset;
}

/** @brief Host-only structural and inter-domain overlap validation helpers. */
namespace kfparticle_gpu_transaction_detail
{
  /**
   * @brief Checks required tables, plan revision, bounds, and work-to-domain associations.
   *
   * Accepts event domains only. Two-track sources must lie inside their domain's
   * track range. Does not inspect numerical track data or check inter-domain overlap.
   * @pre View pointers refer to host-accessible storage of their declared capacities.
   */
  inline KFParticleGpuTransactionValidation ValidateStructure(
  const KFParticleGpuTransactionView& view,
  unsigned long long expectedPlanRevision,
  unsigned int trackCount,
  unsigned int vertexCount,
  unsigned int eventCount,
  unsigned int taskCapacity,
  unsigned int candidateCapacity,
  unsigned int daughterCapacity,
  unsigned int selectedCapacity)
  {
  const auto* transaction = view.Transaction();
  if (transaction == nullptr || view.Domains() == nullptr || view.WorkRanges() == nullptr
      || view.WorkStatuses() == nullptr
      || view.DomainStatuses() == nullptr) {
    return KFParticleGpuTransactionMissingStorage;
  }
  if (transaction->decayPlanRevision != expectedPlanRevision) {
    return KFParticleGpuTransactionPlanRevisionMismatch;
  }
  if (!KFParticleGpuRangeFits(transaction->domains, view.DomainCapacity())
      || !KFParticleGpuRangeFits(transaction->workRanges, view.WorkRangeCapacity())) {
    return KFParticleGpuTransactionRangeOverflow;
  }
  if (transaction->channelCount != 0u
      && (view.DomainChannelStatuses() == nullptr
          || transaction->domains.size > view.DomainChannelStatusCapacity()
                                                / transaction->channelCount)) {
    return KFParticleGpuTransactionMissingStorage;
  }

  for (unsigned int i = 0u; i < transaction->domains.size; ++i) {
    const auto& domain = view.Domains()[transaction->domains.offset + i];
    if (domain.kind != KFParticleGpuEventDomain || domain.inputDescriptorIndex >= eventCount
        || domain.statusIndex >= view.StatusCapacity()
        || !KFParticleGpuRangeFits(domain.sourceTracks, trackCount)
        || !KFParticleGpuRangeFits(domain.primaryVertices, vertexCount)
        || !KFParticleGpuRangeFits(domain.workRanges, view.WorkRangeCapacity())
        || !KFParticleGpuRangeContains(transaction->workRanges, domain.workRanges)
        || !KFParticleGpuRangeFits(domain.taskReservation, taskCapacity)
        || !KFParticleGpuRangeFits(domain.candidateReservation, candidateCapacity)
        || !KFParticleGpuRangeFits(domain.daughterReservation, daughterCapacity)
        || !KFParticleGpuRangeFits(domain.selectedReservation, selectedCapacity)) {
      return KFParticleGpuTransactionInvalidDomain;
    }
    if (transaction->channelCount != 0u
        && (!KFParticleGpuRangeFits(domain.channelStatuses,
                                    view.DomainChannelStatusCapacity())
            || domain.channelStatuses.size != transaction->channelCount)) {
      return KFParticleGpuTransactionInvalidDomain;
    }
    for (unsigned int workIndex = 0u; workIndex < domain.workRanges.size; ++workIndex) {
      const auto& work = view.WorkRanges()[domain.workRanges.offset + workIndex];
      if (work.domainIndex != transaction->domains.offset + i
          || !KFParticleGpuRangeFits(work.sourceTracks, trackCount)
          || !KFParticleGpuRangeFits(work.secondarySourceTracks, trackCount)
          || !KFParticleGpuRangeFits(work.sourceCandidates, candidateCapacity)
          || !KFParticleGpuRangeFits(work.primaryVertices, vertexCount)
          || !KFParticleGpuRangeFits(work.outputReservation, taskCapacity)) {
        return KFParticleGpuTransactionInvalidDomain;
      }
      if (work.family == KFParticleGpuTwoDaughterWork
          && (!KFParticleGpuRangeContains(domain.sourceTracks, work.sourceTracks)
              || !KFParticleGpuRangeContains(domain.sourceTracks,
                                              work.secondarySourceTracks))) {
        return KFParticleGpuTransactionInvalidDomain;
      }
    }
  }
  return KFParticleGpuTransactionValid;
  }

  /**
   * @brief Checks one range family in linear time when ordered, otherwise sorts a temporary copy.
   * @param domains Host domain table, already structurally validated.
   * @param domainCount Number of domains to inspect.
   * @param rangeOf Callable returning the range to compare for each domain.
   * @param stats Optional cumulative counters for this check.
   * @note Ignores empty ranges. The fallback costs O(n log n) time and O(n) storage;
   * domain order is never modified. Range ends must be representable.
   */
  template<typename Accessor>
  inline bool HasOverlapOptimized(const KFParticleGpuDomainDesc* domains,
                                  unsigned int domainCount,
                                  Accessor rangeOf,
                                  KFParticleGpuTransactionValidationStats* stats)
  {
    if (stats) { ++stats->rangeFamilies; }
    bool ordered = true;
    bool havePrevious = false;
    unsigned int previousOffset = 0u;
    for (unsigned int index = 0u; index < domainCount; ++index) {
      const auto range = rangeOf(domains[index]);
      if (range.size == 0u) { continue; }
      if (havePrevious && range.offset < previousOffset) {
        ordered = false;
        break;
      }
      previousOffset = range.offset;
      havePrevious = true;
    }

    std::vector<KFParticleGpuRange> sorted;
    if (!ordered) {
      if (stats) { ++stats->sortedFallbackFamilies; }
      sorted.reserve(domainCount);
      for (unsigned int index = 0u; index < domainCount; ++index) {
        const auto range = rangeOf(domains[index]);
        if (range.size != 0u) { sorted.push_back(range); }
      }
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& first, const auto& second) {
                  return first.offset < second.offset
                         || (first.offset == second.offset
                             && first.size < second.size);
                });
    }

    const unsigned int count = ordered ? domainCount
                                       : static_cast<unsigned int>(sorted.size());
    havePrevious = false;
    unsigned int previousEnd = 0u;
    for (unsigned int index = 0u; index < count; ++index) {
      const auto range = ordered ? rangeOf(domains[index]) : sorted[index];
      if (range.size == 0u) { continue; }
      if (havePrevious) {
        if (stats) { ++stats->adjacentRangeComparisons; }
        if (range.offset < previousEnd) { return true; }
      }
      previousEnd = range.offset + range.size;
      havePrevious = true;
    }
    return false;
  }

  /** @brief Quadratic, allocation-free reference check of every domain pair for one range family. */
  template<typename Accessor>
  inline bool HasOverlapReference(const KFParticleGpuDomainDesc* domains,
                                  unsigned int domainCount,
                                  Accessor rangeOf,
                                  KFParticleGpuTransactionValidationStats* stats)
  {
    if (stats) { ++stats->rangeFamilies; }
    for (unsigned int first = 0u; first < domainCount; ++first) {
      for (unsigned int second = first + 1u; second < domainCount; ++second) {
        if (stats) { ++stats->domainPairComparisons; }
        if (KFParticleGpuRangesOverlap(rangeOf(domains[first]),
                                       rangeOf(domains[second]))) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief Checks track, vertex, task, candidate, daughter, and selected range families.
   * @note Stops at the first overlap; work/status table aliasing is not tested here.
   */
  template<typename OverlapCheck>
  inline bool HasAnyOverlap(const KFParticleGpuDomainDesc* domains,
                            unsigned int domainCount,
                            OverlapCheck overlap)
  {
    return overlap(domains, domainCount,
                   [](const auto& domain) { return domain.sourceTracks; })
           || overlap(domains, domainCount,
                      [](const auto& domain) { return domain.primaryVertices; })
           || overlap(domains, domainCount,
                      [](const auto& domain) { return domain.taskReservation; })
           || overlap(domains, domainCount,
                      [](const auto& domain) { return domain.candidateReservation; })
           || overlap(domains, domainCount,
                      [](const auto& domain) { return domain.daughterReservation; })
           || overlap(domains, domainCount,
                      [](const auto& domain) { return domain.selectedReservation; });
  }
}

/**
 * @brief Validates host transaction structure and rejects overlapping domain storage.
 *
 * Input counts and output capacities bound the corresponding absolute ranges.
 * Overlap checking has an allocation-free O(domains) path for ordered ranges and
 * an O(domains log domains) sorting fallback. Structural work-table checks are
 * additional. This validates storage ranges, not physical track identities.
 * @return First structural error, overlap error, or KFParticleGpuTransactionValid.
 * @note Optional stats accumulate; all tables must be host-accessible.
 */
inline KFParticleGpuTransactionValidation KFParticleGpuValidateTransaction(
  const KFParticleGpuTransactionView& view,
  unsigned long long expectedPlanRevision,
  unsigned int trackCount,
  unsigned int vertexCount,
  unsigned int eventCount,
  unsigned int taskCapacity,
  unsigned int candidateCapacity,
  unsigned int daughterCapacity,
  unsigned int selectedCapacity,
  KFParticleGpuTransactionValidationStats* stats = nullptr)
{
  const auto structural = kfparticle_gpu_transaction_detail::ValidateStructure(
    view, expectedPlanRevision, trackCount, vertexCount, eventCount,
    taskCapacity, candidateCapacity, daughterCapacity, selectedCapacity);
  if (structural != KFParticleGpuTransactionValid) { return structural; }
  const auto* transaction = view.Transaction();
  const auto* domains = view.Domains() + transaction->domains.offset;
  const auto overlap = [stats](const auto* values, unsigned int count, auto rangeOf) {
    return kfparticle_gpu_transaction_detail::HasOverlapOptimized(
      values, count, rangeOf, stats);
  };
  return kfparticle_gpu_transaction_detail::HasAnyOverlap(
           domains, transaction->domains.size, overlap)
           ? KFParticleGpuTransactionOverlappingOutput
           : KFParticleGpuTransactionValid;
}

/**
 * @brief Reference validator with identical structural checks and quadratic overlap checks.
 * @see KFParticleGpuValidateTransaction
 * @note Intended for tests and explicit performance comparisons, not the default launch path.
 */
inline KFParticleGpuTransactionValidation KFParticleGpuValidateTransactionReference(
  const KFParticleGpuTransactionView& view,
  unsigned long long expectedPlanRevision,
  unsigned int trackCount,
  unsigned int vertexCount,
  unsigned int eventCount,
  unsigned int taskCapacity,
  unsigned int candidateCapacity,
  unsigned int daughterCapacity,
  unsigned int selectedCapacity,
  KFParticleGpuTransactionValidationStats* stats = nullptr)
{
  const auto structural = kfparticle_gpu_transaction_detail::ValidateStructure(
    view, expectedPlanRevision, trackCount, vertexCount, eventCount,
    taskCapacity, candidateCapacity, daughterCapacity, selectedCapacity);
  if (structural != KFParticleGpuTransactionValid) { return structural; }
  const auto* transaction = view.Transaction();
  const auto* domains = view.Domains() + transaction->domains.offset;
  const auto overlap = [stats](const auto* values, unsigned int count, auto rangeOf) {
    return kfparticle_gpu_transaction_detail::HasOverlapReference(
      values, count, rangeOf, stats);
  };
  return kfparticle_gpu_transaction_detail::HasAnyOverlap(
           domains, transaction->domains.size, overlap)
           ? KFParticleGpuTransactionOverlappingOutput
           : KFParticleGpuTransactionValid;
}

/** @brief Tests whether KFPARTICLE_GPU_TRANSACTION_VALIDATION_REFERENCE is exactly "1". */
inline bool KFParticleGpuReferenceTransactionValidationRequested()
{
  const char* value = std::getenv("KFPARTICLE_GPU_TRANSACTION_VALIDATION_REFERENCE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

/**
 * @brief Selects the optimized validator unless the environment explicitly requests the reference.
 * @see KFParticleGpuValidateTransaction
 */
inline KFParticleGpuTransactionValidation KFParticleGpuValidateTransactionConfigured(
  const KFParticleGpuTransactionView& view,
  unsigned long long expectedPlanRevision,
  unsigned int trackCount,
  unsigned int vertexCount,
  unsigned int eventCount,
  unsigned int taskCapacity,
  unsigned int candidateCapacity,
  unsigned int daughterCapacity,
  unsigned int selectedCapacity)
{
  return KFParticleGpuReferenceTransactionValidationRequested()
           ? KFParticleGpuValidateTransactionReference(
               view, expectedPlanRevision, trackCount, vertexCount, eventCount,
               taskCapacity, candidateCapacity, daughterCapacity, selectedCapacity)
           : KFParticleGpuValidateTransaction(
               view, expectedPlanRevision, trackCount, vertexCount, eventCount,
               taskCapacity, candidateCapacity, daughterCapacity, selectedCapacity);
}

static_assert(std::is_trivially_copyable<KFParticleGpuTransactionDesc>::value,
              "KFParticle GPU transaction descriptor must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuDomainDesc>::value,
              "KFParticle GPU domain descriptor must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuWorkRange>::value,
              "KFParticle GPU work range must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuPairTile>::value,
              "KFParticle GPU pair tile must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuWorkStatus>::value,
              "KFParticle GPU work status must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuDomainStatus>::value,
              "KFParticle GPU domain status must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuDomainChannelStatus>::value,
              "KFParticle GPU domain-channel status must remain a flat device ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuTransactionView>::value,
              "KFParticle GPU transaction view must remain a flat device ABI");

#endif
