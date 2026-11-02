/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "planning/KFParticleGpuBatch.h"

#include "planning/KFParticleGpuDecayPlan.h"
#include "planning/KFParticleGpuCpuChannelCatalogue.h"
#include "algorithms/KFParticleGpuV0Track.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
  using kfparticle::gpu::GpuPreparedEvent;

  unsigned int CheckedAdd(unsigned int left, unsigned int right, const char* what)
  {
    if (right > std::numeric_limits<unsigned int>::max() - left) { throw std::overflow_error(what); }
    return left + right;
  }

  KFParticleGpuEventDesc OffsetDescriptor(KFParticleGpuEventDesc descriptor,
                                          unsigned int trackOffset,
                                          unsigned int vertexOffset)
  {
    for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
      auto& trackSet = descriptor.trackSets[set];
      trackSet.tracks.offset += trackOffset;
      for (unsigned int species = 0u; species < NumberOfTrackSpecies; ++species) {
        trackSet.species[species].offset += trackOffset;
      }
    }
    descriptor.primaryVertices.offset += vertexOffset;
    return descriptor;
  }

  bool Fits(unsigned int used, unsigned int added, unsigned int limit)
  {
    return added <= limit && used <= limit - added;
  }

}  // namespace

namespace kfparticle::gpu
{
  struct GpuBatchBuilder::CapacityPlan {
    KFParticleGpuDecayPlan value;
    std::unique_ptr<KFParticleGpuCpuFinderCapacityPlanner> planner;
  };

  GpuBatchLimits DefaultGpuBatchLimits(unsigned int events)
  {
    constexpr unsigned int baselineEvents = 8u;
    constexpr unsigned int baselineCandidates = 131072u;
    constexpr unsigned int maximumLineage =
      KFParticleGpuV0TrackLineage::MaximumSize;
    static_assert(maximumLineage >= 2u,
                  "KFParticle GPU lineage storage must cover two-daughter candidates");
    static_assert(baselineCandidates
                    <= std::numeric_limits<unsigned int>::max() / maximumLineage,
                  "KFParticle GPU default daughter admission baseline overflows");

    const unsigned int resourceScale =
      events / baselineEvents
      + static_cast<unsigned int>(events % baselineEvents != 0u);
    const auto scaled = [resourceScale](unsigned int baseline) {
      const auto value = static_cast<std::uint64_t>(baseline) * resourceScale;
      return static_cast<unsigned int>(
        std::min<std::uint64_t>(value,
                                std::numeric_limits<unsigned int>::max()));
    };

    GpuBatchLimits limits;
    limits.events = events;
    limits.tracks = scaled(16384u);
    limits.vertices = scaled(64u);
    limits.twoDaughterTasks = scaled(baselineCandidates);
    limits.candidates = scaled(baselineCandidates);
    // The complete decay graph can publish a lineage of up to 16 source
    // tracks. A legacy two-daughter ratio split otherwise valid large batches
    // before the exact adaptive byte guard could evaluate them.
    limits.daughterIds = scaled(baselineCandidates * maximumLineage);
    limits.selectedCandidates = scaled(baselineCandidates);
    return limits;
  }

  GpuPreparedBatch::GpuPreparedBatch(std::vector<Entry> entries,
                                                         KFParticleGpuBufferCapacities capacities,
                                                         std::vector<int> requestedMotherPdgs)
    : fEntries(std::move(entries)),
      fCapacities(capacities),
      fRequestedMotherPdgs(std::move(requestedMotherPdgs))
  {
  }

  GpuPreparedBatch::GpuPreparedBatch(GpuPreparedBatch&&) noexcept = default;

  GpuPreparedBatch& GpuPreparedBatch::operator=(GpuPreparedBatch&&) noexcept = default;

  GpuPreparedBatch::~GpuPreparedBatch() = default;

  unsigned int GpuPreparedBatch::EventCount() const { return static_cast<unsigned int>(fEntries.size()); }

  unsigned int GpuPreparedBatch::TrackCount() const { return fCapacities.tracks; }

  unsigned int GpuPreparedBatch::VertexCount() const { return fCapacities.vertices; }

  unsigned int GpuPreparedBatch::TaskCapacity() const { return fCapacities.twoDaughterTasks; }

  const KFParticleGpuBufferCapacities& GpuPreparedBatch::Capacities() const { return fCapacities; }

  KFParticleGpuBufferCapacities GpuPreparedBatch::RoutingCapacities() const
  {
    KFParticleGpuBufferCapacities capacities = fCapacities;
    // Counters and views require allocated storage, but routing itself does not
    // write candidate payload. Heavy output is sized from retained tasks.
    capacities.candidates = 1u;
    capacities.daughterIds = 2u;
    capacities.selectedCandidates = 1u;
    // TaskCapacity() remains the logical bound; batch kernels use routed tasks.
    // The legacy explicit-task pool is allocated only by its own entry points.
    capacities.twoDaughterTasks = 0u;
    return capacities;
  }

  const GpuBatchEvent& GpuPreparedBatch::Event(unsigned int index) const
  {
    return fEntries.at(index).event;
  }

  const std::vector<int>& GpuPreparedBatch::RequestedMotherPdgs() const
  {
    return fRequestedMotherPdgs;
  }

  unsigned int GpuPreparedBatch::Write(
    const KFParticleGpuInputTrackSoAView& tracks,
    const KFParticleGpuVertexSoAView& vertices,
    KFParticleGpuEventDesc* events,
    unsigned int packingThreads) const
  {
    if (tracks.Size() != TrackCount() || vertices.Size() != VertexCount()
        || tracks.Stride() < TrackCount()
        || vertices.Stride() < VertexCount()) {
      throw std::invalid_argument("KFParticle GPU batch received views with unexpected logical sizes");
    }
    if (!fEntries.empty() && events == nullptr) {
      throw std::invalid_argument("KFParticle GPU batch requires allocated event descriptors");
    }
    if (packingThreads == 0u
        || packingThreads > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
        "KFParticle packing worker limit must fit a positive int");
    }
    if (EventCount()
        > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
        "KFParticle packing event count exceeds the OpenMP loop index");
    }

    unsigned int workers = std::max(1u, std::min(packingThreads, EventCount()));
#ifndef _OPENMP
    workers = 1u;
#endif
    std::vector<std::exception_ptr> failures(EventCount());
    std::vector<unsigned char> workerUsed(workers, 0u);
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) schedule(static)
#endif
    for (int index = 0; index < static_cast<int>(EventCount()); ++index) {
      const auto& entry = fEntries[static_cast<unsigned int>(index)];
#ifdef _OPENMP
      const unsigned int workerIndex =
        static_cast<unsigned int>(omp_get_thread_num());
#else
      const unsigned int workerIndex = 0u;
#endif
      workerUsed[workerIndex] = 1u;
      try {
        GpuInputPacker::Write(entry.prepared,
                                   entry.event.trackOffset,
                                   entry.event.vertexOffset,
                                   tracks,
                                   vertices,
                                   events[entry.event.eventIndex]);
      }
      catch (...) {
        failures[static_cast<unsigned int>(index)] = std::current_exception();
      }
    }
    for (const auto& failure : failures) {
      if (failure) { std::rethrow_exception(failure); }
    }
    return std::max(
      1u, static_cast<unsigned int>(std::count(
            workerUsed.begin(), workerUsed.end(), static_cast<unsigned char>(1u))));
  }

  void GpuPreparedBatch::WriteTransaction(
    const KFParticleGpuTransactionView& transaction,
    unsigned long long decayPlanRevision,
    unsigned long long transactionId,
    bool compactOutput) const
  {
    if (transaction.Transaction() == nullptr || transaction.Domains() == nullptr
        || transaction.WorkRanges() == nullptr || transaction.DomainStatuses() == nullptr
        || transaction.DomainCapacity() != EventCount()
        || transaction.WorkRangeCapacity() != EventCount()
        || transaction.StatusCapacity() != EventCount()) {
      throw std::invalid_argument("KFParticle GPU batch received an invalid transaction view");
    }

    auto* descriptor = transaction.Transaction();
    auto* domains = transaction.Domains();
    auto* workRanges = transaction.WorkRanges();
    descriptor[0] = KFParticleGpuTransactionDesc{};
    descriptor[0].transactionId = transactionId;
    descriptor[0].decayPlanRevision = decayPlanRevision;
    descriptor[0].domains = {0u, EventCount()};
    descriptor[0].workRanges = {0u, EventCount()};

    for (unsigned int i = 0u; i < EventCount(); ++i) {
      const auto& event = fEntries[i].event;
      auto& domain = domains[i];
      domain = KFParticleGpuDomainDesc{};
      domain.sourceId = event.sourceEventId;
      domain.kind = KFParticleGpuEventDomain;
      domain.inputDescriptorIndex = event.eventIndex;
      unsigned int trackCount = 0u;
      for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
        trackCount += event.descriptor.trackSets[set].tracks.size;
      }
      domain.sourceTracks = {event.trackOffset, trackCount};
      domain.primaryVertices = {event.vertexOffset,
                                event.descriptor.primaryVertices.size};
      domain.workRanges = {i, 1u};
      domain.taskReservation = event.taskReservation;
      domain.candidateReservation = compactOutput
                                      ? KFParticleGpuRange{}
                                      : event.candidateReservation;
      domain.daughterReservation = compactOutput
                                     ? KFParticleGpuRange{}
                                     : event.daughterReservation;
      domain.selectedReservation = compactOutput
                                     ? KFParticleGpuRange{}
                                     : event.selectedReservation;
      domain.statusIndex = i;
      domain.flags = compactOutput ? KFParticleGpuDomainCompactOutput : 0u;

      auto& work = workRanges[i];
      work = KFParticleGpuWorkRange{};
      work.domainIndex = i;
      work.family = KFParticleGpuDomainInputWork;
      work.generation = 0u;
      work.executionGroup = 0u;
      work.sourceTracks = domain.sourceTracks;
      work.primaryVertices = domain.primaryVertices;
      work.outputReservation = domain.taskReservation;
      transaction.WorkStatuses()[i] = KFParticleGpuWorkStatus{};
      transaction.DomainStatuses()[i] = KFParticleGpuDomainStatus{};
    }
  }

  GpuBatchBuilder::GpuBatchBuilder(GpuBatchLimits limits,
                                                       std::vector<int> requestedMotherPdgs)
    : fLimits(limits), fRequestedMotherPdgs(std::move(requestedMotherPdgs))
  {
    fCapacities.nonhomogeneousField = true;
  }

  GpuBatchBuilder::~GpuBatchBuilder() = default;

  GpuBatchAppendResult GpuBatchBuilder::TryAppend(
    std::uint64_t sourceEventId,
    const GpuEventInput& input,
    GpuBatchAppendPolicy policy)
  {
    if (fSealed) { throw std::logic_error("KFParticle GPU batch builder is already sealed"); }

    const auto elapsedMilliseconds = [](std::chrono::steady_clock::time_point started) {
      return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    };
    const auto preparationStarted = std::chrono::steady_clock::now();
    auto prepared = GpuInputPacker::Prepare(input);
    const double preparationMilliseconds = elapsedMilliseconds(preparationStarted);
    if (!fCapacityPlan) {
      const auto planStarted = std::chrono::steady_clock::now();
      auto capacityPlan = std::make_unique<CapacityPlan>();
      if (fRequestedMotherPdgs.empty()) {
        AddCompleteCpuFinderChannels(capacityPlan->value);
      }
      else {
        AddRequestedCpuFinderChannels(capacityPlan->value, fRequestedMotherPdgs);
      }
      capacityPlan->planner =
        std::make_unique<KFParticleGpuCpuFinderCapacityPlanner>(
          capacityPlan->value);
      fCapacityPlan = std::move(capacityPlan);
      fCapacityPlanSetupMilliseconds = elapsedMilliseconds(planStarted);
    }
    const auto eventCapacityStarted = std::chrono::steady_clock::now();
    KFParticleGpuEventCapacityPlan planned;
    try {
      planned = fCapacityPlan->planner->Plan(prepared.Descriptor());
    }
    catch (const std::overflow_error&) {
      return GpuBatchAppendResult::CapacityExceeded;
    }
    const unsigned int tasks = planned.taskCapacity;
    const double eventCapacityPlanningMilliseconds =
      elapsedMilliseconds(eventCapacityStarted);
    const unsigned int candidates = planned.candidates;
    const unsigned int daughterIds = planned.daughterIds;
    const unsigned int selectedCandidates = planned.selectedCandidates;
    const bool eventFits = Fits(fCapacities.events, 1u, fLimits.events);
    const bool resourcesFit =
      Fits(fCapacities.tracks, prepared.TrackCount(), fLimits.tracks)
      && Fits(fCapacities.vertices, prepared.VertexCount(), fLimits.vertices)
      && Fits(fCapacities.twoDaughterTasks, tasks, fLimits.twoDaughterTasks)
      && Fits(fCapacities.candidates, candidates, fLimits.candidates)
      && Fits(fCapacities.daughterIds, daughterIds, fLimits.daughterIds)
      && Fits(fCapacities.selectedCandidates, selectedCandidates,
              fLimits.selectedCandidates);
    const bool allowIsolatedOversizedEvent =
      policy == GpuBatchAppendPolicy::AllowSingleEventBeyondResourceLimits
      && Empty();
    if (!eventFits || (!resourcesFit && !allowIsolatedOversizedEvent)) {
      return GpuBatchAppendResult::CapacityExceeded;
    }

    GpuBatchEvent event;
    event.sourceEventId = sourceEventId;
    event.eventIndex = fCapacities.events;
    event.trackOffset = fCapacities.tracks;
    event.vertexOffset = fCapacities.vertices;
    event.taskCapacity = tasks;
    event.taskReservation = {fCapacities.twoDaughterTasks, tasks};
    event.candidateReservation = {fCapacities.candidates, candidates};
    event.daughterReservation = {fCapacities.daughterIds, daughterIds};
    event.selectedReservation = {fCapacities.selectedCandidates,
                                 selectedCandidates};
    event.inputPreparationMilliseconds = preparationMilliseconds;
    event.capacityPlanSetupMilliseconds =
      fCapacityPlanSetupAccounted ? 0. : fCapacityPlanSetupMilliseconds;
    event.capacityPlanBuilds = fCapacityPlanSetupAccounted ? 0u : 1u;
    event.capacityPlanningMilliseconds =
      eventCapacityPlanningMilliseconds + event.capacityPlanSetupMilliseconds;
    event.descriptor = OffsetDescriptor(prepared.Descriptor(), event.trackOffset, event.vertexOffset);
    const unsigned int eventTrackEnd = CheckedAdd(
      event.trackOffset, prepared.TrackCount(),
      "KFParticle GPU batch event track range exceeds the GPU ABI");
    bool rangesBelongToEvent = prepared.HasValidatedPublicationInput();
    unsigned int nextTrackOffset = event.trackOffset;
    for (unsigned int set = 0u; set < NumberOfTrackSets; ++set) {
      const auto& range = event.descriptor.trackSets[set].tracks;
      rangesBelongToEvent = rangesBelongToEvent
                            && range.offset == nextTrackOffset
                            && range.offset <= eventTrackEnd
                            && range.size <= eventTrackEnd - range.offset;
      if (rangesBelongToEvent) { nextTrackOffset += range.size; }
    }
    rangesBelongToEvent = rangesBelongToEvent
                          && nextTrackOffset == eventTrackEnd;
    for (unsigned int set = 0u; set < 4u; ++set) {
      rangesBelongToEvent = rangesBelongToEvent
                            && event.descriptor.trackSets[set].tracks.size
                                 == event.descriptor.trackSets[set + 4u].tracks.size;
    }
    if (!rangesBelongToEvent) {
      throw std::logic_error(
        "KFParticle GPU batch produced an invalid publication-input range");
    }
    event.validatedPublicationInput = true;
    fEntries.push_back({std::move(event), std::move(prepared)});
    fCapacityPlanSetupAccounted = true;

    fCapacities.events = CheckedAdd(fCapacities.events, 1u, "KFParticle GPU batch event count exceeds the GPU ABI");
    if (fCapacities.transactions == 0u) { fCapacities.transactions = 1u; }
    fCapacities.domains = CheckedAdd(fCapacities.domains, 1u,
                                     "KFParticle GPU batch domain count exceeds the GPU ABI");
    fCapacities.workRanges = CheckedAdd(fCapacities.workRanges, 1u,
                                        "KFParticle GPU batch work-range count exceeds the GPU ABI");
    fCapacities.domainStatuses = CheckedAdd(fCapacities.domainStatuses, 1u,
                                            "KFParticle GPU batch status count exceeds the GPU ABI");
    fCapacities.tracks = CheckedAdd(fCapacities.tracks, fEntries.back().prepared.TrackCount(),
                                    "KFParticle GPU batch track count exceeds the GPU ABI");
    fCapacities.vertices = CheckedAdd(fCapacities.vertices, fEntries.back().prepared.VertexCount(),
                                      "KFParticle GPU batch vertex count exceeds the GPU ABI");
    fCapacities.twoDaughterTasks = CheckedAdd(fCapacities.twoDaughterTasks, tasks,
                                               "KFParticle GPU batch task count exceeds the GPU ABI");
    fCapacities.candidates = CheckedAdd(fCapacities.candidates, candidates,
                                        "KFParticle GPU batch candidate count exceeds the GPU ABI");
    fCapacities.daughterIds = CheckedAdd(fCapacities.daughterIds, daughterIds,
                                         "KFParticle GPU batch daughter count exceeds the GPU ABI");
    fCapacities.selectedCandidates = CheckedAdd(fCapacities.selectedCandidates,
                                                selectedCandidates,
                                                "KFParticle GPU batch selected count exceeds the GPU ABI");
    return GpuBatchAppendResult::Added;
  }

  const KFParticleGpuBufferCapacities& GpuBatchBuilder::CurrentCapacities() const
  {
    return fCapacities;
  }

  unsigned int GpuBatchBuilder::EventCount() const { return fCapacities.events; }

  bool GpuBatchBuilder::Empty() const { return fEntries.empty(); }

  GpuPreparedBatch GpuBatchBuilder::Seal()
  {
    if (fSealed) { throw std::logic_error("KFParticle GPU batch builder is already sealed"); }
    fSealed = true;
    return GpuPreparedBatch(std::move(fEntries), fCapacities,
                                      std::move(fRequestedMotherPdgs));
  }
}  // namespace kfparticle::gpu
