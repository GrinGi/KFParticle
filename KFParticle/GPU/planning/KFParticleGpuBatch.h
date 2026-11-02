/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPUBATCH_H
#define KFPARTICLEGPUBATCH_H

#include "data/KFParticleGpuInputPacker.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace kfparticle::gpu
{
  /** Hard limits for one host-prepared event batch before it touches XPU storage. */
  struct GpuBatchLimits {
    unsigned int events = 0;
    unsigned int tracks = 0;
    unsigned int vertices = 0;
    unsigned int twoDaughterTasks = 0;
    unsigned int candidates = 0;
    unsigned int daughterIds = 0;
    unsigned int selectedCandidates = 0;
  };

  /** Aggregate admission ceilings for a configured event-batch size. */
  GpuBatchLimits DefaultGpuBatchLimits(unsigned int events);

  enum class GpuBatchAppendResult { Added, CapacityExceeded };

  enum class GpuBatchAppendPolicy {
    RespectLimits,
    AllowSingleEventBeyondResourceLimits
  };

  /** Absolute input/output ranges and source identity for one sealed batch member. */
  struct GpuBatchEvent {
    std::uint64_t sourceEventId = 0;
    unsigned int eventIndex = 0;
    unsigned int trackOffset = 0;
    unsigned int vertexOffset = 0;
    unsigned int taskCapacity = 0;
    KFParticleGpuRange taskReservation;
    KFParticleGpuRange candidateReservation;
    KFParticleGpuRange daughterReservation;
    KFParticleGpuRange selectedReservation;
    double inputPreparationMilliseconds = 0.;
    double capacityPlanningMilliseconds = 0.;
    double capacityPlanSetupMilliseconds = 0.;
    unsigned int capacityPlanBuilds = 0u;
    bool validatedPublicationInput = false;
    KFParticleGpuEventDesc descriptor;
  };

  /**
   * A sealed, CPU-only collection of structurally validated packing plans.
   *
   * Source storage remains caller-owned and
   * must outlive the direct write into the XPU host views.
   */
  class GpuPreparedBatch {
   public:
    GpuPreparedBatch(GpuPreparedBatch&&) noexcept;
    GpuPreparedBatch& operator=(GpuPreparedBatch&&) noexcept;
    ~GpuPreparedBatch();

    GpuPreparedBatch(const GpuPreparedBatch&)            = delete;
    GpuPreparedBatch& operator=(const GpuPreparedBatch&) = delete;

    unsigned int EventCount() const;
    unsigned int TrackCount() const;
    unsigned int VertexCount() const;
    unsigned int TaskCapacity() const;
    const KFParticleGpuBufferCapacities& Capacities() const;
    /** Input, transaction and routing storage without speculative candidate pools. */
    KFParticleGpuBufferCapacities RoutingCapacities() const;
    const GpuBatchEvent& Event(unsigned int index) const;
    const std::vector<int>& RequestedMotherPdgs() const;

    /// Validates and writes disjoint events into host views; it performs no XPU transfer.
    unsigned int Write(const KFParticleGpuInputTrackSoAView& tracks,
                       const KFParticleGpuVertexSoAView& vertices,
                       KFParticleGpuEventDesc* events,
                       unsigned int packingThreads = 1u) const;

    /// Writes the structural transaction/domain ABI used by later flattened kernels.
    void WriteTransaction(const KFParticleGpuTransactionView& transaction,
                          unsigned long long decayPlanRevision,
                          unsigned long long transactionId = 0u,
                          bool compactOutput = false) const;

   private:
    friend class GpuBatchBuilder;

    struct Entry {
      GpuBatchEvent event;
      GpuPreparedEvent prepared;
    };

    explicit GpuPreparedBatch(std::vector<Entry> entries,
                                        KFParticleGpuBufferCapacities capacities,
                                        std::vector<int> requestedMotherPdgs);

    std::vector<Entry> fEntries;
    KFParticleGpuBufferCapacities fCapacities;
    std::vector<int> fRequestedMotherPdgs;
  };

  /** Builds one bounded batch atomically: rejected input never changes accumulated ranges. */
  class GpuBatchBuilder {
   public:
    explicit GpuBatchBuilder(GpuBatchLimits limits,
                                       std::vector<int> requestedMotherPdgs = {});
    ~GpuBatchBuilder();

    GpuBatchAppendResult TryAppend(
      std::uint64_t sourceEventId,
      const GpuEventInput& input,
      GpuBatchAppendPolicy policy = GpuBatchAppendPolicy::RespectLimits);
    const KFParticleGpuBufferCapacities& CurrentCapacities() const;
    unsigned int EventCount() const;
    bool Empty() const;

    /// Prevents further append operations and returns the immutable packing plan.
    GpuPreparedBatch Seal();

   private:
    struct CapacityPlan;

    GpuBatchLimits fLimits;
    std::vector<GpuPreparedBatch::Entry> fEntries;
    KFParticleGpuBufferCapacities fCapacities;
    std::vector<int> fRequestedMotherPdgs;
    std::unique_ptr<CapacityPlan> fCapacityPlan;
    double fCapacityPlanSetupMilliseconds = 0.;
    bool fCapacityPlanSetupAccounted = false;
    bool fSealed = false;
  };

}  // namespace kfparticle::gpu

#endif  // KFPARTICLEGPUBATCH_H
