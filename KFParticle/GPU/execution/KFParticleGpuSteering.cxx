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
 * @brief Host sequencing of input upload, event-local GPU reconstruction, and output readback.
 * Device mathematics lives in kernels/helpers; this file owns scheduling, reservations, diagnostics,
 * and host result assembly. Queue operations are ordered, but host calls are not internally locked.
 */
#include "execution/KFParticleGpuSteering.h"
#ifdef KFPARTICLE_USE_XPU
#include "diagnostics/KFParticleGpuStepTrace.h"
#endif

#include "execution/KFParticleGpuBufferManager.h"
#include "diagnostics/KFParticleGpuHostTiming.h"
#include "planning/KFParticleGpuDecayGraphPlan.h"
#include "planning/KFParticleGpuDecayPlan.h"
#include "execution/KFParticleGpuKernels.h"
#include <cstring>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>
#endif

#ifdef KFPARTICLE_USE_XPU
namespace
{
  /**
   * @brief Diagnostic build marker for transaction-reset/output-planning behavior; not a runtime version check.
   */
  constexpr const char* TransactionResetContract =
    "20260824-compact-output-planning-v19";
  /**
   * @brief Maximum base task budget eligible for the combined first-generation route/construct kernel.
   */
  constexpr unsigned int MaximumFusedTwoDaughterTasks = 1024u;

  /**
   * @brief Add two unsigned capacities, throwing std::overflow_error(message) instead of wrapping.
   */
  unsigned int CheckedAddCapacity(unsigned int left, unsigned int right,
                                  const char* message)
  {
    if (right > std::numeric_limits<unsigned int>::max() - left) {
      throw std::overflow_error(message);
    }
    return left + right;
  }

  /**
   * @brief Multiply unsigned capacity values with an overflow check; zero factors produce zero.
   */
  unsigned int CheckedMultiplyCapacity(unsigned int value, unsigned int factor,
                                       const char* message)
  {
    if (factor != 0u && value > std::numeric_limits<unsigned int>::max() / factor) {
      throw std::overflow_error(message);
    }
    return value * factor;
  }

  /**
   * @brief Retrieve XPU's host pointer for a buffer, without copying, synchronizing, or checking it for null.
   */
  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  /**
   * @brief Build a kernel-state record with explicitly empty views for safe publication during teardown.
   */
  KFParticleGpuKernels MakeEmptyKernels()
  {
    const KFParticleGpuInputTrackSoAView emptyInput(
      nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuVertexSoAView emptyVertices(nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuFitSoAView emptyFit(nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuCandidateMetadataSoAView emptyMetadata(nullptr, nullptr, 0);
    const KFParticleGpuDaughterStorageView emptyDaughters(nullptr, nullptr, 0);
    const KFParticleGpuCandidatePoolView emptyCandidates(
      emptyFit, emptyMetadata, emptyDaughters, nullptr, nullptr, 0);

    const KFParticleGpuV0SelectionResultView emptySelectionResults(nullptr, 0u);
    const KFParticleGpuSelectedCandidateIndexView emptySelected(nullptr, nullptr, nullptr, 0);
    return KFParticleGpuKernels(MakeConstView(emptyInput),
                                MakeConstView(emptyVertices),
                                nullptr,
                                nullptr,
                                0u,
                                emptyCandidates,
                                emptySelectionResults,
                                emptySelected);
  }

  /**
   * @brief Read KFPARTICLE_GPU_TRANSACTION_STAGE_FENCE; enabled for a nonempty value not starting with '0'.
   */
  bool TransactionStageFenceEnabled()
  {
    const char* value = std::getenv("KFPARTICLE_GPU_TRANSACTION_STAGE_FENCE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }

  /**
   * @brief Read KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS using the same first-character boolean convention.
   */
  bool TransactionDiagnosticsEnabled()
  {
    const char* value = std::getenv("KFPARTICLE_GPU_TRANSACTION_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }

  /**
   * @brief Map a transaction probe phase to a log label; return "unknown" for an unrecognized value.
   */
  const char* TransactionDiagnosticPhaseName(unsigned int phase)
  {
    switch (phase) {
      case KFGpuTransactionDiagnosticJournal: return "journal";
      case KFGpuTransactionDiagnosticTransaction: return "transaction";
      case KFGpuTransactionDiagnosticWork: return "work";
      case KFGpuTransactionDiagnosticDomain: return "domain";
      case KFGpuTransactionDiagnosticEventRouting: return "event-routing";
      case KFGpuTransactionDiagnosticInputScalars: return "input-scalars";
      case KFGpuTransactionDiagnosticCompatibility: return "compatibility";
      case KFGpuTransactionDiagnosticTrackCuts: return "track-cuts";
      case KFGpuTransactionDiagnosticDcaStateField: return "dca-state-field";
      case KFGpuTransactionDiagnosticDcaRoots: return "dca-roots";
      case KFGpuTransactionDiagnosticSecondTransport: return "second-transport";
      case KFGpuTransactionDiagnosticFirstTransport: return "first-transport";
      case KFGpuTransactionDiagnosticPairDecision: return "pair-decision";
      default: return "unknown";
    }
  }

  /**
   * @brief When enabled, wait for the queue, print a named boundary outcome, and rethrow backend errors.
   * Adds synchronization and stderr output, so timings with fences are not production measurements.
   */
  void TransactionStageFence(xpu::queue& queue, const char* stage,
                             unsigned int generation = 0u)
  {
    if (!TransactionStageFenceEnabled()) { return; }
    try {
      queue.wait();
      std::fprintf(stderr,
                   "KFPARTICLE_GPU_STAGE PASS stage=%s generation=%u\n",
                   stage, generation);
      std::fflush(stderr);
    }
    catch (const std::exception& error) {
      std::fprintf(stderr,
                   "KFPARTICLE_GPU_STAGE FAIL stage=%s generation=%u error=%s\n",
                   stage, generation, error.what());
      std::fflush(stderr);
      throw;
    }
  }

  /**
   * @brief Capture current device pointers, logical sizes, and capacities from the buffer manager.
   * The returned record borrows storage. Rebuild and republish it after relevant buffers or sizes change.
   */
  KFParticleGpuKernels MakeKernelState(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    return KFParticleGpuKernels(MakeConstView(buffers.DeviceInputTracks()),
                                MakeConstView(buffers.DevicePrimaryVertices()),
                                buffers.DeviceEvents(),
                                storage.fTwoDaughterTasks.get(),
                                buffers.Capacities().twoDaughterTasks,
                                buffers.DeviceCandidates(),
                                buffers.DeviceV0SelectionResults(),
                                buffers.DeviceSelectedCandidates(),
                                buffers.DeviceV0TrackRouting(),
                                buffers.DeviceTwoDaughterRouting(),
                                buffers.DeviceCandidateDescriptorIndices(),
                                buffers.DeviceDecayGraph(),
                                buffers.DeviceGraphOperationStorage(),
                                buffers.DeviceTwoDaughterGenerationStorage(),
                                buffers.DeviceV0TrackGenerationStorage(),
                                buffers.DeviceTransaction());
  }

  /**
   * @brief Return the Cartesian product of the two resolved event-local source-range sizes.
   * No multiplication overflow check or same-track/physics rejection is performed here.
   */
  unsigned int CountTwoDaughterPairs(const KFParticleGpuEventDesc& event,
                                     const KFParticleGpuTwoDaughterTaskSource& source)
  {
    const KFParticleGpuRange firstRange =
      ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
    const KFParticleGpuRange secondRange =
      ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
    return firstRange.size * secondRange.size;
  }

  /**
   * @brief Count supported graph nodes carrying the requested payload kind; unsupported nodes are excluded.
   */
  unsigned int CountGraphPayload(const KFParticleGpuDecayGraphPlan& graph,
                                 unsigned int payloadKind)
  {
    unsigned int count = 0u;
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      if (node.supportStatus == KFGpuGraphSupported
          && node.payloadKind == payloadKind) {
        ++count;
      }
    }
    return count;
  }

  /**
   * @brief Host scratch counters and pool boundaries for composite-track work in one event.
   * Per-channel vectors are populated only when detailed snapshots are retained.
   */
  struct KFParticleGpuFusedCascadeSnapshot
  {
    unsigned int groupLaunches = 0u; ///< Logical event/group sequences represented by this snapshot.
    unsigned int visitedPairs = 0u; ///< Reported visited source pairs.
    unsigned int activeChannelBits = 0u; ///< Reported sum of active channel bits.
    unsigned int acceptedTasks = 0u; ///< Tasks accepted before capacity truncation.
    unsigned int storedTasks = 0u; ///< Tasks retained for construction.
    unsigned int blockReservations = 0u; ///< Reported block-level reservations.
    unsigned int taskOverflowFlags = 0u; ///< Routing/task overflow mask.
    unsigned int candidateBegin = 0u; ///< Raw candidate start offset.
    /// Raw candidate end; temporarily a count during transaction first-generation assembly.
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u; ///< Flattened source-ancestry start offset.
    /// Source-ancestry end; temporarily a count during transaction first-generation assembly.
    unsigned int daughterEnd = 0u;
    unsigned int candidateOverflowFlags = 0u; ///< Candidate-pool/domain overflow flags.
    std::vector<unsigned int> channelVisited; ///< Optional per-descriptor visited counts.
    std::vector<unsigned int> channelAccepted; ///< Optional per-descriptor accepted-task counts.
    std::vector<unsigned int> channelStored; ///< Optional per-descriptor retained-task counts.
    std::vector<unsigned int> channelConstructed; ///< Optional per-descriptor construction counts.
  };

  /**
   * @brief Host scratch for first-generation routing/selection counters and event-local pool ranges.
   * In transaction assembly, End fields temporarily hold counts and are later rebased to absolute ends.
   * Per-channel vectors are optional; queued reads must finish before their contents are consumed.
   */
  struct KFParticleGpuFusedTwoDaughterSnapshot
  {
    unsigned int groupLaunches = 0u; ///< Logical event/group sequences represented by this snapshot.
    unsigned int visitedPairs = 0u; ///< Reported visited source pairs.
    unsigned int activeChannelBits = 0u; ///< Reported sum of active channel bits.
    unsigned int acceptedTasks = 0u; ///< Tasks accepted before capacity truncation.
    unsigned int storedTasks = 0u; ///< Tasks retained for construction.
    unsigned int blockReservations = 0u; ///< Reported block-level reservations.
    unsigned int taskOverflowFlags = 0u; ///< Routing/task overflow mask.
    unsigned int candidateBegin = 0u; ///< Raw candidate start offset.
    /// Raw candidate end; temporarily a count during transaction first-generation assembly.
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u; ///< Flattened source-ancestry start offset.
    /// Source-ancestry end; temporarily a count during transaction first-generation assembly.
    unsigned int daughterEnd = 0u;
    unsigned int selectedBegin = 0u; ///< First-generation compact selected start offset.
    unsigned int selectedEnd = 0u; ///< Compact selected end; temporarily a count during transaction assembly.
    unsigned int candidateOverflowFlags = 0u; ///< Candidate-pool/domain overflow flags.
    unsigned int selectedOverflowFlags = 0u; ///< Selected-output overflow flags.
    std::vector<unsigned int> channelVisited; ///< Optional per-descriptor visited counts.
    std::vector<unsigned int> channelAccepted; ///< Optional per-descriptor accepted-task counts.
    std::vector<unsigned int> channelStored; ///< Optional per-descriptor retained-task counts.
    std::vector<unsigned int> channelConstructed; ///< Optional per-descriptor construction counts.
    std::vector<unsigned int> selectionAccepted; ///< Optional per-descriptor accepted-selection counts.
    std::vector<unsigned int> selectionStored; ///< Optional per-descriptor stored-selection counts.
    std::vector<unsigned int> selectionOffsets; ///< Optional per-descriptor compact selection offsets.
  };

  /**
   * @brief Host scratch for dependent graph results: per group in fallback mode, per event in batch mode.
   * In event aggregates, groupIndex/generation can remain default; channel vectors are optional.
   */
  struct KFParticleGpuGraphOperationSnapshot
  {
    unsigned int eventIndex = 0u; ///< Absolute input event index.
    unsigned int groupIndex = 0u; ///< Graph group index for a per-group snapshot; unused for batch event aggregates.
    unsigned int generation = 0u; ///< Graph dependency generation for a per-group snapshot.
    unsigned int visited = 0u; ///< Reported graph combinations visited.
    unsigned int accepted = 0u; ///< Reported accepted graph tasks.
    unsigned int stored = 0u; ///< Reported retained graph tasks.
    unsigned int constructed = 0u; ///< Reported successful graph constructions.
    /// Reported graph rejections; batch aggregation does not populate every rejection counter.
    unsigned int rejected = 0u;
    unsigned int overflowFlags = 0u; ///< Graph routing/execution overflow flags.
    unsigned int candidateBegin = 0u; ///< Raw candidate start offset.
    /// Raw candidate end; temporarily a count during transaction first-generation assembly.
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u; ///< Flattened source-ancestry start offset.
    /// Source-ancestry end; temporarily a count during transaction first-generation assembly.
    unsigned int daughterEnd = 0u;
    unsigned int candidateOverflowFlags = 0u; ///< Candidate-pool/domain overflow flags.
    std::vector<unsigned int> channelVisited; ///< Optional per-descriptor visited counts.
    std::vector<unsigned int> channelAccepted; ///< Optional per-descriptor accepted-task counts.
    std::vector<unsigned int> channelStored; ///< Optional per-descriptor retained-task counts.
    std::vector<unsigned int> channelConstructed; ///< Optional per-descriptor construction counts.
    std::vector<unsigned int> channelRejected; ///< Optional per-descriptor rejection counts.
  };
}
#endif

/**
 * @brief Private coordinator state: owned plans/buffers/results and a borrowed execution queue.
 * No mutex is present. Snapshots retain host storage between runs but are logically reset by full-plan
 * execution. Cached plans follow decay-plan revisions, while published kernel views must also follow
 * buffer layout changes. Non-XPU builds retain only readiness and the owned decay plan.
 */
struct KFParticleGpuSteering::Impl
{
  bool fInitialized; ///< True after steering initialization completes.
#ifdef KFPARTICLE_USE_XPU
  xpu::queue& fQueue; ///< Borrowed runtime queue; must outlive steering and buffer cleanup.
  /// Owned reusable input, work, output, and status storage manager.
  std::unique_ptr<KFParticleGpuBufferManager> fBuffers;
  KFParticleGpuKernels fKernels; ///< Borrowed device-view snapshot published to the finder constant.
  /// Reusable host/device one-record probe journal.
  xpu::buffer<KFParticleGpuTransactionDiagnosticRecord> fTransactionDiagnosticRecord;
  /// Monotonic sequence number for this steering object's diagnostic runs.
  unsigned long long fTransactionDiagnosticSequence;
  /// Detailed two-track channel results, omitted in compact publishable transactions.
  std::vector<KFParticleGpuTwoDaughterChannelResult> fLastDecayPlanResults;
  /// Detailed composite-track channel results when retained.
  std::vector<KFParticleGpuV0TrackChannelResult> fLastV0TrackCascadeResults;
  /// Per-event first-generation scratch counters/ranges.
  std::vector<KFParticleGpuFusedTwoDaughterSnapshot> fFusedTwoDaughterSnapshots;
  /// Per-event composite-track scratch counters/ranges.
  std::vector<KFParticleGpuFusedCascadeSnapshot> fFusedCascadeSnapshots;
  /// Event-local output partitions from the last full plan.
  std::vector<KFParticleGpuDecayPlanEventResult> fLastDecayPlanEventResults;
  /// Host intervals and nested submeasurements of the last full plan.
  KFParticleGpuDecayPlanTiming fLastDecayPlanTiming;
  /// Aggregate first-generation monitoring.
  KFParticleGpuTwoDaughterRoutingMonitorData fLastTwoDaughterRoutingMonitorData;
  KFParticleGpuV0TrackRoutingMonitorData fLastV0TrackRoutingMonitorData; ///< Aggregate composite-track monitoring.
  /// Aggregate dependent-graph monitoring and support coverage.
  KFParticleGpuGraphExecutionMonitorData fLastGraphExecutionMonitorData;
  bool fPerformanceMonitoringEnabled; ///< Opt-in derived performance accounting.
  bool fLastTwoDaughterTransactionBatch; ///< Whether the last full run used prepared transaction work.
  bool fLastTwoDaughterTransactionFused; ///< Whether first-generation fusion was actually eligible and selected.
  bool fLastTwoDaughterSelectionDirect; ///< Whether transaction selection was evaluated during construction.
  bool fLastCascadeTransactionBatch; ///< Whether cascades used the transaction route.
  bool fLastCascadeTransactionFused; ///< Whether transaction cascades constructed during routing.
  unsigned int fLastCascadeGenerationCount; ///< Number of active transaction cascade generations.
  bool fLastGraphTransactionFused; ///< Whether transaction graph operations executed during routing.
  /// Preferred first-generation implementation, subject to eligibility checks.
  KFParticleGpuTwoDaughterTransactionStrategy fTwoDaughterTransactionStrategy;
  /// Preferred transaction selection placement.
  KFParticleGpuTwoDaughterSelectionStrategy fTwoDaughterSelectionStrategy;
  /// Preferred transaction composite-track construction placement.
  KFParticleGpuCascadeTransactionStrategy fCascadeTransactionStrategy;
  KFParticleGpuGraphTransactionStrategy fGraphTransactionStrategy; ///< Preferred transaction graph-operation placement.
  /// Preferred initial administrative reset route.
  KFParticleGpuAdministrativeResetStrategy fAdministrativeResetStrategy;
  /// Last full-plan performance record; not refreshed by isolated helpers.
  KFParticleGpuPerformanceSnapshot fLastPerformanceSnapshot;
  /// Optional detailed event/channel graph counters.
  std::vector<KFParticleGpuGraphChannelMonitorData> fLastGraphChannelMonitorData;
  /// Per-group fallback or per-event batch graph scratch.
  std::vector<KFParticleGpuGraphOperationSnapshot> fGraphOperationSnapshots;
  /// Downloaded first-generation selected-index range.
  KFParticleGpuSelectedCandidateRange fLastDecayPlanSelectedCandidates;
  /// Optional detailed first-generation event/channel selection ranges.
  std::vector<KFParticleGpuSelectedChannelRange> fLastDecayPlanSelectedChannels;
  KFParticleGpuTwoDaughterRoutingPlan fTwoDaughterRoutingPlan; ///< Owned revision-cached two-track routing tables.
  KFParticleGpuV0TrackRoutingPlan fV0TrackRoutingPlan; ///< Owned revision-cached composite-track routing tables.
  KFParticleGpuDecayGraphPlan fDecayGraphPlan; ///< Owned compiled CPU-compatible decay dependency graph.
  unsigned long long fDecayGraphPlanSourceRevision; ///< Revision used to build the cached dependency graph.
#endif
  std::unique_ptr<KFParticleGpuDecayPlan> fDecayPlan; ///< Owned user-configurable source decay plan.

#ifdef KFPARTICLE_USE_XPU
  /**
   * @brief Borrow a queue and choose default fused/direct/device-reset preferences without allocating input buffers.
   */
  explicit Impl(xpu::queue& queue)
    : fInitialized(false), fQueue(queue), fBuffers(), fKernels(),
      fTransactionDiagnosticRecord(), fTransactionDiagnosticSequence(0u),
      fLastDecayPlanResults(),
      fLastV0TrackCascadeResults(),
      fFusedTwoDaughterSnapshots(),
      fFusedCascadeSnapshots(),
      fLastDecayPlanEventResults(),
      fLastDecayPlanTiming(), fLastTwoDaughterRoutingMonitorData(),
      fLastV0TrackRoutingMonitorData(), fLastGraphExecutionMonitorData(),
      fPerformanceMonitoringEnabled(false), fLastTwoDaughterTransactionBatch(false),
      fLastTwoDaughterTransactionFused(false),
      fLastTwoDaughterSelectionDirect(false),
      fLastCascadeTransactionBatch(false),
      fLastCascadeTransactionFused(false),
      fLastCascadeGenerationCount(0u),
      fLastGraphTransactionFused(false),
      fTwoDaughterTransactionStrategy(KFGpuTwoDaughterTransactionFused),
      fTwoDaughterSelectionStrategy(KFGpuTwoDaughterSelectionDirect),
      fCascadeTransactionStrategy(KFGpuCascadeTransactionFused),
      fGraphTransactionStrategy(KFGpuGraphTransactionFused),
      fAdministrativeResetStrategy(KFGpuAdministrativeResetDeviceBatch),
      fLastPerformanceSnapshot(),
      fLastGraphChannelMonitorData(), fGraphOperationSnapshots(),
      fLastDecayPlanSelectedCandidates(), fLastDecayPlanSelectedChannels(),
      fTwoDaughterRoutingPlan(), fV0TrackRoutingPlan(), fDecayGraphPlan(),
      fDecayGraphPlanSourceRevision(0u), fDecayPlan()
  {
  }

  /**
   * @brief Refresh buffer-backed views and publish them as the XPU finder constant, with optional step timing.
   */
  void PublishKernelState()
  {
    KFParticleGpuStepTrace timer("CPU/GPU PublishKernelState (XPU constant)");
    fKernels = MakeKernelState(*fBuffers);
    xpu::set<TheKFParticleFinder>(fKernels);
  }

  /**
   * @brief Opt-in exhaustive phase probes of two-track transaction pairs, with a reusable one-record journal.
   * For every rectangular source pair, runs structural/cut probes, then DCA/transport probes for pre-gated
   * channels. Each phase resets/transfers the journal, waits, and prints results; this is intentionally
   * expensive and not representative of reconstruction performance. Backend failures are logged and rethrown.
   */
  void RunTwoDaughterTransactionDiagnostics()
  {
    if (!TransactionDiagnosticsEnabled()) { return; }
    if (!fTransactionDiagnosticRecord.get()) {
      fTransactionDiagnosticRecord.reset(1u, xpu::buf_io);
    }
    auto* hostRecord = xpu::buffer_prop(fTransactionDiagnosticRecord)
                         .template h_ptr<KFParticleGpuTransactionDiagnosticRecord>();
    const KFParticleGpuTransactionView hostTransaction = fBuffers->HostTransaction();
    const KFParticleGpuTransactionDesc* descriptor = hostTransaction.Transaction();
    if (!hostRecord || !descriptor) {
      throw std::logic_error("KFParticle GPU transaction diagnostic storage is unavailable");
    }

    const unsigned long long diagnosticSequence = ++fTransactionDiagnosticSequence;
    std::fprintf(stderr,
                 "KFPARTICLE_GPU_DIAGNOSTIC LEDGER contract=20260820-transaction-journal-v18 sequence=%llu transaction=%llu revision=%llu domains=%u:%u works=%u:%u tracks=%u vertices=%u events=%u descriptors=%u capacity_allocation_rounds=%llu\n",
                 diagnosticSequence, descriptor->transactionId,
                 descriptor->decayPlanRevision,
                 descriptor->domains.offset, descriptor->domains.size,
                 descriptor->workRanges.offset, descriptor->workRanges.size,
                 fBuffers->TrackSize(), fBuffers->VertexSize(), fBuffers->EventSize(),
                 fKernels.TwoDaughterRouting().DescriptorCount(),
                 static_cast<unsigned long long>(fBuffers->CapacityGrowthCount()));
    std::fflush(stderr);

    /**
     * @brief Reset/upload the journal, run one diagnostic probe, synchronously read it back, and print its result.
     */
    const auto runPhase = [&](unsigned int workIndex, unsigned int pairIndex,
                              unsigned int descriptorIndex, unsigned int phase) {
      *hostRecord = KFParticleGpuTransactionDiagnosticRecord{};
      fQueue.copy(fTransactionDiagnosticRecord, xpu::h2d);
      fQueue.wait();
      std::fprintf(stderr,
                   "KFPARTICLE_GPU_DIAGNOSTIC BEGIN sequence=%llu transaction=%llu work=%u pair=%u descriptor=%u phase=%u:%s grid=1 block=64\n",
                   diagnosticSequence, descriptor->transactionId, workIndex,
                   pairIndex, descriptorIndex,
                   phase, TransactionDiagnosticPhaseName(phase));
      std::fflush(stderr);
      try {
        if (phase == KFGpuTransactionDiagnosticJournal) {
          fQueue.launch<KFParticleGpuProbeTransactionJournal>(
            xpu::n_threads(1u), diagnosticSequence,
            fTransactionDiagnosticRecord.get());
        }
        else if (phase <= KFGpuTransactionDiagnosticDomain) {
          fQueue.launch<KFParticleGpuProbeTransactionStructure>(
            xpu::n_threads(1u), fBuffers->DeviceTransaction(),
            diagnosticSequence, workIndex, pairIndex, phase,
            fTransactionDiagnosticRecord.get());
        }
        else {
          fQueue.launch<KFParticleGpuProbeTwoDaughterTransaction>(
            xpu::n_threads(1u), fKernels, fBuffers->DeviceTransaction(),
            diagnosticSequence, workIndex, pairIndex, descriptorIndex, phase,
            fTransactionDiagnosticRecord.get());
        }
        fQueue.copy(fTransactionDiagnosticRecord, xpu::d2h);
        fQueue.wait();
      }
      catch (const std::exception& error) {
        std::fprintf(stderr,
                     "KFPARTICLE_GPU_DIAGNOSTIC FAIL sequence=%llu transaction=%llu work=%u pair=%u descriptor=%u phase=%u:%s error=%s\n",
                     diagnosticSequence, descriptor->transactionId, workIndex,
                     pairIndex, descriptorIndex,
                     phase, TransactionDiagnosticPhaseName(phase), error.what());
        std::fflush(stderr);
        throw;
      }
      std::fprintf(stderr,
                   "KFPARTICLE_GPU_DIAGNOSTIC PASS sequence=%llu transaction=%llu work=%u domain=%u event=%u group=%u pair=%u tracks=%u,%u sources=%d,%d pdg=%d,%d pv=%d,%d descriptor=%u phase=%u:%s completed=%u status=%u guards=0x%x compatible=%u pregate=%u by=%.9g ds=%.9g,%.9g transport=0x%x,0x%x distance2=%.9g max2=%.9g dot=%.9g momentum2=%.9g,%.9g accepted=%u\n",
                   hostRecord->diagnosticSequence, hostRecord->transactionId,
                   hostRecord->workIndex,
                   hostRecord->domainIndex, hostRecord->eventIndex,
                   hostRecord->groupIndex, hostRecord->pairIndex,
                   hostRecord->firstTrackIndex, hostRecord->secondTrackIndex,
                   hostRecord->firstSourceId, hostRecord->secondSourceId,
                   hostRecord->firstPdg, hostRecord->secondPdg,
                   hostRecord->firstPrimaryVertex, hostRecord->secondPrimaryVertex,
                   hostRecord->descriptorIndex, hostRecord->phase,
                   TransactionDiagnosticPhaseName(hostRecord->phase),
                   hostRecord->completedPhase, hostRecord->status,
                   hostRecord->guardFlags, hostRecord->compatibleChannels.Count(),
                   hostRecord->preGateChannels.Count(), hostRecord->fieldBy,
                   hostRecord->secondDs, hostRecord->firstDs,
                   hostRecord->secondTransportStatus,
                   hostRecord->firstTransportStatus, hostRecord->distance2,
                   hostRecord->maxDistance2, hostRecord->momentumDot,
                   hostRecord->firstMomentum2, hostRecord->secondMomentum2,
                   hostRecord->accepted);
      std::fflush(stderr);
    };

    for (unsigned int localWork = 0u; localWork < descriptor->workRanges.size;
         ++localWork) {
      const unsigned int workIndex = descriptor->workRanges.offset + localWork;
      const KFParticleGpuWorkRange& work = hostTransaction.WorkRanges()[workIndex];
      if (work.family != KFParticleGpuTwoDaughterWork) { continue; }
      const unsigned long long pairCount64 =
        static_cast<unsigned long long>(work.sourceTracks.size)
        * work.secondarySourceTracks.size;
      if (pairCount64 > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("KFParticle GPU diagnostic pair count overflows");
      }
      const unsigned int pairCount = static_cast<unsigned int>(pairCount64);
      for (unsigned int pair = 0u; pair < pairCount; ++pair) {
        for (unsigned int phase = KFGpuTransactionDiagnosticJournal;
             phase <= KFGpuTransactionDiagnosticTrackCuts; ++phase) {
          runPhase(workIndex, pair, KFParticleGpuChannelMask::InvalidBit, phase);
        }
        const KFParticleGpuChannelMask preGate = hostRecord->preGateChannels;
        for (unsigned int bit = preGate.NextSetBit(0u);
             bit != KFParticleGpuChannelMask::InvalidBit;
             bit = preGate.NextSetBit(bit + 1u)) {
          for (unsigned int phase = KFGpuTransactionDiagnosticDcaStateField;
               phase <= KFGpuTransactionDiagnosticPairDecision; ++phase) {
            runPhase(workIndex, pair, bit, phase);
          }
        }
      }
    }
    std::fprintf(stderr,
                 "KFPARTICLE_GPU_DIAGNOSTIC COMPLETE sequence=%llu transaction=%llu\n",
                 diagnosticSequence, descriptor->transactionId);
    std::fflush(stderr);
  }

  /// Buffer-manager reset-launch counter captured at the start of a full run.
  std::uint64_t fSelectionResetLaunchesBefore = 0u;

  /**
   * @brief Assemble derived counters, estimated payloads, and host timings when monitoring is enabled.
   * Uses route formulas for launch/wait counts, buffer-manager accounting for allocation rounds/bytes,
   * and live vector sizes for retained snapshot bytes. Does not perform extra device readbacks or
   * measure an independent memory peak. Optional debug launches/waits are not exhaustively counted.
   */
  void CompletePerformanceSnapshot(unsigned int eventCount,
                                   std::uint64_t growthCountBefore,
                                   unsigned int planUploadWaits,
                                   unsigned int fixedQueueWaits,
                                   unsigned int downloadedCandidates,
                                   unsigned int downloadedDaughters,
                                   unsigned int downloadedSelectedCandidates,
                                   bool fullDiagnostics)
  {
    if (!fPerformanceMonitoringEnabled) { return; }
    KFParticleGpuPerformanceSnapshot& snapshot = fLastPerformanceSnapshot;
    snapshot.enabled = true;
    snapshot.compactOutput = !fullDiagnostics;
    snapshot.events = eventCount;
    snapshot.tracks = fBuffers->TrackSize();
    snapshot.vertices = fBuffers->VertexSize();
    snapshot.descriptorGroups = fLastTwoDaughterRoutingMonitorData.groupLaunches
                                + fLastV0TrackRoutingMonitorData.groupLaunches
                                + fLastGraphExecutionMonitorData.groupLaunches;
    snapshot.visitedCombinations = fLastTwoDaughterRoutingMonitorData.visitedPairs
                                   + fLastV0TrackRoutingMonitorData.visitedPairs
                                   + fLastGraphExecutionMonitorData.visitedCombinations;
    snapshot.activeChannelBits = fLastTwoDaughterRoutingMonitorData.activeChannelBits
                                 + fLastV0TrackRoutingMonitorData.activeChannelBits;
    snapshot.acceptedTasks = fLastTwoDaughterRoutingMonitorData.acceptedTasks
                             + fLastV0TrackRoutingMonitorData.acceptedTasks
                             + fLastGraphExecutionMonitorData.acceptedTasks;
    snapshot.storedTasks = fLastTwoDaughterRoutingMonitorData.storedTasks
                           + fLastV0TrackRoutingMonitorData.storedTasks
                           + fLastGraphExecutionMonitorData.storedTasks;
    snapshot.blockReservations = fLastTwoDaughterRoutingMonitorData.blockReservations
                                 + fLastV0TrackRoutingMonitorData.blockReservations;
    snapshot.rejectedTasks = fLastGraphExecutionMonitorData.rejectedTasks;
    snapshot.rawCandidates = fLastTwoDaughterRoutingMonitorData.candidates
                             + fLastV0TrackRoutingMonitorData.candidates
                             + fLastGraphExecutionMonitorData.candidates;
    snapshot.selectedCandidates = fLastTwoDaughterRoutingMonitorData.selectedCandidates;
    snapshot.daughters = fLastTwoDaughterRoutingMonitorData.daughters
                         + fLastV0TrackRoutingMonitorData.daughters
                         + fLastGraphExecutionMonitorData.daughters;
    snapshot.overflowFlags = fLastTwoDaughterRoutingMonitorData.overflowFlags
                             | fLastV0TrackRoutingMonitorData.overflowFlags
                             | fLastGraphExecutionMonitorData.overflowFlags;
    snapshot.kernelLaunches = (fLastTwoDaughterTransactionBatch
                                 ? (fLastGraphTransactionFused ? 3u : 4u)
                                 : 3u)
                              * fLastGraphExecutionMonitorData.groupLaunches;
    if (fLastTwoDaughterRoutingMonitorData.descriptorCount > 0u) {
      snapshot.kernelLaunches += fLastTwoDaughterTransactionBatch
                                   ? 7u - (fLastTwoDaughterTransactionFused ? 1u : 0u)
                                       - (fLastTwoDaughterSelectionDirect ? 1u : 0u)
                                   : fLastTwoDaughterRoutingMonitorData.groupLaunches
                                       + eventCount * 5u;
    }
    if (fLastV0TrackRoutingMonitorData.descriptorCount > 0u) {
      snapshot.kernelLaunches += fLastCascadeTransactionBatch
                                   ? fLastCascadeGenerationCount
                                       * (fLastCascadeTransactionFused ? 3u : 4u)
                                   : fLastV0TrackRoutingMonitorData.groupLaunches
                                       + eventCount * 2u;
    }
    snapshot.kernelLaunches += fBuffers->SelectionResetLaunches() - fSelectionResetLaunchesBefore;
    snapshot.queueWaits = fixedQueueWaits + planUploadWaits;
    snapshot.capacityGrowths = fBuffers->CapacityGrowthCount() - growthCountBefore;
    snapshot.hostToDeviceBytes = fBuffers->InputPayloadBytes();
    snapshot.deviceToHostBytes = fBuffers->CandidatePayloadBytes(
                                   downloadedCandidates,
                                   downloadedDaughters,
                                   fullDiagnostics)
                                 + fBuffers->SelectedCandidatePayloadBytes(
                                     downloadedSelectedCandidates)
                                 + snapshot.transactionStatusBytes;
    snapshot.retainedSnapshotBytes =
      static_cast<std::uint64_t>(fLastDecayPlanResults.size())
        * sizeof(KFParticleGpuTwoDaughterChannelResult)
      + static_cast<std::uint64_t>(fLastV0TrackCascadeResults.size())
          * sizeof(KFParticleGpuV0TrackChannelResult)
      + static_cast<std::uint64_t>(fFusedTwoDaughterSnapshots.size())
          * sizeof(KFParticleGpuFusedTwoDaughterSnapshot)
      + static_cast<std::uint64_t>(fFusedCascadeSnapshots.size())
          * sizeof(KFParticleGpuFusedCascadeSnapshot)
      + static_cast<std::uint64_t>(fGraphOperationSnapshots.size())
          * sizeof(KFParticleGpuGraphOperationSnapshot)
      + static_cast<std::uint64_t>(fLastDecayPlanEventResults.size())
          * sizeof(KFParticleGpuDecayPlanEventResult)
      + static_cast<std::uint64_t>(fLastGraphChannelMonitorData.size())
          * sizeof(KFParticleGpuGraphChannelMonitorData)
      + static_cast<std::uint64_t>(fLastDecayPlanSelectedChannels.size())
          * sizeof(KFParticleGpuSelectedChannelRange);
    for (const auto& item : fFusedTwoDaughterSnapshots) {
      snapshot.retainedSnapshotBytes += sizeof(unsigned int)
        * (item.channelVisited.size() + item.channelAccepted.size()
           + item.channelStored.size() + item.channelConstructed.size()
           + item.selectionAccepted.size() + item.selectionStored.size()
           + item.selectionOffsets.size());
    }
    for (const auto& item : fFusedCascadeSnapshots) {
      snapshot.retainedSnapshotBytes += sizeof(unsigned int)
        * (item.channelVisited.size() + item.channelAccepted.size()
           + item.channelStored.size() + item.channelConstructed.size());
    }
    for (const auto& item : fGraphOperationSnapshots) {
      snapshot.retainedSnapshotBytes += sizeof(unsigned int)
        * (item.channelVisited.size() + item.channelAccepted.size()
           + item.channelStored.size() + item.channelConstructed.size()
           + item.channelRejected.size());
    }
    snapshot.allocatedBytesHighWater = fBuffers->AllocatedBytes();
    const double maskSlots =
      static_cast<double>(fLastTwoDaughterRoutingMonitorData.visitedPairs)
        * fLastTwoDaughterRoutingMonitorData.descriptorCount
      + static_cast<double>(fLastV0TrackRoutingMonitorData.visitedPairs)
          * fLastV0TrackRoutingMonitorData.descriptorCount;
    snapshot.maskDensity = maskSlots > 0. ? snapshot.activeChannelBits / maskSlots : 0.;
    snapshot.usefulWorkPerLaunch = snapshot.kernelLaunches > 0u
                                     ? static_cast<double>(snapshot.storedTasks)
                                         / snapshot.kernelLaunches
                                     : 0.;
    snapshot.candidatePoolOccupancy = fBuffers->Capacities().candidates > 0u
                                        ? static_cast<double>(snapshot.rawCandidates)
                                            / fBuffers->Capacities().candidates
                                        : 0.;
    snapshot.fitUnrollEnabled = KFPARTICLE_GPU_FIT_UNROLL != 0;
    snapshot.timing = fLastDecayPlanTiming;
  }
#else
  /**
   * @brief Create non-XPU bookkeeping with no initialized decay plan.
   */
  Impl() : fInitialized(false), fDecayPlan() {}
#endif
};

#ifdef KFPARTICLE_USE_XPU
/**
 * @brief Construct private state; XPU construction borrows the supplied queue without taking ownership.
 */
KFParticleGpuSteering::KFParticleGpuSteering(xpu::queue& queue) : fImpl(new Impl(queue)) {}
#else
/**
 * @brief Construct private state; XPU construction borrows the supplied queue without taking ownership.
 */
KFParticleGpuSteering::KFParticleGpuSteering() : fImpl(new Impl) {}
#endif

/**
 * @brief Finalize before releasing private state; no exceptions from backend cleanup are caught here.
 */
KFParticleGpuSteering::~KFParticleGpuSteering()
{
  Finalize();
}

/**
 * @brief Initialize reusable buffers and a fresh decay plan once; revision caches are reset in XPU builds.
 */
void KFParticleGpuSteering::Initialize()
{
  if (fImpl->fInitialized) {
    return;
  }

#ifdef KFPARTICLE_USE_XPU
  fImpl->fBuffers.reset(new KFParticleGpuBufferManager(fImpl->fQueue));
  fImpl->fTwoDaughterRoutingPlan.Clear();
  fImpl->fV0TrackRoutingPlan.Clear();
  fImpl->fDecayGraphPlan.Clear();
  fImpl->fDecayGraphPlanSourceRevision = 0u;
#endif
  fImpl->fDecayPlan = std::make_unique<KFParticleGpuDecayPlan>();
  fImpl->fInitialized = true;
}

/**
 * @brief Wait, clear published device pointers, and release buffers/plan; leave the borrowed queue alive.
 */
void KFParticleGpuSteering::Finalize()
{
  if (!fImpl->fInitialized) {
    return;
  }

#ifdef KFPARTICLE_USE_XPU
  // Clear published device pointers before releasing their owning buffers.
  fImpl->fQueue.wait();
  fImpl->fKernels = MakeEmptyKernels();
  xpu::set<TheKFParticleFinder>(fImpl->fKernels);
  fImpl->fBuffers.reset();
#endif
  fImpl->fDecayPlan.reset();
  fImpl->fInitialized = false;
}

/**
 * @brief Read the local initialization flag without synchronization.
 */
bool KFParticleGpuSteering::IsInitialized() const
{
  return fImpl->fInitialized;
}

/**
 * @brief Require initialization and return borrowed access to the owned decay plan.
 */
KFParticleGpuDecayPlan& KFParticleGpuSteering::GetDecayPlan()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fDecayPlan;
}

/**
 * @brief Require initialization and return borrowed access to the owned decay plan.
 */
const KFParticleGpuDecayPlan& KFParticleGpuSteering::GetDecayPlan() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fDecayPlan;
}

#ifdef KFPARTICLE_USE_XPU
/**
 * @brief Require initialization and return borrowed access to the owned buffer manager.
 */
KFParticleGpuBufferManager& KFParticleGpuSteering::GetBuffers()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fBuffers;
}

/**
 * @brief Require initialization and return borrowed access to the owned buffer manager.
 */
const KFParticleGpuBufferManager& KFParticleGpuSteering::GetBuffers() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fBuffers;
}

/**
 * @brief Upload/reset, publish fresh kernel views, execute the track conversion smoke kernel, and read back.
 */
void KFParticleGpuSteering::RunRoundTrip(float mass, unsigned int eventIndex)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (mass < 0.f) {
    throw std::invalid_argument("KFParticle GPU round-trip mass must be non-negative");
  }
  if (fImpl->fBuffers->TrackSize() > 0 && eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU round-trip event index is out of range");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();
  // Publish fresh views after any capacity or event-size change, including an
  // empty event for which no action is launched.
  fImpl->PublishKernelState();

  if (fImpl->fBuffers->TrackSize() > 0) {
    fImpl->fQueue.launch<KFParticleGpuRoundTrip>(
      xpu::n_threads(fImpl->fBuffers->TrackSize()), mass, eventIndex);
    fImpl->fQueue.wait();
  }

  fImpl->fBuffers->DownloadCandidates();
}

/**
 * @brief Execute the fixed-slot task-generation/construction test route with explicit waits and full readback.
 */
void KFParticleGpuSteering::RunTwoDaughterStage(const KFParticleGpuTwoDaughterTaskSource& source,
                                             unsigned int taskCapacity)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU two-daughter stage task capacity must be positive");
  }

  if (source.eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU two-daughter stage event index is out of range");
  }
  if (taskCapacity > fImpl->fBuffers->Capacities().candidates) {
    throw std::out_of_range("KFParticle GPU two-daughter stage task capacity exceeds candidate capacity");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  fImpl->fQueue.launch<KFParticleGpuGenerateTwoDaughterTasks>(
    xpu::n_threads(taskCapacity),
    fImpl->fBuffers->DeviceEvents(),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    storage.fTwoDaughterTaskCount.get(),
    storage.fTwoDaughterTotalPairCount.get());
  fImpl->fQueue.wait();

  fImpl->fQueue.launch<KFParticleGpuTwoDaughterTaskKernel>(
    xpu::n_threads(taskCapacity),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    fImpl->fBuffers->DeviceCandidates());
  fImpl->fQueue.wait();

  fImpl->fBuffers->DownloadCandidates();
}

/**
 * @brief Delegate compact single-source execution with channel ID zero, then read back the full candidate pool.
 */
void KFParticleGpuSteering::RunTwoDaughterCompactStage(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity)
{
  (void) RunTwoDaughterCompactStageImpl(source, taskCapacity, 0u);
  fImpl->fBuffers->DownloadCandidates();
}

/**
 * @brief Validate isolated-stage settings, upload input, clear candidates, and append one compact channel.
 */
KFParticleGpuTwoDaughterChannelResult KFParticleGpuSteering::RunTwoDaughterCompactStageImpl(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity,
  unsigned int channelId)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU compact two-daughter stage task capacity must be positive");
  }
  if (source.eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU compact two-daughter stage event index is out of range");
  }
  if (taskCapacity > fImpl->fBuffers->Capacities().candidates) {
    throw std::out_of_range(
      "KFParticle GPU compact two-daughter stage task capacity exceeds candidate capacity");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();

  return RunTwoDaughterCompactChannel(source, taskCapacity, channelId, 0u, 0u);
}

/**
 * @brief Route one source's rectangular pair space into bounded tasks, construct, and read counter deltas.
 * The caller supplies coherent pool offsets. Accepted-task overflow is marked on the candidate pool;
 * full fit arrays remain on device until the caller requests readback.
 */
KFParticleGpuTwoDaughterChannelResult KFParticleGpuSteering::RunTwoDaughterCompactChannel(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity,
  unsigned int channelId,
  unsigned int candidateOffset,
  unsigned int daughterOffset)
{
  const unsigned int totalPairs =
    CountTwoDaughterPairs(fImpl->fBuffers->HostEvents()[source.eventIndex], source);
  KFParticleGpuTwoDaughterChannelResult result;
  result.channelId = channelId;
  result.motherPdg = source.motherPdg;
  result.eventIndex = source.eventIndex;
  result.totalPairs = totalPairs;
  result.candidates.offset = candidateOffset;
  result.candidates.daughterOffset = daughterOffset;

  if (totalPairs == 0u) {
    const KFParticleGpuCandidatePoolStatus status = fImpl->fBuffers->DownloadCandidateStatus();
    result.candidates.size = status.candidates - candidateOffset;
    result.candidates.daughterSize = status.daughters - daughterOffset;
    result.candidates.overflowFlags = status.overflowFlags;
    result.constructedCandidates = result.candidates.size;
    result.constructedDaughters = result.candidates.daughterSize;
    result.generationCandidates = result.candidates;
    return result;
  }

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  fImpl->fQueue.launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
    xpu::n_threads(totalPairs),
    fImpl->fBuffers->DeviceEvents(),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    storage.fTwoDaughterTaskCount.get(),
    storage.fTwoDaughterTotalPairCount.get());

  fImpl->fQueue.launch<KFParticleGpuTwoDaughterCompactCandidatePoolKernel>(
      xpu::n_threads(taskCapacity),
      MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
      storage.fTwoDaughterTasks.get(),
      taskCapacity,
      storage.fTwoDaughterTaskCount.get(),
      fImpl->fBuffers->DeviceCandidates());

  const KFParticleGpuTwoDaughterTaskStatus taskStatus =
    fImpl->fBuffers->DownloadTwoDaughterTaskStatus();
  result.acceptedTasks = taskStatus.accepted;
  result.storedTasks = taskStatus.accepted < taskCapacity ? taskStatus.accepted : taskCapacity;
  if (taskStatus.accepted > taskCapacity) {
    fImpl->fBuffers->MarkCandidateOverflow(static_cast<unsigned int>(CandidateCapacityExceeded));
  }
  const KFParticleGpuCandidatePoolStatus status = fImpl->fBuffers->DownloadCandidateStatus();
  result.candidates.size = status.candidates - candidateOffset;
  result.candidates.daughterSize = status.daughters - daughterOffset;
  result.candidates.overflowFlags = status.overflowFlags;
  result.constructedCandidates = result.candidates.size;
  result.constructedDaughters = result.candidates.daughterSize;
  result.generationCandidates = result.candidates;
  return result;
}

/**
 * @brief Append routed two-track fits to resident candidates, with no host counter boundary between kernels.
 * Refreshes the routing plan by revision, then downloads counters, task records, candidates, and descriptors.
 * Input upload and initial candidate reset are the caller's responsibility.
 */
KFParticleGpuTwoDaughterFusedResult
KFParticleGpuSteering::RunTwoDaughterFusedStage(
  unsigned int eventIndex,
  unsigned int taskCapacity,
  KFParticleGpuTwoDaughterRoutingMode routingMode)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range(
      "KFParticle GPU fused two-daughter event index is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument(
      "KFParticle GPU fused two-daughter task capacity must be nonzero");
  }
  if (routingMode != KFGpuTwoDaughterRoutingAtomic
      && routingMode != KFGpuTwoDaughterRoutingBlockScan) {
    throw std::invalid_argument(
      "KFParticle GPU fused two-daughter routing mode is invalid");
  }

  const bool planChanged =
    fImpl->fTwoDaughterRoutingPlan.SourceRevision()
      != fImpl->fDecayPlan->Revision();
  if (planChanged) {
    fImpl->fTwoDaughterRoutingPlan.Compile(*fImpl->fDecayPlan);
  }
  fImpl->fBuffers->UploadTwoDaughterRoutingPlan(
    fImpl->fTwoDaughterRoutingPlan, planChanged);
  fImpl->fBuffers->EnsureTwoDaughterRoutedTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterRoutingStatus();
  fImpl->PublishKernelState();

  const KFParticleGpuCandidatePoolStatus initialCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  const KFParticleGpuEventDesc& event =
    fImpl->fBuffers->HostEvents()[eventIndex];

  KFParticleGpuTwoDaughterFusedResult result;
  result.eventIndex = eventIndex;
  result.descriptorCount = static_cast<unsigned int>(
    fImpl->fTwoDaughterRoutingPlan.Descriptors().size());
  for (const KFParticleGpuTwoDaughterExecutionGroup& group :
       fImpl->fTwoDaughterRoutingPlan.Groups()) {
    const KFParticleGpuRange firstTracks =
      event.TrackSet(group.firstTrackSet).tracks;
    const KFParticleGpuRange secondTracks =
      event.TrackSet(group.secondTrackSet).tracks;
    const unsigned int pairCount = firstTracks.size * secondTracks.size;
    if (pairCount == 0u) {
      continue;
    }
    ++result.groupLaunches;
    if (routingMode == KFGpuTwoDaughterRoutingAtomic) {
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksAtomic>(
        xpu::n_threads(pairCount),
        MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
        fImpl->fBuffers->DeviceEvents(),
        fImpl->fBuffers->DeviceTwoDaughterRouting(),
        eventIndex,
        group.groupIndex,
        storage.fTwoDaughterRoutedTasks.get(),
        taskCapacity,
        storage.fTwoDaughterRoutingVisitedPairCount.get(),
        storage.fTwoDaughterRoutingActiveBitCount.get(),
        storage.fTwoDaughterRoutedAcceptedTaskCount.get(),
        storage.fTwoDaughterRoutedStoredTaskCount.get(),
        storage.fTwoDaughterRoutingBlockReservationCount.get(),
        storage.fTwoDaughterRoutedTaskOverflowFlags.get());
    }
    else {
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        taskCapacity);
    }
  }

  // The routed pool and its device counter are consumed in queue order. No
  // task-count readback or host wait separates routing from construction.
  fImpl->fQueue.launch<KFParticleGpuTwoDaughterRoutedCandidatePoolKernel>(
    xpu::n_threads(taskCapacity),
    taskCapacity);

  result.routing = fImpl->fBuffers->DownloadTwoDaughterRoutingStatus();
  const KFParticleGpuCandidatePoolStatus finalCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  fImpl->fBuffers->DownloadTwoDaughterRoutedTasks();
  fImpl->fBuffers->DownloadCandidates();
  fImpl->fBuffers->DownloadCandidateDescriptorIndices();
  result.candidates.offset = initialCandidates.candidates;
  result.candidates.size =
    finalCandidates.candidates - initialCandidates.candidates;
  result.candidates.daughterOffset = initialCandidates.daughters;
  result.candidates.daughterSize =
    finalCandidates.daughters - initialCandidates.daughters;
  result.candidates.overflowFlags =
    finalCandidates.overflowFlags | result.routing.overflowFlags;
  return result;
}

/**
 * @brief Append composite-track fits from resident selected parents and input tracks.
 * Routing and construction share a device counter in queue order; task/candidate readback follows both.
 * The host selected-size counter must already describe the resident selected list.
 */
KFParticleGpuV0TrackFusedResult KFParticleGpuSteering::RunV0TrackFusedStage(
  unsigned int eventIndex,
  const KFParticleGpuSelectedCandidateRange& selectedRange,
  unsigned int taskCapacity,
  KFParticleGpuV0TrackRoutingMode routingMode)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU fused V0-track event index is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU fused V0-track task capacity must be nonzero");
  }
  if (routingMode != KFGpuV0TrackRoutingAtomic
      && routingMode != KFGpuV0TrackRoutingBlockScan) {
    throw std::invalid_argument("KFParticle GPU fused V0-track routing mode is invalid");
  }
  const unsigned int selectedSize = fImpl->fBuffers->HostSelectedCandidates().Size();
  if (selectedRange.offset > selectedSize
      || selectedRange.size > selectedSize - selectedRange.offset) {
    throw std::out_of_range("KFParticle GPU fused V0-track selected range is out of range");
  }

  const bool planChanged =
    fImpl->fV0TrackRoutingPlan.SourceRevision() != fImpl->fDecayPlan->Revision();
  if (planChanged) {
    fImpl->fV0TrackRoutingPlan.Compile(*fImpl->fDecayPlan);
  }
  fImpl->fBuffers->UploadV0TrackRoutingPlan(fImpl->fV0TrackRoutingPlan, planChanged);
  fImpl->fBuffers->EnsureV0TrackRoutedTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetV0TrackRoutingStatus();
  fImpl->PublishKernelState();

  const KFParticleGpuCandidatePoolStatus initialCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  const KFParticleGpuEventDesc& event = fImpl->fBuffers->HostEvents()[eventIndex];
  for (const KFParticleGpuV0TrackExecutionGroup& group :
       fImpl->fV0TrackRoutingPlan.Groups()) {
    const KFParticleGpuRange bachelors = event.TrackSet(group.bachelorTrackSet).tracks;
    const unsigned int pairCount = selectedRange.size * bachelors.size;
    if (pairCount == 0u) {
      continue;
    }
    if (routingMode == KFGpuV0TrackRoutingAtomic) {
      fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksAtomic>(
        xpu::n_threads(pairCount),
        MakeConstView(fImpl->fBuffers->DeviceCandidates()),
        MakeConstView(fImpl->fBuffers->DeviceSelectedCandidates()),
        MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
        fImpl->fBuffers->DeviceEvents(),
        fImpl->fBuffers->DeviceV0TrackRouting(),
        eventIndex,
        group.groupIndex,
        selectedRange,
        storage.fV0TrackRoutedTasks.get(),
        taskCapacity,
        storage.fV0TrackRoutingVisitedPairCount.get(),
        storage.fV0TrackRoutingActiveBitCount.get(),
        storage.fV0TrackRoutedAcceptedTaskCount.get(),
        storage.fV0TrackRoutedStoredTaskCount.get(),
        storage.fV0TrackRoutingBlockReservationCount.get(),
        storage.fV0TrackRoutedTaskOverflowFlags.get());
    }
    else {
      fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        selectedRange,
        taskCapacity);
    }
  }

  // Queue ordering is the hand-off: construction reads the device counter and
  // routed pool directly, with no scalar D2H synchronization between kernels.
  fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
    xpu::n_threads(taskCapacity),
    taskCapacity);

  KFParticleGpuV0TrackFusedResult result;
  result.eventIndex = eventIndex;
  result.routing = fImpl->fBuffers->DownloadV0TrackRoutingStatus();
  const KFParticleGpuCandidatePoolStatus finalCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  fImpl->fBuffers->DownloadV0TrackRoutedTasks();
  fImpl->fBuffers->DownloadCandidates();
  result.candidates.offset = initialCandidates.candidates;
  result.candidates.size = finalCandidates.candidates - initialCandidates.candidates;
  result.candidates.daughterOffset = initialCandidates.daughters;
  result.candidates.daughterSize = finalCandidates.daughters - initialCandidates.daughters;
  result.candidates.overflowFlags =
    finalCandidates.overflowFlags | result.routing.overflowFlags;
  return result;
}

/**
 * @brief Reset selection output, evaluate one raw range, and download indices and decisions without refitting.
 */
KFParticleGpuSelectedCandidateRange KFParticleGpuSteering::RunV0Selection(
  const KFParticleGpuTwoDaughterChannel& channel,
  const KFParticleGpuCandidateRange& rawCandidates,
  unsigned int eventIndex)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU V0 selection event index is out of range");
  }
  if (rawCandidates.offset > fImpl->fBuffers->Capacities().candidates
      || rawCandidates.size > fImpl->fBuffers->Capacities().candidates - rawCandidates.offset) {
    throw std::out_of_range("KFParticle GPU V0 selection raw candidate range is out of range");
  }
  if (fImpl->fBuffers->Capacities().selectedCandidates == 0u) {
    throw std::logic_error("KFParticle GPU selected-candidate capacity is zero");
  }

  fImpl->fBuffers->ResetSelectedCandidates();
  fImpl->fBuffers->ResetV0SelectionResults();
  if (rawCandidates.size > 0u) {
    fImpl->fQueue.launch<KFParticleGpuSelectV0Candidates>(
      xpu::n_threads(rawCandidates.size),
      MakeConstView(fImpl->fBuffers->DeviceCandidates()),
      MakeConstView(fImpl->fBuffers->DevicePrimaryVertices()),
      fImpl->fBuffers->DeviceEvents(),
      eventIndex,
      rawCandidates.offset,
      rawCandidates.size,
      channel.channelId,
      channel.selection,
      fImpl->fBuffers->DeviceV0SelectionResults(),
      fImpl->fBuffers->DeviceSelectedCandidates());
    fImpl->fQueue.wait();
  }
  fImpl->fBuffers->DownloadSelectedCandidates();
  fImpl->fBuffers->DownloadV0SelectionResults();

  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
  KFParticleGpuSelectedCandidateRange result;
  result.offset = 0u;
  result.size = selected.Size();
  result.overflowFlags = selected.OverflowFlags();
  return result;
}

/**
 * @brief Delegate one-event execution to the full batch path.
 */
const std::vector<KFParticleGpuTwoDaughterChannelResult>& KFParticleGpuSteering::RunDecayPlan(
  unsigned int eventIndex,
  unsigned int taskCapacity,
  KFParticleGpuOutputTransferMode outputMode)
{
  return RunDecayPlanBatch(eventIndex, 1u, taskCapacity, outputMode);
}

/**
 * @brief Plan and execute one full event batch, then assemble host ranges and transfer numerical output.
 *
 * The sequence is input upload, graph/routing compilation, first-generation work planning,
 * routing/construction/selection, composite-track generations, graph-operation generations,
 * completion/status readback, host range assembly, and bounded output readback. A prepared
 * matching transaction enables event-local work and reservations; otherwise submission is per event.
 *
 * Compact output planning retains routed tasks, reads their small status payload, and allocates
 * candidate/ancestry/selected reservations with dependent-generation headroom before construction.
 * It is distinct from publishable transfer mode, which suppresses selection diagnostics and detailed
 * channel snapshots. Isolated helpers and this full path do not share the same reset/readback contract.
 * Returned vectors are owned scratch, overflow is observable, and exceptions do not roll back a run.
 *
 * @verbatim
 * Transaction-batch path
 *
 * [H2D] Upload input tracks, vertices, and event descriptors
 *   |
 * [CPU] Compile/reuse decay graph and routing plans
 *   |
 * [CPU] Prepare event-local work, reservations, and pair tiles
 *   |
 * [GPU] Route track pairs -> Construct two-track candidates
 *   |
 * [GPU] Evaluate selection -> Prepare segments -> Scatter indices
 *   |
 * [GPU] Resolve sources for dependent work
 *   |
 * [GPU] Composite-track generations: route -> construct -> resolve
 *   |
 * [GPU] Graph-operation generations: route -> execute -> resolve
 *   |
 * [D2H + WAIT] Read completion/status data
 *   |
 * [CPU] Assemble event ranges and optional channel diagnostics
 *   |
 * [D2H] Download required candidate/ancestry/selected prefixes
 *
 * Queue order preserves dependencies.
 * Some boxes may be fused; planning may add transfers and waits.
 * @endverbatim
 */
const std::vector<KFParticleGpuTwoDaughterChannelResult>& KFParticleGpuSteering::RunDecayPlanBatch(
  unsigned int firstEventIndex,
  unsigned int eventCount,
  unsigned int taskCapacity,
  KFParticleGpuOutputTransferMode outputMode)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventCount == 0u) {
    throw std::invalid_argument("KFParticle GPU decay-plan batch must contain at least one event");
  }
  if (firstEventIndex >= fImpl->fBuffers->EventSize()
      || eventCount > fImpl->fBuffers->EventSize() - firstEventIndex) {
    throw std::out_of_range("KFParticle GPU decay-plan batch event range is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU decay-plan task capacity must be positive");
  }

  KFParticleGpuProfileScope deviceProfile;
  fImpl->fSelectionResetLaunchesBefore = fImpl->fBuffers->SelectionResetLaunches();
  const std::uint64_t capacityGrowthCountBefore =
    fImpl->fBuffers->CapacityGrowthCount();
  unsigned int planUploadWaits = 0u;
  fImpl->fLastPerformanceSnapshot = KFParticleGpuPerformanceSnapshot();
  fImpl->fLastPerformanceSnapshot.enabled = fImpl->fPerformanceMonitoringEnabled;

  fImpl->fLastDecayPlanResults.clear();
  fImpl->fLastV0TrackCascadeResults.clear();
  fImpl->fFusedTwoDaughterSnapshots.clear();
  fImpl->fFusedCascadeSnapshots.clear();
  fImpl->fLastDecayPlanEventResults.clear();
  fImpl->fLastDecayPlanTiming = KFParticleGpuDecayPlanTiming();
  fImpl->fLastTwoDaughterRoutingMonitorData =
    KFParticleGpuTwoDaughterRoutingMonitorData();
  fImpl->fLastTwoDaughterTransactionBatch = false;
  fImpl->fLastTwoDaughterTransactionFused = false;
  fImpl->fLastTwoDaughterSelectionDirect = false;
  fImpl->fLastCascadeTransactionBatch = false;
  fImpl->fLastCascadeTransactionFused = false;
  fImpl->fLastCascadeGenerationCount = 0u;
  fImpl->fLastGraphTransactionFused = false;
  fImpl->fLastV0TrackRoutingMonitorData = KFParticleGpuV0TrackRoutingMonitorData();
  fImpl->fLastGraphExecutionMonitorData =
    KFParticleGpuGraphExecutionMonitorData();
  fImpl->fLastGraphChannelMonitorData.clear();
  fImpl->fGraphOperationSnapshots.clear();
  fImpl->fLastDecayPlanSelectedCandidates = KFParticleGpuSelectedCandidateRange();
  fImpl->fLastDecayPlanSelectedChannels.clear();
  /**
   * @brief Convert a steady-clock elapsed interval to milliseconds for host-side stage accounting.
   */
  const auto elapsedMilliseconds = [](std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  };
  const auto inputUploadStarted = std::chrono::steady_clock::now();
  KFParticleGpuStepTrace inputTrace("input-upload", true);
  fImpl->fBuffers->UploadInput();
  fImpl->fLastDecayPlanTiming.inputUploadMilliseconds = elapsedMilliseconds(inputUploadStarted);
  inputTrace.Finish();
  const auto constructionStarted = std::chrono::steady_clock::now();
  KFParticleGpuStepTrace constructionTrace("two-track-construction", true);
  constructionTrace.Next("CPU compile decay graph");
  fImpl->fBuffers->SetAllocationTracePhase("construction-setup");
  if (fImpl->fDecayGraphPlanSourceRevision != fImpl->fDecayPlan->Revision()) {
    fImpl->fDecayGraphPlan.Compile(
      MakeDefaultCpuFinderDecayGraphManifest(*fImpl->fDecayPlan));
    fImpl->fDecayGraphPlanSourceRevision = fImpl->fDecayPlan->Revision();
  }
  constructionTrace.Next("CPU/ALLOC/H2D UploadDecayGraphPlan");
  planUploadWaits += fImpl->fBuffers->UploadDecayGraphPlan(fImpl->fDecayGraphPlan) ? 1u : 0u;
  constructionTrace.Next("CPU/ALLOC/H2D UploadGraphOperationPlan");
  planUploadWaits += fImpl->fBuffers->UploadGraphOperationPlan(*fImpl->fDecayPlan) ? 1u : 0u;
  constructionTrace.Next("CPU channel planning and task capacities");
  if (fImpl->fDecayPlan->Empty()) {
    fImpl->fBuffers->ResetCandidates();
    fImpl->fBuffers->ResetV0SelectionResults();
    fImpl->fBuffers->ResetSelectedCandidates();
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult eventResult;
      eventResult.eventIndex = firstEventIndex + eventOffset;
      fImpl->fLastDecayPlanEventResults.push_back(eventResult);
    }
    fImpl->fLastDecayPlanTiming.constructionMilliseconds = elapsedMilliseconds(constructionStarted);
    constructionTrace.Finish();
    const auto downloadStarted = std::chrono::steady_clock::now();
    fImpl->fBuffers->DownloadCandidateOutput(
      0u, 0u, outputMode == KFGpuFullDiagnosticOutput);
    fImpl->fLastDecayPlanTiming.outputDownloadMilliseconds = elapsedMilliseconds(downloadStarted);
    fImpl->CompletePerformanceSnapshot(
      eventCount, capacityGrowthCountBefore, planUploadWaits, 5u,
      0u, 0u, 0u, outputMode == KFGpuFullDiagnosticOutput);
    deviceProfile.Finish();
    return fImpl->fLastDecayPlanResults;
  }

  const unsigned int twoDaughterChannelCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadTwoDaughter);
  if (twoDaughterChannelCount
      != fImpl->fDecayPlan->NumberOfTwoDaughterChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and two-daughter payload tables disagree");
  }
  const unsigned int graphOperationCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadCompositeComposite)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadNeutralDaughter)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadUnaryFinal)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadBinaryFinal);
  if (graphOperationCount
      != fImpl->fDecayPlan->NumberOfGraphOperationChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and later-generation operation tables disagree");
  }
  if (graphOperationCount > 0u) {
    fImpl->fBuffers->EnsureGraphOperationTaskCapacity(taskCapacity);
  }
  unsigned int fusedTaskCapacity = 0u;
  if (twoDaughterChannelCount > 0u) {
    if (taskCapacity > std::numeric_limits<unsigned int>::max()
                         / twoDaughterChannelCount) {
      throw std::overflow_error(
        "KFParticle GPU fused two-daughter task capacity overflows");
    }
    fusedTaskCapacity = taskCapacity * twoDaughterChannelCount;
    const bool planChanged =
      fImpl->fTwoDaughterRoutingPlan.SourceRevision()
        != fImpl->fDecayPlan->Revision();
    if (planChanged) {
      fImpl->fTwoDaughterRoutingPlan.Compile(*fImpl->fDecayPlan);
    }
    planUploadWaits += fImpl->fBuffers->UploadTwoDaughterRoutingPlan(
                         fImpl->fTwoDaughterRoutingPlan, planChanged)
                         ? 1u : 0u;
    fImpl->fBuffers->EnsureTwoDaughterRoutedTaskCapacity(fusedTaskCapacity);
  }

  const unsigned int cascadeChannelCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadCompositeTrack);
  if (cascadeChannelCount
      != fImpl->fDecayPlan->NumberOfV0TrackCascadeChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and composite-track payload tables disagree");
  }
  unsigned int fusedCascadeTaskCapacity = 0u;
  std::chrono::steady_clock::time_point hostAssemblyStarted;
  if (cascadeChannelCount > 0u) {
    if (taskCapacity > std::numeric_limits<unsigned int>::max()
                         / cascadeChannelCount) {
      throw std::overflow_error(
        "KFParticle GPU fused cascade task capacity overflows");
    }
    fusedCascadeTaskCapacity = taskCapacity * cascadeChannelCount;
    const bool planChanged =
      fImpl->fV0TrackRoutingPlan.SourceRevision()
        != fImpl->fDecayPlan->Revision();
    if (planChanged) {
      fImpl->fV0TrackRoutingPlan.Compile(*fImpl->fDecayPlan);
    }
    planUploadWaits += fImpl->fBuffers->UploadV0TrackRoutingPlan(
                         fImpl->fV0TrackRoutingPlan, planChanged)
                         ? 1u : 0u;
    fImpl->fBuffers->EnsureV0TrackRoutedTaskCapacity(
      fusedCascadeTaskCapacity);
  }

  bool transactionBatch = false;
  unsigned int transactionWorkCount = 0u;
  unsigned int maximumTwoDaughterPairCount = 0u;
  std::uint64_t twoDaughterRealPairCount = 0u;
  unsigned int twoDaughterPairTileCount = 0u;
  unsigned int compactCandidateLaunchCapacity = 0u;
  std::vector<KFParticleGpuTaskTile> constructionTiles;
  const char* constructionReference = std::getenv("KFPARTICLE_GPU_CONSTRUCTION_REFERENCE");
  const bool compactConstruction = !constructionReference || std::strcmp(constructionReference, "1") != 0;
  bool compactTransactionOutput = false;
  const unsigned int transactionChannelCount =
    twoDaughterChannelCount + cascadeChannelCount + graphOperationCount;
  if (twoDaughterChannelCount > 0u) {
    const KFParticleGpuTransactionView current = fImpl->fBuffers->HostTransaction();
    constructionTrace.Next("CPU count event-local work and pair tiles");
    transactionBatch = current.Transaction() != nullptr
      && current.Transaction()->decayPlanRevision == fImpl->fDecayPlan->Revision()
      && current.Transaction()->domains.size == eventCount;
    if (transactionBatch) {
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        compactTransactionOutput |=
          (current.Domains()[eventOffset].flags
           & KFParticleGpuDomainCompactOutput) != 0u;
      }
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        const unsigned int eventIndex = firstEventIndex + eventOffset;
        const KFParticleGpuEventDesc& event = fImpl->fBuffers->HostEvents()[eventIndex];
        for (const auto& group : fImpl->fTwoDaughterRoutingPlan.Groups()) {
          const KFParticleGpuRange firstTracks = event.TrackSet(group.firstTrackSet).tracks;
          const KFParticleGpuRange secondTracks = event.TrackSet(group.secondTrackSet).tracks;
          if (secondTracks.size != 0u
              && firstTracks.size > std::numeric_limits<unsigned int>::max()
                                    / secondTracks.size) {
            throw std::overflow_error(
              "KFParticle GPU transaction pair range overflows");
          }
          if (firstTracks.size != 0u && secondTracks.size != 0u) {
            const unsigned int pairCount = firstTracks.size * secondTracks.size;
            maximumTwoDaughterPairCount = std::max(
              maximumTwoDaughterPairCount,
              pairCount);
            twoDaughterRealPairCount += pairCount;
            const unsigned int tiles = KFParticleGpuPairTileCount(pairCount);
            if (twoDaughterPairTileCount
                > std::numeric_limits<unsigned int>::max() - tiles) {
              throw std::overflow_error(
                "KFParticle GPU pair-tile count overflows");
            }
            twoDaughterPairTileCount += tiles;
            if (transactionWorkCount == std::numeric_limits<unsigned int>::max()) {
              throw std::overflow_error(
                "KFParticle GPU transaction work-range count overflows");
            }
            ++transactionWorkCount;
          }
        }
        for (const auto& group : fImpl->fV0TrackRoutingPlan.Groups()) {
          if (event.TrackSet(group.bachelorTrackSet).tracks.size == 0u) { continue; }
          if (transactionWorkCount == std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error(
              "KFParticle GPU transaction work-range count overflows");
          }
          ++transactionWorkCount;
        }
        for (const auto& group : fImpl->fDecayGraphPlan.Groups()) {
          const bool executable =
            group.topology == KFGpuGraphTopologyCompositeComposite
            || group.topology == KFGpuGraphTopologyNeutralDaughter
            || group.topology == KFGpuGraphTopologyUnaryComposite
            || ((group.topology == KFGpuGraphTopologyCompositeTrack
                 || group.topology == KFGpuGraphTopologyTrackComposite)
                && (group.operationMask & KFGpuGraphMatch) != 0u);
          if (!executable) { continue; }
          if (transactionWorkCount == std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error(
              "KFParticle GPU transaction work-range count overflows");
          }
          ++transactionWorkCount;
        }
      }
      if (eventCount > std::numeric_limits<unsigned int>::max()
                         / twoDaughterChannelCount) {
        throw std::overflow_error(
          "KFParticle GPU transaction domain-channel count overflows");
      }
      if (twoDaughterPairTileCount
          > std::numeric_limits<unsigned int>::max()
              / KFParticleGpuPairTileSize) {
        throw std::overflow_error(
          "KFParticle GPU pair-tile launch size overflows");
      }
      constructionTrace.Next("CPU/ALLOC reserve transaction tables");
      const unsigned int domainChannelCount = eventCount * transactionChannelCount;
      KFParticleGpuBufferCapacities requested = fImpl->fBuffers->Capacities();
      requested.workRanges = std::max(requested.workRanges, transactionWorkCount);
      requested.twoDaughterPairTiles = std::max(
        requested.twoDaughterPairTiles, twoDaughterPairTileCount);
      requested.domainChannelStatuses = std::max(
        requested.domainChannelStatuses, domainChannelCount);
      fImpl->fBuffers->EnsureCapacity(requested);
      fImpl->fBuffers->SetTransactionSizes(1u,
                                           eventCount,
                                           transactionWorkCount,
                                           eventCount,
                                           domainChannelCount);
      fImpl->fBuffers->SetTwoDaughterPairTileSize(twoDaughterPairTileCount);
      constructionTrace.Next("CPU fill work ranges and pair tiles");
      const KFParticleGpuTransactionView transaction = fImpl->fBuffers->HostTransaction();
      transaction.Transaction()->workRanges = {0u, transactionWorkCount};
      transaction.Transaction()->channelCount = transactionChannelCount;
      unsigned int workIndex = 0u;
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        const unsigned int eventIndex = firstEventIndex + eventOffset;
        KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
        domain.workRanges.offset = workIndex;
        domain.workRanges.size = 0u;
        domain.channelStatuses = {eventOffset * transactionChannelCount,
                                  transactionChannelCount};
        const KFParticleGpuEventDesc& event = fImpl->fBuffers->HostEvents()[eventIndex];
        for (const auto& group : fImpl->fTwoDaughterRoutingPlan.Groups()) {
          const KFParticleGpuRange firstTracks = event.TrackSet(group.firstTrackSet).tracks;
          const KFParticleGpuRange secondTracks = event.TrackSet(group.secondTrackSet).tracks;
          if (firstTracks.size == 0u || secondTracks.size == 0u) { continue; }
          KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex++];
          work.domainIndex = eventOffset;
          work.family = KFParticleGpuTwoDaughterWork;
          work.generation = 0u;
          work.executionGroup = group.groupIndex;
          work.sourceTracks = firstTracks;
          work.secondarySourceTracks = secondTracks;
          work.sourceCandidates = {};
          work.primaryVertices = event.primaryVertices;
          work.outputReservation = domain.taskReservation;
          ++domain.workRanges.size;
        }
        for (const auto& group : fImpl->fV0TrackRoutingPlan.Groups()) {
          const KFParticleGpuRange bachelors =
            event.TrackSet(group.bachelorTrackSet).tracks;
          if (bachelors.size == 0u) { continue; }
          KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex++];
          work = KFParticleGpuWorkRange{};
          work.domainIndex = eventOffset;
          work.family = KFParticleGpuCompositeTrackWork;
          work.generation = group.generation;
          work.executionGroup = group.groupIndex;
          work.sourceTracks = bachelors;
          work.sourceCandidates = domain.selectedReservation;
          work.primaryVertices = event.primaryVertices;
          work.outputReservation = {
            domain.taskReservation.offset * cascadeChannelCount,
            domain.taskReservation.size * cascadeChannelCount};
          ++domain.workRanges.size;
        }
        for (const auto& group : fImpl->fDecayGraphPlan.Groups()) {
          const bool executable =
            group.topology == KFGpuGraphTopologyCompositeComposite
            || group.topology == KFGpuGraphTopologyNeutralDaughter
            || group.topology == KFGpuGraphTopologyUnaryComposite
            || ((group.topology == KFGpuGraphTopologyCompositeTrack
                 || group.topology == KFGpuGraphTopologyTrackComposite)
                && (group.operationMask & KFGpuGraphMatch) != 0u);
          if (!executable) { continue; }
          KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex++];
          work = KFParticleGpuWorkRange{};
          work.domainIndex = eventOffset;
          work.family = KFParticleGpuGraphOperationWork;
          work.generation = group.generation;
          work.executionGroup = group.groupIndex;
          work.sourceTracks = domain.sourceTracks;
          work.sourceCandidates = domain.candidateReservation;
          work.primaryVertices = event.primaryVertices;
          work.outputReservation = domain.taskReservation;
          ++domain.workRanges.size;
        }
      }
      if (workIndex != transactionWorkCount) {
        throw std::logic_error(
          "KFParticle GPU transaction generation work table is incomplete");
      }
      unsigned int tileIndex = 0u;
      const auto pairTilePlanningStarted = std::chrono::steady_clock::now();
      KFParticleGpuPairTile* pairTiles =
        fImpl->fBuffers->HostTwoDaughterPairTiles();
      for (unsigned int index = 0u; index < transactionWorkCount; ++index) {
        const KFParticleGpuWorkRange& work = transaction.WorkRanges()[index];
        if (work.family != KFParticleGpuTwoDaughterWork) { continue; }
        const unsigned int pairCount =
          work.sourceTracks.size * work.secondarySourceTracks.size;
        if (!KFParticleGpuAppendPairTiles(index,
                                          pairCount,
                                          pairTiles,
                                          twoDaughterPairTileCount,
                                          tileIndex)) {
          throw std::logic_error(
            "KFParticle GPU pair-tile table exceeds its exact capacity");
        }
      }
      if (tileIndex != twoDaughterPairTileCount) {
        throw std::logic_error(
          "KFParticle GPU pair-tile table is incomplete");
      }
      fImpl->fLastDecayPlanTiming.pairTilePlanningMilliseconds +=
        elapsedMilliseconds(pairTilePlanningStarted);
      constructionTrace.Next("CPU initialize statuses and validate transaction");
      for (unsigned int index = 0u; index < transactionWorkCount; ++index) {
        transaction.WorkStatuses()[index] = KFParticleGpuWorkStatus{};
      }
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        const KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
        transaction.DomainStatuses()[domain.statusIndex] = KFParticleGpuDomainStatus{};
      }
      for (unsigned int index = 0u; index < domainChannelCount; ++index) {
        transaction.DomainChannelStatuses()[index] = KFParticleGpuDomainChannelStatus{};
      }
      const auto initialValidationStarted = std::chrono::steady_clock::now();
      const KFParticleGpuTransactionValidation validation =
        KFParticleGpuValidateTransactionConfigured(transaction,
                                         fImpl->fDecayPlan->Revision(),
                                         fImpl->fBuffers->TrackSize(),
                                         fImpl->fBuffers->VertexSize(),
                                         fImpl->fBuffers->EventSize(),
                                         std::max(fusedTaskCapacity,
                                                  fusedCascadeTaskCapacity),
                                         fImpl->fBuffers->Capacities().candidates,
                                         fImpl->fBuffers->Capacities().daughterIds,
                                         fImpl->fBuffers->Capacities().selectedCandidates);
      fImpl->fLastDecayPlanTiming.initialTransactionValidationMilliseconds +=
        elapsedMilliseconds(initialValidationStarted);
      if (validation != KFParticleGpuTransactionValid) {
        throw std::logic_error(
          "KFParticle GPU compiled generation work table violates the transaction contract");
      }
      if (TransactionStageFenceEnabled()) {
        std::fprintf(stderr,
                     "KFPARTICLE_GPU_TRANSACTION reset_contract=%s domains=%u works=%u channels=%u task_capacity=%u candidate_capacity=%u daughter_capacity=%u selected_capacity=%u\n",
                     TransactionResetContract,
                     eventCount, transactionWorkCount, transactionChannelCount,
                     taskCapacity, fImpl->fBuffers->Capacities().candidates,
                     fImpl->fBuffers->Capacities().daughterIds,
                     fImpl->fBuffers->Capacities().selectedCandidates);
        for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
          const KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
          std::fprintf(stderr,
                       "KFPARTICLE_GPU_DOMAIN index=%u event=%u tracks=%u:%u works=%u:%u tasks=%u:%u candidates=%u:%u daughters=%u:%u selected=%u:%u channels=%u:%u\n",
                       eventOffset, domain.inputDescriptorIndex,
                       domain.sourceTracks.offset, domain.sourceTracks.size,
                       domain.workRanges.offset, domain.workRanges.size,
                       domain.taskReservation.offset, domain.taskReservation.size,
                       domain.candidateReservation.offset, domain.candidateReservation.size,
                       domain.daughterReservation.offset, domain.daughterReservation.size,
                       domain.selectedReservation.offset, domain.selectedReservation.size,
                       domain.channelStatuses.offset, domain.channelStatuses.size);
        }
        for (unsigned int workIndex = 0u; workIndex < transactionWorkCount; ++workIndex) {
          const KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex];
          std::fprintf(stderr,
                       "KFPARTICLE_GPU_WORK index=%u domain=%u family=%u generation=%u group=%u tracks=%u:%u secondary=%u:%u candidates=%u:%u vertices=%u:%u output=%u:%u\n",
                       workIndex, work.domainIndex, work.family, work.generation,
                       work.executionGroup,
                       work.sourceTracks.offset, work.sourceTracks.size,
                       work.secondarySourceTracks.offset,
                       work.secondarySourceTracks.size,
                       work.sourceCandidates.offset, work.sourceCandidates.size,
                       work.primaryVertices.offset, work.primaryVertices.size,
                       work.outputReservation.offset, work.outputReservation.size);
        }
        std::fflush(stderr);
      }
      constructionTrace.Next("H2D initial transaction tables");
      fImpl->fBuffers->UploadTransaction();
    }
  }

  const bool retainDetailedSnapshots =
    !transactionBatch || outputMode == KFGpuFullDiagnosticOutput;
  const bool downloadWorkStatuses =
    retainDetailedSnapshots || fImpl->fPerformanceMonitoringEnabled;
  const unsigned int collectDiagnosticCounters =
    downloadWorkStatuses ? 1u : 0u;

  constructionTrace.Next("CPU/H2D initial output reset");
  const bool deviceBatchInitialization =
    transactionBatch
    && fImpl->fAdministrativeResetStrategy
         == KFGpuAdministrativeResetDeviceBatch;
  if (!deviceBatchInitialization) {
    fImpl->fBuffers->ResetCandidates();
    fImpl->fBuffers->ResetV0SelectionResults();
    fImpl->fBuffers->ResetSelectedCandidates();
  }

  // All revision-owned tables and transaction capacities are now stable.
  // Publish one coherent pointer/size snapshot for every state-backed action.
  constructionTrace.Next("CPU/GPU initial kernel state publication");
  fImpl->PublishKernelState();

  constructionTrace.Next("CPU prepare batch launch");
  fImpl->fFusedTwoDaughterSnapshots.resize(eventCount);
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  if (transactionBatch) {
    const std::uint64_t rectangularRouteThreads =
      static_cast<std::uint64_t>(transactionWorkCount)
      * maximumTwoDaughterPairCount;
    const std::uint64_t rectangularRouteLanes =
      (rectangularRouteThreads / KFParticleGpuPairTileSize
       + (rectangularRouteThreads % KFParticleGpuPairTileSize != 0u ? 1u : 0u))
      * KFParticleGpuPairTileSize;
    const std::uint64_t scheduledRouteLanes =
      static_cast<std::uint64_t>(twoDaughterPairTileCount)
      * KFParticleGpuPairTileSize;
    const std::uint64_t avoidedRouteLanes =
      rectangularRouteLanes > scheduledRouteLanes
        ? rectangularRouteLanes - scheduledRouteLanes : 0u;
    fImpl->fLastPerformanceSnapshot.twoDaughterRealPairs =
      twoDaughterRealPairCount;
    fImpl->fLastPerformanceSnapshot.twoDaughterPairTiles =
      twoDaughterPairTileCount;
    fImpl->fLastPerformanceSnapshot.twoDaughterScheduledLanes =
      scheduledRouteLanes;
    fImpl->fLastPerformanceSnapshot.twoDaughterTailLanes =
      scheduledRouteLanes - twoDaughterRealPairCount;
    fImpl->fLastPerformanceSnapshot.twoDaughterRectangularLanesAvoided =
      avoidedRouteLanes;
    fImpl->fLastPerformanceSnapshot.twoDaughterPairTileBytes =
      static_cast<std::uint64_t>(twoDaughterPairTileCount)
      * sizeof(KFParticleGpuPairTile);
    /**
     * @verbatim
     * First-generation construction
     *
     * STAGED: [Route pairs] -> retained tasks -> [Construct candidates]
     * FUSED:  [Route pairs + Construct candidates]
     *
     * Selection
     *
     * STAGED: [Construct] -> [Evaluate selection] -> [Prepare] -> [Scatter]
     * DIRECT: [Construct + Evaluate selection]   -> [Prepare] -> [Scatter]
     *
     * Fused construction is used only when eligible:
     *   requested FUSED && taskCapacity <= MaximumFusedTwoDaughterTasks
     *                   && !compactTransactionOutput
     *
     * DIRECT selection does not eliminate preparation or scatter.
     * @endverbatim
     */
    fImpl->fLastTwoDaughterTransactionBatch = true;
    fImpl->fLastTwoDaughterTransactionFused =
      fImpl->fTwoDaughterTransactionStrategy == KFGpuTwoDaughterTransactionFused
      && taskCapacity <= MaximumFusedTwoDaughterTasks
      && !compactTransactionOutput;
    fImpl->fLastTwoDaughterSelectionDirect =
      fImpl->fTwoDaughterSelectionStrategy == KFGpuTwoDaughterSelectionDirect;
    if (TransactionStageFenceEnabled()) {
      std::fprintf(stderr,
                   "KFPARTICLE_GPU_STRATEGY two_daughter=%s task_capacity=%u fused_limit=%u real_pairs=%llu pair_tiles=%u scheduled_lanes=%llu tail_lanes=%llu rectangular_lanes_avoided=%llu\n",
                   fImpl->fLastTwoDaughterTransactionFused ? "fused" : "staged",
                   taskCapacity, MaximumFusedTwoDaughterTasks,
                   static_cast<unsigned long long>(twoDaughterRealPairCount),
                   twoDaughterPairTileCount,
                   static_cast<unsigned long long>(scheduledRouteLanes),
                   static_cast<unsigned long long>(scheduledRouteLanes
                                                   - twoDaughterRealPairCount),
                   static_cast<unsigned long long>(avoidedRouteLanes));
      std::fflush(stderr);
    }
    const unsigned int evaluateSelection =
      fImpl->fLastTwoDaughterSelectionDirect ? 1u : 0u;
    constructionTrace.Next("KERNEL ResetBatchOutputCounters");
    fImpl->fQueue.launch<KFParticleGpuResetBatchOutputCounters>(
      xpu::n_threads(1u), fImpl->fBuffers->DeviceCandidates(),
      fImpl->fBuffers->DeviceSelectedCandidates());
    TransactionStageFence(fImpl->fQueue, "two-daughter-reset");
    constructionTrace.Next("DEBUG optional transaction probes");
    fImpl->RunTwoDaughterTransactionDiagnostics();
    if (twoDaughterRealPairCount > 0u) {
      if (fImpl->fLastTwoDaughterTransactionFused) {
        constructionTrace.Next("KERNEL RouteConstructTwoDaughterBatch");
        fImpl->fQueue.launch<KFParticleGpuRouteConstructTwoDaughterBatch>(
          xpu::n_blocks(transactionWorkCount),
          fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(), evaluateSelection,
          collectDiagnosticCounters);
        TransactionStageFence(fImpl->fQueue, "two-daughter-route-construct");
      }
      else {
        constructionTrace.Next("KERNEL RouteTwoDaughterBatch");
        fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterBatch>(
          xpu::n_threads(twoDaughterPairTileCount
                         * KFParticleGpuPairTileSize),
          fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(),
          fImpl->fBuffers->DeviceTwoDaughterPairTiles(),
          twoDaughterPairTileCount, collectDiagnosticCounters);
        TransactionStageFence(fImpl->fQueue, "two-daughter-route");
      }
    }
    if (compactTransactionOutput) {
      if (fImpl->fLastTwoDaughterTransactionFused) {
        throw std::logic_error(
          "KFParticle GPU compact output requires retained routing tasks");
      }

      /**
       * @verbatim
       * [GPU] Route pairs and retain accepted tasks
       *          |                            |
       *          | small status payload       | task records stay on GPU
       *          v                            |
       * [D2H + WAIT] Read retained counts      |
       *          |                            |
       * [CPU] Plan output ranges + headroom    |
       *          |                            |
       * [ALLOC] Ensure output capacities      |
       *          |                            |
       * [H2D] Update tables; publish views     |
       *          |                            |
       *          +----------------------------+
       *          v
       * [GPU] Construct candidates from retained tasks
       *
       * No second pair-routing pass. Dependent-stage space is a bounded
       * reservation, not an exact prediction; overflow remains reportable.
       * @endverbatim
       */
      const std::uint64_t countingReadbackBytes =
        fImpl->fBuffers->TransactionStatusPayloadBytes(false, false, false);
      const auto countingReadbackStarted = std::chrono::steady_clock::now();
      constructionTrace.Next("D2H retained-task counting status");
      fImpl->fBuffers->DownloadTransactionStatus(false, false, false);
      fImpl->fLastDecayPlanTiming.countingReadbackBytes +=
        countingReadbackBytes;
      fImpl->fLastDecayPlanTiming.countingReadbackMilliseconds +=
        elapsedMilliseconds(countingReadbackStarted);
      const auto denseOutputPlanningStarted = std::chrono::steady_clock::now();
      constructionTrace.Next("CPU dense output sizes and task tiles");
      fImpl->fBuffers->SetAllocationTracePhase("construction-output");
      if (fImpl->fPerformanceMonitoringEnabled) {
        ++fImpl->fLastPerformanceSnapshot.monitoringReadbacks;
        ++fImpl->fLastPerformanceSnapshot.monitoringReadbackBuffers;
        fImpl->fLastPerformanceSnapshot.transactionStatusBytes +=
          countingReadbackBytes;
      }
      KFParticleGpuTransactionView hostTransaction =
        fImpl->fBuffers->HostTransaction();
      unsigned int candidateCursor = 0u;
      unsigned int daughterCursor = 0u;
      unsigned int selectedCursor = 0u;
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        KFParticleGpuDomainDesc& domain = hostTransaction.Domains()[eventOffset];
        const KFParticleGpuDomainStatus& status =
          hostTransaction.DomainStatuses()[domain.statusIndex];
        const unsigned int retained = status.storedTasks;
        // One slot per retained two-daughter task plus an equally sized arena
        // for dependent generations. Eight slots avoid pathological tiny
        // allocations. Overflow remains observable and is retried by the
        // existing batch-splitting policy.
        const unsigned int dependentFactor = retained < 64u ? 4u : 1u;
        const unsigned int dependentHeadroom = retained == 0u
          ? 0u
          : CheckedAddCapacity(
              CheckedMultiplyCapacity(
                retained, dependentFactor,
                "KFParticle GPU compact dependent headroom overflows"),
              8u, "KFParticle GPU compact candidate headroom overflows");
        const unsigned int candidates = CheckedAddCapacity(
          retained, dependentHeadroom,
          "KFParticle GPU compact candidate reservation overflows");
        const unsigned int daughters = CheckedMultiplyCapacity(
          candidates, 8u,
          "KFParticle GPU compact daughter reservation overflows");
        domain.candidateReservation = {candidateCursor, candidates};
        domain.daughterReservation = {daughterCursor, daughters};
        domain.selectedReservation = {selectedCursor, candidates};
        candidateCursor = CheckedAddCapacity(
          candidateCursor, candidates,
          "KFParticle GPU compact candidate prefix overflows");
        daughterCursor = CheckedAddCapacity(
          daughterCursor, daughters,
          "KFParticle GPU compact daughter prefix overflows");
        selectedCursor = CheckedAddCapacity(
          selectedCursor, candidates,
          "KFParticle GPU compact selected prefix overflows");

        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          KFParticleGpuWorkRange& work =
            hostTransaction.WorkRanges()[domain.workRanges.offset + local];
          if (work.family == KFParticleGpuCompositeTrackWork) {
            work.sourceCandidates = domain.selectedReservation;
          }
          else if (work.family == KFParticleGpuGraphOperationWork) {
            work.sourceCandidates = domain.candidateReservation;
          }
        }
      }

      if (compactConstruction) {
        constructionTiles = KFParticleGpuBuildTaskTiles(
          hostTransaction, fusedTaskCapacity);
      }
      KFParticleGpuBufferCapacities requested = fImpl->fBuffers->Capacities();
      requested.twoDaughterTaskTiles = std::max(requested.twoDaughterTaskTiles,
        static_cast<unsigned int>(constructionTiles.size()));
      requested.candidates = std::max(requested.candidates,
                                      std::max(candidateCursor, 1u));
      requested.daughterIds = std::max(requested.daughterIds,
                                       std::max(daughterCursor, 2u));
      requested.selectedCandidates = std::max(requested.selectedCandidates,
                                              std::max(selectedCursor, 1u));
      constructionTrace.Next("ALLOC dense output capacities");
      fImpl->fBuffers->EnsureCapacity(requested);
      compactCandidateLaunchCapacity = candidateCursor;
      fImpl->fLastDecayPlanTiming.denseOutputPlanningMilliseconds +=
        elapsedMilliseconds(denseOutputPlanningStarted);

      constructionTrace.Next("CPU validate dense output transaction");
      const auto compactValidationStarted = std::chrono::steady_clock::now();
      const KFParticleGpuTransactionValidation compactValidation =
        KFParticleGpuValidateTransactionConfigured(
          hostTransaction, fImpl->fDecayPlan->Revision(),
          fImpl->fBuffers->TrackSize(), fImpl->fBuffers->VertexSize(),
          fImpl->fBuffers->EventSize(),
          std::max(fusedTaskCapacity,
                   fusedCascadeTaskCapacity),
          fImpl->fBuffers->Capacities().candidates,
          fImpl->fBuffers->Capacities().daughterIds,
          fImpl->fBuffers->Capacities().selectedCandidates);
      fImpl->fLastDecayPlanTiming.compactOutputValidationMilliseconds +=
        elapsedMilliseconds(compactValidationStarted);
      if (compactValidation != KFParticleGpuTransactionValid) {
        throw std::logic_error(
          "KFParticle GPU compact output plan violates the transaction contract");
      }
      constructionTrace.Next("CPU/H2D/KERNEL reset dense output");
      fImpl->fBuffers->ResetCandidates();
      fImpl->fBuffers->ResetV0SelectionResults();
      fImpl->fBuffers->ResetSelectedCandidates();
      constructionTrace.Next("CPU/H2D construction task tiles");
      if (!constructionTiles.empty()) {
        std::copy(constructionTiles.begin(), constructionTiles.end(),
                  fImpl->fBuffers->HostTwoDaughterTaskTiles());
        fImpl->fBuffers->UploadTwoDaughterTaskTiles(constructionTiles.size());
      }
      // Update reservations without overwriting device-owned routing counters
      // from host status mirrors that were deliberately not downloaded.
      constructionTrace.Next("H2D updated transaction tables");
      fImpl->fBuffers->UploadTransaction(false);
      constructionTrace.Next("CPU/GPU updated kernel state publication");
      fImpl->PublishKernelState();
      if (TransactionStageFenceEnabled()) {
        std::fprintf(stderr,
                     "KFPARTICLE_GPU_COMPACT_OUTPUT domains=%u retained=%u candidates=%u daughters=%u selected=%u\n",
                     eventCount, selectedCursor, candidateCursor,
                     daughterCursor, selectedCursor);
        std::fflush(stderr);
      }
    }
    if (!fImpl->fLastTwoDaughterTransactionFused) {
      if (compactTransactionOutput && compactConstruction) {
        constructionTrace.Next("KERNEL ConstructTwoDaughterTaskTiles");
        fImpl->fQueue.launch<KFParticleGpuConstructTwoDaughterTaskTiles>(
          xpu::n_blocks(std::max(1u, static_cast<unsigned int>(constructionTiles.size()))),
          fImpl->fKernels, fImpl->fBuffers->DeviceTransaction(), evaluateSelection,
          fImpl->fBuffers->DeviceTwoDaughterTaskTiles(), static_cast<unsigned int>(constructionTiles.size()));
      }
      else {
        constructionTrace.Next("KERNEL ConstructTwoDaughterBatch");
        fImpl->fQueue.launch<KFParticleGpuConstructTwoDaughterBatch>(
          xpu::n_threads(fusedTaskCapacity), fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(), evaluateSelection);
      }
      constructionTrace.Next("CPU construction monitoring");
      if (fImpl->fPerformanceMonitoringEnabled && KFParticleGpuReportRequested()) {
        std::fprintf(stdout,
          "KFP_CONSTRUCTION_MAP contract=1 compact=%u tiles=%zu lanes=%llu reference_lanes=%u upload_bytes=%zu\n",
          compactTransactionOutput && compactConstruction ? 1u : 0u,
          constructionTiles.size(),
          static_cast<unsigned long long>(constructionTiles.size()) * 64u,
          fusedTaskCapacity, constructionTiles.size() * sizeof(KFParticleGpuTaskTile));
      }
      TransactionStageFence(fImpl->fQueue, "two-daughter-construct");
    }
    const auto selectionStarted = std::chrono::steady_clock::now();
    if (!fImpl->fLastTwoDaughterSelectionDirect) {
      constructionTrace.Next("KERNEL EvaluateTwoDaughterBatchSelection");
      fImpl->fQueue.launch<KFParticleGpuEvaluateTwoDaughterBatchSelection>(
        xpu::n_threads(compactCandidateLaunchCapacity != 0u
                         ? compactCandidateLaunchCapacity
                         : fImpl->fBuffers->Capacities().candidates),
        fImpl->fKernels,
        fImpl->fBuffers->DeviceTransaction());
      TransactionStageFence(fImpl->fQueue, "two-daughter-selection-evaluate");
    }
    constructionTrace.Next("KERNEL PrepareTwoDaughterBatchSelection");
    fImpl->fQueue.launch<KFParticleGpuPrepareTwoDaughterBatchSelection>(
      xpu::n_threads(eventCount), fImpl->fKernels,
      fImpl->fBuffers->DeviceTransaction(),
      fImpl->fBuffers->DeviceSelectedCandidates());
    TransactionStageFence(fImpl->fQueue, "two-daughter-selection-prepare");
    constructionTrace.Next("KERNEL ScatterTwoDaughterBatchSelection");
    fImpl->fQueue.launch<KFParticleGpuScatterTwoDaughterBatchSelection>(
      xpu::n_blocks(eventCount),
      fImpl->fKernels,
      fImpl->fBuffers->DeviceTransaction(),
      fImpl->fBuffers->DeviceSelectedCandidates());
    TransactionStageFence(fImpl->fQueue, "two-daughter-selection-scatter");
    if (transactionWorkCount > 0u) {
      constructionTrace.Next("KERNEL ResolveTransactionWorkSources");
      fImpl->fQueue.launch<KFParticleGpuResolveTransactionWorkSources>(
        xpu::n_threads(transactionWorkCount),
        fImpl->fBuffers->DeviceTransaction(), 0u);
    }
    TransactionStageFence(fImpl->fQueue, "two-daughter-source-resolve");
    fImpl->fLastDecayPlanTiming.selectionMilliseconds +=
      elapsedMilliseconds(selectionStarted);
    ++fImpl->fLastTwoDaughterRoutingMonitorData.selectionLaunches;

    constructionTrace.Next("CPU collect two-daughter snapshots");
    const KFParticleGpuTransactionView transaction = fImpl->fBuffers->HostTransaction();
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult eventResult;
      eventResult.eventIndex = firstEventIndex + eventOffset;
      eventResult.channelOffset = eventOffset * twoDaughterChannelCount;
      eventResult.channelCount = twoDaughterChannelCount;
      fImpl->fLastDecayPlanEventResults.push_back(eventResult);
      KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
        fImpl->fFusedTwoDaughterSnapshots[eventOffset];
      if (retainDetailedSnapshots) {
        snapshot.channelVisited.resize(twoDaughterChannelCount);
        snapshot.channelAccepted.resize(twoDaughterChannelCount);
        snapshot.channelStored.resize(twoDaughterChannelCount);
        snapshot.channelConstructed.resize(twoDaughterChannelCount);
        snapshot.selectionAccepted.resize(twoDaughterChannelCount);
        snapshot.selectionStored.resize(twoDaughterChannelCount);
        snapshot.selectionOffsets.resize(twoDaughterChannelCount);
      }
      const KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
      snapshot.candidateBegin = domain.candidateReservation.offset;
      snapshot.daughterBegin = domain.daughterReservation.offset;
      snapshot.selectedBegin = 0u;
      snapshot.candidateEnd = 0u;
      snapshot.daughterEnd = 0u;
      snapshot.selectedEnd = 0u;
    }
  }
  else {
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    constructionTrace.Next("CPU/D2H legacy event setup and counters");
    const unsigned int eventIndex = firstEventIndex + eventOffset;
    KFParticleGpuDecayPlanEventResult eventResult;
    eventResult.eventIndex = eventIndex;
    eventResult.channelOffset = eventOffset * twoDaughterChannelCount;
    eventResult.channelCount = twoDaughterChannelCount;
    fImpl->fLastDecayPlanEventResults.push_back(eventResult);

    if (twoDaughterChannelCount == 0u) {
      continue;
    }
    KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
      fImpl->fFusedTwoDaughterSnapshots[eventOffset];
    if (retainDetailedSnapshots) {
      snapshot.channelVisited.resize(twoDaughterChannelCount);
      snapshot.channelAccepted.resize(twoDaughterChannelCount);
      snapshot.channelStored.resize(twoDaughterChannelCount);
      snapshot.channelConstructed.resize(twoDaughterChannelCount);
      snapshot.selectionAccepted.resize(twoDaughterChannelCount);
      snapshot.selectionStored.resize(twoDaughterChannelCount);
      snapshot.selectionOffsets.resize(twoDaughterChannelCount);
    }
    fImpl->fQueue.memcpy(
      &snapshot.candidateBegin,
      fImpl->fBuffers->DeviceCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.daughterBegin,
      fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedBegin,
      fImpl->fBuffers->DeviceSelectedCandidates().SizeData(),
      sizeof(unsigned int));

    constructionTrace.Next("KERNEL ResetTwoDaughterGenerationState");
    fImpl->fQueue.launch<KFParticleGpuResetTwoDaughterGenerationState>(
      xpu::n_threads(twoDaughterChannelCount));

    constructionTrace.Next("CPU legacy routing group planning");
    const KFParticleGpuEventDesc& event =
      fImpl->fBuffers->HostEvents()[eventIndex];
    for (const KFParticleGpuTwoDaughterExecutionGroup& group :
         fImpl->fTwoDaughterRoutingPlan.Groups()) {
      const KFParticleGpuRange firstTracks =
        event.TrackSet(group.firstTrackSet).tracks;
      const KFParticleGpuRange secondTracks =
        event.TrackSet(group.secondTrackSet).tracks;
      if (secondTracks.size != 0u
          && firstTracks.size > std::numeric_limits<unsigned int>::max()
                                / secondTracks.size) {
        throw std::overflow_error(
          "KFParticle GPU fused two-daughter pair count overflows");
      }
      const unsigned int pairCount = firstTracks.size * secondTracks.size;
      if (pairCount == 0u) {
        continue;
      }
      ++snapshot.groupLaunches;
      constructionTrace.Next("KERNEL RouteTwoDaughterTasksBlockScan");
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        fusedTaskCapacity);
      constructionTrace.Next("CPU legacy next routing group");
    }
    constructionTrace.Next("KERNEL TwoDaughterRoutedCandidatePoolKernel");
    fImpl->fQueue.launch<KFParticleGpuTwoDaughterRoutedCandidatePoolKernel>(
      xpu::n_threads(fusedTaskCapacity),
      fusedTaskCapacity);

    const auto selectionStarted = std::chrono::steady_clock::now();
    constructionTrace.Next("KERNEL EvaluateTwoDaughterGenerationSelection");
    fImpl->fQueue.launch<KFParticleGpuEvaluateTwoDaughterGenerationSelection>(
      xpu::n_threads(fImpl->fBuffers->Capacities().candidates),
      eventIndex);
    constructionTrace.Next("KERNEL PrepareTwoDaughterSelectionSegments");
    fImpl->fQueue.launch<KFParticleGpuPrepareTwoDaughterSelectionSegments>(
      xpu::n_threads(1u));
    constructionTrace.Next("KERNEL ScatterTwoDaughterGenerationSelection");
    fImpl->fQueue.launch<KFParticleGpuScatterTwoDaughterGenerationSelection>(
      xpu::n_threads(fImpl->fBuffers->Capacities().candidates),
      eventIndex);
    constructionTrace.Next("D2H legacy selection and routing counters");
    fImpl->fLastDecayPlanTiming.selectionMilliseconds +=
      elapsedMilliseconds(selectionStarted);
    ++fImpl->fLastTwoDaughterRoutingMonitorData.selectionLaunches;

    const KFParticleGpuTwoDaughterSelectionWorkspaceView selection =
      fImpl->fBuffers->DeviceTwoDaughterSelectionWorkspace();
    fImpl->fQueue.memcpy(
      &snapshot.visitedPairs,
      storage.fTwoDaughterRoutingVisitedPairCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.activeChannelBits,
      storage.fTwoDaughterRoutingActiveBitCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.acceptedTasks,
      storage.fTwoDaughterRoutedAcceptedTaskCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.storedTasks,
      storage.fTwoDaughterRoutedStoredTaskCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.blockReservations,
      storage.fTwoDaughterRoutingBlockReservationCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.taskOverflowFlags,
      storage.fTwoDaughterRoutedTaskOverflowFlags.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelVisited.data(),
      storage.fTwoDaughterRoutingChannelVisitedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelAccepted.data(),
      storage.fTwoDaughterRoutingChannelAcceptedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelStored.data(),
      storage.fTwoDaughterRoutingChannelStoredCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelConstructed.data(),
      storage.fTwoDaughterRoutingChannelConstructedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionAccepted.data(),
      selection.AcceptedData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionStored.data(),
      selection.StoredData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionOffsets.data(),
      selection.OffsetsData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.candidateEnd,
      fImpl->fBuffers->DeviceCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.daughterEnd,
      fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedEnd,
      fImpl->fBuffers->DeviceSelectedCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.candidateOverflowFlags,
      fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedOverflowFlags,
      fImpl->fBuffers->DeviceSelectedCandidates().OverflowFlagsData(),
      sizeof(unsigned int));
  }
  }
  fImpl->fLastDecayPlanTiming.constructionMilliseconds =
    elapsedMilliseconds(constructionStarted)
    - fImpl->fLastDecayPlanTiming.selectionMilliseconds;
  constructionTrace.Finish();

  const auto cascadeStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->SetAllocationTracePhase("dependent-graph");
  const KFParticleGpuTransactionView compiledTransaction =
    fImpl->fBuffers->HostTransaction();
  unsigned int transactionGraphGenerationCount = 0u;
  /**
   * @brief Queue executable dependent graph groups, by generation in transaction mode or by event/group otherwise.
   * Uses device source resolution between generations; the enclosing method performs the completion wait.
   */
  const auto queueGraphOperations = [&]() {
    if (graphOperationCount == 0u) { return; }
    std::vector<std::pair<unsigned int, KFParticleGpuWorkRange>> scheduled;
    if (transactionBatch) {
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        const KFParticleGpuDomainDesc& domain =
          compiledTransaction.Domains()[eventOffset];
        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          const KFParticleGpuWorkRange& work =
            compiledTransaction.WorkRanges()[domain.workRanges.offset + local];
          if (work.family == KFParticleGpuGraphOperationWork) {
            scheduled.emplace_back(domain.inputDescriptorIndex, work);
          }
        }
      }
    }
    else {
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        for (const KFParticleGpuGraphExecutionGroup& group :
             fImpl->fDecayGraphPlan.Groups()) {
          const bool executable =
            group.topology == KFGpuGraphTopologyCompositeComposite
            || group.topology == KFGpuGraphTopologyNeutralDaughter
            || group.topology == KFGpuGraphTopologyUnaryComposite
            || ((group.topology == KFGpuGraphTopologyCompositeTrack
                 || group.topology == KFGpuGraphTopologyTrackComposite)
                && (group.operationMask & KFGpuGraphMatch) != 0u);
          if (!executable) { continue; }
          KFParticleGpuWorkRange work;
          work.domainIndex = eventOffset;
          work.family = KFParticleGpuGraphOperationWork;
          work.generation = group.generation;
          work.executionGroup = group.groupIndex;
          scheduled.emplace_back(firstEventIndex + eventOffset, work);
        }
      }
    }
    if (transactionBatch) {
      fImpl->fLastGraphTransactionFused =
        fImpl->fGraphTransactionStrategy == KFGpuGraphTransactionFused;
      unsigned int maximumGeneration = 0u;
      for (const auto& scheduledWork : scheduled) {
        maximumGeneration =
          std::max(maximumGeneration, scheduledWork.second.generation);
      }
      for (unsigned int generation = 1u;
           generation <= maximumGeneration; ++generation) {
        bool active = false;
        for (const auto& scheduledWork : scheduled) {
          active |= scheduledWork.second.generation == generation;
        }
        if (!active) { continue; }
        ++transactionGraphGenerationCount;
        fImpl->fQueue.launch<KFParticleGpuResetGraphOperationBatch>(
          xpu::n_blocks(1u), fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(), generation);
        TransactionStageFence(fImpl->fQueue, "graph-reset", generation);
        fImpl->fQueue.launch<KFParticleGpuRouteGraphOperationBatch>(
          xpu::n_blocks(transactionWorkCount),
          fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(), generation,
          twoDaughterChannelCount + cascadeChannelCount,
          fImpl->fLastGraphTransactionFused ? 1u : 0u,
          collectDiagnosticCounters);
        TransactionStageFence(fImpl->fQueue, "graph-route-construct", generation);
        if (!fImpl->fLastGraphTransactionFused) {
          fImpl->fQueue.launch<KFParticleGpuExecuteGraphOperationBatch>(
            xpu::n_blocks(eventCount), fImpl->fKernels,
            fImpl->fBuffers->DeviceTransaction(), generation,
            twoDaughterChannelCount + cascadeChannelCount);
          TransactionStageFence(fImpl->fQueue, "graph-construct", generation);
        }
        fImpl->fQueue.launch<KFParticleGpuResolveTransactionWorkSources>(
          xpu::n_threads(transactionWorkCount),
          fImpl->fBuffers->DeviceTransaction(), generation);
        TransactionStageFence(fImpl->fQueue, "graph-source-resolve", generation);
      }
      return;
    }
    fImpl->fGraphOperationSnapshots.resize(scheduled.size());
    unsigned int snapshotIndex = 0u;
    const unsigned int descriptorCount =
      fImpl->fBuffers->GraphOperationDescriptorSize();
    const unsigned int candidateCapacity =
      fImpl->fBuffers->Capacities().candidates;
    const unsigned int sourceCapacity = std::max(
      candidateCapacity, fImpl->fBuffers->Capacities().tracks);
    const KFParticleGpuDeviceStorage& graphStorage =
      fImpl->fBuffers->Storage();
    for (const auto& scheduledWork : scheduled) {
        const unsigned int eventIndex = scheduledWork.first;
        const KFParticleGpuWorkRange& work = scheduledWork.second;
        if (work.domainIndex >= eventCount
            || work.executionGroup >= fImpl->fDecayGraphPlan.Groups().size()) {
          throw std::logic_error(
            "KFParticle GPU graph work range violates its compiled domain");
        }
        const KFParticleGpuGraphExecutionGroup& group =
          fImpl->fDecayGraphPlan.Groups()[work.executionGroup];
        if (group.generation != work.generation) {
          throw std::logic_error(
            "KFParticle GPU graph work range generation is inconsistent");
        }
        if (group.nodeCount > 0u
            && sourceCapacity
                 > std::numeric_limits<unsigned int>::max() / group.nodeCount) {
          throw std::overflow_error(
            "KFParticle GPU graph routing launch size overflows");
        }
        KFParticleGpuGraphOperationSnapshot& snapshot =
          fImpl->fGraphOperationSnapshots[snapshotIndex++];
        snapshot.eventIndex = eventIndex;
        snapshot.groupIndex = group.groupIndex;
        snapshot.generation = group.generation;
        snapshot.channelVisited.resize(descriptorCount);
        snapshot.channelAccepted.resize(descriptorCount);
        snapshot.channelStored.resize(descriptorCount);
        snapshot.channelConstructed.resize(descriptorCount);
        snapshot.channelRejected.resize(descriptorCount);

        fImpl->fQueue.memcpy(
          &snapshot.candidateBegin,
          fImpl->fBuffers->DeviceCandidates().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.daughterBegin,
          fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.launch<KFParticleGpuResetGraphOperationGeneration>(
          xpu::n_threads(descriptorCount > 0u ? descriptorCount : 1u));
        fImpl->fQueue.launch<KFParticleGpuRouteGraphOperationTasks>(
          xpu::n_threads(group.nodeCount * sourceCapacity),
          eventIndex,
          group.groupIndex,
          sourceCapacity,
          taskCapacity);
        fImpl->fQueue.launch<KFParticleGpuExecuteRoutedGraphOperations>(
          xpu::n_threads(taskCapacity),
          taskCapacity);

        fImpl->fQueue.memcpy(
          &snapshot.visited,
          graphStorage.fGraphOperationVisitedCombinations.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.accepted,
          graphStorage.fGraphOperationAcceptedTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.stored,
          graphStorage.fGraphOperationStoredTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.constructed,
          graphStorage.fGraphOperationConstructedCandidates.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.rejected,
          graphStorage.fGraphOperationRejectedTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.overflowFlags,
          graphStorage.fGraphOperationOverflowFlags.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelVisited.data(),
          graphStorage.fGraphOperationChannelVisitedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelAccepted.data(),
          graphStorage.fGraphOperationChannelAcceptedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelStored.data(),
          graphStorage.fGraphOperationChannelStoredCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelConstructed.data(),
          graphStorage.fGraphOperationChannelConstructedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelRejected.data(),
          graphStorage.fGraphOperationChannelRejectedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.candidateEnd,
          fImpl->fBuffers->DeviceCandidates().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.daughterEnd,
          fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.candidateOverflowFlags,
          fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
          sizeof(unsigned int));
    }
  };
  /**
   * @brief Gather already downloaded domain/channel statuses into first-generation event snapshots.
   * Requires the completion wait; does not initiate readback. End fields initially accumulate counts.
   */
  const auto assembleTransactionFirstGeneration = [&]() {
    if (!transactionBatch) { return; }
    const KFParticleGpuTransactionView transaction =
      fImpl->fBuffers->HostTransaction();
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(fImpl->fBuffers->HostCandidates());
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
        fImpl->fFusedTwoDaughterSnapshots[eventOffset];
      const KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
      const KFParticleGpuDomainStatus& domainStatus =
        transaction.DomainStatuses()[domain.statusIndex];
      snapshot.visitedPairs = domainStatus.visitedPairs;
      snapshot.activeChannelBits = domainStatus.activeChannelBits;
      snapshot.acceptedTasks = domainStatus.tasks;
      snapshot.storedTasks = domainStatus.storedTasks;
      snapshot.blockReservations = domainStatus.blockReservations;
      snapshot.taskOverflowFlags = domainStatus.overflowFlags;
      for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
        const unsigned int workIndex = domain.workRanges.offset + local;
        const KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex];
        if (work.family != KFParticleGpuTwoDaughterWork) { continue; }
        ++snapshot.groupLaunches;
      }
      const KFParticleGpuDomainChannelStatus* channels =
        transaction.DomainChannelStatuses() + domain.channelStatuses.offset;
      for (unsigned int channel = 0u; channel < twoDaughterChannelCount; ++channel) {
        if (retainDetailedSnapshots) {
          snapshot.channelVisited[channel] = channels[channel].visited;
          snapshot.channelAccepted[channel] = channels[channel].accepted;
          snapshot.channelStored[channel] = channels[channel].stored;
          snapshot.channelConstructed[channel] = channels[channel].constructed;
          snapshot.selectionAccepted[channel] = channels[channel].selectionAccepted;
          snapshot.selectionStored[channel] = channels[channel].selectionStored;
          snapshot.selectionOffsets[channel] = channels[channel].selectionOffset;
        }
        if (channel == 0u) {
          snapshot.selectedBegin = channels[channel].selectionOffset;
        }
        snapshot.candidateEnd += channels[channel].constructed;
        snapshot.daughterEnd += 2u * channels[channel].constructed;
        snapshot.selectedEnd += channels[channel].selectionStored;
      }
      snapshot.candidateOverflowFlags =
        domainStatus.overflowFlags | candidates.OverflowFlags();
      snapshot.selectedOverflowFlags = selected.OverflowFlags();
    }
  };

  if (cascadeChannelCount > 0u) {
    fImpl->fFusedCascadeSnapshots.resize(eventCount);
    const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

    if (transactionBatch) {
      fImpl->fLastCascadeTransactionBatch = true;
      fImpl->fLastCascadeTransactionFused =
        fImpl->fCascadeTransactionStrategy == KFGpuCascadeTransactionFused;
      std::vector<unsigned int> activeGenerations;
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        KFParticleGpuFusedCascadeSnapshot& snapshot =
          fImpl->fFusedCascadeSnapshots[eventOffset];
        if (retainDetailedSnapshots) {
          snapshot.channelVisited.resize(cascadeChannelCount);
          snapshot.channelAccepted.resize(cascadeChannelCount);
          snapshot.channelStored.resize(cascadeChannelCount);
          snapshot.channelConstructed.resize(cascadeChannelCount);
        }
        const KFParticleGpuDomainDesc& domain =
          compiledTransaction.Domains()[eventOffset];
        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          const KFParticleGpuWorkRange& work =
            compiledTransaction.WorkRanges()[domain.workRanges.offset + local];
          if (work.family == KFParticleGpuCompositeTrackWork) {
            if (std::find(activeGenerations.begin(), activeGenerations.end(),
                          work.generation) == activeGenerations.end()) {
              activeGenerations.push_back(work.generation);
            }
          }
        }
      }
      std::sort(activeGenerations.begin(), activeGenerations.end());
      fImpl->fLastCascadeGenerationCount =
        static_cast<unsigned int>(activeGenerations.size());
      for (const unsigned int generation : activeGenerations) {
        fImpl->fQueue.launch<KFParticleGpuResetCascadeBatch>(
          xpu::n_blocks(1u), fImpl->fBuffers->DeviceTransaction(), generation);
        TransactionStageFence(fImpl->fQueue, "cascade-reset", generation);
        fImpl->fQueue.launch<KFParticleGpuRouteCascadeBatch>(
          xpu::n_blocks(transactionWorkCount),
          fImpl->fKernels,
          fImpl->fBuffers->DeviceTransaction(), generation,
          twoDaughterChannelCount,
          fImpl->fCascadeTransactionStrategy == KFGpuCascadeTransactionFused
            ? 1u : 0u,
          collectDiagnosticCounters);
        TransactionStageFence(fImpl->fQueue, "cascade-route-construct", generation);
        if (fImpl->fCascadeTransactionStrategy == KFGpuCascadeTransactionStaged) {
          fImpl->fQueue.launch<KFParticleGpuConstructCascadeBatch>(
            xpu::n_blocks(eventCount), fImpl->fKernels,
            fImpl->fBuffers->DeviceTransaction(), generation,
            twoDaughterChannelCount);
          TransactionStageFence(fImpl->fQueue, "cascade-construct", generation);
        }
        fImpl->fQueue.launch<KFParticleGpuResolveTransactionWorkSources>(
          xpu::n_threads(transactionWorkCount),
          fImpl->fBuffers->DeviceTransaction(), generation);
        TransactionStageFence(fImpl->fQueue, "cascade-source-resolve", generation);
      }
    }
    else {
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult& eventResult =
        fImpl->fLastDecayPlanEventResults[eventOffset];
      KFParticleGpuFusedCascadeSnapshot& snapshot =
        fImpl->fFusedCascadeSnapshots[eventOffset];
      snapshot.channelVisited.resize(cascadeChannelCount);
      snapshot.channelAccepted.resize(cascadeChannelCount);
      snapshot.channelStored.resize(cascadeChannelCount);
      snapshot.channelConstructed.resize(cascadeChannelCount);
      fImpl->fQueue.memcpy(
        &snapshot.candidateBegin,
        fImpl->fBuffers->DeviceCandidates().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.daughterBegin,
        fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
        sizeof(unsigned int));
      KFParticleGpuDomainDesc domain;
      std::vector<KFParticleGpuWorkRange> scheduled;
      if (transactionBatch) {
        domain = compiledTransaction.Domains()[eventOffset];
        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          const KFParticleGpuWorkRange& work =
            compiledTransaction.WorkRanges()[domain.workRanges.offset + local];
          if (work.family == KFParticleGpuCompositeTrackWork) {
            scheduled.push_back(work);
          }
        }
      }
      else {
        domain.inputDescriptorIndex = eventResult.eventIndex;
        const KFParticleGpuEventDesc& event =
          fImpl->fBuffers->HostEvents()[eventResult.eventIndex];
        for (const KFParticleGpuV0TrackExecutionGroup& group :
             fImpl->fV0TrackRoutingPlan.Groups()) {
          KFParticleGpuWorkRange work;
          work.domainIndex = eventOffset;
          work.family = KFParticleGpuCompositeTrackWork;
          work.generation = group.generation;
          work.executionGroup = group.groupIndex;
          work.sourceTracks = event.TrackSet(group.bachelorTrackSet).tracks;
          work.sourceCandidates = {
            0u, fImpl->fBuffers->Capacities().selectedCandidates};
          scheduled.push_back(work);
        }
      }
      unsigned int activeGeneration = 0u;
      bool generationQueued = false;
      for (const KFParticleGpuWorkRange& work : scheduled) {
        if (work.domainIndex != eventOffset
            || work.executionGroup >= fImpl->fV0TrackRoutingPlan.Groups().size()) {
          throw std::logic_error(
            "KFParticle GPU cascade work range violates its compiled domain");
        }
        const KFParticleGpuV0TrackExecutionGroup& group =
          fImpl->fV0TrackRoutingPlan.Groups()[work.executionGroup];
        if (group.generation != work.generation) {
          throw std::logic_error(
            "KFParticle GPU cascade work range generation is inconsistent");
        }
        if (work.generation != activeGeneration) {
          if (generationQueued) {
            fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
              xpu::n_threads(fusedCascadeTaskCapacity),
              fusedCascadeTaskCapacity);
          }
          fImpl->fQueue.launch<KFParticleGpuResetV0TrackGenerationState>(
            xpu::n_threads(cascadeChannelCount),
            activeGeneration == 0u ? 1u : 0u);
          activeGeneration = work.generation;
          generationQueued = false;
        }
        const KFParticleGpuRange bachelors = work.sourceTracks;
        const KFParticleGpuSelectedCandidateRange selectedRange{
          0u, fImpl->fBuffers->Capacities().selectedCandidates, 0u};
        if (selectedRange.size != 0u
            && bachelors.size > std::numeric_limits<unsigned int>::max()
                                  / selectedRange.size) {
          throw std::overflow_error("KFParticle GPU fused cascade pair count overflows");
        }
        const unsigned int pairCount = selectedRange.size * bachelors.size;
        if (pairCount == 0u) {
          continue;
        }
        ++snapshot.groupLaunches;
        fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksBlockScan>(
          xpu::n_threads(pairCount),
          domain.inputDescriptorIndex,
          work.executionGroup,
          selectedRange,
          fusedCascadeTaskCapacity);
        generationQueued = true;
      }
      if (generationQueued) {
        fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
          xpu::n_threads(fusedCascadeTaskCapacity),
          fusedCascadeTaskCapacity);
      }

      fImpl->fQueue.memcpy(&snapshot.visitedPairs,
                            storage.fV0TrackRoutingVisitedPairCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.activeChannelBits,
                            storage.fV0TrackRoutingActiveBitCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.acceptedTasks,
                            storage.fV0TrackRoutedAcceptedTaskCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.storedTasks,
                            storage.fV0TrackRoutedStoredTaskCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.blockReservations,
                            storage.fV0TrackRoutingBlockReservationCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.taskOverflowFlags,
                            storage.fV0TrackRoutedTaskOverflowFlags.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelVisited.data(),
                            storage.fV0TrackRoutingChannelVisitedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelAccepted.data(),
                            storage.fV0TrackRoutingChannelAcceptedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelStored.data(),
                            storage.fV0TrackRoutingChannelStoredCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelConstructed.data(),
                            storage.fV0TrackRoutingChannelConstructedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.candidateEnd,
        fImpl->fBuffers->DeviceCandidates().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.daughterEnd,
        fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.candidateOverflowFlags,
        fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
        sizeof(unsigned int));
    }
    }
    queueGraphOperations();
    if (transactionBatch) {
      fImpl->fBuffers->QueueDownloadTransactionStatus(
        downloadWorkStatuses, true);
      ++fImpl->fLastPerformanceSnapshot.monitoringReadbacks;
      fImpl->fLastPerformanceSnapshot.monitoringReadbackBuffers +=
        downloadWorkStatuses ? 5u : 4u;
      fImpl->fLastPerformanceSnapshot.transactionStatusBytes +=
        fImpl->fBuffers->TransactionStatusPayloadBytes(
          downloadWorkStatuses, true);
    }
    // All first-generation, selection, cascade, and later graph actions are queued.
    // Compact transaction status reaches the host at this single boundary.
    fImpl->fQueue.wait();

    if (transactionBatch) {
      hostAssemblyStarted = std::chrono::steady_clock::now();
      assembleTransactionFirstGeneration();
      const KFParticleGpuTransactionView transaction =
        fImpl->fBuffers->HostTransaction();
      for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
        KFParticleGpuFusedCascadeSnapshot& snapshot =
          fImpl->fFusedCascadeSnapshots[eventOffset];
        const KFParticleGpuDomainDesc& domain = transaction.Domains()[eventOffset];
        const KFParticleGpuDomainStatus& domainStatus =
          transaction.DomainStatuses()[domain.statusIndex];
        const KFParticleGpuFusedTwoDaughterSnapshot& firstGeneration =
          fImpl->fFusedTwoDaughterSnapshots[eventOffset];
        snapshot.candidateBegin = domain.candidateReservation.offset
                                  + firstGeneration.candidateEnd;
        snapshot.daughterBegin = domain.daughterReservation.offset
                                 + firstGeneration.daughterEnd;
        snapshot.candidateEnd = snapshot.candidateBegin;
        snapshot.daughterEnd = snapshot.daughterBegin;
        snapshot.candidateOverflowFlags = domainStatus.overflowFlags;
        for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
          const unsigned int workIndex = domain.workRanges.offset + local;
          const KFParticleGpuWorkRange& work = transaction.WorkRanges()[workIndex];
          if (work.family != KFParticleGpuCompositeTrackWork) { continue; }
          ++snapshot.groupLaunches;
          if (downloadWorkStatuses) {
            const KFParticleGpuWorkStatus& status =
              transaction.WorkStatuses()[workIndex];
            snapshot.visitedPairs += status.visitedCombinations;
            snapshot.activeChannelBits += status.activeChannelBits;
            snapshot.acceptedTasks += status.acceptedTasks;
            snapshot.storedTasks += status.storedTasks;
            snapshot.blockReservations += status.blockReservations;
            snapshot.taskOverflowFlags |= status.overflowFlags;
          }
        }
        const KFParticleGpuDomainChannelStatus* channels =
          transaction.DomainChannelStatuses() + domain.channelStatuses.offset
          + twoDaughterChannelCount;
        for (unsigned int channel = 0u; channel < cascadeChannelCount; ++channel) {
          if (retainDetailedSnapshots) {
            snapshot.channelVisited[channel] = channels[channel].visited;
            snapshot.channelAccepted[channel] = channels[channel].accepted;
            snapshot.channelStored[channel] = channels[channel].stored;
            snapshot.channelConstructed[channel] = channels[channel].constructed;
          }
          if (!downloadWorkStatuses) {
            snapshot.visitedPairs += channels[channel].visited;
            snapshot.acceptedTasks += channels[channel].accepted;
            snapshot.storedTasks += channels[channel].stored;
          }
          snapshot.candidateEnd += channels[channel].constructed;
          snapshot.daughterEnd +=
            (fImpl->fDecayPlan->V0TrackCascadeChannel(channel).generation + 1u)
            * channels[channel].constructed;
        }
      }
    }

    fImpl->fLastV0TrackRoutingMonitorData.descriptorCount = cascadeChannelCount;
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult& eventResult =
        fImpl->fLastDecayPlanEventResults[eventOffset];
      const KFParticleGpuFusedCascadeSnapshot& snapshot =
        fImpl->fFusedCascadeSnapshots[eventOffset];
      eventResult.cascadeChannelOffset =
        static_cast<unsigned int>(fImpl->fLastV0TrackCascadeResults.size());
      eventResult.cascadeChannelCount = cascadeChannelCount;
      eventResult.cascadeCandidates.offset = snapshot.candidateBegin;
      eventResult.cascadeCandidates.size =
        snapshot.candidateEnd - snapshot.candidateBegin;
      eventResult.cascadeCandidates.daughterOffset = snapshot.daughterBegin;
      eventResult.cascadeCandidates.daughterSize =
        snapshot.daughterEnd - snapshot.daughterBegin;
      eventResult.cascadeCandidates.overflowFlags =
        snapshot.candidateOverflowFlags | snapshot.taskOverflowFlags;
      eventResult.cascadeRouting.visitedPairs = snapshot.visitedPairs;
      eventResult.cascadeRouting.activeChannelBits = snapshot.activeChannelBits;
      eventResult.cascadeRouting.acceptedTasks = snapshot.acceptedTasks;
      eventResult.cascadeRouting.storedTasks = snapshot.storedTasks;
      eventResult.cascadeRouting.blockReservations = snapshot.blockReservations;
      eventResult.cascadeRouting.overflowFlags = snapshot.taskOverflowFlags;
      eventResult.overflowFlags |= eventResult.cascadeCandidates.overflowFlags;

      if (retainDetailedSnapshots) {
        for (unsigned int channelIndex = 0u;
             channelIndex < cascadeChannelCount;
             ++channelIndex) {
          const KFParticleGpuV0TrackCascadeChannel& channel =
            fImpl->fDecayPlan->V0TrackCascadeChannel(channelIndex);
          KFParticleGpuV0TrackChannelResult result;
          result.channelId = channel.channelId;
          result.motherPdg = channel.motherPdg;
          result.eventIndex = eventResult.eventIndex;
          result.totalPairs = snapshot.channelVisited[channelIndex];
          result.acceptedTasks = snapshot.channelAccepted[channelIndex];
          result.storedTasks = snapshot.channelStored[channelIndex];
          result.constructedCandidates = snapshot.channelConstructed[channelIndex];
          result.constructedDaughters =
            (channel.generation + 1u) * result.constructedCandidates;
          result.generationCandidates = eventResult.cascadeCandidates;
          result.candidates.overflowFlags = eventResult.cascadeCandidates.overflowFlags;
          fImpl->fLastV0TrackCascadeResults.push_back(result);
        }

        eventResult.cascadeRouting.visitedPairs = 0u;
        eventResult.cascadeRouting.acceptedTasks = 0u;
        eventResult.cascadeRouting.storedTasks = 0u;
        for (unsigned int channelIndex = 0u;
             channelIndex < cascadeChannelCount;
             ++channelIndex) {
          eventResult.cascadeRouting.visitedPairs += snapshot.channelVisited[channelIndex];
          eventResult.cascadeRouting.acceptedTasks += snapshot.channelAccepted[channelIndex];
          eventResult.cascadeRouting.storedTasks += snapshot.channelStored[channelIndex];
        }
      }

      KFParticleGpuV0TrackRoutingMonitorData& monitoring =
        fImpl->fLastV0TrackRoutingMonitorData;
      monitoring.groupLaunches += snapshot.groupLaunches;
      monitoring.visitedPairs += eventResult.cascadeRouting.visitedPairs;
      monitoring.activeChannelBits += snapshot.activeChannelBits;
      monitoring.acceptedTasks += eventResult.cascadeRouting.acceptedTasks;
      monitoring.storedTasks += eventResult.cascadeRouting.storedTasks;
      monitoring.blockReservations += snapshot.blockReservations;
      monitoring.candidates += eventResult.cascadeCandidates.size;
      monitoring.daughters += eventResult.cascadeCandidates.daughterSize;
      monitoring.overflowFlags |= eventResult.cascadeCandidates.overflowFlags;
    }
  }
  else {
    queueGraphOperations();
    if (transactionBatch) {
      fImpl->fBuffers->QueueDownloadTransactionStatus(
        downloadWorkStatuses, true);
      ++fImpl->fLastPerformanceSnapshot.monitoringReadbacks;
      fImpl->fLastPerformanceSnapshot.monitoringReadbackBuffers +=
        downloadWorkStatuses ? 5u : 4u;
      fImpl->fLastPerformanceSnapshot.transactionStatusBytes +=
        fImpl->fBuffers->TransactionStatusPayloadBytes(
          downloadWorkStatuses, true);
    }
    // Without a cascade generation this is still the sole synchronization
    // between route/construct/select submission and host result assembly.
    fImpl->fQueue.wait();
    if (transactionBatch) {
      hostAssemblyStarted = std::chrono::steady_clock::now();
    }
    assembleTransactionFirstGeneration();
  }

  if (transactionBatch) {
    const KFParticleGpuTransactionView transaction =
      fImpl->fBuffers->HostTransaction();
    fImpl->fGraphOperationSnapshots.resize(eventCount);
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      const KFParticleGpuDomainDesc& domain =
        transaction.Domains()[eventOffset];
      const KFParticleGpuDomainStatus& domainStatus =
        transaction.DomainStatuses()[domain.statusIndex];
      KFParticleGpuGraphOperationSnapshot& graph =
        fImpl->fGraphOperationSnapshots[eventOffset];
      graph.eventIndex = domain.inputDescriptorIndex;
      if (retainDetailedSnapshots) {
        graph.channelVisited.resize(graphOperationCount);
        graph.channelAccepted.resize(graphOperationCount);
        graph.channelStored.resize(graphOperationCount);
        graph.channelConstructed.resize(graphOperationCount);
        graph.channelRejected.resize(graphOperationCount);
      }
      graph.candidateEnd =
        domain.candidateReservation.offset + domainStatus.candidates;
      graph.daughterEnd =
        domain.daughterReservation.offset + domainStatus.daughters;
      graph.candidateOverflowFlags = domainStatus.overflowFlags;
      if (cascadeChannelCount > 0u) {
        graph.candidateBegin =
          fImpl->fFusedCascadeSnapshots[eventOffset].candidateEnd;
        graph.daughterBegin =
          fImpl->fFusedCascadeSnapshots[eventOffset].daughterEnd;
      }
      else {
        graph.candidateBegin =
          domain.candidateReservation.offset
          + fImpl->fFusedTwoDaughterSnapshots[eventOffset].candidateEnd;
        graph.daughterBegin =
          domain.daughterReservation.offset
          + fImpl->fFusedTwoDaughterSnapshots[eventOffset].daughterEnd;
      }
      for (unsigned int local = 0u; local < domain.workRanges.size; ++local) {
        const unsigned int workIndex = domain.workRanges.offset + local;
        const KFParticleGpuWorkRange& work =
          transaction.WorkRanges()[workIndex];
        if (work.family != KFParticleGpuGraphOperationWork) { continue; }
        if (downloadWorkStatuses) {
          const KFParticleGpuWorkStatus& status =
            transaction.WorkStatuses()[workIndex];
          graph.visited += status.visitedCombinations;
          graph.accepted += status.acceptedTasks;
          graph.stored += status.storedTasks;
          graph.constructed += status.constructedCandidates;
          graph.overflowFlags |= status.overflowFlags;
        }
      }
      const KFParticleGpuDomainChannelStatus* channels =
        transaction.DomainChannelStatuses() + domain.channelStatuses.offset
        + twoDaughterChannelCount + cascadeChannelCount;
      for (unsigned int channel = 0u;
           channel < graphOperationCount; ++channel) {
        if (retainDetailedSnapshots) {
          graph.channelVisited[channel] = channels[channel].visited;
          graph.channelAccepted[channel] = channels[channel].accepted;
          graph.channelStored[channel] = channels[channel].stored;
          graph.channelConstructed[channel] = channels[channel].constructed;
        }
        if (!downloadWorkStatuses) {
          graph.visited += channels[channel].visited;
          graph.accepted += channels[channel].accepted;
          graph.stored += channels[channel].stored;
          graph.constructed += channels[channel].constructed;
        }
      }
    }
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
        fImpl->fFusedTwoDaughterSnapshots[eventOffset];
      snapshot.candidateEnd += snapshot.candidateBegin;
      snapshot.daughterEnd += snapshot.daughterBegin;
      const unsigned int selectedCount = snapshot.selectedEnd;
      if (retainDetailedSnapshots) {
        snapshot.selectedBegin = twoDaughterChannelCount > 0u
                                   ? snapshot.selectionOffsets[0] : 0u;
      }
      snapshot.selectedEnd = snapshot.selectedBegin + selectedCount;
    }
  }

  KFParticleGpuGraphExecutionMonitorData& graphMonitorData =
    fImpl->fLastGraphExecutionMonitorData;
  graphMonitorData.descriptorCount = graphOperationCount;
  for (const KFParticleGpuGraphNode& node : fImpl->fDecayGraphPlan.Nodes()) {
    graphMonitorData.unsupportedNodes +=
      node.supportStatus == KFGpuGraphSupported ? 0u : 1u;
  }
  for (const KFParticleGpuGraphFamilyCoverage& family :
       fImpl->fDecayGraphPlan.FamilyCoverage()) {
    graphMonitorData.unsupportedFamilies +=
      family.supportStatus == KFGpuGraphSupported ? 0u : 1u;
  }
  if (retainDetailedSnapshots) {
    fImpl->fLastGraphChannelMonitorData.resize(
      eventCount * graphOperationCount);
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      for (unsigned int descriptorIndex = 0u;
           descriptorIndex < graphOperationCount;
           ++descriptorIndex) {
        KFParticleGpuGraphChannelMonitorData& channel =
          fImpl->fLastGraphChannelMonitorData[
            eventOffset * graphOperationCount + descriptorIndex];
        const KFParticleGpuGraphOperationChannel& configured =
          fImpl->fDecayPlan->GraphOperationChannel(descriptorIndex);
        channel.channelId = configured.descriptor.channelId;
        channel.eventIndex = firstEventIndex + eventOffset;
        channel.generation = configured.node.generation;
      }
    }
  }
  for (const KFParticleGpuGraphOperationSnapshot& snapshot :
       fImpl->fGraphOperationSnapshots) {
    if (!transactionBatch) { ++graphMonitorData.groupLaunches; }
    graphMonitorData.visitedCombinations += snapshot.visited;
    graphMonitorData.acceptedTasks += snapshot.accepted;
    graphMonitorData.storedTasks += snapshot.stored;
    graphMonitorData.constructedCandidates += snapshot.constructed;
    graphMonitorData.rejectedTasks += snapshot.rejected;
    graphMonitorData.candidates += snapshot.candidateEnd - snapshot.candidateBegin;
    graphMonitorData.daughters += snapshot.daughterEnd - snapshot.daughterBegin;
    graphMonitorData.overflowFlags |=
      snapshot.overflowFlags | snapshot.candidateOverflowFlags;
    const unsigned int eventOffset = snapshot.eventIndex - firstEventIndex;
    KFParticleGpuDecayPlanEventResult& eventResult =
      fImpl->fLastDecayPlanEventResults[eventOffset];
    if (eventResult.graphCandidates.size == 0u) {
      eventResult.graphCandidates.offset = snapshot.candidateBegin;
      eventResult.graphCandidates.daughterOffset = snapshot.daughterBegin;
    }
    eventResult.graphCandidates.size +=
      snapshot.candidateEnd - snapshot.candidateBegin;
    eventResult.graphCandidates.daughterSize +=
      snapshot.daughterEnd - snapshot.daughterBegin;
    eventResult.graphCandidates.overflowFlags |=
      snapshot.overflowFlags | snapshot.candidateOverflowFlags;
    eventResult.overflowFlags |= eventResult.graphCandidates.overflowFlags;
    if (retainDetailedSnapshots) {
      for (unsigned int descriptorIndex = 0u;
           descriptorIndex < graphOperationCount;
           ++descriptorIndex) {
        KFParticleGpuGraphChannelMonitorData& channel =
          fImpl->fLastGraphChannelMonitorData[
            eventOffset * graphOperationCount + descriptorIndex];
        channel.visitedCombinations += snapshot.channelVisited[descriptorIndex];
        channel.acceptedTasks += snapshot.channelAccepted[descriptorIndex];
        channel.storedTasks += snapshot.channelStored[descriptorIndex];
        channel.constructedCandidates +=
          snapshot.channelConstructed[descriptorIndex];
        channel.rejectedTasks += snapshot.channelRejected[descriptorIndex];
      }
    }
  }
  if (transactionBatch) {
    graphMonitorData.groupLaunches = transactionGraphGenerationCount;
  }

  fImpl->fLastTwoDaughterRoutingMonitorData.descriptorCount =
    twoDaughterChannelCount;
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    KFParticleGpuDecayPlanEventResult& eventResult =
      fImpl->fLastDecayPlanEventResults[eventOffset];
    if (twoDaughterChannelCount == 0u) {
      continue;
    }
    const KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
      fImpl->fFusedTwoDaughterSnapshots[eventOffset];
    eventResult.candidates.offset = snapshot.candidateBegin;
    eventResult.candidates.size = snapshot.candidateEnd - snapshot.candidateBegin;
    eventResult.candidates.daughterOffset = snapshot.daughterBegin;
    eventResult.candidates.daughterSize =
      snapshot.daughterEnd - snapshot.daughterBegin;
    eventResult.candidates.overflowFlags =
      snapshot.candidateOverflowFlags | snapshot.taskOverflowFlags;
    eventResult.generationRouting.visitedPairs = snapshot.visitedPairs;
    eventResult.generationRouting.activeChannelBits = snapshot.activeChannelBits;
    eventResult.generationRouting.acceptedTasks = snapshot.acceptedTasks;
    eventResult.generationRouting.storedTasks = snapshot.storedTasks;
    eventResult.generationRouting.blockReservations = snapshot.blockReservations;
    eventResult.generationRouting.overflowFlags = snapshot.taskOverflowFlags;
    eventResult.selectedCandidates.offset = snapshot.selectedBegin;
    eventResult.selectedCandidates.size =
      snapshot.selectedEnd - snapshot.selectedBegin;
    eventResult.selectedCandidates.overflowFlags =
      snapshot.selectedOverflowFlags;
    eventResult.overflowFlags |= eventResult.candidates.overflowFlags
                                 | snapshot.selectedOverflowFlags;

    if (retainDetailedSnapshots) {
      for (unsigned int channelIndex = 0u;
           channelIndex < twoDaughterChannelCount;
           ++channelIndex) {
        const KFParticleGpuTwoDaughterChannel& channel =
          fImpl->fDecayPlan->TwoDaughterChannel(channelIndex);
        KFParticleGpuTwoDaughterChannelResult result;
        result.channelId = channel.channelId;
        result.motherPdg = channel.motherPdg;
        result.eventIndex = eventResult.eventIndex;
        result.totalPairs = snapshot.channelVisited[channelIndex];
        result.acceptedTasks = snapshot.channelAccepted[channelIndex];
        result.storedTasks = snapshot.channelStored[channelIndex];
        result.constructedCandidates = snapshot.channelConstructed[channelIndex];
        result.constructedDaughters = 2u * result.constructedCandidates;
        result.generationCandidates = eventResult.candidates;
        result.candidates.overflowFlags = eventResult.candidates.overflowFlags;
        // A single-channel generation remains physically contiguous and can
        // retain the legacy range contract. Multi-channel generations use
        // stable metadata plus constructedCandidates.
        if (twoDaughterChannelCount == 1u) {
          result.candidates = eventResult.candidates;
        }
        fImpl->fLastDecayPlanResults.push_back(result);

        KFParticleGpuSelectedChannelRange selectedChannel;
        selectedChannel.channelId = channel.channelId;
        selectedChannel.eventIndex = eventResult.eventIndex;
        selectedChannel.candidates.offset =
          snapshot.selectionOffsets[channelIndex];
        selectedChannel.candidates.size =
          snapshot.selectionStored[channelIndex];
        selectedChannel.candidates.overflowFlags =
          snapshot.selectedOverflowFlags;
        fImpl->fLastDecayPlanSelectedChannels.push_back(selectedChannel);
      }
    }

    KFParticleGpuTwoDaughterRoutingMonitorData& monitoring =
      fImpl->fLastTwoDaughterRoutingMonitorData;
    monitoring.groupLaunches += snapshot.groupLaunches;
    monitoring.visitedPairs += snapshot.visitedPairs;
    monitoring.activeChannelBits += snapshot.activeChannelBits;
    monitoring.acceptedTasks += snapshot.acceptedTasks;
    monitoring.storedTasks += snapshot.storedTasks;
    monitoring.blockReservations += snapshot.blockReservations;
    monitoring.candidates += eventResult.candidates.size;
    monitoring.daughters += eventResult.candidates.daughterSize;
    monitoring.selectedCandidates += eventResult.selectedCandidates.size;
    monitoring.overflowFlags |= eventResult.candidates.overflowFlags
                               | eventResult.selectedCandidates.overflowFlags;
  }
  fImpl->fLastDecayPlanTiming.cascadeConstructionMilliseconds = elapsedMilliseconds(cascadeStarted);
  if (hostAssemblyStarted != std::chrono::steady_clock::time_point{}) {
    fImpl->fLastDecayPlanTiming.hostAssemblyMilliseconds =
      elapsedMilliseconds(hostAssemblyStarted);
  }
  const auto downloadStarted = std::chrono::steady_clock::now();
  unsigned int candidateOutputSize = 0u;
  unsigned int daughterOutputSize = 0u;
  unsigned int selectedOutputSize = 0u;
  for (const auto& eventResult : fImpl->fLastDecayPlanEventResults) {
    for (const auto& range : {eventResult.candidates, eventResult.cascadeCandidates,
                              eventResult.graphCandidates}) {
      if (range.size != 0u) {
        candidateOutputSize = std::max(candidateOutputSize, range.offset + range.size);
      }
      if (range.daughterSize != 0u) {
        daughterOutputSize = std::max(daughterOutputSize, range.daughterOffset + range.daughterSize);
      }
    }
  }
  // Empty reservations may lie past the populated view. Preserve valid/overflow
  // diagnostics, but do not export out-of-bounds empty spans for materialization.
  const auto normalizeEmptyRange = [&](KFParticleGpuCandidateRange& range) {
    if (range.overflowFlags != 0u || range.size != 0u) { return; }
    if (range.offset > candidateOutputSize) { range.offset = 0u; }
    if (range.daughterSize == 0u && range.daughterOffset > daughterOutputSize) {
      range.daughterOffset = 0u;
    }
  };
  for (auto& channel : fImpl->fLastDecayPlanResults) {
    normalizeEmptyRange(channel.candidates);
    normalizeEmptyRange(channel.generationCandidates);
  }
  for (auto& eventResult : fImpl->fLastDecayPlanEventResults) {
    normalizeEmptyRange(eventResult.candidates);
    normalizeEmptyRange(eventResult.cascadeCandidates);
    normalizeEmptyRange(eventResult.graphCandidates);
    const KFParticleGpuCandidateRange candidateRanges[] = {
      eventResult.candidates,
      eventResult.cascadeCandidates,
      eventResult.graphCandidates};
    for (const auto& range : candidateRanges) {
      candidateOutputSize = std::max(candidateOutputSize, range.offset + range.size);
      daughterOutputSize = std::max(
        daughterOutputSize, range.daughterOffset + range.daughterSize);
    }
    selectedOutputSize = std::max(
      selectedOutputSize,
      eventResult.selectedCandidates.offset + eventResult.selectedCandidates.size);
  }
  fImpl->fBuffers->DownloadSelectedCandidateOutput(selectedOutputSize);
  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
  fImpl->fLastDecayPlanSelectedCandidates.offset = 0u;
  fImpl->fLastDecayPlanSelectedCandidates.size = selected.Size();
  fImpl->fLastDecayPlanSelectedCandidates.overflowFlags = selected.OverflowFlags();
  fImpl->fBuffers->DownloadCandidateOutput(
    candidateOutputSize,
    daughterOutputSize,
    outputMode == KFGpuFullDiagnosticOutput);
  fImpl->fLastDecayPlanTiming.outputDownloadMilliseconds = elapsedMilliseconds(downloadStarted);
  fImpl->CompletePerformanceSnapshot(
    eventCount, capacityGrowthCountBefore, planUploadWaits,
    (deviceBatchInitialization ? 4u : 7u)
      + (compactTransactionOutput ? 1u : 0u),
    candidateOutputSize,
    daughterOutputSize,
    selectedOutputSize,
    outputMode == KFGpuFullDiagnosticOutput);
  deviceProfile.Finish();
  return fImpl->fLastDecayPlanResults;
}

/**
 * @brief Borrow the last full-plan two-track channel results, omitted in publishable transaction mode.
 */
const std::vector<KFParticleGpuTwoDaughterChannelResult>&
KFParticleGpuSteering::LastDecayPlanResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanResults;
}

/**
 * @brief Borrow detailed composite-track channel results, omitted in publishable transaction mode.
 */
const std::vector<KFParticleGpuV0TrackChannelResult>&
KFParticleGpuSteering::LastV0TrackCascadeResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastV0TrackCascadeResults;
}

/**
 * @brief Borrow event-local first-generation, cascade, graph, and selected ranges from the last full plan.
 */
const std::vector<KFParticleGpuDecayPlanEventResult>&
KFParticleGpuSteering::LastDecayPlanEventResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanEventResults;
}

/**
 * @brief Borrow host timing intervals for the last full plan; submeasurements overlap parent intervals.
 */
const KFParticleGpuDecayPlanTiming& KFParticleGpuSteering::LastDecayPlanTiming() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanTiming;
}

/**
 * @brief Borrow aggregated first-generation counters from the last full plan, not isolated stage helpers.
 */
const KFParticleGpuTwoDaughterRoutingMonitorData&
KFParticleGpuSteering::LastTwoDaughterRoutingMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastTwoDaughterRoutingMonitorData;
}

/**
 * @brief Borrow aggregated composite-track counters from the last full plan.
 */
const KFParticleGpuV0TrackRoutingMonitorData&
KFParticleGpuSteering::LastV0TrackRoutingMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastV0TrackRoutingMonitorData;
}

/**
 * @brief Borrow aggregated dependent-graph execution and support-coverage counters.
 */
const KFParticleGpuGraphExecutionMonitorData&
KFParticleGpuSteering::LastGraphExecutionMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastGraphExecutionMonitorData;
}

/**
 * @brief Borrow detailed event/channel graph counters; absent in publishable transaction mode.
 */
const std::vector<KFParticleGpuGraphChannelMonitorData>&
KFParticleGpuSteering::LastGraphChannelMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastGraphChannelMonitorData;
}

/**
 * @brief Change monitoring for later runs; disabling clears the retained performance record immediately.
 */
void KFParticleGpuSteering::SetPerformanceMonitoringEnabled(bool enabled)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  fImpl->fPerformanceMonitoringEnabled = enabled;
  if (!enabled) { fImpl->fLastPerformanceSnapshot = KFParticleGpuPerformanceSnapshot(); }
}

/**
 * @brief Return initialized AND monitoring-enabled, without throwing or locking.
 */
bool KFParticleGpuSteering::PerformanceMonitoringEnabled() const
{
  return fImpl->fInitialized && fImpl->fPerformanceMonitoringEnabled;
}

/**
 * @brief Set the preferred first-generation transaction strategy; reject unknown enum values.
 * Requires initialization and rejects unrecognized enum values; execution uses the setting on subsequent runs.
 */
void KFParticleGpuSteering::SetTwoDaughterTransactionStrategy(
  KFParticleGpuTwoDaughterTransactionStrategy strategy)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (strategy != KFGpuTwoDaughterTransactionStaged
      && strategy != KFGpuTwoDaughterTransactionFused) {
    throw std::invalid_argument("KFParticle GPU two-daughter transaction strategy is invalid");
  }
  fImpl->fTwoDaughterTransactionStrategy = strategy;
}

/**
 * @brief Return the configured preference, not the route actually used in the last transaction.
 */
KFParticleGpuTwoDaughterTransactionStrategy
KFParticleGpuSteering::TwoDaughterTransactionStrategy() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fTwoDaughterTransactionStrategy;
}

/**
 * @brief Choose staged or in-construction selection for transaction batches; reject unknown values.
 * Requires initialization and rejects unrecognized enum values; execution uses the setting on subsequent runs.
 */
void KFParticleGpuSteering::SetTwoDaughterSelectionStrategy(
  KFParticleGpuTwoDaughterSelectionStrategy strategy)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (strategy != KFGpuTwoDaughterSelectionStaged
      && strategy != KFGpuTwoDaughterSelectionDirect) {
    throw std::invalid_argument("KFParticle GPU two-daughter selection strategy is invalid");
  }
  fImpl->fTwoDaughterSelectionStrategy = strategy;
}

/**
 * @brief Return the configured first-generation selection strategy.
 */
KFParticleGpuTwoDaughterSelectionStrategy
KFParticleGpuSteering::TwoDaughterSelectionStrategy() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fTwoDaughterSelectionStrategy;
}

/**
 * @brief Choose separate or routing-integrated cascade construction for transaction batches.
 * Requires initialization and rejects unrecognized enum values; execution uses the setting on subsequent runs.
 */
void KFParticleGpuSteering::SetCascadeTransactionStrategy(
  KFParticleGpuCascadeTransactionStrategy strategy)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (strategy != KFGpuCascadeTransactionStaged
      && strategy != KFGpuCascadeTransactionFused) {
    throw std::invalid_argument("KFParticle GPU cascade transaction strategy is invalid");
  }
  fImpl->fCascadeTransactionStrategy = strategy;
}

/**
 * @brief Return the configured cascade transaction strategy.
 */
KFParticleGpuCascadeTransactionStrategy
KFParticleGpuSteering::CascadeTransactionStrategy() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fCascadeTransactionStrategy;
}

/**
 * @brief Choose separate or routing-integrated graph execution for transaction batches.
 * Requires initialization and rejects unrecognized enum values; execution uses the setting on subsequent runs.
 */
void KFParticleGpuSteering::SetGraphTransactionStrategy(
  KFParticleGpuGraphTransactionStrategy strategy)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (strategy != KFGpuGraphTransactionStaged
      && strategy != KFGpuGraphTransactionFused) {
    throw std::invalid_argument("KFParticle GPU graph transaction strategy is invalid");
  }
  fImpl->fGraphTransactionStrategy = strategy;
}

/**
 * @brief Return the configured dependent-graph transaction strategy.
 */
KFParticleGpuGraphTransactionStrategy
KFParticleGpuSteering::GraphTransactionStrategy() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fGraphTransactionStrategy;
}

/**
 * @brief Choose staged host resets or device-batch initialization where supported.
 * Requires initialization and rejects unrecognized enum values; execution uses the setting on subsequent runs.
 */
void KFParticleGpuSteering::SetAdministrativeResetStrategy(
  KFParticleGpuAdministrativeResetStrategy strategy)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (strategy != KFGpuAdministrativeResetHostStaged
      && strategy != KFGpuAdministrativeResetDeviceBatch) {
    throw std::invalid_argument(
      "KFParticle GPU administrative reset strategy is invalid");
  }
  fImpl->fAdministrativeResetStrategy = strategy;
}

/**
 * @brief Return the configured administrative reset policy.
 */
KFParticleGpuAdministrativeResetStrategy
KFParticleGpuSteering::AdministrativeResetStrategy() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fAdministrativeResetStrategy;
}

/**
 * @brief Borrow performance accounting; check enabled and the documented limits of estimated counters.
 */
const KFParticleGpuPerformanceSnapshot&
KFParticleGpuSteering::LastPerformanceSnapshot() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastPerformanceSnapshot;
}

/**
 * @brief Borrow the downloaded first-generation selected-index range and its overflow mask.
 */
const KFParticleGpuSelectedCandidateRange& KFParticleGpuSteering::LastDecayPlanSelectedCandidates() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanSelectedCandidates;
}

/**
 * @brief Borrow detailed event/channel selected-index ranges; omitted in publishable transactions.
 */
const std::vector<KFParticleGpuSelectedChannelRange>&
KFParticleGpuSteering::LastDecayPlanSelectedChannels() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanSelectedChannels;
}
#endif
