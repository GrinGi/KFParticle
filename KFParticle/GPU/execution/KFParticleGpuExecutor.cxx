/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "execution/KFParticleGpuExecutor.h"

#include "execution/KFParticleGpuRuntimeService.h"
#include "execution/KFParticleGpuMaterializationSchedule.h"

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuCpuChannelCatalogue.h"
#include "planning/KFParticleGpuDecayPlan.h"
#include "algorithms/KFParticleGpuMath.h"
#include "diagnostics/KFParticleGpuParity.h"
#include "execution/KFParticleGpuSteering.h"
#include "diagnostics/KFParticleGpuStepTrace.h"
#include "data/KFParticleGpuSoAView.h"
#include "KFPTrackVector.h"
#include "KFVertex.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
// Use bounded OpenMP workers without a framework Parallel STL/TBB dependency.
#include <omp.h>
#endif

namespace
{
  unsigned int CheckedCountSum(unsigned int left, unsigned int right)
  {
    if (right > std::numeric_limits<unsigned int>::max() - left) {
      throw std::overflow_error("KFParticle GPU diagnostic task count exceeds the GPU ABI");
    }
    return left + right;
  }

  kfparticle::gpu::GpuExecutionStatus ClassifyInputRejection(const std::string& message)
  {
    // Field data is an explicit physics input for the full-field V0 route.
    // Keep it distinguishable from malformed track/PV metadata in reports.
    return message.find("field") != std::string::npos || message.find("Field") != std::string::npos
             ? kfparticle::gpu::GpuExecutionStatus::FieldRejected
             : kfparticle::gpu::GpuExecutionStatus::InputRejected;
  }

  kfparticle::gpu::GpuExecutionStatus ClassifyRuntimeOrFieldFailure(const std::string& message)
  {
    // A missing or non-finite field is an input problem; other logic errors
    // indicate that the diagnostic route itself is unavailable for this run.
    return message.find("field") != std::string::npos || message.find("Field") != std::string::npos
             ? kfparticle::gpu::GpuExecutionStatus::FieldRejected
             : kfparticle::gpu::GpuExecutionStatus::RuntimeUnavailable;
  }

  // Order-independent signature of accepted root identities, not a floating
  // point equivalence proof. Fits remain covered by the numerical regression.
  void RecordOutputSignature(const KFParticleGpuConstCandidatePoolView& candidates,
                             unsigned int index,
                             kfparticle::gpu::GpuEventResult& result)
  {
    const auto& metadata = candidates.Metadata();
    const unsigned int offset = metadata.DaughterOffset(index);
    const unsigned int count = metadata.DaughterCount(index);
    if (offset > candidates.Daughters().Size()
        || count > candidates.Daughters().Size() - offset) {
      throw std::logic_error("invalid signature lineage range");
    }
    if (count > KFGpuParityMaximumLineageSize) {
      throw std::logic_error("signature lineage exceeds the GPU publication contract");
    }
    std::array<int, KFGpuParityMaximumLineageSize> lineage{};
    for (unsigned int i = 0; i < count; ++i) {
      lineage[i] = candidates.Daughters().SourceId(offset + i);
    }
    std::sort(lineage.begin(), lineage.begin() + count);
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&hash](std::uint64_t value) {
      for (unsigned int byte = 0; byte < 8; ++byte) {
        hash = (hash ^ ((value >> (8u * byte)) & 255u)) * 1099511628211ULL;
      }
    };
    mix(result.sourceEventId);
    mix(static_cast<std::uint64_t>(metadata.Pdg(index)));
    mix(metadata.ChannelId(index));
    mix(metadata.OutputClass(index));
    mix(static_cast<std::uint64_t>(metadata.PrimaryVertexIndex(index)));
    mix(count);
    for (unsigned int i = 0; i < count; ++i) {
      mix(static_cast<std::uint64_t>(lineage[i]));
    }
    result.resultSignatureXor ^= hash;
    result.resultSignatureSum += hash;
    ++result.resultSignatureCandidates;
  }

  bool PerformanceMonitoringRequested()
  {
    const char* value = std::getenv("KFPARTICLE_GPU_PERFORMANCE_MONITORING");
    return value && std::string(value) == "1";
  }

  bool InProcessReplayRequested()
  {
    const char* value = std::getenv("KFPARTICLE_GPU_IN_PROCESS_REPLAY");
    return value && std::string(value) == "1";
  }

  unsigned int InProcessRepeatCount()
  {
    // A repeat count may remain exported after a performance campaign.  Keep
    // ordinary reconstruction and tests isolated unless replay was explicitly
    // requested by the dedicated launcher.
    if (!InProcessReplayRequested()) { return 1u; }

    const char* value = std::getenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS");
    if (value == nullptr || value[0] == '\0') {
      throw std::invalid_argument(
        "KFPARTICLE_GPU_IN_PROCESS_REPLAY=1 requires "
        "KFPARTICLE_GPU_IN_PROCESS_REPEATS");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0u || parsed > 20u) {
      throw std::invalid_argument(
        "KFPARTICLE_GPU_IN_PROCESS_REPEATS must be an integer from 1 to 20");
    }
    if (!PerformanceMonitoringRequested()) {
      throw std::invalid_argument(
        "KFParticle in-process repeats require KFPARTICLE_GPU_PERFORMANCE_MONITORING=1");
    }
    return static_cast<unsigned int>(parsed);
  }

  struct RepeatIdentity {
    struct Event {
      std::uint64_t sourceEventId = 0u;
      std::uint64_t signatureCandidates = 0u;
      std::uint64_t signatureXor = 0u;
      std::uint64_t signatureSum = 0u;
      unsigned int status = 0u;
      unsigned int candidates = 0u;
      unsigned int selected = 0u;
      unsigned int overflow = 0u;

      bool operator==(const Event& other) const
      {
        return sourceEventId == other.sourceEventId
               && signatureCandidates == other.signatureCandidates
               && signatureXor == other.signatureXor
               && signatureSum == other.signatureSum && status == other.status
               && candidates == other.candidates && selected == other.selected
               && overflow == other.overflow;
      }
    };

    std::uint64_t sourceXor = 0u;
    std::uint64_t sourceSum = 0u;
    std::uint64_t signatureCandidates = 0u;
    std::uint64_t signatureXor = 0u;
    std::uint64_t signatureSum = 0u;
    std::uint64_t candidates = 0u;
    std::uint64_t selected = 0u;
    unsigned int events = 0u;
    unsigned int succeeded = 0u;
    unsigned int overflow = 0u;
    std::vector<Event> eventIdentity;

    bool operator==(const RepeatIdentity& other) const
    {
      return sourceXor == other.sourceXor && sourceSum == other.sourceSum
             && signatureCandidates == other.signatureCandidates
             && signatureXor == other.signatureXor
             && signatureSum == other.signatureSum
             && candidates == other.candidates && selected == other.selected
             && events == other.events && succeeded == other.succeeded
             && overflow == other.overflow
             && eventIdentity == other.eventIdentity;
    }
  };

  RepeatIdentity MakeRepeatIdentity(
    const std::vector<kfparticle::gpu::GpuEventResult>& results)
  {
    RepeatIdentity identity;
    identity.events = static_cast<unsigned int>(results.size());
    identity.eventIdentity.reserve(results.size());
    for (const auto& result : results) {
      identity.sourceXor ^= result.sourceEventId;
      identity.sourceSum += result.sourceEventId;
      identity.signatureCandidates += result.resultSignatureCandidates;
      identity.signatureXor ^= result.resultSignatureXor;
      identity.signatureSum += result.resultSignatureSum;
      identity.candidates += result.candidates;
      identity.selected += result.selectedCandidates;
      identity.overflow |= result.overflowFlags;
      identity.succeeded += result.Succeeded() ? 1u : 0u;
      identity.eventIdentity.push_back({
        result.sourceEventId,
        result.resultSignatureCandidates,
        result.resultSignatureXor,
        result.resultSignatureSum,
        static_cast<unsigned int>(result.status),
        result.candidates,
        result.selectedCandidates,
        result.overflowFlags});
    }
    return identity;
  }

  double SumMaterializationMilliseconds(
    const std::vector<kfparticle::gpu::GpuEventResult>& results)
  {
    double milliseconds = 0.;
    for (const auto& result : results) {
      milliseconds += result.materializationMilliseconds;
    }
    return milliseconds;
  }

  double SumMaterializationCpuWorkMilliseconds(
    const std::vector<kfparticle::gpu::GpuEventResult>& results)
  {
    double milliseconds = 0.;
    for (const auto& result : results) {
      milliseconds += result.materializationCpuWorkMilliseconds;
    }
    return milliseconds;
  }

  void CopyPerformanceSnapshot(const KFParticleGpuPerformanceSnapshot& source,
                               kfparticle::gpu::GpuEventResult& destination)
  {
    if (!source.enabled) { return; }
    destination.performanceSample = true;
    destination.fitUnrollEnabled = source.fitUnrollEnabled;
    destination.gpuCompactOutput = source.compactOutput;
    destination.gpuKernelLaunches = source.kernelLaunches;
    destination.gpuQueueWaits = source.queueWaits;
    destination.gpuAllocationRounds = source.capacityGrowths;
    destination.gpuHostToDeviceBytes = source.hostToDeviceBytes;
    destination.gpuDeviceToHostBytes = source.deviceToHostBytes;
    destination.gpuTransactionStatusBytes = source.transactionStatusBytes;
    destination.gpuRetainedSnapshotBytes = source.retainedSnapshotBytes;
    destination.gpuHostAssemblyMilliseconds =
      source.timing.hostAssemblyMilliseconds;
    destination.gpuAllocatedBytesHighWater = source.allocatedBytesHighWater;
    destination.gpuTwoDaughterRealPairs = source.twoDaughterRealPairs;
    destination.gpuTwoDaughterPairTiles = source.twoDaughterPairTiles;
    destination.gpuTwoDaughterScheduledLanes = source.twoDaughterScheduledLanes;
    destination.gpuTwoDaughterTailLanes = source.twoDaughterTailLanes;
    destination.gpuTwoDaughterRectangularLanesAvoided =
      source.twoDaughterRectangularLanesAvoided;
    destination.gpuTwoDaughterPairTileBytes = source.twoDaughterPairTileBytes;
    destination.pairTilePlanningMilliseconds =
      source.timing.pairTilePlanningMilliseconds;
    destination.gpuVisitedCombinations = source.visitedCombinations;
    destination.gpuStoredTasks = source.storedTasks;
    destination.gpuBlockReservations = source.blockReservations;
    destination.gpuMaskDensity = source.maskDensity;
    destination.gpuUsefulWorkPerLaunch = source.usefulWorkPerLaunch;
    destination.gpuCandidatePoolOccupancy = source.candidatePoolOccupancy;
    destination.graphConstructionMilliseconds =
      source.timing.cascadeConstructionMilliseconds;
  }

  bool HasSameChannelLayout(const KFParticleGpuDecayPlan& left,
                        const KFParticleGpuDecayPlan& right)
  {
    // Verify every family, but preserve deliberate cut overrides in a complete
    // plan (e.g. the exact-boundary regression). This is not settings equality.
    if (left.NumberOfTwoDaughterChannels() != right.NumberOfTwoDaughterChannels()
        || left.NumberOfV0TrackCascadeChannels() != right.NumberOfV0TrackCascadeChannels()
        || left.NumberOfGraphOperationChannels() != right.NumberOfGraphOperationChannels()) {
      return false;
    }
    for (std::size_t index = 0u; index < left.NumberOfTwoDaughterChannels(); ++index) {
      if (left.TwoDaughterChannel(index).channelId
          != right.TwoDaughterChannel(index).channelId) { return false; }
    }
    for (std::size_t index = 0u; index < left.NumberOfV0TrackCascadeChannels(); ++index) {
      if (left.V0TrackCascadeChannel(index).channelId
          != right.V0TrackCascadeChannel(index).channelId) { return false; }
    }
    for (std::size_t index = 0u; index < left.NumberOfGraphOperationChannels(); ++index) {
      if (left.GraphOperationChannel(index).node.channelId
          != right.GraphOperationChannel(index).node.channelId) { return false; }
    }
    return true;
  }

  void EnsureRequestedDecayPlan(KFParticleGpuDecayPlan& plan,
                       const std::vector<int>& requestedMotherPdgs)
  {
    if (requestedMotherPdgs.empty()) {
      KFParticleGpuDecayPlan complete;
      AddCompleteCpuFinderChannels(complete);
      if (HasSameChannelLayout(plan, complete)) { return; }
      plan.Clear();
      AddCompleteCpuFinderChannels(plan);
      return;
    }
    KFParticleGpuDecayPlan requested;
    AddRequestedCpuFinderChannels(requested, requestedMotherPdgs);
    if (HasSameChannelLayout(plan, requested)) { return; }
    plan.Clear();
    AddRequestedCpuFinderChannels(plan, requestedMotherPdgs);
  }

  void AppendCandidateSnapshots(
    const KFParticleGpuCandidatePoolView& candidates,
    const KFParticleGpuV0SelectionResultView& selectionResults,
    const std::vector<bool>& selectedByRawIndex,
    const KFParticleGpuCandidateRange& range,
    std::uint64_t sourceEventId,
    bool hasSelectionResults,
    std::vector<kfparticle::gpu::GpuRawCandidate>& destination)
  {
    const auto candidateView = MakeConstView(candidates);
    for (unsigned int index = range.offset; index < range.offset + range.size; ++index) {
      const KFParticleGpuV0SelectionResult* selection =
        hasSelectionResults && selectionResults.CanStore(index)
          ? &selectionResults.Result(index)
          : nullptr;
      KFParticleGpuParitySnapshot parity;
      if (!KFParticleGpuParity::BuildSnapshot(candidateView,
                                               index,
                                               sourceEventId,
                                               selection,
                                               index < selectedByRawIndex.size()
                                                 && selectedByRawIndex[index],
                                               parity)) {
        continue;
      }

      kfparticle::gpu::GpuRawCandidate candidate;
      candidate.channelId = parity.key.channelId;
      candidate.pdg = parity.key.pdg;
      for (unsigned int daughter = 0u;
           daughter < 2u && daughter < parity.key.lineageSize;
           ++daughter) {
        candidate.daughterSourceIds[daughter] = parity.key.lineage[daughter];
      }
      candidate.mass = parity.mass;
      candidate.massError = parity.massError;
      candidate.chi2 = parity.chi2;
      candidate.ndf = parity.ndf;
      candidate.primaryVertexIndex = parity.primaryVertexIndex;
      candidate.bestPrimaryVertexIndex = parity.bestPrimaryVertexIndex;
      candidate.candidateFlags = parity.candidateFlags;
      candidate.selectionClass = parity.selectionClass;
      candidate.rejectionReasons = parity.rejectionReasons;
      candidate.topologyStatus = parity.topologyStatus;
      candidate.topology = parity.topology;
      candidate.outputClass = parity.outputClass;
      candidate.operationStatus = parity.operationStatus;
      candidate.selectionObservables = parity.selectionObservables;
      candidate.selected = parity.selected != 0u;
      candidate.massValid = parity.massValid != 0u;
      candidate.parity = parity;
      destination.push_back(candidate);
    }
  }
}  // namespace

namespace kfparticle::gpu
{
  namespace
  {
    struct alignas(64) MaterializationWorkerState {
      KFParticleGpuMaterializationWorkspace workspace;
      KFParticleGpuMaterializationTiming timing;
      std::uint64_t estimatedWeight = 0u;
      std::uint64_t physicalTracks = 0u;
      std::uint64_t candidates = 0u;
      std::uint64_t events = 0u;
      double elapsedMilliseconds = 0.;
      bool used = false;
    };
  }

  GpuExecutor::GpuExecutor(unsigned int materializationThreads,
                                           unsigned int packingThreads)
    : fMaterializationThreads(materializationThreads),
      fPackingThreads(packingThreads)
  {
    if (fMaterializationThreads == 0u
        || fMaterializationThreads
             > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
        "KFParticle materialization worker limit must fit a positive int");
    }
    if (fPackingThreads == 0u
        || fPackingThreads
             > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
        "KFParticle packing worker limit must fit a positive int");
    }
  }

  std::vector<GpuEventResult> GpuExecutor::RunBatch(
    const GpuPreparedBatch& batch,
    GpuOutputMode outputMode) const
  {
    return RunBatchConfigured(batch, outputMode,
      outputMode == GpuOutputMode::PublishableOutput, ExecutionSettings{});
  }

  std::vector<GpuEventResult> GpuExecutor::RunBatchConfigured(
    const GpuPreparedBatch& batch, GpuOutputMode outputMode,
    bool materializeOutput, const ExecutionSettings& settings) const
  {
    const unsigned int repetitions = InProcessRepeatCount();
    std::vector<GpuEventResult> results;
    RepeatIdentity expectedIdentity;
    for (unsigned int sample = 1u; sample <= repetitions; ++sample) {
      results = ExecuteBatchOnce(
        batch, outputMode, materializeOutput, settings);
      const RepeatIdentity identity = MakeRepeatIdentity(results);
      if (sample == 1u) {
        expectedIdentity = identity;
      }
      else if (!(identity == expectedIdentity)) {
        for (auto& result : results) {
          result.status = GpuExecutionStatus::ExecutionFailed;
          result.message = "KFParticle in-process repeat changed output identity";
        }
      }

      if (repetitions > 1u) {
        const GpuEventResult empty;
        const auto& timing = results.empty() ? empty : results.front();
        std::fprintf(stderr,
          "KFP_IN_PROCESS_SAMPLE contract=1 sample=%u/%u mode=%s events=%u succeeded=%u candidates=%llu selected=%llu overflow=%u source_xor=%llu source_sum=%llu signature_count=%llu signature_xor=%llu signature_sum=%llu transaction_ms=%.6f pipeline_ms=%.6f queue_wait_ms=%.6f queue_capacity_ms=%.6f packing_ms=%.6f packing_threads_requested=%u packing_threads_used=%u h2d_ms=%.6f construction_ms=%.6f graph_ms=%.6f selection_ms=%.6f d2h_ms=%.6f extraction_ms=%.6f materialization_ms=%.6f materialization_cpu_work_ms=%.6f materialization_threads_requested=%u materialization_threads_used=%u pair_plan_ms=%.6f counting_readback_ms=%.6f counting_readback_bytes=%llu dense_plan_ms=%.6f validation_packing_ms=%.6f validation_initial_ms=%.6f validation_compact_ms=%.6f initial_allocations=%llu initial_allocation_ms=%.6f replacement_allocations=%llu replacement_allocation_ms=%.6f launches=%u queue_waits=%u real_pairs=%llu h2d_bytes=%llu d2h_bytes=%llu output_mode=%s host_assembly_ms=%.6f status_bytes=%llu retained_snapshot_bytes=%llu\n",
          sample, repetitions,
          KFParticleGpuReferenceTransactionValidationRequested()
            ? "reference" : "optimized",
          identity.events, identity.succeeded,
          static_cast<unsigned long long>(identity.candidates),
          static_cast<unsigned long long>(identity.selected), identity.overflow,
          static_cast<unsigned long long>(identity.sourceXor),
          static_cast<unsigned long long>(identity.sourceSum),
          static_cast<unsigned long long>(identity.signatureCandidates),
          static_cast<unsigned long long>(identity.signatureXor),
          static_cast<unsigned long long>(identity.signatureSum),
          timing.transactionMilliseconds, timing.gpuPipelineMilliseconds,
          timing.queueWaitMilliseconds, timing.queueAndCapacityMilliseconds,
          timing.hostPackingMilliseconds, timing.packingThreadsRequested,
          timing.packingThreadsUsed, timing.inputUploadMilliseconds,
          timing.constructionMilliseconds, timing.graphConstructionMilliseconds,
          timing.selectionMilliseconds, timing.outputDownloadMilliseconds,
          timing.resultExtractionMilliseconds,
          SumMaterializationMilliseconds(results),
          SumMaterializationCpuWorkMilliseconds(results),
          timing.materializationThreadsRequested,
          timing.materializationThreadsUsed,
          timing.pairTilePlanningMilliseconds,
          timing.hostTiming.countingReadbackMilliseconds,
          static_cast<unsigned long long>(timing.hostTiming.countingReadbackBytes),
          timing.hostTiming.denseOutputPlanningMilliseconds,
          timing.hostTiming.packingValidationMilliseconds,
          timing.hostTiming.initialTransactionValidationMilliseconds,
          timing.hostTiming.compactOutputValidationMilliseconds,
          static_cast<unsigned long long>(timing.hostTiming.initialAllocations),
          timing.hostTiming.initialAllocationMilliseconds,
          static_cast<unsigned long long>(timing.hostTiming.replacementAllocations),
          timing.hostTiming.replacementAllocationMilliseconds,
          timing.gpuKernelLaunches, timing.gpuQueueWaits,
          static_cast<unsigned long long>(timing.gpuTwoDaughterRealPairs),
          static_cast<unsigned long long>(timing.gpuHostToDeviceBytes),
          static_cast<unsigned long long>(timing.gpuDeviceToHostBytes),
          timing.gpuCompactOutput ? "compact" : "full",
          timing.gpuHostAssemblyMilliseconds,
          static_cast<unsigned long long>(timing.gpuTransactionStatusBytes),
          static_cast<unsigned long long>(timing.gpuRetainedSnapshotBytes));
      }
      if (identity.succeeded != identity.events || identity.overflow != 0u
          || !(identity == expectedIdentity)) {
        break;
      }
    }
    return results;
  }

  std::vector<GpuEventResult> GpuExecutor::ExecuteBatchOnce(
    const GpuPreparedBatch& batch,
    GpuOutputMode outputMode,
    bool materializeOutput,
    const ExecutionSettings& settings) const
  {
    std::vector<GpuEventResult> results(batch.EventCount());
    for (unsigned int index = 0u; index < batch.EventCount(); ++index) {
      const auto& event = batch.Event(index);
      auto& result = results[index];
      result.sourceEventId = event.sourceEventId;
      result.vertices = event.descriptor.primaryVertices.size;
      result.taskCapacity = event.taskCapacity;
      result.inputPreparationMilliseconds = event.inputPreparationMilliseconds;
      result.capacityPlanningMilliseconds = event.capacityPlanningMilliseconds;
      result.capacityPlanSetupMilliseconds = event.capacityPlanSetupMilliseconds;
      result.capacityPlanBuilds = event.capacityPlanBuilds;
      result.fullDiagnosticSnapshot =
        outputMode == GpuOutputMode::FullDiagnostics;
      result.packingThreadsRequested = fPackingThreads;
      result.packingThreadsUsed = 1u;
      result.materializationThreadsRequested = fMaterializationThreads;
      result.materializationThreadsUsed = 1u;
      for (unsigned int trackSet = 0u; trackSet < NumberOfTrackSets; ++trackSet) {
        result.tracks += event.descriptor.trackSets[trackSet].tracks.size;
      }
    }
    if (batch.EventCount() == 0u) { return results; }

    const auto transactionStarted = std::chrono::steady_clock::now();
    const auto elapsedMilliseconds = [](std::chrono::steady_clock::time_point started) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    };
    try {
      // Empty/no-pair batches are a CPU-side result and must not attach XPU.
      if (batch.TaskCapacity() == 0u
          || std::none_of(results.begin(), results.end(), [](const auto& result) {
               return result.tracks != 0u && result.vertices != 0u;
             })) {
        for (auto& result : results) {
          result.status = result.tracks == 0u || result.vertices == 0u ? GpuExecutionStatus::EmptyInput
                                                                         : GpuExecutionStatus::NoPairs;
          result.message = result.status == GpuExecutionStatus::EmptyInput
                             ? "no physical tracks or primary vertices"
                             : "default CPU finder plan has no input pairs";
          result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
        }
        return results;
      }

      KFParticleGpuStepTrace queueTrace("queue-and-buffers", true);
      queueTrace.Next("CPU acquire serialized steering service");
      const auto queueWaitStarted = std::chrono::steady_clock::now();
      auto transaction = settings.runtime
        ? GpuRuntimeService::Instance().Acquire(*settings.runtime)
        : GpuRuntimeService::Instance().Acquire();
      const double queueWaitMilliseconds = elapsedMilliseconds(queueWaitStarted);
      const auto queueAndCapacityStarted = std::chrono::steady_clock::now();
      queueTrace.Next("CPU ensure decay plan");
      auto& steering = transaction.Steering();
      steering.SetPerformanceMonitoringEnabled(settings.inheritEnvironment
        && PerformanceMonitoringRequested());
      auto& plan = steering.GetDecayPlan();
      EnsureRequestedDecayPlan(plan, batch.RequestedMotherPdgs());
      auto& buffers = steering.GetBuffers();
      const char* allocationTrace = settings.inheritEnvironment
        ? std::getenv("KFPARTICLE_GPU_ALLOCATION_TRACE") : nullptr;
      buffers.BeginAllocationTrace(allocationTrace && std::string(allocationTrace) == "1");
      const std::uint64_t allocationRoundsBefore = buffers.CapacityGrowthCount();
      const auto allocationBefore = buffers.HostTiming();
      queueTrace.Next("CPU/ALLOC input and routing capacities");
      const auto routingCapacities = batch.RoutingCapacities();
      buffers.EnsureCapacity(routingCapacities);
      buffers.SetInputSizes(batch.TrackCount(), batch.VertexCount(), batch.EventCount());
      buffers.SetTransactionSizes(1u, batch.EventCount(), batch.EventCount(), batch.EventCount());
      const double queueAndCapacityMilliseconds =
        elapsedMilliseconds(queueAndCapacityStarted);
      queueTrace.Finish();

      const auto packingStarted = std::chrono::steady_clock::now();
      const unsigned int packingThreadsUsed = batch.Write(
        buffers.HostInputTracks(), buffers.HostPrimaryVertices(),
        buffers.HostEvents(), fPackingThreads);
      batch.WriteTransaction(buffers.HostTransaction(), plan.Revision(), 0u, true);
      const auto packingValidationStarted = std::chrono::steady_clock::now();
      const auto transactionValidation = KFParticleGpuValidateTransactionConfigured(
        buffers.HostTransaction(),
        plan.Revision(),
        batch.TrackCount(),
        batch.VertexCount(),
        batch.EventCount(),
        batch.Capacities().twoDaughterTasks,
        routingCapacities.candidates,
        routingCapacities.daughterIds,
        routingCapacities.selectedCandidates);
      const double packingValidationMilliseconds =
        elapsedMilliseconds(packingValidationStarted);
      if (transactionValidation != KFParticleGpuTransactionValid) {
        throw std::runtime_error("KFParticle GPU batch transaction contract is invalid");
      }
      const double packingMilliseconds = elapsedMilliseconds(packingStarted);

      const auto pipelineStarted = std::chrono::steady_clock::now();
      const auto& channelResults = steering.RunDecayPlanBatch(
        0u,
        batch.EventCount(),
        batch.TaskCapacity(),
        outputMode == GpuOutputMode::FullDiagnostics
          ? KFGpuFullDiagnosticOutput
          : KFGpuPublishableOutput);
      const double pipelineMilliseconds = elapsedMilliseconds(pipelineStarted);
      const auto& timing = steering.LastDecayPlanTiming();
      CopyPerformanceSnapshot(steering.LastPerformanceSnapshot(), results.front());
      results.front().gpuAllocationRounds =
        buffers.CapacityGrowthCount() - allocationRoundsBefore;
      const auto& eventResults = steering.LastDecayPlanEventResults();
      if (eventResults.size() != results.size()) {
        throw std::logic_error("KFParticle GPU batch returned an unexpected event-result count");
      }

      const auto extractionStarted = std::chrono::steady_clock::now();
      const auto candidates = buffers.HostCandidates();
      std::vector<bool> selectedByRawIndex;
      if (outputMode == GpuOutputMode::FullDiagnostics) {
        const auto selected = buffers.HostSelectedCandidates();
        selectedByRawIndex.assign(candidates.Size(), false);
        if (selected.Size() > selected.Capacity()) {
          throw std::logic_error("KFParticle GPU selected output exceeds its capacity");
        }
        // Size() is a high-water mark for disjoint event reservations, not a
        // populated prefix. Padding may still contain indices from an older run.
        for (const auto& eventResult : eventResults) {
          const auto& range = eventResult.selectedCandidates;
          if (range.Empty()) { continue; }
          if (range.offset > selected.Size()
              || range.size > selected.Size() - range.offset) {
            throw std::logic_error("KFParticle GPU event selected range is out of bounds");
          }
          for (unsigned int index = range.offset; index < range.End(); ++index) {
            const unsigned int rawIndex = selected.Index(index);
            if (rawIndex >= selectedByRawIndex.size()
                || rawIndex >= candidates.Capacity()
                || rawIndex < eventResult.candidates.offset
                || rawIndex - eventResult.candidates.offset >= eventResult.candidates.size
                || candidates.Metadata().EventIndex(rawIndex) != eventResult.eventIndex
                || (selected.ChannelIdsData()
                    && selected.ChannelId(index) != candidates.Metadata().ChannelId(rawIndex))
                || selectedByRawIndex[rawIndex]) {
              throw std::logic_error("KFParticle GPU selected entry has invalid event or candidate ownership");
            }
            selectedByRawIndex[rawIndex] = true;
          }
        }
      }
      const auto selectionResults = outputMode == GpuOutputMode::FullDiagnostics
        ? buffers.HostV0SelectionResults()
        : KFParticleGpuV0SelectionResultView(nullptr, 0u);
      const auto candidateView = MakeConstView(candidates);
      const auto trackView = MakeConstView(buffers.HostInputTracks());
      const KFParticleGpuEventDesc* hostEvents = buffers.HostEvents();
      const bool performanceMonitoring = steering.PerformanceMonitoringEnabled();
      const char* referenceOption = settings.inheritEnvironment
        ? std::getenv("KFPARTICLE_GPU_HOST_RESULT_REFERENCE") : nullptr;
      const bool referenceMaterialization = referenceOption && std::string(referenceOption) == "1";
      for (unsigned int eventIndex = 0u; eventIndex < results.size(); ++eventIndex) {
        auto& result = results[eventIndex];
        const auto& eventResult = eventResults[eventIndex];
        result.hostPackingMilliseconds = packingMilliseconds;
        result.packingThreadsUsed = packingThreadsUsed;
        result.queueWaitMilliseconds = queueWaitMilliseconds;
        result.queueAndCapacityMilliseconds = queueAndCapacityMilliseconds;
        result.inputUploadMilliseconds = timing.inputUploadMilliseconds;
        result.constructionMilliseconds = timing.constructionMilliseconds;
        result.graphConstructionMilliseconds =
          timing.cascadeConstructionMilliseconds;
        result.selectionMilliseconds = timing.selectionMilliseconds;
        result.outputDownloadMilliseconds = timing.outputDownloadMilliseconds;
        result.gpuPipelineMilliseconds = pipelineMilliseconds;
        result.candidates = CheckedCountSum(
          CheckedCountSum(eventResult.candidates.size, eventResult.cascadeCandidates.size),
          eventResult.graphCandidates.size);
        result.selectedCandidates = eventResult.selectedCandidates.size;
        result.overflowFlags = eventResult.overflowFlags;

        if (outputMode == GpuOutputMode::FullDiagnostics) {
          AppendCandidateSnapshots(candidates,
                               selectionResults,
                               selectedByRawIndex,
                               eventResult.candidates,
                               result.sourceEventId,
                               true,
                               result.rawCandidates);
          AppendCandidateSnapshots(candidates,
                               selectionResults,
                               selectedByRawIndex,
                               eventResult.cascadeCandidates,
                               result.sourceEventId,
                               false,
                               result.rawCandidates);
          AppendCandidateSnapshots(candidates,
                               selectionResults,
                               selectedByRawIndex,
                               eventResult.graphCandidates,
                               result.sourceEventId,
                               false,
                               result.rawCandidates);
          for (unsigned int channelOffset = 0u; channelOffset < eventResult.channelCount; ++channelOffset) {
            const auto& channel = channelResults[eventResult.channelOffset + channelOffset];
            GpuChannelSummary summary;
            summary.channelId = channel.channelId;
            summary.motherPdg = channel.motherPdg;
            summary.totalPairs = channel.totalPairs;
            summary.storedTasks = channel.storedTasks;
            summary.candidates = channel.constructedCandidates;
            summary.overflowFlags = channel.candidates.overflowFlags;
            summary.truncated = channel.Truncated();
            result.overflowFlags |= summary.overflowFlags;
            result.channels.push_back(summary);
          }
        }
        if (materializeOutput && result.overflowFlags == 0u) {
          const KFParticleGpuCandidateRange ranges[] = {
            eventResult.candidates,
            eventResult.cascadeCandidates,
            eventResult.graphCandidates};
          for (const auto& sourceRange : ranges) {
            for (unsigned int index = sourceRange.offset;
                 index < sourceRange.offset + sourceRange.size;
                 ++index) {
              const auto& metadata = candidateView.Metadata();
              if (metadata.EventIndex(index) != eventIndex
                  || metadata.OperationStatus(index)
                       != KFGpuCandidateOperationAccepted
                  || (metadata.Flags(index) & KFGpuCandidateValid) == 0u) {
                continue;
              }
              if (performanceMonitoring) {
                RecordOutputSignature(candidateView, index, result);
              }
            }
          }
        }
        else if (result.overflowFlags != 0u) {
          result.materializationStatus = KFGpuMaterializationOverflow;
        }
        if (result.tracks == 0u || result.vertices == 0u) {
          result.status = GpuExecutionStatus::EmptyInput;
          result.message = "no physical tracks or primary vertices";
        }
        else if (result.taskCapacity == 0u) {
          result.status = GpuExecutionStatus::NoPairs;
          result.message = "default CPU finder plan has no input pairs";
        }
        else {
          result.status = result.overflowFlags == 0u ? GpuExecutionStatus::Completed : GpuExecutionStatus::Overflow;
          result.message = result.status == GpuExecutionStatus::Completed ? "default CPU finder plan completed"
                                                                            : "GPU candidate pool overflow";
        }
      }

      const auto materializationWallStarted = std::chrono::steady_clock::now();
      std::vector<unsigned int> materializableEvents;
      if (materializeOutput) {
        materializableEvents.reserve(results.size());
        for (unsigned int eventIndex = 0u; eventIndex < results.size(); ++eventIndex) {
          if (results[eventIndex].overflowFlags == 0u) {
            materializableEvents.push_back(eventIndex);
          }
        }
      }

      unsigned int materializationWorkerLimit = 1u;
#ifdef _OPENMP
      if (!referenceMaterialization && !materializableEvents.empty()) {
        materializationWorkerLimit = static_cast<unsigned int>(
          std::min<std::size_t>(fMaterializationThreads,
                                materializableEvents.size()));
      }
#endif
      std::vector<MaterializationWorkerState> materializationWorkers(
        materializationWorkerLimit);
      std::vector<std::exception_ptr> materializationExceptions(results.size());
      const bool validatedPublicationInput =
        !referenceMaterialization
        && std::all_of(
          materializableEvents.begin(), materializableEvents.end(),
          [&batch](unsigned int eventIndex) {
            return batch.Event(eventIndex).validatedPublicationInput;
          });

      std::vector<std::uint64_t> materializationEventWeights;
      std::vector<unsigned int> materializationEventTracks;
      std::vector<unsigned int> materializationEventCandidates;
      materializationEventWeights.reserve(materializableEvents.size());
      if (performanceMonitoring) {
        materializationEventTracks.reserve(materializableEvents.size());
        materializationEventCandidates.reserve(materializableEvents.size());
      }
      for (unsigned int eventIndex : materializableEvents) {
        unsigned int physicalTracks = 0u;
        for (unsigned int set = 0u; set < 4u; ++set) {
          physicalTracks = CheckedCountSum(
            physicalTracks, hostEvents[eventIndex].trackSets[set].tracks.size);
        }
        const auto& eventResult = eventResults[eventIndex];
        const unsigned int candidates = CheckedCountSum(
          eventResult.candidates.size,
          CheckedCountSum(eventResult.cascadeCandidates.size,
                     eventResult.graphCandidates.size));
        materializationEventWeights.push_back(GpuMaterializationEventWeight(
          physicalTracks, eventResult.candidates.size,
          eventResult.cascadeCandidates.size, eventResult.graphCandidates.size));
        if (performanceMonitoring) {
          materializationEventTracks.push_back(physicalTracks);
          materializationEventCandidates.push_back(candidates);
        }
      }
      const char* requestedMaterializationSchedule =
        settings.inheritEnvironment
          ? std::getenv("KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE") : nullptr;
      const auto materializationScheduleMode = ResolveGpuMaterializationSchedule(
        requestedMaterializationSchedule ? requestedMaterializationSchedule : "",
        performanceMonitoring);
      std::vector<GpuMaterializationWorkerRange> materializationSchedule;
      std::exception_ptr materializationScheduleException;

      const auto materializePosition = [&](unsigned int workerIndex,
                                           std::size_t position) {
        const unsigned int eventIndex = materializableEvents[position];
        auto& worker = materializationWorkers[workerIndex];
        worker.used = true;
        std::chrono::steady_clock::time_point materializationStarted;
        if (performanceMonitoring) {
          ++worker.events;
          worker.estimatedWeight = GpuMaterializationSaturatingAdd(
            worker.estimatedWeight, materializationEventWeights[position]);
          worker.physicalTracks = GpuMaterializationSaturatingAdd(
            worker.physicalTracks, materializationEventTracks[position]);
          worker.candidates = GpuMaterializationSaturatingAdd(
            worker.candidates, materializationEventCandidates[position]);
          materializationStarted = std::chrono::steady_clock::now();
        }
        try {
          auto& result = results[eventIndex];
          const auto& eventResult = eventResults[eventIndex];
          const KFParticleGpuCandidateRange ranges[] = {
            eventResult.candidates,
            eventResult.cascadeCandidates,
            eventResult.graphCandidates};
          KFParticleGpuMaterializationRequest request;
          request.tracks = trackView;
          request.candidates = candidateView;
          request.ranges = ranges;
          request.numberOfRanges = sizeof(ranges) / sizeof(ranges[0]);
          request.eventIndex = eventIndex;
          request.event = &hostEvents[eventIndex];
          request.includeAllInputTracks = true;
          request.completePublicationPlan = true;
          request.validatedPublicationInput = validatedPublicationInput;
          request.workspace = referenceMaterialization ? nullptr : &worker.workspace;
          request.timing = performanceMonitoring ? &worker.timing : nullptr;
          result.materializationSucceeded =
            KFParticleGpuMaterializer::CommitToParticles(
              request,
              result.materializedParticles,
              nullptr,
              &result.materializationStatus);
        }
        catch (...) {
          materializationExceptions[eventIndex] = std::current_exception();
        }
        if (performanceMonitoring) {
          const double elapsed = elapsedMilliseconds(materializationStarted);
          results[eventIndex].materializationMilliseconds = elapsed;
          worker.elapsedMilliseconds += elapsed;
        }
      };

      const auto materializeWeightedWorker = [&](unsigned int workerIndex) {
        if (workerIndex >= materializationSchedule.size()) { return; }
        const auto& workerRange = materializationSchedule[workerIndex];
        for (std::size_t position = workerRange.begin;
             position < workerRange.end; ++position) {
          materializePosition(workerIndex, position);
        }
      };

#ifdef _OPENMP
#pragma omp parallel num_threads(materializationWorkerLimit)
      {
        const unsigned int workerIndex =
          static_cast<unsigned int>(omp_get_thread_num());
        const long long eventCount =
          static_cast<long long>(materializableEvents.size());
        if (materializationScheduleMode
            == GpuMaterializationScheduleMode::Static) {
#pragma omp for schedule(static)
          for (long long position = 0; position < eventCount; ++position) {
            materializePosition(workerIndex, static_cast<std::size_t>(position));
          }
        }
        else if (materializationScheduleMode
                 == GpuMaterializationScheduleMode::Dynamic) {
#pragma omp for schedule(dynamic, 1)
          for (long long position = 0; position < eventCount; ++position) {
            materializePosition(workerIndex, static_cast<std::size_t>(position));
          }
        }
        else if (materializationScheduleMode
                 == GpuMaterializationScheduleMode::Guided) {
#pragma omp for schedule(guided, 1)
          for (long long position = 0; position < eventCount; ++position) {
            materializePosition(workerIndex, static_cast<std::size_t>(position));
          }
        }
        else {
#pragma omp single
          {
            try {
              materializationSchedule = PlanGpuMaterializationWorkers(
                materializationEventWeights,
                static_cast<unsigned int>(omp_get_num_threads()));
            }
            catch (...) {
              materializationScheduleException = std::current_exception();
            }
          }
          if (!materializationScheduleException) {
            materializeWeightedWorker(workerIndex);
          }
        }
      }
#else
      if (materializationScheduleMode
          == GpuMaterializationScheduleMode::WeightedContiguous) {
        materializationSchedule = PlanGpuMaterializationWorkers(
          materializationEventWeights, 1u);
        materializeWeightedWorker(0u);
      }
      else {
        for (std::size_t position = 0u;
             position < materializableEvents.size(); ++position) {
          materializePosition(0u, position);
        }
      }
#endif

      if (materializationScheduleException) {
        std::rethrow_exception(materializationScheduleException);
      }

      for (const auto& exception : materializationExceptions) {
        if (exception) {
          for (auto& result : results) {
            result.materializedParticles.clear();
            result.materializationSucceeded = false;
          }
          std::rethrow_exception(exception);
        }
      }

      const double materializationWall = materializableEvents.empty()
                                           ? 0.
                                           : elapsedMilliseconds(materializationWallStarted);
      const unsigned int materializationThreadsUsed = std::max(
        1u,
        static_cast<unsigned int>(std::count_if(
          materializationWorkers.begin(), materializationWorkers.end(),
          [](const MaterializationWorkerState& worker) { return worker.used; })));
      KFParticleGpuMaterializationTiming materializationTiming;
      std::size_t materializationScratchInitialized = 0u;
      std::size_t materializationScratchReset = 0u;
      std::size_t materializationScratchCapacityBytes = 0u;
      std::size_t materializationCandidateMapCapacityEntries = 0u;
      std::size_t materializationTrackMapCapacityEntries = 0u;
      std::uint64_t materializationEstimatedWeightMinimum =
        std::numeric_limits<std::uint64_t>::max();
      std::uint64_t materializationEstimatedWeightMaximum = 0u;
      std::uint64_t materializationEstimatedWeightTotal = 0u;
      double materializationWorkerCpuMinimum =
        std::numeric_limits<double>::max();
      double materializationWorkerCpuMaximum = 0.;
      std::uint64_t materializationWorkerEventsMinimum =
        std::numeric_limits<std::uint64_t>::max();
      std::uint64_t materializationWorkerEventsMaximum = 0u;
      std::uint64_t materializationWorkerTracksMinimum =
        std::numeric_limits<std::uint64_t>::max();
      std::uint64_t materializationWorkerTracksMaximum = 0u;
      std::uint64_t materializationWorkerCandidatesMinimum =
        std::numeric_limits<std::uint64_t>::max();
      std::uint64_t materializationWorkerCandidatesMaximum = 0u;
      for (unsigned int worker = 0u; worker < materializationWorkerLimit; ++worker) {
        const auto& workerState = materializationWorkers[worker];
        materializationTiming.graphMilliseconds +=
          workerState.timing.graphMilliseconds;
        materializationTiming.leavesMilliseconds +=
          workerState.timing.leavesMilliseconds;
        materializationTiming.candidatesMilliseconds +=
          workerState.timing.candidatesMilliseconds;
        materializationTiming.commitMilliseconds +=
          workerState.timing.commitMilliseconds;
        materializationScratchInitialized +=
          workerState.workspace.initializedEntries;
        materializationScratchReset += workerState.workspace.resetEntries;
        materializationScratchCapacityBytes +=
          workerState.workspace.CapacityBytes();
        materializationCandidateMapCapacityEntries +=
          workerState.workspace.CandidateMapCapacityEntries();
        materializationTrackMapCapacityEntries +=
          workerState.workspace.TrackMapCapacityEntries();
        if (workerState.used) {
          const std::uint64_t weight = workerState.estimatedWeight;
          materializationEstimatedWeightMinimum = std::min(
            materializationEstimatedWeightMinimum, weight);
          materializationEstimatedWeightMaximum = std::max(
            materializationEstimatedWeightMaximum, weight);
          materializationEstimatedWeightTotal = GpuMaterializationSaturatingAdd(
            materializationEstimatedWeightTotal, weight);
          materializationWorkerCpuMinimum = std::min(
            materializationWorkerCpuMinimum,
            workerState.elapsedMilliseconds);
          materializationWorkerCpuMaximum = std::max(
            materializationWorkerCpuMaximum,
            workerState.elapsedMilliseconds);
          materializationWorkerEventsMinimum = std::min(
            materializationWorkerEventsMinimum, workerState.events);
          materializationWorkerEventsMaximum = std::max(
            materializationWorkerEventsMaximum, workerState.events);
          materializationWorkerTracksMinimum = std::min(
            materializationWorkerTracksMinimum, workerState.physicalTracks);
          materializationWorkerTracksMaximum = std::max(
            materializationWorkerTracksMaximum, workerState.physicalTracks);
          materializationWorkerCandidatesMinimum = std::min(
            materializationWorkerCandidatesMinimum, workerState.candidates);
          materializationWorkerCandidatesMaximum = std::max(
            materializationWorkerCandidatesMaximum, workerState.candidates);
        }
      }
      if (!performanceMonitoring
          || (materializationThreadsUsed == 1u
              && !materializationWorkers.front().used)) {
        materializationEstimatedWeightMinimum = 0u;
        materializationWorkerCpuMinimum = 0.;
        materializationWorkerEventsMinimum = 0u;
        materializationWorkerTracksMinimum = 0u;
        materializationWorkerCandidatesMinimum = 0u;
      }

      double materializationCpuWork = 0.;
      for (auto& result : results) {
        result.materializationCpuWorkMilliseconds =
          result.materializationMilliseconds;
        materializationCpuWork += result.materializationCpuWorkMilliseconds;
        result.materializationMilliseconds = 0.;
        result.materializationThreadsUsed = materializationThreadsUsed;
      }
      results.front().materializationMilliseconds = materializationWall;
      if (settings.inheritEnvironment && performanceMonitoring
          && outputMode != GpuOutputMode::FullDiagnostics) {
        // Keep machine-readable performance records on the same stream.  When
        // cbmreco redirects stdout and stderr into one log, independently
        // buffered long records can otherwise be spliced together.
        std::fprintf(stderr,
          "KFP_HOST_RESULT contract=1 reference=%u publication_plan=1 event_local_maps=1 validated_publication_input=%u repeated_track_validation=%u materialization_schedule=%s events=%zu materialization_ms=%.6f materialization_cpu_work_ms=%.6f materialization_threads_requested=%u materialization_threads_used=%u worker_estimated_weight_min=%llu worker_estimated_weight_max=%llu worker_estimated_weight_total=%llu worker_events_min=%llu worker_events_max=%llu worker_tracks_min=%llu worker_tracks_max=%llu worker_candidates_min=%llu worker_candidates_max=%llu worker_cpu_min_ms=%.6f worker_cpu_max_ms=%.6f subphase_metric=cpu_work graph_ms=%.6f leaves_ms=%.6f candidates_ms=%.6f commit_ms=%.6f scratch_initialized=%zu scratch_reset=%zu scratch_capacity_bytes=%zu candidate_map_capacity_entries=%zu track_map_capacity_entries=%zu\n",
          referenceMaterialization ? 1u : 0u,
          validatedPublicationInput ? 1u : 0u,
          validatedPublicationInput ? 0u : 1u,
          GpuMaterializationScheduleName(materializationScheduleMode),
          results.size(),
          materializationWall, materializationCpuWork,
          fMaterializationThreads, materializationThreadsUsed,
          static_cast<unsigned long long>(materializationEstimatedWeightMinimum),
          static_cast<unsigned long long>(materializationEstimatedWeightMaximum),
          static_cast<unsigned long long>(materializationEstimatedWeightTotal),
          static_cast<unsigned long long>(materializationWorkerEventsMinimum),
          static_cast<unsigned long long>(materializationWorkerEventsMaximum),
          static_cast<unsigned long long>(materializationWorkerTracksMinimum),
          static_cast<unsigned long long>(materializationWorkerTracksMaximum),
          static_cast<unsigned long long>(materializationWorkerCandidatesMinimum),
          static_cast<unsigned long long>(materializationWorkerCandidatesMaximum),
          materializationWorkerCpuMinimum, materializationWorkerCpuMaximum,
          materializationTiming.graphMilliseconds, materializationTiming.leavesMilliseconds,
          materializationTiming.candidatesMilliseconds, materializationTiming.commitMilliseconds,
          materializationScratchInitialized, materializationScratchReset,
          materializationScratchCapacityBytes,
          materializationCandidateMapCapacityEntries,
          materializationTrackMapCapacityEntries);
        std::fflush(stderr);
      }
      // Include destruction of worker-local scratch in extraction/transaction.
      materializationWorkers.clear();
      results.front().resultExtractionMilliseconds =
        std::max(0., elapsedMilliseconds(extractionStarted) - materializationWall);
      results.front().hostTiming = buffers.HostTiming().Since(allocationBefore);
      results.front().hostTiming.countingReadbackMilliseconds =
        timing.countingReadbackMilliseconds;
      results.front().hostTiming.countingReadbackBytes =
        timing.countingReadbackBytes;
      results.front().hostTiming.denseOutputPlanningMilliseconds =
        timing.denseOutputPlanningMilliseconds;
      results.front().hostTiming.packingValidationMilliseconds =
        packingValidationMilliseconds;
      results.front().hostTiming.initialTransactionValidationMilliseconds =
        timing.initialTransactionValidationMilliseconds;
      results.front().hostTiming.compactOutputValidationMilliseconds =
        timing.compactOutputValidationMilliseconds;
      const double transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      for (auto& result : results) {
        result.transactionMilliseconds = transactionMilliseconds;
      }
      // Print only after closing transaction timers; no per-allocation console IO.
      buffers.PrintAllocationTrace();
      if (settings.inheritEnvironment && PerformanceMonitoringRequested()) {
        std::fprintf(stderr,
          "KFP_VALIDATION_TIMING contract=1 mode=%s events=%zu packing_ms=%.6f initial_ms=%.6f compact_ms=%.6f\n",
          KFParticleGpuReferenceTransactionValidationRequested() ? "reference" : "optimized",
          results.size(), packingValidationMilliseconds,
          timing.initialTransactionValidationMilliseconds,
          timing.compactOutputValidationMilliseconds);
        const auto memory = buffers.MemoryUsage();
        std::fprintf(stderr,
          "KFP_MEMORY contract=1 device_bytes=%llu pinned_host_bytes=%llu legacy_task_capacity=%u\n",
          static_cast<unsigned long long>(memory.deviceBytes),
          static_cast<unsigned long long>(memory.pinnedHostBytes),
          buffers.Capacities().twoDaughterTasks);
      }
    }
    catch (const std::invalid_argument& error) {
      for (auto& result : results) {
        result.status = ClassifyInputRejection(error.what());
        result.message = error.what();
        result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      }
    }
    catch (const std::overflow_error& error) {
      for (auto& result : results) {
        result.status = GpuExecutionStatus::InputRejected;
        result.message = error.what();
        result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      }
    }
    catch (const std::logic_error& error) {
      for (auto& result : results) {
        result.status = ClassifyRuntimeOrFieldFailure(error.what());
        result.message = error.what();
        result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      }
    }
    catch (const std::runtime_error& error) {
      for (auto& result : results) {
        result.status = GpuExecutionStatus::RuntimeUnavailable;
        result.message = error.what();
        result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      }
    }
    catch (const std::exception& error) {
      for (auto& result : results) {
        result.status = GpuExecutionStatus::ExecutionFailed;
        result.message = error.what();
        result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
      }
    }
    return results;
  }

  GpuEventResult GpuExecutor::Run(std::uint64_t sourceEventId,
                                                    const KFPTrackVector& first,
                                                    const KFPTrackVector& last,
                                                    const std::vector<float>& chiToPrimaryVertex,
                                                    const std::vector<KFVertex>& primaryVertices,
                                                    float minSecondaryTrackChiToPrimaryVertex,
                                                    GpuOutputMode outputMode) const
  {
    const GpuEventInput input{static_cast<unsigned int>(sourceEventId), &first, &last,
      &chiToPrimaryVertex, &primaryVertices, minSecondaryTrackChiToPrimaryVertex};
    return RunSingleConfigured(sourceEventId, input, outputMode, {}, true, ExecutionSettings{});
  }

  GpuEventResult GpuExecutor::RunSingleConfigured(
    std::uint64_t sourceEventId, const GpuEventInput& input,
    GpuOutputMode outputMode, const std::vector<int>& motherPdgs,
    bool materializeOutput, const ExecutionSettings& settings) const
  {
    GpuEventResult result;
    result.sourceEventId = sourceEventId;
    result.packingThreadsRequested = fPackingThreads;
    result.packingThreadsUsed = 1u;
    result.materializationThreadsRequested = fMaterializationThreads;
    result.materializationThreadsUsed = 1u;
    result.fullDiagnosticSnapshot =
      outputMode == GpuOutputMode::FullDiagnostics;
    const auto transactionStarted = std::chrono::steady_clock::now();
    const auto elapsedMilliseconds = [](std::chrono::steady_clock::time_point started) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    };

    try {
      // A single call has no batching admission ceiling. Checked GPU-ABI limits
      // still apply, and the prepared input borrows the caller's arrays.
      GpuBatchBuilder builder(DefaultGpuBatchLimits(1u), motherPdgs);
      if (builder.TryAppend(
            sourceEventId, input,
            GpuBatchAppendPolicy::AllowSingleEventBeyondResourceLimits)
          != GpuBatchAppendResult::Added) {
        throw std::overflow_error("KFParticle GPU single event exceeds the GPU ABI");
      }
      // Keep the single-event API's diagnostic materialization and one-shot
      // behavior; only RunBatch enables explicit in-process benchmark replay.
      auto results = ExecuteBatchOnce(builder.Seal(), outputMode, materializeOutput, settings);
      result = std::move(results.front());
    }
    catch (const std::invalid_argument& error) {
      result.status  = ClassifyInputRejection(error.what());
      result.message = error.what();
    }
    catch (const std::overflow_error& error) {
      result.status  = GpuExecutionStatus::InputRejected;
      result.message = error.what();
    }
    catch (const std::logic_error& error) {
      result.status  = ClassifyRuntimeOrFieldFailure(error.what());
      result.message = error.what();
    }
    catch (const std::runtime_error& error) {
      result.status  = GpuExecutionStatus::RuntimeUnavailable;
      result.message = error.what();
    }
    catch (const std::exception& error) {
      result.status  = GpuExecutionStatus::ExecutionFailed;
      result.message = error.what();
    }
    result.transactionMilliseconds = elapsedMilliseconds(transactionStarted);
    return result;
  }
}  // namespace kfparticle::gpu
