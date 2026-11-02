/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPUEXECUTOR_H
#define KFPARTICLEGPUEXECUTOR_H

#include "planning/KFParticleGpuBatch.h"
#include "KFParticle.h"
#include "data/KFParticleGpuMaterializer.h"
#include "diagnostics/KFParticleGpuParity.h"
#include "algorithms/KFParticleGpuSelection.h"

#include <array>
#include "diagnostics/KFParticleGpuHostTiming.h"

#include <cstdint>
#include <string>
#include <vector>

class KFPTrackVector;
class KFVertex;
class KFParticleGpuFinder;
struct KFParticleGpuRuntimeSettings;

namespace kfparticle::gpu
{
  enum class GpuExecutionStatus {
    Completed,
    EmptyInput,
    NoPairs,
    InputRejected,
    FieldRejected,
    RuntimeUnavailable,
    ExecutionFailed,
    Overflow
  };

  enum class GpuBatchFlushReason { None, Capacity, EndOfTimeslice, OversizedEvent };

  enum class GpuOutputMode {
    FullDiagnostics,
    PublishableOutput
  };

  struct GpuChannelSummary {
    unsigned int channelId = 0;
    int motherPdg          = 0;
    unsigned int totalPairs = 0;
    unsigned int storedTasks = 0;
    unsigned int candidates = 0;
    unsigned int overflowFlags = 0;
    bool truncated = false;
  };

  /** Host snapshot of one GPU candidate, including its generalized parity record. */
  struct GpuRawCandidate {
    unsigned int channelId = 0;
    int pdg = 0;
    std::array<int, 2> daughterSourceIds{{-1, -1}};
    float mass = 0.f;
    float massError = 0.f;
    float chi2 = 0.f;
    int ndf = 0;
    int primaryVertexIndex = -1;
    int bestPrimaryVertexIndex = -1;
    unsigned int candidateFlags = 0;
    unsigned int selectionClass = 0;
    unsigned int rejectionReasons = 0;
    unsigned int topologyStatus = KFGpuV0LineTopologyNone;
    unsigned int topology = KFGpuGraphTopologyInvalid;
    unsigned int outputClass = KFGpuGraphOutputInvalid;
    unsigned int operationStatus = KFGpuCandidateOperationPending;
    KFParticleGpuV0SelectionObservables selectionObservables;
    bool selected = false;
    bool massValid = false;
    KFParticleGpuParitySnapshot parity;
  };

  /** Result of one GPU reconstruction attempt, with optional full diagnostics. */
  struct GpuEventResult {
    std::uint64_t sourceEventId = 0;
    GpuExecutionStatus status  = GpuExecutionStatus::InputRejected;
    unsigned int tracks         = 0;
    unsigned int vertices       = 0;
    unsigned int taskCapacity   = 0;
    unsigned int candidates     = 0;
    unsigned int selectedCandidates = 0;
    unsigned int overflowFlags  = 0;
    unsigned int batchLimit = 1;
    unsigned int batchSize = 1;
    unsigned int batchIndex = 0;
    bool sourceOrderViolation = false;
    GpuBatchFlushReason batchFlushReason = GpuBatchFlushReason::None;
    // Host wall-time phases; GPU pipeline includes XPU transfers, launches, and required waits.
    double inputPreparationMilliseconds = 0.;
    double capacityPlanningMilliseconds = 0.;
    double capacityPlanSetupMilliseconds = 0.;
    unsigned int capacityPlanBuilds = 0u;
    double queueAndCapacityMilliseconds = 0.;
    double hostPackingMilliseconds = 0.;
    double inputUploadMilliseconds = 0.;
    double pairTilePlanningMilliseconds = 0.;
    double constructionMilliseconds = 0.;
    double graphConstructionMilliseconds = 0.;
    double selectionMilliseconds = 0.;
    double outputDownloadMilliseconds = 0.;
    double gpuPipelineMilliseconds = 0.;
    double resultExtractionMilliseconds = 0.;
    double materializationMilliseconds = 0.;
    double materializationCpuWorkMilliseconds = 0.;
    unsigned int packingThreadsRequested = 1u;
    unsigned int packingThreadsUsed = 1u;
    unsigned int materializationThreadsRequested = 1u;
    unsigned int materializationThreadsUsed = 1u;
    double cpuReferenceMilliseconds = 0.;
    double comparisonMilliseconds = 0.;
    double transactionMilliseconds = 0.;
    double queueWaitMilliseconds = 0.;
    KFParticleGpuHostTiming hostTiming;
    std::uint64_t resultSignatureXor = 0;
    std::uint64_t resultSignatureSum = 0;
    std::uint64_t resultSignatureCandidates = 0;
    bool performanceSample = false;
    bool fitUnrollEnabled = false;
    bool gpuCompactOutput = false;
    unsigned int gpuKernelLaunches = 0u;
    unsigned int gpuQueueWaits = 0u;
    std::uint64_t gpuAllocationRounds = 0u;
    std::uint64_t gpuHostToDeviceBytes = 0u;
    std::uint64_t gpuDeviceToHostBytes = 0u;
    std::uint64_t gpuTransactionStatusBytes = 0u;
    std::uint64_t gpuRetainedSnapshotBytes = 0u;
    double gpuHostAssemblyMilliseconds = 0.;
    std::uint64_t gpuAllocatedBytesHighWater = 0u;
    std::uint64_t gpuTwoDaughterRealPairs = 0u;
    std::uint64_t gpuTwoDaughterPairTiles = 0u;
    std::uint64_t gpuTwoDaughterScheduledLanes = 0u;
    std::uint64_t gpuTwoDaughterTailLanes = 0u;
    std::uint64_t gpuTwoDaughterRectangularLanesAvoided = 0u;
    std::uint64_t gpuTwoDaughterPairTileBytes = 0u;
    unsigned int gpuVisitedCombinations = 0u;
    unsigned int gpuStoredTasks = 0u;
    unsigned int gpuBlockReservations = 0u;
    double gpuMaskDensity = 0.;
    double gpuUsefulWorkPerLaunch = 0.;
    double gpuCandidatePoolOccupancy = 0.;
    std::vector<GpuChannelSummary> channels;
    std::vector<GpuRawCandidate> rawCandidates;
    std::vector<KFParticle> materializedParticles;
    KFParticleGpuMaterializationStatus materializationStatus =
      KFGpuMaterializationInvalidCandidate;
    bool materializationSucceeded = false;
    bool fullDiagnosticSnapshot = false;
    std::string message;

    bool Succeeded() const
    {
      return status == GpuExecutionStatus::Completed || status == GpuExecutionStatus::NoPairs
             || status == GpuExecutionStatus::EmptyInput;
    }
  };

  /**
   * Executes a CPU-catalogue decay plan over prepared event input.
   *
   * FullDiagnostics supplies snapshots for the current three-V0 CPU oracle.
   * PublishableOutput
   * retains only data required for atomic GPU publication and fallback.
   * GPU failures are always captured as status rather than escaping the event.
   */
  class GpuExecutor {
   public:
    explicit GpuExecutor(unsigned int materializationThreads = 1u,
                                 unsigned int packingThreads = 1u);

    /** Batch-of-one facade; also materializes FullDiagnostics, without benchmark replay. */
    GpuEventResult Run(std::uint64_t sourceEventId,
                                 const KFPTrackVector& first,
                                 const KFPTrackVector& last,
                                 const std::vector<float>& chiToPrimaryVertex,
                                 const std::vector<KFVertex>& primaryVertices,
                                 float minSecondaryTrackChiToPrimaryVertex = -1.f,
                                 GpuOutputMode outputMode =
                                   GpuOutputMode::FullDiagnostics) const;

    /** Executes one sealed batch and returns results in source-event order. */
    std::vector<GpuEventResult> RunBatch(
      const GpuPreparedBatch& batch,
      GpuOutputMode outputMode = GpuOutputMode::FullDiagnostics) const;

   private:
    friend class ::KFParticleGpuFinder;

    struct ExecutionSettings {
      bool inheritEnvironment = true;
      const KFParticleGpuRuntimeSettings* runtime = nullptr;
    };

    std::vector<GpuEventResult> RunBatchConfigured(
      const GpuPreparedBatch& batch, GpuOutputMode outputMode,
      bool materializeOutput, const ExecutionSettings& settings) const;

    GpuEventResult RunSingleConfigured(
      std::uint64_t sourceEventId, const GpuEventInput& input,
      GpuOutputMode outputMode, const std::vector<int>& motherPdgs,
      bool materializeOutput, const ExecutionSettings& settings) const;

    std::vector<GpuEventResult> ExecuteBatchOnce(
      const GpuPreparedBatch& batch,
      GpuOutputMode outputMode,
      bool materializeOutput,
      const ExecutionSettings& settings) const;

    unsigned int fMaterializationThreads = 1u;
    unsigned int fPackingThreads = 1u;
  };
}  // namespace kfparticle::gpu

#endif  // KFPARTICLEGPUEXECUTOR_H
