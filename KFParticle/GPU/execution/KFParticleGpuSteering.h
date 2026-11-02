/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUSTEERING_H
#define KFPARTICLEGPUSTEERING_H

#include <memory>
#include <cstdint>

class KFParticleGpuDecayPlan;
class KFParticleGpuRuntime;

#ifdef KFPARTICLE_USE_XPU
#include "planning/KFParticleGpuDecayPlan.h"
#include "planning/KFParticleGpuChannelRouting.h"
#include "planning/KFParticleGpuTwoDaughterRouting.h"

#include <vector>

class KFParticleGpuBufferManager;
namespace xpu
{
  class queue;
}

/**
 * @brief Host wall-time measurements and submeasurements of one decay-plan invocation.
 *
 * These are steady-clock intervals around host planning, submissions, copies, and waits,
 * not isolated GPU kernel times. Asynchronous work may be charged to a later wait.
 * Construction excludes its separately recorded selection interval; planning/validation
 * fields are nested in construction, and host assembly is nested in the dependent-stage
 * interval. Do not sum all fields as independent stages.
 */
struct KFParticleGpuDecayPlanTiming {
  double inputUploadMilliseconds = 0.; ///< UploadInput() host interval, including its internal transfer/wait costs.
  double pairTilePlanningMilliseconds = 0.; ///< Pair-tile filling time, nested in construction.
  double constructionMilliseconds = 0.; ///< First-generation setup/submission interval excluding selectionMilliseconds.
  double selectionMilliseconds = 0.; ///< Host interval for selection submissions and any waits inside that interval.
  /// Dependent-stage interval, including completion wait and host result assembly.
  double cascadeConstructionMilliseconds = 0.;
  double outputDownloadMilliseconds = 0.; ///< Final output-prefix sizing and readback interval.
  std::uint64_t countingReadbackBytes = 0u; ///< Small retained-task status payload used for dense output planning.
  double countingReadbackMilliseconds = 0.; ///< Counting-status readback interval, nested in construction.
  /// Dense reservation/task-map planning and capacity allocation, nested in construction.
  double denseOutputPlanningMilliseconds = 0.;
  /// Initial transaction-validation interval, nested in construction.
  double initialTransactionValidationMilliseconds = 0.;
  /// Validation after dense reservation planning, nested in construction.
  double compactOutputValidationMilliseconds = 0.;
  /// Transaction result assembly after the completion wait; nested in dependent-stage time.
  double hostAssemblyMilliseconds = 0.;
};

/**
 * @brief Composite-track counters aggregated by the last full plan execution.
 * Group counts describe logical event/group work, not necessarily separate kernel launches.
 * Visited counts may be summed per channel in detailed mode, so hypotheses can count the
 * same source pair more than once. Optional diagnostic counters depend on the active mode.
 */
struct KFParticleGpuV0TrackRoutingMonitorData {
  /// Logical group/generation sequences; interpretation depends on the enclosing record.
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u; ///< Number of channel descriptors for this route, not multiplied by event count.
  unsigned int visitedPairs = 0u; ///< Reported visited source-pair count for this route.
  unsigned int activeChannelBits = 0u; ///< Accumulated compatible channel-bit count across visited combinations.
  unsigned int acceptedTasks = 0u; ///< Tasks accepted by routing before storage-capacity truncation.
  unsigned int storedTasks = 0u; ///< Tasks retained for construction.
  unsigned int blockReservations = 0u; ///< Reported block-level task-reservation count.
  unsigned int candidates = 0u; ///< Appended raw candidate count.
  unsigned int daughters = 0u; ///< Appended flattened source-ancestry entry count.
  /// Combined routing/output overflow flags; callers must inspect these even after normal return.
  unsigned int overflowFlags = 0u;
};

/**
 * @brief First-generation routing, construction, and selection totals for the last full plan.
 * Counts aggregate events. groupLaunches counts logical routing groups, while selectionLaunches
 * counts selection sequences (one for a transaction batch, otherwise one per processed event),
 * not individual GPU kernels. Diagnostic counters can be omitted in compact unmonitored runs.
 */
struct KFParticleGpuTwoDaughterRoutingMonitorData {
  /// Logical group/generation sequences; interpretation depends on the enclosing record.
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u; ///< Number of channel descriptors for this route, not multiplied by event count.
  unsigned int selectionLaunches = 0u; ///< Number of first-generation selection sequences, not individual kernels.
  unsigned int visitedPairs = 0u; ///< Reported visited source-pair count for this route.
  unsigned int activeChannelBits = 0u; ///< Accumulated compatible channel-bit count across visited combinations.
  unsigned int acceptedTasks = 0u; ///< Tasks accepted by routing before storage-capacity truncation.
  unsigned int storedTasks = 0u; ///< Tasks retained for construction.
  unsigned int blockReservations = 0u; ///< Reported block-level task-reservation count.
  unsigned int candidates = 0u; ///< Appended raw candidate count.
  unsigned int daughters = 0u; ///< Appended flattened source-ancestry entry count.
  unsigned int selectedCandidates = 0u; ///< First-generation selected count; not all final graph publications.
  /// Combined routing/output overflow flags; callers must inspect these even after normal return.
  unsigned int overflowFlags = 0u;
};

/**
 * @brief Per-event, per-channel diagnostic counters for dependent graph operations.
 * Available when detailed snapshots are retained; zero counters are not proof that work was absent.
 */
struct KFParticleGpuGraphChannelMonitorData {
  unsigned int channelId = 0u; ///< Configured decay-channel identity.
  unsigned int eventIndex = 0u; ///< Absolute index in the buffer manager's event array.
  unsigned int generation = 0u; ///< Dependency generation of the graph channel.
  /// Reported visited combinations, not necessarily unique physical track tuples.
  unsigned int visitedCombinations = 0u;
  unsigned int acceptedTasks = 0u; ///< Tasks accepted by routing before storage-capacity truncation.
  unsigned int storedTasks = 0u; ///< Tasks retained for construction.
  unsigned int constructedCandidates = 0u; ///< Graph executor's reported successful construction count.
  /// Reported graph-task rejection count; not uniformly populated by all execution routes.
  unsigned int rejectedTasks = 0u;
};

/**
 * @brief Aggregate dependent-graph counters and unsupported-coverage counts.
 * groupLaunches counts active generations for a transaction batch, but event/group sequences
 * for the fallback route. Construction counters and appended pool sizes are tracked separately.
 */
struct KFParticleGpuGraphExecutionMonitorData {
  /// Logical group/generation sequences; interpretation depends on the enclosing record.
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u; ///< Number of channel descriptors for this route, not multiplied by event count.
  /// Reported visited combinations, not necessarily unique physical track tuples.
  unsigned int visitedCombinations = 0u;
  unsigned int acceptedTasks = 0u; ///< Tasks accepted by routing before storage-capacity truncation.
  unsigned int storedTasks = 0u; ///< Tasks retained for construction.
  unsigned int constructedCandidates = 0u; ///< Graph executor's reported successful construction count.
  /// Reported graph-task rejection count; not uniformly populated by all execution routes.
  unsigned int rejectedTasks = 0u;
  unsigned int candidates = 0u; ///< Appended raw candidate count.
  unsigned int daughters = 0u; ///< Appended flattened source-ancestry entry count.
  unsigned int unsupportedNodes = 0u; ///< Compiled graph nodes whose support status is not Supported.
  unsigned int unsupportedFamilies = 0u; ///< Compiled graph families whose support status is not Supported.
  /// Combined routing/output overflow flags; callers must inspect these even after normal return.
  unsigned int overflowFlags = 0u;
};

/**
 * @brief Explanatory performance snapshot assembled by the last full plan invocation.
 *
 * Enable monitoring before execution for complete derived metrics. Kernel/wait counts and
 * payload bytes are accounting estimates, not a trace of every backend operation. In particular,
 * extra debug probes/fences and some administrative transfers are not fully represented.
 * Values describe the executed route; isolated stage helpers do not refresh this snapshot.
 */
struct KFParticleGpuPerformanceSnapshot {
  bool enabled = false; ///< Whether performance monitoring was enabled for this invocation.
  bool fitUnrollEnabled = false; ///< Build-time unroll option; does not prove a particular kernel was unrolled.
  bool compactOutput = false; ///< True for publishable transfer mode, not a statement about reservation layout.
  unsigned int events = 0u; ///< Requested event count for this invocation.
  unsigned int tracks = 0u; ///< Total loaded input-track size, which can exceed the requested event subset.
  unsigned int vertices = 0u; ///< Total loaded vertex size, which can exceed the requested event subset.
  unsigned int descriptorGroups = 0u; ///< Sum of the three route monitors' group counts; not a hardware launch count.
  /// Reported visited combinations, not necessarily unique physical track tuples.
  unsigned int visitedCombinations = 0u;
  unsigned int activeChannelBits = 0u; ///< Accumulated compatible channel-bit count across visited combinations.
  unsigned int acceptedTasks = 0u; ///< Tasks accepted by routing before storage-capacity truncation.
  unsigned int storedTasks = 0u; ///< Tasks retained for construction.
  unsigned int blockReservations = 0u; ///< Reported block-level task-reservation count.
  /// Reported graph-task rejection count; not uniformly populated by all execution routes.
  unsigned int rejectedTasks = 0u;
  unsigned int rawCandidates = 0u; ///< Sum of first-generation, cascade, and graph raw candidate counts.
  unsigned int selectedCandidates = 0u; ///< First-generation selected range/count; not all final graph publications.
  unsigned int daughters = 0u; ///< Appended flattened source-ancestry entry count.
  /// Combined routing/output overflow flags; callers must inspect these even after normal return.
  unsigned int overflowFlags = 0u;
  /// Estimated kernel count from route formulas plus measured selection-reset launch deltas.
  unsigned int kernelLaunches = 0u;
  /// Estimated fixed and plan-upload waits; additional diagnostic waits are not enumerated.
  unsigned int queueWaits = 0u;
  /// Accounted transaction-status readback operations, including required counting readback.
  unsigned int monitoringReadbacks = 0u;
  unsigned int monitoringReadbackBuffers = 0u; ///< Accounted number of buffers in those status readbacks.
  std::uint64_t twoDaughterRealPairs = 0u; ///< Sum of event-local source-range products before pair-level rejection.
  std::uint64_t twoDaughterPairTiles = 0u; ///< Number of host-planned first-generation routing tiles.
  /// Planned tile count times tile size; may differ from the actual fused launch geometry.
  std::uint64_t twoDaughterScheduledLanes = 0u;
  std::uint64_t twoDaughterTailLanes = 0u; ///< Unused tail lanes in the planned routing tiles.
  /// Estimated lanes avoided relative to work-count times maximum pair-count geometry.
  std::uint64_t twoDaughterRectangularLanesAvoided = 0u;
  std::uint64_t twoDaughterPairTileBytes = 0u; ///< Host-planned routing tile-table payload size.
  /// Capacity-allocation rounds, including first allocations; not necessarily resizing existing buffers.
  std::uint64_t capacityGrowths = 0u;
  std::uint64_t hostToDeviceBytes = 0u; ///< Loaded input payload estimate; excludes some plan/administrative transfers.
  std::uint64_t deviceToHostBytes = 0u; ///< Output-prefix payload estimate plus accounted transaction-status bytes.
  /// Accounted status readback payload across counting and completion boundaries.
  std::uint64_t transactionStatusBytes = 0u;
  /// Estimated live record/vector-element bytes; excludes spare vector capacity and allocator overhead.
  std::uint64_t retainedSnapshotBytes = 0u;
  /// Buffer manager's AllocatedBytes() at completion, not an independently sampled peak.
  std::uint64_t allocatedBytesHighWater = 0u;
  /// Active channel bits divided by pair-count times descriptor-count slots; zero for no slots.
  double maskDensity = 0.;
  /// Stored tasks divided by the estimated kernel count; zero when no launches are counted.
  double usefulWorkPerLaunch = 0.;
  /// Raw candidate count divided by allocated candidate capacity; zero when capacity is zero.
  double candidatePoolOccupancy = 0.;
  KFParticleGpuDecayPlanTiming timing; ///< Copy of the full-plan host timing record.
};

/**
 * @brief Event-local ranges and counters from a multi-event decay-plan execution.
 *
 * Candidate/daughter ranges address raw pools; selected ranges address the compact index list.
 * Channel offsets address detailed result vectors only when those vectors are retained.
 * Compact publishable transactions may expose event results while leaving channel vectors empty.
 * These are borrowed-storage coordinates, not independently owned particle collections.
 */
struct KFParticleGpuDecayPlanEventResult {
  unsigned int eventIndex = 0; ///< Absolute index in the buffer manager's event array.
  unsigned int channelOffset = 0; ///< Start in detailed two-track channel results when retained.
  unsigned int channelCount = 0; ///< Configured two-track channels for the event, even if detailed results are omitted.
  unsigned int cascadeChannelOffset = 0; ///< Start in detailed cascade channel results when retained.
  /// Configured cascade channels for the event, even if detailed results are omitted.
  unsigned int cascadeChannelCount = 0;
  KFParticleGpuCandidateRange candidates; ///< Raw candidate and flattened ancestry interval.
  KFParticleGpuTwoDaughterRoutingStatus generationRouting; ///< First-generation routing totals for this event.
  KFParticleGpuCandidateRange cascadeCandidates; ///< Composite-track candidate/ancestry range for this event.
  KFParticleGpuV0TrackRoutingStatus cascadeRouting; ///< Composite-track routing totals for this event.
  KFParticleGpuCandidateRange graphCandidates; ///< Later graph-operation candidate/ancestry range for this event.
  /// First-generation selected range/count; not all final graph publications.
  KFParticleGpuSelectedCandidateRange selectedCandidates;
  /// Combined routing/output overflow flags; callers must inspect these even after normal return.
  unsigned int overflowFlags = 0;
};

/**
 * @brief Routing counters and appended raw-pool range from the isolated composite-track helper.
 */
struct KFParticleGpuV0TrackFusedResult {
  unsigned int eventIndex = 0u; ///< Absolute index in the buffer manager's event array.
  KFParticleGpuV0TrackRoutingStatus routing; ///< Routing counters for this isolated stage.
  KFParticleGpuCandidateRange candidates; ///< Appended raw candidate range/count for this record.
};

/**
 * @brief Routing counters and appended raw-pool range from the isolated two-track helper.
 */
struct KFParticleGpuTwoDaughterFusedResult {
  unsigned int eventIndex = 0u; ///< Absolute index in the buffer manager's event array.
  /// Logical group/generation sequences; interpretation depends on the enclosing record.
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u; ///< Number of channel descriptors for this route, not multiplied by event count.
  KFParticleGpuTwoDaughterRoutingStatus routing; ///< Routing counters for this isolated stage.
  KFParticleGpuCandidateRange candidates; ///< Appended raw candidate range/count for this record.
};

/**
 * @brief Preferred first-generation transaction implementation; fusion is subject to runtime eligibility.
 */
enum KFParticleGpuTwoDaughterTransactionStrategy {
  KFGpuTwoDaughterTransactionStaged = 0, ///< Separate routing and construction kernels.
  /// Request combined routing/construction; currently falls back above 1024 tasks or with compact reservations.
  KFGpuTwoDaughterTransactionFused = 1
};

/**
 * @brief Where first-generation selection is evaluated in a transaction batch.
 */
enum KFParticleGpuTwoDaughterSelectionStrategy {
  KFGpuTwoDaughterSelectionStaged = 0, ///< Evaluate selection in a separate kernel after construction.
  /// Evaluate during construction; segment preparation and scatter still run separately.
  KFGpuTwoDaughterSelectionDirect = 1
};

/**
 * @brief Construction placement for composite-track transaction generations.
 */
enum KFParticleGpuCascadeTransactionStrategy {
  KFGpuCascadeTransactionStaged = 0, ///< Route tasks first, then run a separate construction kernel.
  KFGpuCascadeTransactionFused = 1 ///< Construct during routing; reset and source-resolution kernels remain separate.
};

/**
 * @brief Execution placement for dependent graph-operation transaction generations.
 */
enum KFParticleGpuGraphTransactionStrategy {
  KFGpuGraphTransactionStaged = 0, ///< Route tasks first, then execute them in a separate kernel.
  /// Execute operations during routing; reset and source-resolution kernels remain separate.
  KFGpuGraphTransactionFused = 1
};

/**
 * @brief Initial administrative reset policy; only the transaction route can use device-batch initialization.
 */
enum KFParticleGpuAdministrativeResetStrategy {
  KFGpuAdministrativeResetHostStaged = 0, ///< Use the buffer manager's staged reset operations.
  /// Skip initial staged resets and rely on transaction/device initialization; later compact resets may still run.
  KFGpuAdministrativeResetDeviceBatch = 1
};

/**
 * @brief Result transfer/detail policy, distinct from compact transaction memory planning.
 */
enum KFParticleGpuOutputTransferMode {
  /// Read publication payload prefixes without per-candidate selection diagnostics; suppress detailed transaction
  /// snapshots.
  KFGpuPublishableOutput = 0,
  /// Also read selection diagnostics and retain per-channel snapshots; not a dump of every internal GPU buffer.
  KFGpuFullDiagnosticOutput = 1
};
#endif

/**
 * @brief Host coordinator of GPU reconstruction, from packed event inputs to downloaded numerical output.
 *
 * Created and initialized by KFParticleGpuRuntime. Owns a decay plan, reusable buffer manager,
 * compiled routing/graph plans, and result snapshots; borrows the runtime's queue. Full-plan
 * execution uploads input, plans event-local work, runs construction/selection/dependent
 * generations, then reads back bounded output prefixes. It does not materialize CPU KFParticle
 * objects, choose CPU fallback, or retry overflowed batches.
 *
 * There is no internal mutex: callers must serialize plan/buffer changes and execution.
 * Queue ordering coordinates device stages, but does not make concurrent host calls safe.
 * Returned references belong to this object and may change on the next full-plan run;
 * individual pool entries/views may become invalid after capacity changes or finalization.
 * Last* accessors are not completion certificates after a failed or isolated-stage run.
 *
 * Except IsInitialized() and PerformanceMonitoringEnabled(), public operational accessors
 * require initialization and throw std::logic_error otherwise. Backend/allocation errors
 * propagate; a failed run is not rolled back. A successful return can still report overflow.
 * Without KFPARTICLE_USE_XPU, only lifecycle and decay-plan access are available.
 */
class KFParticleGpuSteering
{
 public:
  /**
   * @brief Finalize owned buffers/plans before destroying private state; the borrowed queue is not destroyed.
   */
  ~KFParticleGpuSteering();

  /**
   * @brief Return the local ready flag without locking.
   */
  bool IsInitialized() const;

  /**
   * @brief Borrow the configured decay plan; edits must be serialized with execution.
   * Compiled tables are refreshed from the plan revision during subsequent full/fused-stage runs.
   */
  KFParticleGpuDecayPlan& GetDecayPlan();

  /**
   * @brief Borrow read-only decay-plan access without extending the plan's lifetime.
   */
  const KFParticleGpuDecayPlan& GetDecayPlan() const;

#ifdef KFPARTICLE_USE_XPU
  /**
   * @brief Borrow the buffer manager used to populate inputs and inspect downloaded output.
   * Changing capacities can invalidate earlier views; serialize all access with execution.
   */
  KFParticleGpuBufferManager& GetBuffers();

  /**
   * @brief Borrow read-only buffer-manager access; no synchronization or lifetime extension is provided.
   */
  const KFParticleGpuBufferManager& GetBuffers() const;

  /**
   * @brief Upload inputs, reset candidates, convert loaded tracks to fit states, and download candidates.
   * A smoke-test path, not decay reconstruction. Uses the supplied mass and event index; previous
   * candidate output is replaced. Rejects negative mass and, when tracks exist, an out-of-range event.
   * The mass check is not a complete finite-value validation.
   */
  void RunRoundTrip(float mass, unsigned int eventIndex = 0);

  /**
   * @brief Run the fixed-slot two-track test path, resetting candidates and reading full candidates back.
   * Uploads input, generates tasks, waits, constructs fits, waits, then downloads results.
   * Requires positive taskCapacity no larger than candidate capacity and a valid source event.
   * Does not run V0 selection or dependent graph stages.
   */
  void RunTwoDaughterStage(const KFParticleGpuTwoDaughterTaskSource& source,
                          unsigned int taskCapacity);

  /**
   * @brief Run one compact two-track source from an empty candidate pool, then download candidates.
   * Uploads input and retains at most taskCapacity accepted tasks; overflow is recorded, not retried.
   * Requires a valid event and positive task capacity not exceeding candidate capacity.
   * Does not update the full-plan Last* result tables.
   */
  void RunTwoDaughterCompactStage(const KFParticleGpuTwoDaughterTaskSource& source,
                                 unsigned int taskCapacity);

  /**
   * @brief Replace selected output with selection of one raw candidate range for the supplied channel/event.
   * Requires current device input/candidate data, a candidate range within capacity, and nonzero
   * selected capacity. Resets selection storage, launches evaluation, then downloads indices and decisions.
   * Does not upload input or reconstruct candidates. Returns the zero-offset selected range and overflow.
   */
  KFParticleGpuSelectedCandidateRange RunV0Selection(
    const KFParticleGpuTwoDaughterChannel& channel,
    const KFParticleGpuCandidateRange& rawCandidates,
    unsigned int eventIndex);

  /**
   * @brief Execute RunDecayPlanBatch() for one event, with the same output and lifetime contract.
   */
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& RunDecayPlan(
    unsigned int eventIndex,
    unsigned int taskCapacity,
    KFParticleGpuOutputTransferMode outputMode = KFGpuFullDiagnosticOutput);

  /**
   * @brief Execute the configured decay plan over a contiguous subset of loaded event descriptors.
   *
   * Uploads the loaded input arrays, rebuilds revision-dependent plans when needed, and replaces
   * prior full-plan result tables/output. A matching prepared transaction enables event-local batch
   * work; otherwise uses the per-event fallback sequence. Compact reservations use retained routing
   * counts to size output with dependent-generation headroom, without rerunning pair routing.
   *
   * @param firstEventIndex First absolute event-array index.
   * @param eventCount Positive number of contiguous events, bounded by the loaded event count.
   * @param taskCapacity Positive base task budget; route-specific capacities may multiply it by channel count.
   * @param outputMode Readback/detail policy; independent of compact transaction memory planning.
   * @return Borrowed detailed two-track channel results. Can be empty for a successful publishable
   * transaction; use LastDecayPlanEventResults() and downloaded buffers in that mode.
   * @note Fused is a preference, not a guarantee. Current two-track fusion requires taskCapacity <= 1024
   * and noncompact output reservations. Overflow is reported in results and is not automatically retried.
   * @throws std::invalid_argument If eventCount or taskCapacity is zero, or plan validation fails.
   * @throws std::out_of_range If the requested event interval exceeds loaded input.
   * @throws std::logic_error If initialization or checked graph/transaction contracts are invalid.
   * @throws std::overflow_error If a checked capacity or launch-size calculation overflows.
   */
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& RunDecayPlanBatch(
    unsigned int firstEventIndex,
    unsigned int eventCount,
    unsigned int taskCapacity,
    KFParticleGpuOutputTransferMode outputMode = KFGpuFullDiagnosticOutput);

  /**
   * @brief Borrow the last full-plan two-track channel results, omitted in publishable transaction mode.
   * For multiple channels, generationCandidates covers shared storage; per-channel candidates need
   * not form contiguous ranges. Use channel identity and constructed counts rather than assuming slices.
   */
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& LastDecayPlanResults() const;

  /**
   * @brief Borrow detailed composite-track channel results, omitted in publishable transaction mode.
   * Generation ranges can be shared by several channels; per-channel counts are not necessarily slices.
   */
  const std::vector<KFParticleGpuV0TrackChannelResult>& LastV0TrackCascadeResults() const;

  /**
   * @brief Borrow event-local first-generation, cascade, graph, and selected ranges from the last full plan.
   */
  const std::vector<KFParticleGpuDecayPlanEventResult>& LastDecayPlanEventResults() const;

  /**
   * @brief Borrow host timing intervals for the last full plan; submeasurements overlap parent intervals.
   */
  const KFParticleGpuDecayPlanTiming& LastDecayPlanTiming() const;

  /**
   * @brief Borrow aggregated first-generation counters from the last full plan, not isolated stage helpers.
   */
  const KFParticleGpuTwoDaughterRoutingMonitorData&
  LastTwoDaughterRoutingMonitorData() const;

  /**
   * @brief Borrow aggregated composite-track counters from the last full plan.
   */
  const KFParticleGpuV0TrackRoutingMonitorData& LastV0TrackRoutingMonitorData() const;

  /**
   * @brief Borrow aggregated dependent-graph execution and support-coverage counters.
   */
  const KFParticleGpuGraphExecutionMonitorData& LastGraphExecutionMonitorData() const;

  /**
   * @brief Borrow detailed event/channel graph counters; absent in publishable transaction mode.
   */
  const std::vector<KFParticleGpuGraphChannelMonitorData>&
  LastGraphChannelMonitorData() const;

  /**
   * @brief Enable derived performance accounting for future full-plan runs.
   * Disabling immediately resets the last performance snapshot; it does not disable separate debug env options.
   */
  void SetPerformanceMonitoringEnabled(bool enabled);

  /**
   * @brief Return true only when initialized and monitoring is enabled; does not throw when uninitialized.
   */
  bool PerformanceMonitoringEnabled() const;

  /**
   * @brief Set the preferred first-generation transaction strategy; reject unknown enum values.
   * The fused preference can fall back to staged execution; does not alter isolated-stage helpers.
   */
  void SetTwoDaughterTransactionStrategy(
    KFParticleGpuTwoDaughterTransactionStrategy strategy);

  /**
   * @brief Return the configured preference, not the route actually used in the last transaction.
   */
  KFParticleGpuTwoDaughterTransactionStrategy TwoDaughterTransactionStrategy() const;

  /**
   * @brief Choose staged or in-construction selection for transaction batches; reject unknown values.
   */
  void SetTwoDaughterSelectionStrategy(
    KFParticleGpuTwoDaughterSelectionStrategy strategy);

  /**
   * @brief Return the configured first-generation selection strategy.
   */
  KFParticleGpuTwoDaughterSelectionStrategy TwoDaughterSelectionStrategy() const;

  /**
   * @brief Choose separate or routing-integrated cascade construction for transaction batches.
   * Unknown enum values throw std::invalid_argument; per-event fallback is unaffected.
   */
  void SetCascadeTransactionStrategy(KFParticleGpuCascadeTransactionStrategy strategy);

  /**
   * @brief Return the configured cascade transaction strategy.
   */
  KFParticleGpuCascadeTransactionStrategy CascadeTransactionStrategy() const;

  /**
   * @brief Choose separate or routing-integrated graph execution for transaction batches.
   * Unknown enum values throw std::invalid_argument; per-event fallback is unaffected.
   */
  void SetGraphTransactionStrategy(KFParticleGpuGraphTransactionStrategy strategy);

  /**
   * @brief Return the configured dependent-graph transaction strategy.
   */
  KFParticleGpuGraphTransactionStrategy GraphTransactionStrategy() const;

  /**
   * @brief Choose staged host resets or device-batch initialization where supported.
   * Unknown enum values throw std::invalid_argument. Compact output allocation may require later resets.
   */
  void SetAdministrativeResetStrategy(
    KFParticleGpuAdministrativeResetStrategy strategy);

  /**
   * @brief Return the configured administrative reset policy.
   */
  KFParticleGpuAdministrativeResetStrategy AdministrativeResetStrategy() const;

  /**
   * @brief Borrow performance accounting; check enabled and the documented limits of estimated counters.
   */
  const KFParticleGpuPerformanceSnapshot& LastPerformanceSnapshot() const;

  /**
   * @brief Borrow the downloaded first-generation selected-index range and its overflow mask.
   * This is not a list of every later-generation publishable particle.
   * For event-reserved batches this is a high-water envelope that may contain gaps.
   * Iterate selectedCandidates of each LastDecayPlanEventResults() entry, or
   * per-channel ranges, not every slot in the envelope.
   */
  const KFParticleGpuSelectedCandidateRange& LastDecayPlanSelectedCandidates() const;

  /**
   * @brief Borrow detailed event/channel selected-index ranges; omitted in publishable transactions.
   */
  const std::vector<KFParticleGpuSelectedChannelRange>& LastDecayPlanSelectedChannels() const;

  /**
   * @brief Append composite-track candidates using resident selected parents and current input data.
   * Requires coherent device inputs/selection and a current host selected-size counter. Does not upload
   * input or reset the candidate pool. Validates event, positive task capacity, routing mode, and selected range.
   * Routes groups, constructs from the device task count in queue order, then downloads tasks/candidates.
   * Returns the appended range; overflow is reported without retry.
   */
  KFParticleGpuV0TrackFusedResult RunV0TrackFusedStage(
    unsigned int eventIndex,
    const KFParticleGpuSelectedCandidateRange& selectedRange,
    unsigned int taskCapacity,
    KFParticleGpuV0TrackRoutingMode routingMode = KFGpuV0TrackRoutingBlockScan);

  /**
   * @brief Append two-track candidates using current device inputs and the configured routing plan.
   * Does not upload input or reset candidates. Validates event, positive task capacity, and routing mode.
   * Routes groups and constructs in queue order without an intermediate task-count readback, then
   * downloads routing status, tasks, candidates, and descriptor indices. Does not perform V0 selection.
   */
  KFParticleGpuTwoDaughterFusedResult RunTwoDaughterFusedStage(
    unsigned int eventIndex,
    unsigned int taskCapacity,
    KFParticleGpuTwoDaughterRoutingMode routingMode =
      KFGpuTwoDaughterRoutingBlockScan);

#endif

 private:
  friend class KFParticleGpuRuntime; ///< Controls construction and lifecycle of the coordinator.

#ifdef KFPARTICLE_USE_XPU
  /**
   * @brief Private construction for KFParticleGpuRuntime; creates bookkeeping, not initialized buffers.
   * In XPU builds the supplied queue is borrowed and must outlive this steering object.
   */
  explicit KFParticleGpuSteering(xpu::queue& queue);

  /**
   * @brief Validate isolated compact-stage inputs, upload them, reset the pool, and execute one channel.
   * Returns counters/ranges without downloading the full candidate payload.
   */
  KFParticleGpuTwoDaughterChannelResult RunTwoDaughterCompactStageImpl(
    const KFParticleGpuTwoDaughterTaskSource& source,
    unsigned int taskCapacity,
    unsigned int channelId);

  /**
   * @brief Append compact tasks/fits for one source and report pool deltas relative to supplied offsets.
   * Requires uploaded input, valid source geometry, positive task capacity, and coherent pool offsets.
   * Does not reset the candidate pool or download its full payload. Pair-count multiplication is unchecked.
   */
  KFParticleGpuTwoDaughterChannelResult RunTwoDaughterCompactChannel(
    const KFParticleGpuTwoDaughterTaskSource& source,
    unsigned int taskCapacity,
    unsigned int channelId,
    unsigned int candidateOffset,
    unsigned int daughterOffset);

#else
  /**
   * @brief Private construction for KFParticleGpuRuntime; creates bookkeeping, not initialized buffers.
   * In XPU builds the supplied queue is borrowed and must outlive this steering object.
   */
  KFParticleGpuSteering();

#endif

  /**
   * @brief Intentionally unavailable copy constructor; execution resources are not copyable.
   */
  KFParticleGpuSteering(const KFParticleGpuSteering&);

  /// @brief Intentionally unavailable copy assignment; resource ownership cannot be copied.
  KFParticleGpuSteering& operator=(const KFParticleGpuSteering&);

  /**
   * @brief Create the buffer manager and an empty decay plan; return immediately if already initialized.
   * Called by the runtime. Without XPU, creates only the decay plan.
   */
  void Initialize();

  /**
   * @brief Wait for queued work, publish empty device views, then release buffers and the decay plan.
   * Does not destroy the borrowed queue. Returns immediately when not initialized; errors propagate.
   */
  void Finalize();

  /// @brief Private buffers, plans, snapshots, and borrowed queue, defined in the implementation file.
  struct Impl;
  std::unique_ptr<Impl> fImpl; ///< Sole owner of the private coordinator state.
};

#endif
