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
 * @brief XPU entry points for KFParticle reconstruction, selection, and device diagnostics.
 *
 * Kernel structs declare launch interfaces; numerical work and device orchestration are implemented
 * in the companion source file. Views and pointers borrow device-accessible storage: kernels do not
 * allocate buffers or perform host/device transfers. Steering owns lifetimes, publishes refreshed
 * constant-memory views, initializes counters, and orders producer/consumer launches.
 *
 * Declared block sizes describe GPU execution geometry, not the number of events in a batch.
 * Capacity guards are not a substitute for valid input views and event indices. Atomic append
 * operations do not guarantee output ordering; subsequent consumers must wait for the producer.
 * Reset/single-thread kernels still require a nonempty launch when their logical input is empty.
 */

#ifndef KFPARTICLEGPUKERNELS_H
#define KFPARTICLEGPUKERNELS_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "execution/KFParticleGpuKernelState.h"
#include "algorithms/KFParticleGpuGraphOperations.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "algorithms/KFParticleGpuTwoDaughter.h"
#include "algorithms/KFParticleGpuV0Track.h"

#ifdef KFPARTICLE_GPU_XPU_ENABLED

class KFParticleGpuKernels;

/**
 * @brief XPU constant-memory binding for the non-owning reconstruction state.
 * Steering republishes this object when buffer views change. It copies views, not the referenced
 * arrays; output buffers remain mutable through their pointers. Referenced storage must outlive launches.
 */
struct TheKFParticleFinder : xpu::constant<KFParticleGpuDeviceImage, KFParticleGpuKernels>
{
};

/**
 * @brief Create one diagnostic particle per input track.
 * Uses the published input/candidate views, the supplied mass, and a common event index.
 * This exercises state layout and candidate storage; it does not reconstruct decays.
 */
struct KFParticleGpuRoundTrip : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Initialize output sizes/overflow flags and fill the capacity-limited track prefix.
   */
  XPU_D void operator()(context& context, float mass, unsigned int eventIndex);
};

/**
 * @brief Clear a contiguous selection-result workspace.
 * One global thread resets one entry; the caller provides storage for at least size results.
 */
struct KFParticleGpuResetSelectionResults : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Assign a default selection result to each index below size.
   */
  XPU_D void operator()(context& context, KFParticleGpuV0SelectionResult* results,
                        unsigned int size);
};

/**
 * @brief Check that a kernel from the KFParticle device image can execute.
 * Only global thread zero writes the fixed marker; no reconstruction state is accessed.
 */
struct KFParticleGpuLaunchSmoke : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write the "KFPG" signature to confirm that the test kernel executed.
   * Only global thread zero writes to a non-null marker.
   * The host reads back and checks this signature in the launch smoke test.
   */
  XPU_D void operator()(context& context, unsigned int* marker);
};

/**
 * @brief Inspect the views published through XPU constant memory.
 * Diagnostic-only, single-thread snapshot of pointer availability and selected state metadata.
 * This is a fixture check, not exhaustive validation of arbitrary device pointers.
 */
struct KFParticleGpuKernelStateProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Write 63 unsigned check values to caller-provided device storage.
   */
  XPU_D void operator()(context& context, unsigned int* checks);
};

/**
 * @brief Inspect input-track, vertex, event, and field layouts on the device.
 * Only global thread zero reads a representative track and the first vertex.
 * The fixture must provide at least one track and one vertex; events may be null.
 */
struct KFParticleGpuInputLayoutProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write 11 floats, 8 integers, and 9 unsigned values to the corresponding check arrays.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuConstVertexSoAView primaryVertices,
                        const KFParticleGpuEventDesc* events,
                        float* floatChecks,
                        int* integerChecks,
                        unsigned int* unsignedChecks);
};

/**
 * @brief Resolve one composite-track task and its physical-track lineage.
 * A single diagnostic thread checks the selected parent and bachelor against one channel.
 * No mother is fitted and no production output is appended.
 */
struct KFParticleGpuV0TrackTaskProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Fill rejection, task, and lineage for selectedV0Index and bachelorTrackIndex.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstCandidatePoolView candidates,
                        KFParticleGpuConstSelectedCandidateIndexView selected,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        KFParticleGpuV0TrackChannel channel,
                        KFParticleGpuSelectedCandidateRange selectedRange,
                        unsigned int selectedV0Index,
                        unsigned int bachelorTrackIndex,
                        unsigned int* rejection,
                        KFParticleGpuV0TrackTask* task,
                        KFParticleGpuV0TrackLineage* lineage);
};

/**
 * @brief Compact accepted selected-parent/bachelor combinations into a task array.
 * One global thread examines one pair in selectedRange times the channel's bachelor range.
 * acceptedTasks counts all accepted tasks, including those beyond taskCapacity;
 * totalPairs counts event/channel-compatible pairs; overflowFlags is optional.
 * Both acceptedTasks and totalPairs must be present and initialized before launch.
 */
struct KFParticleGpuGenerateV0TrackTasksCompact : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Resolve pair eligibility and append tasks within taskCapacity, without fitting mothers.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstCandidatePoolView candidates,
                        KFParticleGpuConstSelectedCandidateIndexView selected,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        KFParticleGpuV0TrackChannel channel,
                        KFParticleGpuSelectedCandidateRange selectedRange,
                        KFParticleGpuV0TrackTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* acceptedTasks,
                        unsigned int* totalPairs,
                        unsigned int* overflowFlags);
};

/**
 * @brief Fit a compact composite-track task list into a raw candidate pool.
 * One global thread handles a task below both taskCapacity and the device-side accepted count.
 * Successful fits reserve candidate and lineage storage; selected indices are not appended.
 */
struct KFParticleGpuV0TrackCompactCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Build and store mothers, recording output-capacity failures in the candidate pool.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        KFParticleGpuConstCandidatePoolView inputCandidates,
                        const KFParticleGpuV0TrackTask* tasks,
                        unsigned int taskCapacity,
                        const unsigned int* acceptedTasks,
                        KFParticleGpuCandidatePoolView outputCandidates);
};

/**
 * @brief Route selected-parent/bachelor pairs with one atomic reservation per accepted channel.
 * One global thread visits one pair and emits its accepted descriptor bits into a bounded worklist.
 * acceptedTasks may exceed capacity; storedTasks counts written entries. The legacy-named
 * blockReservations counter counts task reservations in this variant, not physical blocks.
 */
struct KFParticleGpuRouteV0TrackTasksAtomic : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Filter one event/group, append routed tasks, and update initialized routing counters.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstCandidatePoolView candidates,
                        KFParticleGpuConstSelectedCandidateIndexView selected,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuV0TrackRoutingView routing,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        KFParticleGpuSelectedCandidateRange selectedRange,
                        KFParticleGpuV0TrackRoutedTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* visitedPairs,
                        unsigned int* activeChannelBits,
                        unsigned int* acceptedTasks,
                        unsigned int* storedTasks,
                        unsigned int* blockReservations,
                        unsigned int* overflowFlags);
};

/**
 * @brief Compact composite-track hypotheses using one global reservation per nonempty block.
 * Each lane examines one pair; all lanes participate in the scan and barriers, including padding.
 * The GPU launch must honor ScanBlockSize. Accepted counts can exceed the bounded stored count.
 */
struct KFParticleGpuRouteV0TrackTasksBlockScan : xpu::kernel<KFParticleGpuDeviceImage>
{
  /// @brief Compile-time lane count shared by the GPU launch and prefix-scan implementation.
  static constexpr int ScanBlockSize = 64;
  using block_size = xpu::block_size<ScanBlockSize>;
  /// @brief Exclusive prefix sum of per-lane accepted-hypothesis counts.
  using scan_t = xpu::block_scan<unsigned int, ScanBlockSize>;
  /// @brief Per-block scan scratch and the global output range reserved by lane zero.
  struct shared_data
  {
    scan_t::storage_t scan; ///< XPU scan scratch, shared only within this block.
    unsigned int blockBase; ///< First global task slot reserved for the block.
    unsigned int blockTotal; ///< Accepted hypotheses before capacity clipping.
  };
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = shared_data;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Route event/group pairs through published state, limiting writes to taskLimit and storage capacity.
   */
  XPU_D void operator()(context& context,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        KFParticleGpuSelectedCandidateRange selectedRange,
                        unsigned int taskLimit);
};

/**
 * @brief Fit routed composite-track tasks using published channel descriptors.
 * One global thread consumes one stored task, appends its fitted mother and lineage,
 * and appends a selected index when the candidate passes the construction selection flags.
 */
struct KFParticleGpuV0TrackRoutedCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Consume at most taskLimit stored tasks and report candidate, daughter, or selection overflow.
   */
  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/**
 * @brief Reset reusable composite-track generation counters on the device.
 * Global thread zero clears generation counters; descriptor threads optionally clear channel counters.
 * Task contents and already constructed candidate pools are not cleared.
 */
struct KFParticleGpuResetV0TrackGenerationState
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Clear channel counters as well when resetChannelCounters is nonzero.
   */
  XPU_D void operator()(context& context, unsigned int resetChannelCounters);
};

/**
 * @brief Resolve later-generation inputs from completed transaction-domain outputs.
 * One global thread updates one work status with available candidate/track source ranges and flags.
 * The producer generation must finish before this kernel reads its domain status.
 */
struct KFParticleGpuResolveTransactionWorkSources
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Refresh composite-track and graph sources for generations after completedGeneration.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuTransactionView transaction,
                        unsigned int completedGeneration);
};

/**
 * @brief Prepare domain and work counters for a composite-track batch generation.
 * Only the first block is used; its lanes stride over domains and matching work ranges.
 * Existing candidate data and resolved source ranges are retained.
 */
struct KFParticleGpuResetCascadeBatch : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Reset cascade task counts and monitoring for composite-track work in generation.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation);
};

/**
 * @brief Route composite-track work within independent event domains.
 * One block handles one work range and its lanes traverse selected-parent/bachelor combinations.
 * Tasks either enter reserved domain storage or are constructed immediately; both paths enforce capacity.
 */
struct KFParticleGpuRouteCascadeBatch : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Route generation; constructCandidates enables fused fitting, collectDiagnosticCounters enables monitoring.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation,
                        unsigned int cascadeChannelOffset,
                        unsigned int constructCandidates,
                        unsigned int collectDiagnosticCounters);
};

/**
 * @brief Fit previously routed composite-track tasks in their owning domains.
 * One block handles one domain and strides over its stored tasks for the requested generation.
 * Candidate, daughter, and selected-index writes remain inside domain reservations.
 */
struct KFParticleGpuConstructCascadeBatch : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Construct mothers and update per-channel statuses starting at cascadeChannelOffset.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation,
                        unsigned int cascadeChannelOffset);
};

/**
 * @brief Execute an explicit graph-operation task list without published kernel state.
 * One global thread executes one task using the supplied tracks, candidates, vertices, and descriptors.
 * Each task receives a result; successful outputs reserve candidate and lineage slots.
 */
struct KFParticleGpuExecuteGraphOperationsExplicit
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write taskCount results and append fitted graph outputs, with capacity failures reported per task.
   */
  XPU_D void operator()(
    context& context,
    KFParticleGpuConstInputTrackSoAView inputTracks,
    KFParticleGpuConstCandidatePoolView inputCandidates,
    KFParticleGpuConstVertexSoAView primaryVertices,
    const KFParticleGpuGraphOperationDescriptor* descriptors,
    unsigned int descriptorCount,
    const KFParticleGpuGraphOperationTask* tasks,
    unsigned int taskCount,
    KFParticleGpuCandidatePoolView outputCandidates,
    KFParticleGpuGraphOperationResult* results);
};

/**
 * @brief Clear published graph-operation generation and per-descriptor counters.
 * Global thread zero clears totals; descriptor threads clear channel monitoring.
 * Task/result arrays and candidate contents are left unchanged.
 */
struct KFParticleGpuResetGraphOperationGeneration
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Initialize graph counters before routing a fresh generation.
   */
  XPU_D void operator()(context& context);
};

/**
 * @brief Generate graph tasks for one event and execution group.
 * A global thread identifies one node/first-source slot and loops over its second sources.
 * sourceCapacity is the first-source launch stride, not the number of accepted tasks.
 */
struct KFParticleGpuRouteGraphOperationTasks
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Filter source identity/order and append tasks up to taskLimit and the published task capacity.
   */
  XPU_D void operator()(
    context& context,
    unsigned int eventIndex,
    unsigned int groupIndex,
    unsigned int sourceCapacity,
    unsigned int taskLimit);
};

/**
 * @brief Execute graph tasks from the published bounded worklist.
 * One global thread consumes one stored task and records its result and rejection/constructed counters.
 * Successful fits append raw candidates and physical-track lineage.
 */
struct KFParticleGpuExecuteRoutedGraphOperations
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Execute at most taskLimit stored tasks; report output-capacity failures per task.
   */
  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/**
 * @brief Reset graph-task and monitoring counters for a transaction generation.
 * Only the first block strides over domain/work statuses and graph descriptor counters.
 * Candidate storage and resolved source ranges are retained.
 */
struct KFParticleGpuResetGraphOperationBatch
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Clear graph counters and the matching generation's work counters using the explicit state.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation);
};

/**
 * @brief Generate or directly execute graph operations within event domains.
 * One block handles one work range, distributes first sources across lanes, and loops over second sources.
 * Source channel/event checks and pair ordering constrain the combinations; domain capacity bounds both paths.
 */
struct KFParticleGpuRouteGraphOperationBatch
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Route generation; executeTasks enables fused execution, collectDiagnosticCounters enables monitoring.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation,
                        unsigned int graphChannelOffset,
                        unsigned int executeTasks,
                        unsigned int collectDiagnosticCounters);
};

/**
 * @brief Execute stored graph tasks in bounded transaction domains.
 * One block strides over one domain's stored graph tasks for the requested generation.
 * Results, candidates, lineage, and channel statistics are updated through the explicit state.
 */
struct KFParticleGpuExecuteGraphOperationBatch
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Execute tasks and address graph channel statuses relative to graphChannelOffset.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int generation,
                        unsigned int graphChannelOffset);
};

/**
 * @brief Route track pairs with one atomic task reservation per accepted descriptor.
 * One global thread examines a Cartesian pair in one event/group and applies compatibility and pair gates.
 * acceptedTasks may exceed capacity; storedTasks counts writes. blockReservations counts individual
 * reservations in this variant, despite its name.
 */
struct KFParticleGpuRouteTwoDaughterTasksAtomic : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Append bounded routed tasks and update initialized pair/channel counters.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuTwoDaughterRoutingView routing,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        KFParticleGpuTwoDaughterRoutedTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* visitedPairs,
                        unsigned int* activeChannelBits,
                        unsigned int* acceptedTasks,
                        unsigned int* storedTasks,
                        unsigned int* blockReservations,
                        unsigned int* overflowFlags);
};

/**
 * @brief Compact accepted two-track hypotheses with a block-wide prefix scan.
 * Each lane resolves one pair; padding lanes contribute zero but still participate in scan/barriers.
 * The GPU launch must honor ScanBlockSize. Each nonempty block reserves one global task range.
 */
struct KFParticleGpuRouteTwoDaughterTasksBlockScan
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  /// @brief Compile-time lane count shared by the GPU launch and prefix-scan implementation.
  static constexpr int ScanBlockSize = 64;
  using block_size = xpu::block_size<ScanBlockSize>;
  /// @brief Exclusive prefix sum of per-lane accepted-hypothesis counts.
  using scan_t = xpu::block_scan<unsigned int, ScanBlockSize>;
  /// @brief Per-block scan scratch and the global output range reserved by lane zero.
  struct shared_data
  {
    scan_t::storage_t scan; ///< XPU scan scratch, shared only within this block.
    unsigned int blockBase; ///< First global task slot reserved for the block.
    unsigned int blockTotal; ///< Accepted hypotheses before capacity clipping.
  };
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = shared_data;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Route one event/group through published state, clipping writes to taskLimit and task capacity.
   */
  XPU_D void operator()(context& context,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        unsigned int taskLimit);
};

/**
 * @brief Fit published two-track routed tasks into a shared candidate pool.
 * One global thread expands one stored descriptor/task pair, fits the mother, and stores its descriptor index.
 * Selection evaluation is a separate step; candidate and daughter reservations are bounded.
 */
struct KFParticleGpuTwoDaughterRoutedCandidatePoolKernel
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Consume at most taskLimit stored tasks and update construction counters.
   */
  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/**
 * @brief Clear two-track routing counters and per-descriptor selection workspace.
 * Global thread zero resets generation totals; descriptor threads reset channel counters and segment cursors.
 * This does not reset the candidate pool or individual candidate selection results.
 */
struct KFParticleGpuResetTwoDaughterGenerationState
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Prepare the published generation workspace for routing and selection.
   */
  XPU_D void operator()(context& context);
};

/**
 * @brief Evaluate V0 selection for one event's constructed two-track candidates.
 * One global thread reads one candidate and its descriptor, evaluates against the event's primary vertices,
 * optionally stores diagnostics, and increments the descriptor's accepted-selection count.
 */
struct KFParticleGpuEvaluateTwoDaughterGenerationSelection
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Evaluate candidates belonging to eventIndex before preparing selected-index segments.
   */
  XPU_D void operator()(context& context, unsigned int eventIndex);
};

/**
 * @brief Assign bounded selected-index segments to two-track channel descriptors.
 * A single thread appends channel segments after the existing selected-index prefix, resets scatter cursors,
 * and reports truncation. Run after selection counting and before scattering.
 */
struct KFParticleGpuPrepareTwoDaughterSelectionSegments
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Compute offsets/stored counts from accepted counts and update the selected-index size.
   */
  XPU_D void operator()(context& context);
};

/**
 * @brief Scatter accepted candidate indices into prepared channel segments.
 * One global thread reads one stored selection result for the requested event.
 * A per-channel atomic cursor chooses its output slot; truncated results are marked as rejected.
 */
struct KFParticleGpuScatterTwoDaughterGenerationSelection
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Write indices and optional channel IDs for eventIndex after segment preparation.
   */
  XPU_D void operator()(context& context, unsigned int eventIndex);
};

/**
 * @brief Reset raw-candidate, daughter, and selected-index sizes and overflow flags.
 * Only block zero/lane zero writes counters. Payload buffers and transaction-domain statuses are untouched.
 */
struct KFParticleGpuResetBatchOutputCounters : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Clear non-null output counters before producers start a fresh batch.
   */
  XPU_D void operator()(
    context& context,
    KFParticleGpuCandidatePoolView candidates,
    KFParticleGpuSelectedCandidateIndexView selectedCandidates);
};

/**
 * @brief Route tiled track pairs into event-local transaction task reservations.
 * Global thread indices resolve a pair tile and a pair within that tile, not a cross-event track product.
 * Domain/event track ranges are checked before channel gates and bounded task publication.
 */
struct KFParticleGpuRouteTwoDaughterBatch : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Consume pairTileCount tiles; optionally accumulate detailed pair/channel monitoring.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        const KFParticleGpuPairTile* pairTiles,
                        unsigned int pairTileCount,
                        unsigned int collectDiagnosticCounters);
};

/**
 * @brief Test device diagnostic-journal access independently of transaction data.
 * Only block zero/lane zero writes a fresh record with the supplied sequence and completed journal phase.
 */
struct KFParticleGpuProbeTransactionJournal
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Initialize record without reading event input or fitting candidates.
   */
  XPU_D void operator()(context& context,
                        unsigned long long diagnosticSequence,
                        KFParticleGpuTransactionDiagnosticRecord* record);
};

/**
 * @brief Probe transaction, work, and domain descriptors up to a requested checkpoint.
 * Only block zero/lane zero runs. The record reports the reached phase and implemented guard checks,
 * not exhaustive memory validation; it is separate from production output.
 */
struct KFParticleGpuProbeTransactionStructure
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Trace workIndex/pairIndex with diagnosticSequence through the requested structural phase.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuTransactionView transaction,
                        unsigned long long diagnosticSequence,
                        unsigned int workIndex,
                        unsigned int pairIndex,
                        unsigned int phase,
                        KFParticleGpuTransactionDiagnosticRecord* record);
};

/**
 * @brief Trace one two-track transaction pair through structural and numerical gates.
 * Only block zero/lane zero runs against explicit state. pairIndex is the work's Cartesian pair index;
 * descriptorIndex chooses the hypothesis for deeper checks. No candidate is published.
 */
struct KFParticleGpuProbeTwoDaughterTransaction
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Populate record up to phase, including rejection or invalid-contract outcomes.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned long long diagnosticSequence,
                        unsigned int workIndex,
                        unsigned int pairIndex,
                        unsigned int descriptorIndex,
                        unsigned int phase,
                        KFParticleGpuTransactionDiagnosticRecord* record);
};

/**
 * @brief Fuse two-track routing and mother fitting without storing intermediate tasks.
 * One block strides over the Cartesian pairs of one work range. Domain task/output limits still apply;
 * this variant does not consume the pair-tile array used by separate routing.
 */
struct KFParticleGpuRouteConstructTwoDaughterBatch
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Optionally evaluate selection during fitting and collect detailed diagnostic counters.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int evaluateSelection,
                        unsigned int collectDiagnosticCounters);
};

/**
 * @brief Fit compact two-track task tiles with one block per tile.
 * Each tile refers to a domain and an absolute first-task index. Lanes stride over the tile so the
 * same code also works with XPU's single-thread CPU blocks. Only stored tasks are consumed.
 */
struct KFParticleGpuConstructTwoDaughterTaskTiles : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Build tileCount task tiles, optionally evaluating selection in the construction pass.
   */
  XPU_D void operator()(context& context, KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int evaluateSelection,
                        const KFParticleGpuTaskTile* tiles, unsigned int tileCount);
};

/**
 * @brief Fit two-track tasks by global task index and domain lookup.
 * One global thread locates the domain reservation containing its task index and checks the stored prefix.
 * Unlike the tiled variant, it searches domain descriptors rather than receiving a domain per block.
 */
struct KFParticleGpuConstructTwoDaughterBatch : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Fit stored tasks using explicit state; evaluateSelection enables fused selection evaluation.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction,
                        unsigned int evaluateSelection);
};

/**
 * @brief Evaluate selection for constructed two-track candidates across transaction domains.
 * One global thread locates the candidate's populated domain prefix, loads its descriptor,
 * and updates the per-candidate result and per-channel accepted-selection count.
 */
struct KFParticleGpuEvaluateTwoDaughterBatchSelection : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Evaluate against the owning event's primary vertices after construction completes.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuKernels state,
                        KFParticleGpuTransactionView transaction);
};

/**
 * @brief Prepare bounded per-channel selected-index segments independently in each domain.
 * One global thread handles one domain, computes channel offsets/stored counts, and resets cursors.
 * The global selected size is an occupied-range upper bound, not a sum across reservation gaps.
 */
struct KFParticleGpuPrepareTwoDaughterBatchSelection : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Prepare segments after selection counting and before batch selection scattering.
   */
  XPU_D void operator()(
    context& context,
    KFParticleGpuKernels state,
    KFParticleGpuTransactionView transaction,
    KFParticleGpuSelectedCandidateIndexView selectedCandidates);
};

/**
 * @brief Write selected candidate indices into prepared event-local channel segments.
 * One block strides over one domain's populated candidate prefix. Atomic channel cursors allocate slots;
 * capacity failures update selection rejection and, where checked, overflow flags.
 */
struct KFParticleGpuScatterTwoDaughterBatchSelection : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Scatter stored selection results after domain/channel segment preparation.
   */
  XPU_D void operator()(
    context& context,
    KFParticleGpuKernels state,
    KFParticleGpuTransactionView transaction,
    KFParticleGpuSelectedCandidateIndexView selectedCandidates);
};

/**
 * @brief Fit explicit two-track tasks while preserving task-to-candidate slot correspondence.
 * One global thread fits one task; failed fits produce marked entries rather than compacting the output.
 * Thread zero initializes bounded output sizes and overflow flags.
 */
struct KFParticleGpuTwoDaughterTaskKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Store task i in candidate slot i with daughter slots 2*i and 2*i+1 when capacity permits.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int numberOfTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

/**
 * @brief Fit explicit two-track tasks and compact successful mothers into an output pool.
 * One global thread fits one task; initialized pool counters reserve candidate and daughter slots.
 * A daughter-capacity failure leaves a marked failed candidate in its already reserved slot.
 */
struct KFParticleGpuTwoDaughterCompactCandidateKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Consume numberOfTasks and append bounded candidate/lineage records.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int numberOfTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

/**
 * @brief Fit a compact two-track list whose accepted count remains on the device.
 * One global thread consumes an entry below both taskCapacity and acceptedTasks[0].
 * Candidate and daughter slots are reserved only after a successful fit.
 */
struct KFParticleGpuTwoDaughterCompactCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Append fitted mothers to the initialized output pool, reporting capacity overflow.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        const unsigned int* acceptedTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

/**
 * @brief Select raw candidates against one event's primary vertices and V0 cuts.
 * One global thread evaluates an entry in the requested candidate range, restricted to eventIndex.
 * Optional per-candidate diagnostics and bounded selected-index output are written.
 */
struct KFParticleGpuSelectV0Candidates : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Evaluate candidateOffset/candidateCount with config and the supplied channelId label.
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstCandidatePoolView candidates,
                        KFParticleGpuConstVertexSoAView primaryVertices,
                        const KFParticleGpuEventDesc* events,
                        unsigned int eventIndex,
                        unsigned int candidateOffset,
                        unsigned int candidateCount,
                        unsigned int channelId,
                        KFParticleGpuV0SelectionConfig config,
                        KFParticleGpuV0SelectionResultView selectionResults,
                        KFParticleGpuSelectedCandidateIndexView selectedCandidates);
};

/**
 * @brief Generate a fixed-slot two-track task list for an event/source descriptor.
 * One global thread handles one Cartesian pair. Rejected source cuts produce invalid-track placeholders,
 * so writtenTasks counts bounded output slots, not accepted hypotheses.
 */
struct KFParticleGpuGenerateTwoDaughterTasks : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Fill up to taskCapacity slots and report the full Cartesian count in totalPairs.
   */
  XPU_D void operator()(context& context,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuTwoDaughterTaskSource source,
                        KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* writtenTasks,
                        unsigned int* totalPairs);
};

/**
 * @brief Compact two-track pairs passing source cuts into a bounded task list.
 * One global thread checks one Cartesian pair. acceptedTasks must be initialized before launch;
 * it counts all accepted hypotheses, including those not stored because taskCapacity is exhausted.
 */
struct KFParticleGpuGenerateTwoDaughterTasksCompact : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write accepted tasks and the full pair count; consumers must clamp the accepted count to capacity.
   */
  XPU_D void operator()(context& context,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuTwoDaughterTaskSource source,
                        KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* acceptedTasks,
                        unsigned int* totalPairs);
};

/**
 * @brief Exercise a fixed daughter-addition fixture for device/host numerical comparisons.
 * Only global thread zero runs the Kalman update; no published state or event buffers are used.
 */
struct KFParticleGpuKalmanUpdateProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write 8 floating-point observables and 3 integer checks to caller-provided arrays.
   */
  XPU_D void operator()(context& context, float* floatChecks, int* integerChecks);
};

/**
 * @brief Exercise line, constant-field, and full-field transport and DCA fixtures.
 * Only global thread zero evaluates fixed charged/neutral states and covariance/transport results.
 * The output is diagnostic data, not a production candidate pool.
 */
struct KFParticleGpuFieldTransportProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write 30 floats and 12 integer status/check values to caller-provided arrays.
   */
  XPU_D void operator()(context& context, float* floatChecks, int* integerChecks);
};

/**
 * @brief Exercise a fixed candidate/vertex line-topology fixture.
 * Only global thread zero computes distances, errors, pointing, and topology status.
 */
struct KFParticleGpuV0LineTopologyProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Write 8 floats and 2 unsigned status values when both output pointers are non-null.
   */
  XPU_D void operator()(context& context, float* floatChecks, unsigned int* statusChecks);
};

/**
 * @brief Intermediate numerical states from a single full-field two-track fit probe.
 * stage identifies the last completed checkpoint. Later fields must not be interpreted after an early return;
 * the probe does not initialize the entire record. This diagnostic payload owns no device buffers.
 */
struct KFParticleGpuFullFieldTwoDaughterTrace
{
  float by; ///< Local By at the first daughter's initial position.
  float firstRoots[2]; ///< Two constant-By DCA roots for the first daughter.
  float secondRoots[2]; ///< Corresponding constant-By DCA roots for the second daughter.
  float dS[2]; ///< Chosen transport parameters for the two daughters.
  float dsdr[4][6]; ///< Four derivative rows of the two transport parameters with respect to both states.
  /// @brief Intermediate arithmetic from the CPU-compatible DCA calculation.
  KFParticleGpuMath::KFParticleGpuDcaArithmeticTrace dcaArithmetic;
  KFParticleGpuFitState preliminaryFirst; ///< First daughter after independent full-field transport.
  KFParticleGpuFitState preliminarySecond; ///< Second daughter after independent full-field transport.
  KFParticleGpuFitState secondPassCurrent; ///< Current fit state from the analytic second-pass seed.
  KFParticleGpuMeasurement secondPassMeasurement; ///< Second-pass daughter measurement with propagated covariance.
  KFParticleGpuFitState fittedMother; ///< Trial mother after the energy-fit daughter update.
  unsigned int useMiddlePoint; ///< Nonzero when the CPU-compatible middle DCA point is chosen.
  unsigned int stage; ///< Last completed checkpoint: 0=none, 1=DCA, 2=transport, 3=seed, 4=fit.
};

/**
 * @brief Trace full-field transport and fitting for the first explicit two-track task.
 * Only global thread zero processes tasks[0] using production numerical helpers and fills one trace.
 * It does not append a candidate or apply the complete production publication/selection path.
 */
struct KFParticleGpuFullFieldTwoDaughterProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  /**
   * @brief Record the completed stage and its numerical states in trace[0].
   */
  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        KFParticleGpuFullFieldTwoDaughterTrace* trace);
};

/**
 * @brief Snapshot of a routed task, its descriptor, reconstructed task, and trial fit.
 * status is a progress bit mask: 1=state snapshot, 2=task, 4=descriptor, 8=expanded task,
 * 16=task validation passed, 32=fit succeeded. Read each payload only after its corresponding bit is set.
 * Pool sizes describe the observed state; the probe itself does not publish the fitted mother.
 */
struct KFParticleGpuRoutedTwoDaughterTrace
{
  KFParticleGpuTwoDaughterRoutedTask routed; ///< Stored routed hypothesis, available with status bit 2.
  KFParticleGpuTwoDaughterRoutingDescriptor descriptor; ///< Channel descriptor, available with status bit 4.
  KFParticleGpuTwoDaughterTask task; ///< Expanded numerical task, available with status bit 8.
  KFParticleGpuFitState mother; ///< Trial fit output, valid with status bit 32.
  unsigned int status = 0u; ///< Progress/validation bits described by this structure's contract.
  unsigned int storedTasks = 0u; ///< Device-reported number of written routed tasks.
  unsigned int taskCapacity = 0u; ///< Allocated routed-task capacity before applying taskLimit.
  unsigned int descriptorCount = 0u; ///< Number of published channel descriptors.
  unsigned int candidateSize = 0u; ///< Observed logical raw-candidate size.
  unsigned int candidateCapacity = 0u; ///< Allocated raw-candidate capacity.
  unsigned int daughterSize = 0u; ///< Observed logical daughter-lineage size.
  unsigned int daughterCapacity = 0u; ///< Allocated daughter-lineage capacity.
};

/**
 * @brief Inspect published routed tasks and fit them into diagnostic records only.
 * One global thread initializes one trace entry. Task access is bounded by taskLimit, storage capacity,
 * and the stored count; entries beyond that prefix still report state sizes and capacities.
 */
struct KFParticleGpuRoutedTwoDaughterProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  /**
   * @brief Fill traceCapacity records without modifying the production candidate pool.
   */
  XPU_D void operator()(context& context,
                        unsigned int taskLimit,
                        KFParticleGpuRoutedTwoDaughterTrace* trace,
                        unsigned int traceCapacity);
};

/**
 * @brief Non-owning device reconstruction interface built on the published kernel state.
 *
 * The base KFParticleGpuKernelState holds buffer views, descriptors, and counters; this class adds
 * device methods without owning memory or a queue. It is trivially copyable for XPU constant memory
 * and explicit kernel arguments. Constructing/copying it neither allocates nor transfers payload arrays.
 * Steering must keep the referenced storage alive and publish updated views after buffer replacement.
 *
 * @see KFParticleGpuKernelState
 * @see TheKFParticleFinder
 */
class KFParticleGpuKernels : public KFParticleGpuKernelState
{
 public:
  /// @brief Create an unbound state; bind the views required by a kernel before launching it.
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels() = default;

  /**
   * @brief Bind only input tracks and the raw candidate pool for round-trip probes.
   * All other views retain their defaults; this is not a complete reconstruction configuration.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuCandidatePoolView& candidates)
    : KFParticleGpuKernelState(inputTracks, candidates)
  {
  }

  /**
   * @brief Bind input, output, routing, graph, generation, and transaction views.
   * Arguments are forwarded to KFParticleGpuKernelState without validation or ownership transfer.
   * Default optional views are usable only by kernels that do not require those facilities.
   * @see KFParticleGpuKernelState::KFParticleGpuKernelState
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels(
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
    : KFParticleGpuKernelState(inputTracks,
                               primaryVertices,
                               events,
                               twoDaughterTasks,
                               twoDaughterTaskCapacity,
                               candidates,
                               selectionResults,
                               selectedCandidates,
                               v0TrackRouting,
                               twoDaughterRouting,
                               candidateDescriptorIndices,
                               decayGraph,
                               graphOperations,
                               twoDaughterGeneration,
                               v0TrackGeneration,
                               transaction)
  {
  }

  /**
   * @brief Bind the compatibility state without per-candidate selection results or a transaction.
   * Intended for probes and entry points that do not require either omitted workspace.
   * Other arguments have the same meaning and lifetime requirements as the full constructor.
   */
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuConstVertexSoAView& primaryVertices,
    const KFParticleGpuEventDesc* events,
    const KFParticleGpuTwoDaughterTask* twoDaughterTasks,
    unsigned int twoDaughterTaskCapacity,
    const KFParticleGpuCandidatePoolView& candidates,
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
      KFParticleGpuV0TrackGenerationStorageView())
    : KFParticleGpuKernels(inputTracks,
                           primaryVertices,
                           events,
                           twoDaughterTasks,
                           twoDaughterTaskCapacity,
                           candidates,
                           KFParticleGpuV0SelectionResultView(nullptr, 0u),
                           selectedCandidates,
                           v0TrackRouting,
                           twoDaughterRouting,
                           candidateDescriptorIndices,
                           decayGraph,
                           graphOperations,
                           twoDaughterGeneration,
                           v0TrackGeneration)
  {
  }

  /**
   * @brief Initialize and store independent input-track particles for a layout/transfer check.
   * @param context Launch position used to assign input tracks to global threads.
   * @param mass Mass hypothesis used to initialize each particle's energy.
   * @param eventIndex Common event label written into output metadata.
   * @see KFParticleGpuRoundTrip
   */
  XPU_D void RunRoundTrip(KFParticleGpuRoundTrip::context& context,
                          float mass,
                          unsigned int eventIndex) const;
};

static_assert(std::is_trivially_copyable<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory object must remain trivially copyable");

#else

/// @brief Type placeholder for builds without XPU kernel support.
class KFParticleGpuKernels
{
};

#endif

#endif
