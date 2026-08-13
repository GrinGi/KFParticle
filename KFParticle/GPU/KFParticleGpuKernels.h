/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUKERNELS_H
#define KFPARTICLEGPUKERNELS_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuDeviceImage.h"
#include "KFParticleGpuGraphOperations.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuKernelState.h"
#include "KFParticleGpuTwoDaughter.h"
#include "KFParticleGpuV0Track.h"

#ifdef KFPARTICLE_GPU_XPU_ENABLED

class KFParticleGpuKernels;

// The host steering republishes this lightweight state whenever buffer views
// change. Kernels then share the same pointers through XPU constant memory.
struct TheKFParticleFinder : xpu::constant<KFParticleGpuDeviceImage, KFParticleGpuKernels>
{
};

// XPU actions stay thin: the reconstruction implementation belongs to the
// device-safe KFParticleGpuKernels object, as in the CA and STS GPU chains.
struct KFParticleGpuRoundTrip : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, float mass, unsigned int eventIndex);
};

struct KFParticleGpuLaunchSmoke : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context, unsigned int* marker);
};

// Diagnostic-only constant-memory check covering every published state view.
struct KFParticleGpuKernelStateProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int* checks);
};

struct KFParticleGpuInputLayoutProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuConstVertexSoAView primaryVertices,
                        const KFParticleGpuEventDesc* events,
                        float* floatChecks,
                        int* integerChecks,
                        unsigned int* unsignedChecks);
};

// A one-thread ABI probe; cascade reconstruction itself starts only in Step 14.2.
struct KFParticleGpuV0TrackTaskProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

struct KFParticleGpuGenerateV0TrackTasksCompact : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

struct KFParticleGpuV0TrackCompactCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView tracks,
                        KFParticleGpuConstCandidatePoolView inputCandidates,
                        const KFParticleGpuV0TrackTask* tasks,
                        unsigned int taskCapacity,
                        const unsigned int* acceptedTasks,
                        KFParticleGpuCandidatePoolView outputCandidates);
};

/**
 * One thread visits one selected-V0/bachelor pair and emits all compatible
 * descriptor bits into a single bounded worklist.
 */
struct KFParticleGpuRouteV0TrackTasksAtomic : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

/** Block-scan compaction reserves one global routed-task range per block. */
struct KFParticleGpuRouteV0TrackTasksBlockScan : xpu::kernel<KFParticleGpuDeviceImage>
{
  static constexpr int ScanBlockSize = 64;
  using block_size = xpu::block_size<ScanBlockSize>;
  using scan_t = xpu::block_scan<unsigned int, ScanBlockSize>;
  struct shared_data
  {
    scan_t::storage_t scan;
    unsigned int blockBase;
    unsigned int blockTotal;
  };
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = shared_data;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        KFParticleGpuSelectedCandidateRange selectedRange,
                        unsigned int taskLimit);
};

/** Descriptor-driven construction consumes the routed pool without a host wait. */
struct KFParticleGpuV0TrackRoutedCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/** Resets reusable composite-track routing state entirely on device. */
struct KFParticleGpuResetV0TrackGenerationState
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int resetChannelCounters);
};

/** Explicit-view graph execution oracle retained for isolated device tests. */
struct KFParticleGpuExecuteGraphOperationsExplicit
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

/** Resets reusable later-generation scheduler counters entirely on device. */
struct KFParticleGpuResetGraphOperationGeneration
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context);
};

/** Routes one graph execution group from resident candidate metadata. */
struct KFParticleGpuRouteGraphOperationTasks
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(
    context& context,
    unsigned int eventIndex,
    unsigned int groupIndex,
    unsigned int sourceCapacity,
    unsigned int taskLimit);
};

/** Executes the device-generated task count without a host readback. */
struct KFParticleGpuExecuteRoutedGraphOperations
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/** One thread visits one event-local track pair and emits every active channel bit. */
struct KFParticleGpuRouteTwoDaughterTasksAtomic : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

/** Block-scan compaction performs one bounded global task reservation per block. */
struct KFParticleGpuRouteTwoDaughterTasksBlockScan
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  static constexpr int ScanBlockSize = 64;
  using block_size = xpu::block_size<ScanBlockSize>;
  using scan_t = xpu::block_scan<unsigned int, ScanBlockSize>;
  struct shared_data
  {
    scan_t::storage_t scan;
    unsigned int blockBase;
    unsigned int blockTotal;
  };
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = shared_data;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context,
                        unsigned int eventIndex,
                        unsigned int groupIndex,
                        unsigned int taskLimit);
};

/** Descriptor-driven construction consumes routed tasks directly on the same queue. */
struct KFParticleGpuTwoDaughterRoutedCandidatePoolKernel
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int taskLimit);
};

/** Resets one event's routing and selection counters without a host round-trip. */
struct KFParticleGpuResetTwoDaughterGenerationState
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context);
};

/** Evaluates every raw candidate using its O(1) routing descriptor tag. */
struct KFParticleGpuEvaluateTwoDaughterGenerationSelection
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int eventIndex);
};

/** One device thread assigns bounded contiguous selected-output channel segments. */
struct KFParticleGpuPrepareTwoDaughterSelectionSegments
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context);
};

/** Scatters selected raw indices into the descriptor segments prepared above. */
struct KFParticleGpuScatterTwoDaughterGenerationSelection
  : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context, unsigned int eventIndex);
};

struct KFParticleGpuTwoDaughterTaskKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int numberOfTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

struct KFParticleGpuTwoDaughterCompactCandidateKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int numberOfTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

struct KFParticleGpuTwoDaughterCompactCandidatePoolKernel : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        const unsigned int* acceptedTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

struct KFParticleGpuSelectV0Candidates : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

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

struct KFParticleGpuGenerateTwoDaughterTasks : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuTwoDaughterTaskSource source,
                        KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* writtenTasks,
                        unsigned int* totalPairs);
};

struct KFParticleGpuGenerateTwoDaughterTasksCompact : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        const KFParticleGpuEventDesc* events,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuTwoDaughterTaskSource source,
                        KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int taskCapacity,
                        unsigned int* acceptedTasks,
                        unsigned int* totalPairs);
};

struct KFParticleGpuKalmanUpdateProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context, float* floatChecks, int* integerChecks);
};

struct KFParticleGpuFieldTransportProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context, float* floatChecks, int* integerChecks);
};

// Test-only device call for the Stage 13.1 topology contract. It keeps the
// new helper independently backend-qualified before selection consumes it.
struct KFParticleGpuV0LineTopologyProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context, float* floatChecks, unsigned int* statusChecks);
};

// Test-only stage probe for the coupled full-field two-daughter path. It uses
// the production helpers, but returns compact POD diagnostics instead of
// printing from the device, so host/device divergence stays reproducible.
struct KFParticleGpuFullFieldTwoDaughterTrace
{
  float by;
  float firstRoots[2];
  float secondRoots[2];
  float dS[2];
  float dsdr[4][6];
  KFParticleGpuMath::KFParticleGpuDcaArithmeticTrace dcaArithmetic;
  KFParticleGpuFitState preliminaryFirst;
  KFParticleGpuFitState preliminarySecond;
  KFParticleGpuFitState secondPassCurrent;
  KFParticleGpuMeasurement secondPassMeasurement;
  KFParticleGpuFitState fittedMother;
  unsigned int useMiddlePoint;
  unsigned int stage;
};

struct KFParticleGpuFullFieldTwoDaughterProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        KFParticleGpuFullFieldTwoDaughterTrace* trace);
};

// Test-only snapshot of the exact constant-memory state consumed by the
// routed candidate executor. It distinguishes task/descriptor corruption
// from construction and candidate-pool publication failures.
struct KFParticleGpuRoutedTwoDaughterTrace
{
  KFParticleGpuTwoDaughterRoutedTask routed;
  KFParticleGpuTwoDaughterRoutingDescriptor descriptor;
  KFParticleGpuTwoDaughterTask task;
  KFParticleGpuFitState mother;
  unsigned int status = 0u;
  unsigned int storedTasks = 0u;
  unsigned int taskCapacity = 0u;
  unsigned int descriptorCount = 0u;
  unsigned int candidateSize = 0u;
  unsigned int candidateCapacity = 0u;
  unsigned int daughterSize = 0u;
  unsigned int daughterCapacity = 0u;
};

struct KFParticleGpuRoutedTwoDaughterProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using constants = xpu::cmem<TheKFParticleFinder>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory, constants>;

  XPU_D void operator()(context& context,
                        unsigned int taskLimit,
                        KFParticleGpuRoutedTwoDaughterTrace* trace,
                        unsigned int traceCapacity);
};

// Reads a stored candidate through the same SoA view used by reconstruction.
// Together with the full-field probe it separates construction from store and
// device-to-host transfer without relying on device-side printf.
struct KFParticleGpuCandidatePoolReadbackProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstCandidatePoolView candidates,
                        float* floatChecks,
                        int* integerChecks);
};

// Captures the result of the production two-daughter builder before it reaches
// StoreCandidateFit. This distinguishes a builder stack/codegen issue from a
// candidate-SoA store issue in a single device launch.
struct KFParticleGpuTwoDaughterBuildProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        float* floatChecks,
                        int* integerChecks);
};

// Uses the exact production task-kernel argument layout and writes an
// unmistakable candidate SoA sentinel. This checks XPU/HIP aggregate argument
// ABI independently of reconstruction math and candidate construction.
struct KFParticleGpuTwoDaughterArgumentProbe : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int numberOfTasks,
                        KFParticleGpuCandidatePoolView candidates);
};

/**
 * Device-side KFParticle reconstruction methods.
 *
 * The inherited KFParticleGpuKernelState is the flat non-owning ABI copied to
 * XPU constant memory. Device methods stay here so the published data contract
 * remains independent from reconstruction implementation.
 */
class KFParticleGpuKernels : public KFParticleGpuKernelState
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuKernels(
    const KFParticleGpuConstInputTrackSoAView& inputTracks,
    const KFParticleGpuCandidatePoolView& candidates)
    : KFParticleGpuKernelState(inputTracks, candidates)
  {
  }

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
      KFParticleGpuV0TrackGenerationStorageView())
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
                               v0TrackGeneration)
  {
  }

  // Retain the original published-state constructor for probes that do not
  // consume selection diagnostics.
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

  XPU_D void RunRoundTrip(KFParticleGpuRoundTrip::context& context,
                          float mass,
                          unsigned int eventIndex) const;
};

static_assert(std::is_trivially_copyable<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory object must remain trivially copyable");

#else

class KFParticleGpuKernels
{
};

#endif

#endif
