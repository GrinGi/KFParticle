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
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuKernelState.h"
#include "KFParticleGpuTwoDaughter.h"

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

// Diagnostic-only constant-memory check. Production kernels keep their
// explicit-view ABI until this path is validated on every enabled backend.
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

struct KFParticleGpuRoundTripArgs : xpu::kernel<KFParticleGpuDeviceImage>
{
  using block_size = xpu::block_size<64>;
  using shared_memory = xpu::no_smem;
  using context = xpu::kernel_context<shared_memory>;

  XPU_D void operator()(context& context,
                        KFParticleGpuConstInputTrackSoAView inputTracks,
                        KFParticleGpuCandidatePoolView candidates,
                        float mass,
                        unsigned int eventIndex);
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
    const KFParticleGpuSelectedCandidateIndexView& selectedCandidates)
    : KFParticleGpuKernelState(inputTracks,
                               primaryVertices,
                               events,
                               twoDaughterTasks,
                               twoDaughterTaskCapacity,
                               candidates,
                               selectionResults,
                               selectedCandidates)
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
    const KFParticleGpuSelectedCandidateIndexView& selectedCandidates)
    : KFParticleGpuKernels(inputTracks,
                           primaryVertices,
                           events,
                           twoDaughterTasks,
                           twoDaughterTaskCapacity,
                           candidates,
                           KFParticleGpuV0SelectionResultView(nullptr, 0u),
                           selectedCandidates)
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
