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
 * @brief Lifecycle implementation for the process-wide KFParticle execution runtime.
 * Keeps queue ownership and the lifecycle mutex private; global XPU teardown belongs to the application.
 */

#include "execution/KFParticleGpuRuntime.h"

#include "execution/KFParticleGpuKernelState.h"
#include "execution/KFParticleGpuSteering.h"

#include <mutex>
#include <stdexcept>

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>
#endif

/**
 * @brief Owned runtime resources and flags hidden from users of the public header.
 *
 * Steering borrows the queue and owns its buffer manager. Explicit finalization releases
 * steering first; member declaration order also destroys steering before the queue.
 * The mutex serializes lifecycle methods only, not reads or operations on borrowed resources.
 */
struct KFParticleGpuRuntime::Impl
{
  bool fInitialized; ///< True only after steering initialization completes.
  bool fInitializedXpu; ///< Records this object's successful call to xpu::initialize(), not global ownership.
#ifdef KFPARTICLE_USE_XPU
  std::unique_ptr<xpu::queue> fQueue; ///< Owned queue, kept alive while steering borrows it.
#endif
  std::unique_ptr<KFParticleGpuSteering> fSteering; ///< Owned steering and, indirectly, its buffers and plans.
  mutable std::mutex fMutex; ///< Serializes Initialize()/Finalize(); getters do not acquire it.

  /**
   * @brief Start with false lifecycle flags and no allocated queue or steering.
   */
  Impl()
    : fInitialized(false)
    , fInitializedXpu(false)
#ifdef KFPARTICLE_USE_XPU
    , fQueue()
#endif
    , fSteering()
    , fMutex()
  {
  }
};

/**
 * @brief Lazily construct the singleton; C++ synchronizes its first construction, not all later access.
 */
KFParticleGpuRuntime& KFParticleGpuRuntime::Instance()
{
  static KFParticleGpuRuntime runtime;
  return runtime;
}

/**
 * @brief Allocate private bookkeeping without creating execution resources or initializing XPU.
 */
KFParticleGpuRuntime::KFParticleGpuRuntime() : fImpl(new Impl) {}

/**
 * @brief Finalize owned execution resources before destroying the private state.
 * No exceptions are caught here; the application should complete cleanup while XPU is still usable.
 */
KFParticleGpuRuntime::~KFParticleGpuRuntime()
{
  Finalize();
}

/**
 * @brief Delegate initialization with default runtime settings.
 */
void KFParticleGpuRuntime::Initialize()
{
  Initialize(KFParticleGpuRuntimeSettings());
}

/**
 * @brief Under the lifecycle lock, reuse or initialize XPU, preload kernels, and create queue/steering.
 * Uses a nonempty XPU device list as the initialization test. Settings are forwarded only when
 * XPU must be started; non-XPU builds ignore them. The ready flag is committed last, but earlier
 * allocations and global initialization are not rolled back if a subsequent step throws.
 */
void KFParticleGpuRuntime::Initialize(const KFParticleGpuRuntimeSettings& settings)
{
  std::lock_guard<std::mutex> lock(fImpl->fMutex);
  if (fImpl->fInitialized) {
    return;
  }

#ifdef KFPARTICLE_USE_XPU
  const bool xpuIsInitialized = !xpu::device::all().empty();
  if (!xpuIsInitialized) {
    if (!settings.initializeXpuIfNeeded) {
      throw std::logic_error("KFParticle GPU runtime requires an initialized XPU runtime");
    }
    xpu::settings xpuSettings;
    xpuSettings.device = settings.device;
    xpuSettings.profile = settings.profile;
    xpuSettings.verbose = settings.verbose;
    xpu::initialize(xpuSettings);
    fImpl->fInitializedXpu = true;
  }

  xpu::preload<KFParticleGpuDeviceImage>();
  fImpl->fQueue = std::make_unique<xpu::queue>();
  fImpl->fSteering.reset(new KFParticleGpuSteering(*fImpl->fQueue));
#else
  (void) settings;
  fImpl->fSteering.reset(new KFParticleGpuSteering);
#endif
  fImpl->fSteering->Initialize();
  fImpl->fInitialized = true;
}

/**
 * @brief Under the lifecycle lock, finalize steering before destroying it and its borrowed queue.
 * Resets flags only after cleanup succeeds. If not ready, does nothing, including for partially
 * failed initialization. Never finalizes the global XPU runtime.
 */
void KFParticleGpuRuntime::Finalize()
{
  std::lock_guard<std::mutex> lock(fImpl->fMutex);
  if (!fImpl->fInitialized) {
    return;
  }

  fImpl->fSteering->Finalize();
  fImpl->fSteering.reset();
#ifdef KFPARTICLE_USE_XPU
  fImpl->fQueue.reset();
#endif
  fImpl->fInitialized = false;
  fImpl->fInitializedXpu = false;
}

/**
 * @brief Read the KFParticle ready flag without acquiring the lifecycle mutex.
 */
bool KFParticleGpuRuntime::IsInitialized() const
{
  return fImpl->fInitialized;
}

/**
 * @brief Read the local XPU-initialization record, without querying XPU or acquiring a lock.
 */
bool KFParticleGpuRuntime::InitializedXpu() const
{
  return fImpl->fInitializedXpu;
}

/**
 * @brief Require the ready flag and return a borrowed mutable steering reference without locking.
 * @throws std::logic_error If KFParticle initialization has not completed.
 */
KFParticleGpuSteering& KFParticleGpuRuntime::GetSteering()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fSteering;
}

/**
 * @brief Require the ready flag and return a borrowed const steering reference without locking.
 * @throws std::logic_error If KFParticle initialization has not completed.
 */
const KFParticleGpuSteering& KFParticleGpuRuntime::GetSteering() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fSteering;
}

#ifdef KFPARTICLE_USE_XPU
/**
 * @brief Require the ready flag and borrow the runtime queue for coordinated external submissions.
 * @throws std::logic_error If KFParticle initialization has not completed.
 */
xpu::queue& KFParticleGpuRuntime::GetQueue()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fQueue;
}

/**
 * @brief Require the ready flag and borrow const queue access without locking or transferring ownership.
 * @throws std::logic_error If KFParticle initialization has not completed.
 */
const xpu::queue& KFParticleGpuRuntime::GetQueue() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fQueue;
}
#endif
