/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuRuntime.h"

#include "KFParticleGpuDeviceImage.h"
#include "KFParticleGpuSteering.h"

#include <mutex>
#include <stdexcept>

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>
#endif

struct KFParticleGpuRuntime::Impl
{
  bool fInitialized;
  bool fInitializedXpu;
#ifdef KFPARTICLE_USE_XPU
  std::unique_ptr<xpu::queue> fQueue;
#endif
  std::unique_ptr<KFParticleGpuSteering> fSteering;
  mutable std::mutex fMutex;

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

KFParticleGpuRuntime& KFParticleGpuRuntime::Instance()
{
  static KFParticleGpuRuntime runtime;
  return runtime;
}

KFParticleGpuRuntime::KFParticleGpuRuntime() : fImpl(new Impl) {}

KFParticleGpuRuntime::~KFParticleGpuRuntime()
{
  Finalize();
}

void KFParticleGpuRuntime::Initialize()
{
  Initialize(KFParticleGpuRuntimeSettings());
}

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

bool KFParticleGpuRuntime::IsInitialized() const
{
  return fImpl->fInitialized;
}

bool KFParticleGpuRuntime::InitializedXpu() const
{
  return fImpl->fInitializedXpu;
}

KFParticleGpuSteering& KFParticleGpuRuntime::GetSteering()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fSteering;
}

const KFParticleGpuSteering& KFParticleGpuRuntime::GetSteering() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fSteering;
}

#ifdef KFPARTICLE_USE_XPU
xpu::queue& KFParticleGpuRuntime::GetQueue()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fQueue;
}

const xpu::queue& KFParticleGpuRuntime::GetQueue() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU runtime is not initialized");
  }
  return *fImpl->fQueue;
}
#endif
