/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "execution/KFParticleGpuRuntimeService.h"

#include "execution/KFParticleGpuRuntime.h"
#include "execution/KFParticleGpuSteering.h"

#include <mutex>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>
#endif

namespace kfparticle::gpu
{
  GpuRuntimeService::GpuRuntimeService() = default;

  GpuRuntimeService::~GpuRuntimeService() = default;

  GpuRuntimeService& GpuRuntimeService::Instance()
  {
    static GpuRuntimeService service;
    return service;
  }

  void GpuRuntimeService::Initialize(const KFParticleGpuRuntimeSettings& settings)
  {
    std::lock_guard<std::mutex> lock(fRuntimeMutex);
    InitializeLocked(settings);
  }

  void GpuRuntimeService::InitializeLocked(const KFParticleGpuRuntimeSettings& settings)
  {
    auto& runtime = KFParticleGpuRuntime::Instance();
    const auto caller = std::this_thread::get_id();
    if (runtime.IsInitialized() && fOwnerThread != std::thread::id{}
        && fOwnerThread != caller) {
      throw std::logic_error("KFParticle GPU runtime belongs to another submission thread");
    }
#ifdef KFPARTICLE_USE_XPU
    const char* environmentDevice = std::getenv("XPU_DEVICE");
    if (!settings.device.empty() && xpu::device::all().empty()
        && environmentDevice && environmentDevice[0] != '\0'
        && settings.device != environmentDevice) {
      throw std::invalid_argument("XPU_DEVICE conflicts with the explicit KFParticle device; unset it or use the same selector");
    }
    if (!settings.device.empty() && !xpu::device::all().empty()
        && xpu::device(settings.device).id() != xpu::device::active().id()) {
      throw std::invalid_argument("KFParticle GPU requested device differs from active XPU device");
    }
#endif
    auto adopted = settings;
    if (adopted.device.empty()) { adopted.device = "cpu"; }
    runtime.Initialize(adopted);
    fOwnerThread = caller;
  }

  GpuRuntimeService::Transaction GpuRuntimeService::Acquire()
  {
    std::unique_lock<std::mutex> lock(fRuntimeMutex);

    auto& runtime = KFParticleGpuRuntime::Instance();
    if (!runtime.IsInitialized()) {
      KFParticleGpuRuntimeSettings settings;
      settings.initializeXpuIfNeeded = false;
      runtime.Initialize(settings);
    }
    if (fOwnerThread == std::thread::id{}) { fOwnerThread = std::this_thread::get_id(); }

    return Transaction(std::move(lock));
  }

  GpuRuntimeService::Transaction GpuRuntimeService::Acquire(
    const KFParticleGpuRuntimeSettings& settings)
  {
    std::unique_lock<std::mutex> lock(fRuntimeMutex);
    InitializeLocked(settings);
    return Transaction(std::move(lock));
  }

  void GpuRuntimeService::Finalize()
  {
    Finalize(false);
  }

  void GpuRuntimeService::Finalize(bool requireOwnerThread)
  {
    std::lock_guard<std::mutex> lock(fRuntimeMutex);
    auto& runtime = KFParticleGpuRuntime::Instance();
    if (requireOwnerThread && runtime.IsInitialized()
        && fOwnerThread != std::thread::id{}
        && fOwnerThread != std::this_thread::get_id()) {
      throw std::logic_error("KFParticle GPU cleanup requires its submission thread");
    }
    if (runtime.IsInitialized()) { runtime.Finalize(); }
    fOwnerThread = {};
  }

  GpuRuntimeService::Transaction::Transaction(std::unique_lock<std::mutex> lock)
    : fRuntimeLock(std::move(lock)) {}

  GpuRuntimeService::Transaction::Transaction(Transaction&&) noexcept = default;

  GpuRuntimeService::Transaction& GpuRuntimeService::Transaction::operator=(Transaction&&) noexcept = default;

  GpuRuntimeService::Transaction::~Transaction() = default;

  KFParticleGpuSteering& GpuRuntimeService::Transaction::Steering()
  {
    if (!fRuntimeLock.owns_lock()) { throw std::logic_error("KFParticle GPU diagnostic transaction is not active"); }
    return KFParticleGpuRuntime::Instance().GetSteering();
  }
}  // namespace kfparticle::gpu
