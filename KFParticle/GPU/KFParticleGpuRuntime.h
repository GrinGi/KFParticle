/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPURUNTIME_H
#define KFPARTICLEGPURUNTIME_H

#include <memory>
#include <string>

class KFParticleGpuSteering;

#ifdef KFPARTICLE_USE_XPU
namespace xpu
{
  class queue;
}
#endif

struct KFParticleGpuRuntimeSettings
{
  std::string device = "cpu";
  bool profile = false;
  bool verbose = false;
  bool initializeXpuIfNeeded = true;
};

/**
 * Process-wide owner of the KFParticle GPU execution infrastructure.
 *
 * XPU itself may be initialized by the application. KFParticle owns its queue
 * and buffers, but never assumes ownership of the global XPU runtime.
 */
class KFParticleGpuRuntime
{
 public:
  static KFParticleGpuRuntime& Instance();

  void Initialize();
  void Initialize(const KFParticleGpuRuntimeSettings& settings);
  void Finalize();

  bool IsInitialized() const;
  bool InitializedXpu() const;

  KFParticleGpuSteering& GetSteering();
  const KFParticleGpuSteering& GetSteering() const;

#ifdef KFPARTICLE_USE_XPU
  xpu::queue& GetQueue();
  const xpu::queue& GetQueue() const;
#endif

 private:
  KFParticleGpuRuntime();
  ~KFParticleGpuRuntime();

  KFParticleGpuRuntime(const KFParticleGpuRuntime&);
  KFParticleGpuRuntime& operator=(const KFParticleGpuRuntime&);

  struct Impl;
  std::unique_ptr<Impl> fImpl;
};

#endif
