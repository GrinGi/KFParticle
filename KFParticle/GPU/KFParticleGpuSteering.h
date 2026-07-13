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

class KFParticleGpuDecayPlan;
class KFParticleGpuRuntime;

#ifdef KFPARTICLE_USE_XPU
#include "KFParticleGpuDecayPlan.h"

#include <vector>

class KFParticleGpuBufferManager;
namespace xpu
{
  class queue;
}
#endif

/**
 * Host-side coordinator of KFParticle GPU reconstruction.
 *
 * This class owns reusable device buffers and uses the process-wide queue to
 * launch reconstruction stages.
 */
class KFParticleGpuSteering
{
 public:
  ~KFParticleGpuSteering();

  bool IsInitialized() const;

  KFParticleGpuDecayPlan& GetDecayPlan();
  const KFParticleGpuDecayPlan& GetDecayPlan() const;

#ifdef KFPARTICLE_USE_XPU
  KFParticleGpuBufferManager& GetBuffers();
  const KFParticleGpuBufferManager& GetBuffers() const;
  void RunRoundTrip(float mass, unsigned int eventIndex = 0);
  void RunTwoDaughterStage(const KFParticleGpuTwoDaughterTaskSource& source,
                           unsigned int taskCapacity);
  void RunTwoDaughterCompactStage(const KFParticleGpuTwoDaughterTaskSource& source,
                                  unsigned int taskCapacity);
  KFParticleGpuSelectedCandidateRange RunV0Selection(
    const KFParticleGpuTwoDaughterChannel& channel,
    const KFParticleGpuCandidateRange& rawCandidates,
    unsigned int eventIndex);
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& RunDecayPlan(
    unsigned int eventIndex,
    unsigned int taskCapacity);
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& LastDecayPlanResults() const;
  const KFParticleGpuSelectedCandidateRange& LastDecayPlanSelectedCandidates() const;
  const std::vector<KFParticleGpuSelectedChannelRange>& LastDecayPlanSelectedChannels() const;
#endif

 private:
  friend class KFParticleGpuRuntime;

#ifdef KFPARTICLE_USE_XPU
  explicit KFParticleGpuSteering(xpu::queue& queue);
  KFParticleGpuTwoDaughterChannelResult RunTwoDaughterCompactStageImpl(
    const KFParticleGpuTwoDaughterTaskSource& source,
    unsigned int taskCapacity,
    unsigned int channelId);
  KFParticleGpuTwoDaughterChannelResult RunTwoDaughterCompactChannel(
    const KFParticleGpuTwoDaughterTaskSource& source,
    unsigned int taskCapacity,
    unsigned int channelId,
    unsigned int candidateOffset,
    unsigned int daughterOffset);
#else
  KFParticleGpuSteering();
#endif

  KFParticleGpuSteering(const KFParticleGpuSteering&);
  KFParticleGpuSteering& operator=(const KFParticleGpuSteering&);

  void Initialize();
  void Finalize();

  struct Impl;
  std::unique_ptr<Impl> fImpl;
};

#endif
