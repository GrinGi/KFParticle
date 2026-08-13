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
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuChannelRouting.h"
#include "KFParticleGpuTwoDaughterRouting.h"

#include <vector>

class KFParticleGpuBufferManager;
namespace xpu
{
  class queue;
}

/** Host wall-time of the ordered default-V0 queue sequence. */
struct KFParticleGpuDecayPlanTiming {
  double inputUploadMilliseconds = 0.;
  double constructionMilliseconds = 0.;
  double selectionMilliseconds = 0.;
  double cascadeConstructionMilliseconds = 0.;
  double outputDownloadMilliseconds = 0.;
};

struct KFParticleGpuV0TrackRoutingMonitorData {
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u;
  unsigned int visitedPairs = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int blockReservations = 0u;
  unsigned int candidates = 0u;
  unsigned int daughters = 0u;
  unsigned int overflowFlags = 0u;
};

struct KFParticleGpuTwoDaughterRoutingMonitorData {
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u;
  unsigned int selectionLaunches = 0u;
  unsigned int visitedPairs = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int blockReservations = 0u;
  unsigned int candidates = 0u;
  unsigned int daughters = 0u;
  unsigned int selectedCandidates = 0u;
  unsigned int overflowFlags = 0u;
};

struct KFParticleGpuGraphChannelMonitorData {
  unsigned int channelId = 0u;
  unsigned int eventIndex = 0u;
  unsigned int generation = 0u;
  unsigned int visitedCombinations = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int constructedCandidates = 0u;
  unsigned int rejectedTasks = 0u;
};

struct KFParticleGpuGraphExecutionMonitorData {
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u;
  unsigned int visitedCombinations = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int constructedCandidates = 0u;
  unsigned int rejectedTasks = 0u;
  unsigned int candidates = 0u;
  unsigned int daughters = 0u;
  unsigned int unsupportedNodes = 0u;
  unsigned int unsupportedFamilies = 0u;
  unsigned int overflowFlags = 0u;
};

/** Opt-in explanatory counters for one completed decay-plan transaction. */
struct KFParticleGpuPerformanceSnapshot {
  bool enabled = false;
  unsigned int events = 0u;
  unsigned int tracks = 0u;
  unsigned int vertices = 0u;
  unsigned int descriptorGroups = 0u;
  unsigned int visitedCombinations = 0u;
  unsigned int activeChannelBits = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int blockReservations = 0u;
  unsigned int rejectedTasks = 0u;
  unsigned int rawCandidates = 0u;
  unsigned int selectedCandidates = 0u;
  unsigned int daughters = 0u;
  unsigned int overflowFlags = 0u;
  unsigned int kernelLaunches = 0u;
  unsigned int queueWaits = 0u;
  std::uint64_t capacityGrowths = 0u;
  std::uint64_t hostToDeviceBytes = 0u;
  std::uint64_t deviceToHostBytes = 0u;
  std::uint64_t allocatedBytesHighWater = 0u;
  double maskDensity = 0.;
  double usefulWorkPerLaunch = 0.;
  double candidatePoolOccupancy = 0.;
  KFParticleGpuDecayPlanTiming timing;
};

/** Stable partition of one multi-event decay-plan transaction. */
struct KFParticleGpuDecayPlanEventResult {
  unsigned int eventIndex = 0;
  unsigned int channelOffset = 0;
  unsigned int channelCount = 0;
  unsigned int cascadeChannelOffset = 0;
  unsigned int cascadeChannelCount = 0;
  KFParticleGpuCandidateRange candidates;
  KFParticleGpuTwoDaughterRoutingStatus generationRouting;
  KFParticleGpuCandidateRange cascadeCandidates;
  KFParticleGpuV0TrackRoutingStatus cascadeRouting;
  KFParticleGpuCandidateRange graphCandidates;
  KFParticleGpuSelectedCandidateRange selectedCandidates;
  unsigned int overflowFlags = 0;
};

/** Result of the isolated Stage 15.2 fused cascade route. */
struct KFParticleGpuV0TrackFusedResult {
  unsigned int eventIndex = 0u;
  KFParticleGpuV0TrackRoutingStatus routing;
  KFParticleGpuCandidateRange candidates;
};

/** Result of the isolated Stage 16.2 fused default-V0 generation. */
struct KFParticleGpuTwoDaughterFusedResult {
  unsigned int eventIndex = 0u;
  unsigned int groupLaunches = 0u;
  unsigned int descriptorCount = 0u;
  KFParticleGpuTwoDaughterRoutingStatus routing;
  KFParticleGpuCandidateRange candidates;
};
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
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& RunDecayPlanBatch(
    unsigned int firstEventIndex,
    unsigned int eventCount,
    unsigned int taskCapacity);
  const std::vector<KFParticleGpuTwoDaughterChannelResult>& LastDecayPlanResults() const;
  const std::vector<KFParticleGpuV0TrackChannelResult>& LastV0TrackCascadeResults() const;
  const std::vector<KFParticleGpuDecayPlanEventResult>& LastDecayPlanEventResults() const;
  const KFParticleGpuDecayPlanTiming& LastDecayPlanTiming() const;
  const KFParticleGpuTwoDaughterRoutingMonitorData&
  LastTwoDaughterRoutingMonitorData() const;
  const KFParticleGpuV0TrackRoutingMonitorData& LastV0TrackRoutingMonitorData() const;
  const KFParticleGpuGraphExecutionMonitorData& LastGraphExecutionMonitorData() const;
  const std::vector<KFParticleGpuGraphChannelMonitorData>&
  LastGraphChannelMonitorData() const;
  void SetPerformanceMonitoringEnabled(bool enabled);
  bool PerformanceMonitoringEnabled() const;
  const KFParticleGpuPerformanceSnapshot& LastPerformanceSnapshot() const;
  const KFParticleGpuSelectedCandidateRange& LastDecayPlanSelectedCandidates() const;
  const std::vector<KFParticleGpuSelectedChannelRange>& LastDecayPlanSelectedChannels() const;
  KFParticleGpuV0TrackFusedResult RunV0TrackFusedStage(
    unsigned int eventIndex,
    const KFParticleGpuSelectedCandidateRange& selectedRange,
    unsigned int taskCapacity,
    KFParticleGpuV0TrackRoutingMode routingMode = KFGpuV0TrackRoutingBlockScan);
  KFParticleGpuTwoDaughterFusedResult RunTwoDaughterFusedStage(
    unsigned int eventIndex,
    unsigned int taskCapacity,
    KFParticleGpuTwoDaughterRoutingMode routingMode =
      KFGpuTwoDaughterRoutingBlockScan);
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
  KFParticleGpuV0TrackChannelResult RunV0TrackCompactChannel(
    const KFParticleGpuV0TrackCascadeChannel& channel,
    unsigned int eventIndex,
    unsigned int taskCapacity,
    unsigned int statusIndex);
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
