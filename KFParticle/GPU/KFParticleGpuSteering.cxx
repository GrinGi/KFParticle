/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuSteering.h"

#include "KFParticleGpuBufferManager.h"
#include "KFParticleGpuDecayGraphPlan.h"
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuKernels.h"
#include "KFParticleGpuRoutingPlan.h"

#include <chrono>
#include <limits>
#include <stdexcept>

#ifdef KFPARTICLE_USE_XPU
#include <xpu/host.h>
#endif

#ifdef KFPARTICLE_USE_XPU
namespace
{
  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  KFParticleGpuKernels MakeEmptyKernels()
  {
    const KFParticleGpuInputTrackSoAView emptyInput(
      nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuVertexSoAView emptyVertices(nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuFitSoAView emptyFit(nullptr, nullptr, nullptr, nullptr, 0, 0);
    const KFParticleGpuCandidateMetadataSoAView emptyMetadata(nullptr, nullptr, 0);
    const KFParticleGpuDaughterStorageView emptyDaughters(nullptr, nullptr, 0);
    const KFParticleGpuCandidatePoolView emptyCandidates(
      emptyFit, emptyMetadata, emptyDaughters, nullptr, nullptr, 0);

    const KFParticleGpuV0SelectionResultView emptySelectionResults(nullptr, 0u);
    const KFParticleGpuSelectedCandidateIndexView emptySelected(nullptr, nullptr, nullptr, 0);
    return KFParticleGpuKernels(MakeConstView(emptyInput),
                                MakeConstView(emptyVertices),
                                nullptr,
                                nullptr,
                                0u,
                                emptyCandidates,
                                emptySelectionResults,
                                emptySelected);
  }

  KFParticleGpuKernels MakeKernelState(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    return KFParticleGpuKernels(MakeConstView(buffers.DeviceInputTracks()),
                                MakeConstView(buffers.DevicePrimaryVertices()),
                                buffers.DeviceEvents(),
                                storage.fTwoDaughterTasks.get(),
                                buffers.Capacities().twoDaughterTasks,
                                buffers.DeviceCandidates(),
                                buffers.DeviceV0SelectionResults(),
                                buffers.DeviceSelectedCandidates(),
                                buffers.DeviceV0TrackRouting(),
                                buffers.DeviceTwoDaughterRouting(),
                                buffers.DeviceCandidateDescriptorIndices(),
                                buffers.DeviceDecayGraph(),
                                buffers.DeviceGraphOperationStorage(),
                                buffers.DeviceTwoDaughterGenerationStorage(),
                                buffers.DeviceV0TrackGenerationStorage());
  }

  unsigned int CountTwoDaughterPairs(const KFParticleGpuEventDesc& event,
                                     const KFParticleGpuTwoDaughterTaskSource& source)
  {
    const KFParticleGpuRange firstRange =
      ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
    const KFParticleGpuRange secondRange =
      ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
    return firstRange.size * secondRange.size;
  }

  unsigned int CountGraphPayload(const KFParticleGpuDecayGraphPlan& graph,
                                 unsigned int payloadKind)
  {
    unsigned int count = 0u;
    for (const KFParticleGpuGraphNode& node : graph.Nodes()) {
      if (node.supportStatus == KFGpuGraphSupported
          && node.payloadKind == payloadKind) {
        ++count;
      }
    }
    return count;
  }

  // Host-pinned status snapshots are queued after each cascade channel and
  // consumed together after the final device synchronization.
  struct KFParticleGpuCascadeStatusSnapshot
  {
    bool copied = false;
    unsigned int acceptedTasks = 0u;
    unsigned int totalPairs = 0u;
    unsigned int taskOverflowFlags = 0u;
    unsigned int candidates = 0u;
    unsigned int daughters = 0u;
    unsigned int candidateOverflowFlags = 0u;
  };

  struct KFParticleGpuFusedCascadeSnapshot
  {
    unsigned int groupLaunches = 0u;
    unsigned int visitedPairs = 0u;
    unsigned int activeChannelBits = 0u;
    unsigned int acceptedTasks = 0u;
    unsigned int storedTasks = 0u;
    unsigned int blockReservations = 0u;
    unsigned int taskOverflowFlags = 0u;
    unsigned int candidateBegin = 0u;
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u;
    unsigned int daughterEnd = 0u;
    unsigned int candidateOverflowFlags = 0u;
    std::vector<unsigned int> channelVisited;
    std::vector<unsigned int> channelAccepted;
    std::vector<unsigned int> channelStored;
    std::vector<unsigned int> channelConstructed;
  };

  struct KFParticleGpuFusedTwoDaughterSnapshot
  {
    unsigned int groupLaunches = 0u;
    unsigned int visitedPairs = 0u;
    unsigned int activeChannelBits = 0u;
    unsigned int acceptedTasks = 0u;
    unsigned int storedTasks = 0u;
    unsigned int blockReservations = 0u;
    unsigned int taskOverflowFlags = 0u;
    unsigned int candidateBegin = 0u;
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u;
    unsigned int daughterEnd = 0u;
    unsigned int selectedBegin = 0u;
    unsigned int selectedEnd = 0u;
    unsigned int candidateOverflowFlags = 0u;
    unsigned int selectedOverflowFlags = 0u;
    std::vector<unsigned int> channelVisited;
    std::vector<unsigned int> channelAccepted;
    std::vector<unsigned int> channelStored;
    std::vector<unsigned int> channelConstructed;
    std::vector<unsigned int> selectionAccepted;
    std::vector<unsigned int> selectionStored;
    std::vector<unsigned int> selectionOffsets;
  };

  struct KFParticleGpuGraphOperationSnapshot
  {
    unsigned int eventIndex = 0u;
    unsigned int groupIndex = 0u;
    unsigned int generation = 0u;
    unsigned int visited = 0u;
    unsigned int accepted = 0u;
    unsigned int stored = 0u;
    unsigned int constructed = 0u;
    unsigned int rejected = 0u;
    unsigned int overflowFlags = 0u;
    unsigned int candidateBegin = 0u;
    unsigned int candidateEnd = 0u;
    unsigned int daughterBegin = 0u;
    unsigned int daughterEnd = 0u;
    unsigned int candidateOverflowFlags = 0u;
    std::vector<unsigned int> channelVisited;
    std::vector<unsigned int> channelAccepted;
    std::vector<unsigned int> channelStored;
    std::vector<unsigned int> channelConstructed;
    std::vector<unsigned int> channelRejected;
  };
}
#endif

struct KFParticleGpuSteering::Impl
{
  bool fInitialized;
#ifdef KFPARTICLE_USE_XPU
  xpu::queue& fQueue;
  std::unique_ptr<KFParticleGpuBufferManager> fBuffers;
  KFParticleGpuKernels fKernels;
  std::vector<KFParticleGpuTwoDaughterChannelResult> fLastDecayPlanResults;
  std::vector<KFParticleGpuV0TrackChannelResult> fLastV0TrackCascadeResults;
  std::vector<KFParticleGpuCascadeStatusSnapshot> fCascadeStatusSnapshots;
  std::vector<KFParticleGpuFusedTwoDaughterSnapshot> fFusedTwoDaughterSnapshots;
  std::vector<KFParticleGpuFusedCascadeSnapshot> fFusedCascadeSnapshots;
  std::vector<KFParticleGpuDecayPlanEventResult> fLastDecayPlanEventResults;
  KFParticleGpuDecayPlanTiming fLastDecayPlanTiming;
  KFParticleGpuTwoDaughterRoutingMonitorData fLastTwoDaughterRoutingMonitorData;
  KFParticleGpuV0TrackRoutingMonitorData fLastV0TrackRoutingMonitorData;
  KFParticleGpuGraphExecutionMonitorData fLastGraphExecutionMonitorData;
  bool fPerformanceMonitoringEnabled;
  KFParticleGpuPerformanceSnapshot fLastPerformanceSnapshot;
  std::vector<KFParticleGpuGraphChannelMonitorData> fLastGraphChannelMonitorData;
  std::vector<KFParticleGpuGraphOperationSnapshot> fGraphOperationSnapshots;
  KFParticleGpuSelectedCandidateRange fLastDecayPlanSelectedCandidates;
  std::vector<KFParticleGpuSelectedChannelRange> fLastDecayPlanSelectedChannels;
  KFParticleGpuTwoDaughterRoutingPlan fTwoDaughterRoutingPlan;
  KFParticleGpuV0TrackRoutingPlan fV0TrackRoutingPlan;
  KFParticleGpuDecayGraphPlan fDecayGraphPlan;
  unsigned long long fDecayGraphPlanSourceRevision;
#endif
  std::unique_ptr<KFParticleGpuDecayPlan> fDecayPlan;

#ifdef KFPARTICLE_USE_XPU
  explicit Impl(xpu::queue& queue)
    : fInitialized(false), fQueue(queue), fBuffers(), fKernels(), fLastDecayPlanResults(),
      fLastV0TrackCascadeResults(),
      fCascadeStatusSnapshots(),
      fFusedTwoDaughterSnapshots(),
      fFusedCascadeSnapshots(),
      fLastDecayPlanEventResults(),
      fLastDecayPlanTiming(), fLastTwoDaughterRoutingMonitorData(),
      fLastV0TrackRoutingMonitorData(), fLastGraphExecutionMonitorData(),
      fPerformanceMonitoringEnabled(false), fLastPerformanceSnapshot(),
      fLastGraphChannelMonitorData(), fGraphOperationSnapshots(),
      fLastDecayPlanSelectedCandidates(), fLastDecayPlanSelectedChannels(),
      fTwoDaughterRoutingPlan(), fV0TrackRoutingPlan(), fDecayGraphPlan(),
      fDecayGraphPlanSourceRevision(0u), fDecayPlan()
  {
  }

  void PublishKernelState()
  {
    fKernels = MakeKernelState(*fBuffers);
    xpu::set<TheKFParticleFinder>(fKernels);
  }

  void CompletePerformanceSnapshot(unsigned int eventCount,
                                   std::uint64_t growthCountBefore,
                                   unsigned int planUploadWaits,
                                   unsigned int fixedQueueWaits)
  {
    if (!fPerformanceMonitoringEnabled) { return; }
    KFParticleGpuPerformanceSnapshot& snapshot = fLastPerformanceSnapshot;
    snapshot.enabled = true;
    snapshot.events = eventCount;
    snapshot.tracks = fBuffers->TrackSize();
    snapshot.vertices = fBuffers->VertexSize();
    snapshot.descriptorGroups = fLastTwoDaughterRoutingMonitorData.groupLaunches
                                + fLastV0TrackRoutingMonitorData.groupLaunches
                                + fLastGraphExecutionMonitorData.groupLaunches;
    snapshot.visitedCombinations = fLastTwoDaughterRoutingMonitorData.visitedPairs
                                   + fLastV0TrackRoutingMonitorData.visitedPairs
                                   + fLastGraphExecutionMonitorData.visitedCombinations;
    snapshot.activeChannelBits = fLastTwoDaughterRoutingMonitorData.activeChannelBits
                                 + fLastV0TrackRoutingMonitorData.activeChannelBits;
    snapshot.acceptedTasks = fLastTwoDaughterRoutingMonitorData.acceptedTasks
                             + fLastV0TrackRoutingMonitorData.acceptedTasks
                             + fLastGraphExecutionMonitorData.acceptedTasks;
    snapshot.storedTasks = fLastTwoDaughterRoutingMonitorData.storedTasks
                           + fLastV0TrackRoutingMonitorData.storedTasks
                           + fLastGraphExecutionMonitorData.storedTasks;
    snapshot.blockReservations = fLastTwoDaughterRoutingMonitorData.blockReservations
                                 + fLastV0TrackRoutingMonitorData.blockReservations;
    snapshot.rejectedTasks = fLastGraphExecutionMonitorData.rejectedTasks;
    snapshot.rawCandidates = fLastTwoDaughterRoutingMonitorData.candidates
                             + fLastV0TrackRoutingMonitorData.candidates
                             + fLastGraphExecutionMonitorData.candidates;
    snapshot.selectedCandidates = fLastTwoDaughterRoutingMonitorData.selectedCandidates;
    snapshot.daughters = fLastTwoDaughterRoutingMonitorData.daughters
                         + fLastV0TrackRoutingMonitorData.daughters
                         + fLastGraphExecutionMonitorData.daughters;
    snapshot.overflowFlags = fLastTwoDaughterRoutingMonitorData.overflowFlags
                             | fLastV0TrackRoutingMonitorData.overflowFlags
                             | fLastGraphExecutionMonitorData.overflowFlags;
    snapshot.kernelLaunches = fLastTwoDaughterRoutingMonitorData.groupLaunches
                              + fLastV0TrackRoutingMonitorData.groupLaunches
                              + 3u * fLastGraphExecutionMonitorData.groupLaunches;
    if (fLastTwoDaughterRoutingMonitorData.descriptorCount > 0u) {
      snapshot.kernelLaunches += eventCount * 5u;
    }
    if (fLastV0TrackRoutingMonitorData.descriptorCount > 0u) {
      snapshot.kernelLaunches += eventCount * 2u;
    }
    snapshot.queueWaits = fixedQueueWaits + planUploadWaits;
    snapshot.capacityGrowths = fBuffers->CapacityGrowthCount() - growthCountBefore;
    snapshot.hostToDeviceBytes = fBuffers->InputPayloadBytes();
    snapshot.deviceToHostBytes = fBuffers->CandidatePayloadBytes()
                                 + fBuffers->SelectedCandidatePayloadBytes();
    snapshot.allocatedBytesHighWater = fBuffers->AllocatedBytes();
    const double maskSlots =
      static_cast<double>(fLastTwoDaughterRoutingMonitorData.visitedPairs)
        * fLastTwoDaughterRoutingMonitorData.descriptorCount
      + static_cast<double>(fLastV0TrackRoutingMonitorData.visitedPairs)
          * fLastV0TrackRoutingMonitorData.descriptorCount;
    snapshot.maskDensity = maskSlots > 0. ? snapshot.activeChannelBits / maskSlots : 0.;
    snapshot.usefulWorkPerLaunch = snapshot.kernelLaunches > 0u
                                     ? static_cast<double>(snapshot.storedTasks)
                                         / snapshot.kernelLaunches
                                     : 0.;
    snapshot.candidatePoolOccupancy = fBuffers->Capacities().candidates > 0u
                                        ? static_cast<double>(snapshot.rawCandidates)
                                            / fBuffers->Capacities().candidates
                                        : 0.;
    snapshot.timing = fLastDecayPlanTiming;
  }
#else
  Impl() : fInitialized(false), fDecayPlan() {}
#endif
};

#ifdef KFPARTICLE_USE_XPU
KFParticleGpuSteering::KFParticleGpuSteering(xpu::queue& queue) : fImpl(new Impl(queue)) {}
#else
KFParticleGpuSteering::KFParticleGpuSteering() : fImpl(new Impl) {}
#endif

KFParticleGpuSteering::~KFParticleGpuSteering()
{
  Finalize();
}

void KFParticleGpuSteering::Initialize()
{
  if (fImpl->fInitialized) {
    return;
  }

#ifdef KFPARTICLE_USE_XPU
  fImpl->fBuffers.reset(new KFParticleGpuBufferManager(fImpl->fQueue));
  fImpl->fTwoDaughterRoutingPlan.Clear();
  fImpl->fV0TrackRoutingPlan.Clear();
  fImpl->fDecayGraphPlan.Clear();
  fImpl->fDecayGraphPlanSourceRevision = 0u;
#endif
  fImpl->fDecayPlan = std::make_unique<KFParticleGpuDecayPlan>();
  fImpl->fInitialized = true;
}

void KFParticleGpuSteering::Finalize()
{
  if (!fImpl->fInitialized) {
    return;
  }

#ifdef KFPARTICLE_USE_XPU
  // Clear published device pointers before releasing their owning buffers.
  fImpl->fQueue.wait();
  fImpl->fKernels = MakeEmptyKernels();
  xpu::set<TheKFParticleFinder>(fImpl->fKernels);
  fImpl->fBuffers.reset();
#endif
  fImpl->fDecayPlan.reset();
  fImpl->fInitialized = false;
}

bool KFParticleGpuSteering::IsInitialized() const
{
  return fImpl->fInitialized;
}

KFParticleGpuDecayPlan& KFParticleGpuSteering::GetDecayPlan()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fDecayPlan;
}

const KFParticleGpuDecayPlan& KFParticleGpuSteering::GetDecayPlan() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fDecayPlan;
}

#ifdef KFPARTICLE_USE_XPU
KFParticleGpuBufferManager& KFParticleGpuSteering::GetBuffers()
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fBuffers;
}

const KFParticleGpuBufferManager& KFParticleGpuSteering::GetBuffers() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return *fImpl->fBuffers;
}

void KFParticleGpuSteering::RunRoundTrip(float mass, unsigned int eventIndex)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (mass < 0.f) {
    throw std::invalid_argument("KFParticle GPU round-trip mass must be non-negative");
  }
  if (fImpl->fBuffers->TrackSize() > 0 && eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU round-trip event index is out of range");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();
  // Publish fresh views after any capacity or event-size change, including an
  // empty event for which no action is launched.
  fImpl->PublishKernelState();

  if (fImpl->fBuffers->TrackSize() > 0) {
    fImpl->fQueue.launch<KFParticleGpuRoundTrip>(
      xpu::n_threads(fImpl->fBuffers->TrackSize()), mass, eventIndex);
    fImpl->fQueue.wait();
  }

  fImpl->fBuffers->DownloadCandidates();
}

void KFParticleGpuSteering::RunTwoDaughterStage(const KFParticleGpuTwoDaughterTaskSource& source,
                                                unsigned int taskCapacity)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU two-daughter stage task capacity must be positive");
  }

  if (source.eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU two-daughter stage event index is out of range");
  }
  if (taskCapacity > fImpl->fBuffers->Capacities().candidates) {
    throw std::out_of_range("KFParticle GPU two-daughter stage task capacity exceeds candidate capacity");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  fImpl->fQueue.launch<KFParticleGpuGenerateTwoDaughterTasks>(
    xpu::n_threads(taskCapacity),
    fImpl->fBuffers->DeviceEvents(),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    storage.fTwoDaughterTaskCount.get(),
    storage.fTwoDaughterTotalPairCount.get());
  fImpl->fQueue.wait();

  fImpl->fQueue.launch<KFParticleGpuTwoDaughterTaskKernel>(
    xpu::n_threads(taskCapacity),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    fImpl->fBuffers->DeviceCandidates());
  fImpl->fQueue.wait();

  fImpl->fBuffers->DownloadCandidates();
}

void KFParticleGpuSteering::RunTwoDaughterCompactStage(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity)
{
  (void) RunTwoDaughterCompactStageImpl(source, taskCapacity, 0u);
  fImpl->fBuffers->DownloadCandidates();
}

KFParticleGpuTwoDaughterChannelResult KFParticleGpuSteering::RunTwoDaughterCompactStageImpl(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity,
  unsigned int channelId)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU compact two-daughter stage task capacity must be positive");
  }
  if (source.eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU compact two-daughter stage event index is out of range");
  }
  if (taskCapacity > fImpl->fBuffers->Capacities().candidates) {
    throw std::out_of_range(
      "KFParticle GPU compact two-daughter stage task capacity exceeds candidate capacity");
  }

  fImpl->fBuffers->UploadInput();
  fImpl->fBuffers->ResetCandidates();

  return RunTwoDaughterCompactChannel(source, taskCapacity, channelId, 0u, 0u);
}

KFParticleGpuTwoDaughterChannelResult KFParticleGpuSteering::RunTwoDaughterCompactChannel(
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int taskCapacity,
  unsigned int channelId,
  unsigned int candidateOffset,
  unsigned int daughterOffset)
{
  const unsigned int totalPairs =
    CountTwoDaughterPairs(fImpl->fBuffers->HostEvents()[source.eventIndex], source);
  KFParticleGpuTwoDaughterChannelResult result;
  result.channelId = channelId;
  result.motherPdg = source.motherPdg;
  result.eventIndex = source.eventIndex;
  result.totalPairs = totalPairs;
  result.candidates.offset = candidateOffset;
  result.candidates.daughterOffset = daughterOffset;

  if (totalPairs == 0u) {
    const KFParticleGpuCandidatePoolStatus status = fImpl->fBuffers->DownloadCandidateStatus();
    result.candidates.size = status.candidates - candidateOffset;
    result.candidates.daughterSize = status.daughters - daughterOffset;
    result.candidates.overflowFlags = status.overflowFlags;
    result.constructedCandidates = result.candidates.size;
    result.constructedDaughters = result.candidates.daughterSize;
    result.generationCandidates = result.candidates;
    return result;
  }

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  fImpl->fQueue.launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
    xpu::n_threads(totalPairs),
    fImpl->fBuffers->DeviceEvents(),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    storage.fTwoDaughterTaskCount.get(),
    storage.fTwoDaughterTotalPairCount.get());

  fImpl->fQueue.launch<KFParticleGpuTwoDaughterCompactCandidatePoolKernel>(
      xpu::n_threads(taskCapacity),
      MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
      storage.fTwoDaughterTasks.get(),
      taskCapacity,
      storage.fTwoDaughterTaskCount.get(),
      fImpl->fBuffers->DeviceCandidates());

  const KFParticleGpuTwoDaughterTaskStatus taskStatus =
    fImpl->fBuffers->DownloadTwoDaughterTaskStatus();
  result.acceptedTasks = taskStatus.accepted;
  result.storedTasks = taskStatus.accepted < taskCapacity ? taskStatus.accepted : taskCapacity;
  if (taskStatus.accepted > taskCapacity) {
    fImpl->fBuffers->MarkCandidateOverflow(static_cast<unsigned int>(CandidateCapacityExceeded));
  }
  const KFParticleGpuCandidatePoolStatus status = fImpl->fBuffers->DownloadCandidateStatus();
  result.candidates.size = status.candidates - candidateOffset;
  result.candidates.daughterSize = status.daughters - daughterOffset;
  result.candidates.overflowFlags = status.overflowFlags;
  result.constructedCandidates = result.candidates.size;
  result.constructedDaughters = result.candidates.daughterSize;
  result.generationCandidates = result.candidates;
  return result;
}

KFParticleGpuV0TrackChannelResult KFParticleGpuSteering::RunV0TrackCompactChannel(
  const KFParticleGpuV0TrackCascadeChannel& channel,
  unsigned int eventIndex,
  unsigned int taskCapacity,
  unsigned int statusIndex)
{
  const KFParticleGpuEventDesc& event = fImpl->fBuffers->HostEvents()[eventIndex];
  const KFParticleGpuV0TrackChannel source = MakeV0TrackTaskChannel(channel, event, eventIndex);
  const unsigned int selectedCapacity = fImpl->fBuffers->Capacities().selectedCandidates;
  const unsigned int launchCapacity = selectedCapacity * source.bachelorTracks.size;
  KFParticleGpuV0TrackChannelResult result;
  result.channelId = channel.channelId;
  result.motherPdg = channel.motherPdg;
  result.eventIndex = eventIndex;
  result.totalPairs = 0u;
  if (launchCapacity == 0u) {
    return result;
  }
  if (statusIndex >= fImpl->fCascadeStatusSnapshots.size()) {
    throw std::logic_error("KFParticle GPU cascade status index is out of range");
  }

  fImpl->fBuffers->EnsureV0TrackTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetV0TrackTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  const KFParticleGpuSelectedCandidateRange selectedRange{0u, selectedCapacity, 0u};
  fImpl->fQueue.launch<KFParticleGpuGenerateV0TrackTasksCompact>(
    xpu::n_threads(launchCapacity),
    MakeConstView(fImpl->fBuffers->DeviceCandidates()),
    MakeConstView(fImpl->fBuffers->DeviceSelectedCandidates()),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    selectedRange,
    storage.fV0TrackTasks.get(),
    taskCapacity,
    storage.fV0TrackTaskCount.get(),
    storage.fV0TrackTotalPairCount.get(),
    storage.fV0TrackTaskOverflowFlags.get());
  fImpl->fQueue.launch<KFParticleGpuV0TrackCompactCandidatePoolKernel>(
    xpu::n_threads(taskCapacity),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    MakeConstView(fImpl->fBuffers->DeviceCandidates()),
    storage.fV0TrackTasks.get(),
    taskCapacity,
    storage.fV0TrackTaskCount.get(),
    fImpl->fBuffers->DeviceCandidates());

  KFParticleGpuCascadeStatusSnapshot& snapshot = fImpl->fCascadeStatusSnapshots[statusIndex];
  // These asynchronous copies remain ordered after the channel kernels. The
  // steering loop performs one wait after all channels, not one per channel.
  fImpl->fQueue.memcpy(&snapshot.acceptedTasks, storage.fV0TrackTaskCount.get(), sizeof(unsigned int));
  fImpl->fQueue.memcpy(&snapshot.totalPairs, storage.fV0TrackTotalPairCount.get(), sizeof(unsigned int));
  fImpl->fQueue.memcpy(&snapshot.taskOverflowFlags,
                        storage.fV0TrackTaskOverflowFlags.get(), sizeof(unsigned int));
  fImpl->fQueue.memcpy(&snapshot.candidates, fImpl->fBuffers->DeviceCandidates().SizeData(),
                        sizeof(unsigned int));
  fImpl->fQueue.memcpy(&snapshot.daughters, fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
                        sizeof(unsigned int));
  fImpl->fQueue.memcpy(&snapshot.candidateOverflowFlags,
                        fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(), sizeof(unsigned int));
  snapshot.copied = true;
  return result;
}

KFParticleGpuTwoDaughterFusedResult
KFParticleGpuSteering::RunTwoDaughterFusedStage(
  unsigned int eventIndex,
  unsigned int taskCapacity,
  KFParticleGpuTwoDaughterRoutingMode routingMode)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range(
      "KFParticle GPU fused two-daughter event index is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument(
      "KFParticle GPU fused two-daughter task capacity must be nonzero");
  }
  if (routingMode != KFGpuTwoDaughterRoutingAtomic
      && routingMode != KFGpuTwoDaughterRoutingBlockScan) {
    throw std::invalid_argument(
      "KFParticle GPU fused two-daughter routing mode is invalid");
  }

  const bool planChanged =
    fImpl->fTwoDaughterRoutingPlan.SourceRevision()
      != fImpl->fDecayPlan->Revision();
  if (planChanged) {
    fImpl->fTwoDaughterRoutingPlan.Compile(*fImpl->fDecayPlan);
  }
  fImpl->fBuffers->UploadTwoDaughterRoutingPlan(
    fImpl->fTwoDaughterRoutingPlan, planChanged);
  fImpl->fBuffers->EnsureTwoDaughterRoutedTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterRoutingStatus();
  fImpl->PublishKernelState();

  const KFParticleGpuCandidatePoolStatus initialCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  const KFParticleGpuEventDesc& event =
    fImpl->fBuffers->HostEvents()[eventIndex];

  KFParticleGpuTwoDaughterFusedResult result;
  result.eventIndex = eventIndex;
  result.descriptorCount = static_cast<unsigned int>(
    fImpl->fTwoDaughterRoutingPlan.Descriptors().size());
  for (const KFParticleGpuTwoDaughterExecutionGroup& group :
       fImpl->fTwoDaughterRoutingPlan.Groups()) {
    const KFParticleGpuRange firstTracks =
      event.TrackSet(group.firstTrackSet).tracks;
    const KFParticleGpuRange secondTracks =
      event.TrackSet(group.secondTrackSet).tracks;
    const unsigned int pairCount = firstTracks.size * secondTracks.size;
    if (pairCount == 0u) {
      continue;
    }
    ++result.groupLaunches;
    if (routingMode == KFGpuTwoDaughterRoutingAtomic) {
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksAtomic>(
        xpu::n_threads(pairCount),
        MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
        fImpl->fBuffers->DeviceEvents(),
        fImpl->fBuffers->DeviceTwoDaughterRouting(),
        eventIndex,
        group.groupIndex,
        storage.fTwoDaughterRoutedTasks.get(),
        taskCapacity,
        storage.fTwoDaughterRoutingVisitedPairCount.get(),
        storage.fTwoDaughterRoutingActiveBitCount.get(),
        storage.fTwoDaughterRoutedAcceptedTaskCount.get(),
        storage.fTwoDaughterRoutedStoredTaskCount.get(),
        storage.fTwoDaughterRoutingBlockReservationCount.get(),
        storage.fTwoDaughterRoutedTaskOverflowFlags.get());
    }
    else {
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        taskCapacity);
    }
  }

  // The routed pool and its device counter are consumed in queue order. No
  // task-count readback or host wait separates routing from construction.
  fImpl->fQueue.launch<KFParticleGpuTwoDaughterRoutedCandidatePoolKernel>(
    xpu::n_threads(taskCapacity),
    taskCapacity);

  result.routing = fImpl->fBuffers->DownloadTwoDaughterRoutingStatus();
  const KFParticleGpuCandidatePoolStatus finalCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  fImpl->fBuffers->DownloadTwoDaughterRoutedTasks();
  fImpl->fBuffers->DownloadCandidates();
  fImpl->fBuffers->DownloadCandidateDescriptorIndices();
  result.candidates.offset = initialCandidates.candidates;
  result.candidates.size =
    finalCandidates.candidates - initialCandidates.candidates;
  result.candidates.daughterOffset = initialCandidates.daughters;
  result.candidates.daughterSize =
    finalCandidates.daughters - initialCandidates.daughters;
  result.candidates.overflowFlags =
    finalCandidates.overflowFlags | result.routing.overflowFlags;
  return result;
}

KFParticleGpuV0TrackFusedResult KFParticleGpuSteering::RunV0TrackFusedStage(
  unsigned int eventIndex,
  const KFParticleGpuSelectedCandidateRange& selectedRange,
  unsigned int taskCapacity,
  KFParticleGpuV0TrackRoutingMode routingMode)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU fused V0-track event index is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU fused V0-track task capacity must be nonzero");
  }
  if (routingMode != KFGpuV0TrackRoutingAtomic
      && routingMode != KFGpuV0TrackRoutingBlockScan) {
    throw std::invalid_argument("KFParticle GPU fused V0-track routing mode is invalid");
  }
  const unsigned int selectedSize = fImpl->fBuffers->HostSelectedCandidates().Size();
  if (selectedRange.offset > selectedSize
      || selectedRange.size > selectedSize - selectedRange.offset) {
    throw std::out_of_range("KFParticle GPU fused V0-track selected range is out of range");
  }

  const bool planChanged =
    fImpl->fV0TrackRoutingPlan.SourceRevision() != fImpl->fDecayPlan->Revision();
  if (planChanged) {
    fImpl->fV0TrackRoutingPlan.Compile(*fImpl->fDecayPlan);
  }
  fImpl->fBuffers->UploadV0TrackRoutingPlan(fImpl->fV0TrackRoutingPlan, planChanged);
  fImpl->fBuffers->EnsureV0TrackRoutedTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetV0TrackRoutingStatus();
  fImpl->PublishKernelState();

  const KFParticleGpuCandidatePoolStatus initialCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  const KFParticleGpuEventDesc& event = fImpl->fBuffers->HostEvents()[eventIndex];
  for (const KFParticleGpuV0TrackExecutionGroup& group :
       fImpl->fV0TrackRoutingPlan.Groups()) {
    const KFParticleGpuRange bachelors = event.TrackSet(group.bachelorTrackSet).tracks;
    const unsigned int pairCount = selectedRange.size * bachelors.size;
    if (pairCount == 0u) {
      continue;
    }
    if (routingMode == KFGpuV0TrackRoutingAtomic) {
      fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksAtomic>(
        xpu::n_threads(pairCount),
        MakeConstView(fImpl->fBuffers->DeviceCandidates()),
        MakeConstView(fImpl->fBuffers->DeviceSelectedCandidates()),
        MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
        fImpl->fBuffers->DeviceEvents(),
        fImpl->fBuffers->DeviceV0TrackRouting(),
        eventIndex,
        group.groupIndex,
        selectedRange,
        storage.fV0TrackRoutedTasks.get(),
        taskCapacity,
        storage.fV0TrackRoutingVisitedPairCount.get(),
        storage.fV0TrackRoutingActiveBitCount.get(),
        storage.fV0TrackRoutedAcceptedTaskCount.get(),
        storage.fV0TrackRoutedStoredTaskCount.get(),
        storage.fV0TrackRoutingBlockReservationCount.get(),
        storage.fV0TrackRoutedTaskOverflowFlags.get());
    }
    else {
      fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        selectedRange,
        taskCapacity);
    }
  }

  // Queue ordering is the hand-off: construction reads the device counter and
  // routed pool directly, with no scalar D2H synchronization between kernels.
  fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
    xpu::n_threads(taskCapacity),
    taskCapacity);

  KFParticleGpuV0TrackFusedResult result;
  result.eventIndex = eventIndex;
  result.routing = fImpl->fBuffers->DownloadV0TrackRoutingStatus();
  const KFParticleGpuCandidatePoolStatus finalCandidates =
    fImpl->fBuffers->DownloadCandidateStatus();
  fImpl->fBuffers->DownloadV0TrackRoutedTasks();
  fImpl->fBuffers->DownloadCandidates();
  result.candidates.offset = initialCandidates.candidates;
  result.candidates.size = finalCandidates.candidates - initialCandidates.candidates;
  result.candidates.daughterOffset = initialCandidates.daughters;
  result.candidates.daughterSize = finalCandidates.daughters - initialCandidates.daughters;
  result.candidates.overflowFlags =
    finalCandidates.overflowFlags | result.routing.overflowFlags;
  return result;
}

KFParticleGpuSelectedCandidateRange KFParticleGpuSteering::RunV0Selection(
  const KFParticleGpuTwoDaughterChannel& channel,
  const KFParticleGpuCandidateRange& rawCandidates,
  unsigned int eventIndex)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventIndex >= fImpl->fBuffers->EventSize()) {
    throw std::out_of_range("KFParticle GPU V0 selection event index is out of range");
  }
  if (rawCandidates.offset > fImpl->fBuffers->Capacities().candidates
      || rawCandidates.size > fImpl->fBuffers->Capacities().candidates - rawCandidates.offset) {
    throw std::out_of_range("KFParticle GPU V0 selection raw candidate range is out of range");
  }
  if (fImpl->fBuffers->Capacities().selectedCandidates == 0u) {
    throw std::logic_error("KFParticle GPU selected-candidate capacity is zero");
  }

  fImpl->fBuffers->ResetSelectedCandidates();
  fImpl->fBuffers->ResetV0SelectionResults();
  if (rawCandidates.size > 0u) {
    fImpl->fQueue.launch<KFParticleGpuSelectV0Candidates>(
      xpu::n_threads(rawCandidates.size),
      MakeConstView(fImpl->fBuffers->DeviceCandidates()),
      MakeConstView(fImpl->fBuffers->DevicePrimaryVertices()),
      fImpl->fBuffers->DeviceEvents(),
      eventIndex,
      rawCandidates.offset,
      rawCandidates.size,
      channel.channelId,
      channel.selection,
      fImpl->fBuffers->DeviceV0SelectionResults(),
      fImpl->fBuffers->DeviceSelectedCandidates());
    fImpl->fQueue.wait();
  }
  fImpl->fBuffers->DownloadSelectedCandidates();
  fImpl->fBuffers->DownloadV0SelectionResults();

  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
  KFParticleGpuSelectedCandidateRange result;
  result.offset = 0u;
  result.size = selected.Size();
  result.overflowFlags = selected.OverflowFlags();
  return result;
}

const std::vector<KFParticleGpuTwoDaughterChannelResult>& KFParticleGpuSteering::RunDecayPlan(
  unsigned int eventIndex,
  unsigned int taskCapacity)
{
  return RunDecayPlanBatch(eventIndex, 1u, taskCapacity);
}

const std::vector<KFParticleGpuTwoDaughterChannelResult>& KFParticleGpuSteering::RunDecayPlanBatch(
  unsigned int firstEventIndex,
  unsigned int eventCount,
  unsigned int taskCapacity)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  if (eventCount == 0u) {
    throw std::invalid_argument("KFParticle GPU decay-plan batch must contain at least one event");
  }
  if (firstEventIndex >= fImpl->fBuffers->EventSize()
      || eventCount > fImpl->fBuffers->EventSize() - firstEventIndex) {
    throw std::out_of_range("KFParticle GPU decay-plan batch event range is out of range");
  }
  if (taskCapacity == 0u) {
    throw std::invalid_argument("KFParticle GPU decay-plan task capacity must be positive");
  }

  const std::uint64_t capacityGrowthCountBefore =
    fImpl->fBuffers->CapacityGrowthCount();
  unsigned int planUploadWaits = 0u;
  fImpl->fLastPerformanceSnapshot = KFParticleGpuPerformanceSnapshot();
  fImpl->fLastPerformanceSnapshot.enabled = fImpl->fPerformanceMonitoringEnabled;

  fImpl->fLastDecayPlanResults.clear();
  fImpl->fLastV0TrackCascadeResults.clear();
  fImpl->fCascadeStatusSnapshots.clear();
  fImpl->fFusedTwoDaughterSnapshots.clear();
  fImpl->fFusedCascadeSnapshots.clear();
  fImpl->fLastDecayPlanEventResults.clear();
  fImpl->fLastDecayPlanTiming = KFParticleGpuDecayPlanTiming();
  fImpl->fLastTwoDaughterRoutingMonitorData =
    KFParticleGpuTwoDaughterRoutingMonitorData();
  fImpl->fLastV0TrackRoutingMonitorData = KFParticleGpuV0TrackRoutingMonitorData();
  fImpl->fLastGraphExecutionMonitorData =
    KFParticleGpuGraphExecutionMonitorData();
  fImpl->fLastGraphChannelMonitorData.clear();
  fImpl->fGraphOperationSnapshots.clear();
  fImpl->fLastDecayPlanSelectedCandidates = KFParticleGpuSelectedCandidateRange();
  fImpl->fLastDecayPlanSelectedChannels.clear();
  const auto elapsedMilliseconds = [](std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  };
  const auto inputUploadStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->UploadInput();
  fImpl->fLastDecayPlanTiming.inputUploadMilliseconds = elapsedMilliseconds(inputUploadStarted);
  const auto constructionStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->ResetCandidates();
  fImpl->fBuffers->ResetV0SelectionResults();
  fImpl->fBuffers->ResetSelectedCandidates();
  if (fImpl->fDecayGraphPlanSourceRevision != fImpl->fDecayPlan->Revision()) {
    fImpl->fDecayGraphPlan.Compile(
      MakeDefaultCpuFinderDecayGraphManifest(*fImpl->fDecayPlan));
    fImpl->fDecayGraphPlanSourceRevision = fImpl->fDecayPlan->Revision();
  }
  planUploadWaits += fImpl->fBuffers->UploadDecayGraphPlan(fImpl->fDecayGraphPlan) ? 1u : 0u;
  planUploadWaits += fImpl->fBuffers->UploadGraphOperationPlan(*fImpl->fDecayPlan) ? 1u : 0u;
  if (fImpl->fDecayPlan->Empty()) {
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult eventResult;
      eventResult.eventIndex = firstEventIndex + eventOffset;
      fImpl->fLastDecayPlanEventResults.push_back(eventResult);
    }
    fImpl->fLastDecayPlanTiming.constructionMilliseconds = elapsedMilliseconds(constructionStarted);
    const auto downloadStarted = std::chrono::steady_clock::now();
    fImpl->fBuffers->DownloadCandidates();
    fImpl->fLastDecayPlanTiming.outputDownloadMilliseconds = elapsedMilliseconds(downloadStarted);
    fImpl->CompletePerformanceSnapshot(
      eventCount, capacityGrowthCountBefore, planUploadWaits, 5u);
    return fImpl->fLastDecayPlanResults;
  }

  const unsigned int twoDaughterChannelCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadTwoDaughter);
  if (twoDaughterChannelCount
      != fImpl->fDecayPlan->NumberOfTwoDaughterChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and two-daughter payload tables disagree");
  }
  const unsigned int graphOperationCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadCompositeComposite)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadNeutralDaughter)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadUnaryFinal)
    + CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadBinaryFinal);
  if (graphOperationCount
      != fImpl->fDecayPlan->NumberOfGraphOperationChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and later-generation operation tables disagree");
  }
  if (graphOperationCount > 0u) {
    fImpl->fBuffers->EnsureGraphOperationTaskCapacity(taskCapacity);
  }
  unsigned int fusedTaskCapacity = 0u;
  if (twoDaughterChannelCount > 0u) {
    if (taskCapacity > std::numeric_limits<unsigned int>::max()
                         / twoDaughterChannelCount) {
      throw std::overflow_error(
        "KFParticle GPU fused two-daughter task capacity overflows");
    }
    fusedTaskCapacity = taskCapacity * twoDaughterChannelCount;
    const bool planChanged =
      fImpl->fTwoDaughterRoutingPlan.SourceRevision()
        != fImpl->fDecayPlan->Revision();
    if (planChanged) {
      fImpl->fTwoDaughterRoutingPlan.Compile(*fImpl->fDecayPlan);
    }
    planUploadWaits += fImpl->fBuffers->UploadTwoDaughterRoutingPlan(
                         fImpl->fTwoDaughterRoutingPlan, planChanged)
                         ? 1u : 0u;
    fImpl->fBuffers->EnsureTwoDaughterRoutedTaskCapacity(fusedTaskCapacity);
  }

  const unsigned int cascadeChannelCount =
    CountGraphPayload(fImpl->fDecayGraphPlan, KFGpuGraphPayloadCompositeTrack);
  if (cascadeChannelCount
      != fImpl->fDecayPlan->NumberOfV0TrackCascadeChannels()) {
    throw std::logic_error(
      "KFParticle GPU graph and composite-track payload tables disagree");
  }
  unsigned int fusedCascadeTaskCapacity = 0u;
  if (cascadeChannelCount > 0u) {
    if (taskCapacity > std::numeric_limits<unsigned int>::max()
                         / cascadeChannelCount) {
      throw std::overflow_error(
        "KFParticle GPU fused cascade task capacity overflows");
    }
    fusedCascadeTaskCapacity = taskCapacity * cascadeChannelCount;
    const bool planChanged =
      fImpl->fV0TrackRoutingPlan.SourceRevision()
        != fImpl->fDecayPlan->Revision();
    if (planChanged) {
      fImpl->fV0TrackRoutingPlan.Compile(*fImpl->fDecayPlan);
    }
    planUploadWaits += fImpl->fBuffers->UploadV0TrackRoutingPlan(
                         fImpl->fV0TrackRoutingPlan, planChanged)
                         ? 1u : 0u;
    fImpl->fBuffers->EnsureV0TrackRoutedTaskCapacity(
      fusedCascadeTaskCapacity);
  }

  // All revision-owned tables and transaction capacities are now stable.
  // Publish one coherent pointer/size snapshot for every state-backed action.
  fImpl->PublishKernelState();

  fImpl->fFusedTwoDaughterSnapshots.resize(eventCount);
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    const unsigned int eventIndex = firstEventIndex + eventOffset;
    KFParticleGpuDecayPlanEventResult eventResult;
    eventResult.eventIndex = eventIndex;
    eventResult.channelOffset = eventOffset * twoDaughterChannelCount;
    eventResult.channelCount = twoDaughterChannelCount;
    fImpl->fLastDecayPlanEventResults.push_back(eventResult);

    if (twoDaughterChannelCount == 0u) {
      continue;
    }
    KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
      fImpl->fFusedTwoDaughterSnapshots[eventOffset];
    snapshot.channelVisited.resize(twoDaughterChannelCount);
    snapshot.channelAccepted.resize(twoDaughterChannelCount);
    snapshot.channelStored.resize(twoDaughterChannelCount);
    snapshot.channelConstructed.resize(twoDaughterChannelCount);
    snapshot.selectionAccepted.resize(twoDaughterChannelCount);
    snapshot.selectionStored.resize(twoDaughterChannelCount);
    snapshot.selectionOffsets.resize(twoDaughterChannelCount);
    fImpl->fQueue.memcpy(
      &snapshot.candidateBegin,
      fImpl->fBuffers->DeviceCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.daughterBegin,
      fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedBegin,
      fImpl->fBuffers->DeviceSelectedCandidates().SizeData(),
      sizeof(unsigned int));

    fImpl->fQueue.launch<KFParticleGpuResetTwoDaughterGenerationState>(
      xpu::n_threads(twoDaughterChannelCount));

    const KFParticleGpuEventDesc& event =
      fImpl->fBuffers->HostEvents()[eventIndex];
    for (const KFParticleGpuTwoDaughterExecutionGroup& group :
         fImpl->fTwoDaughterRoutingPlan.Groups()) {
      const KFParticleGpuRange firstTracks =
        event.TrackSet(group.firstTrackSet).tracks;
      const KFParticleGpuRange secondTracks =
        event.TrackSet(group.secondTrackSet).tracks;
      if (secondTracks.size != 0u
          && firstTracks.size > std::numeric_limits<unsigned int>::max()
                                / secondTracks.size) {
        throw std::overflow_error(
          "KFParticle GPU fused two-daughter pair count overflows");
      }
      const unsigned int pairCount = firstTracks.size * secondTracks.size;
      if (pairCount == 0u) {
        continue;
      }
      ++snapshot.groupLaunches;
      fImpl->fQueue.launch<KFParticleGpuRouteTwoDaughterTasksBlockScan>(
        xpu::n_threads(pairCount),
        eventIndex,
        group.groupIndex,
        fusedTaskCapacity);
    }
    fImpl->fQueue.launch<KFParticleGpuTwoDaughterRoutedCandidatePoolKernel>(
      xpu::n_threads(fusedTaskCapacity),
      fusedTaskCapacity);

    const auto selectionStarted = std::chrono::steady_clock::now();
    fImpl->fQueue.launch<KFParticleGpuEvaluateTwoDaughterGenerationSelection>(
      xpu::n_threads(fImpl->fBuffers->Capacities().candidates),
      eventIndex);
    fImpl->fQueue.launch<KFParticleGpuPrepareTwoDaughterSelectionSegments>(
      xpu::n_threads(1u));
    fImpl->fQueue.launch<KFParticleGpuScatterTwoDaughterGenerationSelection>(
      xpu::n_threads(fImpl->fBuffers->Capacities().candidates),
      eventIndex);
    fImpl->fLastDecayPlanTiming.selectionMilliseconds +=
      elapsedMilliseconds(selectionStarted);
    ++fImpl->fLastTwoDaughterRoutingMonitorData.selectionLaunches;

    const KFParticleGpuTwoDaughterSelectionWorkspaceView selection =
      fImpl->fBuffers->DeviceTwoDaughterSelectionWorkspace();
    fImpl->fQueue.memcpy(
      &snapshot.visitedPairs,
      storage.fTwoDaughterRoutingVisitedPairCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.activeChannelBits,
      storage.fTwoDaughterRoutingActiveBitCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.acceptedTasks,
      storage.fTwoDaughterRoutedAcceptedTaskCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.storedTasks,
      storage.fTwoDaughterRoutedStoredTaskCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.blockReservations,
      storage.fTwoDaughterRoutingBlockReservationCount.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.taskOverflowFlags,
      storage.fTwoDaughterRoutedTaskOverflowFlags.get(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelVisited.data(),
      storage.fTwoDaughterRoutingChannelVisitedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelAccepted.data(),
      storage.fTwoDaughterRoutingChannelAcceptedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelStored.data(),
      storage.fTwoDaughterRoutingChannelStoredCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.channelConstructed.data(),
      storage.fTwoDaughterRoutingChannelConstructedCounters.get(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionAccepted.data(),
      selection.AcceptedData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionStored.data(),
      selection.StoredData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      snapshot.selectionOffsets.data(),
      selection.OffsetsData(),
      twoDaughterChannelCount * sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.candidateEnd,
      fImpl->fBuffers->DeviceCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.daughterEnd,
      fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedEnd,
      fImpl->fBuffers->DeviceSelectedCandidates().SizeData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.candidateOverflowFlags,
      fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
      sizeof(unsigned int));
    fImpl->fQueue.memcpy(
      &snapshot.selectedOverflowFlags,
      fImpl->fBuffers->DeviceSelectedCandidates().OverflowFlagsData(),
      sizeof(unsigned int));
  }
  fImpl->fLastDecayPlanTiming.constructionMilliseconds =
    elapsedMilliseconds(constructionStarted)
    - fImpl->fLastDecayPlanTiming.selectionMilliseconds;

  const auto cascadeStarted = std::chrono::steady_clock::now();
  const auto queueGraphOperations = [&]() {
    if (graphOperationCount == 0u) {
      return;
    }
    unsigned int operationGroupCount = 0u;
    for (const KFParticleGpuGraphExecutionGroup& group :
         fImpl->fDecayGraphPlan.Groups()) {
      if (group.topology == KFGpuGraphTopologyCompositeComposite
          || group.topology == KFGpuGraphTopologyNeutralDaughter
          || group.topology == KFGpuGraphTopologyUnaryComposite
          || ((group.topology == KFGpuGraphTopologyCompositeTrack
               || group.topology == KFGpuGraphTopologyTrackComposite)
              && (group.operationMask & KFGpuGraphMatch) != 0u)) {
        ++operationGroupCount;
      }
    }
    fImpl->fGraphOperationSnapshots.resize(eventCount * operationGroupCount);
    unsigned int snapshotIndex = 0u;
    const unsigned int descriptorCount =
      fImpl->fBuffers->GraphOperationDescriptorSize();
    const unsigned int candidateCapacity =
      fImpl->fBuffers->Capacities().candidates;
    const unsigned int sourceCapacity = std::max(
      candidateCapacity, fImpl->fBuffers->Capacities().tracks);
    const KFParticleGpuDeviceStorage& graphStorage =
      fImpl->fBuffers->Storage();
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      const unsigned int eventIndex = firstEventIndex + eventOffset;
      for (const KFParticleGpuGraphExecutionGroup& group :
           fImpl->fDecayGraphPlan.Groups()) {
        if (group.topology != KFGpuGraphTopologyCompositeComposite
            && group.topology != KFGpuGraphTopologyNeutralDaughter
            && group.topology != KFGpuGraphTopologyUnaryComposite
            && !((group.topology == KFGpuGraphTopologyCompositeTrack
                  || group.topology == KFGpuGraphTopologyTrackComposite)
                 && (group.operationMask & KFGpuGraphMatch) != 0u)) {
          continue;
        }
        if (group.nodeCount > 0u
            && sourceCapacity
                 > std::numeric_limits<unsigned int>::max() / group.nodeCount) {
          throw std::overflow_error(
            "KFParticle GPU graph routing launch size overflows");
        }
        KFParticleGpuGraphOperationSnapshot& snapshot =
          fImpl->fGraphOperationSnapshots[snapshotIndex++];
        snapshot.eventIndex = eventIndex;
        snapshot.groupIndex = group.groupIndex;
        snapshot.generation = group.generation;
        snapshot.channelVisited.resize(descriptorCount);
        snapshot.channelAccepted.resize(descriptorCount);
        snapshot.channelStored.resize(descriptorCount);
        snapshot.channelConstructed.resize(descriptorCount);
        snapshot.channelRejected.resize(descriptorCount);

        fImpl->fQueue.memcpy(
          &snapshot.candidateBegin,
          fImpl->fBuffers->DeviceCandidates().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.daughterBegin,
          fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.launch<KFParticleGpuResetGraphOperationGeneration>(
          xpu::n_threads(descriptorCount > 0u ? descriptorCount : 1u));
        fImpl->fQueue.launch<KFParticleGpuRouteGraphOperationTasks>(
          xpu::n_threads(group.nodeCount * sourceCapacity),
          eventIndex,
          group.groupIndex,
          sourceCapacity,
          taskCapacity);
        fImpl->fQueue.launch<KFParticleGpuExecuteRoutedGraphOperations>(
          xpu::n_threads(taskCapacity),
          taskCapacity);

        fImpl->fQueue.memcpy(
          &snapshot.visited,
          graphStorage.fGraphOperationVisitedCombinations.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.accepted,
          graphStorage.fGraphOperationAcceptedTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.stored,
          graphStorage.fGraphOperationStoredTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.constructed,
          graphStorage.fGraphOperationConstructedCandidates.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.rejected,
          graphStorage.fGraphOperationRejectedTasks.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.overflowFlags,
          graphStorage.fGraphOperationOverflowFlags.get(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelVisited.data(),
          graphStorage.fGraphOperationChannelVisitedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelAccepted.data(),
          graphStorage.fGraphOperationChannelAcceptedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelStored.data(),
          graphStorage.fGraphOperationChannelStoredCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelConstructed.data(),
          graphStorage.fGraphOperationChannelConstructedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          snapshot.channelRejected.data(),
          graphStorage.fGraphOperationChannelRejectedCounters.get(),
          descriptorCount * sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.candidateEnd,
          fImpl->fBuffers->DeviceCandidates().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.daughterEnd,
          fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
          sizeof(unsigned int));
        fImpl->fQueue.memcpy(
          &snapshot.candidateOverflowFlags,
          fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
          sizeof(unsigned int));
      }
    }
  };
  if (cascadeChannelCount > 0u) {
    fImpl->fFusedCascadeSnapshots.resize(eventCount);
    const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();
    const KFParticleGpuSelectedCandidateRange selectedRange{
      0u, fImpl->fBuffers->Capacities().selectedCandidates, 0u};

    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult& eventResult =
        fImpl->fLastDecayPlanEventResults[eventOffset];
      KFParticleGpuFusedCascadeSnapshot& snapshot =
        fImpl->fFusedCascadeSnapshots[eventOffset];
      snapshot.channelVisited.resize(cascadeChannelCount);
      snapshot.channelAccepted.resize(cascadeChannelCount);
      snapshot.channelStored.resize(cascadeChannelCount);
      snapshot.channelConstructed.resize(cascadeChannelCount);
      fImpl->fQueue.memcpy(
        &snapshot.candidateBegin,
        fImpl->fBuffers->DeviceCandidates().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.daughterBegin,
        fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
        sizeof(unsigned int));
      const KFParticleGpuEventDesc& event =
        fImpl->fBuffers->HostEvents()[eventResult.eventIndex];
      unsigned int activeGeneration = 0u;
      bool generationQueued = false;
      for (const KFParticleGpuV0TrackExecutionGroup& group :
           fImpl->fV0TrackRoutingPlan.Groups()) {
        if (group.generation != activeGeneration) {
          if (generationQueued) {
            fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
              xpu::n_threads(fusedCascadeTaskCapacity),
              fusedCascadeTaskCapacity);
          }
          fImpl->fQueue.launch<KFParticleGpuResetV0TrackGenerationState>(
            xpu::n_threads(cascadeChannelCount),
            activeGeneration == 0u ? 1u : 0u);
          activeGeneration = group.generation;
          generationQueued = false;
        }
        const KFParticleGpuRange bachelors =
          event.TrackSet(group.bachelorTrackSet).tracks;
        if (selectedRange.size != 0u
            && bachelors.size > std::numeric_limits<unsigned int>::max()
                                  / selectedRange.size) {
          throw std::overflow_error("KFParticle GPU fused cascade pair count overflows");
        }
        const unsigned int pairCount = selectedRange.size * bachelors.size;
        if (pairCount == 0u) {
          continue;
        }
        ++snapshot.groupLaunches;
        fImpl->fQueue.launch<KFParticleGpuRouteV0TrackTasksBlockScan>(
          xpu::n_threads(pairCount),
          eventResult.eventIndex,
          group.groupIndex,
          selectedRange,
          fusedCascadeTaskCapacity);
        generationQueued = true;
      }
      if (generationQueued) {
        fImpl->fQueue.launch<KFParticleGpuV0TrackRoutedCandidatePoolKernel>(
          xpu::n_threads(fusedCascadeTaskCapacity),
          fusedCascadeTaskCapacity);
      }

      fImpl->fQueue.memcpy(&snapshot.visitedPairs,
                            storage.fV0TrackRoutingVisitedPairCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.activeChannelBits,
                            storage.fV0TrackRoutingActiveBitCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.acceptedTasks,
                            storage.fV0TrackRoutedAcceptedTaskCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.storedTasks,
                            storage.fV0TrackRoutedStoredTaskCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.blockReservations,
                            storage.fV0TrackRoutingBlockReservationCount.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(&snapshot.taskOverflowFlags,
                            storage.fV0TrackRoutedTaskOverflowFlags.get(),
                            sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelVisited.data(),
                            storage.fV0TrackRoutingChannelVisitedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelAccepted.data(),
                            storage.fV0TrackRoutingChannelAcceptedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelStored.data(),
                            storage.fV0TrackRoutingChannelStoredCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(snapshot.channelConstructed.data(),
                            storage.fV0TrackRoutingChannelConstructedCounters.get(),
                            cascadeChannelCount * sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.candidateEnd,
        fImpl->fBuffers->DeviceCandidates().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.daughterEnd,
        fImpl->fBuffers->DeviceCandidates().Daughters().SizeData(),
        sizeof(unsigned int));
      fImpl->fQueue.memcpy(
        &snapshot.candidateOverflowFlags,
        fImpl->fBuffers->DeviceCandidates().OverflowFlagsData(),
        sizeof(unsigned int));
    }
    queueGraphOperations();
    // All first-generation, selection, cascade, and later graph actions are queued.
    // Resolve every asynchronous snapshot only after this single transaction
    // boundary so HIP never observes host counters before their D2H copies.
    fImpl->fQueue.wait();

    fImpl->fLastV0TrackRoutingMonitorData.descriptorCount = cascadeChannelCount;
    for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
      KFParticleGpuDecayPlanEventResult& eventResult =
        fImpl->fLastDecayPlanEventResults[eventOffset];
      const KFParticleGpuFusedCascadeSnapshot& snapshot =
        fImpl->fFusedCascadeSnapshots[eventOffset];
      eventResult.cascadeChannelOffset =
        static_cast<unsigned int>(fImpl->fLastV0TrackCascadeResults.size());
      eventResult.cascadeChannelCount = cascadeChannelCount;
      eventResult.cascadeCandidates.offset = snapshot.candidateBegin;
      eventResult.cascadeCandidates.size =
        snapshot.candidateEnd - snapshot.candidateBegin;
      eventResult.cascadeCandidates.daughterOffset = snapshot.daughterBegin;
      eventResult.cascadeCandidates.daughterSize =
        snapshot.daughterEnd - snapshot.daughterBegin;
      eventResult.cascadeCandidates.overflowFlags =
        snapshot.candidateOverflowFlags | snapshot.taskOverflowFlags;
      eventResult.cascadeRouting.visitedPairs = snapshot.visitedPairs;
      eventResult.cascadeRouting.activeChannelBits = snapshot.activeChannelBits;
      eventResult.cascadeRouting.acceptedTasks = snapshot.acceptedTasks;
      eventResult.cascadeRouting.storedTasks = snapshot.storedTasks;
      eventResult.cascadeRouting.blockReservations = snapshot.blockReservations;
      eventResult.cascadeRouting.overflowFlags = snapshot.taskOverflowFlags;
      eventResult.overflowFlags |= eventResult.cascadeCandidates.overflowFlags;

      for (unsigned int channelIndex = 0u;
           channelIndex < cascadeChannelCount;
           ++channelIndex) {
        const KFParticleGpuV0TrackCascadeChannel& channel =
          fImpl->fDecayPlan->V0TrackCascadeChannel(channelIndex);
        KFParticleGpuV0TrackChannelResult result;
        result.channelId = channel.channelId;
        result.motherPdg = channel.motherPdg;
        result.eventIndex = eventResult.eventIndex;
        result.totalPairs = snapshot.channelVisited[channelIndex];
        result.acceptedTasks = snapshot.channelAccepted[channelIndex];
        result.storedTasks = snapshot.channelStored[channelIndex];
        result.constructedCandidates = snapshot.channelConstructed[channelIndex];
        result.constructedDaughters =
          (channel.generation + 1u) * result.constructedCandidates;
        result.generationCandidates = eventResult.cascadeCandidates;
        result.candidates.overflowFlags = eventResult.cascadeCandidates.overflowFlags;
        fImpl->fLastV0TrackCascadeResults.push_back(result);
      }

      eventResult.cascadeRouting.visitedPairs = 0u;
      eventResult.cascadeRouting.acceptedTasks = 0u;
      eventResult.cascadeRouting.storedTasks = 0u;
      for (unsigned int channelIndex = 0u;
           channelIndex < cascadeChannelCount;
           ++channelIndex) {
        eventResult.cascadeRouting.visitedPairs += snapshot.channelVisited[channelIndex];
        eventResult.cascadeRouting.acceptedTasks += snapshot.channelAccepted[channelIndex];
        eventResult.cascadeRouting.storedTasks += snapshot.channelStored[channelIndex];
      }

      KFParticleGpuV0TrackRoutingMonitorData& monitoring =
        fImpl->fLastV0TrackRoutingMonitorData;
      monitoring.groupLaunches += snapshot.groupLaunches;
      monitoring.visitedPairs += eventResult.cascadeRouting.visitedPairs;
      monitoring.activeChannelBits += snapshot.activeChannelBits;
      monitoring.acceptedTasks += eventResult.cascadeRouting.acceptedTasks;
      monitoring.storedTasks += eventResult.cascadeRouting.storedTasks;
      monitoring.blockReservations += snapshot.blockReservations;
      monitoring.candidates += eventResult.cascadeCandidates.size;
      monitoring.daughters += eventResult.cascadeCandidates.daughterSize;
      monitoring.overflowFlags |= eventResult.cascadeCandidates.overflowFlags;
    }
  }
  else {
    queueGraphOperations();
    // Without a cascade generation this is still the sole synchronization
    // between route/construct/select submission and host result assembly.
    fImpl->fQueue.wait();
  }

  KFParticleGpuGraphExecutionMonitorData& graphMonitorData =
    fImpl->fLastGraphExecutionMonitorData;
  graphMonitorData.descriptorCount = graphOperationCount;
  for (const KFParticleGpuGraphNode& node : fImpl->fDecayGraphPlan.Nodes()) {
    graphMonitorData.unsupportedNodes +=
      node.supportStatus == KFGpuGraphSupported ? 0u : 1u;
  }
  for (const KFParticleGpuGraphFamilyCoverage& family :
       fImpl->fDecayGraphPlan.FamilyCoverage()) {
    graphMonitorData.unsupportedFamilies +=
      family.supportStatus == KFGpuGraphSupported ? 0u : 1u;
  }
  fImpl->fLastGraphChannelMonitorData.resize(
    eventCount * graphOperationCount);
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    for (unsigned int descriptorIndex = 0u;
         descriptorIndex < graphOperationCount;
         ++descriptorIndex) {
      KFParticleGpuGraphChannelMonitorData& channel =
        fImpl->fLastGraphChannelMonitorData[
          eventOffset * graphOperationCount + descriptorIndex];
      const KFParticleGpuGraphOperationChannel& configured =
        fImpl->fDecayPlan->GraphOperationChannel(descriptorIndex);
      channel.channelId = configured.descriptor.channelId;
      channel.eventIndex = firstEventIndex + eventOffset;
      channel.generation = configured.node.generation;
    }
  }
  for (const KFParticleGpuGraphOperationSnapshot& snapshot :
       fImpl->fGraphOperationSnapshots) {
    ++graphMonitorData.groupLaunches;
    graphMonitorData.visitedCombinations += snapshot.visited;
    graphMonitorData.acceptedTasks += snapshot.accepted;
    graphMonitorData.storedTasks += snapshot.stored;
    graphMonitorData.constructedCandidates += snapshot.constructed;
    graphMonitorData.rejectedTasks += snapshot.rejected;
    graphMonitorData.candidates += snapshot.candidateEnd - snapshot.candidateBegin;
    graphMonitorData.daughters += snapshot.daughterEnd - snapshot.daughterBegin;
    graphMonitorData.overflowFlags |=
      snapshot.overflowFlags | snapshot.candidateOverflowFlags;
    const unsigned int eventOffset = snapshot.eventIndex - firstEventIndex;
    KFParticleGpuDecayPlanEventResult& eventResult =
      fImpl->fLastDecayPlanEventResults[eventOffset];
    if (eventResult.graphCandidates.size == 0u) {
      eventResult.graphCandidates.offset = snapshot.candidateBegin;
      eventResult.graphCandidates.daughterOffset = snapshot.daughterBegin;
    }
    eventResult.graphCandidates.size +=
      snapshot.candidateEnd - snapshot.candidateBegin;
    eventResult.graphCandidates.daughterSize +=
      snapshot.daughterEnd - snapshot.daughterBegin;
    eventResult.graphCandidates.overflowFlags |=
      snapshot.overflowFlags | snapshot.candidateOverflowFlags;
    eventResult.overflowFlags |= eventResult.graphCandidates.overflowFlags;
    for (unsigned int descriptorIndex = 0u;
         descriptorIndex < graphOperationCount;
         ++descriptorIndex) {
      KFParticleGpuGraphChannelMonitorData& channel =
        fImpl->fLastGraphChannelMonitorData[
          eventOffset * graphOperationCount + descriptorIndex];
      channel.visitedCombinations += snapshot.channelVisited[descriptorIndex];
      channel.acceptedTasks += snapshot.channelAccepted[descriptorIndex];
      channel.storedTasks += snapshot.channelStored[descriptorIndex];
      channel.constructedCandidates +=
        snapshot.channelConstructed[descriptorIndex];
      channel.rejectedTasks += snapshot.channelRejected[descriptorIndex];
    }
  }

  fImpl->fLastTwoDaughterRoutingMonitorData.descriptorCount =
    twoDaughterChannelCount;
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    KFParticleGpuDecayPlanEventResult& eventResult =
      fImpl->fLastDecayPlanEventResults[eventOffset];
    if (twoDaughterChannelCount == 0u) {
      continue;
    }
    const KFParticleGpuFusedTwoDaughterSnapshot& snapshot =
      fImpl->fFusedTwoDaughterSnapshots[eventOffset];
    eventResult.candidates.offset = snapshot.candidateBegin;
    eventResult.candidates.size = snapshot.candidateEnd - snapshot.candidateBegin;
    eventResult.candidates.daughterOffset = snapshot.daughterBegin;
    eventResult.candidates.daughterSize =
      snapshot.daughterEnd - snapshot.daughterBegin;
    eventResult.candidates.overflowFlags =
      snapshot.candidateOverflowFlags | snapshot.taskOverflowFlags;
    eventResult.generationRouting.visitedPairs = snapshot.visitedPairs;
    eventResult.generationRouting.activeChannelBits = snapshot.activeChannelBits;
    eventResult.generationRouting.acceptedTasks = snapshot.acceptedTasks;
    eventResult.generationRouting.storedTasks = snapshot.storedTasks;
    eventResult.generationRouting.blockReservations = snapshot.blockReservations;
    eventResult.generationRouting.overflowFlags = snapshot.taskOverflowFlags;
    eventResult.selectedCandidates.offset = snapshot.selectedBegin;
    eventResult.selectedCandidates.size =
      snapshot.selectedEnd - snapshot.selectedBegin;
    eventResult.selectedCandidates.overflowFlags =
      snapshot.selectedOverflowFlags;
    eventResult.overflowFlags |= eventResult.candidates.overflowFlags
                                 | snapshot.selectedOverflowFlags;

    for (unsigned int channelIndex = 0u;
         channelIndex < twoDaughterChannelCount;
         ++channelIndex) {
      const KFParticleGpuTwoDaughterChannel& channel =
        fImpl->fDecayPlan->TwoDaughterChannel(channelIndex);
      KFParticleGpuTwoDaughterChannelResult result;
      result.channelId = channel.channelId;
      result.motherPdg = channel.motherPdg;
      result.eventIndex = eventResult.eventIndex;
      result.totalPairs = snapshot.channelVisited[channelIndex];
      result.acceptedTasks = snapshot.channelAccepted[channelIndex];
      result.storedTasks = snapshot.channelStored[channelIndex];
      result.constructedCandidates = snapshot.channelConstructed[channelIndex];
      result.constructedDaughters = 2u * result.constructedCandidates;
      result.generationCandidates = eventResult.candidates;
      result.candidates.overflowFlags = eventResult.candidates.overflowFlags;
      // A single-channel generation remains physically contiguous and can
      // retain the legacy range contract. Multi-channel generations use
      // stable metadata plus constructedCandidates.
      if (twoDaughterChannelCount == 1u) {
        result.candidates = eventResult.candidates;
      }
      fImpl->fLastDecayPlanResults.push_back(result);

      KFParticleGpuSelectedChannelRange selectedChannel;
      selectedChannel.channelId = channel.channelId;
      selectedChannel.eventIndex = eventResult.eventIndex;
      selectedChannel.candidates.offset =
        snapshot.selectionOffsets[channelIndex];
      selectedChannel.candidates.size =
        snapshot.selectionStored[channelIndex];
      selectedChannel.candidates.overflowFlags =
        snapshot.selectedOverflowFlags;
      fImpl->fLastDecayPlanSelectedChannels.push_back(selectedChannel);
    }

    KFParticleGpuTwoDaughterRoutingMonitorData& monitoring =
      fImpl->fLastTwoDaughterRoutingMonitorData;
    monitoring.groupLaunches += snapshot.groupLaunches;
    monitoring.visitedPairs += snapshot.visitedPairs;
    monitoring.activeChannelBits += snapshot.activeChannelBits;
    monitoring.acceptedTasks += snapshot.acceptedTasks;
    monitoring.storedTasks += snapshot.storedTasks;
    monitoring.blockReservations += snapshot.blockReservations;
    monitoring.candidates += eventResult.candidates.size;
    monitoring.daughters += eventResult.candidates.daughterSize;
    monitoring.selectedCandidates += eventResult.selectedCandidates.size;
    monitoring.overflowFlags |= eventResult.candidates.overflowFlags
                               | eventResult.selectedCandidates.overflowFlags;
  }
  fImpl->fLastDecayPlanTiming.cascadeConstructionMilliseconds = elapsedMilliseconds(cascadeStarted);
  const auto downloadStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->DownloadSelectedCandidates();
  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
  fImpl->fLastDecayPlanSelectedCandidates.offset = 0u;
  fImpl->fLastDecayPlanSelectedCandidates.size = selected.Size();
  fImpl->fLastDecayPlanSelectedCandidates.overflowFlags = selected.OverflowFlags();
  fImpl->fBuffers->DownloadCandidates();
  fImpl->fLastDecayPlanTiming.outputDownloadMilliseconds = elapsedMilliseconds(downloadStarted);
  fImpl->CompletePerformanceSnapshot(
    eventCount, capacityGrowthCountBefore, planUploadWaits, 7u);
  return fImpl->fLastDecayPlanResults;
}

const std::vector<KFParticleGpuTwoDaughterChannelResult>&
KFParticleGpuSteering::LastDecayPlanResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanResults;
}

const std::vector<KFParticleGpuV0TrackChannelResult>&
KFParticleGpuSteering::LastV0TrackCascadeResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastV0TrackCascadeResults;
}

const std::vector<KFParticleGpuDecayPlanEventResult>&
KFParticleGpuSteering::LastDecayPlanEventResults() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanEventResults;
}

const KFParticleGpuDecayPlanTiming& KFParticleGpuSteering::LastDecayPlanTiming() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanTiming;
}

const KFParticleGpuTwoDaughterRoutingMonitorData&
KFParticleGpuSteering::LastTwoDaughterRoutingMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastTwoDaughterRoutingMonitorData;
}

const KFParticleGpuV0TrackRoutingMonitorData&
KFParticleGpuSteering::LastV0TrackRoutingMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastV0TrackRoutingMonitorData;
}

const KFParticleGpuGraphExecutionMonitorData&
KFParticleGpuSteering::LastGraphExecutionMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastGraphExecutionMonitorData;
}

const std::vector<KFParticleGpuGraphChannelMonitorData>&
KFParticleGpuSteering::LastGraphChannelMonitorData() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastGraphChannelMonitorData;
}

void KFParticleGpuSteering::SetPerformanceMonitoringEnabled(bool enabled)
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  fImpl->fPerformanceMonitoringEnabled = enabled;
  if (!enabled) { fImpl->fLastPerformanceSnapshot = KFParticleGpuPerformanceSnapshot(); }
}

bool KFParticleGpuSteering::PerformanceMonitoringEnabled() const
{
  return fImpl->fInitialized && fImpl->fPerformanceMonitoringEnabled;
}

const KFParticleGpuPerformanceSnapshot&
KFParticleGpuSteering::LastPerformanceSnapshot() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastPerformanceSnapshot;
}

const KFParticleGpuSelectedCandidateRange& KFParticleGpuSteering::LastDecayPlanSelectedCandidates() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanSelectedCandidates;
}

const std::vector<KFParticleGpuSelectedChannelRange>&
KFParticleGpuSteering::LastDecayPlanSelectedChannels() const
{
  if (!fImpl->fInitialized) {
    throw std::logic_error("KFParticle GPU steering is not initialized");
  }
  return fImpl->fLastDecayPlanSelectedChannels;
}
#endif
