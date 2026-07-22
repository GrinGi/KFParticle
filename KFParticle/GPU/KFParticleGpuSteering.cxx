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
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuKernels.h"

#include <chrono>
#include <stdexcept>

#ifdef KFPARTICLE_GPU_TRACE
#include <iostream>
#endif

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

  void Trace(const char* stage)
  {
#ifdef KFPARTICLE_GPU_TRACE
    std::cerr << "[KFParticleGpuSteering] " << stage << std::endl;
#else
    (void) stage;
#endif
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

#ifndef KFPARTICLE_GPU_USE_KERNEL_ARGS
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
                                buffers.DeviceSelectedCandidates());
  }
#endif

  unsigned int CountTwoDaughterPairs(const KFParticleGpuEventDesc& event,
                                     const KFParticleGpuTwoDaughterTaskSource& source)
  {
    const KFParticleGpuRange firstRange =
      ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
    const KFParticleGpuRange secondRange =
      ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
    return firstRange.size * secondRange.size;
  }
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
  std::vector<KFParticleGpuDecayPlanEventResult> fLastDecayPlanEventResults;
  KFParticleGpuDecayPlanTiming fLastDecayPlanTiming;
  KFParticleGpuSelectedCandidateRange fLastDecayPlanSelectedCandidates;
  std::vector<KFParticleGpuSelectedChannelRange> fLastDecayPlanSelectedChannels;
#endif
  std::unique_ptr<KFParticleGpuDecayPlan> fDecayPlan;

#ifdef KFPARTICLE_USE_XPU
  explicit Impl(xpu::queue& queue)
    : fInitialized(false), fQueue(queue), fBuffers(), fKernels(), fLastDecayPlanResults(),
      fLastDecayPlanEventResults(),
      fLastDecayPlanTiming(), fLastDecayPlanSelectedCandidates(), fLastDecayPlanSelectedChannels(), fDecayPlan()
  {
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

  Trace("round-trip: upload input");
  fImpl->fBuffers->UploadInput();
  Trace("round-trip: reset candidates");
  fImpl->fBuffers->ResetCandidates();

  Trace("round-trip: prepare device views");
#ifdef KFPARTICLE_GPU_USE_KERNEL_ARGS
  const KFParticleGpuConstInputTrackSoAView inputTracks =
    MakeConstView(fImpl->fBuffers->DeviceInputTracks());
  const KFParticleGpuCandidatePoolView candidates = fImpl->fBuffers->DeviceCandidates();
  Trace("round-trip: using kernel argument views");
#else
  // Publish fresh views after any capacity or event-size change, including an
  // empty event for which no action is launched.
  fImpl->fKernels = MakeKernelState(*fImpl->fBuffers);
  Trace("round-trip: publish constant memory");
  xpu::set<TheKFParticleFinder>(fImpl->fKernels);
  Trace("round-trip: constant memory published");
#endif

  if (fImpl->fBuffers->TrackSize() > 0) {
    Trace("round-trip: launch kernel");
#ifdef KFPARTICLE_GPU_USE_KERNEL_ARGS
    fImpl->fQueue.launch<KFParticleGpuRoundTripArgs>(
      xpu::n_threads(fImpl->fBuffers->TrackSize()),
      inputTracks,
      candidates,
      mass,
      eventIndex);
#else
    fImpl->fQueue.launch<KFParticleGpuRoundTrip>(
      xpu::n_threads(fImpl->fBuffers->TrackSize()), mass, eventIndex);
#endif
    Trace("round-trip: wait kernel");
    fImpl->fQueue.wait();
    Trace("round-trip: kernel finished");
  }

  Trace("round-trip: download candidates");
  fImpl->fBuffers->DownloadCandidates();
  Trace("round-trip: candidates downloaded");
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

  Trace("two-daughter-stage: upload input");
  fImpl->fBuffers->UploadInput();
  Trace("two-daughter-stage: reset candidates");
  fImpl->fBuffers->ResetCandidates();

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  Trace("two-daughter-stage: generate tasks");
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

  Trace("two-daughter-stage: construct candidates");
  fImpl->fQueue.launch<KFParticleGpuTwoDaughterTaskKernel>(
    xpu::n_threads(taskCapacity),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    fImpl->fBuffers->DeviceCandidates());
  fImpl->fQueue.wait();

  Trace("two-daughter-stage: download candidates");
  fImpl->fBuffers->DownloadCandidates();
  Trace("two-daughter-stage: candidates downloaded");
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

  Trace("two-daughter-compact-stage: upload input");
  fImpl->fBuffers->UploadInput();
  Trace("two-daughter-compact-stage: reset candidates");
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
    Trace("two-daughter-compact-stage: empty pair range");
    const KFParticleGpuCandidatePoolStatus status = fImpl->fBuffers->DownloadCandidateStatus();
    result.candidates.size = status.candidates - candidateOffset;
    result.candidates.daughterSize = status.daughters - daughterOffset;
    result.candidates.overflowFlags = status.overflowFlags;
    return result;
  }

  fImpl->fBuffers->EnsureTwoDaughterTaskCapacity(taskCapacity);
  fImpl->fBuffers->ResetTwoDaughterTaskStatus();
  const KFParticleGpuDeviceStorage& storage = fImpl->fBuffers->Storage();

  Trace("two-daughter-compact-stage: generate compact tasks");
  fImpl->fQueue.launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
    xpu::n_threads(totalPairs),
    fImpl->fBuffers->DeviceEvents(),
    MakeConstView(fImpl->fBuffers->DeviceInputTracks()),
    source,
    storage.fTwoDaughterTasks.get(),
    taskCapacity,
    storage.fTwoDaughterTaskCount.get(),
    storage.fTwoDaughterTotalPairCount.get());

  Trace("two-daughter-compact-stage: construct persistent compact tasks");
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

  Trace("v0-selection: reset selected candidates");
  fImpl->fBuffers->ResetSelectedCandidates();
  fImpl->fBuffers->ResetV0SelectionResults();
  if (rawCandidates.size > 0u) {
    Trace("v0-selection: launch selection kernel");
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
  Trace("v0-selection: download selected candidates");
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

  fImpl->fLastDecayPlanResults.clear();
  fImpl->fLastDecayPlanEventResults.clear();
  fImpl->fLastDecayPlanTiming = KFParticleGpuDecayPlanTiming();
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
    return fImpl->fLastDecayPlanResults;
  }

  unsigned int candidateOffset = 0u;
  unsigned int daughterOffset = 0u;
  for (unsigned int eventOffset = 0u; eventOffset < eventCount; ++eventOffset) {
    const unsigned int eventIndex = firstEventIndex + eventOffset;
    KFParticleGpuDecayPlanEventResult eventResult;
    eventResult.eventIndex = eventIndex;
    eventResult.channelOffset = static_cast<unsigned int>(fImpl->fLastDecayPlanResults.size());
    eventResult.candidates.offset = candidateOffset;
    eventResult.candidates.daughterOffset = daughterOffset;
    for (std::size_t channelIndex = 0u;
         channelIndex < fImpl->fDecayPlan->NumberOfTwoDaughterChannels();
         ++channelIndex) {
      const KFParticleGpuTwoDaughterChannel& channel =
        fImpl->fDecayPlan->TwoDaughterChannel(channelIndex);
      fImpl->fLastDecayPlanResults.push_back(RunTwoDaughterCompactChannel(
        MakeTwoDaughterTaskSource(channel, eventIndex),
        taskCapacity,
        channel.channelId,
        candidateOffset,
        daughterOffset));
      const KFParticleGpuTwoDaughterChannelResult& result = fImpl->fLastDecayPlanResults.back();
      candidateOffset += result.candidates.size;
      daughterOffset += result.candidates.daughterSize;
      eventResult.overflowFlags |= result.candidates.overflowFlags;
    }
    eventResult.channelCount = static_cast<unsigned int>(fImpl->fLastDecayPlanResults.size())
                               - eventResult.channelOffset;
    eventResult.candidates.size = candidateOffset - eventResult.candidates.offset;
    eventResult.candidates.daughterSize = daughterOffset - eventResult.candidates.daughterOffset;
    eventResult.candidates.overflowFlags = eventResult.overflowFlags;
    fImpl->fLastDecayPlanEventResults.push_back(eventResult);
  }
  fImpl->fLastDecayPlanTiming.constructionMilliseconds = elapsedMilliseconds(constructionStarted);

  // Default V0 channels append compact indices while the raw candidate pool is
  // still resident. Generic channels leave expectedMass unset and skip this
  // physics-specific continuation.
  const auto selectionStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->ResetSelectedCandidates();
  for (const auto& eventResult : fImpl->fLastDecayPlanEventResults) {
    for (unsigned int channelOffset = 0u; channelOffset < eventResult.channelCount; ++channelOffset) {
      const KFParticleGpuTwoDaughterChannel& channel =
        fImpl->fDecayPlan->TwoDaughterChannel(channelOffset);
      const KFParticleGpuTwoDaughterChannelResult& result =
        fImpl->fLastDecayPlanResults[eventResult.channelOffset + channelOffset];
      if (channel.selection.expectedMass <= 0.f || result.candidates.size == 0u) {
        continue;
      }
      fImpl->fQueue.launch<KFParticleGpuSelectV0Candidates>(
        xpu::n_threads(result.candidates.size),
        MakeConstView(fImpl->fBuffers->DeviceCandidates()),
        MakeConstView(fImpl->fBuffers->DevicePrimaryVertices()),
        fImpl->fBuffers->DeviceEvents(),
        eventResult.eventIndex,
        result.candidates.offset,
        result.candidates.size,
        channel.channelId,
        channel.selection,
        fImpl->fBuffers->DeviceV0SelectionResults(),
        fImpl->fBuffers->DeviceSelectedCandidates());
    }
  }
  fImpl->fQueue.wait();
  fImpl->fLastDecayPlanTiming.selectionMilliseconds = elapsedMilliseconds(selectionStarted);
  const auto downloadStarted = std::chrono::steady_clock::now();
  fImpl->fBuffers->DownloadSelectedCandidates();
  fImpl->fBuffers->DownloadV0SelectionResults();
  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(fImpl->fBuffers->HostSelectedCandidates());
  fImpl->fLastDecayPlanSelectedCandidates.offset = 0u;
  fImpl->fLastDecayPlanSelectedCandidates.size = selected.Size();
  fImpl->fLastDecayPlanSelectedCandidates.overflowFlags = selected.OverflowFlags();
  unsigned int selectedOffset = 0u;
  for (auto& eventResult : fImpl->fLastDecayPlanEventResults) {
    eventResult.selectedCandidates.offset = selectedOffset;
    for (unsigned int channelOffset = 0u; channelOffset < eventResult.channelCount; ++channelOffset) {
      const KFParticleGpuTwoDaughterChannelResult& channelResult =
        fImpl->fLastDecayPlanResults[eventResult.channelOffset + channelOffset];
      KFParticleGpuSelectedChannelRange channelRange;
      channelRange.channelId = channelResult.channelId;
      channelRange.eventIndex = eventResult.eventIndex;
      channelRange.candidates.offset = selectedOffset;
      while (selectedOffset < selected.Size()
             && selected.ChannelId(selectedOffset) == channelRange.channelId
             && channelResult.candidates.ContainsCandidate(selected.Index(selectedOffset))) {
        ++selectedOffset;
      }
      channelRange.candidates.size = selectedOffset - channelRange.candidates.offset;
      channelRange.candidates.overflowFlags = selected.OverflowFlags();
      fImpl->fLastDecayPlanSelectedChannels.push_back(channelRange);
    }
    eventResult.selectedCandidates.size = selectedOffset - eventResult.selectedCandidates.offset;
    eventResult.selectedCandidates.overflowFlags = selected.OverflowFlags();
    eventResult.overflowFlags |= selected.OverflowFlags();
  }
  fImpl->fBuffers->DownloadCandidates();
  fImpl->fLastDecayPlanTiming.outputDownloadMilliseconds = elapsedMilliseconds(downloadStarted);
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
