/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#include "execution/KFParticleGpuFinder.h"

#include "execution/KFParticleGpuRuntime.h"
#include "execution/KFParticleGpuRuntimeService.h"
#include "planning/KFParticleGpuCpuChannelCatalogue.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
  double Milliseconds(std::chrono::steady_clock::time_point started)
  {
    return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  }

  void ValidatePreparedOptions(const KFParticleGpuFinder::PreparedOptions& run)
  {
    if (run.outputMode != kfparticle::gpu::GpuOutputMode::PublishableOutput
        && run.outputMode != kfparticle::gpu::GpuOutputMode::FullDiagnostics) {
      throw std::invalid_argument("KFParticleGpuFinder received an invalid prepared output mode");
    }
    if ((!run.materializeOutput && run.outputMode == kfparticle::gpu::GpuOutputMode::PublishableOutput)
        || (run.replay && !run.inheritEnvironment)) {
      throw std::invalid_argument("KFParticleGpuFinder received inconsistent prepared execution controls");
    }
  }
}

bool KFParticleGpuFinder::Result::Succeeded() const
{
  return overflowFlags == 0u
         && (status == Status::Completed || status == Status::EmptyInput
             || status == Status::NoPairs);
}

KFParticleGpuFinder::KFParticleGpuFinder() : KFParticleGpuFinder(Options{}) {}

KFParticleGpuFinder::KFParticleGpuFinder(const Options& options)
  : fOptions(options), fOwnerThread(std::this_thread::get_id())
{
#ifndef KFPARTICLE_USE_XPU
  throw std::invalid_argument("KFParticleGpuFinder requires KFPARTICLE_USE_XPU=ON");
#endif
#ifndef NonhomogeneousField
  throw std::invalid_argument("KFParticleGpuFinder host input requires FIXTARGET=ON");
#endif
  const auto& limits = fOptions.limits;
  if (limits.events == 0u || limits.tracks == 0u || limits.vertices == 0u
      || limits.twoDaughterTasks == 0u || limits.candidates == 0u
      || limits.daughterIds == 0u || limits.selectedCandidates == 0u) {
    throw std::invalid_argument("KFParticleGpuFinder requires positive batch resource limits");
  }
  // Reuse the executor's worker validation rather than introducing a second rule.
  const kfparticle::gpu::GpuExecutor checked(fOptions.materializationThreads,
                                            fOptions.packingThreads);
  (void) checked;
  const auto& catalogue = KFParticleGpuCpuChannelCatalogue();
  for (int pdg : fOptions.motherPdgs) {
    bool supported = false;
    for (const auto& channel : catalogue) {
      if (channel.motherPdg != pdg
          || channel.activation == KFGpuCpuChannelConfigurationDisabled) { continue; }
      if (channel.supportStatus != KFGpuGraphSupported) {
        throw std::invalid_argument("KFParticleGpuFinder requested an unsupported channel");
      }
      supported = true;
    }
    if (!supported) {
      throw std::invalid_argument("KFParticleGpuFinder requested an unknown or disabled mother PDG");
    }
  }
}

void KFParticleGpuFinder::CheckThread() const
{
  if (fOwnerThread != std::this_thread::get_id()) {
    throw std::logic_error("KFParticleGpuFinder requires its constructing host thread");
  }
}

void KFParticleGpuFinder::Initialize() const
{
  CheckThread();
  kfparticle::gpu::GpuRuntimeService::Instance().Initialize(RuntimeSettings());
}

KFParticleGpuRuntimeSettings KFParticleGpuFinder::RuntimeSettings() const
{
  KFParticleGpuRuntimeSettings settings;
  settings.device = fOptions.device;
  settings.initializeXpuIfNeeded = fOptions.initializeXpuIfNeeded;
  settings.verbose = fOptions.verbose;
  return settings;
}

void KFParticleGpuFinder::ReleaseResources() const
{
  CheckThread();
  kfparticle::gpu::GpuRuntimeService::Instance().Finalize(true);
}

KFParticleGpuFinder::Result KFParticleGpuFinder::Reconstruct(const Event& event) const
{
  auto results = ReconstructBatch({event});
  return std::move(results.front());
}

std::vector<kfparticle::gpu::GpuEventResult> KFParticleGpuFinder::ReconstructPreparedBatch(
  const kfparticle::gpu::GpuPreparedBatch& batch,
  const PreparedOptions& run) const
{
  CheckThread();
  ValidatePreparedOptions(run);
  const auto runtime = RuntimeSettings();
  const auto settings = kfparticle::gpu::GpuExecutor::ExecutionSettings{run.inheritEnvironment, &runtime};
  kfparticle::gpu::GpuExecutor executor(fOptions.materializationThreads, fOptions.packingThreads);
  return run.replay
    ? executor.RunBatchConfigured(batch, run.outputMode, run.materializeOutput, settings)
    : executor.ExecuteBatchOnce(batch, run.outputMode, run.materializeOutput, settings);
}

kfparticle::gpu::GpuEventResult KFParticleGpuFinder::ReconstructDetailedEvent(
  const Event& event, const PreparedOptions& run) const
{
  CheckThread();
  ValidatePreparedOptions(run);
  if (run.replay) { throw std::invalid_argument("KFParticleGpuFinder individual-event integration cannot replay"); }
  const auto runtime = RuntimeSettings();
  const auto settings = kfparticle::gpu::GpuExecutor::ExecutionSettings{run.inheritEnvironment, &runtime};
  const kfparticle::gpu::GpuEventInput input{static_cast<unsigned int>(event.sourceEventId),
    &event.first, &event.last, &event.chiToPV, &event.vertices, event.minSecondaryTrackChiToPV};
  return kfparticle::gpu::GpuExecutor(fOptions.materializationThreads, fOptions.packingThreads)
    .RunSingleConfigured(event.sourceEventId, input, run.outputMode, fOptions.motherPdgs,
                         run.materializeOutput, settings);
}

std::vector<KFParticleGpuFinder::Result> KFParticleGpuFinder::ReconstructBatch(
  const std::vector<Event>& events) const
{
  CheckThread();
  if (events.empty()) { return {}; }
  const auto started = std::chrono::steady_clock::now();
  std::vector<Result> results(events.size());
  for (std::size_t i = 0u; i < events.size(); ++i) {
    results[i].sourceEventId = events[i].sourceEventId;
    results[i].tracks = events[i].first.Size() > 0
                         ? static_cast<unsigned int>(events[i].first.Size()) : 0u;
  }
  Timing timing;
  Status failure = Status::InputRejected;
  try {
    if (events.size() > fOptions.limits.events) {
      throw std::invalid_argument("KFParticleGpuFinder batch exceeds the configured event limit");
    }
    std::vector<std::uint64_t> identities;
    identities.reserve(events.size());
    for (const auto& event : events) { identities.push_back(event.sourceEventId); }
    std::sort(identities.begin(), identities.end());
    if (std::adjacent_find(identities.begin(), identities.end()) != identities.end()) {
      throw std::invalid_argument("KFParticleGpuFinder batch contains duplicate source event IDs");
    }

    kfparticle::gpu::GpuBatchBuilder builder(fOptions.limits, fOptions.motherPdgs);
    for (std::size_t i = 0u; i < events.size(); ++i) {
      const auto& event = events[i];
      kfparticle::gpu::GpuEventInput input;
      input.eventId = static_cast<unsigned int>(i);
      input.first = &event.first;
      input.last = &event.last;
      input.chiToPrimaryVertex = &event.chiToPV;
      input.primaryVertices = &event.vertices;
      input.minSecondaryTrackChiToPrimaryVertex = event.minSecondaryTrackChiToPV;
      if (builder.TryAppend(event.sourceEventId, input)
          != kfparticle::gpu::GpuBatchAppendResult::Added) {
        throw std::invalid_argument("KFParticleGpuFinder batch exceeds configured resources or GPU ABI");
      }
    }
    auto batch = builder.Seal();
    if (batch.TaskCapacity() != 0u && batch.TrackCount() != 0u && batch.VertexCount() != 0u) {
      failure = Status::RuntimeUnavailable;
      const auto initializationStarted = std::chrono::steady_clock::now();
      Initialize();
      timing.initialization = Milliseconds(initializationStarted);
    }
    failure = Status::ExecutionFailed;
    PreparedOptions run;
    run.outputMode = fOptions.diagnostics ? kfparticle::gpu::GpuOutputMode::FullDiagnostics
                                         : kfparticle::gpu::GpuOutputMode::PublishableOutput;
    auto raw = ReconstructPreparedBatch(batch, run);
    if (raw.size() != results.size()) {
      throw std::runtime_error("KFParticleGpuFinder executor returned an inconsistent result count");
    }
    if (fOptions.timings) {
      const auto& first = raw.front();
      for (const auto& event : raw) {
        timing.inputPreparation += event.inputPreparationMilliseconds;
        timing.capacityPlanning += event.capacityPlanningMilliseconds;
      }
      timing.queueAndCapacity = first.queueAndCapacityMilliseconds;
      timing.packing = first.hostPackingMilliseconds;
      timing.hostToDevice = first.inputUploadMilliseconds;
      timing.construction = first.constructionMilliseconds;
      timing.graph = first.graphConstructionMilliseconds;
      timing.selection = first.selectionMilliseconds;
      timing.deviceToHost = first.outputDownloadMilliseconds;
      timing.pipeline = first.gpuPipelineMilliseconds;
      timing.extraction = first.resultExtractionMilliseconds;
      timing.materialization = first.materializationMilliseconds;
    }
    for (std::size_t i = 0u; i < raw.size(); ++i) {
      auto& source = raw[i];
      auto& result = results[i];
      if (source.sourceEventId != result.sourceEventId) {
        throw std::runtime_error("KFParticleGpuFinder executor changed source event order");
      }
      result.status = source.status;
      result.candidates = source.candidates;
      result.selectedCandidates = source.selectedCandidates;
      result.overflowFlags = source.overflowFlags;
      result.message = std::move(source.message);
      if (result.status == Status::Completed && !source.materializationSucceeded) {
        result.status = Status::ExecutionFailed;
        result.message = "KFParticleGpuFinder could not materialize the complete event";
      }
      if (result.Succeeded()) { result.particles = std::move(source.materializedParticles); }
      if (fOptions.diagnostics) {
        result.diagnostics.emplace();
        result.diagnostics->channels = std::move(source.channels);
        result.diagnostics->candidates = std::move(source.rawCandidates);
      }
    }
  }
  catch (const std::exception& error) {
    for (auto& result : results) {
      result.status = failure;
      result.particles.clear();
      result.diagnostics.reset();
      result.message = error.what();
    }
  }
  if (fOptions.timings) {
    timing.total = Milliseconds(started);
    results.front().timing = timing;
  }
  return results;
}
