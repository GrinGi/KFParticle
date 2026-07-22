/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuBufferManager.h"
#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuRuntime.h"
#include "KFParticleGpuSteering.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef KFPARTICLE_GPU_TEST_DEVICE
#define KFPARTICLE_GPU_TEST_DEVICE "cpu"
#endif

namespace
{
  struct TimingSummary {
    double wallMilliseconds = 0.;
    double uploadMilliseconds = 0.;
    double constructionMilliseconds = 0.;
    double selectionMilliseconds = 0.;
    double downloadMilliseconds = 0.;
    unsigned int rawCandidates = 0;
    unsigned int selectedCandidates = 0;
    unsigned int overflowFlags = 0;
  };

  struct ChannelSummary {
    unsigned int totalPairs = 0;
    unsigned int acceptedTasks = 0;
    unsigned int storedTasks = 0;
    unsigned int candidates = 0;
    unsigned int selectedCandidates = 0;
    unsigned int overflowFlags = 0;
  };

  unsigned int ReadPositiveEnvironment(const char* name, unsigned int fallback)
  {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
      return fallback;
    }

    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    return parsed > 0u && parsed <= 1024u ? static_cast<unsigned int>(parsed) : fallback;
  }

  void StoreTrack(KFParticleGpuInputTrackSoAView& tracks,
                  unsigned int index,
                  float x,
                  float y,
                  float z,
                  float px,
                  float py,
                  float pz,
                  int pdg,
                  int charge,
                  int sourceId)
  {
    KFParticleGpuTrackState track;
    track.X() = x;
    track.Y() = y;
    track.Z() = z;
    track.Px() = px;
    track.Py() = py;
    track.Pz() = pz;

    const float scale = 1.f + 0.05f * static_cast<float>(index);
    track.Covariance(0, 0) = 0.40f * scale;
    track.Covariance(1, 1) = 0.35f * scale;
    track.Covariance(2, 2) = 0.30f * scale;
    track.Covariance(3, 3) = 0.08f * scale;
    track.Covariance(4, 4) = 0.07f * scale;
    track.Covariance(5, 5) = 0.06f * scale;
    track.Covariance(0, 3) = 0.010f * scale;
    track.Covariance(1, 4) = -0.008f * scale;
    track.Covariance(2, 5) = 0.006f * scale;
    StoreTrackState(track, tracks, index);

    tracks.SourceId(index) = sourceId;
    tracks.Pdg(index) = pdg;
    tracks.Charge(index) = charge;
    tracks.PrimaryVertexIndex(index) = -1;
    tracks.NumberOfPixelHits(index) = 6;
    tracks.ChiToPrimaryVertex(index) = 1.f;
  }

  void FillDefaultV0Events(KFParticleGpuBufferManager& buffers, unsigned int eventCount)
  {
    const unsigned int trackCount = 4u * eventCount;
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks = trackCount;
    capacities.events = eventCount;
    capacities.candidates = 3u * eventCount;
    capacities.daughterIds = 6u * eventCount;
    capacities.selectedCandidates = 3u * eventCount;
    capacities.twoDaughterTasks = 3u * eventCount;
    // The benchmark measures batching, not field interpolation.  Keep the
    // default-V0 field-aware path deterministic through its zero-field fallback.
    capacities.nonhomogeneousField = false;
    buffers.EnsureCapacity(capacities);
    buffers.SetInputSizes(trackCount, 0u, eventCount);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int event = 0u; event < eventCount; ++event) {
      const unsigned int offset = 4u * event;
      const float shift = 0.01f * static_cast<float>(event);
      const int sourceBase = 10000 + static_cast<int>(10u * event);
      StoreTrack(tracks, offset, -0.20f + shift, 0.10f, 0.00f, 0.80f, 0.10f, 1.00f,
                 211, 1, sourceBase + 1);
      StoreTrack(tracks, offset + 1u, 0.40f + shift, -0.30f, 0.20f, 0.50f, 0.60f, 0.90f,
                 2212, 1, sourceBase + 2);
      StoreTrack(tracks, offset + 2u, 0.10f + shift, 0.50f, -0.40f, -0.30f, 0.70f, 1.10f,
                 -211, -1, sourceBase + 3);
      StoreTrack(tracks, offset + 3u, -0.60f + shift, -0.20f, 0.30f, -0.40f, 0.20f, 0.80f,
                 -2212, -1, sourceBase + 4);

      events[event] = KFParticleGpuEventDesc();
      events[event].eventId = 9000u + event;
      events[event].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(offset, 2u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(offset + 1u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(offset + 2u, 2u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(offset + 2u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Proton) = KFParticleGpuRange(offset + 3u, 1u);
    }
  }

  ChannelSummary Summarize(const KFParticleGpuTwoDaughterChannelResult& result,
                           const KFParticleGpuSelectedChannelRange& selected)
  {
    ChannelSummary summary;
    summary.totalPairs = result.totalPairs;
    summary.acceptedTasks = result.acceptedTasks;
    summary.storedTasks = result.storedTasks;
    summary.candidates = result.candidates.size;
    summary.selectedCandidates = selected.candidates.size;
    summary.overflowFlags = result.candidates.overflowFlags | selected.candidates.overflowFlags;
    return summary;
  }

  void AddTiming(TimingSummary& summary, const KFParticleGpuDecayPlanTiming& timing)
  {
    summary.uploadMilliseconds += timing.inputUploadMilliseconds;
    summary.constructionMilliseconds += timing.constructionMilliseconds;
    summary.selectionMilliseconds += timing.selectionMilliseconds;
    summary.downloadMilliseconds += timing.outputDownloadMilliseconds;
  }

  TimingSummary RunSerial(KFParticleGpuSteering& steering,
                          KFParticleGpuBufferManager& buffers,
                          unsigned int eventCount,
                          unsigned int iterations,
                          std::vector<ChannelSummary>& result)
  {
    TimingSummary timing;
    result.clear();
    const auto started = std::chrono::steady_clock::now();
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
      FillDefaultV0Events(buffers, eventCount);
      for (unsigned int event = 0u; event < eventCount; ++event) {
        const std::vector<KFParticleGpuTwoDaughterChannelResult>& channels =
          steering.RunDecayPlan(event, 1u);
        if (iteration + 1u == iterations) {
          const std::vector<KFParticleGpuSelectedChannelRange>& selected =
            steering.LastDecayPlanSelectedChannels();
          if (channels.size() != selected.size()) {
            throw std::runtime_error("serial default-V0 selection/channel result size mismatch");
          }
          for (std::size_t channelIndex = 0u; channelIndex < channels.size(); ++channelIndex) {
            result.push_back(Summarize(channels[channelIndex], selected[channelIndex]));
            timing.rawCandidates += channels[channelIndex].candidates.size;
            timing.selectedCandidates += selected[channelIndex].candidates.size;
            timing.overflowFlags |= channels[channelIndex].candidates.overflowFlags;
            timing.overflowFlags |= selected[channelIndex].candidates.overflowFlags;
          }
        }
        AddTiming(timing, steering.LastDecayPlanTiming());
      }
    }
    timing.wallMilliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return timing;
  }

  TimingSummary RunBatch(KFParticleGpuSteering& steering,
                         KFParticleGpuBufferManager& buffers,
                         unsigned int eventCount,
                         unsigned int iterations,
                         std::vector<ChannelSummary>& result)
  {
    TimingSummary timing;
    result.clear();
    const auto started = std::chrono::steady_clock::now();
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
      FillDefaultV0Events(buffers, eventCount);
      const std::vector<KFParticleGpuTwoDaughterChannelResult>& channels =
        steering.RunDecayPlanBatch(0u, eventCount, 1u);
      if (iteration + 1u == iterations) {
        const std::vector<KFParticleGpuSelectedChannelRange>& selected =
          steering.LastDecayPlanSelectedChannels();
        if (channels.size() != selected.size()) {
          throw std::runtime_error("batch default-V0 selection/channel result size mismatch");
        }
        for (std::size_t channelIndex = 0u; channelIndex < channels.size(); ++channelIndex) {
          result.push_back(Summarize(channels[channelIndex], selected[channelIndex]));
        }
        const std::vector<KFParticleGpuDecayPlanEventResult>& events =
          steering.LastDecayPlanEventResults();
        for (const KFParticleGpuDecayPlanEventResult& event : events) {
          timing.rawCandidates += event.candidates.size;
          timing.selectedCandidates += event.selectedCandidates.size;
          timing.overflowFlags |= event.overflowFlags;
        }
      }
      AddTiming(timing, steering.LastDecayPlanTiming());
    }
    timing.wallMilliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    return timing;
  }

  bool SameResults(const std::vector<ChannelSummary>& serial,
                   const std::vector<ChannelSummary>& batch)
  {
    if (serial.size() != batch.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < serial.size(); ++index) {
      const ChannelSummary& first = serial[index];
      const ChannelSummary& second = batch[index];
      if (first.totalPairs != second.totalPairs || first.acceptedTasks != second.acceptedTasks
          || first.storedTasks != second.storedTasks || first.candidates != second.candidates
          || first.selectedCandidates != second.selectedCandidates
          || first.overflowFlags != second.overflowFlags) {
        return false;
      }
    }
    return true;
  }

  void PrintTiming(const char* mode, const TimingSummary& timing, unsigned int iterations)
  {
    const double divisor = static_cast<double>(iterations);
    std::cout << std::fixed << std::setprecision(3)
              << "METRIC mode=" << mode
              << " wall_ms=" << timing.wallMilliseconds / divisor
              << " h2d_ms=" << timing.uploadMilliseconds / divisor
              << " construction_ms=" << timing.constructionMilliseconds / divisor
              << " selection_ms=" << timing.selectionMilliseconds / divisor
              << " d2h_ms=" << timing.downloadMilliseconds / divisor
              << " raw=" << timing.rawCandidates
              << " selected=" << timing.selectedCandidates
              << " overflow=" << timing.overflowFlags << '\n';
  }
}

int main()
{
  const unsigned int eventCount = ReadPositiveEnvironment("KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS", 4u);
  const unsigned int iterations = ReadPositiveEnvironment("KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS", 10u);

  KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
  KFParticleGpuRuntimeSettings settings;
  settings.device = KFPARTICLE_GPU_TEST_DEVICE;
  settings.initializeXpuIfNeeded = true;
  runtime.Initialize(settings);

  try {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);

    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    std::vector<ChannelSummary> serial;
    std::vector<ChannelSummary> batch;

    // Warm-up creates the XPU image and persistent allocations outside timing.
    FillDefaultV0Events(buffers, eventCount);
    steering.RunDecayPlanBatch(0u, eventCount, 1u);

    const TimingSummary serialTiming = RunSerial(steering, buffers, eventCount, iterations, serial);
    const TimingSummary batchTiming = RunBatch(steering, buffers, eventCount, iterations, batch);
    if (!SameResults(serial, batch)) {
      throw std::runtime_error("serial and batch default-V0 channel summaries differ");
    }

    std::cout << "PASS kfparticle-gpu-batch-benchmark"
              << " - events=" << eventCount
              << " iterations=" << iterations
              << " channels=K0S,Lambda,anti-Lambda"
              << " serial_batch_equivalent=1\n";
    PrintTiming("serial", serialTiming, iterations);
    PrintTiming("batch", batchTiming, iterations);

    plan.Clear();
    runtime.Finalize();
    return 0;
  }
  catch (...) {
    runtime.Finalize();
    throw;
  }
}
