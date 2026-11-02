/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Controlled V0 throughput probe. Cascade/routing correctness lives in Lifecycle.
#include "execution/KFParticleGpuRuntime.h"
#include "execution/KFParticleGpuSteering.h"
#include "execution/KFParticleGpuBufferManager.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "planning/KFParticleGpuDecayPlan.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef KFPARTICLE_GPU_TEST_DEVICE
#define KFPARTICLE_GPU_TEST_DEVICE "cpu"
#endif

namespace
{
  void Require(bool condition, const char* message)
  {
    if (!condition) throw std::runtime_error(message);
  }

  unsigned int ReadCount(const char* name, unsigned int fallback, unsigned int maximum)
  {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    const std::string text(value);
    Require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "benchmark counts must be positive decimal integers");
    const auto parsed = std::stoul(text);
    Require(parsed > 0 && parsed <= maximum, "benchmark count is out of range");
    return static_cast<unsigned int>(parsed);
  }

  // One entry per event/channel, independent of candidate-pool storage order.
  using Result = std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int>;
  struct Measurement {
    double wall = 0., upload = 0., construction = 0., selection = 0., download = 0.;
    std::vector<Result> channels;
    KFParticleGpuPerformanceSnapshot performance;
  };

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

    KFParticleGpuFieldRegion field;
    field.Coefficient(0) = 0.4f;
    field.Coefficient(1) = 0.02f;
    field.Coefficient(2) = 0.001f;
    field.Coefficient(3) = 20.f;
    field.Coefficient(4) = 0.5f;
    field.Coefficient(5) = 0.01f;
    field.Coefficient(6) = -0.3f;
    field.Coefficient(7) = 0.015f;
    field.Coefficient(8) = 0.0005f;
    field.Coefficient(9) = z;
    StoreFieldRegion(field, tracks, index);
  }

  void FillDefaultV0Events(KFParticleGpuBufferManager& buffers, unsigned int eventCount)
  {
    const unsigned int trackCount = 4u * eventCount;
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks = trackCount;
    capacities.vertices = eventCount;
    capacities.events = eventCount;
    capacities.candidates = 3u * eventCount;
    capacities.daughterIds = 6u * eventCount;
    capacities.selectedCandidates = 3u * eventCount;
    capacities.twoDaughterTasks = 3u * eventCount;
    capacities.nonhomogeneousField = true;
    buffers.EnsureCapacity(capacities);
    buffers.SetInputSizes(trackCount, eventCount, eventCount);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
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

      KFParticleGpuVertexState vertex;
      vertex.Initialize();
      vertex.X() = 0.f;
      vertex.Y() = 0.f;
      vertex.Z() = 0.f;
      vertex.Covariance(0) = 0.01f;
      vertex.Covariance(2) = 0.01f;
      vertex.Covariance(5) = 0.01f;
      StoreVertexState(vertex, vertices, event);

      events[event] = KFParticleGpuEventDesc();
      events[event].eventId = 9000u + event;
      events[event].primaryVertices = KFParticleGpuRange(event, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(offset, 2u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(offset + 1u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(offset + 2u, 2u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(offset + 2u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Proton) = KFParticleGpuRange(offset + 3u, 1u);
    }
  }

  std::vector<Result> CheckOutput(KFParticleGpuSteering& steering)
  {
    const auto& channels = steering.LastDecayPlanResults();
    const auto& selected = steering.LastDecayPlanSelectedChannels();
    Require(channels.size() == selected.size(), "channel/selection count mismatch");
    std::vector<Result> result;
    for (std::size_t i = 0; i < channels.size(); ++i) {
      const auto& c = channels[i];
      Require(!c.Truncated() && selected[i].candidates.overflowFlags == 0u,
              "benchmark output overflowed");
      result.emplace_back(c.eventIndex, c.channelId, c.totalPairs, c.acceptedTasks,
                          c.storedTasks, c.constructedCandidates, selected[i].candidates.size);
    }
    auto& buffers = steering.GetBuffers();
    const auto candidates = MakeConstView(buffers.HostCandidates());
    const auto topology = MakeConstView(buffers.HostV0SelectionResults());
    Require(candidates.Size() > 0u && candidates.OverflowFlags() == 0u,
            "benchmark produced empty or overflowing candidates");
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      Require(topology.CanStore(i), "candidate lacks topology storage");
      const auto& t = topology.Result(i);
      Require((t.topologyStatus & KFGpuV0LineTopologyValid) && t.bestPrimaryVertexIndex >= 0,
              "benchmark candidate lacks valid primary-vertex topology");
    }
    const auto& routing = steering.LastTwoDaughterRoutingMonitorData();
    Require(routing.overflowFlags == 0u && routing.acceptedTasks == routing.storedTasks,
            "benchmark routing lost accepted work");
    return result;
  }

  Measurement Measure(KFParticleGpuSteering& steering, unsigned int events,
                      unsigned int iterations, unsigned int warmup, bool batch)
  {
    Measurement result;
    std::vector<Result> expected;
    // Each mode warms its own execution path. Repack/upload on every iteration;
    // reuse capacity, not previously reconstructed output.
    for (unsigned int iteration = 0; iteration < warmup + iterations; ++iteration) {
      const bool measured = iteration >= warmup;
      const auto start = std::chrono::steady_clock::now();
      FillDefaultV0Events(steering.GetBuffers(), events);
      std::vector<Result> actual;
      for (unsigned int event = 0; event < events; event += batch ? events : 1u) {
        if (batch) steering.RunDecayPlanBatch(0u, events, 1u);
        else steering.RunDecayPlan(event, 1u);
        const auto& timing = steering.LastDecayPlanTiming();
        if (measured) {
          result.upload += timing.inputUploadMilliseconds;
          result.construction += timing.constructionMilliseconds;
          result.selection += timing.selectionMilliseconds;
          result.download += timing.outputDownloadMilliseconds;
        }
        const auto channels = CheckOutput(steering);
        actual.insert(actual.end(), channels.begin(), channels.end());
      }
      const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      Require(actual.size() == 3u * events, "benchmark did not process all event/channel combinations");
      if (iteration == 0u) expected = actual;
      Require(actual == expected, "benchmark output changed between repetitions");
      if (measured) result.wall += elapsed;
    }
    const double n = iterations;
    result.wall /= n;
    result.upload /= n;
    result.construction /= n;
    result.selection /= n;
    result.download /= n;
    result.channels = std::move(expected);
    result.performance = steering.LastPerformanceSnapshot();
    return result;
  }

  void Print(const char* mode, const Measurement& result, unsigned int events)
  {
    unsigned long long pairs = 0, raw = 0, selected = 0;
    for (const auto& c : result.channels) {
      pairs += std::get<2>(c);
      raw += std::get<5>(c);
      selected += std::get<6>(c);
    }
    Require(std::isfinite(result.wall) && result.wall > 0., "invalid benchmark wall time");
    std::cout << std::fixed << std::setprecision(6)
              << "METRIC mode=" << mode << " events=" << events
              << " wall_ms=" << result.wall << " per_event_ms=" << result.wall / events
              << " h2d_ms=" << result.upload << " construction_ms=" << result.construction
              << " topology_selection_ms=" << result.selection << " d2h_ms=" << result.download
              << " pairs=" << pairs << " raw=" << raw << " selected=" << selected
              << " overflow=0\n";
  }
}

int main()
{
  auto& runtime = KFParticleGpuRuntime::Instance();
  try {
    const auto events = ReadCount("KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS", 4u, 1024u);
    const auto iterations = ReadCount("KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS", 10u, 1024u);
    const auto warmup = ReadCount("KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS", 3u, 1024u);
    KFParticleGpuRuntimeSettings settings;
    settings.device = KFPARTICLE_GPU_TEST_DEVICE;
    settings.initializeXpuIfNeeded = true;
    runtime.Initialize(settings);
    auto& steering = runtime.GetSteering();
    steering.SetPerformanceMonitoringEnabled(true);
    auto& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);
    const auto serial = Measure(steering, events, iterations, warmup, false);
    const auto batch = Measure(steering, events, iterations, warmup, true);
    Require(serial.channels == batch.channels, "serial/batch event-channel results differ");
    Print("serial", serial, events);
    Print("batch", batch, events);
    const auto& p = batch.performance;
    Require(p.enabled && p.overflowFlags == 0u, "invalid batch performance snapshot");
    std::cout << "PERFORMANCE_METRIC mode=batch events=" << p.events << " tracks=" << p.tracks
              << " launches=" << p.kernelLaunches << " queue_waits=" << p.queueWaits
              << " capacity_allocation_rounds=" << p.capacityGrowths
              << " h2d_bytes=" << p.hostToDeviceBytes << " d2h_bytes=" << p.deviceToHostBytes
              << " allocated_high_water_bytes=" << p.allocatedBytesHighWater << '\n';
    runtime.Finalize();
    std::cout << "PASS kfparticle-gpu-batch-benchmark events=" << events
              << " iterations=" << iterations << " warmup_iterations=" << warmup
              << " serial_batch_equivalent=1\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL kfparticle-gpu-batch-benchmark - " << error.what() << '\n';
    runtime.Finalize();
    return 1;
  }
}
