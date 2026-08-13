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
  constexpr unsigned int CascadeBenchmarkTaskCapacity = 32u;
  constexpr unsigned int CascadeBenchmarkSelectedCandidates = 8u;

  struct TimingSummary {
    double wallMilliseconds = 0.;
    double uploadMilliseconds = 0.;
    double constructionMilliseconds = 0.;
    double selectionMilliseconds = 0.;
    double cascadeConstructionMilliseconds = 0.;
    double downloadMilliseconds = 0.;
    unsigned int totalPairs = 0;
    unsigned int acceptedTasks = 0;
    unsigned int storedTasks = 0;
    unsigned int rawCandidates = 0;
    unsigned int selectedCandidates = 0;
    unsigned int topologyCandidates = 0;
    unsigned int validTopologies = 0;
    unsigned int candidatesWithBestPrimaryVertex = 0;
    unsigned int topologyRejected = 0;
    unsigned int overflowFlags = 0;
    unsigned int cascadePairs = 0;
    unsigned int cascadeCandidates = 0;
    unsigned int generationGroupLaunches = 0;
    unsigned int generationDescriptorCount = 0;
    unsigned int generationSelectionLaunches = 0;
    unsigned int generationVisitedPairs = 0;
    unsigned int generationActiveChannelBits = 0;
    unsigned int generationAcceptedTasks = 0;
    unsigned int generationStoredTasks = 0;
    unsigned int generationBlockReservations = 0;
    unsigned int generationDaughters = 0;
    unsigned int generationSelectedCandidates = 0;
    unsigned int routingGroupLaunches = 0;
    unsigned int routingDescriptorCount = 0;
    unsigned int routingVisitedPairs = 0;
    unsigned int routingActiveChannelBits = 0;
    unsigned int routingAcceptedTasks = 0;
    unsigned int routingStoredTasks = 0;
    unsigned int routingBlockReservations = 0;
    unsigned int routingDaughters = 0;
    unsigned int graphGroupLaunches = 0;
    unsigned int graphDescriptorCount = 0;
    unsigned int graphVisitedCombinations = 0;
    unsigned int graphAcceptedTasks = 0;
    unsigned int graphStoredTasks = 0;
    unsigned int graphConstructedCandidates = 0;
    unsigned int graphRejectedTasks = 0;
    unsigned int graphDaughters = 0;
    unsigned int generationPoolOverflowFlags = 0;
    unsigned int generationRoutingOverflowFlags = 0;
    unsigned int cascadeRoutingOverflowFlags = 0;
    unsigned int graphOverflowFlags = 0;
  };

  struct ChannelSummary {
    unsigned int totalPairs = 0;
    unsigned int acceptedTasks = 0;
    unsigned int storedTasks = 0;
    unsigned int candidates = 0;
    unsigned int selectedCandidates = 0;
    unsigned int overflowFlags = 0;
  };

  struct RoutingModeTiming {
    double wallMilliseconds = 0.;
    KFParticleGpuV0TrackRoutingStatus status;
    unsigned int candidates = 0u;
    unsigned int daughters = 0u;
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

  void ConfigureDefaultV0Plan(KFParticleGpuDecayPlan& plan, int transportMode, bool withCascade = false)
  {
    plan.Clear();
    KFParticleGpuTwoDaughterChannel k0 = MakeK0ShortToPiPlusPiMinusChannel();
    KFParticleGpuTwoDaughterChannel lambda = MakeLambdaToProtonPiMinusChannel();
    KFParticleGpuTwoDaughterChannel antiLambda = MakeAntiLambdaToAntiProtonPiPlusChannel();
    k0.transportMode = transportMode;
    lambda.transportMode = transportMode;
    antiLambda.transportMode = transportMode;
    if (withCascade) {
      const auto relax = [](KFParticleGpuTwoDaughterChannel& channel) {
        channel.motherMassSigma = -1.f;
        channel.secondaryMassSigmaCut = -1.f;
        channel.maxSecondaryTopoChi2PerNdf = -1.f;
        channel.minSecondaryLdL = -1.f;
        channel.selection.expectedMass = 1.f;
        channel.selection.expectedMassSigma = -1.f;
        channel.selection.massSigmaCut = -1.f;
        channel.selection.maxGeometricChi2PerNdf = -1.f;
        channel.selection.maxPrimaryVertexDistance = -1.f;
        channel.selection.minSecondaryLdL = -1.f;
        channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
        channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
        channel.selection.requirePrimaryVertex = 0u;
      };
      relax(k0); relax(lambda); relax(antiLambda);
    }
    plan.AddTwoDaughterChannel(k0);
    plan.AddTwoDaughterChannel(lambda);
    plan.AddTwoDaughterChannel(antiLambda);
    if (withCascade) {
      AddDefaultV0TrackCascadeChannels(plan);

      KFParticleGpuGraphOperationChannel composite;
      composite.node.channelId = KFGpuChannelCompositeCompositeProbe;
      composite.node.topology = KFGpuGraphTopologyCompositeComposite;
      composite.node.generation = 2u;
      composite.node.firstSource = {
        KFGpuGraphSourceCandidateGeneration, 1u,
        KFGpuChannelLambdaToProtonPiMinus};
      composite.node.secondSource = {
        KFGpuGraphSourceCandidateGeneration, 1u,
        KFGpuChannelK0ShortToPiPlusPiMinus};
      composite.descriptor.channelId = composite.node.channelId;
      composite.descriptor.topology = composite.node.topology;
      composite.descriptor.operationMask = KFGpuGraphConstruct;
      composite.descriptor.outputClass = KFGpuGraphOutputSecondary;
      composite.descriptor.motherPdg = 9000026;
      composite.descriptor.firstPdg = 3122;
      composite.descriptor.secondPdg = 310;
      plan.AddGraphOperationChannel(composite);

      KFParticleGpuGraphOperationChannel finalState;
      finalState.node.channelId = KFGpuChannelUnaryFinalProbe;
      finalState.node.topology = KFGpuGraphTopologyUnaryComposite;
      finalState.node.generation = 3u;
      finalState.node.firstSource = {
        KFGpuGraphSourceCandidateGeneration, 2u,
        KFGpuChannelCompositeCompositeProbe};
      finalState.descriptor.channelId = finalState.node.channelId;
      finalState.descriptor.topology = finalState.node.topology;
      finalState.descriptor.operationMask = KFGpuGraphSelect;
      finalState.descriptor.outputClass = KFGpuGraphOutputFinal;
      finalState.descriptor.motherPdg = 9000028;
      finalState.descriptor.firstPdg = 9000026;
      plan.AddGraphOperationChannel(finalState);
    }
  }

  void FillCascadeEvents(KFParticleGpuBufferManager& buffers, unsigned int eventCount)
  {
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks = 6u * eventCount;
    capacities.events = eventCount;
    capacities.candidates = 64u * eventCount;
    capacities.daughterIds = 256u * eventCount;
    // The relaxed fixture selects eight V0 candidates. The former six-entry
    // compact pool silently truncated two valid entries and raised
    // KFGpuSelectedCandidateCapacityExceeded, whose numeric bit was then
    // mistaken for graph overflow. Match the selected pool to the candidate
    // pool so selection, cascade routing, and monitoring see the full workload.
    capacities.selectedCandidates = capacities.candidates;
    capacities.twoDaughterTasks = CascadeBenchmarkTaskCapacity * eventCount;
    capacities.v0TrackTasks = CascadeBenchmarkTaskCapacity * eventCount;
    capacities.nonhomogeneousField = true;
    buffers.EnsureCapacity(capacities);
    buffers.SetInputSizes(capacities.tracks, 0u, eventCount);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int event = 0u; event < eventCount; ++event) {
      const unsigned int offset = 6u * event;
      const int source = 20000 + static_cast<int>(10u * event);
      StoreTrack(tracks, offset, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, source + 1);
      StoreTrack(tracks, offset + 1u, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 2212, 1, source + 2);
      StoreTrack(tracks, offset + 2u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, source + 3);
      StoreTrack(tracks, offset + 3u, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, source + 4);
      StoreTrack(tracks, offset + 4u, 0.3f, 0.1f, 0.1f, -0.1f, 0.5f, 0.7f, -211, -1, source + 5);
      StoreTrack(tracks, offset + 5u, -0.1f, 0.2f, -0.2f, 0.2f, 0.3f, 0.9f, -321, -1, source + 6);
      events[event] = KFParticleGpuEventDesc();
      events[event].eventId = 9500u + event;
      events[event].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(offset, 2u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(offset + 1u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(offset + 2u, 4u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(offset + 2u, 3u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(offset + 5u, 1u);
    }
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

  ChannelSummary Summarize(const KFParticleGpuTwoDaughterChannelResult& result,
                           const KFParticleGpuSelectedChannelRange& selected)
  {
    ChannelSummary summary;
    summary.totalPairs = result.totalPairs;
    summary.acceptedTasks = result.acceptedTasks;
    summary.storedTasks = result.storedTasks;
    summary.candidates = result.constructedCandidates;
    summary.selectedCandidates = selected.candidates.size;
    summary.overflowFlags =
      result.generationCandidates.overflowFlags | selected.candidates.overflowFlags;
    return summary;
  }

  void AddTiming(TimingSummary& summary, const KFParticleGpuDecayPlanTiming& timing)
  {
    summary.uploadMilliseconds += timing.inputUploadMilliseconds;
    summary.constructionMilliseconds += timing.constructionMilliseconds;
    summary.selectionMilliseconds += timing.selectionMilliseconds;
    summary.cascadeConstructionMilliseconds += timing.cascadeConstructionMilliseconds;
    summary.downloadMilliseconds += timing.outputDownloadMilliseconds;
  }

  void AddRouting(TimingSummary& summary,
                  const KFParticleGpuV0TrackRoutingMonitorData& routing)
  {
    summary.routingGroupLaunches += routing.groupLaunches;
    if (routing.descriptorCount > summary.routingDescriptorCount) {
      summary.routingDescriptorCount = routing.descriptorCount;
    }
    summary.routingVisitedPairs += routing.visitedPairs;
    summary.routingActiveChannelBits += routing.activeChannelBits;
    summary.routingAcceptedTasks += routing.acceptedTasks;
    summary.routingStoredTasks += routing.storedTasks;
    summary.routingBlockReservations += routing.blockReservations;
    summary.routingDaughters += routing.daughters;
    summary.cascadeRoutingOverflowFlags |= routing.overflowFlags;
    summary.overflowFlags |= routing.overflowFlags;
  }

  void AddGenerationRouting(
    TimingSummary& summary,
    const KFParticleGpuTwoDaughterRoutingMonitorData& routing)
  {
    summary.generationGroupLaunches += routing.groupLaunches;
    if (routing.descriptorCount > summary.generationDescriptorCount) {
      summary.generationDescriptorCount = routing.descriptorCount;
    }
    summary.generationSelectionLaunches += routing.selectionLaunches;
    summary.generationVisitedPairs += routing.visitedPairs;
    summary.generationActiveChannelBits += routing.activeChannelBits;
    summary.generationAcceptedTasks += routing.acceptedTasks;
    summary.generationStoredTasks += routing.storedTasks;
    summary.generationBlockReservations += routing.blockReservations;
    summary.generationDaughters += routing.daughters;
    summary.generationSelectedCandidates += routing.selectedCandidates;
    summary.generationRoutingOverflowFlags |= routing.overflowFlags;
    summary.overflowFlags |= routing.overflowFlags;
  }

  void AddGraphExecution(
    TimingSummary& summary,
    const KFParticleGpuGraphExecutionMonitorData& graph)
  {
    summary.graphGroupLaunches += graph.groupLaunches;
    if (graph.descriptorCount > summary.graphDescriptorCount) {
      summary.graphDescriptorCount = graph.descriptorCount;
    }
    summary.graphVisitedCombinations += graph.visitedCombinations;
    summary.graphAcceptedTasks += graph.acceptedTasks;
    summary.graphStoredTasks += graph.storedTasks;
    summary.graphConstructedCandidates += graph.constructedCandidates;
    summary.graphRejectedTasks += graph.rejectedTasks;
    summary.graphDaughters += graph.daughters;
    summary.graphOverflowFlags |= graph.overflowFlags;
    summary.overflowFlags |= graph.overflowFlags;
  }

  void AddWorkload(TimingSummary& timing, const ChannelSummary& channel)
  {
    timing.totalPairs += channel.totalPairs;
    timing.acceptedTasks += channel.acceptedTasks;
    timing.storedTasks += channel.storedTasks;
    timing.rawCandidates += channel.candidates;
    timing.selectedCandidates += channel.selectedCandidates;
    timing.generationPoolOverflowFlags |= channel.overflowFlags;
    timing.overflowFlags |= channel.overflowFlags;
  }

  void AddTopology(TimingSummary& timing,
                   KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuConstV0SelectionResultView selection =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    for (unsigned int index = 0u; index < candidates.Size(); ++index) {
      if (!selection.CanStore(index)
          || candidates.Metadata().DaughterCount(index) != 2u) {
        continue;
      }
      const KFParticleGpuV0SelectionResult& result = selection.Result(index);
      ++timing.topologyCandidates;
      if ((result.topologyStatus & KFGpuV0LineTopologyValid) != 0u) {
        ++timing.validTopologies;
      }
      if (result.bestPrimaryVertexIndex >= 0) {
        ++timing.candidatesWithBestPrimaryVertex;
      }
      if ((result.rejectionReasons & KFGpuV0SelectionRejectTopology) != 0u) {
        ++timing.topologyRejected;
      }
    }
  }

  TimingSummary RunSerial(KFParticleGpuSteering& steering,
                          KFParticleGpuBufferManager& buffers,
                          unsigned int eventCount,
                          unsigned int iterations,
                          std::vector<ChannelSummary>& result,
                          bool cascade = false)
  {
    TimingSummary timing;
    result.clear();
    const auto started = std::chrono::steady_clock::now();
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
      if (cascade) { FillCascadeEvents(buffers, eventCount); }
      else { FillDefaultV0Events(buffers, eventCount); }
      for (unsigned int event = 0u; event < eventCount; ++event) {
        const std::vector<KFParticleGpuTwoDaughterChannelResult>& channels =
          steering.RunDecayPlan(
            event, cascade ? CascadeBenchmarkTaskCapacity : 1u);
        if (iteration + 1u == iterations) {
          const std::vector<KFParticleGpuSelectedChannelRange>& selected =
            steering.LastDecayPlanSelectedChannels();
          if (channels.size() != selected.size()) {
            throw std::runtime_error("serial default-V0 selection/channel result size mismatch");
          }
          for (std::size_t channelIndex = 0u; channelIndex < channels.size(); ++channelIndex) {
            const ChannelSummary summary = Summarize(channels[channelIndex], selected[channelIndex]);
            result.push_back(summary);
            AddWorkload(timing, summary);
          }
          AddTopology(timing, buffers);
          AddGenerationRouting(
            timing, steering.LastTwoDaughterRoutingMonitorData());
          if (cascade) {
            for (const auto& channel : steering.LastV0TrackCascadeResults()) {
              ChannelSummary summary;
              summary.totalPairs = channel.totalPairs;
              summary.acceptedTasks = channel.acceptedTasks;
              summary.storedTasks = channel.storedTasks;
              summary.candidates = channel.constructedCandidates;
              summary.overflowFlags = channel.generationCandidates.overflowFlags;
              result.push_back(summary);
              timing.cascadePairs += summary.totalPairs;
              timing.cascadeCandidates += summary.candidates;
              timing.overflowFlags |= summary.overflowFlags;
            }
            AddRouting(timing, steering.LastV0TrackRoutingMonitorData());
            AddGraphExecution(
              timing, steering.LastGraphExecutionMonitorData());
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
                         std::vector<ChannelSummary>& result,
                         bool cascade = false)
  {
    TimingSummary timing;
    result.clear();
    const auto started = std::chrono::steady_clock::now();
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
      if (cascade) { FillCascadeEvents(buffers, eventCount); }
      else { FillDefaultV0Events(buffers, eventCount); }
      const std::vector<KFParticleGpuTwoDaughterChannelResult>& channels =
        steering.RunDecayPlanBatch(
          0u, eventCount, cascade ? CascadeBenchmarkTaskCapacity : 1u);
      if (iteration + 1u == iterations) {
        const std::vector<KFParticleGpuSelectedChannelRange>& selected =
          steering.LastDecayPlanSelectedChannels();
        if (channels.size() != selected.size()) {
          throw std::runtime_error("batch default-V0 selection/channel result size mismatch");
        }
        for (std::size_t channelIndex = 0u; channelIndex < channels.size(); ++channelIndex) {
          const ChannelSummary summary = Summarize(channels[channelIndex], selected[channelIndex]);
          result.push_back(summary);
          AddWorkload(timing, summary);
        }
        AddTopology(timing, buffers);
        AddGenerationRouting(
          timing, steering.LastTwoDaughterRoutingMonitorData());
        if (cascade) {
          for (const auto& channel : steering.LastV0TrackCascadeResults()) {
            ChannelSummary summary;
            summary.totalPairs = channel.totalPairs;
            summary.acceptedTasks = channel.acceptedTasks;
            summary.storedTasks = channel.storedTasks;
            summary.candidates = channel.constructedCandidates;
            summary.overflowFlags = channel.generationCandidates.overflowFlags;
            result.push_back(summary);
            timing.cascadePairs += summary.totalPairs;
            timing.cascadeCandidates += summary.candidates;
            timing.overflowFlags |= summary.overflowFlags;
          }
          AddRouting(timing, steering.LastV0TrackRoutingMonitorData());
          AddGraphExecution(
            timing, steering.LastGraphExecutionMonitorData());
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

  bool SameWorkload(const std::vector<ChannelSummary>& first,
                    const std::vector<ChannelSummary>& second)
  {
    const auto sum = [](const std::vector<ChannelSummary>& channels) {
      ChannelSummary total;
      for (const auto& channel : channels) {
        total.totalPairs += channel.totalPairs;
        total.acceptedTasks += channel.acceptedTasks;
        total.storedTasks += channel.storedTasks;
        total.candidates += channel.candidates;
        total.selectedCandidates += channel.selectedCandidates;
        total.overflowFlags |= channel.overflowFlags;
      }
      return total;
    };
    const ChannelSummary lhs = sum(first);
    const ChannelSummary rhs = sum(second);
    return lhs.totalPairs == rhs.totalPairs && lhs.acceptedTasks == rhs.acceptedTasks
           && lhs.storedTasks == rhs.storedTasks && lhs.candidates == rhs.candidates
           && lhs.selectedCandidates == rhs.selectedCandidates
           && lhs.overflowFlags == rhs.overflowFlags;
  }

  RoutingModeTiming RunRoutingModeBenchmark(
    KFParticleGpuSteering& steering,
    KFParticleGpuBufferManager& buffers,
    unsigned int iterations,
    KFParticleGpuV0TrackRoutingMode mode)
  {
    FillCascadeEvents(buffers, 1u);
    steering.RunDecayPlan(0u, CascadeBenchmarkTaskCapacity);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    if (events.size() != 1u || events[0].candidates.Empty()
        || events[0].selectedCandidates.size
             != CascadeBenchmarkSelectedCandidates
        || events[0].selectedCandidates.overflowFlags != 0u) {
      std::cerr << "ROUTING_PREP_ERROR"
                << " events=" << events.size();
      if (!events.empty()) {
        std::cerr << " candidates=" << events[0].candidates.size
                  << " selected=" << events[0].selectedCandidates.size
                  << " selected_overflow="
                  << events[0].selectedCandidates.overflowFlags;
      }
      std::cerr << '\n';
      throw std::runtime_error("routing benchmark could not prepare a selected V0 generation");
    }
    const KFParticleGpuCandidateRange rawCandidates = events[0].candidates;
    const KFParticleGpuSelectedCandidateRange selected =
      events[0].selectedCandidates;
    const unsigned int routedTaskCapacity =
      buffers.Capacities().candidates - rawCandidates.End();
    if (routedTaskCapacity == 0u) {
      throw std::runtime_error("routing benchmark has no candidate capacity for cascades");
    }

    RoutingModeTiming timing;
    const auto runOnce = [&]() {
      KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
      candidates.SizeData()[0] = rawCandidates.End();
      candidates.Daughters().SizeData()[0] = rawCandidates.DaughterEnd();
      candidates.OverflowFlagsData()[0] = 0u;
      buffers.UploadCandidates();
      return steering.RunV0TrackFusedStage(
        0u, selected, routedTaskCapacity, mode);
    };

    // Load the selected XPU action and settle persistent allocations before timing.
    runOnce();
    const auto started = std::chrono::steady_clock::now();
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
      const KFParticleGpuV0TrackFusedResult result = runOnce();
      timing.status = result.routing;
      timing.candidates = result.candidates.size;
      timing.daughters = result.candidates.daughterSize;
    }
    timing.wallMilliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    if (timing.status.acceptedTasks != timing.status.storedTasks
        || timing.status.storedTasks != timing.candidates
        || timing.daughters != 3u * timing.candidates
        || timing.status.overflowFlags != 0u) {
      throw std::runtime_error("routing-mode benchmark produced inconsistent output");
    }
    return timing;
  }

  void PrintRoutingTiming(const char* mode,
                          const RoutingModeTiming& timing,
                          unsigned int iterations)
  {
    std::cout << std::fixed << std::setprecision(3)
              << "ROUTING_METRIC mode=" << mode
              << " wall_ms=" << timing.wallMilliseconds
                                / static_cast<double>(iterations)
              << " visited_pairs=" << timing.status.visitedPairs
              << " active_bits=" << timing.status.activeChannelBits
              << " accepted=" << timing.status.acceptedTasks
              << " stored=" << timing.status.storedTasks
              << " global_reservations=" << timing.status.blockReservations
              << " candidates=" << timing.candidates
              << " daughters=" << timing.daughters
              << " overflow=" << timing.status.overflowFlags << '\n';
  }

  void PrintTiming(const char* mode, const TimingSummary& timing, unsigned int iterations)
  {
    const double divisor = static_cast<double>(iterations);
    std::cout << std::fixed << std::setprecision(3)
              << "METRIC mode=" << mode
              << " wall_ms=" << timing.wallMilliseconds / divisor
              << " h2d_ms=" << timing.uploadMilliseconds / divisor
              << " construction_ms=" << timing.constructionMilliseconds / divisor
              << " topology_selection_ms=" << timing.selectionMilliseconds / divisor
              << " cascade_construction_ms=" << timing.cascadeConstructionMilliseconds / divisor
              << " d2h_ms=" << timing.downloadMilliseconds / divisor
              << " pairs=" << timing.totalPairs
              << " accepted_tasks=" << timing.acceptedTasks
              << " stored_tasks=" << timing.storedTasks
              << " raw=" << timing.rawCandidates
              << " selected=" << timing.selectedCandidates
              << " topology_candidates=" << timing.topologyCandidates
              << " topology_valid=" << timing.validTopologies
              << " topology_best_pv=" << timing.candidatesWithBestPrimaryVertex
              << " topology_rejected=" << timing.topologyRejected
              << " cascade_pairs=" << timing.cascadePairs
              << " cascade_candidates=" << timing.cascadeCandidates
              << " generation_groups=" << timing.generationGroupLaunches
              << " generation_descriptors=" << timing.generationDescriptorCount
              << " generation_selection_launches="
              << timing.generationSelectionLaunches
              << " generation_visited_pairs=" << timing.generationVisitedPairs
              << " generation_active_bits=" << timing.generationActiveChannelBits
              << " generation_accepted=" << timing.generationAcceptedTasks
              << " generation_stored=" << timing.generationStoredTasks
              << " generation_block_reservations="
              << timing.generationBlockReservations
              << " generation_daughters=" << timing.generationDaughters
              << " generation_selected="
              << timing.generationSelectedCandidates
              << " routing_groups=" << timing.routingGroupLaunches
              << " routing_descriptors=" << timing.routingDescriptorCount
              << " routing_visited_pairs=" << timing.routingVisitedPairs
              << " routing_active_bits=" << timing.routingActiveChannelBits
              << " routing_accepted=" << timing.routingAcceptedTasks
              << " routing_stored=" << timing.routingStoredTasks
              << " routing_block_reservations=" << timing.routingBlockReservations
              << " routing_daughters=" << timing.routingDaughters
              << " graph_groups=" << timing.graphGroupLaunches
              << " graph_descriptors=" << timing.graphDescriptorCount
              << " graph_visited=" << timing.graphVisitedCombinations
              << " graph_accepted=" << timing.graphAcceptedTasks
              << " graph_stored=" << timing.graphStoredTasks
              << " graph_constructed=" << timing.graphConstructedCandidates
              << " graph_rejected=" << timing.graphRejectedTasks
              << " graph_daughters=" << timing.graphDaughters
              << " overflow=" << timing.overflowFlags << '\n';
  }

  void PrintPerformance(const KFParticleGpuPerformanceSnapshot& snapshot)
  {
    if (!snapshot.enabled) { return; }
    std::cout << std::fixed << std::setprecision(6)
              << "PERFORMANCE_METRIC mode=batch"
              << " events=" << snapshot.events
              << " tracks=" << snapshot.tracks
              << " vertices=" << snapshot.vertices
              << " descriptor_groups=" << snapshot.descriptorGroups
              << " visited=" << snapshot.visitedCombinations
              << " accepted=" << snapshot.acceptedTasks
              << " stored=" << snapshot.storedTasks
              << " rejected=" << snapshot.rejectedTasks
              << " candidates=" << snapshot.rawCandidates
              << " selected=" << snapshot.selectedCandidates
              << " daughters=" << snapshot.daughters
              << " launches=" << snapshot.kernelLaunches
              << " queue_waits=" << snapshot.queueWaits
              << " capacity_growths=" << snapshot.capacityGrowths
              << " h2d_bytes=" << snapshot.hostToDeviceBytes
              << " d2h_bytes=" << snapshot.deviceToHostBytes
              << " allocated_high_water_bytes=" << snapshot.allocatedBytesHighWater
              << " mask_density=" << snapshot.maskDensity
              << " useful_work_per_launch=" << snapshot.usefulWorkPerLaunch
              << " candidate_pool_occupancy=" << snapshot.candidatePoolOccupancy
              << " overflow=" << snapshot.overflowFlags << '\n';
  }
}

int main()
{
  const unsigned int eventCount = ReadPositiveEnvironment("KFPARTICLE_GPU_BATCH_BENCHMARK_EVENTS", 4u);
  const unsigned int iterations = ReadPositiveEnvironment("KFPARTICLE_GPU_BATCH_BENCHMARK_ITERATIONS", 10u);
  const unsigned int warmupIterations =
    ReadPositiveEnvironment("KFPARTICLE_GPU_BATCH_BENCHMARK_WARMUP_ITERATIONS", 3u);

  KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
  KFParticleGpuRuntimeSettings settings;
  settings.device = KFPARTICLE_GPU_TEST_DEVICE;
  settings.initializeXpuIfNeeded = true;
  runtime.Initialize(settings);

  try {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    steering.SetPerformanceMonitoringEnabled(true);
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    ConfigureDefaultV0Plan(plan, KFGpuTransportFullField);

    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    std::vector<ChannelSummary> serial;
    std::vector<ChannelSummary> batch;

    // Warm-up creates the XPU image and persistent allocations outside timing.
    for (unsigned int iteration = 0u; iteration < warmupIterations; ++iteration) {
      FillDefaultV0Events(buffers, eventCount);
      steering.RunDecayPlanBatch(0u, eventCount, 1u);
    }

    const TimingSummary serialTiming = RunSerial(steering, buffers, eventCount, iterations, serial);
    const TimingSummary batchTiming = RunBatch(steering, buffers, eventCount, iterations, batch);
    const KFParticleGpuPerformanceSnapshot batchPerformance =
      steering.LastPerformanceSnapshot();
    if (!SameResults(serial, batch)) {
      throw std::runtime_error("serial and batch default-V0 channel summaries differ");
    }
    if (serialTiming.topologyCandidates != batchTiming.topologyCandidates
        || serialTiming.validTopologies != batchTiming.validTopologies
        || serialTiming.candidatesWithBestPrimaryVertex != batchTiming.candidatesWithBestPrimaryVertex) {
      throw std::runtime_error("serial and batch default-V0 topology summaries differ");
    }
    if (batchTiming.topologyCandidates != batchTiming.rawCandidates
        || batchTiming.validTopologies != batchTiming.rawCandidates
        || batchTiming.candidatesWithBestPrimaryVertex != batchTiming.rawCandidates) {
      throw std::runtime_error("default-V0 batch did not execute valid line-topology selection for every raw candidate");
    }
    if (serialTiming.generationVisitedPairs != batchTiming.generationVisitedPairs
        || serialTiming.generationActiveChannelBits
             != batchTiming.generationActiveChannelBits
        || serialTiming.generationAcceptedTasks
             != batchTiming.generationAcceptedTasks
        || serialTiming.generationStoredTasks
             != batchTiming.generationStoredTasks
        || serialTiming.generationSelectedCandidates
             != batchTiming.generationSelectedCandidates) {
      throw std::runtime_error(
        "serial and batch default-V0 generation-routing summaries differ");
    }
    if (batchTiming.generationGroupLaunches == 0u
        || batchTiming.generationDescriptorCount != 3u
        || batchTiming.generationSelectionLaunches != eventCount
        || batchTiming.generationAcceptedTasks
             != batchTiming.generationStoredTasks
        || batchTiming.generationStoredTasks != batchTiming.rawCandidates
        || batchTiming.generationBlockReservations == 0u
        || batchTiming.generationBlockReservations
             > batchTiming.generationAcceptedTasks
        || batchTiming.generationDaughters != 2u * batchTiming.rawCandidates
        || batchTiming.generationSelectedCandidates
             != batchTiming.selectedCandidates) {
      throw std::runtime_error(
        "block-scan default-V0 generation monitoring is inconsistent");
    }

    std::vector<ChannelSummary> constantByBatch;
    std::vector<ChannelSummary> approximateFullFieldBatch;
    ConfigureDefaultV0Plan(plan, KFGpuTransportFullFieldApprox);
    const TimingSummary approximateFullFieldTiming =
      RunBatch(steering, buffers, eventCount, iterations, approximateFullFieldBatch);
    if (approximateFullFieldTiming.overflowFlags != 0u) {
      throw std::runtime_error("approximate full-field baseline overflowed its bounded candidate pools");
    }

    ConfigureDefaultV0Plan(plan, KFGpuTransportConstantBy);
    const TimingSummary constantByTiming =
      RunBatch(steering, buffers, eventCount, iterations, constantByBatch);
    if (constantByTiming.overflowFlags != 0u) {
      throw std::runtime_error("constant-By baseline overflowed its bounded candidate pools");
    }

    ConfigureDefaultV0Plan(plan, KFGpuTransportFullField, true);
    std::vector<ChannelSummary> cascadeSerial;
    std::vector<ChannelSummary> cascadeBatch;
    // The controlled cascade fixture isolates one event. Multi-event cascade
    // batching is intentionally qualified by the dedicated lifecycle path.
    const unsigned int cascadeEventCount = 1u;
    const TimingSummary cascadeSerialTiming =
      RunSerial(steering, buffers, cascadeEventCount, iterations, cascadeSerial, true);
    const TimingSummary cascadeBatchTiming =
      RunBatch(steering, buffers, cascadeEventCount, iterations, cascadeBatch, true);
    if (!SameWorkload(cascadeSerial, cascadeBatch)
        || cascadeSerialTiming.cascadePairs != cascadeBatchTiming.cascadePairs
        || cascadeSerialTiming.cascadeCandidates != cascadeBatchTiming.cascadeCandidates
        || cascadeSerialTiming.routingAcceptedTasks
             != cascadeBatchTiming.routingAcceptedTasks
        || cascadeSerialTiming.routingStoredTasks
             != cascadeBatchTiming.routingStoredTasks
        || cascadeSerialTiming.routingVisitedPairs
             != cascadeBatchTiming.routingVisitedPairs
        || cascadeSerialTiming.routingActiveChannelBits
             != cascadeBatchTiming.routingActiveChannelBits
        || cascadeSerialTiming.graphGroupLaunches
             != cascadeBatchTiming.graphGroupLaunches
        || cascadeSerialTiming.graphVisitedCombinations
             != cascadeBatchTiming.graphVisitedCombinations
        || cascadeSerialTiming.graphAcceptedTasks
             != cascadeBatchTiming.graphAcceptedTasks
        || cascadeSerialTiming.graphStoredTasks
             != cascadeBatchTiming.graphStoredTasks
        || cascadeSerialTiming.graphConstructedCandidates
             != cascadeBatchTiming.graphConstructedCandidates
        || cascadeSerialTiming.graphRejectedTasks
             != cascadeBatchTiming.graphRejectedTasks
        || cascadeSerialTiming.graphDaughters
             != cascadeBatchTiming.graphDaughters) {
      throw std::runtime_error("serial and batch V0-track cascade summaries differ");
    }
    if (cascadeBatchTiming.routingGroupLaunches == 0u
        || cascadeBatchTiming.routingDescriptorCount != 4u
        || cascadeBatchTiming.routingAcceptedTasks
             != cascadeBatchTiming.routingStoredTasks
        || cascadeBatchTiming.routingStoredTasks
             != cascadeBatchTiming.cascadeCandidates
        || cascadeBatchTiming.routingBlockReservations == 0u
        || cascadeBatchTiming.routingBlockReservations
             > cascadeBatchTiming.routingAcceptedTasks
        || cascadeBatchTiming.routingDaughters
             != 3u * cascadeBatchTiming.cascadeCandidates) {
      std::cerr << "ROUTING_MONITOR_ERROR"
                << " groups=" << cascadeBatchTiming.routingGroupLaunches
                << " descriptors=" << cascadeBatchTiming.routingDescriptorCount
                << " accepted=" << cascadeBatchTiming.routingAcceptedTasks
                << " stored=" << cascadeBatchTiming.routingStoredTasks
                << " reservations=" << cascadeBatchTiming.routingBlockReservations
                << " candidates=" << cascadeBatchTiming.cascadeCandidates
                << " daughters=" << cascadeBatchTiming.routingDaughters
                << " overflow=" << cascadeBatchTiming.overflowFlags << '\n';
      throw std::runtime_error("block-scan cascade routing monitoring is inconsistent");
    }
    if (cascadeBatchTiming.graphGroupLaunches == 0u
        || cascadeBatchTiming.graphDescriptorCount != 2u
        || cascadeBatchTiming.graphVisitedCombinations == 0u
        || cascadeBatchTiming.graphStoredTasks == 0u
        || cascadeBatchTiming.graphAcceptedTasks
             != cascadeBatchTiming.graphStoredTasks
        || cascadeBatchTiming.graphConstructedCandidates
             + cascadeBatchTiming.graphRejectedTasks
             != cascadeBatchTiming.graphStoredTasks
        || cascadeBatchTiming.overflowFlags != 0u) {
      const KFParticleGpuCandidatePoolView graphCandidates =
        buffers.HostCandidates();
      const KFParticleGpuSelectedCandidateIndexView graphSelected =
        buffers.HostSelectedCandidates();
      std::cerr << "GRAPH_MONITOR_ERROR"
                << " groups=" << cascadeBatchTiming.graphGroupLaunches
                << " descriptors=" << cascadeBatchTiming.graphDescriptorCount
                << " visited=" << cascadeBatchTiming.graphVisitedCombinations
                << " accepted=" << cascadeBatchTiming.graphAcceptedTasks
                << " stored=" << cascadeBatchTiming.graphStoredTasks
                << " constructed=" << cascadeBatchTiming.graphConstructedCandidates
                << " rejected=" << cascadeBatchTiming.graphRejectedTasks
                << " daughters=" << cascadeBatchTiming.graphDaughters
                << " pool_size=" << graphCandidates.Size()
                << " pool_capacity=" << graphCandidates.Capacity()
                << " daughter_size=" << graphCandidates.Daughters().Size()
                << " daughter_capacity="
                << graphCandidates.Daughters().Capacity()
                << " selected_size=" << graphSelected.Size()
                << " selected_capacity=" << graphSelected.Capacity()
                << " selected_overflow=" << graphSelected.OverflowFlags()
                << " generation_pool_overflow="
                << cascadeBatchTiming.generationPoolOverflowFlags
                << " generation_routing_overflow="
                << cascadeBatchTiming.generationRoutingOverflowFlags
                << " cascade_routing_overflow="
                << cascadeBatchTiming.cascadeRoutingOverflowFlags
                << " graph_overflow="
                << cascadeBatchTiming.graphOverflowFlags
                << " overflow=" << cascadeBatchTiming.overflowFlags << '\n';
      throw std::runtime_error(
        "device graph scheduler serial/batch monitoring is inconsistent");
    }
    const RoutingModeTiming atomicRouting = RunRoutingModeBenchmark(
      steering, buffers, iterations, KFGpuV0TrackRoutingAtomic);
    const RoutingModeTiming blockScanRouting = RunRoutingModeBenchmark(
      steering, buffers, iterations, KFGpuV0TrackRoutingBlockScan);
    if (atomicRouting.status.visitedPairs != blockScanRouting.status.visitedPairs
        || atomicRouting.status.activeChannelBits
             != blockScanRouting.status.activeChannelBits
        || atomicRouting.status.acceptedTasks
             != blockScanRouting.status.acceptedTasks
        || atomicRouting.status.storedTasks != blockScanRouting.status.storedTasks
        || atomicRouting.candidates != blockScanRouting.candidates
        || atomicRouting.daughters != blockScanRouting.daughters
        || atomicRouting.status.blockReservations
             != atomicRouting.status.acceptedTasks
        || blockScanRouting.status.blockReservations == 0u
        || blockScanRouting.status.blockReservations
             > atomicRouting.status.blockReservations) {
      throw std::runtime_error("atomic and block-scan routing modes disagree");
    }

    std::cout << "PASS kfparticle-gpu-batch-benchmark"
              << " - events=" << eventCount
              << " iterations=" << iterations
              << " warmup_iterations=" << warmupIterations
              << " channels=K0S,Lambda,anti-Lambda"
              << ",Xi,anti-Xi,Omega,anti-Omega"
              << ",composite-composite,unary-final"
              << " transport=full-field"
              << " serial_batch_equivalent=1\n";
    PrintTiming("serial", serialTiming, iterations);
    PrintTiming("batch", batchTiming, iterations);
    PrintTiming("full-field-approx-batch", approximateFullFieldTiming, iterations);
    PrintTiming("constant-by-batch", constantByTiming, iterations);
    PrintTiming("cascade-serial", cascadeSerialTiming, iterations);
    PrintTiming("cascade-batch", cascadeBatchTiming, iterations);
    PrintPerformance(batchPerformance);
    PrintRoutingTiming("atomic", atomicRouting, iterations);
    PrintRoutingTiming("block-scan", blockScanRouting, iterations);

    plan.Clear();
    runtime.Finalize();
    return 0;
  }
  catch (...) {
    runtime.Finalize();
    throw;
  }
}
