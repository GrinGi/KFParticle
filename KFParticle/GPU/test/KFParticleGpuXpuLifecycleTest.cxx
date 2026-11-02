/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "execution/KFParticleGpuBufferManager.h"
#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuChannelRouting.h"
#include "planning/KFParticleGpuCpuChannelCatalogue.h"
#include "planning/KFParticleGpuDecayGraphPlan.h"
#include "planning/KFParticleGpuDecayPlan.h"
#include "execution/KFParticleGpuDeviceStorage.h"
#include "algorithms/KFParticleGpuGraphOperations.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "execution/KFParticleGpuKernelState.h"
#include "execution/KFParticleGpuKernels.h"
#include "algorithms/KFParticleGpuMath.h"
#include "data/KFParticleGpuMaterializer.h"
#include "diagnostics/KFParticleGpuParity.h"
#include "execution/KFParticleGpuRuntime.h"
#include "execution/KFParticleGpuRuntimeService.h"
#include "algorithms/KFParticleGpuSelection.h"
#include "execution/KFParticleGpuSteering.h"
#include "algorithms/KFParticleGpuTwoDaughter.h"
#include "planning/KFParticleGpuTwoDaughterRouting.h"
#include "algorithms/KFParticleGpuV0Track.h"

#include <xpu/host.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef KFPARTICLE_GPU_TEST_DEVICE
#define KFPARTICLE_GPU_TEST_DEVICE "cpu"
#endif

static_assert(std::is_trivially_copyable<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory state must remain trivially copyable");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernels>::value,
              "KFParticle GPU constant-memory state must not require dynamic initialization");
static_assert(std::is_trivially_copyable<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must remain a flat device ABI");
static_assert(std::is_trivially_default_constructible<KFParticleGpuKernelState>::value,
              "KFParticle GPU kernel state must not require dynamic initialization");
static_assert(std::is_trivially_copyable<KFParticleGpuMeasurement>::value,
              "KFParticle GPU measurement must remain a flat device value type");
static_assert(std::is_trivially_copyable<KFParticleGpuV0SelectionObservables>::value,
              "KFParticle GPU selection observables must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0SelectionResult>::value,
              "KFParticle GPU selection results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0LineTopologyResult>::value,
              "KFParticle GPU line-topology results must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackChannel>::value,
              "KFParticle GPU V0-track channels must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackTask>::value,
              "KFParticle GPU V0-track tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackInputView>::value,
              "KFParticle GPU V0-track input must remain a non-owning device view");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackLineage>::value,
              "KFParticle GPU V0-track lineage must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuChannelMask>::value,
              "KFParticle GPU channel masks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingDescriptor>::value,
              "KFParticle GPU routing descriptors must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackExecutionGroup>::value,
              "KFParticle GPU execution groups must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutedTask>::value,
              "KFParticle GPU routed tasks must remain flat device values");
static_assert(std::is_trivially_copyable<KFParticleGpuV0TrackRoutingStatus>::value,
              "KFParticle GPU routing status must remain a flat device value");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutingDescriptor>::value,
              "KFParticle GPU two-daughter routing descriptors must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterRoutedTask>::value,
              "KFParticle GPU two-daughter routed tasks must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphNode>::value,
              "KFParticle GPU graph nodes must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphExecutionGroup>::value,
              "KFParticle GPU graph groups must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphFamilyCoverage>::value,
              "KFParticle GPU graph coverage must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuDecayGraphView>::value,
              "KFParticle GPU graph view must remain a flat non-owning ABI");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationDescriptor>::value,
              "KFParticle GPU graph operation descriptors must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationTask>::value,
              "KFParticle GPU graph operation tasks must remain flat values");
static_assert(std::is_trivially_copyable<KFParticleGpuGraphOperationResult>::value,
              "KFParticle GPU graph operation results must remain flat values");

namespace
{
  void Pass(const char* name, const char* details);

// Keep the shared fixtures and ordered XPU lifecycle in one translation unit.
#include "lifecycle/InputAndRouting.inc"
#include "lifecycle/MathAndSelection.inc"
#include "lifecycle/TwoDaughter.inc"
#include "lifecycle/DecayGraph.inc"
#include "lifecycle/Publication.inc"
}  // namespace


void TestTaskMapAndCorrelationProducts()
{
  KFParticleGpuTransactionDesc desc;
  desc.domains = {1u, 2u};
  KFParticleGpuDomainDesc domains[3] = {};
  KFParticleGpuDomainStatus statuses[2] = {};
  domains[1].statusIndex = 0u;
  domains[1].taskReservation = {7u, 128u};
  domains[2].statusIndex = 1u;
  domains[2].taskReservation = {135u, 128u};
  KFParticleGpuTransactionView view(&desc, domains, nullptr, nullptr, statuses, 3u, 0u, 2u);
  for (unsigned int count : {0u, 1u, 63u, 64u, 65u, 128u, 129u}) {
    statuses[0].storedTasks = count;
    statuses[1].storedTasks = 1u;
    const auto tiles = KFParticleGpuBuildTaskTiles(view, 263u);
    const unsigned int retained = std::min(count, 128u);
    assert(tiles.size() == KFParticleGpuPairTileCount(retained) + 1u);
    unsigned int covered = 0u;
    for (const auto& tile : tiles) {
      if (tile.domainIndex != 1u) { assert(tile.domainIndex == 2u); continue; }
      const unsigned int offset = tile.firstTask - 7u;
      assert(offset == covered);
      covered += std::min(64u, retained - offset);
    }
    assert(covered == retained && tiles.back().firstTask == 135u);
  }
  domains[1].statusIndex = 2u;
  bool rejected = false;
  try { (void)KFParticleGpuBuildTaskTiles(view, 263u); }
  catch (const std::logic_error&) { rejected = true; }
  assert(rejected);
  domains[1].statusIndex = 0u;
  desc.domains = {3u, 1u};
  rejected = false;
  try { (void)KFParticleGpuBuildTaskTiles(view, 263u); }
  catch (const std::logic_error&) { rejected = true; }
  assert(rejected);

  for (unsigned int sample = 0; sample < 100u; ++sample) {
    KFParticleGpuFitState first, second;
    float a[36], b[36], c[36], d[36];
    for (int i = 0; i < 36; ++i) {
      const float value = static_cast<float>((i * 13 + sample * 7) % 31) - 15.f;
      a[i] = value * 0.01f; b[i] = value * 0.03f;
      c[i] = value * -0.02f; d[i] = value * 0.04f;
      first.Covariance(i) = value * 0.001f;
      second.Covariance(i) = value * 0.002f;
    }
    KFParticleGpuMeasurement compact, reference;
    KFParticleGpuMath::FullFieldDcaCorrelation<true>(first, second, a, b, c, d, compact);
    KFParticleGpuMath::FullFieldDcaCorrelation<false>(first, second, a, b, c, d, reference);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        assert(compact.Correlation(i, j) == reference.Correlation(i, j));
      }
    }
  }
  Pass("task-map-and-correlation-products",
       "empty/tail/offset/overflow map boundaries and 100 exact reduced/full matrix comparisons");
}

void TestHostAllocationTiming(xpu::queue& queue)
{
  KFParticleGpuBufferManager local(queue);
  local.BeginAllocationTrace(true);
  KFParticleGpuBufferCapacities request;
  request.tracks = 4u;
  local.EnsureCapacity(request);
  const auto initial = local.HostTiming();
  assert(initial.initialAllocations > 0u);
  assert(initial.replacementAllocations == 0u);
  assert(local.AllocationTrace().size() == initial.initialAllocations);
  double recordedMilliseconds = 0.;
  for (const auto& record : local.AllocationTrace()) {
    assert(record.bytes > 0u && record.elements > 0u && !record.replacement);
    assert(std::string(record.phase) == "input-routing");
    assert(record.buffer[0] != '\0');
    recordedMilliseconds += record.milliseconds;
  }
  assert(std::abs(recordedMilliseconds - initial.initialAllocationMilliseconds) < 1.e-6);
  const auto initialRecords = local.AllocationTrace().size();
  local.EnsureCapacity(request);
  auto reused = local.HostTiming().Since(initial);
  assert(reused.initialAllocations == 0u && reused.replacementAllocations == 0u);
  assert(reused.initialAllocationMilliseconds == 0.);
  assert(local.AllocationTrace().size() == initialRecords);
  local.SetAllocationTracePhase("test-replacement");
  request.tracks = 8u;
  local.EnsureCapacity(request);
  const auto replacement = local.HostTiming().Since(initial);
  assert(replacement.initialAllocations == 0u);
  assert(replacement.replacementAllocations > 0u);
  assert(replacement.replacementAllocationMilliseconds >= 0.);
  for (std::size_t index = initialRecords; index < local.AllocationTrace().size(); ++index) {
    assert(local.AllocationTrace()[index].replacement);
    assert(std::string(local.AllocationTrace()[index].phase) == "test-replacement");
  }
  local.BeginAllocationTrace(false);
  request.tracks = 16u;
  local.EnsureCapacity(request);
  assert(local.AllocationTrace().empty());
  Pass("allocation-timing-contract", "first IO allocations, replacements and unchanged reuse are distinct");
}


void TestDeviceWorkspaceResidency(xpu::queue& queue)
{
  KFParticleGpuBufferManager local(queue);
  local.DownloadCandidates();
  local.DownloadTwoDaughterRoutedTasks();
  local.DownloadV0TrackRoutedTasks();
  local.DownloadCandidateDescriptorIndices();
  local.DownloadV0SelectionResults();
  assert(local.MemoryUsage().deviceBytes == 0u);
  assert(local.MemoryUsage().pinnedHostBytes == 0u);
  local.BeginAllocationTrace(true);
  KFParticleGpuBufferCapacities request;
  request.tracks = 4u;
  request.candidates = 65u;
  request.twoDaughterTasks = 7u;
  request.v0TrackTasks = 7u;
  request.twoDaughterRoutedTasks = 7u;
  request.v0TrackRoutedTasks = 7u;
  request.graphOperationTasks = 7u;
  request.graphOperationDescriptors = 3u;
  local.EnsureCapacity(request);
  const auto& storage = local.Storage();
  const auto deviceOnly = [](const auto& buffer) {
    assert(buffer.get() != nullptr);
    const xpu::buffer_prop properties(buffer);
    assert(properties.type() == xpu::buf_device);
    assert(properties.h_ptr() == nullptr);
    assert(properties.d_ptr() != nullptr);
  };
  deviceOnly(storage.fTwoDaughterTasks);
  deviceOnly(storage.fV0TrackTasks);
  deviceOnly(storage.fGraphOperationTasks);
  deviceOnly(storage.fGraphOperationResults);
  deviceOnly(storage.fTwoDaughterRoutedTasks);
  deviceOnly(storage.fV0TrackRoutedTasks);
  deviceOnly(storage.fCandidateRoutingDescriptorIndices);
  deviceOnly(storage.fV0SelectionResults);
  deviceOnly(storage.fGraphOperationVisitedCombinations);
  deviceOnly(storage.fGraphOperationAcceptedTasks);
  deviceOnly(storage.fGraphOperationStoredTasks);
  deviceOnly(storage.fGraphOperationConstructedCandidates);
  deviceOnly(storage.fGraphOperationRejectedTasks);
  deviceOnly(storage.fGraphOperationOverflowFlags);
  deviceOnly(storage.fGraphOperationChannelVisitedCounters);
  deviceOnly(storage.fGraphOperationChannelAcceptedCounters);
  deviceOnly(storage.fGraphOperationChannelStoredCounters);
  deviceOnly(storage.fGraphOperationChannelConstructedCounters);
  deviceOnly(storage.fGraphOperationChannelRejectedCounters);
  assert(xpu::buffer_prop(storage.fTrackParameters).type() == xpu::buf_io);
  const auto resident = local.MemoryUsage();
  assert(local.AllocatedBytes() == resident.deviceBytes);
  local.ResetV0SelectionResults();
  assert(local.SelectionResetLaunches() == 1u);
  assert(local.MemoryUsage().pinnedHostBytes == resident.pinnedHostBytes);
  local.DownloadV0SelectionResults();
  const auto selection = local.HostV0SelectionResults();
  for (unsigned int i = 0u; i < request.candidates; ++i) {
    assert(selection.Result(i).bestPrimaryVertexIndex == -1);
    assert(selection.Result(i).selectionClass == KFParticleGpuV0SelectionResult{}.selectionClass);
  }
  const auto mirrored = local.MemoryUsage();
  assert(mirrored.deviceBytes == resident.deviceBytes);
  assert(mirrored.pinnedHostBytes == resident.pinnedHostBytes
    + request.candidates * sizeof(KFParticleGpuV0SelectionResult));
  local.DownloadV0SelectionResults();
  local.EnsureCapacity(request);
  assert(local.MemoryUsage().pinnedHostBytes == mirrored.pinnedHostBytes);
  assert(local.HostV0SelectionResults().Data() == selection.Data());
  request.candidates = 129u;
  local.EnsureCapacity(request);
  const auto beforeReset = local.MemoryUsage();
  local.ResetV0SelectionResults();
  assert(local.MemoryUsage().pinnedHostBytes == beforeReset.pinnedHostBytes);
  local.DownloadV0SelectionResults();
  for (unsigned int i = 0u; i < request.candidates; ++i) {
    assert(local.HostV0SelectionResults().Result(i).bestPrimaryVertexIndex == -1);
  }
  local.HostTwoDaughterRoutedTasks();
  local.HostV0TrackRoutedTasks();
  local.HostCandidateDescriptorIndices();
  local.HostGraphOperationResults();
  bool sawDevice = false, sawPinned = false;
  for (const auto& record : local.AllocationTrace()) {
    sawDevice = sawDevice || std::string(record.memory) == "device";
    sawPinned = sawPinned || std::string(record.memory) == "pinned";
  }
  assert(sawDevice && sawPinned);
  Pass("device-workspace-residency",
       "19 work arrays are device-only; diagnostics allocate host mirrors lazily; reset, growth and reuse preserve defaults");
}


void TestBatchWithoutLegacyTasks(KFParticleGpuRuntime& runtime)
{
  auto& steering = runtime.GetSteering();
  auto& buffers = steering.GetBuffers();
  auto& plan = steering.GetDecayPlan();
  plan.Clear();
  plan.AddTwoDaughterChannel(MakePionPairChannel(91u));
  KFParticleGpuBufferCapacities request;
  request.candidates = 32u;
  request.daughterIds = 256u;
  request.selectedCandidates = 32u;
  request.transactions = 1u;
  request.domains = 2u;
  request.domainStatuses = 2u;
  request.workRanges = 2u;
  buffers.EnsureCapacity(request);
  FillBatchSelectionFixture(buffers);
  buffers.SetTransactionSizes(1u, 2u, 2u, 2u);
  auto transaction = buffers.HostTransaction();
  *transaction.Transaction() = KFParticleGpuTransactionDesc{};
  transaction.Transaction()->transactionId = 12345u;
  transaction.Transaction()->decayPlanRevision = plan.Revision();
  transaction.Transaction()->domains = {0u, 2u};
  transaction.Transaction()->workRanges = {0u, 2u};
  for (unsigned int e = 0u; e < 2u; ++e) {
    auto& domain = transaction.Domains()[e];
    domain = KFParticleGpuDomainDesc{};
    domain.sourceId = buffers.HostEvents()[e].eventId;
    domain.inputDescriptorIndex = e;
    domain.sourceTracks = {2u * e, 2u};
    domain.taskReservation = {e, 1u};
    domain.candidateReservation = {16u * e, 16u};
    domain.daughterReservation = {128u * e, 128u};
    domain.selectedReservation = {16u * e, 16u};
    domain.statusIndex = e;
  }
  steering.SetPerformanceMonitoringEnabled(true);
  std::uint64_t compactSnapshotBytes = 0u;
  int monitoredPdg[2] = {0, 0};
  int monitoredDaughters[2][2] = {{0, 0}, {0, 0}};
  for (unsigned int pass = 0u; pass < 3u; ++pass) {
    buffers.BeginAllocationTrace(true);
    const auto mode = pass < 2u ? KFGpuPublishableOutput : KFGpuFullDiagnosticOutput;
    const auto& results = steering.RunDecayPlanBatch(0u, 2u, 2u, mode);
    if (mode == KFGpuPublishableOutput) {
      assert(results.empty());
      assert(steering.LastDecayPlanSelectedChannels().empty());
      assert(steering.LastGraphChannelMonitorData().empty());
    }
    else {
      assert(results.size() == 2u);
      assert(results[0].constructedCandidates == 1u);
      assert(results[1].constructedCandidates == 1u);
    }
    const auto& performance = steering.LastPerformanceSnapshot();
    assert(performance.enabled);
    assert(performance.compactOutput == (mode == KFGpuPublishableOutput));
    assert(performance.transactionStatusBytes > 0u);
    if (mode == KFGpuPublishableOutput) {
      if (compactSnapshotBytes == 0u) {
        compactSnapshotBytes = performance.retainedSnapshotBytes;
      }
      else {
        assert(performance.retainedSnapshotBytes == compactSnapshotBytes);
      }
    }
    else {
      assert(performance.retainedSnapshotBytes > compactSnapshotBytes);
    }
    assert(buffers.Capacities().twoDaughterTasks == 0u);
    assert(buffers.Storage().fTwoDaughterTasks.get() == nullptr);
    const auto candidates = buffers.HostCandidates();
    const auto& events = steering.LastDecayPlanEventResults();
    assert(events.size() == 2u);
    for (unsigned int e = 0u; e < 2u; ++e) {
      assert(events[e].candidates.size == 1u);
      const auto index = events[e].candidates.offset;
      assert(index < candidates.Size());
      assert(candidates.Metadata().EventIndex(index) == e);
      assert(candidates.Metadata().DaughterCount(index) == 2u);
      if (pass == 0u) {
        monitoredPdg[e] = candidates.Metadata().Pdg(index);
        const unsigned int daughterOffset =
          candidates.Metadata().DaughterOffset(index);
        monitoredDaughters[e][0] =
          candidates.Daughters().SourceId(daughterOffset);
        monitoredDaughters[e][1] =
          candidates.Daughters().SourceId(daughterOffset + 1u);
      }
    }
    bool selectionMirror = false;
    for (const auto& record : buffers.AllocationTrace()) {
      const std::string name(record.buffer);
      if (pass < 2u) { assert(name.find("fHost") != 0u); }
      selectionMirror = selectionMirror || name == "fHostV0SelectionResults";
    }
    if (pass == 2u) { assert(selectionMirror); }
  }
  steering.SetPerformanceMonitoringEnabled(false);
  const auto& productionResults = steering.RunDecayPlanBatch(
    0u, 2u, 2u, KFGpuPublishableOutput);
  assert(productionResults.empty());
  assert(!steering.LastPerformanceSnapshot().enabled);
  const auto productionCandidates = buffers.HostCandidates();
  const auto& productionEvents = steering.LastDecayPlanEventResults();
  assert(productionEvents.size() == 2u);
  transaction = buffers.HostTransaction();
  for (unsigned int e = 0u; e < 2u; ++e) {
    const auto& domain = transaction.Domains()[e];
    const auto& status = transaction.DomainStatuses()[domain.statusIndex];
    assert(status.tasks == 1u);
    assert(status.storedTasks == 1u);
    assert(status.candidates == 1u);
    assert(status.daughters == 2u);
    assert(status.overflowFlags == 0u);
    assert(status.visitedPairs == 0u);
    assert(status.activeChannelBits == 0u);
    assert(status.blockReservations == 0u);
    const auto& channel =
      transaction.DomainChannelStatuses()[domain.channelStatuses.offset];
    assert(channel.visited == 0u);
    assert(channel.accepted == 0u);
    assert(channel.stored == 0u);
    assert(channel.constructed == 1u);
    assert(productionEvents[e].candidates.size == 1u);
    const unsigned int candidate = productionEvents[e].candidates.offset;
    assert(productionCandidates.Metadata().Pdg(candidate) == monitoredPdg[e]);
    assert(productionCandidates.Metadata().DaughterCount(candidate) == 2u);
    const unsigned int daughterOffset =
      productionCandidates.Metadata().DaughterOffset(candidate);
    assert(productionCandidates.Daughters().SourceId(daughterOffset)
           == monitoredDaughters[e][0]);
    assert(productionCandidates.Daughters().SourceId(daughterOffset + 1u)
           == monitoredDaughters[e][1]);
  }
  Pass("batch-without-legacy-task-pool",
       "two-event compact publication preserves output identity with diagnostic counters disabled while full diagnostics retain the oracle and lazy host staging");
}

int main()
{
  KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
  KFParticleGpuRuntimeSettings settings;
  settings.device = KFPARTICLE_GPU_TEST_DEVICE;
  settings.initializeXpuIfNeeded = true;
  settings.verbose = false;

  runtime.Initialize(settings);
  assert(runtime.IsInitialized());
  assert(runtime.InitializedXpu());
  xpu::queue* queue = &runtime.GetQueue();
  assert(queue == &runtime.GetQueue());
  Pass("runtime", "XPU runtime initialized once and exposes a stable process queue");
  auto& service = kfparticle::gpu::GpuRuntimeService::Instance();
  KFParticleGpuRuntimeSettings attachment;
  attachment.device.clear();
  attachment.initializeXpuIfNeeded = false;
  service.Initialize(attachment);
  bool wrongThreadRejected = false;
  std::thread wrongOwner([&] {
    try { service.Initialize(attachment); }
    catch (const std::logic_error&) { wrongThreadRejected = true; }
  });
  wrongOwner.join();
  assert(wrongThreadRejected);
  {
    auto owner = service.Acquire();
    auto moved = std::move(owner);
    bool rejected = false;
    try { owner.Steering(); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
    owner = std::move(moved);
    assert(&owner.Steering() == &runtime.GetSteering());
    assert(&runtime.GetQueue() == queue);
  }
  try {
    auto owner = service.Acquire();
    throw std::runtime_error("test transaction unwind");
  }
  catch (const std::runtime_error&) {}
  { auto owner = service.Acquire(); assert(&owner.Steering() == &runtime.GetSteering()); }
  Pass("runtime-service", "library transaction moves and exception unwinding preserve queue ownership");
  TestHostAllocationTiming(runtime.GetQueue());
  TestDeviceWorkspaceResidency(runtime.GetQueue());
  TestTaskMapAndCorrelationProducts();

  xpu::buffer<unsigned int> launchMarker(1, xpu::buf_device);
  unsigned int launchMarkerHost = 0;
  runtime.GetQueue().memcpy(launchMarker.get(), &launchMarkerHost, sizeof(launchMarkerHost));

  runtime.GetQueue().launch<KFParticleGpuLaunchSmoke>(
    xpu::n_threads(1), launchMarker.get());
  runtime.GetQueue().wait();
  runtime.GetQueue().memcpy(&launchMarkerHost, launchMarker.get(), sizeof(launchMarkerHost));
  runtime.GetQueue().wait();
  assert(launchMarkerHost == 0x4b465047u);
  Pass("kfparticle-image", "KFParticle GPU image kernel launches and writes a device marker");

  KFParticleGpuBufferManager& buffers = runtime.GetSteering().GetBuffers();
  KFParticleGpuBufferCapacities capacities;
  capacities.tracks = 8;
  capacities.vertices = 3;
  capacities.events = 2;
  capacities.candidates = 6;
  capacities.daughterIds = 4;
  capacities.selectedCandidates = 1;
  capacities.nonhomogeneousField = true;
  buffers.EnsureCapacity(capacities);
  FillInput(buffers, capacities);
  Pass("input-packing", "host buffers allocated and filled with packed tracks, vertices, events, and field data");
  TestHostViews(buffers);
  TestDeviceInputProbe(runtime, buffers);
  TestCandidateReset(buffers);
  TestV0TrackRoutingPlan(buffers);
  TestTwoDaughterRoutingPlan(buffers);
  TestCpuFinderChannelCatalogue();
  TestCompleteCpuFinderManifest(buffers);
  TestDecayGraphContract(buffers);
  TestKernelStateContract(buffers);
  TestVisibleDeviceStorage(buffers);
  TestTimeDomainSeam(buffers);
  TestPublishedKernelState(runtime, buffers);
  TestParityAndPromotionContract();
  TestDecayPlanSelectedOutputTruncation(runtime, buffers);
  TestDecayPlanDataTypes();
  TestTwoDaughterPairTilePlan();
  TestV0SelectionResultDataTypes(runtime);
  TestPrimaryVertexTopologyObservables();
  TestV0LineTopologyContract();
  TestV0SelectionDecision();
  TestV0LineTopologySelectionDecision();
  TestSelectedCandidateOutput(runtime, buffers);
  TestDecayPlanChannelList();
  TestDefaultV0DecayPlanBuilders();
  TestTransactionValidationComplexity();
  TestTransactionTableUpdate(runtime, buffers);
  TestKinematicMathSeed();
  TestMeasurementSeed();
  TestFieldMeasurementSeedApprox();
  TestFieldEnergyFit();
  TestFieldTransportPrimitive();
  TestCpuCompatibleByDcaPrimitive();
  TestFullFieldTransportPrimitive();
  TestFullFieldCoupledDcaPrimitive();
  TestFullFieldCoupledEnergyFit();
  TestRealScaleCpuConstructV0Regression();
  TestCpuDerivativeDcaMiddlePointRegression();
  TestFieldDcaKinematicSeed();
  TestFieldDcaReferenceComparison();
  TestFieldTransportProbe(runtime);
  TestV0LineTopologyProbe(runtime);
  TestKalmanUpdateProbe(runtime);
  TestTwoDaughterTaskKernel(runtime, buffers);
  TestTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterCandidateConstruction(runtime, buffers);
  TestCompactTwoDaughterCandidateOverflow(runtime, buffers);
  TestSteeringTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterEdgeCases(runtime, buffers);
  TestDecayPlanOneChannelExecutor(runtime, buffers);
  TestDecayPlanFieldAwareDiagnosticExecutor(runtime, buffers);
  TestDecayPlanMixedTransportExecutor(runtime, buffers);
  TestDecayPlanDefaultK0Executor(runtime, buffers);
  TestCpuFinderTwoDaughterPairGate(runtime, buffers);
  TestDefaultV0FieldAwareFixtures(runtime, buffers);
  TestDefaultV0DeviceReference(runtime, buffers, false);
  TestDefaultV0DeviceReference(runtime, buffers, true);
  TestDecayPlanDefaultV0Executor(runtime, buffers);
  TestDecayPlanSelectionMultiPvBoundary(runtime, buffers);
  TestDecayPlanMultiChannelExecutor(runtime, buffers);
  TestDecayPlanExecutorEdgeCases(runtime, buffers);
  TestDecayPlanBatchExecutor(runtime, buffers);
  TestDecayPlanBatchSelectedOutput(runtime, buffers);
  TestDecayPlanHighOccupancyDomainIsolation(runtime, buffers);
  TestTwoDaughterSyntheticGrid(runtime, buffers);
  TestSelectionHelpers(buffers);
  TestV0TrackTaskContract(runtime, buffers);
  TestV0TrackChannelBuilders();
  TestV0TrackCompactCandidateConstruction(runtime, buffers);
  TestFusedTwoDaughterRoutingAndConstruction(runtime, buffers);
  TestFusedV0TrackRoutingAndConstruction(runtime, buffers);
  TestGenericChargedDecayGraph(runtime, buffers);
  TestCompositeNeutralAndFinalGraphOperations(runtime, buffers);
  TestCpuFinderFinalSelectionGraph(runtime, buffers);
  TestFilteredMissingMassRawTrackGraph(runtime, buffers);
  TestDecayPlanV0TrackCascade(runtime, buffers);
  TestDecayPlanV0TrackCascadeBatch(runtime, buffers);
  TestBoundedOutputMaterialization(buffers);
  FillInput(buffers, capacities);
  TestRoundTrip(runtime.GetSteering());
  TestGuardsAndEmptyRun(runtime.GetSteering());
  TestCapacityPolicy(buffers);

  service.Finalize();
  assert(!runtime.IsInitialized());

  settings.initializeXpuIfNeeded = false;
  runtime.Initialize(settings);
  assert(runtime.IsInitialized());
  assert(!runtime.InitializedXpu());

  KFParticleGpuBufferManager& emptyBuffers = runtime.GetSteering().GetBuffers();
  assert(emptyBuffers.HostInputTracks().Numerical().ParametersData() == nullptr);
  assert(emptyBuffers.HostEvents() == nullptr);
  emptyBuffers.UploadInput();
  bool resetRejected = false;
  try {
    emptyBuffers.ResetCandidates();
  }
  catch (const std::logic_error&) {
    resetRejected = true;
  }
  assert(resetRejected);
  TestBatchWithoutLegacyTasks(runtime);

  service.Finalize();
  service.Finalize();
  assert(!runtime.IsInitialized() && !xpu::device::all().empty());
  Pass("reattach-finalize", "runtime finalizes cleanly and can reattach without owning XPU initialization");
  return 0;
}
