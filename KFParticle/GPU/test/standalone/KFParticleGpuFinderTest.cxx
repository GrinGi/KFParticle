/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

// Minimal external usage and regression: link KFParticle::KFParticle, not CBMRoot.
#include "execution/KFParticleGpuFinder.h"
#include "execution/KFParticleGpuRuntime.h"
#include "KFPVertex.h"

#include <xpu/host.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace {
  using Finder = KFParticleGpuFinder;

  void Require(bool condition, const std::string& message)
  {
    if (!condition) { throw std::runtime_error(message); }
  }

  template<class Function>
  void Rejects(Function function, const char* message)
  {
    bool rejected = false;
    try { function(); }
    catch (const std::logic_error&) { rejected = true; }
    Require(rejected, message);
  }

  struct Input {
    KFPTrackVector first, last;
    std::vector<float> chi{25.f, 25.f};
    std::vector<KFVertex> vertices;

    explicit Input(bool primary = false, float shift = 0.f)
    {
      first.Resize(2u);
      last.Resize(2u);
      const float covariance[21] = {
        0.0100f, 0.0012f, 0.0120f, -0.0008f, 0.0009f, 0.0110f,
        0.0005f, -0.0004f, 0.0003f, 0.0200f, 0.0006f, -0.0005f,
        0.0004f, 0.0007f, 0.0180f, -0.0003f, 0.0004f, -0.0002f,
        0.0008f, -0.0006f, 0.0220f};
      const float field[10] = {0.002f, 0.0002f, 0.00001f, 0.100f, 0.001f,
                                0.00005f, -0.001f, 0.0001f, 0.00001f, 0.f};
      for (auto* tracks : {&first, &last}) {
        for (unsigned int i = 0u; i < 2u; ++i) {
          const float parameters[6] = {
            5.f + shift, i == 0u ? 0.f : 0.01f, 0.f,
            i == 0u ? 0.32f : 0.23f, i == 0u ? 0.10f : -0.10f,
            i == 0u ? 0.05f : -0.05f};
          for (unsigned int j = 0u; j < 6u; ++j) { tracks->SetParameter(parameters[j], j, i); }
          for (unsigned int j = 0u; j < 21u; ++j) { tracks->SetCovariance(covariance[j], j, i); }
          for (unsigned int j = 0u; j < 10u; ++j) { tracks->SetFieldCoefficient(field[j], j, i); }
          tracks->SetPDG(i == 0u ? 211 : -211, i);
          tracks->SetQ(i == 0u ? 1 : -1, i);
          tracks->SetId(601 + static_cast<int>(i), i);
          tracks->SetPVIndex(primary ? 1 : -1, i);
          tracks->SetNPixelHits(4, i);
        }
      }
      KFPVertex vertex;
      vertex.SetXYZ(shift, 0.f, 0.f);
      vertex.SetCovarianceMatrix(0.01f, 0.f, 0.01f, 0.f, 0.f, 0.01f);
      vertex.SetChi2(-100.f);
      vertex.SetNContributors(0);
      if (primary) {
        KFPVertex other = vertex;
        other.SetXYZ(shift + 1000.f, 1000.f, 1000.f);
        vertices.emplace_back(other);
      }
      vertices.emplace_back(vertex);
    }

    Finder::Event Event(std::uint64_t id) const { return {id, first, last, chi, vertices}; }
  };

  void Equivalent(const Finder::Result& first, const Finder::Result& second)
  {
    Require(first.Succeeded() && second.Succeeded(), "reconstruction failed: " + first.message + second.message);
    Require(first.sourceEventId == second.sourceEventId && first.status == second.status
              && first.candidates == second.candidates && first.selectedCandidates == second.selectedCandidates
              && first.particles.size() == second.particles.size(), "event metadata/output count changed");
    std::vector<const KFParticle*> left, right;
    for (const auto& particle : first.particles) { left.push_back(&particle); }
    for (const auto& particle : second.particles) { right.push_back(&particle); }
    const auto less = [](const KFParticle* a, const KFParticle* b) {
      return std::make_tuple(a->GetPDG(), a->DaughterIds())
             < std::make_tuple(b->GetPDG(), b->DaughterIds());
    };
    std::sort(left.begin(), left.end(), less);
    std::sort(right.begin(), right.end(), less);
    for (std::size_t i = 0u; i < left.size(); ++i) {
      Require(left[i]->GetPDG() == right[i]->GetPDG()
                && left[i]->DaughterIds() == right[i]->DaughterIds(), "particle identity changed");
      for (int j = 0; j < 8; ++j) {
        const float a = left[i]->GetParameter(j), b = right[i]->GetParameter(j);
        Require(std::isfinite(a) && std::isfinite(b)
                  && std::abs(a - b) <= 1.e-4f * (1.f + std::abs(a)), "fit changed");
      }
      for (int j = 0; j < 36; ++j) {
        const float a = left[i]->GetCovariance(j), b = right[i]->GetCovariance(j);
        Require(std::isfinite(a) && std::isfinite(b)
                  && std::abs(a - b) <= 1.e-4f * (1.f + std::abs(a)), "covariance changed");
      }
    }
  }

  kfparticle::gpu::GpuPreparedBatch Prepare(const std::vector<Finder::Event>& events)
  {
    kfparticle::gpu::GpuBatchBuilder builder(kfparticle::gpu::DefaultGpuBatchLimits(4u), {310});
    for (std::size_t i = 0u; i < events.size(); ++i) {
      const auto& event = events[i];
      const kfparticle::gpu::GpuEventInput input{static_cast<unsigned int>(i), &event.first,
        &event.last, &event.chiToPV, &event.vertices, event.minSecondaryTrackChiToPV};
      Require(builder.TryAppend(event.sourceEventId, input) == kfparticle::gpu::GpuBatchAppendResult::Added,
              "prepared fixture exceeded its admission limits");
    }
    return builder.Seal();
  }

  Finder::Result Published(kfparticle::gpu::GpuEventResult source)
  {
    Require(source.materializationSucceeded, "prepared event was not materialized");
    Finder::Result result;
    result.sourceEventId = source.sourceEventId;
    result.status = source.status;
    result.candidates = source.candidates;
    result.selectedCandidates = source.selectedCandidates;
    result.overflowFlags = source.overflowFlags;
    result.particles = std::move(source.materializedParticles);
    result.message = std::move(source.message);
    return result;
  }

  void Run(const std::string& device)
  {
    Finder::Options options;
    options.device = device;
    options.motherPdgs = {310};
    options.limits = kfparticle::gpu::DefaultGpuBatchLimits(4u);
    options.packingThreads = 2u;
    options.materializationThreads = 2u;
    auto bad = options;
    bad.packingThreads = 0u;
    Rejects([&] { Finder finder(bad); }, "zero worker limit accepted");
    bad = options;
    bad.limits.events = 0u;
    Rejects([&] { Finder finder(bad); }, "zero resource limit accepted");
    bad = options;
    bad.motherPdgs = {987654321};
    Rejects([&] { Finder finder(bad); }, "unknown channel accepted");
    Require(xpu::device::all().empty(), "option validation initialized XPU");

    Finder finder(options);
    Require(finder.ReconstructBatch({}).empty(), "empty batch returned events");
    KFPTrackVector emptyTracks;
    std::vector<float> emptyChi;
    std::vector<KFVertex> emptyVertices;
    const auto empty = finder.Reconstruct({11u, emptyTracks, emptyTracks, emptyChi, emptyVertices});
    Require(empty.Succeeded() && empty.status == Finder::Status::EmptyInput, "empty event rejected");
    Require(xpu::device::all().empty(), "empty input initialized XPU");
    Input input;
    const auto duplicate = finder.ReconstructBatch({input.Event(9u), input.Event(9u)});
    Require(duplicate.size() == 2u && !duplicate[0].Succeeded() && !duplicate[1].Succeeded(),
            "duplicate source identity accepted");
    Require(xpu::device::all().empty(), "structural rejection initialized XPU");
    bad = options;
    bad.limits.tracks = 1u;
    Finder limited(bad);
    Require(!limited.Reconstruct(input.Event(5u)).Succeeded(), "resource-admission limit ignored");
    Require(xpu::device::all().empty(), "resource rejection initialized XPU");
    const auto tooMany = finder.ReconstructBatch({input.Event(1u), input.Event(2u),
      input.Event(3u), input.Event(4u), input.Event(5u)});
    Require(tooMany.size() == 5u
              && std::all_of(tooMany.begin(), tooMany.end(), [](const auto& r) {
                   return r.status == Finder::Status::InputRejected && r.particles.empty();
                 }), "oversized batch was split, truncated or published");

    auto external = options;
    external.initializeXpuIfNeeded = false;
    Finder attached(external);
    Rejects([&] { attached.Initialize(); }, "uninitialized external runtime accepted");
    const auto preliminary = Prepare({input.Event(8u)});
    const auto unavailable = attached.ReconstructPreparedBatch(preliminary, Finder::PreparedOptions{});
    Require(unavailable.size() == 1u && unavailable[0].status == Finder::Status::RuntimeUnavailable
              && unavailable[0].materializedParticles.empty() && xpu::device::all().empty(),
            "external prepared execution did not return a runtime failure without initialization");
    auto invalidRun = Finder::PreparedOptions{};
    invalidRun.materializeOutput = false;
    Rejects([&] { finder.ReconstructPreparedBatch(preliminary, invalidRun); },
            "compact output without materialization was accepted");

    // A stale benchmarking environment must not repeat or alter standalone work.
    setenv("KFPARTICLE_GPU_IN_PROCESS_REPLAY", "1", 1);
    setenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "invalid", 1);
    setenv("KFPARTICLE_GPU_HOST_RESULT_REFERENCE", "1", 1);
    setenv("KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE", "invalid", 1);
    auto first = finder.Reconstruct(input.Event(1u));
    Require(first.Succeeded(), "standalone reconstruction failed: " + first.message);
    Require(first.tracks == 2u && first.overflowFlags == 0u, "physical track count or overflow is invalid");
    Require(!first.diagnostics && !first.timing, "compact defaults retained diagnostics/timing");
    Require(std::any_of(first.particles.begin(), first.particles.end(),
                        [](const KFParticle& p) { return p.GetPDG() == 310; }), "K0S publication missing");
    const auto saved = first;
    Input shifted(false, 10.f);
    const std::uint64_t secondId = (std::uint64_t{1} << 32u) + 1u;
    auto batch = finder.ReconstructBatch({input.Event(1u), shifted.Event(secondId)});
    Require(batch.size() == 2u && batch[1].sourceEventId == secondId, "64-bit source identity lost");
    Equivalent(first, batch[0]);
    Equivalent(finder.Reconstruct(shifted.Event(secondId)), batch[1]);
    Equivalent(saved, first);
    Input rejectedPair;
    for (auto* tracks : {&rejectedPair.first, &rejectedPair.last}) {
      tracks->SetParameter(1000.f, 1, 1);
    }
    const auto noCandidate = finder.Reconstruct(rejectedPair.Event(3u));
    Require(noCandidate.Succeeded() && noCandidate.candidates == 0u,
            "rejected-pair fixture must complete without a candidate");
    const auto mixed = finder.ReconstructBatch({input.Event(1u), rejectedPair.Event(3u)});
    Equivalent(first, mixed[0]);
    Equivalent(noCandidate, mixed[1]);
    const auto prepared = Prepare({input.Event(1u), shifted.Event(secondId)});
    auto integrationOptions = external;
    integrationOptions.limits.events = 1u;
    integrationOptions.motherPdgs = {113};
    Finder integration(integrationOptions);
    auto preparedResults = integration.ReconstructPreparedBatch(prepared, Finder::PreparedOptions{});
    Require(preparedResults.size() == 2u
              && preparedResults[0].capacityPlanBuilds == prepared.Event(0u).capacityPlanBuilds
              && preparedResults[1].capacityPlanBuilds == prepared.Event(1u).capacityPlanBuilds
              && preparedResults[0].inputPreparationMilliseconds == prepared.Event(0u).inputPreparationMilliseconds,
            "prepared execution changed the admitted batch or its preparation metadata");
    Equivalent(first, Published(std::move(preparedResults[0])));
    Equivalent(batch[1], Published(std::move(preparedResults[1])));

    auto detailedOptions = options;
    detailedOptions.diagnostics = true;
    detailedOptions.timings = true;
    Finder detailed(detailedOptions);
    auto diagnostics = detailed.ReconstructBatch({input.Event(1u), shifted.Event(secondId)});
    Equivalent(first, diagnostics[0]);
    Equivalent(batch[1], diagnostics[1]);
    Require(diagnostics[0].diagnostics && !diagnostics[0].diagnostics->candidates.empty()
              && diagnostics[0].timing && !diagnostics[1].timing, "optional diagnostics/timings missing");
    Require(diagnostics[0].timing->total >= 0. && std::isfinite(diagnostics[0].timing->pipeline),
            "invalid batch timings");
    unsetenv("KFPARTICLE_GPU_IN_PROCESS_REPLAY");
    unsetenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS");
    unsetenv("KFPARTICLE_GPU_HOST_RESULT_REFERENCE");
    unsetenv("KFPARTICLE_GPU_MATERIALIZATION_SCHEDULE");

    Finder::PreparedOptions snapshots;
    snapshots.outputMode = kfparticle::gpu::GpuOutputMode::FullDiagnostics;
    snapshots.materializeOutput = false;
    const auto snapshotResults = attached.ReconstructPreparedBatch(prepared, snapshots);
    Require(snapshotResults.size() == 2u && snapshotResults[0].Succeeded()
              && snapshotResults[1].Succeeded()
              && !snapshotResults[0].rawCandidates.empty()
              && snapshotResults[0].materializedParticles.empty()
              && !snapshotResults[0].materializationSucceeded,
            "snapshot-only integration added materialization or lost diagnostics");

    setenv("KFPARTICLE_GPU_PERFORMANCE_MONITORING", "1", 1);
    setenv("KFPARTICLE_GPU_IN_PROCESS_REPLAY", "1", 1);
    setenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "invalid", 1);
    Finder::PreparedOptions framework;
    framework.inheritEnvironment = true;
    auto individual = attached.ReconstructDetailedEvent(input.Event(1u), framework);
    Equivalent(first, Published(std::move(individual)));
    setenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS", "2", 1);
    framework.replay = true;
    auto replayed = attached.ReconstructPreparedBatch(prepared, framework);
    Require(replayed.size() == 2u && replayed[0].performanceSample
              && replayed[0].gpuCompactOutput && replayed[0].gpuKernelLaunches > 0u,
            "framework monitoring metadata was lost");
    Equivalent(first, Published(std::move(replayed[0])));
    Equivalent(batch[1], Published(std::move(replayed[1])));
    unsetenv("KFPARTICLE_GPU_IN_PROCESS_REPLAY");
    unsetenv("KFPARTICLE_GPU_IN_PROCESS_REPEATS");
    unsetenv("KFPARTICLE_GPU_PERFORMANCE_MONITORING");

    Input oversized;
    oversized.first.Resize(0u);
    oversized.last.Resize(0u);
    oversized.chi.clear();
    oversized.vertices.resize(kfparticle::gpu::DefaultGpuBatchLimits(1u).vertices + 1u);
    const auto unbounded = attached.ReconstructDetailedEvent(oversized.Event(40u), Finder::PreparedOptions{});
    Require(unbounded.status == Finder::Status::EmptyInput
              && unbounded.vertices == oversized.vertices.size(),
            "individual integration acquired a collection admission ceiling");

    Finder complete;
    const auto completeResult = complete.Reconstruct(input.Event(31u));
    Require(completeResult.Succeeded()
              && std::any_of(completeResult.particles.begin(), completeResult.particles.end(),
                            [](const KFParticle& p) { return p.GetPDG() == 310; }),
            "default complete catalogue did not publish K0S");
    Equivalent(first, finder.Reconstruct(input.Event(1u)));

    auto primaryOptions = detailedOptions;
    primaryOptions.motherPdgs = {113};
    Finder primary(primaryOptions);
    Input a(true), b(true, 20.f);
    const auto primaryBatch = primary.ReconstructBatch({a.Event(21u), b.Event(22u)});
    Equivalent(primary.Reconstruct(a.Event(21u)), primaryBatch[0]);
    Equivalent(primary.Reconstruct(b.Event(22u)), primaryBatch[1]);
    Require(primaryBatch[0].candidates > 0u && primaryBatch[1].candidates > 0u,
            "primary/PV fixture did not exercise construction");
    a.last.SetPVIndex(2, 0u);
    const auto invalidPV = primary.Reconstruct(a.Event(23u));
    Require(!invalidPV.Succeeded() && invalidPV.particles.empty(), "invalid local PV was published");
    a.last.SetPVIndex(1, 0u);
    a.first.SetFieldCoefficient(std::numeric_limits<float>::quiet_NaN(), 0u, 0u);
    const auto invalidField = primary.Reconstruct(a.Event(24u));
    Require(invalidField.status == Finder::Status::FieldRejected && invalidField.particles.empty(),
            "invalid field was not isolated");
    Equivalent(first, finder.Reconstruct(input.Event(1u)));

    bool rejectedThread = false;
    std::thread wrong([&] {
      try { finder.Reconstruct(input.Event(25u)); }
      catch (const std::logic_error&) { rejectedThread = true; }
    });
    wrong.join();
    Require(rejectedThread, "Finder allowed GPU submission from another host thread");
    bool rejectedOtherInstance = false;
    std::thread other([&] {
      try { Finder independent(options); independent.Initialize(); }
      catch (const std::logic_error&) { rejectedOtherInstance = true; }
    });
    other.join();
    Require(rejectedOtherInstance, "shared runtime accepted a second submission thread");

    Finder::Result surviving;
    {
      Input temporary;
      Finder temporaryFinder(options);
      surviving = temporaryFinder.Reconstruct(temporary.Event(1u));
    }
    Equivalent(first, surviving);
    finder.ReleaseResources();
    finder.ReleaseResources();
    Require(!KFParticleGpuRuntime::Instance().IsInitialized() && !xpu::device::all().empty(),
            "cleanup leaked KFParticle storage or shut down application XPU");
    Equivalent(saved, first);
    attached.Initialize();
    Require(!KFParticleGpuRuntime::Instance().InitializedXpu(), "external XPU was reinitialized");
    Equivalent(first, attached.Reconstruct(input.Event(1u)));
    attached.ReleaseResources();
    std::cout << "PASS standalone-finder device=" << device
              << " event_batch_parity=1 pv_isolation=1 owned_results=1 explicit_options=1 lifecycle=1 prepared_integration=1\n";
  }
}

int main(int argc, char** argv)
{
  try {
    Run(argc > 1 ? argv[1] : "cpu");
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL standalone-finder - " << error.what() << '\n';
    return 1;
  }
}
