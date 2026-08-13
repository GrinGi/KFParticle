/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Built by CBMRoot with the same KFParticle SIMD and field-layout definitions
// as the CPU finder. It must not be interpreted by ROOT/Cling.

#include "CbmXpu.h"
#include "algo/kfp/gpu/KfpGpuDiagnosticComparison.h"
#include "algo/kfp/gpu/KfpGpuDiagnosticRunner.h"
#include "KFParticleFinder.h"
#include "KFParticleGpuCpuChannelCatalogue.h"
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuKernels.h"
#include "KFParticleGpuRuntime.h"
#include "KFParticleGpuSteering.h"
#include "KFParticleGpuTwoDaughter.h"
#include "KFParticleTopoReconstructor.h"
#include "KFPTrackVector.h"
#include "KFPVertex.h"
#include "KFVertex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  struct V0Case {
    const char* name;
    int firstPdg;
    int secondPdg;
    int motherPdg;
    int firstCharge;
    int secondCharge;
    int firstSourceId;
    int secondSourceId;
  };

  void Require(bool condition, const std::string& message)
  {
    if (!condition) { throw std::runtime_error(message); }
  }

  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  void SelectCbmXpuDevice(const std::string& device)
  {
    if (device == "cpu") {
      cbm::Xpu::Instance().SetDefaultDevice(cbm::Xpu::Driver::Cpu);
      return;
    }

    const std::size_t digit = device.find_first_of("0123456789");
    const std::string backend = device.substr(0, digit);
    const int index = digit == std::string::npos ? 0 : std::stoi(device.substr(digit));
    if (backend == "hip") {
      cbm::Xpu::Instance().SetDefaultDevice(cbm::Xpu::Driver::Hip, index);
    }
    else if (backend == "cuda") {
      cbm::Xpu::Instance().SetDefaultDevice(cbm::Xpu::Driver::Cuda, index);
    }
    else if (backend == "sycl") {
      cbm::Xpu::Instance().SetDefaultDevice(cbm::Xpu::Driver::Sycl, index);
    }
    else {
      throw std::invalid_argument("Unsupported KFPARTICLE_CBMROOT_DEVICE: " + device);
    }
  }

  KFVertex MakePrimaryVertex(float x = 0.f, float y = 0.f, float z = 0.f)
  {
    KFPVertex source;
    source.SetXYZ(x, y, z);
    source.SetCovarianceMatrix(0.01f, 0.f, 0.01f, 0.f, 0.f, 0.01f);
    source.SetChi2(-100.f);
    source.SetNContributors(0);
    return KFVertex(source);
  }

  void FillTrack(KFPTrackVector& tracks,
                 unsigned int index,
                 int pdg,
                 int charge,
                 int sourceId,
                 float px,
                 float py,
                 float pz,
                 float y)
  {
    // A small DCA residual keeps the CPU finder away from its chi2 == 0
    // rejection guard. The parabolic field exercises the shared full-field
    // input ABI rather than a constant-By-only fixture.
    tracks.SetParameter(5.f, 0u, index);
    tracks.SetParameter(y, 1u, index);
    tracks.SetParameter(0.f, 2u, index);
    tracks.SetParameter(px, 3u, index);
    tracks.SetParameter(py, 4u, index);
    tracks.SetParameter(pz, 5u, index);
    // This is positive definite but deliberately non-diagonal. It makes the
    // equivalence fixture sensitive to the coupled full-field covariance path.
    const float covariance[21] = {
      0.0100f,  0.0012f, 0.0120f, -0.0008f, 0.0009f, 0.0110f,
      0.0005f, -0.0004f, 0.0003f,  0.0200f, 0.0006f, -0.0005f,
      0.0004f,  0.0007f, 0.0180f, -0.0003f, 0.0004f, -0.0002f,
      0.0008f, -0.0006f, 0.0220f};
    for (unsigned int component = 0u; component < 21u; ++component) {
      tracks.SetCovariance(covariance[component], component, index);
    }
    const float field[10] = {0.002f, 0.0002f, 0.00001f,
                             0.100f, 0.0010f, 0.00005f,
                             -0.001f, 0.0001f, 0.00001f,
                             0.f};
    for (unsigned int coefficient = 0u; coefficient < 10u; ++coefficient) {
      tracks.SetFieldCoefficient(field[coefficient], coefficient, index);
    }
    tracks.SetId(sourceId, index);
    tracks.SetPDG(pdg, index);
    tracks.SetQ(charge, index);
    tracks.SetPVIndex(-1, index);
    tracks.SetNPixelHits(4, index);
  }

  void MakeInput(const V0Case& test,
                 KFPTrackVector& first,
                 KFPTrackVector& last,
                 std::vector<float>& chiToPrimaryVertex,
                 std::vector<KFVertex>& primaryVertices)
  {
    first.Resize(2u);
    last.Resize(2u);
    // Daughters miss the primary vertex individually, while their summed
    // momentum points from it to the secondary vertex at x = 5 cm.
    FillTrack(first, 0u, test.firstPdg, test.firstCharge, test.firstSourceId, 0.32f, 0.10f, 0.05f, 0.f);
    FillTrack(first, 1u, test.secondPdg, test.secondCharge, test.secondSourceId, 0.23f, -0.10f, -0.05f, 0.01f);
    FillTrack(last, 0u, test.firstPdg, test.firstCharge, test.firstSourceId, 0.32f, 0.10f, 0.05f, 0.f);
    FillTrack(last, 1u, test.secondPdg, test.secondCharge, test.secondSourceId, 0.23f, -0.10f, -0.05f, 0.01f);
    chiToPrimaryVertex = {25.f, 25.f};
    primaryVertices = {MakePrimaryVertex()};
  }

  std::vector<KFParticle> RunCpuFinder(KFPTrackVector first,
                                       KFPTrackVector last,
                                       const KFVertex& vertex,
                                       int motherPdg)
  {
    KFParticleTopoReconstructor reconstructor;
    auto* finder = reconstructor.GetKFParticleFinder();
    finder->AddDecayToReconstructionList(motherPdg);
    finder->SetMaxDistanceBetweenParticlesCut(1000.f);
    finder->SetLCut(-1000.f);
    finder->SetChiPrimaryCut2D(0.f);
    finder->SetChi2Cut2D(1000000.f);
    finder->SetLdLCut2D(-1000000.f);
    finder->SetSecondaryCuts(1000000.f, 1000000.f, -1000000.f);
    reconstructor.Init(first, last);
    reconstructor.AddPV(vertex);
    reconstructor.SortTracks();
    reconstructor.ReconstructParticles();
    return reconstructor.GetParticles();
  }

  KFParticleGpuTwoDaughterChannel ChannelForMother(int motherPdg)
  {
    if (motherPdg == 310) { return MakeK0ShortToPiPlusPiMinusChannel(); }
    if (motherPdg == 3122) { return MakeLambdaToProtonPiMinusChannel(); }
    if (motherPdg == -3122) { return MakeAntiLambdaToAntiProtonPiPlusChannel(); }
    throw std::invalid_argument("unsupported V0 diagnostic mother");
  }

  void DumpFullFieldBuildFailure(int motherPdg)
  {
    auto& runtime = KFParticleGpuRuntime::Instance();
    auto& buffers = runtime.GetSteering().GetBuffers();
    const auto inputTracks = MakeConstView(buffers.HostInputTracks());
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    const KFParticleGpuTwoDaughterChannel channel = ChannelForMother(motherPdg);
    const KFParticleGpuRange firstRange =
      ResolveTaskSourceRange(event, channel.firstTrackSet, channel.firstSpecies);
    const KFParticleGpuRange secondRange =
      ResolveTaskSourceRange(event, channel.secondTrackSet, channel.secondSpecies);
    if (firstRange.size == 0u || secondRange.size == 0u) {
      std::cerr << "TRACE GPU full-field task could not resolve daughter ranges" << std::endl;
      return;
    }

    KFParticleGpuTwoDaughterTask task;
    FillTwoDaughterTask(firstRange,
                        secondRange,
                        0u,
                        MakeTwoDaughterTaskSource(channel, 0u),
                        task);
    KFParticleGpuFitState hostMother;
    hostMother.Initialize();
    const bool hostBuilt = BuildTwoDaughterCandidate(inputTracks, task, hostMother);

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(1u, xpu::buf_io);
    xpu::buffer<KFParticleGpuFullFieldTwoDaughterTrace> traceBuffer(1u, xpu::buf_io);
    HostPointer(taskBuffer)[0] = task;
    HostPointer(traceBuffer)[0] = {};
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().copy(traceBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuFullFieldTwoDaughterProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      traceBuffer.get());
    runtime.GetQueue().copy(traceBuffer, xpu::d2h);
    runtime.GetQueue().wait();
    const auto& trace = HostPointer(traceBuffer)[0];

    std::cerr << "TRACE GPU full-field task channel/tracks/source/flags/transport "
              << channel.channelId << '/' << task.firstTrack << ',' << task.secondTrack << '/'
              << inputTracks.SourceId(task.firstTrack) << ',' << inputTracks.SourceId(task.secondTrack) << '/'
              << task.flags << '/' << task.transportMode << std::endl;
    std::cerr << "TRACE GPU full-field host-built/chi2/ndf/finite "
              << hostBuilt << '/' << hostMother.Chi2() << '/' << hostMother.NDF() << '/'
              << KFParticleGpuMath::IsFiniteState(hostMother) << std::endl;
    std::cerr << "TRACE GPU full-field device-stage/by/middle/roots/ds "
              << trace.stage << '/' << trace.by << '/' << trace.useMiddlePoint << '/'
              << trace.firstRoots[0] << ',' << trace.firstRoots[1] << '/'
              << trace.secondRoots[0] << ',' << trace.secondRoots[1] << '/'
              << trace.dS[0] << ',' << trace.dS[1] << std::endl;
    std::cerr << "TRACE GPU full-field preliminary-first xyz/chi2 "
              << trace.preliminaryFirst.X() << ',' << trace.preliminaryFirst.Y() << ','
              << trace.preliminaryFirst.Z() << '/' << trace.preliminaryFirst.Chi2() << std::endl;
    std::cerr << "TRACE GPU full-field preliminary-second xyz/chi2 "
              << trace.preliminarySecond.X() << ',' << trace.preliminarySecond.Y() << ','
              << trace.preliminarySecond.Z() << '/' << trace.preliminarySecond.Chi2() << std::endl;
    std::cerr << "TRACE GPU full-field second-pass xyz/chi2 "
              << trace.secondPassCurrent.X() << ',' << trace.secondPassCurrent.Y() << ','
              << trace.secondPassCurrent.Z() << '/' << trace.secondPassCurrent.Chi2() << std::endl;
    std::cerr << "TRACE GPU full-field fitted xyz/chi2/ndf/finite/publishable "
              << trace.fittedMother.X() << ',' << trace.fittedMother.Y() << ','
              << trace.fittedMother.Z() << '/' << trace.fittedMother.Chi2() << '/'
              << trace.fittedMother.NDF() << '/'
              << KFParticleGpuMath::IsFiniteState(trace.fittedMother) << '/'
              << (KFParticleGpuMath::IsFinite(trace.fittedMother.Chi2())
                  && trace.fittedMother.Chi2() > 0.f)
              << std::endl;

    const unsigned int routedCapacity = buffers.Capacities().twoDaughterRoutedTasks;
    const unsigned int routedTraceCount = std::min(routedCapacity, 16u);
    if (routedTraceCount == 0u) {
      std::cerr << "TRACE GPU routed executor has zero task capacity" << std::endl;
      return;
    }
    xpu::buffer<KFParticleGpuRoutedTwoDaughterTrace> routedTraceBuffer(
      routedTraceCount, xpu::buf_io);
    for (unsigned int index = 0u; index < routedTraceCount; ++index) {
      HostPointer(routedTraceBuffer)[index] = {};
    }
    runtime.GetQueue().copy(routedTraceBuffer, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuRoutedTwoDaughterProbe>(
      xpu::n_threads(routedTraceCount),
      routedCapacity,
      routedTraceBuffer.get(),
      routedTraceCount);
    runtime.GetQueue().copy(routedTraceBuffer, xpu::d2h);
    runtime.GetQueue().wait();

    for (unsigned int index = 0u; index < routedTraceCount; ++index) {
      const auto& routedTrace = HostPointer(routedTraceBuffer)[index];
      std::cerr << "TRACE GPU routed index/status stored/task-capacity/descriptors "
                << index << '/' << routedTrace.status << ' '
                << routedTrace.storedTasks << '/' << routedTrace.taskCapacity << '/'
                << routedTrace.descriptorCount << std::endl;
      std::cerr << "TRACE GPU routed candidate-size/capacity daughter-size/capacity "
                << routedTrace.candidateSize << '/' << routedTrace.candidateCapacity << ' '
                << routedTrace.daughterSize << '/' << routedTrace.daughterCapacity << std::endl;
      if ((routedTrace.status & 2u) == 0u) {
        continue;
      }
      std::cerr << "TRACE GPU routed descriptor/tracks/event "
                << routedTrace.routed.descriptorIndex << '/'
                << routedTrace.routed.firstTrackIndex << ','
                << routedTrace.routed.secondTrackIndex << '/'
                << routedTrace.routed.eventIndex << std::endl;
      if ((routedTrace.status & 4u) == 0u) {
        continue;
      }
      const auto& descriptor = routedTrace.descriptor;
      const auto& routedTask = routedTrace.task;
      std::cerr << "TRACE GPU descriptor bit/channel/pdg flags/transport "
                << descriptor.channelBit << '/' << descriptor.channelId << '/'
                << descriptor.motherPdg << ' ' << descriptor.flags << '/'
                << descriptor.transportMode << std::endl;
      std::cerr << "TRACE GPU descriptor daughter/source/alternate pdg "
                << descriptor.firstDaughterPdg << ',' << descriptor.secondDaughterPdg << '/'
                << descriptor.firstSourcePdg << ',' << descriptor.secondSourcePdg << '/'
                << descriptor.firstAlternateSourcePdg << ','
                << descriptor.secondAlternateSourcePdg << std::endl;
      std::cerr << "TRACE GPU descriptor masses mother/sigma/cut "
                << descriptor.firstMass << ',' << descriptor.secondMass << ' '
                << descriptor.motherMass << '/' << descriptor.motherMassSigma << '/'
                << descriptor.secondaryMassSigmaCut << std::endl;
      std::cerr << "TRACE GPU routed task channel/tracks/source/flags/transport "
                << routedTask.channelId << '/' << routedTask.firstTrack << ','
                << routedTask.secondTrack << '/'
                << inputTracks.SourceId(routedTask.firstTrack) << ','
                << inputTracks.SourceId(routedTask.secondTrack) << '/'
                << routedTask.flags << '/' << routedTask.transportMode << std::endl;
      std::cerr << "TRACE GPU routed build valid/built xyz/chi2/ndf/finite "
                << ((routedTrace.status & 16u) != 0u) << '/'
                << ((routedTrace.status & 32u) != 0u) << ' '
                << routedTrace.mother.X() << ',' << routedTrace.mother.Y() << ','
                << routedTrace.mother.Z() << '/' << routedTrace.mother.Chi2() << '/'
                << routedTrace.mother.NDF() << '/'
                << KFParticleGpuMath::IsFiniteState(routedTrace.mother) << std::endl;

      xpu::buffer<KFParticleGpuTwoDaughterTask> actualTaskBuffer(1u, xpu::buf_io);
      xpu::buffer<KFParticleGpuFullFieldTwoDaughterTrace> actualTraceBuffer(1u, xpu::buf_io);
      HostPointer(actualTaskBuffer)[0] = routedTask;
      HostPointer(actualTraceBuffer)[0] = {};
      runtime.GetQueue().copy(actualTaskBuffer, xpu::h2d);
      runtime.GetQueue().copy(actualTraceBuffer, xpu::h2d);
      runtime.GetQueue().launch<KFParticleGpuFullFieldTwoDaughterProbe>(
        xpu::n_threads(1),
        MakeConstView(buffers.DeviceInputTracks()),
        actualTaskBuffer.get(),
        actualTraceBuffer.get());
      runtime.GetQueue().copy(actualTraceBuffer, xpu::d2h);
      runtime.GetQueue().wait();
      const auto& actualTrace = HostPointer(actualTraceBuffer)[0];
      std::cerr << "TRACE GPU routed full-field stage/by/middle/ds fitted-chi2/ndf/finite "
                << actualTrace.stage << '/' << actualTrace.by << '/'
                << actualTrace.useMiddlePoint << '/'
                << actualTrace.dS[0] << ',' << actualTrace.dS[1] << ' '
                << actualTrace.fittedMother.Chi2() << '/'
                << actualTrace.fittedMother.NDF() << '/'
                << KFParticleGpuMath::IsFiniteState(actualTrace.fittedMother) << std::endl;
    }
  }

  void CheckEquivalentCase(const V0Case& test, std::uint64_t eventId)
  {
    KFPTrackVector first;
    KFPTrackVector last;
    std::vector<float> chiToPrimaryVertex;
    std::vector<KFVertex> primaryVertices;
    MakeInput(test, first, last, chiToPrimaryVertex, primaryVertices);

    // Run the GPU path from the pristine fixture. The CPU finder sorts and
    // transports its working tracks with SIMD code, so it receives private
    // copies and cannot affect the input used by the independent GPU path.
    const auto gpuResult = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      eventId, first, last, chiToPrimaryVertex, primaryVertices);

    const auto cpuParticles = RunCpuFinder(first, last, primaryVertices.front(), test.motherPdg);
    unsigned int unresolvedCpuCandidates = 0u;
    const auto cpu = cbm::algo::kfp::GpuDiagnosticComparator::ExtractCpuV0Candidates(
      eventId, cpuParticles, unresolvedCpuCandidates);

    auto gpu = cbm::algo::kfp::GpuDiagnosticComparator::MakeGpuCandidateSnapshots(gpuResult);
    gpu.erase(std::remove_if(gpu.begin(),
                             gpu.end(),
                             [&test](const auto& candidate) {
                               return candidate.key.pdg != test.motherPdg;
                             }),
              gpu.end());
    const auto comparison = cbm::algo::kfp::GpuDiagnosticComparator::Compare(
      eventId, cpu, gpu, unresolvedCpuCandidates);

    const std::string label = std::string(test.name) + ": ";
    Require(gpuResult.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            label + "GPU diagnostic did not complete: " + gpuResult.message);
    unsigned int plannedPairs = 0u;
    unsigned int targetPairs = 0u;
    for (const auto& channel : gpuResult.channels) {
      plannedPairs += channel.totalPairs;
      if (channel.motherPdg == test.motherPdg) { targetPairs = channel.totalPairs; }
    }
    Require(gpuResult.channels.size() == 50u,
            label + "complete two-daughter plan did not report all active channels");
    Require(plannedPairs == gpuResult.taskCapacity,
            label + "default-plan channel pairs and task capacity disagree");
    Require(targetPairs == 1u,
            label + "target V0 channel did not receive exactly one daughter pair");
    // A pi+ pi- pair also supports the CPU-compatible proton fallback of the
    // Lambda channel, while explicit proton-pion inputs support only their
    // target channel. The channel-derived sum above is therefore the stable
    // contract; one hard-coded capacity cannot describe all three fixtures.
    Require(unresolvedCpuCandidates == 0u, label + "CPU finder produced unresolved V0 daughters");
    if (cpu.size() != 1u || gpu.size() != 1u || comparison.matched != 1u) {
      std::cerr << "DETAIL cbmroot-cpu-gpu-v0-" << test.name
                << " - CPU particle/V0/unresolved " << cpuParticles.size() << '/' << cpu.size() << '/'
                << unresolvedCpuCandidates
                << ", GPU status/task/raw/selected/overflow "
                << static_cast<int>(gpuResult.status) << '/' << gpuResult.taskCapacity << '/' << gpu.size() << '/'
                << gpuResult.selectedCandidates << '/' << gpuResult.overflowFlags << std::endl;
      std::cerr << "DETAIL GPU monitoring visited/stored "
                << gpuResult.gpuVisitedCombinations << '/' << gpuResult.gpuStoredTasks << std::endl;
      for (const auto& channel : gpuResult.channels) {
        std::cerr << "DETAIL GPU channel id/pdg pairs/stored/candidates/overflow/truncated "
                  << channel.channelId << '/' << channel.motherPdg << ' '
                  << channel.totalPairs << '/' << channel.storedTasks << '/' << channel.candidates << '/'
                  << channel.overflowFlags << '/' << channel.truncated << std::endl;
      }
      std::cerr << "DETAIL GPU raw/result candidates "
                << gpuResult.candidates << '/' << gpuResult.rawCandidates.size() << std::endl;
      for (unsigned int index = 0u; index < gpuResult.rawCandidates.size(); ++index) {
        const auto& raw = gpuResult.rawCandidates[index];
        const auto& observable = raw.selectionObservables;
        std::cerr << "DETAIL GPU raw index/channel/pdg/daughters "
                  << index << '/' << raw.channelId << '/' << raw.pdg << '/'
                  << raw.daughterSourceIds[0] << ',' << raw.daughterSourceIds[1]
                  << " flags/operation/selection/rejections "
                  << raw.candidateFlags << '/' << raw.operationStatus << '/'
                  << raw.selectionClass << '/' << raw.rejectionReasons
                  << " selected/mass-valid " << raw.selected << '/' << raw.massValid
                  << " mass/error/chi2/ndf " << raw.mass << '/' << raw.massError << '/'
                  << raw.chi2 << '/' << raw.ndf
                  << " pv/best/topology-status " << raw.primaryVertexIndex << '/'
                  << raw.bestPrimaryVertexIndex << '/' << raw.topologyStatus
                  << " geo-chi2/distance/error/line-ldl/topo-chi2/decay/error/ldl "
                  << observable.geometricChi2PerNdf << '/'
                  << observable.nearestPrimaryVertexDistance << '/'
                  << observable.nearestPrimaryVertexDistanceError << '/'
                  << observable.nearestPrimaryVertexLdL << '/'
                  << observable.bestPrimaryVertexTopoChi2PerNdf << '/'
                  << observable.bestPrimaryVertexDecayLength << '/'
                  << observable.bestPrimaryVertexDecayLengthError << '/'
                  << observable.bestPrimaryVertexLdL << std::endl;
      }
      auto& diagnosticBuffers = KFParticleGpuRuntime::Instance().GetSteering().GetBuffers();
      const auto diagnosticCandidates = diagnosticBuffers.HostCandidates();
      const auto diagnosticSelection = diagnosticBuffers.HostV0SelectionResults();
      std::cerr << "DETAIL GPU host pool candidates/daughters/capacity "
                << diagnosticCandidates.Size() << '/'
                << diagnosticCandidates.Daughters().Size() << '/'
                << diagnosticCandidates.Capacity() << std::endl;
      for (unsigned int index = 0u; index < diagnosticCandidates.Size(); ++index) {
        const auto& metadata = diagnosticCandidates.Metadata();
        std::cerr << "DETAIL GPU pool index/channel/pdg/event/daughter-range/flags/topology/output/operation "
                  << index << '/' << metadata.ChannelId(index) << '/' << metadata.Pdg(index) << '/'
                  << metadata.EventIndex(index) << '/' << metadata.DaughterOffset(index) << ','
                  << metadata.DaughterCount(index) << '/' << metadata.Flags(index) << '/'
                  << metadata.Topology(index) << '/' << metadata.OutputClass(index) << '/'
                  << metadata.OperationStatus(index);
        if (diagnosticSelection.CanStore(index)) {
          const auto& selection = diagnosticSelection.Result(index);
          std::cerr << " selection/class/rejections/topology-status/best-pv "
                    << selection.selectionClass << '/' << selection.rejectionReasons << '/'
                    << selection.topologyStatus << '/' << selection.bestPrimaryVertexIndex;
        }
        std::cerr << std::endl;
      }
      DumpFullFieldBuildFailure(test.motherPdg);
      if (!cpu.empty()) {
        std::cerr << "DETAIL CPU key channel/daughters " << cpu.front().key.channelId << '/'
                  << cpu.front().key.daughterSourceIds[0] << ',' << cpu.front().key.daughterSourceIds[1]
                  << ", mass/chi2/NDF " << cpu.front().mass << '/' << cpu.front().chi2 << '/'
                  << cpu.front().ndf << std::endl;
      }
      if (!gpu.empty()) {
        std::cerr << "DETAIL GPU key channel/daughters " << gpu.front().key.channelId << '/'
                  << gpu.front().key.daughterSourceIds[0] << ',' << gpu.front().key.daughterSourceIds[1]
                  << ", mass/chi2/NDF " << gpu.front().mass << '/' << gpu.front().chi2 << '/'
                  << gpu.front().ndf << std::endl;
      }
    }
    Require(cpu.size() == 1u && gpu.size() == 1u && comparison.matched == 1u,
            label + "CPU and GPU did not build one matching V0 candidate");
    Require(cpu.front().key.channelId == gpu.front().key.channelId
              && cpu.front().key.daughterSourceIds == gpu.front().key.daughterSourceIds,
            label + "channel or daughter lineage differs");
    Require(cpu.front().massValid && gpu.front().massValid, label + "candidate mass is invalid");
    Require(std::abs(cpu.front().mass - gpu.front().mass) < 0.15f, label + "mass difference exceeds tolerance");
    Require(std::abs(cpu.front().massError - gpu.front().massError) < 0.15f,
            label + "mass-error difference exceeds tolerance");
    Require(std::abs(cpu.front().chi2 - gpu.front().chi2) < 100.f,
            label + "chi2 difference exceeds tolerance");
    Require(std::abs(cpu.front().ndf - gpu.front().ndf) <= 2, label + "NDF difference exceeds tolerance");
    Require(gpu.front().hasSelectionDiagnostics, label + "GPU selection diagnostics are missing");
    const auto rawGpuIt = std::find_if(
      gpuResult.rawCandidates.begin(),
      gpuResult.rawCandidates.end(),
      [&test](const auto& candidate) { return candidate.pdg == test.motherPdg; });
    Require(rawGpuIt != gpuResult.rawCandidates.end(),
            label + "target raw GPU diagnostic record is missing");
    const auto& rawGpu = *rawGpuIt;

    std::cout << "METRIC cbmroot-cpu-gpu-v0-" << test.name
              << " mass_residual=" << std::abs(cpu.front().mass - gpu.front().mass)
              << " mass_error_residual=" << std::abs(cpu.front().massError - gpu.front().massError)
              << " chi2_residual=" << std::abs(cpu.front().chi2 - gpu.front().chi2)
              << " ndf_residual=" << std::abs(cpu.front().ndf - gpu.front().ndf)
              << " selected=" << gpu.front().selected
              << " selection_class=" << gpu.front().selectionClass
              << " rejection_reasons=" << gpu.front().rejectionReasons
              << " topology_status=" << rawGpu.topologyStatus
              << " line_distance=" << rawGpu.selectionObservables.nearestPrimaryVertexDistance
              << " line_chi2_ndf=" << rawGpu.selectionObservables.bestPrimaryVertexTopoChi2PerNdf
              << " overflow=" << gpuResult.overflowFlags << std::endl;

    std::cout << "PASS cbmroot-cpu-gpu-v0-" << test.name
              << " - CPU and GPU agree on V0 lineage and bounded fit observables" << std::endl;
  }

  void ConfigureDefaultV0DistanceCut(float maxDistance)
  {
    KFParticleGpuDecayPlan& plan = KFParticleGpuRuntime::Instance().GetSteering().GetDecayPlan();
    KFParticleGpuDecayPlan completePlan;
    AddCpuFinderTwoDaughterChannels(completePlan);
    plan.Clear();
    for (std::size_t index = 0u;
         index < completePlan.NumberOfTwoDaughterChannels(); ++index) {
      KFParticleGpuTwoDaughterChannel channel =
        completePlan.TwoDaughterChannel(index);
      if (channel.channelId == KFGpuChannelK0ShortToPiPlusPiMinus) {
        channel.selection.maxPrimaryVertexDistance = maxDistance;
      }
      plan.AddTwoDaughterChannel(channel);
    }
  }

  void CheckMultiPrimaryVertexTopologyBoundary()
  {
    KFPTrackVector first;
    KFPTrackVector last;
    std::vector<float> chiToPrimaryVertex;
    std::vector<KFVertex> primaryVertices;
    MakeInput({"multi-pv", 211, -211, 310, 1, -1, 601, 602},
              first,
              last,
              chiToPrimaryVertex,
              primaryVertices);
    // Duplicate PVs make the line-chi2 tie deterministic: the first packed
    // vertex must win, independently of CPU finder output ownership.
    primaryVertices.push_back(MakePrimaryVertex());
    const auto baseline = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      906u, first, last, chiToPrimaryVertex, primaryVertices);
    Require(baseline.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            "multi-PV GPU diagnostic did not complete");
    const auto findK0 = [](const auto& result) {
      return std::find_if(result.rawCandidates.begin(),
                          result.rawCandidates.end(),
                          [](const auto& candidate) { return candidate.pdg == 310; });
    };
    const auto candidateIt = findK0(baseline);
    Require(candidateIt != baseline.rawCandidates.end()
              && std::count_if(baseline.rawCandidates.begin(),
                               baseline.rawCandidates.end(),
                               [](const auto& raw) { return raw.pdg == 310; }) == 1,
            "multi-PV fixture did not build exactly one raw K0S hypothesis");
    const auto& candidate = *candidateIt;
    Require(candidate.bestPrimaryVertexIndex == 0,
            "multi-PV line-topology tie did not keep the lowest primary-vertex index");
    Require((candidate.topologyStatus & KFGpuV0LineTopologyValid) != 0u,
            "multi-PV fixture did not report a valid line topology");
    const float distance = candidate.selectionObservables.nearestPrimaryVertexDistance;
    Require(std::isfinite(distance) && distance > 0.f,
            "multi-PV fixture did not report a finite positive line distance");

    ConfigureDefaultV0DistanceCut(distance);
    const auto exactBoundary = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      907u, first, last, chiToPrimaryVertex, primaryVertices);
    Require(exactBoundary.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            "exact-distance GPU diagnostic did not complete: "
              + exactBoundary.message);
    const auto exactK0 = findK0(exactBoundary);
    Require(exactK0 != exactBoundary.rawCandidates.end(),
            "exact-distance fixture lost its raw K0S hypothesis");
    Require((exactK0->rejectionReasons & KFGpuV0SelectionRejectDistance) != 0u,
            "exact line-distance boundary was not rejected");

    ConfigureDefaultV0DistanceCut(std::nextafter(distance, std::numeric_limits<float>::infinity()));
    const auto aboveBoundary = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      908u, first, last, chiToPrimaryVertex, primaryVertices);
    Require(aboveBoundary.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            "above-boundary GPU diagnostic did not complete: "
              + aboveBoundary.message);
    const auto aboveK0 = findK0(aboveBoundary);
    Require(aboveK0 != aboveBoundary.rawCandidates.end(),
            "above-boundary fixture lost its raw K0S hypothesis");
    Require((aboveK0->rejectionReasons & KFGpuV0SelectionRejectDistance) == 0u,
            "line-distance cut rejected a value strictly below its boundary");
    KFParticleGpuRuntime::Instance().GetSteering().GetDecayPlan().Clear();

    std::cout << "PASS cbmroot-cpu-gpu-v0-multi-pv-boundary"
              << " - stable best_pv=0 line_distance=" << distance
              << " exact_cut_rejected=1 above_cut_accepted=1" << std::endl;
  }

  void FinalizeRuntime() noexcept
  {
    try {
      KFParticleGpuRuntime::Instance().Finalize();
    }
    catch (...) {
    }
  }
}  // namespace

int main(int argc, char** argv)
{
  const std::string device = argc > 1 ? argv[1] : "hip1";
  try {
    SelectCbmXpuDevice(device);
    cbm::Xpu::Instance().Init();
    Require(cbm::Xpu::Instance().IsInitialized(), "CBMRoot did not initialize XPU");

    CheckEquivalentCase({"k0s", 211, -211, 310, 1, -1, 101, 102}, 901u);
    CheckEquivalentCase({"lambda", 2212, -211, 3122, 1, -1, 201, 202}, 902u);
    CheckEquivalentCase({"anti-lambda", 211, -2212, -3122, 1, -1, 301, 302}, 903u);
    CheckMultiPrimaryVertexTopologyBoundary();

    KFPTrackVector invalidFirst;
    KFPTrackVector invalidLast;
    std::vector<float> invalidChi;
    std::vector<KFVertex> invalidVertices;
    MakeInput({"invalid", 211, -211, 310, 1, -1, 401, 402},
              invalidFirst,
              invalidLast,
              invalidChi,
              invalidVertices);
    invalidChi.clear();
    const auto invalid = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      904u, invalidFirst, invalidLast, invalidChi, invalidVertices);
    Require(invalid.status == cbm::algo::kfp::GpuDiagnosticStatus::InputRejected,
            "inconsistent synthetic input was not rejected");
    std::cout << "PASS cbmroot-cpu-gpu-v0-invalid-input - invalid GPU input is isolated" << std::endl;

    KFPTrackVector invalidFieldFirst;
    KFPTrackVector invalidFieldLast;
    std::vector<float> invalidFieldChi;
    std::vector<KFVertex> invalidFieldVertices;
    MakeInput({"invalid-field", 211, -211, 310, 1, -1, 501, 502},
              invalidFieldFirst,
              invalidFieldLast,
              invalidFieldChi,
              invalidFieldVertices);
    invalidFieldFirst.SetFieldCoefficient(std::numeric_limits<float>::quiet_NaN(), 3u, 0u);
    const auto invalidField = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      905u, invalidFieldFirst, invalidFieldLast, invalidFieldChi, invalidFieldVertices);
    Require(invalidField.status == cbm::algo::kfp::GpuDiagnosticStatus::FieldRejected,
            "non-finite field input was not reported as a field rejection");
    std::cout << "PASS cbmroot-cpu-gpu-v0-invalid-field - invalid field metadata is isolated" << std::endl;

    FinalizeRuntime();
    Require(cbm::Xpu::Instance().IsInitialized(), "KFParticle finalized CBMRoot XPU");
    std::cout << "PASS cbmroot-cpu-gpu-v0-equivalence - hermetic CPU/GPU V0 gate completed" << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    FinalizeRuntime();
    std::cerr << "FAIL cbmroot-cpu-gpu-v0-equivalence - " << error.what() << std::endl;
    return 1;
  }
}
