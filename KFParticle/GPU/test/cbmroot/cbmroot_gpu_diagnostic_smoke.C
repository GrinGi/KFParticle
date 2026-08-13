/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Interpreted by ROOT against the built CBMRoot libraries. This exercises the
// diagnostic adapter rather than creating an independent XPU build graph.

#ifndef KFPARTICLE_USE_XPU
#define KFPARTICLE_USE_XPU
#endif

// CBMRoot builds this adapter with the field-aware KFP track layout. Keep
// ROOT's interpreted declarations ABI-compatible with the loaded libraries.
#ifndef NonhomogeneousField
#define NonhomogeneousField
#endif

#include "CbmXpu.h"
#include "KfpGpuDiagnosticComparison.h"
#include "KfpGpuDiagnosticReport.h"
#include "KfpGpuDiagnosticRunner.h"
#include "KfpGpuDiagnosticService.h"
#include "KfpGpuRouting.h"
#include "KFPTrackVector.h"
#include "KFPVertex.h"
#include "KFParticle.h"
#include "KFParticleGpuRuntime.h"
#include "KFVertex.h"

#include <TSystem.h>

#include <xpu/host.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  void Require(bool condition, const char* message)
  {
    if (!condition) { throw std::runtime_error(message); }
  }

  void LoadLibrary(const char* name)
  {
    Require(gSystem->Load(name) >= 0, name);
  }

  void FinalizeKfParticleGpu()
  {
    // This must happen before ROOT unloads the KFParticle HIP image.
    cbm::algo::kfp::GpuDiagnosticService::Instance().Finalize();
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
    tracks.SetParameter(5.f, 0u, index);
    tracks.SetParameter(y, 1u, index);
    tracks.SetParameter(0.f, 2u, index);
    tracks.SetParameter(px, 3u, index);
    tracks.SetParameter(py, 4u, index);
    tracks.SetParameter(pz, 5u, index);
    const float covariance[21] = {
      0.0100f,  0.0012f, 0.0120f, -0.0008f, 0.0009f, 0.0110f,
      0.0005f, -0.0004f, 0.0003f,  0.0200f, 0.0006f, -0.0005f,
      0.0004f,  0.0007f, 0.0180f, -0.0003f, 0.0004f, -0.0002f,
      0.0008f, -0.0006f, 0.0220f};
    for (unsigned int component = 0u; component < 21u; ++component) {
      tracks.SetCovariance(covariance[component], component, index);
    }
#ifdef NonhomogeneousField
    const float field[10] = {0.002f, 0.0002f, 0.00001f,
                             0.100f, 0.0010f, 0.00005f,
                             -0.001f, 0.0001f, 0.00001f,
                             0.f};
    for (unsigned int coefficient = 0u; coefficient < 10u; ++coefficient) {
      tracks.SetFieldCoefficient(field[coefficient], coefficient, index);
    }
#endif
    tracks.SetId(sourceId, index);
    tracks.SetPDG(pdg, index);
    tracks.SetQ(charge, index);
    tracks.SetPVIndex(-1, index);
    tracks.SetNPixelHits(4, index);
  }

  void MakeInput(KFPTrackVector& first,
                 KFPTrackVector& last,
                 std::vector<float>& chiToPrimaryVertex,
                 std::vector<KFVertex>& primaryVertices)
  {
    first.Resize(2u);
    last.Resize(2u);
    FillTrack(first, 0u, 211, 1, 101, 0.32f, 0.10f, 0.05f, 0.f);
    FillTrack(first, 1u, -211, -1, 102, 0.23f, -0.10f, -0.05f, 0.01f);
    FillTrack(last, 0u, 211, 1, 101, 0.32f, 0.10f, 0.05f, 0.f);
    FillTrack(last, 1u, -211, -1, 102, 0.23f, -0.10f, -0.05f, 0.01f);
    chiToPrimaryVertex = {25.f, 25.f};

    KFPVertex source;
    source.SetXYZ(0.f, 0.f, 0.f);
    source.SetCovarianceMatrix(0.01f, 0.f, 0.01f, 0.f, 0.f, 0.01f);
    source.SetChi2(-100.f);
    source.SetNContributors(0);
    primaryVertices = {KFVertex(source)};
  }
}  // namespace

int cbmroot_gpu_diagnostic_smoke(const char* device = "hip1")
{
  try {
    LoadLibrary("libxpu");
    LoadLibrary("libCbmRecoBase");
    LoadLibrary("libKFParticle");
    LoadLibrary("libAlgoOffline");

    SelectCbmXpuDevice(device);
    cbm::Xpu::Instance().Init();
    Require(cbm::Xpu::Instance().IsInitialized(), "CBMRoot did not initialize XPU");

    KFPTrackVector first;
    KFPTrackVector last;
    std::vector<float> chiToPrimaryVertex;
    std::vector<KFVertex> primaryVertices;
    MakeInput(first, last, chiToPrimaryVertex, primaryVertices);

    cbm::algo::kfp::GpuDiagnosticRunner runner;
    const auto result = runner.Run(77u, first, last, chiToPrimaryVertex, primaryVertices);
    Require(result.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            "GPU diagnostic runner did not complete");
    // Capacity is an upper bound before pair/fit rejection. The complete
    // default plan may add raw-track graph hypotheses as channel coverage
    // grows, so this integration smoke checks the required V0 subset instead
    // of freezing the total capacity of the evolving catalogue.
    Require(result.taskCapacity >= 3u,
            "complete default plan did not create the required V0 hypotheses");
    Require(result.candidates > 0u && result.candidates <= result.taskCapacity,
            "GPU diagnostic runner returned an invalid candidate count");
    Require(result.rawCandidates.size() == result.candidates,
            "GPU diagnostic runner candidate counter and downloaded output disagree");
    const auto k0 = std::find_if(result.rawCandidates.begin(), result.rawCandidates.end(), [](const auto& candidate) {
      return candidate.channelId == KFGpuChannelK0ShortToPiPlusPiMinus;
    });
    Require(k0 != result.rawCandidates.end(), "GPU diagnostic runner did not preserve K0S channel identity");
    Require(k0->daughterSourceIds[0] == 101 && k0->daughterSourceIds[1] == 102,
            "GPU diagnostic runner did not preserve K0S daughter lineage");
    Require(result.materializationSucceeded
              && result.materializationStatus == KFGpuMaterializationSucceeded
              && result.materializedParticles.size() == 2u + result.candidates,
            "GPU runner did not atomically materialize the tracks and constructed V0 candidates");
    Require(result.inputUploadMilliseconds >= 0. && result.constructionMilliseconds >= 0.
              && result.selectionMilliseconds >= 0. && result.outputDownloadMilliseconds >= 0.,
            "GPU diagnostic runner returned invalid phase timing");

    unsigned int unresolvedCpuCandidates = 0u;
    const std::vector<KFParticle> noCpuCandidates;
    const auto comparison = cbm::algo::kfp::GpuDiagnosticComparator::Compare(
      result.sourceEventId,
      cbm::algo::kfp::GpuDiagnosticComparator::ExtractCpuV0Candidates(
        result.sourceEventId, noCpuCandidates, unresolvedCpuCandidates),
      cbm::algo::kfp::GpuDiagnosticComparator::MakeGpuCandidateSnapshots(result),
      unresolvedCpuCandidates);
    cbm::algo::kfp::GpuDiagnosticReporter::Instance().Record(result, comparison);
    const auto report = cbm::algo::kfp::GpuDiagnosticReporter::Instance().TakeReport();
    Require(report.completed == 1u && report.gpuOnly == result.candidates && report.channels.size() == 50u,
            "GPU diagnostic reporter did not retain default-channel accounting");

    cbm::algo::kfp::GpuRoutingInput routing;
    routing.mode = cbm::algo::kfp::GpuExecutionMode::QualifiedGpu;
    routing.sampled = true;
    routing.capabilitySupported = cbm::algo::kfp::GpuCapabilityManifest::QualifiedV0().Supports(
      {310, 3122, -3122});
    const auto locked = cbm::algo::kfp::GpuRouter::Decide(routing);
    Require(!locked.publishGpu
              && locked.reason == cbm::algo::kfp::GpuRoutingReason::QualificationLocked,
            "unqualified V0 route was not locked before accelerator attachment");
    routing.qualificationUnlocked = true;
    routing.referenceComparisonRequired = false;
    routing.executionStatus = result.status;
    routing.overflowFlags = result.overflowFlags;
    routing.promotion.blockers = KFGpuPromotionReady;
    routing.materializationSucceeded = result.materializationSucceeded;
    routing.materializationStatus = result.materializationStatus;
    const auto accepted = cbm::algo::kfp::GpuRouter::Decide(routing);
    Require(accepted.publishGpu
              && accepted.reason == cbm::algo::kfp::GpuRoutingReason::GpuAccepted,
            "qualified V0 event was not accepted by the routing policy");
    routing.executionStatus = cbm::algo::kfp::GpuDiagnosticStatus::FieldRejected;
    const auto fallback = cbm::algo::kfp::GpuRouter::Decide(routing);
    Require(!fallback.publishGpu
              && fallback.reason == cbm::algo::kfp::GpuRoutingReason::InvalidField,
            "invalid field did not produce an event-atomic CPU fallback");

    const std::vector<float> invalidChi;
    const auto invalid = runner.Run(78u, first, last, invalidChi, primaryVertices);
    Require(invalid.status == cbm::algo::kfp::GpuDiagnosticStatus::InputRejected,
            "invalid diagnostic input was not rejected");

    // ROOT unloads interpreted code during process teardown. Release the
    // KFParticle-owned queue and buffers while its HIP image is still loaded;
    // CBMRoot continues to own the process-wide XPU runtime.
    FinalizeKfParticleGpu();
    KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
    Require(!runtime.IsInitialized(), "KFParticle GPU runtime did not finalize");
    Require(cbm::Xpu::Instance().IsInitialized(), "KFParticle finalized CBMRoot XPU");

    std::cout << "PASS cbmroot-kfp-gpu-diagnostic - CBMRoot XPU, the KFParticle diagnostic adapter, "
                 "default V0 GPU plan, materialization, qualified routing, event-atomic fallback, monitoring, "
                 "reporting, and invalid-input isolation work together"
              << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL cbmroot-kfp-gpu-diagnostic - " << error.what() << std::endl;
    try {
      FinalizeKfParticleGpu();
    }
    catch (const std::exception& cleanupError) {
      std::cerr << "FAIL cbmroot-kfp-gpu-diagnostic cleanup - " << cleanupError.what() << std::endl;
    }
    return 1;
  }
}
