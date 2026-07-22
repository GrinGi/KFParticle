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
#include "KFPTrackVector.h"
#include "KFPVertex.h"
#include "KFParticle.h"
#include "KFParticleGpuRuntime.h"
#include "KFVertex.h"

#include <TSystem.h>

#include <xpu/host.h>

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

  void FillTrack(KFPTrackVector& tracks, unsigned int index, int pdg, int charge, int sourceId, float px)
  {
    tracks.SetParameter(0.f, 0u, index);
    tracks.SetParameter(0.f, 1u, index);
    tracks.SetParameter(0.f, 2u, index);
    tracks.SetParameter(px, 3u, index);
    tracks.SetParameter(0.1f, 4u, index);
    tracks.SetParameter(0.2f, 5u, index);
    for (unsigned int covariance = 0u; covariance < 21u; ++covariance) {
      tracks.SetCovariance(covariance == 0u || covariance == 2u || covariance == 5u ? 0.01f : 0.f,
                           covariance,
                           index);
    }
#ifdef NonhomogeneousField
    for (unsigned int coefficient = 0u; coefficient < 10u; ++coefficient) {
      tracks.SetFieldCoefficient(0.f, coefficient, index);
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
    FillTrack(first, 0u, 211, 1, 101, 0.30f);
    FillTrack(first, 1u, -211, -1, 102, -0.25f);
    FillTrack(last, 0u, 211, 1, 101, 0.30f);
    FillTrack(last, 1u, -211, -1, 102, -0.25f);
    chiToPrimaryVertex = {10.f, 10.f};

    KFPVertex source;
    source.SetXYZ(0.f, 0.f, 0.f);
    source.SetCovarianceMatrix(0.01f, 0.f, 0.01f, 0.f, 0.f, 0.01f);
    source.SetChi2(1.f);
    source.SetNDF(1);
    source.SetNContributors(2);
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
    Require(result.taskCapacity == 1u, "default V0 plan did not create the expected pion-pair task");
    Require(result.candidates == 1u, "GPU diagnostic runner did not build the expected candidate");
    Require(result.rawCandidates.size() == 1u, "GPU diagnostic runner did not download the raw candidate");
    Require(result.rawCandidates[0].daughterSourceIds[0] == 101
              && result.rawCandidates[0].daughterSourceIds[1] == 102,
            "GPU diagnostic runner did not preserve daughter lineage");
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
    Require(report.completed == 1u && report.gpuOnly == 1u && report.channels.size() == 3u,
            "GPU diagnostic reporter did not retain default-channel accounting");

    const std::vector<float> invalidChi;
    const auto invalid = runner.Run(78u, first, last, invalidChi, primaryVertices);
    Require(invalid.status == cbm::algo::kfp::GpuDiagnosticStatus::InputRejected,
            "invalid diagnostic input was not rejected");

    // ROOT unloads interpreted code during process teardown. Release the
    // KFParticle-owned queue and buffers while its HIP image is still loaded;
    // CBMRoot continues to own the process-wide XPU runtime.
    KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
    runtime.Finalize();
    Require(!runtime.IsInitialized(), "KFParticle GPU runtime did not finalize");
    Require(cbm::Xpu::Instance().IsInitialized(), "KFParticle finalized CBMRoot XPU");

    std::cout << "PASS cbmroot-kfp-gpu-diagnostic - CBMRoot XPU, the KFParticle diagnostic adapter, "
                 "default V0 GPU plan, lineage, telemetry, reporting, and invalid-input isolation work together"
              << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL cbmroot-kfp-gpu-diagnostic - " << error.what() << std::endl;
    return 1;
  }
}
