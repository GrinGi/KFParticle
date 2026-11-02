/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Interpreted by ROOT against the already-built CBMRoot libraries.
// Numerical equivalence and cascade coverage belong to regression/lifecycle.
#ifndef KFPARTICLE_USE_XPU
#define KFPARTICLE_USE_XPU
#endif
// Match the ABI of the CBMRoot adapter loaded below.
#ifndef NonhomogeneousField
#define NonhomogeneousField
#endif

#include "CbmXpu.h"
#include "execution/KFParticleGpuFinder.h"
#include "KFPTrackVector.h"
#include "KFPVertex.h"
#include "KFVertex.h"
#include "execution/KFParticleGpuRuntime.h"

#include <TSystem.h>
#include <xpu/host.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
  void Require(bool condition, const char* message)
  {
    if (!condition) { throw std::runtime_error(message); }
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

  void FillInput(KFPTrackVector& tracks)
  {
    tracks.Resize(2u);
    const float parameters[2][6] = {
      {5.f, 0.f, 0.f, 0.32f, 0.10f, 0.05f},
      {5.f, 0.01f, 0.f, 0.23f, -0.10f, -0.05f}};
    const float covariance[21] = {
      0.0100f, 0.0012f, 0.0120f, -0.0008f, 0.0009f, 0.0110f,
      0.0005f, -0.0004f, 0.0003f, 0.0200f, 0.0006f, -0.0005f,
      0.0004f, 0.0007f, 0.0180f, -0.0003f, 0.0004f, -0.0002f,
      0.0008f, -0.0006f, 0.0220f};
    const float field[10] = {
      0.002f, 0.0002f, 0.00001f, 0.100f, 0.0010f, 0.00005f,
      -0.001f, 0.0001f, 0.00001f, 0.f};
    for (unsigned int i = 0; i < 2u; ++i) {
      for (unsigned int j = 0; j < 6u; ++j) tracks.SetParameter(parameters[i][j], j, i);
      for (unsigned int j = 0; j < 21u; ++j) tracks.SetCovariance(covariance[j], j, i);
      for (unsigned int j = 0; j < 10u; ++j) tracks.SetFieldCoefficient(field[j], j, i);
      tracks.SetId(101 + i, i);
      tracks.SetPDG(i == 0 ? 211 : -211, i);
      tracks.SetQ(i == 0 ? 1 : -1, i);
      tracks.SetPVIndex(-1, i);
      tracks.SetNPixelHits(4, i);
    }
  }
}

int cbmroot_xpu_smoke(const char* device = "hip1")
{
  try {
    for (const char* library : {"libxpu", "libCbmRecoBase", "libKFParticle", "libAlgoOffline"}) {
      Require(gSystem->Load(library) >= 0, library);
    }
    SelectCbmXpuDevice(device);
    cbm::Xpu::Instance().Init();
    Require(cbm::Xpu::Instance().IsInitialized(), "CBMRoot did not initialize XPU");

    auto& runtime = KFParticleGpuRuntime::Instance();
    KFParticleGpuRuntimeSettings settings;
    settings.initializeXpuIfNeeded = false;
    runtime.Initialize(settings);
    Require(!runtime.InitializedXpu(), "KFParticle initialized a second XPU runtime");
    auto* queue = &runtime.GetQueue();

    KFPTrackVector first, last;
    FillInput(first);
    FillInput(last);
    KFPVertex vertex;
    vertex.SetXYZ(0.f, 0.f, 0.f);
    vertex.SetCovarianceMatrix(0.01f, 0.f, 0.01f, 0.f, 0.f, 0.01f);
    vertex.SetChi2(-100.f);
    vertex.SetNContributors(0);
    const std::vector<float> chiToPv = {25.f, 25.f};
    const std::vector<KFVertex> vertices = {KFVertex(vertex)};
    KFParticleGpuFinder::Options options;
    options.initializeXpuIfNeeded = false;
    KFParticleGpuFinder finder(options);
    KFParticleGpuFinder::PreparedOptions run;
    run.outputMode = kfparticle::gpu::GpuOutputMode::FullDiagnostics;
    const auto result = finder.ReconstructDetailedEvent({77u, first, last, chiToPv, vertices}, run);
    Require(result.status == ::kfparticle::gpu::GpuExecutionStatus::Completed,
            "GPU diagnostic runner did not complete");
    Require(result.candidates > 0u && result.rawCandidates.size() == result.candidates
              && result.overflowFlags == 0u,
            "GPU diagnostic runner returned empty, incomplete or overflowing output");
    Require(result.materializationSucceeded, "GPU result materialization failed");
    Require(&runtime.GetQueue() == queue, "adapter replaced the KFParticle queue");
    Require(!runtime.InitializedXpu(), "adapter reinitialized the external XPU runtime");

    // Release KFParticle resources before ROOT unloads the device image.
    finder.ReleaseResources();
    Require(!runtime.IsInitialized(), "KFParticle runtime did not finalize");
    Require(cbm::Xpu::Instance().IsInitialized() && !xpu::device::all().empty(),
            "KFParticle finalized the external CBMRoot XPU runtime");

    std::cout << "PASS cbmroot-xpu-runtime - ROOT libraries, external XPU ownership, "
                 "adapter reconstruction and finalization" << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL cbmroot-xpu-runtime - " << error.what() << std::endl;
    try {
      KFParticleGpuFinder().ReleaseResources();
    }
    catch (const std::exception& cleanupError) {
      std::cerr << "FAIL cbmroot-xpu-runtime cleanup - " << cleanupError.what() << std::endl;
    }
    return 1;
  }
}
