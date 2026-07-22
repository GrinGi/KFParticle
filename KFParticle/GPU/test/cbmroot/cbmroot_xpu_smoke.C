/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Deliberately interpreted by ROOT: this test must use the already-built
// CBMRoot and KFParticle libraries instead of compiling another XPU graph.

#ifndef KFPARTICLE_USE_XPU
#define KFPARTICLE_USE_XPU
#endif

#include "CbmXpu.h"
#include "KFParticleGpuBufferManager.h"
#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuRuntime.h"
#include "KFParticleGpuSteering.h"

#include <TSystem.h>

#include <xpu/host.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

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

  void FillRoundTripInput(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks = 2;
    capacities.events = 1;
    capacities.candidates = 2;
    capacities.daughterIds = 2;
    buffers.EnsureCapacity(capacities);
    buffers.SetInputSizes(2, 0, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    for (unsigned int i = 0; i < 2; ++i) {
      tracks.Numerical().Parameter(0, i) = 1.f + static_cast<float>(i);
      tracks.Numerical().Parameter(1, i) = 0.f;
      tracks.Numerical().Parameter(2, i) = 0.f;
      tracks.Numerical().Parameter(3, i) = 1.f + static_cast<float>(i);
      tracks.Numerical().Parameter(4, i) = 2.f;
      tracks.Numerical().Parameter(5, i) = 3.f;
      for (unsigned int covariance = 0; covariance < 21; ++covariance) {
        tracks.Numerical().Covariance(covariance, i) = 0.f;
      }
      tracks.SourceId(i) = 101 + static_cast<int>(i);
      tracks.Pdg(i) = i == 0 ? 211 : -211;
      tracks.Charge(i) = i == 0 ? 1 : -1;
      tracks.PrimaryVertexIndex(i) = -1;
      tracks.NumberOfPixelHits(i) = 0;
      tracks.ChiToPrimaryVertex(i) = 0.f;
    }

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0].eventId = 1;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
  }
}

int cbmroot_xpu_smoke(const char* device = "hip1")
{
  try {
    LoadLibrary("libxpu");
    LoadLibrary("libCbmRecoBase");
    LoadLibrary("libKFParticle");

    SelectCbmXpuDevice(device);
    cbm::Xpu::Instance().Init();
    Require(cbm::Xpu::Instance().IsInitialized(), "CBMRoot did not initialize XPU");
    Require(!xpu::device::all().empty(), "CBMRoot XPU runtime exposes no devices");

    KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
    KFParticleGpuRuntimeSettings settings;
    settings.initializeXpuIfNeeded = false;
    runtime.Initialize(settings);
    Require(runtime.IsInitialized(), "KFParticle GPU runtime did not initialize");
    Require(!runtime.InitializedXpu(), "KFParticle initialized a second XPU runtime");
    Require(&runtime.GetQueue() == &runtime.GetQueue(), "KFParticle queue is not persistent");

    KFParticleGpuSteering& steering = runtime.GetSteering();
    FillRoundTripInput(steering.GetBuffers());
    steering.RunRoundTrip(0.13957f, 0);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(steering.GetBuffers().HostCandidates());
    Require(candidates.Size() == 2, "KFParticle round-trip did not produce two candidates");
    Require(candidates.Daughters().Size() == 2, "KFParticle round-trip did not preserve daughters");
    Require(candidates.Metadata().EventIndex(0) == 0, "KFParticle candidate event index is invalid");
    Require(candidates.Daughters().SourceId(0) == 101, "first source ID did not survive device round-trip");
    Require(candidates.Daughters().SourceId(1) == 102, "second source ID did not survive device round-trip");
    Require(candidates.OverflowFlags() == 0, "unexpected candidate-pool overflow");

    runtime.Finalize();
    Require(!runtime.IsInitialized(), "KFParticle runtime did not finalize");
    Require(!xpu::device::all().empty(), "KFParticle finalized the global CBMRoot XPU runtime");

    std::cout << "PASS cbmroot-xpu-runtime - CBMRoot initializes XPU; KFParticle reuses it, "
                 "runs a HIP/CPU kernel on its own queue, and leaves the global runtime active"
              << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL cbmroot-xpu-runtime - " << error.what() << std::endl;
    return 1;
  }
}
