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
#include "KFParticleGpuDecayPlan.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuInputDataTransfer.h"
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

  void FillCascadeInput(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities capacities;
    capacities.tracks = 6u;
    capacities.events = 1u;
    // The relaxed fixture can build 9 V0 and 18 Xi/Omega candidates. Keep
    // this smoke test comfortably above that deterministic upper bound; its
    // purpose is runtime integration, not capacity-overflow validation.
    capacities.candidates = 32u;
    capacities.daughterIds = 96u;
    capacities.selectedCandidates = 16u;
    capacities.twoDaughterTasks = 32u;
    capacities.v0TrackTasks = 32u;
    capacities.nonhomogeneousField = true;
    buffers.EnsureCapacity(capacities);
    buffers.SetInputSizes(6u, 0u, 1u);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    const int pdg[6] = {211, 2212, -211, -211, -211, -321};
    const int charge[6] = {1, 1, -1, -1, -1, -1};
    for (unsigned int index = 0u; index < 6u; ++index) {
      KFParticleGpuTrackState track;
      track.X() = 0.1f * static_cast<float>(index);
      track.Y() = -0.05f * static_cast<float>(index);
      track.Z() = 0.02f * static_cast<float>(index);
      track.Px() = 0.2f + 0.1f * static_cast<float>(index);
      track.Py() = 0.1f + 0.05f * static_cast<float>(index);
      track.Pz() = 0.7f + 0.1f * static_cast<float>(index);
      track.Covariance(0, 0) = 0.3f;
      track.Covariance(1, 1) = 0.3f;
      track.Covariance(2, 2) = 0.3f;
      track.Covariance(3, 3) = 0.1f;
      track.Covariance(4, 4) = 0.1f;
      track.Covariance(5, 5) = 0.1f;
      StoreTrackState(track, tracks, index);
      tracks.SourceId(index) = 1400 + static_cast<int>(index);
      tracks.Pdg(index) = pdg[index];
      tracks.Charge(index) = charge[index];
      tracks.PrimaryVertexIndex(index) = -1;
      tracks.ChiToPrimaryVertex(index) = 10.f;
      KFParticleGpuFieldRegion field;
      field.Coefficient(3) = 20.f;
      field.Coefficient(9) = track.Z();
      StoreFieldRegion(field, tracks, index);
    }
    KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    event = KFParticleGpuEventDesc();
    event.eventId = 1414u;
    event.TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0u, 2u);
    event.TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0u, 1u);
    event.TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(1u, 1u);
    event.TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2u, 4u);
    event.TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2u, 3u);
    event.TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(5u, 1u);
  }

  void ConfigureCascadePlan(KFParticleGpuDecayPlan& plan)
  {
    plan.Clear();
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
    KFParticleGpuTwoDaughterChannel k0 = MakeK0ShortToPiPlusPiMinusChannel();
    KFParticleGpuTwoDaughterChannel lambda = MakeLambdaToProtonPiMinusChannel();
    KFParticleGpuTwoDaughterChannel antiLambda = MakeAntiLambdaToAntiProtonPiPlusChannel();
    relax(k0); relax(lambda); relax(antiLambda);
    plan.AddTwoDaughterChannel(k0);
    plan.AddTwoDaughterChannel(lambda);
    plan.AddTwoDaughterChannel(antiLambda);
    AddDefaultV0TrackCascadeChannels(plan);
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

    FillCascadeInput(steering.GetBuffers());
    ConfigureCascadePlan(steering.GetDecayPlan());
    steering.RunDecayPlan(0u, 8u);
    const auto& cascades = steering.LastV0TrackCascadeResults();
    Require(cascades.size() == 4u, "cascade plan did not expose four Xi/Omega channels");
    Require(cascades[0].constructedCandidates > 0u
              && cascades[2].constructedCandidates > 0u,
            "cascade plan did not produce Xi and Omega candidates");
    const KFParticleGpuV0TrackRoutingMonitorData& routing =
      steering.LastV0TrackRoutingMonitorData();
    Require(routing.groupLaunches == 2u && routing.descriptorCount == 4u,
            "cascade plan did not use the fused two-group routing plan");
    Require(routing.acceptedTasks == routing.storedTasks,
            "fused cascade routing lost accepted tasks");
    Require(routing.candidates > 0u
              && routing.candidates <= routing.storedTasks,
            "fused cascade construction produced no valid candidates");
    Require(routing.blockReservations > 0u
              && routing.blockReservations <= routing.acceptedTasks,
            "fused cascade routing did not use bounded block reservations");
    if (routing.overflowFlags != 0u) {
      throw std::runtime_error(
        "fused cascade routing reported overflow: flags="
        + std::to_string(routing.overflowFlags)
        + " accepted=" + std::to_string(routing.acceptedTasks)
        + " stored=" + std::to_string(routing.storedTasks)
        + " candidates=" + std::to_string(routing.candidates));
    }
    Require(steering.GetBuffers().HostCandidates().Size() > 3u,
            "cascade plan did not append candidates after the V0 generation");

    runtime.Finalize();
    Require(!runtime.IsInitialized(), "KFParticle runtime did not finalize");
    Require(!xpu::device::all().empty(), "KFParticle finalized the global CBMRoot XPU runtime");

    std::cout << "PASS cbmroot-xpu-runtime - CBMRoot initializes XPU; KFParticle reuses it, "
                 "runs V0 and Xi/Omega cascade kernels on its own queue, and leaves the global runtime active"
              << std::endl;
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "FAIL cbmroot-xpu-runtime - " << error.what() << std::endl;
    try {
      KFParticleGpuRuntime& runtime = KFParticleGpuRuntime::Instance();
      if (runtime.IsInitialized()) {
        runtime.Finalize();
      }
    }
    catch (...) {
      // Preserve the original smoke-test failure.
    }
    return 1;
  }
}
