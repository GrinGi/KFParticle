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
#include "KFParticleGpuRuntime.h"
#include "KFParticleTopoReconstructor.h"
#include "KFPTrackVector.h"
#include "KFPVertex.h"
#include "KFVertex.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
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

  KFVertex MakePrimaryVertex()
  {
    KFPVertex source;
    source.SetXYZ(0.f, 0.f, 0.f);
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
    // rejection guard. A constant By exercises the field-aware CPU/GPU path.
    tracks.SetParameter(5.f, 0u, index);
    tracks.SetParameter(y, 1u, index);
    tracks.SetParameter(0.f, 2u, index);
    tracks.SetParameter(px, 3u, index);
    tracks.SetParameter(py, 4u, index);
    tracks.SetParameter(pz, 5u, index);
    for (unsigned int covariance = 0u; covariance < 21u; ++covariance) {
      const bool diagonal = covariance == 0u || covariance == 2u || covariance == 5u
                            || covariance == 9u || covariance == 14u || covariance == 20u;
      tracks.SetCovariance(diagonal ? 0.01f : 0.f, covariance, index);
    }
    for (unsigned int coefficient = 0u; coefficient < 10u; ++coefficient) {
      tracks.SetFieldCoefficient(coefficient == 3u ? 0.1f : 0.f, coefficient, index);
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

  std::vector<KFParticle> RunCpuFinder(KFPTrackVector& first,
                                       KFPTrackVector& last,
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

  void CheckEquivalentCase(const V0Case& test, std::uint64_t eventId)
  {
    KFPTrackVector first;
    KFPTrackVector last;
    std::vector<float> chiToPrimaryVertex;
    std::vector<KFVertex> primaryVertices;
    MakeInput(test, first, last, chiToPrimaryVertex, primaryVertices);

    const auto cpuParticles = RunCpuFinder(first, last, primaryVertices.front(), test.motherPdg);
    unsigned int unresolvedCpuCandidates = 0u;
    const auto cpu = cbm::algo::kfp::GpuDiagnosticComparator::ExtractCpuV0Candidates(
      eventId, cpuParticles, unresolvedCpuCandidates);

    const auto gpuResult = cbm::algo::kfp::GpuDiagnosticRunner().Run(
      eventId, first, last, chiToPrimaryVertex, primaryVertices);
    const auto gpu = cbm::algo::kfp::GpuDiagnosticComparator::MakeGpuCandidateSnapshots(gpuResult);
    const auto comparison = cbm::algo::kfp::GpuDiagnosticComparator::Compare(
      eventId, cpu, gpu, unresolvedCpuCandidates);

    const std::string label = std::string(test.name) + ": ";
    Require(gpuResult.status == cbm::algo::kfp::GpuDiagnosticStatus::Completed,
            label + "GPU diagnostic did not complete: " + gpuResult.message);
    Require(gpuResult.taskCapacity == 1u, label + "unexpected default-plan task count");
    Require(unresolvedCpuCandidates == 0u, label + "CPU finder produced unresolved V0 daughters");
    if (cpu.size() != 1u || gpu.size() != 1u || comparison.matched != 1u) {
      std::cerr << "DETAIL cbmroot-cpu-gpu-v0-" << test.name
                << " - CPU particle/V0/unresolved " << cpuParticles.size() << '/' << cpu.size() << '/'
                << ", GPU status/task/raw/selected/overflow "
                << static_cast<int>(gpuResult.status) << '/' << gpuResult.taskCapacity << '/' << gpu.size() << '/'
                << gpuResult.selectedCandidates << '/' << gpuResult.overflowFlags << std::endl;
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

    std::cout << "PASS cbmroot-cpu-gpu-v0-" << test.name
              << " - CPU and GPU agree on V0 lineage and bounded fit observables" << std::endl;
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
