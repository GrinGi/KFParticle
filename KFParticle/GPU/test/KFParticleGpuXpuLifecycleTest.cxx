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
#include "KFParticleGpuDeviceStorage.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuKernelState.h"
#include "KFParticleGpuKernels.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuRuntime.h"
#include "KFParticleGpuSelection.h"
#include "KFParticleGpuSteering.h"
#include "KFParticleGpuTwoDaughter.h"
#include "xpu/KFParticleGpuXpuBaseline.h"

#include <xpu/host.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
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

namespace
{
  constexpr float kCpuReferencePionMass = 0.13957039f;
  constexpr float kCpuReferenceProtonMass = 0.9382720813f;
  constexpr float kCpuReferenceK0ShortMass = 0.497614f;
  constexpr float kCpuReferenceLambdaMass = 1.115683f;

#ifdef PANDA_STT
  constexpr float kCpuReferenceK0ShortMassSigma = 12.0e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 2.7e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = -3.f;
#elif defined ALICE_ITS
  constexpr float kCpuReferenceK0ShortMassSigma = 17.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 5.9e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#elif defined STAR_HFT
  constexpr float kCpuReferenceK0ShortMassSigma = 17.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 5.9e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#elif defined CBM
  constexpr float kCpuReferenceK0ShortMassSigma = 3.7e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 1.5e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#else
  constexpr float kCpuReferenceK0ShortMassSigma = 4.9e-3f;
  constexpr float kCpuReferenceLambdaMassSigma = 2.1e-3f;
  constexpr float kCpuReferenceSecondaryTopoChi2 = 5.f;
#endif

  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  void Pass(const char* name, const char* details)
  {
    static int checkNumber = 0;
    ++checkNumber;
    std::cerr << "PASS [" << std::setw(2) << std::setfill('0') << checkNumber
              << "] " << name << " - " << details << '\n';
  }

  bool AlmostEqual(float lhs, float rhs, float tolerance = 1.e-6f)
  {
    return std::fabs(lhs - rhs) <= tolerance;
  }

  void ExpectSelectionResultClose(const KFParticleGpuV0SelectionResult& actual,
                                  const KFParticleGpuV0SelectionResult& expected,
                                  float tolerance = 1.e-5f)
  {
    assert(actual.candidateIndex == expected.candidateIndex);
    assert(actual.channelId == expected.channelId);
    assert(actual.eventIndex == expected.eventIndex);
    assert(actual.bestPrimaryVertexIndex == expected.bestPrimaryVertexIndex);
    assert(actual.selectionClass == expected.selectionClass);
    assert(actual.rejectionReasons == expected.rejectionReasons);
    assert(AlmostEqual(actual.observables.mass, expected.observables.mass, tolerance));
    assert(AlmostEqual(actual.observables.massError, expected.observables.massError, tolerance));
    assert(AlmostEqual(actual.observables.geometricChi2PerNdf,
                       expected.observables.geometricChi2PerNdf,
                       tolerance));
    assert(AlmostEqual(actual.observables.nearestPrimaryVertexDistance,
                       expected.observables.nearestPrimaryVertexDistance,
                       tolerance));
    assert(AlmostEqual(actual.observables.nearestPrimaryVertexDistanceError,
                       expected.observables.nearestPrimaryVertexDistanceError,
                       tolerance));
    assert(AlmostEqual(actual.observables.nearestPrimaryVertexLdL,
                       expected.observables.nearestPrimaryVertexLdL,
                       tolerance));
    assert(AlmostEqual(actual.observables.bestPrimaryVertexTopoChi2PerNdf,
                       expected.observables.bestPrimaryVertexTopoChi2PerNdf,
                       tolerance));
    assert(AlmostEqual(actual.observables.bestPrimaryVertexDecayLength,
                       expected.observables.bestPrimaryVertexDecayLength,
                       tolerance));
    assert(AlmostEqual(actual.observables.bestPrimaryVertexDecayLengthError,
                       expected.observables.bestPrimaryVertexDecayLengthError,
                       tolerance));
    assert(AlmostEqual(actual.observables.bestPrimaryVertexLdL,
                       expected.observables.bestPrimaryVertexLdL,
                       tolerance));
  }

  unsigned int FieldAwareProfileIterations()
  {
    const char* enabled = std::getenv("KFPARTICLE_GPU_TEST_PROFILE_FIELD_AWARE");
    if (!enabled || enabled[0] == '\0' || enabled[0] == '0') {
      return 0u;
    }

    const char* requested = std::getenv("KFPARTICLE_GPU_TEST_PROFILE_ITERATIONS");
    if (!requested || requested[0] == '\0') {
      return 500u;
    }
    const unsigned long parsed = std::strtoul(requested, nullptr, 10);
    return parsed > 0u && parsed <= 1000000u ? static_cast<unsigned int>(parsed) : 500u;
  }

  constexpr float kDiagnosticFieldBy = 200.f;

  void ExpectFitStateClose(const KFParticleGpuFitState& actual,
                           const KFParticleGpuFitState& expected,
                           float tolerance = 1.e-5f)
  {
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      assert(AlmostEqual(actual.Parameter(i), expected.Parameter(i), tolerance));
    }
    assert(AlmostEqual(actual.Chi2(), expected.Chi2(), tolerance));
    assert(actual.NDF() == expected.NDF());
    assert(actual.Q() == expected.Q());
    assert(AlmostEqual(actual.SumDaughterMass(), expected.SumDaughterMass(), tolerance));
    assert(AlmostEqual(actual.MassHypo(), expected.MassHypo(), tolerance));
  }

  void ExpectFieldAwareSeedClose(const KFParticleGpuFitState& actual,
                                 const KFParticleGpuFitState& expected)
  {
    // HIP and host libm can differ slightly in the diagnostic constant-By path.
    // Keep this comparison strict enough to catch wrong data flow, but do not
    // require bitwise-like agreement from iterative trigonometric transport.
    ExpectFitStateClose(actual, expected, 5.e-4f);
  }

  struct FieldAwareEnergyFitTolerance
  {
    float state = 0.f;
    float chi2 = 0.f;
    float mass = 0.f;
  };

  FieldAwareEnergyFitTolerance DefaultV0EnergyFitTolerance(int motherPdg)
  {
    // These tolerances cover HIP/CPU float and libm differences in the
    // iterative transport plus numerical-Jacobian energy-fit path.
    if (motherPdg == 310) {
      return {7.e-4f, 2.e-3f, 2.e-4f};
    }
    return {1.e-3f, 3.e-3f, 3.e-4f};
  }

  void ExpectDefaultV0EnergyFitClose(const KFParticleGpuFitState& actual,
                                     const KFParticleGpuFitState& expected,
                                     int motherPdg)
  {
    const FieldAwareEnergyFitTolerance tolerance =
      DefaultV0EnergyFitTolerance(motherPdg);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      assert(AlmostEqual(actual.Parameter(i), expected.Parameter(i), tolerance.state));
    }
    assert(AlmostEqual(actual.Chi2(), expected.Chi2(), tolerance.chi2));
    assert(actual.NDF() == expected.NDF());
    assert(actual.Q() == expected.Q());
    assert(AlmostEqual(actual.SumDaughterMass(), expected.SumDaughterMass(), tolerance.mass));
    assert(AlmostEqual(actual.MassHypo(), expected.MassHypo(), tolerance.mass));

    float actualMass = 0.f;
    float actualMassError = 0.f;
    float expectedMass = 0.f;
    float expectedMassError = 0.f;
    assert(KFParticleGpuMath::GetMass(actual, actualMass, actualMassError));
    assert(KFParticleGpuMath::GetMass(expected, expectedMass, expectedMassError));
    assert(AlmostEqual(actualMass, expectedMass, tolerance.mass));
  }

  void StoreDiagnosticField(KFParticleGpuInputTrackSoAView& tracks, unsigned int track)
  {
    KFParticleGpuFieldRegion diagnosticField;
    diagnosticField.Coefficient(3) = kDiagnosticFieldBy;
    diagnosticField.Coefficient(9) = tracks.Numerical().Parameter(2, track);
    StoreFieldRegion(diagnosticField, tracks, track);
  }

  struct ReferenceState
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double energy = 0.0;
    int charge = 0;
  };

  ReferenceState MakeReferenceState(const KFParticleGpuFitState& source)
  {
    ReferenceState state;
    state.x = source.X();
    state.y = source.Y();
    state.z = source.Z();
    state.px = source.Px();
    state.py = source.Py();
    state.pz = source.Pz();
    state.energy = source.E();
    state.charge = source.Q();
    return state;
  }

  ReferenceState MakeReferenceState(const KFParticleGpuTrackState& source,
                                    float mass,
                                    int charge)
  {
    ReferenceState state;
    state.x = source.X();
    state.y = source.Y();
    state.z = source.Z();
    state.px = source.Px();
    state.py = source.Py();
    state.pz = source.Pz();
    state.energy = std::sqrt(state.px * state.px + state.py * state.py
                             + state.pz * state.pz
                             + static_cast<double>(mass) * static_cast<double>(mass));
    state.charge = charge;
    return state;
  }

  void ReferenceGetDStoParticleLine(const ReferenceState& first,
                                    const ReferenceState& second,
                                    double dS[2])
  {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    const double dz = first.z - second.z;
    const double p1p1 = first.px * first.px + first.py * first.py + first.pz * first.pz;
    const double p2p2 = second.px * second.px + second.py * second.py + second.pz * second.pz;
    const double p1p2 = first.px * second.px + first.py * second.py + first.pz * second.pz;
    const double r1p1 = dx * first.px + dy * first.py + dz * first.pz;
    const double r1p2 = dx * second.px + dy * second.py + dz * second.pz;
    double determinant = p1p2 * p1p2 - p1p1 * p2p2;

    if (determinant > -1.e-8 && determinant < 1.e-8) {
      determinant = determinant < 0.0 ? -1.e-8 : 1.e-8;
    }

    dS[0] = (r1p1 * p2p2 - r1p2 * p1p2) / determinant;
    dS[1] = (r1p1 * p1p2 - r1p2 * p1p1) / determinant;
  }

  ReferenceState ReferenceTransportConstantBy(const ReferenceState& particle,
                                              double dS,
                                              double by)
  {
    const double kCLight = 0.000299792458;
    const double bq = by * static_cast<double>(particle.charge) * kCLight;
    ReferenceState transported = particle;

    if (std::fabs(bq) < 1.e-8) {
      transported.x = particle.x + particle.px * dS;
      transported.y = particle.y + particle.py * dS;
      transported.z = particle.z + particle.pz * dS;
      return transported;
    }

    const double bs = bq * dS;
    const double sinBs = std::sin(bs);
    const double cosBs = std::cos(bs);
    double sB = 0.0;
    double cB = 0.0;
    if (std::fabs(bs) > 1.e-8) {
      sB = sinBs / bq;
      cB = (1.0 - cosBs) / bq;
    }
    else {
      sB = dS;
      cB = 0.5 * dS * bs;
    }

    transported.x = particle.x + sB * particle.px - cB * particle.pz;
    transported.y = particle.y + dS * particle.py;
    transported.z = particle.z + cB * particle.px + sB * particle.pz;
    transported.px = cosBs * particle.px - sinBs * particle.pz;
    transported.py = particle.py;
    transported.pz = sinBs * particle.px + cosBs * particle.pz;
    return transported;
  }

  bool ReferenceBuildConstantByDcaKinematicMother(const ReferenceState& referenceFirst,
                                                  const ReferenceState& referenceSecond,
                                                  double by,
                                                  ReferenceState& mother)
  {
    const bool firstIsStraight =
      std::fabs(by * static_cast<double>(referenceFirst.charge)) < 1.e-8;
    const bool secondIsStraight =
      std::fabs(by * static_cast<double>(referenceSecond.charge)) < 1.e-8;

    double dS[2] = {0.0, 0.0};
    ReferenceGetDStoParticleLine(referenceFirst, referenceSecond, dS);
    if (std::fabs(dS[0] * referenceFirst.pz) > 1000.0
        || std::fabs(dS[1] * referenceSecond.pz) > 1000.0) {
      return false;
    }

    ReferenceState firstAtDca;
    ReferenceState secondAtDca;
    if (firstIsStraight && secondIsStraight) {
      firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], 0.0);
      secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], 0.0);
    }
    else {
      for (int iteration = 0; iteration < 2; ++iteration) {
        firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], by);
        secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], by);
        double correction[2] = {0.0, 0.0};
        ReferenceGetDStoParticleLine(firstAtDca, secondAtDca, correction);
        dS[0] += correction[0];
        dS[1] += correction[1];
        if (std::fabs(dS[0] * referenceFirst.pz) > 1000.0
            || std::fabs(dS[1] * referenceSecond.pz) > 1000.0) {
          return false;
        }
      }
      firstAtDca = ReferenceTransportConstantBy(referenceFirst, dS[0], by);
      secondAtDca = ReferenceTransportConstantBy(referenceSecond, dS[1], by);
    }

    mother.x = 0.5 * (firstAtDca.x + secondAtDca.x);
    mother.y = 0.5 * (firstAtDca.y + secondAtDca.y);
    mother.z = 0.5 * (firstAtDca.z + secondAtDca.z);
    mother.px = firstAtDca.px + secondAtDca.px;
    mother.py = firstAtDca.py + secondAtDca.py;
    mother.pz = firstAtDca.pz + secondAtDca.pz;
    mother.energy = firstAtDca.energy + secondAtDca.energy;
    mother.charge = firstAtDca.charge + secondAtDca.charge;
    return true;
  }

  bool ReferenceBuildConstantByDcaKinematicMother(const KFParticleGpuFitState& first,
                                                  const KFParticleGpuFitState& second,
                                                  double by,
                                                  ReferenceState& mother)
  {
    return ReferenceBuildConstantByDcaKinematicMother(
      MakeReferenceState(first), MakeReferenceState(second), by, mother);
  }

  void ExpectFitStateMatchesReference(const KFParticleGpuFitState& actual,
                                      const ReferenceState& expected,
                                      float tolerance)
  {
    assert(AlmostEqual(actual.X(), static_cast<float>(expected.x), tolerance));
    assert(AlmostEqual(actual.Y(), static_cast<float>(expected.y), tolerance));
    assert(AlmostEqual(actual.Z(), static_cast<float>(expected.z), tolerance));
    assert(AlmostEqual(actual.Px(), static_cast<float>(expected.px), tolerance));
    assert(AlmostEqual(actual.Py(), static_cast<float>(expected.py), tolerance));
    assert(AlmostEqual(actual.Pz(), static_cast<float>(expected.pz), tolerance));
    assert(AlmostEqual(actual.E(), static_cast<float>(expected.energy), tolerance));
    assert(actual.Q() == expected.charge);
  }

  void FillInput(KFParticleGpuBufferManager& buffers,
                 const KFParticleGpuBufferCapacities& capacities)
  {
    buffers.SetInputSizes(2, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    for (unsigned int i = 0; i < capacities.tracks; ++i) {
      KFParticleGpuTrackState track;
      track.X() = 1.f + i;
      track.Y() = 10.f + i;
      track.Z() = 20.f + i;
      track.Px() = 2.f + i;
      track.Py() = 3.f + i;
      track.Pz() = 4.f + i;
      track.Covariance(0) = 100.f + i;
      StoreTrackState(track, tracks, i);
      tracks.SourceId(i) = 17 + static_cast<int>(2 * i);
      tracks.Pdg(i) = (i % 2 == 0) ? 211 : -211;
      tracks.Charge(i) = (i % 2 == 0) ? 1 : -1;
      tracks.PrimaryVertexIndex(i) = static_cast<int>(i) - 1;
      tracks.NumberOfPixelHits(i) = 4 + static_cast<int>(i);
      tracks.ChiToPrimaryVertex(i) = 12.f + i;

      KFParticleGpuFieldRegion field;
      field.Coefficient(0) = 3.f + i;
      field.Coefficient(1) = 0.25f;
      field.Coefficient(2) = 0.125f;
      field.Coefficient(3) = 4.f + i;
      field.Coefficient(4) = -0.5f;
      field.Coefficient(5) = 0.25f;
      field.Coefficient(6) = 5.f + i;
      field.Coefficient(7) = 0.75f;
      field.Coefficient(8) = -0.125f;
      field.Coefficient(9) = track.Z();
      StoreFieldRegion(field, tracks, i);
    }

    KFParticleGpuVertexState vertex;
    vertex.X() = 4.f;
    vertex.Covariance(0) = 40.f;
    vertex.Chi2() = 2.5f;
    vertex.NDF() = 5;
    vertex.NContributors() = 6;
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    StoreVertexState(vertex, vertices, 0);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0].eventId = 23;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].primaryVertices = KFParticleGpuRange(0, 1);
  }

  void TestHostViews(KFParticleGpuBufferManager& buffers)
  {
    assert(buffers.TrackSize() == 2);
    assert(buffers.VertexSize() == 1);
    assert(buffers.EventSize() == 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    assert(tracks.Size() == 2);
    assert(tracks.Stride() == buffers.Capacities().tracks);
    assert(tracks.Numerical().Parameter(0, 1) == 2.f);
    assert(tracks.Numerical().Parameter(3, 1) == 3.f);
    assert(tracks.Numerical().Covariance(0, 1) == 101.f);
    assert(tracks.FieldCoefficient(0, 1) == 4.f);
    assert(HasFieldRegions(MakeConstView(tracks)));
    KFParticleGpuFieldRegion loadedField;
    assert(LoadFieldRegionOrZero(MakeConstView(tracks), 1, loadedField));
    assert(loadedField.Coefficient(3) == 5.f);
    const KFParticleGpuFieldValue fieldAtTrack =
      EvaluateTrackField(MakeConstView(tracks), 1, tracks.Numerical().Parameter(2, 1));
    assert(fieldAtTrack.x == 4.f);
    assert(fieldAtTrack.y == 5.f);
    assert(fieldAtTrack.z == 6.f);
    assert(!LoadFieldRegionOrZero(MakeConstView(tracks), tracks.Size(), loadedField));
    assert(loadedField.Get(0.f).x == 0.f);
    assert(tracks.ChiToPrimaryVertex(1) == 13.f);
    assert(tracks.SourceId(1) == 19);
    assert(tracks.Pdg(1) == -211);
    assert(tracks.Charge(1) == -1);
    assert(tracks.PrimaryVertexIndex(1) == 0);
    assert(tracks.NumberOfPixelHits(1) == 5);

    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    assert(vertices.Size() == 1);
    assert(vertices.Stride() == buffers.Capacities().vertices);
    assert(vertices.Parameter(0, 0) == 4.f);
    assert(vertices.Covariance(0, 0) == 40.f);
    assert(vertices.Chi2(0) == 2.5f);
    assert(vertices.NDF(0) == 5);
    assert(vertices.NContributors(0) == 6);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    assert(events[0].eventId == 23);
    assert(events[0].TrackSet(SecondaryPositiveFirst).tracks.End() == 2);
    assert(events[0].TrackSet(SecondaryPositiveFirst).Species(Pion).End() == 2);
    assert(events[0].primaryVertices.End() == 1);
    Pass("host-views", "component-major SoA stride and packed track/vertex/event accessors");
  }

  void TestDeviceInputProbe(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.UploadInput();

    xpu::buffer<float> floatChecks(11, xpu::buf_io);
    xpu::buffer<int> integerChecks(8, xpu::buf_io);
    xpu::buffer<unsigned int> unsignedChecks(9, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuInputLayoutProbe>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      buffers.DeviceEvents(),
      floatChecks.get(),
      integerChecks.get(),
      unsignedChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().copy(unsignedChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    const unsigned int* unsigneds = HostPointer(unsignedChecks);
    assert(floats[0] == 2.f);
    assert(floats[1] == 3.f);
    assert(floats[2] == 101.f);
    assert(floats[3] == 4.f);
    assert(floats[4] == 13.f);
    assert(floats[5] == 4.f);
    assert(floats[6] == 40.f);
    assert(floats[7] == 2.5f);
    assert(floats[8] == 4.f);
    assert(floats[9] == 5.f);
    assert(floats[10] == 6.f);
    assert(integers[0] == 19);
    assert(integers[1] == -211);
    assert(integers[2] == -1);
    assert(integers[3] == 0);
    assert(integers[4] == 5);
    assert(integers[5] == 5);
    assert(integers[6] == 6);
    assert(integers[7] == 23);
    assert(unsigneds[0] == 2);
    assert(unsigneds[1] == buffers.Capacities().tracks);
    assert(unsigneds[2] == 1);
    assert(unsigneds[3] == buffers.Capacities().vertices);
    assert(unsigneds[4] == 0);
    assert(unsigneds[5] == 2);
    assert(unsigneds[6] == 0);
    assert(unsigneds[7] == 1);
    assert(unsigneds[8] == 1);
    Pass("device-input-probe", "host-to-device upload plus device-side SoA/event and field-region reads");
  }

  void TestCandidateReset(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuCandidatePoolView candidates = buffers.HostCandidates();
    candidates.SizeData()[0] = 99;
    candidates.Daughters().SizeData()[0] = 98;
    candidates.OverflowFlagsData()[0] = 97;

    buffers.ResetCandidates();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView resetCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(resetCandidates.Size() == 0);
    assert(resetCandidates.Daughters().Size() == 0);
    assert(resetCandidates.OverflowFlags() == 0);
    Pass("candidate-reset", "candidate, daughter, and overflow counters reset on device and read back");
  }

  void TestKernelStateContract(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuKernelState state(
      MakeConstView(buffers.DeviceInputTracks()), buffers.DeviceCandidates());

    assert(state.InputTracks().Size() == buffers.TrackSize());
    assert(state.InputTracks().Stride() == buffers.Capacities().tracks);
    assert(state.Candidates().Capacity() == buffers.Capacities().candidates);
    assert(state.Candidates().SizeData() == buffers.DeviceCandidates().SizeData());

    Pass("device-state-contract",
         "flat non-owning kernel ABI retains device views without buffer ownership");
  }

  void TestVisibleDeviceStorage(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.twoDaughterTasks = 17u;
    buffers.EnsureCapacity(requested);

    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    assert(storage.fTrackParameters.get() != nullptr);
    assert(storage.fEvents.get() != nullptr);
    assert(storage.fCandidateParameters.get() != nullptr);
    assert(storage.fV0SelectionResults.get() != nullptr);
    assert(storage.fSelectedCandidateIndices.get() != nullptr);
    assert(storage.fSelectedCandidateChannelIds.get() != nullptr);
    assert(storage.fTwoDaughterTasks.get() != nullptr);
    assert(storage.fTwoDaughterTaskCount.get() != nullptr);
    assert(storage.fTwoDaughterTotalPairCount.get() != nullptr);
    assert(storage.fTwoDaughterTaskOverflowFlags.get() != nullptr);
    assert(buffers.Capacities().twoDaughterTasks == requested.twoDaughterTasks);

    Pass("device-storage-owner",
         "visible input, work, raw-output, and selected-output XPU buffers persist together");
  }

  void TestPublishedKernelState(KFParticleGpuRuntime& runtime,
                                KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuDeviceStorage& storage = buffers.Storage();
    const KFParticleGpuKernels state(MakeConstView(buffers.DeviceInputTracks()),
                                     MakeConstView(buffers.DevicePrimaryVertices()),
                                     buffers.DeviceEvents(),
                                     storage.fTwoDaughterTasks.get(),
                                     buffers.Capacities().twoDaughterTasks,
                                     buffers.DeviceCandidates(),
                                     buffers.DeviceSelectedCandidates());
    xpu::set<TheKFParticleFinder>(state);

    xpu::buffer<unsigned int> checks(6, xpu::buf_io);
    unsigned int* hostChecks = HostPointer(checks);
    for (unsigned int i = 0; i < 6; ++i) {
      hostChecks[i] = 0u;
    }
    runtime.GetQueue().copy(checks, xpu::h2d);
    runtime.GetQueue().launch<KFParticleGpuKernelStateProbe>(xpu::n_threads(1), checks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(checks, xpu::d2h);
    runtime.GetQueue().wait();

    assert(hostChecks[0] == buffers.TrackSize());
    assert(hostChecks[1] == buffers.VertexSize());
    assert(hostChecks[2] == buffers.HostEvents()[0].eventId);
    assert(hostChecks[3] == buffers.Capacities().twoDaughterTasks);
    assert(hostChecks[4] == buffers.Capacities().candidates);
    assert(hostChecks[5] == buffers.Capacities().selectedCandidates);
    Pass("published-device-state",
         "XPU constant memory exposes common input, work, and output views to a kernel");
  }

  void TestDecayPlanDataTypes()
  {
    static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterChannel>::value,
                  "Two-daughter channel descriptors must remain device-copyable values");
    static_assert(std::is_trivially_copyable<KFParticleGpuCandidateRange>::value,
                  "Candidate ranges must remain flat host/device bookkeeping values");
    static_assert(std::is_trivially_copyable<KFParticleGpuTwoDaughterChannelResult>::value,
                  "Channel results must remain flat host/device bookkeeping values");

    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = 7;
    channel.firstTrackSet = PrimaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Proton;
    channel.secondSpecies = Pion;
    channel.flags = KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 3122;
    channel.firstDaughterPdg = 2212;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = 3;
    channel.firstMass = 0.938272f;
    channel.secondMass = 0.13957f;
    channel.motherMass = 1.115683f;
    channel.motherMassSigma = 0.0021f;
    channel.secondaryMassSigmaCut = 3.f;
    channel.maxSecondaryTopoChi2PerNdf = 5.f;
    channel.minSecondaryLdL = 10.f;
    channel.transportMode = KFGpuTransportFieldAware;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 4;
    channel.minSecondPixelHits = 5;
    channel.maxFirstChiToPrimaryVertex = 2.5f;
    channel.maxSecondChiToPrimaryVertex = 3.5f;

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 11u);
    assert(source.channelId == channel.channelId);
    assert(source.eventIndex == 11u);
    assert(source.firstTrackSet == PrimaryPositiveFirst);
    assert(source.secondTrackSet == SecondaryNegativeFirst);
    assert(source.firstSpecies == Proton);
    assert(source.secondSpecies == Pion);
    assert(source.flags == KFGpuTwoDaughterUseEnergyFit);
    assert(source.motherPdg == 3122);
    assert(source.firstDaughterPdg == 2212);
    assert(source.secondDaughterPdg == -211);
    assert(source.primaryVertexIndex == 3);
    assert(AlmostEqual(source.firstMass, 0.938272f));
    assert(AlmostEqual(source.secondMass, 0.13957f));
    assert(AlmostEqual(source.motherMass, 1.115683f));
    assert(AlmostEqual(source.motherMassSigma, 0.0021f));
    assert(AlmostEqual(source.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(source.maxSecondaryTopoChi2PerNdf, 5.f));
    assert(AlmostEqual(source.minSecondaryLdL, 10.f));
    assert(source.transportMode == KFGpuTransportFieldAware);
    assert(source.firstCharge == 1);
    assert(source.secondCharge == -1);
    assert(source.minFirstPixelHits == 4);
    assert(source.minSecondPixelHits == 5);
    assert(AlmostEqual(source.maxFirstChiToPrimaryVertex, 2.5f));
    assert(AlmostEqual(source.maxSecondChiToPrimaryVertex, 3.5f));

    KFParticleGpuCandidateRange range;
    assert(range.Empty());
    range.offset = 4;
    range.size = 2;
    range.daughterOffset = 8;
    range.daughterSize = 4;
    range.overflowFlags = CandidateCapacityExceeded;
    assert(!range.Empty());
    assert(range.End() == 6);
    assert(range.ContainsCandidate(4));
    assert(range.ContainsCandidate(5));
    assert(!range.ContainsCandidate(6));
    assert(range.LocalCandidateIndex(5) == 1);
    assert(range.DaughterEnd() == 12);
    assert(range.ContainsDaughter(8));
    assert(range.ContainsDaughter(11));
    assert(!range.ContainsDaughter(12));
    assert(range.LocalDaughterIndex(10) == 2);
    assert(range.HasOverflow(CandidateCapacityExceeded));
    assert(!range.HasOverflow(DaughterCapacityExceeded));
    assert(range.HasAnyOverflow());

    KFParticleGpuTwoDaughterChannelResult result;
    result.channelId = channel.channelId;
    result.motherPdg = channel.motherPdg;
    result.eventIndex = source.eventIndex;
    result.totalPairs = 10;
    result.acceptedTasks = 6;
    result.storedTasks = 4;
    result.candidates = range;
    assert(result.Truncated());
    assert(!result.Empty());
    assert(result.HasOverflow(CandidateCapacityExceeded));
    assert(result.HasAnyOverflow());
    assert(result.ContainsCandidate(4));
    result.storedTasks = result.acceptedTasks;
    assert(!result.Truncated());
    assert(result.candidates.overflowFlags == CandidateCapacityExceeded);
    assert(IsSupportedTwoDaughterTransportMode(KFGpuTransportStraightLine));
    assert(IsSupportedTwoDaughterTransportMode(KFGpuTransportFieldAware));
    assert(IsSupportedTwoDaughterEnergyFitTransportMode(KFGpuTransportStraightLine));
    assert(IsSupportedTwoDaughterEnergyFitTransportMode(KFGpuTransportFieldAware));

    Pass("decay-plan-data-types",
         "standalone two-daughter channel descriptors and result ranges are flat values");
  }

  void TestV0SelectionResultDataTypes(KFParticleGpuRuntime& runtime)
  {
    KFParticleGpuV0SelectionResult defaultResult;
    assert(defaultResult.selectionClass == KFGpuV0SelectionNotEvaluated);
    assert(defaultResult.rejectionReasons == KFGpuV0SelectionRejectNone);
    assert(defaultResult.bestPrimaryVertexIndex == -1);
    assert(!KFParticleGpuSelection::IsSelected(defaultResult));

    KFParticleGpuV0SelectionResult result;
    result.candidateIndex = 17u;
    result.channelId = KFGpuChannelLambdaToProtonPiMinus;
    result.eventIndex = 4u;
    result.bestPrimaryVertexIndex = 2;
    result.selectionClass = KFGpuV0SelectionRejected;
    result.rejectionReasons = static_cast<unsigned int>(
      KFGpuV0SelectionRejectMass | KFGpuV0SelectionRejectTopology);
    result.observables.mass = 1.12f;
    result.observables.massError = 0.004f;
    result.observables.geometricChi2PerNdf = 1.5f;
    result.observables.nearestPrimaryVertexDistance = 2.5f;
    result.observables.nearestPrimaryVertexDistanceError = 0.25f;
    result.observables.nearestPrimaryVertexLdL = 10.f;
    result.observables.bestPrimaryVertexTopoChi2PerNdf = 6.f;
    result.observables.bestPrimaryVertexDecayLength = 3.f;
    result.observables.bestPrimaryVertexDecayLengthError = 0.3f;
    result.observables.bestPrimaryVertexLdL = 10.f;

    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectMass));
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectTopology));
    assert(!KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDistance));
    assert(!KFParticleGpuSelection::IsSelected(result));

    xpu::buffer<KFParticleGpuV0SelectionResult> transfer(1, xpu::buf_device);
    runtime.GetQueue().memcpy(transfer.get(), &result, sizeof(result));
    runtime.GetQueue().wait();
    KFParticleGpuV0SelectionResult copied;
    runtime.GetQueue().memcpy(&copied, transfer.get(), sizeof(copied));
    runtime.GetQueue().wait();

    assert(copied.candidateIndex == result.candidateIndex);
    assert(copied.channelId == result.channelId);
    assert(copied.eventIndex == result.eventIndex);
    assert(copied.bestPrimaryVertexIndex == result.bestPrimaryVertexIndex);
    assert(copied.selectionClass == result.selectionClass);
    assert(copied.rejectionReasons == result.rejectionReasons);
    assert(AlmostEqual(copied.observables.mass, result.observables.mass));
    assert(AlmostEqual(copied.observables.massError, result.observables.massError));
    assert(AlmostEqual(copied.observables.geometricChi2PerNdf,
                       result.observables.geometricChi2PerNdf));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexDistance,
                       result.observables.nearestPrimaryVertexDistance));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexDistanceError,
                       result.observables.nearestPrimaryVertexDistanceError));
    assert(AlmostEqual(copied.observables.nearestPrimaryVertexLdL,
                       result.observables.nearestPrimaryVertexLdL));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexTopoChi2PerNdf,
                       result.observables.bestPrimaryVertexTopoChi2PerNdf));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexDecayLength,
                       result.observables.bestPrimaryVertexDecayLength));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexDecayLengthError,
                       result.observables.bestPrimaryVertexDecayLengthError));
    assert(AlmostEqual(copied.observables.bestPrimaryVertexLdL,
                       result.observables.bestPrimaryVertexLdL));

    result.selectionClass = KFGpuV0SelectionSecondary;
    result.rejectionReasons = KFGpuV0SelectionRejectNone;
    assert(KFParticleGpuSelection::IsSelected(result));
    result.selectionClass = KFGpuV0SelectionPrimary;
    assert(KFParticleGpuSelection::IsSelected(result));

    Pass("v0-selection-data-types",
         "flat V0 observables, classes, rejection masks, and host-device copies preserve selection diagnostics");
  }

  void TestPrimaryVertexTopologyObservables()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 1.f;
    candidate.Y() = 2.f;
    candidate.Z() = 3.f;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    KFParticleGpuVertexState primaryVertex;
    primaryVertex.X() = 4.f;
    primaryVertex.Y() = 6.f;
    primaryVertex.Z() = 3.f;
    primaryVertex.Covariance(0) = 0.6f;
    primaryVertex.Covariance(2) = 0.5f;
    primaryVertex.Covariance(5) = 0.4f;

    KFParticleGpuPrimaryVertexTopologyObservables observables;
    assert(KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
      candidate, primaryVertex, observables));
    assert(observables.valid == 1u);
    assert(AlmostEqual(observables.distance, 5.f));
    assert(AlmostEqual(observables.distanceError, 1.f));
    assert(AlmostEqual(observables.ldL, 5.f));
    assert(AlmostEqual(observables.chi2PerNdf, 25.f / 3.f));

    float vertexParameters[2 * KFParticleGpuVertexState::NumberOfParameters] = {
      4.f, 2.f,
      6.f, 2.f,
      3.f, 3.f
    };
    float vertexCovariances[2 * KFParticleGpuVertexState::NumberOfCovarianceElements] = {};
    vertexCovariances[0] = 0.6f;
    vertexCovariances[1] = 0.6f;
    vertexCovariances[2 * 2] = 0.5f;
    vertexCovariances[2 * 2 + 1] = 0.5f;
    vertexCovariances[5 * 2] = 0.4f;
    vertexCovariances[5 * 2 + 1] = 0.4f;
    float vertexChi2[2] = {0.f, 0.f};
    int vertexIntegers[2 * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 2u, 2u);

    int bestPrimaryVertexIndex = -1;
    KFParticleGpuPrimaryVertexTopologyObservables bestObservables;
    assert(KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      bestPrimaryVertexIndex,
      bestObservables));
    assert(bestPrimaryVertexIndex == 1);
    assert(AlmostEqual(bestObservables.distance, 1.f));
    assert(AlmostEqual(bestObservables.distanceError, 1.f));
    assert(AlmostEqual(bestObservables.ldL, 1.f));
    assert(AlmostEqual(bestObservables.chi2PerNdf, 1.f / 3.f));

    bestPrimaryVertexIndex = 7;
    assert(!KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 0u),
      bestPrimaryVertexIndex,
      bestObservables));
    assert(bestPrimaryVertexIndex == -1);
    assert(!KFParticleGpuSelection::FindBestPrimaryVertexTopology(
      candidate,
      MakeConstView(vertices),
      KFParticleGpuRange(2u, 1u),
      bestPrimaryVertexIndex,
      bestObservables));

    KFParticleGpuFitState invalidCandidate;
    invalidCandidate.Initialize();
    invalidCandidate.X() = 0.f;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      invalidCandidate.Covariance(i) = 0.f;
    }
    KFParticleGpuVertexState invalidVertex;
    invalidVertex.X() = 1.f;
    assert(!KFParticleGpuSelection::BuildPrimaryVertexTopologyObservables(
      invalidCandidate, invalidVertex, observables));
    assert(observables.valid == 0u);
    assert(observables.distanceError == 1.e8f);
    assert(observables.chi2PerNdf == 1.e8f);

    Pass("primary-vertex-topology-observables",
         "candidate-PV distance, uncertainty, L/dL, spatial chi2, and best-PV guards are deterministic");
  }

  void TestV0SelectionDecision()
  {
    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 1.f;
    candidate.Y() = 2.f;
    candidate.Z() = 3.f;
    candidate.E() = 1.f;
    candidate.Chi2() = 3.f;
    candidate.NDF() = 2;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    float vertexParameters[2 * KFParticleGpuVertexState::NumberOfParameters] = {
      4.f, 2.f,
      6.f, 2.f,
      3.f, 3.f
    };
    float vertexCovariances[2 * KFParticleGpuVertexState::NumberOfCovarianceElements] = {};
    vertexCovariances[0] = 0.6f;
    vertexCovariances[1] = 0.6f;
    vertexCovariances[2 * 2] = 0.5f;
    vertexCovariances[2 * 2 + 1] = 0.5f;
    vertexCovariances[5 * 2] = 0.4f;
    vertexCovariances[5 * 2 + 1] = 0.4f;
    float vertexChi2[2] = {0.f, 0.f};
    int vertexIntegers[2 * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents] = {};
    const KFParticleGpuVertexSoAView vertices(
      vertexParameters, vertexCovariances, vertexChi2, vertexIntegers, 2u, 2u);

    KFParticleGpuV0SelectionConfig config;
    config.expectedMass = 1.f;
    config.expectedMassSigma = 0.1f;
    config.massSigmaCut = 3.f;
    config.maxGeometricChi2PerNdf = 2.f;
    config.maxPrimaryVertexDistance = 200.f;
    config.minSecondaryLdL = 0.5f;
    config.maxPrimaryTopologyChi2PerNdf = 1.f;
    config.maxSecondaryTopologyChi2PerNdf = 1.f;
    config.requirePrimaryVertex = 1u;
    const unsigned int validFlags =
      static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit);

    KFParticleGpuV0SelectionResult result;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate,
      validFlags,
      5u,
      KFGpuChannelK0ShortToPiPlusPiMinus,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::IsSelected(result));
    assert(result.selectionClass == KFGpuV0SelectionPrimary);
    assert(result.bestPrimaryVertexIndex == 1);
    assert(result.rejectionReasons == KFGpuV0SelectionRejectNone);
    assert(AlmostEqual(result.observables.mass, 1.f));
    assert(AlmostEqual(result.observables.geometricChi2PerNdf, 1.5f));
    assert(AlmostEqual(result.observables.nearestPrimaryVertexDistance, 1.f));
    assert(AlmostEqual(result.observables.bestPrimaryVertexTopoChi2PerNdf, 1.f / 3.f));

    KFParticleGpuV0SelectionConfig secondaryConfig = config;
    secondaryConfig.maxPrimaryTopologyChi2PerNdf = 0.1f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      secondaryConfig, result);
    assert(KFParticleGpuSelection::IsSelected(result));
    assert(result.selectionClass == KFGpuV0SelectionSecondary);

    KFParticleGpuSelection::EvaluateV0Selection(
      candidate,
      KFGpuCandidateBuildFailed,
      5u,
      1u,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectBuild));
    assert(!KFParticleGpuSelection::IsSelected(result));

    KFParticleGpuV0SelectionConfig massConfig = config;
    massConfig.expectedMass = 2.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      massConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectMass));

    KFParticleGpuV0SelectionConfig geometryConfig = config;
    geometryConfig.maxGeometricChi2PerNdf = 1.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      geometryConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectGeometricChi2));

    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 0u),
      config, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectNoPrimaryVertex));

    KFParticleGpuV0SelectionConfig distanceConfig = config;
    distanceConfig.maxPrimaryVertexDistance = 0.5f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      distanceConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDistance));

    KFParticleGpuV0SelectionConfig decayConfig = secondaryConfig;
    decayConfig.minSecondaryLdL = 2.f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      decayConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectDecayLength));

    KFParticleGpuV0SelectionConfig topologyConfig = secondaryConfig;
    topologyConfig.maxSecondaryTopologyChi2PerNdf = 0.2f;
    KFParticleGpuSelection::EvaluateV0Selection(
      candidate, validFlags, 5u, 1u, 3u, MakeConstView(vertices), KFParticleGpuRange(0u, 2u),
      topologyConfig, result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectTopology));

    KFParticleGpuFitState nonFiniteCandidate = candidate;
    nonFiniteCandidate.X() = std::numeric_limits<float>::quiet_NaN();
    KFParticleGpuSelection::EvaluateV0Selection(
      nonFiniteCandidate,
      validFlags,
      5u,
      1u,
      3u,
      MakeConstView(vertices),
      KFParticleGpuRange(0u, 2u),
      config,
      result);
    assert(KFParticleGpuSelection::HasRejection(result, KFGpuV0SelectionRejectNonFinite));

    Pass("v0-selection-decision",
         "configurable default-V0 decision classifies primary/secondary candidates and reports every rejection reason");
  }

  bool ContainsSelectedCandidate(const KFParticleGpuConstSelectedCandidateIndexView& selected,
                                 unsigned int candidateIndex)
  {
    for (unsigned int i = 0; i < selected.Size(); ++i) {
      if (selected.Index(i) == candidateIndex) {
        return true;
      }
    }
    return false;
  }

  void TestSelectedCandidateOutput(KFParticleGpuRuntime& runtime,
                                   KFParticleGpuBufferManager& buffers)
  {
    FillInput(buffers, buffers.Capacities());
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    vertices.Parameter(0, 0) = 4.f;
    vertices.Parameter(1, 0) = 0.f;
    vertices.Parameter(2, 0) = 0.f;
    for (unsigned int i = 0; i < KFParticleGpuVertexState::NumberOfCovarianceElements; ++i) {
      vertices.Covariance(i, 0) = 0.f;
    }
    vertices.Covariance(0, 0) = 0.6f;
    vertices.Covariance(1, 0) = 0.5f;
    vertices.Covariance(2, 0) = 0.4f;

    KFParticleGpuFitState candidate;
    candidate.Initialize();
    candidate.X() = 3.f;
    candidate.Y() = 0.f;
    candidate.Z() = 0.f;
    candidate.E() = 1.f;
    candidate.Chi2() = 3.f;
    candidate.NDF() = 2;
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      candidate.Covariance(i) = 0.f;
    }
    candidate.Covariance(0, 0) = 0.4f;
    candidate.Covariance(1, 1) = 0.5f;
    candidate.Covariance(2, 2) = 0.6f;

    KFParticleGpuCandidatePoolView rawCandidates = buffers.HostCandidates();
    for (unsigned int i = 0; i < 3u; ++i) {
      StoreCandidateFit(candidate, rawCandidates, i);
      rawCandidates.Metadata().Pdg(i) = 310;
      rawCandidates.Metadata().PrimaryVertexIndex(i) = -1;
      rawCandidates.Metadata().EventIndex(i) = 0u;
      rawCandidates.Metadata().DaughterOffset(i) = 0u;
      rawCandidates.Metadata().DaughterCount(i) = 0u;
      rawCandidates.Metadata().Flags(i) = static_cast<unsigned int>(
        KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit);
    }
    rawCandidates.SizeData()[0] = 3u;
    rawCandidates.Daughters().SizeData()[0] = 0u;
    rawCandidates.OverflowFlagsData()[0] = 0u;

    KFParticleGpuTwoDaughterChannel channel = MakeK0ShortToPiPlusPiMinusChannel();
    channel.selection.expectedMass = 1.f;
    channel.selection.expectedMassSigma = 0.1f;
    channel.selection.massSigmaCut = 3.f;
    channel.selection.maxGeometricChi2PerNdf = 2.f;
    channel.selection.maxPrimaryVertexDistance = 200.f;
    channel.selection.minSecondaryLdL = 0.1f;
    channel.selection.maxPrimaryTopologyChi2PerNdf = 1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = 1.f;
    channel.selection.requirePrimaryVertex = 1u;

    buffers.UploadInput();
    buffers.UploadCandidates();
    const KFParticleGpuCandidateRange rawRange = {0u, 3u, 0u, 0u, 0u};
    const KFParticleGpuSelectedCandidateRange selectedRange =
      runtime.GetSteering().RunV0Selection(channel, rawRange, 0u);
    assert(selectedRange.offset == 0u);
    assert(selectedRange.size == 3u);
    assert(!selectedRange.Truncated());
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    assert(selected.Size() == 3u);
    assert(selected.OverflowFlags() == 0u);
    assert(ContainsSelectedCandidate(selected, 0u));
    assert(ContainsSelectedCandidate(selected, 1u));
    assert(ContainsSelectedCandidate(selected, 2u));
    assert(MakeConstView(buffers.HostCandidates()).Size() == 3u);

    xpu::buffer<unsigned int> limitedIndices(2u, xpu::buf_io);
    xpu::buffer<unsigned int> limitedSize(1u, xpu::buf_io);
    xpu::buffer<unsigned int> limitedOverflow(1u, xpu::buf_io);
    HostPointer(limitedSize)[0] = 0u;
    HostPointer(limitedOverflow)[0] = 0u;
    runtime.GetQueue().copy(limitedSize, xpu::h2d);
    runtime.GetQueue().copy(limitedOverflow, xpu::h2d);
    runtime.GetQueue().wait();
    const KFParticleGpuSelectedCandidateIndexView limitedSelected(
      limitedIndices.get(), limitedSize.get(), limitedOverflow.get(), 2u);
    runtime.GetQueue().launch<KFParticleGpuSelectV0Candidates>(
      xpu::n_threads(rawRange.size),
      MakeConstView(buffers.DeviceCandidates()),
      MakeConstView(buffers.DevicePrimaryVertices()),
      buffers.DeviceEvents(),
      0u,
      rawRange.offset,
      rawRange.size,
      0u,
      channel.selection,
      KFParticleGpuV0SelectionResultView(),
      limitedSelected);
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(limitedIndices, xpu::d2h);
    runtime.GetQueue().copy(limitedSize, xpu::d2h);
    runtime.GetQueue().copy(limitedOverflow, xpu::d2h);
    runtime.GetQueue().wait();
    const KFParticleGpuConstSelectedCandidateIndexView limitedHost(
      HostPointer(limitedIndices), HostPointer(limitedSize), HostPointer(limitedOverflow), 2u);
    assert(limitedHost.Size() == 2u);
    assert(limitedHost.OverflowFlags() == KFGpuSelectedCandidateCapacityExceeded);
    assert(limitedHost.Index(0) < rawRange.size);
    assert(limitedHost.Index(1) < rawRange.size);
    assert(limitedHost.Index(0) != limitedHost.Index(1));

    Pass("v0-selected-candidate-output",
         "selected candidate indices compact independently from raw candidates and report bounded-output overflow");
  }

  KFParticleGpuTwoDaughterChannel MakeTestChannel(unsigned int channelId,
                                                  int motherPdg,
                                                  KFParticleGpuTrackSpecies firstSpecies,
                                                  KFParticleGpuTrackSpecies secondSpecies)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = firstSpecies;
    channel.secondSpecies = secondSpecies;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = motherPdg;
    channel.firstDaughterPdg = 211;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.13957f;
    channel.secondMass = 0.13957f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 5;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  void TestDecayPlanChannelList()
  {
    KFParticleGpuDecayPlan plan;
    assert(plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 0);

    const KFParticleGpuTwoDaughterChannel first =
      MakeTestChannel(3u, 310, Pion, Pion);
    const KFParticleGpuTwoDaughterChannel second =
      MakeTestChannel(4u, 333, Kaon, Kaon);
    plan.AddTwoDaughterChannel(first);
    plan.AddTwoDaughterChannel(second);

    assert(!plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 2);
    assert(plan.TwoDaughterChannel(0).channelId == 3u);
    assert(plan.TwoDaughterChannel(0).motherPdg == 310);
    assert(plan.TwoDaughterChannel(0).firstSpecies == Pion);
    assert(plan.TwoDaughterChannel(1).channelId == 4u);
    assert(plan.TwoDaughterChannel(1).motherPdg == 333);
    assert(plan.TwoDaughterChannel(1).firstSpecies == Kaon);
    assert(plan.TwoDaughterChannel(1).secondSpecies == Kaon);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(plan.TwoDaughterChannel(1), 2u);
    assert(source.eventIndex == 2u);
    assert(source.motherPdg == 333);
    assert(source.firstSpecies == Kaon);

    bool outOfRangeRejected = false;
    try {
      (void) plan.TwoDaughterChannel(2);
    }
    catch (const std::out_of_range&) {
      outOfRangeRejected = true;
    }
    assert(outOfRangeRejected);

    plan.Clear();
    assert(plan.Empty());
    assert(plan.NumberOfTwoDaughterChannels() == 0);

    Pass("decay-plan-channel-list",
         "decay plan preserves ordered two-daughter channels and clear/index guards");
  }

  void TestDefaultV0DecayPlanBuilders()
  {
    KFParticleGpuDecayPlan plan;
    AddDefaultV0TwoDaughterChannels(plan);
    assert(plan.NumberOfTwoDaughterChannels() == 3u);

    const KFParticleGpuTwoDaughterChannel& k0 = plan.TwoDaughterChannel(0);
    assert(k0.channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(k0.motherPdg == 310);
    assert(k0.firstSpecies == Pion);
    assert(k0.secondSpecies == Pion);
    assert(k0.firstDaughterPdg == 211);
    assert(k0.secondDaughterPdg == -211);
    assert(AlmostEqual(k0.firstMass, kCpuReferencePionMass));
    assert(AlmostEqual(k0.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(k0.motherMass, kCpuReferenceK0ShortMass));
    assert(AlmostEqual(k0.motherMassSigma, kCpuReferenceK0ShortMassSigma));
    assert(AlmostEqual(k0.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(k0.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(k0.minSecondaryLdL, 10.f));
    assert(k0.transportMode == KFGpuTransportFieldAware);
    assert(AlmostEqual(k0.selection.expectedMass, kCpuReferenceK0ShortMass));
    assert(AlmostEqual(k0.selection.expectedMassSigma, kCpuReferenceK0ShortMassSigma));
    assert(AlmostEqual(k0.selection.maxGeometricChi2PerNdf, 3.f));
    assert(AlmostEqual(k0.selection.maxPrimaryVertexDistance, 200.f));
    assert(k0.selection.requirePrimaryVertex == 1u);

    const KFParticleGpuTwoDaughterChannel& lambda = plan.TwoDaughterChannel(1);
    assert(lambda.channelId == KFGpuChannelLambdaToProtonPiMinus);
    assert(lambda.motherPdg == 3122);
    assert(lambda.firstSpecies == Proton);
    assert(lambda.secondSpecies == Pion);
    assert(lambda.firstDaughterPdg == 2212);
    assert(lambda.secondDaughterPdg == -211);
    assert(AlmostEqual(lambda.firstMass, kCpuReferenceProtonMass));
    assert(AlmostEqual(lambda.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(lambda.motherMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(lambda.motherMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(lambda.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(lambda.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(lambda.minSecondaryLdL, 10.f));
    assert(lambda.transportMode == KFGpuTransportFieldAware);
    assert(AlmostEqual(lambda.selection.expectedMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(lambda.selection.expectedMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(lambda.selection.minSecondaryLdL, 10.f));

    const KFParticleGpuTwoDaughterChannel& antiLambda = plan.TwoDaughterChannel(2);
    assert(antiLambda.channelId == KFGpuChannelAntiLambdaToAntiProtonPiPlus);
    assert(antiLambda.motherPdg == -3122);
    assert(antiLambda.firstTrackSet == SecondaryNegativeFirst);
    assert(antiLambda.secondTrackSet == SecondaryPositiveFirst);
    assert(antiLambda.firstSpecies == Proton);
    assert(antiLambda.secondSpecies == Pion);
    assert(antiLambda.firstDaughterPdg == -2212);
    assert(antiLambda.secondDaughterPdg == 211);
    assert(AlmostEqual(antiLambda.firstMass, kCpuReferenceProtonMass));
    assert(AlmostEqual(antiLambda.secondMass, kCpuReferencePionMass));
    assert(AlmostEqual(antiLambda.motherMass, kCpuReferenceLambdaMass));
    assert(AlmostEqual(antiLambda.motherMassSigma, kCpuReferenceLambdaMassSigma));
    assert(AlmostEqual(antiLambda.secondaryMassSigmaCut, 3.f));
    assert(AlmostEqual(antiLambda.maxSecondaryTopoChi2PerNdf, kCpuReferenceSecondaryTopoChi2));
    assert(AlmostEqual(antiLambda.minSecondaryLdL, 10.f));
    assert(antiLambda.transportMode == KFGpuTransportFieldAware);
    assert(antiLambda.firstCharge == -1);
    assert(antiLambda.secondCharge == 1);

    Pass("decay-plan-default-v0-builders",
         "default V0 channel builders provide ordered field-aware K0S, Lambda, and anti-Lambda descriptors");
  }

  void TestKinematicMathSeed()
  {
    KFParticleGpuTrackState firstTrack;
    firstTrack.Px() = 3.f;
    firstTrack.Py() = 0.f;
    firstTrack.Pz() = 4.f;
    firstTrack.Covariance(3, 3) = 0.04f;
    firstTrack.Covariance(4, 4) = 0.05f;
    firstTrack.Covariance(5, 5) = 0.06f;
    KFParticleGpuFitState first;
    first.Initialize(firstTrack, 1, 0.5f);

    KFParticleGpuTrackState secondTrack;
    secondTrack.Px() = -1.f;
    secondTrack.Py() = 2.f;
    secondTrack.Pz() = 2.f;
    secondTrack.Covariance(3, 3) = 0.03f;
    secondTrack.Covariance(4, 4) = 0.02f;
    secondTrack.Covariance(5, 5) = 0.01f;
    KFParticleGpuFitState second;
    second.Initialize(secondTrack, -1, 0.25f);

    KFParticleGpuFitState mother;
    KFParticleGpuMath::BuildKinematicMother(first, second, mother);

    assert(mother.Q() == 0);
    assert(mother.Px() == 2.f);
    assert(mother.Py() == 2.f);
    assert(mother.Pz() == 6.f);
    assert(AlmostEqual(mother.E(), first.E() + second.E()));
    assert(AlmostEqual(KFParticleGpuMath::Momentum2(mother), 44.f));
    assert(AlmostEqual(KFParticleGpuMath::TransverseMomentum2(mother), 8.f));
    assert(KFParticleGpuMath::Mass2(mother) > 0.f);

    float mass = 0.f;
    float massError = 0.f;
    const bool validMass = KFParticleGpuMath::GetMass(mother, mass, massError);
    assert(validMass);
    assert(AlmostEqual(mass, KFParticleGpuMath::Mass(mother)));
    assert(massError > 0.f);
    assert(massError < 1.e8f);
    Pass("math-mass-error", "device-safe invariant mass and covariance-based mass error helpers");
    Pass("math-kinematic-seed", "device-safe two-daughter kinematic seed helpers");
  }

  void TestMeasurementSeed()
  {
    KFParticleGpuFitState first;
    first.X() = 0.f;
    first.Y() = 0.f;
    first.Z() = 0.f;
    first.Px() = 1.f;
    first.Py() = 0.f;
    first.Pz() = 0.f;
    first.E() = 1.5f;
    first.Q() = 1;
    first.Covariance(0, 0) = 2.f;
    first.Covariance(3, 3) = 0.2f;

    KFParticleGpuFitState second;
    second.X() = 0.f;
    second.Y() = 1.f;
    second.Z() = 0.f;
    second.Px() = 0.f;
    second.Py() = -1.f;
    second.Pz() = 0.f;
    second.E() = 1.25f;
    second.Q() = -1;
    second.Covariance(1, 1) = 3.f;
    second.Covariance(4, 4) = 0.3f;

    KFParticleGpuFitState currentAtDca;
    KFParticleGpuMeasurement measurement;
    const bool built =
      KFParticleGpuMath::BuildLineDcaMeasurementSeed(first, second, currentAtDca, measurement);
    assert(built);
    assert(currentAtDca.X() == 0.f);
    assert(currentAtDca.Y() == 0.f);
    assert(currentAtDca.Px() == first.Px());
    assert(currentAtDca.Covariance(3, 3) == first.Covariance(3, 3));
    assert(measurement.Parameter(0) == 0.f);
    assert(measurement.Parameter(1) == 0.f);
    assert(measurement.Parameter(4) == second.Py());
    assert(measurement.Covariance(4, 4) == second.Covariance(4, 4));
    assert(KFParticleGpuMath::Abs(measurement.Correlation(0, 0)) > 1.e-6f);
    assert(measurement.Covariance(0, 0) >= 0.f);

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 2001.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedCurrent;
    KFParticleGpuMeasurement rejectedMeasurement;
    assert(!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, farSecond, rejectedCurrent, rejectedMeasurement));
    Pass("measurement-seed", "straight-line DCA measurement arrays, covariance transport, and correlation block");
  }

  void TestFieldMeasurementSeedApprox()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;
    first.NDF() = 0;
    first.NDF() = 0;
    first.Covariance(0, 0) = 0.4f;
    first.Covariance(1, 1) = 0.5f;
    first.Covariance(2, 2) = 0.6f;
    first.Covariance(3, 3) = 0.1f;
    first.Covariance(4, 4) = 0.2f;
    first.Covariance(5, 5) = 0.3f;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;
    second.NDF() = 0;
    second.NDF() = 0;
    second.Covariance(0, 0) = 0.7f;
    second.Covariance(1, 1) = 0.8f;
    second.Covariance(2, 2) = 0.9f;
    second.Covariance(3, 3) = 0.15f;
    second.Covariance(4, 4) = 0.25f;
    second.Covariance(5, 5) = 0.35f;

    KFParticleGpuFitState lineCurrent;
    KFParticleGpuMeasurement lineMeasurement;
    assert(KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, second, lineCurrent, lineMeasurement));

    KFParticleGpuFitState zeroCurrent;
    KFParticleGpuMeasurement zeroMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, second, 0.f, zeroCurrent, zeroMeasurement));
    ExpectFitStateClose(zeroCurrent, lineCurrent, 1.e-5f);
    for (int i = 0; i < KFParticleGpuFitState::NumberOfParameters; ++i) {
      assert(AlmostEqual(zeroMeasurement.Parameter(i), lineMeasurement.Parameter(i), 1.e-5f));
    }

    KFParticleGpuFitState fieldCurrent;
    KFParticleGpuMeasurement fieldMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, second, kDiagnosticFieldBy, fieldCurrent, fieldMeasurement));
    assert(!AlmostEqual(fieldCurrent.X(), lineCurrent.X(), 1.e-6f));
    assert(!AlmostEqual(fieldCurrent.Px(), lineCurrent.Px(), 1.e-6f));
    assert(!AlmostEqual(fieldMeasurement.Parameter(0), lineMeasurement.Parameter(0), 1.e-6f));
    for (int i = 0; i < KFParticleGpuFitState::NumberOfCovarianceElements; ++i) {
      assert(AlmostEqual(fieldCurrent.Covariance(i), lineCurrent.Covariance(i), 1.e-5f));
      assert(AlmostEqual(fieldMeasurement.Covariance(i), lineMeasurement.Covariance(i), 1.e-5f));
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        assert(AlmostEqual(fieldMeasurement.Correlation(row, column),
                           lineMeasurement.Correlation(row, column),
                           1.e-5f));
      }
    }

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedCurrent;
    KFParticleGpuMeasurement rejectedMeasurement;
    assert(!KFParticleGpuMath::BuildConstantByDcaMeasurementSeedApprox(
      first, farSecond, kDiagnosticFieldBy, rejectedCurrent, rejectedMeasurement));

    Pass("field-measurement-seed-approx",
         "field-aware measurement seed updates transported states while preserving line covariance approximation");
  }

  void TestFieldEnergyFit()
  {
    KFParticleGpuFitState first;
    first.Initialize();
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;
    first.NDF() = 0;
    first.Covariance(0, 0) = 0.4f;
    first.Covariance(1, 1) = 0.5f;
    first.Covariance(2, 2) = 0.6f;
    first.Covariance(3, 3) = 0.1f;
    first.Covariance(4, 4) = 0.2f;
    first.Covariance(5, 5) = 0.3f;
    first.Covariance(6, 6) = 0.4f;

    KFParticleGpuFitState second;
    second.Initialize();
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;
    second.NDF() = 0;
    second.Covariance(0, 0) = 0.7f;
    second.Covariance(1, 1) = 0.8f;
    second.Covariance(2, 2) = 0.9f;
    second.Covariance(3, 3) = 0.15f;
    second.Covariance(4, 4) = 0.25f;
    second.Covariance(5, 5) = 0.35f;
    second.Covariance(6, 6) = 0.45f;

    KFParticleGpuFitState lineCurrent;
    KFParticleGpuMeasurement lineMeasurement;
    assert(KFParticleGpuMath::BuildLineDcaMeasurementSeed(
      first, second, lineCurrent, lineMeasurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      lineCurrent, lineMeasurement, second.Q()));

    KFParticleGpuFitState fieldCurrent;
    KFParticleGpuMeasurement fieldMeasurement;
    assert(KFParticleGpuMath::BuildConstantByDcaMeasurementSeed(
      first, second, kDiagnosticFieldBy, fieldCurrent, fieldMeasurement));
    assert(KFParticleGpuMath::AddDaughterWithEnergyFit(
      fieldCurrent, fieldMeasurement, second.Q()));
    assert(fieldCurrent.NDF() == 2);
    assert(fieldCurrent.Q() == 0);
    assert(fieldCurrent.Chi2() >= 0.f);
    assert(!AlmostEqual(fieldCurrent.X(), lineCurrent.X(), 1.e-6f));
    assert(!AlmostEqual(fieldCurrent.Px(), lineCurrent.Px(), 1.e-6f));
    assert(fieldCurrent.Covariance(0, 0) >= 0.f);

    Pass("field-energy-fit",
         "field-aware measurement seed with numerical Jacobian can feed the Kalman energy-fit update");
  }

  void TestFieldTransportPrimitive()
  {
    KFParticleGpuFitState particle;
    particle.X() = 1.f;
    particle.Y() = 2.f;
    particle.Z() = 3.f;
    particle.Px() = 4.f;
    particle.Py() = 5.f;
    particle.Pz() = 6.f;
    particle.Q() = 1;

    KFParticleGpuFitState line;
    KFParticleGpuMath::TransportLine(particle, 0.25f, line);

    KFParticleGpuFitState zeroField;
    KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 0.f, zeroField);
    assert(AlmostEqual(zeroField.X(), line.X()));
    assert(AlmostEqual(zeroField.Y(), line.Y()));
    assert(AlmostEqual(zeroField.Z(), line.Z()));
    assert(AlmostEqual(zeroField.Px(), line.Px()));
    assert(AlmostEqual(zeroField.Pz(), line.Pz()));

    KFParticleGpuFitState chargedField;
    KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 20.f, chargedField);
    assert(AlmostEqual(chargedField.Y(), line.Y()));
    assert(!AlmostEqual(chargedField.X(), line.X()));
    assert(!AlmostEqual(chargedField.Px(), line.Px()));
    assert(AlmostEqual(KFParticleGpuMath::Momentum2(chargedField),
                       KFParticleGpuMath::Momentum2(particle),
                       1.e-5f));

    KFParticleGpuFitState neutral = particle;
    neutral.Q() = 0;
    KFParticleGpuFitState neutralField;
    KFParticleGpuMath::TransportConstantBy(neutral, 0.25f, 20.f, neutralField);
    assert(AlmostEqual(neutralField.X(), line.X()));
    assert(AlmostEqual(neutralField.Z(), line.Z()));

    Pass("field-transport-primitive", "constant-By single-track transport with exact zero-field and neutral fallbacks");
  }

  void TestFieldTransportProbe(KFParticleGpuRuntime& runtime)
  {
    xpu::buffer<float> floatChecks(20, xpu::buf_io);
    xpu::buffer<int> integerChecks(8, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuFieldTransportProbe>(
      xpu::n_threads(1), floatChecks.get(), integerChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    assert(integers[0] == 1);
    assert(AlmostEqual(floats[0], 2.f));
    assert(AlmostEqual(floats[1], 4.5f));
    assert(AlmostEqual(floats[2], floats[0]));
    assert(AlmostEqual(floats[3], floats[1]));
    assert(!AlmostEqual(floats[4], floats[0]));
    assert(!AlmostEqual(floats[6], 4.f));
    assert(AlmostEqual(floats[8], floats[0]));
    assert(AlmostEqual(floats[9], floats[1]));
    assert(integers[1] == 1);
    assert(integers[2] == 1);
    assert(integers[3] == 1);
    assert(integers[4] == 1);
    assert(integers[5] == 1);
    assert(integers[6] == 2);
    assert(integers[7] == 0);
    assert(AlmostEqual(floats[10], floats[11], 1.e-5f));
    assert(!AlmostEqual(floats[12], floats[10], 1.e-6f));
    assert(!AlmostEqual(floats[14], floats[13], 1.e-6f));
    assert(!AlmostEqual(floats[16], floats[15], 1.e-6f));
    assert(floats[18] >= 0.f);
    assert(!AlmostEqual(floats[19], floats[14], 1.e-6f));
    Pass("field-transport-probe",
         "device-side constant-By transport, DCA seed, and field-aware energy-fit checks");
  }

  void TestFieldDcaKinematicSeed()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;

    KFParticleGpuFitState lineMother;
    KFParticleGpuMath::BuildLineDcaKinematicMother(first, second, lineMother);

    KFParticleGpuFitState zeroFieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 0.f, zeroFieldMother));
    ExpectFitStateClose(zeroFieldMother, lineMother, 1.e-5f);

    KFParticleGpuFitState neutralFirst = first;
    KFParticleGpuFitState neutralSecond = second;
    neutralFirst.Q() = 0;
    neutralSecond.Q() = 0;
    KFParticleGpuFitState neutralLineMother;
    KFParticleGpuMath::BuildLineDcaKinematicMother(neutralFirst, neutralSecond, neutralLineMother);
    KFParticleGpuFitState neutralFieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      neutralFirst, neutralSecond, 200.f, neutralFieldMother));
    ExpectFitStateClose(neutralFieldMother, neutralLineMother, 1.e-5f);

    KFParticleGpuFitState fieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 200.f, fieldMother));
    assert(!AlmostEqual(fieldMother.X(), lineMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), lineMother.Px(), 1.e-6f));
    assert(fieldMother.Q() == lineMother.Q());
    assert(AlmostEqual(fieldMother.E(), lineMother.E()));

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    KFParticleGpuFitState rejectedMother;
    assert(!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, farSecond, 200.f, rejectedMother));

    Pass("field-dca-kinematic-seed", "two-daughter constant-By DCA seed with zero-field equivalence and bounded rejection");
  }

  void TestFieldDcaReferenceComparison()
  {
    KFParticleGpuFitState first;
    first.X() = 1.f;
    first.Y() = 2.f;
    first.Z() = 3.f;
    first.Px() = 4.f;
    first.Py() = 5.f;
    first.Pz() = 6.f;
    first.E() = 10.f;
    first.Q() = 1;

    KFParticleGpuFitState second;
    second.X() = -0.5f;
    second.Y() = 1.5f;
    second.Z() = 2.5f;
    second.Px() = -3.f;
    second.Py() = 2.f;
    second.Pz() = 5.f;
    second.E() = 8.f;
    second.Q() = -1;

    ReferenceState zeroReference;
    assert(ReferenceBuildConstantByDcaKinematicMother(first, second, 0.0, zeroReference));
    KFParticleGpuFitState zeroMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, 0.f, zeroMother));
    ExpectFitStateMatchesReference(zeroMother, zeroReference, 1.e-5f);

    ReferenceState fieldReference;
    assert(ReferenceBuildConstantByDcaKinematicMother(
      first, second, kDiagnosticFieldBy, fieldReference));
    KFParticleGpuFitState fieldMother;
    assert(KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, second, kDiagnosticFieldBy, fieldMother));
    ExpectFitStateMatchesReference(fieldMother, fieldReference, 5.e-5f);

    KFParticleGpuFitState farSecond = second;
    farSecond.Z() = 3000.f;
    farSecond.Pz() = -1.f;
    ReferenceState rejectedReference;
    assert(!ReferenceBuildConstantByDcaKinematicMother(
      first, farSecond, kDiagnosticFieldBy, rejectedReference));
    KFParticleGpuFitState rejectedMother;
    assert(!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
      first, farSecond, kDiagnosticFieldBy, rejectedMother));

    Pass("field-dca-reference-comparison",
         "constant-By two-daughter DCA seed matches an independent scalar reference");
  }

  void TestKalmanUpdateProbe(KFParticleGpuRuntime& runtime)
  {
    xpu::buffer<float> floatChecks(8, xpu::buf_io);
    xpu::buffer<int> integerChecks(3, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuKalmanUpdateProbe>(
      xpu::n_threads(1), floatChecks.get(), integerChecks.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(floatChecks, xpu::d2h);
    runtime.GetQueue().copy(integerChecks, xpu::d2h);
    runtime.GetQueue().wait();

    const float* floats = HostPointer(floatChecks);
    const int* integers = HostPointer(integerChecks);
    assert(integers[0] == 1);
    assert(integers[1] == 0);
    assert(integers[2] == 2);
    assert(AlmostEqual(floats[0], 0.8f));
    assert(AlmostEqual(floats[1], 5.f / 3.f));
    assert(AlmostEqual(floats[2], 18.f / 7.f));
    assert(AlmostEqual(floats[3], 1.5f));
    assert(AlmostEqual(floats[4], 5.f));
    assert(AlmostEqual(floats[5], 0.8f));
    assert(AlmostEqual(floats[6], 0.11f));
    assert(AlmostEqual(floats[7], 2.1523809f, 1.e-5f));
    Pass("kalman-energy-fit-probe", "device-side one-daughter energy-fit Kalman update on diagonal covariance");
  }

  void TestTwoDaughterTaskKernel(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(2, 1, 1);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(1, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* taskHost = HostPointer(taskBuffer);
    taskHost[0] = KFParticleGpuTwoDaughterTask();
    taskHost[0].firstTrack = 0;
    taskHost[0].secondTrack = 1;
    taskHost[0].eventIndex = 0;
    taskHost[0].motherPdg = 310;
    taskHost[0].firstDaughterPdg = 211;
    taskHost[0].secondDaughterPdg = -211;
    taskHost[0].primaryVertexIndex = -1;
    taskHost[0].firstMass = 0.13957f;
    taskHost[0].secondMass = 0.13957f;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1);
    assert(candidates.Daughters().Size() == 2);
    assert(candidates.OverflowFlags() == 0);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().PrimaryVertexIndex(0) == -1);
    assert(candidates.Metadata().EventIndex(0) == 0);
    assert(candidates.Metadata().DaughterOffset(0) == 0);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Metadata().Flags(0) == KFGpuCandidateValid);
    assert(candidates.Daughters().SourceId(0) == 17);
    assert(candidates.Daughters().SourceId(1) == 19);

    KFParticleGpuFitState mother;
    LoadCandidateFit(candidates, 0, mother);
    assert(mother.Q() == 0);
    assert(mother.Px() == 5.f);
    assert(mother.Py() == 7.f);
    assert(mother.Pz() == 9.f);
    assert(mother.SumDaughterMass() == 2.f * taskHost[0].firstMass);
    assert(KFParticleGpuMath::Mass2(mother) > 0.f);

    Pass("two-daughter-task-kernel", "prepared pair task builds and stores a two-daughter candidate on device");

    buffers.ResetCandidates();
    taskHost[0].flags = KFGpuTwoDaughterUseLineDca;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceMother;
    const bool referenceBuilt =
      BuildTwoDaughterKinematicCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceMother);
    assert(referenceBuilt);

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView lineDcaCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState lineDcaMother;
    LoadCandidateFit(lineDcaCandidates, 0, lineDcaMother);
    assert(lineDcaCandidates.Size() == 1);
    assert(lineDcaCandidates.Daughters().Size() == 2);
    assert(AlmostEqual(lineDcaMother.X(), referenceMother.X()));
    assert(AlmostEqual(lineDcaMother.Y(), referenceMother.Y()));
    assert(AlmostEqual(lineDcaMother.Z(), referenceMother.Z()));
    assert(AlmostEqual(lineDcaMother.S(), referenceMother.S()));
    assert(AlmostEqual(lineDcaMother.Px(), referenceMother.Px()));
    assert(AlmostEqual(lineDcaMother.E(), referenceMother.E()));

    Pass("two-daughter-line-dca", "prepared pair task transports daughters to straight-line DCA before storing candidate");

    buffers.ResetCandidates();
    taskHost[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceFitMother;
    const bool referenceFitBuilt =
      BuildTwoDaughterEnergyFitCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceFitMother);
    assert(referenceFitBuilt);

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView energyFitCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState energyFitMother;
    LoadCandidateFit(energyFitCandidates, 0, energyFitMother);
    assert(energyFitCandidates.Size() == 1);
    assert(energyFitCandidates.Daughters().Size() == 2);
    assert(energyFitMother.NDF() == 2);
    assert(energyFitMother.Q() == 0);
    assert(energyFitMother.Chi2() >= 0.f);
    assert(AlmostEqual(energyFitMother.X(), referenceFitMother.X()));
    assert(AlmostEqual(energyFitMother.Y(), referenceFitMother.Y()));
    assert(AlmostEqual(energyFitMother.Z(), referenceFitMother.Z()));
    assert(AlmostEqual(energyFitMother.Px(), referenceFitMother.Px()));
    assert(AlmostEqual(energyFitMother.Py(), referenceFitMother.Py()));
    assert(AlmostEqual(energyFitMother.Pz(), referenceFitMother.Pz()));
    assert(AlmostEqual(energyFitMother.E(), referenceFitMother.E()));
    assert(AlmostEqual(energyFitMother.Chi2(), referenceFitMother.Chi2()));
    assert(energyFitCandidates.Metadata().Pdg(0) == 310);
    assert(energyFitCandidates.Metadata().Flags(0)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    assert(energyFitCandidates.Daughters().SourceId(0) == 17);
    assert(energyFitCandidates.Daughters().SourceId(1) == 19);

    Pass("two-daughter-energy-fit", "prepared pair task builds a two-daughter candidate with Kalman energy-fit update");

    buffers.ResetCandidates();
    KFParticleGpuInputTrackSoAView diagnosticTracks = buffers.HostInputTracks();
    StoreDiagnosticField(diagnosticTracks, 0);
    buffers.UploadInput();

    taskHost[0].flags = KFGpuTwoDaughterUseLineDca;
    taskHost[0].transportMode = KFGpuTransportFieldAware;
    taskHost[0].motherPdg = 997;
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    KFParticleGpuFitState referenceFieldMother;
    const bool referenceFieldBuilt =
      BuildTwoDaughterKinematicCandidate(MakeConstView(buffers.HostInputTracks()), taskHost[0], referenceFieldMother);
    assert(referenceFieldBuilt);
    KFParticleGpuTwoDaughterTask lineReferenceTask = taskHost[0];
    lineReferenceTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightReferenceMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), lineReferenceTask, straightReferenceMother));

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(1),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      1u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView fieldCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(fieldCandidates, 0, fieldMother);
    assert(fieldCandidates.Size() == 1);
    assert(fieldCandidates.Daughters().Size() == 2);
    assert(fieldCandidates.Metadata().Pdg(0) == 997);
    assert(fieldCandidates.Metadata().Flags(0)
           == static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca));
    ExpectFieldAwareSeedClose(fieldMother, referenceFieldMother);
    assert(!AlmostEqual(fieldMother.X(), straightReferenceMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightReferenceMother.Px(), 1.e-6f));

    Pass("two-daughter-field-aware-kinematic",
         "explicit field-aware prepared task uses constant-By DCA seed without enabling default V0 channels");

    buffers.ResetCandidates();
    xpu::buffer<KFParticleGpuTwoDaughterTask> guardTaskBuffer(2, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* guardTasks = HostPointer(guardTaskBuffer);
    KFParticleGpuTwoDaughterTask guardBaseTask = taskHost[0];
    guardBaseTask.motherPdg = 310;
    guardBaseTask.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    guardBaseTask.transportMode = KFGpuTransportStraightLine;
    guardTasks[0] = guardBaseTask;
    guardTasks[0].motherPdg = 999;
    guardTasks[0].firstDaughterPdg = -211;
    assert(!ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), guardTasks[0]));
    guardTasks[1] = guardBaseTask;
    guardTasks[1].motherPdg = 999;
    guardTasks[1].transportMode = KFGpuTransportFieldAware;
    runtime.GetQueue().copy(guardTaskBuffer, xpu::h2d);
    runtime.GetQueue().wait();

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(2),
      MakeConstView(buffers.DeviceInputTracks()),
      guardTaskBuffer.get(),
      2u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView guardedCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(guardedCandidates.Size() == 2);
    assert(guardedCandidates.Metadata().Pdg(0) == 999);
    assert(guardedCandidates.Metadata().DaughterCount(0) == 0);
    assert(guardedCandidates.Metadata().Flags(0) & KFGpuCandidateBuildFailed);
    assert(guardedCandidates.Metadata().Flags(0) & KFGpuCandidateEnergyFit);
    assert(guardedCandidates.Metadata().Pdg(1) == 999);
    assert(guardedCandidates.Metadata().DaughterCount(1) == 2);
    assert(guardedCandidates.Metadata().Flags(1)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    KFParticleGpuFitState guardedFieldFitMother;
    LoadCandidateFit(guardedCandidates, 1, guardedFieldFitMother);
    assert(guardedFieldFitMother.NDF() == 2);
    assert(guardedFieldFitMother.Q() == 0);

    Pass("two-daughter-guards", "invalid tasks fail while field-aware energy-fit tasks build candidates");
  }

  void StoreSyntheticTrack(KFParticleGpuInputTrackSoAView& tracks,
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

    const float scale = 1.f + 0.2f * static_cast<float>(index);
    track.Covariance(0, 0) = 0.40f * scale;
    track.Covariance(1, 1) = 0.35f * scale;
    track.Covariance(2, 2) = 0.30f * scale;
    track.Covariance(3, 3) = 0.08f * scale;
    track.Covariance(4, 4) = 0.07f * scale;
    track.Covariance(5, 5) = 0.06f * scale;
    track.Covariance(0, 3) = 0.010f * scale;
    track.Covariance(1, 4) = -0.008f * scale;
    track.Covariance(2, 5) = 0.006f * scale;
    track.Covariance(3, 4) = 0.004f * scale;
    track.Covariance(4, 5) = -0.003f * scale;

    StoreTrackState(track, tracks, index);
    tracks.SourceId(index) = sourceId;
    tracks.Pdg(index) = pdg;
    tracks.Charge(index) = charge;
    tracks.PrimaryVertexIndex(index) = -1;
    tracks.NumberOfPixelHits(index) = 5;
    tracks.ChiToPrimaryVertex(index) = 1.f + static_cast<float>(index);
  }

  void TestTwoDaughterTaskGeneration(KFParticleGpuRuntime& runtime,
                                     KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 501);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 502);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 601);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 602);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 42;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    xpu::buffer<unsigned int> writtenTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasks>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      3u,
      writtenTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(writtenTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    const KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    const unsigned int* written = HostPointer(writtenTasks);
    const unsigned int* total = HostPointer(totalPairs);
    assert(written[0] == 3);
    assert(total[0] == 4);
    assert(tasks[0].firstTrack == 0 && tasks[0].secondTrack == 2);
    assert(tasks[1].firstTrack == 0 && tasks[1].secondTrack == 3);
    assert(tasks[2].firstTrack == 1 && tasks[2].secondTrack == 2);
    for (unsigned int i = 0; i < written[0]; ++i) {
      assert(tasks[i].eventIndex == 0);
      assert(tasks[i].motherPdg == 310);
      assert(tasks[i].firstDaughterPdg == 211);
      assert(tasks[i].secondDaughterPdg == -211);
      assert(tasks[i].flags == source.flags);
      assert(tasks[i].transportMode == source.transportMode);
      assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[i]));
    }

    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minFirstPixelHits = 5;
    source.minSecondPixelHits = 5;
    source.maxSecondChiToPrimaryVertex = 4.f;
    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasks>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      3u,
      writtenTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(writtenTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    tasks = HostPointer(taskBuffer);
    written = HostPointer(writtenTasks);
    total = HostPointer(totalPairs);
    assert(written[0] == 3);
    assert(total[0] == 4);
    assert(tasks[0].firstTrack == 0 && tasks[0].secondTrack == 2);
    assert(tasks[1].firstTrack == buffers.HostInputTracks().Size());
    assert(tasks[2].firstTrack == buffers.HostInputTracks().Size());
    assert(PassTwoDaughterTaskSourceCuts(MakeConstView(buffers.HostInputTracks()),
                                         source,
                                         tasks[0].firstTrack,
                                         tasks[0].secondTrack));
    assert(!ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[1]));

    Pass("two-daughter-task-generation", "event species ranges and pre-fit cuts generate fixed prepared task slots");
  }

  bool HasGeneratedTask(const KFParticleGpuTwoDaughterTask* tasks,
                        unsigned int size,
                        unsigned int firstTrack,
                        unsigned int secondTrack)
  {
    for (unsigned int i = 0; i < size; ++i) {
      if (tasks[i].firstTrack == firstTrack && tasks[i].secondTrack == secondTrack) {
        return true;
      }
    }
    return false;
  }

  bool HasCandidateDaughters(const KFParticleGpuConstCandidatePoolView& candidates,
                             int firstSourceId,
                             int secondSourceId)
  {
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().DaughterCount(i) != 2u) {
        continue;
      }
      const unsigned int offset = candidates.Metadata().DaughterOffset(i);
      if (candidates.Daughters().SourceId(offset) == firstSourceId
          && candidates.Daughters().SourceId(offset + 1u) == secondSourceId) {
        return true;
      }
    }
    return false;
  }

  void ExpectChannelCandidatePdgs(const KFParticleGpuConstCandidatePoolView& candidates,
                                  const KFParticleGpuTwoDaughterChannelResult& result,
                                  int expectedPdg)
  {
    for (unsigned int i = result.candidates.offset; i < result.candidates.End(); ++i) {
      assert(result.ContainsCandidate(i));
      assert(result.candidates.LocalCandidateIndex(i) < result.candidates.size);
      assert(candidates.Metadata().Pdg(i) == expectedPdg);
    }
  }

  void ExpectChannelDaughterRange(const KFParticleGpuConstCandidatePoolView& candidates,
                                  const KFParticleGpuTwoDaughterChannelResult& result)
  {
    for (unsigned int i = result.candidates.offset; i < result.candidates.End(); ++i) {
      const unsigned int daughterOffset = candidates.Metadata().DaughterOffset(i);
      const unsigned int daughterCount = candidates.Metadata().DaughterCount(i);
      for (unsigned int j = 0; j < daughterCount; ++j) {
        assert(result.candidates.ContainsDaughter(daughterOffset + j));
      }
    }
  }

  void ExpectChannelSelectionFlags(const KFParticleGpuConstCandidatePoolView& candidates,
                                   const KFParticleGpuTwoDaughterChannelResult& result,
                                   const KFParticleGpuTwoDaughterChannel& channel)
  {
    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, result.eventIndex);
    for (unsigned int i = result.candidates.offset; i < result.candidates.End(); ++i) {
      KFParticleGpuFitState candidate;
      LoadCandidateFit(candidates, i, candidate);
      const bool expectedRejected = !PassTwoDaughterPostBuildSelection(source, candidate);
      const bool markedRejected =
        (candidates.Metadata().Flags(i) & KFGpuCandidateSelectionRejected) != 0u;
      assert(markedRejected == expectedRejected);
    }
  }

  unsigned int CountChannelSelectionRejected(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuTwoDaughterChannelResult& result)
  {
    unsigned int rejected = 0u;
    for (unsigned int i = result.candidates.offset; i < result.candidates.End(); ++i) {
      if (candidates.Metadata().Flags(i) & KFGpuCandidateSelectionRejected) {
        ++rejected;
      }
    }
    return rejected;
  }

  void ExpectDefaultV0ChannelRegression(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuTwoDaughterChannelResult& result,
    const KFParticleGpuTwoDaughterChannel& channel,
    unsigned int expectedChannelId,
    int expectedMotherPdg,
    unsigned int expectedOffset,
    unsigned int expectedDaughterOffset)
  {
    assert(result.channelId == expectedChannelId);
    assert(result.motherPdg == expectedMotherPdg);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == 1u);
    assert(result.acceptedTasks == 1u);
    assert(result.storedTasks == 1u);
    assert(!result.Truncated());
    assert(!result.HasAnyOverflow());
    assert(result.candidates.offset == expectedOffset);
    assert(result.candidates.size == 1u);
    assert(result.candidates.daughterOffset == expectedDaughterOffset);
    assert(result.candidates.daughterSize == 2u);
    assert(candidates.Metadata().EventIndex(expectedOffset) == 0u);
    assert(candidates.Metadata().Pdg(expectedOffset) == expectedMotherPdg);
    assert(candidates.Metadata().DaughterOffset(expectedOffset) == expectedDaughterOffset);
    assert(candidates.Metadata().DaughterCount(expectedOffset) == 2u);
    assert(candidates.Metadata().Flags(expectedOffset) & KFGpuCandidateValid);
    assert(candidates.Metadata().Flags(expectedOffset) & KFGpuCandidateLineDca);
    assert(candidates.Metadata().Flags(expectedOffset) & KFGpuCandidateEnergyFit);
    assert(!(candidates.Metadata().Flags(expectedOffset) & KFGpuCandidateBuildFailed));
    ExpectChannelCandidatePdgs(candidates, result, expectedMotherPdg);
    ExpectChannelDaughterRange(candidates, result);
    ExpectChannelSelectionFlags(candidates, result, channel);
    assert(CountChannelSelectionRejected(candidates, result) <= result.candidates.size);
  }

  void TestCompactTwoDaughterTaskGeneration(KFParticleGpuRuntime& runtime,
                                            KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1102);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1201);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1202);
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 45;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();

    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;
    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minSecondPixelHits = 5;
    source.minFirstPixelHits = 0;
    source.maxSecondChiToPrimaryVertex = 4.f;

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(2, xpu::buf_io);
    xpu::buffer<unsigned int> acceptedTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

    unsigned int* accepted = HostPointer(acceptedTasks);
    unsigned int* total = HostPointer(totalPairs);
    accepted[0] = 0;
    total[0] = 0;
    runtime.GetQueue().copy(acceptedTasks, xpu::h2d);
    runtime.GetQueue().copy(totalPairs, xpu::h2d);

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      2u,
      acceptedTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(taskBuffer, xpu::d2h);
    runtime.GetQueue().copy(acceptedTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    const KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    assert(accepted[0] == 2);
    assert(total[0] == 4);
    assert(HasGeneratedTask(tasks, 2, 0, 2));
    assert(HasGeneratedTask(tasks, 2, 1, 2));
    for (unsigned int i = 0; i < 2; ++i) {
      assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[i]));
      assert(PassTwoDaughterTaskSourceCuts(MakeConstView(buffers.HostInputTracks()),
                                           source,
                                           tasks[i].firstTrack,
                                           tasks[i].secondTrack));
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> truncatedTaskBuffer(1, xpu::buf_io);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    accepted[0] = 0;
    total[0] = 0;
    runtime.GetQueue().copy(acceptedTasks, xpu::h2d);
    runtime.GetQueue().copy(totalPairs, xpu::h2d);

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasksCompact>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      truncatedTaskBuffer.get(),
      1u,
      acceptedTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();
    runtime.GetQueue().copy(truncatedTaskBuffer, xpu::d2h);
    runtime.GetQueue().copy(acceptedTasks, xpu::d2h);
    runtime.GetQueue().copy(totalPairs, xpu::d2h);
    runtime.GetQueue().wait();

    tasks = HostPointer(truncatedTaskBuffer);
    accepted = HostPointer(acceptedTasks);
    total = HostPointer(totalPairs);
    assert(accepted[0] == 2);
    assert(total[0] == 4);
    assert(ValidateTwoDaughterTask(MakeConstView(buffers.HostInputTracks()), tasks[0]));
    assert((tasks[0].firstTrack == 0 || tasks[0].firstTrack == 1) && tasks[0].secondTrack == 2);

    Pass("two-daughter-task-generation-compact",
         "atomic compact pair generation stores only accepted task slots and reports truncation");
  }

  void TestCompactTwoDaughterCandidateConstruction(KFParticleGpuRuntime& runtime,
                                                   KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1301);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1302);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1401);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1402);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    tasks[0] = KFParticleGpuTwoDaughterTask();
    tasks[0].firstTrack = 0;
    tasks[0].secondTrack = 2;
    tasks[0].eventIndex = 0;
    tasks[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    tasks[0].motherPdg = 310;
    tasks[0].firstDaughterPdg = 211;
    tasks[0].secondDaughterPdg = -211;
    tasks[0].primaryVertexIndex = -1;
    tasks[0].firstMass = 0.13957f;
    tasks[0].secondMass = 0.13957f;

    tasks[1] = tasks[0];
    tasks[1].firstTrack = 1;
    tasks[1].secondTrack = 3;

    tasks[2] = tasks[0];
    tasks[2].secondTrack = 0; // Duplicate track makes this task fail validation.

    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterCompactCandidateKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 1301, 1401));
    assert(HasCandidateDaughters(candidates, 1302, 1402));
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().Pdg(i) == 310);
      assert(candidates.Metadata().DaughterCount(i) == 2u);
      assert(candidates.Metadata().Flags(i)
             == static_cast<unsigned int>(
               KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    }

    Pass("two-daughter-candidate-construction-compact",
         "compact construction stores successful candidates without failed task placeholders");
  }

  void TestCompactTwoDaughterCandidateOverflow(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(6, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1501);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1502);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, 211, 1, 1503);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1601);
    StoreSyntheticTrack(tracks, 4, -0.1f, 0.4f, 0.2f, -0.2f, 0.8f, 0.7f, -211, -1, 1602);
    StoreSyntheticTrack(tracks, 5, 0.3f, -0.5f, 0.1f, -0.6f, 0.4f, 0.9f, -211, -1, 1603);
    buffers.UploadInput();
    buffers.ResetCandidates();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);
    for (unsigned int i = 0; i < 3; ++i) {
      tasks[i] = KFParticleGpuTwoDaughterTask();
      tasks[i].firstTrack = i;
      tasks[i].secondTrack = 3u + i;
      tasks[i].eventIndex = 0;
      tasks[i].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
      tasks[i].motherPdg = 310;
      tasks[i].firstDaughterPdg = 211;
      tasks[i].secondDaughterPdg = -211;
      tasks[i].primaryVertexIndex = -1;
      tasks[i].firstMass = 0.13957f;
      tasks[i].secondMass = 0.13957f;
    }

    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterCompactCandidateKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    unsigned int failedCandidates = 0;
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      if (candidates.Metadata().DaughterCount(i) == 0u) {
        ++failedCandidates;
        assert(candidates.Metadata().Flags(i) & KFGpuCandidateBuildFailed);
      }
    }
    assert(failedCandidates == 1);

    Pass("two-daughter-candidate-overflow-compact",
         "compact construction reports daughter storage overflow without corrupting counters");
  }

  void TestGeneratedTwoDaughterPipeline(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 701);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 702);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 801);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 802);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 43;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    buffers.UploadInput();
    buffers.ResetCandidates();

    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;
    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minFirstPixelHits = 5;
    source.minSecondPixelHits = 5;
    source.maxSecondChiToPrimaryVertex = 4.f;

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(3, xpu::buf_io);
    xpu::buffer<unsigned int> writtenTasks(1, xpu::buf_io);
    xpu::buffer<unsigned int> totalPairs(1, xpu::buf_io);

    runtime.GetQueue().launch<KFParticleGpuGenerateTwoDaughterTasks>(
      xpu::n_threads(4),
      buffers.DeviceEvents(),
      MakeConstView(buffers.DeviceInputTracks()),
      source,
      taskBuffer.get(),
      3u,
      writtenTasks.get(),
      totalPairs.get());
    runtime.GetQueue().wait();

    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(3),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      3u,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    KFParticleGpuTwoDaughterTask referenceTask;
    KFParticleGpuRange firstRange(0, 2);
    KFParticleGpuRange secondRange(2, 2);
    FillTwoDaughterTask(firstRange, secondRange, 0, source, referenceTask);
    KFParticleGpuFitState reference;
    assert(BuildTwoDaughterCandidate(MakeConstView(buffers.HostInputTracks()), referenceTask, reference));

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    KFParticleGpuFitState actual;
    LoadCandidateFit(candidates, 0, actual);
    ExpectFitStateClose(actual, reference, 2.e-5f);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Metadata().Flags(0)
           == static_cast<unsigned int>(
             KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    assert(candidates.Daughters().SourceId(0) == 701);
    assert(candidates.Daughters().SourceId(1) == 801);

    assert(candidates.Metadata().DaughterCount(1) == 0);
    assert(candidates.Metadata().Flags(1) & KFGpuCandidateBuildFailed);
    assert(candidates.Metadata().DaughterCount(2) == 0);
    assert(candidates.Metadata().Flags(2) & KFGpuCandidateBuildFailed);

    Pass("two-daughter-generated-pipeline", "GPU-generated pair tasks feed the two-daughter construction kernel");
  }

  void FillGeneratedPipelineFixture(KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 901);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 902);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 1001);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 1002);
    tracks.NumberOfPixelHits(1) = 3;
    tracks.ChiToPrimaryVertex(3) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 44;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 2);
  }

  KFParticleGpuTwoDaughterTaskSource MakePionPairSource()
  {
    KFParticleGpuTwoDaughterTaskSource source;
    source.eventIndex = 0;
    source.firstTrackSet = SecondaryPositiveFirst;
    source.secondTrackSet = SecondaryNegativeFirst;
    source.firstSpecies = Pion;
    source.secondSpecies = Pion;
    source.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    source.motherPdg = 310;
    source.firstDaughterPdg = 211;
    source.secondDaughterPdg = -211;
    source.primaryVertexIndex = -1;
    source.firstMass = 0.13957f;
    source.secondMass = 0.13957f;
    source.firstCharge = 1;
    source.secondCharge = -1;
    source.minFirstPixelHits = 5;
    source.minSecondPixelHits = 5;
    source.maxSecondChiToPrimaryVertex = 4.f;
    return source;
  }

  KFParticleGpuTwoDaughterChannel MakePionPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Pion;
    channel.secondSpecies = Pion;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 310;
    channel.firstDaughterPdg = 211;
    channel.secondDaughterPdg = -211;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.13957f;
    channel.secondMass = 0.13957f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  KFParticleGpuTwoDaughterChannel MakeFieldAwarePionPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel = MakePionPairChannel(channelId);
    channel.flags = KFGpuTwoDaughterUseLineDca;
    channel.transportMode = KFGpuTransportFieldAware;
    channel.motherPdg = 997;
    return channel;
  }

  KFParticleGpuTwoDaughterChannel MakeKaonPairChannel(unsigned int channelId)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = SecondaryPositiveFirst;
    channel.secondTrackSet = SecondaryNegativeFirst;
    channel.firstSpecies = Kaon;
    channel.secondSpecies = Kaon;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = 333;
    channel.firstDaughterPdg = 321;
    channel.secondDaughterPdg = -321;
    channel.primaryVertexIndex = -1;
    channel.firstMass = 0.493677f;
    channel.secondMass = 0.493677f;
    channel.firstCharge = 1;
    channel.secondCharge = -1;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 5;
    channel.maxSecondChiToPrimaryVertex = 4.f;
    return channel;
  }

  void FillDecayPlanMultiChannelFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 8u ? requested.tracks : 8u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(8, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 2101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 2102);
    StoreSyntheticTrack(tracks, 2, 0.2f, 0.4f, -0.1f, 0.9f, -0.2f, 1.2f, 321, 1, 2201);
    StoreSyntheticTrack(tracks, 3, -0.5f, 0.2f, 0.3f, 0.7f, 0.4f, 1.0f, 321, 1, 2202);
    StoreSyntheticTrack(tracks, 4, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2301);
    StoreSyntheticTrack(tracks, 5, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2302);
    StoreSyntheticTrack(tracks, 6, -0.1f, 0.3f, -0.2f, -0.5f, 0.6f, 1.0f, -321, -1, 2401);
    StoreSyntheticTrack(tracks, 7, 0.5f, -0.4f, 0.1f, -0.2f, 0.3f, 0.9f, -321, -1, 2402);
    tracks.ChiToPrimaryVertex(4) = 1.f;
    tracks.ChiToPrimaryVertex(5) = 9.f;
    tracks.ChiToPrimaryVertex(6) = 1.f;
    tracks.ChiToPrimaryVertex(7) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 61;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 4);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Kaon) = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(4, 4);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(4, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(6, 2);
  }

  void FillDefaultV0Fixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 4u ? requested.tracks : 4u;
    requested.events = requested.events > 1u ? requested.events : 1u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(4, 1, 1);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 3101);
    StoreSyntheticTrack(tracks, 1, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 2212, 1, 3201);
    StoreSyntheticTrack(tracks, 2, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 3301);
    StoreSyntheticTrack(tracks, 3, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -2212, -1, 3401);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 81;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 2);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Proton) = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 2);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Proton) = KFParticleGpuRange(3, 1);
  }

  void FillDecayPlanMultiEventFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 10u ? requested.tracks : 10u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 6u ? requested.candidates : 6u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(10, 1, 2);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.8f, 0.2f, 0.1f, 0.3f, 0.2f, 1.0f, 211, 1, 2501);
    StoreSyntheticTrack(tracks, 1, -0.2f, 0.3f, -0.1f, -0.4f, 0.5f, 0.8f, -211, -1, 2601);
    StoreSyntheticTrack(tracks, 2, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 2701);
    StoreSyntheticTrack(tracks, 3, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 2702);
    StoreSyntheticTrack(tracks, 4, 0.2f, 0.4f, -0.1f, 0.9f, -0.2f, 1.2f, 321, 1, 2801);
    StoreSyntheticTrack(tracks, 5, -0.5f, 0.2f, 0.3f, 0.7f, 0.4f, 1.0f, 321, 1, 2802);
    StoreSyntheticTrack(tracks, 6, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2901);
    StoreSyntheticTrack(tracks, 7, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2902);
    StoreSyntheticTrack(tracks, 8, -0.1f, 0.3f, -0.2f, -0.5f, 0.6f, 1.0f, -321, -1, 3001);
    StoreSyntheticTrack(tracks, 9, 0.5f, -0.4f, 0.1f, -0.2f, 0.3f, 0.9f, -321, -1, 3002);
    tracks.ChiToPrimaryVertex(6) = 1.f;
    tracks.ChiToPrimaryVertex(7) = 9.f;
    tracks.ChiToPrimaryVertex(8) = 1.f;
    tracks.ChiToPrimaryVertex(9) = 9.f;

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 71;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1, 1);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 72;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2, 4);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Kaon) = KFParticleGpuRange(4, 2);
    events[1].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(6, 4);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(6, 2);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Kaon) = KFParticleGpuRange(8, 2);
  }

  void FillBatchSelectionFixture(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.tracks = requested.tracks > 4u ? requested.tracks : 4u;
    requested.events = requested.events > 2u ? requested.events : 2u;
    requested.candidates = requested.candidates > 2u ? requested.candidates : 2u;
    requested.daughterIds = requested.daughterIds > 4u ? requested.daughterIds : 4u;
    requested.selectedCandidates = requested.selectedCandidates > 2u ? requested.selectedCandidates : 2u;
    buffers.EnsureCapacity(requested);
    buffers.SetInputSizes(4u, 0u, 2u);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, 4101);
    StoreSyntheticTrack(tracks, 1u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 4201);
    StoreSyntheticTrack(tracks, 2u, -0.1f, -0.2f, 0.1f, 0.7f, 0.2f, 0.9f, 211, 1, 4301);
    StoreSyntheticTrack(tracks, 3u, 0.2f, 0.4f, -0.3f, -0.2f, 0.6f, 1.0f, -211, -1, 4401);

    KFParticleGpuEventDesc* events = buffers.HostEvents();
    for (unsigned int event = 0u; event < 2u; ++event) {
      const unsigned int offset = 2u * event;
      events[event] = KFParticleGpuEventDesc();
      events[event].eventId = 191u + event;
      events[event].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(offset, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(offset + 1u, 1u);
      events[event].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(offset + 1u, 1u);
    }
  }

  void TestSteeringTwoDaughterStage(KFParticleGpuRuntime& runtime,
                                    KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    const KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();

    KFParticleGpuRange firstRange(0, 2);
    KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask referenceTask;
    FillTwoDaughterTask(firstRange, secondRange, 0, source, referenceTask);
    KFParticleGpuFitState reference;
    assert(BuildTwoDaughterCandidate(MakeConstView(buffers.HostInputTracks()), referenceTask, reference));

    runtime.GetSteering().RunTwoDaughterStage(source, 3u);

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == DaughterCapacityExceeded);

    KFParticleGpuFitState actual;
    LoadCandidateFit(candidates, 0, actual);
    ExpectFitStateClose(actual, reference, 2.e-5f);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().DaughterCount(0) == 2);
    assert(candidates.Daughters().SourceId(0) == 901);
    assert(candidates.Daughters().SourceId(1) == 1001);
    assert(candidates.Metadata().Flags(1) & KFGpuCandidateBuildFailed);
    assert(candidates.Metadata().Flags(2) & KFGpuCandidateBuildFailed);

    bool zeroCapacityRejected = false;
    try {
      runtime.GetSteering().RunTwoDaughterStage(source, 0u);
    }
    catch (const std::invalid_argument&) {
      zeroCapacityRejected = true;
    }
    assert(zeroCapacityRejected);

    Pass("two-daughter-steering-stage", "steering runs upload, task generation, construction, and download");
  }

  void TestSteeringCompactTwoDaughterStage(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();
    source.minFirstPixelHits = 0;

    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);

    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().Pdg(i) == 310);
      assert(candidates.Metadata().DaughterCount(i) == 2);
      assert(candidates.Metadata().Flags(i)
             == static_cast<unsigned int>(
               KFGpuCandidateValid | KFGpuCandidateLineDca | KFGpuCandidateEnergyFit));
    }

    bool zeroCapacityRejected = false;
    try {
      runtime.GetSteering().RunTwoDaughterCompactStage(source, 0u);
    }
    catch (const std::invalid_argument&) {
      zeroCapacityRejected = true;
    }
    assert(zeroCapacityRejected);

    Pass("two-daughter-compact-steering-stage",
         "steering runs compact pair generation and constructs only accepted task slots");
  }

  void TestSteeringCompactTwoDaughterEdgeCases(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuTwoDaughterTaskSource source = MakePionPairSource();
    source.minFirstPixelHits = 0;

    runtime.GetSteering().RunTwoDaughterCompactStage(source, 1u);
    KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1);
    assert(candidates.Daughters().Size() == 2);
    assert(candidates.OverflowFlags() == CandidateCapacityExceeded);
    assert(HasCandidateDaughters(candidates, 901, 1001)
           || HasCandidateDaughters(candidates, 902, 1001));

    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(2, 0);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(2, 0);
    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 0);
    assert(candidates.Daughters().Size() == 0);
    assert(candidates.OverflowFlags() == 0u);

    buffers.SetInputSizes(6, 1, 2);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, -0.8f, 0.2f, 0.1f, 0.3f, 0.2f, 1.0f, 211, 1, 1701);
    StoreSyntheticTrack(tracks, 1, -0.2f, 0.3f, -0.1f, -0.4f, 0.5f, 0.8f, -211, -1, 1801);
    StoreSyntheticTrack(tracks, 2, -0.2f, 0.1f, 0.0f, 0.8f, 0.1f, 1.0f, 211, 1, 1901);
    StoreSyntheticTrack(tracks, 3, 0.4f, -0.3f, 0.2f, 0.5f, 0.6f, 0.9f, 211, 1, 1902);
    StoreSyntheticTrack(tracks, 4, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 2001);
    StoreSyntheticTrack(tracks, 5, -0.6f, -0.2f, 0.3f, -0.4f, 0.2f, 0.8f, -211, -1, 2002);
    tracks.ChiToPrimaryVertex(4) = 1.f;
    tracks.ChiToPrimaryVertex(5) = 9.f;

    events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 51;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0, 1);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1, 1);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1, 1);
    events[1] = KFParticleGpuEventDesc();
    events[1].eventId = 52;
    events[1].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(2, 2);
    events[1].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(4, 2);
    events[1].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(4, 2);

    source.eventIndex = 1;
    runtime.GetSteering().RunTwoDaughterCompactStage(source, 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2);
    assert(candidates.Daughters().Size() == 4);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 1901, 2001));
    assert(HasCandidateDaughters(candidates, 1902, 2001));
    for (unsigned int i = 0; i < candidates.Size(); ++i) {
      assert(candidates.Metadata().EventIndex(i) == 1u);
    }

    Pass("two-daughter-compact-steering-edge-cases",
         "compact steering reports truncation, handles empty ranges, and selects event descriptors");
  }

  void TestDecayPlanOneChannelExecutor(KFParticleGpuRuntime& runtime,
                                       KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& emptyResults =
      steering.RunDecayPlan(0u, 2u);
    assert(emptyResults.empty());
    KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 0u);
    assert(candidates.Daughters().Size() == 0u);
    assert(candidates.OverflowFlags() == 0u);

    const KFParticleGpuTwoDaughterChannel channel = MakePionPairChannel(41u);
    plan.AddTwoDaughterChannel(channel);
    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 1u);
    assert(&results == &steering.LastDecayPlanResults());
    const KFParticleGpuTwoDaughterChannelResult& result = results[0];
    assert(result.channelId == 41u);
    assert(result.motherPdg == 310);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == 4u);
    assert(result.acceptedTasks == 2u);
    assert(result.storedTasks == 2u);
    assert(!result.Truncated());
    assert(result.candidates.offset == 0u);
    assert(result.candidates.size == 2u);
    assert(result.candidates.daughterOffset == 0u);
    assert(result.candidates.daughterSize == 4u);
    assert(result.candidates.overflowFlags == 0u);

    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == result.candidates.size);
    assert(candidates.Daughters().Size() == result.candidates.daughterSize);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    FillGeneratedPipelineFixture(buffers);
    steering.RunTwoDaughterCompactStage(MakeTwoDaughterTaskSource(channel, 0u), 2u);
    candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == result.candidates.size);
    assert(candidates.Daughters().Size() == result.candidates.daughterSize);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    plan.Clear();
    Pass("decay-plan-one-channel-executor",
         "decay plan executor runs empty and one-channel plans through compact steering");
  }

  void TestDecayPlanFieldAwareDiagnosticExecutor(KFParticleGpuRuntime& runtime,
                                                 KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    const KFParticleGpuTwoDaughterChannel channel =
      MakeFieldAwarePionPairChannel(77u);
    plan.AddTwoDaughterChannel(channel);

    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    const KFParticleGpuRange firstRange(0, 2);
    const KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask referenceTask;
    FillTwoDaughterTask(firstRange, secondRange, 0u, source, referenceTask);
    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), referenceTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = referenceTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterKinematicCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 1u);
    const KFParticleGpuTwoDaughterChannelResult& result = results[0];
    assert(result.channelId == 77u);
    assert(result.motherPdg == 997);
    assert(result.eventIndex == 0u);
    assert(result.totalPairs == 4u);
    assert(result.acceptedTasks == 2u);
    assert(result.storedTasks == 2u);
    assert(!result.Truncated());
    assert(result.candidates.offset == 0u);
    assert(result.candidates.size == 2u);
    assert(result.candidates.daughterOffset == 0u);
    assert(result.candidates.daughterSize == 4u);
    assert(result.candidates.overflowFlags == 0u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 2u);
    assert(candidates.Daughters().Size() == 4u);
    assert(candidates.Metadata().Pdg(0) == 997);
    assert(candidates.Metadata().Flags(0)
           == static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca));
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0, fieldMother);
    ExpectFieldAwareSeedClose(fieldMother, referenceMother);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-field-aware-diagnostic",
         "decay plan executor routes an explicit field-aware diagnostic channel through generated tasks");
  }

  void TestDecayPlanMixedTransportExecutor(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    const KFParticleGpuTwoDaughterChannel fieldChannel =
      MakeFieldAwarePionPairChannel(78u);
    const KFParticleGpuTwoDaughterChannel lineChannel =
      MakePionPairChannel(79u);
    plan.AddTwoDaughterChannel(fieldChannel);
    plan.AddTwoDaughterChannel(lineChannel);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 2u);

    assert(results[0].channelId == 78u);
    assert(results[0].motherPdg == 997);
    assert(results[0].candidates.offset == 0u);
    assert(results[0].candidates.size == 2u);
    assert(results[0].candidates.daughterOffset == 0u);
    assert(results[0].candidates.daughterSize == 4u);
    assert(results[0].candidates.overflowFlags == 0u);

    assert(results[1].channelId == 79u);
    assert(results[1].motherPdg == 310);
    assert(results[1].candidates.offset == 2u);
    assert(results[1].candidates.size == 2u);
    assert(results[1].candidates.daughterOffset == 4u);
    assert(results[1].candidates.daughterSize == 4u);
    assert(results[1].candidates.overflowFlags == 0u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Daughters().Size() == 8u);
    assert(candidates.OverflowFlags() == 0u);
    ExpectChannelCandidatePdgs(candidates, results[0], 997);
    ExpectChannelCandidatePdgs(candidates, results[1], 310);
    ExpectChannelDaughterRange(candidates, results[0]);
    ExpectChannelDaughterRange(candidates, results[1]);

    KFParticleGpuFitState fieldMother;
    KFParticleGpuFitState lineMother;
    LoadCandidateFit(candidates, results[0].candidates.offset, fieldMother);
    LoadCandidateFit(candidates, results[1].candidates.offset, lineMother);
    assert(!AlmostEqual(fieldMother.X(), lineMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), lineMother.Px(), 1.e-6f));
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));

    plan.Clear();
    Pass("decay-plan-mixed-transport",
         "decay plan appends field-aware diagnostic and straight-line channels without range overlap");
  }

  void TestDecayPlanDefaultK0Executor(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddTwoDaughterChannel(MakeK0ShortToPiPlusPiMinusChannel());

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(results[0].motherPdg == 310);
    assert(results[0].acceptedTasks == 4u);
    assert(results[0].candidates.size == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(HasCandidateDaughters(candidates, 901, 1001));
    assert(HasCandidateDaughters(candidates, 902, 1001));
    assert(HasCandidateDaughters(candidates, 901, 1002));
    assert(HasCandidateDaughters(candidates, 902, 1002));
    ExpectChannelCandidatePdgs(candidates, results[0], 310);
    ExpectChannelDaughterRange(candidates, results[0]);

    plan.Clear();
    Pass("decay-plan-default-k0-executor",
         "default K0S channel descriptor runs through the standalone decay-plan executor");
  }

  void TestDecayPlanDefaultK0FieldAwareFixture(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const KFParticleGpuTwoDaughterChannel channel =
      MakeK0ShortToPiPlusPiMinusChannel();
    assert(channel.transportMode == KFGpuTransportFieldAware);
    plan.AddTwoDaughterChannel(channel);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 4u ? requested.candidates : 4u;
    requested.daughterIds = requested.daughterIds > 8u ? requested.daughterIds : 8u;
    buffers.EnsureCapacity(requested);
    FillGeneratedPipelineFixture(buffers);

    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    const KFParticleGpuRange firstRange(0, 2);
    const KFParticleGpuRange secondRange(2, 2);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(firstRange, secondRange, 0u, source, fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelK0ShortToPiPlusPiMinus);
    assert(results[0].motherPdg == 310);
    assert(results[0].acceptedTasks == 4u);
    assert(results[0].candidates.size == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Metadata().Pdg(0) == 310);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateValid);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateLineDca);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateEnergyFit);
    assert(HasCandidateDaughters(candidates, 901, 1001));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0, fieldMother);
    ExpectFieldAwareSeedClose(fieldMother, referenceMother);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-k0-field-fixture",
         "default K0S channel uses field-aware transport on a nonzero-field fixture");
  }

  void TestDecayPlanDefaultLambdaFieldAwareFixture(KFParticleGpuRuntime& runtime,
                                                   KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const KFParticleGpuTwoDaughterChannel channel = MakeLambdaToProtonPiMinusChannel();
    assert(channel.transportMode == KFGpuTransportFieldAware);
    plan.AddTwoDaughterChannel(channel);

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // The proton is the first daughter for Lambda and therefore supplies the
    // deterministic nonzero-field transport used by this fixture.
    StoreDiagnosticField(tracks, 1);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(KFParticleGpuRange(1, 1),
                         KFParticleGpuRange(2, 1),
                         0u,
                         source,
                         fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 1u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelLambdaToProtonPiMinus);
    assert(results[0].motherPdg == 3122);
    assert(results[0].acceptedTasks == 1u);
    assert(results[0].candidates.size == 1u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1u);
    assert(candidates.Metadata().Pdg(0) == 3122);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateValid);
    assert(HasCandidateDaughters(candidates, 3201, 3301));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0u, fieldMother);
    ExpectFieldAwareSeedClose(fieldMother, referenceMother);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-lambda-field-fixture",
         "default Lambda channel uses field-aware proton-pion transport on a nonzero-field fixture");
  }

  void TestDecayPlanDefaultAntiLambdaFieldAwareFixture(KFParticleGpuRuntime& runtime,
                                                       KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    const KFParticleGpuTwoDaughterChannel channel =
      MakeAntiLambdaToAntiProtonPiPlusChannel();
    assert(channel.transportMode == KFGpuTransportFieldAware);
    plan.AddTwoDaughterChannel(channel);

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // anti-Lambda reverses the track-set order: the anti-proton remains the
    // first daughter, now taken from the negative secondary track set.
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuTwoDaughterTaskSource source =
      MakeTwoDaughterTaskSource(channel, 0u);
    KFParticleGpuTwoDaughterTask fieldTask;
    FillTwoDaughterTask(KFParticleGpuRange(3, 1),
                         KFParticleGpuRange(0, 1),
                         0u,
                         source,
                         fieldTask);

    KFParticleGpuFitState referenceMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), fieldTask, referenceMother));

    KFParticleGpuTwoDaughterTask straightTask = fieldTask;
    straightTask.transportMode = KFGpuTransportStraightLine;
    KFParticleGpuFitState straightMother;
    assert(BuildTwoDaughterCandidate(
      MakeConstView(buffers.HostInputTracks()), straightTask, straightMother));

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 1u);
    assert(results.size() == 1u);
    assert(results[0].channelId == KFGpuChannelAntiLambdaToAntiProtonPiPlus);
    assert(results[0].motherPdg == -3122);
    assert(results[0].acceptedTasks == 1u);
    assert(results[0].candidates.size == 1u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 1u);
    assert(candidates.Metadata().Pdg(0) == -3122);
    assert(candidates.Metadata().Flags(0) & KFGpuCandidateValid);
    assert(HasCandidateDaughters(candidates, 3401, 3101));

    KFParticleGpuFitState fieldMother;
    LoadCandidateFit(candidates, 0u, fieldMother);
    ExpectFieldAwareSeedClose(fieldMother, referenceMother);
    assert(!AlmostEqual(fieldMother.X(), straightMother.X(), 1.e-6f));
    assert(!AlmostEqual(fieldMother.Px(), straightMother.Px(), 1.e-6f));

    plan.Clear();
    Pass("decay-plan-default-anti-lambda-field-fixture",
         "default anti-Lambda channel uses field-aware anti-proton-pion transport on a nonzero-field fixture");
  }

  void TestDefaultV0FieldAwareScalarReference(KFParticleGpuRuntime& runtime,
                                               KFParticleGpuBufferManager& buffers)
  {
    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    // One field-bearing first daughter per channel covers the track-set and
    // charge ordering of the three default V0 descriptors.
    StoreDiagnosticField(tracks, 0);
    StoreDiagnosticField(tracks, 1);
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuConstInputTrackSoAView inputTracks = MakeConstView(tracks);
    const KFParticleGpuTwoDaughterChannel channels[] = {
      MakeK0ShortToPiPlusPiMinusChannel(),
      MakeLambdaToProtonPiMinusChannel(),
      MakeAntiLambdaToAntiProtonPiPlusChannel()
    };
    constexpr unsigned int numberOfChannels = sizeof(channels) / sizeof(channels[0]);
    KFParticleGpuTwoDaughterTask tasks[numberOfChannels];
    ReferenceState references[numberOfChannels];

    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      const KFParticleGpuTwoDaughterChannel& channel = channels[i];
      assert(channel.transportMode == KFGpuTransportFieldAware);
      const KFParticleGpuTwoDaughterTaskSource source =
        MakeTwoDaughterTaskSource(channel, 0u);
      const KFParticleGpuRange firstRange =
        event.TrackSet(channel.firstTrackSet).Species(channel.firstSpecies);
      const KFParticleGpuRange secondRange =
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies);
      FillTwoDaughterTask(firstRange, secondRange, 0u, source, tasks[i]);

      // Isolate field transport from the later covariance-aware energy fit.
      tasks[i].flags = KFGpuTwoDaughterUseLineDca;

      KFParticleGpuTrackState firstTrack;
      KFParticleGpuTrackState secondTrack;
      LoadTrackState(inputTracks, tasks[i].firstTrack, firstTrack);
      LoadTrackState(inputTracks, tasks[i].secondTrack, secondTrack);
      const ReferenceState firstReference = MakeReferenceState(
        firstTrack, tasks[i].firstMass, inputTracks.Charge(tasks[i].firstTrack));
      const ReferenceState secondReference = MakeReferenceState(
        secondTrack, tasks[i].secondMass, inputTracks.Charge(tasks[i].secondTrack));
      const KFParticleGpuFieldValue field = EvaluateTrackField(
        inputTracks, tasks[i].firstTrack, static_cast<float>(firstReference.z));
      assert(ReferenceBuildConstantByDcaKinematicMother(
        firstReference, secondReference, field.y, references[i]));
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(numberOfChannels, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* hostTasks = HostPointer(taskBuffer);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      hostTasks[i] = tasks[i];
    }

    buffers.UploadInput();
    buffers.ResetCandidates();
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(numberOfChannels),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      numberOfChannels,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == numberOfChannels);
    assert(candidates.Daughters().Size() == 2u * numberOfChannels);
    assert(candidates.OverflowFlags() == 0u);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      KFParticleGpuFitState actual;
      LoadCandidateFit(candidates, i, actual);
      ExpectFitStateMatchesReference(actual, references[i], 8.e-5f);
      assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
      const unsigned int flags = candidates.Metadata().Flags(i);
      const unsigned int requiredFlags =
        static_cast<unsigned int>(KFGpuCandidateValid | KFGpuCandidateLineDca);
      assert((flags & requiredFlags) == requiredFlags);
      assert((flags & static_cast<unsigned int>(KFGpuCandidateEnergyFit)) == 0u);
      // The synthetic fixture is not tuned to pass each physical V0 cut.
      // Selection rejection is valid here; transport and stored fit remain
      // observable for the independent numerical comparison.
      assert((flags & ~static_cast<unsigned int>(
                        requiredFlags | KFGpuCandidateSelectionRejected)) == 0u);
      assert(candidates.Daughters().SourceId(2u * i) == inputTracks.SourceId(tasks[i].firstTrack));
      assert(candidates.Daughters().SourceId(2u * i + 1u)
             == inputTracks.SourceId(tasks[i].secondTrack));
    }

    Pass("default-v0-field-scalar-reference",
         "GPU field-aware K0S, Lambda, and anti-Lambda kinematic seeds match independent double-precision references");
  }

  void TestDefaultV0FieldAwareEnergyFit(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);
    StoreDiagnosticField(tracks, 1);
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuConstInputTrackSoAView inputTracks = MakeConstView(tracks);
    const KFParticleGpuTwoDaughterChannel channels[] = {
      MakeK0ShortToPiPlusPiMinusChannel(),
      MakeLambdaToProtonPiMinusChannel(),
      MakeAntiLambdaToAntiProtonPiPlusChannel()
    };
    constexpr unsigned int numberOfChannels = sizeof(channels) / sizeof(channels[0]);
    KFParticleGpuTwoDaughterTask tasks[numberOfChannels];
    KFParticleGpuFitState references[numberOfChannels];

    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      const KFParticleGpuTwoDaughterChannel& channel = channels[i];
      assert(channel.transportMode == KFGpuTransportFieldAware);
      const KFParticleGpuTwoDaughterTaskSource source =
        MakeTwoDaughterTaskSource(channel, 0u);
      FillTwoDaughterTask(
        event.TrackSet(channel.firstTrackSet).Species(channel.firstSpecies),
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies),
        0u,
        source,
        tasks[i]);
      assert((tasks[i].flags & KFGpuTwoDaughterUseLineDca) != 0u);
      assert((tasks[i].flags & KFGpuTwoDaughterUseEnergyFit) != 0u);
      assert(BuildTwoDaughterCandidate(inputTracks, tasks[i], references[i]));
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(numberOfChannels, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* hostTasks = HostPointer(taskBuffer);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      hostTasks[i] = tasks[i];
    }

    buffers.UploadInput();
    buffers.ResetCandidates();
    runtime.GetQueue().copy(taskBuffer, xpu::h2d);
    runtime.GetQueue().wait();
    runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
      xpu::n_threads(numberOfChannels),
      MakeConstView(buffers.DeviceInputTracks()),
      taskBuffer.get(),
      numberOfChannels,
      buffers.DeviceCandidates());
    runtime.GetQueue().wait();
    buffers.DownloadCandidates();

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == numberOfChannels);
    assert(candidates.Daughters().Size() == 2u * numberOfChannels);
    assert(candidates.OverflowFlags() == 0u);
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      KFParticleGpuFitState actual;
      LoadCandidateFit(candidates, i, actual);
      ExpectDefaultV0EnergyFitClose(actual, references[i], tasks[i].motherPdg);
      assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
      assert(candidates.Metadata().Flags(i)
             == (TwoDaughterCandidateFlags(tasks[i])
                 | (PassTwoDaughterPostBuildSelection(tasks[i], references[i])
                      ? 0u
                      : static_cast<unsigned int>(KFGpuCandidateSelectionRejected))));
      assert(candidates.Daughters().SourceId(2u * i) == inputTracks.SourceId(tasks[i].firstTrack));
      assert(candidates.Daughters().SourceId(2u * i + 1u)
             == inputTracks.SourceId(tasks[i].secondTrack));
    }

    Pass("default-v0-field-energy-fit",
         "GPU field-aware K0S, Lambda, and anti-Lambda energy fits match channel-specific state, chi2, and mass tolerances");
  }

  void ProfileDefaultV0FieldAwareEnergyFit(KFParticleGpuRuntime& runtime,
                                           KFParticleGpuBufferManager& buffers)
  {
    const unsigned int iterations = FieldAwareProfileIterations();
    if (iterations == 0u) {
      return;
    }

    FillDefaultV0Fixture(buffers);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreDiagnosticField(tracks, 0);
    StoreDiagnosticField(tracks, 1);
    StoreDiagnosticField(tracks, 3);

    const KFParticleGpuTwoDaughterChannel channels[] = {
      MakeK0ShortToPiPlusPiMinusChannel(),
      MakeLambdaToProtonPiMinusChannel(),
      MakeAntiLambdaToAntiProtonPiPlusChannel()
    };
    constexpr unsigned int numberOfChannels = sizeof(channels) / sizeof(channels[0]);
    KFParticleGpuTwoDaughterTask fieldTasks[numberOfChannels];
    KFParticleGpuTwoDaughterTask lineTasks[numberOfChannels];
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    for (unsigned int i = 0; i < numberOfChannels; ++i) {
      const KFParticleGpuTwoDaughterChannel& channel = channels[i];
      FillTwoDaughterTask(
        event.TrackSet(channel.firstTrackSet).Species(channel.firstSpecies),
        event.TrackSet(channel.secondTrackSet).Species(channel.secondSpecies),
        0u,
        MakeTwoDaughterTaskSource(channel, 0u),
        fieldTasks[i]);
      lineTasks[i] = fieldTasks[i];
      lineTasks[i].transportMode = KFGpuTransportStraightLine;
    }

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(numberOfChannels, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* hostTasks = HostPointer(taskBuffer);
    buffers.UploadInput();

    const auto runProfile = [&](const KFParticleGpuTwoDaughterTask* tasks) {
      for (unsigned int i = 0; i < numberOfChannels; ++i) {
        hostTasks[i] = tasks[i];
      }
      runtime.GetQueue().copy(taskBuffer, xpu::h2d);
      runtime.GetQueue().wait();

      // Warm-up makes image loading and first-use allocations irrelevant to
      // the reported steady-state queue submission and reconstruction cost.
      for (unsigned int warmup = 0; warmup < 8u; ++warmup) {
        buffers.ResetCandidates();
        runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
          xpu::n_threads(numberOfChannels),
          MakeConstView(buffers.DeviceInputTracks()),
          taskBuffer.get(),
          numberOfChannels,
          buffers.DeviceCandidates());
        runtime.GetQueue().wait();
      }

      const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
      for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
        buffers.ResetCandidates();
        runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
          xpu::n_threads(numberOfChannels),
          MakeConstView(buffers.DeviceInputTracks()),
          taskBuffer.get(),
          numberOfChannels,
          buffers.DeviceCandidates());
        runtime.GetQueue().wait();
      }
      const std::chrono::steady_clock::time_point finished = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(finished - started).count()
             / static_cast<double>(iterations * numberOfChannels);
    };

    const double fieldMicrosecondsPerCandidate = runProfile(fieldTasks);
    const double lineMicrosecondsPerCandidate = runProfile(lineTasks);
    assert(fieldMicrosecondsPerCandidate > 0.0);
    assert(lineMicrosecondsPerCandidate > 0.0);
    std::cerr << "PROFILE field-aware-default-v0-energy-fit"
              << " iterations=" << iterations
              << " candidates-per-iteration=" << numberOfChannels
              << " field-us-per-candidate=" << fieldMicrosecondsPerCandidate
              << " line-us-per-candidate=" << lineMicrosecondsPerCandidate
              << " field-to-line=" << fieldMicrosecondsPerCandidate / lineMicrosecondsPerCandidate
              << '\n';
  }

  void TestDecayPlanDefaultV0Executor(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);

    FillDefaultV0Fixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 3u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 3u);
    assert(candidates.Daughters().Size() == 6u);
    assert(candidates.OverflowFlags() == 0u);
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[0],
                                     plan.TwoDaughterChannel(0),
                                     KFGpuChannelK0ShortToPiPlusPiMinus,
                                     310,
                                     0u,
                                     0u);
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[1],
                                     plan.TwoDaughterChannel(1),
                                     KFGpuChannelLambdaToProtonPiMinus,
                                     3122,
                                     1u,
                                     2u);
    ExpectDefaultV0ChannelRegression(candidates,
                                     results[2],
                                     plan.TwoDaughterChannel(2),
                                     KFGpuChannelAntiLambdaToAntiProtonPiPlus,
                                     -3122,
                                     2u,
                                     4u);
    assert(HasCandidateDaughters(candidates, 3101, 3301));
    assert(HasCandidateDaughters(candidates, 3201, 3301));
    assert(HasCandidateDaughters(candidates, 3401, 3101));
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuSelectedCandidateRange selectedRange =
      steering.LastDecayPlanSelectedCandidates();
    assert(selectedRange.size == selected.Size());
    assert(selectedRange.overflowFlags == selected.OverflowFlags());

    plan.Clear();
    Pass("decay-plan-default-v0-executor",
         "default V0 regression validates K0S, Lambda, and anti-Lambda ranges, flags, and daughters");
  }

  void TestDecayPlanDefaultV0SelectionRegression(KFParticleGpuRuntime& runtime,
                                                 KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);
    FillDefaultV0Fixture(buffers);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    const KFParticleGpuConstVertexSoAView primaryVertices =
      MakeConstView(buffers.HostPrimaryVertices());
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];
    std::vector<bool> expected(candidates.Size(), false);
    std::vector<KFParticleGpuV0SelectionResult> expectedDecisions(candidates.Size());
    unsigned int expectedSize = 0u;

    for (std::size_t channelIndex = 0; channelIndex < results.size(); ++channelIndex) {
      const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(channelIndex);
      const KFParticleGpuTwoDaughterChannelResult& channelResult = results[channelIndex];
      for (unsigned int candidateIndex = channelResult.candidates.offset;
           candidateIndex < channelResult.candidates.End();
           ++candidateIndex) {
        KFParticleGpuFitState candidate;
        LoadCandidateFit(candidates, candidateIndex, candidate);
        KFParticleGpuV0SelectionResult decision;
        KFParticleGpuSelection::EvaluateV0Selection(candidate,
                                                     candidates.Metadata().Flags(candidateIndex),
                                                     candidateIndex,
                                                     channel.channelId,
                                                     0u,
                                                     primaryVertices,
                                                     event.primaryVertices,
                                                     channel.selection,
                                                     decision);
        assert(candidates.Metadata().ChannelId(candidateIndex) == channel.channelId);
        expectedDecisions[candidateIndex] = decision;
        expected[candidateIndex] = KFParticleGpuSelection::IsSelected(decision);
        expectedSize += expected[candidateIndex] ? 1u : 0u;
      }
    }

    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuSelectedCandidateRange selectedRange =
      steering.LastDecayPlanSelectedCandidates();
    assert(selected.Size() == expectedSize);
    assert(selectedRange.size == expectedSize);
    assert(selectedRange.overflowFlags == selected.OverflowFlags());
    const KFParticleGpuConstV0SelectionResultView selectionResults =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuSelectedV0View selectedV0s =
      MakeSelectedV0View(candidates, selected, selectionResults);
    assert(selectedV0s.Size() == selected.Size());
    assert(selectedV0s.OverflowFlags() == selected.OverflowFlags());
    assert(selectedV0s.HasSelectionResults());

    for (unsigned int candidateIndex = 0; candidateIndex < candidates.Size(); ++candidateIndex) {
      ExpectSelectionResultClose(selectionResults.Result(candidateIndex),
                                 expectedDecisions[candidateIndex]);
    }

    const std::vector<KFParticleGpuSelectedChannelRange>& selectedChannels =
      steering.LastDecayPlanSelectedChannels();
    assert(selectedChannels.size() == results.size());
    unsigned int rangeEnd = 0u;
    for (std::size_t channelIndex = 0; channelIndex < selectedChannels.size(); ++channelIndex) {
      const KFParticleGpuSelectedChannelRange& channelRange = selectedChannels[channelIndex];
      assert(channelRange.channelId == results[channelIndex].channelId);
      assert(channelRange.eventIndex == 0u);
      assert(channelRange.candidates.offset == rangeEnd);
      rangeEnd = channelRange.candidates.End();
      for (unsigned int selectedIndex = channelRange.candidates.offset;
           selectedIndex < channelRange.candidates.End();
           ++selectedIndex) {
        assert(selected.ChannelId(selectedIndex) == channelRange.channelId);
      }
    }
    assert(rangeEnd == selected.Size());

    std::vector<bool> observed(candidates.Size(), false);
    for (unsigned int selectedIndex = 0; selectedIndex < selected.Size(); ++selectedIndex) {
      const unsigned int candidateIndex = selected.Index(selectedIndex);
      assert(candidateIndex < candidates.Size());
      assert(expected[candidateIndex]);
      assert(!observed[candidateIndex]);
      assert(selectedV0s.CandidateIndex(selectedIndex) == candidateIndex);
      assert(selectedV0s.EventIndex(selectedIndex) == 0u);
      assert(selectedV0s.DaughterCount(selectedIndex) == 2u);
      assert(selectedV0s.Pdg(selectedIndex) == candidates.Metadata().Pdg(candidateIndex));
      assert(selectedV0s.ChannelId(selectedIndex) == expectedDecisions[candidateIndex].channelId);
      ExpectSelectionResultClose(selectedV0s.Selection(selectedIndex), expectedDecisions[candidateIndex]);
      KFParticleGpuFitState selectedFit;
      selectedV0s.LoadFitState(selectedIndex, selectedFit);
      KFParticleGpuFitState rawFit;
      LoadCandidateFit(candidates, candidateIndex, rawFit);
      ExpectFitStateClose(selectedFit, rawFit);
      assert(selectedV0s.DaughterSourceId(selectedIndex, 0u)
             == candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(candidateIndex)));
      observed[candidateIndex] = true;
    }
    assert(observed == expected);

    plan.Clear();
    Pass("default-v0-selection-regression",
         "device selection diagnostics, channel ranges, and compact membership match host decisions");
  }

  void TestDecayPlanSelectedOutputTruncation(KFParticleGpuRuntime& runtime,
                                             KFParticleGpuBufferManager& buffers)
  {
    assert(buffers.Capacities().selectedCandidates == 1u);
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();

    KFParticleGpuTwoDaughterChannel first = MakeK0ShortToPiPlusPiMinusChannel(81u);
    KFParticleGpuTwoDaughterChannel second = first;
    second.channelId = 82u;
    for (KFParticleGpuTwoDaughterChannel* channel : {&first, &second}) {
      channel->secondaryMassSigmaCut = -1.f;
      channel->maxSecondaryTopoChi2PerNdf = -1.f;
      channel->minSecondaryLdL = -1.f;
      channel->selection.expectedMass = 1.f;
      channel->selection.expectedMassSigma = -1.f;
      channel->selection.massSigmaCut = -1.f;
      channel->selection.maxGeometricChi2PerNdf = -1.f;
      channel->selection.maxPrimaryVertexDistance = -1.f;
      channel->selection.minSecondaryLdL = -1.f;
      channel->selection.maxPrimaryTopologyChi2PerNdf = -1.f;
      channel->selection.maxSecondaryTopologyChi2PerNdf = -1.f;
      channel->selection.requirePrimaryVertex = 0u;
    }
    plan.AddTwoDaughterChannel(first);
    plan.AddTwoDaughterChannel(second);
    FillInput(buffers, buffers.Capacities());
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0u, -0.2f, 0.1f, 0.f, 0.8f, 0.1f, 1.f, 211, 1, 8101);
    StoreSyntheticTrack(tracks, 1u, 0.1f, 0.5f, -0.4f, -0.3f, 0.7f, 1.1f, -211, -1, 8201);
    KFParticleGpuEventDesc* events = buffers.HostEvents();
    events[0] = KFParticleGpuEventDesc();
    events[0].eventId = 181;
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(0u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).tracks = KFParticleGpuRange(1u, 1u);
    events[0].TrackSet(SecondaryNegativeFirst).Species(Pion) = KFParticleGpuRange(1u, 1u);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 4u);
    assert(results.size() == 2u);
    assert(results[0].candidates.size == 1u);
    assert(results[1].candidates.size == 1u);

    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuConstV0SelectionResultView selectionResults =
      MakeConstView(buffers.HostV0SelectionResults());
    assert(selected.Size() == 1u);
    assert(selected.OverflowFlags() == KFGpuSelectedCandidateCapacityExceeded);
    assert(selectionResults.Result(0u).channelId == 81u);
    assert(selectionResults.Result(1u).channelId == 82u);
    assert(KFParticleGpuSelection::IsSelected(selectionResults.Result(0u)));
    assert(KFParticleGpuSelection::HasRejection(
      selectionResults.Result(1u), KFGpuV0SelectionRejectOutputOverflow));

    const std::vector<KFParticleGpuSelectedChannelRange>& ranges =
      steering.LastDecayPlanSelectedChannels();
    assert(ranges.size() == 2u);
    assert(ranges[0].channelId == 81u);
    assert(ranges[0].candidates.offset == 0u);
    assert(ranges[0].candidates.size == 1u);
    assert(ranges[1].channelId == 82u);
    assert(ranges[1].candidates.offset == 1u);
    assert(ranges[1].candidates.size == 0u);

    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.selectedCandidates = requested.candidates;
    buffers.EnsureCapacity(requested);
    plan.Clear();
    Pass("decay-plan-selected-output-truncation",
         "RunDecayPlan records compact-pool overflow and preserves per-channel ranges");
  }

  void TestDecayPlanSelectionMultiPvBoundary(KFParticleGpuRuntime& runtime,
                                             KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    AddDefaultV0TwoDaughterChannels(plan);
    FillDefaultV0Fixture(buffers);
    buffers.SetInputSizes(4u, 2u, 1u);

    KFParticleGpuVertexState tiedVertex;
    tiedVertex.X() = 0.f;
    tiedVertex.Y() = 0.f;
    tiedVertex.Z() = 0.f;
    tiedVertex.Covariance(0) = 1.f;
    tiedVertex.Covariance(2) = 1.f;
    tiedVertex.Covariance(5) = 1.f;
    KFParticleGpuVertexSoAView vertices = buffers.HostPrimaryVertices();
    StoreVertexState(tiedVertex, vertices, 0u);
    StoreVertexState(tiedVertex, vertices, 1u);
    buffers.HostEvents()[0].primaryVertices = KFParticleGpuRange(0u, 2u);

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& initialResults =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstV0SelectionResultView initialSelections =
      MakeConstView(buffers.HostV0SelectionResults());
    assert(initialResults.size() == 3u);
    for (const KFParticleGpuTwoDaughterChannelResult& result : initialResults) {
      for (unsigned int candidateIndex = result.candidates.offset;
           candidateIndex < result.candidates.End();
           ++candidateIndex) {
        assert(initialSelections.Result(candidateIndex).bestPrimaryVertexIndex == 0);
      }
    }

    std::vector<KFParticleGpuTwoDaughterChannel> boundaryChannels;
    for (std::size_t channelIndex = 0; channelIndex < initialResults.size(); ++channelIndex) {
      const KFParticleGpuTwoDaughterChannelResult& result = initialResults[channelIndex];
      assert(result.candidates.size == 1u);
      const float boundary =
        initialSelections.Result(result.candidates.offset).observables.nearestPrimaryVertexDistance;
      KFParticleGpuTwoDaughterChannel channel = plan.TwoDaughterChannel(channelIndex);
      channel.selection.maxPrimaryVertexDistance = boundary;
      boundaryChannels.push_back(channel);
    }

    plan.Clear();
    for (const KFParticleGpuTwoDaughterChannel& channel : boundaryChannels) {
      plan.AddTwoDaughterChannel(channel);
    }

    const std::vector<KFParticleGpuTwoDaughterChannelResult>& boundaryResults =
      steering.RunDecayPlan(0u, 4u);
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    const KFParticleGpuConstVertexSoAView primaryVertices =
      MakeConstView(buffers.HostPrimaryVertices());
    const KFParticleGpuConstV0SelectionResultView deviceSelections =
      MakeConstView(buffers.HostV0SelectionResults());
    const KFParticleGpuEventDesc& event = buffers.HostEvents()[0];

    for (std::size_t channelIndex = 0; channelIndex < boundaryResults.size(); ++channelIndex) {
      const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(channelIndex);
      const KFParticleGpuTwoDaughterChannelResult& result = boundaryResults[channelIndex];
      for (unsigned int candidateIndex = result.candidates.offset;
           candidateIndex < result.candidates.End();
           ++candidateIndex) {
        KFParticleGpuFitState candidate;
        LoadCandidateFit(candidates, candidateIndex, candidate);
        KFParticleGpuV0SelectionResult expected;
        KFParticleGpuSelection::EvaluateV0Selection(candidate,
                                                     candidates.Metadata().Flags(candidateIndex),
                                                     candidateIndex,
                                                     channel.channelId,
                                                     0u,
                                                     primaryVertices,
                                                     event.primaryVertices,
                                                     channel.selection,
                                                     expected);
        ExpectSelectionResultClose(deviceSelections.Result(candidateIndex), expected);
        assert(deviceSelections.Result(candidateIndex).bestPrimaryVertexIndex == 0);
        assert(KFParticleGpuSelection::HasRejection(
          deviceSelections.Result(candidateIndex), KFGpuV0SelectionRejectDistance));
      }
    }

    plan.Clear();
    Pass("decay-plan-selection-multi-pv-boundary",
         "RunDecayPlan preserves deterministic multi-PV ties and exact distance-cut rejection");
  }

  void TestDecayPlanMultiChannelExecutor(KFParticleGpuRuntime& runtime,
                                         KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(51u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(52u));

    FillDecayPlanMultiChannelFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& results =
      steering.RunDecayPlan(0u, 2u);
    assert(results.size() == 2u);

    assert(results[0].channelId == 51u);
    assert(results[0].motherPdg == 310);
    assert(results[0].totalPairs == 4u);
    assert(results[0].acceptedTasks == 2u);
    assert(results[0].storedTasks == 2u);
    assert(!results[0].Empty());
    assert(!results[0].HasAnyOverflow());
    assert(results[0].candidates.offset == 0u);
    assert(results[0].candidates.size == 2u);
    assert(results[0].candidates.daughterOffset == 0u);
    assert(results[0].candidates.daughterSize == 4u);

    assert(results[1].channelId == 52u);
    assert(results[1].motherPdg == 333);
    assert(results[1].totalPairs == 4u);
    assert(results[1].acceptedTasks == 2u);
    assert(results[1].storedTasks == 2u);
    assert(!results[1].Empty());
    assert(!results[1].HasAnyOverflow());
    assert(results[1].candidates.offset == 2u);
    assert(results[1].candidates.size == 2u);
    assert(results[1].candidates.daughterOffset == 4u);
    assert(results[1].candidates.daughterSize == 4u);

    const KFParticleGpuConstCandidatePoolView candidates =
      MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == 4u);
    assert(candidates.Daughters().Size() == 8u);
    assert(candidates.OverflowFlags() == 0u);
    assert(HasCandidateDaughters(candidates, 2101, 2301));
    assert(HasCandidateDaughters(candidates, 2102, 2301));
    assert(HasCandidateDaughters(candidates, 2201, 2401));
    assert(HasCandidateDaughters(candidates, 2202, 2401));
    ExpectChannelCandidatePdgs(candidates, results[0], 310);
    ExpectChannelCandidatePdgs(candidates, results[1], 333);
    ExpectChannelDaughterRange(candidates, results[0]);
    ExpectChannelDaughterRange(candidates, results[1]);

    plan.Clear();
    Pass("decay-plan-multi-channel-executor",
         "decay plan executor appends multiple channels with non-overlapping output ranges");
  }

  void TestDecayPlanBatchExecutor(KFParticleGpuRuntime& runtime,
                                  KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(91u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(92u));

    FillDecayPlanMultiEventFixture(buffers);
    KFParticleGpuBufferCapacities requested = buffers.Capacities();
    requested.candidates = requested.candidates > 10u ? requested.candidates : 10u;
    requested.daughterIds = requested.daughterIds > 18u ? requested.daughterIds : 18u;
    requested.selectedCandidates = requested.selectedCandidates > 10u ? requested.selectedCandidates : 10u;
    buffers.EnsureCapacity(requested);

    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialFirst =
      steering.RunDecayPlan(0u, 4u);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialSecond =
      steering.RunDecayPlan(1u, 4u);
    const unsigned int serialFirstCandidates =
      serialFirst[0u].candidates.size + serialFirst[1u].candidates.size;
    const unsigned int serialFirstDaughters =
      serialFirst[0u].candidates.daughterSize + serialFirst[1u].candidates.daughterSize;
    const unsigned int serialSecondCandidates =
      serialSecond[0u].candidates.size + serialSecond[1u].candidates.size;
    const unsigned int serialSecondDaughters =
      serialSecond[0u].candidates.daughterSize + serialSecond[1u].candidates.daughterSize;

    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& batch =
      steering.RunDecayPlanBatch(0u, 2u, 4u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();

    assert(batch.size() == 4u);
    assert(events.size() == 2u);
    assert(events[0].eventIndex == 0u);
    assert(events[0].channelOffset == 0u);
    assert(events[0].channelCount == 2u);
    assert(events[0].candidates.offset == 0u);
    assert(events[0].candidates.size == serialFirstCandidates);
    assert(events[0].candidates.daughterOffset == 0u);
    assert(events[0].candidates.daughterSize == serialFirstDaughters);
    assert(events[1].eventIndex == 1u);
    assert(events[1].channelOffset == 2u);
    assert(events[1].channelCount == 2u);
    assert(events[1].candidates.offset == serialFirstCandidates);
    assert(events[1].candidates.size == serialSecondCandidates);
    assert(events[1].candidates.daughterOffset == serialFirstDaughters);
    assert(events[1].candidates.daughterSize == serialSecondDaughters);

    for (unsigned int channel = 0u; channel < 2u; ++channel) {
      assert(batch[channel].eventIndex == 0u);
      assert(batch[channel + 2u].eventIndex == 1u);
      assert(batch[channel].totalPairs == serialFirst[channel].totalPairs);
      assert(batch[channel].acceptedTasks == serialFirst[channel].acceptedTasks);
      assert(batch[channel].candidates.size == serialFirst[channel].candidates.size);
      assert(batch[channel + 2u].totalPairs == serialSecond[channel].totalPairs);
      assert(batch[channel + 2u].acceptedTasks == serialSecond[channel].acceptedTasks);
      assert(batch[channel + 2u].candidates.size == serialSecond[channel].candidates.size);
    }

    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
    assert(candidates.Size() == serialFirstCandidates + serialSecondCandidates);
    assert(candidates.Daughters().Size() == serialFirstDaughters + serialSecondDaughters);
    for (unsigned int candidate = 0u; candidate < serialFirstCandidates; ++candidate) {
      assert(candidates.Metadata().EventIndex(candidate) == 0u);
    }
    for (unsigned int candidate = serialFirstCandidates; candidate < candidates.Size(); ++candidate) {
      assert(candidates.Metadata().EventIndex(candidate) == 1u);
    }
    assert(HasCandidateDaughters(candidates, 2501, 2601));
    assert(HasCandidateDaughters(candidates, 2701, 2901));
    assert(HasCandidateDaughters(candidates, 2801, 3001));
    assert(steering.LastDecayPlanSelectedChannels().size() == 4u);

    plan.Clear();
    Pass("decay-plan-batch-executor",
         "one upload executes two isolated event/channel plans with stable event ranges and serial-equivalent lineage");
  }

  void TestDecayPlanBatchSelectedOutput(KFParticleGpuRuntime& runtime,
                                        KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();
    plan.Clear();
    KFParticleGpuTwoDaughterChannel channel = MakeK0ShortToPiPlusPiMinusChannel(93u);
    // Isolate compact-output partitioning from physics cuts: the default
    // descriptor and daughter identity remain intact, while both events must
    // contribute one accepted compact index.
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
    plan.AddTwoDaughterChannel(channel);

    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialFirst =
      steering.RunDecayPlan(0u, 1u);
    const unsigned int serialFirstSelected = steering.LastDecayPlanSelectedCandidates().size;
    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult> serialSecond =
      steering.RunDecayPlan(1u, 1u);
    const unsigned int serialSecondSelected = steering.LastDecayPlanSelectedCandidates().size;

    FillBatchSelectionFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& batch =
      steering.RunDecayPlanBatch(0u, 2u, 1u);
    const std::vector<KFParticleGpuDecayPlanEventResult>& events =
      steering.LastDecayPlanEventResults();
    const std::vector<KFParticleGpuSelectedChannelRange>& ranges =
      steering.LastDecayPlanSelectedChannels();
    const KFParticleGpuConstSelectedCandidateIndexView selected =
      MakeConstView(buffers.HostSelectedCandidates());
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());

    assert(serialFirst.size() == 1u && serialSecond.size() == 1u);
    assert(serialFirstSelected == 1u && serialSecondSelected == 1u);
    assert(batch.size() == 2u && events.size() == 2u && ranges.size() == 2u);
    assert(selected.Size() == 2u && selected.OverflowFlags() == 0u);
    for (unsigned int event = 0u; event < 2u; ++event) {
      assert(batch[event].eventIndex == event);
      assert(batch[event].candidates.size == 1u);
      assert(events[event].selectedCandidates.size == 1u);
      assert(ranges[event].eventIndex == event);
      assert(ranges[event].channelId == 93u);
      assert(ranges[event].candidates.size == 1u);
      const unsigned int selectedIndex = ranges[event].candidates.offset;
      const unsigned int candidateIndex = selected.Index(selectedIndex);
      assert(candidates.Metadata().EventIndex(candidateIndex) == event);
      assert(candidates.Metadata().ChannelId(candidateIndex) == 93u);
      assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(candidateIndex))
             == (event == 0u ? 4101 : 4301));
    }

    plan.Clear();
    Pass("decay-plan-batch-selected-output",
         "two events retain non-empty compact selection ranges, event identity, and source lineage");
  }

  void TestDecayPlanExecutorEdgeCases(KFParticleGpuRuntime& runtime,
                                      KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuSteering& steering = runtime.GetSteering();
    KFParticleGpuDecayPlan& plan = steering.GetDecayPlan();

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(61u));
    FillDecayPlanMultiChannelFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& truncated =
      steering.RunDecayPlan(0u, 1u);
    assert(truncated.size() == 1u);
    assert(truncated[0].acceptedTasks == 2u);
    assert(truncated[0].storedTasks == 1u);
    assert(truncated[0].Truncated());
    assert(truncated[0].HasOverflow(CandidateCapacityExceeded));
    assert(truncated[0].candidates.size == 1u);
    assert(truncated[0].candidates.daughterSize == 2u);
    assert(truncated[0].candidates.overflowFlags == CandidateCapacityExceeded);

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(62u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(63u));
    FillDecayPlanMultiChannelFixture(buffers);
    KFParticleGpuInputTrackSoAView overflowTracks = buffers.HostInputTracks();
    overflowTracks.ChiToPrimaryVertex(5) = 1.f;
    overflowTracks.ChiToPrimaryVertex(7) = 1.f;
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& overflow =
      steering.RunDecayPlan(0u, 4u);
    assert(overflow.size() == 2u);
    assert(overflow[0].acceptedTasks == 4u);
    assert(overflow[0].candidates.size == 4u);
    assert(overflow[0].candidates.daughterSize == 8u);
    assert(overflow[1].acceptedTasks == 4u);
    assert(overflow[1].candidates.offset == 4u);
    assert(overflow[1].candidates.daughterOffset == 8u);
    assert(overflow[1].HasOverflow(CandidateCapacityExceeded));
    assert(overflow[1].HasOverflow(DaughterCapacityExceeded));
    const KFParticleGpuConstCandidatePoolView overflowCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(overflowCandidates.Size() == overflowCandidates.Capacity());
    assert(overflowCandidates.OverflowFlags()
           == static_cast<unsigned int>(CandidateCapacityExceeded | DaughterCapacityExceeded));

    plan.Clear();
    plan.AddTwoDaughterChannel(MakePionPairChannel(64u));
    plan.AddTwoDaughterChannel(MakeKaonPairChannel(65u));
    FillDecayPlanMultiEventFixture(buffers);
    const std::vector<KFParticleGpuTwoDaughterChannelResult>& multiEvent =
      steering.RunDecayPlan(1u, 2u);
    assert(multiEvent.size() == 2u);
    assert(multiEvent[0].eventIndex == 1u);
    assert(multiEvent[1].eventIndex == 1u);
    assert(multiEvent[0].candidates.offset == 0u);
    assert(multiEvent[1].candidates.offset == 2u);
    const KFParticleGpuConstCandidatePoolView eventCandidates =
      MakeConstView(buffers.HostCandidates());
    assert(eventCandidates.Size() == 4u);
    assert(HasCandidateDaughters(eventCandidates, 2701, 2901));
    assert(HasCandidateDaughters(eventCandidates, 2702, 2901));
    assert(HasCandidateDaughters(eventCandidates, 2801, 3001));
    assert(HasCandidateDaughters(eventCandidates, 2802, 3001));
    ExpectChannelCandidatePdgs(eventCandidates, multiEvent[0], 310);
    ExpectChannelCandidatePdgs(eventCandidates, multiEvent[1], 333);
    ExpectChannelDaughterRange(eventCandidates, multiEvent[0]);
    ExpectChannelDaughterRange(eventCandidates, multiEvent[1]);
    for (unsigned int i = 0; i < eventCandidates.Size(); ++i) {
      assert(eventCandidates.Metadata().EventIndex(i) == 1u);
    }

    plan.Clear();
    Pass("decay-plan-executor-edge-cases",
         "decay plan executor reports truncation, overflow, and multi-event channel ranges");
  }

  void TestTwoDaughterSyntheticGrid(KFParticleGpuRuntime& runtime, KFParticleGpuBufferManager& buffers)
  {
    buffers.SetInputSizes(4, 1, 1);
    KFParticleGpuInputTrackSoAView tracks = buffers.HostInputTracks();
    StoreSyntheticTrack(tracks, 0, 0.0f, 0.1f, -0.2f, 0.8f, 0.3f, 1.2f, 211, 1, 301);
    StoreSyntheticTrack(tracks, 1, 0.3f, -0.4f, 0.5f, -0.2f, 0.9f, 0.7f, -211, -1, 302);
    StoreSyntheticTrack(tracks, 2, -0.5f, 0.2f, 0.4f, 1.1f, -0.6f, 0.5f, 321, 1, 401);
    StoreSyntheticTrack(tracks, 3, 0.7f, 0.9f, -0.3f, -0.4f, 0.2f, 1.0f, -321, -1, 402);
    buffers.UploadInput();

    xpu::buffer<KFParticleGpuTwoDaughterTask> taskBuffer(2, xpu::buf_io);
    KFParticleGpuTwoDaughterTask* tasks = HostPointer(taskBuffer);

    for (int pass = 0; pass < 2; ++pass) {
      buffers.ResetCandidates();

      tasks[0] = KFParticleGpuTwoDaughterTask();
      tasks[1] = KFParticleGpuTwoDaughterTask();
      if (pass == 0) {
        tasks[0].firstTrack = 0;
        tasks[0].secondTrack = 1;
        tasks[0].firstDaughterPdg = 211;
        tasks[0].secondDaughterPdg = -211;
        tasks[0].motherPdg = 310;
        tasks[0].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
        tasks[1].firstTrack = 2;
        tasks[1].secondTrack = 3;
        tasks[1].firstDaughterPdg = 321;
        tasks[1].secondDaughterPdg = -321;
        tasks[1].motherPdg = 333;
        tasks[1].flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
      }
      else {
        tasks[0].firstTrack = 0;
        tasks[0].secondTrack = 3;
        tasks[0].firstDaughterPdg = 211;
        tasks[0].secondDaughterPdg = -321;
        tasks[0].motherPdg = 313;
        tasks[0].flags = KFGpuTwoDaughterUseEnergyFit;
        tasks[1].firstTrack = 2;
        tasks[1].secondTrack = 1;
        tasks[1].firstDaughterPdg = 321;
        tasks[1].secondDaughterPdg = -211;
        tasks[1].motherPdg = -313;
        tasks[1].flags = KFGpuTwoDaughterUseLineDca;
      }

      for (unsigned int i = 0; i < 2; ++i) {
        tasks[i].eventIndex = 0;
        tasks[i].primaryVertexIndex = -1;
        tasks[i].firstMass = 0.13957f + 0.05f * static_cast<float>(i + pass);
        tasks[i].secondMass = 0.13957f + 0.03f * static_cast<float>(i + 1);
      }

      KFParticleGpuFitState references[2];
      const KFParticleGpuConstInputTrackSoAView constTracks = MakeConstView(buffers.HostInputTracks());
      for (unsigned int i = 0; i < 2; ++i) {
        const bool built = BuildTwoDaughterCandidate(constTracks, tasks[i], references[i]);
        assert(built);
      }

      runtime.GetQueue().copy(taskBuffer, xpu::h2d);
      runtime.GetQueue().wait();
      runtime.GetQueue().launch<KFParticleGpuTwoDaughterTaskKernel>(
        xpu::n_threads(2),
        MakeConstView(buffers.DeviceInputTracks()),
        taskBuffer.get(),
        2u,
        buffers.DeviceCandidates());
      runtime.GetQueue().wait();
      buffers.DownloadCandidates();

      const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(buffers.HostCandidates());
      assert(candidates.Size() == 2);
      assert(candidates.Daughters().Size() == 4);
      assert(candidates.OverflowFlags() == 0);
      for (unsigned int i = 0; i < 2; ++i) {
        KFParticleGpuFitState actual;
        LoadCandidateFit(candidates, i, actual);
        ExpectFitStateClose(actual, references[i], 2.e-5f);
        assert(candidates.Metadata().Pdg(i) == tasks[i].motherPdg);
        assert(candidates.Metadata().DaughterCount(i) == 2);
        assert(candidates.Metadata().Flags(i) & KFGpuCandidateValid);
        assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(i))
               == tracks.SourceId(tasks[i].firstTrack));
        assert(candidates.Daughters().SourceId(candidates.Metadata().DaughterOffset(i) + 1u)
               == tracks.SourceId(tasks[i].secondTrack));
      }
    }

    Pass("two-daughter-synthetic-grid", "CPU-vs-GPU comparison over multiple pair geometries and covariance cross terms");
  }

  void TestSelectionHelpers(KFParticleGpuBufferManager& buffers)
  {
    KFParticleGpuCandidatePoolView mutableCandidates = buffers.HostCandidates();
    const KFParticleGpuConstCandidatePoolView candidates = MakeConstView(mutableCandidates);

    assert(candidates.Size() >= 1);
    assert(KFParticleGpuSelection::HasDistinctDaughterSourceIds(candidates, 0));

    KFParticleGpuFitState candidate;
    LoadCandidateFit(candidates, 0, candidate);
    candidate.Chi2() = 4.f;
    candidate.NDF() = 2;

    const float mass = KFParticleGpuMath::Mass(candidate);
    assert(KFParticleGpuSelection::PassMassWindow(candidate, mass, 1.e-5f));
    assert(!KFParticleGpuSelection::PassMassWindow(candidate, mass + 0.1f, 1.e-5f));
    assert(AlmostEqual(KFParticleGpuSelection::MassWindowHalfWidth(0.01f, 3.f), 0.03f));
    assert(KFParticleGpuSelection::MassWindowHalfWidth(-1.f, 3.f) < 0.f);
    assert(KFParticleGpuSelection::PassSigmaMassWindow(candidate, mass, 0.01f, 3.f));
    assert(!KFParticleGpuSelection::PassSigmaMassWindow(candidate, mass + 0.1f, 0.01f, 3.f));
    assert(KFParticleGpuSelection::PassChi2PerNdf(candidate, 2.1f));
    assert(!KFParticleGpuSelection::PassChi2PerNdf(candidate, 1.9f));

    KFParticleGpuTwoDaughterTaskSource source;
    source.motherMass = mass + 1.f;
    source.motherMassSigma = 0.f;
    source.secondaryMassSigmaCut = -1.f;
    source.maxSecondaryTopoChi2PerNdf = -1.f;
    assert(PassTwoDaughterPostBuildSelection(source, candidate));
    source.motherMass = mass;
    source.motherMassSigma = 0.01f;
    source.secondaryMassSigmaCut = 3.f;
    source.maxSecondaryTopoChi2PerNdf = 2.1f;
    assert(PassTwoDaughterPostBuildSelection(source, candidate));
    source.motherMass = mass + 0.1f;
    assert(!PassTwoDaughterPostBuildSelection(source, candidate));

    mutableCandidates.Daughters().SourceId(1) = mutableCandidates.Daughters().SourceId(0);
    assert(!KFParticleGpuSelection::HasDistinctDaughterSourceIds(MakeConstView(mutableCandidates), 0));
    Pass("selection-helpers", "mass window, chi2/ndf, post-build selection, and duplicate daughter-source predicates");
  }

  void TestRoundTrip(KFParticleGpuSteering& steering)
  {
    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    KFParticleGpuEventDesc* events = buffers.HostEvents();

    const float pionMass = 0.13957f;
    steering.RunRoundTrip(pionMass, 0);

    const KFParticleGpuConstCandidatePoolView hostCandidates =
      MakeConstView(buffers.HostCandidates());
    KFParticleGpuFitState loaded;
    LoadCandidateFit(hostCandidates, 0, loaded);
    assert(loaded.X() == 1.f);
    assert(loaded.Px() == 2.f);
    assert(AlmostEqual(loaded.E(), std::sqrt(29.f + pionMass * pionMass)));
    assert(loaded.Q() == 1);
    assert(loaded.NDF() == 0);
    assert(loaded.MassHypo() == pionMass);
    assert(hostCandidates.Size() == 2);
    assert(hostCandidates.Daughters().Size() == 2);
    assert(hostCandidates.OverflowFlags() == 0);
    assert(hostCandidates.Metadata().Pdg(0) == 211);
    assert(hostCandidates.Metadata().PrimaryVertexIndex(0) == -1);
    assert(hostCandidates.Metadata().EventIndex(0) == 0);
    assert(hostCandidates.Metadata().DaughterOffset(0) == 0);
    assert(hostCandidates.Metadata().DaughterCount(0) == 1);
    assert(hostCandidates.Daughters().SourceId(0) == 17);
    assert(hostCandidates.Daughters().SourceId(1) == 19);
    Pass("round-trip", "input tracks converted by the KFParticle GPU smoke kernel into candidate pool entries");

    buffers.SetInputSizes(8, 1, 1);
    events[0].TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(0, 8);
    steering.RunRoundTrip(pionMass, 0);

    const KFParticleGpuConstCandidatePoolView overflowCandidates =
      MakeConstView(buffers.HostCandidates());
    const unsigned int expectedOverflowSize =
      buffers.TrackSize() < overflowCandidates.Capacity()
        ? buffers.TrackSize()
        : overflowCandidates.Capacity();
    const unsigned int expectedOutputSize =
      expectedOverflowSize < overflowCandidates.Daughters().Capacity()
        ? expectedOverflowSize
        : overflowCandidates.Daughters().Capacity();
    unsigned int expectedOverflowFlags = 0;
    if (buffers.TrackSize() > overflowCandidates.Capacity()) {
      expectedOverflowFlags |= CandidateCapacityExceeded;
    }
    if (buffers.TrackSize() > overflowCandidates.Daughters().Capacity()) {
      expectedOverflowFlags |= DaughterCapacityExceeded;
    }
    assert(overflowCandidates.Size() == expectedOutputSize);
    assert(overflowCandidates.Daughters().Size() == expectedOutputSize);
    assert(overflowCandidates.OverflowFlags() == expectedOverflowFlags);
    assert(overflowCandidates.Metadata().Pdg(expectedOutputSize - 1u)
           == ((expectedOutputSize - 1u) % 2u == 0u ? 211 : -211));
    assert(overflowCandidates.Daughters().SourceId(expectedOutputSize - 1u)
           == 17 + static_cast<int>(2u * (expectedOutputSize - 1u)));
    Pass("candidate-overflow", "candidate and daughter pool capacity limits set the expected overflow flags");
  }

  void TestGuardsAndEmptyRun(KFParticleGpuSteering& steering)
  {
    KFParticleGpuBufferManager& buffers = steering.GetBuffers();
    const float pionMass = 0.13957f;

    bool invalidMassRejected = false;
    try {
      steering.RunRoundTrip(-1.f, 0);
    }
    catch (const std::invalid_argument&) {
      invalidMassRejected = true;
    }
    assert(invalidMassRejected);

    bool invalidEventRejected = false;
    try {
      steering.RunRoundTrip(pionMass, 1);
    }
    catch (const std::out_of_range&) {
      invalidEventRejected = true;
    }
    assert(invalidEventRejected);

    buffers.SetInputSizes(0, 0, 0);
    steering.RunRoundTrip(pionMass, 0);
    assert(buffers.HostCandidates().Size() == 0);
    assert(buffers.HostCandidates().Daughters().Size() == 0);
    Pass("guards-empty-run", "invalid host inputs are rejected and empty events leave empty candidate pools");
  }

  void TestCapacityPolicy(KFParticleGpuBufferManager& buffers)
  {
    const KFParticleGpuBufferCapacities oldCapacities = buffers.Capacities();
    float* previousParameters = buffers.HostInputTracks().Numerical().ParametersData();
    KFParticleGpuBufferCapacities smaller = oldCapacities;
    smaller.tracks = 4;
    buffers.EnsureCapacity(smaller);
    assert(buffers.HostInputTracks().Numerical().ParametersData() == previousParameters);

    KFParticleGpuBufferCapacities larger = oldCapacities;
    larger.tracks = 12;
    larger.nonhomogeneousField = false;
    buffers.EnsureCapacity(larger);
    assert(buffers.Capacities().tracks == 12);
    assert(buffers.Capacities().nonhomogeneousField);
    Pass("capacity-policy", "buffers reuse smaller requests and grow monotonically without dropping field storage");
  }
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

  xpu::preload<KFParticleGpuXpuBaselineImage>();

  xpu::buffer<unsigned int> baselineMarker(1, xpu::buf_device);
  unsigned int baselineMarkerHost = 0;
  runtime.GetQueue().memcpy(baselineMarker.get(), &baselineMarkerHost, sizeof(baselineMarkerHost));

  runtime.GetQueue().launch<KFParticleGpuXpuBaselineMarker>(
    xpu::n_threads(1), baselineMarker.get());
  runtime.GetQueue().wait();
  runtime.GetQueue().memcpy(&baselineMarkerHost, baselineMarker.get(), sizeof(baselineMarkerHost));
  assert(baselineMarkerHost == 0x58505542u);
  Pass("baseline-image", "independent test image kernel launches and writes a device marker");

  xpu::buffer<unsigned int> launchMarker(1, xpu::buf_device);
  unsigned int launchMarkerHost = 0;
  runtime.GetQueue().memcpy(launchMarker.get(), &launchMarkerHost, sizeof(launchMarkerHost));

  runtime.GetQueue().launch<KFParticleGpuLaunchSmoke>(
    xpu::n_threads(1), launchMarker.get());
  runtime.GetQueue().wait();
  runtime.GetQueue().memcpy(&launchMarkerHost, launchMarker.get(), sizeof(launchMarkerHost));
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
  TestKernelStateContract(buffers);
  TestVisibleDeviceStorage(buffers);
  TestPublishedKernelState(runtime, buffers);
  TestDecayPlanSelectedOutputTruncation(runtime, buffers);
  TestDecayPlanDataTypes();
  TestV0SelectionResultDataTypes(runtime);
  TestPrimaryVertexTopologyObservables();
  TestV0SelectionDecision();
  TestSelectedCandidateOutput(runtime, buffers);
  TestDecayPlanChannelList();
  TestDefaultV0DecayPlanBuilders();
  TestKinematicMathSeed();
  TestMeasurementSeed();
  TestFieldMeasurementSeedApprox();
  TestFieldEnergyFit();
  TestFieldTransportPrimitive();
  TestFieldDcaKinematicSeed();
  TestFieldDcaReferenceComparison();
  TestFieldTransportProbe(runtime);
  TestKalmanUpdateProbe(runtime);
  TestTwoDaughterTaskKernel(runtime, buffers);
  TestTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterTaskGeneration(runtime, buffers);
  TestCompactTwoDaughterCandidateConstruction(runtime, buffers);
  TestCompactTwoDaughterCandidateOverflow(runtime, buffers);
  TestGeneratedTwoDaughterPipeline(runtime, buffers);
  TestSteeringTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterStage(runtime, buffers);
  TestSteeringCompactTwoDaughterEdgeCases(runtime, buffers);
  TestDecayPlanOneChannelExecutor(runtime, buffers);
  TestDecayPlanFieldAwareDiagnosticExecutor(runtime, buffers);
  TestDecayPlanMixedTransportExecutor(runtime, buffers);
  TestDecayPlanDefaultK0Executor(runtime, buffers);
  TestDecayPlanDefaultK0FieldAwareFixture(runtime, buffers);
  TestDecayPlanDefaultLambdaFieldAwareFixture(runtime, buffers);
  TestDecayPlanDefaultAntiLambdaFieldAwareFixture(runtime, buffers);
  TestDefaultV0FieldAwareScalarReference(runtime, buffers);
  TestDefaultV0FieldAwareEnergyFit(runtime, buffers);
  ProfileDefaultV0FieldAwareEnergyFit(runtime, buffers);
  TestDecayPlanDefaultV0Executor(runtime, buffers);
  TestDecayPlanDefaultV0SelectionRegression(runtime, buffers);
  TestDecayPlanSelectionMultiPvBoundary(runtime, buffers);
  TestDecayPlanMultiChannelExecutor(runtime, buffers);
  TestDecayPlanExecutorEdgeCases(runtime, buffers);
  TestDecayPlanBatchExecutor(runtime, buffers);
  TestDecayPlanBatchSelectedOutput(runtime, buffers);
  TestTwoDaughterSyntheticGrid(runtime, buffers);
  TestSelectionHelpers(buffers);
  FillInput(buffers, capacities);
  TestRoundTrip(runtime.GetSteering());
  TestGuardsAndEmptyRun(runtime.GetSteering());
  TestCapacityPolicy(buffers);

  runtime.Finalize();
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

  runtime.Finalize();
  Pass("reattach-finalize", "runtime finalizes cleanly and can reattach without owning XPU initialization");
  return 0;
}
