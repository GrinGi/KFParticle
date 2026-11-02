/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "data/KFParticleGpuFitState.h"
#include "data/KFParticleGpuSoAView.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "diagnostics/KFParticleGpuParity.h"
#include "execution/KFParticleGpuKernelState.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

// Pin the layout of consolidated device-facing types, not merely their size.
template<typename T, std::size_t Size, std::size_t Alignment>
constexpr bool StableLayout = sizeof(T) == Size && alignof(T) == Alignment
                              && std::is_trivially_copyable<T>::value
                              && std::is_standard_layout<T>::value;
static_assert(StableLayout<KFParticleGpuTrackState, 108, 4>, "track ABI");
static_assert(StableLayout<KFParticleGpuFitState, 208, 4>, "fit ABI");
static_assert(StableLayout<KFParticleGpuVertexState, 48, 4>, "vertex ABI");
static_assert(StableLayout<KFParticleGpuFieldValue, 12, 4>, "field value ABI");
static_assert(StableLayout<KFParticleGpuFieldRegion, 40, 4>, "field region ABI");
static_assert(StableLayout<KFParticleGpuMeasurement, 212, 4>, "measurement ABI");
static_assert(StableLayout<KFParticleGpuRange, 8, 4>, "range ABI");
static_assert(StableLayout<KFParticleGpuTrackSetDesc, 112, 4>, "track-set ABI");
static_assert(StableLayout<KFParticleGpuEventDesc, 912, 4>, "event ABI");
static_assert(sizeof(void*) != 8 || StableLayout<KFParticleGpuConstInputTrackSoAView, 48, 8>,
              "input view ABI");
static_assert(sizeof(void*) != 8 || StableLayout<KFParticleGpuConstVertexSoAView, 40, 8>,
              "vertex view ABI");
static_assert(sizeof(void*) != 8 || StableLayout<KFParticleGpuCandidatePoolView, 112, 8>,
              "candidate pool ABI");
static_assert(sizeof(void*) != 8 || StableLayout<KFParticleGpuKernelState, 912, 8>,
              "kernel state ABI");
static_assert(StableLayout<KFParticleGpuTransactionDiagnosticRecord, 216, 8>, "journal ABI");

namespace State
{
  bool AlmostEqual(float lhs, float rhs, float tolerance = 1.e-6f)
  {
    return std::fabs(lhs - rhs) <= tolerance;
  }

  void TestCovarianceIndex()
  {
    int index = 0;
    for (int row = 0; row < KFParticleGpuFitState::NumberOfParameters; ++row) {
      for (int column = 0; column <= row; ++column) {
        assert(KFParticleGpuFitState::CovarianceIndex(row, column) == index);
        assert(KFParticleGpuFitState::CovarianceIndex(column, row) == index);
        ++index;
      }
    }
    assert(index == KFParticleGpuFitState::NumberOfCovarianceElements);
  }

  void TestDefaultFitState()
  {
    const KFParticleGpuFitState state;
    assert(state.NDF() == -3);
    assert(state.Q() == 0);
    assert(state.MassHypo() == -1.f);
    assert(state.Covariance(0) == 100.f);
    assert(state.Covariance(2) == 100.f);
    assert(state.Covariance(5) == 100.f);
    assert(state.Covariance(35) == 1.f);
  }

  void TestTrackInitialization()
  {
    const float parameters[6] = {1.f, 2.f, 3.f, 3.f, 0.f, 0.f};
    float covariance[21] = {};
    covariance[9] = 1.f;
    covariance[14] = 4.f;
    covariance[20] = 9.f;

    KFParticleGpuTrackState track;
    track.Initialize(parameters, covariance);

    KFParticleGpuFitState state;
    state.Initialize(track, 1, 4.f);

    assert(state.X() == 1.f);
    assert(state.Px() == 3.f);
    assert(state.E() == 5.f);
    assert(state.S() == 0.f);
    assert(state.Q() == 1);
    assert(state.NDF() == 0);
    assert(state.SumDaughterMass() == 4.f);
    assert(state.MassHypo() == 4.f);
    assert(AlmostEqual(state.Covariance(24), 0.6f));
    assert(AlmostEqual(state.Covariance(27), 0.36f));
    assert(state.Covariance(35) == 1.f);
  }

  void TestZeroEnergyTrack()
  {
    const float parameters[6] = {};
    const float covariance[21] = {};

    KFParticleGpuTrackState track;
    track.Initialize(parameters, covariance);

    KFParticleGpuFitState state;
    state.Initialize(track, 0, 0.f);
    assert(std::isfinite(state.E()));
    assert(std::isfinite(state.Covariance(27)));
  }

  void TestFieldRegion()
  {
    const float coefficients[10] = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
      10.f
    };
    const KFParticleGpuFieldRegion field(coefficients);
    const KFParticleGpuFieldValue value = field.Get(12.f);

    assert(value.x == 17.f);
    assert(value.y == 38.f);
    assert(value.z == 59.f);

    KFParticleGpuFieldValue combined(1.f, 2.f, 3.f);
    combined.Combine(KFParticleGpuFieldValue(3.f, 6.f, 9.f), 0.5f);
    assert(combined.x == 2.f);
    assert(combined.y == 4.f);
    assert(combined.z == 6.f);
  }

int Run()
{
  static_assert(std::is_trivially_copyable<KFParticleGpuTrackState>::value,
                "GPU track state must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuFitState>::value,
                "GPU fit state must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuFieldRegion>::value,
                "GPU field region must be trivially copyable");
  static_assert(sizeof(KFParticleGpuTrackState) == 108, "Unexpected GPU track state layout");
  static_assert(sizeof(KFParticleGpuFitState) == 208, "Unexpected GPU fit state layout");
  static_assert(sizeof(KFParticleGpuFieldRegion) == 40, "Unexpected GPU field region layout");

  TestCovarianceIndex();
  TestDefaultFitState();
  TestTrackInitialization();
  TestZeroEnergyTrack();
  TestFieldRegion();
  return 0;
}
}

namespace SoAView
{
  const float FloatPadding = -1001.f;
  const int IntegerPadding = -1001;

  void FillTrackState(KFParticleGpuTrackState& state, float offset)
  {
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
      state.Parameter(component) = offset + component;
    }
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfCovarianceElements; ++component) {
      state.Covariance(component) = offset + 100.f + component;
    }
  }

  void AssertTrackState(const KFParticleGpuTrackState& state, float offset)
  {
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
      assert(state.Parameter(component) == offset + component);
    }
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfCovarianceElements; ++component) {
      assert(state.Covariance(component) == offset + 100.f + component);
    }
  }

  void TestTrackRoundTrip()
  {
    const unsigned int size = 2;
    const unsigned int stride = 5;
    std::vector<float> parameters(KFParticleGpuTrackState::NumberOfParameters * stride, FloatPadding);
    std::vector<float> covariances(
      KFParticleGpuTrackState::NumberOfCovarianceElements * stride, FloatPadding);

    KFParticleGpuTrackSoAView view(&parameters[0], &covariances[0], size, stride);
    KFParticleGpuTrackState first;
    KFParticleGpuTrackState second;
    FillTrackState(first, 10.f);
    FillTrackState(second, 20.f);
    StoreTrackState(first, view, 0);
    StoreTrackState(second, view, 1);

    // Equal components of adjacent particles are adjacent in memory.
    assert(parameters[0 * stride + 0] == 10.f);
    assert(parameters[0 * stride + 1] == 20.f);
    assert(parameters[1 * stride + 0] == 11.f);

    KFParticleGpuTrackState loaded;
    LoadTrackState(MakeConstView(view), 1, loaded);
    AssertTrackState(loaded, 20.f);

    for (int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
      for (unsigned int particle = size; particle < stride; ++particle) {
        assert(parameters[component * stride + particle] == FloatPadding);
      }
    }
  }

  void FillFitState(KFParticleGpuFitState& state, float offset)
  {
    for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
      state.Parameter(component) = offset + component;
    }
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      state.Covariance(component) = offset + 100.f + component;
    }
    state.Chi2() = offset + 200.f;
    state.SFromDecay() = offset + 201.f;
    state.SumDaughterMass() = offset + 202.f;
    state.MassHypo() = offset + 203.f;
    state.NDF() = static_cast<int>(offset) + 300;
    state.Q() = static_cast<int>(offset) + 301;
    state.AtProductionVertex() = static_cast<int>(offset) + 302;
    state.ConstructMethod() = static_cast<int>(offset) + 303;
  }

  void AssertFitState(const KFParticleGpuFitState& state, float offset)
  {
    for (int component = 0; component < KFParticleGpuFitState::NumberOfParameters; ++component) {
      assert(state.Parameter(component) == offset + component);
    }
    for (int component = 0; component < KFParticleGpuFitState::NumberOfCovarianceElements; ++component) {
      assert(state.Covariance(component) == offset + 100.f + component);
    }
    assert(state.Chi2() == offset + 200.f);
    assert(state.SFromDecay() == offset + 201.f);
    assert(state.SumDaughterMass() == offset + 202.f);
    assert(state.MassHypo() == offset + 203.f);
    assert(state.NDF() == static_cast<int>(offset) + 300);
    assert(state.Q() == static_cast<int>(offset) + 301);
    assert(state.AtProductionVertex() == static_cast<int>(offset) + 302);
    assert(state.ConstructMethod() == static_cast<int>(offset) + 303);
  }

  void TestFitRoundTrip()
  {
    const unsigned int size = 2;
    const unsigned int stride = 4;
    std::vector<float> parameters(KFParticleGpuFitState::NumberOfParameters * stride, FloatPadding);
    std::vector<float> covariances(
      KFParticleGpuFitState::NumberOfCovarianceElements * stride, FloatPadding);
    std::vector<float> fitScalars(
      KFParticleGpuFitSoALayout::NumberOfFloatComponents * stride, FloatPadding);
    std::vector<int> fitIntegers(
      KFParticleGpuFitSoALayout::NumberOfIntegerComponents * stride, IntegerPadding);

    KFParticleGpuFitSoAView view(
      &parameters[0], &covariances[0], &fitScalars[0], &fitIntegers[0], size, stride);
    KFParticleGpuFitState first;
    KFParticleGpuFitState second;
    FillFitState(first, 30.f);
    FillFitState(second, 40.f);
    StoreFitState(first, view, 0);
    StoreFitState(second, view, 1);

    KFParticleGpuFitState loaded;
    LoadFitState(MakeConstView(view), 0, loaded);
    AssertFitState(loaded, 30.f);

    for (int component = 0; component < KFParticleGpuFitSoALayout::NumberOfFloatComponents; ++component) {
      for (unsigned int particle = size; particle < stride; ++particle) {
        assert(fitScalars[component * stride + particle] == FloatPadding);
      }
    }
    for (int component = 0; component < KFParticleGpuFitSoALayout::NumberOfIntegerComponents; ++component) {
      for (unsigned int particle = size; particle < stride; ++particle) {
        assert(fitIntegers[component * stride + particle] == IntegerPadding);
      }
    }
  }

int Run()
{
  static_assert(std::is_trivially_copyable<KFParticleGpuTrackSoAView>::value,
                "Mutable track view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuConstTrackSoAView>::value,
                "Const track view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuFitSoAView>::value,
                "Mutable fit view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuConstFitSoAView>::value,
                "Const fit view must be trivially copyable");

  TestTrackRoundTrip();
  TestFitRoundTrip();
  return 0;
}
}

namespace InputData
{
  void TestPrimaryVertexIndices()
  {
    int packed = 77;
    assert(KFParticleGpuPackPrimaryVertexIndex(1, {5u, 2u}, packed));
    assert(packed == 6);
    for (int unassigned : {-1, -7}) {
      assert(KFParticleGpuPackPrimaryVertexIndex(unassigned, {5u, 0u}, packed));
      assert(packed == unassigned);
    }
    packed = 77;
    assert(!KFParticleGpuPackPrimaryVertexIndex(2, {5u, 2u}, packed));
    assert(!KFParticleGpuPackPrimaryVertexIndex(0, {0u, 0u}, packed));
    assert(!KFParticleGpuPackPrimaryVertexIndex(1, {static_cast<unsigned int>(INT_MAX), 2u}, packed));
    assert(!KFParticleGpuPackPrimaryVertexIndex(0, {static_cast<unsigned int>(INT_MAX) + 1u, 1u}, packed));
    assert(packed == 77);
    assert(KFParticleGpuPackPrimaryVertexIndex(1, {static_cast<unsigned int>(INT_MAX) - 1u, 2u}, packed));
    assert(packed == INT_MAX);
    assert(KFParticleGpuIsEventPrimaryVertex(5, {5u, 2u}, 7u));
    assert(KFParticleGpuIsEventPrimaryVertex(6, {5u, 2u}, 7u));
    assert(!KFParticleGpuIsEventPrimaryVertex(0, {1u, 1u}, 2u));
    assert(!KFParticleGpuIsEventPrimaryVertex(-1, {0u, 1u}, 1u));
    assert(!KFParticleGpuIsEventPrimaryVertex(7, {5u, 2u}, 7u));
    assert(!KFParticleGpuIsEventPrimaryVertex(0, {0u, 0u}, 0u));
    assert(!KFParticleGpuIsEventPrimaryVertex(5, {5u, 3u}, 7u));
    assert(!KFParticleGpuIsEventPrimaryVertex(0, {UINT_MAX - 1u, 4u}, UINT_MAX));
  }

  void TestInputTrackView()
  {
    const unsigned int size = 3;
    const unsigned int stride = 5;
    std::vector<float> parameters(KFParticleGpuTrackState::NumberOfParameters * stride, -1.f);
    std::vector<float> covariances(KFParticleGpuTrackState::NumberOfCovarianceElements * stride, -1.f);
    std::vector<float> field(KFParticleGpuFieldRegion::NumberOfCoefficients * stride, -1.f);
    std::vector<float> chiToPrimaryVertex(stride, -1.f);
    std::vector<int> integers(KFParticleGpuTrackInputLayout::NumberOfIntegerComponents * stride, -1);

    KFParticleGpuInputTrackSoAView view(&parameters[0],
                                        &covariances[0],
                                        &field[0],
                                        &chiToPrimaryVertex[0],
                                        &integers[0],
                                        size,
                                        stride);

    KFParticleGpuTrackState track;
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfParameters; ++component) {
      track.Parameter(component) = 10.f + component;
    }
    for (int component = 0; component < KFParticleGpuTrackState::NumberOfCovarianceElements; ++component) {
      track.Covariance(component) = 100.f + component;
    }
    KFParticleGpuFieldRegion fieldRegion;
    for (int component = 0; component < KFParticleGpuFieldRegion::NumberOfCoefficients; ++component) {
      fieldRegion.Coefficient(component) = 200.f + component;
    }

    StoreTrackState(track, view, 1);
    StoreFieldRegion(fieldRegion, view, 1);
    view.ChiToPrimaryVertex(1) = 17.f;
    view.SourceId(1) = 42;
    view.Pdg(1) = 211;
    view.Charge(1) = 1;
    view.PrimaryVertexIndex(1) = -1;
    view.NumberOfPixelHits(1) = 4;

    const KFParticleGpuConstInputTrackSoAView constView = MakeConstView(view);
    KFParticleGpuTrackState loadedTrack;
    KFParticleGpuFieldRegion loadedField;
    LoadTrackState(constView, 1, loadedTrack);
    LoadFieldRegion(constView, 1, loadedField);

    assert(loadedTrack.Px() == 13.f);
    assert(loadedTrack.Covariance(20) == 120.f);
    assert(loadedField.Coefficient(9) == 209.f);
    assert(constView.ChiToPrimaryVertex(1) == 17.f);
    assert(constView.SourceId(1) == 42);
    assert(constView.Pdg(1) == 211);
    assert(constView.Charge(1) == 1);
    assert(constView.PrimaryVertexIndex(1) == -1);
    assert(constView.NumberOfPixelHits(1) == 4);

    assert(parameters[3 * stride + 1] == 13.f);
    assert(field[9 * stride + 1] == 209.f);
    assert(integers[KFParticleGpuTrackInputLayout::Pdg * stride + 1] == 211);
  }

  void TestEventDescriptor()
  {
    KFParticleGpuEventDesc event;
    event.eventId = 7;
    event.primaryVertices = KFParticleGpuRange(90, 2);
    event.TrackSet(SecondaryPositiveFirst).tracks = KFParticleGpuRange(10, 12);
    event.TrackSet(SecondaryPositiveFirst).Species(Pion) = KFParticleGpuRange(13, 5);

    assert(event.eventId == 7);
    assert(event.primaryVertices.Contains(91));
    assert(!event.primaryVertices.Contains(92));
    assert(event.TrackSet(SecondaryPositiveFirst).tracks.End() == 22);
    assert(event.TrackSet(SecondaryPositiveFirst).Species(Pion).Contains(15));
  }

  void TestVertexRoundTrip()
  {
    const unsigned int size = 2;
    const unsigned int stride = 4;
    std::vector<float> parameters(KFParticleGpuVertexState::NumberOfParameters * stride, -1.f);
    std::vector<float> covariances(KFParticleGpuVertexState::NumberOfCovarianceElements * stride, -1.f);
    std::vector<float> chi2(stride, -1.f);
    std::vector<int> integers(KFParticleGpuVertexSoALayout::NumberOfIntegerComponents * stride, -1);

    KFParticleGpuVertexSoAView view(
      &parameters[0], &covariances[0], &chi2[0], &integers[0], size, stride);
    const float inputParameters[3] = {1.f, 2.f, 3.f};
    const float inputCovariance[6] = {4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    KFParticleGpuVertexState source;
    source.Initialize(inputParameters, inputCovariance, 10.f, 11, 12);
    StoreVertexState(source, view, 1);

    KFParticleGpuVertexState loaded;
    LoadVertexState(MakeConstView(view), 1, loaded);
    assert(loaded.X() == 1.f);
    assert(loaded.Z() == 3.f);
    assert(loaded.Covariance(5) == 9.f);
    assert(loaded.Chi2() == 10.f);
    assert(loaded.NDF() == 11);
    assert(loaded.NContributors() == 12);
    assert(parameters[2 * stride + 1] == 3.f);
  }

int Run()
{
  static_assert(std::is_trivially_copyable<KFParticleGpuRange>::value,
                "GPU range must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuEventDesc>::value,
                "GPU event descriptor must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuInputTrackSoAView>::value,
                "GPU input track view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuVertexSoAView>::value,
                "GPU vertex view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuVertexState>::value,
                "GPU vertex state must be trivially copyable");

  TestInputTrackView();
  TestPrimaryVertexIndices();
  TestEventDescriptor();
  TestVertexRoundTrip();
  return 0;
}
}

namespace CandidatePool
{
  void TestCandidatePool()
  {
    const unsigned int capacity = 4;
    std::vector<float> parameters(KFParticleGpuFitState::NumberOfParameters * capacity, -1.f);
    std::vector<float> covariances(KFParticleGpuFitState::NumberOfCovarianceElements * capacity, -1.f);
    std::vector<float> fitScalars(KFParticleGpuFitSoALayout::NumberOfFloatComponents * capacity, -1.f);
    std::vector<int> fitIntegers(KFParticleGpuFitSoALayout::NumberOfIntegerComponents * capacity, -1);
    std::vector<int> metadataIntegers(
      KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents * capacity, -1);
    std::vector<unsigned int> metadataUnsigned(
      KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents * capacity, 0);

    const unsigned int daughterCapacity = 12;
    std::vector<int> daughterSourceIds(daughterCapacity, -1);
    unsigned int candidateSize = 0;
    unsigned int daughterSize = 0;
    unsigned int overflowFlags = 0;

    KFParticleGpuFitSoAView fit(&parameters[0],
                                &covariances[0],
                                &fitScalars[0],
                                &fitIntegers[0],
                                capacity,
                                capacity);
    KFParticleGpuCandidateMetadataSoAView metadata(
      &metadataIntegers[0], &metadataUnsigned[0], capacity);
    KFParticleGpuDaughterStorageView daughters(
      &daughterSourceIds[0], &daughterSize, daughterCapacity);
    KFParticleGpuCandidatePoolView pool(
      fit, metadata, daughters, &candidateSize, &overflowFlags, capacity);

    KFParticleGpuFitState state;
    state.X() = 1.f;
    state.Px() = 2.f;
    state.Chi2() = 3.f;
    state.NDF() = 4;
    assert(StoreCandidateFit(state, pool, 1));

    pool.Metadata().Pdg(1) = 3122;
    pool.Metadata().PrimaryVertexIndex(1) = -1;
    pool.Metadata().EventIndex(1) = 7;
    pool.Metadata().DaughterOffset(1) = 3;
    pool.Metadata().DaughterCount(1) = 2;
    pool.Metadata().Flags(1) = 0;
    pool.Metadata().ChannelId(1) = 17u;
    pool.Metadata().Topology(1) = KFGpuGraphTopologyCompositeTrack;
    pool.Metadata().OutputClass(1) = KFGpuGraphOutputSecondary;
    pool.Metadata().OperationStatus(1) = KFGpuCandidateOperationAccepted;
    pool.Metadata().DirectDaughterCount(1) = 2u;
    SetCandidateDirectDaughter(
      pool.Metadata(), 1u, 0u, KFGpuDirectDaughterCandidate, 0u);
    SetCandidateDirectDaughter(
      pool.Metadata(), 1u, 1u, KFGpuDirectDaughterInputTrack, 3u);
    pool.Daughters().SourceId(3) = 101;
    pool.Daughters().SourceId(4) = 205;
    candidateSize = 2;
    daughterSize = 5;

    const KFParticleGpuConstCandidatePoolView constPool = MakeConstView(pool);
    KFParticleGpuFitState loaded;
    LoadCandidateFit(constPool, 1, loaded);

    assert(loaded.X() == 1.f);
    assert(loaded.Px() == 2.f);
    assert(loaded.Chi2() == 3.f);
    assert(loaded.NDF() == 4);
    assert(constPool.Size() == 2);
    assert(constPool.Metadata().Pdg(1) == 3122);
    assert(constPool.Metadata().PrimaryVertexIndex(1) == -1);
    assert(constPool.Metadata().EventIndex(1) == 7);
    assert(constPool.Metadata().DaughterOffset(1) == 3);
    assert(constPool.Metadata().DaughterCount(1) == 2);
    assert(constPool.Metadata().Topology(1) == KFGpuGraphTopologyCompositeTrack);
    assert(constPool.Metadata().OutputClass(1) == KFGpuGraphOutputSecondary);
    assert(constPool.Metadata().OperationStatus(1) == KFGpuCandidateOperationAccepted);
    assert(constPool.Metadata().DirectDaughterCount(1) == 2u);
    assert(constPool.Metadata().DirectFirstKind(1) == KFGpuDirectDaughterCandidate);
    assert(constPool.Metadata().DirectFirstIndex(1) == 0u);
    assert(constPool.Metadata().DirectSecondKind(1) == KFGpuDirectDaughterInputTrack);
    assert(constPool.Metadata().DirectSecondIndex(1) == 3u);
    assert(constPool.Daughters().SourceId(3) == 101);
    assert(constPool.Daughters().SourceId(4) == 205);
    assert(constPool.Daughters().Size() == 5);

    KFParticleGpuParitySnapshot snapshot;
    assert(KFParticleGpuParity::BuildSnapshot(
      constPool, 1u, 91u, nullptr, false, snapshot));
    assert(snapshot.key.eventId == 91u);
    assert(snapshot.key.channelId == 17u);
    assert(snapshot.key.lineageSize == 2u);
    assert(snapshot.key.lineage[0] == 101);
    assert(snapshot.key.lineage[1] == 205);
    assert(snapshot.topology == KFGpuGraphTopologyCompositeTrack);
    assert(snapshot.outputClass == KFGpuGraphOutputSecondary);
    assert(snapshot.operationStatus == KFGpuCandidateOperationAccepted);

    // Component-major metadata keeps adjacent candidates adjacent in memory.
    assert(metadataIntegers[KFParticleGpuCandidateMetadataLayout::Pdg * capacity + 1] == 3122);
    assert(metadataUnsigned[
             KFParticleGpuCandidateMetadataLayout::DaughterOffset * capacity + 1] == 3);

    assert(pool.CanStoreCandidate(capacity - 1));
    assert(!pool.CanStoreCandidate(capacity));
    assert(pool.Daughters().CanStore(10, 2));
    assert(!pool.Daughters().CanStore(11, 2));
    assert(!StoreCandidateFit(state, pool, capacity));

    overflowFlags = CandidateCapacityExceeded | DaughterCapacityExceeded;
    assert(constPool.OverflowFlags() ==
           static_cast<unsigned int>(CandidateCapacityExceeded | DaughterCapacityExceeded));
  }

int Run()
{
  static_assert(std::is_trivially_copyable<KFParticleGpuCandidateMetadataSoAView>::value,
                "Candidate metadata view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuDaughterStorageView>::value,
                "Daughter storage view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuCandidatePoolView>::value,
                "Candidate pool view must be trivially copyable");
  static_assert(std::is_trivially_copyable<KFParticleGpuConstCandidatePoolView>::value,
                "Const candidate pool view must be trivially copyable");

  TestCandidatePool();
  return 0;
}
}

int main(int argc, char** argv)
{
  const char* selected = argc == 1 ? "all" : argv[1];
  if (argc > 2) { return 2; }
  bool found = false;
  const struct { const char* name; int (*run)(); } cases[] = {
    {"State", State::Run}, {"SoAView", SoAView::Run},
    {"InputData", InputData::Run}, {"CandidatePool", CandidatePool::Run}
  };
  for (const auto& test : cases) {
    if (std::strcmp(selected, "all") == 0 || std::strcmp(selected, test.name) == 0) {
      if (test.run() != 0) { return 1; }
      std::cout << "PASS GPU data contract " << test.name << '\n';
      found = true;
    }
  }
  if (!found) { std::cerr << "Unknown data contract: " << selected << '\n'; }
  return found ? 0 : 2;
}
