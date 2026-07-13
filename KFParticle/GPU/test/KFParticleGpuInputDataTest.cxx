/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuInputDataTransfer.h"

#include <cassert>
#include <type_traits>
#include <vector>

namespace
{
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
}

int main()
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
  TestEventDescriptor();
  TestVertexRoundTrip();
  return 0;
}
