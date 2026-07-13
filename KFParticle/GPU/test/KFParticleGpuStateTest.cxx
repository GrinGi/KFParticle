/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuField.h"
#include "KFParticleGpuFitState.h"

#include <cassert>
#include <cmath>
#include <type_traits>

namespace
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
}

int main()
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
