/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuStateTransfer.h"

#include <cassert>
#include <type_traits>
#include <vector>

namespace
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
}

int main()
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
