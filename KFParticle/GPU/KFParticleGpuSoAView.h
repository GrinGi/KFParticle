/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUSOAVIEW_H
#define KFPARTICLEGPUSOAVIEW_H

#include "KFParticleGpuFitState.h"

/**
 * Non-owning component-major view of input track numerical data.
 *
 * A component starts at component * stride. Keeping stride independent from
 * size allows owning buffers to retain capacity between events.
 */
template<typename Value>
class KFParticleGpuTrackSoAViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase(Value* parameters,
                                                           Value* covariances,
                                                           unsigned int size,
                                                           unsigned int stride)
    : fParameters(parameters), fCovariances(covariances), fSize(size), fStride(stride)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE Value& Parameter(unsigned int component, unsigned int particle) const
  {
    return fParameters[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE Value& Covariance(unsigned int component, unsigned int particle) const
  {
    return fCovariances[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE Value* ParametersData() const { return fParameters; }
  KFPARTICLE_GPU_HOST_DEVICE Value* CovariancesData() const { return fCovariances; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  Value* fParameters;
  Value* fCovariances;
  unsigned int fSize;
  unsigned int fStride;
};

typedef KFParticleGpuTrackSoAViewBase<float> KFParticleGpuTrackSoAView;
typedef KFParticleGpuTrackSoAViewBase<const float> KFParticleGpuConstTrackSoAView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstTrackSoAView
MakeConstView(const KFParticleGpuTrackSoAView& view)
{
  return KFParticleGpuConstTrackSoAView(
    view.ParametersData(), view.CovariancesData(), view.Size(), view.Stride());
}

struct KFParticleGpuFitSoALayout
{
  enum FloatComponent
  {
    Chi2 = 0,
    SFromDecay,
    SumDaughterMass,
    MassHypothesis,
    NumberOfFloatComponents
  };

  enum IntegerComponent
  {
    Ndf = 0,
    Charge,
    AtProductionVertex,
    ConstructMethod,
    NumberOfIntegerComponents
  };
};

/**
 * Non-owning component-major view of persistent KFParticle fit data.
 *
 * Fit scalars and integers are separate arrays so kernels can load only the
 * categories required by a reconstruction stage.
 */
template<typename FloatValue, typename IntegerValue>
class KFParticleGpuFitSoAViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitSoAViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuFitSoAViewBase(FloatValue* parameters,
                                                         FloatValue* covariances,
                                                         FloatValue* fitScalars,
                                                         IntegerValue* fitIntegers,
                                                         unsigned int size,
                                                         unsigned int stride)
    : fParameters(parameters)
    , fCovariances(covariances)
    , fFitScalars(fitScalars)
    , fFitIntegers(fitIntegers)
    , fSize(size)
    , fStride(stride)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Parameter(unsigned int component, unsigned int particle) const
  {
    return fParameters[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Covariance(unsigned int component, unsigned int particle) const
  {
    return fCovariances[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& FitScalar(unsigned int component, unsigned int particle) const
  {
    return fFitScalars[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& FitInteger(unsigned int component, unsigned int particle) const
  {
    return fFitIntegers[component * fStride + particle];
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ParametersData() const { return fParameters; }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* CovariancesData() const { return fCovariances; }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* FitScalarsData() const { return fFitScalars; }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* FitIntegersData() const { return fFitIntegers; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  FloatValue* fParameters;
  FloatValue* fCovariances;
  FloatValue* fFitScalars;
  IntegerValue* fFitIntegers;
  unsigned int fSize;
  unsigned int fStride;
};

typedef KFParticleGpuFitSoAViewBase<float, int> KFParticleGpuFitSoAView;
typedef KFParticleGpuFitSoAViewBase<const float, const int> KFParticleGpuConstFitSoAView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstFitSoAView
MakeConstView(const KFParticleGpuFitSoAView& view)
{
  return KFParticleGpuConstFitSoAView(view.ParametersData(),
                                      view.CovariancesData(),
                                      view.FitScalarsData(),
                                      view.FitIntegersData(),
                                      view.Size(),
                                      view.Stride());
}

#endif
