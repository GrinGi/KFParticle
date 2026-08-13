/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUINPUTDATA_H
#define KFPARTICLEGPUINPUTDATA_H

#include "KFParticleGpuField.h"
#include "KFParticleGpuSoAView.h"
#include "KFParticleGpuVertexState.h"

struct KFParticleGpuRange
{
  unsigned int offset;
  unsigned int size;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange() : offset(0), size(0) {}
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange(unsigned int rangeOffset, unsigned int rangeSize)
    : offset(rangeOffset), size(rangeSize)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }
  KFPARTICLE_GPU_HOST_DEVICE bool Contains(unsigned int index) const
  {
    return index >= offset && index < End();
  }
};

enum KFParticleGpuTrackSet
{
  SecondaryPositiveFirst = 0,
  SecondaryNegativeFirst,
  PrimaryPositiveFirst,
  PrimaryNegativeFirst,
  SecondaryPositiveLast,
  SecondaryNegativeLast,
  PrimaryPositiveLast,
  PrimaryNegativeLast,
  NumberOfTrackSets
};

enum KFParticleGpuTrackSpecies
{
  Electron = 0,
  Muon,
  Pion,
  Kaon,
  Proton,
  Deuteron,
  Triton,
  Helium3,
  Helium4,
  Helium6,
  Lithium6,
  Lithium7,
  Beryllium7,
  NumberOfTrackSpecies
};

KFPARTICLE_GPU_HOST_DEVICE inline unsigned int KFParticleGpuGraphTrackSourceId(
  KFParticleGpuTrackSet set,
  KFParticleGpuTrackSpecies species)
{
  return 1u + static_cast<unsigned int>(set)
    * (static_cast<unsigned int>(NumberOfTrackSpecies) + 1u)
    + static_cast<unsigned int>(species);
}

KFPARTICLE_GPU_HOST_DEVICE inline bool KFParticleGpuDecodeGraphTrackSourceId(
  unsigned int sourceId,
  KFParticleGpuTrackSet& set,
  KFParticleGpuTrackSpecies& species)
{
  if (sourceId == 0u) return false;
  const unsigned int value = sourceId - 1u;
  const unsigned int stride = static_cast<unsigned int>(NumberOfTrackSpecies) + 1u;
  const unsigned int setIndex = value / stride;
  const unsigned int speciesIndex = value % stride;
  if (setIndex >= static_cast<unsigned int>(NumberOfTrackSets)
      || speciesIndex > static_cast<unsigned int>(NumberOfTrackSpecies)) {
    return false;
  }
  set = static_cast<KFParticleGpuTrackSet>(setIndex);
  species = static_cast<KFParticleGpuTrackSpecies>(speciesIndex);
  return true;
}

/**
 * Absolute ranges of one CPU KFPTrackVector after packing into shared SoA.
 *
 * GPU ranges contain physical tracks only and do not include CPU SIMD padding.
 */
struct KFParticleGpuTrackSetDesc
{
  KFParticleGpuRange tracks;
  KFParticleGpuRange species[NumberOfTrackSpecies];

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuRange& Species(KFParticleGpuTrackSpecies value) const
  {
    return species[static_cast<int>(value)];
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuRange& Species(KFParticleGpuTrackSpecies value)
  {
    return species[static_cast<int>(value)];
  }
};

/** Offsets for one event in track and primary-vertex input buffers. */
struct KFParticleGpuEventDesc
{
  unsigned int eventId;
  KFParticleGpuTrackSetDesc trackSets[NumberOfTrackSets];
  KFParticleGpuRange primaryVertices;
  float minSecondaryTrackChiToPrimaryVertex;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuEventDesc()
    : eventId(0), trackSets(), primaryVertices(), minSecondaryTrackChiToPrimaryVertex(-1.f)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTrackSetDesc& TrackSet(KFParticleGpuTrackSet value) const
  {
    return trackSets[static_cast<int>(value)];
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSetDesc& TrackSet(KFParticleGpuTrackSet value)
  {
    return trackSets[static_cast<int>(value)];
  }
};

struct KFParticleGpuTrackInputLayout
{
  enum IntegerComponent
  {
    SourceId = 0,
    Pdg,
    Charge,
    PrimaryVertexIndex,
    NumberOfPixelHits,
    NumberOfIntegerComponents
  };
};

/**
 * Non-owning view of packed Finder tracks.
 *
 * The field pointer may be null for homogeneous-field builds. All other arrays
 * share the numerical view stride.
 */
template<typename FloatValue, typename IntegerValue>
class KFParticleGpuInputTrackSoAViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuInputTrackSoAViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuInputTrackSoAViewBase(
    FloatValue* parameters,
    FloatValue* covariances,
    FloatValue* fieldCoefficients,
    FloatValue* chiToPrimaryVertex,
    IntegerValue* integers,
    unsigned int size,
    unsigned int stride)
    : fNumerical(parameters, covariances, size, stride)
    , fFieldCoefficients(fieldCoefficients)
    , fChiToPrimaryVertex(chiToPrimaryVertex)
    , fIntegers(integers)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuTrackSoAViewBase<FloatValue>& Numerical()
  {
    return fNumerical;
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuTrackSoAViewBase<FloatValue>& Numerical() const
  {
    return fNumerical;
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& FieldCoefficient(unsigned int component,
                                                          unsigned int track) const
  {
    return fFieldCoefficients[component * Stride() + track];
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& ChiToPrimaryVertex(unsigned int track) const
  {
    return fChiToPrimaryVertex[track];
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int track) const
  {
    return fIntegers[component * Stride() + track];
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& SourceId(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::SourceId, track);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Pdg(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::Pdg, track);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Charge(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::Charge, track);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& PrimaryVertexIndex(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::PrimaryVertexIndex, track);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NumberOfPixelHits(unsigned int track) const
  {
    return Integer(KFParticleGpuTrackInputLayout::NumberOfPixelHits, track);
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue* FieldCoefficientsData() const { return fFieldCoefficients; }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ChiToPrimaryVertexData() const { return fChiToPrimaryVertex; }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fNumerical.Size(); }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fNumerical.Stride(); }

 private:
  KFParticleGpuTrackSoAViewBase<FloatValue> fNumerical;
  FloatValue* fFieldCoefficients;
  FloatValue* fChiToPrimaryVertex;
  IntegerValue* fIntegers;
};

typedef KFParticleGpuInputTrackSoAViewBase<float, int> KFParticleGpuInputTrackSoAView;
typedef KFParticleGpuInputTrackSoAViewBase<const float, const int> KFParticleGpuConstInputTrackSoAView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstInputTrackSoAView
MakeConstView(const KFParticleGpuInputTrackSoAView& view)
{
  return KFParticleGpuConstInputTrackSoAView(view.Numerical().ParametersData(),
                                             view.Numerical().CovariancesData(),
                                             view.FieldCoefficientsData(),
                                             view.ChiToPrimaryVertexData(),
                                             view.IntegersData(),
                                             view.Size(),
                                             view.Stride());
}

struct KFParticleGpuVertexSoALayout
{
  enum IntegerComponent
  {
    Ndf = 0,
    NumberOfContributors,
    NumberOfIntegerComponents
  };
};

template<typename FloatValue, typename IntegerValue>
class KFParticleGpuVertexSoAViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexSoAViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuVertexSoAViewBase(FloatValue* parameters,
                                                            FloatValue* covariances,
                                                            FloatValue* chi2,
                                                            IntegerValue* integers,
                                                            unsigned int size,
                                                            unsigned int stride)
    : fParameters(parameters)
    , fCovariances(covariances)
    , fChi2(chi2)
    , fIntegers(integers)
    , fSize(size)
    , fStride(stride)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Parameter(unsigned int component, unsigned int vertex) const
  {
    return fParameters[component * fStride + vertex];
  }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Covariance(unsigned int component, unsigned int vertex) const
  {
    return fCovariances[component * fStride + vertex];
  }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue& Chi2(unsigned int vertex) const { return fChi2[vertex]; }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int vertex) const
  {
    return fIntegers[component * fStride + vertex];
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NDF(unsigned int vertex) const
  {
    return Integer(KFParticleGpuVertexSoALayout::Ndf, vertex);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& NContributors(unsigned int vertex) const
  {
    return Integer(KFParticleGpuVertexSoALayout::NumberOfContributors, vertex);
  }

  KFPARTICLE_GPU_HOST_DEVICE FloatValue* ParametersData() const { return fParameters; }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* CovariancesData() const { return fCovariances; }
  KFPARTICLE_GPU_HOST_DEVICE FloatValue* Chi2Data() const { return fChi2; }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const { return fSize; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  FloatValue* fParameters;
  FloatValue* fCovariances;
  FloatValue* fChi2;
  IntegerValue* fIntegers;
  unsigned int fSize;
  unsigned int fStride;
};

typedef KFParticleGpuVertexSoAViewBase<float, int> KFParticleGpuVertexSoAView;
typedef KFParticleGpuVertexSoAViewBase<const float, const int> KFParticleGpuConstVertexSoAView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstVertexSoAView
MakeConstView(const KFParticleGpuVertexSoAView& view)
{
  return KFParticleGpuConstVertexSoAView(view.ParametersData(),
                                         view.CovariancesData(),
                                         view.Chi2Data(),
                                         view.IntegersData(),
                                         view.Size(),
                                         view.Stride());
}

#endif
