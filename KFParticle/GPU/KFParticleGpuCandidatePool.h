/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUCANDIDATEPOOL_H
#define KFPARTICLEGPUCANDIDATEPOOL_H

#include "KFParticleGpuSoAView.h"

struct KFParticleGpuCandidateMetadataLayout
{
  enum IntegerComponent
  {
    Pdg = 0,
    PrimaryVertexIndex,
    NumberOfIntegerComponents
  };

  enum UnsignedComponent
  {
    EventIndex = 0,
    DaughterOffset,
    DaughterCount,
    Flags,
    ChannelId,
    NumberOfUnsignedComponents
  };
};

template<typename IntegerValue, typename UnsignedValue>
class KFParticleGpuCandidateMetadataSoAViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateMetadataSoAViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidateMetadataSoAViewBase(IntegerValue* integers,
                                                                       UnsignedValue* unsignedValues,
                                                                       unsigned int stride)
    : fIntegers(integers), fUnsigned(unsignedValues), fStride(stride)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Integer(unsigned int component, unsigned int candidate) const
  {
    return fIntegers[component * fStride + candidate];
  }

  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Unsigned(unsigned int component, unsigned int candidate) const
  {
    return fUnsigned[component * fStride + candidate];
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& Pdg(unsigned int candidate) const
  {
    return Integer(KFParticleGpuCandidateMetadataLayout::Pdg, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& PrimaryVertexIndex(unsigned int candidate) const
  {
    return Integer(KFParticleGpuCandidateMetadataLayout::PrimaryVertexIndex, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& EventIndex(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::EventIndex, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DaughterOffset(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DaughterOffset, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& DaughterCount(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::DaughterCount, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& Flags(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::Flags, candidate);
  }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue& ChannelId(unsigned int candidate) const
  {
    return Unsigned(KFParticleGpuCandidateMetadataLayout::ChannelId, candidate);
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* IntegersData() const { return fIntegers; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* UnsignedData() const { return fUnsigned; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Stride() const { return fStride; }

 private:
  IntegerValue* fIntegers;
  UnsignedValue* fUnsigned;
  unsigned int fStride;
};

typedef KFParticleGpuCandidateMetadataSoAViewBase<int, unsigned int>
  KFParticleGpuCandidateMetadataSoAView;
typedef KFParticleGpuCandidateMetadataSoAViewBase<const int, const unsigned int>
  KFParticleGpuConstCandidateMetadataSoAView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidateMetadataSoAView
MakeConstView(const KFParticleGpuCandidateMetadataSoAView& view)
{
  return KFParticleGpuConstCandidateMetadataSoAView(
    view.IntegersData(), view.UnsignedData(), view.Stride());
}

/**
 * Flat CSR storage of final source-track identifiers.
 *
 * A candidate owns the range recorded by DaughterOffset and DaughterCount in
 * its metadata. The storage does not own or reserve memory by itself.
 */
template<typename IntegerValue, typename UnsignedValue>
class KFParticleGpuDaughterStorageViewBase
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDaughterStorageViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuDaughterStorageViewBase(IntegerValue* sourceIds,
                                                                  UnsignedValue* size,
                                                                  unsigned int capacity)
    : fSourceIds(sourceIds), fSize(size), fCapacity(capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue& SourceId(unsigned int index) const { return fSourceIds[index]; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE bool CanStore(unsigned int offset, unsigned int count) const
  {
    return offset <= fCapacity && count <= fCapacity - offset;
  }

  KFPARTICLE_GPU_HOST_DEVICE IntegerValue* SourceIdsData() const { return fSourceIds; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }

 private:
  IntegerValue* fSourceIds;
  UnsignedValue* fSize;
  unsigned int fCapacity;
};

typedef KFParticleGpuDaughterStorageViewBase<int, unsigned int> KFParticleGpuDaughterStorageView;
typedef KFParticleGpuDaughterStorageViewBase<const int, const unsigned int>
  KFParticleGpuConstDaughterStorageView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstDaughterStorageView
MakeConstView(const KFParticleGpuDaughterStorageView& view)
{
  return KFParticleGpuConstDaughterStorageView(
    view.SourceIdsData(), view.SizeData(), view.Capacity());
}

enum KFParticleGpuPoolOverflow
{
  CandidateCapacityExceeded = 1u << 0,
  DaughterCapacityExceeded = 1u << 1
};

enum KFParticleGpuCandidateFlags
{
  KFGpuCandidateValid = 1u << 0,
  KFGpuCandidateBuildFailed = 1u << 1,
  KFGpuCandidateLineDca = 1u << 2,
  KFGpuCandidateEnergyFit = 1u << 3,
  KFGpuCandidateSelectionRejected = 1u << 4
};

/**
 * Non-owning persistent candidate pool descriptor.
 *
 * Kernels receive a candidate and daughter index already reserved by their
 * atomic allocation step. This view only validates capacity and exposes data.
 */
template<typename FloatValue, typename IntegerValue, typename UnsignedValue>
class KFParticleGpuCandidatePoolViewBase
{
 public:
  typedef KFParticleGpuFitSoAViewBase<FloatValue, IntegerValue> FitView;
  typedef KFParticleGpuCandidateMetadataSoAViewBase<IntegerValue, UnsignedValue> MetadataView;
  typedef KFParticleGpuDaughterStorageViewBase<IntegerValue, UnsignedValue> DaughterView;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidatePoolViewBase() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuCandidatePoolViewBase(const FitView& fit,
                                                                const MetadataView& metadata,
                                                                const DaughterView& daughters,
                                                                UnsignedValue* size,
                                                                UnsignedValue* overflowFlags,
                                                                unsigned int capacity)
    : fFit(fit)
    , fMetadata(metadata)
    , fDaughters(daughters)
    , fSize(size)
    , fOverflowFlags(overflowFlags)
    , fCapacity(capacity)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE FitView& Fit() { return fFit; }
  KFPARTICLE_GPU_HOST_DEVICE const FitView& Fit() const { return fFit; }
  KFPARTICLE_GPU_HOST_DEVICE MetadataView& Metadata() { return fMetadata; }
  KFPARTICLE_GPU_HOST_DEVICE const MetadataView& Metadata() const { return fMetadata; }
  KFPARTICLE_GPU_HOST_DEVICE DaughterView& Daughters() { return fDaughters; }
  KFPARTICLE_GPU_HOST_DEVICE const DaughterView& Daughters() const { return fDaughters; }

  KFPARTICLE_GPU_HOST_DEVICE unsigned int Size() const
  {
    return fSize ? static_cast<unsigned int>(*fSize) : 0;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int Capacity() const { return fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE bool CanStoreCandidate(unsigned int index) const { return index < fCapacity; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int OverflowFlags() const
  {
    return fOverflowFlags ? static_cast<unsigned int>(*fOverflowFlags) : 0;
  }

  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* SizeData() const { return fSize; }
  KFPARTICLE_GPU_HOST_DEVICE UnsignedValue* OverflowFlagsData() const { return fOverflowFlags; }

 private:
  FitView fFit;
  MetadataView fMetadata;
  DaughterView fDaughters;
  UnsignedValue* fSize;
  UnsignedValue* fOverflowFlags;
  unsigned int fCapacity;
};

typedef KFParticleGpuCandidatePoolViewBase<float, int, unsigned int> KFParticleGpuCandidatePoolView;
typedef KFParticleGpuCandidatePoolViewBase<const float, const int, const unsigned int>
  KFParticleGpuConstCandidatePoolView;

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuConstCandidatePoolView
MakeConstView(const KFParticleGpuCandidatePoolView& view)
{
  return KFParticleGpuConstCandidatePoolView(
    MakeConstView(view.Fit()),
    MakeConstView(view.Metadata()),
    MakeConstView(view.Daughters()),
    view.SizeData(),
    view.OverflowFlagsData(),
    view.Capacity());
}

#endif
