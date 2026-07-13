/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUDECAYPLAN_H
#define KFPARTICLEGPUDECAYPLAN_H

#include "KFParticleGpuTwoDaughter.h"

#include <cstddef>
#include <vector>

/**
 * Standalone description of one two-daughter decay channel.
 *
 * The descriptor intentionally mirrors the compact task source, but keeps the
 * event index out. One channel can therefore be reused for any packed event.
 */
struct KFParticleGpuTwoDaughterChannel
{
  unsigned int channelId = 0;
  KFParticleGpuTrackSet firstTrackSet = SecondaryPositiveFirst;
  KFParticleGpuTrackSet secondTrackSet = SecondaryNegativeFirst;
  KFParticleGpuTrackSpecies firstSpecies = Pion;
  KFParticleGpuTrackSpecies secondSpecies = Pion;
  unsigned int flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
  int motherPdg = 0;
  int firstDaughterPdg = 0;
  int secondDaughterPdg = 0;
  int primaryVertexIndex = -1;
  float firstMass = 0.f;
  float secondMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = 0.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  float minSecondaryLdL = -1.f;
  KFParticleGpuV0SelectionConfig selection;
  int transportMode = KFGpuTransportStraightLine;
  int firstCharge = 0;
  int secondCharge = 0;
  int minFirstPixelHits = 0;
  int minSecondPixelHits = 0;
  float maxFirstChiToPrimaryVertex = -1.f;
  float maxSecondChiToPrimaryVertex = -1.f;
};

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuTwoDaughterTaskSource
MakeTwoDaughterTaskSource(const KFParticleGpuTwoDaughterChannel& channel,
                          unsigned int eventIndex)
{
  KFParticleGpuTwoDaughterTaskSource source;
  source.channelId = channel.channelId;
  source.eventIndex = eventIndex;
  source.firstTrackSet = channel.firstTrackSet;
  source.secondTrackSet = channel.secondTrackSet;
  source.firstSpecies = channel.firstSpecies;
  source.secondSpecies = channel.secondSpecies;
  source.flags = channel.flags;
  source.motherPdg = channel.motherPdg;
  source.firstDaughterPdg = channel.firstDaughterPdg;
  source.secondDaughterPdg = channel.secondDaughterPdg;
  source.primaryVertexIndex = channel.primaryVertexIndex;
  source.firstMass = channel.firstMass;
  source.secondMass = channel.secondMass;
  source.motherMass = channel.motherMass;
  source.motherMassSigma = channel.motherMassSigma;
  source.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
  source.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
  source.minSecondaryLdL = channel.minSecondaryLdL;
  source.transportMode = channel.transportMode;
  source.firstCharge = channel.firstCharge;
  source.secondCharge = channel.secondCharge;
  source.minFirstPixelHits = channel.minFirstPixelHits;
  source.minSecondPixelHits = channel.minSecondPixelHits;
  source.maxFirstChiToPrimaryVertex = channel.maxFirstChiToPrimaryVertex;
  source.maxSecondChiToPrimaryVertex = channel.maxSecondChiToPrimaryVertex;
  return source;
}

struct KFParticleGpuCandidateRange
{
  unsigned int offset = 0;
  unsigned int size = 0;
  unsigned int daughterOffset = 0;
  unsigned int daughterSize = 0;
  unsigned int overflowFlags = 0;

  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return size == 0u; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int End() const { return offset + size; }
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsCandidate(unsigned int index) const
  {
    return index >= offset && index < End();
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int LocalCandidateIndex(unsigned int index) const
  {
    return index - offset;
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int DaughterEnd() const
  {
    return daughterOffset + daughterSize;
  }
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsDaughter(unsigned int index) const
  {
    return index >= daughterOffset && index < DaughterEnd();
  }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int LocalDaughterIndex(unsigned int index) const
  {
    return index - daughterOffset;
  }
  KFPARTICLE_GPU_HOST_DEVICE bool HasOverflow(unsigned int mask) const
  {
    return (overflowFlags & mask) != 0u;
  }
  KFPARTICLE_GPU_HOST_DEVICE bool HasAnyOverflow() const { return overflowFlags != 0u; }
};

struct KFParticleGpuTwoDaughterChannelResult
{
  unsigned int channelId = 0;
  int motherPdg = 0;
  unsigned int eventIndex = 0;
  unsigned int totalPairs = 0;
  unsigned int acceptedTasks = 0;
  unsigned int storedTasks = 0;
  KFParticleGpuCandidateRange candidates;

  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const { return acceptedTasks > storedTasks; }
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return candidates.Empty(); }
  KFPARTICLE_GPU_HOST_DEVICE bool HasOverflow(unsigned int mask) const
  {
    return candidates.HasOverflow(mask);
  }
  KFPARTICLE_GPU_HOST_DEVICE bool HasAnyOverflow() const
  {
    return candidates.HasAnyOverflow();
  }
  KFPARTICLE_GPU_HOST_DEVICE bool ContainsCandidate(unsigned int index) const
  {
    return candidates.ContainsCandidate(index);
  }
};

enum KFParticleGpuDefaultChannelId
{
  KFGpuChannelK0ShortToPiPlusPiMinus = 1u,
  KFGpuChannelLambdaToProtonPiMinus = 2u,
  KFGpuChannelAntiLambdaToAntiProtonPiPlus = 3u
};

/**
 * Host-side description of the ordered reconstruction stages.
 */
class KFParticleGpuDecayPlan
{
 public:
  KFParticleGpuDecayPlan();
  ~KFParticleGpuDecayPlan();

  void Clear();
  bool Empty() const;

  void AddTwoDaughterChannel(const KFParticleGpuTwoDaughterChannel& channel);
  std::size_t NumberOfTwoDaughterChannels() const;
  const KFParticleGpuTwoDaughterChannel& TwoDaughterChannel(std::size_t index) const;

 private:
  KFParticleGpuDecayPlan(const KFParticleGpuDecayPlan&);
  KFParticleGpuDecayPlan& operator=(const KFParticleGpuDecayPlan&);

  std::vector<KFParticleGpuTwoDaughterChannel> fTwoDaughterChannels;
};

KFParticleGpuTwoDaughterChannel MakeK0ShortToPiPlusPiMinusChannel(
  unsigned int channelId = KFGpuChannelK0ShortToPiPlusPiMinus);
KFParticleGpuTwoDaughterChannel MakeLambdaToProtonPiMinusChannel(
  unsigned int channelId = KFGpuChannelLambdaToProtonPiMinus);
KFParticleGpuTwoDaughterChannel MakeAntiLambdaToAntiProtonPiPlusChannel(
  unsigned int channelId = KFGpuChannelAntiLambdaToAntiProtonPiPlus);
void AddDefaultV0TwoDaughterChannels(KFParticleGpuDecayPlan& plan);

#endif
