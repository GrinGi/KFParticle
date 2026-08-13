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

#include "KFParticleGpuGraphOperations.h"
#include "KFParticleGpuV0Track.h"

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
  unsigned int outputClass = KFGpuGraphOutputPrimaryAndSecondary;
  int motherPdg = 0;
  int firstDaughterPdg = 0;
  int secondDaughterPdg = 0;
  // Input identity is independent of the mass hypothesis. CBM's secondary
  // positive pion tracks are also tested with a proton hypothesis for Lambda.
  int firstSourcePdg = 0;
  int firstAlternateSourcePdg = 0;
  int secondSourcePdg = 0;
  int secondAlternateSourcePdg = 0;
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
  float minFirstChiToPrimaryVertex = -1.f;
  float minSecondChiToPrimaryVertex = -1.f;
  float minFirstPt = -1.f;
  float minSecondPt = -1.f;
  float maxDaughterDistance = -1.f;
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
  source.outputClass = channel.outputClass;
  source.motherPdg = channel.motherPdg;
  source.firstDaughterPdg = channel.firstDaughterPdg;
  source.secondDaughterPdg = channel.secondDaughterPdg;
  source.firstSourcePdg = channel.firstSourcePdg;
  source.firstAlternateSourcePdg = channel.firstAlternateSourcePdg;
  source.secondSourcePdg = channel.secondSourcePdg;
  source.secondAlternateSourcePdg = channel.secondAlternateSourcePdg;
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
  source.minFirstChiToPrimaryVertex = channel.minFirstChiToPrimaryVertex;
  source.minSecondChiToPrimaryVertex = channel.minSecondChiToPrimaryVertex;
  source.minFirstPt = channel.minFirstPt;
  source.minSecondPt = channel.minSecondPt;
  source.maxDaughterDistance = channel.maxDaughterDistance;
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
  unsigned int constructedCandidates = 0;
  unsigned int constructedDaughters = 0;
  // The fused first generation is unordered. Stable candidate channelId
  // metadata identifies membership inside this shared event range.
  KFParticleGpuCandidateRange generationCandidates;
  // Retained for isolated explicit-channel APIs. Production fused steering
  // reports counts and generationCandidates instead of pretending contiguity.
  KFParticleGpuCandidateRange candidates;

  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const { return acceptedTasks > storedTasks; }
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const
  {
    return constructedCandidates == 0u && candidates.Empty();
  }
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

/** Reusable cascade descriptor; event-local bachelor ranges are resolved by steering. */
struct KFParticleGpuV0TrackCascadeChannel
{
  unsigned int channelId = 0u;
  unsigned int family = KFGpuCpuFamilyTrackComposite;
  unsigned int generation = 2u;
  unsigned int outputClass = KFGpuGraphOutputSecondary;
  // Zero retains the legacy PDG-only source lookup. Production graph plans
  // publish the stable parent channel ID so equal-PDG generations cannot mix.
  unsigned int parentChannelId = 0u;
  int v0Pdg = 0;
  KFParticleGpuTrackSet bachelorTrackSet = SecondaryNegativeFirst;
  KFParticleGpuTrackSpecies bachelorSpecies = Pion;
  int bachelorPdg = 0;
  int motherPdg = 0;
  int primaryVertexIndex = -1;
  unsigned int flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;
  int transportMode = KFGpuTransportFullField;
  float bachelorMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = -1.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  float minBachelorChiToPrimaryVertex = -1.f;
  float maxV0TrackDistance = -1.f;
  float minBachelorPt = -1.f;
  int minBachelorPixelHits = -1;
  float parentMassConstraint = -1.f;
  float parentMassConstraintSigma = 0.f;
};

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel
MakeV0TrackTaskChannel(const KFParticleGpuV0TrackCascadeChannel& channel,
                       const KFParticleGpuEventDesc& event,
                       unsigned int eventIndex)
{
  KFParticleGpuV0TrackChannel source = KFParticleGpuV0Track::MakeChannel(
    channel.channelId,
    channel.v0Pdg,
    channel.bachelorPdg,
    channel.motherPdg,
    eventIndex,
    ResolveTaskSourceRange(event, channel.bachelorTrackSet, channel.bachelorSpecies),
    channel.bachelorMass,
    channel.motherMass);
  source.parentChannelId = channel.parentChannelId;
  source.outputClass = channel.outputClass;
  source.primaryVertexIndex = channel.primaryVertexIndex;
  source.flags = channel.flags;
  source.transportMode = channel.transportMode;
  source.motherMassSigma = channel.motherMassSigma;
  source.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
  source.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
  source.minBachelorChiToPrimaryVertex = channel.minBachelorChiToPrimaryVertex;
  source.maxV0TrackDistance = channel.maxV0TrackDistance;
  source.minBachelorPt = channel.minBachelorPt;
  source.minBachelorPixelHits = channel.minBachelorPixelHits;
  source.parentMassConstraint = channel.parentMassConstraint;
  source.parentMassConstraintSigma = channel.parentMassConstraintSigma;
  return source;
}

struct KFParticleGpuV0TrackChannelResult
{
  unsigned int channelId = 0u;
  int motherPdg = 0;
  unsigned int eventIndex = 0u;
  unsigned int totalPairs = 0u;
  unsigned int acceptedTasks = 0u;
  unsigned int storedTasks = 0u;
  unsigned int constructedCandidates = 0u;
  unsigned int constructedDaughters = 0u;
  // Fused output is one unordered event generation. This range identifies
  // that generation; channel membership is carried by candidate metadata.
  KFParticleGpuCandidateRange generationCandidates;
  // Retained for the explicit diagnostic path, whose channel launch produces
  // a contiguous range. Fused production output is intentionally unordered.
  KFParticleGpuCandidateRange candidates;

  KFPARTICLE_GPU_HOST_DEVICE bool Truncated() const { return acceptedTasks > storedTasks; }
  KFPARTICLE_GPU_HOST_DEVICE unsigned int CandidateCount() const
  {
    return constructedCandidates != 0u ? constructedCandidates : candidates.size;
  }
  KFPARTICLE_GPU_HOST_DEVICE bool Empty() const { return CandidateCount() == 0u; }
};

enum KFParticleGpuDefaultChannelId
{
  KFGpuChannelK0ShortToPiPlusPiMinus = 1u,
  KFGpuChannelLambdaToProtonPiMinus = 2u,
  KFGpuChannelAntiLambdaToAntiProtonPiPlus = 3u
};

enum KFParticleGpuDefaultCascadeChannelId
{
  KFGpuChannelXiMinusToLambdaPiMinus = 11u,
  KFGpuChannelAntiXiPlusToAntiLambdaPiPlus = 12u,
  KFGpuChannelOmegaMinusToLambdaKMinus = 13u,
  KFGpuChannelAntiOmegaPlusToAntiLambdaKPlus = 14u
};

enum KFParticleGpuRepresentativeChargedChannelId
{
  KFGpuChannelRho0ToPiPlusPiMinus = 21u,
  KFGpuChannelDeltaPlusPlusToProtonPiPlus = 22u,
  KFGpuChannelD0ToPiPlusKMinus = 23u,
  KFGpuChannelDPlusToD0PiPlus = 24u,
  KFGpuChannelHypertritonToDeuteronPiMinus = 25u,
  KFGpuChannelCompositeCompositeProbe = 26u,
  KFGpuChannelNeutralMissingMassProbe = 27u,
  KFGpuChannelUnaryFinalProbe = 28u
};

/**
 * One later-generation graph node and its immutable physical operation.
 *
 * Source dependencies live in the node, while fit and selection parameters
 * live in the descriptor. Their channel IDs form one stable identity.
 */
struct KFParticleGpuGraphOperationChannel
{
  KFParticleGpuGraphNode node;
  KFParticleGpuGraphOperationDescriptor descriptor;
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
  unsigned long long Revision() const;

  void AddTwoDaughterChannel(const KFParticleGpuTwoDaughterChannel& channel);
  std::size_t NumberOfTwoDaughterChannels() const;
  const KFParticleGpuTwoDaughterChannel& TwoDaughterChannel(std::size_t index) const;
  void AddV0TrackCascadeChannel(const KFParticleGpuV0TrackCascadeChannel& channel);
  std::size_t NumberOfV0TrackCascadeChannels() const;
  const KFParticleGpuV0TrackCascadeChannel& V0TrackCascadeChannel(std::size_t index) const;
  void AddGraphOperationChannel(const KFParticleGpuGraphOperationChannel& channel);
  std::size_t NumberOfGraphOperationChannels() const;
  const KFParticleGpuGraphOperationChannel& GraphOperationChannel(
    std::size_t index) const;

 private:
  KFParticleGpuDecayPlan(const KFParticleGpuDecayPlan&);
  KFParticleGpuDecayPlan& operator=(const KFParticleGpuDecayPlan&);

  std::vector<KFParticleGpuTwoDaughterChannel> fTwoDaughterChannels;
  std::vector<KFParticleGpuV0TrackCascadeChannel> fV0TrackCascadeChannels;
  std::vector<KFParticleGpuGraphOperationChannel> fGraphOperationChannels;
  unsigned long long fRevision;
};

KFParticleGpuTwoDaughterChannel MakeK0ShortToPiPlusPiMinusChannel(
  unsigned int channelId = KFGpuChannelK0ShortToPiPlusPiMinus);
KFParticleGpuTwoDaughterChannel MakeLambdaToProtonPiMinusChannel(
  unsigned int channelId = KFGpuChannelLambdaToProtonPiMinus);
KFParticleGpuTwoDaughterChannel MakeAntiLambdaToAntiProtonPiPlusChannel(
  unsigned int channelId = KFGpuChannelAntiLambdaToAntiProtonPiPlus);
void AddDefaultV0TwoDaughterChannels(KFParticleGpuDecayPlan& plan);
void AddDefaultV0TrackCascadeChannels(KFParticleGpuDecayPlan& plan);

#endif
