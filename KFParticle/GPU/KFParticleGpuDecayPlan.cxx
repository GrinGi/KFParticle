/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuDecayPlan.h"

#include <stdexcept>

KFParticleGpuDecayPlan::KFParticleGpuDecayPlan()
  : fTwoDaughterChannels(), fV0TrackCascadeChannels(),
    fGraphOperationChannels(), fRevision(1u)
{
}

KFParticleGpuDecayPlan::~KFParticleGpuDecayPlan() {}

void KFParticleGpuDecayPlan::Clear()
{
  fTwoDaughterChannels.clear();
  fV0TrackCascadeChannels.clear();
  fGraphOperationChannels.clear();
  ++fRevision;
}

bool KFParticleGpuDecayPlan::Empty() const
{
  return fTwoDaughterChannels.empty() && fV0TrackCascadeChannels.empty()
         && fGraphOperationChannels.empty();
}

unsigned long long KFParticleGpuDecayPlan::Revision() const
{
  return fRevision;
}

void KFParticleGpuDecayPlan::AddTwoDaughterChannel(
  const KFParticleGpuTwoDaughterChannel& channel)
{
  fTwoDaughterChannels.push_back(channel);
  ++fRevision;
}

std::size_t KFParticleGpuDecayPlan::NumberOfTwoDaughterChannels() const
{
  return fTwoDaughterChannels.size();
}

const KFParticleGpuTwoDaughterChannel& KFParticleGpuDecayPlan::TwoDaughterChannel(
  std::size_t index) const
{
  if (index >= fTwoDaughterChannels.size()) {
    throw std::out_of_range("KFParticle GPU two-daughter channel index is out of range");
  }
  return fTwoDaughterChannels[index];
}

void KFParticleGpuDecayPlan::AddV0TrackCascadeChannel(
  const KFParticleGpuV0TrackCascadeChannel& channel)
{
  fV0TrackCascadeChannels.push_back(channel);
  ++fRevision;
}

std::size_t KFParticleGpuDecayPlan::NumberOfV0TrackCascadeChannels() const
{
  return fV0TrackCascadeChannels.size();
}

const KFParticleGpuV0TrackCascadeChannel& KFParticleGpuDecayPlan::V0TrackCascadeChannel(
  std::size_t index) const
{
  if (index >= fV0TrackCascadeChannels.size()) {
    throw std::out_of_range("KFParticle GPU V0-track cascade channel index is out of range");
  }
  return fV0TrackCascadeChannels[index];
}

void KFParticleGpuDecayPlan::AddGraphOperationChannel(
  const KFParticleGpuGraphOperationChannel& channel)
{
  if (channel.node.channelId == 0u
      || channel.node.channelId != channel.descriptor.channelId) {
    throw std::invalid_argument(
      "KFParticle GPU graph node and operation descriptor must share a channel ID");
  }
  fGraphOperationChannels.push_back(channel);
  ++fRevision;
}

std::size_t KFParticleGpuDecayPlan::NumberOfGraphOperationChannels() const
{
  return fGraphOperationChannels.size();
}

const KFParticleGpuGraphOperationChannel&
KFParticleGpuDecayPlan::GraphOperationChannel(std::size_t index) const
{
  if (index >= fGraphOperationChannels.size()) {
    throw std::out_of_range(
      "KFParticle GPU graph operation channel index is out of range");
  }
  return fGraphOperationChannels[index];
}

namespace
{
  // Mirrored from KFParticleDatabase. Keeping the values local avoids a CPU
  // database dependency in device-side descriptors while preserving CPU
  // comparison stability.
  const float kKfParticleGpuPionMass = 0.13957039f;
  const float kKfParticleGpuKaonMass = 0.493677f;
  const float kKfParticleGpuProtonMass = 0.9382720813f;
  const float kKfParticleGpuK0ShortMass = 0.497614f;
  const float kKfParticleGpuLambdaMass = 1.115683f;

#ifdef PANDA_STT
  const float kKfParticleGpuK0ShortMassSigma = 12.0e-3f;
  const float kKfParticleGpuLambdaMassSigma = 2.7e-3f;
  const float kKfParticleGpuDefaultSecondaryTopoChi2 = -3.f;
#elif defined ALICE_ITS
  const float kKfParticleGpuK0ShortMassSigma = 17.7e-3f;
  const float kKfParticleGpuLambdaMassSigma = 5.9e-3f;
  const float kKfParticleGpuDefaultSecondaryTopoChi2 = 5.f;
#elif defined STAR_HFT
  const float kKfParticleGpuK0ShortMassSigma = 17.7e-3f;
  const float kKfParticleGpuLambdaMassSigma = 5.9e-3f;
  const float kKfParticleGpuDefaultSecondaryTopoChi2 = 5.f;
#elif defined CBM
  const float kKfParticleGpuK0ShortMassSigma = 3.7e-3f;
  const float kKfParticleGpuLambdaMassSigma = 1.5e-3f;
  const float kKfParticleGpuDefaultSecondaryTopoChi2 = 5.f;
#else
  const float kKfParticleGpuK0ShortMassSigma = 4.9e-3f;
  const float kKfParticleGpuLambdaMassSigma = 2.1e-3f;
  const float kKfParticleGpuDefaultSecondaryTopoChi2 = 5.f;
#endif

  KFParticleGpuTwoDaughterChannel MakeDefaultV0Channel(
    unsigned int channelId,
    int motherPdg,
    float motherMass,
    float motherMassSigma,
    KFParticleGpuTrackSet firstTrackSet,
    KFParticleGpuTrackSet secondTrackSet,
    KFParticleGpuTrackSpecies firstSpecies,
    KFParticleGpuTrackSpecies secondSpecies,
    int firstDaughterPdg,
    int secondDaughterPdg,
    float firstMass,
    float secondMass,
    int firstCharge,
    int secondCharge)
  {
    KFParticleGpuTwoDaughterChannel channel;
    channel.channelId = channelId;
    channel.firstTrackSet = firstTrackSet;
    channel.secondTrackSet = secondTrackSet;
    channel.firstSpecies = firstSpecies;
    channel.secondSpecies = secondSpecies;
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    channel.motherPdg = motherPdg;
    channel.firstDaughterPdg = firstDaughterPdg;
    channel.secondDaughterPdg = secondDaughterPdg;
    channel.firstSourcePdg = firstDaughterPdg;
    channel.secondSourcePdg = secondDaughterPdg;
    channel.primaryVertexIndex = -1;
    channel.firstMass = firstMass;
    channel.secondMass = secondMass;
    channel.motherMass = motherMass;
    channel.motherMassSigma = motherMassSigma;
    channel.secondaryMassSigmaCut = 3.f;
    channel.maxSecondaryTopoChi2PerNdf = kKfParticleGpuDefaultSecondaryTopoChi2;
    channel.minSecondaryLdL = 10.f;
    channel.selection.expectedMass = motherMass;
    channel.selection.expectedMassSigma = motherMassSigma;
    channel.selection.massSigmaCut = channel.secondaryMassSigmaCut;
    channel.selection.maxGeometricChi2PerNdf = 3.f;
    channel.selection.maxPrimaryVertexDistance = 200.f;
    // CPU ConstructV0 publishes the candidate to GetParticles() with the
    // general two-daughter cut fCuts2D[2] (default 5).  The stricter
    // fSecCuts[2] value above (default 10) is only for its secondary-candidate
    // side list and must not suppress the public CPU/GPU comparison boundary.
    channel.selection.minSecondaryLdL = 5.f;
    // CPU ConstructV0 publishes these channels through its secondary path
    // (pvIndex == -1), so the publication L/dL cut must never be bypassed by
    // classifying a candidate as primary here.
    channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
    channel.selection.requirePrimaryVertex = 1u;
    channel.selection.topologyMode = KFGpuV0TopologyLine;
    channel.transportMode = KFGpuTransportFullField;
    channel.firstCharge = firstCharge;
    channel.secondCharge = secondCharge;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 0;
    channel.maxFirstChiToPrimaryVertex = -1.f;
    channel.maxSecondChiToPrimaryVertex = -1.f;
    channel.maxDaughterDistance = 1.f;
    return channel;
  }
}

KFParticleGpuTwoDaughterChannel MakeK0ShortToPiPlusPiMinusChannel(unsigned int channelId)
{
  KFParticleGpuTwoDaughterChannel channel =
    MakeDefaultV0Channel(channelId,
                         310,
                         kKfParticleGpuK0ShortMass,
                         kKfParticleGpuK0ShortMassSigma,
                         SecondaryPositiveFirst,
                         SecondaryNegativeFirst,
                         Pion,
                         Pion,
                         211,
                         -211,
                         kKfParticleGpuPionMass,
                         kKfParticleGpuPionMass,
                         1,
                         -1);
  channel.flags |= KFGpuTwoDaughterReverseFitOrder;
  return channel;
}

KFParticleGpuTwoDaughterChannel MakeLambdaToProtonPiMinusChannel(unsigned int channelId)
{
  KFParticleGpuTwoDaughterChannel channel =
    MakeDefaultV0Channel(channelId,
                         3122,
                         kKfParticleGpuLambdaMass,
                         kKfParticleGpuLambdaMassSigma,
                         SecondaryPositiveFirst,
                         SecondaryNegativeFirst,
                         NumberOfTrackSpecies,
                         Pion,
                         2212,
                         -211,
                         kKfParticleGpuProtonMass,
                         kKfParticleGpuPionMass,
                         1,
                         -1);
  // CPU CBM routing tests a secondary positive pion both as pi+ and as p+.
  // Resolve the full positive set so genuine proton PID and this fallback can
  // coexist without duplicating the packed physical track. Host-side capacity
  // accounting must resolve this sentinel to TrackSet::tracks.
  channel.firstSourcePdg = 2212;
  channel.firstAlternateSourcePdg = 211;
  channel.flags |= KFGpuTwoDaughterReverseFitOrder;
  return channel;
}

KFParticleGpuTwoDaughterChannel MakeAntiLambdaToAntiProtonPiPlusChannel(unsigned int channelId)
{
  return MakeDefaultV0Channel(channelId,
                              -3122,
                              kKfParticleGpuLambdaMass,
                              kKfParticleGpuLambdaMassSigma,
                              SecondaryNegativeFirst,
                              SecondaryPositiveFirst,
                              Proton,
                              Pion,
                              -2212,
                              211,
                              kKfParticleGpuProtonMass,
                              kKfParticleGpuPionMass,
                              -1,
                              1);
}

void AddDefaultV0TwoDaughterChannels(KFParticleGpuDecayPlan& plan)
{
  plan.AddTwoDaughterChannel(MakeK0ShortToPiPlusPiMinusChannel());
  plan.AddTwoDaughterChannel(MakeLambdaToProtonPiMinusChannel());
  plan.AddTwoDaughterChannel(MakeAntiLambdaToAntiProtonPiPlusChannel());
}

void AddDefaultV0TrackCascadeChannels(KFParticleGpuDecayPlan& plan)
{
  KFParticleGpuV0TrackCascadeChannel xiMinus;
  xiMinus.channelId = KFGpuChannelXiMinusToLambdaPiMinus;
  xiMinus.parentChannelId = KFGpuChannelLambdaToProtonPiMinus;
  xiMinus.v0Pdg = 3122;
  xiMinus.bachelorTrackSet = SecondaryNegativeFirst;
  xiMinus.bachelorSpecies = Pion;
  xiMinus.bachelorPdg = -211;
  xiMinus.motherPdg = 3312;
  xiMinus.bachelorMass = kKfParticleGpuPionMass;
  xiMinus.motherMass = 1.32171f;
  plan.AddV0TrackCascadeChannel(xiMinus);

  KFParticleGpuV0TrackCascadeChannel antiXiPlus = xiMinus;
  antiXiPlus.channelId = KFGpuChannelAntiXiPlusToAntiLambdaPiPlus;
  antiXiPlus.parentChannelId = KFGpuChannelAntiLambdaToAntiProtonPiPlus;
  antiXiPlus.v0Pdg = -3122;
  antiXiPlus.bachelorTrackSet = SecondaryPositiveFirst;
  antiXiPlus.bachelorPdg = 211;
  antiXiPlus.motherPdg = -3312;
  plan.AddV0TrackCascadeChannel(antiXiPlus);

  KFParticleGpuV0TrackCascadeChannel omegaMinus = xiMinus;
  omegaMinus.channelId = KFGpuChannelOmegaMinusToLambdaKMinus;
  omegaMinus.bachelorSpecies = Kaon;
  omegaMinus.bachelorPdg = -321;
  omegaMinus.motherPdg = 3334;
  omegaMinus.bachelorMass = kKfParticleGpuKaonMass;
  omegaMinus.motherMass = 1.67245f;
  plan.AddV0TrackCascadeChannel(omegaMinus);

  KFParticleGpuV0TrackCascadeChannel antiOmegaPlus = antiXiPlus;
  antiOmegaPlus.channelId = KFGpuChannelAntiOmegaPlusToAntiLambdaKPlus;
  antiOmegaPlus.bachelorSpecies = Kaon;
  antiOmegaPlus.bachelorPdg = 321;
  antiOmegaPlus.motherPdg = -3334;
  antiOmegaPlus.bachelorMass = kKfParticleGpuKaonMass;
  antiOmegaPlus.motherMass = 1.67245f;
  plan.AddV0TrackCascadeChannel(antiOmegaPlus);
}
