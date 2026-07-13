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

KFParticleGpuDecayPlan::KFParticleGpuDecayPlan() : fTwoDaughterChannels() {}

KFParticleGpuDecayPlan::~KFParticleGpuDecayPlan() {}

void KFParticleGpuDecayPlan::Clear()
{
  fTwoDaughterChannels.clear();
}

bool KFParticleGpuDecayPlan::Empty() const
{
  return fTwoDaughterChannels.empty();
}

void KFParticleGpuDecayPlan::AddTwoDaughterChannel(
  const KFParticleGpuTwoDaughterChannel& channel)
{
  fTwoDaughterChannels.push_back(channel);
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

namespace
{
  // Mirrored from KFParticleDatabase. Keeping the values local avoids a CPU
  // database dependency in device-side descriptors while preserving CPU
  // comparison stability.
  const float kKfParticleGpuPionMass = 0.13957039f;
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
    channel.selection.minSecondaryLdL = channel.minSecondaryLdL;
    channel.selection.maxPrimaryTopologyChi2PerNdf = kKfParticleGpuDefaultSecondaryTopoChi2;
    channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
    channel.selection.requirePrimaryVertex = 1u;
    channel.transportMode = KFGpuTransportFieldAware;
    channel.firstCharge = firstCharge;
    channel.secondCharge = secondCharge;
    channel.minFirstPixelHits = 0;
    channel.minSecondPixelHits = 0;
    channel.maxFirstChiToPrimaryVertex = -1.f;
    channel.maxSecondChiToPrimaryVertex = -1.f;
    return channel;
  }
}

KFParticleGpuTwoDaughterChannel MakeK0ShortToPiPlusPiMinusChannel(unsigned int channelId)
{
  return MakeDefaultV0Channel(channelId,
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
}

KFParticleGpuTwoDaughterChannel MakeLambdaToProtonPiMinusChannel(unsigned int channelId)
{
  return MakeDefaultV0Channel(channelId,
                              3122,
                              kKfParticleGpuLambdaMass,
                              kKfParticleGpuLambdaMassSigma,
                              SecondaryPositiveFirst,
                              SecondaryNegativeFirst,
                              Proton,
                              Pion,
                              2212,
                              -211,
                              kKfParticleGpuProtonMass,
                              kKfParticleGpuPionMass,
                              1,
                              -1);
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
