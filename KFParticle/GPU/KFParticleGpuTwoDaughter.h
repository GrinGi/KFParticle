/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUTWODAUGHTER_H
#define KFPARTICLEGPUTWODAUGHTER_H

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuInputDataTransfer.h"
#include "KFParticleGpuMath.h"
#include "KFParticleGpuSelection.h"

enum KFParticleGpuTwoDaughterTaskFlags
{
  KFGpuTwoDaughterUseLineDca = 1u << 0,
  KFGpuTwoDaughterUseEnergyFit = 1u << 1
};

enum KFParticleGpuTransportMode
{
  KFGpuTransportStraightLine = 0,
  KFGpuTransportFieldAware = 1
};

struct KFParticleGpuTwoDaughterTask
{
  unsigned int channelId = 0;
  unsigned int firstTrack = 0;
  unsigned int secondTrack = 0;
  unsigned int eventIndex = 0;
  unsigned int flags = 0;
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
  int transportMode = KFGpuTransportStraightLine;
};

struct KFParticleGpuTwoDaughterTaskSource
{
  unsigned int channelId = 0;
  unsigned int eventIndex = 0;
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
  int transportMode = KFGpuTransportStraightLine;
  int firstCharge = 0;
  int secondCharge = 0;
  int minFirstPixelHits = 0;
  int minSecondPixelHits = 0;
  float maxFirstChiToPrimaryVertex = -1.f;
  float maxSecondChiToPrimaryVertex = -1.f;
};

KFPARTICLE_GPU_HOST_DEVICE inline unsigned int TwoDaughterCandidateFlags(
  const KFParticleGpuTwoDaughterTask& task)
{
  unsigned int flags = KFGpuCandidateValid;
  if (task.flags & KFGpuTwoDaughterUseLineDca) {
    flags |= KFGpuCandidateLineDca;
  }
  if (task.flags & KFGpuTwoDaughterUseEnergyFit) {
    flags |= KFGpuCandidateEnergyFit;
  }
  return flags;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool IsSupportedTwoDaughterTransportMode(int mode)
{
  return mode == KFGpuTransportStraightLine || mode == KFGpuTransportFieldAware;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool IsSupportedTwoDaughterEnergyFitTransportMode(int mode)
{
  return mode == KFGpuTransportStraightLine || mode == KFGpuTransportFieldAware;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool ValidateTwoDaughterTask(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task)
{
  if (task.firstTrack >= inputTracks.Size() || task.secondTrack >= inputTracks.Size()) {
    return false;
  }
  if (task.firstTrack == task.secondTrack) {
    return false;
  }
  if (inputTracks.SourceId(task.firstTrack) == inputTracks.SourceId(task.secondTrack)) {
    return false;
  }
  if (task.firstDaughterPdg != 0 && inputTracks.Pdg(task.firstTrack) != task.firstDaughterPdg) {
    return false;
  }
  if (task.secondDaughterPdg != 0 && inputTracks.Pdg(task.secondTrack) != task.secondDaughterPdg) {
    return false;
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool PassOptionalTrackCuts(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  unsigned int track,
  int expectedPdg,
  int expectedCharge,
  int minPixelHits,
  float maxChiToPrimaryVertex)
{
  if (track >= inputTracks.Size()) {
    return false;
  }
  if (expectedPdg != 0 && inputTracks.Pdg(track) != expectedPdg) {
    return false;
  }
  if (expectedCharge != 0 && inputTracks.Charge(track) != expectedCharge) {
    return false;
  }
  if (minPixelHits > 0 && inputTracks.NumberOfPixelHits(track) < minPixelHits) {
    return false;
  }
  if (maxChiToPrimaryVertex >= 0.f
      && inputTracks.ChiToPrimaryVertex(track) > maxChiToPrimaryVertex) {
    return false;
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool PassTwoDaughterTaskSourceCuts(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int firstTrack,
  unsigned int secondTrack)
{
  if (firstTrack == secondTrack) {
    return false;
  }
  if (firstTrack >= inputTracks.Size() || secondTrack >= inputTracks.Size()) {
    return false;
  }
  if (inputTracks.SourceId(firstTrack) == inputTracks.SourceId(secondTrack)) {
    return false;
  }
  return PassOptionalTrackCuts(inputTracks,
                               firstTrack,
                               source.firstDaughterPdg,
                               source.firstCharge,
                               source.minFirstPixelHits,
                               source.maxFirstChiToPrimaryVertex)
         && PassOptionalTrackCuts(inputTracks,
                                  secondTrack,
                                  source.secondDaughterPdg,
                                  source.secondCharge,
                                  source.minSecondPixelHits,
                                  source.maxSecondChiToPrimaryVertex);
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuRange ResolveTaskSourceRange(
  const KFParticleGpuEventDesc& event,
  KFParticleGpuTrackSet trackSet,
  KFParticleGpuTrackSpecies species)
{
  const KFParticleGpuRange speciesRange = event.TrackSet(trackSet).Species(species);
  return speciesRange.size > 0 ? speciesRange : event.TrackSet(trackSet).tracks;
}

KFPARTICLE_GPU_HOST_DEVICE inline void FillTwoDaughterTask(
  const KFParticleGpuRange& firstRange,
  const KFParticleGpuRange& secondRange,
  unsigned int pairIndex,
  const KFParticleGpuTwoDaughterTaskSource& source,
  KFParticleGpuTwoDaughterTask& task)
{
  const unsigned int firstOffset = pairIndex / secondRange.size;
  const unsigned int secondOffset = pairIndex - firstOffset * secondRange.size;
  task.firstTrack = firstRange.offset + firstOffset;
  task.secondTrack = secondRange.offset + secondOffset;
  task.channelId = source.channelId;
  task.eventIndex = source.eventIndex;
  task.flags = source.flags;
  task.motherPdg = source.motherPdg;
  task.firstDaughterPdg = source.firstDaughterPdg;
  task.secondDaughterPdg = source.secondDaughterPdg;
  task.primaryVertexIndex = source.primaryVertexIndex;
  task.firstMass = source.firstMass;
  task.secondMass = source.secondMass;
  task.motherMass = source.motherMass;
  task.motherMassSigma = source.motherMassSigma;
  task.secondaryMassSigmaCut = source.secondaryMassSigmaCut;
  task.maxSecondaryTopoChi2PerNdf = source.maxSecondaryTopoChi2PerNdf;
  task.minSecondaryLdL = source.minSecondaryLdL;
  task.transportMode = source.transportMode;
}

template<typename SelectionSource>
KFPARTICLE_GPU_HOST_DEVICE inline bool PassTwoDaughterPostBuildSelection(
  const SelectionSource& source,
  const KFParticleGpuFitState& mother)
{
  if (source.motherMassSigma > 0.f && source.secondaryMassSigmaCut >= 0.f
      && !KFParticleGpuSelection::PassSigmaMassWindow(
           mother, source.motherMass, source.motherMassSigma, source.secondaryMassSigmaCut)) {
    return false;
  }
  if (!KFParticleGpuSelection::PassChi2PerNdf(mother, source.maxSecondaryTopoChi2PerNdf)) {
    return false;
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool BuildTwoDaughterKinematicCandidate(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  KFParticleGpuFitState& mother)
{
  if (!ValidateTwoDaughterTask(inputTracks, task)) {
    return false;
  }
  if (!IsSupportedTwoDaughterTransportMode(task.transportMode)) {
    return false;
  }

  KFParticleGpuTrackState firstTrack;
  KFParticleGpuTrackState secondTrack;
  LoadTrackState(inputTracks, task.firstTrack, firstTrack);
  LoadTrackState(inputTracks, task.secondTrack, secondTrack);

  KFParticleGpuFitState firstDaughter;
  firstDaughter.Initialize(firstTrack, inputTracks.Charge(task.firstTrack), task.firstMass);
  KFParticleGpuFitState secondDaughter;
  secondDaughter.Initialize(secondTrack, inputTracks.Charge(task.secondTrack), task.secondMass);

  if ((task.flags & KFGpuTwoDaughterUseLineDca)
      && task.transportMode == KFGpuTransportFieldAware) {
    const KFParticleGpuFieldValue field =
      EvaluateTrackFieldAtState(inputTracks, task.firstTrack, firstDaughter);
    if (!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
          firstDaughter, secondDaughter, field.y, mother)) {
      return false;
    }
  }
  else if (task.flags & KFGpuTwoDaughterUseLineDca) {
    KFParticleGpuMath::BuildLineDcaKinematicMother(firstDaughter, secondDaughter, mother);
  }
  else {
    KFParticleGpuMath::BuildKinematicMother(firstDaughter, secondDaughter, mother);
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool BuildTwoDaughterEnergyFitCandidate(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  KFParticleGpuFitState& mother)
{
  if (!ValidateTwoDaughterTask(inputTracks, task)) {
    return false;
  }
  if (!IsSupportedTwoDaughterEnergyFitTransportMode(task.transportMode)) {
    return false;
  }

  KFParticleGpuTrackState firstTrack;
  KFParticleGpuTrackState secondTrack;
  LoadTrackState(inputTracks, task.firstTrack, firstTrack);
  LoadTrackState(inputTracks, task.secondTrack, secondTrack);

  KFParticleGpuFitState firstDaughter;
  firstDaughter.Initialize(firstTrack, inputTracks.Charge(task.firstTrack), task.firstMass);
  KFParticleGpuFitState secondDaughter;
  secondDaughter.Initialize(secondTrack, inputTracks.Charge(task.secondTrack), task.secondMass);

  KFParticleGpuMeasurement measurement;
  if ((task.flags & KFGpuTwoDaughterUseLineDca)
      && task.transportMode == KFGpuTransportFieldAware) {
    const KFParticleGpuFieldValue field =
      EvaluateTrackFieldAtState(inputTracks, task.firstTrack, firstDaughter);
    if (!KFParticleGpuMath::BuildConstantByDcaMeasurementSeed(
          firstDaughter, secondDaughter, field.y, mother, measurement)) {
      return false;
    }
  }
  else if (task.flags & KFGpuTwoDaughterUseLineDca) {
    if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(
          firstDaughter, secondDaughter, mother, measurement)) {
      return false;
    }
  }
  else {
    mother = firstDaughter;
    measurement.StoreState(secondDaughter);
  }

  if (!KFParticleGpuMath::AddDaughterWithEnergyFit(
        mother, measurement, inputTracks.Charge(task.secondTrack))) {
    return false;
  }

  mother.SumDaughterMass() = firstDaughter.SumDaughterMass() + secondDaughter.SumDaughterMass();
  mother.MassHypo() = -1.f;
  mother.ConstructMethod() = 0;
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool BuildTwoDaughterCandidate(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  KFParticleGpuFitState& mother)
{
  if (task.flags & KFGpuTwoDaughterUseEnergyFit) {
    return BuildTwoDaughterEnergyFitCandidate(inputTracks, task, mother);
  }
  return BuildTwoDaughterKinematicCandidate(inputTracks, task, mother);
}

KFPARTICLE_GPU_HOST_DEVICE inline bool StoreTwoDaughterCandidate(
  const KFParticleGpuCandidatePoolView& candidates,
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  unsigned int candidateIndex,
  unsigned int daughterOffset,
  const KFParticleGpuFitState& mother)
{
  if (!candidates.CanStoreCandidate(candidateIndex)
      || !candidates.Daughters().CanStore(daughterOffset, 2u)) {
    return false;
  }

  StoreCandidateFit(mother, candidates, candidateIndex);
  candidates.Metadata().Pdg(candidateIndex) = task.motherPdg;
  candidates.Metadata().PrimaryVertexIndex(candidateIndex) = task.primaryVertexIndex;
  candidates.Metadata().EventIndex(candidateIndex) = task.eventIndex;
  candidates.Metadata().DaughterOffset(candidateIndex) = daughterOffset;
  candidates.Metadata().DaughterCount(candidateIndex) = 2u;
  unsigned int flags = TwoDaughterCandidateFlags(task);
  if (!PassTwoDaughterPostBuildSelection(task, mother)) {
    flags |= static_cast<unsigned int>(KFGpuCandidateSelectionRejected);
  }
  candidates.Metadata().Flags(candidateIndex) = flags;
  candidates.Metadata().ChannelId(candidateIndex) = task.channelId;
  candidates.Daughters().SourceId(daughterOffset) = inputTracks.SourceId(task.firstTrack);
  candidates.Daughters().SourceId(daughterOffset + 1u) =
    inputTracks.SourceId(task.secondTrack);
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool StoreTwoDaughterCandidate(
  const KFParticleGpuCandidatePoolView& candidates,
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  unsigned int candidateIndex,
  const KFParticleGpuFitState& mother)
{
  return StoreTwoDaughterCandidate(
    candidates, inputTracks, task, candidateIndex, candidateIndex * 2u, mother);
}

KFPARTICLE_GPU_HOST_DEVICE inline bool StoreFailedTwoDaughterCandidate(
  const KFParticleGpuCandidatePoolView& candidates,
  const KFParticleGpuTwoDaughterTask& task,
  unsigned int candidateIndex)
{
  if (!candidates.CanStoreCandidate(candidateIndex)) {
    return false;
  }

  KFParticleGpuFitState emptyState;
  StoreCandidateFit(emptyState, candidates, candidateIndex);
  candidates.Metadata().Pdg(candidateIndex) = task.motherPdg;
  candidates.Metadata().PrimaryVertexIndex(candidateIndex) = task.primaryVertexIndex;
  candidates.Metadata().EventIndex(candidateIndex) = task.eventIndex;
  candidates.Metadata().DaughterOffset(candidateIndex) = candidateIndex * 2u;
  candidates.Metadata().DaughterCount(candidateIndex) = 0u;
  unsigned int flags = static_cast<unsigned int>(KFGpuCandidateBuildFailed);
  if (task.flags & KFGpuTwoDaughterUseLineDca) {
    flags |= static_cast<unsigned int>(KFGpuCandidateLineDca);
  }
  if (task.flags & KFGpuTwoDaughterUseEnergyFit) {
    flags |= static_cast<unsigned int>(KFGpuCandidateEnergyFit);
  }
  candidates.Metadata().Flags(candidateIndex) = flags;
  candidates.Metadata().ChannelId(candidateIndex) = task.channelId;
  return true;
}

#endif
