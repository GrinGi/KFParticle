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
  KFGpuTwoDaughterUseEnergyFit = 1u << 1,
  // KFParticleFinder constructs CBM V0s by adding the negative daughter first
  // even though routing conventionally enumerates positive-negative pairs.
  KFGpuTwoDaughterReverseFitOrder = 1u << 2
};

enum KFParticleGpuTransportMode
{
  KFGpuTransportStraightLine = 0,
  KFGpuTransportConstantBy = 1,
  // Compatibility name for existing diagnostic channels. It intentionally
  // preserves the former first-field-region constant-By behaviour.
  KFGpuTransportFieldAware = KFGpuTransportConstantBy,
  KFGpuTransportFullField = 2,
  // Diagnostic baseline for the Step 12 coupled covariance route. It keeps
  // full-field state transport but uses the former line-DCA correlation.
  KFGpuTransportFullFieldApprox = 3
};

constexpr int KFGpuPrimaryVertexFromDaughters = -2;

struct KFParticleGpuTwoDaughterTask
{
  unsigned int channelId = 0;
  unsigned int firstTrack = 0;
  unsigned int secondTrack = 0;
  unsigned int eventIndex = 0;
  unsigned int flags = 0;
  unsigned int outputClass = KFGpuGraphOutputPrimaryAndSecondary;
  int motherPdg = 0;
  int firstDaughterPdg = 0;
  int secondDaughterPdg = 0;
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
  unsigned int outputClass = KFGpuGraphOutputPrimaryAndSecondary;
  int motherPdg = 0;
  int firstDaughterPdg = 0;
  int secondDaughterPdg = 0;
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
  return mode == KFGpuTransportStraightLine || mode == KFGpuTransportConstantBy
         || mode == KFGpuTransportFullField || mode == KFGpuTransportFullFieldApprox;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool IsSupportedTwoDaughterEnergyFitTransportMode(int mode)
{
  return mode == KFGpuTransportStraightLine || mode == KFGpuTransportConstantBy
         || mode == KFGpuTransportFullField || mode == KFGpuTransportFullFieldApprox;
}

/** Mirrors KFParticleFinder's fast pair gate before ConstructV0. */
KFPARTICLE_GPU_HOST_DEVICE inline bool PassCpuFinderTwoDaughterPairGate(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  unsigned int firstTrackIndex,
  unsigned int secondTrackIndex,
  int transportMode,
  float maxDaughterDistance)
{
  if (maxDaughterDistance < 0.f) {
    return true;
  }
  if (transportMode != KFGpuTransportFullField || !HasFieldRegions(inputTracks)) {
    return false;
  }

  KFParticleGpuTrackState firstTrack;
  KFParticleGpuTrackState secondTrack;
  LoadTrackState(inputTracks, firstTrackIndex, firstTrack);
  LoadTrackState(inputTracks, secondTrackIndex, secondTrack);
  KFParticleGpuFitState first;
  KFParticleGpuFitState second;
  first.Initialize(firstTrack, inputTracks.Charge(firstTrackIndex), 0.f);
  second.Initialize(secondTrack, inputTracks.Charge(secondTrackIndex), 0.f);

  KFParticleGpuFieldRegion firstField;
  KFParticleGpuFieldRegion secondField;
  LoadFieldRegion(inputTracks, firstTrackIndex, firstField);
  LoadFieldRegion(inputTracks, secondTrackIndex, secondField);

  // CPU routing calls negative.GetDStoParticleFast(positive), so the field
  // and dS ordering are anchored to the second (negative) track here.
  const float by = secondField.Get(second.Z()).y;
  float dS[2] = {0.f, 0.f};
  const bool secondStraight =
    KFParticleGpuMath::Abs(by * static_cast<float>(second.Q())) < 1.e-8f;
  const bool firstStraight =
    KFParticleGpuMath::Abs(by * static_cast<float>(first.Q())) < 1.e-8f;
  if (secondStraight && firstStraight) {
    KFParticleGpuMath::GetDStoParticleLine(second, first, dS);
  }
  else {
    KFParticleGpuMath::GetDStoParticleByCpuCompatible(second, first, by, dS);
  }

  KFParticleGpuFitState secondAtDca;
  KFParticleGpuFitState firstAtDca;
  unsigned int secondStatus = KFParticleGpuMath::KFGpuFullFieldTransportInvalidInput;
  unsigned int firstStatus = KFParticleGpuMath::KFGpuFullFieldTransportInvalidInput;
  if (!KFParticleGpuMath::TransportFullField(
        second, secondField, dS[0], secondAtDca, secondStatus)
      || !KFParticleGpuMath::TransportFullField(
        first, firstField, dS[1], firstAtDca, firstStatus)
      || (secondStatus & KFParticleGpuMath::KFGpuFullFieldTransportPathLimited)
      || (firstStatus & KFParticleGpuMath::KFGpuFullFieldTransportPathLimited)) {
    return false;
  }

  const float dx = secondAtDca.X() - firstAtDca.X();
  const float dy = secondAtDca.Y() - firstAtDca.Y();
  const float dz = secondAtDca.Z() - firstAtDca.Z();
  const float distance2 = dx * dx + dy * dy + dz * dz;
  const float maxDistance2 = maxDaughterDistance * maxDaughterDistance;
  const float momentumDot = firstAtDca.Px() * secondAtDca.Px()
                            + firstAtDca.Py() * secondAtDca.Py()
                            + firstAtDca.Pz() * secondAtDca.Pz();
  const float firstMomentum2 = firstAtDca.Px() * firstAtDca.Px()
                               + firstAtDca.Py() * firstAtDca.Py()
                               + firstAtDca.Pz() * firstAtDca.Pz();
  const float secondMomentum2 = secondAtDca.Px() * secondAtDca.Px()
                                + secondAtDca.Py() * secondAtDca.Py()
                                + secondAtDca.Pz() * secondAtDca.Pz();
  return KFParticleGpuMath::IsFinite(distance2)
         && distance2 < maxDistance2
         && momentumDot > -firstMomentum2
         && momentumDot > -secondMomentum2;
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
  const int firstSourcePdg = task.firstSourcePdg != 0 ? task.firstSourcePdg : task.firstDaughterPdg;
  const int secondSourcePdg = task.secondSourcePdg != 0 ? task.secondSourcePdg : task.secondDaughterPdg;
  const int packedFirstPdg = inputTracks.Pdg(task.firstTrack);
  const int packedSecondPdg = inputTracks.Pdg(task.secondTrack);
  if (firstSourcePdg != 0 && packedFirstPdg != firstSourcePdg
      && packedFirstPdg != task.firstAlternateSourcePdg) {
    return false;
  }
  if (secondSourcePdg != 0 && packedSecondPdg != secondSourcePdg
      && packedSecondPdg != task.secondAlternateSourcePdg) {
    return false;
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool PassOptionalTrackCuts(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  unsigned int track,
  int expectedPdg,
  int alternateExpectedPdg,
  int expectedCharge,
  int minPixelHits,
  float maxChiToPrimaryVertex,
  float minChiToPrimaryVertex = -1.f,
  float minPt = -1.f)
{
  if (track >= inputTracks.Size()) {
    return false;
  }
  if (expectedPdg != 0 && inputTracks.Pdg(track) != expectedPdg
      && inputTracks.Pdg(track) != alternateExpectedPdg) {
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
  if (minChiToPrimaryVertex >= 0.f
      && inputTracks.ChiToPrimaryVertex(track) <= minChiToPrimaryVertex) {
    return false;
  }
  if (minPt >= 0.f) {
    const float px = inputTracks.Numerical().Parameter(3u, track);
    const float py = inputTracks.Numerical().Parameter(4u, track);
    if (px * px + py * py < minPt * minPt) { return false; }
  }
  return true;
}

KFPARTICLE_GPU_HOST_DEVICE inline bool PassTwoDaughterTaskSourceCuts(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTaskSource& source,
  unsigned int firstTrack,
  unsigned int secondTrack,
  float minChiToPrimaryVertex = -1.f)
{
  const bool firstIsPrimary = source.firstTrackSet == PrimaryPositiveFirst
                              || source.firstTrackSet == PrimaryNegativeFirst
                              || source.firstTrackSet == PrimaryPositiveLast
                              || source.firstTrackSet == PrimaryNegativeLast;
  const bool secondIsPrimary = source.secondTrackSet == PrimaryPositiveFirst
                               || source.secondTrackSet == PrimaryNegativeFirst
                               || source.secondTrackSet == PrimaryPositiveLast
                               || source.secondTrackSet == PrimaryNegativeLast;
  if (firstTrack == secondTrack) {
    return false;
  }
  if (firstTrack >= inputTracks.Size() || secondTrack >= inputTracks.Size()) {
    return false;
  }
  if (inputTracks.SourceId(firstTrack) == inputTracks.SourceId(secondTrack)) {
    return false;
  }
  if (source.primaryVertexIndex == KFGpuPrimaryVertexFromDaughters
      && (inputTracks.PrimaryVertexIndex(firstTrack) < 0
          || inputTracks.PrimaryVertexIndex(firstTrack)
               != inputTracks.PrimaryVertexIndex(secondTrack))) {
    return false;
  }
  return PassOptionalTrackCuts(inputTracks,
                               firstTrack,
                               source.firstSourcePdg != 0 ? source.firstSourcePdg : source.firstDaughterPdg,
                               source.firstAlternateSourcePdg,
                               source.firstCharge,
                               source.minFirstPixelHits,
                               source.maxFirstChiToPrimaryVertex,
                               source.minFirstChiToPrimaryVertex >= 0.f
                                 ? source.minFirstChiToPrimaryVertex
                                 : (firstIsPrimary ? -1.f : minChiToPrimaryVertex),
                               source.minFirstPt)
         && PassOptionalTrackCuts(inputTracks,
                                  secondTrack,
                                  source.secondSourcePdg != 0 ? source.secondSourcePdg : source.secondDaughterPdg,
                                  source.secondAlternateSourcePdg,
                                  source.secondCharge,
                                  source.minSecondPixelHits,
                                  source.maxSecondChiToPrimaryVertex,
                                  source.minSecondChiToPrimaryVertex >= 0.f
                                    ? source.minSecondChiToPrimaryVertex
                                    : (secondIsPrimary ? -1.f : minChiToPrimaryVertex),
                                  source.minSecondPt);
}

KFPARTICLE_GPU_HOST_DEVICE inline int ResolveTwoDaughterPrimaryVertexIndex(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  int configuredIndex,
  unsigned int firstTrack,
  unsigned int secondTrack)
{
  if (configuredIndex != KFGpuPrimaryVertexFromDaughters) {
    return configuredIndex;
  }
  const int firstIndex = inputTracks.PrimaryVertexIndex(firstTrack);
  return firstIndex >= 0 && firstIndex == inputTracks.PrimaryVertexIndex(secondTrack)
           ? firstIndex : -1;
}

KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuRange ResolveTaskSourceRange(
  const KFParticleGpuEventDesc& event,
  KFParticleGpuTrackSet trackSet,
  KFParticleGpuTrackSpecies species)
{
  if (species == NumberOfTrackSpecies) {
    return event.TrackSet(trackSet).tracks;
  }
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
  task.outputClass = source.outputClass;
  task.motherPdg = source.motherPdg;
  task.firstDaughterPdg = source.firstDaughterPdg;
  task.secondDaughterPdg = source.secondDaughterPdg;
  task.firstSourcePdg = source.firstSourcePdg;
  task.firstAlternateSourcePdg = source.firstAlternateSourcePdg;
  task.secondSourcePdg = source.secondSourcePdg;
  task.secondAlternateSourcePdg = source.secondAlternateSourcePdg;
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
      && task.transportMode == KFGpuTransportConstantBy) {
    const KFParticleGpuFieldValue field =
      EvaluateTrackFieldAtState(inputTracks, task.firstTrack, firstDaughter);
    if (!KFParticleGpuMath::BuildConstantByDcaKinematicMother(
          firstDaughter, secondDaughter, field.y, mother)) {
      return false;
    }
  }
  else if ((task.flags & KFGpuTwoDaughterUseLineDca)
           && (task.transportMode == KFGpuTransportFullField
               || task.transportMode == KFGpuTransportFullFieldApprox)) {
    if (!HasFieldRegions(inputTracks)) {
      return false;
    }
    KFParticleGpuFieldRegion firstField;
    KFParticleGpuFieldRegion secondField;
    LoadFieldRegion(inputTracks, task.firstTrack, firstField);
    LoadFieldRegion(inputTracks, task.secondTrack, secondField);
    if (!KFParticleGpuMath::BuildFullFieldDcaKinematicMother(
          firstDaughter, secondDaughter, firstField, secondField, mother)) {
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

// Keep the full-field path in a compact device function. Besides making the
// transport boundary explicit, this avoids carrying the unused line-DCA
// branch state through the HIP full-field energy-fit kernel path.
KFPARTICLE_GPU_HOST_DEVICE inline bool BuildFullFieldTwoDaughterEnergyFitCandidate(
  const KFParticleGpuConstInputTrackSoAView& inputTracks,
  const KFParticleGpuTwoDaughterTask& task,
  bool useCpuCompatibleDca,
  KFParticleGpuFitState& mother)
{
  const bool reverse = (task.flags & KFGpuTwoDaughterReverseFitOrder) != 0u;
  const unsigned int firstTrackIndex = reverse ? task.secondTrack : task.firstTrack;
  const unsigned int secondTrackIndex = reverse ? task.firstTrack : task.secondTrack;
  const float firstMass = reverse ? task.secondMass : task.firstMass;
  const float secondMass = reverse ? task.firstMass : task.secondMass;
  KFParticleGpuTrackState firstTrack;
  KFParticleGpuTrackState secondTrack;
  LoadTrackState(inputTracks, firstTrackIndex, firstTrack);
  LoadTrackState(inputTracks, secondTrackIndex, secondTrack);

  KFParticleGpuFitState firstDaughter;
  firstDaughter.Initialize(firstTrack, inputTracks.Charge(firstTrackIndex), firstMass);
  KFParticleGpuFitState secondDaughter;
  secondDaughter.Initialize(secondTrack, inputTracks.Charge(secondTrackIndex), secondMass);

  // CPU Construct() copies the first daughter into a fresh mother and marks
  // that one-daughter state with NDF=-1 before the second Kalman update.
  firstDaughter.NDF() = -1;

  if (!HasFieldRegions(inputTracks)) {
    return false;
  }
  KFParticleGpuFieldRegion firstField;
  KFParticleGpuFieldRegion secondField;
  LoadFieldRegion(inputTracks, firstTrackIndex, firstField);
  LoadFieldRegion(inputTracks, secondTrackIndex, secondField);

  KFParticleGpuMeasurement measurement;
  if (task.flags & KFGpuTwoDaughterUseLineDca) {
    const bool built = useCpuCompatibleDca
      ? KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCpuConstructV0(
          firstDaughter, secondDaughter, firstField, secondField, mother, measurement)
      : KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedApprox(
          firstDaughter, secondDaughter, firstField, secondField, mother, measurement);
    if (!built) {
      return false;
    }
  }
  else {
    mother = firstDaughter;
    measurement.StoreState(secondDaughter);
  }

  if (!KFParticleGpuMath::AddDaughterWithEnergyFit(
        mother, measurement, inputTracks.Charge(secondTrackIndex))) {
    return false;
  }

  // CPU ConstructV0 applies this publication guard immediately after the
  // two-daughter fit. Keep malformed covariance updates out of the compact
  // candidate pool in the exact full-field path as well.
  if (useCpuCompatibleDca
      && (!KFParticleGpuMath::IsFinite(mother.Chi2()) || mother.Chi2() <= 0.f)) {
    return false;
  }

  mother.SumDaughterMass() = firstDaughter.SumDaughterMass() + secondDaughter.SumDaughterMass();
  mother.MassHypo() = -1.f;
  mother.ConstructMethod() = 0;
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
  if (task.transportMode == KFGpuTransportFullField
      || task.transportMode == KFGpuTransportFullFieldApprox) {
    const bool built = BuildFullFieldTwoDaughterEnergyFitCandidate(
      inputTracks, task, task.transportMode == KFGpuTransportFullField, mother);
    return built;
  }

  const bool reverse = (task.flags & KFGpuTwoDaughterReverseFitOrder) != 0u;
  const unsigned int firstTrackIndex = reverse ? task.secondTrack : task.firstTrack;
  const unsigned int secondTrackIndex = reverse ? task.firstTrack : task.secondTrack;
  const float firstMass = reverse ? task.secondMass : task.firstMass;
  const float secondMass = reverse ? task.firstMass : task.secondMass;
  KFParticleGpuTrackState firstTrack;
  KFParticleGpuTrackState secondTrack;
  LoadTrackState(inputTracks, firstTrackIndex, firstTrack);
  LoadTrackState(inputTracks, secondTrackIndex, secondTrack);

  KFParticleGpuFitState firstDaughter;
  firstDaughter.Initialize(firstTrack, inputTracks.Charge(firstTrackIndex), firstMass);
  KFParticleGpuFitState secondDaughter;
  secondDaughter.Initialize(secondTrack, inputTracks.Charge(secondTrackIndex), secondMass);

  firstDaughter.NDF() = -1;

  KFParticleGpuMeasurement measurement;
  if ((task.flags & KFGpuTwoDaughterUseLineDca)
      && task.transportMode == KFGpuTransportConstantBy) {
    const KFParticleGpuFieldValue field =
      EvaluateTrackFieldAtState(inputTracks, firstTrackIndex, firstDaughter);
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
        mother, measurement, inputTracks.Charge(secondTrackIndex))) {
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
    const bool built = BuildTwoDaughterEnergyFitCandidate(inputTracks, task, mother);
    return built;
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
  candidates.Metadata().Topology(candidateIndex) = KFGpuGraphTopologyTrackTrack;
  candidates.Metadata().OutputClass(candidateIndex) = task.outputClass;
  candidates.Metadata().OperationStatus(candidateIndex) = KFGpuCandidateOperationAccepted;
  candidates.Metadata().DirectDaughterCount(candidateIndex) = 2u;
  SetCandidateDirectDaughter(candidates.Metadata(),
                             candidateIndex,
                             0u,
                             KFGpuDirectDaughterInputTrack,
                             task.firstTrack);
  SetCandidateDirectDaughter(candidates.Metadata(),
                             candidateIndex,
                             1u,
                             KFGpuDirectDaughterInputTrack,
                             task.secondTrack);
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
  candidates.Metadata().Topology(candidateIndex) = KFGpuGraphTopologyTrackTrack;
  candidates.Metadata().OutputClass(candidateIndex) = task.outputClass;
  candidates.Metadata().OperationStatus(candidateIndex) = KFGpuCandidateOperationRejected;
  ClearCandidateDirectDaughters(candidates.Metadata(), candidateIndex);
  return true;
}

#endif
