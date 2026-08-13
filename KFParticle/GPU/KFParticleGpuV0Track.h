/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUV0TRACK_H
#define KFPARTICLEGPUV0TRACK_H

#include "KFParticleGpuTwoDaughter.h"

enum KFParticleGpuV0TrackTaskFlags
{
  KFGpuV0TrackUseLineDca = 1u << 0,
  KFGpuV0TrackUseEnergyFit = 1u << 1
};

/**
 * Flat description of one selected-V0 plus bachelor-track reconstruction
 * channel.  The V0 remains in the raw candidate pool; tasks carry indices
 * only, so later cascade kernels do not duplicate its fit state.
 */
struct KFParticleGpuV0TrackChannel
{
  unsigned int channelId = 0u;
  unsigned int parentChannelId = 0u;
  unsigned int outputClass = KFGpuGraphOutputSecondary;
  int v0Pdg = 0;
  int bachelorPdg = 0;
  int motherPdg = 0;
  int primaryVertexIndex = -1;
  unsigned int eventIndex = 0u;
  KFParticleGpuRange bachelorTracks;
  unsigned int flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;
  int transportMode = KFGpuTransportFullField;
  float bachelorMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = 0.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  float maxV0TrackDistance = -1.f;
  float minBachelorChiToPrimaryVertex = -1.f;
  float minBachelorPt = -1.f;
  int minBachelorPixelHits = -1;
  float parentMassConstraint = -1.f;
  float parentMassConstraintSigma = 0.f;
};

/** A compact task with stable references into selected V0 and input-track storage. */
struct KFParticleGpuV0TrackTask
{
  unsigned int selectedV0Index = 0u;
  unsigned int v0CandidateIndex = 0u;
  unsigned int bachelorTrackIndex = 0u;
  unsigned int eventIndex = 0u;
  unsigned int channelId = 0u;
  unsigned int flags = 0u;
  int motherPdg = 0;
  int bachelorPdg = 0;
  int primaryVertexIndex = -1;
  int transportMode = KFGpuTransportFullField;
  float bachelorMass = 0.f;
  float motherMass = 0.f;
  float motherMassSigma = 0.f;
  float secondaryMassSigmaCut = -1.f;
  float maxSecondaryTopoChi2PerNdf = -1.f;
  unsigned int outputClass = KFGpuGraphOutputSecondary;
  float maxV0TrackDistance = -1.f;
  float minBachelorPt = -1.f;
  int minBachelorPixelHits = -1;
  float parentMassConstraint = -1.f;
  float parentMassConstraintSigma = 0.f;
};

/** Non-owning device input for the cascade generation boundary. */
class KFParticleGpuV0TrackInputView
{
 public:
  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackInputView() = default;

  KFPARTICLE_GPU_HOST_DEVICE KFParticleGpuV0TrackInputView(
    const KFParticleGpuSelectedV0View& selectedV0s,
    const KFParticleGpuConstInputTrackSoAView& tracks)
    : fSelectedV0s(selectedV0s), fTracks(tracks)
  {
  }

  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuSelectedV0View& SelectedV0s() const
  {
    return fSelectedV0s;
  }
  KFPARTICLE_GPU_HOST_DEVICE const KFParticleGpuConstInputTrackSoAView& Tracks() const
  {
    return fTracks;
  }

 private:
  KFParticleGpuSelectedV0View fSelectedV0s;
  KFParticleGpuConstInputTrackSoAView fTracks;
};

/** Canonical three-source identity for later comparison and duplicate rejection. */
struct KFParticleGpuV0TrackLineage
{
  static const unsigned int MaximumSize = 16u;
  int firstV0DaughterSourceId = -1;
  int secondV0DaughterSourceId = -1;
  int bachelorSourceId = -1;
  int sources[MaximumSize] = {};
  unsigned int size = 0u;

  KFPARTICLE_GPU_HOST_DEVICE int SourceId(unsigned int index) const
  {
    return sources[index];
  }

  KFPARTICLE_GPU_HOST_DEVICE void SetSourceId(unsigned int index, int source)
  {
    sources[index] = source;
  }
};

enum KFParticleGpuV0TrackTaskRejection
{
  KFGpuV0TrackTaskAccept = 0u,
  KFGpuV0TrackTaskRejectSelectedIndex = 1u << 0,
  KFGpuV0TrackTaskRejectCandidate = 1u << 1,
  KFGpuV0TrackTaskRejectEvent = 1u << 2,
  KFGpuV0TrackTaskRejectBachelorRange = 1u << 3,
  KFGpuV0TrackTaskRejectBachelorPdg = 1u << 4,
  KFGpuV0TrackTaskRejectPrimaryVertex = 1u << 5,
  KFGpuV0TrackTaskRejectDuplicateSource = 1u << 6,
  KFGpuV0TrackTaskRejectCapacity = 1u << 7,
  KFGpuV0TrackTaskRejectBachelorChiToPrimaryVertex = 1u << 8
};

namespace KFParticleGpuV0Track
{
  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel MakeChannel(
    unsigned int channelId,
    int v0Pdg,
    int bachelorPdg,
    int motherPdg,
    unsigned int eventIndex,
    const KFParticleGpuRange& bachelorTracks,
    float bachelorMass,
    float motherMass)
  {
    KFParticleGpuV0TrackChannel channel;
    channel.channelId = channelId;
    channel.v0Pdg = v0Pdg;
    channel.bachelorPdg = bachelorPdg;
    channel.motherPdg = motherPdg;
    channel.primaryVertexIndex = -1;
    channel.eventIndex = eventIndex;
    channel.bachelorTracks = bachelorTracks;
    channel.bachelorMass = bachelorMass;
    channel.motherMass = motherMass;
    return channel;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel MakeXiMinusChannel(
    unsigned int channelId, unsigned int eventIndex, const KFParticleGpuRange& bachelorTracks)
  {
    return MakeChannel(channelId, 3122, -211, 3312, eventIndex, bachelorTracks, 0.13957039f, 1.32171f);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel MakeAntiXiPlusChannel(
    unsigned int channelId, unsigned int eventIndex, const KFParticleGpuRange& bachelorTracks)
  {
    return MakeChannel(channelId, -3122, 211, -3312, eventIndex, bachelorTracks, 0.13957039f, 1.32171f);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel MakeOmegaMinusChannel(
    unsigned int channelId, unsigned int eventIndex, const KFParticleGpuRange& bachelorTracks)
  {
    return MakeChannel(channelId, 3122, -321, 3334, eventIndex, bachelorTracks, 0.493677f, 1.67245f);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline KFParticleGpuV0TrackChannel MakeAntiOmegaPlusChannel(
    unsigned int channelId, unsigned int eventIndex, const KFParticleGpuRange& bachelorTracks)
  {
    return MakeChannel(channelId, -3122, 321, -3334, eventIndex, bachelorTracks, 0.493677f, 1.67245f);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsAccepted(unsigned int rejection)
  {
    return rejection == KFGpuV0TrackTaskAccept;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool IsCanonical(const KFParticleGpuV0TrackLineage& lineage)
  {
    if (lineage.size < 3u || lineage.size > KFParticleGpuV0TrackLineage::MaximumSize) return false;
    for (unsigned int index = 1u; index < lineage.size; ++index) {
      if (lineage.SourceId(index - 1u) >= lineage.SourceId(index)) return false;
    }
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline unsigned int ResolveTask(
    const KFParticleGpuV0TrackInputView& input,
    const KFParticleGpuV0TrackChannel& channel,
    const KFParticleGpuSelectedCandidateRange& selectedRange,
    unsigned int selectedV0Index,
    unsigned int bachelorTrackIndex,
    unsigned int outputIndex,
    unsigned int outputCapacity,
    KFParticleGpuV0TrackTask& task,
    KFParticleGpuV0TrackLineage& lineage)
  {
    if (outputIndex >= outputCapacity) {
      return KFGpuV0TrackTaskRejectCapacity;
    }
    const KFParticleGpuSelectedV0View& selectedV0s = input.SelectedV0s();
    const KFParticleGpuConstInputTrackSoAView& tracks = input.Tracks();
    if (selectedV0Index >= selectedV0s.Size()
        || selectedV0Index < selectedRange.offset
        || selectedV0Index >= selectedRange.End()) {
      return KFGpuV0TrackTaskRejectSelectedIndex;
    }
    if (!channel.bachelorTracks.Contains(bachelorTrackIndex) || bachelorTrackIndex >= tracks.Size()) {
      return KFGpuV0TrackTaskRejectBachelorRange;
    }

    const unsigned int v0CandidateIndex = selectedV0s.CandidateIndex(selectedV0Index);
    if (selectedV0s.Pdg(selectedV0Index) != channel.v0Pdg
        || (channel.parentChannelId != 0u
            && selectedV0s.ChannelId(selectedV0Index) != channel.parentChannelId)
        || selectedV0s.DaughterCount(selectedV0Index) < 2u
        || selectedV0s.DaughterCount(selectedV0Index)
             >= KFParticleGpuV0TrackLineage::MaximumSize) {
      return KFGpuV0TrackTaskRejectCandidate;
    }
    if (selectedV0s.EventIndex(selectedV0Index) != channel.eventIndex) {
      return KFGpuV0TrackTaskRejectEvent;
    }
    if (tracks.Pdg(bachelorTrackIndex) != channel.bachelorPdg) {
      return KFGpuV0TrackTaskRejectBachelorPdg;
    }
    if (tracks.PrimaryVertexIndex(bachelorTrackIndex) != channel.primaryVertexIndex) {
      return KFGpuV0TrackTaskRejectPrimaryVertex;
    }
    if (channel.minBachelorChiToPrimaryVertex >= 0.f
        && tracks.ChiToPrimaryVertex(bachelorTrackIndex) < channel.minBachelorChiToPrimaryVertex) {
      return KFGpuV0TrackTaskRejectBachelorChiToPrimaryVertex;
    }
    if (channel.minBachelorPixelHits > 0
        && tracks.NumberOfPixelHits(bachelorTrackIndex) < channel.minBachelorPixelHits) {
      return KFGpuV0TrackTaskRejectBachelorRange;
    }
    if (channel.minBachelorPt >= 0.f) {
      const float px = tracks.Numerical().Parameter(3u, bachelorTrackIndex);
      const float py = tracks.Numerical().Parameter(4u, bachelorTrackIndex);
      if (px * px + py * py < channel.minBachelorPt * channel.minBachelorPt) {
        return KFGpuV0TrackTaskRejectBachelorRange;
      }
    }
    const int bachelorSource = tracks.SourceId(bachelorTrackIndex);
    const unsigned int parentSize = selectedV0s.DaughterCount(selectedV0Index);
    lineage.firstV0DaughterSourceId =
      selectedV0s.DaughterSourceId(selectedV0Index, 0u);
    lineage.secondV0DaughterSourceId =
      selectedV0s.DaughterSourceId(selectedV0Index, 1u);
    lineage.bachelorSourceId = bachelorSource;
    lineage.size = parentSize + 1u;
    unsigned int output = 0u;
    bool insertedBachelor = false;
    for (unsigned int sourceIndex = 0u; sourceIndex < parentSize; ++sourceIndex) {
      const int source = selectedV0s.DaughterSourceId(selectedV0Index, sourceIndex);
      if (source == bachelorSource
          || (sourceIndex > 0u
              && source <= selectedV0s.DaughterSourceId(selectedV0Index, sourceIndex - 1u))) {
        return KFGpuV0TrackTaskRejectDuplicateSource;
      }
      if (!insertedBachelor && bachelorSource < source) {
        lineage.SetSourceId(output++, bachelorSource);
        insertedBachelor = true;
      }
      lineage.SetSourceId(output++, source);
    }
    if (!insertedBachelor) lineage.SetSourceId(output, bachelorSource);
    task.selectedV0Index = selectedV0Index;
    task.v0CandidateIndex = v0CandidateIndex;
    task.bachelorTrackIndex = bachelorTrackIndex;
    task.eventIndex = selectedV0s.EventIndex(selectedV0Index);
    task.channelId = channel.channelId;
    task.flags = channel.flags;
    task.motherPdg = channel.motherPdg;
    task.bachelorPdg = channel.bachelorPdg;
    task.primaryVertexIndex = channel.primaryVertexIndex;
    task.transportMode = channel.transportMode;
    task.bachelorMass = channel.bachelorMass;
    task.motherMass = channel.motherMass;
    task.motherMassSigma = channel.motherMassSigma;
    task.secondaryMassSigmaCut = channel.secondaryMassSigmaCut;
    task.maxSecondaryTopoChi2PerNdf = channel.maxSecondaryTopoChi2PerNdf;
    task.outputClass = channel.outputClass;
    task.maxV0TrackDistance = channel.maxV0TrackDistance;
    task.minBachelorPt = channel.minBachelorPt;
    task.minBachelorPixelHits = channel.minBachelorPixelHits;
    task.parentMassConstraint = channel.parentMassConstraint;
    task.parentMassConstraintSigma = channel.parentMassConstraintSigma;
    return KFGpuV0TrackTaskAccept;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool ValidateTask(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuV0TrackTask& task,
    KFParticleGpuV0TrackLineage& lineage)
  {
    if (task.v0CandidateIndex >= candidates.Size()
        || task.bachelorTrackIndex >= tracks.Size()
        || candidates.Metadata().EventIndex(task.v0CandidateIndex) != task.eventIndex
        || candidates.Metadata().DaughterCount(task.v0CandidateIndex) < 2u
        || candidates.Metadata().DaughterCount(task.v0CandidateIndex)
             >= KFParticleGpuV0TrackLineage::MaximumSize
        || tracks.Pdg(task.bachelorTrackIndex) != task.bachelorPdg
        || (task.minBachelorPixelHits > 0
            && tracks.NumberOfPixelHits(task.bachelorTrackIndex)
                 < task.minBachelorPixelHits)) {
      return false;
    }
    if (task.minBachelorPt >= 0.f) {
      const float px = tracks.Numerical().Parameter(3u, task.bachelorTrackIndex);
      const float py = tracks.Numerical().Parameter(4u, task.bachelorTrackIndex);
      if (px * px + py * py < task.minBachelorPt * task.minBachelorPt) return false;
    }
    const unsigned int daughterOffset = candidates.Metadata().DaughterOffset(task.v0CandidateIndex);
    const unsigned int parentSize = candidates.Metadata().DaughterCount(task.v0CandidateIndex);
    if (!candidates.Daughters().CanStore(daughterOffset, parentSize)) {
      return false;
    }
    const int bachelorSource = tracks.SourceId(task.bachelorTrackIndex);
    lineage.firstV0DaughterSourceId = candidates.Daughters().SourceId(daughterOffset);
    lineage.secondV0DaughterSourceId = candidates.Daughters().SourceId(daughterOffset + 1u);
    lineage.bachelorSourceId = bachelorSource;
    lineage.size = parentSize + 1u;
    unsigned int output = 0u;
    bool insertedBachelor = false;
    for (unsigned int sourceIndex = 0u; sourceIndex < parentSize; ++sourceIndex) {
      const int source = candidates.Daughters().SourceId(daughterOffset + sourceIndex);
      if (source == bachelorSource
          || (sourceIndex > 0u
              && source <= candidates.Daughters().SourceId(daughterOffset + sourceIndex - 1u))) {
        return false;
      }
      if (!insertedBachelor && bachelorSource < source) {
        lineage.SetSourceId(output++, bachelorSource);
        insertedBachelor = true;
      }
      lineage.SetSourceId(output++, source);
    }
    if (!insertedBachelor) lineage.SetSourceId(output, bachelorSource);
    return IsCanonical(lineage);
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool BuildCandidate(
    const KFParticleGpuConstCandidatePoolView& candidates,
    const KFParticleGpuConstInputTrackSoAView& tracks,
    const KFParticleGpuV0TrackTask& task,
    KFParticleGpuFitState& mother,
    KFParticleGpuV0TrackLineage& lineage)
  {
    if (!ValidateTask(candidates, tracks, task, lineage)
        || !IsSupportedTwoDaughterTransportMode(task.transportMode)) {
      return false;
    }

    KFParticleGpuFitState v0;
    LoadCandidateFit(candidates, task.v0CandidateIndex, v0);
    v0.Q() = 0;
    if (task.parentMassConstraint >= 0.f
        && !KFParticleGpuMath::ApplyLinearMassConstraint(
          v0, task.parentMassConstraint, task.parentMassConstraintSigma)) {
      return false;
    }
    KFParticleGpuTrackState bachelorTrack;
    LoadTrackState(tracks, task.bachelorTrackIndex, bachelorTrack);
    KFParticleGpuFitState bachelor;
    bachelor.Initialize(bachelorTrack, tracks.Charge(task.bachelorTrackIndex), task.bachelorMass);

    if (task.flags & KFGpuV0TrackUseEnergyFit) {
      KFParticleGpuMeasurement measurement;
      if (task.flags & KFGpuV0TrackUseLineDca) {
        if (task.transportMode == KFGpuTransportFullField
            || task.transportMode == KFGpuTransportFullFieldApprox) {
          if (!HasFieldRegions(tracks)) {
            return false;
          }
          const float zeroCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {};
          const KFParticleGpuFieldRegion v0Field(zeroCoefficients);
          KFParticleGpuFieldRegion bachelorField;
          LoadFieldRegion(tracks, task.bachelorTrackIndex, bachelorField);
          const bool built = task.transportMode == KFGpuTransportFullField
            ? KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedCoupled(
                v0, bachelor, v0Field, bachelorField, mother, measurement)
            : KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedApprox(
                v0, bachelor, v0Field, bachelorField, mother, measurement);
          if (!built) {
            return false;
          }
        }
        else if (task.transportMode == KFGpuTransportConstantBy) {
          const KFParticleGpuFieldValue field =
            EvaluateTrackFieldAtState(tracks, task.bachelorTrackIndex, bachelor);
          if (!KFParticleGpuMath::BuildConstantByDcaMeasurementSeed(
                v0, bachelor, field.y, mother, measurement)) {
            return false;
          }
        }
        else if (!KFParticleGpuMath::BuildLineDcaMeasurementSeed(v0, bachelor, mother, measurement)) {
          return false;
        }
      }
      else {
        mother = v0;
        measurement.StoreState(bachelor);
      }
      if (task.maxV0TrackDistance >= 0.f) {
        const float dx = mother.X() - measurement.Parameter(0);
        const float dy = mother.Y() - measurement.Parameter(1);
        const float dz = mother.Z() - measurement.Parameter(2);
        const float maximumDistanceSquared =
          task.maxV0TrackDistance * task.maxV0TrackDistance;
        if (dx * dx + dy * dy + dz * dz >= maximumDistanceSquared) {
          return false;
        }
      }
      if (!KFParticleGpuMath::AddDaughterWithEnergyFit(
            mother, measurement, tracks.Charge(task.bachelorTrackIndex))) {
        return false;
      }
    }
    else if ((task.flags & KFGpuV0TrackUseLineDca)
             && (task.transportMode == KFGpuTransportFullField
                 || task.transportMode == KFGpuTransportFullFieldApprox)) {
      if (!HasFieldRegions(tracks)) {
        return false;
      }
      const float zeroCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {};
      const KFParticleGpuFieldRegion v0Field(zeroCoefficients);
      KFParticleGpuFieldRegion bachelorField;
      LoadFieldRegion(tracks, task.bachelorTrackIndex, bachelorField);
      if (!KFParticleGpuMath::BuildFullFieldDcaKinematicMother(
            v0, bachelor, v0Field, bachelorField, mother)) {
        return false;
      }
    }
    else if ((task.flags & KFGpuV0TrackUseLineDca)
             && task.transportMode == KFGpuTransportConstantBy) {
      const KFParticleGpuFieldValue field =
        EvaluateTrackFieldAtState(tracks, task.bachelorTrackIndex, bachelor);
      if (!KFParticleGpuMath::BuildConstantByDcaKinematicMother(v0, bachelor, field.y, mother)) {
        return false;
      }
    }
    else if (task.flags & KFGpuV0TrackUseLineDca) {
      KFParticleGpuMath::BuildLineDcaKinematicMother(v0, bachelor, mother);
    }
    else {
      KFParticleGpuMath::BuildKinematicMother(v0, bachelor, mother);
    }

    mother.SumDaughterMass() = v0.SumDaughterMass() + bachelor.SumDaughterMass();
    mother.MassHypo() = -1.f;
    mother.ConstructMethod() = 0;
    return true;
  }

  KFPARTICLE_GPU_HOST_DEVICE inline bool StoreCandidate(
    const KFParticleGpuCandidatePoolView& candidates,
    const KFParticleGpuV0TrackTask& task,
    const KFParticleGpuV0TrackLineage& lineage,
    unsigned int candidateIndex,
    unsigned int daughterOffset,
    const KFParticleGpuFitState& mother)
  {
    if (!candidates.CanStoreCandidate(candidateIndex)
        || !candidates.Daughters().CanStore(daughterOffset, lineage.size)) {
      return false;
    }
    StoreCandidateFit(mother, candidates, candidateIndex);
    candidates.Metadata().Pdg(candidateIndex) = task.motherPdg;
    candidates.Metadata().PrimaryVertexIndex(candidateIndex) = task.primaryVertexIndex;
    candidates.Metadata().EventIndex(candidateIndex) = task.eventIndex;
    candidates.Metadata().DaughterOffset(candidateIndex) = daughterOffset;
    candidates.Metadata().DaughterCount(candidateIndex) = lineage.size;
    unsigned int flags = KFGpuCandidateValid;
    if (task.flags & KFGpuV0TrackUseLineDca) {
      flags |= KFGpuCandidateLineDca;
    }
    if (task.flags & KFGpuV0TrackUseEnergyFit) {
      flags |= KFGpuCandidateEnergyFit;
    }
    if (!PassTwoDaughterPostBuildSelection(task, mother)) {
      flags |= KFGpuCandidateSelectionRejected;
    }
    candidates.Metadata().Flags(candidateIndex) = flags;
    candidates.Metadata().ChannelId(candidateIndex) = task.channelId;
    candidates.Metadata().Topology(candidateIndex) = KFGpuGraphTopologyCompositeTrack;
    candidates.Metadata().OutputClass(candidateIndex) = task.outputClass;
    candidates.Metadata().OperationStatus(candidateIndex) = KFGpuCandidateOperationAccepted;
    candidates.Metadata().DirectDaughterCount(candidateIndex) = 2u;
    SetCandidateDirectDaughter(candidates.Metadata(),
                               candidateIndex,
                               0u,
                               KFGpuDirectDaughterCandidate,
                               task.v0CandidateIndex);
    SetCandidateDirectDaughter(candidates.Metadata(),
                               candidateIndex,
                               1u,
                               KFGpuDirectDaughterInputTrack,
                               task.bachelorTrackIndex);
    for (unsigned int source = 0u; source < lineage.size; ++source) {
      candidates.Daughters().SourceId(daughterOffset + source) = lineage.SourceId(source);
    }
    return true;
  }
}

#endif
