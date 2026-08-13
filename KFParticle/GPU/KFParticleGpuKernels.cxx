/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuKernels.h"

#include "KFParticleGpuCandidateTransfer.h"
#include "KFParticleGpuInputDataTransfer.h"

XPU_EXPORT(TheKFParticleFinder);

namespace
{
  template<typename Context>
  XPU_D void RunRoundTripImpl(Context& context,
                              const KFParticleGpuConstInputTrackSoAView& inputTracks,
                              const KFParticleGpuCandidatePoolView& candidates,
                              float mass,
                              unsigned int eventIndex)
  {
    const unsigned int thread =
      static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                                + context.pos().thread_idx_x());
    const unsigned int inputSize = inputTracks.Size();
    const unsigned int candidateLimit =
      inputSize < candidates.Capacity() ? inputSize : candidates.Capacity();
    const unsigned int outputSize =
      candidateLimit < candidates.Daughters().Capacity()
        ? candidateLimit
        : candidates.Daughters().Capacity();

    if (thread == 0) {
      candidates.SizeData()[0] = outputSize;
      candidates.Daughters().SizeData()[0] = outputSize;

      unsigned int overflow = 0;
      if (inputSize > candidates.Capacity()) {
        overflow |= CandidateCapacityExceeded;
      }
      if (inputSize > candidates.Daughters().Capacity()) {
        overflow |= DaughterCapacityExceeded;
      }
      candidates.OverflowFlagsData()[0] = overflow;
    }

    if (thread >= outputSize) {
      return;
    }

    KFParticleGpuTrackState track;
    LoadTrackState(inputTracks, thread, track);

    KFParticleGpuFitState candidate;
    candidate.Initialize(track, inputTracks.Charge(thread), mass);
    StoreCandidateFit(candidate, candidates, thread);

    candidates.Metadata().Pdg(thread) = inputTracks.Pdg(thread);
    candidates.Metadata().PrimaryVertexIndex(thread) = inputTracks.PrimaryVertexIndex(thread);
    candidates.Metadata().EventIndex(thread) = eventIndex;
    candidates.Metadata().DaughterOffset(thread) = thread;
    candidates.Metadata().DaughterCount(thread) = 1;
    candidates.Metadata().Flags(thread) = 0;
    candidates.Metadata().ChannelId(thread) = 0u;
    candidates.Metadata().Topology(thread) = KFGpuGraphTopologyInvalid;
    candidates.Metadata().OutputClass(thread) = KFGpuGraphOutputInvalid;
    candidates.Metadata().OperationStatus(thread) = KFGpuCandidateOperationAccepted;
    candidates.Metadata().DirectDaughterCount(thread) = 1u;
    SetCandidateDirectDaughter(candidates.Metadata(),
                               thread,
                               0u,
                               KFGpuDirectDaughterInputTrack,
                               thread);
    candidates.Metadata().DirectSecondKind(thread) = KFGpuDirectDaughterNone;
    candidates.Metadata().DirectSecondIndex(thread) = 0u;
    candidates.Daughters().SourceId(thread) = inputTracks.SourceId(thread);
  }

  XPU_D bool ReserveBoundedCounter(unsigned int* counter,
                                   unsigned int increment,
                                   unsigned int capacity,
                                   unsigned int& offset)
  {
    unsigned int current = counter ? counter[0] : 0u;
    while (counter && current <= capacity && increment <= capacity - current) {
      const unsigned int previous = xpu::atomic_cas(counter, current, current + increment);
      if (previous == current) {
        offset = current;
        return true;
      }
      current = previous;
    }
    return false;
  }

  struct KFParticleGpuResolvedRoutingPair
  {
    unsigned int selectedV0Index = 0u;
    unsigned int bachelorTrackIndex = 0u;
    KFParticleGpuChannelMask compatibleChannels;
    KFParticleGpuChannelMask acceptedChannels;
  };

  XPU_D bool ResolveV0TrackRoutingPair(
    unsigned int thread,
    KFParticleGpuConstCandidatePoolView candidates,
    KFParticleGpuConstSelectedCandidateIndexView selected,
    KFParticleGpuConstInputTrackSoAView tracks,
    const KFParticleGpuEventDesc* events,
    KFParticleGpuV0TrackRoutingView routing,
    unsigned int eventIndex,
    unsigned int groupIndex,
    KFParticleGpuSelectedCandidateRange selectedRange,
    KFParticleGpuResolvedRoutingPair& result)
  {
    if (!events || groupIndex >= routing.GroupCount()) {
      return false;
    }
    const KFParticleGpuV0TrackExecutionGroup& group = routing.Groups()[groupIndex];
    const KFParticleGpuRange bachelorTracks =
      events[eventIndex].TrackSet(group.bachelorTrackSet).tracks;
    const unsigned int selectedBegin =
      selectedRange.offset < selected.Size() ? selectedRange.offset : selected.Size();
    const unsigned int selectedEnd =
      selectedRange.End() < selected.Size() ? selectedRange.End() : selected.Size();
    const unsigned int selectedCount = selectedEnd - selectedBegin;
    const unsigned int total = selectedCount * bachelorTracks.size;
    if (thread >= total || bachelorTracks.size == 0u) {
      return false;
    }

    result.selectedV0Index = selectedBegin + thread / bachelorTracks.size;
    result.bachelorTrackIndex =
      bachelorTracks.offset + thread % bachelorTracks.size;
    if (result.bachelorTrackIndex >= tracks.Size()) {
      return false;
    }
    const unsigned int v0CandidateIndex = selected.Index(result.selectedV0Index);
    const unsigned int parentDaughterCount =
      v0CandidateIndex < candidates.Size()
        ? candidates.Metadata().DaughterCount(v0CandidateIndex) : 0u;
    if (v0CandidateIndex >= candidates.Size()
        || candidates.Metadata().EventIndex(v0CandidateIndex) != eventIndex
        || parentDaughterCount < 2u
        || parentDaughterCount >= KFParticleGpuV0TrackLineage::MaximumSize) {
      return false;
    }

    const int selectedPdg = candidates.Metadata().Pdg(v0CandidateIndex);
    const int bachelorPdg = tracks.Pdg(result.bachelorTrackIndex);
    const unsigned int parentChannelId =
      candidates.Metadata().ChannelId(v0CandidateIndex);

    const unsigned int daughterOffset =
      candidates.Metadata().DaughterOffset(v0CandidateIndex);
    if (!candidates.Daughters().CanStore(daughterOffset, parentDaughterCount)
        || daughterOffset > candidates.Daughters().Size()
        || parentDaughterCount > candidates.Daughters().Size() - daughterOffset) {
      return false;
    }
    const int bachelorSource = tracks.SourceId(result.bachelorTrackIndex);
    for (unsigned int source = 0u; source < parentDaughterCount; ++source) {
      const int value = candidates.Daughters().SourceId(daughterOffset + source);
      if (value == bachelorSource
          || (source > 0u
              && value <= candidates.Daughters().SourceId(
                            daughterOffset + source - 1u))) return false;
    }

    if (routing.KnownChannelsByPdg(
          parentChannelId, selectedPdg, bachelorPdg).Intersected(group.channels).Empty()) {
      return false;
    }
    result.compatibleChannels = routing.CompatibleChannelsByPdg(
      parentChannelId, selectedPdg, bachelorPdg).Intersected(group.channels);
    for (unsigned int bit = result.compatibleChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = result.compatibleChannels.NextSetBit(bit + 1u)) {
      if (bit >= routing.DescriptorCount()) {
        continue;
      }
      const KFParticleGpuV0TrackRoutingDescriptor& descriptor =
        routing.Descriptors()[bit];
      if (descriptor.channelBit == bit
          && descriptor.bachelorTrackSet == group.bachelorTrackSet
          && (descriptor.parentChannelId == 0u
              || descriptor.parentChannelId == parentChannelId)
          && selectedPdg == descriptor.v0Pdg
          && bachelorPdg == descriptor.bachelorPdg
          && ((descriptor.primaryVertexIndex == KFGpuPrimaryVertexFromDaughters
               && tracks.PrimaryVertexIndex(result.bachelorTrackIndex) >= 0
               && tracks.PrimaryVertexIndex(result.bachelorTrackIndex)
                    == candidates.Metadata().PrimaryVertexIndex(v0CandidateIndex))
              || (descriptor.primaryVertexIndex != KFGpuPrimaryVertexFromDaughters
                  && tracks.PrimaryVertexIndex(result.bachelorTrackIndex)
                       == descriptor.primaryVertexIndex))
          && (descriptor.minBachelorChiToPrimaryVertex < 0.f
              || tracks.ChiToPrimaryVertex(result.bachelorTrackIndex)
                   >= descriptor.minBachelorChiToPrimaryVertex)
          && (descriptor.minBachelorPixelHits <= 0
              || tracks.NumberOfPixelHits(result.bachelorTrackIndex)
                   >= descriptor.minBachelorPixelHits)
          && (descriptor.minBachelorPt < 0.f
              || (tracks.Numerical().Parameter(3u, result.bachelorTrackIndex)
                    * tracks.Numerical().Parameter(3u, result.bachelorTrackIndex)
                  + tracks.Numerical().Parameter(4u, result.bachelorTrackIndex)
                    * tracks.Numerical().Parameter(4u, result.bachelorTrackIndex))
                   >= descriptor.minBachelorPt * descriptor.minBachelorPt)) {
        result.acceptedChannels.Set(bit);
      }
    }
    return true;
  }

  struct KFParticleGpuResolvedTwoDaughterPair
  {
    unsigned int firstTrackIndex = 0u;
    unsigned int secondTrackIndex = 0u;
    KFParticleGpuChannelMask compatibleChannels;
    KFParticleGpuChannelMask acceptedChannels;
  };

  XPU_D bool ResolveTwoDaughterRoutingPair(
    unsigned int thread,
    KFParticleGpuConstInputTrackSoAView tracks,
    const KFParticleGpuEventDesc* events,
    KFParticleGpuTwoDaughterRoutingView routing,
    unsigned int eventIndex,
    unsigned int groupIndex,
    KFParticleGpuResolvedTwoDaughterPair& result)
  {
    if (!events || groupIndex >= routing.GroupCount()) {
      return false;
    }
    const KFParticleGpuEventDesc& event = events[eventIndex];
    const KFParticleGpuTwoDaughterExecutionGroup& group =
      routing.Groups()[groupIndex];
    const KFParticleGpuRange firstTracks =
      event.TrackSet(group.firstTrackSet).tracks;
    const KFParticleGpuRange secondTracks =
      event.TrackSet(group.secondTrackSet).tracks;
    const unsigned int total = firstTracks.size * secondTracks.size;
    if (thread >= total || secondTracks.size == 0u) {
      return false;
    }

    result.firstTrackIndex = firstTracks.offset + thread / secondTracks.size;
    result.secondTrackIndex = secondTracks.offset + thread % secondTracks.size;
    if (result.firstTrackIndex >= tracks.Size()
        || result.secondTrackIndex >= tracks.Size()
        || result.firstTrackIndex == result.secondTrackIndex
        || tracks.SourceId(result.firstTrackIndex)
             == tracks.SourceId(result.secondTrackIndex)) {
      return false;
    }

    const int firstPdg = tracks.Pdg(result.firstTrackIndex);
    const int secondPdg = tracks.Pdg(result.secondTrackIndex);
    if (group.firstTrackSet == group.secondTrackSet
        && firstPdg == secondPdg
        && tracks.SourceId(result.firstTrackIndex)
             > tracks.SourceId(result.secondTrackIndex)) {
      return false;
    }

    if (routing.KnownChannelsByPdg(
          firstPdg, secondPdg).Intersected(group.channels).Empty()) {
      return false;
    }
    result.compatibleChannels =
      routing.CompatibleChannelsByPdg(firstPdg, secondPdg).Intersected(group.channels);
    for (unsigned int bit = result.compatibleChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = result.compatibleChannels.NextSetBit(bit + 1u)) {
      if (bit >= routing.DescriptorCount()) {
        continue;
      }
      const KFParticleGpuTwoDaughterRoutingDescriptor& descriptor =
        routing.Descriptors()[bit];
      const int firstPrimaryVertex =
        tracks.PrimaryVertexIndex(result.firstTrackIndex);
      const bool primaryVertexMatches =
        descriptor.primaryVertexIndex != KFGpuPrimaryVertexFromDaughters
        || (firstPrimaryVertex >= 0
            && firstPrimaryVertex
                 == tracks.PrimaryVertexIndex(result.secondTrackIndex));
      if (descriptor.channelBit == bit
          && descriptor.firstTrackSet == group.firstTrackSet
          && descriptor.secondTrackSet == group.secondTrackSet
          && (descriptor.firstSourcePdg == 0
              || descriptor.firstSourcePdg == firstPdg
              || descriptor.firstAlternateSourcePdg == firstPdg)
          && (descriptor.secondSourcePdg == 0
              || descriptor.secondSourcePdg == secondPdg
              || descriptor.secondAlternateSourcePdg == secondPdg)
          && primaryVertexMatches
          && PassOptionalTrackCuts(
            tracks,
            result.firstTrackIndex,
            descriptor.firstSourcePdg != 0
              ? descriptor.firstSourcePdg : descriptor.firstDaughterPdg,
            descriptor.firstAlternateSourcePdg,
            descriptor.firstCharge,
            descriptor.minFirstPixelHits,
            descriptor.maxFirstChiToPrimaryVertex,
            descriptor.minFirstChiToPrimaryVertex >= 0.f
              ? descriptor.minFirstChiToPrimaryVertex
              : ((descriptor.firstTrackSet == PrimaryPositiveFirst
                  || descriptor.firstTrackSet == PrimaryNegativeFirst
                  || descriptor.firstTrackSet == PrimaryPositiveLast
                  || descriptor.firstTrackSet == PrimaryNegativeLast)
                   ? -1.f : event.minSecondaryTrackChiToPrimaryVertex),
            descriptor.minFirstPt)
          && PassOptionalTrackCuts(
            tracks,
            result.secondTrackIndex,
            descriptor.secondSourcePdg != 0
              ? descriptor.secondSourcePdg : descriptor.secondDaughterPdg,
            descriptor.secondAlternateSourcePdg,
            descriptor.secondCharge,
            descriptor.minSecondPixelHits,
            descriptor.maxSecondChiToPrimaryVertex,
            descriptor.minSecondChiToPrimaryVertex >= 0.f
              ? descriptor.minSecondChiToPrimaryVertex
              : ((descriptor.secondTrackSet == PrimaryPositiveFirst
                  || descriptor.secondTrackSet == PrimaryNegativeFirst
                  || descriptor.secondTrackSet == PrimaryPositiveLast
                  || descriptor.secondTrackSet == PrimaryNegativeLast)
                   ? -1.f : event.minSecondaryTrackChiToPrimaryVertex),
            descriptor.minSecondPt)
          && PassCpuFinderTwoDaughterPairGate(
            tracks,
            result.firstTrackIndex,
            result.secondTrackIndex,
            descriptor.transportMode,
            descriptor.maxDaughterDistance)) {
        result.acceptedChannels.Set(bit);
      }
    }
    return true;
  }
}

XPU_EXPORT(KFParticleGpuRoundTrip);
XPU_D void KFParticleGpuRoundTrip::operator()(context& context,
                                              float mass,
                                              unsigned int eventIndex)
{
  context.cmem<TheKFParticleFinder>().RunRoundTrip(context, mass, eventIndex);
}

XPU_EXPORT(KFParticleGpuLaunchSmoke);
XPU_D void KFParticleGpuLaunchSmoke::operator()(context& context, unsigned int* marker)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread == 0 && marker) {
    marker[0] = 0x4b465047u; // "KFPG": proves the launched device image executed.
  }
}

XPU_EXPORT(KFParticleGpuKernelStateProbe);
XPU_D void KFParticleGpuKernelStateProbe::operator()(context& context, unsigned int* checks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0 || !checks) {
    return;
  }

  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  checks[0] = state.InputTracks().Size();
  checks[1] = state.PrimaryVertices().Size();
  checks[2] = state.Events() ? state.Events()[0].eventId : 0xffffffffu;
  checks[3] = state.TwoDaughterTaskCapacity();
  checks[4] = state.Candidates().Capacity();
  checks[5] = state.SelectedCandidates().Capacity();
  checks[6] = state.V0TrackRouting().DescriptorCount();
  checks[7] = state.V0TrackRouting().GroupCount();
  checks[8] = state.V0TrackRouting()
                .CompatibleChannels(KFGpuSelectedV0RoleLambda, KFGpuBachelorRolePiMinus)
                .Count();
  checks[9] = state.V0TrackRouting().EnabledChannelsData()
                ? state.V0TrackRouting().EnabledChannelsData()[0].Count()
                : 0u;
  checks[10] = state.TwoDaughterRouting().DescriptorCount();
  checks[11] = state.TwoDaughterRouting().GroupCount();
  checks[12] = state.TwoDaughterRouting()
                 .CompatibleChannels(KFGpuTrackRolePiPlus, KFGpuTrackRolePiMinus)
                 .Count();
  checks[13] = state.TwoDaughterRouting().EnabledChannelsData()
                 ? state.TwoDaughterRouting().EnabledChannelsData()[0].Count()
                 : 0u;
  checks[14] = state.CandidateDescriptorIndices().Capacity();
  checks[15] = state.CandidateDescriptorIndices().Capacity() > 0u
                 ? state.CandidateDescriptorIndices().Index(0u)
                 : KFParticleGpuChannelMask::InvalidBit;
  checks[16] = state.DecayGraph().NodeCount();
  checks[17] = state.DecayGraph().GroupCount();
  checks[18] = state.DecayGraph().FamilyCount();
  const KFParticleGpuGraphNode* firstNode =
    state.DecayGraph().NodeCount() > 0u ? &state.DecayGraph().Nodes()[0] : nullptr;
  checks[19] = firstNode ? firstNode->generation : 0u;
  const KFParticleGpuGraphFamilyCoverage* neutral =
    state.DecayGraph().FindFamily(KFGpuCpuFamilyNeutralMissingMass);
  checks[20] = neutral ? neutral->supportStatus : KFGpuGraphSupportStatusCount;
  checks[21] = static_cast<unsigned int>(state.DecayGraph().Revision());
  checks[22] = firstNode ? firstNode->payloadKind : KFGpuGraphPayloadNone;
  checks[23] = firstNode ? firstNode->payloadIndex : 0xffffffffu;
  const KFParticleGpuGraphOperationStorageView& graphOperations =
    state.GraphOperations();
  checks[24] = graphOperations.DescriptorCount();
  checks[25] = graphOperations.DescriptorCapacity();
  checks[26] = graphOperations.TaskCapacity();
  checks[27] = static_cast<unsigned int>(graphOperations.Revision());
  checks[28] = graphOperations.Descriptors() ? 1u : 0u;
  checks[29] = graphOperations.Tasks() ? 1u : 0u;
  checks[30] = graphOperations.Results() ? 1u : 0u;
  checks[31] = graphOperations.Visited() ? 1u : 0u;
  checks[32] = graphOperations.Accepted() ? 1u : 0u;
  checks[33] = graphOperations.Stored() ? 1u : 0u;
  checks[34] = graphOperations.Constructed() ? 1u : 0u;
  checks[35] = graphOperations.Rejected() ? 1u : 0u;
  checks[36] = graphOperations.Overflow() ? 1u : 0u;
  checks[37] = graphOperations.ChannelVisited() ? 1u : 0u;
  checks[38] = graphOperations.ChannelAccepted() ? 1u : 0u;
  checks[39] = graphOperations.ChannelStored() ? 1u : 0u;
  checks[40] = graphOperations.ChannelConstructed() ? 1u : 0u;
  checks[41] = graphOperations.ChannelRejected() ? 1u : 0u;
  const KFParticleGpuTwoDaughterGenerationStorageView& twoDaughter =
    state.TwoDaughterGeneration();
  checks[42] = twoDaughter.TaskCapacity();
  checks[43] = twoDaughter.Tasks() ? 1u : 0u;
  checks[44] = twoDaughter.VisitedPairs() ? 1u : 0u;
  checks[45] = twoDaughter.ActiveChannelBits() ? 1u : 0u;
  checks[46] = twoDaughter.AcceptedTasks() ? 1u : 0u;
  checks[47] = twoDaughter.StoredTasks() ? 1u : 0u;
  checks[48] = twoDaughter.BlockReservations() ? 1u : 0u;
  checks[49] = twoDaughter.OverflowFlags() ? 1u : 0u;
  checks[50] = twoDaughter.Selection().Capacity();
  checks[51] = twoDaughter.Selection().AcceptedData() ? 1u : 0u;
  checks[52] = twoDaughter.Selection().StoredData() ? 1u : 0u;
  checks[53] = twoDaughter.Selection().OffsetsData() ? 1u : 0u;
  checks[54] = twoDaughter.Selection().CursorsData() ? 1u : 0u;
  const KFParticleGpuV0TrackGenerationStorageView& v0Track =
    state.V0TrackGeneration();
  checks[55] = v0Track.TaskCapacity();
  checks[56] = v0Track.Tasks() ? 1u : 0u;
  checks[57] = v0Track.VisitedPairs() ? 1u : 0u;
  checks[58] = v0Track.ActiveChannelBits() ? 1u : 0u;
  checks[59] = v0Track.AcceptedTasks() ? 1u : 0u;
  checks[60] = v0Track.StoredTasks() ? 1u : 0u;
  checks[61] = v0Track.BlockReservations() ? 1u : 0u;
  checks[62] = v0Track.OverflowFlags() ? 1u : 0u;
}

XPU_EXPORT(KFParticleGpuInputLayoutProbe);
XPU_D void KFParticleGpuInputLayoutProbe::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  KFParticleGpuConstVertexSoAView primaryVertices,
  const KFParticleGpuEventDesc* events,
  float* floatChecks,
  int* integerChecks,
  unsigned int* unsignedChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0) {
    return;
  }

  const unsigned int track = inputTracks.Size() > 1 ? 1 : 0;
  const unsigned int vertex = primaryVertices.Size() > 0 ? 0 : 0;
  floatChecks[0] = inputTracks.Numerical().Parameter(0, track);
  floatChecks[1] = inputTracks.Numerical().Parameter(3, track);
  floatChecks[2] = inputTracks.Numerical().Covariance(0, track);
  floatChecks[3] = inputTracks.FieldCoefficientsData()
                     ? inputTracks.FieldCoefficient(0, track)
                     : -1.f;
  floatChecks[4] = inputTracks.ChiToPrimaryVertex(track);
  floatChecks[5] = primaryVertices.Parameter(0, vertex);
  floatChecks[6] = primaryVertices.Covariance(0, vertex);
  floatChecks[7] = primaryVertices.Chi2(vertex);
  const KFParticleGpuFieldValue fieldAtTrack =
    EvaluateTrackField(inputTracks, track, inputTracks.Numerical().Parameter(2, track));
  floatChecks[8] = fieldAtTrack.x;
  floatChecks[9] = fieldAtTrack.y;
  floatChecks[10] = fieldAtTrack.z;

  integerChecks[0] = inputTracks.SourceId(track);
  integerChecks[1] = inputTracks.Pdg(track);
  integerChecks[2] = inputTracks.Charge(track);
  integerChecks[3] = inputTracks.PrimaryVertexIndex(track);
  integerChecks[4] = inputTracks.NumberOfPixelHits(track);
  integerChecks[5] = primaryVertices.NDF(vertex);
  integerChecks[6] = primaryVertices.NContributors(vertex);
  integerChecks[7] = events ? static_cast<int>(events[0].eventId) : -1;

  unsignedChecks[0] = inputTracks.Size();
  unsignedChecks[1] = inputTracks.Stride();
  unsignedChecks[2] = primaryVertices.Size();
  unsignedChecks[3] = primaryVertices.Stride();
  unsignedChecks[4] = events ? events[0].TrackSet(SecondaryPositiveFirst).tracks.offset : 0;
  unsignedChecks[5] = events ? events[0].TrackSet(SecondaryPositiveFirst).tracks.End() : 0;
  unsignedChecks[6] = events ? events[0].TrackSet(SecondaryPositiveFirst).Species(Pion).offset : 0;
  unsignedChecks[7] = events ? events[0].primaryVertices.End() : 0;
  unsignedChecks[8] = HasFieldRegions(inputTracks) ? 1u : 0u;
}

XPU_EXPORT(KFParticleGpuV0TrackTaskProbe);
XPU_D void KFParticleGpuV0TrackTaskProbe::operator()(
  context& context,
  KFParticleGpuConstCandidatePoolView candidates,
  KFParticleGpuConstSelectedCandidateIndexView selected,
  KFParticleGpuConstInputTrackSoAView tracks,
  KFParticleGpuV0TrackChannel channel,
  KFParticleGpuSelectedCandidateRange selectedRange,
  unsigned int selectedV0Index,
  unsigned int bachelorTrackIndex,
  unsigned int* rejection,
  KFParticleGpuV0TrackTask* task,
  KFParticleGpuV0TrackLineage* lineage)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !rejection || !task || !lineage) {
    return;
  }

  const KFParticleGpuSelectedV0View selectedV0s = MakeSelectedV0View(candidates, selected);
  const KFParticleGpuV0TrackInputView input(selectedV0s, tracks);
  rejection[0] = KFParticleGpuV0Track::ResolveTask(
    input, channel, selectedRange, selectedV0Index, bachelorTrackIndex, 0u, 1u,
    task[0], lineage[0]);
}

XPU_EXPORT(KFParticleGpuGenerateV0TrackTasksCompact);
XPU_D void KFParticleGpuGenerateV0TrackTasksCompact::operator()(
  context& context,
  KFParticleGpuConstCandidatePoolView candidates,
  KFParticleGpuConstSelectedCandidateIndexView selected,
  KFParticleGpuConstInputTrackSoAView tracks,
  KFParticleGpuV0TrackChannel channel,
  KFParticleGpuSelectedCandidateRange selectedRange,
  KFParticleGpuV0TrackTask* tasks,
  unsigned int taskCapacity,
  unsigned int* acceptedTasks,
  unsigned int* totalPairs,
  unsigned int* overflowFlags)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !acceptedTasks || !totalPairs) {
    return;
  }
  const unsigned int selectedBegin = selectedRange.offset < selected.Size()
    ? selectedRange.offset
    : selected.Size();
  const unsigned int selectedEnd = selectedRange.End() < selected.Size()
    ? selectedRange.End()
    : selected.Size();
  const unsigned int selectedCount = selectedEnd - selectedBegin;
  const unsigned int total = selectedCount * channel.bachelorTracks.size;
  if (thread >= total || channel.bachelorTracks.size == 0u) {
    return;
  }

  const unsigned int selectedV0Index = selectedBegin + thread / channel.bachelorTracks.size;
  const unsigned int bachelorTrackIndex =
    channel.bachelorTracks.offset + thread % channel.bachelorTracks.size;
  const KFParticleGpuV0TrackInputView input(MakeSelectedV0View(candidates, selected), tracks);
  if (input.SelectedV0s().Pdg(selectedV0Index) != channel.v0Pdg
      || input.SelectedV0s().EventIndex(selectedV0Index) != channel.eventIndex) {
    return;
  }
  // Count only event-local, channel-compatible selected V0/bachelor pairs.
  // This keeps multi-event batch monitoring equivalent to serial execution.
  xpu::atomic_add(totalPairs, 1u);
  KFParticleGpuV0TrackTask task;
  KFParticleGpuV0TrackLineage lineage;
  if (!KFParticleGpuV0Track::IsAccepted(KFParticleGpuV0Track::ResolveTask(
        input, channel, selectedRange, selectedV0Index, bachelorTrackIndex, 0u, taskCapacity,
        task, lineage))) {
    return;
  }
  const unsigned int slot = xpu::atomic_add(acceptedTasks, 1u);
  if (slot < taskCapacity) {
    tasks[slot] = task;
  }
  else if (overflowFlags) {
    xpu::atomic_or(overflowFlags, static_cast<unsigned int>(CandidateCapacityExceeded));
  }
}

XPU_EXPORT(KFParticleGpuV0TrackCompactCandidatePoolKernel);
XPU_D void KFParticleGpuV0TrackCompactCandidatePoolKernel::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView tracks,
  KFParticleGpuConstCandidatePoolView inputCandidates,
  const KFParticleGpuV0TrackTask* tasks,
  unsigned int taskCapacity,
  const unsigned int* acceptedTasks,
  KFParticleGpuCandidatePoolView outputCandidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !acceptedTasks || thread >= taskCapacity || thread >= acceptedTasks[0]) {
    return;
  }

  const KFParticleGpuV0TrackTask task = tasks[thread];
  KFParticleGpuFitState mother;
  KFParticleGpuV0TrackLineage lineage;
  if (!KFParticleGpuV0Track::BuildCandidate(inputCandidates, tracks, task, mother, lineage)) {
    return;
  }

  unsigned int candidateIndex = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.SizeData(), 1u, outputCandidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(), static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }
  unsigned int daughterOffset = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.Daughters().SizeData(), 3u,
        outputCandidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(), static_cast<unsigned int>(DaughterCapacityExceeded));
    outputCandidates.Metadata().Pdg(candidateIndex) = task.motherPdg;
    outputCandidates.Metadata().PrimaryVertexIndex(candidateIndex) = task.primaryVertexIndex;
    outputCandidates.Metadata().EventIndex(candidateIndex) = task.eventIndex;
    outputCandidates.Metadata().DaughterOffset(candidateIndex) = 0u;
    outputCandidates.Metadata().DaughterCount(candidateIndex) = 0u;
    outputCandidates.Metadata().Flags(candidateIndex) = KFGpuCandidateBuildFailed;
    outputCandidates.Metadata().ChannelId(candidateIndex) = task.channelId;
    outputCandidates.Metadata().Topology(candidateIndex) = KFGpuGraphTopologyTrackTrack;
    outputCandidates.Metadata().OutputClass(candidateIndex) =
      KFGpuGraphOutputPrimaryAndSecondary;
    outputCandidates.Metadata().OperationStatus(candidateIndex) =
      KFGpuCandidateOperationRejected;
    ClearCandidateDirectDaughters(outputCandidates.Metadata(), candidateIndex);
    return;
  }
  KFParticleGpuV0Track::StoreCandidate(
    outputCandidates, task, lineage, candidateIndex, daughterOffset, mother);
}

XPU_EXPORT(KFParticleGpuRouteV0TrackTasksAtomic);
XPU_D void KFParticleGpuRouteV0TrackTasksAtomic::operator()(
  context& context,
  KFParticleGpuConstCandidatePoolView candidates,
  KFParticleGpuConstSelectedCandidateIndexView selected,
  KFParticleGpuConstInputTrackSoAView tracks,
  const KFParticleGpuEventDesc* events,
  KFParticleGpuV0TrackRoutingView routing,
  unsigned int eventIndex,
  unsigned int groupIndex,
  KFParticleGpuSelectedCandidateRange selectedRange,
  KFParticleGpuV0TrackRoutedTask* tasks,
  unsigned int taskCapacity,
  unsigned int* visitedPairs,
  unsigned int* activeChannelBits,
  unsigned int* acceptedTasks,
  unsigned int* storedTasks,
  unsigned int* blockReservations,
  unsigned int* overflowFlags)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !visitedPairs || !activeChannelBits || !acceptedTasks
      || !storedTasks || !blockReservations) {
    return;
  }
  KFParticleGpuResolvedRoutingPair pair;
  if (!ResolveV0TrackRoutingPair(
        thread, candidates, selected, tracks, events, routing, eventIndex,
        groupIndex, selectedRange, pair)) {
    return;
  }
  xpu::atomic_add(visitedPairs, 1u);
  xpu::atomic_add(activeChannelBits, pair.compatibleChannels.Count());
  for (unsigned int bit = pair.compatibleChannels.NextSetBit(0u);
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.compatibleChannels.NextSetBit(bit + 1u)) {
    if (routing.ChannelVisitedCounters()) {
      xpu::atomic_add(routing.ChannelVisitedCounters() + bit, 1u);
    }
  }
  for (unsigned int bit = pair.acceptedChannels.NextSetBit(0u);
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.acceptedChannels.NextSetBit(bit + 1u)) {
    if (routing.ChannelAcceptedCounters()) {
      xpu::atomic_add(routing.ChannelAcceptedCounters() + bit, 1u);
    }
    const unsigned int slot = xpu::atomic_add(acceptedTasks, 1u);
    xpu::atomic_add(blockReservations, 1u);
    if (slot < taskCapacity) {
      KFParticleGpuV0TrackRoutedTask task;
      task.descriptorIndex = bit;
      task.selectedV0Index = pair.selectedV0Index;
      task.bachelorTrackIndex = pair.bachelorTrackIndex;
      task.eventIndex = eventIndex;
      tasks[slot] = task;
      xpu::atomic_add(storedTasks, 1u);
      if (routing.ChannelStoredCounters()) {
        xpu::atomic_add(routing.ChannelStoredCounters() + bit, 1u);
      }
    }
    else if (overflowFlags) {
      xpu::atomic_or(
        overflowFlags, static_cast<unsigned int>(CandidateCapacityExceeded));
    }
  }
}

XPU_EXPORT(KFParticleGpuRouteV0TrackTasksBlockScan);
XPU_D void KFParticleGpuRouteV0TrackTasksBlockScan::operator()(
  context& context,
  unsigned int eventIndex,
  unsigned int groupIndex,
  KFParticleGpuSelectedCandidateRange selectedRange,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstCandidatePoolView candidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(state.SelectedCandidates());
  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  const KFParticleGpuEventDesc* events = state.Events();
  const KFParticleGpuV0TrackRoutingView& routing = state.V0TrackRouting();
  const KFParticleGpuV0TrackGenerationStorageView& generation =
    state.V0TrackGeneration();
  KFParticleGpuV0TrackRoutedTask* tasks = generation.Tasks();
  const unsigned int taskCapacity =
    taskLimit < generation.TaskCapacity() ? taskLimit : generation.TaskCapacity();
  unsigned int* visitedPairs = generation.VisitedPairs();
  unsigned int* activeChannelBits = generation.ActiveChannelBits();
  unsigned int* acceptedTasks = generation.AcceptedTasks();
  unsigned int* storedTasks = generation.StoredTasks();
  unsigned int* blockReservations = generation.BlockReservations();
  unsigned int* overflowFlags = generation.OverflowFlags();
  KFParticleGpuResolvedRoutingPair pair;
  const bool pairValid = tasks && visitedPairs && activeChannelBits && acceptedTasks
                         && storedTasks && blockReservations
                         && ResolveV0TrackRoutingPair(
                           thread, candidates, selected, tracks, events, routing,
                           eventIndex, groupIndex, selectedRange, pair);
  const unsigned int acceptedCount =
    pairValid ? pair.acceptedChannels.Count() : 0u;
  if (pairValid) {
    xpu::atomic_add(visitedPairs, 1u);
    xpu::atomic_add(activeChannelBits, pair.compatibleChannels.Count());
    for (unsigned int bit = pair.compatibleChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = pair.compatibleChannels.NextSetBit(bit + 1u)) {
      if (routing.ChannelVisitedCounters()) {
        xpu::atomic_add(routing.ChannelVisitedCounters() + bit, 1u);
      }
    }
    for (unsigned int bit = pair.acceptedChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = pair.acceptedChannels.NextSetBit(bit + 1u)) {
      if (routing.ChannelAcceptedCounters()) {
        xpu::atomic_add(routing.ChannelAcceptedCounters() + bit, 1u);
      }
    }
  }

  scan_t scan(context.pos(), context.smem().scan);
  unsigned int blockOffset = 0u;
  scan.exclusive_sum(acceptedCount, blockOffset);
  if (context.pos().thread_idx_x() + 1 == context.pos().block_dim_x()) {
    context.smem().blockTotal = blockOffset + acceptedCount;
  }
  xpu::barrier(context);
  if (context.pos().thread_idx_x() == 0) {
    const unsigned int blockTotal = context.smem().blockTotal;
    unsigned int blockBase = acceptedTasks[0];
    if (blockTotal > 0u) {
      blockBase = xpu::atomic_add(acceptedTasks, blockTotal);
      xpu::atomic_add(blockReservations, 1u);
      const unsigned int storedInBlock =
        blockBase < taskCapacity
          ? ((blockTotal < taskCapacity - blockBase)
               ? blockTotal : taskCapacity - blockBase)
          : 0u;
      if (storedInBlock > 0u) {
        xpu::atomic_add(storedTasks, storedInBlock);
      }
      if (storedInBlock < blockTotal && overflowFlags) {
        xpu::atomic_or(
          overflowFlags, static_cast<unsigned int>(CandidateCapacityExceeded));
      }
    }
    context.smem().blockBase = blockBase;
  }
  xpu::barrier(context);

  unsigned int localTask = 0u;
  for (unsigned int bit = pairValid
                            ? pair.acceptedChannels.NextSetBit(0u)
                            : KFParticleGpuChannelMask::InvalidBit;
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.acceptedChannels.NextSetBit(bit + 1u), ++localTask) {
    const unsigned int slot = context.smem().blockBase + blockOffset + localTask;
    if (slot >= taskCapacity) {
      continue;
    }
    KFParticleGpuV0TrackRoutedTask task;
    task.descriptorIndex = bit;
    task.selectedV0Index = pair.selectedV0Index;
    task.bachelorTrackIndex = pair.bachelorTrackIndex;
    task.eventIndex = eventIndex;
    tasks[slot] = task;
    if (routing.ChannelStoredCounters()) {
      xpu::atomic_add(routing.ChannelStoredCounters() + bit, 1u);
    }
  }
}

XPU_EXPORT(KFParticleGpuV0TrackRoutedCandidatePoolKernel);
XPU_D void KFParticleGpuV0TrackRoutedCandidatePoolKernel::operator()(
  context& context,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  const KFParticleGpuConstCandidatePoolView inputCandidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstSelectedCandidateIndexView selected =
    MakeConstView(state.SelectedCandidates());
  const KFParticleGpuV0TrackRoutingView& routing = state.V0TrackRouting();
  const KFParticleGpuV0TrackGenerationStorageView& generation =
    state.V0TrackGeneration();
  const KFParticleGpuV0TrackRoutedTask* tasks = generation.Tasks();
  const unsigned int taskCapacity =
    taskLimit < generation.TaskCapacity() ? taskLimit : generation.TaskCapacity();
  const unsigned int* storedTasks = generation.StoredTasks();
  const KFParticleGpuCandidatePoolView& outputCandidates = state.Candidates();
  if (!tasks || !storedTasks || thread >= taskCapacity || thread >= storedTasks[0]) {
    return;
  }

  const KFParticleGpuV0TrackRoutedTask routed = tasks[thread];
  if (routed.descriptorIndex >= routing.DescriptorCount()
      || routed.selectedV0Index >= selected.Size()) {
    return;
  }
  const KFParticleGpuV0TrackRoutingDescriptor& descriptor =
    routing.Descriptors()[routed.descriptorIndex];
  KFParticleGpuV0TrackTask task;
  task.selectedV0Index = routed.selectedV0Index;
  task.v0CandidateIndex = selected.Index(routed.selectedV0Index);
  task.bachelorTrackIndex = routed.bachelorTrackIndex;
  task.eventIndex = routed.eventIndex;
  task.channelId = descriptor.channelId;
  task.flags = descriptor.flags;
  task.motherPdg = descriptor.motherPdg;
  task.bachelorPdg = descriptor.bachelorPdg;
  task.primaryVertexIndex = descriptor.primaryVertexIndex == KFGpuPrimaryVertexFromDaughters
    ? inputCandidates.Metadata().PrimaryVertexIndex(task.v0CandidateIndex)
    : descriptor.primaryVertexIndex;
  task.transportMode = descriptor.transportMode;
  task.bachelorMass = descriptor.bachelorMass;
  task.motherMass = descriptor.motherMass;
  task.motherMassSigma = descriptor.motherMassSigma;
  task.secondaryMassSigmaCut = descriptor.secondaryMassSigmaCut;
  task.maxSecondaryTopoChi2PerNdf = descriptor.maxSecondaryTopoChi2PerNdf;
  task.outputClass = descriptor.outputClass;
  task.maxV0TrackDistance = descriptor.maxV0TrackDistance;
  task.minBachelorPt = descriptor.minBachelorPt;
  task.minBachelorPixelHits = descriptor.minBachelorPixelHits;
  task.parentMassConstraint = descriptor.parentMassConstraint;
  task.parentMassConstraintSigma = descriptor.parentMassConstraintSigma;

  KFParticleGpuFitState mother;
  KFParticleGpuV0TrackLineage lineage;
  if (!KFParticleGpuV0Track::BuildCandidate(
        inputCandidates, tracks, task, mother, lineage)) {
    return;
  }

  unsigned int candidateIndex = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.SizeData(), 1u, outputCandidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(), static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }
  unsigned int daughterOffset = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.Daughters().SizeData(), lineage.size,
        outputCandidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(), static_cast<unsigned int>(DaughterCapacityExceeded));
    outputCandidates.Metadata().Pdg(candidateIndex) = task.motherPdg;
    outputCandidates.Metadata().PrimaryVertexIndex(candidateIndex) = task.primaryVertexIndex;
    outputCandidates.Metadata().EventIndex(candidateIndex) = task.eventIndex;
    outputCandidates.Metadata().DaughterOffset(candidateIndex) = 0u;
    outputCandidates.Metadata().DaughterCount(candidateIndex) = 0u;
    outputCandidates.Metadata().Flags(candidateIndex) = KFGpuCandidateBuildFailed;
    outputCandidates.Metadata().ChannelId(candidateIndex) = task.channelId;
    outputCandidates.Metadata().Topology(candidateIndex) =
      KFGpuGraphTopologyCompositeTrack;
    outputCandidates.Metadata().OutputClass(candidateIndex) = KFGpuGraphOutputSecondary;
    outputCandidates.Metadata().OperationStatus(candidateIndex) =
      KFGpuCandidateOperationRejected;
    ClearCandidateDirectDaughters(outputCandidates.Metadata(), candidateIndex);
    return;
  }
  if (KFParticleGpuV0Track::StoreCandidate(
        outputCandidates, task, lineage, candidateIndex, daughterOffset, mother)) {
    if (routing.ChannelConstructedCounters()) {
      xpu::atomic_add(
        routing.ChannelConstructedCounters() + routed.descriptorIndex, 1u);
    }
    if ((outputCandidates.Metadata().Flags(candidateIndex)
         & KFGpuCandidateSelectionRejected) == 0u) {
      const KFParticleGpuSelectedCandidateIndexView& selected =
        state.SelectedCandidates();
      unsigned int selectedIndex = 0u;
      if (ReserveBoundedCounter(
            selected.SizeData(), 1u, selected.Capacity(), selectedIndex)) {
        selected.Index(selectedIndex) = candidateIndex;
        if (selected.ChannelIdsData()) selected.ChannelId(selectedIndex) = task.channelId;
      }
      else if (selected.OverflowFlagsData()) {
        xpu::atomic_or(selected.OverflowFlagsData(),
                       static_cast<unsigned int>(
                         KFGpuSelectedCandidateCapacityExceeded));
      }
    }
  }
}

XPU_EXPORT(KFParticleGpuResetV0TrackGenerationState);
XPU_D void KFParticleGpuResetV0TrackGenerationState::operator()(
  context& context,
  unsigned int resetChannelCounters)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuV0TrackRoutingView& routing = state.V0TrackRouting();
  const KFParticleGpuV0TrackGenerationStorageView& generation =
    state.V0TrackGeneration();
  if (thread == 0u) {
    if (generation.VisitedPairs()) generation.VisitedPairs()[0] = 0u;
    if (generation.ActiveChannelBits()) generation.ActiveChannelBits()[0] = 0u;
    if (generation.AcceptedTasks()) generation.AcceptedTasks()[0] = 0u;
    if (generation.StoredTasks()) generation.StoredTasks()[0] = 0u;
    if (generation.BlockReservations()) generation.BlockReservations()[0] = 0u;
    if (generation.OverflowFlags()) generation.OverflowFlags()[0] = 0u;
  }
  if (resetChannelCounters != 0u && thread < routing.DescriptorCount()) {
    if (routing.ChannelVisitedCounters()) {
      routing.ChannelVisitedCounters()[thread] = 0u;
    }
    if (routing.ChannelAcceptedCounters()) {
      routing.ChannelAcceptedCounters()[thread] = 0u;
    }
    if (routing.ChannelStoredCounters()) {
      routing.ChannelStoredCounters()[thread] = 0u;
    }
    if (routing.ChannelConstructedCounters()) {
      routing.ChannelConstructedCounters()[thread] = 0u;
    }
  }
}

XPU_EXPORT(KFParticleGpuRouteTwoDaughterTasksAtomic);
XPU_D void KFParticleGpuRouteTwoDaughterTasksAtomic::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView tracks,
  const KFParticleGpuEventDesc* events,
  KFParticleGpuTwoDaughterRoutingView routing,
  unsigned int eventIndex,
  unsigned int groupIndex,
  KFParticleGpuTwoDaughterRoutedTask* tasks,
  unsigned int taskCapacity,
  unsigned int* visitedPairs,
  unsigned int* activeChannelBits,
  unsigned int* acceptedTasks,
  unsigned int* storedTasks,
  unsigned int* blockReservations,
  unsigned int* overflowFlags)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !visitedPairs || !activeChannelBits || !acceptedTasks
      || !storedTasks || !blockReservations) {
    return;
  }
  KFParticleGpuResolvedTwoDaughterPair pair;
  if (!ResolveTwoDaughterRoutingPair(
        thread, tracks, events, routing, eventIndex, groupIndex, pair)) {
    return;
  }
  xpu::atomic_add(visitedPairs, 1u);
  xpu::atomic_add(activeChannelBits, pair.compatibleChannels.Count());
  for (unsigned int bit = pair.compatibleChannels.NextSetBit(0u);
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.compatibleChannels.NextSetBit(bit + 1u)) {
    if (routing.ChannelVisitedCounters()) {
      xpu::atomic_add(routing.ChannelVisitedCounters() + bit, 1u);
    }
  }
  for (unsigned int bit = pair.acceptedChannels.NextSetBit(0u);
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.acceptedChannels.NextSetBit(bit + 1u)) {
    if (routing.ChannelAcceptedCounters()) {
      xpu::atomic_add(routing.ChannelAcceptedCounters() + bit, 1u);
    }
    const unsigned int slot = xpu::atomic_add(acceptedTasks, 1u);
    xpu::atomic_add(blockReservations, 1u);
    if (slot < taskCapacity) {
      KFParticleGpuTwoDaughterRoutedTask task;
      task.descriptorIndex = bit;
      task.firstTrackIndex = pair.firstTrackIndex;
      task.secondTrackIndex = pair.secondTrackIndex;
      task.eventIndex = eventIndex;
      tasks[slot] = task;
      xpu::atomic_add(storedTasks, 1u);
      if (routing.ChannelStoredCounters()) {
        xpu::atomic_add(routing.ChannelStoredCounters() + bit, 1u);
      }
    }
    else if (overflowFlags) {
      xpu::atomic_or(
        overflowFlags, static_cast<unsigned int>(CandidateCapacityExceeded));
    }
  }
}

XPU_EXPORT(KFParticleGpuRouteTwoDaughterTasksBlockScan);
XPU_D void KFParticleGpuRouteTwoDaughterTasksBlockScan::operator()(
  context& context,
  unsigned int eventIndex,
  unsigned int groupIndex,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  const KFParticleGpuEventDesc* events = state.Events();
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuTwoDaughterGenerationStorageView& generation =
    state.TwoDaughterGeneration();
  KFParticleGpuTwoDaughterRoutedTask* tasks = generation.Tasks();
  const unsigned int taskCapacity =
    taskLimit < generation.TaskCapacity() ? taskLimit : generation.TaskCapacity();
  unsigned int* visitedPairs = generation.VisitedPairs();
  unsigned int* activeChannelBits = generation.ActiveChannelBits();
  unsigned int* acceptedTasks = generation.AcceptedTasks();
  unsigned int* storedTasks = generation.StoredTasks();
  unsigned int* blockReservations = generation.BlockReservations();
  unsigned int* overflowFlags = generation.OverflowFlags();
  KFParticleGpuResolvedTwoDaughterPair pair;
  const bool pairValid = tasks && visitedPairs && activeChannelBits && acceptedTasks
                         && storedTasks && blockReservations
                         && ResolveTwoDaughterRoutingPair(
                           thread, tracks, events, routing, eventIndex,
                           groupIndex, pair);
  const unsigned int acceptedCount =
    pairValid ? pair.acceptedChannels.Count() : 0u;
  if (pairValid) {
    xpu::atomic_add(visitedPairs, 1u);
    xpu::atomic_add(activeChannelBits, pair.compatibleChannels.Count());
    for (unsigned int bit = pair.compatibleChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = pair.compatibleChannels.NextSetBit(bit + 1u)) {
      if (routing.ChannelVisitedCounters()) {
        xpu::atomic_add(routing.ChannelVisitedCounters() + bit, 1u);
      }
    }
    for (unsigned int bit = pair.acceptedChannels.NextSetBit(0u);
         bit != KFParticleGpuChannelMask::InvalidBit;
         bit = pair.acceptedChannels.NextSetBit(bit + 1u)) {
      if (routing.ChannelAcceptedCounters()) {
        xpu::atomic_add(routing.ChannelAcceptedCounters() + bit, 1u);
      }
    }
  }

  scan_t scan(context.pos(), context.smem().scan);
  unsigned int blockOffset = 0u;
  scan.exclusive_sum(acceptedCount, blockOffset);
  if (context.pos().thread_idx_x() + 1 == context.pos().block_dim_x()) {
    context.smem().blockTotal = blockOffset + acceptedCount;
  }
  xpu::barrier(context);
  if (context.pos().thread_idx_x() == 0u) {
    const unsigned int blockTotal = context.smem().blockTotal;
    unsigned int blockBase = acceptedTasks[0];
    if (blockTotal > 0u) {
      blockBase = xpu::atomic_add(acceptedTasks, blockTotal);
      xpu::atomic_add(blockReservations, 1u);
      const unsigned int storedInBlock =
        blockBase < taskCapacity
          ? ((blockTotal < taskCapacity - blockBase)
               ? blockTotal : taskCapacity - blockBase)
          : 0u;
      if (storedInBlock > 0u) {
        xpu::atomic_add(storedTasks, storedInBlock);
      }
      if (storedInBlock < blockTotal && overflowFlags) {
        xpu::atomic_or(
          overflowFlags, static_cast<unsigned int>(CandidateCapacityExceeded));
      }
    }
    context.smem().blockBase = blockBase;
  }
  xpu::barrier(context);

  unsigned int localTask = 0u;
  for (unsigned int bit = pairValid
                            ? pair.acceptedChannels.NextSetBit(0u)
                            : KFParticleGpuChannelMask::InvalidBit;
       bit != KFParticleGpuChannelMask::InvalidBit;
       bit = pair.acceptedChannels.NextSetBit(bit + 1u), ++localTask) {
    const unsigned int slot = context.smem().blockBase + blockOffset + localTask;
    if (slot >= taskCapacity) {
      continue;
    }
    KFParticleGpuTwoDaughterRoutedTask task;
    task.descriptorIndex = bit;
    task.firstTrackIndex = pair.firstTrackIndex;
    task.secondTrackIndex = pair.secondTrackIndex;
    task.eventIndex = eventIndex;
    tasks[slot] = task;
    if (routing.ChannelStoredCounters()) {
      xpu::atomic_add(routing.ChannelStoredCounters() + bit, 1u);
    }
  }
}

XPU_EXPORT(KFParticleGpuTwoDaughterRoutedCandidatePoolKernel);
XPU_D void KFParticleGpuTwoDaughterRoutedCandidatePoolKernel::operator()(
  context& context,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuTwoDaughterGenerationStorageView& generation =
    state.TwoDaughterGeneration();
  const KFParticleGpuTwoDaughterRoutedTask* tasks = generation.Tasks();
  const unsigned int taskCapacity =
    taskLimit < generation.TaskCapacity() ? taskLimit : generation.TaskCapacity();
  const unsigned int* storedTasks = generation.StoredTasks();
  const KFParticleGpuCandidatePoolView& candidates = state.Candidates();
  const KFParticleGpuCandidateDescriptorIndexView& descriptorIndices =
    state.CandidateDescriptorIndices();
  if (!tasks || !storedTasks || thread >= taskCapacity || thread >= storedTasks[0]) {
    return;
  }

  const KFParticleGpuTwoDaughterRoutedTask routed = tasks[thread];
  if (routed.descriptorIndex >= routing.DescriptorCount()) {
    return;
  }
  const KFParticleGpuTwoDaughterRoutingDescriptor& descriptor =
    routing.Descriptors()[routed.descriptorIndex];
  KFParticleGpuTwoDaughterTask task;
  task.channelId = descriptor.channelId;
  task.firstTrack = routed.firstTrackIndex;
  task.secondTrack = routed.secondTrackIndex;
  task.eventIndex = routed.eventIndex;
  task.flags = descriptor.flags;
  task.outputClass = descriptor.outputClass;
  task.motherPdg = descriptor.motherPdg;
  task.firstDaughterPdg = descriptor.firstDaughterPdg;
  task.secondDaughterPdg = descriptor.secondDaughterPdg;
  task.firstSourcePdg = descriptor.firstSourcePdg;
  task.firstAlternateSourcePdg = descriptor.firstAlternateSourcePdg;
  task.secondSourcePdg = descriptor.secondSourcePdg;
  task.secondAlternateSourcePdg = descriptor.secondAlternateSourcePdg;
  task.primaryVertexIndex = ResolveTwoDaughterPrimaryVertexIndex(
    state.InputTracks(), descriptor.primaryVertexIndex,
    routed.firstTrackIndex, routed.secondTrackIndex);
  task.firstMass = descriptor.firstMass;
  task.secondMass = descriptor.secondMass;
  task.motherMass = descriptor.motherMass;
  task.motherMassSigma = descriptor.motherMassSigma;
  task.secondaryMassSigmaCut = descriptor.secondaryMassSigmaCut;
  task.maxSecondaryTopoChi2PerNdf = descriptor.maxSecondaryTopoChi2PerNdf;
  task.minSecondaryLdL = descriptor.minSecondaryLdL;
  task.transportMode = descriptor.transportMode;

  KFParticleGpuFitState mother;
  if (!BuildTwoDaughterCandidate(tracks, task, mother)) {
    return;
  }

  unsigned int candidateIndex = 0u;
  if (!ReserveBoundedCounter(
        candidates.SizeData(), 1u, candidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(
      candidates.OverflowFlagsData(),
      static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }
  if (!descriptorIndices.CanStore(candidateIndex)) {
    xpu::atomic_or(
      candidates.OverflowFlagsData(),
      static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }
  descriptorIndices.Index(candidateIndex) = routed.descriptorIndex;

  unsigned int daughterOffset = 0u;
  if (!ReserveBoundedCounter(
        candidates.Daughters().SizeData(), 2u,
        candidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(
      candidates.OverflowFlagsData(),
      static_cast<unsigned int>(DaughterCapacityExceeded));
    StoreFailedTwoDaughterCandidate(candidates, task, candidateIndex);
    return;
  }
  if (StoreTwoDaughterCandidate(
        candidates, tracks, task, candidateIndex, daughterOffset, mother)
      && routing.ChannelConstructedCounters()) {
    xpu::atomic_add(
      routing.ChannelConstructedCounters() + routed.descriptorIndex, 1u);
  }
}

XPU_EXPORT(KFParticleGpuExecuteGraphOperationsExplicit);
XPU_D void KFParticleGpuExecuteGraphOperationsExplicit::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  KFParticleGpuConstCandidatePoolView inputCandidates,
  KFParticleGpuConstVertexSoAView primaryVertices,
  const KFParticleGpuGraphOperationDescriptor* descriptors,
  unsigned int descriptorCount,
  const KFParticleGpuGraphOperationTask* tasks,
  unsigned int taskCount,
  KFParticleGpuCandidatePoolView outputCandidates,
  KFParticleGpuGraphOperationResult* results)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !results || thread >= taskCount) {
    return;
  }

  KFParticleGpuGraphOperationResult result;
  const KFParticleGpuGraphOperationTask task = tasks[thread];
  if (!descriptors || task.descriptorIndex >= descriptorCount) {
    result.status = KFGpuGraphTaskRejectDescriptor;
    results[thread] = result;
    return;
  }

  const KFParticleGpuGraphOperationDescriptor descriptor =
    descriptors[task.descriptorIndex];
  KFParticleGpuFitState output;
  int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
  unsigned int lineageSize = 0u;
  if (!KFParticleGpuGraphOperations::ExecuteTask(
        inputTracks,
        inputCandidates,
        primaryVertices,
        descriptor,
        task,
        output,
        lineage,
        lineageSize,
        result)) {
    results[thread] = result;
    return;
  }

  unsigned int candidateIndex = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.SizeData(), 1u, outputCandidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(),
      static_cast<unsigned int>(CandidateCapacityExceeded));
    result.status = KFGpuGraphTaskRejectCandidateCapacity;
    results[thread] = result;
    return;
  }
  unsigned int daughterOffset = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.Daughters().SizeData(),
        lineageSize,
        outputCandidates.Daughters().Capacity(),
        daughterOffset)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(),
      static_cast<unsigned int>(DaughterCapacityExceeded));
    outputCandidates.Metadata().Pdg(candidateIndex) = descriptor.motherPdg;
    outputCandidates.Metadata().PrimaryVertexIndex(candidateIndex) =
      descriptor.primaryVertexIndex;
    outputCandidates.Metadata().EventIndex(candidateIndex) =
      task.eventIndex;
    outputCandidates.Metadata().DaughterOffset(candidateIndex) = 0u;
    outputCandidates.Metadata().DaughterCount(candidateIndex) = 0u;
    outputCandidates.Metadata().Flags(candidateIndex) = KFGpuCandidateBuildFailed;
    outputCandidates.Metadata().ChannelId(candidateIndex) = descriptor.channelId;
    outputCandidates.Metadata().Topology(candidateIndex) = descriptor.topology;
    outputCandidates.Metadata().OutputClass(candidateIndex) = descriptor.outputClass;
    outputCandidates.Metadata().OperationStatus(candidateIndex) =
      KFGpuCandidateOperationRejected;
    ClearCandidateDirectDaughters(outputCandidates.Metadata(), candidateIndex);
    result.status = KFGpuGraphTaskRejectDaughterCapacity;
    result.outputCandidateIndex = candidateIndex;
    results[thread] = result;
    return;
  }

  KFParticleGpuGraphOperations::StoreOutput(
    outputCandidates,
    descriptor,
    task,
    task.eventIndex,
    result.primaryVertexIndex,
    candidateIndex,
    daughterOffset,
    output,
    lineage,
    lineageSize);
  result.outputCandidateIndex = candidateIndex;
  results[thread] = result;
}

XPU_EXPORT(KFParticleGpuResetGraphOperationGeneration);
XPU_D void KFParticleGpuResetGraphOperationGeneration::operator()(
  context& context)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuGraphOperationStorageView& graphOperations =
    context.cmem<TheKFParticleFinder>().GraphOperations();
  if (thread == 0u) {
    if (graphOperations.Visited()) graphOperations.Visited()[0] = 0u;
    if (graphOperations.Accepted()) graphOperations.Accepted()[0] = 0u;
    if (graphOperations.Stored()) graphOperations.Stored()[0] = 0u;
    if (graphOperations.Constructed()) graphOperations.Constructed()[0] = 0u;
    if (graphOperations.Rejected()) graphOperations.Rejected()[0] = 0u;
    if (graphOperations.Overflow()) graphOperations.Overflow()[0] = 0u;
  }
  if (thread < graphOperations.DescriptorCount()) {
    if (graphOperations.ChannelVisited()) {
      graphOperations.ChannelVisited()[thread] = 0u;
    }
    if (graphOperations.ChannelAccepted()) {
      graphOperations.ChannelAccepted()[thread] = 0u;
    }
    if (graphOperations.ChannelStored()) {
      graphOperations.ChannelStored()[thread] = 0u;
    }
    if (graphOperations.ChannelConstructed()) {
      graphOperations.ChannelConstructed()[thread] = 0u;
    }
    if (graphOperations.ChannelRejected()) {
      graphOperations.ChannelRejected()[thread] = 0u;
    }
  }
}

XPU_EXPORT(KFParticleGpuRouteGraphOperationTasks);
XPU_D void KFParticleGpuRouteGraphOperationTasks::operator()(
  context& context,
  unsigned int eventIndex,
  unsigned int groupIndex,
  unsigned int sourceCapacity,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuDecayGraphView& graph = state.DecayGraph();
  const KFParticleGpuConstCandidatePoolView candidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  const KFParticleGpuEventDesc* events = state.Events();
  const KFParticleGpuGraphOperationStorageView& graphOperations =
    state.GraphOperations();
  const KFParticleGpuGraphOperationDescriptor* descriptors =
    graphOperations.Descriptors();
  KFParticleGpuGraphOperationTask* tasks = graphOperations.Tasks();
  const unsigned int descriptorCount = graphOperations.DescriptorCount();
  const unsigned int taskCapacity =
    taskLimit < graphOperations.TaskCapacity()
      ? taskLimit
      : graphOperations.TaskCapacity();
  if (!descriptors || !tasks || !graphOperations.Visited()
      || !graphOperations.Accepted() || !graphOperations.Stored()
      || groupIndex >= graph.GroupCount()) {
    return;
  }
  const KFParticleGpuGraphExecutionGroup& group = graph.Groups()[groupIndex];
  if (group.nodeCount == 0u || sourceCapacity == 0u || !events) {
    return;
  }
  const unsigned int nodeIndex = thread / sourceCapacity;
  const unsigned int firstSlot = thread % sourceCapacity;
  if (nodeIndex >= group.nodeCount) {
    return;
  }
  const KFParticleGpuGraphNode& node =
    graph.Nodes()[group.nodeOffset + nodeIndex];
  const auto sourceRange = [&](const KFParticleGpuGraphSource& source,
                               unsigned int& begin,
                               unsigned int& end) {
    begin = 0u;
    end = 0u;
    if (source.kind == KFGpuGraphSourceCandidateGeneration) {
      end = candidates.Size();
      return true;
    }
    if (source.kind == KFGpuGraphSourceTrackRange) {
      KFParticleGpuTrackSet set = NumberOfTrackSets;
      KFParticleGpuTrackSpecies species = NumberOfTrackSpecies;
      if (!KFParticleGpuDecodeGraphTrackSourceId(source.sourceId, set, species)) {
        return false;
      }
      const KFParticleGpuTrackSetDesc& trackSet = events[eventIndex].TrackSet(set);
      const KFParticleGpuRange range = species == NumberOfTrackSpecies
        ? trackSet.tracks : trackSet.Species(species);
      begin = range.offset;
      end = range.End();
      return end <= tracks.Size();
    }
    return false;
  };
  const auto sourceMatches = [&](const KFParticleGpuGraphSource& source,
                                 unsigned int index) {
    if (source.kind == KFGpuGraphSourceCandidateGeneration) {
      return index < candidates.Size()
        && candidates.Metadata().EventIndex(index) == eventIndex
        && candidates.Metadata().ChannelId(index) == source.sourceId;
    }
    return source.kind == KFGpuGraphSourceTrackRange && index < tracks.Size();
  };
  unsigned int firstBegin = 0u;
  unsigned int firstEnd = 0u;
  if (node.supportStatus != KFGpuGraphSupported
      || !sourceRange(node.firstSource, firstBegin, firstEnd)
      || firstSlot >= firstEnd - firstBegin) {
    return;
  }
  const unsigned int firstSourceIndex = firstBegin + firstSlot;
  if (!sourceMatches(node.firstSource, firstSourceIndex)) return;
  unsigned int descriptorIndex = descriptorCount;
  for (unsigned int index = 0u; index < descriptorCount; ++index) {
    if (descriptors[index].channelId == node.channelId) {
      descriptorIndex = index;
      break;
    }
  }
  if (descriptorIndex >= descriptorCount) {
    return;
  }
  const KFParticleGpuGraphOperationDescriptor& descriptor =
    descriptors[descriptorIndex];

  const bool unary = node.topology == KFGpuGraphTopologyUnaryComposite;
  unsigned int secondBegin = 0u;
  unsigned int secondEnd = 1u;
  if (!unary && !sourceRange(node.secondSource, secondBegin, secondEnd)) return;
  for (unsigned int second = secondBegin; second < secondEnd; ++second) {
    if (!unary
        && (((descriptor.flags & KFGpuGraphRequireOrderedCandidatePair) != 0u
             && node.firstSource.kind == node.secondSource.kind
             && node.firstSource.sourceId == node.secondSource.sourceId
             && second <= firstSourceIndex)
            || !sourceMatches(node.secondSource, second))) {
      continue;
    }
    xpu::atomic_add(graphOperations.Visited(), 1u);
    if (graphOperations.ChannelVisited()) {
      xpu::atomic_add(
        graphOperations.ChannelVisited() + descriptorIndex, 1u);
    }
    const unsigned int slot = xpu::atomic_add(graphOperations.Accepted(), 1u);
    if (graphOperations.ChannelAccepted()) {
      xpu::atomic_add(
        graphOperations.ChannelAccepted() + descriptorIndex, 1u);
    }
    if (slot < taskCapacity) {
      KFParticleGpuGraphOperationTask task;
      task.descriptorIndex = descriptorIndex;
      task.firstCandidateIndex = firstSourceIndex;
      task.secondCandidateIndex = unary ? 0u : second;
      task.firstSourceKind = node.firstSource.kind;
      task.secondSourceKind = unary ? KFGpuGraphSourceNone : node.secondSource.kind;
      task.eventIndex = eventIndex;
      task.primaryVertexOffset = events[eventIndex].primaryVertices.offset;
      task.primaryVertexCount = events[eventIndex].primaryVertices.size;
      tasks[slot] = task;
      xpu::atomic_add(graphOperations.Stored(), 1u);
      if (graphOperations.ChannelStored()) {
        xpu::atomic_add(
          graphOperations.ChannelStored() + descriptorIndex, 1u);
      }
    }
    else if (graphOperations.Overflow()) {
      xpu::atomic_or(
        graphOperations.Overflow(),
        static_cast<unsigned int>(KFGpuGraphTaskCapacityExceeded));
    }
  }
}

XPU_EXPORT(KFParticleGpuExecuteRoutedGraphOperations);
XPU_D void KFParticleGpuExecuteRoutedGraphOperations::operator()(
  context& context,
  unsigned int taskLimit)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstCandidatePoolView inputCandidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstInputTrackSoAView& inputTracks = state.InputTracks();
  const KFParticleGpuConstVertexSoAView& primaryVertices =
    state.PrimaryVertices();
  const KFParticleGpuCandidatePoolView& outputCandidates = state.Candidates();
  const KFParticleGpuGraphOperationStorageView& graphOperations =
    state.GraphOperations();
  const KFParticleGpuGraphOperationDescriptor* descriptors =
    graphOperations.Descriptors();
  const KFParticleGpuGraphOperationTask* tasks = graphOperations.Tasks();
  KFParticleGpuGraphOperationResult* results = graphOperations.Results();
  const unsigned int descriptorCount = graphOperations.DescriptorCount();
  const unsigned int taskCapacity =
    taskLimit < graphOperations.TaskCapacity()
      ? taskLimit
      : graphOperations.TaskCapacity();
  if (!descriptors || !tasks || !graphOperations.Stored() || !results
      || thread >= taskCapacity || thread >= graphOperations.Stored()[0]) {
    return;
  }
  const KFParticleGpuGraphOperationTask task = tasks[thread];
  KFParticleGpuGraphOperationResult result;
  if (task.descriptorIndex >= descriptorCount) {
    result.status = KFGpuGraphTaskRejectDescriptor;
    results[thread] = result;
    if (graphOperations.Rejected()) {
      xpu::atomic_add(graphOperations.Rejected(), 1u);
    }
    return;
  }
  const KFParticleGpuGraphOperationDescriptor descriptor =
    descriptors[task.descriptorIndex];
  KFParticleGpuFitState output;
  int lineage[KFParticleGpuGraphOperations::MaximumLineageSize];
  unsigned int lineageSize = 0u;
  if (!KFParticleGpuGraphOperations::ExecuteTask(
        inputTracks, inputCandidates, primaryVertices, descriptor, task, output,
        lineage, lineageSize, result)) {
    results[thread] = result;
    if (graphOperations.Rejected()) {
      xpu::atomic_add(graphOperations.Rejected(), 1u);
    }
    if (graphOperations.ChannelRejected()) {
      xpu::atomic_add(
        graphOperations.ChannelRejected() + task.descriptorIndex, 1u);
    }
    return;
  }

  unsigned int candidateIndex = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.SizeData(), 1u,
        outputCandidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(),
      static_cast<unsigned int>(CandidateCapacityExceeded));
    result.status = KFGpuGraphTaskRejectCandidateCapacity;
    results[thread] = result;
    if (graphOperations.Rejected()) {
      xpu::atomic_add(graphOperations.Rejected(), 1u);
    }
    if (graphOperations.ChannelRejected()) {
      xpu::atomic_add(
        graphOperations.ChannelRejected() + task.descriptorIndex, 1u);
    }
    return;
  }
  unsigned int daughterOffset = 0u;
  if (!ReserveBoundedCounter(
        outputCandidates.Daughters().SizeData(), lineageSize,
        outputCandidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(
      outputCandidates.OverflowFlagsData(),
      static_cast<unsigned int>(DaughterCapacityExceeded));
    outputCandidates.Metadata().Pdg(candidateIndex) = descriptor.motherPdg;
    outputCandidates.Metadata().PrimaryVertexIndex(candidateIndex) =
      descriptor.primaryVertexIndex;
    outputCandidates.Metadata().EventIndex(candidateIndex) =
      task.eventIndex;
    outputCandidates.Metadata().DaughterOffset(candidateIndex) = 0u;
    outputCandidates.Metadata().DaughterCount(candidateIndex) = 0u;
    outputCandidates.Metadata().Flags(candidateIndex) = KFGpuCandidateBuildFailed;
    outputCandidates.Metadata().ChannelId(candidateIndex) = descriptor.channelId;
    outputCandidates.Metadata().Topology(candidateIndex) = descriptor.topology;
    outputCandidates.Metadata().OutputClass(candidateIndex) = descriptor.outputClass;
    outputCandidates.Metadata().OperationStatus(candidateIndex) =
      KFGpuCandidateOperationRejected;
    ClearCandidateDirectDaughters(outputCandidates.Metadata(), candidateIndex);
    result.status = KFGpuGraphTaskRejectDaughterCapacity;
    result.outputCandidateIndex = candidateIndex;
    results[thread] = result;
    if (graphOperations.Rejected()) {
      xpu::atomic_add(graphOperations.Rejected(), 1u);
    }
    if (graphOperations.ChannelRejected()) {
      xpu::atomic_add(
        graphOperations.ChannelRejected() + task.descriptorIndex, 1u);
    }
    return;
  }
  KFParticleGpuGraphOperations::StoreOutput(
    outputCandidates, descriptor, task,
    task.eventIndex,
    result.primaryVertexIndex,
    candidateIndex, daughterOffset, output, lineage, lineageSize);
  result.outputCandidateIndex = candidateIndex;
  results[thread] = result;
  if (graphOperations.Constructed()) {
    xpu::atomic_add(graphOperations.Constructed(), 1u);
  }
  if (graphOperations.ChannelConstructed()) {
    xpu::atomic_add(
      graphOperations.ChannelConstructed() + task.descriptorIndex, 1u);
  }
}

XPU_EXPORT(KFParticleGpuResetTwoDaughterGenerationState);
XPU_D void KFParticleGpuResetTwoDaughterGenerationState::operator()(
  context& context)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuTwoDaughterGenerationStorageView& generation =
    state.TwoDaughterGeneration();
  const KFParticleGpuTwoDaughterSelectionWorkspaceView& selection =
    generation.Selection();
  if (thread == 0u) {
    if (generation.VisitedPairs()) {
      generation.VisitedPairs()[0] = 0u;
    }
    if (generation.ActiveChannelBits()) {
      generation.ActiveChannelBits()[0] = 0u;
    }
    if (generation.AcceptedTasks()) {
      generation.AcceptedTasks()[0] = 0u;
    }
    if (generation.StoredTasks()) {
      generation.StoredTasks()[0] = 0u;
    }
    if (generation.BlockReservations()) {
      generation.BlockReservations()[0] = 0u;
    }
    if (generation.OverflowFlags()) {
      generation.OverflowFlags()[0] = 0u;
    }
  }
  if (thread >= routing.DescriptorCount() || thread >= selection.Capacity()) {
    return;
  }
  if (routing.ChannelVisitedCounters()) {
    routing.ChannelVisitedCounters()[thread] = 0u;
  }
  if (routing.ChannelAcceptedCounters()) {
    routing.ChannelAcceptedCounters()[thread] = 0u;
  }
  if (routing.ChannelStoredCounters()) {
    routing.ChannelStoredCounters()[thread] = 0u;
  }
  if (routing.ChannelConstructedCounters()) {
    routing.ChannelConstructedCounters()[thread] = 0u;
  }
  selection.Accepted(thread) = 0u;
  selection.Stored(thread) = 0u;
  selection.Offset(thread) = 0u;
  selection.Cursor(thread) = 0u;
}

XPU_EXPORT(KFParticleGpuEvaluateTwoDaughterGenerationSelection);
XPU_D void KFParticleGpuEvaluateTwoDaughterGenerationSelection::operator()(
  context& context,
  unsigned int eventIndex)
{
  const unsigned int candidateIndex =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstCandidatePoolView candidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstCandidateDescriptorIndexView descriptorIndices =
    MakeConstView(state.CandidateDescriptorIndices());
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuConstVertexSoAView& primaryVertices =
    state.PrimaryVertices();
  const KFParticleGpuEventDesc* events = state.Events();
  const KFParticleGpuV0SelectionResultView& selectionResults =
    state.SelectionResults();
  const KFParticleGpuTwoDaughterSelectionWorkspaceView& selection =
    state.TwoDaughterGeneration().Selection();
  if (!events || candidateIndex >= candidates.Size()
      || candidateIndex >= candidates.Capacity()
      || candidateIndex >= descriptorIndices.Capacity()
      || candidates.Metadata().EventIndex(candidateIndex) != eventIndex) {
    return;
  }
  const unsigned int descriptorIndex = descriptorIndices.Index(candidateIndex);
  if (descriptorIndex >= routing.DescriptorCount()
      || descriptorIndex >= selection.Capacity()) {
    return;
  }
  const KFParticleGpuTwoDaughterRoutingDescriptor& descriptor =
    routing.Descriptors()[descriptorIndex];
  if (descriptor.channelBit != descriptorIndex
      || descriptor.channelId != candidates.Metadata().ChannelId(candidateIndex)
      || descriptor.selection.expectedMass <= 0.f) {
    return;
  }

  KFParticleGpuFitState candidate;
  LoadCandidateFit(candidates, candidateIndex, candidate);
  KFParticleGpuV0SelectionResult result;
  KFParticleGpuSelection::EvaluateV0Selection(
    candidate,
    candidates.Metadata().Flags(candidateIndex),
    candidateIndex,
    descriptor.channelId,
    eventIndex,
    primaryVertices,
    events[eventIndex].primaryVertices,
    descriptor.selection,
    result);
  if (selectionResults.CanStore(candidateIndex)) {
    selectionResults.Result(candidateIndex) = result;
  }
  if (KFParticleGpuSelection::IsSelected(result)) {
    xpu::atomic_add(selection.AcceptedData() + descriptorIndex, 1u);
  }
}

XPU_EXPORT(KFParticleGpuPrepareTwoDaughterSelectionSegments);
XPU_D void KFParticleGpuPrepareTwoDaughterSelectionSegments::operator()(
  context& context)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuTwoDaughterSelectionWorkspaceView& selection =
    state.TwoDaughterGeneration().Selection();
  const KFParticleGpuSelectedCandidateIndexView& selectedCandidates =
    state.SelectedCandidates();
  if (thread != 0u || !selectedCandidates.SizeData()
      || !selectedCandidates.OverflowFlagsData()) {
    return;
  }

  unsigned int output = selectedCandidates.Size();
  for (unsigned int descriptorIndex = 0u;
       descriptorIndex < routing.DescriptorCount()
         && descriptorIndex < selection.Capacity();
       ++descriptorIndex) {
    selection.Offset(descriptorIndex) = output;
    selection.Cursor(descriptorIndex) = 0u;
    const unsigned int accepted = selection.Accepted(descriptorIndex);
    const unsigned int available =
      output < selectedCandidates.Capacity()
        ? selectedCandidates.Capacity() - output : 0u;
    const unsigned int stored = accepted < available ? accepted : available;
    selection.Stored(descriptorIndex) = stored;
    output += stored;
    if (stored < accepted) {
      xpu::atomic_or(
        selectedCandidates.OverflowFlagsData(),
        static_cast<unsigned int>(KFGpuSelectedCandidateCapacityExceeded));
    }
  }
  selectedCandidates.SizeData()[0] = output;
}

XPU_EXPORT(KFParticleGpuScatterTwoDaughterGenerationSelection);
XPU_D void KFParticleGpuScatterTwoDaughterGenerationSelection::operator()(
  context& context,
  unsigned int eventIndex)
{
  const unsigned int candidateIndex =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuConstCandidatePoolView candidates =
    MakeConstView(state.Candidates());
  const KFParticleGpuConstCandidateDescriptorIndexView descriptorIndices =
    MakeConstView(state.CandidateDescriptorIndices());
  const KFParticleGpuTwoDaughterRoutingView& routing =
    state.TwoDaughterRouting();
  const KFParticleGpuV0SelectionResultView& selectionResults =
    state.SelectionResults();
  const KFParticleGpuTwoDaughterSelectionWorkspaceView& selection =
    state.TwoDaughterGeneration().Selection();
  const KFParticleGpuSelectedCandidateIndexView& selectedCandidates =
    state.SelectedCandidates();
  if (candidateIndex >= candidates.Size()
      || candidateIndex >= candidates.Capacity()
      || candidateIndex >= descriptorIndices.Capacity()
      || candidates.Metadata().EventIndex(candidateIndex) != eventIndex
      || !selectionResults.CanStore(candidateIndex)) {
    return;
  }
  KFParticleGpuV0SelectionResult result = selectionResults.Result(candidateIndex);
  if (!KFParticleGpuSelection::IsSelected(result)) {
    return;
  }
  const unsigned int descriptorIndex = descriptorIndices.Index(candidateIndex);
  if (descriptorIndex >= routing.DescriptorCount()
      || descriptorIndex >= selection.Capacity()
      || routing.Descriptors()[descriptorIndex].channelId
           != candidates.Metadata().ChannelId(candidateIndex)) {
    return;
  }
  const unsigned int local =
    xpu::atomic_add(selection.CursorsData() + descriptorIndex, 1u);
  if (local >= selection.Stored(descriptorIndex)) {
    result.selectionClass = KFGpuV0SelectionRejected;
    result.rejectionReasons |= KFGpuV0SelectionRejectOutputOverflow;
    selectionResults.Result(candidateIndex) = result;
    return;
  }
  const unsigned int outputIndex = selection.Offset(descriptorIndex) + local;
  selectedCandidates.Index(outputIndex) = candidateIndex;
  if (selectedCandidates.ChannelIdsData()) {
    selectedCandidates.ChannelId(outputIndex) = result.channelId;
  }
}

XPU_EXPORT(KFParticleGpuTwoDaughterTaskKernel);
XPU_D void KFParticleGpuTwoDaughterTaskKernel::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  unsigned int numberOfTasks,
  KFParticleGpuCandidatePoolView candidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());

  if (thread == 0) {
    const unsigned int candidateSize =
      numberOfTasks < candidates.Capacity() ? numberOfTasks : candidates.Capacity();
    const unsigned int daughterSize =
      candidateSize * 2u < candidates.Daughters().Capacity()
        ? candidateSize * 2u
        : candidates.Daughters().Capacity();
    candidates.SizeData()[0] = candidateSize;
    candidates.Daughters().SizeData()[0] = daughterSize;

    unsigned int overflow = 0;
    if (numberOfTasks > candidates.Capacity()) {
      overflow |= CandidateCapacityExceeded;
    }
    if (numberOfTasks * 2u > candidates.Daughters().Capacity()) {
      overflow |= DaughterCapacityExceeded;
    }
    candidates.OverflowFlagsData()[0] = overflow;
  }

  if (thread >= numberOfTasks || thread >= candidates.Capacity()) {
    return;
  }

  const KFParticleGpuTwoDaughterTask task = tasks[thread];
  if (!candidates.Daughters().CanStore(thread * 2u, 2u)) {
    StoreFailedTwoDaughterCandidate(candidates, task, thread);
    return;
  }

  KFParticleGpuFitState mother;
  if (BuildTwoDaughterCandidate(inputTracks, task, mother)) {
    StoreTwoDaughterCandidate(candidates, inputTracks, task, thread, mother);
  }
  else {
    StoreFailedTwoDaughterCandidate(candidates, task, thread);
  }
}

XPU_EXPORT(KFParticleGpuTwoDaughterCompactCandidateKernel);
XPU_D void KFParticleGpuTwoDaughterCompactCandidateKernel::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  unsigned int numberOfTasks,
  KFParticleGpuCandidatePoolView candidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());

  if (!tasks || thread >= numberOfTasks) {
    return;
  }

  const KFParticleGpuTwoDaughterTask task = tasks[thread];
  KFParticleGpuFitState mother;
  if (!BuildTwoDaughterCandidate(inputTracks, task, mother)) {
    return;
  }

  unsigned int candidateIndex = 0;
  if (!ReserveBoundedCounter(candidates.SizeData(), 1u, candidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(candidates.OverflowFlagsData(), static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }

  unsigned int daughterOffset = 0;
  if (!ReserveBoundedCounter(
        candidates.Daughters().SizeData(), 2u, candidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(candidates.OverflowFlagsData(), static_cast<unsigned int>(DaughterCapacityExceeded));
    StoreFailedTwoDaughterCandidate(candidates, task, candidateIndex);
    return;
  }

  StoreTwoDaughterCandidate(candidates, inputTracks, task, candidateIndex, daughterOffset, mother);
}

XPU_EXPORT(KFParticleGpuTwoDaughterCompactCandidatePoolKernel);
XPU_D void KFParticleGpuTwoDaughterCompactCandidatePoolKernel::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  unsigned int taskCapacity,
  const unsigned int* acceptedTasks,
  KFParticleGpuCandidatePoolView candidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!tasks || !acceptedTasks || thread >= taskCapacity || thread >= acceptedTasks[0]) {
    return;
  }

  const KFParticleGpuTwoDaughterTask task = tasks[thread];
  KFParticleGpuFitState mother;
  if (!BuildTwoDaughterCandidate(inputTracks, task, mother)) {
    return;
  }

  unsigned int candidateIndex = 0;
  if (!ReserveBoundedCounter(candidates.SizeData(), 1u, candidates.Capacity(), candidateIndex)) {
    xpu::atomic_or(candidates.OverflowFlagsData(), static_cast<unsigned int>(CandidateCapacityExceeded));
    return;
  }
  unsigned int daughterOffset = 0;
  if (!ReserveBoundedCounter(
        candidates.Daughters().SizeData(), 2u, candidates.Daughters().Capacity(), daughterOffset)) {
    xpu::atomic_or(candidates.OverflowFlagsData(), static_cast<unsigned int>(DaughterCapacityExceeded));
    StoreFailedTwoDaughterCandidate(candidates, task, candidateIndex);
    return;
  }
  StoreTwoDaughterCandidate(candidates, inputTracks, task, candidateIndex, daughterOffset, mother);
}

XPU_EXPORT(KFParticleGpuSelectV0Candidates);
XPU_D void KFParticleGpuSelectV0Candidates::operator()(
  context& context,
  KFParticleGpuConstCandidatePoolView candidates,
  KFParticleGpuConstVertexSoAView primaryVertices,
  const KFParticleGpuEventDesc* events,
  unsigned int eventIndex,
  unsigned int candidateOffset,
  unsigned int candidateCount,
  unsigned int channelId,
  KFParticleGpuV0SelectionConfig config,
  KFParticleGpuV0SelectionResultView selectionResults,
  KFParticleGpuSelectedCandidateIndexView selectedCandidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!events || thread >= candidateCount) {
    return;
  }

  const unsigned int candidateIndex = candidateOffset + thread;
  if (candidateIndex >= candidates.Size() || candidateIndex >= candidates.Capacity()) {
    return;
  }
  if (candidates.Metadata().EventIndex(candidateIndex) != eventIndex) {
    return;
  }

  KFParticleGpuFitState candidate;
  LoadCandidateFit(candidates, candidateIndex, candidate);
  KFParticleGpuV0SelectionResult selection;
  KFParticleGpuSelection::EvaluateV0Selection(candidate,
                                               candidates.Metadata().Flags(candidateIndex),
                                               candidateIndex,
                                               channelId,
                                               eventIndex,
                                               primaryVertices,
                                               events[eventIndex].primaryVertices,
                                               config,
                                               selection);
  if (selectionResults.CanStore(candidateIndex)) {
    selectionResults.Result(candidateIndex) = selection;
  }
  if (!KFParticleGpuSelection::IsSelected(selection)) {
    return;
  }

  unsigned int selectedIndex = 0u;
  if (!ReserveBoundedCounter(
        selectedCandidates.SizeData(), 1u, selectedCandidates.Capacity(), selectedIndex)) {
    xpu::atomic_or(
      selectedCandidates.OverflowFlagsData(),
      static_cast<unsigned int>(KFGpuSelectedCandidateCapacityExceeded));
    selection.selectionClass = KFGpuV0SelectionRejected;
    selection.rejectionReasons |= KFGpuV0SelectionRejectOutputOverflow;
    if (selectionResults.CanStore(candidateIndex)) {
      selectionResults.Result(candidateIndex) = selection;
    }
    return;
  }
  selectedCandidates.Index(selectedIndex) = candidateIndex;
  if (selectedCandidates.ChannelIdsData()) {
    selectedCandidates.ChannelId(selectedIndex) = selection.channelId;
  }
}

XPU_EXPORT(KFParticleGpuGenerateTwoDaughterTasks);
XPU_D void KFParticleGpuGenerateTwoDaughterTasks::operator()(
  context& context,
  const KFParticleGpuEventDesc* events,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  KFParticleGpuTwoDaughterTaskSource source,
  KFParticleGpuTwoDaughterTask* tasks,
  unsigned int taskCapacity,
  unsigned int* writtenTasks,
  unsigned int* totalPairs)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());

  if (!events || !tasks || !writtenTasks || !totalPairs) {
    return;
  }

  const KFParticleGpuEventDesc event = events[source.eventIndex];
  const KFParticleGpuRange firstRange =
    ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
  const KFParticleGpuRange secondRange =
    ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
  const unsigned int total = firstRange.size * secondRange.size;
  const unsigned int written = total < taskCapacity ? total : taskCapacity;

  if (thread == 0) {
    writtenTasks[0] = written;
    totalPairs[0] = total;
  }

  if (thread >= written || secondRange.size == 0u) {
    return;
  }

  KFParticleGpuTwoDaughterTask task;
  FillTwoDaughterTask(firstRange, secondRange, thread, source, task);
  if (PassTwoDaughterTaskSourceCuts(inputTracks,
                                    source,
                                    task.firstTrack,
                                    task.secondTrack,
                                    event.minSecondaryTrackChiToPrimaryVertex)) {
    task.primaryVertexIndex = ResolveTwoDaughterPrimaryVertexIndex(
      inputTracks, source.primaryVertexIndex, task.firstTrack, task.secondTrack);
    tasks[thread] = task;
  }
  else {
    tasks[thread] = KFParticleGpuTwoDaughterTask();
    tasks[thread].channelId = source.channelId;
    tasks[thread].eventIndex = source.eventIndex;
    tasks[thread].motherPdg = source.motherPdg;
    tasks[thread].firstTrack = inputTracks.Size();
    tasks[thread].secondTrack = inputTracks.Size();
  }
}

XPU_EXPORT(KFParticleGpuGenerateTwoDaughterTasksCompact);
XPU_D void KFParticleGpuGenerateTwoDaughterTasksCompact::operator()(
  context& context,
  const KFParticleGpuEventDesc* events,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  KFParticleGpuTwoDaughterTaskSource source,
  KFParticleGpuTwoDaughterTask* tasks,
  unsigned int taskCapacity,
  unsigned int* acceptedTasks,
  unsigned int* totalPairs)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());

  if (!events || !tasks || !acceptedTasks || !totalPairs) {
    return;
  }

  const KFParticleGpuEventDesc event = events[source.eventIndex];
  const KFParticleGpuRange firstRange =
    ResolveTaskSourceRange(event, source.firstTrackSet, source.firstSpecies);
  const KFParticleGpuRange secondRange =
    ResolveTaskSourceRange(event, source.secondTrackSet, source.secondSpecies);
  const unsigned int total = firstRange.size * secondRange.size;

  if (thread == 0) {
    totalPairs[0] = total;
  }
  if (thread >= total || secondRange.size == 0u) {
    return;
  }

  KFParticleGpuTwoDaughterTask task;
  FillTwoDaughterTask(firstRange, secondRange, thread, source, task);
  if (!PassTwoDaughterTaskSourceCuts(inputTracks,
                                     source,
                                     task.firstTrack,
                                     task.secondTrack,
                                     event.minSecondaryTrackChiToPrimaryVertex)) {
    return;
  }
  task.primaryVertexIndex = ResolveTwoDaughterPrimaryVertexIndex(
    inputTracks, source.primaryVertexIndex, task.firstTrack, task.secondTrack);

  const unsigned int slot = xpu::atomic_add(acceptedTasks, 1u);
  if (slot < taskCapacity) {
    tasks[slot] = task;
  }
}

XPU_EXPORT(KFParticleGpuKalmanUpdateProbe);
XPU_D void KFParticleGpuKalmanUpdateProbe::operator()(context& context,
                                                      float* floatChecks,
                                                      int* integerChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0) {
    return;
  }

  KFParticleGpuFitState particle;
  particle.Initialize();
  particle.X() = 0.f;
  particle.Y() = 0.f;
  particle.Z() = 0.f;
  particle.Px() = 1.f;
  particle.Py() = 2.f;
  particle.Pz() = 3.f;
  particle.E() = 4.f;
  particle.Q() = 1;
  particle.NDF() = 0;
  particle.Covariance(0, 0) = 4.f;
  particle.Covariance(1, 1) = 5.f;
  particle.Covariance(2, 2) = 6.f;
  particle.Covariance(3, 3) = 0.1f;
  particle.Covariance(4, 4) = 0.2f;
  particle.Covariance(5, 5) = 0.3f;
  particle.Covariance(6, 6) = 0.4f;

  KFParticleGpuMeasurement measurement;
  measurement.Parameter(0) = 1.f;
  measurement.Parameter(1) = 2.f;
  measurement.Parameter(2) = 3.f;
  measurement.Parameter(3) = 0.5f;
  measurement.Parameter(4) = 0.25f;
  measurement.Parameter(5) = -0.5f;
  measurement.Parameter(6) = 1.f;
  measurement.Covariance(0, 0) = 1.f;
  measurement.Covariance(1, 1) = 1.f;
  measurement.Covariance(2, 2) = 1.f;
  measurement.Covariance(3, 3) = 0.01f;
  measurement.Covariance(4, 4) = 0.02f;
  measurement.Covariance(5, 5) = 0.03f;
  measurement.Covariance(6, 6) = 0.04f;

  const bool updated = KFParticleGpuMath::AddDaughterWithEnergyFit(particle, measurement, -1);

  floatChecks[0] = particle.X();
  floatChecks[1] = particle.Y();
  floatChecks[2] = particle.Z();
  floatChecks[3] = particle.Px();
  floatChecks[4] = particle.E();
  floatChecks[5] = particle.Covariance(0, 0);
  floatChecks[6] = particle.Covariance(3, 3);
  floatChecks[7] = particle.Chi2();
  integerChecks[0] = updated ? 1 : 0;
  integerChecks[1] = particle.Q();
  integerChecks[2] = particle.NDF();
}

XPU_EXPORT(KFParticleGpuFieldTransportProbe);
XPU_D void KFParticleGpuFieldTransportProbe::operator()(context& context,
                                                        float* floatChecks,
                                                        int* integerChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0) {
    return;
  }

  KFParticleGpuFitState particle;
  particle.Initialize();
  particle.X() = 1.f;
  particle.Y() = 2.f;
  particle.Z() = 3.f;
  particle.Px() = 4.f;
  particle.Py() = 5.f;
  particle.Pz() = 6.f;
  particle.Q() = 1;

  KFParticleGpuFitState line;
  KFParticleGpuMath::TransportLine(particle, 0.25f, line);

  KFParticleGpuFitState zeroField;
  KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 0.f, zeroField);

  KFParticleGpuFitState chargedField;
  KFParticleGpuMath::TransportConstantBy(particle, 0.25f, 20.f, chargedField);

  KFParticleGpuFitState neutral = particle;
  neutral.Q() = 0;
  KFParticleGpuFitState neutralField;
  KFParticleGpuMath::TransportConstantBy(neutral, 0.25f, 20.f, neutralField);

  KFParticleGpuFitState second;
  second.Initialize();
  second.X() = -0.5f;
  second.Y() = 1.5f;
  second.Z() = 2.5f;
  second.Px() = -3.f;
  second.Py() = 2.f;
  second.Pz() = 5.f;
  second.Q() = -1;
  second.E() = 8.f;

  particle.E() = 10.f;
  particle.NDF() = 0;
  second.NDF() = 0;
  particle.Covariance(0, 0) = 0.4f;
  particle.Covariance(1, 1) = 0.5f;
  particle.Covariance(2, 2) = 0.6f;
  particle.Covariance(3, 3) = 0.1f;
  particle.Covariance(4, 4) = 0.2f;
  particle.Covariance(5, 5) = 0.3f;
  particle.Covariance(6, 6) = 0.4f;
  second.Covariance(0, 0) = 0.7f;
  second.Covariance(1, 1) = 0.8f;
  second.Covariance(2, 2) = 0.9f;
  second.Covariance(3, 3) = 0.15f;
  second.Covariance(4, 4) = 0.25f;
  second.Covariance(5, 5) = 0.35f;
  second.Covariance(6, 6) = 0.45f;

  KFParticleGpuFitState lineMother;
  KFParticleGpuMath::BuildLineDcaKinematicMother(particle, second, lineMother);
  KFParticleGpuFitState zeroFieldMother;
  const bool zeroBuilt =
    KFParticleGpuMath::BuildConstantByDcaKinematicMother(particle, second, 0.f, zeroFieldMother);
  KFParticleGpuFitState fieldMother;
  const bool fieldBuilt =
    KFParticleGpuMath::BuildConstantByDcaKinematicMother(particle, second, 200.f, fieldMother);

  KFParticleGpuFitState lineCurrent;
  KFParticleGpuMeasurement lineMeasurement;
  const bool lineMeasurementBuilt =
    KFParticleGpuMath::BuildLineDcaMeasurementSeed(particle, second, lineCurrent, lineMeasurement);
  if (lineMeasurementBuilt) {
    KFParticleGpuMath::AddDaughterWithEnergyFit(lineCurrent, lineMeasurement, second.Q());
  }

  KFParticleGpuFitState fieldCurrent;
  KFParticleGpuMeasurement fieldMeasurement;
  const bool fieldMeasurementBuilt =
    KFParticleGpuMath::BuildConstantByDcaMeasurementSeed(
      particle, second, 200.f, fieldCurrent, fieldMeasurement);
  if (fieldMeasurementBuilt) {
    KFParticleGpuMath::AddDaughterWithEnergyFit(fieldCurrent, fieldMeasurement, second.Q());
  }

  const float fullFieldCoefficients[KFParticleGpuFieldRegion::NumberOfCoefficients] = {
    0.4f, 0.02f, 0.001f, 20.f, 0.5f, 0.01f, -0.3f, 0.015f, 0.0005f, 3.f};
  const KFParticleGpuFieldRegion fullField(fullFieldCoefficients);
  KFParticleGpuFitState fullFieldState;
  unsigned int fullFieldStatus = KFParticleGpuMath::KFGpuFullFieldTransportInvalidInput;
  const bool fullFieldBuilt = KFParticleGpuMath::TransportFullField(
    particle, fullField, 0.25f, fullFieldState, fullFieldStatus);
  KFParticleGpuMath::KFParticleGpuFullFieldDcaResult coupledDca;
  unsigned int coupledDcaStatus = KFParticleGpuMath::KFGpuFullFieldDcaRejected;
  const bool coupledDcaBuilt = KFParticleGpuMath::BuildFullFieldDcaCoupledResult(
    particle, second, fullField, fullField, coupledDca, coupledDcaStatus);

  floatChecks[0] = line.X();
  floatChecks[1] = line.Z();
  floatChecks[2] = zeroField.X();
  floatChecks[3] = zeroField.Z();
  floatChecks[4] = chargedField.X();
  floatChecks[5] = chargedField.Z();
  floatChecks[6] = chargedField.Px();
  floatChecks[7] = chargedField.Pz();
  floatChecks[8] = neutralField.X();
  floatChecks[9] = neutralField.Z();
  floatChecks[10] = lineMother.X();
  floatChecks[11] = zeroFieldMother.X();
  floatChecks[12] = fieldMother.X();
  floatChecks[13] = lineMother.Px();
  floatChecks[14] = fieldMother.Px();
  floatChecks[15] = lineCurrent.X();
  floatChecks[16] = fieldCurrent.X();
  floatChecks[17] = lineCurrent.Chi2();
  floatChecks[18] = fieldCurrent.Chi2();
  floatChecks[19] = fieldCurrent.Px();
  floatChecks[20] = fullFieldState.X();
  floatChecks[21] = fullFieldState.Y();
  floatChecks[22] = fullFieldState.Z();
  floatChecks[23] = fullFieldState.Px();
  floatChecks[24] = fullFieldState.Py();
  floatChecks[25] = fullFieldState.Pz();
  floatChecks[26] = fullFieldState.Covariance(0, 0);
  floatChecks[27] = coupledDca.Correlation(0, 0);
  floatChecks[28] = coupledDca.FirstCovariance(0, 0);
  floatChecks[29] = coupledDca.FirstJacobian(0, 0);
  integerChecks[0] = KFParticleGpuMath::Abs(zeroField.X() - line.X()) < 1.e-6f ? 1 : 0;
  integerChecks[1] = zeroBuilt ? 1 : 0;
  integerChecks[2] = fieldBuilt ? 1 : 0;
  integerChecks[3] = KFParticleGpuMath::Abs(zeroFieldMother.X() - lineMother.X()) < 1.e-5f ? 1 : 0;
  integerChecks[4] = lineMeasurementBuilt ? 1 : 0;
  integerChecks[5] = fieldMeasurementBuilt ? 1 : 0;
  integerChecks[6] = fieldCurrent.NDF();
  integerChecks[7] = fieldCurrent.Q();
  integerChecks[8] = fullFieldBuilt ? 1 : 0;
  integerChecks[9] = static_cast<int>(fullFieldStatus);
  integerChecks[10] = coupledDcaBuilt ? 1 : 0;
  integerChecks[11] = static_cast<int>(coupledDcaStatus);
}

XPU_EXPORT(KFParticleGpuV0LineTopologyProbe);
XPU_D void KFParticleGpuV0LineTopologyProbe::operator()(
  context& context, float* floatChecks, unsigned int* statusChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !floatChecks || !statusChecks) {
    return;
  }

  KFParticleGpuFitState candidate;
  candidate.Initialize();
  candidate.X() = 3.f;
  candidate.Y() = 4.f;
  candidate.Px() = 1.f;
  candidate.Covariance(0, 0) = 0.4f;
  candidate.Covariance(1, 0) = 0.05f;
  candidate.Covariance(1, 1) = 0.5f;
  candidate.Covariance(2, 2) = 0.6f;

  KFParticleGpuVertexState vertex;
  vertex.Initialize();
  vertex.Covariance(0) = 0.6f;
  vertex.Covariance(1) = 0.15f;
  vertex.Covariance(2) = 0.5f;
  vertex.Covariance(5) = 0.4f;

  KFParticleGpuV0LineTopologyResult result;
  const bool built = KFParticleGpuSelection::BuildV0LineTopology(candidate, vertex, result);
  floatChecks[0] = result.pathToPrimaryVertex;
  floatChecks[1] = result.lineDistance;
  floatChecks[2] = result.lineDistanceError;
  floatChecks[3] = result.lineLdL;
  floatChecks[4] = result.decayLength;
  floatChecks[5] = result.decayLengthError;
  floatChecks[6] = result.pointingCosine;
  floatChecks[7] = result.lineChi2PerNdf;
  statusChecks[0] = built ? 1u : 0u;
  statusChecks[1] = result.status;
}

XPU_EXPORT(KFParticleGpuFullFieldTwoDaughterProbe);
XPU_D void KFParticleGpuFullFieldTwoDaughterProbe::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  KFParticleGpuFullFieldTwoDaughterTrace* trace)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !tasks || !trace) {
    return;
  }
  trace[0].stage = 0u;
  const KFParticleGpuTwoDaughterTask task = tasks[0];

  if (!ValidateTwoDaughterTask(inputTracks, task)
      || task.transportMode != KFGpuTransportFullField) {
    return;
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
  KFParticleGpuFitState first;
  KFParticleGpuFitState second;
  first.Initialize(firstTrack, inputTracks.Charge(firstTrackIndex), firstMass);
  second.Initialize(secondTrack, inputTracks.Charge(secondTrackIndex), secondMass);
  first.NDF() = -1;
  KFParticleGpuFieldRegion firstField;
  KFParticleGpuFieldRegion secondField;
  LoadFieldRegion(inputTracks, firstTrackIndex, firstField);
  LoadFieldRegion(inputTracks, secondTrackIndex, secondField);
  trace[0].by = firstField.Get(first.Z()).y;
  float unusedDs[2] = {};
  KFParticleGpuMath::GetDStoParticleByCpuCompatible(
    first,
    second,
    trace[0].by,
    unusedDs,
    false,
    trace[0].firstRoots,
    trace[0].secondRoots);
  trace[0].useMiddlePoint = KFParticleGpuMath::UseCpuCompatibleMiddleDcaPoint(
    first, second, firstField, secondField, trace[0].by) ? 1u : 0u;
  KFParticleGpuMath::GetDStoParticleByCpuCompatibleDerivatives(
    first,
    second,
    trace[0].by,
    trace[0].useMiddlePoint != 0u,
    trace[0].dS,
    trace[0].dsdr,
    &trace[0].dcaArithmetic);
  trace[0].stage = 1u;

  if (!KFParticleGpuMath::BuildFullFieldDcaIndependentTransportStates(
        first,
        second,
        firstField,
        secondField,
        trace[0].preliminaryFirst,
        trace[0].preliminarySecond)) {
    return;
  }
  trace[0].stage = 2u;
  if (!KFParticleGpuMath::BuildFullFieldDcaMeasurementSeedAnalytic(
        trace[0].preliminaryFirst,
        trace[0].preliminarySecond,
        firstField,
        secondField,
        trace[0].secondPassCurrent,
        trace[0].secondPassMeasurement)) {
    return;
  }
  trace[0].stage = 3u;
  trace[0].fittedMother = trace[0].secondPassCurrent;
  if (!KFParticleGpuMath::AddDaughterWithEnergyFit(
        trace[0].fittedMother,
        trace[0].secondPassMeasurement,
        inputTracks.Charge(secondTrackIndex))) {
    return;
  }
  trace[0].fittedMother.SumDaughterMass() =
    first.SumDaughterMass() + second.SumDaughterMass();
  trace[0].fittedMother.MassHypo() = -1.f;
  trace[0].fittedMother.ConstructMethod() = 0;
  trace[0].stage = 4u;
}

XPU_EXPORT(KFParticleGpuCandidatePoolReadbackProbe);
XPU_D void KFParticleGpuCandidatePoolReadbackProbe::operator()(
  context& context,
  KFParticleGpuConstCandidatePoolView candidates,
  float* floatChecks,
  int* integerChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !floatChecks || !integerChecks) {
    return;
  }
  for (unsigned int index = 0; index < 17u; ++index) {
    floatChecks[index] = 0.f;
  }
  integerChecks[0] = candidates.Size() > 0u ? 1 : 0;
  if (!integerChecks[0]) {
    return;
  }

  KFParticleGpuFitState stored;
  LoadCandidateFit(candidates, 0u, stored);
  for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters; ++parameter) {
    floatChecks[parameter] = stored.Parameter(parameter);
    floatChecks[8 + parameter] = stored.Covariance(parameter, parameter);
  }
  floatChecks[16] = stored.Chi2();
  integerChecks[1] = stored.NDF();
  integerChecks[2] = stored.Q();
}

XPU_EXPORT(KFParticleGpuRoutedTwoDaughterProbe);
XPU_D void KFParticleGpuRoutedTwoDaughterProbe::operator()(
  context& context,
  unsigned int taskLimit,
  KFParticleGpuRoutedTwoDaughterTrace* trace,
  unsigned int traceCapacity)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (!trace || thread >= traceCapacity) {
    return;
  }
  KFParticleGpuRoutedTwoDaughterTrace& output = trace[thread];
  output = KFParticleGpuRoutedTwoDaughterTrace();

  const KFParticleGpuKernels& state = context.cmem<TheKFParticleFinder>();
  const KFParticleGpuTwoDaughterGenerationStorageView& generation =
    state.TwoDaughterGeneration();
  const KFParticleGpuTwoDaughterRoutingView& routing = state.TwoDaughterRouting();
  const KFParticleGpuCandidatePoolView& candidates = state.Candidates();
  output.taskCapacity = generation.TaskCapacity();
  output.descriptorCount = routing.DescriptorCount();
  output.candidateSize = candidates.Size();
  output.candidateCapacity = candidates.Capacity();
  output.daughterSize = candidates.Daughters().Size();
  output.daughterCapacity = candidates.Daughters().Capacity();
  if (generation.StoredTasks()) {
    output.storedTasks = generation.StoredTasks()[0];
  }
  output.status = 1u;

  const unsigned int effectiveCapacity =
    taskLimit < generation.TaskCapacity() ? taskLimit : generation.TaskCapacity();
  if (!generation.Tasks() || !generation.StoredTasks()
      || thread >= effectiveCapacity || thread >= output.storedTasks) {
    return;
  }
  output.routed = generation.Tasks()[thread];
  output.status |= 2u;
  if (output.routed.descriptorIndex >= routing.DescriptorCount()) {
    return;
  }
  output.descriptor = routing.Descriptors()[output.routed.descriptorIndex];
  output.status |= 4u;

  const auto& descriptor = output.descriptor;
  auto& task = output.task;
  task.channelId = descriptor.channelId;
  task.firstTrack = output.routed.firstTrackIndex;
  task.secondTrack = output.routed.secondTrackIndex;
  task.eventIndex = output.routed.eventIndex;
  task.flags = descriptor.flags;
  task.outputClass = descriptor.outputClass;
  task.motherPdg = descriptor.motherPdg;
  task.firstDaughterPdg = descriptor.firstDaughterPdg;
  task.secondDaughterPdg = descriptor.secondDaughterPdg;
  task.firstSourcePdg = descriptor.firstSourcePdg;
  task.firstAlternateSourcePdg = descriptor.firstAlternateSourcePdg;
  task.secondSourcePdg = descriptor.secondSourcePdg;
  task.secondAlternateSourcePdg = descriptor.secondAlternateSourcePdg;
  task.primaryVertexIndex = ResolveTwoDaughterPrimaryVertexIndex(
    state.InputTracks(), descriptor.primaryVertexIndex,
    output.routed.firstTrackIndex, output.routed.secondTrackIndex);
  task.firstMass = descriptor.firstMass;
  task.secondMass = descriptor.secondMass;
  task.motherMass = descriptor.motherMass;
  task.motherMassSigma = descriptor.motherMassSigma;
  task.secondaryMassSigmaCut = descriptor.secondaryMassSigmaCut;
  task.maxSecondaryTopoChi2PerNdf = descriptor.maxSecondaryTopoChi2PerNdf;
  task.minSecondaryLdL = descriptor.minSecondaryLdL;
  task.transportMode = descriptor.transportMode;
  output.status |= 8u;

  const KFParticleGpuConstInputTrackSoAView& tracks = state.InputTracks();
  if (ValidateTwoDaughterTask(tracks, task)) {
    output.status |= 16u;
  }
  if (BuildTwoDaughterCandidate(tracks, task, output.mother)) {
    output.status |= 32u;
  }
}

XPU_EXPORT(KFParticleGpuTwoDaughterBuildProbe);
XPU_D void KFParticleGpuTwoDaughterBuildProbe::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  float* floatChecks,
  int* integerChecks)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !tasks || !floatChecks || !integerChecks) {
    return;
  }
  for (unsigned int index = 0; index < 17u; ++index) {
    floatChecks[index] = 0.f;
  }
  integerChecks[0] = 0;
  const KFParticleGpuTwoDaughterTask task = tasks[0];
  KFParticleGpuFitState mother;
  if (!BuildTwoDaughterCandidate(inputTracks, task, mother)) {
    return;
  }
  for (int parameter = 0; parameter < KFParticleGpuFitState::NumberOfParameters; ++parameter) {
    floatChecks[parameter] = mother.Parameter(parameter);
    floatChecks[8 + parameter] = mother.Covariance(parameter, parameter);
  }
  floatChecks[16] = mother.Chi2();
  integerChecks[0] = 1;
  integerChecks[1] = mother.NDF();
  integerChecks[2] = mother.Q();
}

XPU_EXPORT(KFParticleGpuTwoDaughterArgumentProbe);
XPU_D void KFParticleGpuTwoDaughterArgumentProbe::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  const KFParticleGpuTwoDaughterTask* tasks,
  unsigned int numberOfTasks,
  KFParticleGpuCandidatePoolView candidates)
{
  const unsigned int thread =
    static_cast<unsigned int>(context.pos().block_idx_x() * context.pos().block_dim_x()
                              + context.pos().thread_idx_x());
  if (thread != 0u || !tasks || numberOfTasks == 0u || !candidates.CanStoreCandidate(0u)) {
    return;
  }

  const KFParticleGpuTwoDaughterTask task = tasks[0];
  candidates.SizeData()[0] = 1u;
  candidates.Daughters().SizeData()[0] = 0u;
  candidates.OverflowFlagsData()[0] = 0u;
  for (unsigned int component = 0; component < KFParticleGpuFitState::NumberOfParameters;
       ++component) {
    candidates.Fit().Parameter(component, 0u) = 1000.f + static_cast<float>(component);
    candidates.Fit().Covariance(component, 0u) = 2000.f + static_cast<float>(component);
  }
  candidates.Fit().FitScalar(KFParticleGpuFitSoALayout::Chi2, 0u) = 3000.f;
  candidates.Metadata().Pdg(0u) = static_cast<int>(inputTracks.Size());
  candidates.Metadata().PrimaryVertexIndex(0u) = static_cast<int>(inputTracks.Numerical().Stride());
  candidates.Metadata().EventIndex(0u) = task.firstTrack;
  candidates.Metadata().DaughterOffset(0u) = task.secondTrack;
  candidates.Metadata().DaughterCount(0u) = task.flags;
  candidates.Metadata().Flags(0u) = candidates.Fit().Stride();
  candidates.Metadata().ChannelId(0u) = candidates.Capacity();
  candidates.Metadata().Topology(0u) = KFGpuGraphTopologyInvalid;
  candidates.Metadata().OutputClass(0u) = KFGpuGraphOutputInvalid;
  candidates.Metadata().OperationStatus(0u) = KFGpuCandidateOperationAccepted;
  ClearCandidateDirectDaughters(candidates.Metadata(), 0u);
}

XPU_D void KFParticleGpuKernels::RunRoundTrip(KFParticleGpuRoundTrip::context& context,
                                              float mass,
                                              unsigned int eventIndex) const
{
  RunRoundTripImpl(context, fInputTracks, fCandidates, mass, eventIndex);
}
