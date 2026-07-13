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

XPU_EXPORT(KFParticleGpuRoundTripArgs);
XPU_D void KFParticleGpuRoundTripArgs::operator()(
  context& context,
  KFParticleGpuConstInputTrackSoAView inputTracks,
  KFParticleGpuCandidatePoolView candidates,
  float mass,
  unsigned int eventIndex)
{
  RunRoundTripImpl(context, inputTracks, candidates, mass, eventIndex);
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
  if (PassTwoDaughterTaskSourceCuts(inputTracks, source, task.firstTrack, task.secondTrack)) {
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
  if (!PassTwoDaughterTaskSourceCuts(inputTracks, source, task.firstTrack, task.secondTrack)) {
    return;
  }

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
  integerChecks[0] = KFParticleGpuMath::Abs(zeroField.X() - line.X()) < 1.e-6f ? 1 : 0;
  integerChecks[1] = zeroBuilt ? 1 : 0;
  integerChecks[2] = fieldBuilt ? 1 : 0;
  integerChecks[3] = KFParticleGpuMath::Abs(zeroFieldMother.X() - lineMother.X()) < 1.e-5f ? 1 : 0;
  integerChecks[4] = lineMeasurementBuilt ? 1 : 0;
  integerChecks[5] = fieldMeasurementBuilt ? 1 : 0;
  integerChecks[6] = fieldCurrent.NDF();
  integerChecks[7] = fieldCurrent.Q();
}

XPU_D void KFParticleGpuKernels::RunRoundTrip(KFParticleGpuRoundTrip::context& context,
                                              float mass,
                                              unsigned int eventIndex) const
{
  RunRoundTripImpl(context, fInputTracks, fCandidates, mass, eventIndex);
}
