/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "KFParticleGpuBufferManager.h"

#include "KFParticleGpuDecayPlan.h"

#include <xpu/host.h>

#include <cstdint>
#include <stdexcept>

namespace
{
  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    if (!buffer.get()) {
      return nullptr;
    }
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  template<typename T>
  void AllocateIo(xpu::buffer<T>& buffer, unsigned int size)
  {
    buffer.reset(size, xpu::buf_io);
  }

  template<typename T>
  void CopyIfAllocated(xpu::queue& queue, xpu::buffer<T>& buffer, xpu::direction direction)
  {
    if (buffer.get()) {
      queue.copy(buffer, direction);
    }
  }
}

struct KFParticleGpuBufferManager::Impl : KFParticleGpuDeviceStorage
{
  xpu::queue& fQueue;
  KFParticleGpuBufferCapacities fCapacities;
  unsigned int fTrackSize = 0;
  unsigned int fVertexSize = 0;
  unsigned int fEventSize = 0;
  unsigned int fTwoDaughterRoutingDescriptorSize = 0;
  unsigned int fTwoDaughterRoutingCompatibilitySize = 0;
  unsigned int fTwoDaughterRoutingGroupSize = 0;
  unsigned long long fTwoDaughterRoutingRevision = 0u;
  unsigned int fV0TrackRoutingDescriptorSize = 0;
  unsigned int fV0TrackRoutingCompatibilitySize = 0;
  unsigned int fV0TrackRoutingGroupSize = 0;
  unsigned long long fV0TrackRoutingRevision = 0u;
  unsigned int fDecayGraphNodeSize = 0u;
  unsigned int fDecayGraphGroupSize = 0u;
  unsigned int fDecayGraphFamilySize = 0u;
  unsigned long long fDecayGraphRevision = 0u;
  unsigned int fGraphOperationDescriptorSize = 0u;
  unsigned long long fGraphOperationRevision = 0u;
  std::uint64_t fCapacityGrowthCount = 0u;

  explicit Impl(xpu::queue& queue) : fQueue(queue) {}

  void AllocateTracks(unsigned int capacity, bool withField)
  {
    AllocateIo(fTrackParameters, KFParticleGpuTrackState::NumberOfParameters * capacity);
    AllocateIo(fTrackCovariances, KFParticleGpuTrackState::NumberOfCovarianceElements * capacity);
    AllocateIo(fTrackChiToPrimaryVertex, capacity);
    AllocateIo(fTrackIntegers, KFParticleGpuTrackInputLayout::NumberOfIntegerComponents * capacity);
    if (withField) {
      AllocateIo(fTrackField, KFParticleGpuFieldRegion::NumberOfCoefficients * capacity);
    }
    else {
      fTrackField.reset();
    }
  }

  void AllocateVertices(unsigned int capacity)
  {
    AllocateIo(fVertexParameters, KFParticleGpuVertexState::NumberOfParameters * capacity);
    AllocateIo(fVertexCovariances, KFParticleGpuVertexState::NumberOfCovarianceElements * capacity);
    AllocateIo(fVertexChi2, capacity);
    AllocateIo(fVertexIntegers, KFParticleGpuVertexSoALayout::NumberOfIntegerComponents * capacity);
  }

  void AllocateCandidates(unsigned int capacity)
  {
    AllocateIo(fCandidateParameters, KFParticleGpuFitState::NumberOfParameters * capacity);
    AllocateIo(fCandidateCovariances, KFParticleGpuFitState::NumberOfCovarianceElements * capacity);
    AllocateIo(fCandidateFitScalars, KFParticleGpuFitSoALayout::NumberOfFloatComponents * capacity);
    AllocateIo(fCandidateFitIntegers, KFParticleGpuFitSoALayout::NumberOfIntegerComponents * capacity);
    AllocateIo(
      fCandidateMetadataIntegers,
      KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents * capacity);
    AllocateIo(
      fCandidateMetadataUnsigned,
      KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents * capacity);
    AllocateIo(fCandidateRoutingDescriptorIndices, capacity);
    AllocateIo(fV0SelectionResults, capacity);
  }

  void EnsureCounters()
  {
    if (!fCandidateSize.get()) {
      AllocateIo(fCandidateSize, 1);
      AllocateIo(fDaughterSize, 1);
      AllocateIo(fOverflowFlags, 1);
    }
  }

  void AllocateSelectedCandidates(unsigned int capacity)
  {
    AllocateIo(fSelectedCandidateIndices, capacity);
    AllocateIo(fSelectedCandidateChannelIds, capacity);
    if (!fSelectedCandidateSize.get()) {
      AllocateIo(fSelectedCandidateSize, 1);
      AllocateIo(fSelectedCandidateOverflowFlags, 1);
    }
  }

  void AllocateTwoDaughterTasks(unsigned int capacity)
  {
    AllocateIo(fTwoDaughterTasks, capacity);
    if (!fTwoDaughterTaskCount.get()) {
      AllocateIo(fTwoDaughterTaskCount, 1);
      AllocateIo(fTwoDaughterTotalPairCount, 1);
      AllocateIo(fTwoDaughterTaskOverflowFlags, 1);
    }
  }

  void AllocateV0TrackTasks(unsigned int capacity)
  {
    AllocateIo(fV0TrackTasks, capacity);
    if (!fV0TrackTaskCount.get()) {
      AllocateIo(fV0TrackTaskCount, 1);
      AllocateIo(fV0TrackTotalPairCount, 1);
      AllocateIo(fV0TrackTaskOverflowFlags, 1);
    }
  }

  void AllocateTwoDaughterRoutedTasks(unsigned int capacity)
  {
    AllocateIo(fTwoDaughterRoutedTasks, capacity);
    if (!fTwoDaughterRoutedAcceptedTaskCount.get()) {
      AllocateIo(fTwoDaughterRoutingVisitedPairCount, 1u);
      AllocateIo(fTwoDaughterRoutingActiveBitCount, 1u);
      AllocateIo(fTwoDaughterRoutedAcceptedTaskCount, 1u);
      AllocateIo(fTwoDaughterRoutedStoredTaskCount, 1u);
      AllocateIo(fTwoDaughterRoutingBlockReservationCount, 1u);
      AllocateIo(fTwoDaughterRoutedTaskOverflowFlags, 1u);
    }
  }

  void AllocateV0TrackRoutedTasks(unsigned int capacity)
  {
    AllocateIo(fV0TrackRoutedTasks, capacity);
    if (!fV0TrackRoutedAcceptedTaskCount.get()) {
      AllocateIo(fV0TrackRoutingVisitedPairCount, 1u);
      AllocateIo(fV0TrackRoutingActiveBitCount, 1u);
      AllocateIo(fV0TrackRoutedAcceptedTaskCount, 1u);
      AllocateIo(fV0TrackRoutedStoredTaskCount, 1u);
      AllocateIo(fV0TrackRoutingBlockReservationCount, 1u);
      AllocateIo(fV0TrackRoutedTaskOverflowFlags, 1u);
    }
  }

  void EnsureV0TrackRoutingControl()
  {
    if (!fV0TrackRoutingEnabledChannels.get()) {
      AllocateIo(fV0TrackRoutingEnabledChannels, 1u);
    }
  }

  void EnsureTwoDaughterRoutingControl()
  {
    if (!fTwoDaughterRoutingEnabledChannels.get()) {
      AllocateIo(fTwoDaughterRoutingEnabledChannels, 1u);
    }
  }

  void AllocateGraphOperationTasks(unsigned int capacity)
  {
    AllocateIo(fGraphOperationTasks, capacity);
    AllocateIo(fGraphOperationResults, capacity);
    if (!fGraphOperationVisitedCombinations.get()) {
      AllocateIo(fGraphOperationVisitedCombinations, 1u);
      AllocateIo(fGraphOperationAcceptedTasks, 1u);
      AllocateIo(fGraphOperationStoredTasks, 1u);
      AllocateIo(fGraphOperationConstructedCandidates, 1u);
      AllocateIo(fGraphOperationRejectedTasks, 1u);
      AllocateIo(fGraphOperationOverflowFlags, 1u);
    }
  }
};

KFParticleGpuBufferManager::KFParticleGpuBufferManager(xpu::queue& queue) : fImpl(new Impl(queue)) {}

KFParticleGpuBufferManager::~KFParticleGpuBufferManager() = default;

void KFParticleGpuBufferManager::EnsureCapacity(const KFParticleGpuBufferCapacities& requested)
{
  const KFParticleGpuBufferCapacities before = fImpl->fCapacities;
  const bool withField =
    requested.nonhomogeneousField || fImpl->fCapacities.nonhomogeneousField;
  if (requested.tracks > fImpl->fCapacities.tracks
      || (requested.nonhomogeneousField && !fImpl->fCapacities.nonhomogeneousField)) {
    const unsigned int capacity =
      requested.tracks > fImpl->fCapacities.tracks ? requested.tracks : fImpl->fCapacities.tracks;
    fImpl->AllocateTracks(capacity, withField);
    fImpl->fCapacities.tracks = capacity;
    fImpl->fCapacities.nonhomogeneousField = withField;
  }

  if (requested.vertices > fImpl->fCapacities.vertices) {
    fImpl->AllocateVertices(requested.vertices);
    fImpl->fCapacities.vertices = requested.vertices;
  }
  if (requested.events > fImpl->fCapacities.events) {
    AllocateIo(fImpl->fEvents, requested.events);
    fImpl->fCapacities.events = requested.events;
  }
  if (requested.candidates > fImpl->fCapacities.candidates) {
    fImpl->AllocateCandidates(requested.candidates);
    fImpl->fCapacities.candidates = requested.candidates;
  }
  if (requested.daughterIds > fImpl->fCapacities.daughterIds) {
    AllocateIo(fImpl->fDaughterSourceIds, requested.daughterIds);
    fImpl->fCapacities.daughterIds = requested.daughterIds;
  }
  if (requested.twoDaughterTasks > fImpl->fCapacities.twoDaughterTasks) {
    fImpl->AllocateTwoDaughterTasks(requested.twoDaughterTasks);
    fImpl->fCapacities.twoDaughterTasks = requested.twoDaughterTasks;
  }
  if (requested.twoDaughterRoutedTasks > fImpl->fCapacities.twoDaughterRoutedTasks) {
    fImpl->AllocateTwoDaughterRoutedTasks(requested.twoDaughterRoutedTasks);
    fImpl->fCapacities.twoDaughterRoutedTasks = requested.twoDaughterRoutedTasks;
  }
  if (requested.twoDaughterRoutingDescriptors
      > fImpl->fCapacities.twoDaughterRoutingDescriptors) {
    AllocateIo(
      fImpl->fTwoDaughterRoutingDescriptors, requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterRoutingChannelVisitedCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterRoutingChannelAcceptedCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterRoutingChannelStoredCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterRoutingChannelConstructedCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterSelectionAcceptedCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterSelectionStoredCounters,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterSelectionOffsets,
      requested.twoDaughterRoutingDescriptors);
    AllocateIo(
      fImpl->fTwoDaughterSelectionCursors,
      requested.twoDaughterRoutingDescriptors);
    fImpl->fCapacities.twoDaughterRoutingDescriptors =
      requested.twoDaughterRoutingDescriptors;
  }
  if (requested.twoDaughterRoutingCompatibility
      > fImpl->fCapacities.twoDaughterRoutingCompatibility) {
    AllocateIo(
      fImpl->fTwoDaughterRoutingCompatibility,
      requested.twoDaughterRoutingCompatibility);
    fImpl->fCapacities.twoDaughterRoutingCompatibility =
      requested.twoDaughterRoutingCompatibility;
  }
  if (requested.twoDaughterRoutingGroups
      > fImpl->fCapacities.twoDaughterRoutingGroups) {
    AllocateIo(fImpl->fTwoDaughterRoutingGroups, requested.twoDaughterRoutingGroups);
    fImpl->fCapacities.twoDaughterRoutingGroups =
      requested.twoDaughterRoutingGroups;
  }
  if (requested.v0TrackTasks > fImpl->fCapacities.v0TrackTasks) {
    fImpl->AllocateV0TrackTasks(requested.v0TrackTasks);
    fImpl->fCapacities.v0TrackTasks = requested.v0TrackTasks;
  }
  if (requested.v0TrackRoutedTasks > fImpl->fCapacities.v0TrackRoutedTasks) {
    fImpl->AllocateV0TrackRoutedTasks(requested.v0TrackRoutedTasks);
    fImpl->fCapacities.v0TrackRoutedTasks = requested.v0TrackRoutedTasks;
  }
  if (requested.v0TrackRoutingDescriptors > fImpl->fCapacities.v0TrackRoutingDescriptors) {
    AllocateIo(fImpl->fV0TrackRoutingDescriptors, requested.v0TrackRoutingDescriptors);
    AllocateIo(fImpl->fV0TrackRoutingChannelVisitedCounters, requested.v0TrackRoutingDescriptors);
    AllocateIo(fImpl->fV0TrackRoutingChannelAcceptedCounters, requested.v0TrackRoutingDescriptors);
    AllocateIo(fImpl->fV0TrackRoutingChannelStoredCounters, requested.v0TrackRoutingDescriptors);
    AllocateIo(fImpl->fV0TrackRoutingChannelConstructedCounters, requested.v0TrackRoutingDescriptors);
    fImpl->fCapacities.v0TrackRoutingDescriptors = requested.v0TrackRoutingDescriptors;
  }
  if (requested.v0TrackRoutingCompatibility > fImpl->fCapacities.v0TrackRoutingCompatibility) {
    AllocateIo(fImpl->fV0TrackRoutingCompatibility, requested.v0TrackRoutingCompatibility);
    fImpl->fCapacities.v0TrackRoutingCompatibility = requested.v0TrackRoutingCompatibility;
  }
  if (requested.v0TrackRoutingGroups > fImpl->fCapacities.v0TrackRoutingGroups) {
    AllocateIo(fImpl->fV0TrackRoutingGroups, requested.v0TrackRoutingGroups);
    fImpl->fCapacities.v0TrackRoutingGroups = requested.v0TrackRoutingGroups;
  }
  if (requested.decayGraphNodes > fImpl->fCapacities.decayGraphNodes) {
    AllocateIo(fImpl->fDecayGraphNodes, requested.decayGraphNodes);
    fImpl->fCapacities.decayGraphNodes = requested.decayGraphNodes;
  }
  if (requested.decayGraphGroups > fImpl->fCapacities.decayGraphGroups) {
    AllocateIo(fImpl->fDecayGraphGroups, requested.decayGraphGroups);
    fImpl->fCapacities.decayGraphGroups = requested.decayGraphGroups;
  }
  if (requested.decayGraphFamilies > fImpl->fCapacities.decayGraphFamilies) {
    AllocateIo(fImpl->fDecayGraphFamilyCoverage, requested.decayGraphFamilies);
    fImpl->fCapacities.decayGraphFamilies = requested.decayGraphFamilies;
  }
  if (requested.graphOperationDescriptors
      > fImpl->fCapacities.graphOperationDescriptors) {
    AllocateIo(
      fImpl->fGraphOperationDescriptors, requested.graphOperationDescriptors);
    AllocateIo(
      fImpl->fGraphOperationChannelVisitedCounters,
      requested.graphOperationDescriptors);
    AllocateIo(
      fImpl->fGraphOperationChannelAcceptedCounters,
      requested.graphOperationDescriptors);
    AllocateIo(
      fImpl->fGraphOperationChannelStoredCounters,
      requested.graphOperationDescriptors);
    AllocateIo(
      fImpl->fGraphOperationChannelConstructedCounters,
      requested.graphOperationDescriptors);
    AllocateIo(
      fImpl->fGraphOperationChannelRejectedCounters,
      requested.graphOperationDescriptors);
    fImpl->fCapacities.graphOperationDescriptors =
      requested.graphOperationDescriptors;
  }
  if (requested.graphOperationTasks > fImpl->fCapacities.graphOperationTasks) {
    fImpl->AllocateGraphOperationTasks(requested.graphOperationTasks);
    fImpl->fCapacities.graphOperationTasks = requested.graphOperationTasks;
  }
  const unsigned int requestedSelectedCandidates = requested.selectedCandidates > 0u
                                                     ? requested.selectedCandidates
                                                     : requested.candidates;
  if (requestedSelectedCandidates > fImpl->fCapacities.selectedCandidates) {
    fImpl->AllocateSelectedCandidates(requestedSelectedCandidates);
    fImpl->fCapacities.selectedCandidates = requestedSelectedCandidates;
  }
  fImpl->EnsureCounters();
  const KFParticleGpuBufferCapacities& after = fImpl->fCapacities;
  if (before.tracks != after.tracks || before.vertices != after.vertices
      || before.events != after.events || before.candidates != after.candidates
      || before.daughterIds != after.daughterIds
      || before.selectedCandidates != after.selectedCandidates
      || before.twoDaughterTasks != after.twoDaughterTasks
      || before.twoDaughterRoutedTasks != after.twoDaughterRoutedTasks
      || before.twoDaughterRoutingDescriptors != after.twoDaughterRoutingDescriptors
      || before.twoDaughterRoutingCompatibility != after.twoDaughterRoutingCompatibility
      || before.twoDaughterRoutingGroups != after.twoDaughterRoutingGroups
      || before.v0TrackTasks != after.v0TrackTasks
      || before.v0TrackRoutedTasks != after.v0TrackRoutedTasks
      || before.v0TrackRoutingDescriptors != after.v0TrackRoutingDescriptors
      || before.v0TrackRoutingCompatibility != after.v0TrackRoutingCompatibility
      || before.v0TrackRoutingGroups != after.v0TrackRoutingGroups
      || before.decayGraphNodes != after.decayGraphNodes
      || before.decayGraphGroups != after.decayGraphGroups
      || before.decayGraphFamilies != after.decayGraphFamilies
      || before.graphOperationDescriptors != after.graphOperationDescriptors
      || before.graphOperationTasks != after.graphOperationTasks
      || before.nonhomogeneousField != after.nonhomogeneousField) {
    ++fImpl->fCapacityGrowthCount;
  }
}

const KFParticleGpuBufferCapacities& KFParticleGpuBufferManager::Capacities() const
{
  return fImpl->fCapacities;
}

const KFParticleGpuDeviceStorage& KFParticleGpuBufferManager::Storage() const
{
  return *fImpl;
}

std::uint64_t KFParticleGpuBufferManager::CapacityGrowthCount() const
{
  return fImpl->fCapacityGrowthCount;
}

std::uint64_t KFParticleGpuBufferManager::InputPayloadBytes() const
{
  const auto& c = fImpl->fCapacities;
  std::uint64_t bytes = static_cast<std::uint64_t>(c.tracks)
    * (sizeof(float) * (KFParticleGpuTrackState::NumberOfParameters
                        + KFParticleGpuTrackState::NumberOfCovarianceElements + 1u)
       + sizeof(int) * KFParticleGpuTrackInputLayout::NumberOfIntegerComponents);
  if (c.nonhomogeneousField) {
    bytes += static_cast<std::uint64_t>(c.tracks) * sizeof(float)
             * KFParticleGpuFieldRegion::NumberOfCoefficients;
  }
  bytes += static_cast<std::uint64_t>(c.vertices)
    * (sizeof(float) * (KFParticleGpuVertexState::NumberOfParameters
                        + KFParticleGpuVertexState::NumberOfCovarianceElements + 1u)
       + sizeof(int) * KFParticleGpuVertexSoALayout::NumberOfIntegerComponents);
  bytes += static_cast<std::uint64_t>(c.events) * sizeof(KFParticleGpuEventDesc);
  return bytes;
}

std::uint64_t KFParticleGpuBufferManager::CandidatePayloadBytes() const
{
  const auto& c = fImpl->fCapacities;
  return static_cast<std::uint64_t>(c.candidates)
           * (sizeof(float) * (KFParticleGpuFitState::NumberOfParameters
                               + KFParticleGpuFitState::NumberOfCovarianceElements
                               + KFParticleGpuFitSoALayout::NumberOfFloatComponents)
              + sizeof(int) * (KFParticleGpuFitSoALayout::NumberOfIntegerComponents
                               + KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents)
              + sizeof(unsigned int)
                  * KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents
              + sizeof(KFParticleGpuV0SelectionResult))
         + static_cast<std::uint64_t>(c.daughterIds) * sizeof(int)
         + 3u * sizeof(unsigned int);
}

std::uint64_t KFParticleGpuBufferManager::SelectedCandidatePayloadBytes() const
{
  return static_cast<std::uint64_t>(fImpl->fCapacities.selectedCandidates)
           * 2u * sizeof(unsigned int)
         + 2u * sizeof(unsigned int);
}

std::uint64_t KFParticleGpuBufferManager::AllocatedBytes() const
{
  const auto& c = fImpl->fCapacities;
  std::uint64_t bytes = InputPayloadBytes() + CandidatePayloadBytes()
                        + SelectedCandidatePayloadBytes();
  bytes += static_cast<std::uint64_t>(c.twoDaughterTasks)
           * sizeof(KFParticleGpuTwoDaughterTask);
  bytes += static_cast<std::uint64_t>(c.twoDaughterRoutedTasks)
           * sizeof(KFParticleGpuTwoDaughterRoutedTask);
  bytes += static_cast<std::uint64_t>(c.v0TrackTasks)
           * sizeof(KFParticleGpuV0TrackTask);
  bytes += static_cast<std::uint64_t>(c.v0TrackRoutedTasks)
           * sizeof(KFParticleGpuV0TrackRoutedTask);
  bytes += static_cast<std::uint64_t>(c.graphOperationTasks)
           * (sizeof(KFParticleGpuGraphOperationTask)
              + sizeof(KFParticleGpuGraphOperationResult));
  return bytes;
}

void KFParticleGpuBufferManager::SetInputSizes(unsigned int tracks,
                                               unsigned int vertices,
                                               unsigned int events)
{
  if (tracks > fImpl->fCapacities.tracks
      || vertices > fImpl->fCapacities.vertices
      || events > fImpl->fCapacities.events) {
    throw std::out_of_range("KFParticle GPU input size exceeds buffer capacity");
  }
  fImpl->fTrackSize = tracks;
  fImpl->fVertexSize = vertices;
  fImpl->fEventSize = events;
}

void KFParticleGpuBufferManager::EnsureTwoDaughterTaskCapacity(unsigned int capacity)
{
  if (capacity <= fImpl->fCapacities.twoDaughterTasks) {
    return;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.twoDaughterTasks = capacity;
  EnsureCapacity(requested);
}

void KFParticleGpuBufferManager::EnsureTwoDaughterRoutedTaskCapacity(unsigned int capacity)
{
  if (capacity <= fImpl->fCapacities.twoDaughterRoutedTasks) {
    return;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.twoDaughterRoutedTasks = capacity;
  EnsureCapacity(requested);
}

void KFParticleGpuBufferManager::EnsureV0TrackTaskCapacity(unsigned int capacity)
{
  if (capacity <= fImpl->fCapacities.v0TrackTasks) {
    return;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.v0TrackTasks = capacity;
  EnsureCapacity(requested);
}

void KFParticleGpuBufferManager::EnsureV0TrackRoutedTaskCapacity(unsigned int capacity)
{
  if (capacity <= fImpl->fCapacities.v0TrackRoutedTasks) {
    return;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.v0TrackRoutedTasks = capacity;
  EnsureCapacity(requested);
}

void KFParticleGpuBufferManager::EnsureGraphOperationTaskCapacity(
  unsigned int capacity)
{
  if (capacity <= fImpl->fCapacities.graphOperationTasks) {
    return;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.graphOperationTasks = capacity;
  EnsureCapacity(requested);
}

unsigned int KFParticleGpuBufferManager::TrackSize() const
{
  return fImpl->fTrackSize;
}

unsigned int KFParticleGpuBufferManager::VertexSize() const
{
  return fImpl->fVertexSize;
}

unsigned int KFParticleGpuBufferManager::EventSize() const
{
  return fImpl->fEventSize;
}

KFParticleGpuInputTrackSoAView KFParticleGpuBufferManager::HostInputTracks()
{
  return KFParticleGpuInputTrackSoAView(
    HostPointer(fImpl->fTrackParameters),
    HostPointer(fImpl->fTrackCovariances),
    fImpl->fTrackField.get() ? HostPointer(fImpl->fTrackField) : nullptr,
    HostPointer(fImpl->fTrackChiToPrimaryVertex),
    HostPointer(fImpl->fTrackIntegers),
    fImpl->fTrackSize,
    fImpl->fCapacities.tracks);
}

KFParticleGpuInputTrackSoAView KFParticleGpuBufferManager::DeviceInputTracks()
{
  return KFParticleGpuInputTrackSoAView(
    fImpl->fTrackParameters.get(),
    fImpl->fTrackCovariances.get(),
    fImpl->fTrackField.get(),
    fImpl->fTrackChiToPrimaryVertex.get(),
    fImpl->fTrackIntegers.get(),
    fImpl->fTrackSize,
    fImpl->fCapacities.tracks);
}

KFParticleGpuVertexSoAView KFParticleGpuBufferManager::HostPrimaryVertices()
{
  return KFParticleGpuVertexSoAView(
    HostPointer(fImpl->fVertexParameters),
    HostPointer(fImpl->fVertexCovariances),
    HostPointer(fImpl->fVertexChi2),
    HostPointer(fImpl->fVertexIntegers),
    fImpl->fVertexSize,
    fImpl->fCapacities.vertices);
}

KFParticleGpuVertexSoAView KFParticleGpuBufferManager::DevicePrimaryVertices()
{
  return KFParticleGpuVertexSoAView(
    fImpl->fVertexParameters.get(),
    fImpl->fVertexCovariances.get(),
    fImpl->fVertexChi2.get(),
    fImpl->fVertexIntegers.get(),
    fImpl->fVertexSize,
    fImpl->fCapacities.vertices);
}

KFParticleGpuEventDesc* KFParticleGpuBufferManager::HostEvents()
{
  return HostPointer(fImpl->fEvents);
}

KFParticleGpuEventDesc* KFParticleGpuBufferManager::DeviceEvents()
{
  return fImpl->fEvents.get();
}

KFParticleGpuCandidatePoolView KFParticleGpuBufferManager::HostCandidates()
{
  const unsigned int capacity = fImpl->fCapacities.candidates;
  KFParticleGpuFitSoAView fit(
    HostPointer(fImpl->fCandidateParameters),
    HostPointer(fImpl->fCandidateCovariances),
    HostPointer(fImpl->fCandidateFitScalars),
    HostPointer(fImpl->fCandidateFitIntegers),
    capacity,
    capacity);
  KFParticleGpuCandidateMetadataSoAView metadata(
    HostPointer(fImpl->fCandidateMetadataIntegers),
    HostPointer(fImpl->fCandidateMetadataUnsigned),
    capacity);
  KFParticleGpuDaughterStorageView daughters(
    HostPointer(fImpl->fDaughterSourceIds),
    HostPointer(fImpl->fDaughterSize),
    fImpl->fCapacities.daughterIds);
  return KFParticleGpuCandidatePoolView(
    fit,
    metadata,
    daughters,
    HostPointer(fImpl->fCandidateSize),
    HostPointer(fImpl->fOverflowFlags),
    capacity);
}

KFParticleGpuCandidatePoolView KFParticleGpuBufferManager::DeviceCandidates()
{
  const unsigned int capacity = fImpl->fCapacities.candidates;
  KFParticleGpuFitSoAView fit(
    fImpl->fCandidateParameters.get(),
    fImpl->fCandidateCovariances.get(),
    fImpl->fCandidateFitScalars.get(),
    fImpl->fCandidateFitIntegers.get(),
    capacity,
    capacity);
  KFParticleGpuCandidateMetadataSoAView metadata(
    fImpl->fCandidateMetadataIntegers.get(),
    fImpl->fCandidateMetadataUnsigned.get(),
    capacity);
  KFParticleGpuDaughterStorageView daughters(
    fImpl->fDaughterSourceIds.get(),
    fImpl->fDaughterSize.get(),
    fImpl->fCapacities.daughterIds);
  return KFParticleGpuCandidatePoolView(
    fit,
    metadata,
    daughters,
    fImpl->fCandidateSize.get(),
    fImpl->fOverflowFlags.get(),
    capacity);
}

KFParticleGpuV0SelectionResultView KFParticleGpuBufferManager::HostV0SelectionResults()
{
  return KFParticleGpuV0SelectionResultView(
    HostPointer(fImpl->fV0SelectionResults), fImpl->fCapacities.candidates);
}

KFParticleGpuV0SelectionResultView KFParticleGpuBufferManager::DeviceV0SelectionResults()
{
  return KFParticleGpuV0SelectionResultView(
    fImpl->fV0SelectionResults.get(), fImpl->fCapacities.candidates);
}

KFParticleGpuSelectedCandidateIndexView KFParticleGpuBufferManager::HostSelectedCandidates()
{
  return KFParticleGpuSelectedCandidateIndexView(
    HostPointer(fImpl->fSelectedCandidateIndices),
    HostPointer(fImpl->fSelectedCandidateChannelIds),
    HostPointer(fImpl->fSelectedCandidateSize),
    HostPointer(fImpl->fSelectedCandidateOverflowFlags),
    fImpl->fCapacities.selectedCandidates);
}

KFParticleGpuSelectedCandidateIndexView KFParticleGpuBufferManager::DeviceSelectedCandidates()
{
  return KFParticleGpuSelectedCandidateIndexView(
    fImpl->fSelectedCandidateIndices.get(),
    fImpl->fSelectedCandidateChannelIds.get(),
    fImpl->fSelectedCandidateSize.get(),
    fImpl->fSelectedCandidateOverflowFlags.get(),
    fImpl->fCapacities.selectedCandidates);
}

KFParticleGpuCandidateDescriptorIndexView
KFParticleGpuBufferManager::HostCandidateDescriptorIndices()
{
  return KFParticleGpuCandidateDescriptorIndexView(
    HostPointer(fImpl->fCandidateRoutingDescriptorIndices),
    fImpl->fCapacities.candidates);
}

KFParticleGpuCandidateDescriptorIndexView
KFParticleGpuBufferManager::DeviceCandidateDescriptorIndices()
{
  return KFParticleGpuCandidateDescriptorIndexView(
    fImpl->fCandidateRoutingDescriptorIndices.get(),
    fImpl->fCapacities.candidates);
}

KFParticleGpuTwoDaughterRoutingView KFParticleGpuBufferManager::HostTwoDaughterRouting()
{
  return KFParticleGpuTwoDaughterRoutingView(
    HostPointer(fImpl->fTwoDaughterRoutingDescriptors),
    fImpl->fTwoDaughterRoutingDescriptorSize,
    HostPointer(fImpl->fTwoDaughterRoutingCompatibility),
    fImpl->fTwoDaughterRoutingCompatibilitySize,
    HostPointer(fImpl->fTwoDaughterRoutingGroups),
    fImpl->fTwoDaughterRoutingGroupSize,
    HostPointer(fImpl->fTwoDaughterRoutingEnabledChannels),
    HostPointer(fImpl->fTwoDaughterRoutingChannelVisitedCounters),
    HostPointer(fImpl->fTwoDaughterRoutingChannelAcceptedCounters),
    HostPointer(fImpl->fTwoDaughterRoutingChannelStoredCounters),
    HostPointer(fImpl->fTwoDaughterRoutingChannelConstructedCounters));
}

KFParticleGpuTwoDaughterRoutingView KFParticleGpuBufferManager::DeviceTwoDaughterRouting()
{
  return KFParticleGpuTwoDaughterRoutingView(
    fImpl->fTwoDaughterRoutingDescriptors.get(),
    fImpl->fTwoDaughterRoutingDescriptorSize,
    fImpl->fTwoDaughterRoutingCompatibility.get(),
    fImpl->fTwoDaughterRoutingCompatibilitySize,
    fImpl->fTwoDaughterRoutingGroups.get(),
    fImpl->fTwoDaughterRoutingGroupSize,
    fImpl->fTwoDaughterRoutingEnabledChannels.get(),
    fImpl->fTwoDaughterRoutingChannelVisitedCounters.get(),
    fImpl->fTwoDaughterRoutingChannelAcceptedCounters.get(),
    fImpl->fTwoDaughterRoutingChannelStoredCounters.get(),
    fImpl->fTwoDaughterRoutingChannelConstructedCounters.get());
}

KFParticleGpuTwoDaughterSelectionWorkspaceView
KFParticleGpuBufferManager::HostTwoDaughterSelectionWorkspace()
{
  return KFParticleGpuTwoDaughterSelectionWorkspaceView(
    HostPointer(fImpl->fTwoDaughterSelectionAcceptedCounters),
    HostPointer(fImpl->fTwoDaughterSelectionStoredCounters),
    HostPointer(fImpl->fTwoDaughterSelectionOffsets),
    HostPointer(fImpl->fTwoDaughterSelectionCursors),
    fImpl->fTwoDaughterRoutingDescriptorSize);
}

KFParticleGpuTwoDaughterSelectionWorkspaceView
KFParticleGpuBufferManager::DeviceTwoDaughterSelectionWorkspace()
{
  return KFParticleGpuTwoDaughterSelectionWorkspaceView(
    fImpl->fTwoDaughterSelectionAcceptedCounters.get(),
    fImpl->fTwoDaughterSelectionStoredCounters.get(),
    fImpl->fTwoDaughterSelectionOffsets.get(),
    fImpl->fTwoDaughterSelectionCursors.get(),
    fImpl->fTwoDaughterRoutingDescriptorSize);
}

KFParticleGpuTwoDaughterGenerationStorageView
KFParticleGpuBufferManager::DeviceTwoDaughterGenerationStorage()
{
  return KFParticleGpuTwoDaughterGenerationStorageView(
    fImpl->fTwoDaughterRoutedTasks.get(),
    fImpl->fCapacities.twoDaughterRoutedTasks,
    fImpl->fTwoDaughterRoutingVisitedPairCount.get(),
    fImpl->fTwoDaughterRoutingActiveBitCount.get(),
    fImpl->fTwoDaughterRoutedAcceptedTaskCount.get(),
    fImpl->fTwoDaughterRoutedStoredTaskCount.get(),
    fImpl->fTwoDaughterRoutingBlockReservationCount.get(),
    fImpl->fTwoDaughterRoutedTaskOverflowFlags.get(),
    DeviceTwoDaughterSelectionWorkspace());
}

KFParticleGpuV0TrackRoutingView KFParticleGpuBufferManager::HostV0TrackRouting()
{
  return KFParticleGpuV0TrackRoutingView(
    HostPointer(fImpl->fV0TrackRoutingDescriptors),
    fImpl->fV0TrackRoutingDescriptorSize,
    HostPointer(fImpl->fV0TrackRoutingCompatibility),
    fImpl->fV0TrackRoutingCompatibilitySize,
    HostPointer(fImpl->fV0TrackRoutingGroups),
    fImpl->fV0TrackRoutingGroupSize,
    HostPointer(fImpl->fV0TrackRoutingEnabledChannels),
    HostPointer(fImpl->fV0TrackRoutingChannelVisitedCounters),
    HostPointer(fImpl->fV0TrackRoutingChannelAcceptedCounters),
    HostPointer(fImpl->fV0TrackRoutingChannelStoredCounters),
    HostPointer(fImpl->fV0TrackRoutingChannelConstructedCounters));
}

KFParticleGpuV0TrackRoutingView KFParticleGpuBufferManager::DeviceV0TrackRouting()
{
  return KFParticleGpuV0TrackRoutingView(
    fImpl->fV0TrackRoutingDescriptors.get(),
    fImpl->fV0TrackRoutingDescriptorSize,
    fImpl->fV0TrackRoutingCompatibility.get(),
    fImpl->fV0TrackRoutingCompatibilitySize,
    fImpl->fV0TrackRoutingGroups.get(),
    fImpl->fV0TrackRoutingGroupSize,
    fImpl->fV0TrackRoutingEnabledChannels.get(),
    fImpl->fV0TrackRoutingChannelVisitedCounters.get(),
    fImpl->fV0TrackRoutingChannelAcceptedCounters.get(),
    fImpl->fV0TrackRoutingChannelStoredCounters.get(),
    fImpl->fV0TrackRoutingChannelConstructedCounters.get());
}

KFParticleGpuV0TrackGenerationStorageView
KFParticleGpuBufferManager::DeviceV0TrackGenerationStorage()
{
  return KFParticleGpuV0TrackGenerationStorageView(
    fImpl->fV0TrackRoutedTasks.get(),
    fImpl->fCapacities.v0TrackRoutedTasks,
    fImpl->fV0TrackRoutingVisitedPairCount.get(),
    fImpl->fV0TrackRoutingActiveBitCount.get(),
    fImpl->fV0TrackRoutedAcceptedTaskCount.get(),
    fImpl->fV0TrackRoutedStoredTaskCount.get(),
    fImpl->fV0TrackRoutingBlockReservationCount.get(),
    fImpl->fV0TrackRoutedTaskOverflowFlags.get());
}

KFParticleGpuDecayGraphView KFParticleGpuBufferManager::HostDecayGraph()
{
  return KFParticleGpuDecayGraphView(
    HostPointer(fImpl->fDecayGraphNodes),
    fImpl->fDecayGraphNodeSize,
    HostPointer(fImpl->fDecayGraphGroups),
    fImpl->fDecayGraphGroupSize,
    HostPointer(fImpl->fDecayGraphFamilyCoverage),
    fImpl->fDecayGraphFamilySize,
    fImpl->fDecayGraphRevision);
}

KFParticleGpuDecayGraphView KFParticleGpuBufferManager::DeviceDecayGraph()
{
  return KFParticleGpuDecayGraphView(
    fImpl->fDecayGraphNodes.get(),
    fImpl->fDecayGraphNodeSize,
    fImpl->fDecayGraphGroups.get(),
    fImpl->fDecayGraphGroupSize,
    fImpl->fDecayGraphFamilyCoverage.get(),
    fImpl->fDecayGraphFamilySize,
    fImpl->fDecayGraphRevision);
}

bool KFParticleGpuBufferManager::UploadTwoDaughterRoutingPlan(
  const KFParticleGpuTwoDaughterRoutingPlan& plan,
  bool force)
{
  if (plan.SourceRevision() == 0u) {
    throw std::invalid_argument(
      "KFParticle GPU two-daughter routing plan must be compiled before upload");
  }
  if (!force && fImpl->fTwoDaughterRoutingRevision == plan.SourceRevision()) {
    return false;
  }

  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.twoDaughterRoutingDescriptors =
    static_cast<unsigned int>(plan.Descriptors().size());
  requested.twoDaughterRoutingCompatibility =
    static_cast<unsigned int>(plan.CompatibilityEntries().size());
  requested.twoDaughterRoutingGroups = static_cast<unsigned int>(plan.Groups().size());
  EnsureCapacity(requested);
  fImpl->EnsureTwoDaughterRoutingControl();

  fImpl->fTwoDaughterRoutingDescriptorSize =
    static_cast<unsigned int>(plan.Descriptors().size());
  fImpl->fTwoDaughterRoutingCompatibilitySize =
    static_cast<unsigned int>(plan.CompatibilityEntries().size());
  fImpl->fTwoDaughterRoutingGroupSize =
    static_cast<unsigned int>(plan.Groups().size());

  for (unsigned int index = 0u;
       index < fImpl->fTwoDaughterRoutingDescriptorSize;
       ++index) {
    HostPointer(fImpl->fTwoDaughterRoutingDescriptors)[index] =
      plan.Descriptors()[index];
    HostPointer(fImpl->fTwoDaughterRoutingChannelVisitedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelStoredCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelConstructedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionStoredCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionOffsets)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionCursors)[index] = 0u;
  }
  for (unsigned int index = 0u;
       index < fImpl->fTwoDaughterRoutingCompatibilitySize;
       ++index) {
    HostPointer(fImpl->fTwoDaughterRoutingCompatibility)[index] =
      plan.CompatibilityEntries()[index];
  }
  for (unsigned int index = 0u;
       index < fImpl->fTwoDaughterRoutingGroupSize;
       ++index) {
    HostPointer(fImpl->fTwoDaughterRoutingGroups)[index] = plan.Groups()[index];
  }
  HostPointer(fImpl->fTwoDaughterRoutingEnabledChannels)[0] =
    plan.EnabledChannels();

  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterRoutingDescriptors, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterRoutingCompatibility, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterRoutingGroups, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterRoutingEnabledChannels, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelVisitedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelAcceptedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelStoredCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelConstructedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterSelectionAcceptedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterSelectionStoredCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterSelectionOffsets, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterSelectionCursors, xpu::h2d);
  fImpl->fQueue.wait();
  fImpl->fTwoDaughterRoutingRevision = plan.SourceRevision();
  return true;
}

unsigned long long KFParticleGpuBufferManager::TwoDaughterRoutingRevision() const
{
  return fImpl->fTwoDaughterRoutingRevision;
}

void KFParticleGpuBufferManager::SetTwoDaughterRoutingEnabledChannels(
  const KFParticleGpuChannelMask& enabled)
{
  fImpl->EnsureTwoDaughterRoutingControl();
  HostPointer(fImpl->fTwoDaughterRoutingEnabledChannels)[0] = enabled;
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingEnabledChannels, xpu::h2d);
  fImpl->fQueue.wait();
}

bool KFParticleGpuBufferManager::UploadV0TrackRoutingPlan(
  const KFParticleGpuV0TrackRoutingPlan& plan,
  bool force)
{
  if (plan.SourceRevision() == 0u) {
    throw std::invalid_argument(
      "KFParticle GPU V0-track routing plan must be compiled before upload");
  }
  if (!force && fImpl->fV0TrackRoutingRevision == plan.SourceRevision()) {
    return false;
  }

  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.v0TrackRoutingDescriptors =
    static_cast<unsigned int>(plan.Descriptors().size());
  requested.v0TrackRoutingCompatibility =
    static_cast<unsigned int>(plan.CompatibilityEntries().size());
  requested.v0TrackRoutingGroups = static_cast<unsigned int>(plan.Groups().size());
  EnsureCapacity(requested);
  fImpl->EnsureV0TrackRoutingControl();

  fImpl->fV0TrackRoutingDescriptorSize =
    static_cast<unsigned int>(plan.Descriptors().size());
  fImpl->fV0TrackRoutingCompatibilitySize =
    static_cast<unsigned int>(plan.CompatibilityEntries().size());
  fImpl->fV0TrackRoutingGroupSize = static_cast<unsigned int>(plan.Groups().size());

  for (unsigned int index = 0u; index < fImpl->fV0TrackRoutingDescriptorSize; ++index) {
    HostPointer(fImpl->fV0TrackRoutingDescriptors)[index] = plan.Descriptors()[index];
    HostPointer(fImpl->fV0TrackRoutingChannelVisitedCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelStoredCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelConstructedCounters)[index] = 0u;
  }
  for (unsigned int index = 0u; index < fImpl->fV0TrackRoutingCompatibilitySize; ++index) {
    HostPointer(fImpl->fV0TrackRoutingCompatibility)[index] =
      plan.CompatibilityEntries()[index];
  }
  for (unsigned int index = 0u; index < fImpl->fV0TrackRoutingGroupSize; ++index) {
    HostPointer(fImpl->fV0TrackRoutingGroups)[index] = plan.Groups()[index];
  }
  HostPointer(fImpl->fV0TrackRoutingEnabledChannels)[0] = plan.EnabledChannels();

  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingDescriptors, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingCompatibility, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingGroups, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingEnabledChannels, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelVisitedCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelAcceptedCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelStoredCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelConstructedCounters, xpu::h2d);
  fImpl->fQueue.wait();
  fImpl->fV0TrackRoutingRevision = plan.SourceRevision();
  return true;
}

unsigned long long KFParticleGpuBufferManager::V0TrackRoutingRevision() const
{
  return fImpl->fV0TrackRoutingRevision;
}

bool KFParticleGpuBufferManager::UploadDecayGraphPlan(
  const KFParticleGpuDecayGraphPlan& plan,
  bool force)
{
  if (plan.SourceRevision() == 0u) {
    throw std::invalid_argument(
      "KFParticle GPU decay graph must be compiled before upload");
  }
  if (!force && fImpl->fDecayGraphRevision == plan.SourceRevision()) {
    return false;
  }

  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.decayGraphNodes = static_cast<unsigned int>(plan.Nodes().size());
  requested.decayGraphGroups = static_cast<unsigned int>(plan.Groups().size());
  requested.decayGraphFamilies =
    static_cast<unsigned int>(plan.FamilyCoverage().size());
  EnsureCapacity(requested);

  fImpl->fDecayGraphNodeSize = static_cast<unsigned int>(plan.Nodes().size());
  fImpl->fDecayGraphGroupSize = static_cast<unsigned int>(plan.Groups().size());
  fImpl->fDecayGraphFamilySize =
    static_cast<unsigned int>(plan.FamilyCoverage().size());

  for (unsigned int index = 0u; index < fImpl->fDecayGraphNodeSize; ++index) {
    HostPointer(fImpl->fDecayGraphNodes)[index] = plan.Nodes()[index];
  }
  for (unsigned int index = 0u; index < fImpl->fDecayGraphGroupSize; ++index) {
    HostPointer(fImpl->fDecayGraphGroups)[index] = plan.Groups()[index];
  }
  for (unsigned int index = 0u; index < fImpl->fDecayGraphFamilySize; ++index) {
    HostPointer(fImpl->fDecayGraphFamilyCoverage)[index] =
      plan.FamilyCoverage()[index];
  }

  CopyIfAllocated(fImpl->fQueue, fImpl->fDecayGraphNodes, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDecayGraphGroups, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDecayGraphFamilyCoverage, xpu::h2d);
  fImpl->fQueue.wait();
  fImpl->fDecayGraphRevision = plan.SourceRevision();
  return true;
}

unsigned long long KFParticleGpuBufferManager::DecayGraphRevision() const
{
  return fImpl->fDecayGraphRevision;
}

bool KFParticleGpuBufferManager::UploadGraphOperationPlan(
  const KFParticleGpuDecayPlan& plan,
  bool force)
{
  if (!force && fImpl->fGraphOperationRevision == plan.Revision()) {
    return false;
  }
  KFParticleGpuBufferCapacities requested = fImpl->fCapacities;
  requested.graphOperationDescriptors =
    static_cast<unsigned int>(plan.NumberOfGraphOperationChannels());
  EnsureCapacity(requested);
  fImpl->fGraphOperationDescriptorSize =
    static_cast<unsigned int>(plan.NumberOfGraphOperationChannels());
  for (unsigned int index = 0u;
       index < fImpl->fGraphOperationDescriptorSize;
       ++index) {
    HostPointer(fImpl->fGraphOperationDescriptors)[index] =
      plan.GraphOperationChannel(index).descriptor;
    HostPointer(fImpl->fGraphOperationChannelVisitedCounters)[index] = 0u;
    HostPointer(fImpl->fGraphOperationChannelAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fGraphOperationChannelStoredCounters)[index] = 0u;
    HostPointer(fImpl->fGraphOperationChannelConstructedCounters)[index] = 0u;
    HostPointer(fImpl->fGraphOperationChannelRejectedCounters)[index] = 0u;
  }
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationDescriptors, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationChannelVisitedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationChannelAcceptedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationChannelStoredCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationChannelConstructedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationChannelRejectedCounters, xpu::h2d);
  fImpl->fQueue.wait();
  fImpl->fGraphOperationRevision = plan.Revision();
  return true;
}

unsigned long long KFParticleGpuBufferManager::GraphOperationRevision() const
{
  return fImpl->fGraphOperationRevision;
}

unsigned int KFParticleGpuBufferManager::GraphOperationDescriptorSize() const
{
  return fImpl->fGraphOperationDescriptorSize;
}

KFParticleGpuGraphOperationDescriptor*
KFParticleGpuBufferManager::HostGraphOperationDescriptors()
{
  return HostPointer(fImpl->fGraphOperationDescriptors);
}

KFParticleGpuGraphOperationDescriptor*
KFParticleGpuBufferManager::DeviceGraphOperationDescriptors()
{
  return fImpl->fGraphOperationDescriptors.get();
}

KFParticleGpuGraphOperationTask*
KFParticleGpuBufferManager::DeviceGraphOperationTasks()
{
  return fImpl->fGraphOperationTasks.get();
}

KFParticleGpuGraphOperationResult*
KFParticleGpuBufferManager::HostGraphOperationResults()
{
  return HostPointer(fImpl->fGraphOperationResults);
}

KFParticleGpuGraphOperationResult*
KFParticleGpuBufferManager::DeviceGraphOperationResults()
{
  return fImpl->fGraphOperationResults.get();
}

KFParticleGpuGraphOperationStorageView
KFParticleGpuBufferManager::DeviceGraphOperationStorage()
{
  return KFParticleGpuGraphOperationStorageView(
    fImpl->fGraphOperationDescriptors.get(),
    fImpl->fGraphOperationDescriptorSize,
    fImpl->fCapacities.graphOperationDescriptors,
    fImpl->fGraphOperationTasks.get(),
    fImpl->fCapacities.graphOperationTasks,
    fImpl->fGraphOperationResults.get(),
    fImpl->fGraphOperationVisitedCombinations.get(),
    fImpl->fGraphOperationAcceptedTasks.get(),
    fImpl->fGraphOperationStoredTasks.get(),
    fImpl->fGraphOperationConstructedCandidates.get(),
    fImpl->fGraphOperationRejectedTasks.get(),
    fImpl->fGraphOperationOverflowFlags.get(),
    fImpl->fGraphOperationChannelVisitedCounters.get(),
    fImpl->fGraphOperationChannelAcceptedCounters.get(),
    fImpl->fGraphOperationChannelStoredCounters.get(),
    fImpl->fGraphOperationChannelConstructedCounters.get(),
    fImpl->fGraphOperationChannelRejectedCounters.get(),
    fImpl->fGraphOperationRevision);
}

void KFParticleGpuBufferManager::SetV0TrackRoutingEnabledChannels(
  const KFParticleGpuChannelMask& enabled)
{
  fImpl->EnsureV0TrackRoutingControl();
  HostPointer(fImpl->fV0TrackRoutingEnabledChannels)[0] = enabled;
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingEnabledChannels, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::UploadInput()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fTrackParameters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTrackCovariances, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTrackField, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTrackChiToPrimaryVertex, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTrackIntegers, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fVertexParameters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fVertexCovariances, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fVertexChi2, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fVertexIntegers, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fEvents, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::UploadCandidates()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateParameters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateCovariances, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateFitScalars, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateFitIntegers, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateMetadataIntegers, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateMetadataUnsigned, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateRoutingDescriptorIndices, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0SelectionResults, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSourceIds, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateSize, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSize, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fOverflowFlags, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::UploadSelectedCandidates()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateIndices, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateChannelIds, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateSize, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateOverflowFlags, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::ResetCandidates()
{
  if (!fImpl->fCandidateSize.get()) {
    throw std::logic_error("KFParticle GPU buffers must be initialized before candidate reset");
  }
  unsigned int* candidateSize = HostPointer(fImpl->fCandidateSize);
  unsigned int* daughterSize = HostPointer(fImpl->fDaughterSize);
  unsigned int* overflowFlags = HostPointer(fImpl->fOverflowFlags);
  candidateSize[0] = 0;
  daughterSize[0] = 0;
  overflowFlags[0] = 0;
  fImpl->fQueue.copy(fImpl->fCandidateSize, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fDaughterSize, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fOverflowFlags, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::ResetTwoDaughterTaskStatus()
{
  if (!fImpl->fTwoDaughterTaskCount.get()) {
    throw std::logic_error("KFParticle GPU task buffers must be initialized before reset");
  }
  HostPointer(fImpl->fTwoDaughterTaskCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterTotalPairCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterTaskOverflowFlags)[0] = 0u;
  fImpl->fQueue.copy(fImpl->fTwoDaughterTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterTotalPairCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterTaskOverflowFlags, xpu::h2d);
}

KFParticleGpuTwoDaughterTaskStatus KFParticleGpuBufferManager::DownloadTwoDaughterTaskStatus()
{
  KFParticleGpuTwoDaughterTaskStatus status;
  if (!fImpl->fTwoDaughterTaskCount.get()) {
    return status;
  }
  fImpl->fQueue.copy(fImpl->fTwoDaughterTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterTotalPairCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterTaskOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
  status.accepted = HostPointer(fImpl->fTwoDaughterTaskCount)[0];
  status.totalPairs = HostPointer(fImpl->fTwoDaughterTotalPairCount)[0];
  status.overflowFlags = HostPointer(fImpl->fTwoDaughterTaskOverflowFlags)[0];
  return status;
}

void KFParticleGpuBufferManager::ResetTwoDaughterRoutingStatus()
{
  if (!fImpl->fTwoDaughterRoutedAcceptedTaskCount.get()) {
    throw std::logic_error(
      "KFParticle GPU routed two-daughter task buffers must be initialized before reset");
  }
  HostPointer(fImpl->fTwoDaughterRoutingVisitedPairCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterRoutingActiveBitCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterRoutedAcceptedTaskCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterRoutedStoredTaskCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterRoutingBlockReservationCount)[0] = 0u;
  HostPointer(fImpl->fTwoDaughterRoutedTaskOverflowFlags)[0] = 0u;
  for (unsigned int index = 0u;
       index < fImpl->fTwoDaughterRoutingDescriptorSize;
       ++index) {
    HostPointer(fImpl->fTwoDaughterRoutingChannelVisitedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelStoredCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterRoutingChannelConstructedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionStoredCounters)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionOffsets)[index] = 0u;
    HostPointer(fImpl->fTwoDaughterSelectionCursors)[index] = 0u;
  }
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingVisitedPairCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingActiveBitCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedAcceptedTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedStoredTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingBlockReservationCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedTaskOverflowFlags, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelVisitedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelAcceptedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelStoredCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelConstructedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterSelectionAcceptedCounters, xpu::h2d);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterSelectionStoredCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterSelectionOffsets, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterSelectionCursors, xpu::h2d);
}

KFParticleGpuTwoDaughterRoutingStatus
KFParticleGpuBufferManager::DownloadTwoDaughterRoutingStatus()
{
  KFParticleGpuTwoDaughterRoutingStatus status;
  if (!fImpl->fTwoDaughterRoutedAcceptedTaskCount.get()) {
    return status;
  }
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingVisitedPairCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingActiveBitCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedAcceptedTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedStoredTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutingBlockReservationCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fTwoDaughterRoutedTaskOverflowFlags, xpu::d2h);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelVisitedCounters, xpu::d2h);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelAcceptedCounters, xpu::d2h);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelStoredCounters, xpu::d2h);
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fTwoDaughterRoutingChannelConstructedCounters, xpu::d2h);
  fImpl->fQueue.wait();
  status.visitedPairs = HostPointer(fImpl->fTwoDaughterRoutingVisitedPairCount)[0];
  status.activeChannelBits = HostPointer(fImpl->fTwoDaughterRoutingActiveBitCount)[0];
  status.acceptedTasks = HostPointer(fImpl->fTwoDaughterRoutedAcceptedTaskCount)[0];
  status.storedTasks = HostPointer(fImpl->fTwoDaughterRoutedStoredTaskCount)[0];
  status.blockReservations =
    HostPointer(fImpl->fTwoDaughterRoutingBlockReservationCount)[0];
  status.overflowFlags = HostPointer(fImpl->fTwoDaughterRoutedTaskOverflowFlags)[0];
  return status;
}

void KFParticleGpuBufferManager::DownloadTwoDaughterRoutedTasks()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fTwoDaughterRoutedTasks, xpu::d2h);
  fImpl->fQueue.wait();
}

KFParticleGpuTwoDaughterRoutedTask*
KFParticleGpuBufferManager::HostTwoDaughterRoutedTasks()
{
  return HostPointer(fImpl->fTwoDaughterRoutedTasks);
}

void KFParticleGpuBufferManager::ResetV0TrackTaskStatus()
{
  if (!fImpl->fV0TrackTaskCount.get()) {
    throw std::logic_error("KFParticle GPU V0-track task buffers must be initialized before reset");
  }
  HostPointer(fImpl->fV0TrackTaskCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackTotalPairCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackTaskOverflowFlags)[0] = 0u;
  fImpl->fQueue.copy(fImpl->fV0TrackTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackTotalPairCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackTaskOverflowFlags, xpu::h2d);
}

KFParticleGpuV0TrackTaskStatus KFParticleGpuBufferManager::DownloadV0TrackTaskStatus()
{
  KFParticleGpuV0TrackTaskStatus status;
  if (!fImpl->fV0TrackTaskCount.get()) {
    return status;
  }
  fImpl->fQueue.copy(fImpl->fV0TrackTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackTotalPairCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackTaskOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
  status.accepted = HostPointer(fImpl->fV0TrackTaskCount)[0];
  status.totalPairs = HostPointer(fImpl->fV0TrackTotalPairCount)[0];
  status.overflowFlags = HostPointer(fImpl->fV0TrackTaskOverflowFlags)[0];
  return status;
}

void KFParticleGpuBufferManager::ResetV0TrackRoutingStatus()
{
  if (!fImpl->fV0TrackRoutedAcceptedTaskCount.get()) {
    throw std::logic_error(
      "KFParticle GPU routed V0-track task buffers must be initialized before reset");
  }
  HostPointer(fImpl->fV0TrackRoutingVisitedPairCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackRoutingActiveBitCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackRoutedAcceptedTaskCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackRoutedStoredTaskCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackRoutingBlockReservationCount)[0] = 0u;
  HostPointer(fImpl->fV0TrackRoutedTaskOverflowFlags)[0] = 0u;
  for (unsigned int index = 0u; index < fImpl->fV0TrackRoutingDescriptorSize; ++index) {
    HostPointer(fImpl->fV0TrackRoutingChannelVisitedCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelAcceptedCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelStoredCounters)[index] = 0u;
    HostPointer(fImpl->fV0TrackRoutingChannelConstructedCounters)[index] = 0u;
  }
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingVisitedPairCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingActiveBitCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedAcceptedTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedStoredTaskCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingBlockReservationCount, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedTaskOverflowFlags, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelVisitedCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelAcceptedCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelStoredCounters, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelConstructedCounters, xpu::h2d);
}

KFParticleGpuV0TrackRoutingStatus KFParticleGpuBufferManager::DownloadV0TrackRoutingStatus()
{
  KFParticleGpuV0TrackRoutingStatus status;
  if (!fImpl->fV0TrackRoutedAcceptedTaskCount.get()) {
    return status;
  }
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingVisitedPairCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingActiveBitCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedAcceptedTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedStoredTaskCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutingBlockReservationCount, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fV0TrackRoutedTaskOverflowFlags, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelVisitedCounters, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelAcceptedCounters, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelStoredCounters, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutingChannelConstructedCounters, xpu::d2h);
  fImpl->fQueue.wait();
  status.visitedPairs = HostPointer(fImpl->fV0TrackRoutingVisitedPairCount)[0];
  status.activeChannelBits = HostPointer(fImpl->fV0TrackRoutingActiveBitCount)[0];
  status.acceptedTasks = HostPointer(fImpl->fV0TrackRoutedAcceptedTaskCount)[0];
  status.storedTasks = HostPointer(fImpl->fV0TrackRoutedStoredTaskCount)[0];
  status.blockReservations = HostPointer(fImpl->fV0TrackRoutingBlockReservationCount)[0];
  status.overflowFlags = HostPointer(fImpl->fV0TrackRoutedTaskOverflowFlags)[0];
  return status;
}

void KFParticleGpuBufferManager::DownloadV0TrackRoutedTasks()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0TrackRoutedTasks, xpu::d2h);
  fImpl->fQueue.wait();
}

KFParticleGpuV0TrackRoutedTask* KFParticleGpuBufferManager::HostV0TrackRoutedTasks()
{
  return HostPointer(fImpl->fV0TrackRoutedTasks);
}

void KFParticleGpuBufferManager::ResetSelectedCandidates()
{
  if (!fImpl->fSelectedCandidateSize.get()) {
    throw std::logic_error("KFParticle GPU selected-candidate buffers must be initialized before reset");
  }
  unsigned int* size = HostPointer(fImpl->fSelectedCandidateSize);
  unsigned int* overflowFlags = HostPointer(fImpl->fSelectedCandidateOverflowFlags);
  size[0] = 0u;
  overflowFlags[0] = 0u;
  fImpl->fQueue.copy(fImpl->fSelectedCandidateSize, xpu::h2d);
  fImpl->fQueue.copy(fImpl->fSelectedCandidateOverflowFlags, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::ResetV0SelectionResults()
{
  if (!fImpl->fV0SelectionResults.get()) {
    throw std::logic_error("KFParticle GPU selection-result buffers must be initialized before reset");
  }
  KFParticleGpuV0SelectionResult* results = HostPointer(fImpl->fV0SelectionResults);
  for (unsigned int i = 0; i < fImpl->fCapacities.candidates; ++i) {
    results[i] = KFParticleGpuV0SelectionResult();
  }
  fImpl->fQueue.copy(fImpl->fV0SelectionResults, xpu::h2d);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::MarkCandidateOverflow(unsigned int flags)
{
  if (!fImpl->fOverflowFlags.get()) {
    throw std::logic_error("KFParticle GPU buffers must be initialized before marking overflow");
  }
  unsigned int* overflowFlags = HostPointer(fImpl->fOverflowFlags);
  overflowFlags[0] |= flags;
  fImpl->fQueue.copy(fImpl->fOverflowFlags, xpu::h2d);
  fImpl->fQueue.wait();
}

KFParticleGpuCandidatePoolStatus KFParticleGpuBufferManager::DownloadCandidateStatus()
{
  KFParticleGpuCandidatePoolStatus status;
  if (!fImpl->fCandidateSize.get()) {
    return status;
  }
  fImpl->fQueue.copy(fImpl->fCandidateSize, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fDaughterSize, xpu::d2h);
  fImpl->fQueue.copy(fImpl->fOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
  status.candidates = HostPointer(fImpl->fCandidateSize)[0];
  status.daughters = HostPointer(fImpl->fDaughterSize)[0];
  status.overflowFlags = HostPointer(fImpl->fOverflowFlags)[0];
  return status;
}

void KFParticleGpuBufferManager::DownloadCandidates()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateParameters, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateCovariances, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateFitScalars, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateFitIntegers, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateMetadataIntegers, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateMetadataUnsigned, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0SelectionResults, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSourceIds, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadCandidateDescriptorIndices()
{
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fCandidateRoutingDescriptorIndices, xpu::d2h);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadV0SelectionResults()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0SelectionResults, xpu::d2h);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadSelectedCandidates()
{
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateIndices, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateChannelIds, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
}
