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

#include <xpu/host.h>

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
};

KFParticleGpuBufferManager::KFParticleGpuBufferManager(xpu::queue& queue) : fImpl(new Impl(queue)) {}

KFParticleGpuBufferManager::~KFParticleGpuBufferManager() = default;

void KFParticleGpuBufferManager::EnsureCapacity(const KFParticleGpuBufferCapacities& requested)
{
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
  const unsigned int requestedSelectedCandidates = requested.selectedCandidates > 0u
                                                     ? requested.selectedCandidates
                                                     : requested.candidates;
  if (requestedSelectedCandidates > fImpl->fCapacities.selectedCandidates) {
    fImpl->AllocateSelectedCandidates(requestedSelectedCandidates);
    fImpl->fCapacities.selectedCandidates = requestedSelectedCandidates;
  }
  fImpl->EnsureCounters();
}

const KFParticleGpuBufferCapacities& KFParticleGpuBufferManager::Capacities() const
{
  return fImpl->fCapacities;
}

const KFParticleGpuDeviceStorage& KFParticleGpuBufferManager::Storage() const
{
  return *fImpl;
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
  CopyIfAllocated(fImpl->fQueue, fImpl->fV0SelectionResults, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSourceIds, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateSize, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSize, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fOverflowFlags, xpu::h2d);
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
