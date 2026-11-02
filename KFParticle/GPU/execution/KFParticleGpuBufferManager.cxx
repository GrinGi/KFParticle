/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @file
 * @brief Buffer allocation, optional readback mirrors, and XPU queue transfers.
 */

#include "execution/KFParticleGpuBufferManager.h"

#include "planning/KFParticleGpuDecayPlan.h"
#include "execution/KFParticleGpuKernels.h"
#include "diagnostics/KFParticleGpuStepTrace.h"

#include <xpu/host.h>

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace
{
  /**
   * @brief Returns the host pointer of an IO buffer, or nullptr if unallocated.
   * @note Does not allocate, copy data, or synchronize the queue.
   */
  template<typename T>
  T* HostPointer(xpu::buffer<T>& buffer)
  {
    if (!buffer.get()) {
      return nullptr;
    }
    return xpu::buffer_prop(buffer).template h_ptr<T>();
  }

  /** @brief Queues a full IO-buffer copy in @p direction if allocated; does not wait. */
  template<typename T>
  void CopyIfAllocated(xpu::queue& queue, xpu::buffer<T>& buffer, xpu::direction direction)
  {
    if (buffer.get()) {
      queue.copy(buffer, direction);
    }
  }

  /**
   * @brief Queues D2H copying of an IO-buffer prefix; skips empty or unallocated buffers.
   * @pre The allocated buffer has at least @p size elements. Does not wait.
   */
  template<typename T>
  void CopyPrefix(xpu::queue& queue, xpu::buffer<T>& buffer, unsigned int size)
  {
    if (size == 0u || !buffer.get()) { return; }
    queue.copy(buffer.get(), HostPointer(buffer), size);
  }

  /**
   * @brief Queues D2H prefixes for all components of a structure-of-arrays buffer.
   * @param queue Queue receiving the copies; this function does not wait.
   * @param buffer IO buffer containing consecutive component arrays.
   * @param components Number of component arrays.
   * @param stride Allocated elements per component, on both host and device.
   * @param size Elements to copy per component; must not exceed @p stride.
   * @pre An allocated buffer contains at least components * stride elements.
   */
  template<typename T>
  void CopySoAPrefix(xpu::queue& queue,
                     xpu::buffer<T>& buffer,
                     unsigned int components,
                     unsigned int stride,
                     unsigned int size)
  {
    if (size == 0u || !buffer.get()) { return; }
    T* host = HostPointer(buffer);
    T* device = buffer.get();
    for (unsigned int component = 0u; component < components; ++component) {
      queue.copy(device + component * stride,
                 host + component * stride,
                 size);
    }
  }
}

/**
 * @brief Private owner of XPU storage and host bookkeeping for the buffer manager.
 *
 * The base holds transfer buffers and device workspaces. This implementation
 * adds optional pinned readback mirrors, active sizes, cached plan revisions,
 * allocation statistics, and a non-owning queue reference. Mirrors are allocated
 * only by explicit host-view or readback requests; obtaining a mirror does not
 * populate it. No internal locking or queue-lifetime management is provided.
 */
struct KFParticleGpuBufferManager::Impl : KFParticleGpuDeviceStorage
{
  xpu::queue& fQueue; ///< Borrowed execution queue; must outlive all managed buffers and queued operations.
  // Optional host staging is allocated only by an explicit host-view/download request.
  xpu::buffer<KFParticleGpuTwoDaughterRoutedTask> fHostTwoDaughterRoutedTasks;
  xpu::buffer<KFParticleGpuV0TrackRoutedTask> fHostV0TrackRoutedTasks;
  xpu::buffer<unsigned int> fHostCandidateRoutingDescriptorIndices;
  xpu::buffer<KFParticleGpuGraphOperationResult> fHostGraphOperationResults;
  xpu::buffer<KFParticleGpuV0SelectionResult> fHostV0SelectionResults;
  std::uint64_t fSelectionResetLaunches = 0u;
  KFParticleGpuBufferCapacities fCapacities;
  unsigned int fTrackSize = 0;
  unsigned int fVertexSize = 0;
  unsigned int fEventSize = 0;
  unsigned int fTransactionSize = 0;
  unsigned int fDomainSize = 0;
  unsigned int fWorkRangeSize = 0;
  unsigned int fTwoDaughterPairTileSize = 0;
  unsigned int fDomainStatusSize = 0;
  unsigned int fDomainChannelStatusSize = 0;
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

  KFParticleGpuHostTiming fHostTiming;
  bool fAllocationTraceEnabled = false;
  const char* fAllocationPhase = "input-routing";
  std::vector<KFParticleGpuAllocationRecord> fAllocationTrace;
  unsigned int fAllocationTraceDropped = 0u;

  /// @brief Binds the borrowed queue without allocating payload buffers.
  explicit Impl(xpu::queue& queue) : fQueue(queue)
  {
  }

  /**
   * @brief Allocates or replaces one buffer and records nonzero allocations in host timing.
   * @param name Static buffer label used by optional step/allocation tracing.
   * @param buffer Buffer whose old contents may be discarded; never copied to the new allocation.
   * @param size Requested elements; zero resets the buffer without recording a timed allocation.
   * @param type XPU memory kind: IO, device-only, or pinned host.
   * @pre No outstanding operation uses the old allocation. This function does not wait.
   */
  template<typename T>
  void Allocate(const char* name, xpu::buffer<T>& buffer, unsigned int size,
                xpu::buffer_type type)
  {
    if (size == 0u) {
      buffer.reset(size, type);
      return;
    }
    const bool replacement = buffer.get() != nullptr;
    char timerName[256] = {};
    if (KFParticleGpuStepTrace::Enabled()) {
      std::snprintf(timerName, sizeof(timerName), "ALLOC %s phase=%s type=%s action=%s bytes=%llu",
        name, fAllocationPhase,
        type == xpu::buf_device ? "device" : (type == xpu::buf_pinned ? "pinned" : "io"),
        replacement ? "replace" : "initial",
        static_cast<unsigned long long>(size) * sizeof(T));
    }
    KFParticleGpuStepTrace allocationTimer(timerName);
    const auto started = std::chrono::steady_clock::now();
    buffer.reset(size, type);
    const double milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    if (replacement) {
      ++fHostTiming.replacementAllocations;
      fHostTiming.replacementAllocationMilliseconds += milliseconds;
    }
    else {
      ++fHostTiming.initialAllocations;
      fHostTiming.initialAllocationMilliseconds += milliseconds;
    }
    if (fAllocationTraceEnabled) {
      if (fAllocationTrace.size() < 256u) {
        fAllocationTrace.push_back({name, fAllocationPhase,
          static_cast<std::uint64_t>(size) * sizeof(T), size, replacement, milliseconds,
          type == xpu::buf_device ? "device" : (type == xpu::buf_pinned ? "pinned" : "io")});
      }
      else { ++fAllocationTraceDropped; }
    }
  }

  /// @brief Allocates host/device transfer storage using Allocate() and its replacement rules.
  template<typename T>
  void AllocateIo(const char* name, xpu::buffer<T>& buffer, unsigned int size)
  {
    Allocate(name, buffer, size, xpu::buf_io);
  }

  /// @brief Allocates device-only storage using Allocate() and its replacement rules.
  template<typename T>
  void AllocateDevice(const char* name, xpu::buffer<T>& buffer, unsigned int size)
  {
    Allocate(name, buffer, size, xpu::buf_device);
  }

  /**
   * @brief Ensures a pinned host mirror large enough for the entire device buffer.
   * @return Host pointer, or nullptr if the device buffer is unallocated.
   * @note No readback or wait is performed; a replacement mirror discards its old contents.
   */
  template<typename T>
  T* HostMirror(const char* name, xpu::buffer<T>& device, xpu::buffer<T>& host)
  {
    if (!device.get()) { return nullptr; }
    const auto count = xpu::buffer_prop(device).size();
    if (!host.get() || xpu::buffer_prop(host).size() < count) {
      Allocate(name, host, static_cast<unsigned int>(count), xpu::buf_pinned);
    }
    return host.get();
  }

  /**
   * @brief Queues a device prefix readback into an on-demand, full-capacity pinned mirror.
   * @param name Static mirror label used for allocation tracing.
   * @param device Source device buffer.
   * @param host Destination mirror, allocated or replaced by HostMirror() if needed.
   * @param count Number of elements to copy; zero is a no-op.
   * @throws std::out_of_range If a nonempty readback exceeds available device storage.
   * @note The caller must wait before reading the mirror or reusing its storage.
   */
  template<typename T>
  void DownloadMirror(const char* name, xpu::buffer<T>& device, xpu::buffer<T>& host,
                      unsigned int count)
  {
    if (count == 0u) { return; }
    if (!device.get() || count > xpu::buffer_prop(device).size()) {
      throw std::out_of_range("KFParticle GPU diagnostic readback exceeds device capacity");
    }
    fQueue.copy(device.get(), HostMirror(name, device, host), count);
  }

  /// @brief Queues full-capacity readback into a pinned mirror if the device buffer exists; does not wait.
  template<typename T>
  void DownloadMirror(const char* name, xpu::buffer<T>& device, xpu::buffer<T>& host)
  {
    if (device.get()) {
      DownloadMirror(name, device, host, static_cast<unsigned int>(xpu::buffer_prop(device).size()));
    }
  }

  /**
   * @brief Allocates track SoA transfer arrays for @p capacity tracks, discarding replaced data.
   * @param capacity Number of track slots and the stride of each component array.
   * @param withField Allocate field coefficients when true; otherwise release that array.
   */
  void AllocateTracks(unsigned int capacity, bool withField)
  {
    AllocateIo("fTrackParameters", fTrackParameters, KFParticleGpuTrackState::NumberOfParameters * capacity);
    AllocateIo("fTrackCovariances", fTrackCovariances, KFParticleGpuTrackState::NumberOfCovarianceElements * capacity);
    AllocateIo("fTrackChiToPrimaryVertex", fTrackChiToPrimaryVertex, capacity);
    AllocateIo("fTrackIntegers", fTrackIntegers, KFParticleGpuTrackInputLayout::NumberOfIntegerComponents * capacity);
    if (withField) {
      AllocateIo("fTrackField", fTrackField, KFParticleGpuFieldRegion::NumberOfCoefficients * capacity);
    }
    else {
      fTrackField.reset();
    }
  }

  /// @brief Allocates primary-vertex SoA transfer arrays, discarding replaced data.
  void AllocateVertices(unsigned int capacity)
  {
    AllocateIo("fVertexParameters", fVertexParameters, KFParticleGpuVertexState::NumberOfParameters * capacity);
    AllocateIo("fVertexCovariances", fVertexCovariances,
               KFParticleGpuVertexState::NumberOfCovarianceElements * capacity);
    AllocateIo("fVertexChi2", fVertexChi2, capacity);
    AllocateIo("fVertexIntegers", fVertexIntegers, KFParticleGpuVertexSoALayout::NumberOfIntegerComponents * capacity);
  }

  /// @brief Allocates candidate fit/metadata IO arrays and device-only routing indices/selection results.
  void AllocateCandidates(unsigned int capacity)
  {
    AllocateIo("fCandidateParameters", fCandidateParameters, KFParticleGpuFitState::NumberOfParameters * capacity);
    AllocateIo("fCandidateCovariances", fCandidateCovariances,
               KFParticleGpuFitState::NumberOfCovarianceElements * capacity);
    AllocateIo("fCandidateFitScalars", fCandidateFitScalars,
               KFParticleGpuFitSoALayout::NumberOfFloatComponents * capacity);
    AllocateIo("fCandidateFitIntegers", fCandidateFitIntegers,
               KFParticleGpuFitSoALayout::NumberOfIntegerComponents * capacity);
    AllocateIo(
      "fCandidateMetadataIntegers", fCandidateMetadataIntegers,
      KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents * capacity);
    AllocateIo(
      "fCandidateMetadataUnsigned", fCandidateMetadataUnsigned,
      KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents * capacity);
    AllocateDevice("fCandidateRoutingDescriptorIndices", fCandidateRoutingDescriptorIndices, capacity);
    AllocateDevice("fV0SelectionResults", fV0SelectionResults, capacity);
  }

  /// @brief Lazily allocates the three candidate-pool IO counters without initializing their values.
  void EnsureCounters()
  {
    if (!fCandidateSize.get()) {
      AllocateIo("fCandidateSize", fCandidateSize, 1);
      AllocateIo("fDaughterSize", fDaughterSize, 1);
      AllocateIo("fOverflowFlags", fOverflowFlags, 1);
    }
  }

  /// @brief Allocates selected-index/channel IO arrays and lazily allocates their two counters.
  void AllocateSelectedCandidates(unsigned int capacity)
  {
    AllocateIo("fSelectedCandidateIndices", fSelectedCandidateIndices, capacity);
    AllocateIo("fSelectedCandidateChannelIds", fSelectedCandidateChannelIds, capacity);
    if (!fSelectedCandidateSize.get()) {
      AllocateIo("fSelectedCandidateSize", fSelectedCandidateSize, 1);
      AllocateIo("fSelectedCandidateOverflowFlags", fSelectedCandidateOverflowFlags, 1);
    }
  }

  /// @brief Allocates explicit two-daughter tasks on device and lazily allocates their IO counters.
  void AllocateTwoDaughterTasks(unsigned int capacity)
  {
    AllocateDevice("fTwoDaughterTasks", fTwoDaughterTasks, capacity);
    if (!fTwoDaughterTaskCount.get()) {
      AllocateIo("fTwoDaughterTaskCount", fTwoDaughterTaskCount, 1);
      AllocateIo("fTwoDaughterTotalPairCount", fTwoDaughterTotalPairCount, 1);
      AllocateIo("fTwoDaughterTaskOverflowFlags", fTwoDaughterTaskOverflowFlags, 1);
    }
  }

  /// @brief Allocates explicit V0-plus-track tasks on device and lazily allocates their IO counters.
  void AllocateV0TrackTasks(unsigned int capacity)
  {
    AllocateDevice("fV0TrackTasks", fV0TrackTasks, capacity);
    if (!fV0TrackTaskCount.get()) {
      AllocateIo("fV0TrackTaskCount", fV0TrackTaskCount, 1);
      AllocateIo("fV0TrackTotalPairCount", fV0TrackTotalPairCount, 1);
      AllocateIo("fV0TrackTaskOverflowFlags", fV0TrackTaskOverflowFlags, 1);
    }
  }

  /// @brief Allocates routed two-daughter tasks on device and lazily allocates their IO counters.
  void AllocateTwoDaughterRoutedTasks(unsigned int capacity)
  {
    AllocateDevice("fTwoDaughterRoutedTasks", fTwoDaughterRoutedTasks, capacity);
    if (!fTwoDaughterRoutedAcceptedTaskCount.get()) {
      AllocateIo("fTwoDaughterRoutingVisitedPairCount", fTwoDaughterRoutingVisitedPairCount, 1u);
      AllocateIo("fTwoDaughterRoutingActiveBitCount", fTwoDaughterRoutingActiveBitCount, 1u);
      AllocateIo("fTwoDaughterRoutedAcceptedTaskCount", fTwoDaughterRoutedAcceptedTaskCount, 1u);
      AllocateIo("fTwoDaughterRoutedStoredTaskCount", fTwoDaughterRoutedStoredTaskCount, 1u);
      AllocateIo("fTwoDaughterRoutingBlockReservationCount", fTwoDaughterRoutingBlockReservationCount, 1u);
      AllocateIo("fTwoDaughterRoutedTaskOverflowFlags", fTwoDaughterRoutedTaskOverflowFlags, 1u);
    }
  }

  /// @brief Allocates routed V0-plus-track tasks on device and lazily allocates their IO counters.
  void AllocateV0TrackRoutedTasks(unsigned int capacity)
  {
    AllocateDevice("fV0TrackRoutedTasks", fV0TrackRoutedTasks, capacity);
    if (!fV0TrackRoutedAcceptedTaskCount.get()) {
      AllocateIo("fV0TrackRoutingVisitedPairCount", fV0TrackRoutingVisitedPairCount, 1u);
      AllocateIo("fV0TrackRoutingActiveBitCount", fV0TrackRoutingActiveBitCount, 1u);
      AllocateIo("fV0TrackRoutedAcceptedTaskCount", fV0TrackRoutedAcceptedTaskCount, 1u);
      AllocateIo("fV0TrackRoutedStoredTaskCount", fV0TrackRoutedStoredTaskCount, 1u);
      AllocateIo("fV0TrackRoutingBlockReservationCount", fV0TrackRoutingBlockReservationCount, 1u);
      AllocateIo("fV0TrackRoutedTaskOverflowFlags", fV0TrackRoutedTaskOverflowFlags, 1u);
    }
  }

  /// @brief Lazily allocates the V0-plus-track enabled-channel IO mask without initializing it.
  void EnsureV0TrackRoutingControl()
  {
    if (!fV0TrackRoutingEnabledChannels.get()) {
      AllocateIo("fV0TrackRoutingEnabledChannels", fV0TrackRoutingEnabledChannels, 1u);
    }
  }

  /// @brief Lazily allocates the two-daughter enabled-channel IO mask without initializing it.
  void EnsureTwoDaughterRoutingControl()
  {
    if (!fTwoDaughterRoutingEnabledChannels.get()) {
      AllocateIo("fTwoDaughterRoutingEnabledChannels", fTwoDaughterRoutingEnabledChannels, 1u);
    }
  }

  /// @brief Allocates graph-operation tasks/results and lazily allocates counters, all device-only.
  void AllocateGraphOperationTasks(unsigned int capacity)
  {
    AllocateDevice("fGraphOperationTasks", fGraphOperationTasks, capacity);
    AllocateDevice("fGraphOperationResults", fGraphOperationResults, capacity);
    if (!fGraphOperationVisitedCombinations.get()) {
      AllocateDevice("fGraphOperationVisitedCombinations", fGraphOperationVisitedCombinations, 1u);
      AllocateDevice("fGraphOperationAcceptedTasks", fGraphOperationAcceptedTasks, 1u);
      AllocateDevice("fGraphOperationStoredTasks", fGraphOperationStoredTasks, 1u);
      AllocateDevice("fGraphOperationConstructedCandidates", fGraphOperationConstructedCandidates, 1u);
      AllocateDevice("fGraphOperationRejectedTasks", fGraphOperationRejectedTasks, 1u);
      AllocateDevice("fGraphOperationOverflowFlags", fGraphOperationOverflowFlags, 1u);
    }
  }
};

KFParticleGpuBufferManager::KFParticleGpuBufferManager(xpu::queue& queue) : fImpl(new Impl(queue))
{
}

KFParticleGpuBufferManager::~KFParticleGpuBufferManager() = default;

const KFParticleGpuHostTiming& KFParticleGpuBufferManager::HostTiming() const
{
  return fImpl->fHostTiming;
}

void KFParticleGpuBufferManager::BeginAllocationTrace(bool enabled)
{
  fImpl->fAllocationTraceEnabled = enabled;
  fImpl->fAllocationTrace.clear();
  fImpl->fAllocationTraceDropped = 0u;
  fImpl->fAllocationPhase = "input-routing";
  if (enabled) { fImpl->fAllocationTrace.reserve(256u); }
}

void KFParticleGpuBufferManager::SetAllocationTracePhase(const char* phase)
{
  fImpl->fAllocationPhase = phase;
}

const std::vector<KFParticleGpuAllocationRecord>& KFParticleGpuBufferManager::AllocationTrace() const
{
  return fImpl->fAllocationTrace;
}

void KFParticleGpuBufferManager::PrintAllocationTrace() const
{
  if (!fImpl->fAllocationTraceEnabled) { return; }
  // stdout block buffering can splice logger output into an allocation row.
  // Keep each short record on stderr, outside the transaction timing interval.
  for (const auto& record : fImpl->fAllocationTrace) {
    std::fprintf(stderr,
      "KFP_ALLOC contract=1 phase=%s buffer=%s kind=%s elements=%u bytes=%llu ms=%.6f memory=%s\n",
      record.phase, record.buffer, record.replacement ? "replacement" : "initial",
      record.elements, static_cast<unsigned long long>(record.bytes), record.milliseconds, record.memory);
  }
  std::fprintf(stderr, "KFP_ALLOC_SUMMARY contract=1 records=%zu dropped=%u\n",
    fImpl->fAllocationTrace.size(), fImpl->fAllocationTraceDropped);
  std::fflush(stderr);
}

void KFParticleGpuBufferManager::EnsureCapacity(const KFParticleGpuBufferCapacities& requested)
{
  KFParticleGpuStepTrace timer("CPU EnsureCapacity (nested ALLOC rows)");
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
    fImpl->AllocateIo("fEvents", fImpl->fEvents, requested.events);
    fImpl->fCapacities.events = requested.events;
  }
  if (requested.transactions > fImpl->fCapacities.transactions) {
    fImpl->AllocateIo("fTransactions", fImpl->fTransactions, requested.transactions);
    fImpl->fCapacities.transactions = requested.transactions;
  }
  if (requested.domains > fImpl->fCapacities.domains) {
    fImpl->AllocateIo("fDomains", fImpl->fDomains, requested.domains);
    fImpl->fCapacities.domains = requested.domains;
  }
  if (requested.workRanges > fImpl->fCapacities.workRanges) {
    fImpl->AllocateIo("fWorkRanges", fImpl->fWorkRanges, requested.workRanges);
    fImpl->AllocateIo("fWorkStatuses", fImpl->fWorkStatuses, requested.workRanges);
    fImpl->fCapacities.workRanges = requested.workRanges;
  }
  if (requested.twoDaughterPairTiles > fImpl->fCapacities.twoDaughterPairTiles) {
    fImpl->AllocateIo("fTwoDaughterPairTiles", fImpl->fTwoDaughterPairTiles, requested.twoDaughterPairTiles);
    fImpl->fCapacities.twoDaughterPairTiles = requested.twoDaughterPairTiles;
  }
  if (requested.twoDaughterTaskTiles > fImpl->fCapacities.twoDaughterTaskTiles) {
    fImpl->AllocateIo("fTwoDaughterTaskTiles", fImpl->fTwoDaughterTaskTiles, requested.twoDaughterTaskTiles);
    fImpl->fCapacities.twoDaughterTaskTiles = requested.twoDaughterTaskTiles;
  }
  if (requested.domainStatuses > fImpl->fCapacities.domainStatuses) {
    fImpl->AllocateIo("fDomainStatuses", fImpl->fDomainStatuses, requested.domainStatuses);
    fImpl->fCapacities.domainStatuses = requested.domainStatuses;
  }
  if (requested.domainChannelStatuses > fImpl->fCapacities.domainChannelStatuses) {
    fImpl->AllocateIo("fDomainChannelStatuses", fImpl->fDomainChannelStatuses, requested.domainChannelStatuses);
    fImpl->fCapacities.domainChannelStatuses = requested.domainChannelStatuses;
  }
  if (requested.candidates > fImpl->fCapacities.candidates) {
    fImpl->AllocateCandidates(requested.candidates);
    fImpl->fCapacities.candidates = requested.candidates;
  }
  if (requested.daughterIds > fImpl->fCapacities.daughterIds) {
    fImpl->AllocateIo("fDaughterSourceIds", fImpl->fDaughterSourceIds, requested.daughterIds);
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
    fImpl->AllocateIo(
      "fTwoDaughterRoutingDescriptors", fImpl->fTwoDaughterRoutingDescriptors, requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterRoutingChannelVisitedCounters", fImpl->fTwoDaughterRoutingChannelVisitedCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterRoutingChannelAcceptedCounters", fImpl->fTwoDaughterRoutingChannelAcceptedCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterRoutingChannelStoredCounters", fImpl->fTwoDaughterRoutingChannelStoredCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterRoutingChannelConstructedCounters", fImpl->fTwoDaughterRoutingChannelConstructedCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterSelectionAcceptedCounters", fImpl->fTwoDaughterSelectionAcceptedCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterSelectionStoredCounters", fImpl->fTwoDaughterSelectionStoredCounters,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterSelectionOffsets", fImpl->fTwoDaughterSelectionOffsets,
      requested.twoDaughterRoutingDescriptors);
    fImpl->AllocateIo(
      "fTwoDaughterSelectionCursors", fImpl->fTwoDaughterSelectionCursors,
      requested.twoDaughterRoutingDescriptors);
    fImpl->fCapacities.twoDaughterRoutingDescriptors =
      requested.twoDaughterRoutingDescriptors;
  }
  if (requested.twoDaughterRoutingCompatibility
      > fImpl->fCapacities.twoDaughterRoutingCompatibility) {
    fImpl->AllocateIo(
      "fTwoDaughterRoutingCompatibility", fImpl->fTwoDaughterRoutingCompatibility,
      requested.twoDaughterRoutingCompatibility);
    fImpl->fCapacities.twoDaughterRoutingCompatibility =
      requested.twoDaughterRoutingCompatibility;
  }
  if (requested.twoDaughterRoutingGroups
      > fImpl->fCapacities.twoDaughterRoutingGroups) {
    fImpl->AllocateIo("fTwoDaughterRoutingGroups", fImpl->fTwoDaughterRoutingGroups,
                      requested.twoDaughterRoutingGroups);
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
    fImpl->AllocateIo("fV0TrackRoutingDescriptors", fImpl->fV0TrackRoutingDescriptors,
                      requested.v0TrackRoutingDescriptors);
    fImpl->AllocateIo("fV0TrackRoutingChannelVisitedCounters", fImpl->fV0TrackRoutingChannelVisitedCounters,
                      requested.v0TrackRoutingDescriptors);
    fImpl->AllocateIo("fV0TrackRoutingChannelAcceptedCounters", fImpl->fV0TrackRoutingChannelAcceptedCounters,
                      requested.v0TrackRoutingDescriptors);
    fImpl->AllocateIo("fV0TrackRoutingChannelStoredCounters", fImpl->fV0TrackRoutingChannelStoredCounters,
                      requested.v0TrackRoutingDescriptors);
    fImpl->AllocateIo("fV0TrackRoutingChannelConstructedCounters", fImpl->fV0TrackRoutingChannelConstructedCounters,
                      requested.v0TrackRoutingDescriptors);
    fImpl->fCapacities.v0TrackRoutingDescriptors = requested.v0TrackRoutingDescriptors;
  }
  if (requested.v0TrackRoutingCompatibility > fImpl->fCapacities.v0TrackRoutingCompatibility) {
    fImpl->AllocateIo("fV0TrackRoutingCompatibility", fImpl->fV0TrackRoutingCompatibility,
                      requested.v0TrackRoutingCompatibility);
    fImpl->fCapacities.v0TrackRoutingCompatibility = requested.v0TrackRoutingCompatibility;
  }
  if (requested.v0TrackRoutingGroups > fImpl->fCapacities.v0TrackRoutingGroups) {
    fImpl->AllocateIo("fV0TrackRoutingGroups", fImpl->fV0TrackRoutingGroups, requested.v0TrackRoutingGroups);
    fImpl->fCapacities.v0TrackRoutingGroups = requested.v0TrackRoutingGroups;
  }
  if (requested.decayGraphNodes > fImpl->fCapacities.decayGraphNodes) {
    fImpl->AllocateIo("fDecayGraphNodes", fImpl->fDecayGraphNodes, requested.decayGraphNodes);
    fImpl->fCapacities.decayGraphNodes = requested.decayGraphNodes;
  }
  if (requested.decayGraphGroups > fImpl->fCapacities.decayGraphGroups) {
    fImpl->AllocateIo("fDecayGraphGroups", fImpl->fDecayGraphGroups, requested.decayGraphGroups);
    fImpl->fCapacities.decayGraphGroups = requested.decayGraphGroups;
  }
  if (requested.decayGraphFamilies > fImpl->fCapacities.decayGraphFamilies) {
    fImpl->AllocateIo("fDecayGraphFamilyCoverage", fImpl->fDecayGraphFamilyCoverage, requested.decayGraphFamilies);
    fImpl->fCapacities.decayGraphFamilies = requested.decayGraphFamilies;
  }
  if (requested.graphOperationDescriptors
      > fImpl->fCapacities.graphOperationDescriptors) {
    fImpl->AllocateIo(
      "fGraphOperationDescriptors", fImpl->fGraphOperationDescriptors, requested.graphOperationDescriptors);
    fImpl->AllocateDevice(
      "fGraphOperationChannelVisitedCounters", fImpl->fGraphOperationChannelVisitedCounters,
      requested.graphOperationDescriptors);
    fImpl->AllocateDevice(
      "fGraphOperationChannelAcceptedCounters", fImpl->fGraphOperationChannelAcceptedCounters,
      requested.graphOperationDescriptors);
    fImpl->AllocateDevice(
      "fGraphOperationChannelStoredCounters", fImpl->fGraphOperationChannelStoredCounters,
      requested.graphOperationDescriptors);
    fImpl->AllocateDevice(
      "fGraphOperationChannelConstructedCounters", fImpl->fGraphOperationChannelConstructedCounters,
      requested.graphOperationDescriptors);
    fImpl->AllocateDevice(
      "fGraphOperationChannelRejectedCounters", fImpl->fGraphOperationChannelRejectedCounters,
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
      || before.transactions != after.transactions || before.domains != after.domains
      || before.workRanges != after.workRanges
      || before.twoDaughterPairTiles != after.twoDaughterPairTiles
      || before.twoDaughterTaskTiles != after.twoDaughterTaskTiles
      || before.domainStatuses != after.domainStatuses
      || before.domainChannelStatuses != after.domainChannelStatuses
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
  bytes += static_cast<std::uint64_t>(c.transactions) * sizeof(KFParticleGpuTransactionDesc);
  bytes += static_cast<std::uint64_t>(c.domains) * sizeof(KFParticleGpuDomainDesc);
  bytes += static_cast<std::uint64_t>(c.workRanges)
           * (sizeof(KFParticleGpuWorkRange) + sizeof(KFParticleGpuWorkStatus));
  bytes += static_cast<std::uint64_t>(c.twoDaughterPairTiles)
           * sizeof(KFParticleGpuPairTile);
  bytes += static_cast<std::uint64_t>(c.twoDaughterTaskTiles) * sizeof(KFParticleGpuTaskTile);
  bytes += static_cast<std::uint64_t>(c.domainStatuses) * sizeof(KFParticleGpuDomainStatus);
  bytes += static_cast<std::uint64_t>(c.domainChannelStatuses)
           * sizeof(KFParticleGpuDomainChannelStatus);
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

std::uint64_t KFParticleGpuBufferManager::CandidatePayloadBytes(
  unsigned int candidates,
  unsigned int daughters,
  bool includeSelectionDiagnostics) const
{
  const auto& c = fImpl->fCapacities;
  if (candidates > c.candidates || daughters > c.daughterIds) {
    throw std::out_of_range("KFParticle GPU candidate output exceeds allocated capacity");
  }
  std::uint64_t bytes = static_cast<std::uint64_t>(candidates)
    * (sizeof(float) * (KFParticleGpuFitState::NumberOfParameters
                        + KFParticleGpuFitState::NumberOfCovarianceElements
                        + KFParticleGpuFitSoALayout::NumberOfFloatComponents)
       + sizeof(int) * (KFParticleGpuFitSoALayout::NumberOfIntegerComponents
                        + KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents)
       + sizeof(unsigned int)
           * KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents);
  if (includeSelectionDiagnostics) {
    bytes += static_cast<std::uint64_t>(candidates)
             * sizeof(KFParticleGpuV0SelectionResult);
  }
  return bytes + static_cast<std::uint64_t>(daughters) * sizeof(int)
         + 3u * sizeof(unsigned int);
}

std::uint64_t KFParticleGpuBufferManager::SelectedCandidatePayloadBytes() const
{
  return static_cast<std::uint64_t>(fImpl->fCapacities.selectedCandidates)
           * 2u * sizeof(unsigned int)
         + 2u * sizeof(unsigned int);
}

std::uint64_t KFParticleGpuBufferManager::SelectedCandidatePayloadBytes(
  unsigned int candidates) const
{
  if (candidates > fImpl->fCapacities.selectedCandidates) {
    throw std::out_of_range("KFParticle GPU selected output exceeds allocated capacity");
  }
  return static_cast<std::uint64_t>(candidates) * 2u * sizeof(unsigned int)
         + 2u * sizeof(unsigned int);
}

std::uint64_t KFParticleGpuBufferManager::TransactionStatusPayloadBytes(
  bool includeWorkStatuses,
  bool includeChannelStatuses,
  bool includeOverflowFlags) const
{
  return (includeWorkStatuses
            ? static_cast<std::uint64_t>(fImpl->fWorkRangeSize)
                * sizeof(KFParticleGpuWorkStatus)
            : 0u)
         + static_cast<std::uint64_t>(fImpl->fDomainStatusSize)
             * sizeof(KFParticleGpuDomainStatus)
         + (includeChannelStatuses
              ? static_cast<std::uint64_t>(fImpl->fDomainChannelStatusSize)
                  * sizeof(KFParticleGpuDomainChannelStatus)
              : 0u)
         + (includeOverflowFlags ? 2u * sizeof(unsigned int) : 0u);
}

KFParticleGpuMemoryUsage KFParticleGpuBufferManager::MemoryUsage() const
{
  KFParticleGpuMemoryUsage usage;
  const auto add = [&](const auto& buffer) {
    if (!buffer.get()) { return; }
    const xpu::buffer_prop property(buffer);
    if (property.type() != xpu::buf_pinned) { usage.deviceBytes += property.size_bytes(); }
    if (property.type() == xpu::buf_io || property.type() == xpu::buf_pinned) {
      usage.pinnedHostBytes += property.size_bytes();
    }
  };
  add(fImpl->fTrackParameters);
  add(fImpl->fTrackCovariances);
  add(fImpl->fTrackField);
  add(fImpl->fTrackChiToPrimaryVertex);
  add(fImpl->fTrackIntegers);
  add(fImpl->fVertexParameters);
  add(fImpl->fVertexCovariances);
  add(fImpl->fVertexChi2);
  add(fImpl->fVertexIntegers);
  add(fImpl->fEvents);
  add(fImpl->fTransactions);
  add(fImpl->fDomains);
  add(fImpl->fWorkRanges);
  add(fImpl->fWorkStatuses);
  add(fImpl->fTwoDaughterPairTiles);
  add(fImpl->fTwoDaughterTaskTiles);
  add(fImpl->fDomainStatuses);
  add(fImpl->fDomainChannelStatuses);
  add(fImpl->fTwoDaughterTasks);
  add(fImpl->fTwoDaughterTaskCount);
  add(fImpl->fTwoDaughterTotalPairCount);
  add(fImpl->fTwoDaughterTaskOverflowFlags);
  add(fImpl->fTwoDaughterRoutedTasks);
  add(fImpl->fTwoDaughterRoutingVisitedPairCount);
  add(fImpl->fTwoDaughterRoutingActiveBitCount);
  add(fImpl->fTwoDaughterRoutedAcceptedTaskCount);
  add(fImpl->fTwoDaughterRoutedStoredTaskCount);
  add(fImpl->fTwoDaughterRoutingBlockReservationCount);
  add(fImpl->fTwoDaughterRoutedTaskOverflowFlags);
  add(fImpl->fTwoDaughterRoutingDescriptors);
  add(fImpl->fTwoDaughterRoutingCompatibility);
  add(fImpl->fTwoDaughterRoutingGroups);
  add(fImpl->fTwoDaughterRoutingEnabledChannels);
  add(fImpl->fTwoDaughterRoutingChannelVisitedCounters);
  add(fImpl->fTwoDaughterRoutingChannelAcceptedCounters);
  add(fImpl->fTwoDaughterRoutingChannelStoredCounters);
  add(fImpl->fTwoDaughterRoutingChannelConstructedCounters);
  add(fImpl->fTwoDaughterSelectionAcceptedCounters);
  add(fImpl->fTwoDaughterSelectionStoredCounters);
  add(fImpl->fTwoDaughterSelectionOffsets);
  add(fImpl->fTwoDaughterSelectionCursors);
  add(fImpl->fV0TrackTasks);
  add(fImpl->fV0TrackTaskCount);
  add(fImpl->fV0TrackTotalPairCount);
  add(fImpl->fV0TrackTaskOverflowFlags);
  add(fImpl->fV0TrackRoutedTasks);
  add(fImpl->fV0TrackRoutingVisitedPairCount);
  add(fImpl->fV0TrackRoutingActiveBitCount);
  add(fImpl->fV0TrackRoutedAcceptedTaskCount);
  add(fImpl->fV0TrackRoutedStoredTaskCount);
  add(fImpl->fV0TrackRoutingBlockReservationCount);
  add(fImpl->fV0TrackRoutedTaskOverflowFlags);
  add(fImpl->fV0TrackRoutingDescriptors);
  add(fImpl->fV0TrackRoutingCompatibility);
  add(fImpl->fV0TrackRoutingGroups);
  add(fImpl->fV0TrackRoutingEnabledChannels);
  add(fImpl->fV0TrackRoutingChannelVisitedCounters);
  add(fImpl->fV0TrackRoutingChannelAcceptedCounters);
  add(fImpl->fV0TrackRoutingChannelStoredCounters);
  add(fImpl->fV0TrackRoutingChannelConstructedCounters);
  add(fImpl->fDecayGraphNodes);
  add(fImpl->fDecayGraphGroups);
  add(fImpl->fDecayGraphFamilyCoverage);
  add(fImpl->fGraphOperationDescriptors);
  add(fImpl->fGraphOperationTasks);
  add(fImpl->fGraphOperationResults);
  add(fImpl->fGraphOperationVisitedCombinations);
  add(fImpl->fGraphOperationAcceptedTasks);
  add(fImpl->fGraphOperationStoredTasks);
  add(fImpl->fGraphOperationConstructedCandidates);
  add(fImpl->fGraphOperationRejectedTasks);
  add(fImpl->fGraphOperationOverflowFlags);
  add(fImpl->fGraphOperationChannelVisitedCounters);
  add(fImpl->fGraphOperationChannelAcceptedCounters);
  add(fImpl->fGraphOperationChannelStoredCounters);
  add(fImpl->fGraphOperationChannelConstructedCounters);
  add(fImpl->fGraphOperationChannelRejectedCounters);
  add(fImpl->fCandidateParameters);
  add(fImpl->fCandidateCovariances);
  add(fImpl->fCandidateFitScalars);
  add(fImpl->fCandidateFitIntegers);
  add(fImpl->fCandidateMetadataIntegers);
  add(fImpl->fCandidateMetadataUnsigned);
  add(fImpl->fCandidateRoutingDescriptorIndices);
  add(fImpl->fDaughterSourceIds);
  add(fImpl->fCandidateSize);
  add(fImpl->fDaughterSize);
  add(fImpl->fOverflowFlags);
  add(fImpl->fV0SelectionResults);
  add(fImpl->fSelectedCandidateIndices);
  add(fImpl->fSelectedCandidateChannelIds);
  add(fImpl->fSelectedCandidateSize);
  add(fImpl->fSelectedCandidateOverflowFlags);
  add(fImpl->fHostTwoDaughterRoutedTasks);
  add(fImpl->fHostV0TrackRoutedTasks);
  add(fImpl->fHostCandidateRoutingDescriptorIndices);
  add(fImpl->fHostGraphOperationResults);
  add(fImpl->fHostV0SelectionResults);
  return usage;
}

std::uint64_t KFParticleGpuBufferManager::AllocatedBytes() const
{
  return MemoryUsage().deviceBytes;
}

std::uint64_t KFParticleGpuBufferManager::SelectionResetLaunches() const
{
  return fImpl->fSelectionResetLaunches;
}

std::uint64_t KFParticleGpuBufferManager::EstimateAllocatedBytes(
  const KFParticleGpuBufferCapacities& c)
{
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
  bytes += static_cast<std::uint64_t>(c.transactions) * sizeof(KFParticleGpuTransactionDesc);
  bytes += static_cast<std::uint64_t>(c.domains) * sizeof(KFParticleGpuDomainDesc);
  bytes += static_cast<std::uint64_t>(c.workRanges)
           * (sizeof(KFParticleGpuWorkRange) + sizeof(KFParticleGpuWorkStatus));
  bytes += static_cast<std::uint64_t>(c.twoDaughterPairTiles)
           * sizeof(KFParticleGpuPairTile);
  bytes += static_cast<std::uint64_t>(c.twoDaughterTaskTiles) * sizeof(KFParticleGpuTaskTile);
  bytes += static_cast<std::uint64_t>(c.domainStatuses) * sizeof(KFParticleGpuDomainStatus);
  bytes += static_cast<std::uint64_t>(c.domainChannelStatuses)
           * sizeof(KFParticleGpuDomainChannelStatus);
  bytes += static_cast<std::uint64_t>(c.candidates)
           * (sizeof(float) * (KFParticleGpuFitState::NumberOfParameters
                               + KFParticleGpuFitState::NumberOfCovarianceElements
                               + KFParticleGpuFitSoALayout::NumberOfFloatComponents)
              + sizeof(int) * (KFParticleGpuFitSoALayout::NumberOfIntegerComponents
                               + KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents)
              + sizeof(unsigned int)
                  * KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents
              + sizeof(KFParticleGpuV0SelectionResult));
  bytes += static_cast<std::uint64_t>(c.daughterIds) * sizeof(int)
           + 3u * sizeof(unsigned int);
  bytes += static_cast<std::uint64_t>(c.selectedCandidates)
             * 2u * sizeof(unsigned int)
           + 2u * sizeof(unsigned int);
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

void KFParticleGpuBufferManager::SetTransactionSizes(unsigned int transactions,
                                                    unsigned int domains,
                                                    unsigned int workRanges,
                                                    unsigned int domainStatuses,
                                                    unsigned int domainChannelStatuses)
{
  if (transactions > fImpl->fCapacities.transactions
      || domains > fImpl->fCapacities.domains
      || workRanges > fImpl->fCapacities.workRanges
      || domainStatuses > fImpl->fCapacities.domainStatuses) {
    throw std::out_of_range("KFParticle GPU transaction size exceeds buffer capacity");
  }
  if (domainChannelStatuses > fImpl->fCapacities.domainChannelStatuses) {
    throw std::out_of_range("KFParticle GPU domain-channel status size exceeds buffer capacity");
  }
  fImpl->fTransactionSize = transactions;
  fImpl->fDomainSize = domains;
  fImpl->fWorkRangeSize = workRanges;
  fImpl->fDomainStatusSize = domainStatuses;
  fImpl->fDomainChannelStatusSize = domainChannelStatuses;
}

void KFParticleGpuBufferManager::SetTwoDaughterPairTileSize(unsigned int pairTiles)
{
  if (pairTiles > fImpl->fCapacities.twoDaughterPairTiles) {
    throw std::out_of_range("KFParticle GPU pair-tile size exceeds buffer capacity");
  }
  fImpl->fTwoDaughterPairTileSize = pairTiles;
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

KFParticleGpuTransactionView KFParticleGpuBufferManager::HostTransaction()
{
  return KFParticleGpuTransactionView(HostPointer(fImpl->fTransactions),
                                      HostPointer(fImpl->fDomains),
                                      HostPointer(fImpl->fWorkRanges),
                                      HostPointer(fImpl->fWorkStatuses),
                                      HostPointer(fImpl->fDomainStatuses),
                                      fImpl->fDomainSize,
                                      fImpl->fWorkRangeSize,
                                      fImpl->fDomainStatusSize,
                                      HostPointer(fImpl->fDomainChannelStatuses),
                                      fImpl->fDomainChannelStatusSize);
}

KFParticleGpuTransactionView KFParticleGpuBufferManager::DeviceTransaction()
{
  return KFParticleGpuTransactionView(fImpl->fTransactions.get(),
                                      fImpl->fDomains.get(),
                                      fImpl->fWorkRanges.get(),
                                      fImpl->fWorkStatuses.get(),
                                      fImpl->fDomainStatuses.get(),
                                      fImpl->fDomainSize,
                                      fImpl->fWorkRangeSize,
                                      fImpl->fDomainStatusSize,
                                      fImpl->fDomainChannelStatuses.get(),
                                      fImpl->fDomainChannelStatusSize);
}

KFParticleGpuPairTile* KFParticleGpuBufferManager::HostTwoDaughterPairTiles()
{
  return HostPointer(fImpl->fTwoDaughterPairTiles);
}

const KFParticleGpuPairTile* KFParticleGpuBufferManager::DeviceTwoDaughterPairTiles()
{
  return fImpl->fTwoDaughterPairTiles.get();
}

KFParticleGpuTaskTile* KFParticleGpuBufferManager::HostTwoDaughterTaskTiles()
{
  return HostPointer(fImpl->fTwoDaughterTaskTiles);
}

const KFParticleGpuTaskTile* KFParticleGpuBufferManager::DeviceTwoDaughterTaskTiles()
{
  return fImpl->fTwoDaughterTaskTiles.get();
}

void KFParticleGpuBufferManager::UploadTwoDaughterTaskTiles(unsigned int count)
{
  KFParticleGpuStepTrace timer("H2D UploadTwoDaughterTaskTiles");
  if (count > fImpl->fCapacities.twoDaughterTaskTiles) {
    throw std::out_of_range("construction task-map upload exceeds capacity");
  }
  if (count != 0u) {
    fImpl->fQueue.copy(HostPointer(fImpl->fTwoDaughterTaskTiles),
                      fImpl->fTwoDaughterTaskTiles.get(), count);
  }
}

void KFParticleGpuBufferManager::UploadTransaction(bool includeStatuses)
{
  KFParticleGpuStepTrace timer("H2D UploadTransaction + wait");
  CopyIfAllocated(fImpl->fQueue, fImpl->fTransactions, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDomains, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fWorkRanges, xpu::h2d);
  if (fImpl->fTwoDaughterPairTileSize != 0u) {
    fImpl->fQueue.copy(HostPointer(fImpl->fTwoDaughterPairTiles),
                       fImpl->fTwoDaughterPairTiles.get(),
                       fImpl->fTwoDaughterPairTileSize);
  }
  if (includeStatuses) {
    CopyIfAllocated(fImpl->fQueue, fImpl->fWorkStatuses, xpu::h2d);
    CopyIfAllocated(fImpl->fQueue, fImpl->fDomainStatuses, xpu::h2d);
    CopyIfAllocated(fImpl->fQueue, fImpl->fDomainChannelStatuses, xpu::h2d);
  }
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::QueueDownloadTransactionStatus(
  bool includeWorkStatuses,
  bool includeChannelStatuses,
  bool includeOverflowFlags)
{
  if (includeWorkStatuses) {
    CopyPrefix(fImpl->fQueue, fImpl->fWorkStatuses, fImpl->fWorkRangeSize);
  }
  CopyPrefix(fImpl->fQueue, fImpl->fDomainStatuses, fImpl->fDomainStatusSize);
  if (includeChannelStatuses) {
    CopyPrefix(fImpl->fQueue, fImpl->fDomainChannelStatuses,
               fImpl->fDomainChannelStatusSize);
  }
  if (includeOverflowFlags) {
    CopyPrefix(fImpl->fQueue, fImpl->fOverflowFlags, 1u);
    CopyPrefix(fImpl->fQueue, fImpl->fSelectedCandidateOverflowFlags, 1u);
  }
}

void KFParticleGpuBufferManager::DownloadTransactionStatus(
  bool includeWorkStatuses,
  bool includeChannelStatuses,
  bool includeOverflowFlags)
{
  KFParticleGpuStepTrace timer("D2H DownloadTransactionStatus + wait");
  QueueDownloadTransactionStatus(includeWorkStatuses,
                                 includeChannelStatuses,
                                 includeOverflowFlags);
  fImpl->fQueue.wait();
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
    fImpl->HostMirror("fHostV0SelectionResults", fImpl->fV0SelectionResults, fImpl->fHostV0SelectionResults),
    fImpl->fCapacities.candidates);
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
    fImpl->HostMirror("fHostCandidateRoutingDescriptorIndices", fImpl->fCandidateRoutingDescriptorIndices,
                      fImpl->fHostCandidateRoutingDescriptorIndices),
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
  KFParticleGpuStepTrace timer("CPU/ALLOC/H2D UploadTwoDaughterRoutingPlan");
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
  KFParticleGpuStepTrace timer("CPU/ALLOC/H2D UploadV0TrackRoutingPlan");
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
  }
  CopyIfAllocated(
    fImpl->fQueue, fImpl->fGraphOperationDescriptors, xpu::h2d);
  if (fImpl->fGraphOperationChannelVisitedCounters.get()) {
    fImpl->fQueue.memset(fImpl->fGraphOperationChannelVisitedCounters, 0);
    fImpl->fQueue.memset(fImpl->fGraphOperationChannelAcceptedCounters, 0);
    fImpl->fQueue.memset(fImpl->fGraphOperationChannelStoredCounters, 0);
    fImpl->fQueue.memset(fImpl->fGraphOperationChannelConstructedCounters, 0);
    fImpl->fQueue.memset(fImpl->fGraphOperationChannelRejectedCounters, 0);
  }
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

KFParticleGpuGraphOperationResult*
KFParticleGpuBufferManager::HostGraphOperationResults()
{
  return fImpl->HostMirror("fHostGraphOperationResults", fImpl->fGraphOperationResults,
                           fImpl->fHostGraphOperationResults);
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
  KFParticleGpuStepTrace timer("H2D UploadInput + wait");
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
  CopyIfAllocated(fImpl->fQueue, fImpl->fTransactions, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDomains, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fWorkRanges, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDomainStatuses, xpu::h2d);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDomainChannelStatuses, xpu::h2d);
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
  if (fImpl->fCandidateRoutingDescriptorIndices.get()) {
    fImpl->fQueue.copy(HostCandidateDescriptorIndices().Data(),
      fImpl->fCandidateRoutingDescriptorIndices.get(), fImpl->fCapacities.candidates);
  }
  if (fImpl->fV0SelectionResults.get()) {
    fImpl->fQueue.copy(HostV0SelectionResults().Data(),
      fImpl->fV0SelectionResults.get(), fImpl->fCapacities.candidates);
  }
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
  KFParticleGpuStepTrace timer("CPU/H2D ResetCandidates + wait");
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
  fImpl->DownloadMirror("fHostTwoDaughterRoutedTasks", fImpl->fTwoDaughterRoutedTasks,
                        fImpl->fHostTwoDaughterRoutedTasks);
  fImpl->fQueue.wait();
}

KFParticleGpuTwoDaughterRoutedTask*
KFParticleGpuBufferManager::HostTwoDaughterRoutedTasks()
{
  return fImpl->HostMirror("fHostTwoDaughterRoutedTasks", fImpl->fTwoDaughterRoutedTasks,
                           fImpl->fHostTwoDaughterRoutedTasks);
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
  fImpl->DownloadMirror("fHostV0TrackRoutedTasks", fImpl->fV0TrackRoutedTasks,
                        fImpl->fHostV0TrackRoutedTasks);
  fImpl->fQueue.wait();
}

KFParticleGpuV0TrackRoutedTask* KFParticleGpuBufferManager::HostV0TrackRoutedTasks()
{
  return fImpl->HostMirror("fHostV0TrackRoutedTasks", fImpl->fV0TrackRoutedTasks, fImpl->fHostV0TrackRoutedTasks);
}

void KFParticleGpuBufferManager::ResetSelectedCandidates()
{
  KFParticleGpuStepTrace timer("CPU/H2D ResetSelectedCandidates + wait");
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
  KFParticleGpuStepTrace timer("KERNEL ResetSelectionResults + wait");
  if (!fImpl->fV0SelectionResults.get()) {
    throw std::logic_error("KFParticle GPU selection-result buffers must be initialized before reset");
  }
  fImpl->fQueue.launch<KFParticleGpuResetSelectionResults>(
    xpu::n_threads(fImpl->fCapacities.candidates),
    fImpl->fV0SelectionResults.get(), fImpl->fCapacities.candidates);
  ++fImpl->fSelectionResetLaunches;
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
  fImpl->DownloadMirror("fHostV0SelectionResults", fImpl->fV0SelectionResults,
                        fImpl->fHostV0SelectionResults);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSourceIds, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadCandidateOutput(
  unsigned int candidates,
  unsigned int daughters,
  bool includeSelectionDiagnostics)
{
  if (candidates > fImpl->fCapacities.candidates
      || daughters > fImpl->fCapacities.daughterIds) {
    throw std::out_of_range("KFParticle GPU candidate output exceeds allocated capacity");
  }
  const unsigned int stride = fImpl->fCapacities.candidates;
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateParameters,
                KFParticleGpuFitState::NumberOfParameters, stride, candidates);
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateCovariances,
                KFParticleGpuFitState::NumberOfCovarianceElements, stride, candidates);
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateFitScalars,
                KFParticleGpuFitSoALayout::NumberOfFloatComponents, stride, candidates);
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateFitIntegers,
                KFParticleGpuFitSoALayout::NumberOfIntegerComponents, stride, candidates);
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateMetadataIntegers,
                KFParticleGpuCandidateMetadataLayout::NumberOfIntegerComponents, stride, candidates);
  CopySoAPrefix(fImpl->fQueue, fImpl->fCandidateMetadataUnsigned,
                KFParticleGpuCandidateMetadataLayout::NumberOfUnsignedComponents, stride, candidates);
  if (includeSelectionDiagnostics) {
    fImpl->DownloadMirror("fHostV0SelectionResults", fImpl->fV0SelectionResults,
                          fImpl->fHostV0SelectionResults, candidates);
  }
  CopyPrefix(fImpl->fQueue, fImpl->fDaughterSourceIds, daughters);
  CopyIfAllocated(fImpl->fQueue, fImpl->fCandidateSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fDaughterSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadCandidateDescriptorIndices()
{
  fImpl->DownloadMirror("fHostCandidateRoutingDescriptorIndices", fImpl->fCandidateRoutingDescriptorIndices,
                        fImpl->fHostCandidateRoutingDescriptorIndices);
  fImpl->fQueue.wait();
}

void KFParticleGpuBufferManager::DownloadV0SelectionResults()
{
  fImpl->DownloadMirror("fHostV0SelectionResults", fImpl->fV0SelectionResults,
                        fImpl->fHostV0SelectionResults);
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

void KFParticleGpuBufferManager::DownloadSelectedCandidateOutput(unsigned int candidates)
{
  if (candidates > fImpl->fCapacities.selectedCandidates) {
    throw std::out_of_range("KFParticle GPU selected output exceeds allocated capacity");
  }
  CopyPrefix(fImpl->fQueue, fImpl->fSelectedCandidateIndices, candidates);
  CopyPrefix(fImpl->fQueue, fImpl->fSelectedCandidateChannelIds, candidates);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateSize, xpu::d2h);
  CopyIfAllocated(fImpl->fQueue, fImpl->fSelectedCandidateOverflowFlags, xpu::d2h);
  fImpl->fQueue.wait();
}
