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
 * @brief Reusable GPU buffer ownership, capacity accounting, and host/device transfer API.
 */

#ifndef KFPARTICLEGPUBUFFERMANAGER_H
#define KFPARTICLEGPUBUFFERMANAGER_H

#include "data/KFParticleGpuCandidateTransfer.h"
#include "planning/KFParticleGpuDecayGraphPlan.h"
#include "execution/KFParticleGpuDeviceStorage.h"
#include "data/KFParticleGpuInputDataTransfer.h"
#include "diagnostics/KFParticleGpuHostTiming.h"
#include "algorithms/KFParticleGpuSelection.h"
#include "data/KFParticleGpuTransaction.h"

#include <cstdint>
#include <memory>
#include <vector>

/** @brief One timed buffer allocation, recorded only when allocation tracing is enabled. */
struct KFParticleGpuAllocationRecord
{
  const char* buffer = "";     ///< Buffer label with static storage duration.
  const char* phase = "";      ///< Execution phase label with static storage duration.
  std::uint64_t bytes = 0u;    ///< Requested payload bytes; not the sum of host and device storage.
  unsigned int elements = 0u;  ///< Requested number of elements.
  bool replacement = false;    ///< True when replacing an existing allocation, not allocating initially.
  double milliseconds = 0.;    ///< Host wall time spent allocating the buffer.
  const char* memory = "io";   ///< Allocation kind: "io", "device", or "pinned".
};

/**
 * @brief Allocated payload bytes, separated by logical memory domain.
 *
 * IO buffers contribute to both fields. On the CPU backend their host and device
 * pointers alias, so adding the fields does not describe physical memory usage.
 */
struct KFParticleGpuMemoryUsage
{
  std::uint64_t deviceBytes = 0u;      ///< Payload of device-only buffers and device sides of IO buffers.
  std::uint64_t pinnedHostBytes = 0u;  ///< Payload of IO host sides and optional pinned host mirrors.
};

namespace xpu
{
  class queue;
}

/**
 * @brief Element capacities for the buffer families managed by KFParticleGpuBufferManager.
 *
 * These are allocation limits, not current event sizes or numbers of produced
 * candidates. SoA component arrays use the corresponding object capacity as
 * their stride. KFParticleGpuBufferManager::EnsureCapacity() retains allocations
 * when requests get smaller.
 */
struct KFParticleGpuBufferCapacities
{
  unsigned int tracks = 0;                           ///< Input track states and metadata.
  unsigned int vertices = 0;                         ///< Primary vertex states and metadata.
  unsigned int events = 0;                           ///< Event descriptors.
  unsigned int transactions = 0;                     ///< Transaction descriptors.
  unsigned int domains = 0;                          ///< Independent reconstruction domains.
  unsigned int workRanges = 0;                       ///< Work ranges and their status records.
  unsigned int twoDaughterPairTiles = 0;             ///< Input pair-generation tiles.
  unsigned int twoDaughterTaskTiles = 0;             ///< Construction tiles over routed tasks.
  unsigned int domainStatuses = 0;                   ///< Per-domain status records.
  unsigned int domainChannelStatuses = 0;            ///< Per-domain, per-channel status records.
  unsigned int candidates = 0;                       ///< Fit states, metadata, and selection diagnostics.
  unsigned int daughterIds = 0;                      ///< Flattened daughter identifiers.
  unsigned int selectedCandidates = 0;               ///< Selected indices; zero defaults to requested candidate capacity.
  unsigned int twoDaughterTasks = 0;                 ///< Explicit two-daughter construction tasks.
  unsigned int twoDaughterRoutedTasks = 0;           ///< Channel-routed two-daughter tasks.
  unsigned int twoDaughterRoutingDescriptors = 0;    ///< Two-daughter channel descriptors and counters.
  unsigned int twoDaughterRoutingCompatibility = 0;  ///< Two-daughter compatibility table entries.
  unsigned int twoDaughterRoutingGroups = 0;         ///< Two-daughter routing groups.
  unsigned int v0TrackTasks = 0;                     ///< Explicit V0-plus-track construction tasks.
  unsigned int v0TrackRoutedTasks = 0;               ///< Channel-routed V0-plus-track tasks.
  unsigned int v0TrackRoutingDescriptors = 0;        ///< V0-plus-track channel descriptors and counters.
  unsigned int v0TrackRoutingCompatibility = 0;      ///< V0-plus-track compatibility table entries.
  unsigned int v0TrackRoutingGroups = 0;             ///< V0-plus-track routing groups.
  unsigned int decayGraphNodes = 0;                  ///< Decay dependency graph nodes.
  unsigned int decayGraphGroups = 0;                 ///< Decay dependency graph groups.
  unsigned int decayGraphFamilies = 0;               ///< Decay-family coverage records.
  unsigned int graphOperationDescriptors = 0;        ///< Graph-operation channel descriptors and counters.
  unsigned int graphOperationTasks = 0;              ///< Graph-operation tasks and their results.
  bool nonhomogeneousField = false;                  ///< Request per-track field coefficients; retained once enabled.
};

/** @brief Host snapshot of explicit two-daughter task-generation counters. */
struct KFParticleGpuTwoDaughterTaskStatus
{
  unsigned int accepted = 0;       ///< Accepted task count; may exceed storage capacity on overflow.
  unsigned int totalPairs = 0;     ///< Pair count reported by the generation kernel.
  unsigned int overflowFlags = 0;  ///< Task-generation overflow bit mask.
};

/** @brief Host snapshot of explicit V0-plus-track task-generation counters. */
struct KFParticleGpuV0TrackTaskStatus
{
  unsigned int accepted = 0;       ///< Accepted task count; may exceed storage capacity on overflow.
  unsigned int totalPairs = 0;     ///< Pair count reported by the generation kernel.
  unsigned int overflowFlags = 0;  ///< Task-generation overflow bit mask.
};

/** @brief Host snapshot of candidate-pool counters, without downloading particle data. */
struct KFParticleGpuCandidatePoolStatus
{
  unsigned int candidates = 0;     ///< Candidate counter; check capacity and overflow before reading output.
  unsigned int daughters = 0;      ///< Flattened daughter-storage counter.
  unsigned int overflowFlags = 0;  ///< Candidate-pool overflow bit mask.
};

/**
 * @brief Owns reusable host/device buffers and transfers for KFParticle GPU reconstruction.
 *
 * Stores input tracks and vertices, transaction metadata, routing plans, task
 * workspaces, and candidate output. IO buffers serve host/device transfers;
 * device-only workspaces receive pinned host mirrors only on explicit request.
 * Capacities are retained across events and batches, independently of active sizes.
 * Replacing a buffer does not preserve its previous contents.
 *
 * The supplied XPU queue is borrowed and must outlive the manager. Views and
 * pointers are non-owning: obtaining one does not copy data or wait for the GPU,
 * and replacing its buffer or destroying the manager invalidates it.
 *
 * @note The manager has no internal locking; callers must serialize mutations.
 * Complete outstanding operations before replacing buffers or destroying the
 * manager, and do not modify host memory while a queued transfer is using it.
 * Transfer and reset methods below state whether they wait for the queue.
 */
class KFParticleGpuBufferManager
{
 public:
  /// @brief Creates an empty manager borrowing @p queue; payload allocation is deferred.
  explicit KFParticleGpuBufferManager(xpu::queue& queue);

  /// @brief Releases owned buffers without an explicit queue wait; the caller must finish queued work first.
  ~KFParticleGpuBufferManager();

  /// @brief Copying is disabled: allocations and mutable queue-related state have one owner.
  KFParticleGpuBufferManager(const KFParticleGpuBufferManager&) = delete;

  /// @brief Copy assignment is disabled for the same ownership reason as copy construction.
  KFParticleGpuBufferManager& operator=(const KFParticleGpuBufferManager&) = delete;

  /// @name Allocation and accounting
  /// @{

  /**
   * @brief Allocates missing buffers or replaces undersized ones without shrinking others.
   * @param requested Minimum capacities, in elements of each buffer family.
   * @note Replaced contents are discarded; active sizes and plan revisions are not reset.
   * Complete outstanding buffer use first, then refill affected data and refresh views.
   */
  void EnsureCapacity(const KFParticleGpuBufferCapacities& requested);

  /// @brief Returns current allocation capacities, not the number of valid elements.
  const KFParticleGpuBufferCapacities& Capacities() const;

  /// @brief Returns a borrowed reference to the underlying XPU buffer handles; performs no transfer.
  const KFParticleGpuDeviceStorage& Storage() const;

  /// @brief Counts EnsureCapacity() calls that changed capacities, not individual allocations or copies.
  std::uint64_t CapacityGrowthCount() const;

  /// @brief Returns cumulative host allocation counts and times for this manager.
  const KFParticleGpuHostTiming& HostTiming() const;

  /// @brief Clears trace records and enables or disables recording; cumulative allocation timing is retained.
  void BeginAllocationTrace(bool enabled);

  /// @brief Sets the allocation phase label; @p phase must be non-null and have static storage duration.
  void SetAllocationTracePhase(const char* phase);

  /// @brief Returns up to 256 records from the current trace, retained until BeginAllocationTrace().
  const std::vector<KFParticleGpuAllocationRecord>& AllocationTrace() const;

  /// @brief Prints enabled allocation tracing and the dropped-record count to stderr; performs no GPU work.
  void PrintAllocationTrace() const;

  /**
   * @brief Estimates principal device payload bytes for planning, without allocating.
   * @note Omits routing/graph-plan tables, some auxiliary buffers, host mirrors,
   * and allocator overhead. Use MemoryUsage() to inspect actual managed payloads.
   */
  static std::uint64_t EstimateAllocatedBytes(
    const KFParticleGpuBufferCapacities& capacities);

  /// @brief Returns MemoryUsage().deviceBytes, excluding host mirrors and allocator overhead.
  std::uint64_t AllocatedBytes() const;

  /// @brief Sums currently allocated payloads by logical memory domain, including optional host mirrors.
  KFParticleGpuMemoryUsage MemoryUsage() const;

  /// @brief Returns the number of selection-result reset kernels submitted by this manager.
  std::uint64_t SelectionResetLaunches() const;

  /// @brief Returns capacity-based input/transaction payload bytes, not measured transfer traffic or active-size bytes.
  std::uint64_t InputPayloadBytes() const;

  /// @brief Returns full-capacity candidate readback bytes, including selection diagnostics and pool counters.
  std::uint64_t CandidatePayloadBytes() const;

  /**
   * @brief Returns candidate readback bytes for the specified prefixes, including pool counters.
   * @param candidates Number of candidate records.
   * @param daughters Number of flattened daughter identifiers.
   * @param includeSelectionDiagnostics Include per-candidate V0 selection results.
   * @throws std::out_of_range If either count exceeds its allocated capacity.
   */
  std::uint64_t CandidatePayloadBytes(unsigned int candidates,
                                      unsigned int daughters,
                                      bool includeSelectionDiagnostics) const;

  /// @brief Returns full-capacity selected-index readback bytes, including channel IDs and counters.
  std::uint64_t SelectedCandidatePayloadBytes() const;

  /**
   * @brief Returns selected-index readback bytes for @p candidates entries, including channel IDs and counters.
   * @throws std::out_of_range If the count exceeds selected-candidate capacity.
   */
  std::uint64_t SelectedCandidatePayloadBytes(unsigned int candidates) const;

  /**
   * @brief Returns active-size status readback bytes; domain statuses are always included.
   * @param includeWorkStatuses Include per-work-range statuses.
   * @param includeChannelStatuses Include per-domain, per-channel statuses.
   * @param includeOverflowFlags Include candidate-pool and selected-output overflow flags.
   */
  std::uint64_t TransactionStatusPayloadBytes(
    bool includeWorkStatuses = true,
    bool includeChannelStatuses = true,
    bool includeOverflowFlags = true) const;

  /// @}
  /// @name Active sizes and task capacities
  /// @{

  /**
   * @brief Sets active track, primary-vertex, and event counts without allocating or copying.
   * @throws std::out_of_range If any count exceeds its allocated capacity.
   */
  void SetInputSizes(unsigned int tracks, unsigned int vertices, unsigned int events);

  /**
   * @brief Sets active transaction, domain, work-range, and status counts without allocating or copying.
   * @throws std::out_of_range If any count exceeds its allocated capacity.
   */
  void SetTransactionSizes(unsigned int transactions,
                           unsigned int domains,
                           unsigned int workRanges,
                           unsigned int domainStatuses,
                           unsigned int domainChannelStatuses = 0u);

  /**
   * @brief Sets the pair-tile prefix transferred by UploadTransaction().
   * @throws std::out_of_range If @p pairTiles exceeds allocated pair-tile capacity.
   */
  void SetTwoDaughterPairTileSize(unsigned int pairTiles);

  /// @brief Ensures at least @p capacity explicit two-daughter tasks; follows EnsureCapacity() replacement rules.
  void EnsureTwoDaughterTaskCapacity(unsigned int capacity);

  /// @brief Ensures at least @p capacity routed two-daughter tasks; follows EnsureCapacity() replacement rules.
  void EnsureTwoDaughterRoutedTaskCapacity(unsigned int capacity);

  /// @brief Ensures at least @p capacity explicit V0-plus-track tasks; follows EnsureCapacity() replacement rules.
  void EnsureV0TrackTaskCapacity(unsigned int capacity);

  /// @brief Ensures at least @p capacity routed V0-plus-track tasks; follows EnsureCapacity() replacement rules.
  void EnsureV0TrackRoutedTaskCapacity(unsigned int capacity);

  /// @brief Ensures task/result capacity for graph operations; follows EnsureCapacity() replacement rules.
  void EnsureGraphOperationTaskCapacity(unsigned int capacity);

  /// @brief Returns the active input track count.
  unsigned int TrackSize() const;

  /// @brief Returns the active primary-vertex count.
  unsigned int VertexSize() const;

  /// @brief Returns the active event count.
  unsigned int EventSize() const;

  /// @}
  /// @name Input and transaction views and transfers
  /// @{

  /// @brief Returns a host track SoA view with active size and allocated stride.
  KFParticleGpuInputTrackSoAView HostInputTracks();

  /// @brief Returns a device track SoA view with active size and allocated stride.
  KFParticleGpuInputTrackSoAView DeviceInputTracks();

  /// @brief Returns a host primary-vertex SoA view with active size and allocated stride.
  KFParticleGpuVertexSoAView HostPrimaryVertices();

  /// @brief Returns a device primary-vertex SoA view with active size and allocated stride.
  KFParticleGpuVertexSoAView DevicePrimaryVertices();

  /// @brief Returns host event storage, or nullptr if unallocated; EventSize() gives the active count.
  KFParticleGpuEventDesc* HostEvents();

  /// @brief Returns device event storage, or nullptr if unallocated; EventSize() gives the active count.
  KFParticleGpuEventDesc* DeviceEvents();

  /// @brief Returns host transaction descriptors and statuses with their active bounds; does not refresh statuses.
  KFParticleGpuTransactionView HostTransaction();

  /// @brief Returns device transaction descriptors and statuses with their active bounds.
  KFParticleGpuTransactionView DeviceTransaction();

  /// @brief Returns writable host pair-tile storage, or nullptr if unallocated.
  KFParticleGpuPairTile* HostTwoDaughterPairTiles();

  /// @brief Returns device pair-tile storage, or nullptr if unallocated.
  const KFParticleGpuPairTile* DeviceTwoDaughterPairTiles();

  /// @brief Returns writable host construction-task tile storage, or nullptr if unallocated.
  KFParticleGpuTaskTile* HostTwoDaughterTaskTiles();

  /// @brief Returns device construction-task tile storage, or nullptr if unallocated.
  const KFParticleGpuTaskTile* DeviceTwoDaughterTaskTiles();

  /**
   * @brief Queues H2D copying of @p count construction-task tiles, without waiting.
   * @throws std::out_of_range If the count exceeds allocated task-tile capacity.
   */
  void UploadTwoDaughterTaskTiles(unsigned int count);

  /**
   * @brief Copies transaction tables and the active pair-tile prefix H2D, then waits.
   * @param includeStatuses Also upload host status buffers for initial submission.
   * Pass false when updating tables of an in-flight transaction: device-produced
   * work, domain and channel statuses must survive the update.
   * @pre With includeStatuses=false, status buffers must retain their allocation and layout.
   */
  void UploadTransaction(bool includeStatuses = true);

  /**
   * @brief Queues D2H status prefixes selected by TransactionStatusPayloadBytes(), without waiting.
   * @note Wait on the supplied queue before reading or modifying the destination host buffers.
   */
  void QueueDownloadTransactionStatus(bool includeWorkStatuses = true,
                                      bool includeChannelStatuses = true,
                                      bool includeOverflowFlags = true);

  /// @brief Calls QueueDownloadTransactionStatus() and waits before returning host-readable statuses.
  void DownloadTransactionStatus(bool includeWorkStatuses = true,
                                 bool includeChannelStatuses = true,
                                 bool includeOverflowFlags = true);

  /// @}
  /// @name Candidate, routing, and graph views
  /// @{

  /// @brief Returns host candidate/daughter storage and counters; download device results before reading them.
  KFParticleGpuCandidatePoolView HostCandidates();

  /// @brief Returns device candidate/daughter storage and counters, bounded by allocated capacities.
  KFParticleGpuCandidatePoolView DeviceCandidates();

  /// @brief Returns a host selection-result view, allocating a pinned mirror if needed but not downloading data.
  KFParticleGpuV0SelectionResultView HostV0SelectionResults();

  /// @brief Returns device V0 selection diagnostics for candidate-capacity slots.
  KFParticleGpuV0SelectionResultView DeviceV0SelectionResults();

  /// @brief Returns host selected indices, channel IDs, and counters without refreshing them.
  KFParticleGpuSelectedCandidateIndexView HostSelectedCandidates();

  /// @brief Returns device selected indices, channel IDs, and counters.
  KFParticleGpuSelectedCandidateIndexView DeviceSelectedCandidates();

  /// @brief Returns host candidate routing indices, allocating a pinned mirror if needed but not downloading data.
  KFParticleGpuCandidateDescriptorIndexView HostCandidateDescriptorIndices();

  /// @brief Returns the device mapping from candidate slots to routing descriptors.
  KFParticleGpuCandidateDescriptorIndexView DeviceCandidateDescriptorIndices();

  /// @brief Returns the host two-daughter routing plan and channel counters without refreshing counters.
  KFParticleGpuTwoDaughterRoutingView HostTwoDaughterRouting();

  /// @brief Returns the device two-daughter routing plan and channel counters.
  KFParticleGpuTwoDaughterRoutingView DeviceTwoDaughterRouting();

  /// @brief Returns device per-channel selection counts, offsets, and write cursors.
  KFParticleGpuTwoDaughterSelectionWorkspaceView DeviceTwoDaughterSelectionWorkspace();

  /// @brief Returns device routed two-daughter tasks, generation counters, and selection workspace.
  KFParticleGpuTwoDaughterGenerationStorageView
  DeviceTwoDaughterGenerationStorage();

  /// @brief Returns the host V0-plus-track routing plan and channel counters without refreshing counters.
  KFParticleGpuV0TrackRoutingView HostV0TrackRouting();

  /// @brief Returns the device V0-plus-track routing plan and channel counters.
  KFParticleGpuV0TrackRoutingView DeviceV0TrackRouting();

  /// @brief Returns device routed V0-plus-track tasks and generation counters.
  KFParticleGpuV0TrackGenerationStorageView DeviceV0TrackGenerationStorage();

  /// @brief Returns the host decay dependency graph with active sizes and cached revision.
  KFParticleGpuDecayGraphView HostDecayGraph();

  /// @brief Returns the device decay dependency graph with active sizes and cached revision.
  KFParticleGpuDecayGraphView DeviceDecayGraph();

  /// @brief Returns device graph-operation descriptors, tasks, results, and counters.
  KFParticleGpuGraphOperationStorageView DeviceGraphOperationStorage();

  /// @}
  /// @name Input upload, resets, and output readback
  /// @{

  /// @brief Copies full allocated input and transaction buffers H2D, excluding work statuses and tiles, then waits.
  void UploadInput();

  /**
   * @brief Copies full candidate storage, diagnostics, routing indices, and pool counters H2D, then waits.
   * @pre All copied host data, including optional mirrors, has been initialized by the caller.
   */
  void UploadCandidates();

  /// @brief Copies full host selected-index/channel arrays and their counters H2D, then waits.
  void UploadSelectedCandidates();

  /**
   * @brief Zeros host/device candidate and daughter counters and overflow flags, then waits; leaves payload intact.
   * @throws std::logic_error If candidate counters have not been allocated.
   */
  void ResetCandidates();

  /**
   * @brief Zeros host explicit two-daughter task counters and queues H2D copies without waiting.
   * @throws std::logic_error If task counters have not been allocated.
   */
  void ResetTwoDaughterTaskStatus();

  /// @brief Downloads explicit two-daughter task counters and waits; returns zeros if unallocated.
  KFParticleGpuTwoDaughterTaskStatus DownloadTwoDaughterTaskStatus();

  /**
   * @brief Zeros host two-daughter routing/selection counters and queues H2D copies without waiting.
   * @throws std::logic_error If routed-task counters have not been allocated.
   */
  void ResetTwoDaughterRoutingStatus();

  /// @brief Downloads two-daughter routing totals and channel counters, then waits; returns zeros if unallocated.
  KFParticleGpuTwoDaughterRoutingStatus DownloadTwoDaughterRoutingStatus();

  /// @brief Downloads the full routed two-daughter task buffer to an on-demand pinned mirror and waits.
  void DownloadTwoDaughterRoutedTasks();

  /**
   * @brief Returns a pinned routed-task mirror, allocating if needed but not downloading data.
   * @return Host pointer, or nullptr if the device task buffer is unallocated.
   */
  KFParticleGpuTwoDaughterRoutedTask* HostTwoDaughterRoutedTasks();

  /// @brief Updates the host two-daughter channel mask, copies it H2D, and waits; leaves plan revision unchanged.
  void SetTwoDaughterRoutingEnabledChannels(const KFParticleGpuChannelMask& enabled);

  /**
   * @brief Zeros host explicit V0-plus-track task counters and queues H2D copies without waiting.
   * @throws std::logic_error If task counters have not been allocated.
   */
  void ResetV0TrackTaskStatus();

  /// @brief Downloads explicit V0-plus-track task counters and waits; returns zeros if unallocated.
  KFParticleGpuV0TrackTaskStatus DownloadV0TrackTaskStatus();

  /**
   * @brief Zeros host V0-plus-track routing counters and queues H2D copies without waiting.
   * @throws std::logic_error If routed-task counters have not been allocated.
   */
  void ResetV0TrackRoutingStatus();

  /// @brief Downloads V0-plus-track routing totals and channel counters, then waits; returns zeros if unallocated.
  KFParticleGpuV0TrackRoutingStatus DownloadV0TrackRoutingStatus();

  /// @brief Downloads the full routed V0-plus-track task buffer to an on-demand pinned mirror and waits.
  void DownloadV0TrackRoutedTasks();

  /**
   * @brief Returns a pinned V0-track task mirror, allocating if needed but not downloading data.
   * @return Host pointer, or nullptr if the device task buffer is unallocated.
   */
  KFParticleGpuV0TrackRoutedTask* HostV0TrackRoutedTasks();

  /// @brief Updates the host V0-track enabled-channel mask, copies it H2D, and waits; leaves plan revision unchanged.
  void SetV0TrackRoutingEnabledChannels(const KFParticleGpuChannelMask& enabled);

  /**
   * @brief Zeros host/device selected-output count and overflow flags, then waits; leaves index arrays intact.
   * @throws std::logic_error If selected-output counters have not been allocated.
   */
  void ResetSelectedCandidates();

  /**
   * @brief Launches a device reset over all V0 selection-result slots and waits; does not refresh a host mirror.
   * @throws std::logic_error If selection-result storage has not been allocated.
   */
  void ResetV0SelectionResults();

  /**
   * @brief ORs @p flags into the host candidate overflow mask, copies it H2D, and waits.
   * @pre The host mask is current; download device flags first if they must be preserved.
   * @throws std::logic_error If candidate counters have not been allocated.
   */
  void MarkCandidateOverflow(unsigned int flags);

  /// @brief Downloads candidate/daughter counters and overflow flags, then waits; returns zeros if unallocated.
  KFParticleGpuCandidatePoolStatus DownloadCandidateStatus();

  /**
   * @brief Downloads full candidate/daughter storage, selection diagnostics, and counters, then waits.
   * @note Routing descriptor indices require DownloadCandidateDescriptorIndices().
   */
  void DownloadCandidates();

  /**
   * @brief Downloads candidate/daughter prefixes and pool counters, then waits; preserves allocated SoA strides.
   * @param candidates Number of candidate records to copy.
   * @param daughters Number of flattened daughter identifiers to copy.
   * @param includeSelectionDiagnostics Also copy V0 selection results to an on-demand pinned mirror.
   * @throws std::out_of_range If either count exceeds its allocated capacity.
   */
  void DownloadCandidateOutput(unsigned int candidates,
                               unsigned int daughters,
                               bool includeSelectionDiagnostics);

  /// @brief Downloads all candidate routing indices to an on-demand pinned mirror and waits.
  void DownloadCandidateDescriptorIndices();

  /// @brief Downloads all V0 selection results to an on-demand pinned mirror and waits.
  void DownloadV0SelectionResults();

  /// @brief Downloads full selected-index/channel arrays and their counters, then waits.
  void DownloadSelectedCandidates();

  /**
   * @brief Downloads @p candidates selected-index/channel entries and their counters, then waits.
   * @throws std::out_of_range If the count exceeds selected-candidate capacity.
   */
  void DownloadSelectedCandidateOutput(unsigned int candidates);

  /// @}
  /// @name Revision-cached plan uploads
  /// @{

  /**
   * @brief Uploads a compiled two-daughter routing plan, resets its counters, and waits.
   * @param plan Compiled plan with a nonzero source revision.
   * @param force Upload even if its revision is already cached.
   * @return True if uploaded; false if the revision matched and no work was submitted.
   * @throws std::invalid_argument If the plan has not been compiled.
   */
  bool UploadTwoDaughterRoutingPlan(const KFParticleGpuTwoDaughterRoutingPlan& plan,
                                    bool force = false);

  /// @brief Returns the cached two-daughter plan revision, or zero before the first upload.
  unsigned long long TwoDaughterRoutingRevision() const;

  /**
   * @brief Uploads a compiled V0-plus-track routing plan, resets its counters, and waits.
   * @param plan Compiled plan with a nonzero source revision.
   * @param force Upload even if its revision is already cached.
   * @return True if uploaded; false if the revision matched and no work was submitted.
   * @throws std::invalid_argument If the plan has not been compiled.
   */
  bool UploadV0TrackRoutingPlan(const KFParticleGpuV0TrackRoutingPlan& plan,
                                bool force = false);

  /// @brief Returns the cached V0-plus-track plan revision, or zero before the first upload.
  unsigned long long V0TrackRoutingRevision() const;

  /**
   * @brief Uploads compiled decay graph nodes, groups, and family coverage, then waits.
   * @param plan Compiled dependency graph with a nonzero source revision.
   * @param force Upload even if its revision is already cached.
   * @return True if uploaded; false if the revision matched and no work was submitted.
   * @throws std::invalid_argument If the graph has not been compiled.
   */
  bool UploadDecayGraphPlan(const KFParticleGpuDecayGraphPlan& plan,
                            bool force = false);

  /// @brief Returns the cached decay graph revision, or zero before the first upload.
  unsigned long long DecayGraphRevision() const;

  /**
   * @brief Uploads graph-operation descriptors, zeros device channel counters, and waits.
   * @param plan Decay plan supplying graph-operation channels and their revision.
   * @param force Upload even if its revision is already cached.
   * @return True if uploaded; false if the revision matched and no work was submitted.
   */
  bool UploadGraphOperationPlan(const KFParticleGpuDecayPlan& plan,
                                bool force = false);

  /// @brief Returns the cached graph-operation plan revision, initially zero.
  unsigned long long GraphOperationRevision() const;

  /// @brief Returns the active number of uploaded graph-operation descriptors.
  unsigned int GraphOperationDescriptorSize() const;

  /// @brief Returns host graph-operation descriptor storage, or nullptr if unallocated.
  KFParticleGpuGraphOperationDescriptor* HostGraphOperationDescriptors();

  /**
   * @brief Returns a pinned graph-result mirror, allocating if needed but not downloading data.
   * @return Host pointer, or nullptr if the device result buffer is unallocated.
   */
  KFParticleGpuGraphOperationResult* HostGraphOperationResults();

  /// @}

 private:
  /// @brief Private storage, active sizes, plan revisions, and borrowed queue, defined in the implementation file.
  struct Impl;

  /// @brief Sole owner of the implementation and its buffers, but not of the supplied queue.
  std::unique_ptr<Impl> fImpl;
};

#endif
