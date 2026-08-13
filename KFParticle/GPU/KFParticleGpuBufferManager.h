/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef KFPARTICLEGPUBUFFERMANAGER_H
#define KFPARTICLEGPUBUFFERMANAGER_H

#include "KFParticleGpuCandidatePool.h"
#include "KFParticleGpuDecayGraphPlan.h"
#include "KFParticleGpuDeviceStorage.h"
#include "KFParticleGpuInputData.h"
#include "KFParticleGpuRoutingPlan.h"
#include "KFParticleGpuSelection.h"

#include <cstdint>
#include <memory>

namespace xpu
{
  class queue;
}

struct KFParticleGpuBufferCapacities
{
  unsigned int tracks = 0;
  unsigned int vertices = 0;
  unsigned int events = 0;
  unsigned int candidates = 0;
  unsigned int daughterIds = 0;
  unsigned int selectedCandidates = 0;
  unsigned int twoDaughterTasks = 0;
  unsigned int twoDaughterRoutedTasks = 0;
  unsigned int twoDaughterRoutingDescriptors = 0;
  unsigned int twoDaughterRoutingCompatibility = 0;
  unsigned int twoDaughterRoutingGroups = 0;
  unsigned int v0TrackTasks = 0;
  unsigned int v0TrackRoutedTasks = 0;
  unsigned int v0TrackRoutingDescriptors = 0;
  unsigned int v0TrackRoutingCompatibility = 0;
  unsigned int v0TrackRoutingGroups = 0;
  unsigned int decayGraphNodes = 0;
  unsigned int decayGraphGroups = 0;
  unsigned int decayGraphFamilies = 0;
  unsigned int graphOperationDescriptors = 0;
  unsigned int graphOperationTasks = 0;
  bool nonhomogeneousField = false;
};

struct KFParticleGpuTwoDaughterTaskStatus
{
  unsigned int accepted = 0;
  unsigned int totalPairs = 0;
  unsigned int overflowFlags = 0;
};

struct KFParticleGpuV0TrackTaskStatus
{
  unsigned int accepted = 0;
  unsigned int totalPairs = 0;
  unsigned int overflowFlags = 0;
};

struct KFParticleGpuCandidatePoolStatus
{
  unsigned int candidates = 0;
  unsigned int daughters = 0;
  unsigned int overflowFlags = 0;
};

/**
 * Process-persistent owner of KFParticle XPU buffers.
 *
 * Host and device views are temporary non-owning descriptors. Capacity grows
 * on demand and is not reduced between events.
 */
class KFParticleGpuBufferManager
{
 public:
  explicit KFParticleGpuBufferManager(xpu::queue& queue);
  ~KFParticleGpuBufferManager();

  KFParticleGpuBufferManager(const KFParticleGpuBufferManager&) = delete;
  KFParticleGpuBufferManager& operator=(const KFParticleGpuBufferManager&) = delete;

  void EnsureCapacity(const KFParticleGpuBufferCapacities& requested);
  const KFParticleGpuBufferCapacities& Capacities() const;
  const KFParticleGpuDeviceStorage& Storage() const;
  std::uint64_t CapacityGrowthCount() const;
  std::uint64_t AllocatedBytes() const;
  std::uint64_t InputPayloadBytes() const;
  std::uint64_t CandidatePayloadBytes() const;
  std::uint64_t SelectedCandidatePayloadBytes() const;

  void SetInputSizes(unsigned int tracks, unsigned int vertices, unsigned int events);
  void EnsureTwoDaughterTaskCapacity(unsigned int capacity);
  void EnsureTwoDaughterRoutedTaskCapacity(unsigned int capacity);
  void EnsureV0TrackTaskCapacity(unsigned int capacity);
  void EnsureV0TrackRoutedTaskCapacity(unsigned int capacity);
  void EnsureGraphOperationTaskCapacity(unsigned int capacity);
  unsigned int TrackSize() const;
  unsigned int VertexSize() const;
  unsigned int EventSize() const;

  KFParticleGpuInputTrackSoAView HostInputTracks();
  KFParticleGpuInputTrackSoAView DeviceInputTracks();
  KFParticleGpuVertexSoAView HostPrimaryVertices();
  KFParticleGpuVertexSoAView DevicePrimaryVertices();
  KFParticleGpuEventDesc* HostEvents();
  KFParticleGpuEventDesc* DeviceEvents();

  KFParticleGpuCandidatePoolView HostCandidates();
  KFParticleGpuCandidatePoolView DeviceCandidates();
  KFParticleGpuV0SelectionResultView HostV0SelectionResults();
  KFParticleGpuV0SelectionResultView DeviceV0SelectionResults();
  KFParticleGpuSelectedCandidateIndexView HostSelectedCandidates();
  KFParticleGpuSelectedCandidateIndexView DeviceSelectedCandidates();
  KFParticleGpuCandidateDescriptorIndexView HostCandidateDescriptorIndices();
  KFParticleGpuCandidateDescriptorIndexView DeviceCandidateDescriptorIndices();
  KFParticleGpuTwoDaughterRoutingView HostTwoDaughterRouting();
  KFParticleGpuTwoDaughterRoutingView DeviceTwoDaughterRouting();
  KFParticleGpuTwoDaughterSelectionWorkspaceView HostTwoDaughterSelectionWorkspace();
  KFParticleGpuTwoDaughterSelectionWorkspaceView DeviceTwoDaughterSelectionWorkspace();
  KFParticleGpuTwoDaughterGenerationStorageView
  DeviceTwoDaughterGenerationStorage();
  KFParticleGpuV0TrackRoutingView HostV0TrackRouting();
  KFParticleGpuV0TrackRoutingView DeviceV0TrackRouting();
  KFParticleGpuV0TrackGenerationStorageView DeviceV0TrackGenerationStorage();
  KFParticleGpuDecayGraphView HostDecayGraph();
  KFParticleGpuDecayGraphView DeviceDecayGraph();
  KFParticleGpuGraphOperationStorageView DeviceGraphOperationStorage();

  void UploadInput();
  void UploadCandidates();
  void UploadSelectedCandidates();
  void ResetCandidates();
  void ResetTwoDaughterTaskStatus();
  KFParticleGpuTwoDaughterTaskStatus DownloadTwoDaughterTaskStatus();
  void ResetTwoDaughterRoutingStatus();
  KFParticleGpuTwoDaughterRoutingStatus DownloadTwoDaughterRoutingStatus();
  void DownloadTwoDaughterRoutedTasks();
  KFParticleGpuTwoDaughterRoutedTask* HostTwoDaughterRoutedTasks();
  void SetTwoDaughterRoutingEnabledChannels(const KFParticleGpuChannelMask& enabled);
  void ResetV0TrackTaskStatus();
  KFParticleGpuV0TrackTaskStatus DownloadV0TrackTaskStatus();
  void ResetV0TrackRoutingStatus();
  KFParticleGpuV0TrackRoutingStatus DownloadV0TrackRoutingStatus();
  void DownloadV0TrackRoutedTasks();
  KFParticleGpuV0TrackRoutedTask* HostV0TrackRoutedTasks();
  void SetV0TrackRoutingEnabledChannels(const KFParticleGpuChannelMask& enabled);
  void ResetSelectedCandidates();
  void ResetV0SelectionResults();
  void MarkCandidateOverflow(unsigned int flags);
  KFParticleGpuCandidatePoolStatus DownloadCandidateStatus();
  void DownloadCandidates();
  void DownloadCandidateDescriptorIndices();
  void DownloadV0SelectionResults();
  void DownloadSelectedCandidates();
  bool UploadTwoDaughterRoutingPlan(const KFParticleGpuTwoDaughterRoutingPlan& plan,
                                    bool force = false);
  unsigned long long TwoDaughterRoutingRevision() const;
  bool UploadV0TrackRoutingPlan(const KFParticleGpuV0TrackRoutingPlan& plan,
                                bool force = false);
  unsigned long long V0TrackRoutingRevision() const;
  bool UploadDecayGraphPlan(const KFParticleGpuDecayGraphPlan& plan,
                            bool force = false);
  unsigned long long DecayGraphRevision() const;
  bool UploadGraphOperationPlan(const KFParticleGpuDecayPlan& plan,
                                bool force = false);
  unsigned long long GraphOperationRevision() const;
  unsigned int GraphOperationDescriptorSize() const;
  KFParticleGpuGraphOperationDescriptor* HostGraphOperationDescriptors();
  KFParticleGpuGraphOperationDescriptor* DeviceGraphOperationDescriptors();
  KFParticleGpuGraphOperationTask* DeviceGraphOperationTasks();
  KFParticleGpuGraphOperationResult* HostGraphOperationResults();
  KFParticleGpuGraphOperationResult* DeviceGraphOperationResults();

 private:
  struct Impl;
  std::unique_ptr<Impl> fImpl;
};

#endif
